"""
Training-Skript für den RIAWELC-Schweißnaht-Klassifikator.

Ablauf:
  1. Config definieren (alle Hyperparameter an einem Ort)
  2. Seed setzen für Reproduzierbarkeit
  3. W&B initialisieren
  4. Dataset + DataLoader erstellen
  5. LightningModule erstellen (mit optionalem freeze_backbone)
  6. Trainer mit Callbacks erstellen
  7. trainer.fit() starten
  8. Bestes Modell auf dem Test-Set evaluieren

Drei geplante Runs:
  run_1_baseline    : CE-Loss, Backbone eingefroren, 10 Epochen
  run_2_finetune    : CE-Loss, Backbone aufgetaut,   20 Epochen, niedrige LR
  run_3_focal       : Focal Loss, Backbone aufgetaut, 20 Epochen, niedrige LR
"""

import random
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from .dataset import CLASS_NAMES, WeldDefectDataset, get_train_transforms, get_val_transforms
from .model import WeldDefectModule

# ---------------------------------------------------------------------------
# Pfade
# ---------------------------------------------------------------------------

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
MODELS_DIR  = OUTPUTS_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Run-Konfigurationen
# ---------------------------------------------------------------------------

# Jede Config ist ein Dict — alle Hyperparameter an einem Ort.
# W&B loggt das gesamte Dict, damit jeder Run vollständig reproduzierbar ist.

RUN_CONFIGS: dict[str, dict] = {
    "run_1_baseline": {
        "run_name":      "baseline-ce-frozen",
        "loss_type":     "ce",
        "freeze":        True,    # Backbone eingefroren: nur conv1 + fc trainieren
        "lr":            1e-3,
        "weight_decay":  1e-4,
        "gamma":         2.0,     # nur relevant für focal, hier ignoriert
        "batch_size":    32,
        "max_epochs":    10,
        "seed":          42,
        "num_workers":   4,
    },
    "run_2_finetune": {
        "run_name":      "finetune-ce-unfrozen",
        "loss_type":     "ce",
        "freeze":        False,   # Gesamtes Netz trainieren
        "lr":            3e-4,    # Niedrigere LR — Backbone vorsichtig anpassen
        "weight_decay":  1e-4,
        "gamma":         2.0,
        "batch_size":    32,
        "max_epochs":    20,
        "seed":          42,
        "num_workers":   4,
    },
    "run_3_focal": {
        "run_name":      "focal-loss-unfrozen",
        "loss_type":     "focal",
        "freeze":        False,
        "lr":            3e-4,
        "weight_decay":  1e-4,
        "gamma":         2.0,
        "batch_size":    32,
        "max_epochs":    20,
        "seed":          42,
        "num_workers":   4,
    },
}


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """
    Setzt den Zufallsgenerator-Seed für vollständige Reproduzierbarkeit.

    Warum alle drei (random, numpy, torch)?
    PyTorch nutzt intern numpy und Pythons random — wenn nur einer gesetzt ist,
    können Datenaugmentation oder DataLoader-Shuffling trotzdem variieren.
    pl.seed_everything() macht das alles auf einmal + setzt PYTHONHASHSEED.

    Args:
        seed: Ganzzahl, z.B. 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pl.seed_everything(seed, workers=True)


# ---------------------------------------------------------------------------
# DataLoader-Erstellung
# ---------------------------------------------------------------------------

def build_dataloaders(cfg: dict) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Erstellt Train-, Val- und Test-DataLoader aus dem RIAWELC-Datensatz.

    Args:
        cfg: Run-Config-Dict (braucht batch_size und num_workers).

    Returns:
        Tuple (train_loader, val_loader, test_loader).
    """
    train_ds = WeldDefectDataset(split="training",   transform=get_train_transforms())
    val_ds   = WeldDefectDataset(split="validation", transform=get_val_transforms())
    test_ds  = WeldDefectDataset(split="testing",    transform=get_val_transforms())

    # pin_memory=True: Daten im CPU-RAM als "pinned" (nicht-auslagerbar) halten.
    # Das beschleunigt den Transfer zur GPU, weil kein Extra-Kopier-Schritt nötig.
    # Nur sinnvoll wenn eine GPU verfügbar ist.
    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,         # Reihenfolge jede Epoche zufällig — wichtig für Training
        num_workers=cfg["num_workers"],
        pin_memory=pin,
        persistent_workers=cfg["num_workers"] > 0,  # Worker-Prozesse zwischen Epochen behalten
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,        # Val-Reihenfolge konstant — Metriken bleiben vergleichbar
        num_workers=cfg["num_workers"],
        pin_memory=pin,
        persistent_workers=cfg["num_workers"] > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=pin,
        persistent_workers=cfg["num_workers"] > 0,
    )
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Haupt-Trainingsfunktion
# ---------------------------------------------------------------------------

def train(run_key: str) -> dict:
    """
    Führt einen vollständigen Training-Run durch.

    Ablauf:
      1. Config laden + Seed setzen
      2. W&B + Lightning-Logger initialisieren
      3. DataLoader erstellen + Klassenverteilung ermitteln
      4. LightningModule erstellen
      5. Callbacks: ModelCheckpoint + EarlyStopping
      6. Trainer starten
      7. Bestes Modell auf Test-Set evaluieren

    Args:
        run_key: Schlüssel in RUN_CONFIGS, z.B. "run_1_baseline".

    Returns:
        Dict mit Test-Metriken des besten Modells.
    """
    cfg = RUN_CONFIGS[run_key]
    set_seed(cfg["seed"])

    # --- W&B Logger ---
    # WandbLogger ist der Lightning-native Weg, W&B zu integrieren.
    # Er fängt alle self.log()-Aufrufe aus dem LightningModule ab
    # und schickt sie an das W&B-Dashboard.
    wandb_logger = WandbLogger(
        project="riawelc-weld-defect",  # Projektname im W&B-Dashboard
        name=cfg["run_name"],            # Name dieses Runs
        config=cfg,                      # Alle Hyperparameter loggen
        log_model=False,                 # Wir speichern Modelle selbst via Checkpoint
    )

    # --- DataLoader ---
    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    # Klassenverteilung aus dem Training-Split für Verlust-Gewichte
    # WeldDefectDataset.samples ist eine Liste von (path, label)-Tupeln
    labels = [label for _, label in train_loader.dataset.samples]
    class_counts = [labels.count(i) for i in range(len(CLASS_NAMES))]
    print(f"\nKlassenverteilung Training: { {n: c for n, c in zip(CLASS_NAMES, class_counts)} }")

    # --- LightningModule ---
    module = WeldDefectModule(
        class_counts=class_counts,
        loss_type=cfg["loss_type"],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        gamma=cfg["gamma"],
        max_epochs=cfg["max_epochs"],
    )

    # Backbone einfrieren wenn konfiguriert (Run 1)
    if cfg["freeze"]:
        module.model.freeze_backbone()
        trainable = sum(p.numel() for p in module.model.parameters() if p.requires_grad)
        print(f"Backbone eingefroren. Trainierbare Parameter: {trainable:,}")

    # --- Callbacks ---

    # ModelCheckpoint: speichert das Modell wenn val/loss ein neues Minimum erreicht.
    # save_top_k=1: nur das beste Modell behalten (nicht jede Epoche).
    # filename enthält Epoche und val_loss im Dateinamen für schnelle Orientierung.
    checkpoint_cb = ModelCheckpoint(
        dirpath=MODELS_DIR / cfg["run_name"],
        filename="{epoch:02d}-{val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        verbose=True,
    )

    # EarlyStopping: bricht ab wenn val/loss sich patience=7 Epochen nicht verbessert.
    # min_delta=1e-4: Verbesserungen kleiner als 0.0001 zählen nicht.
    # Verhindert sinnloses Weitertrainieren bei Plateau oder Overfitting.
    early_stop_cb = EarlyStopping(
        monitor="val/loss",
        patience=7,
        min_delta=1e-4,
        mode="min",
        verbose=True,
    )

    # --- Trainer ---
    # Der Trainer übernimmt: Trainingsschleife, Device-Management,
    # Gradient-Accumulation, Mixed Precision, Logging, Callbacks.
    trainer = pl.Trainer(
        max_epochs=cfg["max_epochs"],
        accelerator="auto",    # GPU wenn vorhanden, sonst CPU
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_cb, early_stop_cb],
        log_every_n_steps=10,  # Nicht jeden einzelnen Step loggen (zu viel Rauschen)
        deterministic=True,    # Reproduzierbare Ergebnisse (etwas langsamer)
    )

    # --- Training ---
    print(f"\nStarte Run: {cfg['run_name']}")
    trainer.fit(module, train_loader, val_loader)

    # --- Test-Evaluation mit bestem Checkpoint ---
    # ckpt_path="best" lädt automatisch das von ModelCheckpoint gespeicherte
    # beste Modell — nicht die Gewichte vom Ende des Trainings.
    print(f"\nEvaluiere bestes Modell auf Test-Set ...")
    test_results = trainer.test(module, test_loader, ckpt_path="best", verbose=True)

    # W&B Run sauber beenden
    wandb.finish()

    return test_results[0] if test_results else {}


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Welchen Run starten? Argument von der Kommandozeile oder Default.
    run_key = sys.argv[1] if len(sys.argv) > 1 else "run_1_baseline"

    if run_key not in RUN_CONFIGS:
        print(f"Unbekannter Run-Key: '{run_key}'")
        print(f"Verfügbar: {list(RUN_CONFIGS.keys())}")
        sys.exit(1)

    print(f"Starte: {run_key}")
    results = train(run_key)
    print(f"\nTest-Ergebnisse: {results}")

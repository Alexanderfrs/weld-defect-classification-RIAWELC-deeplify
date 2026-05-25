"""
DataLoader-Test und Batch-Visualisierung (Phase 2.4).

Zweck dieses Skripts:
  - Sicherstellen, dass Dataset + DataLoader korrekt zusammenspielen
  - Visuell prüfen, ob Augmentationen plausibel aussehen
  - Batch-Shape und Wertebereich bestätigen

Ausgabe: outputs/plots/04_dataloader_batch.png
"""

from pathlib import Path
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from src.dataset import (
    WeldDefectDataset,
    get_train_transforms,
    get_val_transforms,
    CLASS_NAMES,
    PIXEL_MEAN,
    PIXEL_STD,
)

OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_dataloaders(batch_size: int = 32, num_workers: int = 4) -> dict[str, DataLoader]:
    """
    Erstellt DataLoader für alle drei Splits.

    Warum unterschiedliche num_workers-Empfehlung pro Split?
    Beim Training ist schnelles Laden kritisch — der DataLoader muss
    Batches vorausladen, während die GPU rechnet. Für Val/Test reicht
    weniger Parallelisierung.

    Args:
        batch_size:  Samples pro Batch (32 passt gut auf 4GB VRAM).
        num_workers: Parallele Lade-Threads. 0 = kein Multiprocessing
                     (nützlich für Debugging). 4 ist ein guter Standard.

    Returns:
        Dict mit Keys "training", "validation", "testing".
    """
    train_ds = WeldDefectDataset(split="training", transform=get_train_transforms())
    val_ds   = WeldDefectDataset(split="validation", transform=get_val_transforms())
    test_ds  = WeldDefectDataset(split="testing",  transform=get_val_transforms())

    return {
        "training": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,          # zufällige Reihenfolge beim Training
            num_workers=num_workers,
            pin_memory=True,       # schnellerer CPU→GPU Transfer
            persistent_workers=(num_workers > 0),  # Worker-Prozesse zwischen Batches halten
        ),
        "validation": DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,         # deterministische Reihenfolge für Metriken
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        ),
        "testing": DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        ),
    }


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """
    Macht die Normalisierung rückgängig — für die Visualisierung.

    Nach Normalize(mean, std) liegen Pixelwerte im Bereich ~[-3, +3].
    Matplotlib erwartet [0, 1] oder [0, 255]. Wir kehren die Formel um:
      pixel_original = pixel_normalized * std + mean

    Args:
        tensor: Normalisierter Tensor der Form (1, H, W).

    Returns:
        NumPy-Array (H, W), geclippt auf [0, 1].
    """
    arr = tensor.squeeze().numpy()          # (1, H, W) → (H, W)
    arr = arr * PIXEL_STD + PIXEL_MEAN      # Normalisierung rückgängig machen
    return np.clip(arr, 0.0, 1.0)           # Auf [0,1] begrenzen für imshow


def plot_batch(loader: DataLoader, title: str, out_path: Path, n_images: int = 16) -> None:
    """
    Visualisiert die ersten n_images Bilder aus einem DataLoader-Batch.

    Warum ist diese Visualisierung wichtig?
    Sie ist ein "Sanity Check" — wir sehen, ob:
    - die Augmentationen vernünftig aussehen (Rotation nicht zu extrem)
    - die Bilder den richtigen Klassen zugeordnet sind
    - nichts mit der Bildvorverarbeitung schiefgelaufen ist

    Args:
        loader:    DataLoader, aus dem ein Batch gezogen wird.
        title:     Titel des Plots.
        out_path:  Pfad zum Speichern des PNG.
        n_images:  Anzahl der Bilder im Plot (Standard: 16 → 4×4-Grid).
    """
    # Einen einzigen Batch ziehen (iter + next, weil DataLoader kein Index hat)
    images, labels = next(iter(loader))

    cols = 4
    rows = int(np.ceil(n_images / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.atleast_1d(axes).reshape(-1)
    max_images = min(n_images, len(images))

    for i, ax in enumerate(axes):
        if i < max_images:
            ax.imshow(denormalize(images[i]), cmap="gray", vmin=0, vmax=1)
            ax.set_title(CLASS_NAMES[labels[i].item()], fontsize=10)
        ax.axis("off")

    fig.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Gespeichert: {out_path}")


if __name__ == "__main__":
    print("Erstelle DataLoader...")
    loaders = build_dataloaders(batch_size=32, num_workers=4)

    # Batch-Shape und Wertebereich prüfen
    images, labels = next(iter(loaders["training"]))
    print(f"\nBatch-Shape:      {images.shape}")          # erwartet: (32, 1, 224, 224)
    print(f"Labels-Shape:     {labels.shape}")            # erwartet: (32,)
    print(f"Pixel-Range:      [{images.min():.3f}, {images.max():.3f}]")
    print(f"Label-Verteilung: {dict(zip(*labels.unique(return_counts=True)))}")

    # Anzahl Batches pro Split
    print(f"\nAnzahl Batches (batch_size=32):")
    for split, loader in loaders.items():
        print(f"  {split:12s}: {len(loader)} Batches  ({len(loader.dataset)} Samples)")

    # Visualisierung: Training (mit Augmentation) und Validation (ohne)
    print("\nPlots erstellen...")
    plot_batch(
        loaders["training"],
        title="DataLoader Sanity Check — Training-Batch (mit Augmentation)",
        out_path=OUTPUT_DIR / "04_dataloader_batch_train.png",
    )
    plot_batch(
        loaders["validation"],
        title="DataLoader Sanity Check — Validation-Batch (ohne Augmentation)",
        out_path=OUTPUT_DIR / "04_dataloader_batch_val.png",
    )

    print("\nFertig.")

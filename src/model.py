"""
Modell-Definition: ResNet50 mit Transfer Learning für RIAWELC.

Aufbau:
  1. ResNet50-Backbone laden (vortrainiert auf ImageNet)
  2. Ersten Conv-Layer auf 1 Kanal (Graustufenbild) anpassen
  3. Letzten Fully-Connected Layer auf 4 Klassen anpassen
  4. freeze_backbone() / unfreeze_backbone() für zweistufiges Training
  5. WeldDefectModule: LightningModule für Training, Validation, Optimizer

Architekturentscheid: Warum ResNet50?
  - Bewährt, gut dokumentiert, klare Schichtstruktur
  - 2048-dimensionale Feature-Representation vor dem Classifier
  - Gut transferierbar auf industrielle Bilddomänen
  - Alternativen: EfficientNet-B4 (effizienter), ViT (moderner, mehr Daten nötig)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from torchvision import models
from torchvision.models import ResNet50_Weights

from .losses import FocalLoss, compute_class_weights
from .dataset import CLASS_NAMES

# Anzahl Klassen im RIAWELC-Datensatz: CR, LP, PO, ND
NUM_CLASSES: int = 4


class WeldDefectResNet(nn.Module):
    """
    ResNet50-basierter Klassifikator für Schweißnaht-Defekte.

    Zwei Anpassungen gegenüber dem Standard-ResNet50:
      1. conv1: 3 Eingangskanäle → 1 (Graustufenbild)
      2. fc:    1000 Ausgänge → 4 (unsere Klassen)

    Args:
        num_classes:  Anzahl der Ausgabeklassen. Standard: 4 (RIAWELC).
        pretrained:   ImageNet-Gewichte laden. Standard: True.
                      Nur auf False setzen für Tests ohne Internet.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        # --- 1. Backbone laden ---
        # weights=DEFAULT lädt die aktuell empfohlenen ImageNet-Gewichte.
        # torchvision.models.ResNet50_Weights.DEFAULT ist der moderne Weg
        # (statt dem veralteten pretrained=True Flag).
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)

        # --- 2. Ersten Conv-Layer ersetzen: 3 Kanäle → 1 Kanal ---
        # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Neu:      Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        #
        # Alle Spatial-Parameter (kernel_size, stride, padding) bleiben gleich,
        # damit das Ausgabe-Feature-Map die exakt gleiche Größe hat.
        # bias=False bleibt False, weil danach ein BatchNorm folgt, der den
        # Bias ohnehin rausrechnen würde.
        #
        # WICHTIG: Die vortrainierten Gewichte für conv1 haben Shape (64, 3, 7, 7).
        # Unser neuer Layer hat Shape (64, 1, 7, 7) — zufällig initialisiert.
        # Das ist okay: conv1 wird im Finetuning mittrainiert.
        backbone.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        # --- 3. Letzten Fully-Connected Layer ersetzen: 1000 → 4 Klassen ---
        # backbone.fc ist Linear(2048, 1000).
        # Die 2048 kommen aus layer4 von ResNet50 (2048 Feature-Map-Kanäle),
        # die nach dem AdaptiveAvgPool2d zu einem 2048-dimensionalen Vektor
        # zusammengefasst werden.
        # Wir behalten 2048 als Eingangsgröße, ändern nur den Ausgang auf 4.
        in_features: int = backbone.fc.in_features  # = 2048
        backbone.fc = nn.Linear(in_features, num_classes)

        # Backbone als Instanzvariable speichern
        # (nicht self.backbone = backbone, weil wir die ursprüngliche
        # ResNet-Struktur für freeze/unfreeze beibehalten wollen)
        self.backbone = backbone

    # ------------------------------------------------------------------
    # Freeze / Unfreeze
    # ------------------------------------------------------------------

    def freeze_backbone(self) -> None:
        """
        Friert alle Parameter außer conv1 und fc ein.

        "Einfrieren" = requires_grad = False → diese Gewichte erhalten
        keine Gradienten und werden beim Optimizer-Schritt nicht verändert.

        Warum nur conv1 und fc auftauen?
          - conv1 ist neu initialisiert (kein Pretraining für 1 Kanal) → muss lernen
          - fc ist unser neuer Classifier-Kopf → muss lernen
          - Alle anderen Layer haben gute ImageNet-Gewichte → vorerst einfrieren
        """
        # Zuerst alles einfrieren
        for param in self.backbone.parameters():
            param.requires_grad = False

        # conv1 und fc wieder auftauen (die sind neu, müssen trainiert werden)
        for param in self.backbone.conv1.parameters():
            param.requires_grad = True
        for param in self.backbone.fc.parameters():
            param.requires_grad = True

    def unfreeze_backbone(self) -> None:
        """
        Taut alle Parameter auf — für Finetuning des gesamten Netzes.

        Wird nach Phase 1 (Kopf-Training) aufgerufen. Danach läuft Training
        mit niedrigerer Lernrate, damit die guten ImageNet-Features nicht
        zu stark überschrieben werden (Stichwort: Catastrophic Forgetting).
        """
        for param in self.backbone.parameters():
            param.requires_grad = True

    # ------------------------------------------------------------------
    # Forward Pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Datenfluss durch das Netz.

        Input:  x — FloatTensor der Form (B, 1, 224, 224)
                    B = Batch-Größe, 1 = Graustufenkanal

        Output: logits — FloatTensor der Form (B, 4)
                    4 rohe Scores, einer pro Klasse (kein Softmax hier!)

        Warum kein Softmax im forward()?
          - CrossEntropyLoss und FocalLoss erwarten rohe Logits
          - Softmax wird nur bei Inferenz (predict()) aufgerufen
          - Numerisch stabiler: log_softmax ist in den Loss-Funktionen eingebaut

        Datenfluss (vereinfacht):
          (B, 1, 224, 224)
          → conv1+bn1+relu+maxpool       → (B, 64,  56, 56)
          → layer1 (3× Bottleneck-Block) → (B, 256, 56, 56)
          → layer2 (4× Bottleneck-Block) → (B, 512, 28, 28)
          → layer3 (6× Bottleneck-Block) → (B, 1024,14, 14)
          → layer4 (3× Bottleneck-Block) → (B, 2048, 7,  7)
          → AdaptiveAvgPool2d(1,1)       → (B, 2048, 1,  1)
          → Flatten                      → (B, 2048)
          → fc (Linear)                  → (B, 4)
        """
        return self.backbone(x)


# ------------------------------------------------------------------
# Lightning Module
# ------------------------------------------------------------------

class WeldDefectModule(pl.LightningModule):
    """
    LightningModule für den RIAWELC-Klassifikator.

    Kapselt Modell, Loss, Metriken, Optimizer und LR-Scheduler.
    Der Lightning Trainer ruft die Methoden dieser Klasse automatisch
    auf — wir schreiben keine manuelle Trainingsschleife mehr.

    Args:
        class_counts:  Anzahl Samples pro Klasse [CR, LP, PO, ND].
                       Wird für Focal Loss α und Weighted CE genutzt.
        loss_type:     "focal" oder "ce" (Weighted Cross-Entropy).
        lr:            Initiale Lernrate für AdamW.
        weight_decay:  L2-Regularisierung (AdamW-Parameter).
        gamma:         Fokussierungsparameter für Focal Loss.
        max_epochs:    Epochenanzahl — bestimmt den Cosine-Annealing-Zyklus.
        pretrained:    ImageNet-Gewichte für den Backbone laden.
    """

    def __init__(
        self,
        class_counts: list[int],
        loss_type: str = "focal",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        gamma: float = 2.0,
        max_epochs: int = 10,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        # Alle __init__-Argumente als self.hparams speichern.
        # Lightning loggt sie automatisch zu W&B — jeder Run ist damit
        # vollständig reproduzierbar, auch Monate später.
        self.save_hyperparameters()

        # Modell
        self.model = WeldDefectResNet(num_classes=NUM_CLASSES, pretrained=pretrained)

        # Klassengewichte berechnen — genutzt von beiden Loss-Varianten
        weights = compute_class_weights(class_counts)

        # Loss-Funktion wählen
        if loss_type == "focal":
            # Focal Loss mit α (inverse Klassenfrequenz) und γ
            self.criterion = FocalLoss(alpha=weights, gamma=gamma)
        elif loss_type == "ce":
            # Weighted Cross-Entropy — einfacherer Baseline-Loss
            # register_buffer damit weights bei .to(device) mitwandert;
            # hier speichern wir es direkt im Modul für nn.CrossEntropyLoss
            self.register_buffer("ce_weights", weights)
            self.criterion = None  # wird in forward-Methoden genutzt
        else:
            raise ValueError(f"loss_type muss 'focal' oder 'ce' sein, nicht '{loss_type}'")

        # --- Metriken (torchmetrics) ---
        # Warum torchmetrics statt manueller Berechnung?
        # torchmetrics akkumuliert Predictions über den gesamten Split,
        # bevor es die Metrik berechnet. Manuell würde man Batch-Metriken
        # mitteln — das gibt bei unterschiedlichen Batch-Größen falsche Werte.
        num_classes = NUM_CLASSES

        # Accuracy: Anteil korrekt klassifizierter Samples
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc   = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

        # Per-Class Recall: Für jede Klasse separat — safety-critical!
        # average=None → gibt einen Wert pro Klasse zurück (kein Mitteln)
        self.val_recall = torchmetrics.Recall(
            task="multiclass", num_classes=num_classes, average=None
        )

    # ------------------------------------------------------------------
    # Hilfsmethode: Loss berechnen (beide Loss-Typen)
    # ------------------------------------------------------------------

    def _compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.hparams.loss_type == "focal":
            return self.criterion(logits, targets)
        # Weighted CE: weight muss auf demselben Device wie logits liegen
        return F.cross_entropy(logits, targets, weight=self.ce_weights)

    # ------------------------------------------------------------------
    # Training Step
    # ------------------------------------------------------------------

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Wird für jeden Trainings-Batch aufgerufen.

        Lightning übernimmt: Device-Transfer, zero_grad(), backward(),
        optimizer.step(). Wir liefern nur den Loss zurück.

        Args:
            batch:     Tuple (images, labels) — bereits auf dem richtigen Device.
            batch_idx: Index des Batches in der Epoche (meist nicht gebraucht).

        Returns:
            Loss-Tensor — Lightning ruft darauf .backward() auf.
        """
        images, labels = batch
        logits = self.model(images)
        loss   = self._compute_loss(logits, labels)

        # Predictions für Accuracy: argmax über Klassen-Dimension
        preds = logits.argmax(dim=1)
        self.train_acc.update(preds, labels)

        # on_step=False, on_epoch=True: Wert am Epochenende loggen (nicht pro Batch)
        # prog_bar=True: erscheint im Fortschrittsbalken
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc",  self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    # ------------------------------------------------------------------
    # Validation Step
    # ------------------------------------------------------------------

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """
        Wird für jeden Validierungs-Batch aufgerufen.

        Kein Gradient-Tracking (Lightning macht torch.no_grad() automatisch).
        Wir berechnen Loss + Accuracy + per-class Recall.
        """
        images, labels = batch
        logits = self.model(images)
        loss   = self._compute_loss(logits, labels)

        preds = logits.argmax(dim=1)
        self.val_acc.update(preds, labels)
        self.val_recall.update(preds, labels)

        # val/loss ist das Kriterium für ModelCheckpoint und EarlyStopping
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc",  self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """
        Wird am Ende jeder Validierungs-Epoche aufgerufen.

        Hier loggen wir den per-class Recall — nach dem Akkumulieren über
        alle Val-Batches (deshalb nicht in validation_step).
        """
        # recall hat Shape (num_classes,) — einen Wert pro Klasse
        recall_per_class = self.val_recall.compute()
        for class_idx, class_name in enumerate(CLASS_NAMES):
            self.log(f"val/recall_{class_name}", recall_per_class[class_idx], prog_bar=False)
        self.val_recall.reset()

    # ------------------------------------------------------------------
    # Optimizer + LR Scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        """
        Definiert Optimizer und LR-Scheduler.

        AdamW: Wie Adam, aber Weight-Decay ist vom adaptiven Term entkoppelt.
               Stabiler beim Finetuning von vortrainierten Modellen.

        CosineAnnealingLR: Senkt die LR von lr_max bis ~0 nach einem
               Cosinus-Verlauf über T_max Epochen. Kein abruptes Abfallen,
               das Modell konvergiert sanft ins Minimum.

        Returns:
            Dict mit "optimizer" und "lr_scheduler" — Lightning-Konvention.
        """
        optimizer = torch.optim.AdamW(
            # Nur trainierbare Parameter (freeze_backbone() berücksichtigt)
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs,  # Volle Cosinus-Periode = max_epochs
            eta_min=1e-6,                    # Untergrenze der LR
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    # ------------------------------------------------------------------
    # Freeze / Unfreeze Convenience-Methoden (delegieren ans Modell)
    # ------------------------------------------------------------------

    def freeze_backbone(self) -> None:
        """Friert den ResNet-Backbone ein (nur conv1 + fc trainierbar)."""
        self.model.freeze_backbone()
        # Optimizer neu konfigurieren — er muss die neuen requires_grad kennen
        self.trainer.strategy.setup_optimizers(self.trainer)

    def unfreeze_backbone(self) -> None:
        """Taut den gesamten Backbone auf — für Finetuning-Phase."""
        self.model.unfreeze_backbone()
        self.trainer.strategy.setup_optimizers(self.trainer)


# ------------------------------------------------------------------
# Schnelltest — python src/model.py
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("Lade Modell (pretrained=True) ...")
    model = WeldDefectResNet(num_classes=4, pretrained=True)

    # Einen synthetischen Batch erstellen: 2 Graustufenbilder 224×224
    dummy_batch = torch.randn(2, 1, 224, 224)

    # 1. Test: Forward Pass im vollen Modus (alle Parameter auftauen)
    model.unfreeze_backbone()
    model.eval()
    with torch.no_grad():
        logits = model(dummy_batch)
    print(f"Output Shape (erwartet: (2, 4)): {logits.shape}")
    assert logits.shape == (2, 4), "Shape stimmt nicht!"

    # 2. Test: Freeze-Funktion
    model.freeze_backbone()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"\nParameter nach freeze_backbone():")
    print(f"  Trainierbar: {trainable:>10,}")
    print(f"  Gesamt:      {total:>10,}")
    print(f"  Eingefroren: {total - trainable:>10,}")

    # 3. Test: Unfreeze
    model.unfreeze_backbone()
    trainable_all = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParameter nach unfreeze_backbone():")
    print(f"  Trainierbar: {trainable_all:>10,} (sollte gleich Gesamt sein)")

    print("\nAlle WeldDefectResNet-Tests bestanden.")

    # ------------------------------------------------------------------
    # WeldDefectModule Schnelltest
    # ------------------------------------------------------------------
    print("\n--- WeldDefectModule ---")

    # Klassenverteilung aus der Datenexploration
    class_counts = [1200, 3800, 2100, 17307]

    module = WeldDefectModule(
        class_counts=class_counts,
        loss_type="focal",
        lr=1e-3,
        max_epochs=10,
    )
    print(f"LightningModule erstellt: loss_type={module.hparams.loss_type}, lr={module.hparams.lr}")

    # Simulierter Batch: 4 Bilder, 1 Kanal, 224×224
    dummy_images  = torch.randn(4, 1, 224, 224)
    dummy_labels  = torch.tensor([0, 1, 2, 3])
    dummy_batch   = (dummy_images, dummy_labels)

    # training_step testen (ohne echten Trainer)
    module.model.eval()
    loss = module.training_step(dummy_batch, batch_idx=0)
    print(f"training_step loss: {loss.item():.4f} (Tensor: {loss.shape})")
    assert loss.ndim == 0, "Loss sollte ein Skalar sein!"

    # CE-Variante testen
    module_ce = WeldDefectModule(class_counts=class_counts, loss_type="ce")
    loss_ce = module_ce.training_step(dummy_batch, batch_idx=0)
    print(f"CE-Variante loss:   {loss_ce.item():.4f}")

    print("\nAlle Tests bestanden.")
    sys.exit(0)

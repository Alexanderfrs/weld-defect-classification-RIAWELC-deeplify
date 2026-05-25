"""
Modell-Definition: ResNet50 mit Transfer Learning für RIAWELC.

Aufbau:
  1. ResNet50-Backbone laden (vortrainiert auf ImageNet)
  2. Ersten Conv-Layer auf 1 Kanal (Graustufenbild) anpassen
  3. Letzten Fully-Connected Layer auf 4 Klassen anpassen
  4. freeze_backbone() / unfreeze_backbone() für zweistufiges Training

Architekturentscheid: Warum ResNet50?
  - Bewährt, gut dokumentiert, klare Schichtstruktur
  - 2048-dimensionale Feature-Representation vor dem Classifier
  - Gut transferierbar auf industrielle Bilddomänen
  - Alternativen: EfficientNet-B4 (effizienter), ViT (moderner, mehr Daten nötig)
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

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

    print("\nAlle Tests bestanden.")
    sys.exit(0)

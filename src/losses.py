"""
Focal Loss für Klassenungleichgewicht im RIAWELC-Datensatz.

Motivation:
  Standard Cross-Entropy behandelt alle Samples gleich. Bei unbalancierten
  Klassen (z.B. viele ND-Bilder) dominiert die Mehrheitsklasse den Gradienten.
  Focal Loss dämpft einfache, korrekt klassifizierte Samples und lässt das
  Training sich auf schwierige Fälle konzentrieren.

Formel:
  FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

  α_t : Klassengewicht (inverse Frequenz), kompensiert Ungleichgewicht
  γ   : Fokussierungsparameter (Standard: 2), dämpft einfache Samples
  p_t : vorhergesagte Wahrscheinlichkeit der korrekten Klasse

Referenz: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss für Multi-Klassen-Klassifikation.

    Args:
        alpha:      Klassengewichte als 1D-Tensor der Länge num_classes.
                    None → alle Klassen gleich gewichtet.
        gamma:      Fokussierungsparameter. 0 = normaler Cross-Entropy Loss.
                    Empfehlung: 2.0 (aus dem Original-Paper).
        reduction:  Wie die Losses über den Batch aggregiert werden.
                    "mean" (Standard) oder "sum".
    """

    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        # alpha als Buffer registrieren — damit er bei .to(device) mitwandert,
        # aber nicht als trainierbarer Parameter gilt.
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None
        self.gamma = gamma
        if reduction not in {"mean", "sum"}:
            raise ValueError(
                f"Unsupported reduction='{reduction}'. Expected one of: 'mean', 'sum'."
            )
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Berechnet den Focal Loss.

        Args:
            logits:  Rohe Modellausgaben, Shape (B, C) — kein Softmax.
            targets: Ganzzahlige Klassenindizes, Shape (B,).

        Returns:
            Skalarer Loss-Wert (wenn reduction="mean").
        """
        # Schritt 1: Standard Cross-Entropy pro Sample (nicht reduziert)
        # cross_entropy gibt -log(p_t) zurück — also CE ohne Aggregation.
        # weight=self.alpha wendet α_t pro Sample an (α der wahren Klasse).
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")

        # Schritt 2: p_t berechnen — Wahrscheinlichkeit der korrekten Klasse
        # exp(-ce) kehrt -log(p_t) um → p_t = e^(-CE)
        pt = torch.exp(-ce)

        # Schritt 3: Fokussierungsterm (1 - p_t)^γ anwenden
        # Einfache Samples (pt ≈ 1) → Term ≈ 0  → Loss wird stark gedämpft
        # Schwere Samples  (pt ≈ 0) → Term ≈ 1  → Loss bleibt fast unverändert
        focal_loss = (1.0 - pt) ** self.gamma * ce

        # Schritt 4: Aggregieren
        if self.reduction == "mean":
            return focal_loss.mean()
        return focal_loss.sum()


# ------------------------------------------------------------------
# Hilfsfunktion: α aus Klassenfrequenzen berechnen
# ------------------------------------------------------------------

def compute_class_weights(class_counts: list[int]) -> torch.Tensor:
    """
    Berechnet inverse Klassenfrequenzen als α-Gewichte für Focal Loss.

    Formel: α_c = N_gesamt / (C · N_c)
      N_gesamt : Gesamtzahl aller Samples
      C        : Anzahl der Klassen
      N_c      : Anzahl der Samples in Klasse c

    Ergebnis ist normiert, sodass der Durchschnitt der Gewichte = 1.

    Beispiel: [100, 200, 150, 550] → seltene Klassen bekommen höhere Gewichte.

    Args:
        class_counts: Liste mit Anzahl Samples pro Klasse (Index = Label).

    Returns:
        1D-FloatTensor der Länge len(class_counts).
    """
    if any(count <= 0 for count in class_counts):
        raise ValueError("class_counts must contain only positive values.")

    counts = torch.tensor(class_counts, dtype=torch.float)
    n_total = counts.sum()
    n_classes = len(class_counts)
    # Inverse Frequenz, normiert auf Mittelwert = 1
    weights = n_total / (n_classes * counts)
    return weights / weights.mean()


# ------------------------------------------------------------------
# Schnelltest — python src/losses.py
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    torch.manual_seed(42)

    # Synthetischer Batch: 8 Samples, 4 Klassen
    logits  = torch.randn(8, 4)
    targets = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])

    # 1. Focal Loss ohne α (γ=0 muss identisch zu CE sein)
    fl_gamma0 = FocalLoss(gamma=0.0)
    ce_ref    = F.cross_entropy(logits, targets)
    fl_val    = fl_gamma0(logits, targets)
    print(f"γ=0  → FL={fl_val:.4f}, CE={ce_ref:.4f} (sollten gleich sein)")
    assert torch.allclose(fl_val, ce_ref, atol=1e-5), "γ=0 stimmt nicht mit CE überein!"

    # 2. Focal Loss mit γ=2 muss kleiner sein als CE
    # (einfache Samples werden gedämpft → insgesamt kleinerer Loss)
    fl_gamma2 = FocalLoss(gamma=2.0)
    fl_val2   = fl_gamma2(logits, targets)
    print(f"γ=2  → FL={fl_val2:.4f} (erwartet: < CE)")
    assert fl_val2 < ce_ref, "Focal Loss mit γ=2 sollte kleiner als CE sein!"

    # 3. Klassengewichte berechnen
    class_counts = [1200, 3800, 2100, 17307]  # aus RIAWELC-Exploration
    weights = compute_class_weights(class_counts)
    print(f"\nKlassengewichte (CR, LP, PO, ND):")
    for name, w in zip(["CR", "LP", "PO", "ND"], weights):
        print(f"  {name}: {w:.3f}")
    print(f"  Mittelwert: {weights.mean():.3f} (sollte ≈ 1.0 sein)")

    # 4. Focal Loss mit α
    fl_alpha = FocalLoss(alpha=weights, gamma=2.0)
    fl_val3  = fl_alpha(logits, targets)
    print(f"\nFL mit α und γ=2: {fl_val3:.4f}")

    print("\nAlle Tests bestanden.")
    sys.exit(0)

"""
Datenexploration für den RIAWELC-Datensatz.

Dieses Skript analysiert die Bilder bevor wir mit dem Training anfangen.
Ziel: Klassenverteilung, visuelle Beispiele und Pixelstatistiken verstehen.
"""

from pathlib import Path
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

# --- Pfade und Klassen-Mapping -------------------------------------------

DATA_ROOT = Path(__file__).parent.parent / "data" / "RIAWELC" / "images" / "DB - Copy"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Ordnernamen im Datensatz → lesbare Klassen-Namen
# (Difetto3 existiert nicht im Datensatz)
CLASS_MAP: dict[str, str] = {
    "Difetto1": "CR (Cracks)",
    "Difetto2": "LP (Lack of Penetration)",
    "Difetto4": "PO (Porosity)",
    "NoDifetto": "ND (No Defect)",
}

SPLITS = ["training", "validation", "testing"]

# -------------------------------------------------------------------------


def load_dataset_index() -> dict[str, dict[str, list[Path]]]:
    """
    Liest alle Bildpfade aus dem Dateisystem und gibt sie strukturiert zurück.

    Returns:
        dict mit Struktur: {split: {klassenname: [pfad1, pfad2, ...]}}
        Beispiel: {"training": {"CR (Cracks)": [Path(...), ...]}, ...}

    Warum diese Struktur? Wir brauchen die Pfade getrennt nach Split und Klasse,
    damit wir sowohl die Gesamtverteilung als auch Split-spezifische Verteilungen
    analysieren können.
    """
    index: dict[str, dict[str, list[Path]]] = {}

    for split in SPLITS:
        index[split] = {}
        for folder_name, class_name in CLASS_MAP.items():
            class_dir = DATA_ROOT / split / folder_name
            paths = sorted(class_dir.glob("*.png"))
            index[split][class_name] = paths

    return index


def plot_class_distribution(index: dict[str, dict[str, list[Path]]]) -> None:
    """
    Plottet die Klassenverteilung als gruppierten Balkendiagramm (pro Split).

    Warum ist die Klassenverteilung wichtig?
    Wenn eine Klasse z.B. 70% der Daten ausmacht, tendiert das Modell dazu,
    diese Klasse bevorzugt vorherzusagen — nicht weil es gut ist, sondern weil
    es statistisch "sicher" ist. Das nennen wir Class Imbalance.
    Als Gegenmaßnahme gibt es: Focal Loss, Class Weights, oder Oversampling.

    Args:
        index: Datensatz-Index aus load_dataset_index()
    """
    class_names = list(CLASS_MAP.values())
    x = np.arange(len(class_names))
    width = 0.25  # Breite eines Balkens

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, split in enumerate(SPLITS):
        counts = [len(index[split][cls]) for cls in class_names]
        bars = ax.bar(x + i * width, counts, width, label=split.capitalize())
        # Zahlenwert über jeden Balken schreiben
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 20,
                str(count),
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xlabel("Klasse")
    ax.set_ylabel("Anzahl Bilder")
    ax.set_title("RIAWELC Klassenverteilung pro Split")
    ax.set_xticks(x + width)
    ax.set_xticklabels(class_names, rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.4)

    # Gesamtzahlen als Text im Plot
    totals = {cls: sum(len(index[s][cls]) for s in SPLITS) for cls in class_names}
    total_all = sum(totals.values())
    info = "  |  ".join(f"{cls.split()[0]}: {n} ({n/total_all:.0%})" for cls, n in totals.items())
    fig.text(0.5, 0.01, f"Gesamt: {total_all}  —  {info}", ha="center", fontsize=9, color="gray")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out_path = OUTPUT_DIR / "01_class_distribution.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Gespeichert: {out_path}")


def plot_sample_images(index: dict[str, dict[str, list[Path]]], n_per_class: int = 8) -> None:
    """
    Plottet n_per_class zufällige Beispielbilder für jede Klasse in einem 2×4-Grid.

    Warum? Bevor wir trainieren, sollten wir mit den Augen sehen, wie die Defekte
    aussehen. Das hilft uns einzuschätzen:
    - Wie schwierig ist die Unterscheidung visuell?
    - Gibt es Klassen, die sich sehr ähnlich sehen? (→ Modell wird dort mehr Fehler machen)
    - Sind die Bilder qualitativ okay (kein Rauschen, kein Artefakt)?

    Args:
        index: Datensatz-Index aus load_dataset_index()
        n_per_class: Anzahl Beispielbilder pro Klasse (Standard: 8 → 2×4-Grid)
    """
    class_names = list(CLASS_MAP.values())
    cols = 4
    rows = n_per_class // cols  # 8 Bilder → 2 Zeilen

    fig, axes = plt.subplots(
        rows * len(class_names), cols,
        figsize=(cols * 3, rows * len(class_names) * 3),
    )

    for cls_idx, cls_name in enumerate(class_names):
        # Bilder aus dem Training-Split ziehen (dort haben wir die meisten)
        all_paths = index["training"][cls_name]
        sample_paths = random.sample(all_paths, min(n_per_class, len(all_paths)))

        for img_idx, img_path in enumerate(sample_paths):
            row = cls_idx * rows + img_idx // cols
            col = img_idx % cols
            ax = axes[row, col]

            img = Image.open(img_path)
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            ax.axis("off")

            # Klassentitel nur über dem ersten Bild jeder Klasse
            if img_idx == 0:
                ax.set_title(cls_name, fontsize=11, fontweight="bold", pad=6)

    fig.suptitle("RIAWELC — 8 Beispiele pro Klasse (Training-Split)", fontsize=14, y=1.01)
    plt.tight_layout()
    out_path = OUTPUT_DIR / "02_sample_images.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Gespeichert: {out_path}")


def compute_pixel_statistics(index: dict[str, dict[str, list[Path]]]) -> tuple[float, float]:
    """
    Berechnet Mean und Std der Pixelwerte über den gesamten Training-Split.
    Plottet außerdem ein Histogramm der Pixelwertverteilung.

    Warum brauchen wir das?
    Neuronale Netze trainieren besser, wenn die Eingabedaten normalisiert sind —
    d.h. Mittelwert ≈ 0 und Standardabweichung ≈ 1. Ohne Normalisierung sind
    die Gradienten in frühen Schichten oft zu groß oder zu klein (Stichwort:
    vanishing/exploding gradients). Wir berechnen Mean und Std auf dem
    Training-Split (nicht dem ganzen Datensatz!), damit wir keine Information
    aus Val/Test ins Training "leaken".

    Die berechneten Werte (mean, std) kommen später als Parameter in
    torchvision.transforms.Normalize(mean=[...], std=[...]).

    Args:
        index: Datensatz-Index aus load_dataset_index()

    Returns:
        (mean, std) als float, skaliert auf [0, 1]
    """
    print("  Berechne Pixelstatistiken über Training-Split (das dauert kurz)...")

    all_paths: list[Path] = []
    for cls_paths in index["training"].values():
        all_paths.extend(cls_paths)

    # Für den Mittelwert: einmal über alle Bilder iterieren
    # Wir arbeiten mit float32 und normalisieren auf [0, 1]
    pixel_sum = 0.0
    pixel_sq_sum = 0.0
    n_pixels = 0

    # Histogramm-Akkumulator (256 Bins für 8-bit Bilder)
    histogram = np.zeros(256, dtype=np.int64)

    for path in all_paths:
        arr = np.array(Image.open(path), dtype=np.float32) / 255.0  # → [0, 1]
        pixel_sum += arr.sum()
        pixel_sq_sum += (arr ** 2).sum()
        n_pixels += arr.size

        # Für das Histogramm: Rohwerte [0, 255]
        raw = np.array(Image.open(path))
        hist, _ = np.histogram(raw, bins=256, range=(0, 255))
        histogram += hist

    mean = pixel_sum / n_pixels
    # Varianz = E[X²] - E[X]² (Verschiebungssatz)
    std = np.sqrt(pixel_sq_sum / n_pixels - mean ** 2)

    print(f"  Mean = {mean:.4f}  |  Std = {std:.4f}  (skaliert auf [0,1])")

    # Histogramm plotten
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(np.arange(256), histogram / histogram.sum(), width=1, color="steelblue", alpha=0.8)
    ax.axvline(mean * 255, color="red", linestyle="--", label=f"Mean = {mean*255:.1f}")
    ax.axvline((mean - std) * 255, color="orange", linestyle=":", alpha=0.7)
    ax.axvline((mean + std) * 255, color="orange", linestyle=":", alpha=0.7, label=f"±1 Std = {std*255:.1f}")
    ax.set_xlabel("Pixelwert (0–255)")
    ax.set_ylabel("Relative Häufigkeit")
    ax.set_title("Pixelwertverteilung — RIAWELC Training-Split")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    out_path = OUTPUT_DIR / "03_pixel_distribution.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Gespeichert: {out_path}")

    return float(mean), float(std)


# -------------------------------------------------------------------------

if __name__ == "__main__":
    print("Lade Datensatz-Index...")
    index = load_dataset_index()

    # Kurze Zusammenfassung in der Konsole
    total = sum(len(index[s][cls]) for s in SPLITS for cls in CLASS_MAP.values())
    print(f"Datensatz geladen: {total} Bilder")
    for split in SPLITS:
        n = sum(len(index[split][cls]) for cls in CLASS_MAP.values())
        print(f"  {split:12s}: {n} Bilder")

    print("\n[1/3] Klassenverteilung plotten...")
    plot_class_distribution(index)

    print("[2/3] Beispielbilder plotten...")
    random.seed(42)  # reproduzierbar — gleiche Bilder bei jedem Lauf
    plot_sample_images(index)

    print("[3/3] Pixelstatistiken berechnen...")
    mean, std = compute_pixel_statistics(index)

    print(f"""
=== Ergebnisse für dataset.py ===
Normalisierungs-Parameter (Training-Split, skaliert auf [0,1]):
  mean = [{mean:.4f}]
  std  = [{std:.4f}]
Diese Werte in transforms.Normalize(mean=[{mean:.4f}], std=[{std:.4f}]) eintragen.
""")

    print("Fertig. Plots gespeichert in outputs/plots/")


# =============================================================================
# ERKENNTNISSE AUS DER DATENEXPLORATION (Schritt 1.2)
# =============================================================================
#
# KLASSENVERTEILUNG:
#   CR (Cracks):             7.635 Bilder  (31%) — größte Klasse
#   LP (Lack of Penetration):6.320 Bilder  (26%)
#   ND (No Defect):          6.000 Bilder  (25%)
#   PO (Porosity):           4.452 Bilder  (18%) — kleinste Klasse
#
#   → Mild unbalanciert (CR ist 1.7× häufiger als PO), KEIN dramatisches
#     Ungleichgewicht. NoDifetto ist entgegen der Erwartung NICHT dominant —
#     der Datensatz wurde vermutlich absichtlich ausgeglichen.
#   → Focal Loss bleibt sinnvoll (PO ist auch visuell schwieriger), aber
#     Class-Weights spielen eine kleinere Rolle als erwartet.
#
# VISUELLE UNTERSCHIEDE DER KLASSEN:
#   CR (Cracks): Scharfe, dunkle Linien quer durch die Schweißnaht.
#                Gut erkennbar, wenn ausgeprägt — aber feine Risse können
#                subtil sein.
#   LP (Lack of Penetration): Längliche, helle oder dunkle Bereiche an der
#                Nahtwurzel. Kann CR ähneln → erwartete Verwechslungsquelle.
#   PO (Porosity): Runde, dunkle Punkte (Gaseinschlüsse). Oft mehrere auf
#                einmal. Visuell am distinktivsten, aber klein.
#   ND (No Defect): Gleichmäßige Textur, keine deutlichen Merkmale.
#                Herausforderung: schwache Defekte in CR/LP/PO könnten wie
#                ND aussehen.
#
#   → Kritische Verwechslungspaare: CR↔LP (beide lineare Strukturen),
#     schwache PO↔ND (kleine Poren vs. gleichmäßige Textur).
#
# PIXELSTATISTIKEN (Training-Split, skaliert auf [0,1]):
#   mean = 0.6023  (entspricht Pixelwert ~154 von 255)
#   std  = 0.1951
#
#   → Bilder sind generell hell (Röntgenstrahlen werden vom Metall absorbiert,
#     d.h. gesundes Metall = helle Pixel; Defekte = dunklere Bereiche).
#   → Diese Werte in transforms.Normalize(mean=[0.6023], std=[0.1951])
#     verwenden. Damit werden Pixelwerte auf ~N(0,1) normalisiert.
#
# PREDEFINED SPLIT (von den Dataset-Autoren):
#   Training: 15.863 (65%) | Validation: 6.101 (25%) | Test: 2.443 (10%)
#   → Ungewöhnlich viele Validation-Daten (25% statt üblicher 10-15%).
#   → Wir verwenden den vordefinierten Split (nicht sklearn), um Vergleich
#     mit der Originalpublikation zu ermöglichen.
#
# IMPLIKATIONEN FÜR DAS TRAINING:
#   - Focal Loss: γ=2, α als inverse Klassenfrequenz (PO bekommt höheres Gewicht)
#   - Augmentation: Rotation/Flip sinnvoll (kein natürliches "oben" bei Patches)
#   - Eval-Fokus: Recall pro Klasse, besonders für PO (seltene, schwierige Klasse)
# =============================================================================

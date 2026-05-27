"""
Custom PyTorch Dataset für den RIAWELC-Datensatz.

Verantwortlichkeiten dieser Datei:
  - Bildpfade und Labels aus dem Dateisystem einlesen
  - Augmentations-Pipelines (Training vs. Val/Test) definieren
  - Einzelne Samples (Tensor, Label) auf Anfrage des DataLoaders liefern

Was hier NICHT passiert:
  - Batching (das macht der DataLoader)
  - Modell-Logik, Loss-Berechnung, Training
"""

import os
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# ---------------------------------------------------------------------------
# Konstanten
# ---------------------------------------------------------------------------

# Gleiche Pfad-Logik wie in der Datenexploration — DATA_ROOT zeigt auf den
# Ordner, der die Unterordner "training", "validation", "testing" enthält.
DATA_ROOT = Path(
    os.environ.get("RIAWELC_DATA_ROOT")
    or Path(__file__).parent.parent / "data" / "RIAWELC" / "images" / "DB - Copy"
)

# Ordnernamen im Dateisystem → Integer-Label (0-3) für PyTorch
# Reihenfolge ist alphabetisch, damit sie stabil und reproduzierbar ist.
CLASS_MAP: dict[str, int] = {
    "Difetto1": 0,   # CR — Cracks / Risse (unregelmäßige dunkle Linien)
    "Difetto2": 1,   # PO — Porosity / Poren (runde dunkle Flecken)
    "Difetto4": 2,   # LP — Lack of Penetration (horizontaler dunkler Streifen)
    "NoDifetto": 3,  # ND — No Defect
}

# Klartext-Namen in der gleichen Reihenfolge (Index = Label-Integer)
# Nützlich für Plots, Confusion Matrix, W&B-Logs.
CLASS_NAMES: list[str] = ["CR", "PO", "LP", "ND"]

# Normalisierungsparameter aus der Datenexploration (Training-Split, [0,1])
# mean=0.6023, std=0.1951 — ermittelt in notebooks/01_data_exploration.py
PIXEL_MEAN: float = 0.6023
PIXEL_STD: float = 0.1951

# Gültige Split-Namen (entsprechen Ordnernamen im Dateisystem)
VALID_SPLITS = ("training", "validation", "testing")


# ---------------------------------------------------------------------------
# Augmentation Pipelines
# ---------------------------------------------------------------------------

def get_train_transforms() -> Callable:
    """
    Augmentation-Pipeline für den Training-Split.

    Warum Augmentation?
    Das Modell soll lernen, Defekte zu erkennen — egal ob das Bild leicht
    rotiert, gespiegelt oder minimal verändert ist. Augmentation erzeugt
    künstliche Varianten jedes Bildes und macht das Modell robuster.

    Welche Augmentationen wählen wir und warum?

    - RandomHorizontalFlip / RandomVerticalFlip:
        Schweißnaht-Patches haben kein natürliches "oben" oder "links".
        Ein Riss sieht von links nach rechts genauso aus wie umgekehrt.
        → Verdoppelt/vervierfacht die effektive Datenmenge kostenlos.

    - RandomRotation(15°):
        Kleine Rotationen simulieren leichte Kamerawinkel-Abweichungen.
        Wir wählen ±15°, nicht 90° — zu starke Rotation könnte lineare
        Defekte (CR, LP) unkenntlich machen.

    - ColorJitter (Brightness, Contrast):
        Leichte Helligkeits-/Kontrastveränderungen simulieren unterschied-
        liche Röntgenbelichtungen und Scanner-Kalibrierungen.

    - Normalize:
        Zentriert die Pixelwerte auf ~N(0,1) mit unseren berechneten
        Werten. Das stabilisiert den Gradient-Fluss durch das Netz.

    Returns:
        torchvision.transforms.Compose — anwendbar auf ein PIL-Image.
    """
    return transforms.Compose([
        # Bilder sind 227×227 — auf 224×224 bringen (ResNet-Standard)
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),          # PIL Image → FloatTensor [0,1], Shape: (1, 224, 224)
        transforms.Normalize(mean=[PIXEL_MEAN], std=[PIXEL_STD]),
    ])


def get_val_transforms() -> Callable:
    """
    Transformation für Validation- und Test-Split (keine Augmentation).

    Warum keine Augmentation bei Val/Test?
    Die Validierung misst, wie gut das Modell auf echten, unveränderten
    Bildern ist. Wenn wir Val-Bilder augmentieren, messen wir die Performance
    auf künstlichen Varianten — das verfälscht die Metrik.
    Nur die Normalisierung bleibt (die ist kein Augmentation, sondern
    eine fixe Vorverarbeitung, die immer angewendet werden muss).

    Returns:
        torchvision.transforms.Compose — anwendbar auf ein PIL-Image.
    """
    return transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[PIXEL_MEAN], std=[PIXEL_STD]),
    ])


# ---------------------------------------------------------------------------
# Dataset-Klasse
# ---------------------------------------------------------------------------

class WeldDefectDataset(Dataset):
    """
    PyTorch Dataset für den RIAWELC Schweißnaht-Datensatz.

    Liest beim Initialisieren alle (Pfad, Label)-Paare ein,
    öffnet Bilder aber erst auf Anfrage des DataLoaders (__getitem__).

    Args:
        split:     Welcher Split — "training", "validation" oder "testing".
                   Wird ignoriert wenn `samples` übergeben wird.
        transform: Eine torchvision.transforms.Compose-Pipeline.
                   Wenn None: nur ToTensor() ohne Normalisierung (für Tests).
        data_root: Pfad zum Datensatz-Wurzelverzeichnis. Standard: DATA_ROOT.
        samples:   Vorberechnete Liste von (Path, label_int) Tupeln.
                   Wenn übergeben, wird split/data_root ignoriert (z.B. build_clean_splits()).
    """

    def __init__(
        self,
        split: str | None = None,
        transform: Callable | None = None,
        data_root: Path = DATA_ROOT,
        samples: list[tuple[Path, int]] | None = None,
    ) -> None:
        self.data_root = data_root
        self.transform = transform if transform is not None else transforms.ToTensor()

        if samples is not None:
            # Vorberechnete Sample-Liste direkt verwenden (z.B. build_clean_splits())
            self.split = "custom"
            self.samples = samples
        else:
            if split not in VALID_SPLITS:
                raise ValueError(f"split muss einer von {VALID_SPLITS} sein, nicht '{split}'")
            self.split = split
            # Alle (pfad, label) Paare einlesen — Bilder werden noch NICHT geöffnet
            self.samples: list[tuple[Path, int]] = self._load_samples()

    def _load_samples(self) -> list[tuple[Path, int]]:
        """
        Scannt das Dateisystem und gibt eine Liste von (Pfad, Label) zurück.

        Warum eine separate Methode statt direkt in __init__?
        Sauberere Trennung: __init__ koordiniert, _load_samples macht die
        Dateiarbeit. Leichter zu testen und zu überschreiben.

        Returns:
            Liste von (Path, int) Tupeln, sortiert für Reproduzierbarkeit.
        """
        samples: list[tuple[Path, int]] = []

        for folder_name, label in CLASS_MAP.items():
            class_dir = self.data_root / self.split / folder_name
            if not class_dir.exists():
                raise FileNotFoundError(
                    f"Klassen-Ordner nicht gefunden: {class_dir}\n"
                    f"Ist der Datensatz unter {self.data_root} entpackt?"
                )
            # sorted() → stabile Reihenfolge unabhängig vom Dateisystem
            for img_path in sorted(class_dir.glob("*.png")):
                samples.append((img_path, label))

        return samples

    def __len__(self) -> int:
        """Gibt die Anzahl der Samples in diesem Split zurück."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """
        Lädt ein einzelnes Sample und gibt (Tensor, Label) zurück.

        Dieser Methode wird vom DataLoader aufgerufen — für jeden Index
        in einem Batch. Sie wird deshalb sehr oft aufgerufen und sollte
        schnell sein (kein unnötiger Overhead).

        Args:
            idx: Index des gewünschten Samples (0 bis len-1).

        Returns:
            Tuple aus:
            - tensor: FloatTensor der Form (1, 224, 224), normalisiert
            - label:  Integer 0-3 (CR=0, PO=1, LP=2, ND=3)
        """
        img_path, label = self.samples[idx]

        # Bild als Graustufenbild öffnen (L = Luminance = 1 Kanal)
        # PIL öffnet lazy — erst .load() oder eine Operation lädt Pixel.
        # Über den Context-Manager wird das File-Handle deterministisch
        # geschlossen, was insbesondere mit DataLoader-Multiprocessing
        # offene Handles vermeidet.
        with Image.open(img_path) as img:
            img = img.convert("L")

        # Transform anwenden: PIL Image → FloatTensor (1, 224, 224)
        tensor = self.transform(img)

        return tensor, label

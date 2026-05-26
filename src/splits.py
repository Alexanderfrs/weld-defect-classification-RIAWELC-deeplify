"""
Split-Hilfsfunktionen für RIAWELC.

Zwei Funktionen:

1. build_clean_splits()  ← wird für das Training verwendet
   Problem: Der originale training/-Ordner enthält 2.443 Patches, die
   byte-identisch auch im testing/-Ordner vorkommen. Ein Modell, das auf
   training/ trainiert und auf testing/ evaluiert wird, sieht also
   Testbilder bereits im Training — 100 % Accuracy ist die Folge.
   Lösung: Beim Laden des Training-Sets werden diese Duplikate herausgefiltert.
   Ergebnis: 13.420 Trainings-Patches (statt 15.863), 2.443 echte Test-Patches.

2. build_weld_level_splits()  ← experimentell, für spätere Analyse
   Teilt alle Patches nach Weld-ID auf, damit kein Schweißstück in mehr
   als einem Split vorkommt.
"""

from pathlib import Path

from .dataset import CLASS_MAP, CLASS_NAMES, DATA_ROOT


# ---------------------------------------------------------------------------
# Clean Split (Option B fix)
# ---------------------------------------------------------------------------

def build_clean_splits(
    data_root: Path = DATA_ROOT,
) -> tuple[list, list, list]:
    """
    Gibt saubere Train/Val/Test-Listen zurück, bei denen die 2.443 Duplikate
    aus dem Training-Set entfernt wurden.

    Das originale training/-Verzeichnis enthält 2.443 Patches, die
    byte-identisch auch in testing/ liegen. Diese Funktion filtert sie heraus,
    sodass train und test disjunkt sind.

    Splits nach dem Fix:
      Training:   13.420 Patches  (originale training/ minus Duplikate)
      Validation:  6.101 Patches  (originale validation/, unverändert)
      Testing:     2.443 Patches  (originale testing/, jetzt genuiner Hold-out)

    Returns:
        Tuple (train_samples, val_samples, test_samples).
        Jedes Element ist eine Liste von (Path, label_int) Tupeln.
    """
    # Dateinamen des Test-Sets einsammeln — dienen als Filter
    test_fnames: set[tuple[str, str]] = set()
    for folder in CLASS_MAP:
        class_dir = data_root / "testing" / folder
        if class_dir.exists():
            for p in class_dir.glob("*.png"):
                test_fnames.add((folder, p.name))

    # Training: Patches die NICHT im Test-Set sind
    train_samples: list[tuple[Path, int]] = []
    for folder, label in CLASS_MAP.items():
        class_dir = data_root / "training" / folder
        if not class_dir.exists():
            raise FileNotFoundError(f"Ordner nicht gefunden: {class_dir}")
        for p in sorted(class_dir.glob("*.png")):
            if (folder, p.name) not in test_fnames:
                train_samples.append((p, label))

    # Validation: unverändert
    val_samples: list[tuple[Path, int]] = []
    for folder, label in CLASS_MAP.items():
        class_dir = data_root / "validation" / folder
        if not class_dir.exists():
            raise FileNotFoundError(f"Ordner nicht gefunden: {class_dir}")
        for p in sorted(class_dir.glob("*.png")):
            val_samples.append((p, label))

    # Testing: unverändert (jetzt genuiner Hold-out)
    test_samples: list[tuple[Path, int]] = []
    for folder, label in CLASS_MAP.items():
        class_dir = data_root / "testing" / folder
        if not class_dir.exists():
            raise FileNotFoundError(f"Ordner nicht gefunden: {class_dir}")
        for p in sorted(class_dir.glob("*.png")):
            test_samples.append((p, label))

    # Sicherheits-Check: kein Patch in Training UND Test
    train_fnames = {(p.parent.name, p.name) for p, _ in train_samples}
    overlap = len(train_fnames & test_fnames)
    assert overlap == 0, f"Noch {overlap} Duplikate zwischen Train und Test!"

    _print_split_summary(train_samples, val_samples, test_samples)
    return train_samples, val_samples, test_samples


def _print_split_summary(
    train_samples: list, val_samples: list, test_samples: list
) -> None:
    print("\n=== Clean Split ===")
    for name, samples in [("Training", train_samples), ("Validation", val_samples), ("Testing", test_samples)]:
        counts = [sum(1 for _, l in samples if l == i) for i in range(len(CLASS_NAMES))]
        class_str = ", ".join(f"{n}={c}" for n, c in zip(CLASS_NAMES, counts))
        print(f"  {name:10}: {len(samples):>6} Patches  [{class_str}]")



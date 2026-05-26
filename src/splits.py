"""
Weld-Level Train/Val/Test Split für RIAWELC.

Problem mit den vordefinierten Splits:
  Die originalen training/validation/testing Ordner wurden patch-weise aufgeteilt.
  Alle 29 physischen Schweißstücke (Weld-IDs) erscheinen in allen drei Ordnern.
  Das Modell hat beim Training also Patches von denselben Stücken gesehen,
  die später auch im Test-Set auftauchen → das ist kein fairer Generalisierungstest.

Dieser Fix:
  Wir lesen alle 24.407 Patches aus allen drei Ordnern zusammen,
  gruppieren nach Weld-ID, und teilen auf Weld-Ebene auf.
  Kein Weld-ID erscheint in mehr als einem Split.
  Das misst die echte Generalisierung auf ungesehene Schweißstücke.
"""

import random
from pathlib import Path

from .dataset import CLASS_MAP, DATA_ROOT


def parse_weld_id(filename: str) -> str:
    """
    Extrahiert die Weld-ID aus einem Dateinamen.

    Beispiel: 'RRT-40R_Img1_A80_S1_[3][36].png' → 'RRT-40R'
              'bam5_Img2_A80_S4_[1][23].png'     → 'bam5'

    Das Trennzeichen '_Img' trennt die Weld-ID vom Rest des Namens.
    """
    return filename.split("_Img")[0]


def build_weld_level_splits(
    data_root: Path = DATA_ROOT,
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    seed: int = 42,
) -> tuple[list, list, list]:
    """
    Liest alle Patches aus allen drei originalen Split-Ordnern zusammen,
    gruppiert nach Weld-ID, und teilt auf Weld-Ebene auf.

    Args:
        data_root:   Pfad zum Datensatz-Wurzelverzeichnis (enthält training/validation/testing).
        train_ratio: Anteil der Weld-IDs für Training (default: 70%).
        val_ratio:   Anteil der Weld-IDs für Validation (default: 15%).
                     Test-Anteil = 1 - train_ratio - val_ratio (default: 15%).
        seed:        Zufallsgenerator-Seed für Reproduzierbarkeit.

    Returns:
        Tuple (train_samples, val_samples, test_samples).
        Jedes Element ist eine Liste von (Path, label_int) Tupeln.
        Kein Weld-ID erscheint in mehr als einem Split.
    """
    # --- 1. Alle Patches einsammeln ---
    all_samples: list[tuple[Path, int, str]] = []  # (path, label, weld_id)

    for split_folder in ("training", "validation", "testing"):
        for folder_name, label in CLASS_MAP.items():
            class_dir = data_root / split_folder / folder_name
            if not class_dir.exists():
                raise FileNotFoundError(f"Ordner nicht gefunden: {class_dir}")
            for img_path in sorted(class_dir.glob("*.png")):
                weld_id = parse_weld_id(img_path.stem)
                all_samples.append((img_path, label, weld_id))

    # --- 2. Weld-IDs pro Klasse gruppieren ---
    # Ziel: jede Klasse soll in jedem Split vertreten sein (stratifiziert).
    # Dazu sammeln wir pro Klasse, welche Weld-IDs dort vorkommen.
    from collections import defaultdict
    weld_label: dict[str, int] = {}  # weld_id → dominante Klasse
    weld_class_counts: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    for _, label, weld_id in all_samples:
        weld_class_counts[weld_id][label] += 1

    # Dominante Klasse = die mit den meisten Patches für diesen Weld
    for weld_id, counts in weld_class_counts.items():
        weld_label[weld_id] = max(counts, key=counts.get)

    # --- 3. Weld-IDs stratifiziert aufteilen ---
    # Stratifizierung: Weld-IDs nach Klasse gruppieren, dann proportional ziehen,
    # damit alle Klassen in allen Splits vertreten sind.
    rng = random.Random(seed)

    class_welds: dict[int, list[str]] = defaultdict(list)
    for weld_id, label in weld_label.items():
        class_welds[label].append(weld_id)

    train_welds, val_welds, test_welds = [], [], []

    for label in sorted(class_welds.keys()):
        welds = sorted(class_welds[label])
        rng.shuffle(welds)
        n = len(welds)
        n_train = max(1, round(n * train_ratio))
        n_val   = max(1, round(n * val_ratio))
        # Rest geht in Test — mindestens 1
        n_test  = max(1, n - n_train - n_val)
        # Korrektur wenn Summe nicht aufgeht
        if n_train + n_val + n_test > n:
            n_test = max(1, n - n_train - n_val)
            if n_train + n_val + n_test > n:
                n_val = max(1, n - n_train - 1)

        train_welds.extend(welds[:n_train])
        val_welds.extend(welds[n_train:n_train + n_val])
        test_welds.extend(welds[n_train + n_val:n_train + n_val + n_test])

    train_set = set(train_welds)
    val_set   = set(val_welds)
    test_set  = set(test_welds)

    # Sicherheits-Check: keine Überschneidungen
    assert not (train_set & val_set),  "Überschneidung Train/Val!"
    assert not (train_set & test_set), "Überschneidung Train/Test!"
    assert not (val_set   & test_set), "Überschneidung Val/Test!"

    # --- 4. Patches den Splits zuordnen ---
    train_samples = [(p, l) for p, l, w in all_samples if w in train_set]
    val_samples   = [(p, l) for p, l, w in all_samples if w in val_set]
    test_samples  = [(p, l) for p, l, w in all_samples if w in test_set]

    # --- 5. Zusammenfassung ausgeben ---
    print("\n=== Weld-Level Split ===")
    print(f"Gesamt:     {len(all_samples):>6} Patches, {len(weld_label):>2} Weld-IDs")
    print(f"Training:   {len(train_samples):>6} Patches, {len(train_welds):>2} Weld-IDs: {sorted(train_welds)}")
    print(f"Validation: {len(val_samples):>6} Patches, {len(val_welds):>2} Weld-IDs: {sorted(val_welds)}")
    print(f"Test:       {len(test_samples):>6} Patches, {len(test_welds):>2} Weld-IDs: {sorted(test_welds)}")
    print()

    from .dataset import CLASS_NAMES
    for split_name, samples in [("Train", train_samples), ("Val", val_samples), ("Test", test_samples)]:
        counts = [sum(1 for _, l in samples if l == i) for i in range(len(CLASS_NAMES))]
        class_str = ", ".join(f"{n}={c}" for n, c in zip(CLASS_NAMES, counts))
        print(f"  {split_name} Klassenverteilung: {class_str}")

    return train_samples, val_samples, test_samples

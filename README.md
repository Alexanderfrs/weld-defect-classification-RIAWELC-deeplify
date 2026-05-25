# Weld Defect Classification on RIAWELC

Klassifikation von Schweißnaht-Defekten in Röntgenbildern mit ResNet50 und Transfer Learning.

---

## Datensatz

**RIAWELC** — Radiographic Images for Automatic Weld defects CLassification  
Quelle: [github.com/stefyste/RIAWELC](https://github.com/stefyste/RIAWELC)  
Lizenz: Frei verfügbar (für akademische Nutzung)

| Eigenschaft | Wert |
|---|---|
| Bilder gesamt | 24.407 |
| Auflösung | 224 × 224 px, 8-bit Grayscale PNG |
| Klassen | 4 (CR, LP, PO, ND) |
| Split | Training 65% / Validation 25% / Test 10% (vordefiniert) |

**Klassen:**
| Kürzel | Name | Beschreibung | Bilder |
|---|---|---|---|
| CR | Cracks / Risse | Scharfe, lineare Helligkeitsunterschiede | 7.635 (31%) |
| LP | Lack of Penetration | Unvollständige Durchschweißung, an der Nahtwurzel | 6.320 (26%) |
| ND | No Defect | Defektfreie Schweißnaht | 6.000 (25%) |
| PO | Porosity / Poren | Runde, dunkle Gaseinschlüsse | 4.452 (18%) |

**Wichtige Erkenntnisse aus der Exploration:**
- Mild unbalanciert (CR 1.7× häufiger als PO), kein dramatisches Ungleichgewicht
- Pixelstatistiken (Training): mean = 0.602, std = 0.195 — Bilder sind generell hell
- Kritische Verwechslungspaare: CR↔LP (beide lineare Strukturen)

---

## Methode

*(wird nach Abschluss des Trainings ergänzt)*

---

## Ergebnisse

*(wird nach Evaluation ergänzt)*

---

## Setup & Reproduktion

**Voraussetzungen:** `uv`, NVIDIA GPU mit CUDA-Support

```bash
# Repository klonen
git clone <repo-url>
cd ndt-defect-classification

# Python 3.12 Umgebung erstellen und Dependencies installieren
uv venv --python 3.12 .venv
uv sync

# Datensatz herunterladen (24.407 Bilder, ~1.5 GB nach Extraktion)
git clone https://github.com/stefyste/RIAWELC data/RIAWELC
cd data/RIAWELC/Dataset_partitioned
unrar x RIAWELC_dataset.part01.rar ../images/
```

**Datenexploration:**
```bash
.venv/bin/python notebooks/01_data_exploration.py
# Plots werden in outputs/plots/ gespeichert
```

*(Training, Evaluation und Grad-CAM-Anweisungen werden ergänzt)*

---

## Limitierungen

- Klassifikation auf vorgeschnittenen **Patches**, keine Lokalisierung auf Vollbildern
- Nur Radiografie (RT) — kein PAUT, UT oder andere NDT-Verfahren
- Datensatz aus einer einzigen Quelle — Generalisierung auf andere Anlagen unbekannt
- Kleiner Datensatz für industriellen Einsatz (~24k Bilder)

---

## Referenzen

[1] Totino, Spagnolo, Perri. *RIAWELC: A Novel Dataset of Radiographic Images for Automatic Weld Defects Classification.* ICMECE 2022.

[2] Perri, Spagnolo, Frustaci, Corsonello. *Welding Defects Classification Through a Convolutional Neural Network.* Manufacturing Letters, Elsevier.

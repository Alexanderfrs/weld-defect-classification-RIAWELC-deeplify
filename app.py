"""
Gradio Demo — RIAWELC Weld Defect Classifier

Lädt ein Graustufenbild eines Schweißnaht-Patches hoch und erhält:
  - Vorhergesagte Klasse + Konfidenz
  - Klassenwahrscheinlichkeiten als Balkendiagramm
  - Grad-CAM Overlay (wohin schaut das Modell?)
  - Hinweis ob manuelle Prüfung empfohlen wird

Modellgewichte werden beim Start von HuggingFace Hub geladen.
"""

import random
from pathlib import Path

import gradio as gr
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from src.dataset import CLASS_NAMES
from src.inference import REVIEW_THRESHOLD, load_model, predict

# ---------------------------------------------------------------------------
# Modell einmal beim App-Start laden (nicht bei jeder Anfrage)
# ---------------------------------------------------------------------------

MODEL_REPO = "Alexfrs/riawelc-weld-defect-resnet50"
DEVICE     = torch.device("cpu")   # HF Spaces Free Tier: CPU

weights_path = Path(
    hf_hub_download(repo_id=MODEL_REPO, filename="model_weights.pth")
)
model = load_model(weights_path, DEVICE)
print(f"Modell geladen von {MODEL_REPO}")

_EXAMPLE_POOL = sorted(Path("examples").glob("*.png"))


def load_random_example() -> Image.Image | None:
    if not _EXAMPLE_POOL:
        return None
    with Image.open(random.choice(_EXAMPLE_POOL)) as img:
        return img.copy()


# ---------------------------------------------------------------------------
# Inferenz-Funktion (wird von Gradio aufgerufen)
# ---------------------------------------------------------------------------

def run_prediction(image: Image.Image):
    """
    Gradio-Callback: nimmt ein PIL-Image, gibt 3 Outputs zurück.

    Returns:
        label_dict:    dict {class: probability} für gr.Label
        overlay_image: PIL-Image für gr.Image
        status_text:   Markdown-String für gr.Markdown
    """
    if image is None:
        return None, None, "Bitte ein Bild hochladen."

    result = predict(image, model, DEVICE)

    # Output 1: Wahrscheinlichkeiten als Balkendiagramm
    label_dict = {
        f"{name}": float(prob)
        for name, prob in result["class_probabilities"].items()
    }

    # Output 2: Grad-CAM Overlay
    overlay = result["gradcam_overlay"]

    # Output 3: Status-Text
    cls   = result["predicted_class"]
    conf  = result["confidence"] * 100
    emoji = "⚠️" if result["needs_review"] else "✅"

    cls_descriptions = {
        "CR": "Riss (Crack)",
        "PO": "Pore (Porosity)",
        "LP": "Unvollständige Durchschweißung (Lack of Penetration)",
        "ND": "Kein Defekt (No Defect)",
    }

    if result["needs_review"]:
        status = (
            f"**{emoji} Manuelle Prüfung empfohlen**\n\n"
            f"Klasse: **{cls}** — {cls_descriptions[cls]}  \n"
            f"Konfidenz: **{conf:.1f}%** (unter {REVIEW_THRESHOLD*100:.0f}%-Schwellenwert)"
        )
    else:
        status = (
            f"**{emoji} Automatisch klassifizierbar**\n\n"
            f"Klasse: **{cls}** — {cls_descriptions[cls]}  \n"
            f"Konfidenz: **{conf:.1f}%**"
        )

    return label_dict, overlay, status


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

CLASS_DESCRIPTIONS = (
    "**CR** — Cracks / Risse: scharfe, lineare Helligkeitsunterschiede  \n"
    "**PO** — Porosity / Poren: runde, dunkle Gaseinschlüsse  \n"
    "**LP** — Lack of Penetration: unvollständige Durchschweißung an der Nahtwurzel  \n"
    "**ND** — No Defect: defektfreie Schweißnaht"
)

with gr.Blocks(title="RIAWELC Weld Defect Classifier") as demo:

    gr.Markdown(
        "# RIAWELC Weld Defect Classifier\n"
        "Lade einen 224×224-Röntgen-Patch einer Schweißnaht hoch. "
        "Das Modell (ResNet50, Transfer Learning von ImageNet) klassifiziert ihn in eine von 4 Klassen "
        "und zeigt per **Grad-CAM**, welche Bildregion die Entscheidung beeinflusst hat.\n\n"
        + CLASS_DESCRIPTIONS
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                type="pil",
                label="Röntgen-Patch hochladen",
                image_mode="L",
            )
            shuffle_btn = gr.Button("🔀 Zufälliges Beispiel laden", variant="secondary")
            run_btn = gr.Button("Klassifizieren", variant="primary")

        with gr.Column():
            output_label   = gr.Label(num_top_classes=4, label="Klassenwahrscheinlichkeiten")
            output_overlay = gr.Image(label="Grad-CAM — Modellfokus")
            output_status  = gr.Markdown()

    shuffle_btn.click(fn=load_random_example, inputs=[], outputs=input_image)
    run_btn.click(
        fn=run_prediction,
        inputs=input_image,
        outputs=[output_label, output_overlay, output_status],
    )

    # Beispielbilder (werden nach dem Upload ins Space-Repo hinzugefügt)
    gr.Examples(
        examples=[
            ["examples/cr_example.png"],
            ["examples/po_example.png"],
            ["examples/lp_example.png"],
            ["examples/nd_example.png"],
        ],
        inputs=input_image,
        label="Beispielbilder (je eine Klasse)",
    )

    gr.Markdown(
        "---\n"
        "Modell: [Alexfrs/riawelc-weld-defect-resnet50](https://huggingface.co/Alexfrs/riawelc-weld-defect-resnet50)  \n"
        "Code: [GitHub](https://github.com/Alexanderfrs/weld-defect-classification-RIAWELC-deeplify)"
    )

if __name__ == "__main__":
    demo.launch()

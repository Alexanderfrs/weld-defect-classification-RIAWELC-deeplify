"""
Phase 8.2 — Inference-Pipeline für HuggingFace Spaces

Kapselt den gesamten Weg von einem PIL-Bild zur Vorhersage:
  1. Preprocessing (identisch zum Training — wichtig!)
  2. Forward Pass → Softmax-Wahrscheinlichkeiten
  3. Grad-CAM Heatmap
  4. Overlay als PIL-Image

Warum muss das Preprocessing exakt gleich sein wie beim Training?
  Das Modell hat gelernt, auf normalisierte Eingaben zu reagieren.
  Wenn wir z.B. anders normalisieren (andere mean/std), verschieben wir
  die Pixelwerte in einen Bereich den das Modell nie gesehen hat —
  die Vorhersagen werden zufällig oder systematisch falsch.

Run (Schnelltest):
  uv run python -m src.inference
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from .dataset import CLASS_NAMES, get_val_transforms
from .model import WeldDefectResNet

# Schwellenwert unter dem wir "manuelle Prüfung empfohlen" ausgeben
REVIEW_THRESHOLD = 0.85


def load_model(weights_path: Path, device: torch.device) -> WeldDefectResNet:
    """
    Lädt die exportierten Gewichte in ein WeldDefectResNet.

    Warum pretrained=False?
    Wir laden unsere eigenen trainierten Gewichte — die ImageNet-Initialisierung
    wird sofort überschrieben, also ist pretrained=True nur unnötiger Download.

    Args:
        weights_path: Pfad zur .pth-Datei (state_dict, erzeugt von src/export.py).
        device:       CPU oder CUDA.

    Returns:
        Modell im eval()-Modus auf dem angegebenen Device.
    """
    model = WeldDefectResNet(pretrained=False)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def predict(
    image: Image.Image,
    model: WeldDefectResNet,
    device: torch.device,
) -> dict:
    """
    Vollständige Inferenz für ein einzelnes PIL-Bild.

    Schritte:
      1. Bild auf Graustufenbild konvertieren (1 Kanal, wie beim Training)
      2. Val-Transforms anwenden (Resize → ToTensor → Normalize)
      3. Forward Pass → Logits → Softmax → Wahrscheinlichkeiten
      4. Grad-CAM Heatmap berechnen
      5. Heatmap auf Originalbild überlagern

    Args:
        image:  PIL-Image beliebiger Größe und Modus.
        model:  Geladenes WeldDefectResNet im eval()-Modus.
        device: CPU oder CUDA.

    Returns:
        Dict mit:
          predicted_class     — str, z.B. "CR"
          confidence          — float in [0, 1]
          class_probabilities — dict {class_name: probability}
          gradcam_overlay     — PIL.Image (RGB, 224×224) mit Heatmap
          needs_review        — bool, True wenn confidence < REVIEW_THRESHOLD
    """
    # Schritt 1+2: Preprocessing — exakt wie get_val_transforms() beim Training
    transform = get_val_transforms()
    tensor = transform(image.convert("L"))   # (1, 224, 224)
    inp = tensor.unsqueeze(0).to(device)     # (1, 1, 224, 224)

    # Schritt 3: Forward Pass
    with torch.no_grad():
        logits = model(inp)
        probs  = torch.softmax(logits, dim=1).squeeze()  # (4,)

    probs_np  = probs.cpu().numpy()
    pred_idx  = int(probs_np.argmax())
    confidence = float(probs_np[pred_idx])

    # Schritt 4: Grad-CAM
    # enable_grad() nötig weil wir uns in einem no_grad()-Kontext befinden könnten
    cam = GradCAM(model=model, target_layers=[model.backbone.layer4[-1]])
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    with torch.enable_grad():
        grayscale_cam = cam(
            input_tensor=inp,
            targets=[ClassifierOutputTarget(pred_idx)],
        )[0]  # (H, W) in [0, 1]

    # Schritt 5: Overlay als RGB-PIL-Image
    # _to_rgb_numpy: normalisierter Tensor → [0,1] RGB-Array für show_cam_on_image
    img_np  = _to_rgb_numpy(tensor)
    overlay = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)  # uint8
    overlay_pil = Image.fromarray(overlay)

    return {
        "predicted_class":     CLASS_NAMES[pred_idx],
        "confidence":          confidence,
        "class_probabilities": {
            name: float(p) for name, p in zip(CLASS_NAMES, probs_np)
        },
        "gradcam_overlay":     overlay_pil,
        "needs_review":        confidence < REVIEW_THRESHOLD,
    }


def _to_rgb_numpy(tensor_1c: torch.Tensor) -> np.ndarray:
    """Normalisierter 1-Kanal-Tensor → [0,1] RGB-Array für Grad-CAM-Overlay."""
    img = tensor_1c.squeeze().cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return np.stack([img, img, img], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# Schnelltest
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from .splits import build_clean_splits

    weights_path = Path(__file__).parent.parent / "outputs" / "model_weights.pth"
    if not weights_path.exists():
        print("Erst exportieren: uv run python -m src.export")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(weights_path, device)
    print("Modell geladen.")

    # Ein Testbild aus dem Test-Split laden
    _, _, test_samples = build_clean_splits()
    test_path, true_label = test_samples[0]

    with Image.open(test_path) as img:
        result = predict(img, model, device)

    print(f"\nVorhersage:  {result['predicted_class']}  ({result['confidence']*100:.1f}%)")
    print(f"Wahrheit:    {CLASS_NAMES[true_label]}")
    print(f"Prüfung nötig: {result['needs_review']}")
    print(f"Alle Klassen: { {k: f'{v:.3f}' for k, v in result['class_probabilities'].items()} }")
    print(f"Grad-CAM:    {result['gradcam_overlay'].size} PIL-Image")
    print("\nInference-Test bestanden.")

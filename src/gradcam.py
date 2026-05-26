"""
Phase 6 — Grad-CAM Visualisierungen

Grad-CAM (Gradient-weighted Class Activation Mapping) zeigt, welche
Bildregionen das Modell für seine Entscheidung genutzt hat:
  - Warme Farben (rot/gelb) = hohe Aktivierung → wichtig für die Vorhersage
  - Kühle Farben (blau)     = geringe Aktivierung → unwichtig

Target-Layer: backbone.layer4[-1] — letzter Residual-Block von ResNet50,
kurz vor dem Global Average Pooling. Hier sind die semantisch reichsten
Feature-Maps.

Outputs:
  outputs/plots/gradcam_per_class.png   — je 4 korrekte Beispiele pro Klasse
  outputs/plots/gradcam_errors.png      — alle falsch klassifizierten Patches

Run:
  uv run python -m src.gradcam
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch.utils.data import DataLoader

from .dataset import CLASS_NAMES, WeldDefectDataset, get_val_transforms
from .model import WeldDefectModule
from .splits import build_clean_splits

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT  = Path(__file__).parent.parent
PLOTS_DIR  = REPO_ROOT / "outputs" / "plots"
CKPT_PATH  = REPO_ROOT / "outputs" / "models" / "finetune-ce-unfrozen" / "best.ckpt"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

EXAMPLES_PER_CLASS = 4   # correct examples shown per class row


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_rgb_numpy(tensor_1c: torch.Tensor) -> np.ndarray:
    """Convert a normalised 1-channel tensor to a [0,1] RGB numpy array."""
    img = tensor_1c.squeeze().cpu().numpy()           # (H, W)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)  # rescale to [0,1]
    return np.stack([img, img, img], axis=-1).astype(np.float32)  # (H, W, 3)


def _overlay(tensor_1c: torch.Tensor, cam_map: np.ndarray) -> np.ndarray:
    """Return a Grad-CAM overlay as uint8 RGB numpy array."""
    rgb = _to_rgb_numpy(tensor_1c)
    return show_cam_on_image(rgb, cam_map, use_rgb=True)


# ---------------------------------------------------------------------------
# Inference + Grad-CAM in one pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_samples(
    module: WeldDefectModule,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list, list]:
    """
    Returns:
        correct: list of (tensor, true_label, pred_label, confidence)
        errors:  list of (tensor, true_label, pred_label, confidence)
    """
    correct, errors = [], []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        logits = module.model(imgs)
        probs  = torch.softmax(logits, dim=1)
        preds  = probs.argmax(dim=1)
        confs  = probs.max(dim=1).values

        for i in range(len(imgs)):
            entry = (
                imgs[i].cpu(),
                int(labels[i]),
                int(preds[i]),
                float(confs[i]),
            )
            if preds[i] == labels[i]:
                correct.append(entry)
            else:
                errors.append(entry)

    return correct, errors


def compute_gradcam(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    target_class: int,
    cam: GradCAM,
) -> np.ndarray:
    """Run Grad-CAM for a single image tensor, returns (H, W) heatmap in [0,1]."""
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    model_device = next(model.parameters()).device
    inp = tensor.unsqueeze(0).to(model_device)
    targets = [ClassifierOutputTarget(target_class)]
    with torch.enable_grad():
        grayscale_cam = cam(input_tensor=inp, targets=targets)
    return grayscale_cam[0]


# ---------------------------------------------------------------------------
# Plot 1: correct examples per class
# ---------------------------------------------------------------------------

def plot_per_class(correct: list, cam: GradCAM) -> None:
    n_cols = EXAMPLES_PER_CLASS
    n_rows = len(CLASS_NAMES)
    # Each example = original + overlay → 2 sub-columns per example
    fig, axes = plt.subplots(
        n_rows, n_cols * 2,
        figsize=(n_cols * 4, n_rows * 2.2),
    )
    fig.suptitle("Grad-CAM — Correct Predictions per Class", fontsize=13, y=1.01)

    # Group correct samples by class
    by_class: dict[int, list] = {i: [] for i in range(len(CLASS_NAMES))}
    for entry in correct:
        by_class[entry[1]].append(entry)

    for row, cls_idx in enumerate(range(len(CLASS_NAMES))):
        samples = by_class[cls_idx][:n_cols]
        cls_name = CLASS_NAMES[cls_idx]

        for col, (tensor, true_lbl, pred_lbl, conf) in enumerate(samples):
            ax_orig    = axes[row, col * 2]
            ax_overlay = axes[row, col * 2 + 1]

            # Original image
            img_np = _to_rgb_numpy(tensor)
            ax_orig.imshow(img_np, cmap="gray")
            ax_orig.axis("off")
            if col == 0:
                # ax.text() in Achsen-Koordinaten ist zuverlässiger als set_ylabel
                # bei tight_layout: x=-0.15 = links außerhalb der Achse
                ax_orig.text(
                    -0.15, 0.5, cls_name,
                    transform=ax_orig.transAxes,
                    fontsize=11, fontweight="bold",
                    va="center", ha="right",
                )

            # Grad-CAM overlay
            heatmap = compute_gradcam(None, tensor, pred_lbl, cam)
            overlay = _overlay(tensor, heatmap)
            ax_overlay.imshow(overlay)
            ax_overlay.set_title(f"{conf*100:.1f}%", fontsize=8, pad=2)
            ax_overlay.axis("off")

        # Fill empty slots if fewer samples than n_cols
        for col in range(len(samples), n_cols):
            axes[row, col * 2].axis("off")
            axes[row, col * 2 + 1].axis("off")

    # Column headers on top row
    for col in range(n_cols):
        axes[0, col * 2].set_title("Original", fontsize=8)
        axes[0, col * 2 + 1].set_title("Grad-CAM", fontsize=8)

    fig.tight_layout()
    save_path = PLOTS_DIR / "gradcam_per_class.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Plot 2: misclassified examples
# ---------------------------------------------------------------------------

def plot_errors(errors: list, cam: GradCAM) -> None:
    if not errors:
        print("No errors to plot — model classified everything correctly.")
        return

    n = len(errors)
    fig, axes = plt.subplots(n, 2, figsize=(5, n * 2.5))
    if n == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle("Grad-CAM — Misclassified Patches", fontsize=12)

    for i, (tensor, true_lbl, pred_lbl, conf) in enumerate(errors):
        ax_orig    = axes[i, 0]
        ax_overlay = axes[i, 1]

        img_np = _to_rgb_numpy(tensor)
        ax_orig.imshow(img_np, cmap="gray")
        ax_orig.set_title(f"True: {CLASS_NAMES[true_lbl]}", fontsize=9)
        ax_orig.axis("off")

        heatmap = compute_gradcam(None, tensor, pred_lbl, cam)
        overlay = _overlay(tensor, heatmap)
        ax_overlay.imshow(overlay)
        ax_overlay.set_title(f"Pred: {CLASS_NAMES[pred_lbl]}  ({conf*100:.1f}%)", fontsize=9)
        ax_overlay.axis("off")

    axes[0, 0].set_xlabel("Original", fontsize=9)
    axes[0, 1].set_xlabel("Grad-CAM", fontsize=9)

    fig.tight_layout()
    save_path = PLOTS_DIR / "gradcam_errors.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}  ({n} errors shown)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    module = WeldDefectModule.load_from_checkpoint(str(CKPT_PATH), map_location=device)
    module.eval()
    module.to(device)

    # Grad-CAM setup — target: last ResNet50 residual block
    target_layers = [module.model.backbone.layer4[-1]]
    cam = GradCAM(model=module.model, target_layers=target_layers)

    # Test set (genuine hold-out)
    _, _, test_samples = build_clean_splits()
    test_ds = WeldDefectDataset(samples=test_samples, transform=get_val_transforms())
    loader  = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)
    print(f"Test patches: {len(test_ds)}")

    print("Running inference...")
    correct, errors = collect_samples(module, loader, device)
    print(f"  Correct: {len(correct)}  |  Errors: {len(errors)}")

    print("Generating Grad-CAM visualisations...")
    plot_per_class(correct, cam)
    plot_errors(errors, cam)

    print("Done.")


if __name__ == "__main__":
    main()

"""
Phase 5 Evaluation — RIAWELC Weld Defect Classifier

Loads the best checkpoint (finetune-ce-unfrozen/best.ckpt) and runs a full
evaluation on the pre-defined test split. Produces:

  outputs/plots/confusion_matrix.png       — normalised heatmap
  outputs/plots/classification_report.png  — per-class Precision / Recall / F1
  outputs/plots/confidence_distribution.png — softmax-confidence: correct vs wrong
  outputs/evaluation_results.json          — raw numbers for downstream use

Run:
  uv run python -m src.evaluate
  uv run python -m src.evaluate --ckpt outputs/models/focal-loss-unfrozen/best.ckpt
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from .dataset import CLASS_NAMES, WeldDefectDataset, get_val_transforms
from .model import WeldDefectModule

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT   = Path(__file__).parent.parent
MODELS_DIR  = REPO_ROOT / "outputs" / "models"
PLOTS_DIR   = REPO_ROOT / "outputs" / "plots"
RESULTS_PATH = REPO_ROOT / "outputs" / "evaluation_results.json"

DEFAULT_CKPT = MODELS_DIR / "finetune-ce-unfrozen" / "best.ckpt"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(ckpt_path: Path, batch_size: int = 64) -> dict:
    """
    Loads checkpoint, runs inference on the test split.

    Returns dict with:
      all_preds    — (N,) int array of predicted class indices
      all_labels   — (N,) int array of ground-truth labels
      all_confs    — (N,) float array of max softmax probability
      all_probs    — (N, 4) float array of full softmax distribution
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading checkpoint: {ckpt_path}")

    module = WeldDefectModule.load_from_checkpoint(str(ckpt_path), map_location=device)
    module.eval()
    module.to(device)

    test_ds = WeldDefectDataset(split="testing", transform=get_val_transforms())
    loader  = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )

    print(f"Test samples: {len(test_ds)}")

    all_preds, all_labels, all_probs = [], [], []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        logits = module.model(imgs)
        probs  = torch.softmax(logits, dim=1)
        preds  = probs.argmax(dim=1)

        all_preds.append(preds.cpu())
        all_labels.append(labels)
        all_probs.append(probs.cpu())

    preds  = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    probs  = torch.cat(all_probs).numpy()
    confs  = probs.max(axis=1)

    return {
        "all_preds":  preds,
        "all_labels": labels,
        "all_confs":  confs,
        "all_probs":  probs,
    }


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _class_label(i: int) -> str:
    """Return full label with index, e.g. 'CR (0)'."""
    return f"{CLASS_NAMES[i]} ({i})"

CLASS_LABELS = [_class_label(i) for i in range(len(CLASS_NAMES))]


def plot_confusion_matrix(preds: np.ndarray, labels: np.ndarray, save_path: Path) -> None:
    cm = confusion_matrix(labels, preds, labels=list(range(len(CLASS_NAMES))))

    # Row-normalise: each cell = fraction of true-class samples predicted as that class
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Fraction of true class", fontsize=9)

    n = len(CLASS_NAMES)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(CLASS_LABELS, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(CLASS_LABELS, fontsize=9)
    ax.set_xlabel("Predicted label", fontsize=10)
    ax.set_ylabel("True label", fontsize=10)
    ax.set_title("Confusion Matrix (row-normalised)", fontsize=11, pad=10)

    # Annotate each cell with fraction + raw count
    thresh = 0.5
    for r in range(n):
        for c in range(n):
            color = "white" if cm_norm[r, c] > thresh else "black"
            ax.text(
                c, r,
                f"{cm_norm[r, c]:.2f}\n({cm[r, c]})",
                ha="center", va="center",
                fontsize=8, color=color,
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_classification_report(preds: np.ndarray, labels: np.ndarray, save_path: Path) -> dict:
    report = classification_report(
        labels, preds,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    metrics = ["precision", "recall", "f1-score"]
    x = np.arange(len(CLASS_NAMES))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#4C72B0", "#55A868", "#C44E52"]

    for i, metric in enumerate(metrics):
        values = [report[cls][metric] for cls in CLASS_NAMES]
        bars = ax.bar(x + i * width, values, width, label=metric.replace("-score", ""), color=colors[i])
        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{v:.2f}",
                ha="center", va="bottom", fontsize=7.5,
            )

    ax.set_xticks(x + width)
    ax.set_xticklabels(CLASS_NAMES, fontsize=10)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.set_title("Per-class Precision / Recall / F1", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Print table to console
    print("\n" + classification_report(labels, preds, target_names=CLASS_NAMES, zero_division=0))

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")

    return report


def plot_confidence_distribution(
    confs: np.ndarray,
    preds: np.ndarray,
    labels: np.ndarray,
    save_path: Path,
) -> None:
    correct = confs[preds == labels]
    wrong   = confs[preds != labels]

    bins = np.linspace(0, 1, 26)
    fig, ax = plt.subplots(figsize=(7, 4))

    # density=True raises RuntimeWarning when a group is empty; skip density for empty arrays
    ax.hist(correct, bins=bins, alpha=0.65, color="#55A868",
            label=f"Correct  (n={len(correct):,})", density=len(correct) > 0)
    if len(wrong) > 0:
        ax.hist(wrong, bins=bins, alpha=0.65, color="#C44E52",
                label=f"Wrong    (n={len(wrong):,})", density=True)
    else:
        ax.plot([], [], color="#C44E52", label=f"Wrong    (n=0)")

    ax.axvline(0.5, color="grey", linestyle="--", linewidth=0.8, label="Confidence = 0.5")
    ax.set_xlabel("Max softmax confidence", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Confidence Distribution: Correct vs. Wrong Predictions", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(ckpt_path: Path) -> None:
    results = run_inference(ckpt_path)
    preds   = results["all_preds"]
    labels  = results["all_labels"]
    confs   = results["all_confs"]

    accuracy = (preds == labels).mean()
    print(f"\nOverall accuracy: {accuracy:.4f}  ({(preds == labels).sum()}/{len(labels)})")

    plot_confusion_matrix(preds, labels, PLOTS_DIR / "confusion_matrix.png")
    report = plot_classification_report(preds, labels, PLOTS_DIR / "classification_report.png")
    plot_confidence_distribution(confs, preds, labels, PLOTS_DIR / "confidence_distribution.png")

    # Save raw numbers
    summary = {
        "checkpoint": str(ckpt_path),
        "n_samples":  int(len(labels)),
        "accuracy":   float(accuracy),
        "per_class":  {
            cls: {k: float(report[cls][k]) for k in ["precision", "recall", "f1-score", "support"]}
            for cls in CLASS_NAMES
        },
        "macro_avg": {k: float(report["macro avg"][k]) for k in ["precision", "recall", "f1-score"]},
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {RESULTS_PATH}")
    print(f"\nMacro F1: {summary['macro_avg']['f1-score']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained RIAWELC checkpoint.")
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=DEFAULT_CKPT,
        help=f"Path to .ckpt file (default: {DEFAULT_CKPT})",
    )
    args = parser.parse_args()

    if not args.ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.ckpt}\n"
            "Download it from Kaggle outputs and place it under outputs/models/."
        )

    main(args.ckpt)

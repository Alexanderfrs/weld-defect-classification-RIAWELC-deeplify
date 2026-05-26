"""
Phase 5.2 — Weld-Level Honest Evaluation

Bewertet das beste Modell auf einem weld-level Test-Split:
  - Kein Weld-ID des Test-Sets wurde während des Trainings gesehen
  - Das misst echte Generalisierung auf unbekannte Schweißstücke

Vergleicht außerdem die Ergebnisse mit dem patch-level Test-Split (Pre-Defined),
um das Ausmaß des Leakage-Effekts quantitativ sichtbar zu machen.

Run:
  uv run python -m src.evaluate_weld_split
"""

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
from .splits import build_weld_level_splits

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT    = Path(__file__).parent.parent
MODELS_DIR   = REPO_ROOT / "outputs" / "models"
PLOTS_DIR    = REPO_ROOT / "outputs" / "plots"
RESULTS_PATH = REPO_ROOT / "outputs" / "weld_split_evaluation.json"

DEFAULT_CKPT = MODELS_DIR / "finetune-ce-unfrozen" / "best.ckpt"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Inference (shared helper)
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(module: WeldDefectModule, loader: DataLoader, device: torch.device) -> dict:
    preds_list, labels_list, probs_list = [], [], []

    for imgs, labels in loader:
        imgs   = imgs.to(device)
        logits = module.model(imgs)
        probs  = torch.softmax(logits, dim=1)
        preds  = probs.argmax(dim=1)

        preds_list.append(preds.cpu())
        labels_list.append(labels)
        probs_list.append(probs.cpu())

    preds  = torch.cat(preds_list).numpy()
    labels = torch.cat(labels_list).numpy()
    probs  = torch.cat(probs_list).numpy()

    return {
        "preds":  preds,
        "labels": labels,
        "confs":  probs.max(axis=1),
    }


# ---------------------------------------------------------------------------
# Plot: side-by-side confusion matrices
# ---------------------------------------------------------------------------

def plot_comparison_confusion(
    preds_biased:  np.ndarray, labels_biased:  np.ndarray,
    preds_honest:  np.ndarray, labels_honest:  np.ndarray,
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    titles = ["Patch-Level Split (biased)", "Weld-Level Split (honest)"]

    for ax, preds, labels, title in zip(
        axes,
        [preds_biased, preds_honest],
        [labels_biased, labels_honest],
        titles,
    ):
        cm = confusion_matrix(labels, preds, labels=list(range(len(CLASS_NAMES))))
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        n = len(CLASS_NAMES)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right", fontsize=9)
        ax.set_yticklabels(CLASS_NAMES, fontsize=9)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("True", fontsize=9)
        ax.set_title(title, fontsize=10, pad=8)

        thresh = 0.5
        for r in range(n):
            for c in range(n):
                color = "white" if cm_norm[r, c] > thresh else "black"
                ax.text(c, r, f"{cm_norm[r, c]:.2f}\n({cm[r, c]})",
                        ha="center", va="center", fontsize=7.5, color=color)

    fig.suptitle("Confusion Matrix: Biased vs. Honest Evaluation", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Plot: per-class recall comparison bar chart
# ---------------------------------------------------------------------------

def plot_recall_comparison(
    report_biased: dict,
    report_honest: dict,
    save_path: Path,
) -> None:
    x = np.arange(len(CLASS_NAMES))
    width = 0.35

    biased_recall = [report_biased[cls]["recall"] for cls in CLASS_NAMES]
    honest_recall = [report_honest[cls]["recall"] for cls in CLASS_NAMES]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars1 = ax.bar(x - width / 2, biased_recall, width, label="Patch-level (biased)", color="#4C72B0", alpha=0.85)
    bars2 = ax.bar(x + width / 2, honest_recall, width, label="Weld-level (honest)",  color="#C44E52", alpha=0.85)

    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, fontsize=10)
    ax.set_ylabel("Recall", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.set_title("Per-class Recall: Biased vs. Honest Evaluation", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not DEFAULT_CKPT.exists():
        raise FileNotFoundError(
            f"Checkpoint nicht gefunden: {DEFAULT_CKPT}\n"
            "Lade ihn von Kaggle herunter und lege ihn unter outputs/models/ ab."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading checkpoint: {DEFAULT_CKPT}")

    module = WeldDefectModule.load_from_checkpoint(str(DEFAULT_CKPT), map_location=device)
    module.eval()
    module.to(device)

    transform = get_val_transforms()

    # --- Biased evaluation: pre-defined patch-level test split ---
    print("\n--- Biased Evaluation (pre-defined test split) ---")
    biased_ds     = WeldDefectDataset(split="testing", transform=transform)
    biased_loader = DataLoader(biased_ds, batch_size=64, shuffle=False, num_workers=4)
    print(f"Patches: {len(biased_ds)}")

    biased = run_inference(module, biased_loader, device)

    # --- Honest evaluation: weld-level test split ---
    print("\n--- Honest Evaluation (weld-level test split) ---")
    _, _, test_samples = build_weld_level_splits()
    honest_ds     = WeldDefectDataset(samples=test_samples, transform=transform)
    honest_loader = DataLoader(honest_ds, batch_size=64, shuffle=False, num_workers=4)
    print(f"Patches: {len(honest_ds)}")

    honest = run_inference(module, honest_loader, device)

    # --- Metrics ---
    biased_acc = (biased["preds"] == biased["labels"]).mean()
    honest_acc = (honest["preds"] == honest["labels"]).mean()

    biased_report = classification_report(
        biased["labels"], biased["preds"], target_names=CLASS_NAMES,
        output_dict=True, zero_division=0,
    )
    honest_report = classification_report(
        honest["labels"], honest["preds"], target_names=CLASS_NAMES,
        output_dict=True, zero_division=0,
    )

    print("\n" + "=" * 55)
    print(f"{'':30} {'Biased':>10} {'Honest':>10}")
    print("-" * 55)
    print(f"{'Overall Accuracy':30} {biased_acc:>10.4f} {honest_acc:>10.4f}")
    print(f"{'Macro F1':30} {biased_report['macro avg']['f1-score']:>10.4f} {honest_report['macro avg']['f1-score']:>10.4f}")
    print(f"{'Macro Recall':30} {biased_report['macro avg']['recall']:>10.4f} {honest_report['macro avg']['recall']:>10.4f}")
    print("-" * 55)
    for cls in CLASS_NAMES:
        b = biased_report[cls]["recall"]
        h = honest_report[cls]["recall"]
        print(f"  Recall {cls:4}                      {b:>10.4f} {h:>10.4f}")
    print("=" * 55)

    # --- Plots ---
    plot_comparison_confusion(
        biased["preds"], biased["labels"],
        honest["preds"], honest["labels"],
        PLOTS_DIR / "weld_split_confusion_comparison.png",
    )
    plot_recall_comparison(
        biased_report, honest_report,
        PLOTS_DIR / "weld_split_recall_comparison.png",
    )

    # --- Save results ---
    results = {
        "biased": {
            "n_samples": int(len(biased["labels"])),
            "accuracy":  float(biased_acc),
            "macro_f1":  float(biased_report["macro avg"]["f1-score"]),
            "per_class_recall": {cls: float(biased_report[cls]["recall"]) for cls in CLASS_NAMES},
        },
        "honest": {
            "n_samples": int(len(honest["labels"])),
            "accuracy":  float(honest_acc),
            "macro_f1":  float(honest_report["macro avg"]["f1-score"]),
            "per_class_recall": {cls: float(honest_report[cls]["recall"]) for cls in CLASS_NAMES},
        },
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()

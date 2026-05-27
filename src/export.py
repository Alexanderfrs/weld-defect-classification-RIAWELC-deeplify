"""
Phase 8.1 — Modell-Export für HuggingFace

Lädt den trainierten Lightning-Checkpoint und speichert nur die reinen
Modellgewichte (state_dict). Das reduziert die Dateigröße erheblich, weil
Optimizer-Zustände und Lightning-Metadaten wegfallen.

Warum state_dict statt TorchScript?
  - state_dict ist einfacher zu laden und zu debuggen
  - Kein Tracing nötig, das bei dynamischen Modellen fehleranfällig sein kann
  - Für HF-Spaces-Inferenz völlig ausreichend

Output:
  outputs/model_weights.pth   — reine Gewichte, ~100 MB

Run:
  uv run python -m src.export
"""

from pathlib import Path

import torch

from .model import WeldDefectModule

REPO_ROOT   = Path(__file__).parent.parent
CKPT_PATH   = REPO_ROOT / "outputs" / "models" / "finetune-ce-unfrozen" / "best.ckpt"
OUTPUT_PATH = REPO_ROOT / "outputs" / "model_weights.pth"


def export_weights(ckpt_path: Path = CKPT_PATH, output_path: Path = OUTPUT_PATH) -> None:
    """
    Extrahiert Modellgewichte aus einem Lightning-Checkpoint.

    Lightning-Checkpoints enthalten:
      - model.state_dict()   ← das wollen wir
      - optimizer state_dict  ← für Inferenz unnötig
      - hparams               ← für Inferenz unnötig
      - lr_scheduler state    ← für Inferenz unnötig

    Args:
        ckpt_path:   Pfad zum .ckpt-File von PyTorch Lightning.
        output_path: Zielpfad für die exportierten Gewichte (.pth).
    """
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint nicht gefunden: {ckpt_path}")

    print(f"Lade Checkpoint: {ckpt_path}")
    # map_location="cpu": Gewichte auf CPU laden, unabhängig davon wo trainiert wurde
    module = WeldDefectModule.load_from_checkpoint(str(ckpt_path), map_location="cpu")
    module.eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(module.model.state_dict(), output_path)

    size_mb = output_path.stat().st_size / 1e6
    print(f"Exportiert: {output_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    export_weights()

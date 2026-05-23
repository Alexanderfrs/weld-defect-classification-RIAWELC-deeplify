# deeplify-exploration-task

PyTorch Lightning project for **RIAWELC weld defect classification** (LP, CR, PO, ND) on 224x224 grayscale PNG radiographic images.

## Stack
- PyTorch
- PyTorch Lightning
- torchvision
- Weights & Biases (W&B)
- pytorch-grad-cam

## Installation
```bash
pip install -r requirements.txt
```

## Expected dataset layout
The training script supports either explicit splits (`train/`, `val/`, `test/`) or a single folder of class subfolders.

```text
RIAWELC/
  train/
    LP/*.png
    CR/*.png
    PO/*.png
    ND/*.png
  val/
    LP/*.png
    ...
  test/
    LP/*.png
    ...
```

If `train/`, `val/`, `test/` are not present, the datamodule performs an automatic random split from class folders under `data_dir`.

## Train
```bash
python train.py --data_dir /path/to/RIAWELC --max_epochs 30 --batch_size 32
```

Useful flags:
- `--fast_dev_run` for one quick sanity iteration
- `--no_pretrained` to disable ImageNet initialization

## Grad-CAM usage
```python
from deeplify.gradcam import generate_gradcam_overlay

overlay = generate_gradcam_overlay(model, image_tensor)  # image_tensor shape: [1, 1, H, W]
```

Dataset: https://github.com/stefyste/RIAWELC

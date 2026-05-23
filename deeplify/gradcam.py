from __future__ import annotations

import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


def generate_gradcam_overlay(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    target_category: int | None = None,
) -> np.ndarray:
    """Generate a Grad-CAM heatmap overlay for a single grayscale image tensor."""

    if image_tensor.ndim != 4 or image_tensor.shape[0] != 1:
        raise ValueError("image_tensor must be a batched tensor with shape [1, 1, H, W]")

    target_layers = [model.backbone.layer4[-1]]
    with GradCAM(model=model, target_layers=target_layers) as cam:
        targets = (
            None
            if target_category is None
            else [ClassifierOutputTarget(target_category)]
        )
        grayscale_cam = cam(input_tensor=image_tensor, targets=targets)[0]

    image = image_tensor[0, 0].detach().cpu().float()
    image = (image * 0.5 + 0.5).clamp(0.0, 1.0).numpy()
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    image_rgb = np.repeat(image[..., None], 3, axis=2)
    return show_cam_on_image(image_rgb, grayscale_cam, use_rgb=True)

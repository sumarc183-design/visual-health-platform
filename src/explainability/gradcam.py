"""Cartes d'explicabilité Grad-CAM pour justifier une prédiction du modèle."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def generate_gradcam(
    model: nn.Module,
    target_layer: nn.Module,
    image: np.ndarray,
    input_tensor: torch.Tensor,
    target_class: int | None = None,
) -> np.ndarray:
    """Retourne l'image superposée à la carte d'activation (0-1, RGB float)."""
    cam = GradCAM(model=model, target_layers=[target_layer])
    targets = None if target_class is None else [target_class]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    image_float = image.astype(np.float32) / 255.0
    return show_cam_on_image(image_float, grayscale_cam, use_rgb=True)

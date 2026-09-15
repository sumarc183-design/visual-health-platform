"""Cartes d'explicabilité Grad-CAM pour justifier une prédiction du modèle.

RETFound est un ViT, pas un CNN : Grad-CAM standard suppose une carte
d'activation spatiale (sortie d'une conv), alors qu'un bloc de transformer
sort une séquence de tokens (1 CLS + N patches). Il faut donc un
`reshape_transform` qui retire le token CLS et replie la séquence de patches
en grille 2D avant de calculer la CAM -- sinon la carte produite est un
artefact sans rapport avec l'image (piège classique documenté par la lib
pytorch_grad_cam elle-même pour les ViT)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def vit_reshape_transform(tensor: torch.Tensor, patch_grid_size: int) -> torch.Tensor:
    """(batch, 1 + patch_grid_size**2, dim) -> (batch, dim, grid, grid), en
    retirant le token CLS. `patch_grid_size` = img_size / patch_size (ex.
    224/16 = 14 pour RETFound à résolution d'entraînement standard)."""
    result = tensor[:, 1:, :].reshape(tensor.size(0), patch_grid_size, patch_grid_size, tensor.size(-1))
    return result.permute(0, 3, 1, 2)


def generate_gradcam(
    model: nn.Module,
    target_layer: nn.Module,
    image: np.ndarray,
    input_tensor: torch.Tensor,
    target_class: int | None = None,
    patch_grid_size: int | None = None,
) -> np.ndarray:
    """Retourne l'image superposée à la carte d'activation (0-1, RGB float).

    Pour un CNN classique : laisser `patch_grid_size=None`.
    Pour RETFound/un ViT : passer `patch_grid_size` (ex. 14 pour du 224x224
    en patches 16x16) et `target_layer` = la LayerNorm/dernier bloc du ViT.
    """
    reshape_transform = (
        (lambda t: vit_reshape_transform(t, patch_grid_size)) if patch_grid_size is not None else None
    )
    cam = GradCAM(model=model, target_layers=[target_layer], reshape_transform=reshape_transform)
    targets = None if target_class is None else [ClassifierOutputTarget(target_class)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    image_float = image.astype(np.float32) / 255.0
    return show_cam_on_image(image_float, grayscale_cam, use_rgb=True)

"""Localisation de la rétinopathie : modèles figés et mesures contre des
masques d'experts (scripts/localization_maps.py, localization_analysis.py)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score

from src.models.retfound_backbone import IMAGENET_MEAN, IMAGENET_STD, load_pretrained_backbone
from src.models.retfound_green import GREEN_IMAGE_SIZE, RETFoundGreenBackbone

MODELS = {
    "RETFound": {"image_size": 224, "patch": 32, "stride": 16},
    # Même fraction de l'image cachée et même nombre de positions qu'à 224 px
    "RETFound-Green": {"image_size": GREEN_IMAGE_SIZE, "patch": 56, "stride": 28},
}


class FrozenClassifier(nn.Module):
    """Images dans [0, 1] -> logit de la rétinopathie (N, 1), et tokens de patch."""

    def __init__(self, name: str, checkpoint: str | Path, weight: np.ndarray, bias: float, trust: bool | None = None):
        super().__init__()
        self.name = name
        size = MODELS[name]["image_size"]
        if name == "RETFound-Green":
            self.vit = RETFoundGreenBackbone(checkpoint, img_size=size).backbone
            mean, std = RETFoundGreenBackbone.input_mean, RETFoundGreenBackbone.input_std
        else:
            self.vit = load_pretrained_backbone(checkpoint, img_size=size, trust_checkpoint=trust)
            mean, std = IMAGENET_MEAN, IMAGENET_STD
        self.register_buffer("mean", mean.clone())
        self.register_buffer("std", std.clone())
        self.head = nn.Linear(len(weight), 1)
        self.head.weight.data = torch.as_tensor(weight, dtype=torch.float32)[None]
        self.head.bias.data = torch.tensor([float(bias)], dtype=torch.float32)
        self.eval()

    def _normalized(self, images: torch.Tensor) -> torch.Tensor:
        return (images - self.mean) / self.std

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self._normalized(images)
        features = self.vit(x) if self.name == "RETFound-Green" else self.vit.forward_features(x)
        return self.head(features)

    @torch.no_grad()
    def tokens(self, images: torch.Tensor) -> torch.Tensor:
        """Tokens de patch (N, T, D), avant la moyenne qui donne l'embedding."""
        x = self._normalized(images)
        if self.name == "RETFound-Green":
            return self.vit.forward_features(x)[:, self.vit.num_prefix_tokens :]
        vit = self.vit
        x = vit.patch_embed(x)
        x = torch.cat((vit.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = vit.pos_drop(x + vit.pos_embed)
        for block in vit.blocks:
            x = block(x)
        return x[:, 1:]


def resize_map(heat: np.ndarray, size: int) -> np.ndarray:
    """Carte (H, W) ramenée à (size, size), par moyenne en réduction."""
    image = Image.fromarray(heat.astype(np.float32), mode="F")
    return np.asarray(image.resize((size, size), Image.BOX), dtype=np.float32)


def map_auroc(heat: np.ndarray, target: np.ndarray, retina: np.ndarray) -> float:
    """Probabilité qu'un pixel de lésion ait un score plus haut qu'un pixel
    sain de la rétine. NaN si la lésion est absente de la rétine."""
    y = target[retina]
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, heat[retina]))


def top_enrichment(heat: np.ndarray, target: np.ndarray, retina: np.ndarray, fraction: float = 0.05) -> float:
    """Densité de lésion dans exactement `fraction` des pixels de la rétine, ceux
    de plus fort score, divisée par la densité sur toute la rétine. 1 = pas mieux
    que le hasard. Ex æquo départagés par l'ordre des pixels, jamais par
    l'annotation."""
    scores, y = heat[retina], target[retina]
    if y.sum() == 0:
        return float("nan")
    count = max(1, round(fraction * len(scores)))
    top = np.argsort(-scores, kind="stable")[:count]
    return float(y[top].mean() / y.mean())


def cell_targets(mask: np.ndarray, grid: int, retina: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Masque (S, S) -> (grille booléenne : la case contient de la lésion,
    grille booléenne : la case est au moins à moitié dans la rétine)."""
    step = mask.shape[0] // grid
    if step * grid != mask.shape[0]:
        raise ValueError(f"grille {grid} incompatible avec {mask.shape[0]} pixels")
    blocks = mask.reshape(grid, step, grid, step)
    inside = retina.reshape(grid, step, grid, step).mean(axis=(1, 3)) >= 0.5
    return blocks.max(axis=(1, 3)) > 0, inside

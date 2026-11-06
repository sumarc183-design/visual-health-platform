"""RETFound-Green (Engelmann & Bernabeu, arXiv:2405.00117) comme extracteur
d'embeddings figé : ViT-S/14 DINOv2 avec 4 registres, 392×392, 384 dimensions,
moyenne des tokens de patch. Poids : release v0.1 du dépôt
justinengelmann/RETFound_Green, non versionnés (licence dans
THIRD_PARTY_NOTICES.md).
"""

from __future__ import annotations

from pathlib import Path

import timm
import torch
import torch.nn as nn

GREEN_IMAGE_SIZE = 392
GREEN_EMBEDDING_DIM = 384
GREEN_MEAN = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
GREEN_STD = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)


class RETFoundGreenBackbone(nn.Module):
    """`input_mean` et `input_std` remplacent la normalisation ImageNet dans
    run_baseline.collect_embeddings."""

    input_mean = GREEN_MEAN
    input_std = GREEN_STD
    embedding_dim = GREEN_EMBEDDING_DIM

    def __init__(self, checkpoint_path: str | Path, img_size: int = GREEN_IMAGE_SIZE):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_small_patch14_reg4_dinov2",
            img_size=(img_size, img_size),
            num_classes=0,
            checkpoint_path=str(checkpoint_path),
        )
        self.backbone.global_pool = "avg"  # comme le dépôt d'origine
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    def train(self, mode: bool = True):
        return super().train(False)  # toujours figé

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

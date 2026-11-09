"""RETFound-Green (Engelmann & Bernabeu, arXiv:2405.00117) comme extracteur
d'embeddings figé : ViT-S/14 DINOv2 avec 4 registres, 392×392, 384 dimensions,
moyenne des tokens de patch. Poids : release v0.1 du dépôt
justinengelmann/RETFound_Green, non versionnés (licence dans
THIRD_PARTY_NOTICES.md).
"""

from __future__ import annotations

from pathlib import Path

import torch

from .timm_backbones import FrozenTimmBackbone

GREEN_IMAGE_SIZE = 392
GREEN_EMBEDDING_DIM = 384
GREEN_MEAN = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
GREEN_STD = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)


class RETFoundGreenBackbone(FrozenTimmBackbone):
    """ViT-S/14 DINOv2 à 4 registres, moyenne des tokens de patch (comme le dépôt
    d'origine), normalisation 0,5."""

    model_name = "vit_small_patch14_reg4_dinov2"
    global_pool = "avg"
    embedding_dim = GREEN_EMBEDDING_DIM
    input_mean = GREEN_MEAN
    input_std = GREEN_STD

    def __init__(self, checkpoint_path: str | Path, img_size: int = GREEN_IMAGE_SIZE):
        super().__init__(checkpoint_path, img_size)

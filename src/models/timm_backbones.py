"""Modèles figés construits avec timm : RETFound-Green (rétine) et DINOv2
(généraliste, pour la dermatologie). Les poids sont lus depuis un fichier local ;
si sa résolution diffère, les encodages de position sont interpolés (timm)."""

from __future__ import annotations

from pathlib import Path

import timm
import torch
import torch.nn as nn
from timm.layers import resample_abs_pos_embed

from .retfound_backbone import IMAGENET_MEAN, IMAGENET_STD


class FrozenTimmBackbone(nn.Module):
    """`input_mean` et `input_std` remplacent la normalisation ImageNet dans
    run_baseline.collect_embeddings."""

    model_name: str
    global_pool: str
    embedding_dim: int
    input_mean = IMAGENET_MEAN
    input_std = IMAGENET_STD

    def __init__(self, checkpoint_path: str | Path, img_size: int):
        super().__init__()
        self.backbone = timm.create_model(self.model_name, img_size=(img_size, img_size), num_classes=0)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if state["pos_embed"].shape != self.backbone.pos_embed.shape:
            # Autre résolution que celle des poids : encodages de position interpolés.
            # Le modèle n'a pas été entraîné à cette résolution.
            prefix = self.backbone.pos_embed.shape[1] - self.backbone.patch_embed.num_patches
            side = int(round((state["pos_embed"].shape[1] - prefix) ** 0.5))
            state["pos_embed"] = resample_abs_pos_embed(
                state["pos_embed"],
                new_size=list(self.backbone.patch_embed.grid_size),
                old_size=[side, side],
                num_prefix_tokens=prefix,
            )
        self.backbone.load_state_dict(state, strict=True)
        self.backbone.global_pool = self.global_pool
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.eval()

    def train(self, mode: bool = True):
        return super().train(False)  # toujours figé

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


DINOV2_MODEL = "vit_base_patch14_dinov2.lvd142m"
DINOV2_IMAGE_SIZE = 224


class DINOv2Backbone(FrozenTimmBackbone):
    """DINOv2 ViT-B/14 (Meta, Apache 2.0), token de classe, 768 dimensions.
    Poids : `timm.create_model(DINOV2_MODEL, pretrained=True).state_dict()`
    enregistré dans un fichier (voir le notebook de dermatologie)."""

    model_name = DINOV2_MODEL
    global_pool = "token"
    embedding_dim = 768

    def __init__(self, checkpoint_path: str | Path, img_size: int = DINOV2_IMAGE_SIZE):
        super().__init__(checkpoint_path, img_size)

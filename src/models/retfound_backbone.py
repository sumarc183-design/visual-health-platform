"""Loads the RETFound ViT-Large/16 backbone and its pretrained checkpoint.

Architecture and checkpoint-loading logic adapted from rmaphoh/RETFound_MAE
(main_finetune.py + models_vit.py). See THIRD_PARTY_NOTICES.md: RETFound is
CC BY-NC 4.0 -- research/internal use only until a commercial license (or an
alternative backbone) is in place.

Requires `timm` (already in requirements.txt).
"""
from __future__ import annotations

import pickle
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import timm.models.vision_transformer

from .pos_embed import interpolate_pos_embed

# RETFound est préentraîné (et fine-tuné dans main_finetune.py) sur des images
# normalisées ImageNet. Les datasets du projet renvoient des images dans
# [0, 1] (le contrôle qualité en a besoin) : normaliser juste avant le backbone.
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class RETFoundViT(timm.models.vision_transformer.VisionTransformer):
    """ViT with global average pooling instead of the CLS token, matching
    RETFound's fine-tuning setup."""

    def __init__(self, global_pool: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs["norm_layer"]
            embed_dim = kwargs["embed_dim"]
            self.fc_norm = norm_layer(embed_dim)
            del self.norm

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x)
        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1, keepdim=True)
            return self.fc_norm(x)[:, 0]
        return self.norm(x)[:, 0]


def build_retfound_vit_large(img_size: int = 224, global_pool: bool = True) -> RETFoundViT:
    return RETFoundViT(
        img_size=img_size, patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        global_pool=global_pool, num_classes=0,
    )


def _extract_state_dict(checkpoint) -> dict:
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            return checkpoint["model"]
        if "teacher" in checkpoint:
            return checkpoint["teacher"]
    return checkpoint


def load_pretrained_backbone(checkpoint_path: str | Path, img_size: int = 224, global_pool: bool = True) -> RETFoundViT:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Poids RETFound introuvables : {checkpoint_path}. "
            "Les télécharger avec scripts/download_retfound_checkpoint.py (voir README)."
        )
    model = build_retfound_vit_large(img_size=img_size, global_pool=global_pool)

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        # Les checkpoints MAE embarquent parfois des objets non-tenseurs (ex.
        # argparse.Namespace dans "args"), refusés par weights_only=True --
        # défaut de torch>=2.6. Repli acceptable : fichier officiel RETFound.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_model = _extract_state_dict(checkpoint)
    checkpoint_model = {k.replace("backbone.", ""): v for k, v in checkpoint_model.items()}

    # Only the classifier head can legitimately mismatch (different
    # num_classes); anything else mismatching (notably pos_embed, when the
    # checkpoint was pretrained at a different image size) must go through
    # interpolate_pos_embed below instead of being dropped here.
    state_dict = model.state_dict()
    for key in ("head.weight", "head.bias"):
        if key in checkpoint_model and key in state_dict and checkpoint_model[key].shape != state_dict[key].shape:
            print(f"Skipping mismatched-shape key: {key}")
            del checkpoint_model[key]

    interpolate_pos_embed(model, checkpoint_model)
    missing, unexpected = model.load_state_dict(checkpoint_model, strict=False)
    print(f"RETFound loaded -- missing: {len(missing)} keys, unexpected: {len(unexpected)} keys")

    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model


class RETFoundBackbone(nn.Module):
    """Thin wrapper exposing a frozen RETFound encoder as a feature extractor."""

    def __init__(self, checkpoint_path: str | Path, img_size: int = 224, embedding_dim: int = 1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.backbone = load_pretrained_backbone(checkpoint_path, img_size=img_size, global_pool=True)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_features(x)

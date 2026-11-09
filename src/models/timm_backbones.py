"""Modèles figés construits avec timm : RETFound-Green (rétine), DINOv2
(généraliste) et PanDerm (dermatologie). Les poids sont lus depuis un fichier local ;
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
    model_kwargs: dict = {}

    def __init__(self, checkpoint_path: str | Path, img_size: int):
        super().__init__()
        self.backbone = timm.create_model(
            self.model_name, img_size=(img_size, img_size), num_classes=0, **self.model_kwargs
        )
        state = self.prepare_state(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
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

    def prepare_state(self, state: dict) -> dict:
        return state

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


DINOV2_LARGE_MODEL = "vit_large_patch14_dinov2.lvd142m"


class DINOv2LargeBackbone(DINOv2Backbone):
    """DINOv2 ViT-L/14 (Apache 2.0), token de classe, 1 024 dimensions : même taille
    que PanDerm-Large, pour séparer l'effet de la spécialisation de celui de la taille."""

    model_name = DINOV2_LARGE_MODEL
    embedding_dim = 1024


PANDERM_MODEL = "beit_large_patch16_224"
PANDERM_IMAGE_SIZE = 224
# Poids PanDerm-Large (CC BY-NC-ND 4.0, recherche académique non commerciale) :
# https://github.com/SiyuanYan1/PanDerm, fichier Google Drive ci-dessous.
PANDERM_GDRIVE_ID = "1SwEzaOlFV_gBKf2UzeowMC8z9UH7AQbE"


class PanDermBackbone(FrozenTimmBackbone):
    """PanDerm-Large (Yan et al., Nature Medicine 2025) : ViT-L/16 de type BEiT/CAE
    (biais q et v séparés, layer scale, encodages de position absolus fixes, pas de
    biais de position relatif). Comme dans leur évaluation linéaire
    (`modeling_finetune.VisionTransformer`, `lin_probe=True`) : token de classe après
    la normalisation finale, 1 024 dimensions, entrée 224 px.

    Le checkpoint contient tout le modèle CAE : seules les clés `encoder.*` sont
    gardées, puis chargées en mode strict (leur code utilise `strict=False`)."""

    model_name = PANDERM_MODEL
    global_pool = "token"
    embedding_dim = 1024
    # Constantes de leur code (builder.py) : écart-type rouge 0,228 et non 0,229.
    input_std = torch.tensor([0.228, 0.224, 0.225]).view(1, 3, 1, 1)
    model_kwargs = {"use_abs_pos_emb": True, "use_rel_pos_bias": False, "init_values": 1e-5, "global_pool": "token"}

    def __init__(self, checkpoint_path: str | Path, img_size: int = PANDERM_IMAGE_SIZE):
        super().__init__(checkpoint_path, img_size)

    def prepare_state(self, state: dict) -> dict:
        prefix = "encoder."
        encoder = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
        if not encoder:
            raise ValueError("checkpoint PanDerm attendu : aucune clé « encoder. »")
        return encoder

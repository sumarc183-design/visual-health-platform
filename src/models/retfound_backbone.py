"""Charge le backbone RETFound ViT-Large/16 et son checkpoint préentraîné.

Architecture et logique de chargement adaptées de rmaphoh/RETFound_MAE
(main_finetune.py + models_vit.py). Voir THIRD_PARTY_NOTICES.md : RETFound est
sous licence CC BY-NC 4.0 -- usage recherche/interne uniquement tant qu'une
licence commerciale (ou un autre backbone) n'est pas en place.

Nécessite `timm` (dans requirements.txt).
"""

from __future__ import annotations

import argparse
import os
import pickle
from functools import partial
from pathlib import Path

import timm.models.vision_transformer
import torch
import torch.nn as nn

from .pos_embed import interpolate_pos_embed

# RETFound est préentraîné (et fine-tuné dans main_finetune.py) sur des images
# normalisées ImageNet. Les datasets du projet renvoient des images dans
# [0, 1] (le contrôle qualité en a besoin) : normaliser juste avant le backbone.
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

ALLOWED_MISSING_KEYS = {"fc_norm.weight", "fc_norm.bias"}

# Dossier où scripts/download_retfound_checkpoint.py range les poids officiels
# (Hugging Face, dépôt RETFound). Seul un fichier de ce dossier peut, par
# défaut, être chargé sans la protection weights_only.
TRUSTED_CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "third_party" / "retfound" / "checkpoints"


def load_checkpoint_file(checkpoint_path: Path, trust_checkpoint: bool | None = None):
    """torch.load protégé. 1) weights_only=True ; 2) idem en autorisant
    argparse.Namespace, objet inoffensif que les checkpoints MAE embarquent
    dans "args" (torch >= 2.4) ; 3) en dernier recours, chargement complet
    (pickle : peut exécuter du code), seulement pour un fichier de confiance :
    `trust_checkpoint=True`, ou par défaut un fichier du dossier officiel
    TRUSTED_CHECKPOINT_DIR."""
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as error:
        first_error = error
    if hasattr(torch.serialization, "safe_globals"):
        try:
            with torch.serialization.safe_globals([argparse.Namespace]):
                return torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except pickle.UnpicklingError:
            pass
    if trust_checkpoint is None:
        # Dossier qui contient le fichier (pas la cible d'un lien symbolique :
        # le téléchargement Hugging Face peut lier le fichier à son cache).
        trust_checkpoint = Path(os.path.abspath(checkpoint_path)).parent.resolve() == TRUSTED_CHECKPOINT_DIR.resolve()
    if not trust_checkpoint:
        raise RuntimeError(
            f"{checkpoint_path} contient des objets que torch.load(weights_only=True) refuse "
            f"({first_error}). Le charger complètement peut exécuter du code : le faire seulement pour "
            "un fichier de confiance, avec trust_checkpoint=True (--trust-checkpoint), ou en le plaçant "
            f"dans {TRUSTED_CHECKPOINT_DIR} via scripts/download_retfound_checkpoint.py."
        )
    print(f"⚠ Checkpoint chargé sans weights_only (fichier de confiance) : {checkpoint_path}")
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


class RETFoundViT(timm.models.vision_transformer.VisionTransformer):
    """ViT avec moyenne globale des tokens au lieu du token CLS, comme dans le
    fine-tuning de RETFound."""

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
        img_size=img_size,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        global_pool=global_pool,
        num_classes=0,
    )


def _extract_state_dict(checkpoint) -> dict:
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            return checkpoint["model"]
        if "teacher" in checkpoint:
            return checkpoint["teacher"]
    return checkpoint


def load_pretrained_backbone(
    checkpoint_path: str | Path, img_size: int = 224, global_pool: bool = True, trust_checkpoint: bool | None = None
) -> RETFoundViT:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Poids RETFound introuvables : {checkpoint_path}. "
            "Les télécharger avec scripts/download_retfound_checkpoint.py (voir README)."
        )
    model = build_retfound_vit_large(img_size=img_size, global_pool=global_pool)

    checkpoint = load_checkpoint_file(checkpoint_path, trust_checkpoint)
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
    # Un checkpoint MAE préentraîné n'a pas fc_norm (créée pour le global
    # pool, restée à l'identité) ; ses clés en trop (décodeur, mask_token,
    # norm) sont ignorées. Toute autre clé manquante = encodeur partiellement
    # aléatoire, qui donnerait des chiffres faux sans planter : on refuse.
    unexplained = sorted(set(missing) - ALLOWED_MISSING_KEYS)
    if unexplained:
        raise RuntimeError(
            f"Checkpoint incompatible avec RETFound ViT-L/16 : {len(unexplained)} clé(s) de l'encodeur "
            f"absente(s), ex. {unexplained[:5]}. Vérifier le fichier (poids MAE ou fine-tunés RETFound)."
        )

    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model


class RETFoundBackbone(nn.Module):
    """Enveloppe fine qui expose l'encodeur RETFound figé comme extracteur d'embeddings."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        img_size: int = 224,
        embedding_dim: int = 1024,
        trust_checkpoint: bool | None = None,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.backbone = load_pretrained_backbone(
            checkpoint_path, img_size=img_size, global_pool=True, trust_checkpoint=trust_checkpoint
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_features(x)

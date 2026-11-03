"""Interpolation des position embeddings, pour charger un checkpoint ViT
préentraîné à une autre résolution d'image que celle utilisée ici.

Repris de rmaphoh/RETFound_MAE (util/pos_embed.py), lui-même adapté du code de
référence MAE/DeiT de Meta. Voir THIRD_PARTY_NOTICES.md -- RETFound est sous
licence CC BY-NC 4.0 (usage recherche / non commercial uniquement).
"""

from __future__ import annotations

import torch


def interpolate_pos_embed(model, checkpoint_model: dict) -> None:
    if "pos_embed" not in checkpoint_model:
        return
    pos_embed_checkpoint = checkpoint_model["pos_embed"]
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = model.patch_embed.num_patches
    num_extra_tokens = model.pos_embed.shape[-2] - num_patches
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    new_size = int(num_patches**0.5)
    if orig_size == new_size:
        return

    print(f"Position embedding interpolated from {orig_size}x{orig_size} to {new_size}x{new_size}")
    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
    pos_tokens = torch.nn.functional.interpolate(
        pos_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False
    )
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    checkpoint_model["pos_embed"] = torch.cat((extra_tokens, pos_tokens), dim=1)

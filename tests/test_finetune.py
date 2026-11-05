"""Vérifie le fine-tuning partiel (src/models/finetune.py) sur un petit ViT."""

from functools import partial

import pytest
import torch
import torch.nn as nn

from src.data.fundus_preprocess import common_mask
from src.models.finetune import PartialFinetune, augment_batch, cosine_with_warmup
from src.models.retfound_backbone import RETFoundViT


def tiny_vit() -> RETFoundViT:
    return RETFoundViT(
        global_pool=True,
        img_size=32,
        patch_size=16,
        embed_dim=32,
        depth=4,
        num_heads=4,
        mlp_ratio=2,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )


def test_only_the_last_blocks_and_head_are_trained():
    torch.manual_seed(0)
    model = PartialFinetune(tiny_vit(), num_classes=3, trainable_blocks=2)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    model.train()
    loss = nn.BCEWithLogitsLoss()(model(torch.rand(4, 3, 32, 32)), torch.ones(4, 3))
    loss.backward()
    optimizer.step()
    changed = {n for n, p in model.named_parameters() if not torch.equal(p, before[n])}
    assert any(n.startswith("vit.blocks.3.") for n in changed)
    assert any(n.startswith("vit.blocks.2.") for n in changed)
    assert any(n.startswith("head.") for n in changed)
    frozen = ("vit.blocks.0.", "vit.blocks.1.", "vit.patch_embed", "vit.pos_embed")
    assert not any(n.startswith(frozen) for n in changed)
    assert not model.vit.blocks[0].training and model.vit.blocks[3].training  # partie figée en mode évaluation


def test_frozen_and_trainable_features_match_the_plain_vit():
    """Sans entraînement, les features du modèle partiel sont celles du ViT d'origine."""
    vit = tiny_vit().eval()
    model = PartialFinetune(vit, num_classes=2, trainable_blocks=2).eval()
    images = torch.rand(2, 3, 32, 32)
    expected = vit.forward_features((images - model.mean) / model.std)
    assert torch.allclose(model.features(images), expected, atol=1e-6)


def test_augmentation_keeps_the_common_mask_and_is_reproducible():
    mask = torch.from_numpy(common_mask(64))
    images = torch.rand(3, 3, 64, 64) * mask
    a = augment_batch(images, mask, torch.Generator().manual_seed(1))
    b = augment_batch(images, mask, torch.Generator().manual_seed(1))
    assert torch.equal(a, b)
    assert a[:, :, ~mask].abs().max() == 0  # rien hors du masque après rotation
    assert a.min() >= 0 and a.max() <= 1
    assert not torch.equal(a, images)


def test_learning_rate_schedule():
    assert cosine_with_warmup(0, 100, 10) == pytest.approx(0.1)
    assert cosine_with_warmup(10, 100, 10) == pytest.approx(1.0)
    assert cosine_with_warmup(100, 100, 10) == pytest.approx(0.0, abs=1e-9)
    with pytest.raises(ValueError):
        PartialFinetune(tiny_vit(), num_classes=2, trainable_blocks=5)

"""Vérifie le backbone RETFound (architecture, chargement de checkpoint,
interpolation des position embeddings)."""

import pytest
import torch

from src.models.classifier_head import MultiLabelHead
from src.models.retfound_backbone import RETFoundBackbone, build_retfound_vit_large


def test_vit_large_param_count_and_forward():
    model = build_retfound_vit_large(img_size=224, global_pool=True)
    n_params = sum(p.numel() for p in model.parameters())
    assert 280e6 < n_params < 320e6, f"{n_params / 1e6:.1f}M params, attendu ~304M pour un ViT-L/16"

    out = model.forward_features(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 1024)


def test_backbone_loads_checkpoint_with_model_key(tmp_path):
    model = build_retfound_vit_large(img_size=224, global_pool=True)
    ckpt_path = tmp_path / "ckpt.pth"
    torch.save({"model": model.state_dict(), "epoch": 50}, ckpt_path)

    backbone = RETFoundBackbone(ckpt_path, img_size=224, embedding_dim=1024)
    out = backbone(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 1024)


def test_backbone_accepts_mae_checkpoint_without_fc_norm(tmp_path):
    """Les poids MAE préentraînés n'ont pas fc_norm : cas normal, pas une erreur."""
    state = build_retfound_vit_large(img_size=224, global_pool=True).state_dict()
    for key in ("fc_norm.weight", "fc_norm.bias"):
        del state[key]
    ckpt_path = tmp_path / "ckpt.pth"
    torch.save({"model": state}, ckpt_path)

    RETFoundBackbone(ckpt_path, img_size=224, embedding_dim=1024)


def test_backbone_rejects_checkpoint_missing_encoder_keys(tmp_path):
    """Un checkpoint où manquent des poids de l'encodeur ne doit pas se
    charger en silence avec des couches aléatoires."""
    state = build_retfound_vit_large(img_size=224, global_pool=True).state_dict()
    state = {k: v for k, v in state.items() if not k.startswith("blocks.0.")}
    ckpt_path = tmp_path / "ckpt.pth"
    torch.save({"model": state}, ckpt_path)

    with pytest.raises(RuntimeError, match="Checkpoint incompatible"):
        RETFoundBackbone(ckpt_path, img_size=224, embedding_dim=1024)


def test_pos_embed_interpolation_on_resolution_change(tmp_path):
    """Un checkpoint pré-entraîné à 224x224 doit se charger sans erreur sur
    un backbone à 256x256 (interpolation des position embeddings), et NE
    DOIT PAS silencieusement jeter pos_embed comme une clé mismatchée."""
    model_224 = build_retfound_vit_large(img_size=224, global_pool=True)
    ckpt_path = tmp_path / "ckpt_224.pth"
    torch.save({"model": model_224.state_dict()}, ckpt_path)

    backbone_256 = RETFoundBackbone(ckpt_path, img_size=256, embedding_dim=1024)
    out = backbone_256(torch.randn(1, 3, 256, 256))
    assert out.shape == (1, 1024)


def test_classifier_head_on_backbone_features(tmp_path):
    model = build_retfound_vit_large(img_size=224, global_pool=True)
    ckpt_path = tmp_path / "ckpt.pth"
    torch.save({"model": model.state_dict()}, ckpt_path)
    backbone = RETFoundBackbone(ckpt_path, img_size=224, embedding_dim=1024)

    head = MultiLabelHead(embedding_dim=1024, num_classes=28)
    features = backbone(torch.randn(2, 3, 224, 224))
    logits = head(features)
    assert logits.shape == (2, 28)


class NotATensor:
    """Objet arbitraire : torch.load(weights_only=True) le refuse."""


def test_untrusted_checkpoint_with_arbitrary_objects_is_refused(tmp_path):
    """Hors du dossier officiel, pas de chargement complet (pickle) sans
    accord explicite : il pourrait exécuter du code."""
    from src.models.retfound_backbone import load_checkpoint_file

    path = tmp_path / "suspect.pth"
    torch.save({"model": {"w": torch.zeros(1)}, "extra": NotATensor()}, path)
    with pytest.raises(RuntimeError, match="weights_only"):
        load_checkpoint_file(path)
    assert "model" in load_checkpoint_file(path, trust_checkpoint=True)


def test_plain_checkpoint_loads_safely_anywhere(tmp_path):
    from src.models.retfound_backbone import load_checkpoint_file

    path = tmp_path / "weights.pth"
    torch.save({"model": {"w": torch.ones(2)}}, path)
    assert torch.equal(load_checkpoint_file(path)["model"]["w"], torch.ones(2))

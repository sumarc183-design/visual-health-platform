"""Backbone RETFound-Green : poids chargés, pooling, normalisation propre."""

import numpy as np
import pytest
import timm
import torch

from run_baseline import collect_embeddings
from src.models.retfound_green import GREEN_EMBEDDING_DIM, RETFoundGreenBackbone

SIZE = 98  # multiple de 14 : assez petit pour un test rapide


@pytest.fixture
def checkpoint(tmp_path):
    torch.manual_seed(0)
    model = timm.create_model("vit_small_patch14_reg4_dinov2", img_size=(SIZE, SIZE), num_classes=0)
    path = tmp_path / "green.pth"
    torch.save(model.state_dict(), path)
    return path


def test_loads_weights_and_stays_frozen(checkpoint):
    state = torch.load(checkpoint, weights_only=True)
    backbone = RETFoundGreenBackbone(checkpoint, img_size=SIZE).train()
    name = "blocks.0.attn.proj.weight"
    assert torch.equal(backbone.backbone.state_dict()[name], state[name])
    assert not backbone.training and not backbone.backbone.training
    assert not any(p.requires_grad for p in backbone.parameters())


def test_embedding_is_the_mean_of_patch_tokens_without_prefix_tokens(checkpoint):
    backbone = RETFoundGreenBackbone(checkpoint, img_size=SIZE)
    x = torch.randn(2, 3, SIZE, SIZE)
    out = backbone(x)
    assert out.shape == (2, GREEN_EMBEDDING_DIM)
    tokens = backbone.backbone.forward_features(x)
    assert backbone.backbone.num_prefix_tokens == 5  # classe + 4 registres
    assert torch.allclose(out, tokens[:, 5:].mean(dim=1), atol=1e-5)


class _Recorder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = None

    def forward(self, x):
        self.seen = x
        return x.flatten(1)[:, :2]


def _images():
    return [(torch.full((3, 4, 4), 0.5), torch.zeros(1))]


def test_collect_embeddings_uses_the_backbone_normalisation():
    recorder = _Recorder()
    recorder.input_mean, recorder.input_std = RETFoundGreenBackbone.input_mean, RETFoundGreenBackbone.input_std
    collect_embeddings(recorder, _images())
    assert torch.equal(recorder.seen, torch.zeros_like(recorder.seen))  # (0,5 - 0,5) / 0,5


def test_collect_embeddings_keeps_imagenet_by_default():
    recorder = _Recorder()
    collect_embeddings(recorder, _images())
    expected = (0.5 - 0.485) / 0.229
    assert np.isclose(recorder.seen[0, 0, 0, 0].item(), expected, atol=1e-5)

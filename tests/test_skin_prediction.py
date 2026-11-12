"""Chaîne peau de predict.py (tête DINOv2, sans contrôle qualité), avec un faux modèle
léger à la place de DINOv2-B : vérifie le câblage, pas les poids."""

import json
import sys

import numpy as np
import pytest
import torch
from PIL import Image

import src.pipeline.inference as inference
from src.data.fundus_preprocess import PREPROCESS_VERSION
from src.data.isic_dataset import CLASS_NAMES, load_skin_image
from src.models.classifier_head import MultiLabelHead
from src.models.embedding_cache import file_fingerprint

DIM = 8


class FakeDINOv2(torch.nn.Module):
    """Moyenne des pixels par canal, projetée sur DIM dimensions."""

    embedding_dim = DIM

    def __init__(self, checkpoint_path, img_size=224):
        super().__init__()
        torch.manual_seed(0)
        self.proj = torch.nn.Linear(3, DIM)

    def forward(self, x):
        return self.proj(x.mean(dim=(2, 3)))


@pytest.fixture
def skin(tmp_path, monkeypatch):
    monkeypatch.setattr(inference, "DINOv2Backbone", FakeDINOv2)
    checkpoint = tmp_path / "dinov2.pth"
    checkpoint.write_bytes(b"faux poids")
    torch.manual_seed(1)
    head = MultiLabelHead(DIM, len(CLASS_NAMES))
    head_path = tmp_path / "skin_head.pt"
    torch.save(
        {
            "state_dict": head.state_dict(),
            "class_names": CLASS_NAMES,
            "modality": "dermoscopy",
            "backbone": "dinov2",
            "image_size": 224,
            "preprocess": "skin",
            "preprocess_version": PREPROCESS_VERSION,
            "checkpoint_sha256": file_fingerprint(checkpoint),
        },
        head_path,
    )
    image = tmp_path / "lesion.png"
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (300, 400, 3), dtype=np.uint8)).save(image)
    return {"head": head, "head_path": head_path, "checkpoint": checkpoint, "image": image}


def test_skin_head_uses_skin_preprocessing_and_skips_fundus_quality(skin, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("contrôle qualité fond d'œil appelé sur une image de peau")

    monkeypatch.setattr(inference, "check_fundus_file", forbidden)
    predictor = inference.Predictor(skin["head_path"], skin["checkpoint"])
    assert predictor.modality == "dermoscopy"
    prediction = predictor.predict(skin["image"])
    assert prediction.quality == {"checked": False} and not prediction.rejected
    x = load_skin_image(skin["image"], 224).unsqueeze(0)
    expected = torch.sigmoid(skin["head"](FakeDINOv2(None)((x - inference.IMAGENET_MEAN) / inference.IMAGENET_STD)))[0]
    got = [prediction.findings[c]["probability"] for c in CLASS_NAMES]
    assert np.allclose(got, expected.detach().numpy(), atol=1e-6)
    assert all(f["decision"] is None for f in prediction.findings.values())


def test_skin_head_refuses_another_checkpoint(skin, tmp_path):
    other = tmp_path / "autre.pth"
    other.write_bytes(b"autres poids")
    with pytest.raises(ValueError, match="checkpoint"):
        inference.Predictor(skin["head_path"], other)


def test_cli_prints_probabilities_without_thresholds(skin, monkeypatch, capsys, tmp_path):
    import predict

    out = tmp_path / "pred.json"
    argv = ["predict.py", str(skin["image"]), "--head", str(skin["head_path"])]
    monkeypatch.setattr(sys, "argv", [*argv, "--checkpoint", str(skin["checkpoint"]), "--out", str(out)])
    predict.main()
    assert "probabilité seule (pas de seuil)" in capsys.readouterr().out
    assert json.loads(out.read_text())[0]["quality"] == {"checked": False}

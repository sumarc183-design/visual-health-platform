"""ISIC 2019 : classes, regroupement par lésion, découpages, et DINOv2 figé."""

import numpy as np
import pandas as pd
import pytest
import timm
import torch
from PIL import Image

from src.data.isic_dataset import CLASS_NAMES, ISIC2019Dataset
from src.models.timm_backbones import DINOV2_MODEL, DINOv2Backbone


def _write(root, part, rows, with_lesion=True):
    images = root / f"ISIC_2019_{part}_Input"
    images.mkdir(parents=True)
    labels, meta = [], []
    for name, cls, lesion in rows:
        Image.fromarray(np.full((30, 40, 3), 120, np.uint8)).save(images / f"{name}.jpg")
        labels.append({"image": name, **{c: float(c == cls) for c in [*CLASS_NAMES, "UNK"]}})
        meta.append({"image": name, "age_approx": 50, **({"lesion_id": lesion} if with_lesion else {})})
    pd.DataFrame(labels).to_csv(root / f"ISIC_2019_{part}_GroundTruth.csv", index=False)
    pd.DataFrame(meta).to_csv(root / f"ISIC_2019_{part}_Metadata.csv", index=False)


@pytest.fixture
def isic(tmp_path):
    rows = []
    for i in range(70):
        lesion = f"L{i // 2}" if i < 60 else None  # la plupart des lésions ont deux images
        rows.append((f"ISIC_{i:07d}", CLASS_NAMES[i % 2], lesion))
    _write(tmp_path, "Training", rows)
    _write(tmp_path, "Test", [("ISIC_9000000", "MEL", None), ("ISIC_9000001", "UNK", None)], with_lesion=False)
    return tmp_path


def test_validation_never_shares_a_lesion_with_training(isic):
    train, val = ISIC2019Dataset(isic, "train"), ISIC2019Dataset(isic, "val")
    assert len(train) + len(val) == 70 and len(val) > 0
    assert not set(train.groups) & set(val.groups)
    assert train.groups.count("L0") in (0, 2)


def test_test_split_drops_unknown_and_returns_one_hot(isic):
    test = ISIC2019Dataset(isic, "test", image_size=32)
    assert len(test) == 1 and test.groups == ["ISIC_9000000"]
    image, labels = test[0]
    assert image.shape == (3, 32, 32) and labels.tolist() == [1.0] + [0.0] * 7


def test_dinov2_loads_its_weights_at_another_resolution(tmp_path):
    torch.manual_seed(0)
    model = timm.create_model(DINOV2_MODEL, img_size=(70, 70), num_classes=0, pretrained=False)
    path = tmp_path / "dinov2.pth"
    torch.save(model.state_dict(), path)
    backbone = DINOv2Backbone(path, img_size=42)
    assert backbone.backbone.pos_embed.shape[1] == 1 + 9  # token de classe + grille 3×3
    name = "blocks.0.attn.proj.weight"
    assert torch.equal(backbone.backbone.state_dict()[name], model.state_dict()[name])
    out = backbone(torch.rand(2, 3, 42, 42))
    assert out.shape == (2, 768) and not backbone.training

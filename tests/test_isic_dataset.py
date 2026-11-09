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


def test_skin_image_keeps_the_aspect_ratio_and_crops_the_centre(tmp_path):
    from src.data.isic_dataset import load_skin_image

    array = np.zeros((60, 120, 3), np.uint8)
    array[:, 30:90] = 255  # bande centrale blanche, bords noirs
    Image.fromarray(array).save(tmp_path / "wide.png")
    image = load_skin_image(tmp_path / "wide.png", 56)  # petit côté 64, image 128 × 64, recadrage 56
    assert image.shape == (3, 56, 56)
    assert image[:, :, 4:52].min() > 0.95  # la bande centrale remplit le recadrage


def _tiny_panderm():
    from src.models.timm_backbones import PanDermBackbone

    class TinyPanDerm(PanDermBackbone):
        embedding_dim = 64
        model_kwargs = {**PanDermBackbone.model_kwargs, "embed_dim": 64, "depth": 2, "num_heads": 2}

    return TinyPanDerm


def test_panderm_keeps_only_the_encoder_and_returns_the_normalised_cls_token(tmp_path):
    cls = _tiny_panderm()
    torch.manual_seed(0)
    source = timm.create_model(cls.model_name, img_size=(32, 32), num_classes=0, **cls.model_kwargs)
    for name, param in source.named_parameters():  # layer scale non trivial
        if "gamma" in name:
            param.data.uniform_(0.5, 1.5)
    state = {f"encoder.{k}": v for k, v in source.state_dict().items()}
    state.update({"regresser.norm.weight": torch.ones(64), "decoder.head.weight": torch.zeros(8, 64)})
    state["mask_token"] = torch.zeros(1, 1, 64)
    torch.save(state, tmp_path / "panderm.pth")
    backbone = cls(tmp_path / "panderm.pth", img_size=32)
    x = torch.rand(2, 3, 32, 32)
    with torch.no_grad():
        expected = source.norm(_tokens(source, x))[:, 0]
    assert torch.allclose(backbone(x), expected, atol=1e-5)
    assert backbone.input_std.flatten().tolist() == pytest.approx([0.228, 0.224, 0.225])


def _tokens(model, x):
    """Tokens avant la normalisation finale (timm applique déjà `norm` dans forward_features)."""
    x = model.patch_embed(x)
    x = torch.cat([model.cls_token.expand(len(x), -1, -1), x], dim=1) + model.pos_embed
    for block in model.blocks:
        x = block(x, shared_rel_pos_bias=None)
    return x


def test_panderm_rejects_a_checkpoint_without_encoder(tmp_path):
    torch.save({"pos_embed": torch.zeros(1, 5, 64)}, tmp_path / "other.pth")
    with pytest.raises(ValueError, match="encoder"):
        _tiny_panderm()(tmp_path / "other.pth", img_size=32)

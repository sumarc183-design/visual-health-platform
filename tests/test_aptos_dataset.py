"""Vérifie le chargeur APTOS 2019 (validation externe)."""

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.data.aptos_dataset import APTOSDataset


@pytest.fixture
def aptos_root(tmp_path):
    (tmp_path / "train_images").mkdir()
    grades = [0, 1, 2, 3, 4, 0]
    rng = np.random.default_rng(0)
    for i in range(len(grades)):
        Image.fromarray((rng.random((64, 64, 3)) * 255).astype(np.uint8)).save(tmp_path / "train_images" / f"im{i}.png")
    pd.DataFrame({"id_code": [f"im{i}" for i in range(len(grades))], "diagnosis": grades}).to_csv(
        tmp_path / "train.csv", index=False
    )
    return tmp_path


def test_binary_labels_follow_the_grade(aptos_root):
    ds = APTOSDataset(aptos_root, image_size=32)
    assert ds.df["any_dr"].tolist() == [0, 1, 1, 1, 1, 0]
    assert ds.df["referable_dr"].tolist() == [0, 0, 1, 1, 1, 0]
    assert ds.image_keys[0] == "aptos/im0" and ds.groups == ds.image_keys
    image, labels = ds[2]
    assert image.shape == (3, 32, 32) and labels.tolist() == [1.0, 1.0]
    assert ds.image_path(1).name == "im1.png"


def test_invalid_grade_is_refused(aptos_root):
    pd.DataFrame({"id_code": ["im0"], "diagnosis": [7]}).to_csv(aptos_root / "train.csv", index=False)
    with pytest.raises(ValueError, match="stade"):
        APTOSDataset(aptos_root)

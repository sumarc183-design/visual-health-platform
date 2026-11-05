"""Vérifie le chargeur EyePACS (2e validation externe)."""

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.data.eyepacs_dataset import EyePACSDataset


@pytest.fixture
def eyepacs_root(tmp_path):
    (tmp_path / "train").mkdir()
    rows = [("10_left", 0), ("10_right", 2), ("13_left", 4), ("13_right", 1), ("15_left", 0)]
    rng = np.random.default_rng(0)
    for name, _ in rows[:-1]:  # 15_left absente : échantillon partiel extrait
        Image.fromarray((rng.random((48, 64, 3)) * 255).astype(np.uint8)).save(tmp_path / "train" / f"{name}.jpeg")
    pd.DataFrame(rows, columns=["image", "level"]).to_csv(tmp_path / "trainLabels.csv", index=False)
    return tmp_path


def test_only_extracted_images_are_kept_and_grouped_by_patient(eyepacs_root):
    ds = EyePACSDataset(eyepacs_root, image_size=32)
    assert len(ds) == 4
    assert ds.groups == ["10", "10", "13", "13"]
    assert ds.df["any_dr"].tolist() == [0, 1, 1, 1] and ds.df["referable_dr"].tolist() == [0, 1, 1, 0]
    image, labels = ds[1]
    assert image.shape == (3, 32, 32) and labels.tolist() == [1.0, 1.0]
    assert ds.image_keys[0] == "eyepacs/10_left"


def test_invalid_level_is_refused(eyepacs_root):
    pd.DataFrame({"image": ["10_left"], "level": [9]}).to_csv(eyepacs_root / "trainLabels.csv", index=False)
    with pytest.raises(ValueError, match="stade"):
        EyePACSDataset(eyepacs_root)

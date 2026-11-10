"""Chargeur DeepDRiD : colonnes, patients, chemins, valeurs refusées."""

import pandas as pd
import pytest

from src.data.deepdrid import load_split


def _write(root, rows):
    base = root / "regular-fundus-training"
    base.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(base / "regular-fundus-training.csv", index=False)


def test_training_split_gives_patients_paths_and_integer_scores(tmp_path):
    row = {
        "patient_id": 1,
        "image_id": "1_l1",
        "image_path": "x",
        "Overall quality": 0,
        "Artifact": 4,
        "Clarity": 8,
        "Field definition": 8,
    }
    _write(tmp_path, [row, {**row, "image_id": "1_r1", "Overall quality": 1}])
    df = load_split(tmp_path, "train")
    assert df["patient"].tolist() == ["1", "1"]
    assert df["path"][0] == tmp_path / "regular-fundus-training" / "Images" / "1" / "1_l1.jpg"
    assert df["Overall quality"].tolist() == [0, 1]


def test_invalid_quality_or_unknown_split_is_refused(tmp_path):
    row = {"image_id": "1_l1", "Overall quality": 2, "Artifact": 0, "Clarity": 8, "Field definition": 8}
    _write(tmp_path, [row])
    with pytest.raises(ValueError):
        load_split(tmp_path, "train")
    with pytest.raises(ValueError):
        load_split(tmp_path, "test")

"""Confirmation DeepDRiD : lecture des stades, couverture et séparation des patients."""

import pandas as pd
import pytest

from scripts.confirmation_deepdrid import deepdrid_table, grades

QUALITY = {"Overall quality": 1, "Artifact": 0, "Clarity": 8, "Field definition": 8}


def _split(root, folder, csv, ids, left, right):
    base = root / folder
    rows = [
        {"image_id": i, "left_eye_DR_Level": lv, "right_eye_DR_Level": rv, **QUALITY}
        for i, lv, rv in zip(ids, left, right, strict=True)
    ]
    base.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(base / csv, index=False)
    for i in ids:
        (base / "Images" / i.split("_")[0]).mkdir(parents=True, exist_ok=True)
        (base / "Images" / i.split("_")[0] / f"{i}.jpg").write_bytes(b"x")


def _eval(root, ids, levels):
    """Remplace la lecture des classeurs .xlsx de l'évaluation par des tableaux en mémoire."""
    base = root / "Online-Challenge1&2-Evaluation"
    for i in ids:
        (base / "Images" / i.split("_")[0]).mkdir(parents=True, exist_ok=True)
        (base / "Images" / i.split("_")[0] / f"{i}.jpg").write_bytes(b"x")
    stades = pd.DataFrame({"image_id": ids, "DR_Levels": levels})
    qualite = pd.DataFrame([{"image_id": i, **QUALITY} for i in ids])
    return lambda path: stades if path.name.startswith("Challenge1") else qualite


def test_grade_comes_from_the_only_filled_eye_column(tmp_path):
    _split(tmp_path, "regular-fundus-training", "regular-fundus-training.csv", ["1_l1", "1_r3"], [2, 1], [None, None])
    assert grades(tmp_path, "train").to_dict() == {"1_l1": 2, "1_r3": 1}


def test_two_or_no_eye_grade_is_refused(tmp_path):
    _split(tmp_path, "regular-fundus-training", "regular-fundus-training.csv", ["1_l1"], [2], [1])
    with pytest.raises(ValueError, match="exactement un"):
        grades(tmp_path, "train")


def test_table_covers_three_splits_and_groups_by_patient(tmp_path, monkeypatch):
    _split(tmp_path, "regular-fundus-training", "regular-fundus-training.csv", ["1_l1", "1_r1"], [0, None], [None, 3])
    _split(tmp_path, "regular-fundus-validation", "regular-fundus-validation.csv", ["2_l1"], [1], [None])
    reader = _eval(tmp_path, ["3_l1"], [4])
    monkeypatch.setattr("scripts.confirmation_deepdrid.read_xlsx", reader)
    monkeypatch.setattr("src.data.deepdrid.read_xlsx", reader)
    table = deepdrid_table(tmp_path)
    assert table["grade"].tolist() == [0, 3, 1, 4]
    assert table["group"].tolist() == ["deepdrid/1", "deepdrid/1", "deepdrid/2", "deepdrid/3"]


def test_patient_shared_between_splits_is_refused(tmp_path, monkeypatch):
    _split(tmp_path, "regular-fundus-training", "regular-fundus-training.csv", ["1_l1"], [0], [None])
    _split(tmp_path, "regular-fundus-validation", "regular-fundus-validation.csv", ["1_l2"], [0], [None])
    reader = _eval(tmp_path, ["3_l1"], [4])
    monkeypatch.setattr("scripts.confirmation_deepdrid.read_xlsx", reader)
    monkeypatch.setattr("src.data.deepdrid.read_xlsx", reader)
    with pytest.raises(ValueError, match="patients communs"):
        deepdrid_table(tmp_path)

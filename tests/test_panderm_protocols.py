"""Constat F (audit du 19 nov.) : correspondances, séparation par patient et contrôles d'entrée des études PanDerm."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.panderm_derm12345 import CRITERION, MEL, VAL_FRACTION, derm_targets, held_out_patients
from scripts.panderm_snapshot import check_derm_metadata, check_unique_stems
from src.data.isic_dataset import CLASS_NAMES


def test_derm_targets_mapping():
    y = derm_targets(
        pd.Series(
            ["Melanoma in situ", "Nevus", "Squamous cell carcinoma in situ", "Solar lentigo", "Lymphangioma", None]
        )
    )
    assert y[0, MEL] == 1 and y[0].sum() == 1
    assert y[1, CLASS_NAMES.index("NV")] == 1
    assert y[2, CLASS_NAMES.index("SCC")] == 1
    assert y[3, CLASS_NAMES.index("BKL")] == 1
    assert y[4].sum() == 0 and y[5].sum() == 0  # hors périmètre et maligne sans diagnostic : négatifs


def test_held_out_patients_split_by_patient():
    patients = pd.Series([f"P{i % 50}" for i in range(500)])
    held = held_out_patients(patients)
    assert not set(patients[held]) & set(patients[~held])  # aucun patient des deux côtés
    assert patients[held].nunique() == round(VAL_FRACTION * 50)
    assert np.array_equal(held, held_out_patients(patients.sample(frac=1, random_state=1).sort_index()))
    assert np.array_equal(held, held_out_patients(patients))  # graine fixe


def test_criterion_threshold():
    assert CRITERION == 0.80


def _md(**changes):
    md = pd.DataFrame(
        {"isic_id": ["A", "B", "C"], "patient_id": ["p1", "p1", "p2"], "diagnosis_3": ["Nevus", "Lymphangioma", None]}
    )
    for column, values in changes.items():
        md[column] = values
    return md


def test_derm_metadata_checks():
    check_derm_metadata(_md())
    with pytest.raises(ValueError, match="inattendus"):
        check_derm_metadata(_md(diagnosis_3=["Nevus", "Melanoma invasive", None]))  # faute de casse d'une catégorie MEL
    with pytest.raises(ValueError, match="double"):
        check_derm_metadata(_md(isic_id=["A", "A", "C"]))
    with pytest.raises(ValueError, match="sans patient"):
        check_derm_metadata(_md(patient_id=["p1", None, "p2"]))


def test_duplicate_image_names_refused():
    check_unique_stems([Path("a/x.jpg"), Path("b/y.jpg")], "base")
    with pytest.raises(ValueError, match="double"):
        check_unique_stems([Path("a/x.jpg"), Path("b/x.jpg")], "base")

"""Effectifs rares et patients appariés dans l'étude des défauts réels."""

import numpy as np
import pytest

from scripts.quality_real_defects import auc_with_groups, exact_rate, patient_rate


def test_zero_detection_on_two_patients_does_not_get_a_zero_width_interval():
    result = exact_rate([False, False])
    assert result["rate"] == 0
    assert result["ci"][0] == 0 and result["ci"][1] > 0.8
    assert exact_rate([])["rate"] is None


def test_two_eyes_are_counted_as_one_patient():
    result = patient_rate([{"group": "a"}, {"group": "a"}, {"group": "b"}], np.array([False, True, False]))
    assert result["patients"] == 2
    assert result["detected"] == 1
    assert result["rate"] == 0.5


def test_auc_interval_is_withheld_for_too_few_positive_patients():
    rows = [{"group": str(i)} for i in range(12)]
    y = np.array([1, 1] + [0] * 10)
    result = auc_with_groups(rows, y, y.astype(float))
    assert result["value"] == pytest.approx(1)
    assert result["ci"] is None and "moins de 10" in result["reason"]


def test_review_selection_is_balanced_and_mixed():
    from scripts.quality_real_defects import review_selection

    cases = [{"split": "train", "tags": ["low image quality"], "filename": f"c{i}"} for i in range(5)]
    dust = [{"split": "train", "tags": ["lens dust"], "filename": f"d{i}"} for i in range(5)]
    controls = [{"split": "train", "tags": [], "filename": f"t{i}"} for i in range(40)]
    val = [{"split": "val", "tags": ["low image quality"], "filename": "v0"}]
    chosen = review_selection(cases + dust + controls + val, seed=0)
    names = [r["filename"] for r in chosen]
    assert sum(n.startswith("c") for n in names) == 5 and sum(n.startswith("t") for n in names) == 5
    assert not any(n.startswith(("d", "v")) for n in names)
    assert names[:5] != [f"c{i}" for i in range(5)]  # cas et témoins mélangés
    assert review_selection(cases + dust + controls + val, seed=0) == chosen  # reproductible

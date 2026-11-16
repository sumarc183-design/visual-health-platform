import numpy as np

from scripts.quality_deepdrid_grouped_ci import grouped_rates


def test_grouped_rates_keep_points_and_count_patients():
    y = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    flags = np.array([True, False, False, True, True, False, True, False])
    groups = ["p1", "p1", "p1", "p1", "p2", "p2", "p2", "p2"]
    r = grouped_rates(y, flags, groups)
    assert r["unit"] == "image" and r["patients"] == 2
    assert r["sensitivity"] == 0.75 and r["specificity"] == 0.75
    low, high = r["sensitivity_ci_patient_bootstrap"]
    assert 0 <= low <= high <= 1


def test_odir_patient_from_key():
    import pytest

    from scripts.operating_points_grouped_ci import odir_patient

    assert odir_patient("40_right.jpg") == "40" and odir_patient("odir/7_left.jpg") == "7"
    with pytest.raises(ValueError):
        odir_patient("image.jpg")

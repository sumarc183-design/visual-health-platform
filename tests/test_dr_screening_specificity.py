import numpy as np

from scripts.dr_screening_specificity import paired_specificity_diff, select


def test_select_prefers_the_served_head_on_ties():
    assert select({"green": 0.4, "retfound": 0.4, "fusion": 0.3, "moyenne": 0.2}) == "green"
    assert select({"green": 0.4, "retfound": 0.3, "fusion": 0.5, "moyenne": 0.45}) == "fusion"


def test_paired_specificity_diff_counts_healthy_only():
    y = np.array([0, 0, 0, 0, 1, 1])
    old = np.array([True, True, False, False, True, False])  # spécificité 0,5
    new = np.array([True, False, False, False, False, False])  # spécificité 0,75
    out = paired_specificity_diff(y, new, old, [str(i) for i in range(6)])
    assert np.isclose(out["diff"], 0.25)
    assert out["ci"][0] <= 0.25 <= out["ci"][1]

"""Compromis par maladie : seuil choisi sur la validation, comptes pour 1 000 personnes."""

import numpy as np

from scripts.operating_points import operating_point


def test_operating_point_counts_add_up_and_follow_the_validation_threshold():
    rng = np.random.default_rng(0)
    y_val, y_test = rng.integers(0, 2, 400), rng.integers(0, 2, 400)
    p_val = np.clip(0.3 * y_val + rng.uniform(0, 0.7, 400), 0, 1)
    p_test = np.clip(0.3 * y_test + rng.uniform(0, 0.7, 400), 0, 1)
    point = operating_point(y_val, p_val, y_test, p_test, 0.9)
    c = point["counts"]
    assert c["tp"] + c["fn"] == y_test.sum() and c["fp"] + c["tn"] == (y_test == 0).sum()
    per = point["per_1000"]
    assert per["detected"] + per["missed"] == per["sick"]
    assert (p_val[y_val == 1] >= point["threshold"]).mean() >= 0.9

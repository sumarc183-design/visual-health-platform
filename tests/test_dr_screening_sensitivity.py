import numpy as np

from scripts.dr_screening_sensitivity import rate_ci


def test_rate_ci_brackets_sensitivity_and_specificity():
    rng = np.random.default_rng(0)
    y = np.array([1] * 200 + [0] * 200)
    decided = np.concatenate([rng.random(200) < 0.8, rng.random(200) < 0.3])
    groups = [str(i) for i in range(len(y))]
    low, high = rate_ci(y, decided, groups, positive=True)
    assert low < decided[:200].mean() < high
    low, high = rate_ci(y, decided, groups, positive=False)
    assert low < (~decided[200:]).mean() < high

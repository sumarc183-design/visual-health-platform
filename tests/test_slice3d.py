"""AUC partielle au-dessus de 80 % de sensibilité (mesure officielle d'ISIC 2024)."""

import numpy as np
import pytest

from scripts.slice3d_feasibility import partial_auc


def test_partial_auc_bounds():
    y = np.array([0] * 50 + [1] * 50)
    perfect = y.astype(float)
    assert partial_auc(y, perfect) == pytest.approx(0.2)
    assert partial_auc(y, 1 - perfect) == pytest.approx(0.0, abs=1e-9)
    rng = np.random.default_rng(0)
    random = partial_auc(np.repeat([0, 1], 5000), rng.uniform(size=10000))
    assert random == pytest.approx(0.02, abs=0.005)  # hasard : 0,2² / 2

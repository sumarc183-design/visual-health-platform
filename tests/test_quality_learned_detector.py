"""Défauts simulés : le champ rétinien change, le fond reste noir."""

import numpy as np
import pytest

from scripts.quality_learned_detector import DEFECTS, degrade


@pytest.mark.parametrize("defect", DEFECTS)
def test_degrade_changes_the_field_and_keeps_the_background_black(defect):
    yy, xx = np.mgrid[:128, :128]
    field = (yy - 64) ** 2 + (xx - 64) ** 2 < 55**2
    image = np.zeros((128, 128, 3), np.uint8)
    image[field] = (150, 80, 40)
    image[field & (xx % 8 == 0)] = (90, 30, 20)  # « vaisseaux » pour que le flou se voie
    out = degrade(image, defect, np.random.default_rng(1))
    assert out.dtype == np.uint8 and out.shape == image.shape
    assert (out[~field] == 0).all()
    assert np.abs(out[field].astype(int) - image[field]).mean() > 1


def test_unknown_defect_is_refused():
    with pytest.raises(ValueError):
        degrade(np.zeros((8, 8, 3), np.uint8), "rain", np.random.default_rng(0))

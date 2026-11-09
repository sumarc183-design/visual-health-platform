"""Diagnostic d'exposition : champ utile, sens des variations et entrées invalides."""

import numpy as np
import pytest

from src.pipeline.exposure import exposure_statistics


def test_black_border_is_excluded_from_exposure_statistics():
    image = np.zeros((12, 12, 3), np.uint8)
    mask = np.zeros((12, 12), bool)
    mask[4:8, 4:8] = True
    image[mask] = 120
    scores = exposure_statistics(image, mask)
    assert scores["median_luminance"] == 120
    assert scores["dark_pixel_share"] == 0
    image[~mask] = 255
    assert exposure_statistics(image, mask) == scores


def test_dimming_and_clipping_have_the_expected_direction():
    image = np.full((10, 10, 3), 160, np.uint8)
    mask = np.ones((10, 10), bool)
    original = exposure_statistics(image, mask)
    dark = exposure_statistics((image * 0.05).astype(np.uint8), mask)
    bright = exposure_statistics(np.full_like(image, 255), mask)
    assert dark["median_luminance"] < original["median_luminance"] < bright["median_luminance"]
    assert dark["dark_pixel_share"] == 1
    assert bright["bright_pixel_share"] == 1


@pytest.mark.parametrize("bad_mask", [np.zeros((4, 4), bool), np.ones((3, 3), bool), np.ones((4, 4), np.uint8)])
def test_empty_or_incompatible_field_masks_are_refused(bad_mask):
    with pytest.raises(ValueError, match="masque"):
        exposure_statistics(np.zeros((4, 4, 3), np.uint8), bad_mask)


def test_float_image_is_refused_instead_of_mixing_luminance_scales():
    with pytest.raises(ValueError, match="uint8"):
        exposure_statistics(np.ones((4, 4, 3), np.float32), np.ones((4, 4), bool))

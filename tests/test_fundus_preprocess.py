"""Vérifie le prétraitement commun des images de fond d'œil."""

import numpy as np
import pytest
from PIL import Image

from src.data.fundus_preprocess import crop_mask, fundus_bbox, load_fundus_image


def fundus_photo(width: int, height: int, radius: int, center_x: int, white_band: int = 0) -> Image.Image:
    """Disque orangé (rétine) sur fond noir, éventuellement décentré, avec
    une bande blanche de remplissage à gauche comme certaines images ODIR."""
    yy, xx = np.mgrid[:height, :width]
    disc = (yy - height / 2) ** 2 + (xx - center_x) ** 2 < radius**2
    array = np.zeros((height, width, 3), dtype=np.uint8)
    array[disc] = (200, 90, 40)
    array[:, :white_band] = 255
    return Image.fromarray(array)


def test_bbox_finds_the_disc_and_ignores_white_band():
    photo = fundus_photo(600, 400, radius=150, center_x=350, white_band=40)
    left, top, right, bottom = fundus_bbox(photo)
    assert abs(left - 200) <= 4 and abs(right - 500) <= 4
    assert abs(top - 50) <= 4 and abs(bottom - 350) <= 4


def test_bbox_none_on_blank_image():
    assert fundus_bbox(Image.new("RGB", (100, 80))) is None


@pytest.mark.parametrize("width, height, center_x", [(600, 400, 300), (400, 400, 200), (900, 500, 600)])
def test_crop_mask_gives_the_same_geometry_whatever_the_camera(width, height, center_x):
    """Formats et cadrages différents -> même disque centré, mêmes coins noirs :
    la géométrie ne trahit plus l'appareil."""
    out = crop_mask(fundus_photo(width, height, radius=min(height // 2 - 10, 180), center_x=center_x), 64)
    assert out.shape == (64, 64, 3) and out.dtype == np.uint8
    assert (out[0, 0] == 0).all() and (out[-1, -1] == 0).all() and (out[0, -1] == 0).all()
    assert (out[1, 32] == 0).all() and (out[-2, 32] == 0).all()  # coupe commune haut/bas
    assert out[32, 32, 0] > 150  # centre = rétine
    assert out[32, 3, 0] > 150 and out[32, 60, 0] > 150  # le disque remplit la largeur


def test_load_fundus_image_both_modes(tmp_path):
    path = tmp_path / "eye.png"
    fundus_photo(300, 200, radius=90, center_x=150).save(path)
    for preprocess in ("none", "crop-mask"):
        image = load_fundus_image(path, 32, preprocess)
        assert image.shape == (3, 32, 32)
        assert 0.0 <= float(image.min()) and float(image.max()) <= 1.0
    with pytest.raises(ValueError):
        load_fundus_image(path, 32, "autre")

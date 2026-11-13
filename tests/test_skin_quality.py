import numpy as np
from PIL import Image

from scripts.calibrate_skin_quality import blurred
from src.pipeline.quality_control import skin_blur_score


def test_skin_blur_score_drops_with_blur_and_ignores_the_dark_border():
    rng = np.random.default_rng(0)
    array = rng.integers(0, 255, (600, 800, 3), dtype=np.uint8)
    sharp = Image.fromarray(array)
    assert skin_blur_score(blurred(sharp, 4)) < skin_blur_score(sharp) / 10
    framed = array.copy()
    framed[:, :120] = 0  # bord noir hors du carré central : sans effet sur le score
    assert np.isclose(skin_blur_score(Image.fromarray(framed)), skin_blur_score(sharp))

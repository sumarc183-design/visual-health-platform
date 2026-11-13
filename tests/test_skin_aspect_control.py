from PIL import Image

from scripts.skin_aspect_control import crop_ratio, squash


def test_crop_ratio_gives_landscape_3_2_from_square_and_4_3():
    for size in ((1024, 1024), (600, 450), (300, 900)):
        w, h = crop_ratio(Image.new("RGB", size)).size
        assert abs(w / h - 1.5) < 0.01


def test_squash_ignores_proportions():
    assert squash(Image.new("RGB", (600, 400))).size == (256, 256)

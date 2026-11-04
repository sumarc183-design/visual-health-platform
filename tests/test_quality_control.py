"""Vérifie le contrôle qualité (flou, résolution)."""

import numpy as np

from src.pipeline.quality_control import check_fundus_file, check_quality, compute_blur_score


def test_sharp_image_is_usable():
    rng = np.random.default_rng(0)
    sharp = (rng.random((300, 300, 3)) * 255).astype(np.uint8)
    report = check_quality(sharp, blur_threshold=50)
    assert report.is_usable
    assert not report.is_blurry


def test_uniform_image_is_flagged_blurry():
    blurry = np.full((300, 300, 3), 128, dtype=np.uint8)
    report = check_quality(blurry, blur_threshold=50)
    assert report.is_blurry
    assert not report.is_usable


def test_low_resolution_image_is_rejected():
    rng = np.random.default_rng(0)
    small = (rng.random((50, 50, 3)) * 255).astype(np.uint8)
    report = check_quality(small, blur_threshold=1)
    assert not report.is_usable
    assert any("résolution" in reason for reason in report.reasons)


def test_blur_score_higher_for_sharp_than_uniform():
    rng = np.random.default_rng(0)
    sharp = (rng.random((100, 100, 3)) * 255).astype(np.uint8)
    uniform = np.full((100, 100, 3), 128, dtype=np.uint8)
    assert compute_blur_score(sharp) > compute_blur_score(uniform)


def test_blur_score_uses_rgb_convention():
    """Les images du projet sont en RGB : le rouge pèse plus que le bleu dans
    le gris (0,299 contre 0,114). Avec la convention BGR, ce serait l'inverse."""
    rng = np.random.default_rng(0)
    pattern = (rng.random((100, 100)) * 255).astype(np.uint8)
    red, blue = np.zeros((100, 100, 3), np.uint8), np.zeros((100, 100, 3), np.uint8)
    red[..., 0], blue[..., 2] = pattern, pattern
    assert compute_blur_score(red) > compute_blur_score(blue)


def _fundus_file(path, size, textured=True):
    """Disque « rétinien » sur fond noir, texturé (net) ou uniforme (flou)."""
    from PIL import Image

    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:size, :size]
    disc = (yy - size / 2) ** 2 + (xx - size / 2) ** 2 <= (0.45 * size) ** 2
    image = np.zeros((size, size, 3), np.uint8)
    image[disc] = (rng.random((disc.sum(), 3)) * 200 + 40) if textured else 150
    Image.fromarray(image).save(path)
    return path


def test_fundus_file_resolution_is_checked_before_resizing(tmp_path):
    """Une image d'origine trop petite est signalée, même si le modèle la
    recevrait agrandie en 224x224."""
    report = check_fundus_file(_fundus_file(tmp_path / "small.png", 120), blur_threshold=0)
    assert not report.is_usable
    assert any("résolution" in reason for reason in report.reasons)
    assert check_fundus_file(_fundus_file(tmp_path / "big.png", 600), blur_threshold=0).is_usable


def test_fundus_blur_ignores_the_disc_edge(tmp_path):
    """Un disque uniforme n'a que son bord de net : il doit rester « flou »,
    car le bord est exclu du calcul ; un disque texturé est net."""
    uniform = check_fundus_file(_fundus_file(tmp_path / "u.png", 600, textured=False), blur_threshold=1.0)
    textured = check_fundus_file(_fundus_file(tmp_path / "t.png", 600, textured=True), blur_threshold=1.0)
    assert uniform.is_blurry and uniform.blur_score < 1e-6
    assert not textured.is_blurry

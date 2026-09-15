"""Vérifie le contrôle qualité (flou, résolution)."""
import numpy as np

from src.pipeline.quality_control import check_quality, compute_blur_score


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

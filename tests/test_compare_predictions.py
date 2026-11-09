"""Vérifie l'alignement de deux runs avant comparaison appariée."""

import numpy as np
import pytest

from scripts.compare_predictions import align, load_predictions


def save_run(path, keys, y, probs, classes=("a", "b")):
    np.savez(path, keys=np.array(keys), groups=np.array(keys), y_true=y, probs=probs, class_names=np.array(classes))
    return load_predictions(path)


def test_align_reorders_b_on_a(tmp_path):
    y = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
    a = save_run(tmp_path / "a.npz", ["k1", "k2", "k3"], y, np.zeros((3, 2)))
    b = save_run(tmp_path / "b.npz", ["k3", "k1", "k2"], y[[2, 0, 1]], np.array([[0.3, 0.3], [0.1, 0.1], [0.2, 0.2]]))
    assert np.array_equal(align(a, b), [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]])


def test_align_refuses_different_images_or_labels(tmp_path):
    y = np.array([[1, 0], [0, 1]], dtype=np.float32)
    a = save_run(tmp_path / "a.npz", ["k1", "k2"], y, np.zeros((2, 2)))
    with pytest.raises(SystemExit):
        align(a, save_run(tmp_path / "b.npz", ["k1", "k9"], y, np.zeros((2, 2))))
    with pytest.raises(SystemExit):
        align(a, save_run(tmp_path / "c.npz", ["k1", "k2"], 1 - y, np.zeros((2, 2))))

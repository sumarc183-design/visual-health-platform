"""Plis internes de l'adaptation few-shot et liaison des seuils à leur tête."""

import subprocess
import sys

import numpy as np

from scripts.site_adaptation import internal_folds


def test_internal_folds_keep_duplicate_groups_together():
    rng = np.random.default_rng(0)
    groups = rng.integers(0, 60, 200)
    y = (groups % 3 == 0).astype(float)  # toutes les copies d'un groupe ont la même étiquette
    folds = internal_folds(y, groups)
    assert len(folds) == 5
    seen = []
    for fit, held_out in folds:
        assert not set(groups[fit]) & set(groups[held_out])
        seen.extend(held_out.tolist())
    assert sorted(seen) == list(range(200))


def test_internal_folds_use_the_number_of_groups_not_of_images():
    # 3 images malades, mais toutes copies d'un seul groupe : impossible de les séparer
    groups = np.array([0, 0, 0, 1, 2, 3, 4])
    y = np.array([1, 1, 1, 0, 0, 0, 0], dtype=float)
    assert internal_folds(y, groups) == []


def test_decision_metrics_refuses_a_head_without_proof(tmp_path):
    result = subprocess.run(
        [sys.executable, "scripts/decision_metrics.py", "--dataset", "rfmid", "--head", "h.pt", "--out", "o.json"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "--embeddings" in result.stderr

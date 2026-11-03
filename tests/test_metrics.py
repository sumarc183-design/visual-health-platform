"""Vérifie les intervalles de confiance et la comparaison appariée des AUROC."""

import numpy as np
from sklearn.metrics import roc_auc_score

from src.eval.metrics import _auroc, bootstrap_auroc_ci, evaluate_multilabel, paired_bootstrap_diff

CLASSES = ["a", "b"]


def labelled_scores(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=(n, 2)).astype(np.float32)
    good = y + rng.normal(0, 0.6, size=y.shape)  # séparable
    noisy = y + rng.normal(0, 3.0, size=y.shape)  # presque au hasard
    groups = [str(i // 2) for i in range(n)]  # deux images par « patient »
    return y, good, noisy, groups


def test_rank_auroc_matches_sklearn():
    y, good, _, _ = labelled_scores()
    for c in range(2):
        assert abs(_auroc(y[:, c], good[:, c]) - roc_auc_score(y[:, c], good[:, c])) < 1e-9
    assert np.isnan(_auroc(np.zeros(5), np.arange(5.0)))


def test_evaluate_multilabel_counts_positives():
    y, good, _, _ = labelled_scores()
    assert evaluate_multilabel(y, good, CLASSES)["positives_per_class"] == {
        "a": int(y[:, 0].sum()),
        "b": int(y[:, 1].sum()),
    }


def test_bootstrap_ci_brackets_the_estimate():
    y, good, _, groups = labelled_scores()
    point = evaluate_multilabel(y, good, CLASSES)
    ci = bootstrap_auroc_ci(y, good, CLASSES, groups, n_boot=200)
    lo, hi = ci["auroc_macro_ci"]
    assert lo < point["auroc_macro"] < hi and hi - lo < 0.1
    for name in CLASSES:
        lo, hi = ci["auroc_per_class_ci"][name]
        assert lo < point["auroc_per_class"][name] < hi


def test_paired_diff_is_zero_for_identical_runs_and_detects_a_better_one():
    y, good, noisy, groups = labelled_scores()
    same = paired_bootstrap_diff(y, good, good, CLASSES, groups, n_boot=100)
    assert same["macro"]["diff"] == 0 and same["macro"]["ci"] == [0.0, 0.0]

    better = paired_bootstrap_diff(y, noisy, good, CLASSES, groups, n_boot=200)
    assert better["macro"]["diff"] > 0 and better["macro"]["ci"][0] > 0
    assert all(better[name]["ci"][0] > 0 for name in CLASSES)

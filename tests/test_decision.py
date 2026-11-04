"""Vérifie les seuils, métriques de décision et la calibration (src/eval/decision.py)."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.eval.decision import (
    apply_temperature,
    brier_score,
    choose_threshold,
    decision_metrics,
    expected_calibration_error,
    fit_temperature,
    per_class_report,
)


def scores(n=2000, prevalence=0.1, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < prevalence).astype(float)
    p = 1 / (1 + np.exp(-(3 * y - 1.5 + rng.normal(0, 1, n))))
    return y, p


def test_decision_metrics_counts():
    y = np.array([1, 1, 0, 0, 0])
    p = np.array([0.9, 0.2, 0.6, 0.1, 0.05])
    m = decision_metrics(y, p, 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 2)
    assert m["sensitivity"] == 0.5 and m["specificity"] == pytest.approx(2 / 3) and m["ppv"] == 0.5


def test_sensitivity_policy_reaches_the_target_on_its_own_data():
    y, p = scores()
    threshold = choose_threshold(y, p, "sensitivity", target=0.9)
    assert decision_metrics(y, p, threshold)["sensitivity"] >= 0.9
    # Le seuil le plus haut possible : un cran plus strict passerait sous la cible.
    stricter = np.unique(p)[np.unique(p) > threshold].min()
    assert decision_metrics(y, p, stricter)["sensitivity"] < 0.9


def test_f1_policy_beats_a_fixed_half_threshold_for_a_rare_class():
    y, p = scores(prevalence=0.03)
    assert decision_metrics(y, p, choose_threshold(y, p, "f1"))["f1"] >= decision_metrics(y, p, 0.5)["f1"]


def test_temperature_fixes_overconfidence_without_changing_the_ranking():
    # Probabilités calibrées par construction : y tiré avec la probabilité p.
    rng = np.random.default_rng(0)
    p = 1 / (1 + np.exp(-rng.normal(-1, 1.5, 5000)))
    y = (rng.random(5000) < p).astype(float)
    overconfident = apply_temperature(p, 1 / 3)  # logits multipliés par 3
    t = fit_temperature(y, overconfident)
    assert t == pytest.approx(3, rel=0.15)
    calibrated = apply_temperature(overconfident, t)
    assert brier_score(y, calibrated) < brier_score(y, overconfident)
    assert expected_calibration_error(y, calibrated) < expected_calibration_error(y, overconfident)
    assert roc_auc_score(y, calibrated) == pytest.approx(roc_auc_score(y, overconfident), abs=1e-9)


def test_rare_classes_are_flagged_as_insufficient_evidence():
    y_common, p_common = scores(n=500, prevalence=0.3, seed=1)
    y_rare = np.zeros(500)
    y_rare[:3] = 1
    y = np.column_stack([y_common, y_rare])
    p = np.column_stack([p_common, np.full(500, 0.1)])
    report = per_class_report(y, p, y, p, ["common", "rare"])
    assert not report["classes"]["common"]["insufficient_evidence"]
    assert report["classes"]["rare"]["insufficient_evidence"]

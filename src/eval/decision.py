"""Métriques de décision et de calibration, par classe.

L'AUROC mesure le classement, indépendamment de tout seuil. Un outil de
dépistage prend pourtant des décisions oui/non : il faut un seuil par
maladie, choisi sur la validation (jamais sur le test), et des mesures de ce
que ces décisions donnent (sensibilité, spécificité, valeurs prédictives,
faux positifs et faux négatifs). La calibration vérifie qu'une probabilité
de 0,8 correspond bien à ~80 % de malades.

Politiques de seuil :
- `f1` : seuil qui maximise le F1 sur la validation ;
- `sensitivity` : seuil le plus haut qui garde une sensibilité d'au moins
  `target` sur la validation (dépistage : rater peu de malades, au prix de
  fausses alertes).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import average_precision_score

MIN_POSITIVES_FOR_THRESHOLD = 10  # en dessous, un seuil choisi sur la validation n'est pas fiable


def choose_threshold(y: np.ndarray, p: np.ndarray, policy: str = "f1", target: float = 0.9) -> float:
    """Seuil de décision (score >= seuil -> positif) choisi sur (y, p)."""
    if y.sum() == 0:
        return float("nan")
    candidates = np.unique(p)[::-1]  # du plus strict au plus permissif
    positives = y.sum()
    best, best_value = float(candidates[-1]), -1.0
    for threshold in candidates:
        predicted = p >= threshold
        tp = (predicted & (y == 1)).sum()
        sensitivity = tp / positives
        if policy == "sensitivity":
            if sensitivity >= target:
                return float(threshold)
            continue
        if policy != "f1":
            raise ValueError(f"politique inconnue : {policy!r}")
        precision = tp / max(predicted.sum(), 1)
        f1 = 2 * precision * sensitivity / max(precision + sensitivity, 1e-12)
        if f1 > best_value:
            best, best_value = float(threshold), f1
    return best


def decision_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    """Confusion et métriques dérivées au seuil donné."""
    predicted = p >= threshold
    positive = y == 1
    tp, fp = int((predicted & positive).sum()), int((predicted & ~positive).sum())
    fn, tn = int((~predicted & positive).sum()), int((~predicted & ~positive).sum())

    def ratio(a: int, b: int) -> float | None:
        return a / b if b else None

    sensitivity, precision = ratio(tp, tp + fn), ratio(tp, tp + fp)
    f1 = (
        2 * precision * sensitivity / (precision + sensitivity)
        if precision is not None and sensitivity is not None and precision + sensitivity > 0
        else 0.0
    )
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "sensitivity": sensitivity,
        "specificity": ratio(tn, tn + fp),
        "ppv": precision,
        "npv": ratio(tn, tn + fn),
        "f1": f1,
    }


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """ECE à classes d'égal effectif : écart moyen |fréquence observée −
    probabilité moyenne| pondéré par la taille des classes. Des classes
    d'égal effectif évitent des cases vides quand les probabilités sont
    toutes proches de 0 (maladies rares)."""
    order = np.argsort(p)
    error = 0.0
    for chunk in np.array_split(order, n_bins):
        if len(chunk):
            error += len(chunk) * abs(y[chunk].mean() - p[chunk].mean())
    return float(error / len(p))


def reliability_curve(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> list[dict]:
    order = np.argsort(p)
    return [
        {"mean_probability": float(p[c].mean()), "observed_rate": float(y[c].mean()), "n": int(len(c))}
        for c in np.array_split(order, n_bins)
        if len(c)
    ]


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1 - 1e-12)  # écrêtage minimal : préserve l'ordre des scores extrêmes
    return np.log(p / (1 - p))


def fit_temperature(y: np.ndarray, p: np.ndarray) -> float:
    """Température unique (1 paramètre, toutes classes) qui minimise la
    log-vraisemblance sur (y, p) : p' = sigmoïde(logit(p) / T). Ne change
    pas le classement (AUROC), seulement l'échelle des probabilités ; un
    seul paramètre évite de surajuster les maladies rares."""
    z = _logit(p)

    def nll(log_t: float) -> float:
        q = np.clip(1 / (1 + np.exp(-z / np.exp(log_t))), 1e-7, 1 - 1e-7)
        return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))

    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))


def apply_temperature(p: np.ndarray, temperature: float) -> np.ndarray:
    return 1 / (1 + np.exp(-_logit(p) / temperature))


def per_class_report(
    y_val: np.ndarray,
    p_val: np.ndarray,
    y_test: np.ndarray,
    p_test: np.ndarray,
    class_names: list[str],
    target_sensitivity: float = 0.9,
) -> dict:
    """Seuils choisis sur la validation, appliqués au test ; AP et
    calibration (avant / après température ajustée sur la validation)."""
    temperature = fit_temperature(y_val, p_val)
    p_test_cal = apply_temperature(p_test, temperature)
    report = {"temperature": temperature, "classes": {}}
    for c, name in enumerate(class_names):
        yv, pv, yt, pt = y_val[:, c], p_val[:, c], y_test[:, c], p_test[:, c]
        entry = {
            "positives_val": int(yv.sum()),
            "positives_test": int(yt.sum()),
            "insufficient_evidence": bool(min(yv.sum(), yt.sum()) < MIN_POSITIVES_FOR_THRESHOLD),
            "brier_test": brier_score(yt, pt),
            "brier_test_calibrated": brier_score(yt, p_test_cal[:, c]),
            "ece_test": expected_calibration_error(yt, pt),
            "ece_test_calibrated": expected_calibration_error(yt, p_test_cal[:, c]),
            "prevalence_test": float(yt.mean()),
        }
        if yt.sum() > 0:
            entry["average_precision_test"] = float(average_precision_score(yt, pt))
        entry["at_0.5"] = decision_metrics(yt, pt, 0.5)
        for policy in ("f1", "sensitivity"):
            if yv.sum() == 0:
                continue
            threshold = choose_threshold(yv, pv, policy, target_sensitivity)
            entry[f"threshold_{policy}"] = decision_metrics(yt, pt, threshold)
        report["classes"][name] = entry
    return report

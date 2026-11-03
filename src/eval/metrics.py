"""Métriques d'évaluation pour la classification multi-label d'images médicales."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import f1_score, roc_auc_score


def evaluate_multilabel(y_true: np.ndarray, y_prob: np.ndarray, class_names: list[str]) -> dict:
    per_class_auroc = {}
    for i, name in enumerate(class_names):
        if len(np.unique(y_true[:, i])) < 2:
            continue  # AUROC non défini si une seule classe présente
        per_class_auroc[name] = float(roc_auc_score(y_true[:, i], y_prob[:, i]))

    macro_auroc = float(np.mean(list(per_class_auroc.values()))) if per_class_auroc else float("nan")
    y_pred = (y_prob >= 0.5).astype(int)

    return {
        "auroc_macro": macro_auroc,
        "auroc_per_class": per_class_auroc,
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "positives_per_class": {name: int(y_true[:, i].sum()) for i, name in enumerate(class_names)},
    }


def _auroc(y: np.ndarray, p: np.ndarray) -> float:
    """AUROC par les rangs (statistique de Mann-Whitney) : équivalent à
    roc_auc_score, assez rapide pour des milliers de tirages bootstrap."""
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(p)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _group_resamples(groups: list[str], n_boot: int, seed: int):
    """Tirages avec remise des groupes (patients) : les images d'un même
    patient sont corrélées, les tirer une à une rétrécirait les intervalles."""
    labels, inverse = np.unique(np.asarray(groups), return_inverse=True)
    members = [np.flatnonzero(inverse == g) for g in range(len(labels))]
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        yield np.concatenate([members[g] for g in rng.integers(0, len(labels), len(labels))])


def _per_class(y_true: np.ndarray, y_prob: np.ndarray, idx: np.ndarray, columns: list[int]) -> np.ndarray:
    return np.array([_auroc(y_true[idx, c], y_prob[idx, c]) for c in columns])


def bootstrap_auroc_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    groups: list[str],
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """Intervalles de confiance à 95 % (percentiles) de l'AUROC macro et par
    classe, par bootstrap sur les groupes. Classes = celles qui ont une AUROC
    sur l'échantillon complet, comme dans evaluate_multilabel."""
    columns = [i for i in range(len(class_names)) if len(np.unique(y_true[:, i])) == 2]
    draws = np.array([_per_class(y_true, y_prob, idx, columns) for idx in _group_resamples(groups, n_boot, seed)])
    low, high = np.nanpercentile(draws, [2.5, 97.5], axis=0)
    macro = np.nanmean(draws, axis=1)
    return {
        "auroc_macro_ci": [float(np.percentile(macro, 2.5)), float(np.percentile(macro, 97.5))],
        "auroc_per_class_ci": {
            class_names[c]: [float(lo), float(hi)] for c, lo, hi in zip(columns, low, high, strict=True)
        },
    }


def paired_bootstrap_diff(
    y_true: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    class_names: list[str],
    groups: list[str],
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """Différence d'AUROC B - A sur les mêmes images, mêmes tirages pour les
    deux : la variance commune aux deux modèles s'annule, l'intervalle est
    bien plus étroit que la différence de deux intervalles séparés."""
    columns = [i for i in range(len(class_names)) if len(np.unique(y_true[:, i])) == 2]
    full = np.arange(len(y_true))
    point = _per_class(y_true, prob_b, full, columns) - _per_class(y_true, prob_a, full, columns)
    draws = np.array(
        [
            _per_class(y_true, prob_b, idx, columns) - _per_class(y_true, prob_a, idx, columns)
            for idx in _group_resamples(groups, n_boot, seed)
        ]
    )
    low, high = np.nanpercentile(draws, [2.5, 97.5], axis=0)
    macro = np.nanmean(draws, axis=1)
    result = {
        name: {"diff": float(d), "ci": [float(lo), float(hi)]}
        for name, d, lo, hi in zip([class_names[c] for c in columns], point, low, high, strict=True)
    }
    result["macro"] = {
        "diff": float(np.mean(point)),
        "ci": [float(np.percentile(macro, 2.5)), float(np.percentile(macro, 97.5))],
    }
    return result

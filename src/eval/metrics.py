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


def _mann_whitney_pairs(y: np.ndarray, p: np.ndarray, strata: np.ndarray) -> tuple[float, float]:
    """Paires (positif, négatif) bien ordonnées et nombre de paires, en ne
    comparant que des images de la même strate."""
    ordered, pairs = 0.0, 0.0
    for stratum in np.unique(strata):
        mask = strata == stratum
        n_pos = y[mask].sum()
        n_neg = mask.sum() - n_pos
        if n_pos == 0 or n_neg == 0:
            continue
        ranks = rankdata(p[mask])
        ordered += ranks[y[mask] == 1].sum() - n_pos * (n_pos + 1) / 2
        pairs += n_pos * n_neg
    return ordered, pairs


def stratified_auroc(y: np.ndarray, p: np.ndarray, strata) -> float:
    """AUROC « à strate égale » : probabilité qu'un positif soit classé
    au-dessus d'un négatif **de la même strate** (ex. même format d'image).
    L'écart avec l'AUROC globale montre une dépendance aux strates, sans en
    identifier la cause : les paires comparées et leurs populations diffèrent."""
    ordered, pairs = _mann_whitney_pairs(np.asarray(y), np.asarray(p), np.asarray(strata))
    return float(ordered / pairs) if pairs else float("nan")


def stratification_gap(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    strata,
    groups: list[str],
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """Par classe : AUROC globale, AUROC à strate égale, écart (globale −
    strate égale) avec son IC à 95 % par bootstrap sur les groupes, et part
    des paires positif/négatif comparables à strate égale."""
    strata = np.asarray(strata)
    result = {}
    for c, name in enumerate(class_names):
        y, p = y_true[:, c], y_prob[:, c]
        if len(np.unique(y)) < 2:
            continue
        pooled, within = _auroc(y, p), stratified_auroc(y, p, strata)
        draws = [
            _auroc(y[idx], p[idx]) - stratified_auroc(y[idx], p[idx], strata[idx])
            for idx in _group_resamples(groups, n_boot, seed)
        ]
        low, high = np.nanpercentile(draws, [2.5, 97.5])
        n_pos = y.sum()
        result[name] = {
            "auroc": pooled,
            "auroc_within_strata": within,
            "gap": pooled - within,
            "gap_ci": [float(low), float(high)],
            "comparable_pairs": float(_mann_whitney_pairs(y, p, strata)[1] / (n_pos * (len(y) - n_pos))),
            "positives": int(n_pos),
        }
    return result

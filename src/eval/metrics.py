"""Métriques d'évaluation pour la classification multi-label d'images médicales."""
from __future__ import annotations

import numpy as np
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
    }

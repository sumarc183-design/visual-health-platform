"""Validation des archives de prédictions avant tout calcul ou statut « vérifié »."""

from __future__ import annotations

import numpy as np


def validate_predictions(pred: dict, name: str = "prédictions") -> None:
    """Clés uniques, dimensions cohérentes, probabilités finies dans [0, 1],
    étiquettes binaires. Lève ValueError sinon."""
    keys = [str(k) for k in np.asarray(pred["keys"]).tolist()]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{name} : {len(keys) - len(set(keys))} clé(s) en double")
    probs = np.asarray(pred["probs"], dtype=np.float64)
    expected = (len(keys), len(pred["class_names"]))
    if probs.shape != expected:
        raise ValueError(f"{name} : probabilités de forme {probs.shape}, attendu {expected}")
    if not np.isfinite(probs).all() or probs.min() < 0 or probs.max() > 1:
        raise ValueError(f"{name} : probabilités non finies ou hors de [0, 1]")
    if "y_true" in pred:
        y = np.asarray(pred["y_true"])
        if y.shape != expected or not np.isin(y, (0, 1)).all():
            raise ValueError(f"{name} : étiquettes de forme {y.shape} ou non binaires")

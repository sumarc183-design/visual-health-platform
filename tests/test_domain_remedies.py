"""Vérifie l'effacement linéaire d'un concept (LEACE) de scripts/domain_remedies.py."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from scripts.domain_remedies import leace_eraser


def test_leace_removes_the_device_but_keeps_an_independent_disease_signal():
    """Embeddings = appareil (3 décalages) + maladie (une autre direction) +
    bruit. Après effacement, l'appareil n'est plus prévisible linéairement ;
    la maladie, indépendante de l'appareil, le reste."""
    rng = np.random.default_rng(0)
    n, d = 3000, 20
    device = rng.integers(0, 3, n)
    disease = rng.integers(0, 2, n)
    offsets = rng.normal(0, 3, (3, d))
    X = offsets[device] + np.outer(disease, np.eye(d)[0]) * 2 + rng.normal(0, 1, (n, d))
    train, test = np.arange(n) < 2000, np.arange(n) >= 2000

    erase = leace_eraser(X[train], device[train])
    X_erased = erase(X)

    def device_score(features):
        clf = LogisticRegression(max_iter=2000).fit(features[train], device[train])
        return balanced_accuracy_score(device[test], clf.predict(features[test]))

    def disease_auroc(features):
        clf = LogisticRegression(max_iter=2000).fit(features[train], disease[train])
        return roc_auc_score(disease[test], clf.decision_function(features[test]))

    assert device_score(X) > 0.95
    assert device_score(X_erased) < 0.45  # hasard = 1/3
    assert disease_auroc(X_erased) > 0.85

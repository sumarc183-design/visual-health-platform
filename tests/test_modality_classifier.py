"""Routeur de modalité : tête linéaire, seuil de confiance et sauvegarde."""

import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.pipeline.modality_classifier import MODALITIES, ModalityHead, linear_from_sklearn


def _fitted_head(threshold=0.9):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 8)) * 3 + 5
    y = np.repeat([0, 1, 2], 100)
    X[:, 0] += 4 * y
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(max_iter=2000).fit(scaler.transform(X), y)
    weight, bias = linear_from_sklearn(scaler, model)
    head = ModalityHead(8, threshold)
    head.linear.weight.data[:] = torch.from_numpy(weight)
    head.linear.bias.data[:] = torch.from_numpy(bias)
    return head, scaler, model, X


def test_folded_head_matches_scikit_learn():
    head, scaler, model, X = _fitted_head()
    expected = model.predict_proba(scaler.transform(X))
    assert np.allclose(head.probabilities(X), expected, atol=1e-4)


def test_low_confidence_is_routed_to_unknown():
    head = ModalityHead(2, threshold=0.9)
    probs = np.array([[0.95, 0.03, 0.02], [0.6, 0.35, 0.05], [0.01, 0.97, 0.02]])
    assert head.decide(probs) == ["fundus", "unknown", "dermoscopy"]


def test_save_and_load_keep_weights_threshold_and_metadata(tmp_path):
    head, _, _, X = _fitted_head(threshold=0.8)
    head.save(tmp_path / "router.pt", {"backbone": "dinov2", "checkpoint_sha256": "abc"})
    loaded, artifact = ModalityHead.load(tmp_path / "router.pt")
    assert loaded.threshold == 0.8 and artifact["checkpoint_sha256"] == "abc"
    assert loaded.predict(X) == head.predict(X)
    assert set(loaded.predict(X)) <= set(MODALITIES)


def test_load_rejects_other_classes(tmp_path):

    torch.save({"state_dict": {}, "threshold": 0.9, "classes": ["a", "b"]}, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="classes"):
        ModalityHead.load(tmp_path / "bad.pt")


def test_corrupted_router_is_refused(tmp_path):
    """Constat G (audit du 19 nov.) : pas d'avis du routeur avec des probabilités NaN."""
    head = ModalityHead(4)
    with torch.no_grad():
        head.linear.weight.fill_(float("nan"))
    head.save(tmp_path / "nan.pt", {})
    with pytest.raises(ValueError, match="non finis"):
        ModalityHead.load(tmp_path / "nan.pt")
    for bad in (float("nan"), 0.0, 1.5):
        ModalityHead(4, bad).save(tmp_path / "seuil.pt", {})
        with pytest.raises(ValueError, match="seuil"):
            ModalityHead.load(tmp_path / "seuil.pt")
    with pytest.raises(ValueError, match="non finies"):
        ModalityHead(4).probabilities(np.full((1, 4), np.nan))

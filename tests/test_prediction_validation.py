"""Archives de prédictions invalides refusées avant tout statut « vérifié » (audit du 9 nov.)."""

import numpy as np
import pytest
import torch

from scripts import decision_metrics
from scripts.compare_predictions import align
from src.eval.predictions import validate_predictions


def _pred(keys=("a", "b", "c"), probs=None):
    probs = np.array([[0.1, 0.9], [0.8, 0.2], [0.4, 0.6]]) if probs is None else probs
    return {
        "keys": np.array(keys),
        "probs": probs,
        "y_true": np.array([[0, 1], [1, 0], [0, 1]]),
        "class_names": np.array(["X", "Y"]),
    }


@pytest.mark.parametrize(
    "bad",
    [
        _pred(keys=("a", "a", "c")),
        _pred(probs=np.array([[0.1, np.nan], [0.8, 0.2], [0.4, 0.6]])),
        _pred(probs=np.array([[0.1, 1.2], [0.8, 0.2], [0.4, 0.6]])),
        _pred(probs=np.zeros((2, 2))),
    ],
)
def test_invalid_archives_are_refused(bad):
    with pytest.raises(ValueError):
        validate_predictions(bad)


def test_comparison_refuses_duplicated_keys():
    with pytest.raises(SystemExit, match="double"):
        align(_pred(), _pred(keys=("a", "b", "b")))
    assert align(_pred(), _pred()).shape == (3, 2)


def test_head_link_refuses_nan_predictions(tmp_path, monkeypatch):
    head = {
        "class_names": ["X", "Y"],
        "image_size": 224,
        "state_dict": {"fc.weight": torch.zeros(2, 4), "fc.bias": torch.zeros(2)},
    }
    torch.save(head, tmp_path / "head.pt")
    monkeypatch.setattr(decision_metrics, "load_embeddings", lambda *a, **k: np.zeros((3, 4)))
    nan = _pred(probs=np.array([[0.5, np.nan], [0.5, 0.5], [0.5, 0.5]]))
    report = {}
    with pytest.raises(ValueError):
        decision_metrics.link_head(
            report, tmp_path / "head.pt", tmp_path, "rfmid", {"val": nan, "test": nan}, ["X", "Y"]
        )
    assert "head_sha256" not in report
    ok = _pred(probs=np.full((3, 2), 0.5))
    decision_metrics.link_head(report, tmp_path / "head.pt", tmp_path, "rfmid", {"val": ok, "test": ok}, ["X", "Y"])
    assert report["head_check_max_abs_diff"] == {"val": 0.0, "test": 0.0}

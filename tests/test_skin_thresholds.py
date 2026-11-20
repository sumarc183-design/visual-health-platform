"""Seuils de la tête peau commitée : liés à elle, lisibles par predict.py."""

from pathlib import Path

import torch

from src.pipeline.inference import load_thresholds

ROOT = Path(__file__).resolve().parents[1]


def test_skin_thresholds_are_bound_to_the_committed_head():
    head = torch.load(ROOT / "heads/isic2019_dinov2b_head.pt", map_location="cpu", weights_only=True)
    path = ROOT / "results/decision_isic2019_dinov2b.json"
    for policy in ("f1", "sensitivity"):
        thresholds, verified = load_thresholds(path, head["class_names"], policy, head=head)
        assert verified and set(thresholds) == set(head["class_names"])
        assert all(t is not None and 0.0 < t < 1.0 for t in thresholds.values())


def test_served_panderm_head_and_thresholds_are_bound():
    """Tête servie depuis le 20 nov. (P) : seuils liés, mélanome au seuil de l'étude."""
    head = torch.load(ROOT / "heads/isic2019_panderm_head.pt", map_location="cpu", weights_only=True)
    assert head["backbone"] == "panderm" and head["modality"] == "dermoscopy"
    path = ROOT / "results/decision_isic2019_panderm.json"
    thresholds, verified = load_thresholds(path, head["class_names"], "sensitivity", head=head)
    assert verified and all(t is not None and 0.0 < t < 1.0 for t in thresholds.values())
    assert abs(thresholds["MEL"] - 0.061183370649814606) < 1e-12


def test_predict_skin_preset_points_to_the_served_head():
    import predict

    assert predict.SKIN["head"] == Path("heads/isic2019_panderm_head.pt")
    assert predict.SKIN["thresholds"] == Path("results/decision_isic2019_panderm.json")
    assert predict.SKIN["policy"] == "sensitivity"

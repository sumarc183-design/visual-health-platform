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

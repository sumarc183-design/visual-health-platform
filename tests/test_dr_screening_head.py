"""Réglage « dépistage RD » : la tête commitée et son seuil sont liés, et `--dr-screening`
les charge avec la politique sensitivity (sans charger Green)."""

import sys
from pathlib import Path

import pytest
import torch

import predict
from src.pipeline.inference import load_thresholds

ROOT = Path(__file__).resolve().parents[1]
HEAD = ROOT / predict.DR_SCREENING["head"]
THRESHOLDS = ROOT / predict.DR_SCREENING["thresholds"]


def test_committed_head_and_threshold_are_bound():
    head = torch.load(HEAD, map_location="cpu", weights_only=True)
    assert head["class_names"] == ["DR"] and head["backbone"] == "green" and head["preprocess"] == "crop-mask"
    thresholds, verified = load_thresholds(THRESHOLDS, head["class_names"], "sensitivity", head=head)
    assert verified and 0.0 < thresholds["DR"] < 0.5
    assert load_thresholds(THRESHOLDS, head["class_names"], "f1", head=head)[0] == {"DR": None}


class Stop(Exception):
    pass


def test_cli_preset_fills_head_thresholds_and_policy(monkeypatch):
    seen = {}

    def fake_predictor(head, checkpoint, thresholds, policy, **kwargs):
        seen.update(head=head, checkpoint=checkpoint, thresholds=thresholds, policy=policy)
        raise Stop

    monkeypatch.setattr(predict, "Predictor", fake_predictor)
    monkeypatch.setattr(sys, "argv", ["predict.py", "image.png", "--dr-screening"])
    with pytest.raises(Stop):
        predict.main()
    assert seen == predict.DR_SCREENING


def test_cli_preset_refuses_the_f1_policy(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["predict.py", "image.png", "--dr-screening", "--policy", "f1"])
    with pytest.raises(SystemExit):
        predict.main()


def test_cli_requires_a_head_without_preset(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["predict.py", "image.png"])
    with pytest.raises(SystemExit):
        predict.main()

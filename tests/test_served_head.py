"""Liaison stricte tête/seuils dans les scripts d'analyse (audit du 15 nov., constat 3)."""

import json

import pytest
import torch

from src.models.classifier_head import MultiLabelHead, head_contract, head_sha256
from src.pipeline.served_head import load_served_head


def _head(tmp_path, name, seed):
    torch.manual_seed(seed)
    artifact = {
        "state_dict": MultiLabelHead(4, 1).state_dict(),
        "class_names": ["DR"],
        "backbone": "green",
        "image_size": 392,
        "preprocess": "crop-mask",
        "preprocess_version": 1,
        "checkpoint_sha256": "abcd",
    }
    path = tmp_path / name
    torch.save(artifact, path)
    return path, artifact


def _thresholds(tmp_path, artifact):
    report = {
        "classes": {"DR": {"insufficient_evidence": False, "threshold_sensitivity": {"threshold": 0.3}}},
        "head_sha256": head_sha256(artifact["state_dict"]),
        "head_contract": head_contract(artifact),
    }
    path = tmp_path / "seuils.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_served_pair_is_accepted_and_identity_recorded(tmp_path):
    head, artifact = _head(tmp_path, "tete.pt", 0)
    served = load_served_head(head, "abcd", _thresholds(tmp_path, artifact))
    assert served.thresholds["classes"]["DR"]["threshold_sensitivity"]["threshold"] == 0.3
    assert served.identity["head_sha256"] == head_sha256(artifact["state_dict"])
    assert served.probabilities(torch.zeros(2, 4)).shape == (2, 1)


def test_other_head_with_same_dimensions_is_refused(tmp_path):
    _, artifact = _head(tmp_path, "tete.pt", 0)
    thresholds = _thresholds(tmp_path, artifact)
    other, _ = _head(tmp_path, "autre.pt", 1)
    with pytest.raises(ValueError, match="autre tête"):
        load_served_head(other, "abcd", thresholds)


def test_other_checkpoint_is_refused(tmp_path):
    head, _ = _head(tmp_path, "tete.pt", 0)
    with pytest.raises(ValueError, match="autre modèle figé"):
        load_served_head(head, "efgh")

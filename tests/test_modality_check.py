import numpy as np
import pytest
import torch

from src.pipeline.inference import Prediction
from src.pipeline.modality_check import opinion
from src.pipeline.modality_classifier import ModalityHead


def head_favouring(index: int) -> ModalityHead:
    head = ModalityHead(4)
    with torch.no_grad():
        head.linear.weight.zero_()
        head.linear.bias.zero_()
        head.linear.bias[index] = 10.0
    return head


def test_agreement_gives_no_warning():
    seen = opinion(head_favouring(0), np.zeros((1, 4)))
    assert seen.decision == "fundus" and seen.agrees and seen.warning is None


def test_disagreement_warns_without_blocking():
    seen = opinion(head_favouring(1), np.zeros((1, 4)))
    assert seen.decision == "dermoscopy" and not seen.agrees
    assert "déclaré fond d'œil" in seen.warning and "quand même" in seen.warning


def test_low_confidence_is_reported_as_unknown():
    head = head_favouring(0)
    with torch.no_grad():
        head.linear.bias.zero_()  # trois classes à 1/3 : sous le seuil de 0,9
    seen = opinion(head, np.zeros((1, 4)))
    assert seen.decision == "unknown" and "type non reconnu" in seen.warning


def test_declared_type_must_be_a_specialist():
    with pytest.raises(ValueError):
        opinion(head_favouring(0), np.zeros((1, 4)), declared="unknown")


def test_prediction_dict_includes_modality_only_when_checked():
    p = Prediction(image="a.jpg", quality={}, findings={})
    assert "modality" not in p.to_dict()
    p.modality = {"declared": "fundus", "decision": "fundus"}
    assert p.to_dict()["modality"]["decision"] == "fundus"

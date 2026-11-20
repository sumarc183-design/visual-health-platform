"""Validation qualité sur DDR : empreinte des poids et taux au seuil."""

import numpy as np
import torch

from scripts.quality_ddr_embeddings import tensor_digest
from scripts.quality_ddr_validation import flags_report


def test_tensor_digest_ignores_serialization_but_not_values(tmp_path):
    state = {"b": torch.ones(2, 3), "a": torch.arange(4.0)}
    torch.save(state, tmp_path / "x.pth")
    torch.save({"state_dict": dict(reversed(list(state.items())))}, tmp_path / "y.pth")
    assert tensor_digest(tmp_path / "x.pth") == tensor_digest(tmp_path / "y.pth")
    state["a"][0] = 1e-3
    torch.save(state, tmp_path / "z.pth")
    assert tensor_digest(tmp_path / "z.pth") != tensor_digest(tmp_path / "x.pth")


def test_flags_report_counts_detections_and_false_alerts():
    y = np.array([1, 1, 0, 0, 0, 0])
    p = np.array([0.99, 0.10, 0.995, 0.2, 0.1, 0.0])
    r = flags_report(y, p, 0.989, [str(i) for i in range(6)])
    assert r["ungradable_flagged"] == 0.5 and r["false_alert_rate"] == 0.25
    low, high = r["false_alert_rate_ci"]
    assert 0.0 <= low <= 0.25 <= high <= 1.0

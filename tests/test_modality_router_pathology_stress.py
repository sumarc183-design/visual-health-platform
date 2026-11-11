import numpy as np

from scripts.modality_router_pathology_stress import burden_bin, rates


def test_burden_bin_groups_three_or_more():
    assert [burden_bin(n) for n in (0, 1, 2, 3, 7)] == ["0", "1", "2", "3+", "3+"]


def test_rates_counts_dermoscopy_with_exact_interval():
    r = rates(np.array(["fundus", "dermoscopy", "unknown", "fundus"]))
    assert r["decisions"] == {"fundus": 2, "dermoscopy": 1, "unknown": 1}
    assert r["to_dermoscopy"]["count"] == 1 and r["to_dermoscopy"]["value"] == 0.25
    low, high = r["to_dermoscopy"]["ci"]
    assert 0 < low < 0.25 < high < 1

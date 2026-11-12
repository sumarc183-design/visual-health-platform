import numpy as np

from scripts.modality_router_skin_photo_stress import to_fundus


def test_no_fundus_meets_criterion_with_enough_images():
    r = to_fundus(np.array(["dermoscopy"] * 3000 + ["unknown"] * 3000))
    assert r["count"] == 0 and r["met"]
    assert r["ci"][0] == 0 and r["ci"][1] < 0.005


def test_fundus_errors_fail_criterion():
    r = to_fundus(np.array(["fundus"] * 40 + ["unknown"] * 960))
    assert r["count"] == 40 and not r["met"]

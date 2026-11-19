"""Constat E (audit du 19 nov.) : les empreintes des entrées PanDerm détectent un changement à effectif égal."""

import numpy as np

from scripts.panderm_snapshot import differences, digest


def test_digest_detects_order_and_label_changes():
    keys, groups = ["a", "b", "c"], ["p1", "p1", "p2"]
    assert digest(keys) == digest(list(keys))
    assert digest(keys) != digest(["b", "a", "c"])  # ordre
    assert digest(groups) != digest(["p1", "p2", "p2"])  # patient changé, même effectif
    y = np.array([[1, 0], [0, 1]], np.float32)
    assert digest(y) != digest(y[::-1].copy())
    assert digest(np.array(["x", float("nan")], object)) == digest(np.array(["x", float("nan")], object))


def test_differences_lists_changed_inputs():
    old = {"bases": {"derm12345": {"groups": "aa", "n": 3}}, "annotations": {"m.csv": "f1"}}
    new = {"bases": {"derm12345": {"groups": "bb", "n": 3}}, "annotations": {"m.csv": "f1"}}
    assert differences(old, old) == []
    assert differences(old, new) == ["bases/derm12345/groups : aa → bb"]

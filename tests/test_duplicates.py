"""Vérifie le repérage des doublons d'images et les groupes indépendants."""

import numpy as np

from src.eval.duplicates import duplicate_report, exact_duplicate_groups, independent_groups, merge_groups


def test_exact_and_near_duplicates_are_grouped_with_patients():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(6, 16)).astype(np.float32)
    X[3] = X[0]  # doublon exact
    X[5] = X[1] + 1e-5  # quasi-doublon
    exact = exact_duplicate_groups(X)
    assert exact[3] == exact[0] and len(set(exact)) == 5
    patients = ["p0", "p1", "p2", "p2", "p4", "p5"]  # images 2 et 3 : même patient
    groups, near_pairs = independent_groups(X, patients)
    assert near_pairs == 2  # (1, 5) quasi identiques, et (0, 3) identiques comptent aussi
    assert groups[0] == groups[2] == groups[3]  # 0~3 (doublon) et 2~3 (patient) -> une seule unité
    assert groups[1] == groups[5] and groups[4] not in (groups[0], groups[1])


def test_duplicate_report_flags_contradictory_grades():
    groups = merge_groups(4, ["a", "a", "b", "c"])
    report = duplicate_report(groups, np.array([0, 2, 1, 1]))
    assert report == {
        "groups_with_several_images": 1,
        "images_in_such_groups": 2,
        "groups_with_different_grades": 1,
        "groups_with_contradictory_any_dr": 1,
    }

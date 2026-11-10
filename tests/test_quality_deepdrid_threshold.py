"""Seuil choisi pour une spécificité cible, sur la validation seulement."""

import numpy as np

from scripts.quality_deepdrid_threshold import threshold_for_specificity


def test_threshold_reaches_the_target_specificity_with_the_smallest_value():
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.92, 0.95])
    y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    t = threshold_for_specificity(scores, y, 0.9)
    assert t == 0.92  # 9 négatifs sur 10 sous le seuil ; 0,9 n'en laisserait que 8
    assert (scores[y == 0] < t).mean() >= 0.9
    assert threshold_for_specificity(scores, y, 1.0) == 0.95

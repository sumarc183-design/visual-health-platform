import pandas as pd

from scripts.skin_transfer_labeled import labeled


def test_labeled_keeps_only_mapped_diagnoses():
    d = pd.Series(["unknown", "melanoma", "nevus", "cafe-au-lait macule", "solar lentigo"])
    assert labeled(d).tolist() == [False, True, True, False, True]

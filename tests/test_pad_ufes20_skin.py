import numpy as np
import pandas as pd
import pytest

from scripts.pad_ufes20_skin import EVALUATED, cancer_score, targets


def test_targets_map_pad_classes_to_isic_columns():
    y = targets(pd.Series(["ACK", "SEK", "NEV", "BCC", "SCC", "MEL"]))
    assert y.shape == (6, len(EVALUATED)) and (y.sum(axis=1) == 1).all()
    assert y[0, EVALUATED.index("AK")] == 1 and y[1, EVALUATED.index("BKL")] == 1 and y[2, EVALUATED.index("NV")] == 1


def test_unknown_diagnostic_is_rejected():
    with pytest.raises(ValueError):
        targets(pd.Series(["BCC", "XYZ"]))


def test_cancer_score_sums_bcc_scc_mel():
    probs = np.zeros((1, len(EVALUATED)))
    for c, v in (("BCC", 0.1), ("SCC", 0.2), ("MEL", 0.3), ("NV", 0.9)):
        probs[0, EVALUATED.index(c)] = v
    assert np.isclose(cancer_score(probs)[0], 0.6)

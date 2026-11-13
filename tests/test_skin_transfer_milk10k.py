import numpy as np
import pandas as pd

from scripts.skin_transfer_milk10k import isic2020_targets, milk_targets
from src.data.isic_dataset import CLASS_NAMES


def test_milk_classes_map_to_isic2019_and_unmapped_are_negative():
    cols = ["AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF", "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"]
    gt = pd.DataFrame(np.eye(len(cols), dtype=int), columns=cols)
    y = milk_targets(gt)
    assert y[cols.index("AKIEC"), CLASS_NAMES.index("AK")] == 1
    assert y[cols.index("SCCKA"), CLASS_NAMES.index("SCC")] == 1
    for unmapped in ("BEN_OTH", "INF", "MAL_OTH"):
        assert y[cols.index(unmapped)].sum() == 0


def test_isic2020_labels():
    y = isic2020_targets(pd.Series(["melanoma", "nevus", "lentigo NOS", "unknown"]))
    assert y[0, CLASS_NAMES.index("MEL")] == 1 and y[1, CLASS_NAMES.index("NV")] == 1
    assert y[2, CLASS_NAMES.index("BKL")] == 1 and y[3].sum() == 0

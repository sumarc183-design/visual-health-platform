"""Protocole 2 du routeur : aucun groupe commun entre ajustement et contrôle."""

import numpy as np
from PIL import Image

from scripts.train_modality_router_v2 import Item, split_by_group


def test_split_never_shares_a_group_even_when_a_group_is_cut():
    items = [Item(f"k{g}_{i}", f"g{g}", lambda: Image.new("RGB", (4, 4))) for g in range(40) for i in range(3)]
    for seed in range(5):
        fit, control = split_by_group(items, n_control=10, n_fit=60, rng=np.random.default_rng(seed))
        assert len(control) == 10 and len(fit) == 60
        assert not {it.group for it in fit} & {it.group for it in control}

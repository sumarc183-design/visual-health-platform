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


def test_embed_records_source_fingerprints(tmp_path):
    import torch

    from scripts.train_modality_router_v2 import embed
    from src.models.embedding_cache import EmbeddingCache

    image = Image.fromarray(np.zeros((40, 40, 3), dtype=np.uint8))
    items = [Item(f"k{i}", f"g{i}", lambda: image, lambda i=i: f"empreinte{i}") for i in range(3)]
    cache = EmbeddingCache(tmp_path / "c.npz")
    embed(items, lambda x: torch.zeros(len(x), 4), cache)
    assert EmbeddingCache(tmp_path / "c.npz").sources == {f"k{i}": f"empreinte{i}" for i in range(3)}
    without = [Item("z", "z", lambda: image)]  # sans source : embedding gardé, aucune empreinte
    embed(without, lambda x: torch.zeros(len(x), 4), cache)
    assert "z" not in EmbeddingCache(tmp_path / "c.npz").sources


def test_embed_uses_the_backbone_normalization(tmp_path):
    """PanDerm normalise avec son propre écart-type (0,228) : embed doit l'utiliser."""
    import torch

    from scripts.train_modality_router_v2 import embed
    from src.models.embedding_cache import EmbeddingCache

    class Recorder:
        input_mean = torch.zeros(1, 3, 1, 1)
        input_std = torch.full((1, 3, 1, 1), 0.5)

        def __call__(self, x):
            self.seen = x
            return torch.zeros(len(x), 2)

    image = Image.fromarray(np.full((40, 40, 3), 255, dtype=np.uint8))  # pixels à 1,0 après mise à l'échelle
    backbone = Recorder()
    embed([Item("k", "g", lambda: image)], backbone, EmbeddingCache(tmp_path / "c.npz"))
    assert torch.allclose(backbone.seen, torch.full_like(backbone.seen, 2.0))

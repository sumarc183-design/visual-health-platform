import numpy as np

from scripts.isic2020_external import near_duplicates


def test_near_duplicates_flags_only_almost_identical_embeddings():
    rng = np.random.default_rng(0)
    reference = rng.normal(size=(50, 16)).astype(np.float32)
    X = np.vstack([reference[3] * 2.0 + 1e-4, rng.normal(size=16)]).astype(np.float32)
    assert near_duplicates(X, reference).tolist() == [True, False]

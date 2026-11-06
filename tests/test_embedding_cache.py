"""Vérifie le cache d'embeddings et son usage par run_baseline.py."""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from run_baseline import dataset_embeddings
from src.models.embedding_cache import ASSUMED_KEY, CacheMismatchError, EmbeddingCache, cache_file_name


class TinyDataset(torch.utils.data.Dataset):
    """Images constantes dont la valeur identifie l'image."""

    class_names = ["x", "y"]

    def __init__(self, n: int, offset: int = 0):
        self.values = np.arange(offset, offset + n, dtype=np.float32)
        self.df = pd.DataFrame({"x": self.values % 2, "y": 1 - self.values % 2})
        self.image_keys = [f"img{int(v)}" for v in self.values]

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, idx: int):
        image = torch.full((3, 4, 4), float(self.values[idx]))
        return image, torch.tensor(self.df.iloc[idx].to_numpy(dtype=np.float32))


class CountingBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.seen += len(x)
        return x.mean(dim=(2, 3))  # (B, 3)


def test_cache_roundtrip(tmp_path):
    cache = EmbeddingCache(tmp_path / "c.npz")
    assert cache.missing(["a", "b"]) == ["a", "b"]
    cache.add(["a", "b"], np.array([[1.0, 2.0], [3.0, 4.0]]))
    cache.save()
    reloaded = EmbeddingCache(tmp_path / "c.npz")
    assert reloaded.missing(["a", "b", "c"]) == ["c"]
    assert np.array_equal(reloaded.get(["b", "a"]), [[3.0, 4.0], [1.0, 2.0]])
    assert [p.name for p in tmp_path.iterdir()] == ["c.npz"]  # pas de fichier temporaire laissé


def test_cache_file_name_depends_on_what_changes_the_embedding():
    names = {
        cache_file_name("odir", "none", 224, "ckpt/RETFound_mae_natureCFP.pth"),
        cache_file_name("odir", "crop-mask", 224, "ckpt/RETFound_mae_natureCFP.pth"),
        cache_file_name("odir", "none", 256, "ckpt/RETFound_mae_natureCFP.pth"),
        cache_file_name("rfmid", "none", 224, "ckpt/RETFound_mae_natureCFP.pth"),
    }
    assert len(names) == 4


def test_dataset_embeddings_only_runs_missing_images(tmp_path):
    backbone = CountingBackbone()
    cache = EmbeddingCache(tmp_path / "c.npz")
    kwargs = dict(batch_size=4, device="cpu", num_workers=0)

    X1, y1 = dataset_embeddings(TinyDataset(6), "train", lambda: backbone, cache, kwargs)
    assert backbone.seen == 6 and X1.shape == (6, 3)

    # 4 images déjà vues + 3 nouvelles : seules les nouvelles passent dans le backbone
    X2, y2 = dataset_embeddings(TinyDataset(7, offset=2), "val", lambda: backbone, EmbeddingCache(cache.path), kwargs)
    assert backbone.seen == 6 + 3
    assert np.allclose(X2[:4], X1[2:])
    assert np.array_equal(y2, TinyDataset(7, offset=2).df[["x", "y"]].to_numpy(dtype=np.float32))


def test_dataset_embeddings_does_not_load_backbone_when_all_cached(tmp_path):
    cache = EmbeddingCache(tmp_path / "c.npz")
    kwargs = dict(batch_size=4, device="cpu", num_workers=0)
    dataset_embeddings(TinyDataset(3), "train", CountingBackbone, cache, kwargs)

    def must_not_load():
        raise AssertionError("le backbone ne doit pas être chargé si tout est en cache")

    X, _ = dataset_embeddings(TinyDataset(3), "train", must_not_load, EmbeddingCache(cache.path), kwargs)
    assert X.shape == (3, 3)


META = {"checkpoint_sha256": "abc", "preprocess": "crop-mask", "preprocess_version": 1, "image_size": 224}


def test_cache_refuses_embeddings_from_another_configuration(tmp_path):
    cache = EmbeddingCache(tmp_path / "c.npz", meta=META)
    cache.add(["a"], np.ones((1, 2)))
    cache.save()
    assert EmbeddingCache(tmp_path / "c.npz", meta=META).missing(["a"]) == []
    for changed in ({"checkpoint_sha256": "autre"}, {"preprocess_version": 2}, {"image_size": 256}):
        with pytest.raises(CacheMismatchError):
            EmbeddingCache(tmp_path / "c.npz", meta={**META, **changed})


def test_legacy_cache_is_adopted_then_protected(tmp_path):
    """Un cache sans empreinte (ancien format) reste utilisable ; l'empreinte
    est alors enregistrée, et un changement ultérieur est détecté."""
    legacy = EmbeddingCache(tmp_path / "c.npz")
    legacy.add(["a"], np.ones((1, 2)))
    legacy.save()
    assert EmbeddingCache(tmp_path / "c.npz", meta=META).missing(["a"]) == []
    with pytest.raises(CacheMismatchError):
        EmbeddingCache(tmp_path / "c.npz", meta={**META, "preprocess_version": 2})


def test_unknown_checkpoint_fingerprint_is_not_checked(tmp_path):
    """Checkpoint absent de la machine (tête seule, sur CPU) : None n'est pas comparé."""
    cache = EmbeddingCache(tmp_path / "c.npz", meta=META)
    cache.add(["a"], np.ones((1, 2)))
    cache.save()
    reloaded = EmbeddingCache(tmp_path / "c.npz", meta={**META, "checkpoint_sha256": None})
    assert reloaded.meta["checkpoint_sha256"] == "abc"


def test_malformed_caches_are_refused(tmp_path):
    """Clés en double (une ligne écraserait l'autre) et valeurs non finies : refusées."""
    np.savez(tmp_path / "dup.npz", keys=np.array(["a", "a"]), X=np.ones((2, 3)))
    with pytest.raises(CacheMismatchError, match="double"):
        EmbeddingCache(tmp_path / "dup.npz")
    np.savez(tmp_path / "nan.npz", keys=np.array(["a", "b"]), X=np.array([[1.0, np.nan], [0.0, 1.0]]))
    with pytest.raises(CacheMismatchError, match="finies"):
        EmbeddingCache(tmp_path / "nan.npz")


def test_adopted_metadata_stays_marked_as_not_verified(tmp_path):
    """Un cache qui ne déclare pas son checkpoint : l'empreinte demandée sert à
    détecter les changements, sans devenir une preuve de son origine."""
    path = tmp_path / "c.npz"
    np.savez(path, keys=np.array(["a"]), X=np.zeros((1, 2), np.float32), meta=np.array(json.dumps({"preprocess": "p"})))
    adopted = EmbeddingCache(path, meta={"preprocess": "p", "checkpoint_sha256": "abc"})
    assert adopted.meta[ASSUMED_KEY] == ["checkpoint_sha256"]
    again = EmbeddingCache(path, meta={"preprocess": "p", "checkpoint_sha256": "abc"})
    assert again.meta[ASSUMED_KEY] == ["checkpoint_sha256"]
    with pytest.raises(CacheMismatchError):
        EmbeddingCache(path, meta={"preprocess": "p", "checkpoint_sha256": "autre"})


def test_cache_with_verified_metadata_has_no_mark(tmp_path):
    cache = EmbeddingCache(tmp_path / "c.npz", meta={"preprocess": "p"})
    cache.add(["a"], np.zeros((1, 2)))
    cache.save()
    assert ASSUMED_KEY not in EmbeddingCache(tmp_path / "c.npz", meta={"preprocess": "p"}).meta

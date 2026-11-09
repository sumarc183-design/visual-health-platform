"""Analyse dermatologie de bout en bout sur des caches synthétiques (sans images)."""

import json
import sys

import numpy as np

from scripts import dermatology_analysis
from src.data.fundus_preprocess import PREPROCESS_VERSION
from src.data.isic_dataset import CLASS_NAMES, ISIC2019Dataset
from src.models.embedding_cache import CHECKPOINT_STEMS, KNOWN_CHECKPOINTS, EmbeddingCache
from tests.test_isic_dataset import _write


def test_analysis_compares_the_three_models_on_the_same_images(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    train = [(f"ISIC_{i:07d}", CLASS_NAMES[i % 2], f"L{i // 2}") for i in range(140)]
    test = [(f"ISIC_{9000000 + i}", CLASS_NAMES[i % 2], None) for i in range(40)]
    _write(tmp_path, "Training", train)
    _write(tmp_path, "Test", test, with_lesion=False)
    keys = [f"isic2019/{name}" for name, _, _ in train + test]
    labels = np.array([CLASS_NAMES.index(c) for _, c, _ in train + test])
    for backbone, signal in (("panderm", 3.0), ("dinov2-large", 1.0), ("dinov2", 0.5)):
        meta = {
            "checkpoint_sha256": KNOWN_CHECKPOINTS.get(backbone),
            "preprocess": "none",
            "preprocess_version": PREPROCESS_VERSION,
            "image_size": 224,
        }
        cache = EmbeddingCache(tmp_path / "emb" / f"isic2019_none_224_{CHECKPOINT_STEMS[backbone]}.npz", meta)
        X = rng.normal(size=(len(keys), 16)).astype(np.float32)
        X[:, 0] += signal * labels
        cache.add(keys, X)
        cache.save()
    out = tmp_path / "derm.json"
    argv = ["x", "--isic-root", str(tmp_path), "--embeddings", str(tmp_path / "emb"), "--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    dermatology_analysis.main()

    report = json.loads(out.read_text())
    n_val = len(ISIC2019Dataset(tmp_path, "val"))
    assert report["images"] == {"train": 140 - n_val, "val": n_val, "test": 40}
    test_auroc = {m: report["models"][m]["test"]["auroc_macro"] for m in dermatology_analysis.MODELS}
    assert test_auroc["PanDerm"] > test_auroc["DINOv2-B"]
    main = report["comparisons"]["PanDerm - DINOv2-L"]["test"]
    assert main["auroc_macro"]["diff"] > 0 and len(main["balanced_accuracy"]["ci"]) == 2
    archive = np.load(out.with_suffix(".npz"))
    assert archive["PanDerm_test"].shape == (40, len(CLASS_NAMES))

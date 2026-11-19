"""Constat C (audit du 19 nov.) : le vérificateur des empreintes ne conclut « ok » qu'après contrôle complet."""

import json

import numpy as np

from scripts.verify_cache_sources import main
from src.models.embedding_cache import EmbeddingCache, file_fingerprint


def _setup(tmp_path, base="derm12345", folder="images"):
    images = tmp_path / "raw" / base / folder
    images.mkdir(parents=True)
    for name in ("ISIC_1", "ISIC_2"):
        (images / f"{name}.jpg").write_bytes(name.encode())
    caches = tmp_path / "emb"
    cache = EmbeddingCache(caches / f"{base}_panderm_large.npz", meta={"x": 1})
    keys = [f"{base}/ISIC_1", f"{base}/ISIC_2"]
    cache.add(keys, np.ones((2, 3)), sources=[file_fingerprint(images / f"{k.split('/')[1]}.jpg") for k in keys])
    cache.save()
    return tmp_path / "raw", caches, images


def _run(tmp_path, data, caches, *extra):
    out = tmp_path / "bilan.json"
    code = main(["--data-root", str(data), "--caches", str(caches), "--out", str(out), *extra])
    return code, json.loads(out.read_text(encoding="utf-8"))


def test_derm12345_is_covered_and_modified_image_detected(tmp_path):
    data, caches, images = _setup(tmp_path)
    code, report = _run(tmp_path, data, caches)
    assert code == 0 and report["derm12345_panderm_large.npz"]["status"] == "ok"
    (images / "ISIC_2.jpg").write_bytes(b"autre")
    code, report = _run(tmp_path, data, caches)
    assert code == 1 and report["derm12345_panderm_large.npz"]["status"] == "IMAGES MODIFIÉES"


def test_missing_images_or_cache_folder_fail(tmp_path):
    data, caches, images = _setup(tmp_path)
    (images / "ISIC_2.jpg").unlink()
    code, report = _run(tmp_path, data, caches)
    assert code == 1 and report["derm12345_panderm_large.npz"]["status"] == "incomplet"
    code, report = _run(tmp_path, data, caches, "--allow-partial")
    assert code == 0 and report["derm12345_panderm_large.npz"]["status"] == "partiel"
    for p in images.iterdir():
        p.unlink()
    code, report = _run(tmp_path, data, caches, "--allow-partial")  # aucune comparaison : jamais accepté
    assert code == 1 and report["derm12345_panderm_large.npz"]["status"] == "incomplet"
    code, report = _run(tmp_path, data, tmp_path / "absent")
    assert code == 1 and report[str(tmp_path / "absent")]["status"] == "incomplet"

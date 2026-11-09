#!/usr/bin/env python3
"""Routeur de modalité : fond d'œil, dermoscopie ou autre (CPU, environ 15 min).

Plan fixé le 9 novembre 2026, avant tout run :
- embeddings DINOv2 ViT-B/14 figés (Apache 2.0), images au petit côté 256 puis
  recadrage central 224 (comme pour ISIC) : le routeur voit l'image brute, avant
  tout prétraitement propre à une modalité ;
- tête : standardisation puis régression logistique multinomiale (scikit-learn,
  C = 1, réglages par défaut), sans réglage ;
- seuil de confiance fixé à 0,9 : en dessous, « unknown » (aucun modèle spécialisé) ;
- tirages fixés (graine 0) :
  - ajustement, 1 000 images par classe : ODIR (découpage d'entraînement), ISIC 2019
    (début de l'archive d'entraînement), Imagenette (entraînement) ;
  - contrôle, 300 autres images par classe, des mêmes sources ;
  - test, 500 images par classe, d'autres sources ou d'autres découpages :
    RFMiD (test officiel, autre source de fond d'œil), ISIC 2019 (début de l'archive
    de test), Imagenette (validation) ;
- critères de réussite sur le test (fixés avant le run) : rappel du fond d'œil et de
  la dermoscopie >= 0,95 ; aucune image envoyée au mauvais modèle spécialisé (fond
  d'œil vers dermoscopie ou l'inverse) ; images « autre » envoyées à un modèle
  spécialisé <= 2 %. Intervalles binomiaux exacts à 95 %.

Réserves : « autre » n'est représenté que par des photos d'Imagenette ; des photos
de peau sans dermoscope, des radiographies ou des captures d'écran ne sont pas
testées. Le début des archives ISIC n'est pas un tirage aléatoire de toute la base.

Exemple :
    python scripts/train_modality_router.py --odir-root data/raw/odir --rfmid-root data/raw/rfmid \\
        --modality-root data/raw/modality --checkpoint third_party/dinov2/dinov2_vitb14_timm.pth
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.robustness_checks import load  # noqa: E402
from src.data.isic_dataset import load_skin_image  # noqa: E402
from src.eval.provenance import run_info  # noqa: E402
from src.models.embedding_cache import EmbeddingCache, file_fingerprint  # noqa: E402
from src.models.retfound_backbone import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from src.models.timm_backbones import DINOV2_IMAGE_SIZE, DINOv2Backbone  # noqa: E402
from src.pipeline.modality_classifier import (  # noqa: E402
    CONFIDENCE_THRESHOLD,
    MODALITIES,
    ModalityHead,
    linear_from_sklearn,
)

SIZES = {"fit": 1000, "control": 300, "test": 500}
SPECIALISTS = ("fundus", "dermoscopy")


def exact_ci(k: int, n: int) -> list[float]:
    low = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    high = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return [low, high]


def sample(paths: list[Path], n: int, rng) -> list[Path]:
    if len(paths) < n:
        raise ValueError(f"{len(paths)} images disponibles, {n} demandées")
    return [paths[i] for i in sorted(rng.choice(len(paths), n, replace=False))]


def image_lists(args) -> dict[str, dict[str, list[Path]]]:
    """{partie: {modalité: chemins}} ; ajustement et contrôle disjoints."""
    rng = np.random.default_rng(0)
    odir = load("odir", args.odir_root, "train")
    rfmid = load("rfmid", args.rfmid_root, "test")
    root = args.modality_root
    pools = {
        "fundus": [odir.image_path(i) for i in range(len(odir))],
        "dermoscopy": sorted((root / "isic_train").rglob("ISIC_*.jpg")),
        "unknown": sorted((root / "imagenette/imagenette2-320/train").rglob("*.JPEG")),
    }
    tests = {
        "fundus": [rfmid.image_path(i) for i in range(len(rfmid))],
        "dermoscopy": sorted((root / "isic_test").rglob("ISIC_*.jpg")),
        "unknown": sorted((root / "imagenette/imagenette2-320/val").rglob("*.JPEG")),
    }
    lists = {"fit": {}, "control": {}, "test": {}}
    for modality in MODALITIES:
        chosen = sample(pools[modality], SIZES["fit"] + SIZES["control"], rng)
        order = rng.permutation(len(chosen))
        lists["fit"][modality] = [chosen[i] for i in order[: SIZES["fit"]]]
        lists["control"][modality] = [chosen[i] for i in order[SIZES["fit"] :]]
        lists["test"][modality] = sample(tests[modality], SIZES["test"], rng)
    return lists


def embed(paths: list[Path], backbone, cache: EmbeddingCache, batch_size: int = 32) -> np.ndarray:
    keys = [f"modality/{p.parent.name}/{p.name}" for p in paths]
    missing = set(cache.missing(keys))
    todo = [i for i, k in enumerate(keys) if k in missing]
    start = time.time()
    for b in range(0, len(todo), batch_size):
        idx = todo[b : b + batch_size]
        x = torch.stack([load_skin_image(paths[i], DINOV2_IMAGE_SIZE) for i in idx])
        cache.add([keys[i] for i in idx], backbone((x - IMAGENET_MEAN) / IMAGENET_STD).numpy())
        if (b // batch_size) % 20 == 0:
            cache.save()
            print(f"  {b + len(idx)}/{len(todo)} en {time.time() - start:.0f} s", flush=True)
    cache.save()
    return cache.get(keys)


def evaluate(head: ModalityHead, X: dict[str, np.ndarray]) -> dict:
    report = {"confusion": {}, "recall": {}, "wrong_specialist": {}}
    for truth in MODALITIES:
        decided = head.predict(X[truth])
        n = len(decided)
        report["confusion"][truth] = {m: decided.count(m) for m in MODALITIES}
        k = decided.count(truth)
        report["recall"][truth] = {"value": k / n, "ci": exact_ci(k, n), "n": n}
        wrong = sum(d in SPECIALISTS and d != truth for d in decided)
        report["wrong_specialist"][truth] = {"value": wrong / n, "ci": exact_ci(wrong, n), "count": wrong}
    r, w = report["recall"], report["wrong_specialist"]
    report["criteria"] = {
        "fundus_recall_ge_0.95": r["fundus"]["value"] >= 0.95,
        "dermoscopy_recall_ge_0.95": r["dermoscopy"]["value"] >= 0.95,
        "no_cross_specialist_error": w["fundus"]["count"] == 0 and w["dermoscopy"]["count"] == 0,
        "unknown_to_specialist_le_0.02": w["unknown"]["value"] <= 0.02,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--odir-root", type=Path, required=True)
    parser.add_argument("--rfmid-root", type=Path, required=True)
    parser.add_argument("--modality-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("embeddings/modality_dinov2_vitb14.npz"))
    parser.add_argument("--out", type=Path, default=Path("results/modality_router.json"))
    parser.add_argument("--head", type=Path, default=Path("heads/modality_router.pt"))
    args = parser.parse_args()

    lists = image_lists(args)
    fingerprint = file_fingerprint(args.checkpoint)
    cache = EmbeddingCache(args.cache, meta={"checkpoint_sha256": fingerprint, "image_size": DINOV2_IMAGE_SIZE})
    backbone = DINOv2Backbone(args.checkpoint)
    X = {}
    for part, by_modality in lists.items():
        X[part] = {}
        for modality, paths in by_modality.items():
            print(f"{part} / {modality} : {len(paths)} images", flush=True)
            X[part][modality] = embed(paths, backbone, cache)

    features = np.vstack([X["fit"][m] for m in MODALITIES])
    target = np.concatenate([np.full(len(X["fit"][m]), i) for i, m in enumerate(MODALITIES)])
    scaler = StandardScaler().fit(features)
    model = LogisticRegression(C=1.0, max_iter=5000).fit(scaler.transform(features), target)
    weight, bias = linear_from_sklearn(scaler, model)
    head = ModalityHead(weight.shape[1], CONFIDENCE_THRESHOLD)
    head.linear.weight.data = torch.from_numpy(weight)
    head.linear.bias.data = torch.from_numpy(bias)

    report = {
        "run": run_info(),
        "plan": "fixé le 9 novembre 2026 avant tout run (docstring du script)",
        "backbone": {"model": "dinov2_vitb14", "checkpoint_sha256": fingerprint, "image_size": DINOV2_IMAGE_SIZE},
        "threshold": CONFIDENCE_THRESHOLD,
        "sizes": SIZES,
        "sources": {
            "fit_control": {"fundus": "ODIR train", "dermoscopy": "ISIC 2019 train", "unknown": "Imagenette train"},
            "test": {"fundus": "RFMiD test", "dermoscopy": "ISIC 2019 test", "unknown": "Imagenette val"},
        },
        "control": evaluate(head, X["control"]),
        "test": evaluate(head, X["test"]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    args.head.parent.mkdir(parents=True, exist_ok=True)
    head.save(args.head, {"backbone": "dinov2", "checkpoint_sha256": fingerprint, "image_size": DINOV2_IMAGE_SIZE})
    for truth, row in report["test"]["confusion"].items():
        print(f"test {truth:10s} -> {row}")
    print("critères :", report["test"]["criteria"])
    print(f"-> {args.out}, {args.head}")


if __name__ == "__main__":
    main()

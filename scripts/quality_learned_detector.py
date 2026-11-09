#!/usr/bin/env python3
"""Détecteur de défauts techniques appris sur des défauts simulés, testé sur des vrais (CPU).

Plan fixé le 9 novembre 2026, avant tout run :
- base d'entraînement : 600 images ODIR du découpage d'entraînement sans mot-clé de
  qualité, tirées au hasard (graine 0), hors patients de l'étude des défauts réels ;
- chaque image donne une version propre et une version dégradée ; le défaut est tiré
  au hasard parmi cinq, d'intensité aléatoire : flou gaussien, sous-exposition,
  voile clair, taches (poussière), bande claire en bas du champ (paupière, reflet) ;
- géométrie `crop-mask` à 512 px, défaut appliqué dans le champ rétinien, puis
  embeddings DINOv2 ViT-B/14 figés à 224 px ; régression logistique (C = 1) ;
  seuil fixé à 0,5 ;
- aucun vrai défaut ne sert à l'entraînement. Tests :
  A. validation ODIR de l'étude des défauts réels : images avec mot-clé technique
     (y compris poussière) contre les 120 témoins ; AUROC par image (bootstrap par
     patient), sensibilité et fausses alertes par patient ;
  B. les 36 images relues par Marc : défaut vu par le relecteur contre aucun défaut ;
- comparateur, sur les mêmes images : détecteur actuel (luminance moyenne sous le P2
  des témoins train, ou flou sous 4,16).

Réserves : défauts simulés plus simples que les vrais ; mots-clés d'ODIR non
spécifiques ; relecture d'un seul humain non spécialiste, sur le train. Exploratoire :
aucune règle de rejet n'est ajoutée au pipeline.

Exemple :
    python scripts/quality_learned_detector.py --odir-root data/raw/odir \\
        --checkpoint third_party/dinov2/dinov2_vitb14_timm.pth
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.quality_real_defects import auc_with_groups, exact_rate, patient_rate  # noqa: E402
from src.data.fundus_preprocess import crop_mask  # noqa: E402
from src.data.odir_quality import source_records  # noqa: E402
from src.eval.provenance import run_info  # noqa: E402
from src.models.embedding_cache import EmbeddingCache, file_fingerprint  # noqa: E402
from src.models.retfound_backbone import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from src.models.timm_backbones import DINOV2_IMAGE_SIZE, DINOv2Backbone  # noqa: E402

SIZE = 512
N_BASE = 600
DEFECTS = ("blur", "dark", "washed", "spots", "glare_band")
THRESHOLD = 0.5
LUMINANCE_P2 = 49.107588866789065  # results/quality_real_defects.json
BLUR_THRESHOLD = 4.16


def field_mask(image: np.ndarray) -> np.ndarray:
    return image.max(axis=2) > 10


def degrade(image: np.ndarray, defect: str, rng) -> np.ndarray:
    """Défaut simulé dans le champ rétinien ; le fond reste noir."""
    mask = field_mask(image)
    x = image.astype(np.float32)
    if defect == "blur":
        x = cv2.GaussianBlur(x, (0, 0), rng.uniform(2.0, 6.0))
    elif defect == "dark":
        x = x * rng.uniform(0.2, 0.45)
    elif defect == "washed":
        a = rng.uniform(0.35, 0.6)
        x = x * (1 - a) + 255 * a
    elif defect == "spots":
        layer = np.zeros(x.shape[:2], np.float32)
        ys, xs = np.nonzero(mask)
        for _ in range(rng.integers(3, 13)):
            i = rng.integers(len(ys))
            cv2.circle(layer, (int(xs[i]), int(ys[i])), int(rng.integers(4, 16)), 1.0, -1)
        layer = cv2.GaussianBlur(layer, (0, 0), 2.0)[..., None] * rng.uniform(0.4, 0.8)
        target = 255.0 if rng.random() < 0.5 else 0.0
        x = x * (1 - layer) + target * layer
    elif defect == "glare_band":
        rows = np.arange(x.shape[0], dtype=np.float32)[:, None, None]
        start = x.shape[0] * (1 - rng.uniform(0.15, 0.35))
        alpha = np.clip((rows - start) / 20.0, 0, 1)
        x = x * (1 - alpha) + 255 * alpha
    else:
        raise ValueError(defect)
    x[~mask] = 0
    return np.clip(x, 0, 255).astype(np.uint8)


def retina(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return crop_mask(im.convert("RGB"), SIZE)


def embed(images: list[np.ndarray], backbone) -> np.ndarray:
    x = torch.stack(
        [
            torch.from_numpy(cv2.resize(im, (DINOV2_IMAGE_SIZE, DINOV2_IMAGE_SIZE), interpolation=cv2.INTER_AREA))
            .permute(2, 0, 1)
            .float()
            / 255.0
            for im in images
        ]
    )
    return backbone((x - IMAGENET_MEAN) / IMAGENET_STD).numpy()


def embed_keys(keys, make_image, backbone, cache, batch: int = 32) -> np.ndarray:
    missing = set(cache.missing(keys))
    todo = [k for k in keys if k in missing]
    start = time.time()
    for b in range(0, len(todo), batch):
        part = todo[b : b + batch]
        cache.add(part, embed([make_image(k) for k in part], backbone))
        if (b // batch) % 10 == 0:
            cache.save()
            print(f"  {b + len(part)}/{len(todo)} en {time.time() - start:.0f} s", flush=True)
    cache.save()
    return cache.get(keys)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--odir-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--study-dir", type=Path, default=Path("data/quality_review"))
    parser.add_argument("--review-dir", type=Path, default=Path("data/quality_review_v2"))
    parser.add_argument("--cache", type=Path, default=Path("embeddings/quality_dinov2_vitb14.npz"))
    parser.add_argument("--out", type=Path, default=Path("results/quality_learned_detector.json"))
    args = parser.parse_args()

    images = args.odir_root / "ODIR-5K" / "ODIR-5K" / "Training Images"
    study = json.loads((args.study_dir / "source_manifest.json").read_text())["records"]
    measures = {r["filename"]: r["scores"] for r in json.loads((args.study_dir / "measurements.json").read_text())}
    study_patients = {r["group"] for r in study}
    records, _ = source_records(args.odir_root)
    clean = [r for r in records if r["split"] == "train" and not r["tags"] and r["group"] not in study_patients]
    rng = np.random.default_rng(0)
    base = [clean[i] for i in sorted(rng.choice(len(clean), N_BASE, replace=False))]
    plan = {r["filename"]: (DEFECTS[rng.integers(len(DEFECTS))], int(rng.integers(2**31))) for r in base}

    def make(key: str) -> np.ndarray:
        kind, name = key.split("/", 1)
        image = retina(images / name)
        if kind == "clean" or kind == "real":
            return image
        defect, seed = plan[name]
        return degrade(image, defect, np.random.default_rng(seed))

    fingerprint = file_fingerprint(args.checkpoint)
    cache = EmbeddingCache(args.cache, meta={"checkpoint_sha256": fingerprint, "image_size": DINOV2_IMAGE_SIZE})
    backbone = DINOv2Backbone(args.checkpoint)
    names = [r["filename"] for r in base]
    print("Entraînement : images propres et dégradées", flush=True)
    X_train = np.vstack(
        [
            embed_keys([f"clean/{n}" for n in names], make, backbone, cache),
            embed_keys([f"degraded/{n}" for n in names], make, backbone, cache),
        ]
    )
    y_train = np.repeat([0, 1], len(names))
    scaler = StandardScaler().fit(X_train)
    model = LogisticRegression(C=1.0, max_iter=5000).fit(scaler.transform(X_train), y_train)

    def proba(rows: list[dict]) -> np.ndarray:
        X = embed_keys([f"real/{r['filename']}" for r in rows], make, backbone, cache)
        return model.predict_proba(scaler.transform(X))[:, 1]

    def current(rows: list[dict]) -> np.ndarray:
        s = [measures[r["filename"]] for r in rows]
        return np.array([m["mean_luminance"] < LUMINANCE_P2 or m["blur_score"] < BLUR_THRESHOLD for m in s])

    print("Test A : validation ODIR", flush=True)
    val = [r for r in study if r["split"] == "val"]
    y_val = np.array([bool(r["tags"]) for r in val])
    p_val = proba(val)
    test_a = {"positive_images": int(y_val.sum()), "control_images": int((~y_val).sum()), "detectors": {}}
    for name, flags in (("learned", p_val >= THRESHOLD), ("current", current(val))):
        pos = [r for r, y in zip(val, y_val, strict=True) if y]
        neg = [r for r, y in zip(val, y_val, strict=True) if not y]
        test_a["detectors"][name] = {
            "sensitivity": patient_rate(pos, flags[y_val]),
            "false_alert_rate": patient_rate(neg, flags[~y_val]),
        }
    test_a["detectors"]["learned"]["auroc_image"] = auc_with_groups(val, y_val.astype(int), p_val)

    print("Test B : relecture de Marc", flush=True)
    mapping = json.loads((args.review_dir / "review_mapping.json").read_text())
    reviews = list(csv.DictReader((args.review_dir / "review.csv").open()))
    rows = [{"filename": mapping[r["code"]]} for r in reviews]
    seen = np.array(
        [
            r["fundus_visible"] == "no"
            or any(r[c] == "yes" for c in ("underexposed", "overexposed", "blur", "artifacts"))
            for r in reviews
        ]
    )
    p_rev = proba(rows)
    test_b = {"defect_seen": int(seen.sum()), "no_defect": int((~seen).sum()), "detectors": {}}
    for name, flags in (("learned", p_rev >= THRESHOLD), ("current", current(rows))):
        test_b["detectors"][name] = {
            "sensitivity": exact_rate(list(flags[seen])),
            "false_alert_rate": exact_rate(list(flags[~seen])),
        }

    report = {
        "run": run_info(),
        "plan": "fixé le 9 novembre 2026 avant tout run (docstring du script)",
        "backbone": {"model": "dinov2_vitb14", "checkpoint_sha256": fingerprint},
        "training": {"base_images": N_BASE, "defects": {d: sum(v[0] == d for v in plan.values()) for d in DEFECTS}},
        "threshold": THRESHOLD,
        "test_a_odir_validation": test_a,
        "test_b_human_review": test_b,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    for test, r in (("A", test_a), ("B", test_b)):
        for name, d in r["detectors"].items():
            s, f = d["sensitivity"], d["false_alert_rate"]
            print(
                f"test {test} {name:8s} sensibilité {s['detected']}/{s['patients']}  "
                f"fausses alertes {f['detected']}/{f['patients']}"
            )
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()

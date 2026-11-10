#!/usr/bin/env python3
"""Détecteur de qualité appris sur les notes d'experts de DeepDRiD (CPU).

Plan fixé le 10 novembre 2026, avant tout run :
- données : DeepDRiD, photos de fond d'œil classiques (CC BY-SA 4.0) ; cible = qualité
  globale insuffisante pour le diagnostic (`Overall quality` = 0, note d'experts) ;
  entraînement 1 200 images (300 patients), validation 400 (100), évaluation 400 (100),
  patients disjoints, vérifié ;
- mêmes embeddings que les détecteurs précédents (DINOv2 ViT-B/14 figé, `crop-mask`
  512 px puis 224 px) ; régression logistique standardisée, C = 1, sans réglage ;
  seuil fixé à 0,5 ;
- mesures : AUROC par image (bootstrap par patient, 500 tirages), sensibilité et
  spécificité au seuil, sur la validation puis sur l'évaluation, lue une seule fois ;
- comparateur, sur les mêmes images : détecteur actuel du projet (luminance moyenne
  sous 49,1, P2 des témoins ODIR, ou flou sous 4,16), seuils inchangés ;
- transfert vers une autre source, sans réentraînement : validation de l'étude ODIR
  (mots-clés techniques contre témoins) et relecture humaine de Marc (patients avec
  défaut vu contre sans défaut confirmé ; indéterminés à part) ;
- descriptif : sensibilité selon la note d'artefact, de netteté et de cadrage.

Réserves : DeepDRiD vient surtout d'un site (Nicheng, 1 140 des 1 200 images
d'entraînement) ; la notion de « qualité insuffisante » est celle des annotateurs de
DeepDRiD ; ODIR et la relecture de Marc ont servi aux études précédentes. Exploratoire :
aucune règle de rejet ajoutée au pipeline.

Exemple :
    python scripts/quality_deepdrid_detector.py --deepdrid-root data/raw/deepdrid/regular_fundus_images \\
        --checkpoint third_party/dinov2/dinov2_vitb14_timm.pth
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.quality_learned_detector import (  # noqa: E402
    BLUR_THRESHOLD,
    LUMINANCE_P2,
    SIZE,
    TRANSFORM,
    embed_keys,
    retina,
    review_state,
    sha256_file,
)
from scripts.quality_real_defects import auc_with_groups, exact_rate, patient_rate  # noqa: E402
from src.data.deepdrid import load_split  # noqa: E402
from src.data.fundus_preprocess import crop_mask  # noqa: E402
from src.eval.provenance import run_info  # noqa: E402
from src.models.embedding_cache import EmbeddingCache, file_fingerprint  # noqa: E402
from src.models.timm_backbones import DINOV2_IMAGE_SIZE, DINOv2Backbone  # noqa: E402
from src.pipeline.exposure import exposure_statistics  # noqa: E402
from src.pipeline.quality_control import compute_blur_score, retina_blur_mask  # noqa: E402

THRESHOLD = 0.5
STATES = ("defect", "clear", "indeterminate")


def current_flags(paths: list[Path]) -> np.ndarray:
    """Détecteur actuel du projet (luminance ou flou), seuils inchangés."""
    mask = retina_blur_mask()
    flags = []
    for path in paths:
        with Image.open(path) as im:
            image = crop_mask(im.convert("RGB"), SIZE)
        lum = exposure_statistics(image, mask)["mean_luminance"]
        flags.append(lum < LUMINANCE_P2 or compute_blur_score(image, mask) < BLUR_THRESHOLD)
    return np.array(flags)


def image_metrics(y: np.ndarray, flags: np.ndarray, groups: list[str], scores: np.ndarray | None = None) -> dict:
    out = {
        "sensitivity": exact_rate([bool(f) for f in flags[y == 1]]),
        "specificity": exact_rate([not bool(f) for f in flags[y == 0]]),
        "unit": "image",
    }
    if scores is not None:
        rows = [{"group": g} for g in groups]
        out["auroc_image"] = auc_with_groups(rows, y, scores)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--deepdrid-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--study-dir", type=Path, default=Path("data/quality_review"))
    parser.add_argument("--review-dir", type=Path, default=Path("data/quality_review_v2"))
    parser.add_argument("--odir-root", type=Path, default=Path("data/raw/odir"))
    parser.add_argument("--cache", type=Path, default=Path("embeddings/quality_dinov2_vitb14.npz"))
    parser.add_argument("--out", type=Path, default=Path("results/quality_deepdrid_detector.json"))
    args = parser.parse_args()

    splits = {s: load_split(args.deepdrid_root, s) for s in ("train", "val", "eval")}
    patients = {s: set(d["patient"]) for s, d in splits.items()}
    if (
        patients["train"] & patients["val"]
        or patients["train"] & patients["eval"]
        or patients["val"] & patients["eval"]
    ):
        raise ValueError("patients partagés entre découpages DeepDRiD")

    fingerprint = file_fingerprint(args.checkpoint)
    cache = EmbeddingCache(
        args.cache, meta={"checkpoint_sha256": fingerprint, "image_size": DINOV2_IMAGE_SIZE, "transform": TRANSFORM}
    )
    backbone = DINOv2Backbone(args.checkpoint)
    odir_images = args.odir_root / "ODIR-5K" / "ODIR-5K" / "Training Images"
    paths = {f"deepdrid/{s}/{i}": p for s, d in splits.items() for i, p in zip(d["image_id"], d["path"], strict=True)}

    def make(key: str) -> np.ndarray:
        if key.startswith("deepdrid/"):
            return retina(paths[key])
        return retina(odir_images / key.split("/", 1)[1])

    X, y = {}, {}
    for s, d in splits.items():
        print(f"DeepDRiD {s} : {len(d)} images", flush=True)
        X[s] = embed_keys([f"deepdrid/{s}/{i}" for i in d["image_id"]], make, backbone, cache)
        y[s] = (d["Overall quality"] == 0).astype(int).to_numpy()
    scaler = StandardScaler().fit(X["train"])
    model = LogisticRegression(C=1.0, max_iter=5000).fit(scaler.transform(X["train"]), y["train"])

    def proba(Z: np.ndarray) -> np.ndarray:
        return model.predict_proba(scaler.transform(Z))[:, 1]

    report = {
        "run": run_info(),
        "plan": "fixé le 10 novembre 2026 avant tout run (docstring du script)",
        "backbone": {"model": "dinov2_vitb14", "checkpoint_sha256": fingerprint, "transform": TRANSFORM},
        "threshold": THRESHOLD,
        "deepdrid": {
            s: {"images": len(d), "patients": d["patient"].nunique(), "insufficient_quality": int(y[s].sum())}
            for s, d in splits.items()
        },
        "results": {},
    }
    scores = {}
    for s in ("val", "eval"):
        print(f"Mesures sur DeepDRiD {s}", flush=True)
        p = proba(X[s])
        scores[s] = p
        groups = list(splits[s]["patient"])
        d = splits[s]
        bad = y[s] == 1
        report["results"][f"deepdrid_{s}"] = {
            "learned": image_metrics(y[s], p >= THRESHOLD, groups, p),
            "current": image_metrics(y[s], current_flags(list(d["path"])), groups),
            "learned_sensitivity_by_subscore": {
                "artifact_ge_6": exact_rate([bool(f) for f in (p >= THRESHOLD)[bad & (d["Artifact"] >= 6).to_numpy()]]),
                "clarity_le_4": exact_rate([bool(f) for f in (p >= THRESHOLD)[bad & (d["Clarity"] <= 4).to_numpy()]]),
                "field_le_4": exact_rate(
                    [bool(f) for f in (p >= THRESHOLD)[bad & (d["Field definition"] <= 4).to_numpy()]]
                ),
            },
        }

    print("Transfert vers ODIR", flush=True)
    study = json.loads((args.study_dir / "source_manifest.json").read_text())["records"]
    measures = {r["filename"]: r["scores"] for r in json.loads((args.study_dir / "measurements.json").read_text())}
    val = [r for r in study if r["split"] == "val"]
    y_val = np.array([bool(r["tags"]) for r in val])
    p_val = proba(embed_keys([f"real/{r['filename']}" for r in val], make, backbone, cache))

    def current_odir(rows):
        s = [measures[r["filename"]] for r in rows]
        return np.array([m["mean_luminance"] < LUMINANCE_P2 or m["blur_score"] < BLUR_THRESHOLD for m in s])

    odir = {"positive_images": int(y_val.sum()), "control_images": int((~y_val).sum()), "detectors": {}}
    pos = [r for r, t in zip(val, y_val, strict=True) if t]
    neg = [r for r, t in zip(val, y_val, strict=True) if not t]
    for name, flags in (("learned_deepdrid", p_val >= THRESHOLD), ("current", current_odir(val))):
        odir["detectors"][name] = {
            "sensitivity": patient_rate(pos, flags[y_val]),
            "false_alert_rate": patient_rate(neg, flags[~y_val]),
        }
    odir["detectors"]["learned_deepdrid"]["auroc_image"] = auc_with_groups(val, y_val.astype(int), p_val)
    report["results"]["odir_validation_transfer"] = odir

    mapping = json.loads((args.review_dir / "review_mapping.json").read_text())
    groups = {
        r["filename"]: r["group"] for r in json.loads((args.review_dir / "source_manifest.json").read_text())["records"]
    }
    reviews = list(csv.DictReader((args.review_dir / "review.csv").open()))
    rows = [{"filename": mapping[r["code"]], "group": groups[mapping[r["code"]]]} for r in reviews]
    states = np.array([review_state(r) for r in reviews])
    p_rev = proba(embed_keys([f"real/{r['filename']}" for r in rows], make, backbone, cache))
    review = {"images": {s: int((states == s).sum()) for s in STATES}, "unit": "patient", "detectors": {}}
    for name, flags in (("learned_deepdrid", p_rev >= THRESHOLD), ("current", current_odir(rows))):
        part = {s: ([r for r, t in zip(rows, states, strict=True) if t == s], flags[states == s]) for s in STATES}
        review["detectors"][name] = {
            "sensitivity": patient_rate(*part["defect"]),
            "false_alert_rate": patient_rate(*part["clear"]),
            "flagged_indeterminate": patient_rate(*part["indeterminate"]),
        }
    report["results"]["human_review_transfer"] = review
    report["inputs_sha256"] = {
        "review": sha256_file(args.review_dir / "review.csv"),
        "study_manifest": sha256_file(args.study_dir / "source_manifest.json"),
    }
    report["local_archive"] = f"{args.out.with_suffix('.npz').name} (scores, modèle ; local, non versionné)"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    np.savez(
        args.out.with_suffix(".npz"),
        val_scores=scores["val"],
        eval_scores=scores["eval"],
        odir_val_scores=p_val,
        review_scores=p_rev,
        coef=model.coef_,
        intercept=model.intercept_,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
    )
    for key, r in report["results"].items():
        print(
            key,
            json.dumps({k: v for k, v in r.items() if k != "learned_sensitivity_by_subscore"}, ensure_ascii=False)[
                :600
            ],
        )
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()

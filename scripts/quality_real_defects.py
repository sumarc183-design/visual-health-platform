#!/usr/bin/env python3
"""Défauts réels ODIR : revue locale du train et évaluation exploratoire séparée.

Plan du 9 novembre 2026 : conserver les patients train/val du projet ; exclure
le test et les patients absents du CSV nettoyé ; reprendre les annotations
techniques des deux yeux dans data.xlsx. Garder tous les yeux avec mot-clé
technique, plus 120 témoins sans mot-clé par split (patients distincts des cas).
Critère principal : « low image quality » ; secondaire : « lens dust ».

Seuil de luminance moyenne : percentile 2 des témoins train. Comparateur :
seuil de flou existant 4,16. Union des deux alarmes également rapportée, sans
réglage. Les décisions restent externes au pipeline de prédiction.

Sensibilité et fausses alertes : patient signalé si au moins un œil sélectionné
est signalé, intervalles binomiaux exacts à 95 %. AUROC par image ; bootstrap
par patient (500 tirages), intervalle absent si moins de 10 patients positifs
ou négatifs. Évaluation de mots-clés source, pas d'une vérité clinique.

--prepare-only prépare 24 images du train avec codes anonymes, sans mots-clés
ni scores visibles. Les préannotations visuelles restent locales et provisoires ;
elles ne servent ni à régler le seuil ni à mesurer les performances.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.stats import beta
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.fundus_preprocess import PREPROCESS_VERSION, crop_mask  # noqa: E402
from src.data.odir_quality import QUALITY_TAGS, select_records, source_records  # noqa: E402
from src.eval.metrics import _group_resamples  # noqa: E402
from src.eval.provenance import run_info  # noqa: E402
from src.pipeline.exposure import exposure_statistics  # noqa: E402
from src.pipeline.quality_control import (  # noqa: E402
    BLUR_ANALYSIS_SIZE,
    DEFAULT_BLUR_THRESHOLD,
    compute_blur_score,
    retina_blur_mask,
)

TARGETS = ("low image quality", "lens dust")
REVIEW_FIELDS = (
    "code",
    "fundus_visible",
    "underexposed",
    "overexposed",
    "blur",
    "artifacts",
    "comment",
    "reviewer",
    "status",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_rate(flags: list[bool]) -> dict:
    n, k = len(flags), sum(flags)
    if not n:
        return {"detected": 0, "patients": 0, "rate": None, "ci": None}
    return {
        "detected": k,
        "patients": n,
        "rate": k / n,
        "ci": [
            0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1)),
            1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k)),
        ],
    }


def patient_rate(rows: list[dict], flags: np.ndarray) -> dict:
    by_patient = {}
    for row, flag in zip(rows, flags, strict=True):
        by_patient[row["group"]] = by_patient.get(row["group"], False) or bool(flag)
    return exact_rate(list(by_patient.values()))


def auc_with_groups(rows: list[dict], y: np.ndarray, scores: np.ndarray) -> dict:
    if len(np.unique(y)) != 2:
        return {"value": None, "ci": None, "reason": "une seule classe présente"}
    point = float(roc_auc_score(y, scores))
    groups = np.array([r["group"] for r in rows])
    if min(len(set(groups[y == label])) for label in (0, 1)) < 10:
        return {"value": point, "ci": None, "reason": "moins de 10 patients dans une classe"}
    draws = [roc_auc_score(y[i], scores[i]) for i in _group_resamples(groups, 500, 0) if len(np.unique(y[i])) == 2]
    return {"value": point, "ci": np.percentile(draws, [2.5, 97.5]).tolist(), "valid_bootstrap_draws": len(draws)}


def evaluate_target(records: list[dict], tag: str, threshold: float) -> dict:
    positive = [r for r in records if r["split"] == "val" and tag in r["tags"]]
    negative = [r for r in records if r["split"] == "val" and not r["tags"]]
    rows = positive + negative
    y = np.array([1] * len(positive) + [0] * len(negative))
    low = np.array([r["scores"]["mean_luminance"] < threshold for r in rows])
    blur = np.array([r["scores"]["blur_score"] < DEFAULT_BLUR_THRESHOLD for r in rows])
    out = {"positive_images": len(positive), "control_images": len(negative), "detectors": {}}
    for name, flags in (("low_luminance", low), ("blur", blur), ("either", low | blur)):
        out["detectors"][name] = {
            "sensitivity": patient_rate(positive, flags[: len(positive)]),
            "false_alert_rate": patient_rate(negative, flags[len(positive) :]),
        }
    for name, score in (("low_luminance", "mean_luminance"), ("blur", "blur_score")):
        out["detectors"][name]["auroc_image"] = auc_with_groups(rows, y, -np.array([r["scores"][score] for r in rows]))
    return out


def prepare_review(records: list[dict], images: Path, directory: Path, seed: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    train = [r for r in records if r["split"] == "train"]
    cases = [r for r in train if any(t in r["tags"] for t in QUALITY_TAGS if t != "lens dust")]
    controls = [r for r in train if not r["tags"]]
    rng = np.random.default_rng(seed)
    chosen = cases[:24] + [
        controls[i] for i in rng.choice(len(controls), min(max(0, 24 - len(cases)), len(controls)), False)
    ]
    rng.shuffle(chosen)
    mapping = {f"Q{i + 1:02d}": r["filename"] for i, r in enumerate(chosen)}
    mapping_path = directory / "review_mapping.json"
    if mapping_path.exists() and json.loads(mapping_path.read_text()) != mapping:
        raise ValueError("sélection de revue différente : utiliser un autre dossier pour préserver les annotations")
    mapping_path.write_text(json.dumps(mapping, indent=2) + "\n")
    path = directory / "review.csv"
    if not path.exists():
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=REVIEW_FIELDS)
            writer.writeheader()
            writer.writerows({"code": code, "status": "pending"} for code in mapping)
    pairs = list(mapping.items())
    for start in range(0, len(pairs), 6):
        fig, axes = plt.subplots(2, 3, figsize=(12, 8), facecolor="white")
        for ax in axes.flat:
            ax.axis("off")
        for ax, (code, filename) in zip(axes.flat, pairs[start : start + 6], strict=False):
            with Image.open(images / filename) as im:
                im = im.convert("RGB")
                im.thumbnail((700, 500))
                ax.imshow(im)
            ax.set_title(code)
        fig.tight_layout()
        fig.savefig(directory / f"plate_{start // 6 + 1}.png", dpi=140)
        plt.close(fig)
    print(f"Revue du train : {len(mapping)} images, {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--odir-root", type=Path, required=True)
    parser.add_argument("--local-dir", type=Path, default=Path("data/quality_review"))
    parser.add_argument("--out", type=Path, default=Path("results/quality_real_defects.json"))
    parser.add_argument("--controls", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    source, excluded = source_records(args.odir_root)
    records = select_records(source, args.controls, args.seed)
    images = args.odir_root / "ODIR-5K" / "ODIR-5K" / "Training Images"
    args.local_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"records": records, "seed": args.seed, "controls_per_split": args.controls}
    manifest_path = args.local_dir / "source_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("manifeste local différent : choisir un autre --local-dir")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if args.prepare_only:
        prepare_review(records, images, args.local_dir, args.seed)
        return
    mask = retina_blur_mask()
    digest = hashlib.sha256()
    for i, row in enumerate(records):
        path = images / row["filename"]
        digest.update(row["filename"].encode() + b"\0" + bytes.fromhex(sha256(path)))
        with Image.open(path) as im:
            retina = crop_mask(im.convert("RGB"), BLUR_ANALYSIS_SIZE)
        row["scores"] = {**exposure_statistics(retina, mask), "blur_score": compute_blur_score(retina, mask)}
        if (i + 1) % 40 == 0:
            print(f"Mesures : {i + 1}/{len(records)}", flush=True)
    controls_train = [r["scores"]["mean_luminance"] for r in records if r["split"] == "train" and not r["tags"]]
    threshold = float(np.percentile(controls_train, 2))
    local_scores = args.local_dir / "measurements.json"
    local_scores.write_text(json.dumps(records, indent=2) + "\n")
    review_path = args.local_dir / "review.csv"
    review = None
    if review_path.exists():
        with review_path.open(newline="") as f:
            reviews = list(csv.DictReader(f))
        review = {
            "sha256": sha256(review_path),
            "images": len(reviews),
            "provisional_reviews": sum(r["status"] == "provisional" for r in reviews),
            "pending": sum(r["status"] == "pending" for r in reviews),
            "used_for_calibration_or_validation": False,
        }
    report = {
        "run": run_info(),
        "plan_date": "2026-11-09",
        "seed": args.seed,
        "controls_requested_per_split": args.controls,
        "preprocess_version": PREPROCESS_VERSION,
        "analysis_size": BLUR_ANALYSIS_SIZE,
        "source_workbook_sha256": sha256(args.odir_root / "ODIR-5K" / "ODIR-5K" / "data.xlsx"),
        "split_csv_sha256": sha256(args.odir_root / "full_df.csv"),
        "selection_sha256": sha256(manifest_path),
        "input_images_sha256": digest.hexdigest(),
        "local_visual_review": review,
        "excluded_source_patients_test_or_unmapped": excluded,
        "counts": {
            s: {
                "images": sum(r["split"] == s for r in records),
                "source_tags": {t: sum(r["split"] == s and t in r["tags"] for r in records) for t in QUALITY_TAGS},
            }
            for s in ("train", "val")
        },
        "thresholds": {
            "mean_luminance_below": threshold,
            "calibration": "P2 des témoins train",
            "calibration_images": len(controls_train),
            "blur_below": DEFAULT_BLUR_THRESHOLD,
        },
        "validation": {tag: evaluate_target(records, tag, threshold) for tag in TARGETS},
        "limitations": [
            "Mots-clés source de qualité générale/poussière ; pas d'étiquettes spécifiques d'exposition.",
            "Témoins sans mot-clé, pas des images certifiées acceptables ; échantillon enrichi en défauts.",
            "Validation exploratoire déjà utilisée dans le projet ; aucun résultat prospectif ou clinique.",
            "Trop peu de patients low image quality pour une conclusion solide ; pas d'IC AUROC dans ce cas.",
            "Aucune modification des seuils ni des rejets du pipeline de prédiction.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"Rapport agrégé : {args.out}")


if __name__ == "__main__":
    main()

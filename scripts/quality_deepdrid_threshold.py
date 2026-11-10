#!/usr/bin/env python3
"""Seuil d'avertissement du détecteur de qualité DeepDRiD (analyse secondaire, CPU).

Plan fixé le 10 novembre 2026, après le run principal (`quality_deepdrid_detector.py`,
seuil 0,5) et avant ce calcul : le seuil 0,5 signale trop d'images correctes pour un
usage réel. Règle : plus petit seuil dont la spécificité sur la VALIDATION DeepDRiD
atteint 0,90 (au plus 10 % d'images correctes signalées). Ce seuil est ensuite appliqué
sans changement à l'évaluation DeepDRiD, à la validation ODIR et à la relecture humaine.
Seconde lecture de l'évaluation, annoncée comme telle : le résultat principal reste
celui du seuil 0,5. Aucun réentraînement : scores archivés localement par le run
principal (`results/quality_deepdrid_detector.npz`).

Exemple :
    python scripts/quality_deepdrid_threshold.py --deepdrid-root data/raw/deepdrid/regular_fundus_images
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.quality_learned_detector import review_state  # noqa: E402
from scripts.quality_real_defects import exact_rate, patient_rate  # noqa: E402
from src.data.deepdrid import load_split  # noqa: E402
from src.eval.provenance import run_info  # noqa: E402

TARGET_SPECIFICITY = 0.90


def threshold_for_specificity(scores: np.ndarray, y: np.ndarray, target: float) -> float:
    """Plus petit seuil (parmi les scores observés) dont la spécificité atteint la cible."""
    negatives = np.sort(scores[y == 0])
    for t in np.unique(scores):
        if (negatives < t).mean() >= target:
            return float(t)
    return float(np.inf)


def rates(y: np.ndarray, flags: np.ndarray) -> dict:
    return {
        "sensitivity": exact_rate([bool(f) for f in flags[y == 1]]),
        "specificity": exact_rate([not bool(f) for f in flags[y == 0]]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--deepdrid-root", type=Path, required=True)
    parser.add_argument("--scores", type=Path, default=Path("results/quality_deepdrid_detector.npz"))
    parser.add_argument("--study-dir", type=Path, default=Path("data/quality_review"))
    parser.add_argument("--review-dir", type=Path, default=Path("data/quality_review_v2"))
    parser.add_argument("--out", type=Path, default=Path("results/quality_deepdrid_threshold.json"))
    args = parser.parse_args()

    archive = np.load(args.scores)
    y = {s: (load_split(args.deepdrid_root, s)["Overall quality"] == 0).astype(int).to_numpy() for s in ("val", "eval")}
    threshold = threshold_for_specificity(archive["val_scores"], y["val"], TARGET_SPECIFICITY)

    study = json.loads((args.study_dir / "source_manifest.json").read_text())["records"]
    val = [r for r in study if r["split"] == "val"]
    y_odir = np.array([bool(r["tags"]) for r in val])
    flags_odir = archive["odir_val_scores"] >= threshold
    mapping = json.loads((args.review_dir / "review_mapping.json").read_text())
    groups = {
        r["filename"]: r["group"] for r in json.loads((args.review_dir / "source_manifest.json").read_text())["records"]
    }
    reviews = list(csv.DictReader((args.review_dir / "review.csv").open()))
    rows = [{"filename": mapping[r["code"]], "group": groups[mapping[r["code"]]]} for r in reviews]
    states = np.array([review_state(r) for r in reviews])
    flags_rev = archive["review_scores"] >= threshold

    def part(state):
        return [r for r, t in zip(rows, states, strict=True) if t == state], flags_rev[states == state]

    report = {
        "run": run_info(),
        "plan": "fixé le 10 novembre 2026 après le run principal, avant ce calcul (docstring)",
        "rule": f"plus petit seuil de spécificité >= {TARGET_SPECIFICITY} sur la validation DeepDRiD",
        "threshold": threshold,
        "deepdrid_val": rates(y["val"], archive["val_scores"] >= threshold),
        "deepdrid_eval": rates(y["eval"], archive["eval_scores"] >= threshold),
        "odir_validation": {
            "sensitivity": patient_rate([r for r, t in zip(val, y_odir, strict=True) if t], flags_odir[y_odir]),
            "false_alert_rate": patient_rate(
                [r for r, t in zip(val, y_odir, strict=True) if not t], flags_odir[~y_odir]
            ),
        },
        "human_review": {
            "sensitivity": patient_rate(*part("defect")),
            "false_alert_rate": patient_rate(*part("clear")),
            "flagged_indeterminate": patient_rate(*part("indeterminate")),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"seuil {threshold:.3f}")
    for key in ("deepdrid_val", "deepdrid_eval", "odir_validation", "human_review"):
        print(key, {k: (v["detected"], v["patients"]) for k, v in report[key].items()})
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()

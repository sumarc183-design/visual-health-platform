#!/usr/bin/env python3
"""Détecteur de qualité DeepDRiD : intervalles de confiance groupés par patient (CPU).

Correctif d'unité (audit du 15 novembre 2026, constat 4), écrit le 16 novembre avant ce
calcul. Les runs `quality_deepdrid_detector.py` (seuil 0,5, principal) et
`quality_deepdrid_threshold.py` (seuil de spécificité 0,90, secondaire) donnent la
sensibilité et la spécificité **par image** avec des IC binomiaux qui supposent des images
indépendantes ; or chaque découpage compte 400 images pour 100 patients (4 images par
patient). Les champs nommés `patients` de ces résultats comptent en fait des images.

Ici, rien ne change dans les points (mêmes scores archivés, mêmes seuils, mêmes images) :
seuls les IC sont recalculés par bootstrap sur les patients (1 000 tirages, graine 0,
`rate_ci`). Les résultats d'origine ne sont pas modifiés ; ce fichier les complète.
Les scores archivés sont associés aux images par position, dans l'ordre de `load_split`
(celui du run principal) : le nombre d'images et la proportion d'images de qualité
insuffisante sont vérifiés contre le run principal avant tout calcul.

Exemple :
    python scripts/quality_deepdrid_grouped_ci.py --deepdrid-root data/raw/deepdrid/regular_fundus_images
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.dr_screening_sensitivity import rate_ci  # noqa: E402
from src.data.deepdrid import load_split  # noqa: E402
from src.eval.provenance import run_info  # noqa: E402

PRINCIPAL_THRESHOLD = 0.5


def grouped_rates(y: np.ndarray, flags: np.ndarray, groups: list[str]) -> dict:
    """Taux par image ; IC bootstrap sur les patients. Image positive = qualité insuffisante."""
    return {
        "unit": "image",
        "images_positive": int(y.sum()),
        "images_negative": int((1 - y).sum()),
        "patients": len(set(groups)),
        "sensitivity": float(flags[y == 1].mean()),
        "sensitivity_ci_patient_bootstrap": rate_ci(y, flags, groups, positive=True),
        "specificity": float((~flags[y == 0]).mean()),
        "specificity_ci_patient_bootstrap": rate_ci(y, flags, groups, positive=False),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--deepdrid-root", type=Path, required=True)
    parser.add_argument("--scores", type=Path, default=Path("results/quality_deepdrid_detector.npz"))
    parser.add_argument("--principal", type=Path, default=Path("results/quality_deepdrid_detector.json"))
    parser.add_argument("--secondary", type=Path, default=Path("results/quality_deepdrid_threshold.json"))
    parser.add_argument("--out", type=Path, default=Path("results/quality_deepdrid_grouped_ci.json"))
    args = parser.parse_args()

    archive = np.load(args.scores)
    principal = json.loads(args.principal.read_text(encoding="utf-8"))
    secondary = json.loads(args.secondary.read_text(encoding="utf-8"))
    thresholds = {"principal_0.5": PRINCIPAL_THRESHOLD, "secondary_specificity_0.90": secondary["threshold"]}

    report = {
        "run": run_info(),
        "plan": "correctif d'unité, écrit le 16 novembre 2026 avant ce calcul (docstring) ; points inchangés",
        "thresholds": thresholds,
        "splits": {},
    }
    for split in ("val", "eval"):
        d = load_split(args.deepdrid_root, split)
        y = (d["Overall quality"] == 0).astype(int).to_numpy()
        scores = archive[f"{split}_scores"]
        groups = [str(g) for g in d["patient"]]
        expected = principal["deepdrid"][split]
        if (len(scores), len(set(groups)), int(y.sum())) != (
            expected["images"],
            expected["patients"],
            expected["insufficient_quality"],
        ):
            raise ValueError(f"{split} : images, patients ou étiquettes différents du run principal")
        report["splits"][split] = {name: grouped_rates(y, scores >= t, groups) for name, t in thresholds.items()}
        learned = principal["results"][f"deepdrid_{split}"]["learned"]
        got = report["splits"][split]["principal_0.5"]
        if abs(got["sensitivity"] - learned["sensitivity"]["rate"]) > 1e-12:
            raise ValueError(f"{split} : points différents du run principal, association des scores à revoir")

    s = report["splits"]["eval"]["secondary_specificity_0.90"]
    ref = secondary["deepdrid_eval"]
    if (
        abs(s["sensitivity"] - ref["sensitivity"]["rate"]) > 1e-12
        or abs(s["specificity"] - ref["specificity"]["rate"]) > 1e-12
    ):
        raise ValueError("points différents de l'analyse secondaire : association des scores à revoir")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    for split, by_threshold in report["splits"].items():
        for name, r in by_threshold.items():
            print(
                f"{split:4s} {name:28s} sens {r['sensitivity']:.3f}"
                f" {np.round(r['sensitivity_ci_patient_bootstrap'], 3)}"
                f" | spéc {r['specificity']:.3f} {np.round(r['specificity_ci_patient_bootstrap'], 3)}"
            )
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()

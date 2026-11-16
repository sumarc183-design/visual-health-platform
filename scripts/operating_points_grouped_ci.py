#!/usr/bin/env python3
"""Points de fonctionnement ODIR : intervalles de confiance groupés par patient (CPU).

Correctif d'unité (audit du 15 novembre 2026, constat 4), écrit le 16 novembre avant ce
calcul. `operating_points.py` donne sensibilité et spécificité par image (œil) avec des IC
binomiaux exacts, qui supposent des images indépendantes ; le test ODIR compte 965 images
pour 504 patients (souvent les deux yeux).

Ici, rien ne change dans les points : mêmes prédictions archivées, seuils lus dans
`results/operating_points_odir_<variante>.json` (vérifiés identiques aux comptes
enregistrés). Seuls les IC sont recalculés par bootstrap sur les patients (1 000 tirages,
graine 0, `rate_ci`) ; patient = numéro avant « _left/_right ». Le résultat d'origine
n'est pas modifié. RFMiD n'a pas d'identifiant patient public : ses IC restent par image.

Exemple :
    python scripts/operating_points_grouped_ci.py --variant crop-mask
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.dr_screening_sensitivity import rate_ci  # noqa: E402
from scripts.operating_points import load  # noqa: E402
from src.eval.provenance import run_info  # noqa: E402


def odir_patient(key: str) -> str:
    """« odir/40_right.jpg » ou « 40_right.jpg » -> « 40 »."""
    name = key.rsplit("/", 1)[-1]
    patient, _, side = name.partition("_")
    if not patient.isdigit() or not side.startswith(("left", "right")):
        raise ValueError(f"clé ODIR inattendue : {key}")
    return patient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", default="crop-mask")
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()

    original = json.loads((args.results / f"operating_points_odir_{args.variant}.json").read_text(encoding="utf-8"))
    test = load(args.results / f"pred_odir_{args.variant}_test.npz")
    classes = [str(c) for c in test["class_names"]]
    groups = [odir_patient(str(k)) for k in test["keys"]]
    report = {
        "run": run_info(),
        "plan": "correctif d'unité, écrit le 16 novembre 2026 avant ce calcul (docstring) ; points inchangés",
        "dataset": "odir",
        "variant": args.variant,
        "unit": "image (œil)",
        "images_test": len(groups),
        "patients_test": len(set(groups)),
        "classes": {},
    }
    for i, name in enumerate(classes):
        entry = original["classes"][name]
        if entry["status"] != "ok":
            continue
        y, p = test["y_true"][:, i].astype(int), test["probs"][:, i]
        points = []
        for point in entry["points"]:
            flags = p >= point["threshold"]
            counts = {
                "tp": int((flags & (y == 1)).sum()),
                "fn": int((~flags & (y == 1)).sum()),
                "fp": int((flags & (y == 0)).sum()),
                "tn": int((~flags & (y == 0)).sum()),
            }
            if counts != point["counts"]:
                raise ValueError(f"{name} : comptes différents du résultat d'origine")
            points.append(
                {
                    "target_sensitivity_on_validation": point["target_sensitivity_on_validation"],
                    "threshold": point["threshold"],
                    "test_sensitivity": point["test_sensitivity"]["value"],
                    "sensitivity_ci_binomial_image": point["test_sensitivity"]["ci"],
                    "sensitivity_ci_patient_bootstrap": rate_ci(y, flags, groups, positive=True),
                    "test_specificity": point["test_specificity"]["value"],
                    "specificity_ci_binomial_image": point["test_specificity"]["ci"],
                    "specificity_ci_patient_bootstrap": rate_ci(y, flags, groups, positive=False),
                }
            )
        report["classes"][name] = points

    out = args.results / f"operating_points_odir_{args.variant}_grouped_ci.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    for name, points in report["classes"].items():
        q = points[1]  # sensibilité visée 0,90
        print(
            f"{name:28s} 90 % : sens {q['test_sensitivity']:.2f}"
            f" binomial {np.round(q['sensitivity_ci_binomial_image'], 2)}"
            f" groupé {np.round(q['sensitivity_ci_patient_bootstrap'], 2)}"
        )
    print(f"-> {out}")


if __name__ == "__main__":
    main()

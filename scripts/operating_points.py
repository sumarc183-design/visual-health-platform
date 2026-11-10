#!/usr/bin/env python3
"""Compromis cas manqués / fausses alertes par maladie, pour choisir un seuil (CPU).

Plan fixé le 10 novembre 2026, avant ce calcul : pour chaque maladie, seuils choisis
sur la VALIDATION pour trois sensibilités visées (0,80, 0,90, 0,95 ; `choose_threshold`,
politique « sensibilité »), appliqués sans changement au test. Rapportés au test :
sensibilité et spécificité (IC binomiaux exacts) et, pour 1 000 personnes à la
proportion de malades observée dans le test, cas repérés, cas manqués et fausses
alertes. Maladies avec moins de 10 cas en validation ou en test : « preuve
insuffisante ». Aucun réentraînement : prédictions sauvegardées (mêmes têtes que les
rapports de décision). Le test a déjà servi à d'autres analyses : chiffres descriptifs,
pour aider à choisir un compromis, pas une confirmation.

Exemple :
    python scripts/operating_points.py --dataset odir --variant crop-mask
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_modality_router import exact_ci  # noqa: E402
from src.eval.decision import choose_threshold  # noqa: E402
from src.eval.predictions import validate_predictions  # noqa: E402
from src.eval.provenance import run_info  # noqa: E402

TARGETS = (0.80, 0.90, 0.95)
MIN_CASES = 10
PER = 1000


def load(path: Path) -> dict:
    with np.load(path) as data:
        pred = {k: data[k] for k in data.files}
    validate_predictions(pred, path.name)
    return pred


def operating_point(y_val, p_val, y_test, p_test, target: float) -> dict:
    t = choose_threshold(y_val, p_val, "sensitivity", target)
    flags = p_test >= t
    pos, neg = y_test == 1, y_test == 0
    tp, fn, fp, tn = (
        int((flags & pos).sum()),
        int((~flags & pos).sum()),
        int((flags & neg).sum()),
        int((~flags & neg).sum()),
    )
    prevalence = pos.mean()
    sens, spec = tp / max(pos.sum(), 1), tn / max(neg.sum(), 1)
    sick = PER * prevalence
    return {
        "target_sensitivity_on_validation": target,
        "threshold": float(t),
        "test_sensitivity": {"value": sens, "ci": exact_ci(tp, int(pos.sum()))},
        "test_specificity": {"value": spec, "ci": exact_ci(tn, int(neg.sum()))},
        "counts": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
        f"per_{PER}": {
            "sick": round(sick),
            "detected": round(sick * sens),
            "missed": round(sick * (1 - sens)),
            "false_alerts": round((PER - sick) * (1 - spec)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["rfmid", "odir"], required=True)
    parser.add_argument("--variant", default="crop-mask")
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    val = load(args.results / f"pred_{args.dataset}_{args.variant}_val.npz")
    test = load(args.results / f"pred_{args.dataset}_{args.variant}_test.npz")
    classes = [str(c) for c in val["class_names"]]
    if classes != [str(c) for c in test["class_names"]]:
        raise ValueError("validation et test n'ont pas les mêmes classes")
    report = {
        "run": run_info(),
        "plan": "fixé le 10 novembre 2026 avant ce calcul (docstring du script)",
        "dataset": args.dataset,
        "variant": args.variant,
        "unit": "image",
        "classes": {},
    }
    for i, name in enumerate(classes):
        yv, yt = val["y_true"][:, i].astype(int), test["y_true"][:, i].astype(int)
        entry = {"cases_val": int(yv.sum()), "cases_test": int(yt.sum()), "images_test": len(yt)}
        if min(yv.sum(), yt.sum()) < MIN_CASES:
            entry["status"] = "preuve insuffisante"
        else:
            entry["status"] = "ok"
            entry["points"] = [operating_point(yv, val["probs"][:, i], yt, test["probs"][:, i], t) for t in TARGETS]
        report["classes"][name] = entry
    out = args.out or args.results / f"operating_points_{args.dataset}_{args.variant}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    for name, e in report["classes"].items():
        if e["status"] != "ok":
            continue
        line = "  ".join(
            f"{int(p['target_sensitivity_on_validation'] * 100)}%: sens {p['test_sensitivity']['value']:.2f} "
            f"manqués {p[f'per_{PER}']['missed']} fausses {p[f'per_{PER}']['false_alerts']}"
            for p in e["points"]
        )
        print(f"{name:28s} malades/1000 {e['points'][0][f'per_{PER}']['sick']:4d}  {line}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()

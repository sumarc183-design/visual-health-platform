#!/usr/bin/env python3
"""Métriques de décision et calibration à partir des prédictions sauvegardées
(sur CPU, sans réentraîner).

Seuils par maladie choisis sur la validation (F1 maximal, ou sensibilité
d'au moins 90 % pour le dépistage), appliqués une fois au test ;
sensibilité, spécificité, VPP, VPN, faux positifs et faux négatifs ;
précision moyenne (AP) ; calibration (Brier, ECE) avant et après une
température unique ajustée sur la validation. Les maladies avec moins de 10
positifs en validation ou en test sont marquées « preuve insuffisante ».

Les prédictions de validation et de test viennent de la même tête (même
train, même graine 42, entraînement déterministe).

Exemple :
    python scripts/decision_metrics.py --dataset rfmid --out results/decision_rfmid.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.decision import per_class_report  # noqa: E402


def summarize(report: dict) -> dict:
    """Moyennes sur les maladies avec assez de positifs pour juger un seuil."""
    rows = [r for r in report["classes"].values() if not r["insufficient_evidence"]]
    summary = {"classes_evaluable": len(rows), "classes_insufficient": len(report["classes"]) - len(rows)}
    for key in ("at_0.5", "threshold_f1", "threshold_sensitivity"):
        values = [r[key] for r in rows if key in r]
        summary[key] = {
            metric: float(np.mean([v[metric] for v in values if v[metric] is not None]))
            for metric in ("sensitivity", "specificity", "ppv", "f1")
        }
    for key in ("average_precision_test", "brier_test", "brier_test_calibrated", "ece_test", "ece_test_calibrated"):
        summary[key] = float(np.mean([r[key] for r in rows]))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["rfmid", "odir"], required=True)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--target-sensitivity", type=float, default=0.9)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    val = np.load(args.results / f"pred_{args.dataset}_crop-mask_val.npz")
    test = np.load(args.results / f"pred_{args.dataset}_crop-mask_test.npz")
    classes = list(val["class_names"])
    if classes != list(test["class_names"]):
        raise ValueError("classes différentes entre validation et test")

    report = per_class_report(
        val["y_true"], val["probs"], test["y_true"], test["probs"], classes, args.target_sensitivity
    )
    report["dataset"] = args.dataset
    report["target_sensitivity"] = args.target_sensitivity
    report["summary"] = summary = summarize(report)

    print(
        f"{args.dataset} : {summary['classes_evaluable']} maladies évaluables, "
        f"{summary['classes_insufficient']} en preuve insuffisante ; température {report['temperature']:.2f}"
    )
    for key, label in (
        ("at_0.5", "seuil 0,5"),
        ("threshold_f1", "seuil F1 (val)"),
        ("threshold_sensitivity", "seuil dépistage"),
    ):
        s = summary[key]
        print(
            f"  {label:16s} sensibilité {s['sensitivity']:.2f} | spécificité {s['specificity']:.2f} | "
            f"VPP {s['ppv']:.2f} | F1 {s['f1']:.2f}"
        )
    print(
        f"  AP {summary['average_precision_test']:.3f} | Brier {summary['brier_test']:.4f} -> "
        f"{summary['brier_test_calibrated']:.4f} | ECE {summary['ece_test']:.4f} -> "
        f"{summary['ece_test_calibrated']:.4f}"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"Résultats : {args.out}")


if __name__ == "__main__":
    main()

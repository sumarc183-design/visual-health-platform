#!/usr/bin/env python3
"""Compare le fine-tuning partiel (scripts/finetune_retfound.py) à la tête
linéaire figée (CPU, à partir des prédictions sauvegardées).

1. Validation RFMiD : différence appariée fine-tuning − tête linéaire (mêmes
   images ; la configuration du fine-tuning était fixée à l'avance).
2. Test RFMiD (lu une fois par chaque modèle) : même comparaison.
3. APTOS et EyePACS, si leurs prédictions sont présentes (rétinopathie
   diabétique, toute forme) : AUROC globale et à
   format égal (format d'image d'origine, indicateur indirect de
   l'acquisition), pour le fine-tuning et pour la tête linéaire RFMiD
   (recalculée à partir des embeddings en cache), sur les mêmes
   images ; différences appariées fine-tuning − tête linéaire (globale et à
   format égal) et AUROC par format. Bootstrap par groupes indépendants
   (patients, doublons et quasi-doublons regroupés).

Exemple :
    python scripts/finetune_analysis.py --finetune-dir results/finetune --rfmid-root data/raw/rfmid \\
        --embeddings embeddings --out results/finetune_analysis.json
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.robustness_checks import embeddings, head_scores, load  # noqa: E402
from src.eval.duplicates import independent_groups  # noqa: E402
from src.eval.metrics import (  # noqa: E402
    _auroc,
    _group_resamples,
    evaluate_multilabel,
    paired_bootstrap_diff,
    stratification_gap,
    stratified_auroc,
)
from src.models.embedding_cache import load_embeddings  # noqa: E402


def compare(linear_path: Path, finetune_path: Path) -> dict:
    a, b = np.load(linear_path), np.load(finetune_path)
    if list(a["class_names"]) != list(b["class_names"]):
        raise ValueError("classes différentes entre les deux fichiers de prédictions")
    if set(b["keys"]) != set(a["keys"]):
        raise ValueError(
            f"images différentes : {len(b['keys'])} contre {len(a['keys'])} (essai partiel ?) ; "
            "la comparaison exige le même jeu complet"
        )
    order = {k: i for i, k in enumerate(a["keys"])}
    idx = np.array([order[k] for k in b["keys"]])
    if not np.array_equal(a["y_true"][idx], b["y_true"]):
        raise ValueError("étiquettes différentes entre les deux fichiers de prédictions")
    names = list(b["class_names"])
    diff = paired_bootstrap_diff(b["y_true"], a["probs"][idx], b["probs"], names, list(b["groups"]), n_boot=1000)
    return {
        "linear_auroc_macro": evaluate_multilabel(b["y_true"], a["probs"][idx], names)["auroc_macro"],
        "finetune_auroc_macro": evaluate_multilabel(b["y_true"], b["probs"], names)["auroc_macro"],
        "finetune_minus_linear": diff["macro"],
        "per_class": {k: v for k, v in diff.items() if k != "macro"},
    }


def paired_external_diff(y, p_linear, p_finetune, formats, groups, n_boot=1000, seed=0) -> dict:
    """Différences fine-tuning − tête linéaire, globale et à format égal,
    avec IC à 95 % par bootstrap apparié sur les groupes."""
    out = {}
    for name, metric in (
        ("global", lambda i: _auroc(y[i], p_finetune[i]) - _auroc(y[i], p_linear[i])),
        (
            "within_format",
            lambda i: (
                stratified_auroc(y[i], p_finetune[i], formats[i]) - stratified_auroc(y[i], p_linear[i], formats[i])
            ),
        ),
    ):
        draws = [metric(idx) for idx in _group_resamples(groups, n_boot, seed)]
        low, high = np.nanpercentile(draws, [2.5, 97.5])
        out[name] = {"diff": float(metric(np.arange(len(y)))), "ci": [float(low), float(high)]}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--finetune-dir", type=Path, required=True)
    parser.add_argument("--rfmid-root", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    report = {
        "validation": compare(
            args.results / "pred_rfmid_crop-mask_val.npz", args.finetune_dir / "pred_rfmid_finetune_val.npz"
        )
    }
    test_path = args.finetune_dir / "pred_rfmid_finetune_test.npz"
    if test_path.exists():
        report["test"] = compare(args.results / "pred_rfmid_crop-mask_test.npz", test_path)

    rfmid_train = load("rfmid", args.rfmid_root, "train")
    for name in ("aptos", "eyepacs"):
        path = args.finetune_dir / f"pred_{name}_finetune.npz"
        if not path.exists():
            continue
        ft = np.load(path)
        grade = ft["grade"]
        y = (grade >= 1).astype(np.float32)
        keys, formats = list(ft["keys"]), ft["formats"]
        X = load_embeddings(args.embeddings, name, keys)
        groups, _ = independent_groups(X, list(ft["groups"]) if "groups" in ft.files else keys)
        with contextlib.redirect_stdout(io.StringIO()):
            linear = head_scores(
                embeddings(args.embeddings, "rfmid", "crop-mask", rfmid_train.image_keys),
                rfmid_train.df[["DR"]].to_numpy(np.float32),
                X,
            )[:, 0]
        finetune = ft["prob_dr"]
        scores = (("tête linéaire RFMiD", linear), ("fine-tuning partiel", finetune))
        report[name] = {
            label: {
                **stratification_gap(y[:, None], p[:, None], ["any_dr"], formats, groups, n_boot=1000)["any_dr"],
                "referable_dr_auroc": _auroc((grade >= 2).astype(float), p),
                "auroc_grade_vs_0": {
                    int(g): _auroc((grade[np.isin(grade, [0, g])] == g).astype(float), p[np.isin(grade, [0, g])])
                    for g in np.unique(grade)
                    if g > 0
                },
            }
            for label, p in scores
        }
        report[f"{name}_finetune_minus_linear"] = paired_external_diff(y, linear, finetune, formats, groups)
        by_format = {}
        for f in np.unique(formats):
            m = formats == f
            pos = int(y[m].sum())
            if pos and pos < m.sum():
                by_format[str(f)] = {
                    "images": int(m.sum()),
                    "positives": pos,
                    "comparable_pairs": pos * int(m.sum() - pos),
                    "auroc_linear": _auroc(y[m], linear[m]),
                    "auroc_finetune": _auroc(y[m], finetune[m]),
                }
        report[f"{name}_by_format"] = by_format

    for split in ("validation", "test"):
        if split in report:
            r = report[split]
            d = r["finetune_minus_linear"]
            print(
                f"{split:10s} : tête linéaire {r['linear_auroc_macro']:.4f} | "
                f"fine-tuning {r['finetune_auroc_macro']:.4f} "
                f"| écart apparié {d['diff']:+.4f} [{d['ci'][0]:+.4f} ; {d['ci'][1]:+.4f}]"
            )
    for base in ("aptos", "eyepacs"):
        for name, g in report.get(base, {}).items():
            print(
                f"{base} {name:20s} : AUROC globale {g['auroc']:.3f} | à format égal {g['auroc_within_strata']:.3f} "
                f"| écart {g['gap']:+.3f} [{g['gap_ci'][0]:+.3f} ; {g['gap_ci'][1]:+.3f}] "
                f"| paires comparables {g['comparable_pairs']:.0%} | à adresser {g['referable_dr_auroc']:.3f}"
            )
        for kind, d in report.get(f"{base}_finetune_minus_linear", {}).items():
            print(f"{base} fine-tuning − linéaire ({kind}) : {d['diff']:+.3f} [{d['ci'][0]:+.3f} ; {d['ci'][1]:+.3f}]")
        formats_sorted = sorted(report.get(f"{base}_by_format", {}).items(), key=lambda kv: -kv[1]["comparable_pairs"])
        for f, c in formats_sorted[:5]:
            print(
                f"  {f:>11s} : {c['images']} images, {c['positives']} malades | "
                f"linéaire {c['auroc_linear']:.3f} | fine-tuning {c['auroc_finetune']:.3f}"
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"Résultats : {args.out}")


if __name__ == "__main__":
    main()

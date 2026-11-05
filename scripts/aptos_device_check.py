#!/usr/bin/env python3
"""APTOS : le diagnostic est-il lié au format d'acquisition ? Les têtes
tiennent-elles à format égal ? (CPU, analyse exploratoire)

Les embeddings d'APTOS se regroupent presque parfaitement selon le
diagnostic (scripts/external_validation.py), signe possible d'un biais
d'acquisition. La strate est le **format d'image d'origine**, indicateur
indirect des conditions d'acquisition : ce n'est pas un identifiant
d'appareil validé (un format peut regrouper plusieurs appareils ou
traitements, un appareil produire plusieurs formats). Un écart entre AUROC
globale et à format égal montre une dépendance aux strates, sans en
identifier la cause (appareil, sévérité, sélection, population).

1. Prévalence de la rétinopathie par format, et lien format/diagnostic.
2. AUROC globale contre AUROC à format égal (`stratification_gap`) sur la
   moitié « rapport » d'APTOS (même découpage que site_adaptation.py) :
   - tête RFMiD (n'a jamais vu APTOS) ;
   - tête réentraînée avec N images étiquetées de la moitié « site » ;
   - tête entraînée sur toute la moitié « site ».
Un écart positif = une partie du classement vient des différences de
prévalence entre formats. Moitiés et bootstrap par groupes indépendants
(doublons d'images regroupés). Couverture rapportée par format : effectifs,
malades, paires comparables et AUROC dans le format.

Exemple :
    python scripts/aptos_device_check.py --rfmid-root data/raw/rfmid --embeddings embeddings \\
        --formats embeddings/aptos_formats.csv --out results/aptos_device_check.json
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.robustness_checks import embeddings, head_scores, load  # noqa: E402
from scripts.site_adaptation import split_site_report  # noqa: E402
from src.eval.duplicates import independent_groups  # noqa: E402
from src.eval.metrics import stratification_gap  # noqa: E402
from src.models.embedding_cache import EmbeddingCache  # noqa: E402

MIN_FORMAT_IMAGES = 30


def quiet_scores(X_train, y_train, X_eval) -> np.ndarray:
    with contextlib.redirect_stdout(io.StringIO()):
        return head_scores(X_train.astype(np.float32), y_train.astype(np.float32), X_eval.astype(np.float32))[:, 0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rfmid-root", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--formats", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    labels = pd.read_csv(args.embeddings / "aptos_labels.csv").merge(pd.read_csv(args.formats), on="id_code")
    labels["format"] = labels["width"].astype(str) + "x" + labels["height"].astype(str)
    X = EmbeddingCache(args.embeddings / "aptos_crop-mask_224_RETFound_mae_natureCFP.npz").get(list(labels["key"]))
    grade = labels["diagnosis"].to_numpy()
    y = (grade >= 1).astype(np.float32)
    formats = labels["format"].to_numpy()

    by_format = labels.assign(dr=y).groupby("format")["dr"].agg(["mean", "size"]).sort_values("size", ascending=False)
    frequent = by_format[by_format["size"] >= MIN_FORMAT_IMAGES]
    report = {
        "images": int(len(labels)),
        "formats": int(len(by_format)),
        "adjusted_mutual_information_format_dr": float(adjusted_mutual_info_score(y.astype(int), formats)),
        "prevalence_by_format": {
            f: {"prevalence": float(r["mean"]), "images": int(r["size"])} for f, r in frequent.iterrows()
        },
        "heads": {},
    }

    ds = load("rfmid", args.rfmid_root, "train")
    X_src = embeddings(args.embeddings, "rfmid", "crop-mask", ds.image_keys)
    y_src = ds.df[["DR"]].to_numpy(np.float32)
    site, rep, _ = split_site_report(X, grade, args.seed)
    groups_all, _ = independent_groups(X, [str(i) for i in range(len(X))])
    rng = np.random.default_rng(args.seed)
    heads = {"tête RFMiD (jamais vu APTOS)": quiet_scores(X_src, y_src, X[rep])}
    for n in (50, 200):
        pick = site[rng.choice(len(site), n, replace=False)]
        heads[f"tête RFMiD + {n} images du site"] = quiet_scores(
            np.vstack([X_src, X[pick]]), np.vstack([y_src, y[pick, None]]), X[rep]
        )
    heads["tête entraînée sur la moitié site"] = quiet_scores(X[site], y[site, None], X[rep])

    groups = list(np.array(groups_all)[rep])
    for name, p in heads.items():
        gap = stratification_gap(y[rep, None], p[:, None], ["any_dr"], formats[rep], groups, n_boot=1000)["any_dr"]
        report["heads"][name] = gap

    # Couverture par format pour la tête RFMiD (moitié rapport)
    p_src = heads["tête RFMiD (jamais vu APTOS)"]
    coverage = {}
    for f in np.unique(formats[rep]):
        m = formats[rep] == f
        pos = int(y[rep][m].sum())
        neg = int(m.sum() - pos)
        entry = {"images": int(m.sum()), "positives": pos, "comparable_pairs": pos * neg}
        if pos and neg:
            from sklearn.metrics import roc_auc_score

            entry["auroc_rfmid_head"] = float(roc_auc_score(y[rep][m], p_src[m]))
        coverage[str(f)] = entry
    total_pairs = sum(c["comparable_pairs"] for c in coverage.values())
    report["coverage_by_format_report_half"] = {
        f: {**c, "share_of_comparable_pairs": c["comparable_pairs"] / total_pairs if total_pairs else 0.0}
        for f, c in coverage.items()
    }
    print("Couverture à format égal (moitié rapport, formats avec malades et sains) :")
    for f, c in sorted(report["coverage_by_format_report_half"].items(), key=lambda kv: -kv[1]["comparable_pairs"]):
        if c["comparable_pairs"]:
            print(
                f"  {f:>11s} : {c['images']} images, {c['positives']} malades, "
                f"{c['share_of_comparable_pairs']:.0%} des paires, AUROC tête RFMiD {c['auroc_rfmid_head']:.3f}"
            )
    print(
        f"{report['images']} images, {report['formats']} formats ; lien format/diagnostic (info mutuelle) "
        f"{report['adjusted_mutual_information_format_dr']:.3f}"
    )
    print("Prévalence de la rétinopathie par format (>= 30 images) :")
    for f, r in report["prevalence_by_format"].items():
        print(f"  {f:>11s} : {r['prevalence']:.0%} sur {r['images']} images")
    for name, g in report["heads"].items():
        print(
            f"{name:36s} AUROC globale {g['auroc']:.3f} | à format égal {g['auroc_within_strata']:.3f} | "
            f"écart {g['gap']:+.3f} [{g['gap_ci'][0]:+.3f} ; {g['gap_ci'][1]:+.3f}] | "
            f"paires comparables {g['comparable_pairs']:.0%}"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"Résultats : {args.out}")


if __name__ == "__main__":
    main()

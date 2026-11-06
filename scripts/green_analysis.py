#!/usr/bin/env python3
"""Compare RETFound-Green à RETFound, à tête et découpages identiques (CPU).

Plan fixé avant de lire les résultats, sans réglage :
- même prétraitement `crop-mask` (celui du projet, et non le simple
  redimensionnement de l'article de Green) ; Green en 392 px, RETFound en
  224 px, résolutions d'origine de chaque modèle ;
- même tête (`head_scores`, réglages de la baseline), différences appariées
  sur les mêmes images ;
- RFMiD multi-label : validation, puis test lu une fois pour Green ;
- rétinopathie diabétique, tête RFMiD appliquée à APTOS et EyePACS : AUROC
  globale, à format égal et « à adresser » ; sensibilité et spécificité au
  seuil « dépistage » (sensibilité >= 0,90) choisi sur la validation RFMiD,
  jamais sur la base externe (mesure ajoutée après une première lecture des
  AUROC : elle ne change ni les modèles ni les têtes) ;
- diagnostic dans la base : tête linéaire entraînée dans la base externe,
  5 plis groupés identiques pour les deux modèles. C'est la performance de cette
  sonde, pas le plafond de l'information contenue dans les embeddings.

Scores archivés dans `green_predictions.npz` : les AUROC se recalculent sans
réentraîner. Les bases externes ont déjà servi à d'autres analyses : ces mesures
sont exploratoires, pas une confirmation indépendante.

Bootstrap par groupes indépendants (patients, doublons regroupés, repérés
sur les embeddings RETFound et communs aux deux modèles).

Deux réserves sur l'indépendance : le pré-entraînement de RETFound inclut
Kaggle EyePACS, ce qui pourrait l'avantager sur cette base ; celui de Green
inclut ODIR, d'où l'absence d'ODIR ici (article de Green, Supp. Table 1).

Exemple :
    python scripts/green_analysis.py --rfmid-root data/raw/rfmid --embeddings embeddings \\
        --out results/green_analysis.json
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
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.external_validation import auroc_with_ci, quiet_scores  # noqa: E402
from scripts.finetune_analysis import paired_external_diff  # noqa: E402
from scripts.robustness_checks import CHECKPOINT_STEM, head_scores, load  # noqa: E402
from src.data.fundus_preprocess import PREPROCESS_VERSION  # noqa: E402
from src.eval.decision import choose_threshold, decision_metrics  # noqa: E402
from src.eval.duplicates import independent_groups  # noqa: E402
from src.eval.metrics import evaluate_multilabel, paired_bootstrap_diff, stratification_gap  # noqa: E402
from src.eval.provenance import run_info  # noqa: E402
from src.models.embedding_cache import EmbeddingCache  # noqa: E402
from src.models.retfound_green import GREEN_IMAGE_SIZE  # noqa: E402

GREEN_FILE = "{}_crop-mask_392_retfoundgreen_statedict.npz"
RETFOUND_FILE = "{}_crop-mask_224_" + CHECKPOINT_STEM + ".npz"
FILES = {"RETFound": RETFOUND_FILE, "RETFound-Green": GREEN_FILE}
GREEN_CHECKPOINT_SHA = "431de5dbc1bebbb3"


def cached(directory: Path, model: str, dataset: str, keys: list[str]) -> np.ndarray:
    """Refuse un cache d'une autre configuration ; l'empreinte du checkpoint n'est
    contrôlée que pour Green, RETFound n'étant pas toujours sur la machine."""
    meta = {
        "checkpoint_sha256": GREEN_CHECKPOINT_SHA if model == "RETFound-Green" else None,
        "preprocess": "crop-mask",
        "preprocess_version": PREPROCESS_VERSION,
        "image_size": GREEN_IMAGE_SIZE if model == "RETFound-Green" else 224,
    }
    return EmbeddingCache(directory / FILES[model].format(dataset), meta=meta).get(keys).astype(np.float32)


def external_labels(directory: Path, name: str) -> pd.DataFrame:
    """Clé, stade, groupe et format. APTOS : formats relevés à part."""
    labels = pd.read_csv(directory / f"{name}_labels.csv")
    if "width" not in labels and (directory / f"{name}_formats.csv").exists():
        labels = labels.merge(pd.read_csv(directory / f"{name}_formats.csv"), on="id_code")
    labels["format"] = labels["width"].astype(str) + "x" + labels["height"].astype(str)
    return labels


def rfmid_comparison(root: Path, directory: Path) -> dict:
    sets = {s: load("rfmid", root, s) for s in ("train", "val", "test")}
    names = sets["train"].class_names
    targets = {s: ds.df[names].to_numpy(np.float32) for s, ds in sets.items()}
    report, scores = {}, {}
    for model in FILES:
        X = {s: cached(directory, model, "rfmid", ds.image_keys) for s, ds in sets.items()}
        with contextlib.redirect_stdout(io.StringIO()):
            p = head_scores(X["train"], targets["train"], np.vstack([X["val"], X["test"]]))
        scores[model] = {"val": p[: len(X["val"])], "test": p[len(X["val"]) :]}
    for split in ("val", "test"):
        y, groups = targets[split], sets[split].groups
        a, b = scores["RETFound"][split], scores["RETFound-Green"][split]
        diff = paired_bootstrap_diff(y, a, b, names, groups, n_boot=1000)
        report[split] = {
            "retfound_auroc_macro": evaluate_multilabel(y, a, names)["auroc_macro"],
            "green_auroc_macro": evaluate_multilabel(y, b, names)["auroc_macro"],
            "green_minus_retfound": diff["macro"],
            "per_class": {k: v for k, v in diff.items() if k != "macro"},
        }
    return report


def external_comparison(root: Path, directory: Path, name: str, archive: dict) -> dict:
    labels = external_labels(directory, name)
    keys = list(labels["key"])
    grade = labels["diagnosis"].to_numpy()
    any_dr, referable = (grade >= 1).astype(np.float32), (grade >= 2).astype(np.float32)
    formats = labels["format"].to_numpy()
    X = {model: cached(directory, model, name, keys) for model in FILES}
    base = list(labels["group"].astype(str)) if "group" in labels else keys
    groups, _ = independent_groups(X["RETFound"], base)
    train = load("rfmid", root, "train")
    y_train = train.df[["DR"]].to_numpy(np.float32)
    val = load("rfmid", root, "val")
    y_val = val.df["DR"].to_numpy(np.float32)
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(keys, grade, groups=groups))

    transfer, ceiling, report = {}, {}, {"images": len(keys), "models": {}}
    for model in FILES:
        X_train = cached(directory, model, "rfmid", train.image_keys)
        transfer[model] = quiet_scores(X_train, y_train, X[model])
        p_val = quiet_scores(X_train, y_train, cached(directory, model, "rfmid", val.image_keys))
        threshold = choose_threshold(y_val, p_val, "sensitivity", 0.9)
        ceiling[model] = np.zeros(len(keys))
        for fit, held_out in folds:
            ceiling[model][held_out] = quiet_scores(X[model][fit], any_dr[fit, None], X[model][held_out])
        report["models"][model] = {
            "transfer_any_dr": auroc_with_ci(any_dr, transfer[model], groups, "any_dr"),
            "transfer_referable_dr": auroc_with_ci(referable, transfer[model], groups, "referable_dr"),
            "format_check": stratification_gap(
                any_dr[:, None], transfer[model][:, None], ["any_dr"], formats, groups, n_boot=1000
            )["any_dr"],
            "screening_threshold": {
                "threshold": threshold,
                "any_dr": decision_metrics(any_dr, transfer[model], threshold),
                "detected_by_grade": {
                    int(g): float((transfer[model][grade == g] >= threshold).mean()) for g in np.unique(grade)
                },
            },
            "in_domain_any_dr": auroc_with_ci(any_dr, ceiling[model], groups, "any_dr"),
            "in_domain_referable_dr": auroc_with_ci(referable, ceiling[model], groups, "referable_dr"),
        }
    report["transfer_green_minus_retfound"] = paired_external_diff(
        any_dr, transfer["RETFound"], transfer["RETFound-Green"], formats, groups
    )
    report["in_domain_green_minus_retfound"] = paired_external_diff(
        any_dr, ceiling["RETFound"], ceiling["RETFound-Green"], formats, groups
    )
    archive.update(
        {
            f"{name}_keys": np.array(keys),
            f"{name}_grade": grade,
            f"{name}_groups": np.array(groups),
            f"{name}_formats": formats,
            f"{name}_folds": np.concatenate([np.full(len(h), i) for i, (_, h) in enumerate(folds)])[
                np.argsort(np.concatenate([h for _, h in folds]))
            ],
        }
    )
    for model, short in (("RETFound", "retfound"), ("RETFound-Green", "green")):
        archive[f"{name}_{short}_transfer"] = transfer[model].astype(np.float32)
        archive[f"{name}_{short}_in_domain"] = ceiling[model].astype(np.float32)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rfmid-root", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    report = {"run": run_info(), "rfmid": rfmid_comparison(args.rfmid_root, args.embeddings)}
    archive: dict = {}
    for split in ("val", "test"):
        r = report["rfmid"][split]
        d = r["green_minus_retfound"]
        print(
            f"RFMiD {split:4s} : RETFound {r['retfound_auroc_macro']:.4f} | Green {r['green_auroc_macro']:.4f} "
            f"| écart {d['diff']:+.4f} [{d['ci'][0]:+.4f} ; {d['ci'][1]:+.4f}]"
        )
    for name in ("aptos", "eyepacs"):
        report[name] = external_comparison(args.rfmid_root, args.embeddings, name, archive)
        for model, m in report[name]["models"].items():
            t, g, c = m["transfer_any_dr"], m["format_check"], m["in_domain_any_dr"]
            s = m["screening_threshold"]["any_dr"]
            print(
                f"{name} {model:15s} : tête RFMiD {t['auroc']:.3f} [{t['ci'][0]:.3f} ; {t['ci'][1]:.3f}] | "
                f"à format égal {g['auroc_within_strata']:.3f} | "
                f"à adresser {m['transfer_referable_dr']['auroc']:.3f} | "
                f"dans la base {c['auroc']:.3f} [{c['ci'][0]:.3f} ; {c['ci'][1]:.3f}] | "
                f"seuil dépistage : sensibilité {s['sensitivity']:.2f}, spécificité {s['specificity']:.2f}"
            )
        for label in ("transfer", "in_domain"):
            for kind, d in report[name][f"{label}_green_minus_retfound"].items():
                print(f"  Green − RETFound, {label}, {kind} : {d['diff']:+.3f} [{d['ci'][0]:+.3f} ; {d['ci'][1]:+.3f}]")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    np.savez_compressed(args.out.parent / "green_predictions.npz", **archive)
    print(f"Résultats : {args.out}")


if __name__ == "__main__":
    main()

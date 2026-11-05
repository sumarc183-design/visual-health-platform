#!/usr/bin/env python3
"""Validation externe de la rétinopathie diabétique sur APTOS 2019 ou EyePACS (CPU).

Têtes « rétinopathie diabétique » (une sortie, réglages de la baseline :
Adam lr 1e-3, 30 epochs, mini-batchs de 16, graine 42) entraînées sur les
embeddings `crop-mask` de :
- RFMiD (classe DR), - ODIR (classe D), - les deux ensemble (multi-source).

Seuils de décision choisis sur la **validation de la base d'origine**
(politique F1, et dépistage : sensibilité >= 0,90), jamais sur la base
externe. Rien n'y est réglé : c'est une mesure, lue une fois.

Sur la base externe : AUROC pour toute rétinopathie (stade >= 1) et pour la
rétinopathie à adresser (stade >= 2), IC à 95 % par bootstrap (par patient
quand la base le donne, sinon par image) ; sensibilité et spécificité aux
seuils hérités ; part détectée par stade. En référence, la même tête
évaluée sur le test de sa propre base.

Diagnostic (pas une mesure de généralisation) : une tête entraînée **dans**
la base externe par validation croisée à 5 plis (groupés par patient). Une
AUROC élevée montre que les embeddings portent bien l'information de
rétinopathie : un échec du transfert vient alors du changement de base, pas
d'un problème de préparation des images.

Structure des embeddings (sans étiquette) : regroupement en 8 groupes
(k-moyennes) de chaque base, puis lien entre groupes et diagnostic (info
mutuelle ajustée) et part des images dans un groupe « pur » (moins de 10 %
ou plus de 90 % de malades). Un lien fort indique que la variation
dominante des images suit le diagnostic, signe possible d'un biais
d'acquisition (malades et sains photographiés dans des conditions
différentes) : une tête entraînée sur cette base pourrait s'en servir.

Prérequis : embeddings de la base externe (scripts/extract_embeddings.py,
sur GPU) et `<base>_labels.csv` dans le dossier des embeddings. EyePACS :
`--external eyepacs` ; bootstrap et plis groupés par patient.

Exemple :
    python scripts/external_validation.py --rfmid-root data/raw/rfmid --odir-root data/raw/odir \\
        --embeddings embeddings --out results/external_aptos.json
    python scripts/external_validation.py ... --external eyepacs --out results/external_eyepacs.json
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
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.robustness_checks import embeddings, head_scores, load  # noqa: E402
from src.eval.decision import choose_threshold, decision_metrics  # noqa: E402
from src.eval.duplicates import (  # noqa: E402
    NEAR_DUPLICATE_COSINE,
    duplicate_report,
    exact_duplicate_groups,
    independent_groups,
    merge_groups,
)
from src.eval.metrics import bootstrap_auroc_ci, evaluate_multilabel  # noqa: E402
from src.models.embedding_cache import EmbeddingCache  # noqa: E402

DR_COLUMN = {"rfmid": "DR", "odir": "D"}
CHECKPOINT_STEM = "RETFound_mae_natureCFP"


def dr_split(name: str, root: Path, split: str, cache_dir: Path) -> dict:
    ds = load(name, root, split)
    return {
        "X": embeddings(cache_dir, name, "crop-mask", ds.image_keys),
        "y": ds.df[[DR_COLUMN[name]]].to_numpy(np.float32),
        "groups": ds.groups,
    }


def stack(parts: list[dict]) -> dict:
    return {
        "X": np.vstack([p["X"] for p in parts]),
        "y": np.vstack([p["y"] for p in parts]),
        "groups": [g for p in parts for g in p["groups"]],
    }


def quiet_scores(X_train, y_train, X_eval) -> np.ndarray:
    with contextlib.redirect_stdout(io.StringIO()):  # train_head affiche la loss à chaque epoch
        return head_scores(X_train, y_train, X_eval)[:, 0]


def auroc_with_ci(y: np.ndarray, p: np.ndarray, groups: list[str], name: str) -> dict:
    metrics = evaluate_multilabel(y[:, None], p[:, None], [name])
    ci = bootstrap_auroc_ci(y[:, None], p[:, None], [name], groups, n_boot=1000)
    return {"auroc": metrics["auroc_per_class"][name], "ci": ci["auroc_per_class_ci"][name]}


def cluster_structure(X: np.ndarray, y: np.ndarray, k: int = 8) -> dict:
    clusters = KMeans(k, n_init=10, random_state=0).fit_predict(X)
    prevalence = pd.Series(y).groupby(clusters).mean()
    sizes = pd.Series(clusters).value_counts()
    pure = sum(sizes[c] for c in prevalence.index if prevalence[c] < 0.1 or prevalence[c] > 0.9)
    return {
        "adjusted_mutual_information": float(adjusted_mutual_info_score(y, clusters)),
        "share_in_pure_clusters": float(pure / len(y)),
        "prevalence_by_cluster": [float(v) for v in prevalence.values],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rfmid-root", type=Path, required=True)
    parser.add_argument("--odir-root", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--external", choices=["aptos", "eyepacs"], default="aptos", help="Base externe")
    parser.add_argument("--target-sensitivity", type=float, default=0.9)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    roots = {"rfmid": args.rfmid_root, "odir": args.odir_root}

    data = {(n, s): dr_split(n, roots[n], s, args.embeddings) for n in roots for s in ("train", "val", "test")}
    sources = {
        "rfmid": ([data["rfmid", "train"]], [data["rfmid", "val"]], data["rfmid", "test"]),
        "odir": ([data["odir", "train"]], [data["odir", "val"]], data["odir", "test"]),
        "rfmid+odir": (
            [data["rfmid", "train"], data["odir", "train"]],
            [data["rfmid", "val"], data["odir", "val"]],
            None,
        ),
    }

    ext = args.external
    labels = pd.read_csv(args.embeddings / f"{ext}_labels.csv")
    cache = EmbeddingCache(args.embeddings / f"{ext}_crop-mask_224_{CHECKPOINT_STEM}.npz")
    X_ext = cache.get(list(labels["key"]))
    grade = labels["diagnosis"].to_numpy()
    any_dr, referable = (grade >= 1).astype(float), (grade >= 2).astype(float)
    # Unités indépendantes : patient quand la base le donne (EyePACS : deux yeux
    # par patient), sinon l'image ; fusionnées avec les doublons d'images
    # (embeddings identiques ou quasi identiques), qui ne doivent pas être
    # séparés par les plis ni tirés séparément par le bootstrap.
    base_groups = list(labels["group"].astype(str)) if "group" in labels else list(labels["key"])
    ext_groups, near_pairs = independent_groups(X_ext, base_groups)

    in_domain = np.zeros(len(grade))
    # Plis groupés par patient : les deux yeux d'un patient restent du même côté
    folds = StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X_ext, grade, groups=ext_groups)
    for train_idx, test_idx in folds:
        in_domain[test_idx] = quiet_scores(X_ext[train_idx], any_dr[train_idx, None], X_ext[test_idx])
    report = {
        "diagnostic_in_external_5fold": {
            "any_dr": auroc_with_ci(any_dr, in_domain, ext_groups, "any_dr"),
            "referable_dr": auroc_with_ci(referable, in_domain, ext_groups, "referable_dr"),
        },
        "external": ext,
        "duplicates": {
            **duplicate_report(merge_groups(len(grade), exact_duplicate_groups(X_ext)), grade),
            "near_duplicate_pairs": near_pairs,
            "near_duplicate_cosine": NEAR_DUPLICATE_COSINE,
            "independent_groups": len(set(ext_groups)),
        },
        "external_images": int(len(labels)),
        "external_grades": {int(g): int((grade == g).sum()) for g in range(5)},
        "target_sensitivity": args.target_sensitivity,
        "heads": {},
    }
    for name, (train_parts, val_parts, own_test) in sources.items():
        train, val = stack(train_parts), stack(val_parts)
        p_val = quiet_scores(train["X"], train["y"], val["X"])
        p_ext = quiet_scores(train["X"], train["y"], X_ext)
        entry = {
            "external_any_dr": auroc_with_ci(any_dr, p_ext, ext_groups, "any_dr"),
            "external_referable_dr": auroc_with_ci(referable, p_ext, ext_groups, "referable_dr"),
            "source_val_auroc": auroc_with_ci(val["y"][:, 0], p_val, val["groups"], "dr"),
        }
        if own_test is not None:
            p_test = quiet_scores(train["X"], train["y"], own_test["X"])
            entry["own_test_auroc"] = auroc_with_ci(own_test["y"][:, 0], p_test, own_test["groups"], "dr")
        for policy in ("f1", "sensitivity"):
            threshold = choose_threshold(val["y"][:, 0], p_val, policy, args.target_sensitivity)
            decided = p_ext >= threshold
            entry[f"threshold_{policy}"] = {
                "threshold": threshold,
                "external_any_dr": decision_metrics(any_dr, p_ext, threshold),
                "detected_by_grade": {int(g): float(decided[grade == g].mean()) for g in np.unique(grade)},
            }
        report["heads"][name] = entry

        own = entry.get("own_test_auroc")
        own_text = f" | test de la base d'origine {own['auroc']:.3f}" if own else ""
        print(
            f"{name:11s} {ext.upper()} toute RD {entry['external_any_dr']['auroc']:.3f} "
            f"[{entry['external_any_dr']['ci'][0]:.3f}-{entry['external_any_dr']['ci'][1]:.3f}] | "
            f"RD à adresser {entry['external_referable_dr']['auroc']:.3f}{own_text}"
        )
        for policy in ("f1", "sensitivity"):
            t = entry[f"threshold_{policy}"]
            m = t["external_any_dr"]
            by_grade = " ".join(f"{g}:{v:.0%}" for g, v in t["detected_by_grade"].items())
            print(
                f"    seuil {policy:11s} (val d'origine) : sensibilité {m['sensitivity']:.2f}, "
                f"spécificité {m['specificity']:.2f} | détectées par stade {by_grade}"
            )
    report["cluster_structure"] = {
        name: cluster_structure(
            np.vstack([data[name, s]["X"] for s in ("train", "val", "test")]),
            np.concatenate([data[name, s]["y"][:, 0] for s in ("train", "val", "test")]).astype(int),
        )
        for name in roots
    }
    report["cluster_structure"][ext] = cluster_structure(X_ext, any_dr.astype(int))
    for name, c in report["cluster_structure"].items():
        print(
            f"Structure {name:6s} : info mutuelle groupes/diagnostic {c['adjusted_mutual_information']:.3f}, "
            f"images en groupe pur {c['share_in_pure_clusters']:.0%}"
        )
    diag = report["diagnostic_in_external_5fold"]
    print(
        f"Diagnostic, tête entraînée dans {ext.upper()} (5 plis) : toute RD {diag['any_dr']['auroc']:.3f}, "
        f"à adresser {diag['referable_dr']['auroc']:.3f}"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"Résultats : {args.out}")


if __name__ == "__main__":
    main()

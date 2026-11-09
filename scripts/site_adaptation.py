#!/usr/bin/env python3
"""Adapter la détection de la rétinopathie diabétique à un nouveau site (CPU).

La validation externe (scripts/external_validation.py) montre qu'une tête
entraînée sur RFMiD perd sur APTOS (AUROC 0,947 -> 0,827) et que ses seuils
ne se transposent pas. Question pratique : que faut-il au nouveau site pour
que le modèle y fonctionne ?

Protocole : APTOS est coupé en deux moitiés stratifiées par stade (graine
fixe), **groupées** : les doublons d'images (embeddings identiques ou quasi
identiques, 6,8 % d'APTOS) restent du même côté. La moitié **site** joue les images dont dispose l'hôpital (sans ou
avec quelques étiquettes) : tout réglage s'y fait. La moitié **rapport**
n'est lue que pour mesurer.

Pistes (tête source : RFMiD, classe DR, réglages de la baseline) :
1. Sans étiquette du site : centrage des embeddings par domaine, ou CORAL
   (alignement des moyennes et covariances de la source sur celles du site),
   à partir des images de la moitié site seulement.
2. Avec N images étiquetées du site : seuil recalibré pour une sensibilité
   de 0,90 (la tête ne change pas).
3. Avec N images étiquetées du site : tête réentraînée sur source + N ;
   seuil choisi hors échantillon (validation croisée interne sur les N).
Pour 2 et 3, N parmi 25 à 400, tirages répétés (moyenne et écart-type).

Exemple :
    python scripts/site_adaptation.py --rfmid-root data/raw/rfmid --embeddings embeddings \\
        --out results/site_adaptation_aptos.json
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.robustness_checks import embeddings, head_scores, load  # noqa: E402
from src.eval.decision import choose_threshold, decision_metrics  # noqa: E402
from src.eval.duplicates import (  # noqa: E402
    duplicate_report,
    exact_duplicate_groups,
    independent_groups,
    merge_groups,
)
from src.eval.provenance import run_info  # noqa: E402
from src.models.embedding_cache import load_embeddings  # noqa: E402

SHOTS = (25, 50, 100, 200, 400)


def quiet_scores(X_train, y_train, X_eval) -> np.ndarray:
    with contextlib.redirect_stdout(io.StringIO()):
        return head_scores(X_train.astype(np.float32), y_train.astype(np.float32), X_eval.astype(np.float32))[:, 0]


def coral(source: np.ndarray, target: np.ndarray, eps: float = 1.0) -> np.ndarray:
    """Aligne moyenne et covariance de la source sur celles de la cible
    (Sun et al. 2016), covariances régularisées par eps * identité."""

    def sqrtm(cov, inverse=False):
        values, vectors = np.linalg.eigh(cov)
        values = np.clip(values, 1e-8, None)
        return vectors @ np.diag(values ** (-0.5 if inverse else 0.5)) @ vectors.T

    d = source.shape[1]
    cov_s = np.cov(source, rowvar=False) + eps * np.eye(d)
    cov_t = np.cov(target, rowvar=False) + eps * np.eye(d)
    return (source - source.mean(0)) @ sqrtm(cov_s, inverse=True) @ sqrtm(cov_t) + target.mean(0)


def split_site_report(X: np.ndarray, grade: np.ndarray, seed: int):
    """Moitiés « site » et « rapport » stratifiées par stade et groupées : les
    doublons d'images (embeddings identiques ou quasi identiques) restent du
    même côté (second audit, 5 novembre 2026)."""
    groups, near_pairs = independent_groups(X, [str(i) for i in range(len(X))])
    splitter = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed)
    site, report = next(splitter.split(X, grade, groups=groups))
    report_info = duplicate_report(merge_groups(len(X), exact_duplicate_groups(X)), grade)
    report_info.update(near_duplicate_pairs=near_pairs, independent_groups=len(set(groups)))
    shared = set(np.array(groups)[site]) & set(np.array(groups)[report])
    assert not shared, f"{len(shared)} groupes de doublons partagés entre site et rapport"
    return site, report, report_info


def internal_folds(y_few: np.ndarray, groups: np.ndarray, folds: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """Plis internes groupés : un groupe de doublons reste d'un seul côté. Le
    nombre de plis dépend des groupes malades et sains disponibles ; liste vide
    s'il y en a moins de 2."""
    positives = len(set(groups[y_few == 1]))
    negatives = len(set(groups[y_few == 0]))
    k = int(min(folds, positives, negatives))
    if k < 2:
        return []
    splitter = StratifiedGroupKFold(k, shuffle=True, random_state=0)
    result = list(splitter.split(y_few, y_few, groups=groups))
    for fit, held_out in result:
        assert not set(groups[fit]) & set(groups[held_out]), "groupe partagé dans un pli interne"
    return result


def out_of_fold_scores(X_src, y_src, X_few, y_few, groups: np.ndarray, folds: int = 5) -> np.ndarray:
    """Scores des N images du site, chacune prédite par une tête entraînée sur
    la source et les autres images du site, hors de son groupe de doublons."""
    splits = internal_folds(y_few, groups, folds)
    if not splits:  # trop peu de malades ou de sains pour séparer : repli sur la tête source seule
        return quiet_scores(X_src, y_src, X_few)
    scores = np.zeros(len(y_few))
    for train_idx, test_idx in splits:
        X_mix = np.vstack([X_src, X_few[train_idx]])
        y_mix = np.vstack([y_src, y_few[train_idx, None]])
        scores[test_idx] = quiet_scores(X_mix, y_mix, X_few[test_idx])
    return scores


def summary(values: list[float]) -> dict:
    return {"mean": float(np.mean(values)), "sd": float(np.std(values)), "n": len(values)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rfmid-root", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-sensitivity", type=float, default=0.9)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    def rfmid(split):
        ds = load("rfmid", args.rfmid_root, split)
        return embeddings(args.embeddings, "rfmid", "crop-mask", ds.image_keys), ds.df[["DR"]].to_numpy(np.float32)

    (X_src, y_src), (X_src_val, y_src_val) = rfmid("train"), rfmid("val")
    labels = pd.read_csv(args.embeddings / "aptos_labels.csv")
    X_apt = load_embeddings(args.embeddings, "aptos", list(labels["key"]))
    grade = labels["diagnosis"].to_numpy()
    y_apt = (grade >= 1).astype(np.float32)
    site, report, dup = split_site_report(X_apt, grade, args.seed)
    X_site, y_site, X_rep, y_rep = X_apt[site], y_apt[site], X_apt[report], y_apt[report]
    all_groups, _ = independent_groups(X_apt, [str(i) for i in range(len(X_apt))])
    groups_site = np.array(all_groups)[site]

    def evaluate(p_rep, threshold=None) -> dict:
        out = {"auroc": float(roc_auc_score(y_rep, p_rep))}
        if threshold is not None:
            m = decision_metrics(y_rep, p_rep, threshold)
            out.update(sensitivity=m["sensitivity"], specificity=m["specificity"])
        return out

    result = {
        "site_images": len(site),
        "report_images": len(report),
        "split": "par groupes indépendants (doublons d'images regroupés), stratifié par stade",
        "duplicates": dup,
        "few_shot_threshold": "choisi hors échantillon (validation croisée interne groupée sur les N images)",
        "run": run_info(),
        "few_shot_draws": {},
        "unsupervised": {},
        "few_shot": {},
    }

    # 0. Référence : tête source telle quelle, seuil de la validation RFMiD
    p_val = quiet_scores(X_src, y_src, X_src_val)
    p_rep = quiet_scores(X_src, y_src, X_rep)
    source_threshold = choose_threshold(y_src_val[:, 0], p_val, "sensitivity", args.target_sensitivity)
    result["reference"] = evaluate(p_rep, source_threshold)

    # 1. Sans étiquette du site : choix sur la moitié site (AUROC), puis rapport
    unsupervised = {
        "centrage": (X_src - X_src.mean(0), X_site - X_site.mean(0), X_rep - X_site.mean(0)),
        "CORAL": (coral(X_src, X_site), X_site, X_rep),
    }
    for name, (Xs, Xsite, Xrep) in unsupervised.items():
        p_site = quiet_scores(Xs, y_src, Xsite)
        p_rep_u = quiet_scores(Xs, y_src, Xrep)
        result["unsupervised"][name] = {
            "site_auroc": float(roc_auc_score(y_site, p_site)),
            "report": evaluate(p_rep_u),
        }

    # 2 et 3. Avec N étiquettes du site
    p_site_source = quiet_scores(X_src, y_src, X_site)
    for n in SHOTS:
        draws = []
        recal, retrain = [], []
        for _ in range(args.repeats):
            pick = rng.choice(len(site), size=n, replace=False)
            while y_site[pick].min() == y_site[pick].max():  # il faut des malades et des sains
                pick = rng.choice(len(site), size=n, replace=False)
            draws.append([str(k) for k in labels["key"].to_numpy()[site][pick]])
            threshold = choose_threshold(y_site[pick], p_site_source[pick], "sensitivity", args.target_sensitivity)
            recal.append(evaluate(p_rep, threshold))
            # Seuil hors échantillon : scores des N images prédits par des têtes
            # qui ne les ont pas vues (validation croisée interne), pas par la
            # tête finale, entraînée sur elles.
            oof = out_of_fold_scores(X_src, y_src, X_site[pick], y_site[pick], groups_site[pick])
            threshold_mix = choose_threshold(y_site[pick], oof, "sensitivity", args.target_sensitivity)
            X_mix = np.vstack([X_src, X_site[pick]])
            y_mix = np.vstack([y_src, y_site[pick, None]])
            retrain.append(evaluate(quiet_scores(X_mix, y_mix, X_rep), threshold_mix))
        result["few_shot_draws"][str(n)] = draws
        result["few_shot"][str(n)] = {
            "threshold_recalibrated": {k: summary([r[k] for r in recal]) for k in recal[0]},
            "head_retrained": {k: summary([r[k] for r in retrain]) for k in retrain[0]},
        }

    ref = result["reference"]
    print(
        f"Référence (tête RFMiD, seuil de la validation RFMiD) : AUROC {ref['auroc']:.3f}, "
        f"sensibilité {ref['sensitivity']:.2f}, spécificité {ref['specificity']:.2f}"
    )
    for name, r in result["unsupervised"].items():
        print(f"Sans étiquette, {name:8s} : AUROC site {r['site_auroc']:.3f}, rapport {r['report']['auroc']:.3f}")
    for n, r in result["few_shot"].items():
        a, b = r["threshold_recalibrated"], r["head_retrained"]
        print(
            f"N = {n:>3s} étiquettes : seuil recalibré -> sensibilité {a['sensitivity']['mean']:.2f}"
            f"±{a['sensitivity']['sd']:.2f}, spécificité {a['specificity']['mean']:.2f} | tête réentraînée -> "
            f"AUROC {b['auroc']['mean']:.3f}±{b['auroc']['sd']:.3f}, sensibilité {b['sensitivity']['mean']:.2f}, "
            f"spécificité {b['specificity']['mean']:.2f}"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Les tirages contiennent des identifiants APTOS (règlement Kaggle) : fichier local
    # séparé, ignoré par git ; le rapport versionné ne garde que les agrégats.
    draws_path = args.out.with_name(args.out.stem + "_draws.json")
    draws_path.write_text(json.dumps(result.pop("few_shot_draws"), ensure_ascii=False))
    result["few_shot_draws_file"] = f"{draws_path.name} (local, non versionné)"
    args.out.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(f"Résultats : {args.out} ; tirages : {draws_path}")


if __name__ == "__main__":
    main()

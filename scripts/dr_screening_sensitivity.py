#!/usr/bin/env python3
"""Dépistage de la rétinopathie diabétique : relever la sensibilité sur une base nouvelle (CPU).

Plan fixé le 12 novembre 2026, avant tout calcul, après la décision du tuteur (« tout le
monde » comme utilisateurs, améliorer les performances pour dépasser 27 rétinopathies
repérées sur 100 dans DDR avec la tête et le seuil RFMiD).

Pourquoi : la tête Green apprise sur RFMiD classe correctement DDR (AUROC 0,794) mais son
seuil « dépistage » (sensibilité visée 0,90 sur la validation RFMiD) n'y repère que 27 %
des rétinopathies. Les essais précédents écartent ODIR (étiquettes de rétinopathie qui se
transposent mal) et APTOS (format d'image lié au diagnostic) comme sources
d'apprentissage. EyePACS (programme de dépistage, format peu lié au diagnostic) est
gardé en complément par décision du tuteur ; le préentraînement de Green ne l'inclut pas.

Méthode (RETFound-Green 392 px, `crop-mask`, embeddings figés ; tête et réglages de la
baseline, graine 42) :
- EyePACS (5 000 images, 2 500 patients) coupé par patient : 80 % apprentissage, 20 %
  validation (StratifiedGroupKFold à 5 plis, graine 0, premier pli en validation) ;
- tête principale « RFMiD + EyePACS » : apprise sur l'entraînement RFMiD (colonne DR) et
  l'apprentissage EyePACS (stade >= 1) ;
- seuil choisi sur la validation EyePACS pour une sensibilité de 0,90, jamais sur DDR ;
- mesure unique sur les 12 522 images de DDR (toute rétinopathie, stade >= 1), mêmes
  groupes de doublons que la confirmation ; IC par bootstrap sur les groupes (1 000
  tirages), contrôle avec les groupes horaires.

Critère principal (fixé) : sensibilité sur DDR au seuil choisi, borne basse de l'IC à
95 % >= 0,80. Rapportés avec : spécificité, valeur prédictive positive, AUROC, et les
mêmes mesures pour la tête RFMiD seule avec son seuil de validation RFMiD (référence :
sensibilité 0,27). Secondaires, sans critère : écart d'AUROC apparié (tête principale −
tête RFMiD) ; tête « EyePACS seul » au même seuil de méthode ; sensibilité « à adresser »
(stade >= 2) ; sensibilité par stade.

Conséquence prévue : critère atteint → la tête RFMiD + EyePACS et son seuil deviennent le
réglage « dépistage » par défaut de la rétinopathie (au prix de la spécificité mesurée) ;
sinon, la sensibilité ne se transpose toujours pas et il faudra recalibrer sur des images
du site.

Réserves : licence d'EyePACS limitée à sa compétition (décision du tuteur de le garder en
complément) ; DDR a déjà servi à la confirmation et à l'analyse de résolution, mais
aucune tête ni aucun seuil n'y a été réglé ; une seule base de test.

Exemple :
    python scripts/dr_screening_sensitivity.py --rfmid-root data/raw/rfmid --embeddings embeddings \\
        --reference results/confirmation_ddr.npz --out results/dr_screening_sensitivity.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.external_validation import quiet_scores  # noqa: E402
from scripts.finetune_analysis import paired_external_diff  # noqa: E402
from scripts.green_analysis import cached, external_labels  # noqa: E402
from scripts.robustness_checks import load  # noqa: E402
from src.data.ddr_dataset import time_groups  # noqa: E402
from src.eval.decision import choose_threshold, decision_metrics  # noqa: E402
from src.eval.metrics import _group_resamples  # noqa: E402
from src.eval.provenance import run_info  # noqa: E402

MODEL = "RETFound-Green"
TARGET = 0.90
CRITERION = 0.80


def rate_ci(y: np.ndarray, decided: np.ndarray, groups: list[str], positive: bool, seed: int = 0) -> list[float]:
    """IC bootstrap (groupes) de la sensibilité (positive=True) ou de la spécificité."""
    keep = y == (1 if positive else 0)
    draws = []
    for idx in _group_resamples(groups, 1000, seed):
        k = idx[keep[idx]]
        if len(k):
            draws.append(float((decided[k] == positive).mean()))
    return np.percentile(draws, [2.5, 97.5]).tolist()


def screening(y: np.ndarray, p: np.ndarray, threshold: float, groups: list[str], grade: np.ndarray) -> dict:
    decided = p >= threshold
    m = decision_metrics(y, p, threshold)
    return {
        **m,
        "sensitivity_ci": rate_ci(y, decided, groups, True),
        "specificity_ci": rate_ci(y, decided, groups, False),
        "auroc": float(roc_auc_score(y, p)),
        "referable_sensitivity": float(decided[grade >= 2].mean()),
        "sensitivity_by_grade": {int(g): float(decided[grade == g].mean()) for g in np.unique(grade) if g >= 1},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rfmid-root", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=Path("results/confirmation_ddr.npz"))
    parser.add_argument("--out", type=Path, default=Path("results/dr_screening_sensitivity.json"))
    args = parser.parse_args()

    train, val = load("rfmid", args.rfmid_root, "train"), load("rfmid", args.rfmid_root, "val")
    X_rf = cached(args.embeddings, MODEL, "rfmid", train.image_keys)
    y_rf = train.df["DR"].to_numpy(np.float32)
    X_rf_val = cached(args.embeddings, MODEL, "rfmid", val.image_keys)
    y_rf_val = val.df["DR"].to_numpy(np.float32)

    eye = external_labels(args.embeddings, "eyepacs")
    X_eye = cached(args.embeddings, MODEL, "eyepacs", list(eye["key"]))
    y_eye = (eye["diagnosis"].to_numpy() >= 1).astype(np.float32)
    fit, held = next(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X_eye, y_eye, groups=eye["group"]))

    ddr = external_labels(args.embeddings, "ddr")
    keys = list(ddr["key"])
    X_ddr = cached(args.embeddings, MODEL, "ddr", keys)
    grade = ddr["diagnosis"].to_numpy()
    y_ddr = (grade >= 1).astype(np.float32)
    with np.load(args.reference) as ref:
        if list(ref["ddr_keys"]) != keys:
            raise ValueError("clés DDR différentes de la confirmation")
        groups = list(ref["ddr_groups"])
    by_time = time_groups([k.rsplit("/", 1)[1] for k in keys])
    formats = ddr["format"].to_numpy()

    heads = {
        "rfmid_eyepacs": (np.vstack([X_rf, X_eye[fit]]), np.concatenate([y_rf, y_eye[fit]])),
        "eyepacs_only": (X_eye[fit], y_eye[fit]),
        "rfmid_only": (X_rf, y_rf),
    }
    report = {
        "run": run_info(),
        "plan": "fixé le 12 novembre 2026 avant tout calcul (docstring du script)",
        "eyepacs_split": {
            "fit": len(fit),
            "validation": len(held),
            "validation_patients": int(eye["group"].iloc[held].nunique()),
        },
        "heads": {},
    }
    scores = {}
    for name, (Xa, ya) in heads.items():
        p_eval = quiet_scores(Xa, ya[:, None], np.vstack([X_eye[held], X_rf_val, X_ddr]))
        p_eye_val, p_rf_val, p_ddr = np.split(p_eval, [len(held), len(held) + len(X_rf_val)])
        if name == "rfmid_only":  # référence : seuil de la validation RFMiD, comme la confirmation
            threshold = choose_threshold(y_rf_val, p_rf_val, "sensitivity", TARGET)
            calibration = "validation RFMiD"
        else:
            threshold = choose_threshold(y_eye[held], p_eye_val, "sensitivity", TARGET)
            calibration = "validation EyePACS"
        scores[name] = p_ddr
        report["heads"][name] = {
            "threshold_from": calibration,
            "eyepacs_validation": decision_metrics(y_eye[held], p_eye_val, threshold),
            "ddr": screening(y_ddr, p_ddr, threshold, groups, grade),
            "ddr_time_groups_sensitivity_ci": rate_ci(y_ddr, p_ddr >= threshold, by_time, True),
        }
    main_head = report["heads"]["rfmid_eyepacs"]
    low = min(main_head["ddr"]["sensitivity_ci"][0], main_head["ddr_time_groups_sensitivity_ci"][0])
    report["criterion"] = {
        "sensitivity": main_head["ddr"]["sensitivity"],
        "lower_bound": low,
        "threshold": CRITERION,
        "met": low >= CRITERION,
    }
    report["auroc_rfmid_eyepacs_minus_rfmid"] = paired_external_diff(
        y_ddr, scores["rfmid_only"], scores["rfmid_eyepacs"], formats, groups
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    for name, r in report["heads"].items():
        d = r["ddr"]
        print(
            f"{name:14s} seuil ({r['threshold_from']}) : DDR sensibilité {d['sensitivity']:.2f} {d['sensitivity_ci']}, "
            f"spécificité {d['specificity']:.2f}, VPP {d['ppv']:.2f}, AUROC {d['auroc']:.3f}"
        )
    print("Critère :", report["criterion"])
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Enregistre la tête « dépistage RD » (RETFound-Green, RFMiD + EyePACS) et son seuil (CPU).

Conséquence fixée par le plan de `dr_screening_sensitivity.py` (critère atteint le
12 novembre 2026 : sensibilité sur DDR 0,87 [0,86 ; 0,88]) : cette tête et son seuil
deviennent le réglage « dépistage » de la rétinopathie diabétique.

Aucune nouvelle mesure : la tête est réentraînée à l'identique (mêmes données, réglages
et graine 42) et le script vérifie qu'elle redonne exactement le seuil et la sensibilité
sur DDR archivés dans `results/dr_screening_sensitivity.json`.

Sorties :
- `heads/dr_screening_green_head.pt` : une classe (DR), Green 392 px, `crop-mask` ;
- `results/decision_dr_screening.json` : seuil de la politique `sensitivity` (choisi sur
  la validation EyePACS), lié à la tête par `head_sha256`, lu par `predict.py --policy
  sensitivity`. Pas de seuil `f1` : seule la politique dépistage a été évaluée.

Réserves : licence d'EyePACS limitée à sa compétition (gardé en complément par décision
du tuteur) ; spécificité mesurée sur DDR 0,52 (une fausse alerte pour deux personnes
saines environ).

Exemple :
    python scripts/save_dr_screening_head.py --rfmid-root data/raw/rfmid --embeddings embeddings \\
        --checkpoint third_party/retfound_green/retfoundgreen_statedict.pth
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.dr_screening_sensitivity import MODEL, TARGET  # noqa: E402
from scripts.green_analysis import cached, external_labels  # noqa: E402
from scripts.robustness_checks import fit_head, load  # noqa: E402
from src.data.fundus_preprocess import PREPROCESS_VERSION  # noqa: E402
from src.eval.decision import choose_threshold, decision_metrics  # noqa: E402
from src.eval.provenance import run_info  # noqa: E402
from src.models.classifier_head import head_contract, head_sha256  # noqa: E402
from src.models.embedding_cache import file_fingerprint  # noqa: E402
from src.models.retfound_green import GREEN_IMAGE_SIZE  # noqa: E402

SEED = 42
TOLERANCE = 1e-6


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rfmid-root", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="RETFound-Green local")
    parser.add_argument("--reference", type=Path, default=Path("results/dr_screening_sensitivity.json"))
    parser.add_argument("--head-out", type=Path, default=Path("heads/dr_screening_green_head.pt"))
    parser.add_argument("--thresholds-out", type=Path, default=Path("results/decision_dr_screening.json"))
    args = parser.parse_args()

    train = load("rfmid", args.rfmid_root, "train")
    X_rf = cached(args.embeddings, MODEL, "rfmid", train.image_keys)
    y_rf = train.df["DR"].to_numpy(np.float32)
    eye = external_labels(args.embeddings, "eyepacs")
    X_eye = cached(args.embeddings, MODEL, "eyepacs", list(eye["key"]))
    y_eye = (eye["diagnosis"].to_numpy() >= 1).astype(np.float32)
    fit, held = next(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X_eye, y_eye, groups=eye["group"]))
    ddr = external_labels(args.embeddings, "ddr")
    X_ddr = cached(args.embeddings, MODEL, "ddr", list(ddr["key"]))
    y_ddr = (ddr["diagnosis"].to_numpy() >= 1).astype(np.float32)

    X_fit = np.vstack([X_rf, X_eye[fit]])
    y_fit = np.concatenate([y_rf, y_eye[fit]])[:, None]
    with contextlib.redirect_stdout(io.StringIO()):
        head = fit_head(X_fit, y_fit, SEED)
    with torch.no_grad():
        p_val = torch.sigmoid(head(torch.from_numpy(X_eye[held]))).numpy()[:, 0]
        p_ddr = torch.sigmoid(head(torch.from_numpy(X_ddr))).numpy()[:, 0]
    threshold = choose_threshold(y_eye[held], p_val, "sensitivity", TARGET)
    ddr_metrics = decision_metrics(y_ddr, p_ddr, threshold)

    archived = json.loads(args.reference.read_text())["heads"]["rfmid_eyepacs"]
    checks = {
        "threshold": abs(threshold - archived["eyepacs_validation"]["threshold"]),
        "ddr_sensitivity": abs(ddr_metrics["sensitivity"] - archived["ddr"]["sensitivity"]),
        "ddr_specificity": abs(ddr_metrics["specificity"] - archived["ddr"]["specificity"]),
    }
    print("écarts à l'analyse archivée :", {k: f"{v:.1e}" for k, v in checks.items()})
    if max(checks.values()) > TOLERANCE:
        raise ValueError("la tête ne redonne pas le seuil et les mesures de l'analyse du 12 novembre")

    artifact = {
        "state_dict": head.cpu().state_dict(),
        "class_names": ["DR"],
        "dataset": "rfmid+eyepacs",
        "modality": "fundus",
        "backbone": "green",
        "image_size": GREEN_IMAGE_SIZE,
        "preprocess": "crop-mask",
        "preprocess_version": PREPROCESS_VERSION,
        "checkpoint_sha256": file_fingerprint(args.checkpoint),
        "head_training": {"lr": 1e-3, "epochs": 30, "batch_size": 16},
        "training_images": {"rfmid": len(X_rf), "eyepacs": len(fit)},
        "seed": SEED,
        "git_commit": run_info()["git_commit"],
    }
    args.head_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.head_out)

    validation = decision_metrics(y_eye[held], p_val, threshold)
    report = {
        "dataset": "rfmid+eyepacs",
        "head_file": args.head_out.name,
        "head_sha256": head_sha256(artifact["state_dict"]),
        "head_contract": head_contract(artifact),
        "run": run_info(),
        "target_sensitivity": TARGET,
        "threshold_from": f"validation EyePACS ({len(held)} images, coupée par patient)",
        "source": "scripts/dr_screening_sensitivity.py (plan 16238ba, résultat 53400d0)",
        "reproduction_gaps": checks,
        "classes": {
            "DR": {
                "insufficient_evidence": False,
                "threshold_sensitivity": validation,
                "threshold_f1": None,
                "ddr_external": {k: ddr_metrics[k] for k in ("sensitivity", "specificity", "ppv", "npv")},
            }
        },
    }
    args.thresholds_out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(
        f"seuil {threshold:.4f} ; DDR sensibilité {ddr_metrics['sensitivity']:.3f}, "
        f"spécificité {ddr_metrics['specificity']:.3f}"
    )
    print(f"-> {args.head_out}, {args.thresholds_out}")


if __name__ == "__main__":
    main()

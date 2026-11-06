#!/usr/bin/env python3
"""Tête RFMiD sur les embeddings RETFound-Green, pour predict.py (CPU).

Mêmes réglages et même graine que la tête comparée par
scripts/green_analysis.py : les AUROC sont identiques. Écrit la tête
(`heads/rfmid_green_head.pt`, qui indique son modèle et l'empreinte de son
checkpoint) et les prédictions (`results/pred_rfmid_green_{val,test}.npz`), à
passer ensuite à scripts/decision_metrics.py pour les seuils.

Pas de tête ODIR : le pré-entraînement de Green inclut ODIR.

Exemple :
    python scripts/train_green_head.py --rfmid-root data/raw/rfmid --embeddings embeddings \\
        --checkpoint third_party/retfound_green/retfoundgreen_statedict.pth
    python scripts/decision_metrics.py --dataset rfmid --variant green --head heads/rfmid_green_head.pt \\
        --embeddings embeddings --out results/decision_rfmid_green.json
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.green_analysis import cached  # noqa: E402
from scripts.robustness_checks import fit_head, load  # noqa: E402
from src.data.fundus_preprocess import PREPROCESS_VERSION  # noqa: E402
from src.eval.metrics import evaluate_multilabel  # noqa: E402
from src.models.embedding_cache import file_fingerprint  # noqa: E402
from src.models.retfound_green import GREEN_IMAGE_SIZE  # noqa: E402

SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rfmid-root", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--heads", type=Path, default=Path("heads"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()

    sets = {s: load("rfmid", args.rfmid_root, s) for s in ("train", "val", "test")}
    names = sets["train"].class_names
    X = {s: cached(args.embeddings, "RETFound-Green", "rfmid", ds.image_keys) for s, ds in sets.items()}
    y = {s: ds.df[names].to_numpy(np.float32) for s, ds in sets.items()}
    with contextlib.redirect_stdout(io.StringIO()):
        head = fit_head(X["train"], y["train"], SEED)
    aurocs = {}
    for split in ("val", "test"):
        with torch.no_grad():
            probs = torch.sigmoid(head(torch.from_numpy(X[split]))).numpy()
        np.savez(
            args.results / f"pred_rfmid_green_{split}.npz",
            keys=np.array(sets[split].image_keys),
            groups=np.array(sets[split].groups),
            y_true=y[split],
            probs=probs,
            class_names=np.array(names),
        )
        aurocs[split] = evaluate_multilabel(y[split], probs, names)
        print(f"{split} : AUROC macro {aurocs[split]['auroc_macro']:.4f}")
    args.heads.mkdir(parents=True, exist_ok=True)
    path = args.heads / "rfmid_green_head.pt"
    torch.save(
        {
            "state_dict": head.state_dict(),
            "class_names": list(names),
            "backbone": "green",
            "dataset": "rfmid",
            "image_size": GREEN_IMAGE_SIZE,
            "preprocess": "crop-mask",
            "auroc_per_class": aurocs["val"]["auroc_per_class"],
            "preprocess_version": PREPROCESS_VERSION,
            "checkpoint_sha256": file_fingerprint(args.checkpoint),
            "head_training": {"lr": 1e-3, "epochs": 30, "batch_size": 16},
            "seed": SEED,
        },
        path,
    )
    print(f"Tête sauvegardée : {path}")


if __name__ == "__main__":
    main()

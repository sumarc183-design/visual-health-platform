#!/usr/bin/env python3
"""Balayage des réglages de la tête linéaire, sur les embeddings en cache (CPU).

Archive l'expérience du 4 novembre 2026 : l'écart train/validation de la tête
RFMiD (AUROC 0,98 contre 0,86) peut-il se réduire en régularisant la tête ?
Protocole :
- même tête que run_baseline.py (linéaire, BCE, Adam lr 1e-3, mini-batchs de
  16, graine 42), sur les embeddings `crop-mask` en cache ;
- variantes : weight decay découplé (AdamW), pénalité L2 explicite sur les
  poids, pondération des positifs par classe (racine de (1-p)/p, adoucie),
  standardisation des embeddings, AUROC relevée à 5/10/15/20/30/50 epochs ;
- **lit uniquement train et validation** : le test n'est pas touché.

Le meilleur score parmi des dizaines de variantes est biaisé vers le haut
(maximum de mesures bruitées) : un gain de quelques millièmes n'est pas une
amélioration démontrée. Pour conclure sur une variante, la relancer seule et
la comparer à la base avec scripts/compare_predictions.py (différence
appariée).

Exemple :
    python scripts/sweep_head.py --dataset rfmid --data-root data/raw/rfmid \\
        --embeddings embeddings/rfmid_crop-mask_224_RETFound_mae_natureCFP.npz \\
        --out results/sweep_head_rfmid.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.odir_dataset import ODIRDataset  # noqa: E402
from src.data.rfmid_dataset import RFMiDDataset  # noqa: E402
from src.eval.metrics import evaluate_multilabel  # noqa: E402
from src.models.embedding_cache import EmbeddingCache  # noqa: E402

EPOCH_CHECKPOINTS = (5, 10, 15, 20, 30, 50)
GRID = (
    # (mode de régularisation, force, pondération des positifs, standardisation)
    *[
        ("adamw", wd, pw, std)
        for wd, pw, std in itertools.product((0.0, 1e-3, 1e-2, 1e-1), (False, True), (False, True))
    ],
    *[("l2", lam, pw, False) for lam, pw in itertools.product((1e-5, 1e-4, 3e-4, 1e-3), (False, True))],
)


def train_and_track(X_train, y_train, X_val, y_val, class_names, mode, strength, pos_weight, standardize, seed=42):
    """AUROC macro (val, train) à chaque epoch de EPOCH_CHECKPOINTS."""
    if standardize:
        mean, std = X_train.mean(0), X_train.std(0) + 1e-6
        X_train, X_val = (X_train - mean) / std, (X_val - mean) / std
    torch.manual_seed(seed)
    loader = DataLoader(
        list(zip(torch.from_numpy(X_train), torch.from_numpy(y_train), strict=True)), batch_size=16, shuffle=True
    )
    head = nn.Linear(X_train.shape[1], len(class_names))
    if mode == "adamw" and strength > 0:
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=strength)
    else:
        optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    weights = None
    if pos_weight:
        prevalence = y_train.mean(0).clip(1e-3)
        weights = torch.from_numpy(np.sqrt((1 - prevalence) / prevalence)).float()
    criterion = nn.BCEWithLogitsLoss(pos_weight=weights)

    scores = {}
    X_val_t, X_train_t = torch.from_numpy(X_val), torch.from_numpy(X_train)
    for epoch in range(1, max(EPOCH_CHECKPOINTS) + 1):
        head.train()
        for x, y in loader:
            optimizer.zero_grad()
            loss = criterion(head(x), y)
            if mode == "l2":
                loss = loss + strength * head.weight.pow(2).sum()
            loss.backward()
            optimizer.step()
        if epoch in EPOCH_CHECKPOINTS:
            with torch.no_grad():
                p_val, p_train = torch.sigmoid(head(X_val_t)).numpy(), torch.sigmoid(head(X_train_t)).numpy()
            scores[epoch] = (
                evaluate_multilabel(y_val, p_val, class_names)["auroc_macro"],
                evaluate_multilabel(y_train, p_train, class_names)["auroc_macro"],
            )
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["rfmid", "odir"], required=True)
    parser.add_argument("--data-root", type=Path, required=True, help="Pour les étiquettes et le découpage")
    parser.add_argument("--embeddings", type=Path, required=True, help="Cache .npz crop-mask (embeddings/)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    dataset_cls = RFMiDDataset if args.dataset == "rfmid" else ODIRDataset
    train, val = (dataset_cls(args.data_root, split=s, preprocess="crop-mask") for s in ("train", "val"))
    cache = EmbeddingCache(args.embeddings)
    X_train, X_val = cache.get(train.image_keys), cache.get(val.image_keys)
    y_train = train.df[train.class_names].to_numpy(np.float32)
    y_val = val.df[val.class_names].to_numpy(np.float32)
    print(f"{args.dataset} : train {X_train.shape}, val {X_val.shape}, {len(train.class_names)} classes")

    rows = []
    for mode, strength, pos_weight, standardize in GRID:
        scores = train_and_track(
            X_train, y_train, X_val, y_val, train.class_names, mode, strength, pos_weight, standardize
        )
        for epoch, (auroc_val, auroc_train) in scores.items():
            rows.append(
                {
                    "mode": mode,
                    "strength": strength,
                    "pos_weight": pos_weight,
                    "standardize": standardize,
                    "epochs": epoch,
                    "auroc_val": auroc_val,
                    "auroc_train": auroc_train,
                }
            )
        print(
            f"{mode:5s} {strength:<7g} pos_weight={pos_weight!s:5s} std={standardize!s:5s} | "
            + "  ".join(f"{e}:{v:.4f}" for e, (v, _) in scores.items()),
            flush=True,
        )

    baseline = next(
        r
        for r in rows
        if r["mode"] == "adamw"
        and r["strength"] == 0
        and not r["pos_weight"]
        and not r["standardize"]
        and r["epochs"] == 30
    )
    best = max(rows, key=lambda r: r["auroc_val"])
    summary = {
        "dataset": args.dataset,
        "split_read": ["train", "val"],
        "n_configurations": len(GRID),
        "n_measures": len(rows),
        "baseline": baseline,
        "best": best,
        "best_gain": best["auroc_val"] - baseline["auroc_val"],
        "note": "Meilleur score = maximum de mesures bruitées, biaisé vers le haut ; "
        "à confirmer par une différence appariée avant toute adoption.",
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(
        f"Base (30 epochs) : {baseline['auroc_val']:.4f} | meilleur : {best['auroc_val']:.4f} "
        f"({best['mode']} {best['strength']:g}, pos_weight={best['pos_weight']}, std={best['standardize']}, "
        f"{best['epochs']} epochs) | gain {summary['best_gain']:+.4f}"
    )
    print(f"Résultats : {args.out}")


if __name__ == "__main__":
    main()

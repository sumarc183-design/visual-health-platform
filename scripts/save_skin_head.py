#!/usr/bin/env python3
"""Enregistre la tête « peau » du socle : DINOv2-B figé, ISIC 2019, 8 classes (CPU).

Même tête et mêmes réglages que l'analyse du 9 novembre (`dermatology_analysis.py`,
`head_scores`, graine 42), entraînée sur le découpage d'entraînement. Aucune nouvelle
mesure : le script vérifie seulement que la tête enregistrée redonne exactement les
scores archivés de validation et de test (`results/dermatology_isic2019.npz`).

Pourquoi DINOv2-B plutôt que PanDerm (meilleur de +0,046 d'AUROC macro) : même modèle
que le routeur de modalité (un seul calcul d'embedding par image), licence Apache 2.0,
poids locaux vérifiés équivalents à ceux du run Kaggle. PanDerm (CC BY-NC-ND, 1,4 Go)
reste l'option de recherche. Les données ISIC 2019 sont CC BY-NC : tête non commerciale.

Exemple :
    python scripts/save_skin_head.py --isic-root data/raw/isic2019 --embeddings embeddings_derm \\
        --checkpoint third_party/dinov2/dinov2_vitb14_timm.pth --out heads/isic2019_dinov2b_head.pt
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

from scripts.dermatology_analysis import ImageList  # noqa: E402
from scripts.robustness_checks import fit_head  # noqa: E402
from src.data.fundus_preprocess import PREPROCESS_VERSION  # noqa: E402
from src.data.isic_dataset import CLASS_NAMES  # noqa: E402
from src.eval.provenance import run_info  # noqa: E402
from src.models.embedding_cache import file_fingerprint, load_embeddings  # noqa: E402

KAGGLE_FINGERPRINT = "388c468b37acb34e"  # poids du run Kaggle des embeddings ISIC


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--isic-root", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="DINOv2-B local (celui du routeur)")
    parser.add_argument("--reference", type=Path, default=Path("results/dermatology_isic2019.npz"))
    parser.add_argument("--out", type=Path, default=Path("heads/isic2019_dinov2b_head.pt"))
    args = parser.parse_args()

    sets = {s: ImageList(args.isic_root, s) for s in ("train", "val", "test")}
    X = {s: load_embeddings(args.embeddings, "isic2019", d.keys, "dinov2", preprocess="none") for s, d in sets.items()}
    with contextlib.redirect_stdout(io.StringIO()):
        head = fit_head(X["train"], sets["train"].targets)
    with torch.no_grad(), np.load(args.reference) as ref:
        for split in ("val", "test"):
            if list(ref[f"{split}_keys"]) != sets[split].keys:
                raise ValueError(f"clés {split} différentes de l'archive")
            scores = torch.sigmoid(head(torch.from_numpy(X[split]))).numpy()
            gap = float(np.abs(scores - ref[f"DINOv2-B_{split}"]).max())
            print(f"{split} : écart maximal aux scores archivés {gap:.2e}")
            if gap > 1e-5:
                raise ValueError("la tête ne redonne pas les scores de l'analyse du 9 novembre")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": head.cpu().state_dict(),
            "class_names": list(CLASS_NAMES),
            "dataset": "isic2019",
            "modality": "dermoscopy",
            "backbone": "dinov2",
            "image_size": 224,
            "preprocess": "skin",
            "preprocess_version": PREPROCESS_VERSION,
            "checkpoint_sha256": file_fingerprint(args.checkpoint),
            "embeddings_checkpoint_sha256": KAGGLE_FINGERPRINT,
            "head_training": {"lr": 1e-3, "epochs": 30, "batch_size": 16},
            "seed": 42,
            "git_commit": run_info()["git_commit"],
        },
        args.out,
    )
    print(f"Tête enregistrée : {args.out}")


if __name__ == "__main__":
    main()

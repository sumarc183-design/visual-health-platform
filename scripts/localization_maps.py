#!/usr/bin/env python3
"""Cartes d'occlusion et tokens de patch sur la segmentation IDRiD (GPU).

Pour chaque image et chaque modèle (RETFound, RETFound-Green) : carte
d'occlusion du logit de rétinopathie, avec la tête RFMiD servie par predict.py
(jamais entraînée sur IDRiD), et tokens de patch pour
scripts/localization_analysis.py, qui fait le reste sur CPU.

Exemple (GPU T4, ~10 min) :
    python scripts/localization_maps.py --data-root <IDRiD> --dr-heads results/localization/dr_heads.npz \\
        --retfound-checkpoint third_party/retfound/checkpoints/RETFound_mae_natureCFP.pth \\
        --green-checkpoint third_party/retfound_green/retfoundgreen_statedict.pth --out-dir localization
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.idrid_dataset import IDRiDSegmentation  # noqa: E402
from src.explainability.localization import MODELS, FrozenClassifier, resize_map  # noqa: E402
from src.explainability.occlusion import occlusion_map  # noqa: E402

MAP_SIZE = 56
FILE_NAMES = {"RETFound": "retfound", "RETFound-Green": "green"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dr-heads", type=Path, required=True)
    parser.add_argument("--retfound-checkpoint", type=Path, required=True)
    parser.add_argument("--green-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--limit", type=int, default=None, help="Essai rapide : nombre d'images par découpage")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--trust-checkpoint", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    heads = np.load(args.dr_heads)
    checkpoints = {"RETFound": args.retfound_checkpoint, "RETFound-Green": args.green_checkpoint}

    for name in args.models:
        cfg = MODELS[name]
        file = FILE_NAMES[name]
        model = FrozenClassifier(
            name,
            checkpoints[name],
            heads[f"{file}_weight"],
            float(heads[f"{file}_bias"]),
            trust=True if args.trust_checkpoint else None,
        ).to(args.device)
        ids, splits, maps, logits, tokens = [], [], [], [], []
        t0 = time.time()
        for split in ("train", "test"):
            dataset = IDRiDSegmentation(args.data_root, split=split, image_size=cfg["image_size"])
            count = len(dataset) if args.limit is None else min(args.limit, len(dataset))
            for i in range(count):
                image = dataset.image(i)
                heat, base = occlusion_map(
                    model, image, 0, patch=cfg["patch"], stride=cfg["stride"], batch_size=args.batch_size
                )
                ids.append(dataset.ids[i])
                splits.append(split)
                maps.append(resize_map(heat, MAP_SIZE))
                logits.append(base)
                tokens.append(model.tokens(image[None].to(args.device))[0].cpu().numpy().astype(np.float32))
                print(f"{name} {split} {i + 1}/{count} | logit {base:+.2f} | {time.time() - t0:.0f} s", flush=True)
        np.savez_compressed(
            args.out_dir / f"localization_{file}.npz",
            ids=np.array(ids),
            splits=np.array(splits),
            maps=np.stack(maps),
            logits=np.array(logits),
            tokens=np.stack(tokens),
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"Terminé -> {args.out_dir}")


if __name__ == "__main__":
    main()

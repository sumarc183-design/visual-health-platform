#!/usr/bin/env python3
"""Extrait les embeddings RETFound d'un jeu externe (APTOS, EyePACS) dans le cache.

Même extraction que run_baseline.py (`dataset_embeddings` : normalisation
ImageNet, RETFound figé) et même cache avec empreinte
(`<dataset>_<prétraitement>_<taille>_<checkpoint>.npz`). Écrit aussi
`<dataset>_labels.csv` à côté du cache (clé, stade, patient, format
d'image d'origine, indicateur de l'appareil) : l'évaluation
(scripts/external_validation.py) se fait ensuite sur CPU, sans les images.

Exemple (GPU, ~10 min sur T4 pour les 3 662 images APTOS) :
    python scripts/extract_embeddings.py --dataset aptos \\
        --data-root /kaggle/input/aptos2019-blindness-detection \\
        --checkpoint third_party/retfound/checkpoints/RETFound_mae_natureCFP.pth \\
        --out-dir embeddings
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_baseline import dataset_embeddings  # noqa: E402
from src.data.aptos_dataset import APTOSDataset  # noqa: E402
from src.data.eyepacs_dataset import EyePACSDataset  # noqa: E402
from src.data.fundus_preprocess import PREPROCESS_CHOICES, PREPROCESS_VERSION  # noqa: E402
from src.models.embedding_cache import EmbeddingCache, cache_file_name, file_fingerprint  # noqa: E402
from src.models.retfound_backbone import RETFoundBackbone  # noqa: E402

DATASETS = {"aptos": APTOSDataset, "eyepacs": EyePACSDataset}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True, help="Dossier du cache d'embeddings")
    parser.add_argument("--preprocess", choices=PREPROCESS_CHOICES, default="crop-mask")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--trust-checkpoint", action="store_true")
    args = parser.parse_args()

    dataset = DATASETS[args.dataset](args.data_root, image_size=args.image_size, preprocess=args.preprocess)
    cache = EmbeddingCache(
        args.out_dir / cache_file_name(args.dataset, args.preprocess, args.image_size, args.checkpoint),
        meta={
            "checkpoint_sha256": file_fingerprint(args.checkpoint),
            "preprocess": args.preprocess,
            "preprocess_version": PREPROCESS_VERSION,
            "image_size": args.image_size,
        },
    )
    holder: list[RETFoundBackbone] = []

    def get_backbone() -> RETFoundBackbone:
        if not holder:
            holder.append(
                RETFoundBackbone(
                    args.checkpoint,
                    img_size=args.image_size,
                    embedding_dim=1024,
                    trust_checkpoint=True if args.trust_checkpoint else None,
                ).to(args.device)
            )
        return holder[0]

    embed_kwargs = dict(batch_size=args.batch_size, device=args.device, num_workers=args.num_workers)
    X, _ = dataset_embeddings(dataset, args.dataset, get_backbone, cache, embed_kwargs)
    sizes = []
    for i in range(len(dataset)):
        with Image.open(dataset.image_path(i)) as image:  # en-tête seulement
            sizes.append((image.width, image.height))
    labels = pd.DataFrame(
        {
            "key": dataset.image_keys,
            "diagnosis": dataset.df["diagnosis"].to_numpy(),
            "group": dataset.groups,
            "width": [w for w, _ in sizes],
            "height": [h for _, h in sizes],
        }
    )
    if "id_code" in dataset.df:
        labels.insert(0, "id_code", dataset.df["id_code"].to_numpy())
    labels.to_csv(args.out_dir / f"{args.dataset}_labels.csv", index=False)
    print(f"{len(X)} embeddings -> {cache.path}")
    counts = labels["diagnosis"].value_counts().sort_index().to_dict()
    print(f"Stades -> {args.out_dir / f'{args.dataset}_labels.csv'} : {counts}")


if __name__ == "__main__":
    main()

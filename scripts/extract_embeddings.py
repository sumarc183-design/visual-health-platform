#!/usr/bin/env python3
"""Extrait les embeddings d'un jeu (APTOS, EyePACS, RFMiD, ISIC 2019) dans le cache.

Même extraction que run_baseline.py (`dataset_embeddings`, backbone figé) et
même cache avec empreinte
(`<dataset>_<prétraitement>_<taille>_<checkpoint>.npz`). Écrit aussi
`<dataset>_labels.csv` à côté du cache (clé, stade, patient, format
d'image d'origine, indicateur de l'appareil) : l'évaluation
(scripts/external_validation.py) se fait ensuite sur CPU, sans les images.

`--backbone retfound` (défaut : 224 px, normalisation ImageNet) ou `green`
(RETFound-Green : 392 px, normalisation 0,5) ; pour la dermatologie, `dinov2`
(ViT-B/14), `dinov2-large` (ViT-L/14) ou `panderm` (PanDerm-Large), 224 px, normalisation ImageNet. RFMiD et
ISIC 2019 : les trois découpages, sans fichier d'étiquettes (elles restent dans la
base).

Exemple (GPU, ~10 min sur T4 pour les 3 662 images APTOS) :
    python scripts/extract_embeddings.py --dataset aptos \\
        --data-root /kaggle/input/aptos2019-blindness-detection \\
        --checkpoint third_party/retfound/checkpoints/RETFound_mae_natureCFP.pth \\
        --out-dir embeddings
    python scripts/extract_embeddings.py --dataset rfmid --backbone green --data-root <RFMiD> \\
        --checkpoint third_party/retfound_green/retfoundgreen_statedict.pth --out-dir embeddings
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
from src.data.isic_dataset import ISIC2019Dataset  # noqa: E402
from src.data.rfmid_dataset import RFMiDDataset  # noqa: E402
from src.models.embedding_cache import EmbeddingCache, cache_file_name, file_fingerprint  # noqa: E402
from src.models.retfound_backbone import RETFoundBackbone  # noqa: E402
from src.models.retfound_green import GREEN_IMAGE_SIZE, RETFoundGreenBackbone  # noqa: E402
from src.models.timm_backbones import (  # noqa: E402
    DINOV2_IMAGE_SIZE,
    PANDERM_IMAGE_SIZE,
    DINOv2Backbone,
    DINOv2LargeBackbone,
    PanDermBackbone,
)

DATASETS = {"aptos": APTOSDataset, "eyepacs": EyePACSDataset, "rfmid": RFMiDDataset, "isic2019": ISIC2019Dataset}
SPLIT_DATASETS = ("rfmid", "isic2019")  # trois découpages, sans fichier d'étiquettes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True, help="Dossier du cache d'embeddings")
    parser.add_argument("--preprocess", choices=PREPROCESS_CHOICES, default="crop-mask")
    parser.add_argument(
        "--backbone", choices=["retfound", "green", "dinov2", "dinov2-large", "panderm"], default="retfound"
    )
    parser.add_argument("--image-size", type=int, default=None, help="Défaut : 392 (green), 224 (autres)")
    parser.add_argument("--limit", type=int, default=None, help="Essai rapide : nombre d'images par jeu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--trust-checkpoint", action="store_true")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val", "test"],
        default=["train", "val", "test"],
        help="RFMiD et ISIC : découpages à extraire (le test peut se télécharger pendant ce temps)",
    )
    args = parser.parse_args()
    if args.image_size is None:
        native = {
            "green": GREEN_IMAGE_SIZE,
            "dinov2": DINOV2_IMAGE_SIZE,
            "dinov2-large": DINOV2_IMAGE_SIZE,
            "panderm": PANDERM_IMAGE_SIZE,
        }
        args.image_size = native.get(args.backbone, 224)
    if args.dataset == "isic2019" and args.preprocess != "none":
        parser.error("ISIC : --preprocess none (crop-mask est propre au fond d'œil)")

    cache = EmbeddingCache(
        args.out_dir / cache_file_name(args.dataset, args.preprocess, args.image_size, args.checkpoint),
        meta={
            "checkpoint_sha256": file_fingerprint(args.checkpoint),
            "preprocess": args.preprocess,
            "preprocess_version": PREPROCESS_VERSION,
            "image_size": args.image_size,
        },
    )
    holder: list[torch.nn.Module] = []

    def get_backbone() -> torch.nn.Module:
        if not holder:
            if args.backbone == "green":
                backbone = RETFoundGreenBackbone(args.checkpoint, img_size=args.image_size)
            elif args.backbone == "dinov2":
                backbone = DINOv2Backbone(args.checkpoint, img_size=args.image_size)
            elif args.backbone == "dinov2-large":
                backbone = DINOv2LargeBackbone(args.checkpoint, img_size=args.image_size)
            elif args.backbone == "panderm":
                backbone = PanDermBackbone(args.checkpoint, img_size=args.image_size)
            else:
                backbone = RETFoundBackbone(
                    args.checkpoint,
                    img_size=args.image_size,
                    embedding_dim=1024,
                    trust_checkpoint=True if args.trust_checkpoint else None,
                )
            holder.append(backbone.to(args.device))
        return holder[0]

    embed_kwargs = dict(batch_size=args.batch_size, device=args.device, num_workers=args.num_workers)
    kwargs = dict(image_size=args.image_size, preprocess=args.preprocess)
    if args.dataset in SPLIT_DATASETS:
        for split in args.splits:
            if args.dataset == "isic2019":
                dataset = ISIC2019Dataset(args.data_root, split=split, image_size=args.image_size)
            else:
                dataset = RFMiDDataset(args.data_root, split=split, **kwargs)
            if args.limit is not None:
                dataset.df = dataset.df.iloc[: args.limit].reset_index(drop=True)
            X, _ = dataset_embeddings(dataset, f"{args.dataset} {split}", get_backbone, cache, embed_kwargs)
            print(f"{len(X)} embeddings -> {cache.path}")
        return
    dataset = DATASETS[args.dataset](args.data_root, **kwargs)
    if args.limit is not None:
        dataset.df = dataset.df.iloc[: args.limit].reset_index(drop=True)
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

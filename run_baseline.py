#!/usr/bin/env python3
"""Point d'entrée unique pour la baseline Visual Health (RETFound sur RFMiD/ODIR).

Enchaîne contrôle qualité -> backbone RETFound -> tête de classification ->
évaluation en réutilisant exclusivement le code déjà testé
(src/pipeline/quality_control.py, src/models/retfound_backbone.py,
src/models/classifier_head.py, src/data/rfmid_dataset.py,
src/data/odir_dataset.py) : pas de réimplémentation parallèle, pas de
taxonomie de classes inventée.

Sans --data-root : génère des images synthétiques (comme
tests/test_pipeline_smoke.py) pour un smoke test rapide.
Sans --checkpoint : utilise un backbone à poids aléatoires (résultats non
interprétables médicalement, plomberie uniquement -- clairement indiqué en
sortie).

Exemples :
    # Smoke test, tout synthétique
    python run_baseline.py --dataset rfmid

    # Vos vraies données RFMiD téléchargées, backbone aléatoire
    python run_baseline.py --dataset rfmid --data-root data/raw/rfmid

    # Vos vraies données ODIR + vrai checkpoint RETFound (premier vrai AUROC)
    python run_baseline.py --dataset odir --data-root data/raw/odir \\
        --checkpoint third_party/retfound/checkpoints/RETFound_mae_natureCFP.pth
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.generate_synthetic_fundus_dataset import generate_dataset
from src.data.odir_dataset import ODIRDataset
from src.data.rfmid_dataset import RFMiDDataset
from src.eval.metrics import evaluate_multilabel
from src.models.classifier_head import MultiLabelHead, train_head
from src.models.retfound_backbone import IMAGENET_MEAN, IMAGENET_STD, RETFoundBackbone, build_retfound_vit_large
from src.pipeline.quality_control import check_quality


def build_random_checkpoint(path: Path) -> None:
    model = build_retfound_vit_large(img_size=224, global_pool=True)
    torch.save({"model": model.state_dict()}, path)


class SyntheticFundusDataset(torch.utils.data.Dataset):
    """Adapte la sortie CSV de generate_synthetic_fundus_dataset au même
    format (image, labels) que RFMiDDataset/ODIRDataset, pour partager
    exactement le même code de collecte d'embeddings ci-dessous."""

    def __init__(self, data_dir: Path, image_size: int = 224):
        import csv

        from PIL import Image

        self.Image = Image
        self.data_dir = data_dir
        with open(data_dir / "labels.csv") as f:
            reader = csv.reader(f)
            self.class_names = next(reader)[1:]
            self.rows = list(reader)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        image = self.Image.open(self.data_dir / "images" / f"{row[0]}.png").convert("RGB")
        image = np.array(image.resize((self.image_size, self.image_size))).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        labels = torch.tensor([int(v) for v in row[1:]], dtype=torch.float32)
        return image, labels


def run_quality_control(dataset, n_samples: int = 50) -> None:
    """Sonde sur un échantillon, sans filtrer : le seuil de flou n'est pas
    encore calibré sur de vraies images, retirer des images fausserait le jeu."""
    n_check = min(n_samples, len(dataset))
    n_flagged = 0
    for i in range(n_check):
        image_t, _ = dataset[i]
        raw = (image_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        if not check_quality(raw, blur_threshold=1.0).is_usable:
            n_flagged += 1
    print(f"Contrôle qualité (échantillon de {n_check}) : {n_flagged} signalée(s), aucune retirée")


def collect_embeddings(
    backbone: RETFoundBackbone, dataset, batch_size: int = 16, device: str = "cpu", num_workers: int = 0
):
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    mean, std = IMAGENET_MEAN.to(device), IMAGENET_STD.to(device)
    embeddings, labels = [], []
    with torch.no_grad():
        for images, batch_labels in loader:
            images = (images.to(device) - mean) / std
            embeddings.append(backbone(images).cpu().numpy())
            labels.append(batch_labels.numpy())
    return np.concatenate(embeddings), np.concatenate(labels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["rfmid", "odir"], required=True)
    parser.add_argument(
        "--data-root", type=Path, default=None, help="Dossier des données téléchargées. Omis = synthétique."
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=None, help="Checkpoint RETFound officiel. Omis = poids aléatoires."
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda (GPU Colab/Kaggle) ou cpu. Par défaut : cuda si disponible.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--odir-labels",
        choices=["eye", "patient"],
        default="eye",
        help="ODIR : étiquette de l'œil photographié (défaut) ou diagnostic patient "
        "recopié sur les deux yeux (comportement d'origine, ~21 %% d'images mal "
        "étiquetées -- voir src/data/odir_dataset.py).",
    )
    parser.add_argument(
        "--eval-split",
        choices=["val", "test"],
        default="val",
        help="Jeu d'évaluation. test = jeu de test officiel RFMiD, à n'utiliser qu'une fois les "
        "réglages figés sur val. ODIR n'a pas de test étiqueté : val uniquement.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Graine de l'entraînement de la tête (résultats reproductibles)"
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        default=None,
        help="Enregistre les métriques (validation, train, gap) dans ce fichier JSON",
    )
    parser.add_argument(
        "--save-head",
        type=Path,
        default=None,
        help="Sauvegarde la tête entraînée (+ classes, dataset, taille d'image) pour scripts/gradcam_examples.py.",
    )
    args = parser.parse_args()
    if args.eval_split == "test" and (args.dataset != "rfmid" or args.data_root is None):
        parser.error("--eval-split test n'existe que pour RFMiD réel (ODIR : pas de test étiqueté).")

    is_synthetic = args.data_root is None
    is_random_checkpoint = args.checkpoint is None

    with tempfile.TemporaryDirectory() as tmp:
        checkpoint_path = args.checkpoint
        if is_random_checkpoint:
            checkpoint_path = Path(tmp) / "random_checkpoint.pth"
            build_random_checkpoint(checkpoint_path)

        if is_synthetic:
            data_dir = Path(tmp) / "synthetic"
            print(f"[synthétique] Génération d'images {args.dataset} factices...")
            generate_dataset(str(data_dir), dataset=args.dataset, num_images=40, image_size=args.image_size, seed=42)
            train_dataset = SyntheticFundusDataset(data_dir, image_size=args.image_size)
            val_dataset = train_dataset  # un seul jeu synthétique : plomberie uniquement, annoncé plus bas
        elif args.dataset == "rfmid":
            train_dataset = RFMiDDataset(args.data_root, split="train", image_size=args.image_size)
            val_dataset = RFMiDDataset(args.data_root, split=args.eval_split, image_size=args.image_size)
        else:
            train_dataset = ODIRDataset(
                args.data_root, split="train", image_size=args.image_size, label_level=args.odir_labels
            )
            val_dataset = ODIRDataset(
                args.data_root, split="val", image_size=args.image_size, label_level=args.odir_labels
            )

        class_names = train_dataset.class_names
        print(f"Classes ({len(class_names)}) : {class_names}")
        print(f"Train : {len(train_dataset)} images | {args.eval_split.capitalize()} : {len(val_dataset)} images")

        run_quality_control(train_dataset)

        backbone = RETFoundBackbone(checkpoint_path, img_size=args.image_size, embedding_dim=1024).to(args.device)
        print(f"Backbone sur : {args.device}")
        embed_kwargs = dict(batch_size=args.batch_size, device=args.device, num_workers=args.num_workers)

        print("Extraction des embeddings (train)...")
        X_train, y_train = collect_embeddings(backbone, train_dataset, **embed_kwargs)
        print("Extraction des embeddings (val)...")
        X_val, y_val = collect_embeddings(backbone, val_dataset, **embed_kwargs)

        torch.manual_seed(args.seed)  # initialisation de la tête et ordre des mini-batchs
        train_loader = DataLoader(
            list(zip(torch.from_numpy(X_train), torch.from_numpy(y_train), strict=True)), batch_size=16, shuffle=True
        )
        head = MultiLabelHead(embedding_dim=1024, num_classes=len(class_names))
        head = train_head(head, train_loader, epochs=args.epochs, lr=1e-3, device=args.device)

        with torch.no_grad():
            probs = torch.sigmoid(head(torch.from_numpy(X_val).to(args.device))).cpu().numpy()
            train_probs = torch.sigmoid(head(torch.from_numpy(X_train).to(args.device))).cpu().numpy()
        results = evaluate_multilabel(y_val, probs, class_names)
        train_results = evaluate_multilabel(y_train, train_probs, class_names)

        if args.save_head is not None:
            args.save_head.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": head.cpu().state_dict(),
                    "class_names": list(class_names),
                    "dataset": args.dataset,
                    "image_size": args.image_size,
                    "odir_labels": args.odir_labels if args.dataset == "odir" else None,
                    "auroc_per_class": results["auroc_per_class"],
                },
                args.save_head,
            )
            print(f"Tête sauvegardée : {args.save_head}")

    print("=" * 60)
    labels_desc = (
        f", étiquettes {'par œil' if args.odir_labels == 'eye' else 'patient'}" if args.dataset == "odir" else ""
    )
    print(f"Dataset          : {args.dataset.upper()} ({'SYNTHÉTIQUE' if is_synthetic else 'RÉEL'}{labels_desc})")
    checkpoint_desc = "ALÉATOIRE (pas de poids RETFound officiels)" if is_random_checkpoint else args.checkpoint
    print(f"Checkpoint       : {checkpoint_desc}")
    print(f"Évaluation       : {args.eval_split}")
    n_auroc = len(results["auroc_per_class"])
    if n_auroc < len(class_names):
        print(
            f"AUROC macro sur {n_auroc}/{len(class_names)} classes "
            f"(les autres n'ont aucun positif en {args.eval_split})"
        )
    print(
        f"AUROC macro      : {results['auroc_macro']:.4f}  (train {train_results['auroc_macro']:.4f}, "
        f"gap {train_results['auroc_macro'] - results['auroc_macro']:+.4f})"
    )
    print(
        f"F1 macro (0,5)   : {results['f1_macro']:.4f}  (train {train_results['f1_macro']:.4f}, "
        f"gap {train_results['f1_macro'] - results['f1_macro']:+.4f})"
    )
    for name, value in results["auroc_per_class"].items():
        print(f"  {name:>20s} : {value:.4f}")
    print("=" * 60)
    if args.results_json is not None:
        args.results_json.parent.mkdir(parents=True, exist_ok=True)
        args.results_json.write_text(
            json.dumps(
                {
                    "dataset": args.dataset,
                    "odir_labels": args.odir_labels if args.dataset == "odir" else None,
                    "synthetic": is_synthetic,
                    "checkpoint": None if is_random_checkpoint else str(args.checkpoint),
                    "eval_split": args.eval_split,
                    "seed": args.seed,
                    "epochs": args.epochs,
                    "val": results,
                    "train": {k: v for k, v in train_results.items() if k != "auroc_per_class"},
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        print(f"Résultats JSON : {args.results_json}")
    if is_synthetic or is_random_checkpoint:
        print(
            "Ces résultats ne sont PAS interprétables médicalement (données et/ou "
            "checkpoint non réels) -- ils valident uniquement que le pipeline tourne."
        )


if __name__ == "__main__":
    main()

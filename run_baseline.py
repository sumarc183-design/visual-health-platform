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
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.generate_synthetic_fundus_dataset import generate_dataset
from src.data.fundus_preprocess import PREPROCESS_CHOICES, PREPROCESS_VERSION
from src.data.odir_dataset import TEST_FRACTION, ODIRDataset
from src.data.rfmid_dataset import RFMiDDataset
from src.eval.metrics import bootstrap_auroc_ci, evaluate_multilabel
from src.models.classifier_head import MultiLabelHead, train_head
from src.models.embedding_cache import EmbeddingCache, cache_file_name, file_fingerprint
from src.models.retfound_backbone import IMAGENET_MEAN, IMAGENET_STD, RETFoundBackbone, build_retfound_vit_large
from src.pipeline.quality_control import check_fundus_file, check_quality


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


def run_quality_control(dataset, n_samples: int = 50, synthetic_blur_threshold: float = 1.0) -> None:
    """Sonde sur un échantillon, sans filtrer : retirer des images changerait
    le jeu d'évaluation. Données réelles : contrôle sur le fichier d'origine
    (résolution avant redimensionnement, flou sur la rétine à échelle fixe,
    seuil calibré DEFAULT_BLUR_THRESHOLD). Synthétique : sur l'image chargée,
    seuil propre aux images générées."""
    n_check = min(n_samples, len(dataset))
    n_flagged = 0
    for i in range(n_check):
        if hasattr(dataset, "image_path"):
            report = check_fundus_file(dataset.image_path(i))  # seuil de flou calibré par défaut
        else:
            image_t, _ = dataset[i]
            raw = (image_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            report = check_quality(raw, blur_threshold=synthetic_blur_threshold)
        n_flagged += not report.is_usable
    print(f"Contrôle qualité (échantillon de {n_check}) : {n_flagged} signalée(s), aucune retirée")


def collect_embeddings(
    backbone: RETFoundBackbone, dataset, batch_size: int = 16, device: str = "cpu", num_workers: int = 0
):
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    # Normalisation du backbone quand il en déclare une (RETFound-Green), sinon ImageNet
    mean = getattr(backbone, "input_mean", IMAGENET_MEAN).to(device)
    std = getattr(backbone, "input_std", IMAGENET_STD).to(device)
    embeddings, labels = [], []
    with torch.no_grad():
        for images, batch_labels in loader:
            images = (images.to(device) - mean) / std
            embeddings.append(backbone(images).cpu().numpy())
            labels.append(batch_labels.numpy())
    return np.concatenate(embeddings), np.concatenate(labels)


def dataset_embeddings(dataset, label: str, get_backbone, cache: EmbeddingCache | None, embed_kwargs: dict):
    """Embeddings et étiquettes d'un jeu. Avec cache, seules les images
    absentes passent dans le backbone (chargé seulement s'il le faut)."""
    if cache is None:
        print(f"Extraction des embeddings ({label})...")
        return collect_embeddings(get_backbone(), dataset, **embed_kwargs)
    keys = dataset.image_keys
    missing = set(cache.missing(keys))  # calculé une fois : sinon quadratique en nombre d'images
    todo = [i for i, key in enumerate(keys) if key in missing]
    print(f"Embeddings ({label}) : {len(keys) - len(todo)} en cache, {len(todo)} à extraire")
    if todo:
        X_new, _ = collect_embeddings(get_backbone(), Subset(dataset, todo), **embed_kwargs)
        cache.add([keys[i] for i in todo], X_new)
        cache.save()
    return cache.get(keys), dataset.df[dataset.class_names].to_numpy(dtype=np.float32)


def git_commit() -> str | None:
    """Commit du code exécuté (+ « -dirty » si des fichiers suivis sont modifiés)."""
    repo = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return commit + ("-dirty" if dirty else "")


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
    parser.add_argument("--epochs", type=int, default=30, help="Epochs de la tête (30 = réglage des résultats publiés)")
    parser.add_argument(
        "--head-batch-size",
        type=int,
        default=16,
        help="Taille des mini-batchs de la tête (16 = résultats publiés). --batch-size ne règle que l'extraction.",
    )
    parser.add_argument(
        "--trust-checkpoint",
        action="store_true",
        help="Autorise le chargement complet (pickle) d'un checkpoint hors de third_party/retfound/checkpoints/, "
        "si torch.load(weights_only=True) le refuse. Seulement pour un fichier de confiance.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda (GPU Colab/Kaggle) ou cpu. Par défaut : cuda si disponible.",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Taille des batchs d'extraction des embeddings")
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
        "--preprocess",
        choices=PREPROCESS_CHOICES,
        default="crop-mask",
        help="crop-mask (défaut) = recadrage sur la rétine + masque commun, contre le raccourci "
        "d'acquisition (voir src/data/fundus_preprocess.py) ; none = redimensionnement direct "
        "(comportement d'origine, déforme selon l'appareil).",
    )
    parser.add_argument(
        "--eval-split",
        choices=["val", "test"],
        default="val",
        help="Jeu d'évaluation. test = jeu de test (officiel pour RFMiD, 15 %% des patients mis de côté "
        "pour ODIR), à n'utiliser qu'une fois les réglages figés sur val.",
    )
    parser.add_argument(
        "--odir-test-fraction",
        type=float,
        default=TEST_FRACTION,
        help="ODIR : part des images mise de côté pour le test (défaut 0,15 : split 70/15/15). "
        "0 = split d'origine 85/15 sans test (premiers résultats publiés).",
    )
    parser.add_argument(
        "--embeddings-cache",
        type=Path,
        default=None,
        help="Dossier du cache d'embeddings (Google Drive dans Colab) : les images déjà vues ne repassent "
        "pas dans le backbone. Ignoré en synthétique ou avec des poids aléatoires.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Enregistre les probabilités image par image du jeu évalué (.npz) : intervalles de confiance, "
        "comparaison appariée (scripts/compare_predictions.py), seuils, calibration sans relancer le GPU.",
    )
    parser.add_argument(
        "--bootstrap", type=int, default=1000, help="Tirages bootstrap pour les IC à 95 %% (0 = désactivé)"
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
    if args.eval_split == "test" and args.data_root is None:
        parser.error("--eval-split test demande des données réelles (--data-root).")
    if args.eval_split == "test" and args.dataset == "odir" and args.odir_test_fraction == 0:
        parser.error("--eval-split test impossible avec --odir-test-fraction 0 (split d'origine sans test).")

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
            train_dataset = RFMiDDataset(
                args.data_root, split="train", image_size=args.image_size, preprocess=args.preprocess
            )
            val_dataset = RFMiDDataset(
                args.data_root, split=args.eval_split, image_size=args.image_size, preprocess=args.preprocess
            )
        else:
            odir_kwargs = dict(
                image_size=args.image_size,
                label_level=args.odir_labels,
                preprocess=args.preprocess,
                test_fraction=args.odir_test_fraction,
            )
            train_dataset = ODIRDataset(args.data_root, split="train", **odir_kwargs)
            val_dataset = ODIRDataset(args.data_root, split=args.eval_split, **odir_kwargs)

        class_names = train_dataset.class_names
        print(f"Classes ({len(class_names)}) : {class_names}")
        print(f"Train : {len(train_dataset)} images | {args.eval_split.capitalize()} : {len(val_dataset)} images")

        run_quality_control(train_dataset)

        backbone_holder: list[RETFoundBackbone] = []

        def get_backbone() -> RETFoundBackbone:
            if not backbone_holder:
                backbone_holder.append(
                    RETFoundBackbone(
                        checkpoint_path,
                        img_size=args.image_size,
                        embedding_dim=1024,
                        trust_checkpoint=True if args.trust_checkpoint else None,
                    ).to(args.device)
                )
                print(f"Backbone sur : {args.device}")
            return backbone_holder[0]

        checkpoint_fingerprint = None
        if not is_random_checkpoint:
            if args.checkpoint.exists():
                checkpoint_fingerprint = file_fingerprint(args.checkpoint)
            else:  # possible si tous les embeddings sont en cache (tête seule, sur CPU)
                print(f"⚠ Checkpoint absent ({args.checkpoint}) : son empreinte ne peut pas être vérifiée.")
        cache = None
        if args.embeddings_cache is not None and not (is_synthetic or is_random_checkpoint):
            cache_name = cache_file_name(args.dataset, args.preprocess, args.image_size, args.checkpoint)
            cache = EmbeddingCache(
                args.embeddings_cache / cache_name,
                meta={
                    "checkpoint_sha256": checkpoint_fingerprint,
                    "preprocess": args.preprocess,
                    "preprocess_version": PREPROCESS_VERSION,
                    "image_size": args.image_size,
                },
            )
        embed_kwargs = dict(batch_size=args.batch_size, device=args.device, num_workers=args.num_workers)
        X_train, y_train = dataset_embeddings(train_dataset, "train", get_backbone, cache, embed_kwargs)
        X_val, y_val = dataset_embeddings(val_dataset, args.eval_split, get_backbone, cache, embed_kwargs)

        torch.manual_seed(args.seed)  # initialisation de la tête et ordre des mini-batchs
        train_loader = DataLoader(
            list(zip(torch.from_numpy(X_train), torch.from_numpy(y_train), strict=True)),
            batch_size=args.head_batch_size,
            shuffle=True,
        )
        head = MultiLabelHead(embedding_dim=1024, num_classes=len(class_names))
        head = train_head(head, train_loader, epochs=args.epochs, lr=1e-3, device=args.device)

        with torch.no_grad():
            probs = torch.sigmoid(head(torch.from_numpy(X_val).to(args.device))).cpu().numpy()
            train_probs = torch.sigmoid(head(torch.from_numpy(X_train).to(args.device))).cpu().numpy()
        results = evaluate_multilabel(y_val, probs, class_names)
        train_results = evaluate_multilabel(y_train, train_probs, class_names)
        eval_keys = getattr(val_dataset, "image_keys", [str(i) for i in range(len(val_dataset))])
        eval_groups = getattr(val_dataset, "groups", eval_keys)
        if args.bootstrap > 0:
            results.update(bootstrap_auroc_ci(y_val, probs, class_names, eval_groups, n_boot=args.bootstrap))
        if args.predictions is not None:
            args.predictions.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                args.predictions,
                keys=np.array(eval_keys),
                groups=np.array(eval_groups),
                y_true=y_val,
                probs=probs,
                class_names=np.array(class_names),
            )
            print(f"Prédictions : {args.predictions}")

        if args.save_head is not None:
            args.save_head.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": head.cpu().state_dict(),
                    "class_names": list(class_names),
                    "dataset": args.dataset,
                    "image_size": args.image_size,
                    "preprocess": args.preprocess,
                    "odir_labels": args.odir_labels if args.dataset == "odir" else None,
                    "odir_test_fraction": args.odir_test_fraction if args.dataset == "odir" else None,
                    "auroc_per_class": results["auroc_per_class"],
                    "preprocess_version": PREPROCESS_VERSION,
                    "checkpoint_sha256": checkpoint_fingerprint,
                    "head_training": {"lr": 1e-3, "epochs": args.epochs, "batch_size": args.head_batch_size},
                    "seed": args.seed,
                    "git_commit": git_commit(),
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
    print(f"Prétraitement    : {args.preprocess}")
    n_auroc = len(results["auroc_per_class"])
    if n_auroc < len(class_names):
        print(
            f"AUROC macro sur {n_auroc}/{len(class_names)} classes "
            f"(les autres n'ont aucun positif en {args.eval_split})"
        )
    macro_ci = results.get("auroc_macro_ci")
    ci_text = f" [IC 95 % {macro_ci[0]:.3f}-{macro_ci[1]:.3f}]" if macro_ci else ""
    print(
        f"AUROC macro      : {results['auroc_macro']:.4f}{ci_text}  (train {train_results['auroc_macro']:.4f}, "
        f"gap {train_results['auroc_macro'] - results['auroc_macro']:+.4f})"
    )
    print(
        f"F1 macro (0,5)   : {results['f1_macro']:.4f}  (train {train_results['f1_macro']:.4f}, "
        f"gap {train_results['f1_macro'] - results['f1_macro']:+.4f})"
    )
    class_ci = results.get("auroc_per_class_ci", {})
    for name, value in results["auroc_per_class"].items():
        ci = f"  [{class_ci[name][0]:.3f}-{class_ci[name][1]:.3f}]" if name in class_ci else ""
        n_pos = results["positives_per_class"][name]
        unstable = "  ⚠ moins de 10 positifs : IC peu fiable" if n_pos < 10 else ""
        print(f"  {name:>20s} : {value:.4f}{ci}  (n+ = {n_pos}){unstable}")
    print("=" * 60)
    if args.results_json is not None:
        args.results_json.parent.mkdir(parents=True, exist_ok=True)
        args.results_json.write_text(
            json.dumps(
                {
                    "dataset": args.dataset,
                    "odir_labels": args.odir_labels if args.dataset == "odir" else None,
                    "odir_test_fraction": args.odir_test_fraction if args.dataset == "odir" else None,
                    "synthetic": is_synthetic,
                    "checkpoint": None if is_random_checkpoint else str(args.checkpoint),
                    "eval_split": args.eval_split,
                    "preprocess": args.preprocess,
                    "seed": args.seed,
                    "epochs": args.epochs,
                    # Métriques du jeu évalué (eval_split). "val" est gardé pour la
                    # compatibilité avec les JSON déjà publiés : il contient le test
                    # quand eval_split vaut "test". Préférer "eval".
                    "eval": results,
                    "val": results,
                    "train": {k: v for k, v in train_results.items() if k in ("auroc_macro", "f1_macro")},
                    "manifest": {
                        "git_commit": git_commit(),
                        "checkpoint_sha256": checkpoint_fingerprint,
                        "preprocess_version": PREPROCESS_VERSION,
                        "image_size": args.image_size,
                        "head": {
                            "optimizer": "Adam",
                            "lr": 1e-3,
                            "epochs": args.epochs,
                            "batch_size": args.head_batch_size,
                            "seed": args.seed,
                        },
                        "n_train": int(len(y_train)),
                        "n_eval": int(len(y_val)),
                        "python": platform.python_version(),
                        "torch": torch.__version__,
                        "numpy": np.__version__,
                    },
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

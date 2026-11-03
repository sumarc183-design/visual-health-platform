"""Génère un jeu de données SYNTHÉTIQUE d'images de fond d'œil + labels
multi-label au format RFMiD/ODIR, pour faire tourner tout le pipeline
(contrôle qualité -> backbone -> classification -> Grad-CAM) sans attendre
le vrai téléchargement.

A REMPLACER par de vraies images RFMiD/ODIR dès que le téléchargement est
possible (scripts/download_rfmid_odir.py) -- les images générées ici sont
du bruit avec un léger vignettage circulaire pour ressembler grossièrement
à un fond d'œil, pas de la vraie physiologie. Ne sert qu'à valider la
plomberie du pipeline.
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np
from PIL import Image

RFMID_CLASSES = [
    f"C{i}" for i in range(28)
]  # noms génériques -- remplacer par la vraie taxonomie RFMiD au téléchargement réel
ODIR_CLASSES = ["N", "D", "G", "C", "A", "H", "M", "O"]


def generate_fundus_image(size: int, rng: np.random.Generator) -> np.ndarray:
    """Bruit RGB avec un vignettage circulaire, pour au moins avoir une
    zone centrale "utile" et des bords sombres comme une vraie photo de
    fond d'œil (sans quoi le contrôle qualité rejetterait tout comme
    hors-cadre)."""
    image = (rng.random((size, size, 3)) * 255).astype(np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    center = size / 2
    dist = np.sqrt((xx - center) ** 2 + (yy - center) ** 2) / center
    vignette = np.clip(1.2 - dist, 0, 1)[..., None]
    image = (image * vignette).astype(np.uint8)
    return image


def generate_dataset(out_dir: str, dataset: str, num_images: int, image_size: int, seed: int) -> None:
    classes = RFMID_CLASSES if dataset == "rfmid" else ODIR_CLASSES
    rng = np.random.default_rng(seed)

    images_dir = os.path.join(out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    rows = []
    for i in range(num_images):
        image_id = f"{dataset}_{i:05d}"
        image = generate_fundus_image(image_size, rng)
        Image.fromarray(image).save(os.path.join(images_dir, f"{image_id}.png"))

        # Multi-label déséquilibré (comme les vraies bases) : ~85% de
        # "normal" (aucun label positif), le reste avec 1-3 labels.
        labels = np.zeros(len(classes), dtype=int)
        if rng.random() > 0.85:
            n_positive = rng.integers(1, 4)
            positive_idx = rng.choice(len(classes), size=n_positive, replace=False)
            labels[positive_idx] = 1
        rows.append([image_id] + labels.tolist())

    labels_path = os.path.join(out_dir, "labels.csv")
    with open(labels_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID"] + classes)
        writer.writerows(rows)

    print(f"{num_images} image(s) {dataset} générée(s) dans {images_dir}, labels dans {labels_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--dataset", choices=["rfmid", "odir"], default="rfmid")
    parser.add_argument("--num-images", type=int, default=40)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_dataset(args.out, args.dataset, args.num_images, args.image_size, args.seed)


if __name__ == "__main__":
    main()

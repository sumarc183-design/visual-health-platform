#!/usr/bin/env python3
"""Diagnostic exploratoire d'exposition sur le train ODIR uniquement (CPU).

Plan du 9 novembre 2026 : jusqu'à 120 images d'entraînement, graine 0 ;
recadrage commun à 512 px, masque intérieur du contrôle de flou. Sur les mêmes
images, atténuer/amplifier les canaux RGB par 0,25, 0,5, 1, 1,5 et 2, avec
écrêtage à 255. La géométrie est fixée avant ces perturbations.

Sortie : médiane et quartiles de la luminance moyenne/médiane et des fractions
de pixels sombres/clairs. Aucun choix de seuil, aucune évaluation de maladie,
aucune lecture des images de validation ou de test, aucune nouvelle règle de rejet.
Les perturbations contrôlées ne valident pas la détection de défauts réels.
L'archive JSON ne contient ni identifiant ni mesure individuelle.

Exemple :
    python scripts/quality_exposure_diagnostics.py --odir-root data/raw/odir \\
        --out results/quality_exposure_diagnostics.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.fundus_preprocess import PREPROCESS_VERSION, crop_mask  # noqa: E402
from src.data.odir_dataset import ODIRDataset  # noqa: E402
from src.eval.provenance import run_info  # noqa: E402
from src.pipeline.exposure import BRIGHT_LEVEL, DARK_LEVEL, exposure_statistics  # noqa: E402
from src.pipeline.quality_control import BLUR_ANALYSIS_SIZE, retina_blur_mask  # noqa: E402

RGB_GAINS = (0.25, 0.5, 1.0, 1.5, 2.0)


def summarize(records: list[dict[str, float]]) -> dict:
    return {
        key: {
            "median": float(np.median(values := [row[key] for row in records])),
            "quartiles": np.percentile(values, [25, 75]).tolist(),
        }
        for key in records[0]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--odir-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.sample_size < 1:
        parser.error("--sample-size doit être positif")

    dataset = ODIRDataset(args.odir_root, split="train")
    if not len(dataset):
        parser.error("jeu d'entraînement vide")
    indices = np.random.default_rng(args.seed).choice(len(dataset), min(args.sample_size, len(dataset)), replace=False)
    mask = retina_blur_mask()
    records = {gain: [] for gain in RGB_GAINS}
    digest = hashlib.sha256()
    for index in indices:
        path = dataset.image_path(int(index))
        digest.update(path.name.encode() + b"\0" + hashlib.sha256(path.read_bytes()).digest())
        with Image.open(path) as image:
            retina = crop_mask(image.convert("RGB"), BLUR_ANALYSIS_SIZE)
        for gain in RGB_GAINS:
            adjusted = np.clip(retina.astype(np.float32) * gain, 0, 255).round().astype(np.uint8)
            records[gain].append(exposure_statistics(adjusted, mask))
    report = {
        "run": run_info(),
        "plan_date": "2026-11-09",
        "dataset": "ODIR",
        "split": "train",
        "seed": args.seed,
        "images": len(indices),
        "input_sample_sha256": digest.hexdigest(),
        "preprocess": "crop-mask",
        "preprocess_version": PREPROCESS_VERSION,
        "analysis_size": BLUR_ANALYSIS_SIZE,
        "mask": "retina_blur_mask (champ intérieur, géométrie fixée avant perturbation)",
        "luminance_scale": [0, 255],
        "dark_level_inclusive": DARK_LEVEL,
        "bright_level_inclusive": BRIGHT_LEVEL,
        "rgb_gains": {str(gain): summarize(rows) for gain, rows in records.items()},
        "limitations": [
            "Diagnostic descriptif sur des perturbations synthétiques du train ; aucune validation clinique.",
            "Aucune annotation de qualité réelle utilisée ; aucune performance de détection estimée.",
            "Aucun seuil de rejet choisi ; contrôle qualité et prédictions inchangés.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"{len(indices)} images du train ODIR ; sortie : {args.out}")


if __name__ == "__main__":
    main()

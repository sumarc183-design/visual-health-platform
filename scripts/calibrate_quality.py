#!/usr/bin/env python3
"""Calibration du seuil de flou du contrôle qualité (CPU, ~10 min).

Le score de flou (variance du Laplacien sur la rétine recadrée à 512 px,
src/pipeline/quality_control.py) n'avait pas de seuil validé. Données :
ODIR (images d'origine et annotations data.xlsx) et RFMiD.

1. **Seuil** : percentile bas du score sur les images d'entraînement ODIR
   jugées acceptables (gardées dans le jeu nettoyé `full_df.csv`, sans
   mot-clé de qualité) : `--percentile 2` = 2 % de fausses alertes attendues
   sur des images comparables. Choisi sur le train seulement.
2. **Images réellement mauvaises** : les 21 yeux annotés « low image
   quality » et les 405 « lens dust » (data.xlsx d'ODIR, dont la plupart
   ont été retirés du jeu nettoyé) : part détectée, AUROC du score.
3. **Flou simulé** : flou gaussien d'intensité connue sur des images de
   validation ODIR (pas celles du train) : part détectée par intensité.
4. **Impact sur le modèle** : perte moyenne des prédictions sauvegardées
   (validation et test, `crop-mask`) par quintile de netteté, et images
   signalées contre les autres (IC par bootstrap).

Lire data.xlsx demande `openpyxl` (requirements-dev.txt).

Exemple :
    python scripts/calibrate_quality.py --odir-root data/raw/odir --rfmid-root data/raw/rfmid \\
        --out results/quality_calibration.json
"""

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.fundus_preprocess import crop_mask  # noqa: E402
from src.data.odir_dataset import ODIRDataset  # noqa: E402
from src.pipeline.quality_control import (  # noqa: E402
    BLUR_ANALYSIS_SIZE,
    check_fundus_file,
    compute_blur_score,
    retina_blur_mask,
)

QUALITY_KEYWORDS = ("low image quality", "lens dust")
RFMID_DIRS = {
    "train": ("Training_Set", "Training_Set", "Training"),
    "val": ("Evaluation_Set", "Evaluation_Set", "Validation"),
    "test": ("Test_Set", "Test_Set", "Test"),
}


def blur_score(path: str) -> float:
    return check_fundus_file(path, blur_threshold=0).blur_score


def odir_keywords(odir_root: Path) -> dict[str, str]:
    table = pd.read_excel(odir_root / "ODIR-5K" / "ODIR-5K" / "data.xlsx")
    keywords = {}
    for side in ("Left", "Right"):
        for fundus, words in zip(table[f"{side}-Fundus"], table[f"{side}-Diagnostic Keywords"], strict=True):
            keywords[fundus] = str(words).replace("，", ",").lower()
    return keywords


def model_loss_by_blur(results_dir: Path, name: str, scores: dict, threshold: float, rng) -> dict:
    ys, ps, blur = [], [], []
    for split in ("val", "test"):
        pred = np.load(results_dir / f"pred_{name}_crop-mask_{split}.npz")
        ys.append(pred["y_true"])
        ps.append(pred["probs"])
        blur += [scores[k] for k in pred["keys"]]
    y, p, blur = np.vstack(ys), np.clip(np.vstack(ps), 1e-6, 1 - 1e-6), np.array(blur)
    loss = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean(1)
    quintile = pd.qcut(blur, 5, labels=False)
    flagged = blur < threshold
    out = {
        "images": int(len(blur)),
        "loss_by_quintile_least_to_most_sharp": [float(loss[quintile == q].mean()) for q in range(5)],
        "flagged": int(flagged.sum()),
    }
    if flagged.sum() and (~flagged).sum():
        diffs = []
        for _ in range(2000):
            i = rng.integers(0, len(blur), len(blur))
            if flagged[i].any() and (~flagged[i]).any():
                diffs.append(loss[i][flagged[i]].mean() - loss[i][~flagged[i]].mean())
        out["loss_flagged_minus_others"] = float(loss[flagged].mean() - loss[~flagged].mean())
        out["loss_flagged_minus_others_ci"] = [float(x) for x in np.percentile(diffs, [2.5, 97.5])]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--odir-root", type=Path, required=True)
    parser.add_argument("--rfmid-root", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--percentile", type=float, default=2.0)
    parser.add_argument("--synthetic-images", type=int, default=120)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(0)

    odir_dir = args.odir_root / "ODIR-5K" / "ODIR-5K" / "Training Images"
    odir_files = sorted(odir_dir.glob("*.jpg"))
    rfmid_files = [
        (f"{split}/{p.stem}", p)
        for split, parts in RFMID_DIRS.items()
        for p in sorted((args.rfmid_root.joinpath(*parts)).glob("*.png"))
    ]
    with Pool(args.workers) as pool:
        odir_values = pool.map(blur_score, [str(p) for p in odir_files], 16)
        rfmid_values = pool.map(blur_score, [str(p) for _, p in rfmid_files], 16)
    odir_scores = dict(zip([p.name for p in odir_files], odir_values, strict=True))
    rfmid_scores = dict(zip([k for k, _ in rfmid_files], rfmid_values, strict=True))
    print(f"Scores : {len(odir_scores)} images ODIR, {len(rfmid_scores)} RFMiD")

    keywords = odir_keywords(args.odir_root)
    kept = set(pd.read_csv(args.odir_root / "full_df.csv")["filename"])
    train = set(ODIRDataset(args.odir_root, split="train").image_keys)
    names = np.array(list(odir_scores))
    score = np.array([odir_scores[n] for n in names])
    has = {k: np.array([k in keywords.get(n, "") for n in names]) for k in QUALITY_KEYWORDS}
    acceptable_train = (
        np.array([n in train and n in kept for n in names]) & ~has["low image quality"] & ~has["lens dust"]
    )
    threshold = float(np.percentile(score[acceptable_train], args.percentile))

    report = {
        "threshold": threshold,
        "percentile_of_acceptable_train": args.percentile,
        "acceptable_train_images": int(acceptable_train.sum()),
        "odir_flagged_share": float((score < threshold).mean()),
        "rfmid_flagged_share": float((np.array(list(rfmid_scores.values())) < threshold).mean()),
        "real_quality_labels": {},
    }
    for key, mask in has.items():
        report["real_quality_labels"][key] = {
            "images": int(mask.sum()),
            "detected_share": float((score[mask] < threshold).mean()),
            "median_score": float(np.median(score[mask])),
            "auroc_low_score": float(roc_auc_score(mask, -score)),
        }
    report["median_score_acceptable"] = float(np.median(score[acceptable_train]))

    val_keys = ODIRDataset(args.odir_root, split="val").image_keys
    sample = rng.choice(val_keys, size=min(args.synthetic_images, len(val_keys)), replace=False)
    images = [Image.open(odir_dir / k).convert("RGB") for k in sample]
    mask = retina_blur_mask()
    report["synthetic_blur_detected_share"] = {}
    for sigma in (0, 2, 3, 4, 6, 8):
        blurred = [im.filter(ImageFilter.GaussianBlur(sigma)) if sigma else im for im in images]
        values = np.array([compute_blur_score(crop_mask(im, BLUR_ANALYSIS_SIZE), mask) for im in blurred])
        report["synthetic_blur_detected_share"][str(sigma)] = float((values < threshold).mean())

    report["model_loss"] = {
        "odir": model_loss_by_blur(args.results, "odir", odir_scores, threshold, rng),
        "rfmid": model_loss_by_blur(args.results, "rfmid", rfmid_scores, threshold, rng),
    }

    real = report["real_quality_labels"]
    print(
        f"Seuil {threshold:.2f} (P{args.percentile:g} des images acceptables du train ODIR) : signale "
        f"{report['odir_flagged_share']:.1%} d'ODIR, {report['rfmid_flagged_share']:.1%} de RFMiD"
    )
    for key, r in real.items():
        print(
            f"  « {key} » ({r['images']} images) : {r['detected_share']:.0%} détectées, "
            f"AUROC {r['auroc_low_score']:.2f}"
        )
    print(
        "  Flou simulé détecté : "
        + " | ".join(f"σ {s} : {v:.0%}" for s, v in report["synthetic_blur_detected_share"].items())
    )
    for name, m in report["model_loss"].items():
        quintiles = " ".join(f"{v:.3f}" for v in m["loss_by_quintile_least_to_most_sharp"])
        print(f"  Perte du modèle {name} par quintile de netteté : {quintiles}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"Résultats : {args.out}")


if __name__ == "__main__":
    main()

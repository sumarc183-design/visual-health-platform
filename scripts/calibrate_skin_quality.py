#!/usr/bin/env python3
"""Contrôle qualité des images de peau : seuil de flou calibré et évalué (CPU).

Plan fixé le 13 novembre 2026, avant tout calcul. La chaîne peau de `predict.py` n'a
aucun contrôle qualité (celui du fond d'œil n'a pas de sens sur la peau).

Score : `skin_blur_score` (variance du Laplacien, petit côté ramené à 512 px, carré
central de 256 px pour éviter le bord noir du dermatoscope).

Calibration (même règle que le fond d'œil, `calibrate_quality.py`) : seuil = 2e
percentile du score sur les images d'apprentissage d'ISIC 2019.

Évaluation sur 500 images du test ISIC 2019 tirées au hasard (graine 0), jamais vues par
la calibration :
- fausses alertes : part des images d'origine signalées ;
- détection : part signalée après flou gaussien de sigma 2, 4 et 8 px (à l'échelle
  512 px, ramené à la taille d'origine).
Critère (fixé) : détection >= 0,80 à sigma 4 ET fausses alertes <= 5 %.

Secondaire, sans critère (pourquoi contrôler) : effet du flou sigma 4 sur la tête peau
(AUROC un contre tous sur ces 500 images, classes avec au moins 10 positifs, d'origine
contre floutées, DINOv2-B local).

Conséquence prévue : critère atteint → le contrôle de flou est activé pour la peau dans
`predict.py` (signale, et refuse avec `--reject-unusable`, comme le fond d'œil) ; sinon,
la peau reste sans contrôle qualité et c'est documenté.

Réserves : flou synthétique (pas de vraies photos ratées étiquetées) ; exposition non
contrôlée ; dermoscopie seulement.

Exemple :
    python scripts/calibrate_skin_quality.py --isic-root data/raw/isic2019 \\
        --checkpoint third_party/dinov2/dinov2_vitb14_timm.pth
"""

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.isic_dataset import CLASS_NAMES, ISIC2019Dataset, skin_tensor  # noqa: E402
from src.eval.provenance import run_info  # noqa: E402
from src.models.classifier_head import MultiLabelHead  # noqa: E402
from src.models.embedding_cache import file_fingerprint  # noqa: E402
from src.models.retfound_backbone import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from src.models.timm_backbones import DINOV2_IMAGE_SIZE, DINOv2Backbone  # noqa: E402
from src.pipeline.quality_control import SKIN_ANALYSIS_SIZE, skin_blur_score  # noqa: E402

SIGMAS = (2, 4, 8)
SAMPLE = 500
PERCENTILE = 2
DETECTION = 0.80
MAX_FALSE_ALERTS = 0.05
SKIN_HEAD = Path("heads/isic2019_dinov2b_head.pt")


def blurred(image: Image.Image, sigma: float) -> Image.Image:
    """Flou gaussien de `sigma` px à l'échelle d'analyse, ramené à la taille d'origine."""
    return image.filter(ImageFilter.GaussianBlur(sigma * min(image.size) / SKIN_ANALYSIS_SIZE))


def score_file(path: Path) -> float:
    return skin_blur_score(Image.open(path))


def scores_with_blur(path: Path) -> list[float]:
    image = Image.open(path).convert("RGB")
    return [skin_blur_score(image)] + [skin_blur_score(blurred(image, s)) for s in SIGMAS]


@torch.no_grad()
def head_probs(paths: list[Path], sigma: float | None, backbone, head) -> np.ndarray:
    out = []
    for b in range(0, len(paths), 32):
        images = [Image.open(p).convert("RGB") for p in paths[b : b + 32]]
        if sigma is not None:
            images = [blurred(im, sigma) for im in images]
        x = torch.stack([skin_tensor(im, DINOV2_IMAGE_SIZE) for im in images])
        out.append(torch.sigmoid(head(backbone((x - IMAGENET_MEAN) / IMAGENET_STD))).numpy())
    return np.concatenate(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--isic-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("results/skin_quality_calibration.json"))
    args = parser.parse_args()

    train = ISIC2019Dataset(args.isic_root, "train")
    test = ISIC2019Dataset(args.isic_root, "test")
    pick = np.sort(np.random.default_rng(0).choice(len(test), SAMPLE, replace=False))
    test_paths = [test.image_path(int(i)) for i in pick]
    with Pool(args.workers) as pool:
        train_scores = np.array(pool.map(score_file, [train.image_path(i) for i in range(len(train))], 64))
        test_scores = np.array(pool.map(scores_with_blur, test_paths, 16))
    threshold = float(np.percentile(train_scores, PERCENTILE))
    flagged = test_scores < threshold
    detection = {str(s): float(flagged[:, k + 1].mean()) for k, s in enumerate(SIGMAS)}
    false_alerts = float(flagged[:, 0].mean())

    artifact = torch.load(SKIN_HEAD, map_location="cpu", weights_only=True)
    # Poids local ou poids Kaggle (vérifiés équivalents, cosinus médian 1,0 ; les deux sont dans la tête)
    known = {artifact["checkpoint_sha256"], artifact.get("embeddings_checkpoint_sha256")}
    if file_fingerprint(args.checkpoint) not in known:
        raise ValueError("tête peau calculée avec un autre DINOv2-B")
    backbone = DINOv2Backbone(args.checkpoint).eval()
    head = MultiLabelHead(backbone.embedding_dim, len(CLASS_NAMES))
    head.load_state_dict(artifact["state_dict"])
    head.eval()
    y = test.df[CLASS_NAMES].to_numpy()[pick]
    p_orig, p_blur = head_probs(test_paths, None, backbone, head), head_probs(test_paths, 4, backbone, head)
    effect = {
        c: {
            "positives": int(y[:, i].sum()),
            "original": float(roc_auc_score(y[:, i], p_orig[:, i])),
            "blur_sigma4": float(roc_auc_score(y[:, i], p_blur[:, i])),
        }
        for i, c in enumerate(CLASS_NAMES)
        if y[:, i].sum() >= 10
    }

    report = {
        "run": run_info(),
        "plan": "fixé le 13 novembre 2026 avant tout calcul (docstring du script)",
        "train_images": len(train),
        "threshold": threshold,
        "threshold_rule": f"{PERCENTILE}e percentile du score sur l'apprentissage ISIC 2019",
        "test_sample": SAMPLE,
        "false_alerts": false_alerts,
        "detection_by_sigma": detection,
        "criterion": {
            "detection_sigma4": detection["4"],
            "false_alerts": false_alerts,
            "met": bool(detection["4"] >= DETECTION and false_alerts <= MAX_FALSE_ALERTS),
        },
        "head_auroc_effect": effect,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"seuil {threshold:.2f} ; fausses alertes {false_alerts:.3f} ; détection {detection}")
    print("critère :", report["criterion"])
    for c, e in effect.items():
        print(f"  {c:5s} ({e['positives']} positifs) AUROC {e['original']:.3f} -> {e['blur_sigma4']:.3f}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()

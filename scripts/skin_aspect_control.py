#!/usr/bin/env python3
"""Contrôle : l'écrasement en 256 × 256 explique-t-il la baisse sur ISIC 2020 ? (CPU)

Plan fixé le 13 novembre 2026, avant tout calcul. Seules les proportions des images ont
été relevées (test ISIC 2019 : 79 % carrées, 21 % en 4:3 ; ISIC 2020 d'origine : surtout
3:2 et 16:9).

Question : sur ISIC 2020, l'AUROC du mélanome tombe à 0,788 (0,874 au test ISIC 2019).
La version Kaggle utilisée écrase les images en 256 × 256 sans garder les proportions ;
la chaîne de la tête peau, elle, garde les proportions (petit côté 256, recadrage central
224). Une part de la baisse pourrait venir de cette déformation, pas d'un vrai défaut de
transfert.

Méthode (test ISIC 2019, 8 238 images, tête peau inchangée, DINOv2-B local) :
- chaque image est d'abord recadrée au centre en 3:2 paysage (proportion la plus
  fréquente d'ISIC 2020) ;
- bras « proportions gardées » : chaîne habituelle sur l'image recadrée ;
- bras « écrasé » : image recadrée redimensionnée en 256 × 256 (bilinéaire), puis chaîne
  habituelle, comme une image ISIC 2020 de la version Kaggle ;
- écart apparié d'AUROC (écrasé − gardé), bootstrap sur les lésions (1 000 tirages).

Lecture fixée : mélanome, borne haute de l'IC de l'écart < −0,02 → le format contribue à
la baisse, et ISIC 2020 doit être refait avec les images d'origine avant de conclure ;
sinon → la baisse sur ISIC 2020 ne s'explique pas par le format. Secondaire : AUROC macro
des 8 classes.

Réserves : déformation simulée (l'outil de redimensionnement de la version Kaggle n'est
pas documenté) ; le recadrage 3:2 coupe une partie des lésions carrées, de la même façon
dans les deux bras.

Exemple :
    python scripts/skin_aspect_control.py --isic-root data/raw/isic2019 \\
        --checkpoint third_party/dinov2/dinov2_vitb14_timm.pth
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.pad_ufes20_skin import auroc_ci  # noqa: E402
from src.data.isic_dataset import CLASS_NAMES, ISIC2019Dataset, skin_tensor  # noqa: E402
from src.eval.metrics import bootstrap_auroc_ci, paired_bootstrap_diff  # noqa: E402
from src.eval.provenance import run_info  # noqa: E402
from src.models.classifier_head import MultiLabelHead  # noqa: E402
from src.models.embedding_cache import file_fingerprint  # noqa: E402
from src.models.retfound_backbone import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from src.models.timm_backbones import DINOV2_IMAGE_SIZE, DINOv2Backbone  # noqa: E402

SKIN_HEAD = Path("heads/isic2019_dinov2b_head.pt")
RATIO = 1.5  # 3:2 paysage
SQUASH = 256
READING = -0.02


def crop_ratio(image: Image.Image, ratio: float = RATIO) -> Image.Image:
    """Recadrage central au rapport largeur/hauteur `ratio`."""
    w, h = image.size
    if w / h > ratio:
        new_w = round(h * ratio)
        left = (w - new_w) // 2
        return image.crop((left, 0, left + new_w, h))
    new_h = round(w / ratio)
    top = (h - new_h) // 2
    return image.crop((0, top, w, top + new_h))


def squash(image: Image.Image, size: int = SQUASH) -> Image.Image:
    return image.resize((size, size), Image.BILINEAR)


@torch.no_grad()
def probabilities(paths: list[Path], squashed: bool, backbone, head) -> np.ndarray:
    out, start = [], time.time()
    for b in range(0, len(paths), 32):
        images = [crop_ratio(Image.open(p).convert("RGB")) for p in paths[b : b + 32]]
        if squashed:
            images = [squash(im) for im in images]
        x = torch.stack([skin_tensor(im, DINOV2_IMAGE_SIZE) for im in images])
        out.append(torch.sigmoid(head(backbone((x - IMAGENET_MEAN) / IMAGENET_STD))).numpy())
        if (b // 32) % 40 == 0:
            print(
                f"  {'écrasé' if squashed else 'gardé'} {b + len(images)}/{len(paths)} en {time.time() - start:.0f} s"
            )
    return np.concatenate(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--isic-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/skin_aspect_control.json"))
    args = parser.parse_args()

    artifact = torch.load(SKIN_HEAD, map_location="cpu", weights_only=True)
    known = {artifact["checkpoint_sha256"], artifact.get("embeddings_checkpoint_sha256")}
    if file_fingerprint(args.checkpoint) not in known:
        raise ValueError("tête peau calculée avec un autre DINOv2-B")
    backbone = DINOv2Backbone(args.checkpoint).eval()
    head = MultiLabelHead(backbone.embedding_dim, len(CLASS_NAMES))
    head.load_state_dict(artifact["state_dict"])
    head.eval()

    test = ISIC2019Dataset(args.isic_root, "test")
    paths = [test.image_path(i) for i in range(len(test))]
    y = test.df[CLASS_NAMES].to_numpy(np.float32)
    groups = test.groups
    kept = probabilities(paths, False, backbone, head)
    squashed = probabilities(paths, True, backbone, head)

    mel = CLASS_NAMES.index("MEL")
    diff = paired_bootstrap_diff(y, kept, squashed, list(CLASS_NAMES), groups)
    report = {
        "run": run_info(),
        "plan": "fixé le 13 novembre 2026 avant tout calcul (docstring du script)",
        "images": len(paths),
        "melanomas": int(y[:, mel].sum()),
        "kept": {
            "melanoma": auroc_ci(y[:, mel], kept[:, mel], groups),
            **bootstrap_auroc_ci(y, kept, list(CLASS_NAMES), groups),
        },
        "squashed": {
            "melanoma": auroc_ci(y[:, mel], squashed[:, mel], groups),
            **bootstrap_auroc_ci(y, squashed, list(CLASS_NAMES), groups),
        },
        "squashed_minus_kept": diff,
    }
    high = diff["MEL"]["ci"][1]
    report["reading"] = {
        "melanoma_diff": diff["MEL"],
        "threshold": READING,
        "format_contributes": bool(high < READING),
        "text": "le format contribue : refaire ISIC 2020 avec les images d'origine"
        if high < READING
        else "la baisse sur ISIC 2020 ne s'explique pas par le format",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print("Mélanome gardé :", report["kept"]["melanoma"], "| écrasé :", report["squashed"]["melanoma"])
    print("Écart mélanome :", diff["MEL"], "| macro :", diff["macro"])
    print("Lecture :", report["reading"]["text"])
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()

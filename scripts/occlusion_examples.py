#!/usr/bin/env python3
"""Planche d'occlusion : contrôle des cartes Grad-CAM sur les mêmes images.

Mêmes classes et mêmes images que scripts/gradcam_examples.py (vrais
positifs de validation les plus sûrs), pour comparer case par case. Pour
chaque image : l'image, puis la carte des zones dont l'occlusion fait le
plus chuter le logit de la classe (rouge = zone utile à la prédiction).

Légende de chaque case :
    fond Δ    : variation du logit quand tout le hors-champ (marges noires
                ou blanches) est remplacé par du gris. Proche de 0 = le fond
                ne compte pas ; fortement négatif = raccourci sur le fond.
    hors champ: part de l'effet d'occlusion qui tombe hors de la rétine /
                part de l'image hors rétine. Nettement au-dessus = le modèle
                s'appuie sur le fond plus que sa surface ne l'explique.

Exemple :
    python scripts/occlusion_examples.py --dataset rfmid --data-root data/raw/rfmid \\
        --checkpoint third_party/retfound/checkpoints/RETFound_mae_natureCFP.pth \\
        --head heads/rfmid_head.pt --out results/occlusion_rfmid.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.gradcam_examples import caption, load_val_dataset, pick_default_classes, top_true_positives
from src.explainability.gradcam import RETFoundClassifier
from src.explainability.occlusion import background_test, fundus_mask, occlusion_map, outside_share
from src.models.classifier_head import MultiLabelHead
from src.models.retfound_backbone import load_pretrained_backbone


def percent(value: float) -> str:
    return "n/a" if value != value else f"{value:.0%}"


def mean_share(shares: list[float | None]) -> float:
    known = [s for s in shares if s is not None]
    return float(np.mean(known)) if known else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["rfmid", "odir"], required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint RETFound (le même que pour la tête)")
    parser.add_argument("--head", type=Path, required=True, help="Tête sauvegardée par run_baseline.py --save-head")
    parser.add_argument("--classes", nargs="*", default=None, help="Défaut : les mêmes que gradcam_examples.py")
    parser.add_argument("--num-classes", type=int, default=4, help="Nombre de classes si --classes est omis")
    parser.add_argument("--per-class", type=int, default=3, help="Images par classe")
    parser.add_argument("--patch", type=int, default=32, help="Côté du carré masqué (pixels)")
    parser.add_argument("--stride", type=int, default=16, help="Pas de glissement du carré (pixels)")
    parser.add_argument("--out", type=Path, default=Path("results/occlusion.png"))
    parser.add_argument("--json", type=Path, default=None, help="Mesures par image dans ce fichier JSON")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    head_ckpt = torch.load(args.head, map_location="cpu", weights_only=True)
    if head_ckpt["dataset"] != args.dataset:
        parser.error(f"La tête a été entraînée sur {head_ckpt['dataset']}, pas sur {args.dataset}")
    class_names, image_size = head_ckpt["class_names"], head_ckpt["image_size"]
    dataset, labels = load_val_dataset(args.dataset, args.data_root, head_ckpt)
    positives_per_class = {c: int(labels[:, i].sum()) for i, c in enumerate(class_names)}

    classes = args.classes or pick_default_classes(
        head_ckpt["auroc_per_class"], positives_per_class, args.num_classes, min_positives=args.per_class * 3
    )
    unknown = [c for c in classes if c not in class_names]
    if unknown:
        parser.error(f"Classes inconnues : {unknown} (disponibles : {class_names})")

    head = MultiLabelHead(embedding_dim=1024, num_classes=len(class_names))
    head.load_state_dict(head_ckpt["state_dict"])
    vit = load_pretrained_backbone(args.checkpoint, img_size=image_size, global_pool=True)
    model = RETFoundClassifier(vit, head).to(args.device).eval()

    rows, records = [], []
    for class_name in classes:
        class_idx = class_names.index(class_name)
        positive_idx = np.flatnonzero(labels[:, class_idx] == 1)
        if len(positive_idx) == 0:
            print(f"{class_name} : aucun positif en validation, ignorée")
            continue
        picks = top_true_positives(
            model, dataset, positive_idx, class_idx, args.per_class, args.device, args.batch_size
        )
        tiles = []
        for idx, prob in picks:
            image_t, _ = dataset[idx]
            heat, _ = occlusion_map(model, image_t, class_idx, args.patch, args.stride, args.batch_size)
            bg = background_test(model, image_t, class_idx)
            share = outside_share(heat, fundus_mask(image_t).numpy())
            delta = bg["logit_fond_gris"] - bg["logit"]
            # nan = aucune occlusion ne fait baisser le logit : rien à répartir (None en JSON)
            records.append(
                {
                    "classe": class_name,
                    "index": idx,
                    "p": prob,
                    **bg,
                    "part_effet_hors_champ": share if share == share else None,
                }
            )

            image = (image_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            positive = np.clip(heat, 0, None)
            scaled = positive / positive.max() if positive.max() > 0 else positive
            overlay = show_cam_on_image(image.astype(np.float32) / 255.0, scaled.astype(np.float32), use_rgb=True)
            area = bg["part_hors_champ"]
            text = f"{class_name} p={prob:.2f}  fond Δ={delta:+.1f}  hors champ {percent(share)}/{area:.0%}"
            tiles.append(caption(np.concatenate([image, overlay], axis=1), text))
        while len(tiles) < args.per_class:  # moins de positifs que demandé : case vide
            tiles.append(np.full_like(tiles[0], 255))
        rows.append(np.concatenate(tiles, axis=1))
        mine = [r for r in records if r["classe"] == class_name]
        print(
            f"{class_name} : logit {np.mean([r['logit'] for r in mine]):+.1f}, "
            f"fond gris Δ {np.mean([r['logit_fond_gris'] - r['logit'] for r in mine]):+.1f}, "
            f"effet hors champ {percent(mean_share([r['part_effet_hors_champ'] for r in mine]))} "
            f"pour {np.mean([r['part_hors_champ'] for r in mine]):.0%} de l'image"
        )

    if not rows:
        raise SystemExit("Aucune classe affichable.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(rows, axis=0)).save(args.out)
    print(f"Planche d'occlusion : {args.out}")
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(records, indent=2, ensure_ascii=False))
        print(f"Mesures JSON : {args.json}")


if __name__ == "__main__":
    main()

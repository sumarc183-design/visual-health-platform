#!/usr/bin/env python3
"""Planche Grad-CAM sur les vraies prédictions RETFound (validation RFMiD/ODIR).

Pour chaque classe retenue, prend les images de validation réellement
positives auxquelles la tête donne la plus forte probabilité (vrais positifs
les plus sûrs), et affiche côte à côte l'image et sa carte Grad-CAM pour
cette classe. Sert à vérifier que le modèle regarde des zones plausibles
(lésions, disque optique, macula...) et pas un artefact (bord, texte,
vignettage).

Nécessite la tête entraînée par run_baseline.py --save-head.

Exemple :
    python run_baseline.py --dataset rfmid --data-root data/raw/rfmid \\
        --checkpoint third_party/retfound/checkpoints/RETFound_mae_natureCFP.pth \\
        --epochs 30 --save-head heads/rfmid_head.pt
    python scripts/gradcam_examples.py --dataset rfmid --data-root data/raw/rfmid \\
        --checkpoint third_party/retfound/checkpoints/RETFound_mae_natureCFP.pth \\
        --head heads/rfmid_head.pt --out results/gradcam_rfmid.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.odir_dataset import ODIRDataset
from src.data.rfmid_dataset import RFMiDDataset
from src.explainability.gradcam import RETFoundClassifier, generate_gradcam
from src.models.classifier_head import MultiLabelHead
from src.models.retfound_backbone import load_pretrained_backbone

CAPTION_HEIGHT = 22


def pick_default_classes(auroc_per_class: dict, positives_per_class: dict, n_classes: int, min_positives: int) -> list[str]:
    """Classes les mieux classées en AUROC parmi celles qui ont assez de
    positifs en validation (une AUROC sur 2 positifs ne veut rien dire)."""
    eligible = [c for c, auroc in auroc_per_class.items() if positives_per_class.get(c, 0) >= min_positives]
    return sorted(eligible, key=lambda c: auroc_per_class[c], reverse=True)[:n_classes]


def top_true_positives(model: RETFoundClassifier, dataset, positive_idx: np.ndarray, class_idx: int,
                       k: int, device: str, batch_size: int) -> list[tuple[int, float]]:
    probs = []
    with torch.no_grad():
        for start in range(0, len(positive_idx), batch_size):
            batch = torch.stack([dataset[int(i)][0] for i in positive_idx[start:start + batch_size]]).to(device)
            probs.append(torch.sigmoid(model(batch))[:, class_idx].cpu().numpy())
    probs = np.concatenate(probs)
    order = np.argsort(-probs)[:k]
    return [(int(positive_idx[i]), float(probs[i])) for i in order]


def caption(tile: np.ndarray, text: str) -> np.ndarray:
    canvas = Image.new("RGB", (tile.shape[1], tile.shape[0] + CAPTION_HEIGHT), "white")
    canvas.paste(Image.fromarray(tile), (0, CAPTION_HEIGHT))
    try:
        font = ImageFont.load_default(size=14)
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    ImageDraw.Draw(canvas).text((4, 3), text, fill="black", font=font)
    return np.array(canvas)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["rfmid", "odir"], required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint RETFound (le même que pour la tête)")
    parser.add_argument("--head", type=Path, required=True, help="Tête sauvegardée par run_baseline.py --save-head")
    parser.add_argument("--classes", nargs="*", default=None,
                        help="Classes à montrer. Défaut : les mieux classées en AUROC (avec assez de positifs).")
    parser.add_argument("--num-classes", type=int, default=4, help="Nombre de classes si --classes est omis")
    parser.add_argument("--per-class", type=int, default=3, help="Images par classe")
    parser.add_argument("--out", type=Path, default=Path("results/gradcam.png"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    head_ckpt = torch.load(args.head, map_location="cpu", weights_only=True)
    if head_ckpt["dataset"] != args.dataset:
        parser.error(f"La tête a été entraînée sur {head_ckpt['dataset']}, pas sur {args.dataset}")
    class_names, image_size = head_ckpt["class_names"], head_ckpt["image_size"]

    dataset_cls = RFMiDDataset if args.dataset == "rfmid" else ODIRDataset
    dataset = dataset_cls(args.data_root, split="val", image_size=image_size)
    if list(dataset.class_names) != class_names:
        raise SystemExit("Les classes du dataset ne correspondent pas à celles de la tête sauvegardée.")
    labels = dataset.df[class_names].to_numpy(dtype=np.float32)
    positives_per_class = {c: int(labels[:, i].sum()) for i, c in enumerate(class_names)}

    classes = args.classes or pick_default_classes(
        head_ckpt["auroc_per_class"], positives_per_class, args.num_classes, min_positives=args.per_class * 3)
    unknown = [c for c in classes if c not in class_names]
    if unknown:
        parser.error(f"Classes inconnues : {unknown} (disponibles : {class_names})")

    head = MultiLabelHead(embedding_dim=1024, num_classes=len(class_names))
    head.load_state_dict(head_ckpt["state_dict"])
    vit = load_pretrained_backbone(args.checkpoint, img_size=image_size, global_pool=True)
    model = RETFoundClassifier(vit, head).to(args.device).eval()

    rows = []
    for class_name in classes:
        class_idx = class_names.index(class_name)
        positive_idx = np.flatnonzero(labels[:, class_idx] == 1)
        if len(positive_idx) == 0:
            print(f"{class_name} : aucun positif en validation, ignorée")
            continue
        picks = top_true_positives(model, dataset, positive_idx, class_idx, args.per_class, args.device, args.batch_size)
        tiles = []
        for idx, prob in picks:
            image_t, _ = dataset[idx]
            image = (image_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            overlay = generate_gradcam(model, model.gradcam_target_layer, image, image_t[None].to(args.device),
                                       target_class=class_idx, patch_grid_size=image_size // 16)
            auroc = head_ckpt["auroc_per_class"].get(class_name, float("nan"))
            tiles.append(caption(np.concatenate([image, overlay], axis=1),
                                 f"{class_name}  p={prob:.2f}  (AUROC classe {auroc:.2f})"))
        while len(tiles) < args.per_class:  # moins de positifs que demandé : case vide
            tiles.append(np.full_like(tiles[0], 255))
        rows.append(np.concatenate(tiles, axis=1))
        print(f"{class_name} : {len(picks)} image(s), p = {', '.join(f'{p:.2f}' for _, p in picks)}")

    if not rows:
        raise SystemExit("Aucune classe affichable.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(rows, axis=0)).save(args.out)
    print(f"Planche Grad-CAM : {args.out}")


if __name__ == "__main__":
    main()

"""Fait tourner tout le pipeline Visual Health de bout en bout sur des
données synthétiques (contrôle qualité -> RETFound -> classification
multi-label -> Grad-CAM), pour valider la plomberie avant l'arrivée des
vraies données. Ne produit aucun résultat médicalement significatif --
seul le "ça tourne sans erreur" compte ici.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.explainability.gradcam import generate_gradcam
from src.models.classifier_head import MultiLabelHead, train_head
from src.models.retfound_backbone import RETFoundBackbone
from src.pipeline.quality_control import check_quality


class SyntheticFundusDataset(Dataset):
    def __init__(self, data_dir: str, image_size: int = 224):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        with open(self.data_dir / "labels.csv") as f:
            reader = csv.reader(f)
            header = next(reader)
            self.class_names = header[1:]
            self.rows = list(reader)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        image = Image.open(self.data_dir / "images" / f"{row[0]}.png").convert("RGB")
        image = np.array(image.resize((self.image_size, self.image_size))).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        labels = torch.tensor([int(v) for v in row[1:]], dtype=torch.float32)
        return image, labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", default="third_party/retfound/checkpoints/RETFound_mae_natureCFP.pth")
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    dataset = SyntheticFundusDataset(args.data_dir, image_size=args.image_size)
    print(f"{len(dataset)} images, {len(dataset.class_names)} classes")

    n_rejected = 0
    for row in dataset.rows:
        image = np.array(Image.open(Path(args.data_dir) / "images" / f"{row[0]}.png"))
        if not check_quality(image, blur_threshold=5.0).is_usable:
            n_rejected += 1
    print(f"Contrôle qualité : {n_rejected}/{len(dataset)} rejetée(s)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Chargement du backbone RETFound...")
    backbone = RETFoundBackbone(args.checkpoint, img_size=args.image_size, embedding_dim=1024).to(device)

    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    print("Extraction des embeddings...")
    all_embeddings, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            features = backbone(images.to(device))
            all_embeddings.append(features.cpu())
            all_labels.append(labels)
    X = torch.cat(all_embeddings)
    y = torch.cat(all_labels)

    n_train = int(0.7 * len(X))
    train_loader = DataLoader(list(zip(X[:n_train], y[:n_train])), batch_size=8)
    test_X, test_y = X[n_train:], y[n_train:]

    print("Entraînement de la tête de classification...")
    head = MultiLabelHead(embedding_dim=1024, num_classes=len(dataset.class_names))
    head = train_head(head, train_loader, train_loader, epochs=3, lr=1e-3, device=device)

    with torch.no_grad():
        probs = torch.sigmoid(head(test_X.to(device))).cpu()
    print(f"Prédictions test : shape {tuple(probs.shape)}, plage [{probs.min():.3f}, {probs.max():.3f}]")

    print("Génération d'une carte Grad-CAM...")
    sample_image, _ = dataset[0]
    input_tensor = sample_image.unsqueeze(0).to(device).requires_grad_(True)

    class Wrapper(torch.nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone, self.head = backbone, head

        def forward(self, x):
            return self.head(self.backbone.backbone.forward_features(x))

    wrapper = Wrapper(backbone, head)
    target_layer = backbone.backbone.blocks[-1].norm1
    raw_image = (sample_image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    patch_grid = args.image_size // 16
    cam_image = generate_gradcam(wrapper, target_layer, raw_image, input_tensor, target_class=0, patch_grid_size=patch_grid)
    print(f"Carte Grad-CAM générée : shape {cam_image.shape}")

    print("\nOK -- le pipeline Visual Health tourne de bout en bout (données synthétiques).")


if __name__ == "__main__":
    main()

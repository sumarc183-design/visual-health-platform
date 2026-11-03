"""Fait tourner tout le pipeline Visual Health (génération synthétique ->
contrôle qualité -> RETFound -> classification multi-label -> Grad-CAM) de
bout en bout. Marqué "slow" : exclure avec `pytest -m "not slow"`.
"""

import csv
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.generate_synthetic_fundus_dataset import generate_dataset
from src.explainability.gradcam import generate_gradcam
from src.models.classifier_head import MultiLabelHead, train_head
from src.models.retfound_backbone import RETFoundBackbone, build_retfound_vit_large
from src.pipeline.quality_control import check_quality


@pytest.mark.slow
def test_full_pipeline_end_to_end(tmp_path):
    data_dir = tmp_path / "synthetic_rfmid"
    generate_dataset(str(data_dir), dataset="rfmid", num_images=16, image_size=224, seed=1)

    with open(data_dir / "labels.csv") as f:
        reader = csv.reader(f)
        class_names = next(reader)[1:]
        rows = list(reader)
    assert len(rows) == 16
    assert len(class_names) == 28

    from PIL import Image

    for row in rows:
        image = np.array(Image.open(data_dir / "images" / f"{row[0]}.png"))
        assert check_quality(image, blur_threshold=1.0).is_usable

    backbone_model = build_retfound_vit_large(img_size=224, global_pool=True)
    ckpt_path = tmp_path / "ckpt.pth"
    torch.save({"model": backbone_model.state_dict()}, ckpt_path)
    backbone = RETFoundBackbone(ckpt_path, img_size=224, embedding_dim=1024)

    embeddings, labels = [], []
    for row in rows:
        image = Image.open(data_dir / "images" / f"{row[0]}.png").convert("RGB")
        image_t = torch.from_numpy(np.array(image).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            embeddings.append(backbone(image_t))
        labels.append(torch.tensor([int(v) for v in row[1:]], dtype=torch.float32))
    X, y = torch.cat(embeddings), torch.stack(labels)

    loader = torch.utils.data.DataLoader(list(zip(X, y, strict=True)), batch_size=8)
    head = MultiLabelHead(embedding_dim=1024, num_classes=len(class_names))
    head = train_head(head, loader, epochs=2, lr=1e-3)

    with torch.no_grad():
        probs = torch.sigmoid(head(X))
    assert probs.shape == (16, 28)
    assert (probs >= 0).all() and (probs <= 1).all()

    class Wrapper(torch.nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone, self.head = backbone, head

        def forward(self, x):
            return self.head(self.backbone.backbone.forward_features(x))

    wrapper = Wrapper(backbone, head)
    target_layer = backbone.backbone.blocks[-1].norm1
    sample_image = Image.open(data_dir / "images" / f"{rows[0][0]}.png").convert("RGB")
    raw_image = np.array(sample_image)
    input_tensor = (
        torch.from_numpy(raw_image.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).requires_grad_(True)
    )

    cam_image = generate_gradcam(wrapper, target_layer, raw_image, input_tensor, target_class=0, patch_grid_size=14)
    assert cam_image.shape == (224, 224, 3)

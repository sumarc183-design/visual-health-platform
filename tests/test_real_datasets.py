"""Fait tourner le pipeline (contrôle qualité -> RETFound -> classification ->
Grad-CAM) sur de VRAIES images RFMiD/ODIR, en remplacement du smoke test
synthétique de test_pipeline_smoke.py. Backbone toujours instancié avec des
poids aléatoires (le vrai checkpoint RETFound n'est pas versionné, cf.
THIRD_PARTY_NOTICES.md) : ceci valide la plomberie sur données réelles, pas
la qualité des prédictions.

Sauté si data/raw/{rfmid,odir}/ est absent (jeux de données non versionnés,
~10 Go -- cf. scripts/download_rfmid_odir.py) : la CI n'a pas ces données et
continue de tourner sur le smoke test synthétique.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.odir_dataset import ODIRDataset
from src.data.rfmid_dataset import RFMiDDataset
from src.explainability.gradcam import generate_gradcam
from src.models.classifier_head import MultiLabelHead, train_head
from src.models.retfound_backbone import RETFoundBackbone, build_retfound_vit_large
from src.pipeline.quality_control import check_quality

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw"
RFMID_ROOT = DATA_ROOT / "rfmid"
ODIR_ROOT = DATA_ROOT / "odir"

requires_real_data = pytest.mark.skipif(
    not (RFMID_ROOT.exists() and ODIR_ROOT.exists()),
    reason="data/raw/{rfmid,odir}/ absent -- voir scripts/download_rfmid_odir.py",
)


@pytest.mark.slow
@requires_real_data
@pytest.mark.parametrize(
    "dataset_cls, root, expected_classes",
    [(RFMiDDataset, RFMID_ROOT, 28), (ODIRDataset, ODIR_ROOT, 8)],
)
def test_real_dataset_end_to_end(dataset_cls, root, expected_classes):
    dataset = dataset_cls(root, split="train", image_size=224)
    assert len(dataset) > 0
    assert len(dataset.class_names) == expected_classes

    n_samples = min(16, len(dataset))
    images, labels = zip(*(dataset[i] for i in range(n_samples)))

    for image_t in images:
        raw_image = (image_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        assert check_quality(raw_image, blur_threshold=1.0).is_usable

    backbone_model = build_retfound_vit_large(img_size=224, global_pool=True)
    ckpt_path = Path(dataset.root) / "_test_ckpt.pth"
    torch.save({"model": backbone_model.state_dict()}, ckpt_path)
    try:
        backbone = RETFoundBackbone(ckpt_path, img_size=224, embedding_dim=1024)

        embeddings = [backbone(image_t.unsqueeze(0)).detach() for image_t in images]
    finally:
        ckpt_path.unlink(missing_ok=True)

    X = torch.cat(embeddings)
    y = torch.stack(labels)

    loader = torch.utils.data.DataLoader(list(zip(X, y)), batch_size=8)
    head = MultiLabelHead(embedding_dim=1024, num_classes=expected_classes)
    head = train_head(head, loader, loader, epochs=2, lr=1e-3)

    with torch.no_grad():
        probs = torch.sigmoid(head(X))
    assert probs.shape == (n_samples, expected_classes)
    assert (probs >= 0).all() and (probs <= 1).all()

    class Wrapper(torch.nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone, self.head = backbone, head

        def forward(self, x):
            return self.head(self.backbone.backbone.forward_features(x))

    wrapper = Wrapper(backbone, head)
    target_layer = backbone.backbone.blocks[-1].norm1
    sample_image = images[0]
    raw_image = (sample_image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    input_tensor = sample_image.unsqueeze(0).clone().requires_grad_(True)

    cam_image = generate_gradcam(wrapper, target_layer, raw_image, input_tensor, target_class=0, patch_grid_size=14)
    assert cam_image.shape == (224, 224, 3)

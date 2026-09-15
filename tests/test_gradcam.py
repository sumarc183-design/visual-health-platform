"""Vérifie Grad-CAM adapté aux ViT (reshape_transform, API ClassifierOutputTarget)."""
import numpy as np
import torch

from src.explainability.gradcam import generate_gradcam
from src.models.classifier_head import MultiLabelHead
from src.models.retfound_backbone import RETFoundBackbone, build_retfound_vit_large


def test_gradcam_on_vit_produces_valid_heatmap(tmp_path):
    model = build_retfound_vit_large(img_size=224, global_pool=True)
    ckpt_path = tmp_path / "ckpt.pth"
    torch.save({"model": model.state_dict()}, ckpt_path)
    backbone = RETFoundBackbone(ckpt_path, img_size=224, embedding_dim=1024)
    head = MultiLabelHead(embedding_dim=1024, num_classes=28)

    class Wrapper(torch.nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone, self.head = backbone, head

        def forward(self, x):
            return self.head(self.backbone.backbone.forward_features(x))

    wrapper = Wrapper(backbone, head)
    target_layer = backbone.backbone.blocks[-1].norm1

    image = (np.random.default_rng(0).random((224, 224, 3)) * 255).astype(np.uint8)
    input_tensor = torch.randn(1, 3, 224, 224, requires_grad=True)

    cam_image = generate_gradcam(wrapper, target_layer, image, input_tensor, target_class=0, patch_grid_size=14)
    assert cam_image.shape == (224, 224, 3)
    assert cam_image.dtype == np.uint8

"""Identifie le type d'image médicale (fond d'œil, dermato, etc.) avant routage
vers le prétraitement et la tête de classification adaptés.

Phase 1 : un classifieur léger fine-tuné sur un petit set étiqueté manuellement
par modalité. Placeholder tant que ce set n'existe pas.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tv_models

MODALITIES = ["fundus", "dermoscopy", "unknown"]


class ModalityClassifier(nn.Module):
    def __init__(self, num_modalities: int = len(MODALITIES)):
        super().__init__()
        backbone = tv_models.mobilenet_v3_small(weights=tv_models.MobileNet_V3_Small_Weights.DEFAULT)
        backbone.classifier[-1] = nn.Linear(backbone.classifier[-1].in_features, num_modalities)
        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def predict_modality(self, x: torch.Tensor) -> list[str]:
        logits = self.forward(x)
        indices = logits.argmax(dim=-1).tolist()
        return [MODALITIES[i] for i in indices]

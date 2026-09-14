"""Wrapper pour charger le backbone RETFound préentraîné (ViT-Large, MAE) et
en extraire des représentations, sans réentraîner le backbone en phase 1.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class RETFoundBackbone(nn.Module):
    def __init__(self, checkpoint_path: str | Path, freeze: bool = True):
        super().__init__()
        self.backbone = self._load(checkpoint_path)
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def _load(self, checkpoint_path: str | Path) -> nn.Module:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Poids RETFound introuvables : {checkpoint_path}. "
                "Suivre third_party/retfound/README.md pour les récupérer."
            )
        # NOTE : RETFound est basé sur un ViT-Large (architecture MAE). Le
        # chargement exact du state_dict dépend de la version du dépôt cloné
        # dans third_party/retfound — importer sa définition de modèle plutôt
        # que d'en réécrire une ici, pour rester compatible avec les poids.
        raise NotImplementedError(
            "Brancher ici la construction du ViT-Large et le chargement du "
            "state_dict via third_party/retfound (voir leur script d'inférence)."
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_features(x)

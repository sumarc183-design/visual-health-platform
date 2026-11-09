"""Identifie le type d'image (fond d'œil, dermoscopie, autre) avant le routage vers
le prétraitement et la tête adaptés.

Embeddings DINOv2 ViT-B/14 figés (Apache 2.0) et tête linéaire à trois classes
(`scripts/train_modality_router.py`). Une image dont la probabilité maximale reste
sous le seuil est classée « unknown » : elle n'est envoyée à aucun modèle
spécialisé. Le seuil est fixé à l'avance, pas réglé sur les données.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

MODALITIES = ["fundus", "dermoscopy", "unknown"]
CONFIDENCE_THRESHOLD = 0.9


class ModalityHead(nn.Module):
    """Tête linéaire (standardisation déjà intégrée aux poids) et seuil de confiance."""

    def __init__(self, embedding_dim: int, threshold: float = CONFIDENCE_THRESHOLD):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, len(MODALITIES))
        self.threshold = threshold

    @torch.no_grad()
    def probabilities(self, X: torch.Tensor | np.ndarray) -> np.ndarray:
        X = torch.as_tensor(np.asarray(X, dtype=np.float32))
        return torch.softmax(self.linear(X), dim=-1).numpy()

    def decide(self, probs: np.ndarray) -> list[str]:
        """Classe la plus probable, ou « unknown » si sa probabilité est sous le seuil."""
        best = probs.argmax(axis=1)
        return [MODALITIES[i] if p[i] >= self.threshold else "unknown" for i, p in zip(best, probs, strict=True)]

    def predict(self, X) -> list[str]:
        return self.decide(self.probabilities(X))

    def save(self, path: str | Path, metadata: dict) -> None:
        state = {k: v.clone() for k, v in self.linear.state_dict().items()}
        torch.save({"state_dict": state, "threshold": self.threshold, "classes": MODALITIES, **metadata}, path)

    @classmethod
    def load(cls, path: str | Path) -> tuple[ModalityHead, dict]:
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        if artifact["classes"] != MODALITIES:
            raise ValueError(f"classes inattendues : {artifact['classes']}")
        weight = artifact["state_dict"]["weight"]
        head = cls(weight.shape[1], artifact["threshold"])
        head.linear.load_state_dict(artifact["state_dict"])
        return head, artifact


def linear_from_sklearn(scaler, model) -> tuple[np.ndarray, np.ndarray]:
    """Replie StandardScaler + LogisticRegression multinomiale en une couche linéaire."""
    weight = model.coef_ / scaler.scale_
    bias = model.intercept_ - (model.coef_ * scaler.mean_ / scaler.scale_).sum(axis=1)
    return weight.astype(np.float32), bias.astype(np.float32)

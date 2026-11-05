"""Tête de classification multi-label branchée sur les features RETFound gelées."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class MultiLabelHead(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)  # logits ; appliquer sigmoid pour les probabilités


def train_head(
    head: MultiLabelHead,
    train_loader: DataLoader,
    epochs: int,
    lr: float,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> MultiLabelHead:
    head.to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        head.train()
        total, count = 0.0, 0
        for embeddings, labels in train_loader:
            embeddings, labels = embeddings.to(device), labels.float().to(device)
            optimizer.zero_grad()
            loss = criterion(head(embeddings), labels)
            loss.backward()
            optimizer.step()
            total, count = total + loss.item() * len(labels), count + len(labels)
        # Moyenne sur l'epoch : la loss du dernier batch seul oscille au hasard du tirage.
        print(f"epoch {epoch + 1}/{epochs} — train loss: {total / count:.4f}")

    return head


PROBE_TOLERANCE = 1e-3


def head_probe(state_dict: dict) -> list[float]:
    """Signature d'une tête : ses logits pour un vecteur de référence fixe
    (graine 0). Deux entraînements identiques sur CPU et GPU diffèrent de
    ~1e-7 : une empreinte au bit près les déclarerait différents, la sonde
    comparée avec une tolérance (`probes_match`) non. Sert à lier un fichier
    de seuils à la tête qui a produit ses prédictions."""
    weight = state_dict["fc.weight"].double().cpu().numpy()
    bias = state_dict["fc.bias"].double().cpu().numpy()
    reference = np.random.default_rng(0).standard_normal(weight.shape[1])
    return (weight @ reference + bias).tolist()


def probes_match(a: list[float], b: list[float], tolerance: float = PROBE_TOLERANCE) -> bool:
    return len(a) == len(b) and bool(np.allclose(a, b, rtol=tolerance, atol=tolerance))

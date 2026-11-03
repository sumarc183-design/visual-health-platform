"""Tête de classification multi-label branchée sur les features RETFound gelées."""

from __future__ import annotations

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
        for embeddings, labels in train_loader:
            embeddings, labels = embeddings.to(device), labels.float().to(device)
            optimizer.zero_grad()
            loss = criterion(head(embeddings), labels)
            loss.backward()
            optimizer.step()
        print(f"epoch {epoch + 1}/{epochs} — train loss: {loss.item():.4f}")

    return head

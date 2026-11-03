"""Cache des embeddings RETFound, un fichier .npz par configuration.

Le backbone est figé : l'embedding d'une image ne dépend que de l'image, de
son prétraitement, de la taille et du checkpoint. Le recalculer à chaque run
coûtait ~10 min de GPU pour n'entraîner qu'une tête linéaire. Avec le cache
(sur Google Drive dans Colab), les expériences sur la tête tournent en
quelques secondes, sur CPU, et le backbone n'est même pas chargé.

Indexé par image (`dataset.image_keys`), pas par split : changer de split ou
d'étiquettes réutilise les embeddings déjà calculés.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


class EmbeddingCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.embeddings: dict[str, np.ndarray] = {}
        if self.path.exists():
            with np.load(self.path) as data:
                self.embeddings = dict(zip(data["keys"].tolist(), data["X"], strict=True))

    def missing(self, keys: list[str]) -> list[str]:
        return [k for k in keys if k not in self.embeddings]

    def add(self, keys: list[str], X: np.ndarray) -> None:
        self.embeddings.update(zip(keys, X.astype(np.float32), strict=True))

    def get(self, keys: list[str]) -> np.ndarray:
        return np.stack([self.embeddings[k] for k in keys])

    def save(self) -> None:
        """Écriture dans un fichier temporaire puis renommage : un run coupé
        (déconnexion Colab) ne laisse pas un cache à moitié écrit."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp.npz")
        keys = list(self.embeddings)
        np.savez(tmp, keys=np.array(keys), X=np.stack([self.embeddings[k] for k in keys]))
        os.replace(tmp, self.path)


def cache_file_name(dataset: str, preprocess: str, image_size: int, checkpoint: str | Path) -> str:
    return f"{dataset}_{preprocess}_{image_size}_{Path(checkpoint).stem}.npz"

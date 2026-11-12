"""Contrôle du type d'image par le routeur, en mode avertissement (jamais bloquant).

L'utilisateur déclare le type d'image (ici : fond d'œil). Le routeur de modalité
(DINOv2-B figé + tête linéaire, `heads/modality_router_v2.pt`) donne son avis ; s'il
n'est pas d'accord, la prédiction est faite quand même, avec un avertissement.

Pourquoi avertir plutôt que bloquer (proposition au tuteur, décision attendue) :
- l'erreur restante est rare mais concentrée sur une affection (hyalose astéroïde :
  2 images sur 25 envoyées en dermoscopie, `docs/MODALITE.md`) ; bloquer écarterait ces
  patients ;
- le routeur ne distingue pas une photo de peau ordinaire d'une dermoscopie (SLICE-3D,
  12 novembre) : il ne peut pas remplacer le type déclaré.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from src.data.isic_dataset import skin_tensor
from src.models.embedding_cache import file_fingerprint
from src.models.retfound_backbone import IMAGENET_MEAN, IMAGENET_STD
from src.models.timm_backbones import DINOV2_IMAGE_SIZE, DINOv2Backbone
from src.pipeline.modality_classifier import MODALITIES, ModalityHead

LABELS = {"fundus": "fond d'œil", "dermoscopy": "dermoscopie", "unknown": "type non reconnu"}


@dataclass
class ModalityOpinion:
    declared: str
    decision: str
    probabilities: dict[str, float]

    @property
    def agrees(self) -> bool:
        return self.decision == self.declared

    @property
    def warning(self) -> str | None:
        if self.agrees:
            return None
        return (
            f"Type d'image : déclaré {LABELS[self.declared]}, le routeur voit {LABELS[self.decision]} "
            f"(probabilité {self.probabilities[self.declared]:.2f} pour {LABELS[self.declared]}). "
            "Vérifier l'image ; la prédiction est donnée quand même."
        )

    def to_dict(self) -> dict:
        return {
            "declared": self.declared,
            "decision": self.decision,
            "agrees": self.agrees,
            "probabilities": self.probabilities,
        }


def opinion(head: ModalityHead, embedding, declared: str = "fundus") -> ModalityOpinion:
    """Avis du routeur sur un embedding DINOv2 (une image)."""
    if declared not in MODALITIES[:-1]:
        raise ValueError(f"type déclaré inconnu : {declared}")
    probs = head.probabilities(embedding)
    return ModalityOpinion(
        declared, head.decide(probs)[0], {m: float(p) for m, p in zip(MODALITIES, probs[0], strict=True)}
    )


class ModalityCheck:
    """Charge le routeur et vérifie que le checkpoint DINOv2 est celui de la tête."""

    def __init__(self, head_path: str | Path, checkpoint: str | Path):
        self.head, meta = ModalityHead.load(head_path)
        expected = meta.get("checkpoint_sha256")
        actual = file_fingerprint(checkpoint)
        if expected is not None and actual != expected:
            raise ValueError(f"checkpoint DINOv2 différent de celui du routeur ({actual} au lieu de {expected})")
        self.backbone = DINOv2Backbone(checkpoint)

    @torch.no_grad()
    def __call__(self, image_path: str | Path, declared: str = "fundus") -> ModalityOpinion:
        x = skin_tensor(Image.open(image_path), DINOV2_IMAGE_SIZE).unsqueeze(0)
        embedding = self.backbone((x - IMAGENET_MEAN) / IMAGENET_STD).numpy()
        return opinion(self.head, embedding, declared)

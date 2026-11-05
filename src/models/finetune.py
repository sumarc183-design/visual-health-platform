"""Fine-tuning partiel de RETFound : les `k` derniers blocs du ViT et la tête
sont entraînés, les autres restent figés.

Pourquoi partiel : avec quelques milliers d'images, entraîner tout le ViT-L
(~300 M de paramètres) surapprend vite et ne tient pas en mémoire sur un GPU
T4 ; les derniers blocs portent les représentations les plus spécifiques à
la tâche. Les blocs figés tournent sans gradient (mémoire économisée).

Augmentations (`augment_batch`) : retournement horizontal, petite rotation,
variations de luminosité, contraste et saturation. Les variations de couleur
visent la signature d'appareil, lisible dans les embeddings par la couleur
et la netteté. Le masque commun de `crop-mask` est réappliqué après la
rotation : sinon les coins tournés réintroduiraient une forme d'image
variable, le raccourci que le prétraitement supprime.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torchvision.transforms.v2.functional as TF

from .retfound_backbone import IMAGENET_MEAN, IMAGENET_STD, RETFoundViT


class PartialFinetune(nn.Module):
    """Normalisation ImageNet + ViT RETFound (k derniers blocs entraînables) + tête linéaire.
    Entrée : images dans [0, 1] ; sortie : logits."""

    def __init__(self, vit: RETFoundViT, num_classes: int, trainable_blocks: int = 4):
        super().__init__()
        if not 0 <= trainable_blocks <= len(vit.blocks):
            raise ValueError(f"trainable_blocks doit être entre 0 et {len(vit.blocks)}")
        self.vit = vit
        self.trainable_blocks = trainable_blocks
        self.head = nn.Linear(vit.embed_dim, num_classes)
        self.register_buffer("mean", IMAGENET_MEAN.clone())
        self.register_buffer("std", IMAGENET_STD.clone())
        for p in self.vit.parameters():
            p.requires_grad = False
        trainable = list(self.vit.blocks[len(self.vit.blocks) - trainable_blocks :])
        trainable.append(self.vit.fc_norm if self.vit.global_pool else self.vit.norm)
        for module in trainable:
            for p in module.parameters():
                p.requires_grad = True

    def backbone_parameters(self):
        return [p for p in self.vit.parameters() if p.requires_grad]

    def features(self, images: torch.Tensor) -> torch.Tensor:
        vit = self.vit
        x = (images - self.mean) / self.std
        split = len(vit.blocks) - self.trainable_blocks
        with torch.no_grad():  # partie figée : pas de graphe de gradient
            x = vit.patch_embed(x)
            x = torch.cat((vit.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = vit.pos_drop(x + vit.pos_embed)
            for block in vit.blocks[:split]:
                x = block(x)
        for block in vit.blocks[split:]:
            x = block(x)
        if vit.global_pool:
            return vit.fc_norm(x[:, 1:, :].mean(dim=1))
        return vit.norm(x)[:, 0]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(images))

    def train(self, mode: bool = True):
        super().train(mode)
        # Partie figée toujours en mode évaluation (dropout coupé, comme pour les embeddings).
        split = len(self.vit.blocks) - self.trainable_blocks
        for module in (self.vit.patch_embed, self.vit.pos_drop, *self.vit.blocks[:split]):
            module.eval()
        return self


def augment_batch(
    images: torch.Tensor,
    mask: torch.Tensor | None,
    generator: torch.Generator,
    max_rotation: float = 15.0,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.15,
) -> torch.Tensor:
    """Augmente un batch (B, 3, H, W) dans [0, 1], image par image (tirages
    reproductibles via `generator`). `mask` (H, W) booléen : zone conservée
    par crop-mask, remise à zéro autour après la rotation."""

    def uniform(low: float, high: float) -> float:
        return low + (high - low) * torch.rand(1, generator=generator).item()

    out = []
    for image in images:
        if torch.rand(1, generator=generator).item() < 0.5:
            image = TF.horizontal_flip(image)
        image = TF.rotate(image, uniform(-max_rotation, max_rotation))
        image = TF.adjust_brightness(image, uniform(1 - brightness, 1 + brightness))
        image = TF.adjust_contrast(image, uniform(1 - contrast, 1 + contrast))
        image = TF.adjust_saturation(image, uniform(1 - saturation, 1 + saturation))
        out.append(image.clamp(0, 1))
    batch = torch.stack(out)
    if mask is not None:
        batch = batch * mask.to(batch.device, batch.dtype)
    return batch


def cosine_with_warmup(step: int, total_steps: int, warmup_steps: int) -> float:
    """Facteur multiplicatif du taux d'apprentissage."""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def load_finetuned_state(model: PartialFinetune, artifact: dict) -> PartialFinetune:
    """Applique le delta de poids sauvegardé par scripts/finetune_retfound.py
    (blocs dégelés, normalisation finale, tête) à un modèle partiel construit
    sur le même backbone de base. Refuse un artefact incompatible : autre
    nombre de blocs, autres classes, paramètres manquants ou en trop."""
    if artifact["trainable_blocks"] != model.trainable_blocks:
        raise ValueError(
            f"artefact pour {artifact['trainable_blocks']} blocs dégelés, modèle à {model.trainable_blocks}"
        )
    if len(artifact["class_names"]) != model.head.out_features:
        raise ValueError("nombre de classes différent entre l'artefact et le modèle")
    expected = {name for name, p in model.named_parameters() if p.requires_grad}
    stored = set(artifact["state_dict"])
    if stored != expected:
        raise ValueError(
            f"paramètres incompatibles : manquants {sorted(expected - stored)[:3]}, "
            f"en trop {sorted(stored - expected)[:3]}"
        )
    model.load_state_dict(artifact["state_dict"], strict=False)
    return model

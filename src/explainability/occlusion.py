"""Sensibilité à l'occlusion : ce que le modèle perd quand on cache une zone.

Contrôle de Grad-CAM, qui sur RETFound (ViT MAE) s'allume surtout sur le
fond noir et les bords : effet des tokens « artefacts » de forte norme des
ViT, ou vrai raccourci (forme du masque, caméra) ? L'occlusion ne passe pas
par les gradients ni par les activations internes : elle masque un carré de
l'image et mesure directement la chute du logit de la classe. Si cacher la
lésion fait chuter le logit, le modèle s'en sert ; si c'est cacher le fond,
il prend un raccourci.

Logit plutôt que probabilité : à p = 1,00 la sigmoïde est saturée et une
occlusion, même décisive, ne la ferait presque pas bouger.

Le carré est rempli avec la moyenne ImageNet (gris, 0 après normalisation),
pas en noir : du noir sur le fond noir ne changerait rien, et un raccourci
fondé sur la forme du masque passerait inaperçu.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.models.retfound_backbone import IMAGENET_MEAN

FILL = IMAGENET_MEAN.view(3, 1, 1)


def fundus_mask(image_t: torch.Tensor, dark: float = 0.06, bright: float = 0.94) -> torch.Tensor:
    """(3, H, W) dans [0, 1] -> masque booléen (H, W) du champ rétinien.

    Hors champ = quasi noir (marges) ou quasi blanc (bandes de remplissage de
    certaines images ODIR), sur tous les canaux."""
    is_dark = image_t.max(dim=0).values < dark
    is_bright = image_t.min(dim=0).values > bright
    return ~(is_dark | is_bright)


@torch.no_grad()
def class_logit(model: nn.Module, images: torch.Tensor, class_idx: int, batch_size: int = 32) -> torch.Tensor:
    return torch.cat([model(images[i : i + batch_size])[:, class_idx] for i in range(0, len(images), batch_size)])


@torch.no_grad()
def occlusion_map(
    model: nn.Module,
    image_t: torch.Tensor,
    class_idx: int,
    patch: int = 32,
    stride: int = 16,
    batch_size: int = 32,
) -> tuple[np.ndarray, float]:
    """Carte (H, W) de la chute moyenne du logit quand chaque pixel est caché,
    et le logit d'origine. Positif = zone utile à la prédiction."""
    device = next(model.parameters()).device
    _, H, W = image_t.shape
    image_t = image_t.to(device)
    fill = FILL.to(device)
    positions = [(y, x) for y in range(0, H - patch + 1, stride) for x in range(0, W - patch + 1, stride)]

    occluded = image_t.repeat(len(positions), 1, 1, 1)
    for n, (y, x) in enumerate(positions):
        occluded[n, :, y : y + patch, x : x + patch] = fill
    base = class_logit(model, image_t[None], class_idx).item()
    drops = base - class_logit(model, occluded, class_idx, batch_size).cpu().numpy()

    heat, count = np.zeros((H, W)), np.zeros((H, W))
    for drop, (y, x) in zip(drops, positions, strict=True):
        heat[y : y + patch, x : x + patch] += drop
        count[y : y + patch, x : x + patch] += 1
    return heat / np.maximum(count, 1), base


@torch.no_grad()
def background_test(model: nn.Module, image_t: torch.Tensor, class_idx: int) -> dict:
    """Logit d'origine, logit avec tout le hors-champ remplacé par du gris,
    et part de l'image hors champ. Une forte chute = raccourci sur le fond."""
    device = next(model.parameters()).device
    image_t = image_t.to(device)
    outside = ~fundus_mask(image_t)
    grayed = torch.where(outside, FILL.to(device), image_t)
    base, gray = class_logit(model, torch.stack([image_t, grayed]), class_idx).tolist()
    return {"logit": base, "logit_fond_gris": gray, "part_hors_champ": float(outside.float().mean())}


def outside_share(heat: np.ndarray, mask: np.ndarray) -> float:
    """Part de l'effet positif de l'occlusion qui tombe hors du champ rétinien."""
    positive = np.clip(heat, 0, None)
    total = positive.sum()
    return float(positive[~mask].sum() / total) if total > 0 else float("nan")

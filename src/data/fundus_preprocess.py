"""Chargement des images de fond d'œil, avec ou sans prétraitement commun.

Pourquoi un prétraitement : ODIR mélange 87 formats d'image (appareils
différents), et certains sont fortement liés à une maladie (60 % des images
3456x3456 sont des glaucomes, contre 4,9 % en moyenne). Redimensionner
directement en 224x224 (`none`, comportement d'origine) déforme la rétine en
un ovale dont l'aplatissement et les marges dépendent de l'appareil : le
modèle peut reconnaître l'appareil au lieu de la maladie. Le test
d'occlusion l'a montré sur la DMLA d'ODIR (89 à 97 % de l'effet hors de la
rétine).

`crop-mask` rend toutes les images géométriquement identiques hors rétine :
recadrage sur le champ rétinien, carré par ajout de noir (sans déformer),
puis masque commun (disque coupé en haut et en bas) qui efface les marges,
l'encoche, les reflets de bord et les troncatures propres à chaque appareil.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

PREPROCESS_CHOICES = ("none", "crop-mask")
# À incrémenter à chaque modification qui change les pixels produits par
# load_fundus_image : l'empreinte du cache d'embeddings l'inclut, et les
# embeddings calculés avec l'ancienne version sont alors refusés.
PREPROCESS_VERSION = 1

DARK = 20  # /255 : en dessous sur tous les canaux = marge noire
BRIGHT = 235  # /255 : au-dessus sur tous les canaux = bande blanche de remplissage
MIN_FILL = 0.02  # ligne/colonne retenue si au moins 2 % de ses pixels sont de la rétine
MASK_RADIUS = 0.97  # rayon du masque / demi-côté : rogne le liseré de bord et l'encoche
# Demi-hauteur du masque / demi-côté. 19 % des images ODIR ont la rétine
# coupée en haut et en bas par l'appareil (bords plats) ; 90 % ont un rapport
# hauteur/largeur du champ >= 0,89. Couper toutes les images à cette hauteur
# donne la même forme à 90 % d'entre elles, pour ~5 % de rétine perdue en
# haut et en bas des images complètes.
MASK_BAND = 0.89


def _longest_run(flags: np.ndarray) -> tuple[int, int] | None:
    """Début et fin (exclue) du plus long bloc continu de True : la rétine,
    pas un liseré isolé (frontière blanc/noir, texte, encoche)."""
    best, start = None, None
    for i, flag in enumerate(np.append(flags, False)):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if best is None or i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    return best


def fundus_bbox(image: Image.Image, probe_size: int = 512) -> tuple[int, int, int, int] | None:
    """Boîte (gauche, haut, droite, bas) du champ rétinien en pixels de
    l'image d'origine, ou None si rien n'est détecté. Calculée sur une
    vignette, réduite sans interpolation : un mélange blanc/noir donnerait
    du gris pris pour de la rétine."""
    scale = probe_size / max(image.size)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    probe = np.asarray(image.resize(size, Image.NEAREST))
    retina = (probe.max(axis=2) > DARK) & (probe.min(axis=2) < BRIGHT)
    rows = _longest_run(retina.mean(axis=1) > MIN_FILL)
    cols = _longest_run(retina.mean(axis=0) > MIN_FILL)
    if rows is None or cols is None:
        return None
    return (
        int(cols[0] / scale),
        int(rows[0] / scale),
        min(image.width, int(cols[1] / scale)),
        min(image.height, int(rows[1] / scale)),
    )


def common_mask(image_size: int) -> np.ndarray:
    """Masque booléen (H, W) de la rétine conservée par crop-mask, identique
    pour toutes les images."""
    yy, xx = np.mgrid[:image_size, :image_size]
    center = (image_size - 1) / 2
    inside = (yy - center) ** 2 + (xx - center) ** 2 <= (MASK_RADIUS * image_size / 2) ** 2
    return inside & (np.abs(yy - center) <= MASK_BAND * image_size / 2)


def crop_mask(image: Image.Image, image_size: int) -> np.ndarray:
    """Recadre sur la rétine, complète en carré noir centré, redimensionne,
    puis noircit tout ce qui est hors du masque commun. Sortie uint8 (H, W, 3)."""
    bbox = fundus_bbox(image)
    if bbox is not None:
        image = image.crop(bbox)
    side = max(image.size)
    square = Image.new("RGB", (side, side))
    square.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    array = np.asarray(square.resize((image_size, image_size), Image.BILINEAR)).copy()
    array[~common_mask(image_size)] = 0
    return array


def crop_mask_annotation(image: Image.Image, annotation: np.ndarray, image_size: int) -> np.ndarray:
    """Applique à une annotation binaire (H, W) la géométrie de `crop_mask`
    pour cette image. Sortie float32 (image_size, image_size) : part de chaque
    pixel couverte par l'annotation, réduite par moyenne pour garder les
    petites lésions ; zéro hors du masque commun."""
    if annotation.shape != (image.height, image.width):
        raise ValueError(f"annotation {annotation.shape} pour une image {image.size}")
    mask = Image.fromarray(annotation.astype(np.uint8) * 255)
    bbox = fundus_bbox(image)
    if bbox is not None:
        mask = mask.crop(bbox)
    side = max(mask.size)
    square = Image.new("L", (side, side))
    square.paste(mask, ((side - mask.width) // 2, (side - mask.height) // 2))
    array = np.asarray(square.resize((image_size, image_size), Image.BOX), dtype=np.float32) / 255
    return array * common_mask(image_size)


def load_fundus_image(path: str | Path, image_size: int, preprocess: str = "none") -> torch.Tensor:
    """Image (3, H, W) dans [0, 1], comme l'attendent le contrôle qualité et
    run_baseline.py (la normalisation ImageNet est faite juste avant RETFound)."""
    if preprocess not in PREPROCESS_CHOICES:
        raise ValueError(f"preprocess doit être {PREPROCESS_CHOICES}, reçu {preprocess!r}")
    image = Image.open(path).convert("RGB")
    if preprocess == "none":
        array = np.asarray(image.resize((image_size, image_size)))
    else:
        array = crop_mask(image, image_size)
    return torch.from_numpy(array.astype(np.float32) / 255.0).permute(2, 0, 1)

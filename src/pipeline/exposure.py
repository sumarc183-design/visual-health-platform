"""Statistiques descriptives d'exposition, sans seuil de rejet validé.

Mesurer uniquement le champ rétinien : le fond noir ne doit pas être compté
comme une sous-exposition. Ces statistiques ne déterminent pas si une image
est utilisable et ne modifient pas le contrôle qualité actuel.
"""

from __future__ import annotations

import cv2
import numpy as np

DARK_LEVEL = 10
BRIGHT_LEVEL = 245


def exposure_statistics(image: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Image RGB uint8 et masque booléen H×W ; luminance sur l'échelle 0–255.

    Les niveaux 10 et 245 définissent des fractions descriptives de pixels
    sombres/clairs, pas des seuils de qualité pour l'image entière.
    """
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image RGB uint8 attendue (H, W, 3)")
    if mask.dtype != np.bool_ or mask.shape != image.shape[:2] or not mask.any():
        raise ValueError("masque booléen non vide attendu, aux dimensions de l'image")
    luminance = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)[mask]
    return {
        "mean_luminance": float(luminance.mean()),
        "median_luminance": float(np.median(luminance)),
        "dark_pixel_share": float((luminance <= DARK_LEVEL).mean()),
        "bright_pixel_share": float((luminance >= BRIGHT_LEVEL).mean()),
    }

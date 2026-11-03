"""Contrôle qualité automatique des images avant tout traitement en aval."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class QualityReport:
    is_blurry: bool
    is_usable: bool
    blur_score: float
    reasons: list[str]


def compute_blur_score(image: np.ndarray) -> float:
    """Variance du Laplacien — plus c'est bas, plus l'image est floue."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def check_quality(image: np.ndarray, blur_threshold: float = 100.0) -> QualityReport:
    reasons: list[str] = []
    blur_score = compute_blur_score(image)
    is_blurry = blur_score < blur_threshold
    if is_blurry:
        reasons.append(f"flou détecté (score={blur_score:.1f} < seuil={blur_threshold})")

    h, w = image.shape[:2]
    if h < 224 or w < 224:
        reasons.append(f"résolution trop faible ({w}x{h})")

    return QualityReport(
        is_blurry=is_blurry,
        is_usable=len(reasons) == 0,
        blur_score=blur_score,
        reasons=reasons,
    )

"""Contrôle qualité automatique des images avant tout traitement en aval.

Les images sont en RGB (comme partout dans le projet : PIL `convert("RGB")`).

Pour un fichier de fond d'œil, `check_fundus_file` mesure ce qui a un sens
sur l'image d'origine, pas sur l'entrée du modèle :
- la résolution, avant tout redimensionnement (en 224x224, une image trop
  petite au départ ne serait plus détectable) ;
- le flou, à une échelle fixe et seulement à l'intérieur de la rétine : la
  variance du Laplacien dépend de la taille de l'image, et le bord net du
  disque rétinien la gonflerait sur une image pourtant floue.

Seuil de flou calibré le 5 novembre 2026 (scripts/calibrate_quality.py,
results/quality_calibration.json) : 2e percentile du score sur les images
d'entraînement ODIR jugées acceptables. Il détecte le flou franc (79 à 90 %
des images floutées à σ ≥ 4 px), signale ~2 % d'ODIR et 0,1 % de RFMiD, et
aucune image nette d'origine. Il ne détecte pas les images « low image
quality » d'ODIR (AUROC 0,51) : leur défaut n'est pas le flou (surtout un
manque de luminosité). Le contrôle signale, il ne retire rien par défaut.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.data.fundus_preprocess import MASK_RADIUS, common_mask, crop_mask

MIN_RESOLUTION = 224  # entrée de RETFound
BLUR_ANALYSIS_SIZE = 512  # échelle fixe du calcul de flou
BLUR_MASK_MARGIN = 0.9  # rayon du masque de flou / rayon du masque commun : écarte le bord du disque
DEFAULT_BLUR_THRESHOLD = 4.16  # calibré : voir l'en-tête et results/quality_calibration.json


@dataclass
class QualityReport:
    is_blurry: bool
    is_usable: bool
    blur_score: float
    reasons: list[str]


def compute_blur_score(image: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Variance du Laplacien (image RGB ou niveaux de gris) — plus c'est bas,
    plus l'image est floue. Avec `mask` (booléen H x W), seuls les pixels du
    masque comptent."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian[mask].var() if mask is not None else laplacian.var())


def _report(blur_score: float, blur_threshold: float, width: int, height: int) -> QualityReport:
    reasons: list[str] = []
    is_blurry = blur_score < blur_threshold
    if is_blurry:
        reasons.append(f"flou détecté (score={blur_score:.1f} < seuil={blur_threshold})")
    if height < MIN_RESOLUTION or width < MIN_RESOLUTION:
        reasons.append(f"résolution trop faible ({width}x{height})")
    return QualityReport(is_blurry=is_blurry, is_usable=not reasons, blur_score=blur_score, reasons=reasons)


def check_quality(image: np.ndarray, blur_threshold: float = 100.0) -> QualityReport:
    """Contrôle d'une image déjà chargée (RGB, uint8), telle quelle."""
    h, w = image.shape[:2]
    return _report(compute_blur_score(image), blur_threshold, w, h)


def retina_blur_mask(size: int = BLUR_ANALYSIS_SIZE) -> np.ndarray:
    """Intérieur du masque commun, sans son bord (cercle plus petit)."""
    yy, xx = np.mgrid[:size, :size]
    center = (size - 1) / 2
    inner = (yy - center) ** 2 + (xx - center) ** 2 <= (BLUR_MASK_MARGIN * MASK_RADIUS * size / 2) ** 2
    return common_mask(size) & inner


def check_fundus_file(path: str | Path, blur_threshold: float = DEFAULT_BLUR_THRESHOLD) -> QualityReport:
    """Contrôle d'un fichier de fond d'œil d'origine : résolution d'origine,
    flou mesuré sur la rétine recadrée à BLUR_ANALYSIS_SIZE."""
    image = Image.open(path).convert("RGB")
    retina = crop_mask(image, BLUR_ANALYSIS_SIZE)
    blur_score = compute_blur_score(retina, retina_blur_mask())
    return _report(blur_score, blur_threshold, image.width, image.height)

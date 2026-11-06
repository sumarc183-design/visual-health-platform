"""IDRiD, partie segmentation (Porwal et al., CC BY 4.0) : 81 images de fond
d'œil avec rétinopathie diabétique, découpage officiel 54 / 27, lésions
dessinées par des experts au pixel près, disque optique en plus.

Un masque absent signifie que la lésion n'a pas été annotée sur cette image :
il compte comme vide (cas de 41 images pour les exsudats mous).

Structure attendue, cherchée sous `root` :
<dossier>/1. Original Images/{a. Training Set,b. Testing Set}/IDRiD_01.jpg et
<dossier>/2. All Segmentation Groundtruths/{...}/{1. Microaneurysms,...}/IDRiD_01_MA.tif
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .fundus_preprocess import crop_mask, crop_mask_annotation

LESION_DIRS = {
    "MA": "1. Microaneurysms",
    "HE": "2. Haemorrhages",
    "EX": "3. Hard Exudates",
    "SE": "4. Soft Exudates",
    "OD": "5. Optic Disc",
}
LESIONS = ("MA", "HE", "EX", "SE")  # le disque optique est une structure normale, pas une lésion
SPLIT_DIRS = {"train": "a. Training Set", "test": "b. Testing Set"}
_MASKS_DIR = "2. All Segmentation Groundtruths"


def find_segmentation_root(root: str | Path) -> Path:
    root = Path(root)
    if (root / _MASKS_DIR).is_dir():
        return root
    hits = sorted(p.parent for p in root.rglob(_MASKS_DIR) if p.is_dir())
    if not hits:
        raise FileNotFoundError(f"Segmentation IDRiD introuvable sous {root}")
    return hits[0]


def read_annotation(path: Path) -> np.ndarray:
    """Masque booléen (H, W). Presque tous les fichiers sont en palette ; un
    masque au moins (IDRiD_81_EX) est en RGBA, lésion dans le rouge et alpha
    plein partout : l'alpha n'est jamais lu."""
    with Image.open(path) as mask:
        array = np.asarray(mask.convert("RGB") if mask.mode in ("RGB", "RGBA") else mask)
    return array.max(axis=2) > 0 if array.ndim == 3 else array > 0


class IDRiDSegmentation:
    def __init__(self, root: str | Path, split: str = "train", image_size: int = 224):
        if split not in SPLIT_DIRS:
            raise ValueError(f"split doit être {tuple(SPLIT_DIRS)}, reçu {split!r}")
        self.root = find_segmentation_root(root)
        self.split = split
        self.image_size = image_size
        self.images_dir = self.root / "1. Original Images" / SPLIT_DIRS[split]
        self.masks_dir = self.root / _MASKS_DIR / SPLIT_DIRS[split]
        self.ids = sorted(p.stem for p in self.images_dir.glob("IDRiD_*.jpg"))
        if not self.ids:
            raise ValueError(f"Aucune image dans {self.images_dir}")

    def __len__(self) -> int:
        return len(self.ids)

    def image_path(self, idx: int) -> Path:
        return self.images_dir / f"{self.ids[idx]}.jpg"

    def image(self, idx: int) -> torch.Tensor:
        """Image `crop-mask` (3, H, W) dans [0, 1], comme load_fundus_image."""
        with Image.open(self.image_path(idx)) as image:
            array = crop_mask(image.convert("RGB"), self.image_size)
        return torch.from_numpy(array.astype(np.float32) / 255.0).permute(2, 0, 1)

    def masks(self, idx: int) -> dict[str, np.ndarray]:
        """Lésion -> part de chaque pixel couverte, même géométrie que l'image."""
        result = {}
        with Image.open(self.image_path(idx)) as image:
            image = image.convert("RGB")
            for lesion, folder in LESION_DIRS.items():
                path = self.masks_dir / folder / f"{self.ids[idx]}_{lesion}.tif"
                if path.exists():
                    result[lesion] = crop_mask_annotation(image, read_annotation(path), self.image_size)
                else:
                    result[lesion] = np.zeros((self.image_size, self.image_size), np.float32)
        return result

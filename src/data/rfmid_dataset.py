"""Dataset RFMiD réel -- consomme la structure Kaggle/RIADD telle qu'organisée
par scripts/download_rfmid_odir.py dans data/raw/rfmid/.

Les CSV réels ont 45 colonnes de pathologies (+ Disease_Risk, label binaire
séparé), pas les C0..C27 génériques du générateur synthétique. Beaucoup de
ces 45 classes ont trop peu d'exemples positifs pour être apprises -- le
sous-ensemble de 28 classes de configs/config.yaml est reproduit ici en
gardant les classes avec >= MIN_CLASS_POSITIVES positifs dans le train set
(vérifié : ce seuil donne exactement 28 classes). Toujours dérivé du train
seul, jamais de val/test, pour éviter toute fuite.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

MIN_CLASS_POSITIVES = 5  # seuil qui donne les 28 classes de configs/config.yaml

# (dossier_niveau1, dossier_niveau2, dossier_images, nom_csv) -- la structure
# Kaggle imbrique deux fois le même nom de dossier (ex. Training_Set/Training_Set/).
_SPLIT_DIRS = {
    "train": ("Training_Set", "Training_Set", "Training", "RFMiD_Training_Labels.csv"),
    "val": ("Evaluation_Set", "Evaluation_Set", "Validation", "RFMiD_Validation_Labels.csv"),
    "test": ("Test_Set", "Test_Set", "Test", "RFMiD_Testing_Labels.csv"),
}


def _labels_csv_path(root: Path, split: str) -> Path:
    top, mid, _, csv_name = _SPLIT_DIRS[split]
    return root / top / mid / csv_name


def _images_dir(root: Path, split: str) -> Path:
    top, mid, images, _ = _SPLIT_DIRS[split]
    return root / top / mid / images


def select_classes(root: Path, min_positives: int = MIN_CLASS_POSITIVES) -> list[str]:
    """Classes retenues, dérivées du train set uniquement."""
    train_df = pd.read_csv(_labels_csv_path(root, "train"))
    pathology_cols = [c for c in train_df.columns if c not in ("ID", "Disease_Risk")]
    counts = train_df[pathology_cols].sum()
    return [c for c in pathology_cols if counts[c] >= min_positives]


class RFMiDDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        image_size: int = 224,
        min_class_positives: int = MIN_CLASS_POSITIVES,
    ):
        if split not in _SPLIT_DIRS:
            raise ValueError(f"split doit être train/val/test, reçu {split!r}")
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.class_names = select_classes(self.root, min_class_positives)
        self.df = pd.read_csv(_labels_csv_path(self.root, split))
        self.images_dir = _images_dir(self.root, split)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(self.images_dir / f"{int(row['ID'])}.png").convert("RGB")
        image = np.array(image.resize((self.image_size, self.image_size))).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        labels = torch.tensor(row[self.class_names].to_numpy(dtype=np.float32))
        return image, labels

"""Dataset EyePACS (compétition Kaggle « diabetic-retinopathy-detection ») :
2e validation externe de la rétinopathie diabétique.

35 126 images d'entraînement annotées en 5 stades (0 pas de rétinopathie à
4 proliférante), issues d'un programme de dépistage américain (EyePACS) :
le format d'image, indicateur indirect de l'acquisition, y est peu lié au
diagnostic, contrairement à APTOS où il le donne presque. Les deux yeux d'un patient sont
deux images (`<patient>_left.jpeg`, `<patient>_right.jpeg`) : le bootstrap
se fait par patient.

Seule une partie des images est extraite en pratique (échantillon de
patients, notebooks/colab_external_eyepacs.ipynb) : le dataset ne garde que
les lignes de `trainLabels.csv` dont l'image est présente.

Mêmes étiquettes binaires qu'APTOS : `any_dr` (stade >= 1) et
`referable_dr` (stade >= 2).

Structure attendue : <root>/trainLabels.csv (image, level) et
<root>/train/<image>.jpeg.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from .fundus_preprocess import load_fundus_image

CLASS_NAMES = ["any_dr", "referable_dr"]


class EyePACSDataset(Dataset):
    def __init__(self, root: str | Path, image_size: int = 224, preprocess: str = "crop-mask"):
        self.root = Path(root)
        self.image_size = image_size
        self.preprocess = preprocess
        self.class_names = CLASS_NAMES
        df = pd.read_csv(self.root / "trainLabels.csv")
        if not {"image", "level"} <= set(df.columns):
            raise ValueError(f"{self.root / 'trainLabels.csv'} : colonnes image et level attendues")
        if not df["level"].between(0, 4).all():
            raise ValueError("level doit être un stade entre 0 et 4")
        present = df["image"].map(lambda name: (self.root / "train" / f"{name}.jpeg").exists())
        df = df[present].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"aucune image de trainLabels.csv trouvée dans {self.root / 'train'}")
        df["diagnosis"] = df["level"]
        df["any_dr"] = (df["level"] >= 1).astype(int)
        df["referable_dr"] = (df["level"] >= 2).astype(int)
        self.df = df

    @property
    def image_keys(self) -> list[str]:
        return [f"eyepacs/{i}" for i in self.df["image"]]

    @property
    def groups(self) -> list[str]:
        """Patient : partie du nom avant « _left » / « _right »."""
        return [str(i).rsplit("_", 1)[0] for i in self.df["image"]]

    def image_path(self, idx: int) -> Path:
        return self.root / "train" / f"{self.df.iloc[idx]['image']}.jpeg"

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        image = load_fundus_image(self.image_path(idx), self.image_size, self.preprocess)
        labels = torch.tensor(self.df.iloc[idx][self.class_names].to_numpy(dtype="float32"))
        return image, labels

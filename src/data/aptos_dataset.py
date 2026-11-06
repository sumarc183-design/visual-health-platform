"""Dataset APTOS 2019 (compétition Kaggle « aptos2019-blindness-detection ») :
validation externe de la rétinopathie diabétique.

3 662 images d'entraînement annotées par des cliniciens (Aravind Eye
Hospital, Inde) en 5 stades : 0 pas de rétinopathie, 1 légère, 2 modérée,
3 sévère, 4 proliférante. Autres patients et autres annotateurs que RFMiD et ODIR, appareils probablement
différents : base sans rôle dans le développement initial, mais déjà utilisée pour
plusieurs analyses.

Étiquettes binaires dérivées du stade :
- `any_dr` : stade >= 1 (toute rétinopathie, comme « DR » de RFMiD et « D »
  d'ODIR, qui incluent les formes légères) ;
- `referable_dr` : stade >= 2 (rétinopathie à adresser à un spécialiste,
  critère usuel du dépistage).

Seul `train.csv` est annoté (les étiquettes du test de la compétition ne
sont pas publiques). APTOS ne fournit pas d'identifiant patient : un même
patient peut avoir plusieurs images, le bootstrap par image est donc un peu
optimiste sur la largeur des intervalles.

Structure attendue (input Kaggle) : <root>/train.csv (id_code, diagnosis)
et <root>/train_images/<id_code>.png.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from .fundus_preprocess import load_fundus_image

CLASS_NAMES = ["any_dr", "referable_dr"]


class APTOSDataset(Dataset):
    def __init__(self, root: str | Path, image_size: int = 224, preprocess: str = "crop-mask"):
        self.root = Path(root)
        self.image_size = image_size
        self.preprocess = preprocess
        self.class_names = CLASS_NAMES
        df = pd.read_csv(self.root / "train.csv")
        if not {"id_code", "diagnosis"} <= set(df.columns):
            raise ValueError(f"{self.root / 'train.csv'} : colonnes id_code et diagnosis attendues")
        if not df["diagnosis"].between(0, 4).all():
            raise ValueError("diagnosis doit être un stade entre 0 et 4")
        df["any_dr"] = (df["diagnosis"] >= 1).astype(int)
        df["referable_dr"] = (df["diagnosis"] >= 2).astype(int)
        self.df = df.reset_index(drop=True)

    @property
    def image_keys(self) -> list[str]:
        return [f"aptos/{i}" for i in self.df["id_code"]]

    @property
    def groups(self) -> list[str]:
        """Pas d'identifiant patient dans APTOS : chaque image compte seule."""
        return self.image_keys

    def image_path(self, idx: int) -> Path:
        return self.root / "train_images" / f"{self.df.iloc[idx]['id_code']}.png"

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        image = load_fundus_image(self.image_path(idx), self.image_size, self.preprocess)
        labels = torch.tensor(self.df.iloc[idx][self.class_names].to_numpy(dtype="float32"))
        return image, labels

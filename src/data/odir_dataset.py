"""Dataset ODIR-5K réel -- consomme full_df.csv tel qu'organisé par
scripts/download_rfmid_odir.py dans data/raw/odir/.

Piège de ce CSV : deux étiquettes différentes par ligne. Les colonnes
N/D/G/C/A/H/M/O sont le diagnostic PATIENT (agrégé sur les deux yeux) ; la
colonne `target`/`labels` est un label par œil, dérivé d'un parsing des
mots-clés diagnostiques. On garde ici N..O (conforme à configs/config.yaml),
dupliqué sur les deux images du patient -- pas `target`, qui mélangerait
deux conventions différentes dans le même jeu d'entraînement.

`Testing Images/` (1000 fichiers) n'a pas d'étiquette dans full_df.csv --
c'est le jeu de test caché du challenge Kaggle d'origine, inutilisable pour
une évaluation supervisée. Le split train/val est donc fait ici sur les
images annotées de `Training Images/`.

Piège n°2 : chaque patient a 2 lignes (œil gauche + œil droit) qui
partagent le même label N..O. Un split ligne-par-ligne peut mettre l'œil
gauche d'un patient en train et l'œil droit du même patient en val --
fuite qui gonfle artificiellement les métriques (le modèle "reconnaît" en
partie le patient, pas la pathologie). Le split est donc groupé par
patient (colonne `ID`) via GroupShuffleSplit, jamais par ligne.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset

CLASS_NAMES = ["N", "D", "G", "C", "A", "H", "M", "O"]
PATIENT_ID_COLUMN = "ID"
VAL_FRACTION = 0.15
SPLIT_SEED = 42


class ODIRDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        image_size: int = 224,
        val_fraction: float = VAL_FRACTION,
        seed: int = SPLIT_SEED,
    ):
        if split not in ("train", "val"):
            raise ValueError(f"split doit être train/val, reçu {split!r}")
        self.root = Path(root)
        self.image_size = image_size
        self.class_names = CLASS_NAMES

        df = pd.read_csv(self.root / "full_df.csv")
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
        train_idx, val_idx = next(splitter.split(df, groups=df[PATIENT_ID_COLUMN]))

        # Un même patient ne doit jamais apparaître des deux côtés du split.
        train_patients = set(df.iloc[train_idx][PATIENT_ID_COLUMN])
        val_patients = set(df.iloc[val_idx][PATIENT_ID_COLUMN])
        assert train_patients.isdisjoint(val_patients), "fuite patient train/val détectée"

        self.df = df.iloc[train_idx if split == "train" else val_idx].reset_index(drop=True)
        self.images_dir = self.root / "ODIR-5K" / "ODIR-5K" / "Training Images"

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(self.images_dir / row["filename"]).convert("RGB")
        image = np.array(image.resize((self.image_size, self.image_size))).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        labels = torch.tensor(row[self.class_names].to_numpy(dtype=np.float32))
        return image, labels

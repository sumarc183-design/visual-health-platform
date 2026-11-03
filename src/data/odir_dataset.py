"""Dataset ODIR-5K réel -- consomme full_df.csv tel qu'organisé par
scripts/download_rfmid_odir.py dans data/raw/odir/.

Piège de ce CSV : les colonnes N/D/G/C/A/H/M/O sont le diagnostic PATIENT
(agrégé sur les deux yeux), alors que chaque ligne est l'image d'UN œil.
Les recopier sur l'image (label_level="patient", comportement d'origine)
étiquette faux 20,8 % des images (1 329 / 6 392) : un œil sain d'un
patient diabétique devient "D" (401 images), "O" est faux sur 45 % de ses
positifs (708 / 1 588).

Par défaut (label_level="eye"), l'étiquette est donc reconstruite pour
l'œil photographié, depuis ses propres mots-clés diagnostiques
(`Left-/Right-Diagnostic Keywords`) via keywords_to_classes(). Validation :
le OU des étiquettes des deux yeux redonne exactement les colonnes patient
N..O pour 3 358 / 3 358 patients (test_real_datasets.py). La colonne
`target`/`labels` du CSV n'est pas utilisée : mono-label, elle ne peut pas
représenter un œil à plusieurs pathologies.

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

# Mots-clés qui décrivent la prise de vue, pas l'œil : ne donnent aucune classe.
NON_DIAGNOSTIC_KEYWORDS = {
    "lens dust", "low image quality", "image offset", "anterior segment image",
    "no fundus image", "optic disk photographically invisible",
}


def _keyword_class(keyword: str) -> str | None:
    if keyword == "normal fundus":
        return "N"
    if keyword in NON_DIAGNOSTIC_KEYWORDS:
        return None
    if "nonproliferative retinopathy" in keyword or "non proliferative retinopathy" in keyword \
            or "diabetic retinopathy" in keyword:
        return "D"
    if "glaucoma" in keyword:
        return "G"
    if "cataract" in keyword:
        return "C"
    if "age-related macular degeneration" in keyword:
        return "A"
    if "hypertensive retinopathy" in keyword:
        return "H"
    if keyword in ("pathological myopia", "myopia retinopathy", "myopic maculopathy"):
        return "M"
    return "O"


def keywords_to_classes(keywords: str) -> set[str]:
    """Mots-clés diagnostiques d'un œil ("moderate non proliferative
    retinopathy，lens dust") -> classes ODIR ({"D"}). N n'est gardé que si
    aucune pathologie n'est présente, comme dans les colonnes patient."""
    classes = {_keyword_class(k.strip()) for k in str(keywords).replace("，", ",").split(",")} - {None}
    if len(classes) > 1:
        classes.discard("N")
    return classes


def eye_level_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Remplace les colonnes N..O (patient) par l'étiquette de l'œil de chaque
    ligne, d'après `filename` (<id>_left.jpg / <id>_right.jpg)."""
    df = df.copy()
    is_left = df["filename"].str.contains("_left")
    keywords = df["Left-Diagnostic Keywords"].where(is_left, df["Right-Diagnostic Keywords"])
    eye_classes = keywords.map(keywords_to_classes)
    for class_name in CLASS_NAMES:
        df[class_name] = eye_classes.map(lambda classes: int(class_name in classes))
    return df


class ODIRDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        image_size: int = 224,
        val_fraction: float = VAL_FRACTION,
        seed: int = SPLIT_SEED,
        label_level: str = "eye",
    ):
        if split not in ("train", "val"):
            raise ValueError(f"split doit être train/val, reçu {split!r}")
        if label_level not in ("eye", "patient"):
            raise ValueError(f"label_level doit être eye/patient, reçu {label_level!r}")
        self.label_level = label_level
        self.root = Path(root)
        self.image_size = image_size
        self.class_names = CLASS_NAMES

        df = pd.read_csv(self.root / "full_df.csv")
        if label_level == "eye":
            df = eye_level_labels(df)
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

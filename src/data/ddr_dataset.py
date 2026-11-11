"""Dataset DDR (gradation de la rétinopathie diabétique) : cohorte de confirmation.

13 673 photos de fond d'œil de Chine (Li et al., Information Sciences, 2019), gradées
de 0 (pas de rétinopathie) à 4 (proliférante), 5 = non gradable. Licence CC BY-NC-SA 4.0
(pas d'usage commercial sans accord écrit des auteurs). Les images non gradables sont
écartées.

Pas d'identifiant patient publié : chaque image forme son groupe ; l'analyse regroupe
ensuite les doublons (embeddings) et, en contrôle, les images prises à moins de deux
minutes d'intervalle (`time_groups`, nom de fichier = date et heure de prise).

Structure attendue : <root>/DR_grading/{train,valid,test}.txt (fichier, stade) et
<root>/DR_grading/{train,valid,test}/<fichier>.jpg. Les trois découpages sont réunis :
le projet n'apprend rien sur DDR.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from .fundus_preprocess import load_fundus_image

CLASS_NAMES = ["any_dr", "referable_dr"]
SPLITS = ("train", "valid", "test")
UNGRADABLE = 5


def read_grades(root: Path) -> pd.DataFrame:
    frames = []
    for split in SPLITS:
        df = pd.read_csv(root / "DR_grading" / f"{split}.txt", sep=" ", header=None, names=["image", "level"])
        df["split"] = split
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    if not df["level"].between(0, UNGRADABLE).all():
        raise ValueError("stade DDR attendu entre 0 et 5")
    if df["image"].duplicated().any():
        raise ValueError("nom d'image DDR en double entre découpages")
    return df


def capture_time(name: str) -> datetime | None:
    """Date et heure de prise lues dans le nom (AAAAMMJJhhmmss + millisecondes), sinon None."""
    stem = Path(name).stem
    try:
        return datetime.strptime(stem[:14], "%Y%m%d%H%M%S")
    except ValueError:
        return None


def time_groups(names: list[str], gap_seconds: int = 120) -> list[str]:
    """Chaîne les images prises à moins de `gap_seconds` d'intervalle (même groupe).

    Approximation du patient (les deux yeux sont photographiés l'un après l'autre) ;
    une image sans horodatage lisible reste seule.
    """
    order = sorted(range(len(names)), key=lambda i: (capture_time(names[i]) is None, capture_time(names[i]) or 0))
    groups = [""] * len(names)
    current, previous = None, None
    for i in order:
        t = capture_time(names[i])
        if t is None:
            groups[i] = f"solo/{names[i]}"
            continue
        if previous is None or (t - previous).total_seconds() > gap_seconds:
            current = f"t/{names[i]}"
        groups[i] = current
        previous = t
    return groups


class DDRDataset(Dataset):
    def __init__(self, root: str | Path, image_size: int = 224, preprocess: str = "crop-mask"):
        self.root = Path(root)
        self.image_size = image_size
        self.preprocess = preprocess
        self.class_names = CLASS_NAMES
        df = read_grades(self.root)
        df = df[df["level"] != UNGRADABLE].reset_index(drop=True)
        df["diagnosis"] = df["level"]
        df["any_dr"] = (df["level"] >= 1).astype(int)
        df["referable_dr"] = (df["level"] >= 2).astype(int)
        self.df = df

    @property
    def image_keys(self) -> list[str]:
        return [f"ddr/{s}/{i}" for s, i in zip(self.df["split"], self.df["image"], strict=True)]

    @property
    def groups(self) -> list[str]:
        return self.image_keys

    def image_path(self, idx: int) -> Path:
        row = self.df.iloc[idx]
        return self.root / "DR_grading" / row["split"] / row["image"]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        image = load_fundus_image(self.image_path(idx), self.image_size, self.preprocess)
        labels = torch.tensor(self.df.iloc[idx][self.class_names].to_numpy(dtype="float32"))
        return image, labels

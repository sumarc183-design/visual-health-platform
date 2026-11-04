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
une évaluation supervisée. Le split est donc fait ici sur les images
annotées de `Training Images/` : 70 % train, 15 % val, 15 % test par
défaut. Le test est mis de côté : les choix (étiquettes, prétraitement,
réglages de la tête) se font sur val, le test n'est lu qu'une fois à la
fin. val est identique au split d'origine 85/15 ; le test est pris dans
l'ancien train (voir split_by_patient). `test_fraction=0` redonne le split
d'origine sans test (premiers résultats publiés).

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
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset

from .fundus_preprocess import load_fundus_image

CLASS_NAMES = ["N", "D", "G", "C", "A", "H", "M", "O"]
PATIENT_ID_COLUMN = "ID"
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
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
    aucune pathologie n'est présente, comme dans les colonnes patient.

    Tout mot-clé diagnostique non reconnu va dans O (« autres maladies et
    anomalies » dans ODIR : drusen, occlusions veineuses, membrane
    épirétinienne...). Une cellule vide ou manquante est en revanche une
    erreur : sans annotation, l'œil ne doit pas devenir « autre »."""
    text = "" if pd.isna(keywords) else str(keywords)
    parts = [k.strip().lower() for k in text.replace("，", ",").split(",")]
    parts = [k for k in parts if k]
    if not parts:
        raise ValueError(f"mots-clés diagnostiques absents ({keywords!r})")
    classes = {_keyword_class(k) for k in parts} - {None}
    if len(classes) > 1:
        classes.discard("N")
    return classes


def eye_level_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Remplace les colonnes N..O (patient) par l'étiquette de l'œil de chaque
    ligne, d'après `filename` (<id>_left.jpg / <id>_right.jpg)."""
    df = df.copy()
    is_left = df["filename"].str.contains("_left")
    is_right = df["filename"].str.contains("_right")
    ambiguous = df.loc[is_left == is_right, "filename"]
    if len(ambiguous):
        raise ValueError(
            f"{len(ambiguous)} nom(s) de fichier sans latéralité claire (_left ou _right), "
            f"ex. {list(ambiguous[:3])}"
        )
    keywords = df["Left-Diagnostic Keywords"].where(is_left, df["Right-Diagnostic Keywords"])
    eye_classes = keywords.map(keywords_to_classes)
    for class_name in CLASS_NAMES:
        df[class_name] = eye_classes.map(lambda classes: int(class_name in classes))
    return df


def split_by_patient(df: pd.DataFrame, val_fraction: float, test_fraction: float, seed: int) -> dict:
    """Indices train/val(/test) groupés par patient.

    val est le split d'origine (85/15), inchangé : c'est sur lui qu'ont été
    prises les décisions jusqu'ici (étiquettes par œil, prétraitement). Le
    test est donc tiré dans l'ancien train, jamais évalué : tiré ailleurs, il
    recouvrirait des images déjà utilisées pour décider."""
    groups = df[PATIENT_ID_COLUMN]
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(splitter.split(df, groups=groups))
    if test_fraction == 0:
        return {"train": train_idx, "val": val_idx}
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_fraction / (1 - val_fraction), random_state=seed)
    train_rel, test_rel = next(splitter.split(train_idx, groups=groups.iloc[train_idx]))
    return {"train": train_idx[train_rel], "val": val_idx, "test": train_idx[test_rel]}


class ODIRDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        image_size: int = 224,
        val_fraction: float = VAL_FRACTION,
        test_fraction: float = TEST_FRACTION,
        seed: int = SPLIT_SEED,
        label_level: str = "eye",
        preprocess: str = "none",
    ):
        if split not in ("train", "val", "test"):
            raise ValueError(f"split doit être train/val/test, reçu {split!r}")
        if split == "test" and test_fraction == 0:
            raise ValueError("pas de split test avec test_fraction=0 (split d'origine 85/15)")
        if label_level not in ("eye", "patient"):
            raise ValueError(f"label_level doit être eye/patient, reçu {label_level!r}")
        self.label_level = label_level
        self.root = Path(root)
        self.image_size = image_size
        self.preprocess = preprocess
        self.class_names = CLASS_NAMES

        df = pd.read_csv(self.root / "full_df.csv")
        if label_level == "eye":
            df = eye_level_labels(df)
        indices = split_by_patient(df, val_fraction, test_fraction, seed)

        # Un même patient ne doit jamais apparaître dans deux splits.
        patients = {name: set(df.iloc[idx][PATIENT_ID_COLUMN]) for name, idx in indices.items()}
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            assert patients.get(a, set()).isdisjoint(patients.get(b, set())), f"fuite patient {a}/{b} détectée"

        self.df = df.iloc[indices[split]].reset_index(drop=True)
        self.images_dir = self.root / "ODIR-5K" / "ODIR-5K" / "Training Images"

    @property
    def image_keys(self) -> list[str]:
        """Identifiant stable de chaque image (cache d'embeddings, prédictions)."""
        return list(self.df["filename"])

    @property
    def groups(self) -> list[str]:
        """Patient de chaque image : unité du bootstrap (les deux yeux sont liés)."""
        return [str(i) for i in self.df[PATIENT_ID_COLUMN]]

    def __len__(self) -> int:
        return len(self.df)

    def image_path(self, idx: int) -> Path:
        """Fichier d'origine (contrôle qualité sur l'image non redimensionnée)."""
        return self.images_dir / self.df.iloc[idx]["filename"]

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = load_fundus_image(self.image_path(idx), self.image_size, self.preprocess)
        labels = torch.tensor(row[self.class_names].to_numpy(dtype=np.float32))
        return image, labels

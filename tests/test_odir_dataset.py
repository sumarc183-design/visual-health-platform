"""Vérifie que ODIRDataset ne fuit pas de patient entre train et val (un
même patient a 2 lignes -- œil gauche/droit -- partageant le même label ;
un split ligne-par-ligne mettrait parfois les deux yeux d'un même patient
de part et d'autre du split)."""

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.data.odir_dataset import CLASS_NAMES, ODIRDataset, keywords_to_classes


@pytest.fixture
def synthetic_odir_root(tmp_path):
    root = tmp_path / "odir"
    images_dir = root / "ODIR-5K" / "ODIR-5K" / "Training Images"
    images_dir.mkdir(parents=True)

    # Œil gauche malade (cataracte), œil droit sain : le diagnostic patient
    # (colonnes N..O) est C, l'étiquette par œil diffère selon l'image.
    rng = np.random.default_rng(0)
    rows = []
    for patient_id in range(50):
        labels = {c: 0 for c in CLASS_NAMES}
        labels["C"] = 1
        keywords = {"Left-Diagnostic Keywords": "cataract", "Right-Diagnostic Keywords": "normal fundus，lens dust"}
        for eye in ("left", "right"):
            filename = f"{patient_id}_{eye}.jpg"
            Image.fromarray((rng.random((32, 32, 3)) * 255).astype(np.uint8)).save(images_dir / filename)
            rows.append({"ID": patient_id, "filename": filename, **keywords, **labels})

    pd.DataFrame(rows).to_csv(root / "full_df.csv", index=False)
    return root


def test_no_patient_appears_in_both_splits(synthetic_odir_root):
    train_ds = ODIRDataset(synthetic_odir_root, split="train", image_size=32)
    val_ds = ODIRDataset(synthetic_odir_root, split="val", image_size=32)

    train_patients = set(train_ds.df["ID"])
    val_patients = set(val_ds.df["ID"])
    assert train_patients.isdisjoint(val_patients)
    assert len(train_patients) > 0 and len(val_patients) > 0


def test_getitem_shapes(synthetic_odir_root):
    dataset = ODIRDataset(synthetic_odir_root, split="train", image_size=32)
    image, labels = dataset[0]
    assert image.shape == (3, 32, 32)
    assert labels.shape == (len(CLASS_NAMES),)


def test_eye_level_labels_follow_the_photographed_eye(synthetic_odir_root):
    eye_ds = ODIRDataset(synthetic_odir_root, split="train", image_size=32, label_level="eye")
    patient_ds = ODIRDataset(synthetic_odir_root, split="train", image_size=32, label_level="patient")
    assert list(eye_ds.df["filename"]) == list(patient_ds.df["filename"])  # même split

    is_left = eye_ds.df["filename"].str.contains("_left")
    assert (eye_ds.df.loc[is_left, "C"] == 1).all() and (eye_ds.df.loc[is_left, "N"] == 0).all()
    assert (eye_ds.df.loc[~is_left, "N"] == 1).all() and (eye_ds.df.loc[~is_left, "C"] == 0).all()
    assert (patient_ds.df["C"] == 1).all()  # comportement d'origine : diagnostic patient sur les deux yeux


@pytest.mark.parametrize(
    "keywords, expected",
    [
        ("normal fundus", {"N"}),
        ("normal fundus，lens dust", {"N"}),
        ("moderate non proliferative retinopathy", {"D"}),
        ("mild nonproliferative retinopathy，hypertensive retinopathy", {"D", "H"}),
        ("suspected glaucoma", {"G"}),
        ("dry age-related macular degeneration", {"A"}),
        ("pathological myopia", {"M"}),
        ("epiretinal membrane", {"O"}),
        ("normal fundus，drusen", {"O"}),  # une pathologie efface N
        ("low image quality", set()),
    ],
)
def test_keywords_to_classes(keywords, expected):
    assert keywords_to_classes(keywords) == expected

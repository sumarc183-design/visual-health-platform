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


def test_three_way_split_keeps_original_val_and_draws_test_from_old_train(synthetic_odir_root):
    """val = split d'origine 85/15 (sur lequel on a déjà décidé) ; le test
    vient de l'ancien train, jamais évalué ; aucun patient partagé."""
    old_train = set(ODIRDataset(synthetic_odir_root, split="train", test_fraction=0).df["ID"])
    old_val = set(ODIRDataset(synthetic_odir_root, split="val", test_fraction=0).df["ID"])
    new = {s: set(ODIRDataset(synthetic_odir_root, split=s).df["ID"]) for s in ("train", "val", "test")}

    assert new["val"] == old_val
    assert new["test"] and new["test"] <= old_train
    assert new["train"].isdisjoint(new["test"]) and new["train"] | new["test"] == old_train


def test_no_test_split_without_test_fraction(synthetic_odir_root):
    with pytest.raises(ValueError):
        ODIRDataset(synthetic_odir_root, split="test", test_fraction=0)


def test_image_keys_and_groups(synthetic_odir_root):
    dataset = ODIRDataset(synthetic_odir_root, split="val")
    assert dataset.image_keys == list(dataset.df["filename"])
    assert dataset.groups == [str(i) for i in dataset.df["ID"]]


def test_missing_keywords_are_an_error_not_class_o():
    """Sans annotation, l'œil ne doit pas devenir « autre maladie »."""
    for empty in ("", " ", float("nan"), " , "):
        with pytest.raises(ValueError):
            keywords_to_classes(empty)
    assert keywords_to_classes("Cataract") == {"C"}  # casse normalisée
    assert keywords_to_classes("drusen") == {"O"}  # vraie pathologie non listée -> O


def test_filename_without_laterality_is_rejected(synthetic_odir_root):
    csv = synthetic_odir_root / "full_df.csv"
    df = pd.read_csv(csv)
    df.loc[0, "filename"] = "0.jpg"
    df.to_csv(csv, index=False)
    with pytest.raises(ValueError, match="latéralité"):
        ODIRDataset(synthetic_odir_root, split="train", image_size=32)

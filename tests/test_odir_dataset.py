"""Vérifie que ODIRDataset ne fuit pas de patient entre train et val (un
même patient a 2 lignes -- œil gauche/droit -- partageant le même label ;
un split ligne-par-ligne mettrait parfois les deux yeux d'un même patient
de part et d'autre du split)."""
import os

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.data.odir_dataset import CLASS_NAMES, ODIRDataset


@pytest.fixture
def synthetic_odir_root(tmp_path):
    root = tmp_path / "odir"
    images_dir = root / "ODIR-5K" / "ODIR-5K" / "Training Images"
    images_dir.mkdir(parents=True)

    rng = np.random.default_rng(0)
    rows = []
    for patient_id in range(50):
        labels = {c: 0 for c in CLASS_NAMES}
        labels[rng.choice(CLASS_NAMES)] = 1
        for eye in ("left", "right"):
            filename = f"{patient_id}_{eye}.jpg"
            Image.fromarray((rng.random((32, 32, 3)) * 255).astype(np.uint8)).save(images_dir / filename)
            rows.append({"ID": patient_id, "filename": filename, **labels})

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

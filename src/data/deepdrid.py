"""DeepDRiD, photos de fond d'œil classiques (CC BY-SA 4.0) : notes de qualité d'experts.

Liu et al., « DeepDRiD: Diabetic Retinopathy—Grading and Image Quality Estimation
Challenge », Patterns 2022 (doi:10.1016/j.patter.2022.100512),
https://github.com/deepdrdoc/DeepDRiD. Trois découpages, patients disjoints :
entraînement (1 200 images, 300 patients), validation (400, 100) et évaluation
(400, 100 ; notes dans `Challenge2_labels.xlsx`).

Notes (Readme officiel) : `Overall quality` 0 = qualité insuffisante pour le diagnostic,
1 = suffisante ; `Artifact` 0 (aucun) à 10 (tout le pôle postérieur couvert) ;
`Clarity` 1 (seule l'arcade de niveau 1 visible) à 10 (niveau 3 et toutes les lésions) ;
`Field definition` 1 (ni disque ni macula) à 10 (centrés à moins d'1 diamètre papillaire).
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pandas as pd

SPLITS = {
    "train": ("regular-fundus-training", "regular-fundus-training.csv"),
    "val": ("regular-fundus-validation", "regular-fundus-validation.csv"),
    "eval": ("Online-Challenge1&2-Evaluation", "Challenge2_labels.xlsx"),
}
QUALITY_COLUMNS = ["Overall quality", "Artifact", "Clarity", "Field definition"]
_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def read_xlsx(path: str | Path) -> pd.DataFrame:
    """Première feuille d'un classeur .xlsx simple (en-tête en première ligne), sans openpyxl."""
    with zipfile.ZipFile(path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            shared = ["".join(t.text or "" for t in si.iter(f"{{{_NS['m']}}}t")) for si in root.findall("m:si", _NS)]
        sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    rows = []
    for r in sheet.iter(f"{{{_NS['m']}}}row"):
        values = {}
        for c in r.findall("m:c", _NS):
            v = c.find("m:v", _NS)
            if v is not None:
                column = re.match(r"[A-Z]+", c.get("r")).group()
                values[column] = shared[int(v.text)] if c.get("t") == "s" else v.text
        rows.append(values)
    if not rows:
        raise ValueError(f"{path} : feuille vide")
    header = rows[0]
    return pd.DataFrame([{header[k]: v for k, v in r.items() if k in header} for r in rows[1:]])


def load_split(root: str | Path, split: str) -> pd.DataFrame:
    """Une ligne par image : image_id, patient, path, notes de qualité (entiers)."""
    if split not in SPLITS:
        raise ValueError(f"découpage {split!r} inconnu, attendu parmi {sorted(SPLITS)}")
    folder, labels = SPLITS[split]
    base = Path(root) / folder
    df = read_xlsx(base / labels) if labels.endswith(".xlsx") else pd.read_csv(base / labels)
    missing = [c for c in ["image_id", *QUALITY_COLUMNS] if c not in df]
    if missing:
        raise ValueError(f"DeepDRiD {split} : colonnes absentes {missing}")
    df = df.copy()
    df["image_id"] = df["image_id"].astype(str)
    df["patient"] = df["image_id"].str.split("_").str[0]
    for c in QUALITY_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="raise").astype(int)
    if not df["Overall quality"].isin([0, 1]).all():
        raise ValueError(f"DeepDRiD {split} : qualité globale hors de {{0, 1}}")
    if df["image_id"].duplicated().any():
        raise ValueError(f"DeepDRiD {split} : identifiants d'images dupliqués")
    df["path"] = [base / "Images" / p / f"{i}.jpg" for p, i in zip(df["patient"], df["image_id"], strict=True)]
    return df[["image_id", "patient", "path", *QUALITY_COLUMNS]].reset_index(drop=True)

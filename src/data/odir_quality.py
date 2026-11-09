"""Annotations techniques originales d'ODIR, y compris les yeux exclus du CSV nettoyé.

Lecture limitée au tableau simple de la première feuille de data.xlsx, sans
dépendance Excel optionnelle. Les formules et colonnes manquantes sont refusées.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from src.data.odir_dataset import ODIRDataset

QUALITY_TAGS = (
    "low image quality",
    "lens dust",
    "image offset",
    "no fundus image",
    "anterior segment image",
    "optic disk photographically invisible",
)
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def read_quality_workbook(path: Path) -> list[dict[str, str]]:
    with zipfile.ZipFile(path) as archive:
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared = [
                "".join(t.text or "" for t in cell.findall(".//m:t", NS))
                for cell in ET.fromstring(archive.read("xl/sharedStrings.xml")).findall("m:si", NS)
            ]
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    rows = []
    for row in sheet.findall(".//m:sheetData/m:row", NS):
        values = {}
        for cell in row.findall("m:c", NS):
            if cell.find("m:f", NS) is not None:
                raise ValueError("ODIR : formules Excel non prises en charge")
            column = "".join(c for c in cell.attrib["r"] if c.isalpha())
            value = cell.findtext("m:v", default="", namespaces=NS)
            if cell.attrib.get("t") == "s":
                value = shared[int(value)]
            elif cell.attrib.get("t") == "inlineStr":
                value = "".join(t.text or "" for t in cell.findall(".//m:t", NS))
            values[column] = value
        rows.append(values)
    required = {"ID", *(f"{side}-{field}" for side in ("Left", "Right") for field in ("Fundus", "Diagnostic Keywords"))}
    if not rows or not required <= set(rows[0].values()):
        raise ValueError("ODIR : colonnes requises absentes de la première feuille Excel")
    header = rows[0]
    return [{header[c]: value for c, value in row.items() if c in header} for row in rows[1:]]


def source_records(root: Path) -> tuple[list[dict], int]:
    """Conserve les splits du projet ; aucun patient test n'entre dans la sélection."""
    assignment = {}
    for split in ("train", "val", "test"):
        for group in ODIRDataset(root, split=split).groups:
            if group in assignment and assignment[group] != split:
                raise ValueError("patient partagé entre découpages")
            assignment[group] = split
    records, excluded = [], 0
    table = read_quality_workbook(root / "ODIR-5K" / "ODIR-5K" / "data.xlsx")
    for row in table:
        group = str(int(row["ID"]))
        split = assignment.get(group)
        if split not in ("train", "val"):
            excluded += 1
            continue
        for side in ("Left", "Right"):
            filename = row.get(f"{side}-Fundus", "")
            words = row.get(f"{side}-Diagnostic Keywords", "").replace("，", ",").lower()
            if not filename or not words:
                raise ValueError("ODIR : image ou annotation technique manquante")
            if Path(filename).name != filename:
                raise ValueError("ODIR : nom d'image invalide")
            parts = {w.strip() for w in words.split(",")}
            records.append(
                {"filename": filename, "group": group, "split": split, "tags": [t for t in QUALITY_TAGS if t in parts]}
            )
    if len({r["filename"] for r in records}) != len(records):
        raise ValueError("ODIR : identifiants d'images dupliqués")
    return records, excluded


def select_records(records: list[dict], controls: int = 120, seed: int = 0) -> list[dict]:
    import numpy as np

    if controls < 1:
        raise ValueError("au moins un témoin requis")
    selected = []
    rng = np.random.default_rng(seed)
    for split in ("train", "val"):
        rows = sorted((r for r in records if r["split"] == split), key=lambda r: r["filename"])
        positive_groups = {r["group"] for r in rows if r["tags"]}
        # Un patient avec un défaut connu ne devient pas un témoin via son autre œil.
        negatives = [r for r in rows if not r["tags"] and r["group"] not in positive_groups]
        if not negatives:
            raise ValueError(f"{split} : aucun témoin sans mot-clé technique")
        selected.extend(r for r in rows if r["tags"])
        selected.extend(negatives[i] for i in rng.choice(len(negatives), min(controls, len(negatives)), replace=False))
    groups = {s: {r["group"] for r in selected if r["split"] == s} for s in ("train", "val")}
    if not groups["train"].isdisjoint(groups["val"]):
        raise ValueError("patients partagés entre calibration et validation")
    return selected

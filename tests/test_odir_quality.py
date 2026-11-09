"""Annotations qualité : préserver les patients et ne pas inventer de témoins."""

import zipfile
from xml.etree import ElementTree as ET

import pytest

from src.data.odir_quality import read_quality_workbook, select_records


def row(filename, group, split, tags=()):
    return {"filename": filename, "group": group, "split": split, "tags": list(tags)}


def test_other_eye_of_a_known_quality_case_is_not_a_control():
    records = [
        row("a_left.jpg", "a", "train", ["low image quality"]),
        row("a_right.jpg", "a", "train"),
        row("b.jpg", "b", "train"),
        row("c.jpg", "c", "val"),
        row("test.jpg", "d", "test", ["low image quality"]),
    ]
    selected = select_records(records)
    assert {r["filename"] for r in selected} == {"a_left.jpg", "b.jpg", "c.jpg"}


def test_patient_shared_between_calibration_and_validation_is_refused():
    with pytest.raises(ValueError, match="partagés"):
        select_records([row("a.jpg", "shared", "train"), row("b.jpg", "shared", "val")])


def test_incomplete_excel_table_is_refused(tmp_path):
    path = tmp_path / "source.xlsx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData/></worksheet>',
        )
    with pytest.raises(ValueError, match="colonnes"):
        read_quality_workbook(path)


@pytest.mark.parametrize("shared_strings", [False, True])
def test_workbook_preserves_the_eye_specific_quality_keyword(tmp_path, shared_strings):
    headers = ["ID", "Left-Fundus", "Right-Fundus", "Left-Diagnostic Keywords", "Right-Diagnostic Keywords"]
    values = ["12", "12_left.jpg", "12_right.jpg", "low image quality", "normal fundus"]
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    root = ET.Element("worksheet", xmlns=ns)
    data = ET.SubElement(root, "sheetData")
    for row_number, items in enumerate((headers, values), 1):
        row_element = ET.SubElement(data, "row")
        for column, value in enumerate(items):
            cell = ET.SubElement(
                row_element, "c", r=f"{chr(65 + column)}{row_number}", t="s" if shared_strings else "inlineStr"
            )
            if shared_strings:
                ET.SubElement(cell, "v").text = str((row_number - 1) * len(headers) + column)
            else:
                ET.SubElement(ET.SubElement(cell, "is"), "t").text = value
    path = tmp_path / "source.xlsx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("xl/worksheets/sheet1.xml", ET.tostring(root))
        if shared_strings:
            shared = ET.Element("sst", xmlns=ns)
            for value in headers + values:
                ET.SubElement(ET.SubElement(shared, "si"), "t").text = value
            archive.writestr("xl/sharedStrings.xml", ET.tostring(shared))
    result = read_quality_workbook(path)
    assert result == [dict(zip(headers, values, strict=True))]

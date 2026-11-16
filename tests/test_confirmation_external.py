from pathlib import Path

import pandas as pd
import pytest

from scripts.confirmation_external import idrid_table, messidor_table


def test_idrid_table_reads_both_official_splits(tmp_path):
    base = tmp_path / "B. Disease Grading"
    (base / "2. Groundtruths").mkdir(parents=True)
    for labels, names in (
        ("a. IDRiD_Disease Grading_Training Labels.csv", ["IDRiD_001"]),
        ("b. IDRiD_Disease Grading_Testing Labels.csv", ["IDRiD_001"]),
    ):
        pd.DataFrame({"Image name": names, "Retinopathy grade": [2], "Risk of macular edema ": [0]}).to_csv(
            base / "2. Groundtruths" / labels, index=False
        )
    table = idrid_table(tmp_path)
    assert len(table) == 2 and table["grade"].tolist() == [2, 2]
    assert table["key"].is_unique and table["group"].is_unique  # mêmes noms dans les deux découpages
    assert Path(table["path"][1]).as_posix().endswith("b. Testing Set/IDRiD_001.jpg")


def test_messidor_table_groups_by_exam_ignoring_case(tmp_path):
    (tmp_path / "IMAGES").mkdir()
    for name in ("a.JPG", "b.JPG", "c.png", "d.png"):
        (tmp_path / "IMAGES" / name).touch()
    pd.DataFrame(
        {
            "image_id": ["a.jpg", "b.jpg", "c.png", "d.png"],
            "adjudicated_dr_grade": [0, 2, 1, 0],
            "adjudicated_dme": [0, 0, 0, 0],
            "adjudicated_gradable": [1, 1, 1, 0],
        }
    ).to_csv(tmp_path / "messidor_data.csv", index=False)
    (tmp_path / "messidor-2.csv").write_text("left;right \na.JPG;b.JPG\nc.png;d.png\n")
    table = messidor_table(tmp_path)
    assert len(table) == 3 and table["grade"].tolist() == [0, 2, 1]
    assert table["group"].tolist() == ["messidor2/a.jpg", "messidor2/a.jpg", "messidor2/c.png"]
    assert table["path"][0].name == "a.JPG"


def test_image_format_reads_size_and_lowercase_suffix(tmp_path):
    from PIL import Image

    from scripts.messidor_format_check import image_format

    Image.new("RGB", (30, 20)).save(tmp_path / "x.JPG", format="JPEG")
    assert image_format(tmp_path / "x.JPG") == "30x20.jpg"


def test_messidor_table_refuses_ambiguous_file_names(tmp_path):
    for sub in ("IMAGES", "copie"):
        (tmp_path / sub).mkdir()
        (tmp_path / sub / "a.png").touch()
    pd.DataFrame(
        {"image_id": ["a.png"], "adjudicated_dr_grade": [0], "adjudicated_dme": [0], "adjudicated_gradable": [1]}
    ).to_csv(tmp_path / "messidor_data.csv", index=False)
    with pytest.raises(ValueError, match="ambigu"):
        messidor_table(tmp_path)


def test_images_list_fingerprint_detects_permutation():
    from scripts.confirmation_external import file_fingerprint_text

    assert file_fingerprint_text("a\t1\nb\t2") != file_fingerprint_text("b\t2\na\t1")
    assert len(file_fingerprint_text("a\t1")) == 16

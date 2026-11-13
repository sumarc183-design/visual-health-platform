import pandas as pd

from scripts.confirmation_external import idrid_table


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
    assert str(table["path"][1]).endswith("b. Testing Set/IDRiD_001.jpg")

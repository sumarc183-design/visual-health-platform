from src.data.ddr_dataset import capture_time, read_grades, time_groups


def test_capture_time_reads_file_name():
    assert capture_time("20170413102628830.jpg").strftime("%Y-%m-%d %H:%M:%S") == "2017-04-13 10:26:28"
    assert capture_time("image_a.jpg") is None


def test_time_groups_chain_close_shots_only():
    names = ["20170413100000000.jpg", "20170413100100000.jpg", "20170413101000000.jpg", "x.jpg"]
    g = time_groups(names, gap_seconds=120)
    assert g[0] == g[1]
    assert g[2] != g[1]
    assert g[3].startswith("solo/")


def test_read_grades_rejects_duplicates(tmp_path):
    d = tmp_path / "DR_grading"
    d.mkdir()
    (d / "train.txt").write_text("a.jpg 0\nb.jpg 5\n")
    (d / "valid.txt").write_text("c.jpg 2\n")
    (d / "test.txt").write_text("a.jpg 1\n")
    try:
        read_grades(tmp_path)
    except ValueError:
        return
    raise AssertionError("doublon non détecté")

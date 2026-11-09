"""Aucune donnée individuelle APTOS/EyePACS (règlements Kaggle) dans les fichiers suivis.

Vérifie le contenu, pas seulement le nom : identifiants APTOS (12 caractères
hexadécimaux), images EyePACS (`<n>_left`/`<n>_right`), clés `aptos/…`/`eyepacs/…`
et tableaux NPZ nommés d'après ces bases.
"""

import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
APTOS_ID = re.compile(r'"(?:aptos/)?[0-9a-f]{12}"')
EYEPACS_ID = re.compile(r'"(?:eyepacs/)?[0-9]+_(?:left|right)"')


def tracked_results() -> list[Path]:
    try:
        out = subprocess.run(["git", "ls-files", "results"], cwd=ROOT, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git indisponible")
    return [ROOT / f for f in out.stdout.split()]


def restricted_in_npz(path: Path) -> list[str]:
    found = []
    with np.load(path, allow_pickle=False) as data:
        for name in data.files:
            if re.search(r"aptos|eyepacs", name, re.I):
                found.append(name)
            elif data[name].dtype.kind in "US" and any(
                str(v).startswith(("aptos/", "eyepacs/")) for v in data[name].ravel()[:1000]
            ):
                found.append(name)
    return found


def test_no_aptos_or_eyepacs_individual_data_is_tracked():
    problems = []
    for path in tracked_results():
        if path.suffix == ".json":
            text = path.read_text(encoding="utf-8")
            if APTOS_ID.search(text) or EYEPACS_ID.search(text):
                problems.append(path.name)
        elif path.suffix == ".npz":
            if restricted_in_npz(path):
                problems.append(path.name)
    assert not problems, f"données individuelles APTOS/EyePACS suivies : {problems}"


def test_detector_catches_a_restricted_archive(tmp_path):
    np.savez(tmp_path / "x.npz", aptos_keys=np.array(["aptos/0a1b2c3d4e5f"]), other=np.zeros(2))
    np.savez(tmp_path / "y.npz", keys=np.array(["eyepacs/10_left"]))
    np.savez(tmp_path / "z.npz", keys=np.array(["rfmid/1"]))
    assert restricted_in_npz(tmp_path / "x.npz") == ["aptos_keys"]
    assert restricted_in_npz(tmp_path / "y.npz") == ["keys"]
    assert restricted_in_npz(tmp_path / "z.npz") == []

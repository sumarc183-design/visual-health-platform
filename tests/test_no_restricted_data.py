"""Aucune donnée individuelle APTOS/EyePACS (règlements Kaggle) dans les fichiers suivis.

Vérifie le contenu, pas seulement le nom : identifiants APTOS (12 caractères
hexadécimaux), images EyePACS (`<n>_left`/`<n>_right`), clés `aptos/…`/`eyepacs/…`
et tableaux NPZ nommés d'après ces bases.

Portée (audit du 15 nov., constat 9) : JSON et NPZ de `results/` (tableaux de chaînes lus
en entier, par blocs) ; tous les autres fichiers texte suivis (CSV, TXT, Markdown,
notebooks, code…) pour les clés `aptos/…`/`eyepacs/…` et les identifiants APTOS. Exclus :
`tests/` (identifiants synthétiques des tests) ; le motif `<n>_left` hors `results/` JSON,
car ODIR nomme ses images de la même façon. L'historique Git n'est pas inspecté.
"""

import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
APTOS_ID = re.compile(r'"(?:aptos/)?[0-9a-f]{12}"')
EYEPACS_ID = re.compile(r'"(?:eyepacs/)?[0-9]+_(?:left|right)"')


TEXT_SUFFIXES = {".json", ".csv", ".tsv", ".txt", ".md", ".ipynb", ".py", ".yaml", ".yml", ".cff", ".toml", ".sh"}
PREFIXED_KEY = re.compile(r"(?:aptos|eyepacs)/[0-9A-Za-z_]+")
BARE_APTOS_ID = re.compile(r"(?<![0-9a-zA-Z_/.-])[0-9a-f]{12}(?![0-9a-zA-Z_])")
CHUNK = 100_000


def tracked(*paths: str) -> list[Path]:
    try:
        out = subprocess.run(["git", "ls-files", *paths], cwd=ROOT, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git indisponible")
    return [ROOT / f for f in out.stdout.splitlines()]


def tracked_results() -> list[Path]:
    return tracked("results")


def restricted_in_text(text: str) -> list[str]:
    """Clés préfixées et identifiants APTOS nus (12 hexadécimaux isolés, avec au moins une lettre)."""
    hits = PREFIXED_KEY.findall(text)
    hits += [h for h in BARE_APTOS_ID.findall(text) if re.search("[a-f]", h) and re.search("[0-9]", h)]
    return hits


def restricted_in_npz(path: Path) -> list[str]:
    found = []
    with np.load(path, allow_pickle=False) as data:
        for name in data.files:
            if re.search(r"aptos|eyepacs", name, re.I):
                found.append(name)
            elif data[name].dtype.kind in "US":
                values = data[name].ravel()
                for start in range(0, len(values), CHUNK):  # tableau lu en entier
                    if any(str(v).startswith(("aptos/", "eyepacs/")) for v in values[start : start + CHUNK]):
                        found.append(name)
                        break
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


def test_no_restricted_identifier_in_other_tracked_text_files():
    problems = {}
    for path in tracked():
        rel = path.relative_to(ROOT)
        if rel.parts[0] == "tests" or path.suffix not in TEXT_SUFFIXES or not path.exists():
            continue
        hits = restricted_in_text(path.read_text(encoding="utf-8", errors="replace"))
        if hits:
            problems[str(rel)] = hits[:3]
    assert not problems, f"identifiants APTOS/EyePACS dans des fichiers suivis : {problems}"


def test_detector_reads_whole_arrays_and_text(tmp_path):
    keys = np.array([f"rfmid/{i}" for i in range(1500)] + ["eyepacs/10_left"])
    np.savez(tmp_path / "long.npz", keys=keys)
    assert restricted_in_npz(tmp_path / "long.npz") == ["keys"]  # au-delà du rang 1 000
    assert restricted_in_text("image,grade\naptos/0a1b2c3d4e5f,2\n")
    assert restricted_in_text("id;grade\n0a1b2c3d4e5f;2\n")
    assert not restricted_in_text("commit 8d5cd74 ; empreinte 431de5dbc1bebbb3 ; 3401_left.jpg")

"""Manifeste d'exécution des notebooks : empreintes des fichiers, refus des cas invalides."""

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_manifest.py"


def run(archive, *extra):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--archive", str(archive), *extra], capture_output=True, text=True
    )


def test_manifest_lists_each_file_with_its_hash(tmp_path):
    archive = tmp_path / "out.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("embeddings/a.npz", b"abc")
    assert run(archive, "--requested-revision", "main").returncode == 0
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read("run_manifest.json"))
    assert manifest["requested_revision"] == "main"
    assert manifest["files"]["embeddings/a.npz"]["sha256"] == hashlib.sha256(b"abc").hexdigest()
    assert "python" in manifest["versions"]
    assert run(archive).returncode != 0  # pas de second manifeste


def test_missing_or_empty_archive_fails(tmp_path):
    assert run(tmp_path / "absent.zip").returncode != 0
    with zipfile.ZipFile(tmp_path / "vide.zip", "w"):
        pass
    assert run(tmp_path / "vide.zip").returncode != 0

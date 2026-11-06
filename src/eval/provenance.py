"""État du code et versions d'un run, à enregistrer dans ses résultats."""

from __future__ import annotations

import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PACKAGES = ("numpy", "scikit-learn", "torch", "timm")


def git_commit() -> str | None:
    """Commit du code exécuté (+ « -dirty » si des fichiers suivis sont modifiés)."""
    repo = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return commit + ("-dirty" if dirty else "")


def run_info() -> dict:
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return {"git_commit": git_commit(), "python": platform.python_version(), "versions": versions}

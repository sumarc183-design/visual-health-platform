#!/usr/bin/env python3
"""Ajoute `run_manifest.json` dans le zip de sortie d'un notebook (audit du 9 nov., constat 11).

Le manifeste relie les fichiers produits à leur exécution : commit réellement utilisé
(et fichiers suivis modifiés éventuels), révision demandée, plateforme, GPU, versions
des bibliothèques, empreinte SHA-256 et taille de chaque fichier du zip. Échec explicite
(code de retour non nul) si le zip est absent, vide ou déjà muni d'un manifeste.

Exemple (dernière cellule d'un notebook) :
    subprocess.run(["python", "scripts/run_manifest.py", "--archive", archive,
                    "--requested-revision", REVISION], check=True)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

NAME = "run_manifest.json"
PACKAGES = ("numpy", "torch", "timm", "scikit-learn", "pandas", "Pillow", "opencv-python-headless")


def git_state() -> dict:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()

    try:
        return {
            "commit": run("rev-parse", "HEAD"),
            "modified_tracked_files": run("status", "--porcelain", "-uno").splitlines(),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "modified_tracked_files": None}


def versions() -> dict:
    out = {"python": platform.python_version()}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return out


def gpu() -> str | None:
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        return None


def build_manifest(archive: Path, requested_revision: str | None) -> dict:
    with zipfile.ZipFile(archive) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
        if not names:
            raise SystemExit(f"{archive} est vide : aucun résultat à décrire")
        if NAME in names:
            raise SystemExit(f"{archive} contient déjà {NAME}")
        files = {n: {"sha256": hashlib.sha256(z.read(n)).hexdigest(), "bytes": z.getinfo(n).file_size} for n in names}
    platform_name = "kaggle" if os.path.exists("/kaggle") else "colab" if "COLAB_RELEASE_TAG" in os.environ else "local"
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "requested_revision": requested_revision,
        "git": git_state(),
        "platform": platform_name,
        "gpu": gpu(),
        "versions": versions(),
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--requested-revision", default=None)
    args = parser.parse_args()
    if not args.archive.is_file():
        raise SystemExit(f"zip absent : {args.archive}")
    manifest = build_manifest(args.archive, args.requested_revision)
    with zipfile.ZipFile(args.archive, "a") as z:
        z.writestr(NAME, json.dumps(manifest, indent=2, ensure_ascii=False))
    commit = manifest["git"]["commit"]
    print(
        f"{NAME} ajouté : {len(manifest['files'])} fichiers, commit {commit[:7] if commit else 'inconnu'}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

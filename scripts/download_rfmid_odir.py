"""Récupère RFMiD et ODIR-5K et les range dans data/raw/{rfmid,odir}/.

Par défaut, téléchargement anonyme depuis Kaggle (kagglehub, comme le
notebook Colab), puis lien symbolique vers le cache kagglehub : pas de
copie de ~10 Go. `--copy` copie à la place (disque séparé, pas de liens).
`--rfmid-from` / `--odir-from` rangent une archive déjà extraite à la main
(copiée, jamais déplacée : la source reste intacte).

Les dossiers obtenus ont la structure attendue par src/data/rfmid_dataset.py
et src/data/odir_dataset.py, et activent tests/test_real_datasets.py.

Exemple :
    pip install kagglehub
    python scripts/download_rfmid_odir.py --out data/raw
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

KAGGLE_HANDLES = {
    "rfmid": "andrewmvd/retinal-disease-classification",
    "odir": "andrewmvd/ocular-disease-recognition-odir5k",
}


def kaggle_download(handle: str) -> Path:
    try:
        import kagglehub
    except ImportError as error:
        raise SystemExit("kagglehub manquant : pip install kagglehub") from error
    return Path(kagglehub.dataset_download(handle))


def place(source: Path, dest: Path, copy: bool) -> None:
    if dest.exists() or dest.is_symlink():
        print(f"{dest} existe déjà, laissé tel quel")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copytree(source, dest)
        print(f"{dest} : copie de {source}")
    else:
        dest.symlink_to(source.resolve(), target_is_directory=True)
        print(f"{dest} -> {source}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data/raw"), help="Dossier de sortie (défaut data/raw)")
    parser.add_argument("--datasets", nargs="+", choices=list(KAGGLE_HANDLES), default=list(KAGGLE_HANDLES))
    parser.add_argument("--copy", action="store_true", help="Copier au lieu de créer un lien symbolique")
    parser.add_argument("--rfmid-from", type=Path, default=None, help="Archive RFMiD déjà extraite (pas de Kaggle)")
    parser.add_argument("--odir-from", type=Path, default=None, help="Archive ODIR déjà extraite (pas de Kaggle)")
    args = parser.parse_args()

    manual = {"rfmid": args.rfmid_from, "odir": args.odir_from}
    for name in args.datasets:
        source = manual[name]
        if source is None:
            print(f"Téléchargement {name} ({KAGGLE_HANDLES[name]})...")
            source = kaggle_download(KAGGLE_HANDLES[name])
            place(source, args.out / name, args.copy)
        else:
            place(source, args.out / name, copy=True)


if __name__ == "__main__":
    main()

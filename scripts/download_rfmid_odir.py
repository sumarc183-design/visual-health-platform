"""Instructions/scaffold pour récupérer RFMiD et ODIR.

Ces deux bases sont distribuées via des plateformes de challenge (pas un
simple lien direct) : téléchargement à finaliser manuellement une fois les
comptes créés, ce script structure ensuite les données téléchargées.

RFMiD : https://riadd.grand-challenge.org/  (ou Kaggle "RFMiD")
ODIR  : https://odir2019.grandchallenge.org/  (ou Kaggle "ocular-disease-recognition-odir5k")
"""
import argparse
import shutil
from pathlib import Path


def organize_dataset(source_dir: Path, dest_dir: Path, dataset_name: str) -> None:
    dest = dest_dir / dataset_name
    dest.mkdir(parents=True, exist_ok=True)
    for item in source_dir.iterdir():
        shutil.move(str(item), str(dest / item.name))
    print(f"{dataset_name} organisé dans {dest}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Répertoire de sortie data/raw")
    parser.add_argument(
        "--rfmid-download",
        help="Répertoire où le zip RFMiD téléchargé manuellement a été extrait",
    )
    parser.add_argument(
        "--odir-download",
        help="Répertoire où le zip ODIR téléchargé manuellement a été extrait",
    )
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.rfmid_download:
        organize_dataset(Path(args.rfmid_download), out, "rfmid")
    else:
        print("--rfmid-download non fourni : télécharger RFMiD manuellement d'abord "
              "(voir docstring de ce script)")

    if args.odir_download:
        organize_dataset(Path(args.odir_download), out, "odir")
    else:
        print("--odir-download non fourni : télécharger ODIR manuellement d'abord "
              "(voir docstring de ce script)")


if __name__ == "__main__":
    main()

"""Télécharge un sous-ensemble ISIC via l'API officielle (isic-cli).

pip install isic-cli
"""

import argparse
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Répertoire de sortie")
    parser.add_argument("--limit", type=int, default=1000, help="Nombre d'images")
    parser.add_argument(
        "--collection",
        default="",
        help="ID de collection ISIC spécifique (optionnel, sinon échantillon global)",
    )
    args = parser.parse_args()

    cmd = ["isic", "image", "download", args.out, "--limit", str(args.limit)]
    if args.collection:
        cmd += ["--search", f"collections:{args.collection}"]

    print("Commande :", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

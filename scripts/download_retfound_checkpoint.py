"""Télécharge un checkpoint RETFound officiel depuis HuggingFace
(YukunZhou/RETFound_*, dépôts gated -- nécessite un compte HF + un token
avec les conditions d'utilisation acceptées sur la page du modèle).

huggingface.co est hors de la liste d'hôtes autorisés de certains sandboxs
de développement (pare-feu d'entreprise ou politique d'environnement cloud) :
si `huggingface-cli login` ou ce script échouent avec une erreur réseau/403
depuis un tel environnement, lancez-les depuis un poste avec accès internet
normal.

Étapes préalables (une seule fois) :
  1. Créer un compte sur https://huggingface.co et un token sur
     https://huggingface.co/settings/tokens
  2. Visiter la page du modèle voulu (ex. la CFP tabulée dans le README de
     RETFound_MAE : https://huggingface.co/YukunZhou/RETFound_mae_natureCFP)
     et accepter les conditions d'accès (dépôt "gated")
  3. pip install huggingface_hub
  4. huggingface-cli login --token VOTRE_TOKEN

Exemple :
    python scripts/download_retfound_checkpoint.py \\
        --repo-id YukunZhou/RETFound_mae_natureCFP \\
        --out third_party/retfound/checkpoints
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

WEIGHT_EXTENSIONS = (".pth", ".pt", ".bin", ".safetensors")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--repo-id", default="YukunZhou/RETFound_mae_natureCFP",
        help="Dépôt HuggingFace du checkpoint. Seules les variantes MAE (ViT-L/16) sont compatibles "
        "avec src/models/retfound_backbone.py : RETFound_mae_natureCFP, RETFound_mae_natureOCT, "
        "RETFound_mae_meh, RETFound_mae_shanghai. Les variantes dinov2 (ViT-L/14) ne le sont pas.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Dossier de destination")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    snapshot_dir = Path(snapshot_download(repo_id=args.repo_id))

    weight_files = [p for p in snapshot_dir.rglob("*") if p.suffix in WEIGHT_EXTENSIONS]
    if not weight_files:
        raise RuntimeError(
            f"Aucun fichier de poids ({WEIGHT_EXTENSIONS}) trouvé dans le snapshot de {args.repo_id} "
            f"({snapshot_dir}). Vérifier manuellement le contenu du dépôt HuggingFace."
        )

    for weight_file in weight_files:
        dest = args.out / weight_file.name
        shutil.copy2(weight_file, dest)
        print(f"Copié : {dest}")

    print(f"\nTerminé. Utiliser --checkpoint {args.out / weight_files[0].name} avec run_baseline.py.")


if __name__ == "__main__":
    main()

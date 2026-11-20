#!/usr/bin/env python3
"""Embeddings DINOv2-B « qualité » de toutes les images de gradation DDR, non gradables comprises (CPU).

Sert la validation du détecteur de qualité (`scripts/quality_ddr_validation.py`, plan
fixé le 20 novembre 2026). Préparation identique au détecteur du 10 novembre
(`scripts/quality_learned_detector.py` : `crop-mask` à 512 px, réduction par aire à
224 px, normalisation ImageNet, DINOv2 ViT-B/14 figé). Aucune mesure ici.

Les poids DINOv2-B doivent avoir exactement les mêmes valeurs que ceux du détecteur :
l'empreinte des tenseurs (indépendante de la sérialisation du fichier) est vérifiée
avant tout calcul. Le fichier local du projet (`ba7954cb…`) et celui retéléchargé par timm
(`ed724e34…`) ont la même empreinte de tenseurs, `df4a90fee93f84ed`.

Sorties : cache `embeddings/quality_ddr_dinov2_vitb14.npz` (clés `ddr/<découpage>/<image>`)
et `embeddings/quality_ddr_labels.csv` (clé, découpage, stade, non gradable, groupe
horaire approchant le patient).

Exemple (Kaggle) :
    python scripts/quality_ddr_embeddings.py --ddr-root /kaggle/input/.../DDR-dataset \\
        --checkpoint third_party/dinov2/dinov2_vitb14_timm.pth
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.quality_learned_detector import SIZE, TRANSFORM, embed_keys, retina  # noqa: E402
from src.data.ddr_dataset import UNGRADABLE, read_grades, time_groups  # noqa: E402
from src.models.embedding_cache import EmbeddingCache  # noqa: E402
from src.models.timm_backbones import DINOV2_IMAGE_SIZE, DINOv2Backbone  # noqa: E402

EXPECTED_TENSOR_DIGEST = "df4a90fee93f84ed"


def tensor_digest(path: str | Path) -> str:
    """Empreinte des valeurs des poids (clés triées, formes, float32) : deux fichiers de
    sérialisation différente mais de mêmes tenseurs ont la même empreinte."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state)
    h = hashlib.sha256()
    for key in sorted(state):
        t = state[key].detach().contiguous().to(torch.float32).numpy()
        h.update(key.encode())
        h.update(str(t.shape).encode())
        h.update(t.tobytes())
    return h.hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ddr-root", type=Path, required=True, help="dossier contenant DR_grading/")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("embeddings/quality_ddr_dinov2_vitb14.npz"))
    parser.add_argument("--labels", type=Path, default=Path("embeddings/quality_ddr_labels.csv"))
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    digest = tensor_digest(args.checkpoint)
    if digest != EXPECTED_TENSOR_DIGEST:
        raise SystemExit(f"poids DINOv2-B différents de ceux du détecteur ({digest}) : arrêt avant tout calcul")

    df = read_grades(args.ddr_root)
    df["key"] = "ddr/" + df["split"] + "/" + df["image"]
    paths = {
        k: args.ddr_root / "DR_grading" / s / i for k, s, i in zip(df["key"], df["split"], df["image"], strict=True)
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images DDR absentes, ex. {missing[:3]}")
    df["ungradable"] = (df["level"] == UNGRADABLE).astype(int)
    df["group"] = time_groups(list(df["image"]))
    args.labels.parent.mkdir(parents=True, exist_ok=True)
    df[["key", "split", "level", "ungradable", "group"]].to_csv(args.labels, index=False)
    print(f"{len(df)} images, dont {int(df['ungradable'].sum())} non gradables", flush=True)

    meta = {"checkpoint_tensor_sha256": digest, "image_size": DINOV2_IMAGE_SIZE, "transform": TRANSFORM}
    cache = EmbeddingCache(args.out, meta=meta)
    backbone = DINOv2Backbone(args.checkpoint)
    X = embed_keys(list(df["key"]), lambda k: retina(paths[k]), backbone, cache)
    if X.shape != (len(df), 768) or not torch.isfinite(torch.from_numpy(X)).all():
        raise ValueError(f"embeddings inattendus : forme {X.shape}")
    print(f"-> {args.out} ({X.shape[0]} embeddings, crop-mask {SIZE} px puis {DINOV2_IMAGE_SIZE} px)")
    assert pd.read_csv(args.labels).shape[0] == X.shape[0]


if __name__ == "__main__":
    main()

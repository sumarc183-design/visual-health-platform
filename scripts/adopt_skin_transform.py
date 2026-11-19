#!/usr/bin/env python3
"""Adoption explicite de la transformation peau par les caches ISIC 2019 historiques.

Constat D de l'audit du 19 novembre 2026 : le chargeur commun ne vérifiait pas la
transformation des embeddings peau (`SKIN_TRANSFORM`). Il l'exige désormais, sans
adoption implicite. Les trois caches ISIC 2019 (DINOv2-B, DINOv2-L, PanDerm), calculés
avant que ce champ existe, sont donc refusés tant qu'ils ne l'ont pas adopté ici.

Pour chaque cache sans champ `transform` :
- poids présents : recalcul des N premières images
  d'apprentissage avec la chaîne actuelle (`skin_tensor` + normalisation du modèle) et
  comparaison aux embeddings en cache ; adoption seulement si le cosinus minimal est
  >= 0,9999. Note `transform_sample_check` dans les métadonnées ;
- poids absents : refus, sauf `--declare-without-check` (note
  `transform_declared_without_check`).

Dans les deux cas, `transform` est ajouté à `assumed_not_verified` : un échantillon ne
prouve pas l'origine de tous les embeddings. Clés, embeddings et empreintes des sources
sont inchangés.

Exemple :
    python scripts/adopt_skin_transform.py --isic2019-root data/raw/isic2019 \\
        --dinov2 third_party/dinov2/dinov2_vitb14_timm.pth --panderm third_party/panderm/panderm_large.pth
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_modality_router_v2 import Item, embed  # noqa: E402
from src.data.isic_dataset import SKIN_TRANSFORM, ISIC2019Dataset  # noqa: E402
from src.models.embedding_cache import (  # noqa: E402
    ASSUMED_KEY,
    CHECKPOINT_STEMS,
    EmbeddingCache,
    file_fingerprint,
)
from src.models.timm_backbones import DINOv2Backbone, DINOv2LargeBackbone, PanDermBackbone  # noqa: E402

MIN_COSINE = 0.9999
BACKBONES = {"dinov2": DINOv2Backbone, "dinov2-large": DINOv2LargeBackbone, "panderm": PanDermBackbone}


def sample_check(cache: EmbeddingCache, backbone, root: Path, n: int) -> dict:
    dataset = ISIC2019Dataset(root, "train")
    keys = dataset.image_keys[:n]
    paths = [dataset.image_path(i) for i in range(n)]
    items = [Item(k, k, lambda p=p: Image.open(p)) for k, p in zip(keys, paths, strict=True)]
    scratch = EmbeddingCache(cache.path.with_name("_adopt_skin_transform.npz"))
    try:
        X = embed(items, backbone, scratch)
    finally:
        scratch.path.unlink(missing_ok=True)
    ref = cache.get(keys)
    cosine = (X * ref).sum(1) / (np.linalg.norm(X, axis=1) * np.linalg.norm(ref, axis=1))
    return {"images": n, "min_cosine": float(cosine.min()), "max_abs_diff": float(np.abs(X - ref).max())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--isic2019-root", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, default=Path("embeddings_derm"))
    parser.add_argument("--dinov2", type=Path, help="poids DINOv2-B")
    parser.add_argument("--dinov2-large", type=Path, help="poids DINOv2-L")
    parser.add_argument("--panderm", type=Path, help="poids PanDerm-Large")
    parser.add_argument("--images", type=int, default=16)
    parser.add_argument("--declare-without-check", action="store_true", help="adopter sans poids (déclaration)")
    args = parser.parse_args()
    weights = {"dinov2": args.dinov2, "dinov2-large": args.dinov2_large, "panderm": args.panderm}

    report, refused = {}, []
    for name, stem in CHECKPOINT_STEMS.items():
        if name not in BACKBONES:
            continue
        path = args.embeddings / f"isic2019_none_224_{stem}.npz"
        if not path.exists():
            continue
        cache = EmbeddingCache(path)  # sans contrat demandé : lecture seule des métadonnées
        meta = dict(cache.meta or {})
        if meta.get("transform") is not None:
            report[path.name] = {"status": "déjà présent", "transform": meta["transform"]}
            continue
        w = weights[name]
        if w is not None and w.exists():
            check = sample_check(cache, BACKBONES[name](w), args.isic2019_root, args.images)
            fp = file_fingerprint(w)
            if fp != meta.get("checkpoint_sha256"):
                # Autre sérialisation des poids (DINOv2-B : Kaggle 388c…, local ba79…) : le
                # recalcul ci-dessus juge l'équivalence ; l'empreinte du cache est gardée.
                check["local_checkpoint_sha256"] = fp
            if check["min_cosine"] < MIN_COSINE:
                refused.append(path.name)
                report[path.name] = {"status": "refusé", "transform_sample_check": check}
                continue
            meta["transform_sample_check"] = check
        elif args.declare_without_check:
            meta["transform_declared_without_check"] = True
        else:
            refused.append(path.name)
            report[path.name] = {"status": "refusé (poids absents)"}
            continue
        meta["transform"] = SKIN_TRANSFORM
        meta[ASSUMED_KEY] = sorted(set(meta.get(ASSUMED_KEY, [])) | {"transform"})
        cache.meta = meta
        cache.save()
        report[path.name] = {"status": "adopté", **{k: meta[k] for k in meta if k.startswith("transform")}}
    print(json.dumps(report, indent=1, ensure_ascii=False))
    if refused:
        raise SystemExit(f"non adoptés : {refused}")


if __name__ == "__main__":
    main()

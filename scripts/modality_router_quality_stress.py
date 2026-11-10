#!/usr/bin/env python3
"""Routeur de modalité face à des fonds d'œil de mauvaise qualité (test de résistance, CPU).

Plan fixé le 10 novembre 2026, avant ce calcul : les têtes des protocoles 1 et 2
(`heads/modality_router.pt`, `heads/modality_router_v2.pt`), inchangées et sans
entraînement, sont appliquées aux 800 photos de validation et d'évaluation de DeepDRiD
(CC BY-SA 4.0), séparées selon la qualité notée par ses experts (suffisante ou
insuffisante pour le diagnostic). Mêmes embeddings et même transformation que le routeur
(DINOv2 ViT-B/14, petit côté 256, recadrage 224). Critères repris du protocole : au moins
95 % de fonds d'œil reconnus, aucun envoyé en dermoscopie. Les images insuffisantes
peuvent légitimement être classées « inconnu » ; elles ne doivent pas aller en dermoscopie.

Exemple :
    python scripts/modality_router_quality_stress.py --deepdrid-root data/raw/deepdrid/regular_fundus_images \\
        --checkpoint third_party/dinov2/dinov2_vitb14_timm.pth
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_modality_router import exact_ci  # noqa: E402
from scripts.train_modality_router_v2 import Item, embed  # noqa: E402
from src.data.deepdrid import load_split  # noqa: E402
from src.eval.provenance import run_info  # noqa: E402
from src.models.embedding_cache import EmbeddingCache, file_fingerprint  # noqa: E402
from src.models.timm_backbones import DINOV2_IMAGE_SIZE, DINOv2Backbone  # noqa: E402
from src.pipeline.modality_classifier import MODALITIES, ModalityHead  # noqa: E402

HEADS = {"protocol_1": Path("heads/modality_router.pt"), "protocol_2": Path("heads/modality_router_v2.pt")}


def summary(decided: list[str], probs: np.ndarray) -> dict:
    n = len(decided)
    derm = decided.count("dermoscopy")
    fundus = decided.count("fundus")
    return {
        "n": n,
        "decisions": {m: decided.count(m) for m in MODALITIES},
        "fundus_recall": {"value": fundus / n, "ci": exact_ci(fundus, n)},
        "to_dermoscopy": {"count": derm, "value": derm / n, "ci": exact_ci(derm, n)},
        "max_dermoscopy_probability": float(probs[:, MODALITIES.index("dermoscopy")].max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--deepdrid-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("embeddings/modality_dinov2_vitb14.npz"))
    parser.add_argument("--out", type=Path, default=Path("results/modality_router_quality_stress.json"))
    args = parser.parse_args()

    frames = {s: load_split(args.deepdrid_root, s) for s in ("val", "eval")}
    items = [
        Item(f"deepdrid/{s}/{i}", p_id, lambda path=path: Image.open(path))
        for s, d in frames.items()
        for i, p_id, path in zip(d["image_id"], d["patient"], d["path"], strict=True)
    ]
    insufficient = np.concatenate([(d["Overall quality"] == 0).to_numpy() for d in frames.values()])
    fingerprint = file_fingerprint(args.checkpoint)
    cache = EmbeddingCache(args.cache, meta={"checkpoint_sha256": fingerprint, "image_size": DINOV2_IMAGE_SIZE})
    X = embed(items, DINOv2Backbone(args.checkpoint), cache)

    report = {
        "run": run_info(),
        "plan": "fixé le 10 novembre 2026 avant ce calcul (docstring du script)",
        "images": {"sufficient": int((~insufficient).sum()), "insufficient": int(insufficient.sum())},
        "heads": {},
    }
    for name, path in HEADS.items():
        head, _ = ModalityHead.load(path)
        probs = head.probabilities(X)
        decided = np.array(head.decide(probs))
        report["heads"][name] = {
            "all": summary(list(decided), probs),
            "sufficient": summary(list(decided[~insufficient]), probs[~insufficient]),
            "insufficient": summary(list(decided[insufficient]), probs[insufficient]),
        }
        report["heads"][name]["criteria"] = {
            "fundus_recall_ge_0.95_sufficient": report["heads"][name]["sufficient"]["fundus_recall"]["value"] >= 0.95,
            "no_fundus_to_dermoscopy": report["heads"][name]["all"]["to_dermoscopy"]["count"] == 0,
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    for name, r in report["heads"].items():
        print(
            name,
            "suffisante",
            r["sufficient"]["decisions"],
            "insuffisante",
            r["insufficient"]["decisions"],
            r["criteria"],
        )
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()

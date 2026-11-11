#!/usr/bin/env python3
"""Routeur de modalité face aux fonds d'œil pathologiques de RFMiD (test de résistance, CPU).

Plan fixé le 11 novembre 2026, avant ce calcul. Question : l'erreur résiduelle des
protocoles 1 et 2 (RFMiD test 443, rétine très pathologique envoyée en dermoscopie)
est-elle un cas isolé, ou se concentre-t-elle dans certaines maladies ?

- têtes des protocoles 1 et 2 (`heads/modality_router.pt`, `heads/modality_router_v2.pt`)
  inchangées, sans entraînement ni changement de seuil (0,9) ;
- images : les 3 200 photos de RFMiD (entraînement 1 920, validation 640, test 640).
  Le routeur n'a jamais été ajusté sur RFMiD (ajustement sur ODIR, ISIC, Imagenette,
  MedMNIST) ; 500 images du test ont déjà servi à son évaluation et sont incluses ;
- mêmes embeddings et même transformation que le routeur (DINOv2 ViT-B/14, image brute,
  petit côté 256, recadrage 224) ;
- mesures, par tête : décisions (fond d'œil, dermoscopie, inconnu) sur l'ensemble, selon
  `Disease_Risk` (0 ou 1) et selon le nombre de pathologies annotées (0, 1, 2, 3 ou plus) ;
  taux envoyé en dermoscopie avec IC exact à 95 % ; pour chaque pathologie annotée sur au
  moins une image envoyée en dermoscopie, le nombre d'envois sur le nombre d'images
  porteuses ; liste des images envoyées en dermoscopie et leur probabilité ;
- critère fixé : erreur « rare » si la borne haute de l'IC à 95 % du taux d'envoi en
  dermoscopie est au plus 0,5 % parmi les images à `Disease_Risk` = 1, pour chaque tête ;
  sinon, le routeur ne peut pas servir de garde-fou bloquant sur les rétines malades.

Réserves : une seule source (Inde, trois caméras) ; les pathologies rares comptent peu
d'images, leurs taux restent indicatifs ; pas de correction pour comparaisons multiples
(lecture descriptive).

Exemple :
    python scripts/modality_router_pathology_stress.py --rfmid-root data/raw/rfmid \\
        --checkpoint third_party/dinov2/dinov2_vitb14_timm.pth
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_modality_router import exact_ci  # noqa: E402
from scripts.train_modality_router_v2 import Item, embed  # noqa: E402
from src.data.rfmid_dataset import RFMiDDataset  # noqa: E402
from src.eval.provenance import run_info  # noqa: E402
from src.models.embedding_cache import EmbeddingCache, file_fingerprint  # noqa: E402
from src.models.timm_backbones import DINOV2_IMAGE_SIZE, DINOv2Backbone  # noqa: E402
from src.pipeline.modality_classifier import MODALITIES, ModalityHead  # noqa: E402

HEADS = {"protocol_1": Path("heads/modality_router.pt"), "protocol_2": Path("heads/modality_router_v2.pt")}
SPLITS = ("train", "val", "test")
RARE_UPPER_BOUND = 0.005


def burden_bin(n_labels: int) -> str:
    """Classe du nombre de pathologies annotées : 0, 1, 2 ou 3+."""
    return str(n_labels) if n_labels < 3 else "3+"


def rates(decided: np.ndarray) -> dict:
    n = len(decided)
    derm = int((decided == "dermoscopy").sum())
    return {
        "n": n,
        "decisions": {m: int((decided == m).sum()) for m in MODALITIES},
        "to_dermoscopy": {"count": derm, "value": derm / n if n else None, "ci": exact_ci(derm, n) if n else None},
    }


def labels_frame(root: Path) -> pd.DataFrame:
    frames = []
    for split in SPLITS:
        ds = RFMiDDataset(root, split=split)
        df = ds.df.copy()
        df["key"] = ds.image_keys
        df["path"] = [ds.image_path(i) for i in range(len(ds))]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True).fillna(0)
    pathologies = [c for c in df.columns if c not in ("ID", "Disease_Risk", "key", "path")]
    df["n_labels"] = df[pathologies].sum(axis=1).astype(int)
    df.attrs["pathologies"] = pathologies
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rfmid-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("embeddings/modality_dinov2_vitb14.npz"))
    parser.add_argument("--out", type=Path, default=Path("results/modality_router_pathology_stress.json"))
    args = parser.parse_args()

    df = labels_frame(args.rfmid_root)
    pathologies = df.attrs["pathologies"]
    items = [
        Item(f"rfmid_all/{k}", k, lambda path=p: Image.open(path)) for k, p in zip(df["key"], df["path"], strict=True)
    ]
    fingerprint = file_fingerprint(args.checkpoint)
    cache = EmbeddingCache(args.cache, meta={"checkpoint_sha256": fingerprint, "image_size": DINOV2_IMAGE_SIZE})
    X = embed(items, DINOv2Backbone(args.checkpoint), cache)

    risk = df["Disease_Risk"].to_numpy().astype(int)
    bins = np.array([burden_bin(n) for n in df["n_labels"]])
    report = {
        "run": run_info(),
        "plan": "fixé le 11 novembre 2026 avant ce calcul (docstring du script)",
        "backbone": {"model": "dinov2_vitb14", "checkpoint_sha256": fingerprint},
        "images": {"total": len(df), "by_split": df["key"].str.split("/").str[0].value_counts().to_dict()},
        "heads": {},
    }
    for name, path in HEADS.items():
        head, _ = ModalityHead.load(path)
        probs = head.probabilities(X)
        decided = np.array(head.decide(probs))
        derm = decided == "dermoscopy"
        p_derm = probs[:, MODALITIES.index("dermoscopy")]
        by_pathology = {
            c: {"to_dermoscopy": int((derm & (df[c] == 1)).sum()), "n": int((df[c] == 1).sum())}
            for c in pathologies
            if (derm & (df[c] == 1)).any()
        }
        sick = rates(decided[risk == 1])
        report["heads"][name] = {
            "all": rates(decided),
            "disease_risk": {"0": rates(decided[risk == 0]), "1": sick},
            "n_labels": {b: rates(decided[bins == b]) for b in ("0", "1", "2", "3+")},
            "pathologies_of_errors": by_pathology,
            "errors": [
                {
                    "key": df["key"][i],
                    "p_dermoscopy": float(p_derm[i]),
                    "pathologies": [c for c in pathologies if df[c][i] == 1],
                }
                for i in np.flatnonzero(derm)
            ],
            "max_dermoscopy_probability_not_sent": float(p_derm[~derm].max()),
            "criterion_rare": {
                "upper_bound": sick["to_dermoscopy"]["ci"][1],
                "threshold": RARE_UPPER_BOUND,
                "met": sick["to_dermoscopy"]["ci"][1] <= RARE_UPPER_BOUND,
            },
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    for name, r in report["heads"].items():
        print(name, r["all"]["decisions"], "malades :", r["disease_risk"]["1"]["to_dermoscopy"], r["criterion_rare"])
        for e in r["errors"]:
            print("  ", e)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()

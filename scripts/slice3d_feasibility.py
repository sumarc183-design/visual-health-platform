#!/usr/bin/env python3
"""SLICE-3D Permissive (ISIC 2024, CC BY / CC0) : étude de faisabilité, cancer cutané (CPU).

Plan fixé le 11 novembre 2026, avant tout run :
- données : entraînement « Permissive » d'ISIC 2024 (recadrages de photos du corps
  entier, pas de dermoscopie), seule grande base de peau compatible a priori avec un
  usage commercial ; cible = malin (294 images sur 217 477) ; images « Indeterminate »
  écartées ;
- pas d'identifiant patient publié : découpage par CENTRE. Apprentissage : Memorial Sloan
  Kettering et Frazer Institute (Queensland) ; test externe : ACEMID MIA et Athènes
  (centres jamais vus) ;
- échantillon : tous les malins, 4 000 bénins tirés dans les centres d'apprentissage et
  2 000 dans les centres de test (graine 0). L'AUROC ne dépend pas de la proportion de
  malades ; les fausses alertes, si : elles ne sont pas rapportées ici ;
- modèle : DINOv2 ViT-B/14 figé (Apache 2.0), image au petit côté 256 puis recadrage 224,
  régression logistique standardisée à classes équilibrées, C = 1, sans réglage.
  PanDerm est écarté : son préentraînement inclut une partie d'ISIC 2024 ;
- mesures : AUROC et AUC partielle au-dessus de 80 % de sensibilité (mesure officielle
  d'ISIC 2024, de 0 à 0,2), bootstrap par image (1 000 tirages) ; référence dans les
  centres d'apprentissage par validation croisée stratifiée à 5 plis.

Réserves : sonde linéaire seulement ; 39 cas malins au test, intervalles larges ; plusieurs
images d'un même patient peuvent se trouver dans un même centre (validation croisée
optimiste), pas entre centres.

Exemple :
    python scripts/slice3d_feasibility.py --root data/raw/slice3d \\
        --checkpoint third_party/dinov2/dinov2_vitb14_timm.pth
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_modality_router_v2 import Item, embed  # noqa: E402
from src.eval.provenance import run_info  # noqa: E402
from src.models.embedding_cache import EmbeddingCache, file_fingerprint  # noqa: E402
from src.models.timm_backbones import DINOV2_IMAGE_SIZE, DINOv2Backbone  # noqa: E402

TRAIN_SITES = ("Memorial Sloan Kettering", "Frazer Institute")
TEST_SITES = ("ACEMID MIA", "Department of Dermatology, University of Athens")
N_BENIGN = {"train": 4000, "test": 2000}
N_BOOT = 1000


def partial_auc(y: np.ndarray, p: np.ndarray, min_tpr: float = 0.80) -> float:
    """AUC partielle au-dessus d'une sensibilité minimale, comme le calcul officiel
    d'ISIC 2024 (étiquettes et scores inversés, aire sous FPR <= 1 - min_tpr)."""
    v_gt, v_pred = 1 - y, -p
    max_fpr = 1 - min_tpr
    fpr, tpr, _ = roc_curve(v_gt, v_pred)
    stop = np.searchsorted(fpr, max_fpr, "right")
    tpr = np.append(tpr[:stop], np.interp(max_fpr, [fpr[stop - 1], fpr[stop]], [tpr[stop - 1], tpr[stop]]))
    fpr = np.append(fpr[:stop], max_fpr)
    return float(auc(fpr, tpr))


def with_ci(y: np.ndarray, p: np.ndarray, fn, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            draws.append(fn(y[i], p[i]))
    return {"value": fn(y, p), "ci": np.percentile(draws, [2.5, 97.5]).tolist()}


def select(root: Path) -> pd.DataFrame:
    g = pd.read_csv(root / "ISIC_2024_Permissive_Training_GroundTruth.csv")
    s = pd.read_csv(root / "ISIC_2024_Permissive_Training_Supplement.csv", low_memory=False)
    d = g.merge(s, on="isic_id", validate="one_to_one")
    d = d[d["iddx_1"] != "Indeterminate"].copy()
    d["part"] = np.where(
        d["attribution"].str.startswith(TRAIN_SITES),
        "train",
        np.where(d["attribution"].str.startswith(TEST_SITES), "test", ""),
    )
    if (d["part"] == "").any():
        raise ValueError("centre non attribué : vérifier les noms de centres")
    rng = np.random.default_rng(0)
    keep = []
    for part, n in N_BENIGN.items():
        benign = d[(d["part"] == part) & (d["malignant"] == 0)]
        keep.append(d[(d["part"] == part) & (d["malignant"] == 1)])
        keep.append(benign.iloc[np.sort(rng.choice(len(benign), n, replace=False))])
    return pd.concat(keep).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("embeddings/slice3d_dinov2_vitb14.npz"))
    parser.add_argument("--out", type=Path, default=Path("results/slice3d_feasibility.json"))
    args = parser.parse_args()

    d = select(args.root)
    archive = zipfile.ZipFile(args.root / "ISIC_2024_Permissive_Training_Input.zip")
    names = {Path(n).stem: n for n in archive.namelist() if n.endswith(".jpg")}
    items = [Item(f"slice3d/{i}", i, lambda n=names[i]: Image.open(io.BytesIO(archive.read(n)))) for i in d["isic_id"]]
    fingerprint = file_fingerprint(args.checkpoint)
    cache = EmbeddingCache(args.cache, meta={"checkpoint_sha256": fingerprint, "image_size": DINOV2_IMAGE_SIZE})
    X = embed(items, DINOv2Backbone(args.checkpoint), cache)
    y = d["malignant"].astype(int).to_numpy()
    tr, te = (d["part"] == "train").to_numpy(), (d["part"] == "test").to_numpy()

    def fit(Xa, ya):
        scaler = StandardScaler().fit(Xa)
        model = LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced").fit(scaler.transform(Xa), ya)
        return lambda Z: model.predict_proba(scaler.transform(Z))[:, 1]

    in_domain = np.zeros(tr.sum())
    for fit_idx, held in StratifiedKFold(5, shuffle=True, random_state=0).split(X[tr], y[tr]):
        in_domain[held] = fit(X[tr][fit_idx], y[tr][fit_idx])(X[tr][held])
    external = fit(X[tr], y[tr])(X[te])

    report = {
        "run": run_info(),
        "plan": "fixé le 11 novembre 2026 avant tout run (docstring du script)",
        "backbone": {"model": "dinov2_vitb14", "checkpoint_sha256": fingerprint},
        "sample": {
            part: {"malignant": int(y[mask].sum()), "benign": int((y[mask] == 0).sum())}
            for part, mask in (("train_sites", tr), ("test_sites", te))
        },
        "in_domain_5fold": {
            "auroc": with_ci(y[tr], in_domain, roc_auc_score),
            "pauc_tpr80": with_ci(y[tr], in_domain, partial_auc),
        },
        "external_sites": {
            "auroc": with_ci(y[te], external, roc_auc_score),
            "pauc_tpr80": with_ci(y[te], external, partial_auc),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    for key in ("in_domain_5fold", "external_sites"):
        r = report[key]
        print(
            f"{key:16s} AUROC {r['auroc']['value']:.3f} {np.round(r['auroc']['ci'], 3)}  "
            f"pAUC {r['pauc_tpr80']['value']:.3f} {np.round(r['pauc_tpr80']['ci'], 3)}"
        )
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()

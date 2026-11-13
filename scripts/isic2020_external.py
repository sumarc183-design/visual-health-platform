#!/usr/bin/env python3
"""Validation externe de la tête peau sur ISIC 2020 (mélanome, dermoscopie) (CPU).

Plan fixé le 13 novembre 2026, avant tout téléchargement de la base. La tête peau
(`heads/isic2019_dinov2b_head.pt`) n'a été mesurée que sur le test d'ISIC 2019, de même
source que son apprentissage.

Données : ISIC 2020 (challenge SIIM-ISIC, 33 126 dermoscopies d'apprentissage, 2 056
patients, 584 mélanomes confirmés ; données d'origine CC BY-NC), version JPEG 256 × 256
de Kaggle `cdeotte/jpeg-melanoma-256x256`. Jamais utilisée dans le projet. Seuls les
mélanomes sont étiquetés de façon fiable (cible binaire `target`) : seule la classe MEL
est évaluée.

Préparation :
- embeddings DINOv2-B local (celui du routeur et de la tête peau), petit côté 256 px,
  recadrage 224 px ;
- quasi-doublons d'ISIC 2019 exclus avant toute mesure : image d'ISIC 2020 dont le
  cosinus à une image d'apprentissage ISIC 2019 est >= 0,999 (même règle qu'en
  dermatologie) ; nombre rapporté.

Critère principal (fixé) : AUROC du mélanome (probabilité MEL de la tête, inchangée), IC
par bootstrap sur les patients (1 000 tirages), borne basse >= 0,80 (référence interne :
0,874 au test d'ISIC 2019).

Secondaires, sans critère : aux seuils MEL figés de `results/decision_isic2019_dinov2b.json`
(F1 et dépistage), sensibilité, spécificité et VPP avec IC (patients) ; routeur de
modalité (têtes inchangées, seuil 0,9) : part envoyée en fond d'œil, borne haute de l'IC
exact attendue <= 0,5 % (critère de sécurité déjà utilisé).

Conséquence prévue : critère atteint → le classement du mélanome se transpose à une
autre source de dermoscopie (les seuils restent à vérifier : prévalence de 1,8 % contre
18 % dans ISIC 2019, la VPP baissera) ; sinon, la tête peau reste limitée à sa source.

Réserves : images redimensionnées à 256 × 256 par l'auteur de la version Kaggle (rapport
largeur/hauteur non conservé) ; ISIC 2020 et ISIC 2019 partagent des centres (Barcelone,
Vienne, MSK) : externe par la période et les patients, pas par le centre ; mélanomes
rares (1,8 %), autres lésions surtout « nevus » ou « inconnu ».

Exemple :
    python scripts/isic2020_external.py --root data/raw/isic2020 --isic2019-root data/raw/isic2019 \\
        --isic2019-embeddings embeddings_derm --checkpoint third_party/dinov2/dinov2_vitb14_timm.pth
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.dermatology_analysis import ImageList  # noqa: E402
from scripts.dr_screening_sensitivity import rate_ci  # noqa: E402
from scripts.modality_router_pathology_stress import HEADS, rates  # noqa: E402
from scripts.modality_router_skin_photo_stress import to_fundus  # noqa: E402
from scripts.pad_ufes20_skin import auroc_ci  # noqa: E402
from scripts.train_modality_router_v2 import Item, embed  # noqa: E402
from src.data.isic_dataset import CLASS_NAMES  # noqa: E402
from src.eval.decision import decision_metrics  # noqa: E402
from src.eval.provenance import run_info  # noqa: E402
from src.models.classifier_head import MultiLabelHead  # noqa: E402
from src.models.embedding_cache import EmbeddingCache, file_fingerprint, load_embeddings  # noqa: E402
from src.models.timm_backbones import DINOV2_IMAGE_SIZE, DINOv2Backbone  # noqa: E402
from src.pipeline.modality_classifier import ModalityHead  # noqa: E402

SKIN_HEAD = Path("heads/isic2019_dinov2b_head.pt")
THRESHOLDS = Path("results/decision_isic2019_dinov2b.json")
DUPLICATE_COSINE = 0.999
CRITERION = 0.80


def near_duplicates(X: np.ndarray, reference: np.ndarray, threshold: float = DUPLICATE_COSINE) -> np.ndarray:
    """Masque des lignes de X dont le cosinus maximal à `reference` atteint le seuil."""
    a = X / np.linalg.norm(X, axis=1, keepdims=True)
    b = reference / np.linalg.norm(reference, axis=1, keepdims=True)
    best = np.concatenate([(a[i : i + 2048] @ b.T).max(axis=1) for i in range(0, len(a), 2048)])
    return best >= threshold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="train.csv + images JPEG d'apprentissage")
    parser.add_argument("--isic2019-root", type=Path, required=True)
    parser.add_argument("--isic2019-embeddings", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("embeddings/isic2020_dinov2_vitb14.npz"))
    parser.add_argument("--out", type=Path, default=Path("results/isic2020_external.json"))
    args = parser.parse_args()

    meta = pd.read_csv(args.root / "train.csv")
    paths = {p.stem: p for p in args.root.rglob("*.jpg")}
    missing = set(meta["image_name"]) - set(paths)
    if missing:
        raise FileNotFoundError(f"{len(missing)} images absentes, ex. {sorted(missing)[:3]}")
    items = [
        Item(f"isic2020/{n}", g, lambda p=paths[n]: Image.open(p))
        for n, g in zip(meta["image_name"], meta["patient_id"], strict=True)
    ]
    fingerprint = file_fingerprint(args.checkpoint)
    cache = EmbeddingCache(args.cache, meta={"checkpoint_sha256": fingerprint, "image_size": DINOV2_IMAGE_SIZE})
    X = embed(items, DINOv2Backbone(args.checkpoint), cache)

    train_2019 = ImageList(args.isic2019_root, "train")
    X_2019 = load_embeddings(args.isic2019_embeddings, "isic2019", train_2019.keys, "dinov2", preprocess="none")
    duplicate = near_duplicates(X, X_2019)
    keep = ~duplicate
    y = meta["target"].to_numpy()[keep]
    groups = list(meta["patient_id"].to_numpy()[keep])

    artifact = torch.load(SKIN_HEAD, map_location="cpu", weights_only=True)
    if artifact["checkpoint_sha256"] != fingerprint:
        raise ValueError("tête peau calculée avec un autre DINOv2-B")
    head = MultiLabelHead(X.shape[1], len(CLASS_NAMES))
    head.load_state_dict(artifact["state_dict"])
    with torch.no_grad():
        p_mel = torch.sigmoid(head(torch.from_numpy(X[keep]))).numpy()[:, CLASS_NAMES.index("MEL")]

    mel = json.loads(THRESHOLDS.read_text())["classes"]["MEL"]
    report = {
        "run": run_info(),
        "plan": "fixé le 13 novembre 2026 avant tout téléchargement d'ISIC 2020 (docstring du script)",
        "backbone": {"model": "dinov2_vitb14", "checkpoint_sha256": fingerprint},
        "images": {
            "total": len(meta),
            "near_duplicates_of_isic2019_train": int(duplicate.sum()),
            "evaluated": int(keep.sum()),
            "patients": len(set(groups)),
            "melanomas": int(y.sum()),
        },
        "melanoma": auroc_ci(y, p_mel, groups),
        "thresholds": {},
        "router": {},
    }
    for policy in ("f1", "sensitivity"):
        threshold = mel[f"threshold_{policy}"]["threshold"]
        decided = p_mel >= threshold
        report["thresholds"][policy] = {
            **decision_metrics(y, p_mel, threshold),
            "sensitivity_ci": rate_ci(y, decided, groups, True),
            "specificity_ci": rate_ci(y, decided, groups, False),
        }
    low = report["melanoma"]["ci"][0]
    report["criterion"] = {"auroc": report["melanoma"]["auroc"], "lower_bound": low, "threshold": CRITERION}
    report["criterion"]["met"] = bool(low >= CRITERION)
    for name, path in HEADS.items():
        router, _ = ModalityHead.load(path)
        decisions = np.array(router.decide(router.probabilities(X)))
        report["router"][name] = {"all": rates(decisions), "criterion_safe": to_fundus(decisions)}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(report["images"])
    print("Mélanome :", report["melanoma"], "| critère :", report["criterion"])
    for policy, m in report["thresholds"].items():
        print(
            f"  seuil {policy:11s} sens. {m['sensitivity']:.2f} {m['sensitivity_ci']}, "
            f"spéc. {m['specificity']:.2f}, VPP {m['ppv']:.3f}"
        )
    for name, r in report["router"].items():
        print(name, r["all"]["decisions"], "fond d'œil :", r["criterion_safe"])
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()

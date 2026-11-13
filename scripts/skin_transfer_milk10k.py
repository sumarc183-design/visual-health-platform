#!/usr/bin/env python3
"""Tête peau : apprendre aussi sur ISIC 2020 améliore-t-il le transfert ? Mesure sur MILK10k (CPU).

Plan fixé le 13 novembre 2026, avant tout téléchargement d'images MILK10k. Seuls les
en-têtes de ses fichiers CSV ont été lus.

Contexte : la tête peau apprise sur ISIC 2019 tombe à 0,788 d'AUROC du mélanome sur
ISIC 2020 (critère ≥ 0,80 non atteint), et ce n'est pas un effet du format des images.
Même idée que pour la rétinopathie (RFMiD + EyePACS) : élargir les sources
d'apprentissage. ISIC 2020 ne peut alors plus servir de test : la mesure se fait sur une
troisième base.

Base de test : MILK10k (ISIC, 2025, CC BY-NC, doi:10.34970/648456), 5 240 lésions, une
dermoscopie par lésion (les photos cliniques ne sont pas utilisées), 96 % confirmées par
histologie. Jamais utilisée dans le projet. Classes MILK10k → ISIC 2019 : MEL, NV, BCC,
BKL, DF, VASC à l'identique ; AKIEC → AK ; SCCKA → SCC ; INF, BEN_OTH, MAL_OTH sans
équivalent (négatifs pour toutes les classes).

Têtes (DINOv2-B figé, tête et réglages de la baseline, graine 42) :
- A : la tête servie (`heads/isic2019_dinov2b_head.pt`, ISIC 2019 seul, inchangée) ;
- B : même tête apprise sur l'apprentissage ISIC 2019 + les 33 126 images d'ISIC 2020.
  Étiquettes ISIC 2020 : MEL = mélanome ; NV = nævus ; BKL = kératose séborrhéique,
  lentigo, kératose lichénoïde ; autres diagnostics (dont « unknown », bénins non
  précisés) : négatifs pour toutes les classes.

Avant toute mesure : les images MILK10k quasi identiques (cosinus >= 0,999) à une image
d'apprentissage (ISIC 2019 ou ISIC 2020) sont exclues ; nombre rapporté.

Critère principal (fixé) : sur MILK10k, écart apparié d'AUROC du mélanome (B − A), IC par
bootstrap sur les lésions (1 000 tirages), borne basse > 0.
Secondaires, sans critère : AUROC du mélanome de A et de B avec IC (repère 0,80) ; AUROC
macro des classes présentes ; test ISIC 2019 : écart B − A du mélanome (B ne doit pas
perdre sur la source d'origine) ; routeur sur MILK10k (part envoyée en fond d'œil).

Conséquence prévue : critère atteint → B remplace la tête peau servie (enregistrée et
vérifiée, seuils recalculés sur une validation) ; sinon, ajouter ISIC 2020 n'aide pas et
la tête actuelle reste.

Réserves : ISIC 2020 surtout fait de bénins « unknown » (27 124 images) et de nævus ; les
centres de MILK10k ne sont pas tous connus et peuvent recouvrir ceux d'ISIC ; un seul
découpage, une seule graine.

Exemple :
    python scripts/skin_transfer_milk10k.py --milk-root data/raw/milk10k --isic2019-root data/raw/isic2019 \\
        --isic2019-embeddings embeddings_derm --isic2020-root data/raw/isic2020 \\
        --checkpoint third_party/dinov2/dinov2_vitb14_timm.pth
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.dermatology_analysis import ImageList  # noqa: E402
from scripts.isic2020_external import near_duplicates  # noqa: E402
from scripts.modality_router_pathology_stress import HEADS, rates  # noqa: E402
from scripts.modality_router_skin_photo_stress import to_fundus  # noqa: E402
from scripts.pad_ufes20_skin import auroc_ci  # noqa: E402
from scripts.robustness_checks import fit_head  # noqa: E402
from scripts.train_modality_router_v2 import Item, embed  # noqa: E402
from src.data.isic_dataset import CLASS_NAMES  # noqa: E402
from src.eval.metrics import bootstrap_auroc_ci, paired_bootstrap_diff  # noqa: E402
from src.eval.provenance import run_info  # noqa: E402
from src.models.classifier_head import MultiLabelHead  # noqa: E402
from src.models.embedding_cache import EmbeddingCache, file_fingerprint, load_embeddings  # noqa: E402
from src.models.timm_backbones import DINOV2_IMAGE_SIZE, DINOv2Backbone  # noqa: E402
from src.pipeline.modality_classifier import ModalityHead  # noqa: E402

SKIN_HEAD = Path("heads/isic2019_dinov2b_head.pt")
MILK_TO_ISIC = {"MEL": "MEL", "NV": "NV", "BCC": "BCC", "BKL": "BKL", "DF": "DF", "VASC": "VASC", "AKIEC": "AK"}
MILK_TO_ISIC["SCCKA"] = "SCC"
ISIC2020_TO_ISIC = {
    "melanoma": "MEL",
    "nevus": "NV",
    "seborrheic keratosis": "BKL",
    "lentigo NOS": "BKL",
    "solar lentigo": "BKL",
    "lichenoid keratosis": "BKL",
}


def milk_targets(gt: pd.DataFrame) -> np.ndarray:
    """Matrice (lésions, 8 classes ISIC 2019) ; classes sans équivalent : négatives partout."""
    y = np.zeros((len(gt), len(CLASS_NAMES)), np.float32)
    for milk, isic in MILK_TO_ISIC.items():
        y[:, CLASS_NAMES.index(isic)] = gt[milk].to_numpy()
    return y


def isic2020_targets(diagnosis: pd.Series) -> np.ndarray:
    y = np.zeros((len(diagnosis), len(CLASS_NAMES)), np.float32)
    for i, d in enumerate(diagnosis):
        if d in ISIC2020_TO_ISIC:
            y[i, CLASS_NAMES.index(ISIC2020_TO_ISIC[d])] = 1
    return y


def scores(head: MultiLabelHead, X: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return torch.sigmoid(head(torch.from_numpy(X))).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--milk-root", type=Path, required=True, help="archive MILK10k extraite (metadata.csv, supplements/, images/)")
    parser.add_argument("--isic2019-root", type=Path, required=True)
    parser.add_argument("--isic2019-embeddings", type=Path, required=True)
    parser.add_argument("--isic2020-root", type=Path, required=True)
    parser.add_argument("--isic2020-cache", type=Path, default=Path("embeddings/isic2020_dinov2_vitb14.npz"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("embeddings/milk10k_dinov2_vitb14.npz"))
    parser.add_argument("--out", type=Path, default=Path("results/skin_transfer_milk10k.json"))
    args = parser.parse_args()

    fingerprint = file_fingerprint(args.checkpoint)
    backbone = DINOv2Backbone(args.checkpoint)
    meta = {"checkpoint_sha256": fingerprint, "image_size": DINOV2_IMAGE_SIZE}

    # MILK10k : une dermoscopie par lésion
    images = pd.read_csv(args.milk_root / "metadata.csv")
    derm = images[images["image_type"] == "dermoscopic"].drop_duplicates("lesion_id")
    gt = pd.read_csv(args.milk_root / "supplements" / "training_gt.csv").merge(derm[["lesion_id", "isic_id"]], on="lesion_id")
    paths = {p.stem: p for p in args.milk_root.rglob("*.jpg")}
    missing = set(gt["isic_id"]) - set(paths)
    if missing:
        raise FileNotFoundError(f"{len(missing)} images MILK10k absentes, ex. {sorted(missing)[:3]}")
    items = [
        Item(f"milk10k/{i}", g, lambda p=paths[i]: Image.open(p))
        for i, g in zip(gt["isic_id"], gt["lesion_id"], strict=True)
    ]
    X_milk = embed(items, backbone, EmbeddingCache(args.cache, meta=meta))

    # Apprentissage
    train19, test19 = ImageList(args.isic2019_root, "train"), ImageList(args.isic2019_root, "test")
    X19 = load_embeddings(args.isic2019_embeddings, "isic2019", train19.keys, "dinov2", preprocess="none")
    X19_test = load_embeddings(args.isic2019_embeddings, "isic2019", test19.keys, "dinov2", preprocess="none")
    m20 = pd.read_csv(args.isic2020_root / "train.csv")
    paths20 = {p.stem: p for p in args.isic2020_root.rglob("*.jpg")}
    items20 = [
        Item(f"isic2020/{n}", g, lambda p=paths20[n]: Image.open(p))
        for n, g in zip(m20["image_name"], m20["patient_id"], strict=True)
    ]
    X20 = embed(items20, backbone, EmbeddingCache(args.isic2020_cache, meta=meta))
    y20 = isic2020_targets(m20["diagnosis"])

    duplicate = near_duplicates(X_milk, np.vstack([X19, X20]))
    keep = ~duplicate
    X_eval, y_eval = X_milk[keep], milk_targets(gt)[keep]
    groups = list(gt["lesion_id"].to_numpy()[keep])

    artifact = torch.load(SKIN_HEAD, map_location="cpu", weights_only=True)
    if fingerprint not in {artifact["checkpoint_sha256"], artifact.get("embeddings_checkpoint_sha256")}:
        raise ValueError("tête peau calculée avec un autre DINOv2-B")
    head_a = MultiLabelHead(X_milk.shape[1], len(CLASS_NAMES))
    head_a.load_state_dict(artifact["state_dict"])
    with contextlib.redirect_stdout(io.StringIO()):
        head_b = fit_head(np.vstack([X19, X20]), np.vstack([train19.targets, y20]))

    mel = CLASS_NAMES.index("MEL")
    p_a, p_b = scores(head_a, X_eval), scores(head_b, X_eval)
    present = [c for i, c in enumerate(CLASS_NAMES) if y_eval[:, i].sum() > 0]
    cols = [CLASS_NAMES.index(c) for c in present]
    diff = paired_bootstrap_diff(y_eval[:, cols], p_a[:, cols], p_b[:, cols], present, groups)
    t_a, t_b = scores(head_a, X19_test), scores(head_b, X19_test)
    test19_diff = paired_bootstrap_diff(test19.targets, t_a, t_b, list(CLASS_NAMES), list(test19.groups))

    report = {
        "run": run_info(),
        "plan": "fixé le 13 novembre 2026 avant tout téléchargement d'images MILK10k (docstring du script)",
        "milk10k": {
            "lesions": len(gt),
            "near_duplicates_of_training": int(duplicate.sum()),
            "evaluated": int(keep.sum()),
            "positives": {c: int(y_eval[:, CLASS_NAMES.index(c)].sum()) for c in present},
        },
        "training_b": {"isic2019": len(X19), "isic2020": len(X20)},
        "head_a": {"melanoma": auroc_ci(y_eval[:, mel], p_a[:, mel], groups)},
        "head_b": {"melanoma": auroc_ci(y_eval[:, mel], p_b[:, mel], groups)},
        "b_minus_a": diff,
        "isic2019_test_b_minus_a": {"MEL": test19_diff["MEL"], "macro": test19_diff["macro"]},
        "router": {},
    }
    for name, p in {"head_a": p_a, "head_b": p_b}.items():
        report[name].update(bootstrap_auroc_ci(y_eval[:, cols], p[:, cols], present, groups))
    low = diff["MEL"]["ci"][0]
    report["criterion"] = {"melanoma_diff": diff["MEL"], "met": bool(low > 0)}
    for name, path in HEADS.items():
        router, _ = ModalityHead.load(path)
        decisions = np.array(router.decide(router.probabilities(X_milk)))
        report["router"][name] = {"all": rates(decisions), "criterion_safe": to_fundus(decisions)}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(report["milk10k"])
    print("Mélanome A :", report["head_a"]["melanoma"], "| B :", report["head_b"]["melanoma"])
    print("B − A :", diff["MEL"], "| macro :", diff["macro"], "| critère :", report["criterion"]["met"])
    print("Test ISIC 2019, B − A :", report["isic2019_test_b_minus_a"])
    for name, r in report["router"].items():
        print(name, r["all"]["decisions"], "fond d'œil :", r["criterion_safe"])
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()

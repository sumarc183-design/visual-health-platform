#!/usr/bin/env python3
"""Tête peau PanDerm en validation externe : ISIC 2020 (critère principal) et MILK10k (confirmation) (CPU).

Plan fixé le 18 novembre 2026, avant tout calcul d'embedding PanDerm sur ISIC 2020 ou
MILK10k. Brouillon du 17 novembre validé au Point du 18-11 : les deux bases, et critère
de J90 maintenu.

Contexte : la tête servie (A, DINOv2-B figé, apprise sur ISIC 2019) obtient 0,788
[0,765 ; 0,808] d'AUROC du mélanome sur ISIC 2020 (critère : borne basse >= 0,80, non
atteint). Sur le test d'ISIC 2019, la même sonde sur PanDerm dépasse DINOv2-B (+0,037
[+0,030 ; +0,043] en AUROC macro).

Données de pré-entraînement de PanDerm (article, Methods et « Data availability », relus
le 17 novembre) : elles contiennent ISIC 2024, la source de SLICE-3D (donc exclue ici).
ISIC 2019, ISIC 2020, MILK10k et PAD-UFES-20 n'y figurent pas ; aucun dédoublonnage n'est
décrit.

Têtes :
- P : PanDerm-Large figé, même sonde et mêmes réglages que la tête servie (baseline,
  graine 42 ; `fit_head`), apprise sur l'apprentissage ISIC 2019 (8 classes, sans UNK).
  Embeddings ISIC 2019 existants (`isic2019_none_224_panderm_large.npz`, Kaggle) ; P doit
  redonner les scores PanDerm archivés du 9 novembre (écart <= 1e-5) ;
- A : la tête servie (`heads/isic2019_dinov2b_head.pt`), inchangée, sur les embeddings
  DINOv2-B déjà calculés.

Préparation :
- embeddings PanDerm d'ISIC 2020 (33 126) et de MILK10k (5 240, une dermoscopie par
  lésion), même transformation que pour ISIC 2019 (petit côté 256 px, recadrage 224 px)
  et normalisation de PanDerm. Contrôle préalable : 16 images d'apprentissage ISIC 2019
  recalculées ici doivent redonner les embeddings Kaggle (cosinus >= 0,9999) ;
- quasi-doublons exclus avant toute mesure : image dont le cosinus à une image
  d'apprentissage ISIC 2019 est >= 0,999, dans l'espace PanDerm OU dans l'espace DINOv2-B
  (réunion des deux, pour comparer A et P sur les mêmes images) ; nombres rapportés.

Critère principal (option 1, ISIC 2020) : AUROC du mélanome de P, IC par bootstrap sur
les patients (1 000 tirages), borne basse >= 0,80.

Confirmation (option 2, MILK10k, critère lu à part) : écart apparié d'AUROC du mélanome
P − A, IC par bootstrap sur les lésions (1 000 tirages), borne basse > 0.

Secondaires, sans critère : écart apparié P − A sur ISIC 2020 ; AUROC du mélanome de P et
de A sur MILK10k ; AUROC macro des classes présentes sur MILK10k (P, A, écart) ; test
ISIC 2019, écart P − A (mélanome et macro) ; sensibilité et spécificité de P (IC) aux
seuils MEL choisis sur la validation ISIC 2019 (F1 et dépistage, même règle que la tête
servie).

Conséquences prévues : critère principal atteint → P devient candidate à la tête peau
servie (`predict.py`) : seuils et contrôle qualité peau à revérifier, routeur inchangé
(DINOv2-B) ; non atteint → le défaut de transfert ne vient pas seulement du modèle de
base, piste suivante à rediscuter.

Réserves : ISIC 2020 a déjà servi une fois (évaluation de A, sans réglage) ; MILK10k deux
fois pour comparer des têtes DINOv2 (A, B, C), jamais pour PanDerm ; sources de PanDerm
pas toutes publiques (recouvrement non exclu) ; licence CC BY-NC-ND (poids non modifiés,
usage de démonstration) ; une seule graine.

Exemple :
    python scripts/panderm_external.py --isic2019-root data/raw/isic2019 --isic2020-root data/raw/isic2020 \\
        --milk-root data/raw/milk10k --panderm third_party/panderm/panderm_large.pth \\
        --dinov2 third_party/dinov2/dinov2_vitb14_timm.pth
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
from scripts.dr_screening_sensitivity import rate_ci  # noqa: E402
from scripts.isic2020_external import near_duplicates  # noqa: E402
from scripts.pad_ufes20_skin import auroc_ci  # noqa: E402
from scripts.robustness_checks import fit_head  # noqa: E402
from scripts.skin_decision_thresholds import TARGET  # noqa: E402
from scripts.skin_transfer_milk10k import milk_targets  # noqa: E402
from scripts.train_modality_router_v2 import Item, embed  # noqa: E402
from src.data.isic_dataset import CLASS_NAMES, SKIN_TRANSFORM, ISIC2019Dataset  # noqa: E402
from src.eval.decision import decision_metrics, per_class_report  # noqa: E402
from src.eval.metrics import bootstrap_auroc_ci, paired_bootstrap_diff  # noqa: E402
from src.eval.predictions import check_gap, require_finite  # noqa: E402
from src.eval.provenance import run_info  # noqa: E402
from src.models.classifier_head import head_sha256  # noqa: E402
from src.models.embedding_cache import KNOWN_CHECKPOINTS, EmbeddingCache, file_fingerprint, load_embeddings  # noqa: E402
from src.models.timm_backbones import DINOV2_IMAGE_SIZE, PANDERM_IMAGE_SIZE, PanDermBackbone  # noqa: E402
from src.pipeline.served_head import load_served_head  # noqa: E402

SKIN_HEAD = Path("heads/isic2019_dinov2b_head.pt")
REFERENCE = Path("results/dermatology_isic2019.npz")
CRITERION = 0.80
MEL = CLASS_NAMES.index("MEL")
PREFLIGHT_IMAGES, PREFLIGHT_COSINE = 16, 0.9999


def scores(head, X: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return require_finite(torch.sigmoid(head(torch.from_numpy(X))).numpy(), "scores")


def cached(path: Path, keys: list[str], meta: dict) -> np.ndarray:
    """Embeddings déjà calculés (DINOv2-B de la tête A) : aucun recalcul, toutes les clés exigées."""
    cache = EmbeddingCache(path, meta=meta)
    missing = cache.missing(keys)
    if missing:
        raise FileNotFoundError(f"{path} : {len(missing)} embedding(s) absent(s), dont {missing[:3]}")
    return cache.get(keys)


def preflight(backbone, isic2019_root: Path, reference: np.ndarray, keys: list[str]) -> dict:
    """PanDerm local + chaîne de ce script redonnent-ils les embeddings ISIC 2019 de Kaggle ?"""
    dataset = ISIC2019Dataset(isic2019_root, "train")
    paths = [dataset.image_path(i) for i in range(PREFLIGHT_IMAGES)]
    if dataset.image_keys[:PREFLIGHT_IMAGES] != keys[:PREFLIGHT_IMAGES]:
        raise ValueError("clés ISIC 2019 dans un autre ordre que les embeddings")
    items = [Item(k, k, lambda p=p: Image.open(p)) for k, p in zip(keys, paths, strict=False)]
    scratch = EmbeddingCache(Path("embeddings") / "_panderm_preflight.npz")
    try:
        X = embed(items, backbone, scratch)
    finally:
        scratch.path.unlink(missing_ok=True)
    ref = reference[:PREFLIGHT_IMAGES]
    cosine = (X * ref).sum(1) / (np.linalg.norm(X, axis=1) * np.linalg.norm(ref, axis=1))
    result = {"images": len(X), "min_cosine": float(cosine.min()), "max_abs_diff": float(np.abs(X - ref).max())}
    if result["min_cosine"] < PREFLIGHT_COSINE:
        raise ValueError(f"PanDerm local ne redonne pas les embeddings Kaggle : {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--isic2019-root", type=Path, required=True)
    parser.add_argument("--isic2019-embeddings", type=Path, default=Path("embeddings_derm"))
    parser.add_argument("--isic2020-root", type=Path, required=True)
    parser.add_argument("--milk-root", type=Path, required=True)
    parser.add_argument("--panderm", type=Path, required=True, help="third_party/panderm/panderm_large.pth")
    parser.add_argument("--dinov2", type=Path, required=True, help="DINOv2-B local (tête A)")
    parser.add_argument("--out", type=Path, default=Path("results/panderm_external.json"))
    args = parser.parse_args()

    panderm_fp, dinov2_fp = file_fingerprint(args.panderm), file_fingerprint(args.dinov2)
    if panderm_fp != KNOWN_CHECKPOINTS["panderm"]:
        raise ValueError(f"poids PanDerm {panderm_fp}, attendu {KNOWN_CHECKPOINTS['panderm']}")
    p_meta = {"checkpoint_sha256": panderm_fp, "image_size": PANDERM_IMAGE_SIZE, "transform": SKIN_TRANSFORM}
    a_meta = {"checkpoint_sha256": dinov2_fp, "image_size": DINOV2_IMAGE_SIZE}

    # ISIC 2019 (apprentissage de P, validation pour ses seuils, test secondaire)
    lists = {s: ImageList(args.isic2019_root, s) for s in ("train", "val", "test")}
    P19 = {
        s: load_embeddings(args.isic2019_embeddings, "isic2019", d.keys, "panderm", preprocess="none")
        for s, d in lists.items()
    }
    A19 = {
        s: load_embeddings(args.isic2019_embeddings, "isic2019", lists[s].keys, "dinov2", preprocess="none")
        for s in ("train", "test")
    }

    backbone = PanDermBackbone(args.panderm)
    check = preflight(backbone, args.isic2019_root, P19["train"], lists["train"].keys)
    print("Contrôle préalable :", check, flush=True)

    # ISIC 2020 : une image par ligne, groupes = patients
    m20 = pd.read_csv(args.isic2020_root / "train.csv")
    paths20 = {p.stem: p for p in args.isic2020_root.rglob("*.jpg")}
    missing = set(m20["image_name"]) - set(paths20)
    if missing:
        raise FileNotFoundError(f"{len(missing)} images ISIC 2020 absentes, ex. {sorted(missing)[:3]}")
    keys20 = [f"isic2020/{n}" for n in m20["image_name"]]
    items20 = [
        Item(k, g, lambda p=paths20[n]: Image.open(p), lambda p=paths20[n]: file_fingerprint(p))
        for k, n, g in zip(keys20, m20["image_name"], m20["patient_id"], strict=True)
    ]
    print("Embeddings PanDerm ISIC 2020...", flush=True)
    P20 = embed(items20, backbone, EmbeddingCache(Path("embeddings/isic2020_panderm_large.npz"), meta=p_meta))
    A20 = cached(Path("embeddings/isic2020_dinov2_vitb14.npz"), keys20, a_meta)

    # MILK10k : une dermoscopie par lésion
    images = pd.read_csv(args.milk_root / "metadata.csv")
    derm = images[images["image_type"] == "dermoscopic"].drop_duplicates("lesion_id")
    gt = pd.read_csv(args.milk_root / "supplements" / "training_gt.csv").merge(
        derm[["lesion_id", "isic_id"]], on="lesion_id"
    )
    paths_m = {p.stem: p for p in args.milk_root.rglob("*.jpg")}
    missing = set(gt["isic_id"]) - set(paths_m)
    if missing:
        raise FileNotFoundError(f"{len(missing)} images MILK10k absentes, ex. {sorted(missing)[:3]}")
    keys_m = [f"milk10k/{i}" for i in gt["isic_id"]]
    items_m = [
        Item(k, g, lambda p=paths_m[i]: Image.open(p), lambda p=paths_m[i]: file_fingerprint(p))
        for k, i, g in zip(keys_m, gt["isic_id"], gt["lesion_id"], strict=True)
    ]
    print("Embeddings PanDerm MILK10k...", flush=True)
    Pm = embed(items_m, backbone, EmbeddingCache(Path("embeddings/milk10k_panderm_large.npz"), meta=p_meta))
    Am = cached(Path("embeddings/milk10k_dinov2_vitb14.npz"), keys_m, a_meta)

    # Quasi-doublons d'apprentissage ISIC 2019, dans l'un ou l'autre espace
    def duplicates(P: np.ndarray, A: np.ndarray) -> dict:
        in_p, in_a = near_duplicates(P, P19["train"]), near_duplicates(A, A19["train"])
        return {"mask": in_p | in_a, "panderm": int(in_p.sum()), "dinov2": int(in_a.sum())}

    dup20, dup_m = duplicates(P20, A20), duplicates(Pm, Am)

    # Têtes
    with contextlib.redirect_stdout(io.StringIO()):
        head_p = fit_head(P19["train"], lists["train"].targets)
    gaps = {}
    with np.load(REFERENCE) as ref:
        for split in ("val", "test"):
            if list(ref[f"{split}_keys"]) != lists[split].keys:
                raise ValueError(f"clés {split} différentes de l'archive")
            archived = require_finite(ref[f"PanDerm_{split}"], f"scores PanDerm archivés {split}")
            gaps[split] = float(np.abs(scores(head_p, P19[split]) - archived).max())
            check_gap(gaps[split], 1e-5, f"P ne redonne pas les scores PanDerm archivés ({split})")
    served = load_served_head(SKIN_HEAD, dinov2_fp)
    head_a = served.head

    p_val, p_test19 = scores(head_p, P19["val"]), scores(head_p, P19["test"])
    thresholds = per_class_report(
        lists["val"].targets, p_val, lists["test"].targets, p_test19, list(CLASS_NAMES), TARGET
    )["classes"]["MEL"]

    # Option 1 : ISIC 2020
    keep = ~dup20["mask"]
    y20 = m20["target"].to_numpy()[keep]
    g20 = list(m20["patient_id"].to_numpy()[keep])
    s20_p, s20_a = scores(head_p, P20[keep])[:, MEL], scores(head_a, A20[keep])[:, MEL]
    isic2020 = {
        "images": {
            "total": len(m20),
            "near_duplicates_panderm": dup20["panderm"],
            "near_duplicates_dinov2": dup20["dinov2"],
            "excluded": int(dup20["mask"].sum()),
            "evaluated": int(keep.sum()),
            "patients": len(set(g20)),
            "melanomas": int(y20.sum()),
        },
        "head_p": auroc_ci(y20, s20_p, g20),
        "head_a": auroc_ci(y20, s20_a, g20),
        "p_minus_a": paired_bootstrap_diff(y20[:, None], s20_a[:, None], s20_p[:, None], ["MEL"], g20)["MEL"],
        "thresholds_p": {},
    }
    for policy in ("f1", "sensitivity"):
        threshold = thresholds[f"threshold_{policy}"]["threshold"]
        decided = s20_p >= threshold
        isic2020["thresholds_p"][policy] = {
            **decision_metrics(y20, s20_p, threshold),
            "sensitivity_ci": rate_ci(y20, decided, g20, True),
            "specificity_ci": rate_ci(y20, decided, g20, False),
        }
    low = isic2020["head_p"]["ci"][0]
    criterion = {"auroc": isic2020["head_p"]["auroc"], "lower_bound": low, "threshold": CRITERION}
    criterion["met"] = bool(low >= CRITERION)

    # Option 2 : MILK10k
    keep_m = ~dup_m["mask"]
    y_m = milk_targets(gt)[keep_m]
    g_m = list(gt["lesion_id"].to_numpy()[keep_m])
    sm_p, sm_a = scores(head_p, Pm[keep_m]), scores(head_a, Am[keep_m])
    present = [c for i, c in enumerate(CLASS_NAMES) if y_m[:, i].sum() > 0]
    cols = [CLASS_NAMES.index(c) for c in present]
    diff_m = paired_bootstrap_diff(y_m[:, cols], sm_a[:, cols], sm_p[:, cols], present, g_m)
    milk = {
        "lesions": {
            "total": len(gt),
            "near_duplicates_panderm": dup_m["panderm"],
            "near_duplicates_dinov2": dup_m["dinov2"],
            "excluded": int(dup_m["mask"].sum()),
            "evaluated": int(keep_m.sum()),
            "positives": {c: int(y_m[:, CLASS_NAMES.index(c)].sum()) for c in present},
        },
        "head_p": {"melanoma": auroc_ci(y_m[:, MEL], sm_p[:, MEL], g_m)},
        "head_a": {"melanoma": auroc_ci(y_m[:, MEL], sm_a[:, MEL], g_m)},
        "p_minus_a": diff_m,
    }
    for name, s in (("head_p", sm_p), ("head_a", sm_a)):
        milk[name].update(bootstrap_auroc_ci(y_m[:, cols], s[:, cols], present, g_m))
    confirmation = {"melanoma_diff": diff_m["MEL"], "met": bool(diff_m["MEL"]["ci"][0] > 0)}

    test19 = paired_bootstrap_diff(
        lists["test"].targets, scores(head_a, A19["test"]), p_test19, list(CLASS_NAMES), list(lists["test"].lesions)
    )
    report = {
        "run": run_info(),
        "plan": "fixé le 18 novembre 2026 avant tout embedding PanDerm d'ISIC 2020 et de MILK10k (docstring)",
        "backbones": {"panderm_large": panderm_fp, "dinov2_vitb14": dinov2_fp},
        "preflight": check,
        "head_p": {
            "head_sha256": head_sha256(head_p.state_dict()),
            "archive_check_max_abs_diff": gaps,
            "mel_thresholds_val": {p: thresholds[f"threshold_{p}"]["threshold"] for p in ("f1", "sensitivity")},
        },
        "head_a_identity": served.identity,
        "criterion_isic2020": criterion,
        "confirmation_milk10k": confirmation,
        "isic2020": isic2020,
        "milk10k": milk,
        "isic2019_test_p_minus_a": {"MEL": test19["MEL"], "macro": test19["macro"]},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print("ISIC 2020 :", isic2020["images"])
    print("  P :", isic2020["head_p"], "| A :", isic2020["head_a"], "| P − A :", isic2020["p_minus_a"])
    print("  Critère principal :", criterion)
    print("MILK10k :", milk["lesions"])
    print("  P − A mélanome :", diff_m["MEL"], "| macro :", diff_m["macro"], "| confirmation :", confirmation["met"])
    print("Test ISIC 2019, P − A :", report["isic2019_test_p_minus_a"])
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()

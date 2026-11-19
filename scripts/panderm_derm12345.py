#!/usr/bin/env python3
"""Tête peau PanDerm apprise sur ISIC 2019 + ISIC 2020, validation externe sur DERM12345 (CPU).

Plan fixé le 19 novembre 2026, avant tout calcul d'embedding sur DERM12345. Seuls ont été
lus : les comptes par diagnostic de la collection (API ISIC) et une fiche d'exemple.

Contexte (`results/panderm_external.json`, 18 nov.) : la tête PanDerm apprise sur ISIC
2019 seul (P) obtient 0,811 [0,788 ; 0,829] d'AUROC du mélanome sur ISIC 2020 ; critère de
J90 (borne basse >= 0,80) non atteint. Décision de Marc le 19 nov. : option B de la
proposition du jour (apprendre aussi sur ISIC 2020, nouvelle base externe), avec l'option
C (seuils choisis à la prévalence d'un dépistage) ; PanDerm retenu comme modèle de base.

Base externe : DERM12345 (collection ISIC 399, CC BY, publiée en 2024), 12 345
dermoscopies, patients identifiés ; 400 mélanomes confirmés par histologie (3,2 %).
Absente de l'article PanDerm (préentraînement et évaluations, texte intégral relu le
19 nov.), d'ISIC 2019 et d'ISIC 2020. Jamais utilisée dans le projet.

Têtes (PanDerm-Large figé, même sonde et mêmes réglages que la tête servie ; `fit_head`,
graine 42) :
- Q : apprise sur l'apprentissage ISIC 2019 + 80 % des patients d'ISIC 2020 (tirage
  par patient, graine 42). Étiquettes ISIC 2020 comme pour la tête B du 13 nov.
  (`isic2020_targets` : MEL, NV, BKL ; autres diagnostics, dont « unknown », négatifs
  pour toutes les classes) ;
- P : la tête du 18 nov. (ISIC 2019 seul), refaite à l'identique (scores archivés
  redonnés à 1e-5 près), pour mesurer l'apport d'ISIC 2020.

Seuils de Q pour le mélanome (option C) : choisis sur les 20 % de patients d'ISIC 2020
tenus à l'écart (prévalence d'un dépistage, environ 1,8 %), mêmes règles que la tête
servie : F1 maximal, et sensibilité >= 0,90.

Avant toute mesure : images DERM12345 quasi identiques (cosinus >= 0,999 dans l'espace
PanDerm) à une image d'ISIC 2019 (apprentissage) ou d'ISIC 2020 exclues ; nombre rapporté.

Critère principal (J90) : AUROC du mélanome de Q sur DERM12345, IC par bootstrap sur les
patients (1 000 tirages), borne basse >= 0,80.

Secondaires, sans critère : AUROC du mélanome de P et écart apparié Q − P sur DERM12345 ;
AUROC macro des classes présentes (correspondance ci-dessous) pour Q et P ; sensibilité,
spécificité et VPP de Q (IC par patient) aux seuils d'ISIC 2020 et, pour comparaison, aux
seuils choisis sur la validation ISIC 2019 ; MILK10k et test ISIC 2019 : écart Q − P
(mélanome et macro), pour vérifier que Q ne perd pas ailleurs.

Correspondance DERM12345 (diagnosis_3) → classes ISIC 2019 : mélanomes (NOS, invasif,
in situ) → MEL ; nævus → NV ; carcinome basocellulaire → BCC ; carcinome épidermoïde (NOS,
in situ) → SCC ; kératose actinique → AK ; kératose séborrhéique, lentigo solaire,
kératose lichénoïde → BKL ; dermatofibrome → DF ; hémangiome, angiokératome, granulome
pyogénique → VASC ; autres : négatifs pour toutes les classes.

Conséquences prévues : critère atteint → Q devient la tête peau servie (`predict.py`,
PanDerm en second modèle d'embedding, routeur DINOv2-B inchangé), avec les seuils
d'ISIC 2020 ; contrôle qualité peau à revérifier. Non atteint → conclusion A : la peau
reste démontrée sur la dermoscopie sans atteindre le niveau visé en validation externe.

Réserves : bénins de DERM12345 pas tous confirmés par histologie (les malins le sont) ;
un seul pays (Turquie) ; MILK10k déjà vu pour P (secondaire seulement) ; licence PanDerm
CC BY-NC-ND (usage de démonstration) ; une seule graine, un seul tirage des patients.

Exemple :
    python scripts/panderm_derm12345.py --isic2019-root data/raw/isic2019 --isic2020-root data/raw/isic2020 \\
        --milk-root data/raw/milk10k --derm-root data/raw/derm12345 --panderm third_party/panderm/panderm_large.pth
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
from scripts.panderm_external import preflight  # noqa: E402
from scripts.robustness_checks import fit_head  # noqa: E402
from scripts.skin_decision_thresholds import TARGET  # noqa: E402
from scripts.skin_transfer_milk10k import isic2020_targets, milk_targets  # noqa: E402
from scripts.train_modality_router_v2 import Item, embed  # noqa: E402
from src.data.isic_dataset import CLASS_NAMES, SKIN_TRANSFORM  # noqa: E402
from src.eval.decision import choose_threshold, decision_metrics  # noqa: E402
from src.eval.metrics import bootstrap_auroc_ci, paired_bootstrap_diff  # noqa: E402
from src.eval.predictions import check_gap, require_finite  # noqa: E402
from src.eval.provenance import run_info  # noqa: E402
from src.models.classifier_head import head_sha256  # noqa: E402
from src.models.embedding_cache import KNOWN_CHECKPOINTS, EmbeddingCache, file_fingerprint, load_embeddings  # noqa: E402
from src.models.timm_backbones import PANDERM_IMAGE_SIZE, PanDermBackbone  # noqa: E402

REFERENCE = Path("results/dermatology_isic2019.npz")
CRITERION = 0.80
MEL = CLASS_NAMES.index("MEL")
VAL_FRACTION, SEED = 0.20, 42
DERM_TO_ISIC = {
    "Melanoma, NOS": "MEL",
    "Melanoma Invasive": "MEL",
    "Melanoma in situ": "MEL",
    "Nevus": "NV",
    "Basal cell carcinoma": "BCC",
    "Squamous cell carcinoma, NOS": "SCC",
    "Squamous cell carcinoma in situ": "SCC",
    "Solar or actinic keratosis": "AK",
    "Seborrheic keratosis": "BKL",
    "Solar lentigo": "BKL",
    "Lichen planus like keratosis": "BKL",
    "Dermatofibroma": "DF",
    "Hemangioma": "VASC",
    "Angiokeratoma": "VASC",
    "Pyogenic granuloma": "VASC",
}


def scores(head, X: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return require_finite(torch.sigmoid(head(torch.from_numpy(X))).numpy(), "scores")


def derm_targets(diagnosis: pd.Series) -> np.ndarray:
    y = np.zeros((len(diagnosis), len(CLASS_NAMES)), np.float32)
    for i, d in enumerate(diagnosis):
        if d in DERM_TO_ISIC:
            y[i, CLASS_NAMES.index(DERM_TO_ISIC[d])] = 1
    return y


def held_out_patients(patients: pd.Series) -> np.ndarray:
    """Masque des images dont le patient tombe dans les 20 % tenus à l'écart (graine 42)."""
    unique = np.array(sorted(patients.unique()))
    order = np.random.default_rng(SEED).permutation(len(unique))
    held = set(unique[order[: round(VAL_FRACTION * len(unique))]])
    return patients.isin(held).to_numpy()


def at_threshold(y: np.ndarray, s: np.ndarray, threshold: float, groups: list[str]) -> dict:
    decided = s >= threshold
    return {
        **decision_metrics(y, s, threshold),
        "sensitivity_ci": rate_ci(y, decided, groups, True),
        "specificity_ci": rate_ci(y, decided, groups, False),
    }


def compare(y: np.ndarray, s_p: np.ndarray, s_q: np.ndarray, groups: list[str]) -> dict:
    """AUROC (P, Q) et écart apparié Q − P sur les classes présentes."""
    present = [c for i, c in enumerate(CLASS_NAMES) if y[:, i].sum() > 0]
    cols = [CLASS_NAMES.index(c) for c in present]
    out = {"positives": {c: int(y[:, CLASS_NAMES.index(c)].sum()) for c in present}}
    for name, s in (("head_p", s_p), ("head_q", s_q)):
        out[name] = {"melanoma": auroc_ci(y[:, MEL], s[:, MEL], groups)}
        out[name].update(bootstrap_auroc_ci(y[:, cols], s[:, cols], present, groups))
    out["q_minus_p"] = paired_bootstrap_diff(y[:, cols], s_p[:, cols], s_q[:, cols], present, groups)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--isic2019-root", type=Path, required=True)
    parser.add_argument("--isic2019-embeddings", type=Path, default=Path("embeddings_derm"))
    parser.add_argument("--isic2020-root", type=Path, required=True)
    parser.add_argument("--milk-root", type=Path, required=True)
    parser.add_argument("--derm-root", type=Path, required=True, help="metadata.csv + images/ (collection ISIC 399)")
    parser.add_argument("--panderm", type=Path, required=True, help="third_party/panderm/panderm_large.pth")
    parser.add_argument("--out", type=Path, default=Path("results/panderm_derm12345.json"))
    args = parser.parse_args()

    fp = file_fingerprint(args.panderm)
    if fp != KNOWN_CHECKPOINTS["panderm"]:
        raise ValueError(f"poids PanDerm {fp}, attendu {KNOWN_CHECKPOINTS['panderm']}")
    meta = {"checkpoint_sha256": fp, "image_size": PANDERM_IMAGE_SIZE, "transform": SKIN_TRANSFORM}

    lists = {s: ImageList(args.isic2019_root, s) for s in ("train", "val", "test")}
    P19 = {
        s: load_embeddings(args.isic2019_embeddings, "isic2019", d.keys, "panderm", preprocess="none")
        for s, d in lists.items()
    }
    backbone = PanDermBackbone(args.panderm)
    check = preflight(backbone, args.isic2019_root, P19["train"], lists["train"].keys)
    print("Contrôle préalable :", check, flush=True)

    # ISIC 2020 (embeddings du 18 nov., déjà en cache)
    m20 = pd.read_csv(args.isic2020_root / "train.csv")
    paths20 = {p.stem: p for p in args.isic2020_root.rglob("*.jpg")}
    items20 = [
        Item(f"isic2020/{n}", g, lambda p=paths20[n]: Image.open(p), lambda p=paths20[n]: file_fingerprint(p))
        for n, g in zip(m20["image_name"], m20["patient_id"], strict=True)
    ]
    P20 = embed(items20, backbone, EmbeddingCache(Path("embeddings/isic2020_panderm_large.npz"), meta=meta))
    held = held_out_patients(m20["patient_id"])
    y20 = isic2020_targets(m20["diagnosis"])

    # Têtes
    with contextlib.redirect_stdout(io.StringIO()):
        head_p = fit_head(P19["train"], lists["train"].targets)
        head_q = fit_head(np.vstack([P19["train"], P20[~held]]), np.vstack([lists["train"].targets, y20[~held]]))
    gaps = {}
    with np.load(REFERENCE) as ref:
        for split in ("val", "test"):
            if list(ref[f"{split}_keys"]) != lists[split].keys:
                raise ValueError(f"clés {split} différentes de l'archive")
            archived = require_finite(ref[f"PanDerm_{split}"], f"scores PanDerm archivés {split}")
            gaps[split] = float(np.abs(scores(head_p, P19[split]) - archived).max())
            check_gap(gaps[split], 1e-5, f"P ne redonne pas les scores PanDerm archivés ({split})")

    # Seuils de Q : validation ISIC 2020 (option C) et validation ISIC 2019 (comparaison)
    v20_y, v20_s = m20["target"].to_numpy()[held], scores(head_q, P20[held])[:, MEL]
    v19_y, v19_s = lists["val"].targets[:, MEL], scores(head_q, P19["val"])[:, MEL]
    thresholds = {
        source: {pol: choose_threshold(y, s, pol, TARGET) for pol in ("f1", "sensitivity")}
        for source, (y, s) in {"isic2020_val": (v20_y, v20_s), "isic2019_val": (v19_y, v19_s)}.items()
    }

    # DERM12345
    md = pd.read_csv(args.derm_root / "metadata.csv")
    md = md[md["image_type"] == "dermoscopic"].reset_index(drop=True)
    paths_d = {p.stem: p for p in (args.derm_root / "images").glob("*.jpg")}
    missing = set(md["isic_id"]) - set(paths_d)
    if missing:
        raise FileNotFoundError(f"{len(missing)} images DERM12345 absentes, ex. {sorted(missing)[:3]}")
    items_d = [
        Item(f"derm12345/{i}", g, lambda p=paths_d[i]: Image.open(p), lambda p=paths_d[i]: file_fingerprint(p))
        for i, g in zip(md["isic_id"], md["patient_id"], strict=True)
    ]
    print("Embeddings PanDerm DERM12345...", flush=True)
    Pd = embed(items_d, backbone, EmbeddingCache(Path("embeddings/derm12345_panderm_large.npz"), meta=meta))
    duplicate = near_duplicates(Pd, np.vstack([P19["train"], P20]))
    keep = ~duplicate
    y_d, g_d = derm_targets(md["diagnosis_3"])[keep], list(md["patient_id"].to_numpy()[keep])
    sd_p, sd_q = scores(head_p, Pd[keep]), scores(head_q, Pd[keep])

    derm = {
        "images": {
            "total": len(md),
            "near_duplicates": int(duplicate.sum()),
            "evaluated": int(keep.sum()),
            "patients": len(set(g_d)),
            "melanomas": int(y_d[:, MEL].sum()),
        },
        **compare(y_d, sd_p, sd_q, g_d),
        "thresholds_q": {
            source: {pol: at_threshold(y_d[:, MEL], sd_q[:, MEL], t, g_d) for pol, t in pols.items()}
            for source, pols in thresholds.items()
        },
    }
    low = derm["head_q"]["melanoma"]["ci"][0]
    criterion = {"auroc": derm["head_q"]["melanoma"]["auroc"], "lower_bound": low, "threshold": CRITERION}
    criterion["met"] = bool(low >= CRITERION)

    # MILK10k (embeddings du 18 nov.) et test ISIC 2019 : Q ne doit pas perdre ailleurs
    images = pd.read_csv(args.milk_root / "metadata.csv")
    dm = images[images["image_type"] == "dermoscopic"].drop_duplicates("lesion_id")
    gt = pd.read_csv(args.milk_root / "supplements" / "training_gt.csv").merge(
        dm[["lesion_id", "isic_id"]], on="lesion_id"
    )
    keys_m = [f"milk10k/{i}" for i in gt["isic_id"]]
    cache_m = EmbeddingCache(Path("embeddings/milk10k_panderm_large.npz"), meta=meta)
    if cache_m.missing(keys_m):
        raise FileNotFoundError("embeddings PanDerm MILK10k incomplets")
    Pm = cache_m.get(keys_m)
    milk = compare(milk_targets(gt), scores(head_p, Pm), scores(head_q, Pm), list(gt["lesion_id"]))
    test19 = paired_bootstrap_diff(
        lists["test"].targets,
        scores(head_p, P19["test"]),
        scores(head_q, P19["test"]),
        list(CLASS_NAMES),
        list(lists["test"].lesions),
    )

    report = {
        "run": run_info(),
        "plan": "fixé le 19 novembre 2026 avant tout embedding DERM12345 (docstring)",
        "backbone_panderm": fp,
        "preflight": check,
        "training_q": {
            "isic2019": len(P19["train"]),
            "isic2020": int((~held).sum()),
            "isic2020_held_out": int(held.sum()),
            "isic2020_held_out_melanomas": int(v20_y.sum()),
        },
        "head_p": {"head_sha256": head_sha256(head_p.state_dict()), "archive_check_max_abs_diff": gaps},
        "head_q": {"head_sha256": head_sha256(head_q.state_dict()), "mel_thresholds": thresholds},
        "criterion_derm12345": criterion,
        "derm12345": derm,
        "milk10k_q_minus_p": {"MEL": milk["q_minus_p"]["MEL"], "macro": milk["q_minus_p"]["macro"]},
        "milk10k": milk,
        "isic2019_test_q_minus_p": {"MEL": test19["MEL"], "macro": test19["macro"]},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print("DERM12345 :", derm["images"])
    print("  Q :", derm["head_q"]["melanoma"], "| P :", derm["head_p"]["melanoma"])
    print("  Q − P :", derm["q_minus_p"]["MEL"], "| macro :", derm["q_minus_p"]["macro"])
    print("  Critère principal :", criterion)
    for source, pols in derm["thresholds_q"].items():
        for pol, r in pols.items():
            print(f"  seuil {source}/{pol} : sens {r['sensitivity']:.3f} spéc {r['specificity']:.3f}")
    print("MILK10k Q − P :", report["milk10k_q_minus_p"])
    print("Test ISIC 2019 Q − P :", report["isic2019_test_q_minus_p"])
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()

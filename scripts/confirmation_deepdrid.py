#!/usr/bin/env python3
"""Troisième confirmation rétine, sur DeepDRiD (photos classiques, CPU).

Plan fixé le 20 novembre 2026, avant tout calcul d'embedding RETFound ou Green sur
DeepDRiD. Seuls la structure des fichiers et le décompte des stades ont été lus :
2 000 images, 500 patients (4 images chacun, 2 par œil), stades 0 à 4 :
914 / 222 / 398 / 354 / 112 ; qualité jugée insuffisante par les experts sur 1 062 images.

Pourquoi : Green n'est confirmé que sur une base indépendante sur deux (Messidor-2 oui,
IDRiD sans écart) ; le tuteur a accepté une troisième base si le calcul reste léger
(17 et 18 nov.). DeepDRiD n'apparaît dans aucun des deux pré-entraînements (Green :
AIROGS, DDR, ODIR ; RETFound : MEH-MIDAS, EyePACS). Dans le projet, elle n'a servi
qu'à la qualité d'image (embeddings DINOv2) et au test du routeur : ses stades de
rétinopathie n'ont jamais été lus par une tête, ses images jamais passées dans RETFound
ni dans Green. Licence CC BY-SA 4.0.

Données : les trois découpages officiels réunis (rien n'y est appris) : entraînement
(1 200) et validation (400), stade de l'œil photographié (`left_eye_DR_Level` ou
`right_eye_DR_Level`, une seule des deux colonnes remplie par image ; on prend celle qui
l'est : sur 11 images d'entraînement, elle ne correspond pas à l'œil du nom de fichier,
stade patient cohérent) ; évaluation (400), `Challenge1_labels.xlsx`.
Groupe = patient (découpages à patients disjoints, vérifié).

Mesures, identiques à Messidor-2 (`confirmation_external.py`) :
1. Modèle par défaut : tête « rétinopathie » apprise sur l'entraînement RFMiD pour chaque
   modèle (réglages de la baseline, graine 42). Critère : écart d'AUROC « toute
   rétinopathie » (stade >= 1), Green − RETFound, bootstrap apparié sur les patients
   (1 000 tirages), borne basse > 0.
2. Dépistage (tête et seuil servis par `predict.py --dr-screening`, inchangés) :
   sensibilité « toute rétinopathie », IC sur les patients ; critère : borne basse >= 0,80.
   Rapportés : spécificité, VPP, sensibilité « à adresser » (stade >= 2).
Secondaires, sans critère : les deux mesures sur les seules images de qualité suffisante ;
AUROC de chaque modèle, « à adresser ».

Conséquences prévues : mesure 1 confirmée → Green confirmé sur deux bases indépendantes
sur trois ; sinon → une sur trois, choix de Green par défaut rediscuté avec le tuteur.
Mesure 2 atteinte → réglage documenté avec ces chiffres ; sinon → limites ajoutées, seuil
inchangé.

Réserves : une seule source chinoise ; beaucoup d'images de mauvaise qualité ; aucun
dédoublonnage décrit dans les articles de Green et de RETFound.

Exemple :
    python scripts/confirmation_deepdrid.py --root data/raw/deepdrid/regular_fundus_images \\
        --rfmid-root data/raw/rfmid --embeddings embeddings
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.confirmation_external import (  # noqa: E402
    CACHE_MODEL,
    CHECKPOINTS,
    CRITERION_SENSITIVITY,
    SCREENING_HEAD,
    SCREENING_THRESHOLDS,
    SIZES,
    embeddings,
    file_fingerprint_text,
    screening,
)
from scripts.external_validation import quiet_scores  # noqa: E402
from scripts.green_analysis import cached  # noqa: E402
from scripts.pad_ufes20_skin import auroc_ci  # noqa: E402
from scripts.robustness_checks import load  # noqa: E402
from src.data.deepdrid import SPLITS, load_split, read_xlsx  # noqa: E402
from src.eval.metrics import paired_bootstrap_diff  # noqa: E402
from src.eval.provenance import run_info  # noqa: E402
from src.models.embedding_cache import file_fingerprint  # noqa: E402
from src.pipeline.served_head import load_served_head  # noqa: E402

PLAN = "fixé le 20 novembre 2026 avant tout calcul d'embedding RETFound/Green sur DeepDRiD (docstring du script)"


def grades(root: Path, split: str) -> pd.Series:
    """Stade de rétinopathie de chaque image (index : image_id)."""
    folder, labels = SPLITS[split]
    if split == "eval":
        df = read_xlsx(root / folder / "Challenge1_labels.xlsx")
        df["image_id"] = df["image_id"].astype(str)
        grade = pd.to_numeric(df["DR_Levels"], errors="raise")
    else:
        df = pd.read_csv(root / folder / labels)
        df["image_id"] = df["image_id"].astype(str)
        left, right = df["left_eye_DR_Level"], df["right_eye_DR_Level"]
        if (left.notna() == right.notna()).any():
            raise ValueError(f"DeepDRiD {split} : il faut exactement un stade d'œil par image")
        grade = left.fillna(right)
    if grade.isna().any() or not grade.isin([0, 1, 2, 3, 4]).all():
        raise ValueError(f"DeepDRiD {split} : stade manquant ou hors de 0 à 4")
    if df["image_id"].duplicated().any():
        raise ValueError(f"DeepDRiD {split} : image en double dans les stades")
    return pd.Series(grade.astype(int).to_numpy(), index=df["image_id"])


def deepdrid_table(root: Path) -> pd.DataFrame:
    """Clé, chemin, stade, patient et qualité des 2 000 images des trois découpages."""
    frames, patients = [], {}
    for split in ("train", "val", "eval"):
        df = load_split(root, split)
        g = grades(root, split)
        if set(g.index) != set(df["image_id"]):
            raise ValueError(f"DeepDRiD {split} : stades et notes de qualité ne couvrent pas les mêmes images")
        df["grade"] = df["image_id"].map(g).astype(int)
        patients[split] = set(df["patient"])
        df["key"] = f"deepdrid/{split}/" + df["image_id"]
        frames.append(df)
    for a, b in (("train", "val"), ("train", "eval"), ("val", "eval")):
        if patients[a] & patients[b]:
            raise ValueError(f"patients communs aux découpages {a} et {b}")
    df = pd.concat(frames, ignore_index=True)
    missing = [str(p) for p in df["path"] if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images DeepDRiD absentes, ex. {missing[:3]}")
    df["group"] = "deepdrid/" + df["patient"]
    df["good_quality"] = df["Overall quality"] == 1
    return df[["key", "path", "grade", "group", "good_quality"]]


def measures(y, grade, groups, p_ret, p_green, p_screen, threshold) -> dict:
    diff = paired_bootstrap_diff(y[:, None], p_ret[:, None], p_green[:, None], ["DR"], groups)["DR"]
    return {
        "images": len(y),
        "groups": len(set(groups)),
        "by_grade": {int(g): int((grade == g).sum()) for g in np.unique(grade)},
        "auroc_any_dr": {"retfound": auroc_ci(y, p_ret, groups), "green": auroc_ci(y, p_green, groups)},
        "auroc_referable": {
            m: auroc_ci((grade >= 2).astype(np.float32), p, groups)
            for m, p in (("retfound", p_ret), ("green", p_green))
        },
        "green_minus_retfound": diff,
        "dr_screening": screening(y, p_screen, threshold, groups, grade),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--rfmid-root", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, default=Path("embeddings"))
    parser.add_argument("--out", type=Path, default=Path("results/confirmation_deepdrid.json"))
    parser.add_argument("--snapshot", type=Path, default=Path("results/confirmation_deepdrid.npz"))
    args = parser.parse_args()

    table = deepdrid_table(args.root)
    grade = table["grade"].to_numpy()
    y = (grade >= 1).astype(np.float32)
    groups = list(table["group"])

    train, val = load("rfmid", args.rfmid_root, "train"), load("rfmid", args.rfmid_root, "val")
    p, X_green = {}, None
    for model in ("retfound", "green"):
        X = embeddings(table, model, args.embeddings / f"deepdrid_crop-mask_{SIZES[model]}_{model}.npz")
        X_tr = cached(args.embeddings, CACHE_MODEL[model], "rfmid", train.image_keys)
        X_val = cached(args.embeddings, CACHE_MODEL[model], "rfmid", val.image_keys)
        scores = quiet_scores(X_tr, train.df["DR"].to_numpy(np.float32)[:, None], np.vstack([X_val, X]))
        p[model] = scores[len(X_val) :]
        if model == "green":
            X_green = X

    served = load_served_head(SCREENING_HEAD, file_fingerprint(CHECKPOINTS["green"]), SCREENING_THRESHOLDS)
    p_screen = served.probabilities(X_green)[:, 0]
    threshold = served.thresholds["classes"]["DR"]["threshold_sensitivity"]["threshold"]

    main_m = measures(y, grade, groups, p["retfound"], p["green"], p_screen, threshold)
    good = table["good_quality"].to_numpy()
    good_groups = [g for g, k in zip(groups, good, strict=True) if k]
    good_m = measures(
        y[good], grade[good], good_groups, p["retfound"][good], p["green"][good], p_screen[good], threshold
    )
    report = {
        "run": run_info(),
        "plan": PLAN,
        "dataset": "deepdrid",
        "all_images": main_m,
        "good_quality_only": good_m,
        "dr_screening_threshold": threshold,
        "dr_screening_identity": served.identity,
        "criteria": {
            "default_model": {
                "diff": main_m["green_minus_retfound"],
                "confirmed": bool(main_m["green_minus_retfound"]["ci"][0] > 0),
            },
            "screening": {
                "sensitivity": main_m["dr_screening"]["sensitivity"],
                "lower_bound": main_m["dr_screening"]["sensitivity_ci"][0],
                "threshold": CRITERION_SENSITIVITY,
                "met": bool(main_m["dr_screening"]["sensitivity_ci"][0] >= CRITERION_SENSITIVITY),
            },
        },
        "inputs": {
            "annotations_fingerprint": {
                p.name: file_fingerprint(p) for p in sorted(args.root.rglob("*")) if p.suffix in (".csv", ".xlsx")
            },
            "images_list_fingerprint": file_fingerprint_text(
                "\n".join(f"{k}\t{g}\t{s}" for k, g, s in zip(table["key"], groups, grade, strict=True))
            ),
        },
        "local_snapshot": f"{args.snapshot.name} (clés, groupes, stades, qualité, scores ; local, non versionné)",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    np.savez(
        args.snapshot,
        keys=table["key"].to_numpy(str),
        groups=np.array(groups),
        grade=grade,
        good_quality=good,
        p_retfound=p["retfound"],
        p_green=p["green"],
        p_dr_screening=p_screen,
    )
    for name, m in (("toutes", main_m), ("qualité suffisante", good_m)):
        s = m["dr_screening"]
        print(f"[{name}] {m['images']} images, {m['groups']} patients, stades {m['by_grade']}")
        print(f"  AUROC toute RD : {m['auroc_any_dr']}")
        print(f"  Green − RETFound : {m['green_minus_retfound']}")
        print(
            f"  Dépistage : sensibilité {s['sensitivity']:.3f} {s['sensitivity_ci']}, "
            f"spécificité {s['specificity']:.3f}, VPP {s['ppv']:.3f}, à adresser {s['referable_sensitivity']:.3f}"
        )
    print("Critères :", report["criteria"])
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()

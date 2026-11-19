#!/usr/bin/env python3
"""Instantané des entrées et des scores des études PanDerm (constat E de l'audit du 19 novembre 2026).

Les trois études PanDerm (`panderm_external.py`, `panderm_derm12345.py`,
`panderm_p_thresholds.py`) enregistrent des résultats agrégés, sans empreinte des
annotations, des listes ordonnées clés/groupes, des exclusions ni des scores. Une étiquette
ou un patient modifié, à effectif égal, passerait inaperçu à la relance. Même besoin que
pour `confirmation_external.py` (16 nov.). Les scripts et plans exécutés ne sont pas
modifiés : ce script reconstruit leurs entrées avec leurs propres fonctions.

Mode par défaut : reconstruit entrées, exclusions (quasi-doublons) et têtes A, P, Q ;
vérifie qu'elles redonnent les AUROC du mélanome et les seuils archivés (tolérance 1e-4) ;
écrit
- `results/panderm_inputs.json` (versionné) : empreintes des annotations, des listes
  ordonnées clés/groupes/cibles, des masques d'exclusion et des embeddings utilisés ;
  identités des têtes ; arguments, environnement (`pip freeze`) ; contrôle de reproduction ;
- `results/panderm_snapshot_<base>.npz` (local, non versionné) : clés, groupes, cibles,
  exclusions et scores des têtes, base par base.

`--check` : recalcule seulement les empreintes des entrées et les compare au manifeste ;
code de sortie non nul et liste des écarts au premier changement, avant toute métrique.

Exemple :
    python scripts/panderm_snapshot.py --isic2019-root data/raw/isic2019 --isic2020-root data/raw/isic2020 \\
        --milk-root data/raw/milk10k --derm-root data/raw/derm12345 \\
        --panderm third_party/panderm/panderm_large.pth --dinov2 third_party/dinov2/dinov2_vitb14_timm.pth
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.dermatology_analysis import ImageList  # noqa: E402
from scripts.isic2020_external import near_duplicates  # noqa: E402
from scripts.panderm_derm12345 import MEL, derm_targets, held_out_patients, scores  # noqa: E402
from scripts.robustness_checks import fit_head  # noqa: E402
from scripts.skin_decision_thresholds import TARGET  # noqa: E402
from scripts.skin_transfer_milk10k import isic2020_targets, milk_targets  # noqa: E402
from src.data.isic_dataset import SKIN_TRANSFORM  # noqa: E402
from src.eval.decision import choose_threshold  # noqa: E402
from src.eval.provenance import run_info  # noqa: E402
from src.models.classifier_head import head_sha256  # noqa: E402
from src.models.embedding_cache import KNOWN_CHECKPOINTS, EmbeddingCache, file_fingerprint, load_embeddings  # noqa: E402
from src.models.timm_backbones import DINOV2_IMAGE_SIZE, PANDERM_IMAGE_SIZE  # noqa: E402
from src.pipeline.served_head import load_served_head  # noqa: E402

MANIFEST = Path("results/panderm_inputs.json")
SKIN_HEAD = Path("heads/isic2019_dinov2b_head.pt")
TOLERANCE = 1e-4


def digest(*arrays) -> str:
    """SHA-256 (16 caractères) d'une suite de tableaux ou de listes, ordre compris."""
    h = hashlib.sha256()
    for a in arrays:
        a = np.asarray(a)
        h.update(str((a.dtype.str, a.shape)).encode())
        h.update(a.tobytes() if a.dtype != object else "\x1f".join(map(str, a.ravel())).encode())
    return h.hexdigest()[:16]


# Constat F (audit du 19 nov.) : diagnostics DERM12345 (diagnosis_3) hors des huit classes,
# comptés négatifs partout par le plan du 19 nov. ; « » = lésion maligne sans diagnostic
# précis (12 lésions). Toute autre valeur est une entrée inattendue (faute de frappe d'une
# catégorie MEL, nouvelle version de la base) et arrête l'analyse.
DERM_OUT_OF_SCOPE = {
    "Lymphangioma",
    "Lentigo simplex",
    "Kaposi sarcoma",
    "Vascular spider",
    "Paget disease",
    "Ink-spot lentigo",
    "Dermatofibrosarcoma protuberans",
    "",
}


def check_derm_metadata(md: pd.DataFrame) -> None:
    """Identifiants uniques, patients présents, diagnostics tous connus (classe ou hors périmètre)."""
    from scripts.panderm_derm12345 import DERM_TO_ISIC

    if not md["isic_id"].is_unique:
        raise ValueError(f"DERM12345 : {int(md['isic_id'].duplicated().sum())} identifiant(s) en double")
    if md["patient_id"].isna().any():
        raise ValueError(f"DERM12345 : {int(md['patient_id'].isna().sum())} image(s) sans patient")
    unexpected = set(md["diagnosis_3"].fillna("")) - set(DERM_TO_ISIC) - DERM_OUT_OF_SCOPE
    if unexpected:
        raise ValueError(f"DERM12345 : diagnostics inattendus {sorted(unexpected)}")


def check_unique_stems(paths: list[Path], base: str) -> None:
    """Les scripts associent clé et image par `{p.stem: p}` : un doublon serait écrasé en silence."""
    stems = [p.stem for p in paths]
    if len(set(stems)) != len(stems):
        raise ValueError(f"{base} : {len(stems) - len(set(stems))} nom(s) d'image en double")


def cached(path: Path, keys: list[str], meta: dict) -> np.ndarray:
    cache = EmbeddingCache(path, meta=meta)
    if cache.missing(keys):
        raise FileNotFoundError(f"embeddings incomplets dans {path}")
    return cache.get(keys)


def load_inputs(args) -> dict:
    """Entrées des trois études, lues exactement comme dans leurs scripts."""
    p_meta = {"checkpoint_sha256": KNOWN_CHECKPOINTS["panderm"], "image_size": PANDERM_IMAGE_SIZE}
    p_meta["transform"] = SKIN_TRANSFORM
    a_meta = {"checkpoint_sha256": file_fingerprint(args.dinov2), "image_size": DINOV2_IMAGE_SIZE}
    lists = {s: ImageList(args.isic2019_root, s) for s in ("train", "val", "test")}
    m20 = pd.read_csv(args.isic2020_root / "train.csv")
    keys20 = [f"isic2020/{n}" for n in m20["image_name"]]
    images = pd.read_csv(args.milk_root / "metadata.csv")
    dm = images[images["image_type"] == "dermoscopic"].drop_duplicates("lesion_id")
    gt = pd.read_csv(args.milk_root / "supplements" / "training_gt.csv").merge(
        dm[["lesion_id", "isic_id"]], on="lesion_id"
    )
    keys_m = [f"milk10k/{i}" for i in gt["isic_id"]]
    md = pd.read_csv(args.derm_root / "metadata.csv")
    md = md[md["image_type"] == "dermoscopic"].reset_index(drop=True)
    check_derm_metadata(md)
    for base, root in (("isic2020", args.isic2020_root), ("milk10k", args.milk_root), ("derm12345", args.derm_root)):
        check_unique_stems(sorted(root.rglob("*.jpg")), base)
    keys_d = [f"derm12345/{i}" for i in md["isic_id"]]
    held = held_out_patients(m20["patient_id"])
    return {
        "annotations": {
            **{f"isic2019/{p.name}": p for p in sorted(args.isic2019_root.glob("ISIC_2019_*.csv"))},
            "isic2020/train.csv": args.isic2020_root / "train.csv",
            "milk10k/metadata.csv": args.milk_root / "metadata.csv",
            "milk10k/training_gt.csv": args.milk_root / "supplements" / "training_gt.csv",
            "derm12345/metadata.csv": args.derm_root / "metadata.csv",
        },
        "bases": {
            **{
                f"isic2019_{s}": {"keys": d.keys, "groups": list(d.lesions), "targets": d.targets}
                for s, d in lists.items()
            },
            "isic2020": {
                "keys": keys20,
                "groups": list(m20["patient_id"]),
                "targets": isic2020_targets(m20["diagnosis"]),
                "melanoma": m20["target"].to_numpy(),
                "threshold_split_held_out": held,
            },
            "milk10k": {"keys": keys_m, "groups": list(gt["lesion_id"]), "targets": milk_targets(gt)},
            "derm12345": {"keys": keys_d, "groups": list(md["patient_id"]), "targets": derm_targets(md["diagnosis_3"])},
        },
        "embeddings": {
            **{
                f"P_isic2019_{s}": load_embeddings(args.isic2019_embeddings, "isic2019", d.keys, "panderm", "none")
                for s, d in lists.items()
            },
            **{
                f"A_isic2019_{s}": load_embeddings(
                    args.isic2019_embeddings, "isic2019", lists[s].keys, "dinov2", "none"
                )
                for s in ("train", "test")
            },
            "P_isic2020": cached(Path("embeddings/isic2020_panderm_large.npz"), keys20, p_meta),
            "A_isic2020": cached(Path("embeddings/isic2020_dinov2_vitb14.npz"), keys20, a_meta),
            "P_milk10k": cached(Path("embeddings/milk10k_panderm_large.npz"), keys_m, p_meta),
            "A_milk10k": cached(Path("embeddings/milk10k_dinov2_vitb14.npz"), keys_m, a_meta),
            "P_derm12345": cached(Path("embeddings/derm12345_panderm_large.npz"), keys_d, p_meta),
        },
    }


def fingerprints(inputs: dict) -> dict:
    out = {
        "annotations": {name: file_fingerprint(p) for name, p in inputs["annotations"].items()},
        "bases": {
            name: {k: digest(v) for k, v in base.items()} | {"n": len(base["keys"])}
            for name, base in inputs["bases"].items()
        },
        "embeddings": {name: digest(X.astype(np.float32)) for name, X in inputs["embeddings"].items()},
    }
    return out


def differences(old: dict, new: dict, prefix: str = "") -> list[str]:
    diffs = []
    for k in sorted(set(old) | set(new)):
        a, b = old.get(k), new.get(k)
        if isinstance(a, dict) and isinstance(b, dict):
            diffs += differences(a, b, f"{prefix}{k}/")
        elif a != b:
            diffs.append(f"{prefix}{k} : {a} → {b}")
    return diffs


def pip_freeze() -> list[str]:
    run = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=False)
    return sorted(run.stdout.split())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--isic2019-root", type=Path, required=True)
    parser.add_argument("--isic2019-embeddings", type=Path, default=Path("embeddings_derm"))
    parser.add_argument("--isic2020-root", type=Path, required=True)
    parser.add_argument("--milk-root", type=Path, required=True)
    parser.add_argument("--derm-root", type=Path, required=True)
    parser.add_argument("--panderm", type=Path, required=True)
    parser.add_argument("--dinov2", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="comparer les entrées au manifeste, sans métrique")
    args = parser.parse_args()
    if file_fingerprint(args.panderm) != KNOWN_CHECKPOINTS["panderm"]:
        raise ValueError("poids PanDerm inattendus")

    inputs = load_inputs(args)
    current = fingerprints(inputs)
    if args.check:
        if not MANIFEST.exists():
            raise SystemExit(f"{MANIFEST} absent : lancer d'abord sans --check")
        saved = json.loads(MANIFEST.read_text(encoding="utf-8"))["inputs"]
        saved.pop("exclusions", None)  # dérivées des embeddings, déjà comparés
        diffs = differences(saved, current)
        if diffs:
            raise SystemExit("Entrées modifiées depuis l'instantané :\n  " + "\n  ".join(diffs))
        print("Entrées identiques au manifeste.")
        return

    E, B = inputs["embeddings"], inputs["bases"]
    with contextlib.redirect_stdout(io.StringIO()):
        head_p = fit_head(E["P_isic2019_train"], B["isic2019_train"]["targets"])
        held = B["isic2020"]["threshold_split_held_out"]
        head_q = fit_head(
            np.vstack([E["P_isic2019_train"], E["P_isic2020"][~held]]),
            np.vstack([B["isic2019_train"]["targets"], B["isic2020"]["targets"][~held]]),
        )
    head_a = load_served_head(SKIN_HEAD, file_fingerprint(args.dinov2)).head

    # Exclusions des scripts : quasi-doublons d'apprentissage
    excluded = {
        "isic2020": near_duplicates(E["P_isic2020"], E["P_isic2019_train"])
        | near_duplicates(E["A_isic2020"], E["A_isic2019_train"]),
        "milk10k": near_duplicates(E["P_milk10k"], E["P_isic2019_train"])
        | near_duplicates(E["A_milk10k"], E["A_isic2019_train"]),
        "derm12345": near_duplicates(E["P_derm12345"], np.vstack([E["P_isic2019_train"], E["P_isic2020"]])),
    }
    current["exclusions"] = {k: {"count": int(m.sum()), "mask": digest(m)} for k, m in excluded.items()}

    s = {
        "isic2020": {"P": scores(head_p, E["P_isic2020"]), "Q": scores(head_q, E["P_isic2020"])},
        "milk10k": {"P": scores(head_p, E["P_milk10k"]), "Q": scores(head_q, E["P_milk10k"])},
        "derm12345": {"P": scores(head_p, E["P_derm12345"]), "Q": scores(head_q, E["P_derm12345"])},
        "isic2019_val": {"P": scores(head_p, E["P_isic2019_val"]), "Q": scores(head_q, E["P_isic2019_val"])},
        "isic2019_test": {"P": scores(head_p, E["P_isic2019_test"]), "Q": scores(head_q, E["P_isic2019_test"])},
    }
    s["isic2020"]["A"] = scores(head_a, E["A_isic2020"])
    s["milk10k"]["A"] = scores(head_a, E["A_milk10k"])
    s["isic2019_test"]["A"] = scores(head_a, E["A_isic2019_test"])

    def auroc(base: str, head: str, y: np.ndarray) -> float:
        keep = ~excluded.get(base, np.zeros(len(y), bool))
        return float(roc_auc_score(y[keep], s[base][head][keep, MEL]))

    y20, ym, yd = B["isic2020"]["melanoma"], B["milk10k"]["targets"][:, MEL], B["derm12345"]["targets"][:, MEL]
    ext = json.loads(Path("results/panderm_external.json").read_text(encoding="utf-8"))
    d12 = json.loads(Path("results/panderm_derm12345.json").read_text(encoding="utf-8"))
    thr = json.loads(Path("results/panderm_p_thresholds.json").read_text(encoding="utf-8"))
    v20 = (y20[held], {h: s["isic2020"][h][held, MEL] for h in ("P", "Q")})
    v19 = (B["isic2019_val"]["targets"][:, MEL], {h: s["isic2019_val"][h][:, MEL] for h in ("P", "Q")})
    checks = {
        "isic2020_P": (auroc("isic2020", "P", y20), ext["isic2020"]["head_p"]["auroc"]),
        "isic2020_A": (auroc("isic2020", "A", y20), ext["isic2020"]["head_a"]["auroc"]),
        "milk10k_P": (auroc("milk10k", "P", ym), ext["milk10k"]["head_p"]["melanoma"]["auroc"]),
        "milk10k_A": (auroc("milk10k", "A", ym), ext["milk10k"]["head_a"]["melanoma"]["auroc"]),
        "derm12345_P": (auroc("derm12345", "P", yd), d12["derm12345"]["head_p"]["melanoma"]["auroc"]),
        "derm12345_Q": (auroc("derm12345", "Q", yd), d12["derm12345"]["head_q"]["melanoma"]["auroc"]),
    }
    for src, (y, by_head) in (("isic2020_val", v20), ("isic2019_val", v19)):
        for pol in ("f1", "sensitivity"):
            checks[f"threshold_P_{src}_{pol}"] = (
                choose_threshold(y, by_head["P"], pol, TARGET),
                thr["mel_thresholds"][src][pol],
            )
            checks[f"threshold_Q_{src}_{pol}"] = (
                choose_threshold(y, by_head["Q"], pol, TARGET),
                d12["head_q"]["mel_thresholds"][src][pol],
            )
    reproduction = {k: {"recomputed": a, "archived": b, "gap": abs(a - b)} for k, (a, b) in checks.items()}
    failed = [k for k, r in reproduction.items() if not r["gap"] <= TOLERANCE]

    snapshots = {}
    for base in ("isic2020", "milk10k", "derm12345", "isic2019_val", "isic2019_test"):
        path = Path(f"results/panderm_snapshot_{base}.npz")
        b = B[base]
        extra = {f"scores_{h}": v for h, v in s[base].items()}
        if base in excluded:
            extra["excluded"] = excluded[base]
        if base == "isic2020":
            extra["melanoma"], extra["threshold_split_held_out"] = b["melanoma"], held
        np.savez(path, keys=np.array(b["keys"]), groups=np.array(b["groups"], str), targets=b["targets"], **extra)
        snapshots[path.name] = file_fingerprint(path)

    freeze = pip_freeze()
    report = {
        "run": run_info(),
        "purpose": "constat E de l'audit du 19 novembre 2026 : entrées et scores des études PanDerm",
        "argv": sys.argv[1:],
        "environment": {"pip_freeze_sha256": digest(np.array(freeze, object)), "packages": len(freeze)},
        "inputs": current,
        "heads": {
            "P": head_sha256(head_p.state_dict()),
            "Q": head_sha256(head_q.state_dict()),
            "A": head_sha256(head_a.state_dict()),
            "P_matches_archive": head_sha256(head_p.state_dict()) == d12["head_p"]["head_sha256"],
            "Q_matches_archive": head_sha256(head_q.state_dict()) == d12["head_q"]["head_sha256"],
        },
        "reproduction": {"tolerance": TOLERANCE, "failed": failed, "checks": reproduction},
        "local_snapshots": snapshots,
    }
    MANIFEST.write_text(json.dumps(report, indent=1, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    for k, r in reproduction.items():
        print(f"{k:38s} recalculé {r['recomputed']:.6f}  archivé {r['archived']:.6f}  écart {r['gap']:.1e}")
    print("Exclusions :", {k: v["count"] for k, v in current["exclusions"].items()})
    print("Têtes identiques à l'archive :", report["heads"]["P_matches_archive"], report["heads"]["Q_matches_archive"])
    print(f"-> {MANIFEST}")
    if failed:
        raise SystemExit(f"reproduction hors tolérance : {failed}")


if __name__ == "__main__":
    main()

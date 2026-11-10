#!/usr/bin/env python3
"""Détecteur de défauts techniques appris sur de VRAIS défauts annotés (CPU).

Plan fixé le 10 novembre 2026, avant tout run, après l'échec du détecteur appris sur
des défauts simulés (`scripts/quality_learned_detector.py`) :
- mêmes embeddings DINOv2 ViT-B/14 figés, même géométrie (`crop-mask` 512 px, 224 px)
  et même cache ;
- positifs d'entraînement : images du découpage d'entraînement ODIR portant un
  mot-clé technique (poussière comprise), hors patients de la relecture humaine ;
  négatifs : les 600 images propres du train déjà tirées pour le détecteur simulé
  (hors patients de l'étude des défauts réels) ;
- régression logistique standardisée, C = 1, classes équilibrées ; seuil 0,5 ;
- tests identiques au détecteur simulé, avec le même comparateur (luminance sous le
  P2 des témoins train, ou flou sous 4,16) :
  A. validation de l'étude des défauts réels (mots-clés, poussière comprise) contre
     ses 120 témoins ; AUROC par image (bootstrap par patient), taux par patient ;
  B. relecture de Marc : patients avec défaut vu contre patients sans défaut
     confirmé ; indéterminés rapportés à part.
- secondaire, descriptif : sensibilité du test A par type de mot-clé (poussière ou
  défaut général), effectifs trop petits pour conclure sur les défauts généraux.

Réserves : mots-clés d'ODIR non spécifiques (la poussière domine : 165 des 183 images
annotées du train) ; un seul relecteur humain, non spécialiste ; validation déjà
utilisée par les études précédentes, exploratoire. Aucune règle de rejet ajoutée.

Exemple :
    python scripts/quality_real_detector.py --odir-root data/raw/odir \\
        --checkpoint third_party/dinov2/dinov2_vitb14_timm.pth
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.quality_learned_detector import (  # noqa: E402
    BLUR_THRESHOLD,
    LUMINANCE_P2,
    N_BASE,
    TRANSFORM,
    embed_keys,
    retina,
    review_state,
    sha256_file,
)
from scripts.quality_real_defects import auc_with_groups, patient_rate  # noqa: E402
from src.data.odir_quality import source_records  # noqa: E402
from src.eval.provenance import run_info  # noqa: E402
from src.models.embedding_cache import EmbeddingCache, file_fingerprint  # noqa: E402
from src.models.timm_backbones import DINOV2_IMAGE_SIZE, DINOv2Backbone  # noqa: E402

THRESHOLD = 0.5
STATES = ("defect", "clear", "indeterminate")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--odir-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--study-dir", type=Path, default=Path("data/quality_review"))
    parser.add_argument("--review-dir", type=Path, default=Path("data/quality_review_v2"))
    parser.add_argument("--cache", type=Path, default=Path("embeddings/quality_dinov2_vitb14.npz"))
    parser.add_argument("--out", type=Path, default=Path("results/quality_real_detector.json"))
    args = parser.parse_args()

    images = args.odir_root / "ODIR-5K" / "ODIR-5K" / "Training Images"
    study = json.loads((args.study_dir / "source_manifest.json").read_text())["records"]
    measures = {r["filename"]: r["scores"] for r in json.loads((args.study_dir / "measurements.json").read_text())}
    review_records = json.loads((args.review_dir / "source_manifest.json").read_text())["records"]
    mapping = json.loads((args.review_dir / "review_mapping.json").read_text())
    groups = {r["filename"]: r["group"] for r in review_records}
    review_patients = {groups[f] for f in mapping.values()}

    # Négatifs : même tirage que le détecteur simulé (graine 0, 600 images).
    study_patients = {r["group"] for r in study}
    records, _ = source_records(args.odir_root)
    clean = [r for r in records if r["split"] == "train" and not r["tags"] and r["group"] not in study_patients]
    rng = np.random.default_rng(0)
    negatives = [clean[i] for i in sorted(rng.choice(len(clean), N_BASE, replace=False))]
    positives = [r for r in records if r["split"] == "train" and r["tags"] and r["group"] not in review_patients]

    fingerprint = file_fingerprint(args.checkpoint)
    cache = EmbeddingCache(
        args.cache, meta={"checkpoint_sha256": fingerprint, "image_size": DINOV2_IMAGE_SIZE, "transform": TRANSFORM}
    )
    backbone = DINOv2Backbone(args.checkpoint)

    def make(key: str) -> np.ndarray:
        kind, name = key.split("/", 1)
        if kind not in ("clean", "real"):
            raise ValueError(key)
        return retina(images / name)

    def features(rows: list[dict], kind: str = "real") -> np.ndarray:
        return embed_keys([f"{kind}/{r['filename']}" for r in rows], make, backbone, cache)

    print(f"Entraînement : {len(positives)} images annotées, {len(negatives)} propres", flush=True)
    X_train = np.vstack([features(positives), features(negatives, "clean")])
    y_train = np.concatenate([np.ones(len(positives)), np.zeros(len(negatives))])
    scaler = StandardScaler().fit(X_train)
    model = LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced").fit(scaler.transform(X_train), y_train)

    def proba(rows: list[dict]) -> np.ndarray:
        return model.predict_proba(scaler.transform(features(rows)))[:, 1]

    def current(rows: list[dict]) -> np.ndarray:
        s = [measures[r["filename"]] for r in rows]
        return np.array([m["mean_luminance"] < LUMINANCE_P2 or m["blur_score"] < BLUR_THRESHOLD for m in s])

    print("Test A : validation ODIR", flush=True)
    val = [r for r in study if r["split"] == "val"]
    y_val = np.array([bool(r["tags"]) for r in val])
    p_val = proba(val)
    test_a = {"positive_images": int(y_val.sum()), "control_images": int((~y_val).sum()), "detectors": {}}
    pos = [r for r, y in zip(val, y_val, strict=True) if y]
    neg = [r for r, y in zip(val, y_val, strict=True) if not y]
    for name, flags in (("learned_real", p_val >= THRESHOLD), ("current", current(val))):
        by_type = {}
        for label, keep in (
            ("dust_only", lambda r: r["tags"] == ["lens dust"]),
            ("general", lambda r: any(t != "lens dust" for t in r["tags"])),
        ):
            sel = [(r, f) for r, f, y in zip(val, flags, y_val, strict=True) if y and keep(r)]
            by_type[label] = patient_rate([r for r, _ in sel], np.array([f for _, f in sel], dtype=bool))
        test_a["detectors"][name] = {
            "sensitivity": patient_rate(pos, flags[y_val]),
            "false_alert_rate": patient_rate(neg, flags[~y_val]),
            "sensitivity_by_type": by_type,
        }
    test_a["detectors"]["learned_real"]["auroc_image"] = auc_with_groups(val, y_val.astype(int), p_val)

    print("Test B : relecture de Marc", flush=True)
    reviews = list(csv.DictReader((args.review_dir / "review.csv").open()))
    rows = [{"filename": mapping[r["code"]], "group": groups[mapping[r["code"]]]} for r in reviews]
    states = np.array([review_state(r) for r in reviews])
    p_rev = proba(rows)
    test_b = {
        "images": {s: int((states == s).sum()) for s in STATES},
        "unit": "patient (au moins un œil signalé) ; indéterminés rapportés à part",
        "detectors": {},
    }
    for name, flags in (("learned_real", p_rev >= THRESHOLD), ("current", current(rows))):
        part = {s: ([r for r, t in zip(rows, states, strict=True) if t == s], flags[states == s]) for s in STATES}
        test_b["detectors"][name] = {
            "sensitivity": patient_rate(*part["defect"]),
            "false_alert_rate": patient_rate(*part["clear"]),
            "flagged_indeterminate": patient_rate(*part["indeterminate"]),
        }

    report = {
        "run": run_info(),
        "plan": "fixé le 10 novembre 2026 avant tout run (docstring du script)",
        "backbone": {"model": "dinov2_vitb14", "checkpoint_sha256": fingerprint, "transform": TRANSFORM},
        "training": {
            "positive_images": len(positives),
            "positive_tags": {
                t: sum(t in r["tags"] for r in positives) for t in sorted({t for r in positives for t in r["tags"]})
            },
            "negative_images": len(negatives),
            "excluded_review_patients": len(review_patients),
        },
        "threshold": THRESHOLD,
        "test_a_odir_validation": test_a,
        "test_b_human_review": test_b,
        "inputs_sha256": {
            "study_manifest": sha256_file(args.study_dir / "source_manifest.json"),
            "review": sha256_file(args.review_dir / "review.csv"),
            "review_mapping": sha256_file(args.review_dir / "review_mapping.json"),
        },
        "local_archive": f"{args.out.with_suffix('.npz').name} (scores, modèle ; local, non versionné)",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    np.savez(
        args.out.with_suffix(".npz"),
        val_files=np.array([r["filename"] for r in val]),
        val_scores=p_val,
        review_codes=np.array([r["code"] for r in reviews]),
        review_scores=p_rev,
        coef=model.coef_,
        intercept=model.intercept_,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
    )
    for test, r in (("A", test_a), ("B", test_b)):
        for name, d in r["detectors"].items():
            s, f = d["sensitivity"], d["false_alert_rate"]
            print(
                f"test {test} {name:12s} sensibilité {s['detected']}/{s['patients']}  "
                f"fausses alertes {f['detected']}/{f['patients']}"
            )
    print(f"AUROC test A : {test_a['detectors']['learned_real']['auroc_image']}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()

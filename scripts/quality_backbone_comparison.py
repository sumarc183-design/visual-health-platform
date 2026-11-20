#!/usr/bin/env python3
"""Détecteur de qualité DeepDRiD : DINOv2-B, Green ou RETFound ? (CPU, exploratoire)

Plan fixé le 20 novembre 2026, avant tout calcul de détecteur sur les embeddings Green ou
RETFound. Aucune nouvelle image n'est passée dans un modèle : les embeddings des 2 000
images DeepDRiD existent déjà (DINOv2-B : détecteur du 10 nov. ; Green et RETFound :
confirmation rétinopathie du 20 nov.).

Pourquoi : le détecteur actuel (DINOv2-B, notes d'experts DeepDRiD, 10 nov.) classe les
images avec une AUROC d'environ 0,7 ; à 10 % de fausses alertes, il ne repère que 40 % des
images insuffisantes. Un meilleur classement améliorerait les deux à la fois. Green et
RETFound sont spécialisés dans la rétine ; la confirmation du 20 nov. suggère que Green
réagit à la qualité de l'image (son avance sur la rétinopathie tenait aux mauvaises
photos). Rien ne garantit qu'ils distinguent mieux la qualité.

Statut : **exploratoire**. L'évaluation DeepDRiD a déjà été lue le 10 nov. pour le
détecteur DINOv2-B, et il n'existe pas ici de base propre pour vérifier le transfert : ODIR
et la relecture de Marc ont déjà servi, et ODIR fait partie du pré-entraînement de Green
(un transfert sur ODIR l'avantagerait). Le transfert n'est donc pas mesuré.

Méthode, identique pour les trois modèles et à celle du 10 nov. : cible = « Overall
quality » = 0 (insuffisante) ; standardisation puis régression logistique (C = 1),
apprises sur l'entraînement DeepDRiD (1 200 images, 300 patients). Le détecteur DINOv2-B
recalculé doit redonner les scores archivés (`results/quality_deepdrid_detector.npz`).

Correction du contrôle, avant toute lecture (20 nov., premier lancement arrêté par ce
contrôle ; scores Green et RETFound calculés en mémoire, ni affichés ni enregistrés) : le détecteur DINOv2-B recalculé sur ce
PC (Windows) s'écarte de l'archive (Mac) de 9e-3 en probabilité, avec la même version de
scikit-learn et une standardisation identique ; seuls les coefficients de la régression
diffèrent (3e-3) : écart numérique de l'optimisation entre machines. Embeddings gardés en
float32 comme le 10 nov. ; tolérance 1e-2 sur les probabilités et 1e-3 sur l'AUROC de
validation archivée (0,721).

Mesures sur la **validation** (400 images, 100 patients) :
1. Principal : écart d'AUROC à DINOv2-B, IC par bootstrap apparié sur les patients
   (1 000 tirages) ; un modèle « fait mieux » si la borne basse est > 0.
2. Seuil choisi sur la validation (règle du 10 nov. : plus petit seuil atteignant 0,90 de
   spécificité) ; part des images insuffisantes repérées à ce seuil, IC par patient.
Seconde lecture de l'évaluation (400 images), annoncée comme telle : AUROC et, au seuil de
la validation, images insuffisantes repérées et fausses alertes.

Conséquences prévues : si Green ou RETFound fait mieux (critère 1), ce détecteur est
proposé au tuteur au Point du 25-11 à la place de DINOv2-B, comme résultat exploratoire ;
si les deux font mieux, celui à la plus forte AUROC de validation. Sinon, la proposition
actuelle reste (DINOv2-B, seuil 0,989).

Exemple :
    python scripts/quality_backbone_comparison.py --deepdrid-root data/raw/deepdrid/regular_fundus_images
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.dr_screening_sensitivity import rate_ci  # noqa: E402
from scripts.pad_ufes20_skin import auroc_ci  # noqa: E402
from scripts.quality_deepdrid_threshold import threshold_for_specificity  # noqa: E402
from src.data.deepdrid import load_split  # noqa: E402
from src.eval.metrics import paired_bootstrap_diff  # noqa: E402
from src.eval.predictions import check_gap  # noqa: E402
from src.eval.provenance import run_info  # noqa: E402
from src.models.embedding_cache import file_fingerprint  # noqa: E402

PLAN = "fixé le 20 novembre 2026 avant tout calcul de détecteur sur Green ou RETFound (docstring du script)"
CACHES = {
    "dinov2": Path("embeddings/quality_dinov2_vitb14.npz"),
    "green": Path("embeddings/deepdrid_crop-mask_392_green.npz"),
    "retfound": Path("embeddings/deepdrid_crop-mask_224_retfound.npz"),
}
ARCHIVE = Path("results/quality_deepdrid_detector.npz")
TARGET_SPECIFICITY = 0.90
MAX_GAP = 1e-2  # probabilités : écart d'optimisation entre machines (voir docstring)
MAX_AUROC_GAP = 1e-3


def embeddings(path: Path, keys: list[str]) -> tuple[np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as z:
        index = {k: i for i, k in enumerate(z["keys"])}
        missing = [k for k in keys if k not in index]
        if missing:
            raise KeyError(f"{path} : {len(missing)} clés absentes, ex. {missing[:3]}")
        X = z["X"][[index[k] for k in keys]].astype(np.float32)
        meta = json.loads(str(z["meta"]))
    if not np.isfinite(X).all():
        raise ValueError(f"{path} : embeddings non finis")
    return X, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--deepdrid-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/quality_backbone_comparison.json"))
    args = parser.parse_args()

    splits = {s: load_split(args.deepdrid_root, s) for s in ("train", "val", "eval")}
    patients = {s: set(d["patient"]) for s, d in splits.items()}
    pairs = (("train", "val"), ("train", "eval"), ("val", "eval"))
    if any(patients[a] & patients[b] for a, b in pairs):
        raise ValueError("patients partagés entre découpages DeepDRiD")
    keys = {s: [f"deepdrid/{s}/{i}" for i in d["image_id"]] for s, d in splits.items()}
    y = {s: (d["Overall quality"] == 0).astype(int).to_numpy() for s, d in splits.items()}
    groups = {s: list(d["patient"]) for s, d in splits.items()}

    scores, metas = {}, {}
    for name, path in CACHES.items():
        X = {}
        for s in splits:
            X[s], metas[name] = embeddings(path, keys[s])
        scaler = StandardScaler().fit(X["train"])
        model = LogisticRegression(C=1.0, max_iter=5000).fit(scaler.transform(X["train"]), y["train"])
        scores[name] = {s: model.predict_proba(scaler.transform(X[s]))[:, 1] for s in ("val", "eval")}

    with np.load(ARCHIVE) as archive:
        gap = max(float(np.abs(scores["dinov2"][s] - archive[f"{s}_scores"]).max()) for s in ("val", "eval"))
    check_gap(gap, MAX_GAP, "le détecteur DINOv2-B recalculé ne redonne pas les scores du 10 novembre")
    archived_auroc = json.loads(Path("results/quality_deepdrid_detector.json").read_text(encoding="utf-8"))
    archived_auroc = archived_auroc["results"]["deepdrid_val"]["learned"]["auroc_image"]["value"]
    auroc_gap = abs(float(roc_auc_score(y["val"], scores["dinov2"]["val"])) - archived_auroc)
    check_gap(auroc_gap, MAX_AUROC_GAP, "AUROC de validation DINOv2-B différente de l'archive")

    report = {
        "run": run_info(),
        "plan": PLAN,
        "status": "exploratoire (évaluation déjà lue le 10 nov. ; transfert non mesuré)",
        "dinov2_reproduction_max_gap": gap,
        "dinov2_val_auroc_gap_to_archive": auroc_gap,
        "inputs": {
            "caches": {n: {"file": p.name, "sha256": file_fingerprint(p), "meta": metas[n]} for n, p in CACHES.items()},
            "deepdrid": {
                s: {"images": len(d), "patients": len(patients[s]), "insufficient": int(y[s].sum())}
                for s, d in splits.items()
            },
        },
        "models": {},
    }
    for name in CACHES:
        threshold = threshold_for_specificity(scores[name]["val"], y["val"], TARGET_SPECIFICITY)
        entry = {"threshold_val_spec_0_90": threshold}
        for s in ("val", "eval"):
            flags = scores[name][s] >= threshold
            entry[s] = {
                "auroc": auroc_ci(y[s], scores[name][s], groups[s]),
                "sensitivity": float(flags[y[s] == 1].mean()),
                "sensitivity_ci": rate_ci(y[s], flags, groups[s], True),
                "false_alert_rate": float(flags[y[s] == 0].mean()),
            }
        if name != "dinov2":
            for s in ("val", "eval"):
                entry[s]["auroc_minus_dinov2"] = paired_bootstrap_diff(
                    y[s][:, None], scores["dinov2"][s][:, None], scores[name][s][:, None], ["q"], groups[s]
                )["q"]
        report["models"][name] = entry

    better = {n: report["models"][n]["val"]["auroc_minus_dinov2"]["ci"][0] > 0 for n in ("green", "retfound")}
    winners = [n for n, ok in better.items() if ok]
    chosen = max(winners, key=lambda n: report["models"][n]["val"]["auroc"]["auroc"]) if winners else "dinov2"
    report["criterion"] = {"better_than_dinov2_on_val": better, "proposed_detector": chosen}

    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"DINOv2-B recalculé : écart {gap:.1e} aux scores archivés")
    for name, m in report["models"].items():
        for s in ("val", "eval"):
            r = m[s]
            diff = r.get("auroc_minus_dinov2")
            extra = f", écart {diff['diff']:+.3f} [{diff['ci'][0]:+.3f} ; {diff['ci'][1]:+.3f}]" if diff else ""
            print(
                f"{name:9s} {s:4s} AUROC {r['auroc']['auroc']:.3f}{extra} | repérées {r['sensitivity']:.2f} "
                f"{[round(c, 2) for c in r['sensitivity_ci']]}, fausses alertes {r['false_alert_rate']:.2f}"
            )
    print("Critère :", report["criterion"])
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()

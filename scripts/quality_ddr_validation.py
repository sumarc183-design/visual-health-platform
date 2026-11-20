#!/usr/bin/env python3
"""Le détecteur de qualité proposé tient-il sur une base indépendante ? Validation sur DDR (CPU).

Plan fixé le 20 novembre 2026, avant tout calcul d'embedding sur DDR pour la qualité.

Pourquoi : le détecteur proposé au tuteur pour le Point du 25-11 (DINOv2-B figé,
régression logistique apprise le 10 nov. sur les notes d'experts de DeepDRiD, seuil
d'avertissement 0,989) n'a été mesuré que sur DeepDRiD (évaluation déjà lue) et, en
transfert, sur ODIR (mots-clés, déjà utilisés). DDR (Chine, CC BY-NC-SA 4.0) contient
1 151 images que ses lecteurs n'ont pas pu grader (stade 5, « non gradable ») parmi
13 673 ; jamais utilisées dans le projet pour la qualité (la confirmation du 12 nov. ne
gardait que les gradables). DINOv2-B n'a pas vu DDR. C'est une base indépendante pour
**ce** détecteur ; elle ne l'est pas pour Green (DDR dans son pré-entraînement), qui n'est
pas mesuré ici.

Ce que DDR mesure : la cible « non gradable » ne couvre que les images trop mauvaises
pour être gradées, pas les défauts légers qui comptent aussi dans « qualité insuffisante »
de DeepDRiD. On mesure donc surtout la détection des défauts graves.

Méthode :
- détecteur exactement celui du 10 nov. : coefficients, ordonnée et standardisation lus
  dans `results/quality_deepdrid_detector.npz`, vérifiés en redonnant les scores archivés
  de la validation DeepDRiD ; aucun réapprentissage ;
- embeddings DDR calculés sur Kaggle (`scripts/quality_ddr_embeddings.py`), même
  préparation, poids de mêmes tenseurs (empreinte vérifiée) ;
- cible : stade 5 ; toutes les images des trois découpages DDR ;
- IC par bootstrap (1 000 tirages) sur des groupes horaires approchant le patient (images
  prises à moins de 2 min d'intervalle, `time_groups`) : DDR ne donne pas d'identifiant
  patient.

Critères, fixés avant calcul :
1. AUROC « non gradable », borne basse de l'IC ≥ 0,70 (niveau du détecteur sur
   DeepDRiD) ;
2. au seuil d'avertissement 0,989 : fausses alertes sur les images gradables ≤ 20 %
   (estimation ponctuelle ; 10 % sur DeepDRiD, 21 % au transfert ODIR).
Rapportés : part des non gradables signalées à 0,989 et à 0,5, IC ; fausses alertes par
stade de rétinopathie ; mêmes mesures par découpage.

Conséquences prévues : les deux critères atteints → la proposition du 25-11 est présentée
comme validée sur une base indépendante pour les défauts graves, avec le taux mesuré ;
critère 1 seul → le classement tient mais le seuil se transfère mal : seuil à recalibrer
par site, avertissement présenté comme tel ; critère 1 non atteint → la proposition est
retirée en l'état et la question au tuteur le dit.

Exemple :
    python scripts/quality_ddr_validation.py --embeddings embeddings
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.dr_screening_sensitivity import rate_ci  # noqa: E402
from scripts.pad_ufes20_skin import auroc_ci  # noqa: E402
from scripts.quality_ddr_embeddings import EXPECTED_TENSOR_DIGEST  # noqa: E402
from scripts.quality_learned_detector import TRANSFORM  # noqa: E402
from src.data.deepdrid import load_split  # noqa: E402
from src.eval.predictions import check_gap  # noqa: E402
from src.eval.provenance import run_info  # noqa: E402
from src.models.embedding_cache import file_fingerprint  # noqa: E402

PLAN = "fixé le 20 novembre 2026 avant tout calcul d'embedding qualité sur DDR (docstring du script)"
DETECTOR = Path("results/quality_deepdrid_detector.npz")
WARNING_THRESHOLD = 0.989  # seuil d'avertissement du 10 nov. (results/quality_deepdrid_threshold.json)
CRITERION_AUROC = 0.70
CRITERION_FALSE_ALERTS = 0.20


class Detector:
    """Régression logistique archivée du 10 novembre, appliquée telle quelle."""

    def __init__(self, path: Path):
        with np.load(path) as z:
            self.coef = z["coef"].astype(np.float64)[0]
            self.intercept = float(z["intercept"][0])
            self.mean = z["scaler_mean"].astype(np.float64)
            self.scale = z["scaler_scale"].astype(np.float64)
            self.val_scores = z["val_scores"]

    def __call__(self, X: np.ndarray) -> np.ndarray:
        z = ((X.astype(np.float64) - self.mean) / self.scale) @ self.coef + self.intercept
        return 1 / (1 + np.exp(-z))


def load_cache(path: Path, keys: list[str]) -> tuple[np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as z:
        index = {k: i for i, k in enumerate(z["keys"])}
        missing = [k for k in keys if k not in index]
        if missing:
            raise KeyError(f"{path} : {len(missing)} clés absentes, ex. {missing[:3]}")
        return z["X"][[index[k] for k in keys]], json.loads(str(z["meta"]))


def flags_report(y: np.ndarray, p: np.ndarray, threshold: float, groups: list[str]) -> dict:
    flags = p >= threshold
    return {
        "threshold": threshold,
        "ungradable_flagged": float(flags[y == 1].mean()),
        "ungradable_flagged_ci": rate_ci(y, flags, groups, True),
        "false_alert_rate": float(flags[y == 0].mean()),
        "false_alert_rate_ci": [1 - v for v in reversed(rate_ci(y, flags, groups, False))],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embeddings", type=Path, default=Path("embeddings"))
    parser.add_argument("--deepdrid-root", type=Path, default=Path("data/raw/deepdrid/regular_fundus_images"))
    parser.add_argument("--out", type=Path, default=Path("results/quality_ddr_validation.json"))
    args = parser.parse_args()

    detector = Detector(DETECTOR)
    val = load_split(args.deepdrid_root, "val")
    X_val, _ = load_cache(args.embeddings / "quality_dinov2_vitb14.npz", [f"deepdrid/val/{i}" for i in val["image_id"]])
    gap = float(np.abs(detector(X_val) - detector.val_scores).max())
    check_gap(gap, 1e-5, "le détecteur archivé ne redonne pas ses scores de validation DeepDRiD")

    labels = pd.read_csv(args.embeddings / "quality_ddr_labels.csv")
    if labels["key"].duplicated().any() or len(labels) != 13673:
        raise ValueError(f"étiquettes DDR inattendues ({len(labels)} lignes)")
    X, meta = load_cache(args.embeddings / "quality_ddr_dinov2_vitb14.npz", list(labels["key"]))
    if meta.get("checkpoint_tensor_sha256") != EXPECTED_TENSOR_DIGEST or meta.get("transform") != TRANSFORM:
        raise ValueError(f"cache DDR incompatible avec le détecteur : {meta}")
    if not np.isfinite(X).all():
        raise ValueError("embeddings DDR non finis")

    y = labels["ungradable"].to_numpy()
    groups = list(labels["group"].astype(str))
    p = detector(X)
    report = {
        "run": run_info(),
        "plan": PLAN,
        "detector": {"file": DETECTOR.name, "sha256": file_fingerprint(DETECTOR), "deepdrid_val_gap": gap},
        "inputs": {
            "labels_sha256": file_fingerprint(args.embeddings / "quality_ddr_labels.csv"),
            "embeddings_sha256": file_fingerprint(args.embeddings / "quality_ddr_dinov2_vitb14.npz"),
            "meta": meta,
        },
        "images": len(y),
        "ungradable": int(y.sum()),
        "groups": len(set(groups)),
        "auroc": auroc_ci(y, p, groups),
        "at_warning_threshold": flags_report(y, p, WARNING_THRESHOLD, groups),
        "at_0_5": flags_report(y, p, 0.5, groups),
        "false_alerts_by_dr_grade": {
            int(g): float((p[(labels["level"] == g).to_numpy()] >= WARNING_THRESHOLD).mean()) for g in range(5)
        },
        "by_split": {},
    }
    for split in ("train", "valid", "test"):
        m = (labels["split"] == split).to_numpy()
        g = [x for x, k in zip(groups, m, strict=True) if k]
        report["by_split"][split] = {
            "images": int(m.sum()),
            "ungradable": int(y[m].sum()),
            "auroc": auroc_ci(y[m], p[m], g),
            "at_warning_threshold": flags_report(y[m], p[m], WARNING_THRESHOLD, g),
        }
    w = report["at_warning_threshold"]
    report["criteria"] = {
        "auroc_lower_bound": report["auroc"]["ci"][0],
        "auroc_met": bool(report["auroc"]["ci"][0] >= CRITERION_AUROC),
        "false_alert_rate": w["false_alert_rate"],
        "false_alerts_met": bool(w["false_alert_rate"] <= CRITERION_FALSE_ALERTS),
    }
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"Détecteur archivé vérifié (écart {gap:.1e}) ; {len(y)} images DDR, {int(y.sum())} non gradables")
    print(f"AUROC non gradable : {report['auroc']}")
    for name in ("at_warning_threshold", "at_0_5"):
        r = report[name]
        print(
            f"Seuil {r['threshold']} : non gradables signalées {r['ungradable_flagged']:.2f} "
            f"{[round(v, 2) for v in r['ungradable_flagged_ci']]}, fausses alertes {r['false_alert_rate']:.2f}"
        )
    print("Fausses alertes par stade (seuil 0,989) :", report["false_alerts_by_dr_grade"])
    print("Critères :", report["criteria"])
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()

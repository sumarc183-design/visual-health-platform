#!/usr/bin/env python3
"""Dépistage RD : réduire les fausses alertes sans perdre la sensibilité (CPU).

Plan fixé le 13 novembre 2026, avant tout calcul. Suite de `dr_screening_sensitivity.py`
(critère atteint : sensibilité sur DDR 0,87, mais spécificité 0,52 sur DDR et 0,34 sur la
validation EyePACS, soit beaucoup de fausses alertes).

Idée : à sensibilité fixée, la spécificité dépend du pouvoir de classement. RETFound
(gardé par décision du tuteur tant qu'il est meilleur quelque part) voit les images
autrement que Green ; le combiner peut mieux classer.

Candidats (mêmes données que la tête actuelle : entraînement RFMiD colonne DR +
apprentissage EyePACS stade >= 1, même coupe EyePACS par patient, tête et réglages de la
baseline, graine 42 ; embeddings figés `crop-mask`) :
- A « green » : la tête servie (Green 392 px) ;
- B « retfound » : RETFound 224 px ;
- C « fusion » : embeddings Green et RETFound concaténés, une seule tête ;
- D « moyenne » : moyenne des probabilités de A et B.

Seuil de chaque candidat : sensibilité 0,90 sur la validation EyePACS (comme avant).
Sélection (fixée, sur la validation EyePACS seulement) : candidat de plus grande
spécificité en validation à son seuil ; A en cas d'égalité.

Mesure unique sur DDR (12 522 images, mêmes groupes de doublons que la confirmation, IC
par bootstrap des groupes, 1 000 tirages) :
- critère principal : écart apparié de spécificité (sélectionné − A), borne basse de
  l'IC à 95 % > 0, ET sensibilité du sélectionné à borne basse >= 0,80 (critère
  précédent conservé) ;
- si A est sélectionné, rien ne change et le critère est sans objet ;
- secondaires, sans critère : mêmes mesures pour tous les candidats ; AUROC ;
  sensibilité « à adresser » (stade >= 2).

Conséquence prévue : critère atteint → le candidat sélectionné remplace la tête de
`predict.py --dr-screening` (enregistrement vérifié comme pour la tête actuelle) ; sinon
la tête actuelle reste, et réduire les fausses alertes demandera d'autres données
(images du site, stade 1 mieux étiqueté).

Réserves : la validation EyePACS sert au seuil et à la sélection (léger optimisme en
validation, pas sur DDR) ; DDR a déjà servi aux mesures précédentes, sans réglage ;
RETFound et Green ne sont pas à la même résolution (chacun à sa taille d'usage) ; une
seule base de test.

Exemple :
    python scripts/dr_screening_specificity.py --rfmid-root data/raw/rfmid --embeddings embeddings \\
        --reference results/confirmation_ddr.npz --out results/dr_screening_specificity.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.dr_screening_sensitivity import CRITERION, TARGET, screening  # noqa: E402
from scripts.external_validation import quiet_scores  # noqa: E402
from scripts.green_analysis import cached, external_labels  # noqa: E402
from scripts.robustness_checks import load  # noqa: E402
from src.eval.decision import choose_threshold, decision_metrics  # noqa: E402
from src.eval.metrics import _group_resamples  # noqa: E402
from src.eval.provenance import run_info  # noqa: E402

MODELS = {"green": "RETFound-Green", "retfound": "RETFound"}
CANDIDATES = ("green", "retfound", "fusion", "moyenne")


def select(validation_specificity: dict[str, float]) -> str:
    """Plus grande spécificité en validation ; « green » (tête servie) en cas d'égalité."""
    best = max(validation_specificity.values())
    if validation_specificity["green"] == best:
        return "green"
    return max(validation_specificity, key=validation_specificity.get)


def paired_specificity_diff(y: np.ndarray, new: np.ndarray, old: np.ndarray, groups: list[str], seed: int = 0) -> dict:
    """Écart apparié de spécificité (décisions booléennes), IC bootstrap sur les groupes."""
    healthy = y == 0
    draws = []
    for idx in _group_resamples(groups, 1000, seed):
        k = idx[healthy[idx]]
        if len(k):
            draws.append(float((~new[k]).mean() - (~old[k]).mean()))
    point = float((~new[healthy]).mean() - (~old[healthy]).mean())
    return {"diff": point, "ci": np.percentile(draws, [2.5, 97.5]).tolist()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rfmid-root", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=Path("results/confirmation_ddr.npz"))
    parser.add_argument("--out", type=Path, default=Path("results/dr_screening_specificity.json"))
    args = parser.parse_args()

    train = load("rfmid", args.rfmid_root, "train")
    y_rf = train.df["DR"].to_numpy(np.float32)
    eye = external_labels(args.embeddings, "eyepacs")
    y_eye = (eye["diagnosis"].to_numpy() >= 1).astype(np.float32)
    ddr = external_labels(args.embeddings, "ddr")
    keys = list(ddr["key"])
    grade = ddr["diagnosis"].to_numpy()
    y_ddr = (grade >= 1).astype(np.float32)
    with np.load(args.reference) as ref:
        if list(ref["ddr_keys"]) != keys:
            raise ValueError("clés DDR différentes de la confirmation")
        groups = list(ref["ddr_groups"])

    X = {
        name: {
            "rfmid": cached(args.embeddings, model, "rfmid", train.image_keys),
            "eyepacs": cached(args.embeddings, model, "eyepacs", list(eye["key"])),
            "ddr": cached(args.embeddings, model, "ddr", keys),
        }
        for name, model in MODELS.items()
    }
    X["fusion"] = {s: np.hstack([X["green"][s], X["retfound"][s]]) for s in ("rfmid", "eyepacs", "ddr")}
    fit, held = next(
        StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X["green"]["eyepacs"], y_eye, groups=eye["group"])
    )
    y_fit = np.concatenate([y_rf, y_eye[fit]])

    val, ext = {}, {}
    for name in ("green", "retfound", "fusion"):
        Xa = np.vstack([X[name]["rfmid"], X[name]["eyepacs"][fit]])
        p = quiet_scores(Xa, y_fit[:, None], np.vstack([X[name]["eyepacs"][held], X[name]["ddr"]]))
        val[name], ext[name] = p[: len(held)], p[len(held) :]
    val["moyenne"] = (val["green"] + val["retfound"]) / 2
    ext["moyenne"] = (ext["green"] + ext["retfound"]) / 2

    report = {
        "run": run_info(),
        "plan": "fixé le 13 novembre 2026 avant tout calcul (docstring du script)",
        "candidates": {},
    }
    decided = {}
    for name in CANDIDATES:
        threshold = choose_threshold(y_eye[held], val[name], "sensitivity", TARGET)
        decided[name] = ext[name] >= threshold
        report["candidates"][name] = {
            "eyepacs_validation": decision_metrics(y_eye[held], val[name], threshold),
            "ddr": screening(y_ddr, ext[name], threshold, groups, grade),
        }
    chosen = select({n: c["eyepacs_validation"]["specificity"] for n, c in report["candidates"].items()})
    report["selected"] = chosen
    if chosen == "green":
        report["criterion"] = {"applicable": False, "reading": "la tête servie reste (sélectionnée en validation)"}
    else:
        diff = paired_specificity_diff(y_ddr, decided[chosen], decided["green"], groups)
        sens_low = report["candidates"][chosen]["ddr"]["sensitivity_ci"][0]
        met = diff["ci"][0] > 0 and sens_low >= CRITERION
        report["criterion"] = {
            "applicable": True,
            "specificity_diff_vs_green": diff,
            "sensitivity_lower_bound": sens_low,
            "sensitivity_threshold": CRITERION,
            "met": bool(met),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    for name, c in report["candidates"].items():
        v, d = c["eyepacs_validation"], c["ddr"]
        print(
            f"{name:9s} validation spéc. {v['specificity']:.3f} | DDR sens. {d['sensitivity']:.3f} "
            f"{d['sensitivity_ci']}, spéc. {d['specificity']:.3f} {d['specificity_ci']}, AUROC {d['auroc']:.3f}"
        )
    print("Sélectionné :", chosen, "| critère :", report["criterion"])
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()

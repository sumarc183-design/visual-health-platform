#!/usr/bin/env python3
"""DDR : l'avance de RETFound-Green tient-elle à résolution égale ? (exploratoire, CPU)

Plan fixé le 12 novembre 2026, après la confirmation sur DDR (`scripts/confirmation_ddr.py`,
critère confirmé : Green 392 px − RETFound 224 px = +0,139 [+0,130 ; +0,149]) et avant
tout calcul de Green en 224 px sur DDR.

Analyse **exploratoire** : elle ne modifie ni le critère ni la conclusion de la
confirmation. Question : sur EyePACS, l'avance de Green fondait à 224 px (+0,019
[−0,001 ; +0,038]) ; qu'en est-il sur DDR ?

Méthode, identique à la confirmation et à l'analyse « résolution égale » du 8 novembre
(`green_analysis.py --green-size 224`) : Green reçoit des images de 224 px, encodages de
position interpolés depuis 392 px ; tête « rétinopathie » apprise sur l'entraînement RFMiD
avec les embeddings Green 224 px ; mêmes 12 522 images, mêmes groupes de doublons,
bootstrap apparié par groupes (1 000 tirages). Scores RETFound et Green 392 px repris de
l'archive de la confirmation (`results/confirmation_ddr.npz`) ; les scores RETFound
recalculés ici doivent être identiques (contrôle de reproductibilité).

Mesures (toute rétinopathie, tête RFMiD transférée) :
1. Green 224 px − RETFound 224 px, avec IC (groupes de doublons, puis groupes horaires) ;
2. Green 392 px − Green 224 px, avec IC (gain dû à la résolution, ou perte due à
   l'interpolation) ;
3. AUROC de Green 224 px, à format égal et « à adresser ».

Lecture fixée à l'avance : si la borne basse de l'IC de la mesure 1 est > 0 avec les
deux regroupements, l'avance de Green ne s'explique pas seulement par la résolution sur
DDR ; si l'IC contient 0, elle tient surtout à la résolution (ou à l'interpolation),
comme sur EyePACS ; si la borne haute est < 0, Green fait moins bien que RETFound à 224 px.

Réserve : Green n'a pas été entraîné à 224 px ; une baisse peut venir de la résolution
comme de l'interpolation. RETFound à haute résolution (384 px) n'est pas refait sur DDR
(sur EyePACS il ne gagnait rien : 0,651 contre 0,661).

Exemple :
    python scripts/ddr_resolution.py --rfmid-root data/raw/rfmid --embeddings embeddings \\
        --reference results/confirmation_ddr.npz --out results/ddr_resolution.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import green_analysis  # noqa: E402
from scripts.finetune_analysis import paired_external_diff  # noqa: E402
from src.data.ddr_dataset import time_groups  # noqa: E402
from src.eval.provenance import run_info  # noqa: E402

NAME = "ddr"
SIZE = 224


def reading(ci_dup: list[float], ci_time: list[float]) -> str:
    """Lecture fixée à l'avance de la mesure 1 (Green 224 − RETFound)."""
    if ci_dup[0] > 0 and ci_time[0] > 0:
        return "avance de Green au-delà de la résolution"
    if ci_dup[1] < 0 and ci_time[1] < 0:
        return "Green moins bon que RETFound à 224 px"
    return "avance liée surtout à la résolution (ou à l'interpolation)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rfmid-root", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=Path("results/confirmation_ddr.npz"))
    parser.add_argument("--out", type=Path, default=Path("results/ddr_resolution.json"))
    args = parser.parse_args()

    green_analysis.GREEN_SIZE["value"] = SIZE
    archive: dict = {}
    result = green_analysis.external_comparison(args.rfmid_root, args.embeddings, NAME, archive)
    with np.load(args.reference) as ref:
        reference = {k: ref[k] for k in ref.files}
    for k in ("keys", "groups"):
        if not np.array_equal(reference[f"{NAME}_{k}"], archive[f"{NAME}_{k}"]):
            raise ValueError(f"{k} DDR différents de la confirmation")
    retfound_gap = float(np.abs(reference[f"{NAME}_retfound_transfer"] - archive[f"{NAME}_retfound_transfer"]).max())

    keys = list(archive[f"{NAME}_keys"])
    any_dr = (archive[f"{NAME}_grade"] >= 1).astype(np.float32)
    formats = archive[f"{NAME}_formats"]
    groups = list(archive[f"{NAME}_groups"])
    by_time = time_groups([k.rsplit("/", 1)[1] for k in keys])
    retfound = reference[f"{NAME}_retfound_transfer"]
    green224 = archive[f"{NAME}_green_transfer"]
    green392 = reference[f"{NAME}_green_transfer"]

    vs_retfound = result["transfer_green_minus_retfound"]["global"]
    vs_retfound_time = paired_external_diff(any_dr, retfound, green224, formats, by_time)["global"]
    report = {
        "run": run_info(),
        "plan": "fixé le 12 novembre 2026 avant le calcul de Green 224 px sur DDR (docstring) ; exploratoire",
        "reproducibility": {"retfound_transfer_max_abs_diff": retfound_gap},
        "green_224": result["models"]["RETFound-Green"],
        "green224_minus_retfound": {"duplicate_groups": vs_retfound, "time_groups": vs_retfound_time},
        "green392_minus_green224": paired_external_diff(any_dr, green224, green392, formats, groups),
        "within_format_green224_minus_retfound": result["transfer_green_minus_retfound"]["within_format"],
        "reading": reading(vs_retfound["ci"], vs_retfound_time["ci"]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    g = report["green_224"]["transfer_any_dr"]
    print(f"Green 224 px : {g['auroc']:.3f} [{g['ci'][0]:.3f} ; {g['ci'][1]:.3f}]")
    print("Green 224 − RETFound :", report["green224_minus_retfound"])
    print("Green 392 − Green 224 :", report["green392_minus_green224"]["global"])
    print("Écart max. scores RETFound (reproductibilité) :", retfound_gap)
    print("Lecture :", report["reading"])
    print(f"Résultats : {args.out}")


if __name__ == "__main__":
    main()

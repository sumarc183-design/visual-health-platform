#!/usr/bin/env python3
"""Confirmation sur DDR : RETFound-Green contre RETFound, tête RFMiD transférée (CPU).

Plan fixé le 11 novembre 2026, avant tout calcul d'embeddings sur DDR et avant toute
lecture de résultat. Seule la distribution des stades a été comptée (pour écarter les
images non gradables).

Pourquoi : la base externe EyePACS a déjà servi plusieurs fois (choix de variantes) et
son règlement limite l'usage à la compétition. DDR (Chine, CC BY-NC-SA 4.0) n'a jamais
été vue par le projet : c'est la confirmation demandée par l'audit du 6 novembre.

Cohorte : les 12 522 images gradables de DDR (stades 0 à 4 ; 1 151 non gradables
écartées), trois découpages officiels réunis : rien n'est appris sur DDR, sauf la sonde
« dans la base » (secondaire).

Méthode, inchangée par rapport à EyePACS (`green_analysis.external_comparison`, même
code) : prétraitement `crop-mask`, RETFound 224 px, Green 392 px, modèles figés ; tête
« rétinopathie » (colonne DR) apprise sur l'entraînement RFMiD avec les réglages de la
baseline ; seuil « dépistage » choisi sur la validation RFMiD (sensibilité visée 0,90),
jamais sur DDR ; bootstrap apparié par groupes (doublons repérés sur les embeddings
regroupés), 1 000 tirages.

Critère principal (fixé) : écart d'AUROC « toute rétinopathie » (stade >= 1),
Green − RETFound, tête RFMiD transférée. Confirmé si la borne basse de l'IC à 95 % est
> 0 (sur EyePACS : +0,065 [+0,043 ; +0,085]). Il doit aussi l'être avec le contrôle de
regroupement ci-dessous ; sinon, « non confirmé ».

Mesures secondaires, sans critère : AUROC de chaque modèle (toute rétinopathie, « à
adresser » stade >= 2) ; écart à format d'image égal ; sensibilité et spécificité au seuil
RFMiD (sur EyePACS, la sensibilité est tombée à 0,02 et 0,13) ; sonde entraînée dans
DDR (5 plis groupés) et son écart Green − RETFound.

Contrôle de regroupement : DDR ne publie pas d'identifiant patient. Les deux yeux d'un
patient pouvant tomber de part et d'autre d'un tirage, l'écart principal est recalculé en
regroupant les images prises à moins de deux minutes d'intervalle (`time_groups`,
heure lue dans le nom de fichier : 9 320 groupes pour 12 522 images, l'article annonce
9 598 patients). Précision ajoutée le même jour, toujours avant
tout calcul : seules 5 423 images ont un nom horodaté ; les 7 099 autres
(`007-NNNN-NNN.jpg`, un seul centre, aucun préfixe répété) restent chacune seule, faute
d'information sur le patient.

Conséquence prévue : confirmé → Green est le meilleur candidat « modèle par défaut »
pour la rétinopathie diabétique (sous réserve de sa licence) ; non confirmé → aucun
changement de modèle par défaut justifié par la rétinopathie.

Réserves : une seule cohorte (Chine, plusieurs centres et caméras) ; gradation par des
lecteurs formés, pas un consensus de spécialistes ; licence non commerciale.

Exemple :
    python scripts/confirmation_ddr.py --rfmid-root data/raw/rfmid --embeddings embeddings \\
        --out results/confirmation_ddr.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.finetune_analysis import paired_external_diff  # noqa: E402
from scripts.green_analysis import external_comparison, external_labels  # noqa: E402
from src.data.ddr_dataset import time_groups  # noqa: E402
from src.eval.provenance import run_info  # noqa: E402

NAME = "ddr"


def primary(diff: dict) -> dict:
    """Critère principal : borne basse de l'IC de l'écart > 0."""
    return {"diff": diff["diff"], "ci": diff["ci"], "confirmed": diff["ci"][0] > 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rfmid-root", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/confirmation_ddr.json"))
    args = parser.parse_args()

    archive: dict = {}
    report = {
        "run": run_info(),
        "plan": "fixé le 11 novembre 2026 avant tout calcul sur DDR (docstring du script)",
        "ddr": external_comparison(args.rfmid_root, args.embeddings, NAME, archive),
    }
    labels = external_labels(args.embeddings, NAME)
    keys = list(labels["key"])
    if list(archive[f"{NAME}_keys"]) != keys:
        raise ValueError("ordre des clés DDR incohérent entre étiquettes et scores")
    grade = labels["diagnosis"].to_numpy()
    any_dr = (grade >= 1).astype(np.float32)
    formats = labels["format"].to_numpy()
    by_time = time_groups([k.rsplit("/", 1)[1] for k in keys])
    control = paired_external_diff(
        any_dr, archive[f"{NAME}_retfound_transfer"], archive[f"{NAME}_green_transfer"], formats, by_time
    )
    main_diff = report["ddr"]["transfer_green_minus_retfound"]
    first = "global"  # AUROC « toute rétinopathie », toutes images
    report["criterion"] = {
        "measure": first,
        "duplicate_groups": primary(main_diff[first]),
        "time_groups": {"n_groups": len(set(by_time)), **primary(control[first])},
    }
    report["criterion"]["confirmed"] = (
        report["criterion"]["duplicate_groups"]["confirmed"] and report["criterion"]["time_groups"]["confirmed"]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    np.savez_compressed(args.out.with_suffix(".npz"), **archive)
    for model, m in report["ddr"]["models"].items():
        t, s = m["transfer_any_dr"], m["screening_threshold"]["any_dr"]
        print(
            f"{model:15s} : tête RFMiD {t['auroc']:.3f} [{t['ci'][0]:.3f} ; {t['ci'][1]:.3f}] | "
            f"à adresser {m['transfer_referable_dr']['auroc']:.3f} | seuil dépistage : "
            f"sensibilité {s['sensitivity']:.2f}, spécificité {s['specificity']:.2f}"
        )
    print("Critère :", report["criterion"])
    print(f"Résultats : {args.out}")


if __name__ == "__main__":
    main()

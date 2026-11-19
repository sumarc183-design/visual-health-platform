#!/usr/bin/env python3
"""Seuils du mélanome de la tête PanDerm P choisis à la prévalence d'ISIC 2020 (CPU, embeddings en cache).

Plan fixé le 19 novembre 2026, après les résultats de `scripts/panderm_derm12345.py`
(`results/panderm_derm12345.json`) et avant tout calcul de seuil pour P.

Contexte : sur DERM12345, la tête Q (ISIC 2019 + ISIC 2020) atteint le critère de J90
(mélanome 0,953 [0,937 ; 0,967]) mais n'apporte rien au mélanome par rapport à P
(ISIC 2019 seul, Q − P +0,006 [−0,000 ; +0,012]) et dégrade le nævus (−0,263). P atteint
lui aussi la borne de 0,80 sur DERM12345 (0,947 [0,930 ; 0,962]). Décision de Marc le
19 nov. : proposer au tuteur de servir P, avec des seuils fixés à la prévalence d'un
dépistage. Ce choix de tête est fait après avoir vu les résultats (le plan du 19 nov.
prévoyait Q) ; il est documenté comme tel.

Tête : P, refaite à l'identique (`fit_head`, graine 42, apprentissage ISIC 2019 ;
scores archivés redonnés à 1e-5 près). Aucun nouvel apprentissage.

Seuils du mélanome : choisis sur les mêmes 20 % de patients d'ISIC 2020 que pour Q
(`held_out_patients`, graine 42), règles de la tête servie : F1 maximal, et sensibilité
>= 0,90. Pour comparaison, seuils choisis sur la validation ISIC 2019.

Mesures, sans critère (DERM12345 a déjà servi à évaluer P) : sensibilité, spécificité et
VPP du mélanome (IC par patient, 1 000 tirages) à chacun des quatre seuils, sur
DERM12345 et sur les 80 % de patients restants d'ISIC 2020 (jamais vus par P, patients
disjoints de ceux qui fixent les seuils).

Conséquence prévue : résultats soumis au tuteur avec la proposition de servir P ; aucune
modification de `predict.py` avant son accord.

Réserves : bénins de DERM12345 pas tous confirmés par histologie ; un seul tirage des
patients d'ISIC 2020 ; 113 mélanomes seulement pour fixer les seuils.

Exemple :
    python scripts/panderm_p_thresholds.py --isic2019-root data/raw/isic2019 --isic2020-root data/raw/isic2020 \\
        --derm-root data/raw/derm12345 --panderm third_party/panderm/panderm_large.pth
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.dermatology_analysis import ImageList  # noqa: E402
from scripts.panderm_derm12345 import MEL, REFERENCE, at_threshold, derm_targets, held_out_patients, scores  # noqa: E402
from scripts.robustness_checks import fit_head  # noqa: E402
from scripts.skin_decision_thresholds import TARGET  # noqa: E402
from src.data.isic_dataset import SKIN_TRANSFORM  # noqa: E402
from src.eval.decision import choose_threshold  # noqa: E402
from src.eval.predictions import check_gap, require_finite  # noqa: E402
from src.eval.provenance import run_info  # noqa: E402
from src.models.classifier_head import head_sha256  # noqa: E402
from src.models.embedding_cache import KNOWN_CHECKPOINTS, EmbeddingCache, file_fingerprint, load_embeddings  # noqa: E402
from src.models.timm_backbones import PANDERM_IMAGE_SIZE  # noqa: E402


def cached(path: Path, keys: list[str], meta: dict) -> np.ndarray:
    cache = EmbeddingCache(path, meta=meta)
    if cache.missing(keys):
        raise FileNotFoundError(f"embeddings incomplets dans {path}")
    return cache.get(keys)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--isic2019-root", type=Path, required=True)
    parser.add_argument("--isic2019-embeddings", type=Path, default=Path("embeddings_derm"))
    parser.add_argument("--isic2020-root", type=Path, required=True)
    parser.add_argument("--derm-root", type=Path, required=True)
    parser.add_argument("--panderm", type=Path, required=True, help="poids vérifiés, pour l'empreinte des caches")
    parser.add_argument("--out", type=Path, default=Path("results/panderm_p_thresholds.json"))
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

    m20 = pd.read_csv(args.isic2020_root / "train.csv")
    P20 = cached(Path("embeddings/isic2020_panderm_large.npz"), [f"isic2020/{n}" for n in m20["image_name"]], meta)
    held = held_out_patients(m20["patient_id"])
    y20, s20, g20 = m20["target"].to_numpy(), scores(head_p, P20)[:, MEL], list(m20["patient_id"])

    md = pd.read_csv(args.derm_root / "metadata.csv")
    md = md[md["image_type"] == "dermoscopic"].reset_index(drop=True)
    Pd = cached(Path("embeddings/derm12345_panderm_large.npz"), [f"derm12345/{i}" for i in md["isic_id"]], meta)
    y_d, s_d, g_d = derm_targets(md["diagnosis_3"])[:, MEL], scores(head_p, Pd)[:, MEL], list(md["patient_id"])

    sources = {
        "isic2020_val": (y20[held], s20[held]),
        "isic2019_val": (lists["val"].targets[:, MEL], scores(head_p, P19["val"])[:, MEL]),
    }
    thresholds = {
        src: {pol: choose_threshold(y, s, pol, TARGET) for pol in ("f1", "sensitivity")}
        for src, (y, s) in sources.items()
    }
    rest = ~held
    g20_rest = [g for g, r in zip(g20, rest, strict=True) if r]
    evaluations = {
        "derm12345": {
            src: {pol: at_threshold(y_d, s_d, t, g_d) for pol, t in pols.items()} for src, pols in thresholds.items()
        },
        "isic2020_rest": {
            src: {pol: at_threshold(y20[rest], s20[rest], t, g20_rest) for pol, t in pols.items()}
            for src, pols in thresholds.items()
        },
    }

    report = {
        "run": run_info(),
        "plan": "fixé le 19 novembre 2026 avant tout calcul de seuil pour P (docstring)",
        "backbone_panderm": fp,
        "head_p": {"head_sha256": head_sha256(head_p.state_dict()), "archive_check_max_abs_diff": gaps},
        "isic2020": {
            "held_out_images": int(held.sum()),
            "held_out_melanomas": int(y20[held].sum()),
            "rest_images": int(rest.sum()),
            "rest_melanomas": int(y20[rest].sum()),
        },
        "derm12345": {"images": len(md), "melanomas": int(y_d.sum())},
        "mel_thresholds": thresholds,
        "evaluations": evaluations,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    for base, by_src in evaluations.items():
        for src, pols in by_src.items():
            for pol, r in pols.items():
                print(
                    f"{base} | seuil {src}/{pol} ({thresholds[src][pol]:.3f}) : sens {r['sensitivity']:.3f} "
                    f"{[round(x, 3) for x in r['sensitivity_ci']]} spéc {r['specificity']:.3f} "
                    f"{[round(x, 3) for x in r['specificity_ci']]} VPP {r['ppv']:.3f}"
                )
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()

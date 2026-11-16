#!/usr/bin/env python3
"""Messidor-2 : les AUROC basses viennent-elles d'un décalage des scores d'un format à l'autre ? (CPU)

Contrôle **exploratoire**, sans critère ni décision, écrit le 16 novembre 2026 après
`confirmation_external.py --dataset messidor2` (Green 0,747, RETFound 0,667) et avant
ce calcul. Deuxième lecture de Messidor-2 : rien n'y est choisi ni appris.

Constat préalable (lu avant ce script, tailles d'image et stades seulement) : cinq formats
(1440×960 png, 2240×1488 png/jpg, 2304×1536 png/jpg), prévalence de la rétinopathie de
17 % à 62 % selon le format. Un recentrage des scores par appareil (piste du contrôle 8)
supposerait une prévalence comparable : il n'est donc pas testé ici.

Mesures, mêmes scores que la confirmation (têtes RFMiD de chaque modèle, tête et seuil
`--dr-screening`) :
- AUROC globale, AUROC à format égal et leur écart (IC bootstrap sur les examens) ;
- AUROC dans chaque format.
Lecture prévue : écart nettement négatif (borne haute < 0) → une partie de la baisse
vient d'un décalage entre formats ; sinon, la baisse est dans les formats eux-mêmes.

Exemple :
    python scripts/messidor_format_check.py --root data/raw/messidor2 --rfmid-root data/raw/rfmid
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.confirmation_external import (  # noqa: E402
    CACHE_MODEL,
    SCREENING_HEAD,
    SIZES,
    embeddings,
    messidor_table,
)
from scripts.external_validation import quiet_scores  # noqa: E402
from scripts.green_analysis import cached  # noqa: E402
from scripts.pad_ufes20_skin import auroc_ci  # noqa: E402
from scripts.robustness_checks import load  # noqa: E402
from src.eval.metrics import stratification_gap  # noqa: E402
from src.eval.provenance import run_info  # noqa: E402
from src.models.classifier_head import MultiLabelHead  # noqa: E402


def image_format(path: Path) -> str:
    """Format d'origine : « largeur×hauteur extension »."""
    with Image.open(path) as im:
        return f"{im.size[0]}x{im.size[1]}{path.suffix.lower()}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--rfmid-root", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, default=Path("embeddings"))
    parser.add_argument("--out", type=Path, default=Path("results/messidor_format_check.json"))
    args = parser.parse_args()

    table = messidor_table(args.root)
    y = (table["grade"].to_numpy() >= 1).astype(np.float32)
    groups = list(table["group"].astype(str))
    formats = np.array([image_format(p) for p in table["path"]])

    train = load("rfmid", args.rfmid_root, "train")
    scores = {}
    for model in ("retfound", "green"):
        X = embeddings(table, model, args.embeddings / f"messidor2_crop-mask_{SIZES[model]}_{model}.npz")
        X_tr = cached(args.embeddings, CACHE_MODEL[model], "rfmid", train.image_keys)
        scores[model] = quiet_scores(X_tr, train.df["DR"].to_numpy(np.float32)[:, None], X)
        if model == "green":
            head = MultiLabelHead(X.shape[1], 1)
            head.load_state_dict(torch.load(SCREENING_HEAD, map_location="cpu", weights_only=True)["state_dict"])
            with torch.no_grad():
                scores["dr_screening"] = torch.sigmoid(head(torch.from_numpy(X))).numpy()[:, 0]

    report = {
        "run": run_info(),
        "plan": "exploratoire, écrit le 16 novembre 2026 avant ce calcul (docstring)",
        "formats": {
            f: {"images": int((formats == f).sum()), "prevalence": float(y[formats == f].mean())}
            for f in sorted(set(formats))
        },
        "models": {},
    }
    for name, p in scores.items():
        gap = stratification_gap(y[:, None], np.asarray(p).reshape(-1, 1), ["DR"], formats, groups)["DR"]
        per_format = {
            f: auroc_ci(y[m], np.asarray(p).reshape(-1)[m], [g for g, k in zip(groups, m, strict=True) if k])
            for f in sorted(set(formats))
            if len(np.unique(y[m := formats == f])) == 2
        }
        report["models"][name] = {"stratification": gap, "per_format": per_format}
        print(name, {k: gap[k] for k in gap if k != "per_stratum"})
        print("  par format :", {f: round(v["auroc"], 3) for f, v in per_format.items()})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()

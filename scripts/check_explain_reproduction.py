#!/usr/bin/env python3
"""Vérifie que `predict.py --explain` redonne les cartes d'occlusion de la mesure IDRiD (CPU).

La mesure de localisation du 6 novembre (`scripts/localization_maps.py`, archive
`results/localization/localization_green.npz`) a calculé, sur les 81 images de
segmentation IDRiD, la carte de la rétinopathie avec la colonne DR de la tête servie
`heads/rfmid_green_head.pt` (identique à `dr_heads.npz`). Les images de segmentation ne
sont pas toutes sur ce PC : le script cherche, par empreinte de fichier, celles qui sont
identiques octet pour octet à une image de gradation IDRiD présente, et compare pour
celles-là la carte de `Predictor.explain` (ramenée à 56×56 comme l'archive) et le logit
d'origine. Aucune nouvelle mesure.

Exemple :
    python scripts/check_explain_reproduction.py --idrid-root data/raw/idrid_grading
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.predictions import check_gap  # noqa: E402
from src.eval.provenance import run_info  # noqa: E402
from src.explainability.localization import resize_map  # noqa: E402
from src.models.embedding_cache import file_fingerprint  # noqa: E402
from src.pipeline.inference import Predictor  # noqa: E402

ARCHIVE = Path("results/localization/localization_green.npz")
HEADS = Path("results/localization/dr_heads.npz")
HEAD = Path("heads/rfmid_green_head.pt")
CHECKPOINT = Path("third_party/retfound_green/retfoundgreen_statedict.pth")
MAX_GAP = 1e-3  # archive calculée sur GPU, recalcul sur CPU


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--idrid-root", type=Path, required=True, help="IDRiD, partie gradation")
    parser.add_argument("--out", type=Path, default=Path("results/predict_check_explain.json"))
    args = parser.parse_args()

    head = torch.load(HEAD, map_location="cpu", weights_only=True)
    dr = list(head["class_names"]).index("DR")
    with np.load(HEADS) as archived_head:
        gap_head = float(np.abs(head["state_dict"]["fc.weight"][dr].numpy() - archived_head["green_weight"]).max())
    check_gap(gap_head, 0.0, "la tête servie n'est pas celle de la mesure IDRiD")

    archive = np.load(ARCHIVE)
    manifest = json.loads(str(archive["manifest"]))
    wanted = {f: name for name, f in manifest["inputs"]["files"].items() if name.startswith("1. Original Images")}
    found = {}
    for path in sorted(args.idrid_root.rglob("*.jpg")):
        name = wanted.get(file_fingerprint(path))
        if name is not None:
            found[Path(name).stem] = path
    if not found:
        raise SystemExit("aucune image de la mesure IDRiD trouvée sur ce PC")

    predictor = Predictor(HEAD, CHECKPOINT)
    ids = list(archive["ids"])
    rows = []
    for image_id, path in sorted(found.items()):
        i = ids.index(image_id)
        result = predictor.explain(path, "DR")
        heat = resize_map(result["heat"], archive["maps"].shape[1])
        rows.append(
            {
                "image": image_id,
                "file": path.name,
                "logit": result["logit"],
                "logit_archived": float(archive["logits"][i]),
                "map_max_abs_diff": float(np.abs(heat - archive["maps"][i]).max()),
                "map_scale": float(np.abs(archive["maps"][i]).max()),
            }
        )
        print(rows[-1])
    worst = max(max(r["map_max_abs_diff"], abs(r["logit"] - r["logit_archived"])) for r in rows)
    check_gap(worst, MAX_GAP, "predict.py --explain ne redonne pas les cartes de la mesure IDRiD")
    report = {
        "run": run_info(),
        "purpose": "predict.py --explain redonne les cartes d'occlusion de la mesure IDRiD du 6 novembre",
        "archive": ARCHIVE.name,
        "head_sha_matches_archive": gap_head == 0.0,
        "images_available": len(rows),
        "images_in_archive": len(ids),
        "tolerance": MAX_GAP,
        "max_gap": worst,
        "rows": rows,
    }
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"Écart maximal {worst:.1e} sur {len(rows)} images -> {args.out}")


if __name__ == "__main__":
    main()

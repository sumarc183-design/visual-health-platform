#!/usr/bin/env python3
"""Compare le fine-tuning partiel (scripts/finetune_retfound.py) à la tête
linéaire figée (CPU, à partir des prédictions sauvegardées).

1. Validation RFMiD : différence appariée fine-tuning − tête linéaire (mêmes
   images ; la configuration du fine-tuning était fixée à l'avance).
2. Test RFMiD (lu une fois par chaque modèle) : même comparaison.
3. APTOS (rétinopathie diabétique, toute forme) : AUROC globale et à
   appareil égal (format d'origine), pour le fine-tuning et pour la tête
   linéaire RFMiD (recalculée à partir des embeddings en cache), sur les
   mêmes 3 662 images.

Exemple :
    python scripts/finetune_analysis.py --finetune-dir results/finetune --rfmid-root data/raw/rfmid \\
        --embeddings embeddings --out results/finetune_analysis.json
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.robustness_checks import embeddings, head_scores, load  # noqa: E402
from src.eval.metrics import evaluate_multilabel, paired_bootstrap_diff, stratification_gap  # noqa: E402
from src.models.embedding_cache import EmbeddingCache  # noqa: E402


def compare(linear_path: Path, finetune_path: Path) -> dict:
    a, b = np.load(linear_path), np.load(finetune_path)
    if list(a["class_names"]) != list(b["class_names"]):
        raise ValueError("classes différentes entre les deux fichiers de prédictions")
    if set(b["keys"]) != set(a["keys"]):
        raise ValueError(
            f"images différentes : {len(b['keys'])} contre {len(a['keys'])} (essai partiel ?) ; "
            "la comparaison exige le même jeu complet"
        )
    order = {k: i for i, k in enumerate(a["keys"])}
    idx = np.array([order[k] for k in b["keys"]])
    if not np.array_equal(a["y_true"][idx], b["y_true"]):
        raise ValueError("étiquettes différentes entre les deux fichiers de prédictions")
    names = list(b["class_names"])
    diff = paired_bootstrap_diff(b["y_true"], a["probs"][idx], b["probs"], names, list(b["groups"]), n_boot=1000)
    return {
        "linear_auroc_macro": evaluate_multilabel(b["y_true"], a["probs"][idx], names)["auroc_macro"],
        "finetune_auroc_macro": evaluate_multilabel(b["y_true"], b["probs"], names)["auroc_macro"],
        "finetune_minus_linear": diff["macro"],
        "per_class": {k: v for k, v in diff.items() if k != "macro"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--finetune-dir", type=Path, required=True)
    parser.add_argument("--rfmid-root", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    report = {
        "validation": compare(
            args.results / "pred_rfmid_crop-mask_val.npz", args.finetune_dir / "pred_rfmid_finetune_val.npz"
        )
    }
    test_path = args.finetune_dir / "pred_rfmid_finetune_test.npz"
    if test_path.exists():
        report["test"] = compare(args.results / "pred_rfmid_crop-mask_test.npz", test_path)

    aptos_path = args.finetune_dir / "pred_aptos_finetune.npz"
    if aptos_path.exists():
        ft = np.load(aptos_path)
        y = (ft["grade"] >= 1).astype(np.float32)
        keys, formats = list(ft["keys"]), ft["formats"]
        ds = load("rfmid", args.rfmid_root, "train")
        with contextlib.redirect_stdout(io.StringIO()):
            linear = head_scores(
                embeddings(args.embeddings, "rfmid", "crop-mask", ds.image_keys),
                ds.df[["DR"]].to_numpy(np.float32),
                EmbeddingCache(args.embeddings / "aptos_crop-mask_224_RETFound_mae_natureCFP.npz").get(keys),
            )[:, 0]
        report["aptos"] = {
            name: stratification_gap(y[:, None], p[:, None], ["any_dr"], formats, keys, n_boot=1000)["any_dr"]
            for name, p in (("tête linéaire RFMiD", linear), ("fine-tuning partiel", ft["prob_dr"]))
        }

    for split in ("validation", "test"):
        if split in report:
            r = report[split]
            d = r["finetune_minus_linear"]
            print(
                f"{split:10s} : tête linéaire {r['linear_auroc_macro']:.4f} | "
                f"fine-tuning {r['finetune_auroc_macro']:.4f} "
                f"| écart apparié {d['diff']:+.4f} [{d['ci'][0]:+.4f} ; {d['ci'][1]:+.4f}]"
            )
    for name, g in report.get("aptos", {}).items():
        print(
            f"APTOS {name:20s} : AUROC globale {g['auroc']:.3f} | à appareil égal {g['auroc_within_strata']:.3f} "
            f"| écart {g['gap']:+.3f} [{g['gap_ci'][0]:+.3f} ; {g['gap_ci'][1]:+.3f}]"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"Résultats : {args.out}")


if __name__ == "__main__":
    main()

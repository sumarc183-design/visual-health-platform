#!/usr/bin/env python3
"""Dermatologie (ISIC 2019) : modèle spécialisé contre modèles généralistes (CPU).

Plan fixé le 9 novembre 2026, avant tout run, sans réglage :
- trois modèles figés, mêmes images (petit côté à 256 px, recadrage central 224 px) :
  PanDerm-Large (spécialisé peau, CC BY-NC-ND), DINOv2 ViT-L/14 et ViT-B/14
  (généralistes, Apache 2.0) ;
- même tête que le reste du projet (`head_scores`, réglages de la baseline), entraînée
  sur le découpage d'entraînement, 8 classes en un-contre-tous ;
- mesures : AUROC macro des 8 classes (principale) et exactitude équilibrée par
  argmax (mesure officielle du challenge ISIC 2019), sur la validation puis le test
  officiel (sans les « UNK »), lu une seule fois ;
- comparaison principale : PanDerm moins DINOv2 ViT-L (même taille, la différence
  mesure la spécialisation) ; secondaires : PanDerm moins DINOv2 ViT-B, et ViT-L moins
  ViT-B (effet de la taille chez le généraliste) ;
- différences appariées, bootstrap par groupes : lésions (entraînement et validation)
  et quasi-doublons repérés sur les embeddings DINOv2 ViT-L, fixés à l'avance et
  communs aux trois modèles (le test n'a pas d'identifiant de lésion).

Réserves :
- c'est la performance d'une sonde linéaire, pas celle d'un modèle ajusté ;
- ISIC 2019 n'est pas déclaré dans le préentraînement de PanDerm (article, Methods),
  mais ses sources ne sont pas toutes publiques : un recouvrement d'images ne peut pas
  être exclu ;
- PanDerm a été évalué par ses auteurs sur des jeux proches (HAM10000, sous-ensemble
  d'ISIC 2019) : ce n'est pas une mesure aveugle pour lui.

Exemple :
    python scripts/dermatology_analysis.py --isic-root data/raw/isic2019 \\
        --embeddings embeddings --out results/dermatology_isic2019.json
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

from scripts.robustness_checks import head_scores  # noqa: E402
from src.data.isic_dataset import CLASS_NAMES, ISIC2019Dataset  # noqa: E402
from src.eval.duplicates import independent_groups  # noqa: E402
from src.eval.metrics import (  # noqa: E402
    _group_resamples,
    bootstrap_auroc_ci,
    evaluate_multilabel,
    paired_bootstrap_diff,
)
from src.eval.provenance import run_info  # noqa: E402
from src.models.embedding_cache import load_embeddings  # noqa: E402

MODELS = {"PanDerm": "panderm", "DINOv2-L": "dinov2-large", "DINOv2-B": "dinov2"}
GROUPING_MODEL = "DINOv2-L"
COMPARISONS = [("DINOv2-L", "PanDerm"), ("DINOv2-B", "PanDerm"), ("DINOv2-B", "DINOv2-L")]
N_BOOT = 1000


class ImageList:
    """Découpage ISIC sans ouvrir les images (seuls les CSV sont lus)."""

    def __init__(self, root: Path, split: str):
        dataset = ISIC2019Dataset(root, split)
        self.keys = dataset.image_keys
        self.lesions = dataset.groups
        self.targets = dataset.df[CLASS_NAMES].to_numpy(np.float32)


def balanced_accuracy(y: np.ndarray, p: np.ndarray) -> float:
    """Moyenne des rappels par classe, classe prédite = score le plus haut."""
    truth, pred = y.argmax(1), p.argmax(1)
    return float(np.mean([(pred[truth == c] == c).mean() for c in np.unique(truth)]))


def balanced_accuracy_ci(y, p, groups, seed: int = 0) -> list[float]:
    draws = [balanced_accuracy(y[i], p[i]) for i in _group_resamples(groups, N_BOOT, seed)]
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def paired_balanced_accuracy(y, a, b, groups, seed: int = 0) -> dict:
    draws = [
        balanced_accuracy(y[i], b[i]) - balanced_accuracy(y[i], a[i]) for i in _group_resamples(groups, N_BOOT, seed)
    ]
    return {
        "diff": balanced_accuracy(y, b) - balanced_accuracy(y, a),
        "ci": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--isic-root", type=Path, required=True, help="Dossier des quatre CSV officiels")
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/dermatology_isic2019.json"))
    args = parser.parse_args()

    sets = {s: ImageList(args.isic_root, s) for s in ("train", "val", "test")}
    X = {
        model: {
            s: load_embeddings(args.embeddings, "isic2019", d.keys, backbone, preprocess="none")
            for s, d in sets.items()
        }
        for model, backbone in MODELS.items()
    }
    groups, duplicates = {}, {}
    for split in ("val", "test"):
        groups[split], duplicates[split] = independent_groups(X[GROUPING_MODEL][split], sets[split].lesions)

    scores = {}
    for model in MODELS:
        with contextlib.redirect_stdout(io.StringIO()):
            p = head_scores(X[model]["train"], sets["train"].targets, np.vstack([X[model]["val"], X[model]["test"]]))
        n_val = len(sets["val"].keys)
        scores[model] = {"val": p[:n_val], "test": p[n_val:]}

    report = {
        "run": run_info(),
        "plan": "fixé le 9 novembre 2026 avant tout run (docstring du script)",
        "images": {s: len(d.keys) for s, d in sets.items()},
        "classes": CLASS_NAMES,
        "test_class_counts": dict(zip(CLASS_NAMES, sets["test"].targets.sum(0).astype(int).tolist(), strict=True)),
        "groups": {s: {"n": len(set(groups[s])), "near_duplicate_pairs": duplicates[s]} for s in groups},
        "models": {},
        "comparisons": {},
    }
    for model in MODELS:
        report["models"][model] = {}
        for split in ("val", "test"):
            y, p, g = sets[split].targets, scores[model][split], groups[split]
            metrics = evaluate_multilabel(y, p, CLASS_NAMES)
            metrics.update(bootstrap_auroc_ci(y, p, CLASS_NAMES, g, n_boot=N_BOOT))
            report["models"][model][split] = {
                "auroc_macro": metrics["auroc_macro"],
                "auroc_macro_ci": metrics["auroc_macro_ci"],
                "auroc_per_class": metrics["auroc_per_class"],
                "balanced_accuracy": balanced_accuracy(y, p),
                "balanced_accuracy_ci": balanced_accuracy_ci(y, p, g),
            }
    for a, b in COMPARISONS:
        name = f"{b} - {a}"
        report["comparisons"][name] = {}
        for split in ("val", "test"):
            y, g = sets[split].targets, groups[split]
            pa, pb = scores[a][split], scores[b][split]
            diff = paired_bootstrap_diff(y, pa, pb, CLASS_NAMES, g, n_boot=N_BOOT)
            report["comparisons"][name][split] = {
                "auroc_macro": diff["macro"],
                "auroc_per_class": {k: v for k, v in diff.items() if k != "macro"},
                "balanced_accuracy": paired_balanced_accuracy(y, pa, pb, g),
            }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    archive = {f"{s}_keys": np.array(d.keys) for s, d in sets.items() if s != "train"}
    archive.update({f"{s}_groups": np.array(groups[s]) for s in groups})
    archive.update({f"{m}_{s}": scores[m][s].astype(np.float32) for m in MODELS for s in ("val", "test")})
    np.savez_compressed(args.out.with_suffix(".npz"), **archive)
    for model in MODELS:
        t = report["models"][model]["test"]
        print(f"{model:9s} test AUROC macro {t['auroc_macro']:.3f}  exactitude équilibrée {t['balanced_accuracy']:.3f}")
    for name, c in report["comparisons"].items():
        d = c["test"]["auroc_macro"]
        print(f"{name:20s} AUROC {d['diff']:+.3f} [{d['ci'][0]:+.3f} ; {d['ci'][1]:+.3f}]")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()

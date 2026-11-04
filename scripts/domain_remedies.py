#!/usr/bin/env python3
"""Remèdes au biais d'appareil et au mauvais transfert entre bases, sur les
embeddings en cache (CPU). Suite de scripts/robustness_checks.py.

Les contrôles de robustesse ont montré que l'appareil photo reste lisible
dans les embeddings `crop-mask`, et qu'une tête entraînée sur une base perd
nettement sur l'autre (rétinopathie diabétique surtout). Remèdes comparés,
**sur la validation uniquement**, pour les maladies communes (rétinopathie
diabétique, DMLA, myopie) :

- `base` : embeddings tels quels ;
- `center` / `zscore` : normalisation par base (moyenne, et écart-type),
  calculée sur les images d'entraînement de chaque base, sans étiquette ;
- `leace` : effacement linéaire de l'appareil (LEACE, Belrose et al. 2023) :
  projection qui retire des embeddings toute information sur l'appareil
  récupérable par un modèle linéaire, ajustée sur les deux bases ;
- `center+leace` : les deux ;
- et, pour chaque variante, une tête entraînée **sur les deux bases
  ensemble** (multi-source).

Pour chacune : appareil retrouvé à partir des embeddings (exactitude
équilibrée), AUROC macro dans chaque base (toutes classes, pour vérifier que
le remède ne dégrade pas la tâche), AUROC des maladies communes en
transfert et en multi-source. Le multi-source est comparé à la tête de la
base par différence appariée (mêmes images de validation).

Le remède retenu au vu de la validation (multi-source sur embeddings bruts)
est ensuite évalué **une fois** sur le test de chaque base (`--report-test`).

Exemple :
    python scripts/domain_remedies.py --rfmid-root data/raw/rfmid --odir-root data/raw/odir \\
        --embeddings embeddings --out results/domain_remedies.json --report-test
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.robustness_checks import (  # noqa: E402
    MIN_DEVICE_IMAGES,
    SHARED_CLASSES,
    embeddings,
    head_scores,
    image_formats,
    load,
)
from src.eval.metrics import bootstrap_auroc_ci, evaluate_multilabel, paired_bootstrap_diff  # noqa: E402

DATASETS = ("rfmid", "odir")
SHARED = {"rfmid": list(SHARED_CLASSES), "odir": list(SHARED_CLASSES.values())}
NAMES = ["rétinopathie diabétique", "DMLA", "myopie"]
VARIANTS = ("base", "center", "zscore", "leace", "center+leace")
RETAINED = "base"  # multi-source sur embeddings bruts : seul remède sans perte en validation


def leace_eraser(X: np.ndarray, concept: np.ndarray, eps: float = 1e-3):
    """Effacement linéaire d'un concept catégoriel (LEACE, forme fermée) :
    après projection, aucun classifieur linéaire ne prédit le concept mieux
    qu'une constante (sur la distribution d'ajustement), avec le moins de
    déformation possible des embeddings."""
    mean = X.mean(0)
    centered = X - mean
    categories = np.unique(concept)
    Z = (concept[:, None] == categories[None]).astype(float)
    Z -= Z.mean(0)
    cov = centered.T @ centered / len(X)
    cov += eps * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])
    eigval, eigvec = np.linalg.eigh(cov)
    whiten = eigvec @ np.diag(eigval**-0.5) @ eigvec.T
    unwhiten = eigvec @ np.diag(eigval**0.5) @ eigvec.T
    U, s, _ = np.linalg.svd(whiten @ (centered.T @ Z / len(X)), full_matrices=False)
    basis = U[:, s > 1e-8 * s.max()]
    projection = unwhiten @ basis @ basis.T @ whiten
    return lambda Y: Y - (Y - mean) @ projection.T


def device_accuracy(X_train, dev_train, X_val, dev_val) -> float:
    devices = [d for d, n in Counter(dev_train).items() if n >= MIN_DEVICE_IMAGES]
    a, b = np.isin(dev_train, devices), np.isin(dev_val, devices)
    mean, std = X_train[a].mean(0), X_train[a].std(0) + 1e-6
    probe = LogisticRegression(max_iter=3000).fit((X_train[a] - mean) / std, dev_train[a])
    return float(balanced_accuracy_score(dev_val[b], probe.predict((X_val[b] - mean) / std)))


def quiet_head(X_train, y_train, X_eval):
    with contextlib.redirect_stdout(io.StringIO()):  # train_head affiche la loss à chaque epoch
        return head_scores(X_train.astype(np.float32), y_train, X_eval.astype(np.float32))


def make_transform(variant: str, data: dict):
    if variant == "base":
        return lambda name, X: X
    stats = {n: (data[n, "train"]["X"].mean(0), data[n, "train"]["X"].std(0) + 1e-6) for n in DATASETS}
    center = lambda name, X: X - stats[name][0]  # noqa: E731
    if variant == "center":
        return center
    if variant == "zscore":
        return lambda name, X: (X - stats[name][0]) / stats[name][1]
    first = center if variant == "center+leace" else (lambda name, X: X)
    pooled = np.vstack([first(n, data[n, "train"]["X"]) for n in DATASETS])
    devices = np.concatenate([data[n, "train"]["device"] for n in DATASETS])
    erase = leace_eraser(pooled, devices)
    return lambda name, X: erase(first(name, X))


def load_split(name: str, root: Path, split: str, cache_dir: Path) -> dict:
    ds = load(name, root, split)
    return {
        "X": embeddings(cache_dir, name, "crop-mask", ds.image_keys).astype(np.float64),
        "y_shared": ds.df[SHARED[name]].to_numpy(np.float32),
        "y_all": ds.df[ds.class_names].to_numpy(np.float32),
        "classes": ds.class_names,
        "device": np.array([f"{name}:{f}" for f in image_formats(ds)]),
        "groups": ds.groups,
    }


def shared_auroc(y, p) -> dict:
    return evaluate_multilabel(y, p, NAMES)["auroc_per_class"]


def evaluate_variant(variant: str, data: dict) -> dict:
    T = make_transform(variant, data)
    X = {(n, s): T(n, data[n, s]["X"]) for n in DATASETS for s in ("train", "val")}
    both_X = np.vstack([X[n, "train"] for n in DATASETS])
    both_y = np.vstack([data[n, "train"]["y_shared"] for n in DATASETS])
    out = {"device_balanced_accuracy": {}, "in_domain_auroc_macro": {}, "shared": {}}
    for n in DATASETS:
        val = data[n, "val"]
        out["device_balanced_accuracy"][n] = device_accuracy(
            X[n, "train"], data[n, "train"]["device"], X[n, "val"], val["device"]
        )
        probs = quiet_head(X[n, "train"], data[n, "train"]["y_all"], X[n, "val"])
        out["in_domain_auroc_macro"][n] = evaluate_multilabel(val["y_all"], probs, val["classes"])["auroc_macro"]
        other = "odir" if n == "rfmid" else "rfmid"
        in_domain = quiet_head(X[n, "train"], data[n, "train"]["y_shared"], X[n, "val"])
        transfer = quiet_head(X[other, "train"], data[other, "train"]["y_shared"], X[n, "val"])
        multi = quiet_head(both_X, both_y, X[n, "val"])
        out["shared"][n] = {
            "in_domain": shared_auroc(val["y_shared"], in_domain),
            "transfer": shared_auroc(val["y_shared"], transfer),
            "multi_source": shared_auroc(val["y_shared"], multi),
            "multi_source_minus_in_domain": paired_bootstrap_diff(
                val["y_shared"], in_domain, multi, NAMES, val["groups"], n_boot=1000
            ),
        }
    return out


def report_test(data: dict) -> dict:
    """Remède retenu, lu une fois sur le test de chaque base."""
    T = make_transform(RETAINED, data)
    both_X = np.vstack([T(n, data[n, "train"]["X"]) for n in DATASETS])
    both_y = np.vstack([data[n, "train"]["y_shared"] for n in DATASETS])
    out = {}
    for n in DATASETS:
        test = data[n, "test"]
        other = "odir" if n == "rfmid" else "rfmid"
        X_test = T(n, test["X"])
        runs = {
            "in_domain": quiet_head(T(n, data[n, "train"]["X"]), data[n, "train"]["y_shared"], X_test),
            "transfer": quiet_head(T(other, data[other, "train"]["X"]), data[other, "train"]["y_shared"], X_test),
            "multi_source": quiet_head(both_X, both_y, X_test),
        }
        out[n] = {
            label: {
                "auroc_per_class": shared_auroc(test["y_shared"], probs),
                "auroc_per_class_ci": bootstrap_auroc_ci(test["y_shared"], probs, NAMES, test["groups"], n_boot=1000)[
                    "auroc_per_class_ci"
                ],
            }
            for label, probs in runs.items()
        }
        out[n]["multi_source_minus_in_domain"] = paired_bootstrap_diff(
            test["y_shared"], runs["in_domain"], runs["multi_source"], NAMES, test["groups"], n_boot=1000
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rfmid-root", type=Path, required=True)
    parser.add_argument("--odir-root", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report-test", action="store_true", help="Évalue le remède retenu une fois sur le test")
    args = parser.parse_args()
    roots = {"rfmid": args.rfmid_root, "odir": args.odir_root}
    splits = ("train", "val", "test") if args.report_test else ("train", "val")
    data = {(n, s): load_split(n, roots[n], s, args.embeddings) for n in DATASETS for s in splits}

    report = {"selection_split": "val", "variants": {}}
    for variant in VARIANTS:
        report["variants"][variant] = r = evaluate_variant(variant, data)
        dev, macro = r["device_balanced_accuracy"], r["in_domain_auroc_macro"]
        print(
            f"{variant:13s} appareil {dev['rfmid']:.2f}/{dev['odir']:.2f}"
            f" | macro dans la base {macro['rfmid']:.4f}/{macro['odir']:.4f}"
        )
        for n in DATASETS:
            s = r["shared"][n]
            print(
                f"    val {n:5s} "
                + " | ".join(
                    f"{label} " + " ".join(f"{v:.3f}" for v in s[label].values())
                    for label in ("in_domain", "transfer", "multi_source")
                )
                + f" | multi − base : {s['multi_source_minus_in_domain']['macro']['diff']:+.3f} "
                f"[{s['multi_source_minus_in_domain']['macro']['ci'][0]:+.3f} ; "
                f"{s['multi_source_minus_in_domain']['macro']['ci'][1]:+.3f}]"
            )
    if args.report_test:
        report["retained"] = f"multi_source sur embeddings {RETAINED}"
        report["test"] = test = report_test(data)
        for n in DATASETS:
            d = test[n]["multi_source_minus_in_domain"]["macro"]
            print(
                f"test {n:5s} "
                + " | ".join(
                    f"{label} " + " ".join(f"{v:.3f}" for v in test[n][label]["auroc_per_class"].values())
                    for label in ("in_domain", "transfer", "multi_source")
                )
                + f" | multi − base : {d['diff']:+.3f} [{d['ci'][0]:+.3f} ; {d['ci'][1]:+.3f}]"
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"Résultats : {args.out}")


if __name__ == "__main__":
    main()

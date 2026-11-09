#!/usr/bin/env python3
"""Recalibrer les scores par appareil sur un nouveau site, sans étiquette (CPU).

Sur APTOS, la tête RFMiD reconnaît la rétinopathie diabétique à appareil
égal (0,942) mais ses scores se décalent d'un appareil à l'autre, d'où une
AUROC globale de 0,830 (scripts/aptos_device_check.py). Remède testé ici :
normaliser les scores (ou les embeddings) appareil par appareil, avec les
seules images du site, sans diagnostic.

Banc d'essai : transfert entre RFMiD et ODIR (tête entraînée sur une base,
évaluée sur l'autre), pour les maladies communes. Contrairement à APTOS, les
appareils de ces bases photographient malades et sains, ce qui rend la
comparaison interprétable. L'appareil est repéré par le format d'image
d'origine ; les formats à moins de `--min-device-images` images de
référence sont regroupés.

Méthodes (statistiques calculées sur les images train + validation de la
base cible, **sans leurs étiquettes**, appliquées à la validation et au
test) :
- `aucune` : scores tels quels ;
- `centrage_scores` : logit − médiane du logit de l'appareil ;
- `zscore_scores` : (logit − moyenne) / écart-type de l'appareil ;
- `centrage_embeddings` : embedding − moyenne de l'appareil + moyenne de la
  base source, avant la tête.
Le choix se fait sur la validation cible ; le test donne la mesure, avec la
différence appariée par rapport à `aucune`. Une normalisation par appareil
suppose une prévalence comparable d'un appareil à l'autre : la prévalence
par appareil est rapportée pour juger ce risque.

Exemple :
    python scripts/device_recalibration.py --rfmid-root data/raw/rfmid --odir-root data/raw/odir \\
        --embeddings embeddings --out results/device_recalibration.json
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.robustness_checks import SHARED_CLASSES, embeddings, image_formats, load  # noqa: E402
from src.eval.metrics import evaluate_multilabel, paired_bootstrap_diff, stratification_gap  # noqa: E402
from src.eval.provenance import run_info  # noqa: E402
from src.models.classifier_head import MultiLabelHead, train_head  # noqa: E402

COLUMNS = {"rfmid": list(SHARED_CLASSES), "odir": list(SHARED_CLASSES.values())}
NAMES = ["rétinopathie diabétique", "DMLA", "myopie"]
METHODS = ("aucune", "centrage_scores", "zscore_scores", "centrage_embeddings")


def train_logits_fn(X_train: np.ndarray, y_train: np.ndarray, seed: int = 42):
    """Tête de la baseline ; renvoie une fonction embeddings -> logits."""
    torch.manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        list(zip(torch.from_numpy(X_train), torch.from_numpy(y_train), strict=True)), batch_size=16, shuffle=True
    )
    with contextlib.redirect_stdout(io.StringIO()):
        head = train_head(MultiLabelHead(X_train.shape[1], y_train.shape[1]), loader, epochs=30, lr=1e-3, device="cpu")
    head.eval()

    def logits(X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return head(torch.from_numpy(X.astype(np.float32))).numpy()

    return logits


def device_ids(formats: list[str], reference: list[str], min_images: int) -> np.ndarray:
    """Format d'image ; les formats rares dans la référence sont regroupés en « autres »."""
    counts = {f: reference.count(f) for f in set(reference)}
    return np.array([f if counts.get(f, 0) >= min_images else "autres" for f in formats])


def per_device_stats(values: np.ndarray, devices: np.ndarray) -> dict:
    return {d: values[devices == d] for d in np.unique(devices)}


def normalize_scores(logits, devices, reference_logits, reference_devices, method):
    stats = per_device_stats(reference_logits, reference_devices)
    out = logits.copy()
    for d in np.unique(devices):
        ref = stats.get(d, reference_logits)  # appareil absent de la référence : statistiques globales
        mask = devices == d
        if method == "centrage_scores":
            out[mask] = logits[mask] - np.median(ref, axis=0)
        else:  # zscore_scores
            out[mask] = (logits[mask] - ref.mean(0)) / (ref.std(0) + 1e-6)
    return out


def center_embeddings(X, devices, reference_X, reference_devices, source_mean):
    stats = per_device_stats(reference_X, reference_devices)
    out = X.copy()
    for d in np.unique(devices):
        ref = stats.get(d, reference_X)
        out[devices == d] = X[devices == d] - ref.mean(0) + source_mean
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rfmid-root", type=Path, required=True)
    parser.add_argument("--odir-root", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--min-device-images", type=int, default=30)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    roots = {"rfmid": args.rfmid_root, "odir": args.odir_root}

    data = {}
    for name in roots:
        for split in ("train", "val", "test"):
            ds = load(name, roots[name], split)
            data[name, split] = {
                "X": embeddings(args.embeddings, name, "crop-mask", ds.image_keys),
                "y": ds.df[COLUMNS[name]].to_numpy(np.float32),
                "formats": image_formats(ds),
                "groups": ds.groups,
            }

    report = {"run": run_info(), "min_device_images": args.min_device_images, "directions": {}}
    for source, target in (("odir", "rfmid"), ("rfmid", "odir")):
        src = data[source, "train"]
        logits_fn = train_logits_fn(src["X"], src["y"])
        reference_formats = data[target, "train"]["formats"] + data[target, "val"]["formats"]
        ref_X = np.vstack([data[target, "train"]["X"], data[target, "val"]["X"]])
        ref_devices = device_ids(reference_formats, reference_formats, args.min_device_images)
        ref_y = np.vstack([data[target, "train"]["y"], data[target, "val"]["y"]])
        ref_logits = logits_fn(ref_X)

        direction = {
            "devices": int(len(set(ref_devices))),
            "prevalence_by_device": {
                d: [float(v) for v in ref_y[ref_devices == d].mean(0)] for d in sorted(set(ref_devices))
            },
            "splits": {},
        }
        for split in ("val", "test"):
            part = data[target, split]
            devices = device_ids(part["formats"], reference_formats, args.min_device_images)
            raw = logits_fn(part["X"])
            scores = {
                "aucune": raw,
                "centrage_scores": normalize_scores(raw, devices, ref_logits, ref_devices, "centrage_scores"),
                "zscore_scores": normalize_scores(raw, devices, ref_logits, ref_devices, "zscore_scores"),
                "centrage_embeddings": logits_fn(
                    center_embeddings(part["X"], devices, ref_X, ref_devices, src["X"].mean(0))
                ),
            }
            split_report = {}
            for method, s in scores.items():
                metrics = evaluate_multilabel(part["y"], s, NAMES)
                entry = {"auroc": metrics["auroc_per_class"], "auroc_macro": metrics["auroc_macro"]}
                if method == "aucune":
                    gap = stratification_gap(part["y"], s, NAMES, devices, part["groups"], n_boot=500)
                    entry["auroc_within_device"] = {c: g["auroc_within_strata"] for c, g in gap.items()}
                else:
                    entry["minus_aucune"] = paired_bootstrap_diff(
                        part["y"], raw, s, NAMES, part["groups"], n_boot=1000
                    )["macro"]
                split_report[method] = entry
            direction["splits"][split] = split_report
        val = direction["splits"]["val"]
        direction["chosen_on_val"] = max(METHODS, key=lambda m: val[m]["auroc_macro"])
        report["directions"][f"{source}_to_{target}"] = direction

        print(f"== {source} -> {target} ({direction['devices']} appareils de référence)")
        for split in ("val", "test"):
            r = direction["splits"][split]
            within = r["aucune"]["auroc_within_device"]
            print(
                f"  {split:4s} aucune : macro {r['aucune']['auroc_macro']:.3f} | par maladie "
                + " ".join(f"{v:.3f}" for v in r["aucune"]["auroc"].values())
                + " | à appareil égal "
                + " ".join(f"{v:.3f}" for v in within.values())
            )
            for method in METHODS[1:]:
                d = r[method]["minus_aucune"]
                print(
                    f"       {method:20s} macro {r[method]['auroc_macro']:.3f} "
                    f"(écart {d['diff']:+.3f} [{d['ci'][0]:+.3f} ; {d['ci'][1]:+.3f}])"
                )
        print(f"  Choisie sur la validation : {direction['chosen_on_val']}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"Résultats : {args.out}")


if __name__ == "__main__":
    main()

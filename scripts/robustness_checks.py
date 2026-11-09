#!/usr/bin/env python3
"""Trois contrôles de robustesse, sur CPU, à partir des embeddings en cache et
des prédictions sauvegardées (aucun réglage n'est choisi ici : la tête est
celle de la baseline, Adam lr 1e-3, 30 epochs, mini-batchs de 16, graine 42).

1. **L'appareil se lit-il encore dans les embeddings ?** Une régression
   logistique tente de retrouver le format d'image d'origine (indicateur de
   l'appareil photo) à partir des embeddings, avec `none` puis `crop-mask`.
   Entraînée sur train, évaluée sur validation. Si l'appareil reste lisible
   avec `crop-mask`, l'information passe par la rétine elle-même (couleur,
   netteté), pas par la géométrie.
2. **Le modèle s'aide-t-il de l'appareil pour classer les malades ?** AUROC
   globale contre AUROC à appareil égal (paires malade/sain d'un même
   appareil, `src.eval.metrics.stratification_gap`), sur les prédictions
   sauvegardées. Un écart positif = une partie du classement vient des
   différences de prévalence entre appareils.
3. **Les résultats tiennent-ils hors de leur base ?** Pour les maladies
   présentes dans les deux bases (rétinopathie diabétique, DMLA, myopie), une
   tête entraînée sur une base est évaluée sur le test de l'autre (autres
   patients, autres appareils, autres annotateurs), comparée à la même tête
   entraînée dans la base cible. Lecture du test : aucune décision n'en
   dépend, c'est une mesure rapportée.

Exemple :
    python scripts/robustness_checks.py --rfmid-root data/raw/rfmid --odir-root data/raw/odir \\
        --embeddings embeddings --out results/robustness_checks.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.fundus_preprocess import PREPROCESS_VERSION  # noqa: E402
from src.data.odir_dataset import ODIRDataset  # noqa: E402
from src.data.rfmid_dataset import RFMiDDataset  # noqa: E402
from src.eval.metrics import bootstrap_auroc_ci, evaluate_multilabel, stratification_gap  # noqa: E402
from src.models.classifier_head import MultiLabelHead, train_head  # noqa: E402
from src.models.embedding_cache import load_embeddings  # noqa: E402

CHECKPOINT_STEM = "RETFound_mae_natureCFP"
MIN_DEVICE_IMAGES = 30  # format retenu comme « appareil » s'il a au moins 30 images en train
# Maladies communes aux deux bases : RFMiD -> ODIR. Définitions proches mais pas identiques
# (ODIR M = myopie pathologique ; RFMiD MYA = myopie visible au fond d'œil).
SHARED_CLASSES = {"DR": "D", "ARMD": "A", "MYA": "M"}


def load(name: str, root: Path, split: str):
    cls = RFMiDDataset if name == "rfmid" else ODIRDataset
    return cls(root, split=split, preprocess="crop-mask")


def image_formats(dataset) -> list[str]:
    sizes = []
    for i in range(len(dataset)):
        with Image.open(dataset.image_path(i)) as image:  # lit seulement l'en-tête
            sizes.append(f"{image.width}x{image.height}")
    return sizes


def embeddings(cache_dir: Path, name: str, preprocess: str, keys: list[str]) -> np.ndarray:
    """Embeddings en cache, avec la même vérification de configuration que le
    pipeline principal (prétraitement, sa version, taille d'image ; checkpoint
    non vérifiable ici, faute du fichier sur la machine)."""
    meta = {
        "checkpoint_sha256": None,
        "preprocess": preprocess,
        "preprocess_version": PREPROCESS_VERSION,
        "image_size": 224,
    }
    del meta  # contrat porté par le chargeur commun
    return load_embeddings(cache_dir, name, keys, "retfound", preprocess)


def device_probe(name: str, root: Path, cache_dir: Path) -> dict:
    train, val = load(name, root, "train"), load(name, root, "val")
    fmt_train, fmt_val = image_formats(train), image_formats(val)
    devices = sorted(d for d, n in Counter(fmt_train).items() if n >= MIN_DEVICE_IMAGES)
    keep_train = np.array([f in devices for f in fmt_train])
    keep_val = np.array([f in devices for f in fmt_val])
    y_train, y_val = np.array(fmt_train)[keep_train], np.array(fmt_val)[keep_val]
    result = {
        "devices": len(devices),
        "train_images": int(keep_train.sum()),
        "val_images": int(keep_val.sum()),
        "chance_balanced_accuracy": 1 / len(devices),
    }
    for preprocess in ("none", "crop-mask"):
        X_train = embeddings(cache_dir, name, preprocess, train.image_keys)[keep_train]
        X_val = embeddings(cache_dir, name, preprocess, val.image_keys)[keep_val]
        mean, std = X_train.mean(0), X_train.std(0) + 1e-6
        probe = LogisticRegression(max_iter=3000, C=1.0)
        probe.fit((X_train - mean) / std, y_train)
        predicted = probe.predict((X_val - mean) / std)
        result[preprocess] = {
            "balanced_accuracy": float(balanced_accuracy_score(y_val, predicted)),
            "accuracy": float((predicted == y_val).mean()),
        }
    return result


def device_gap(name: str, root: Path, results_dir: Path) -> dict:
    out = {}
    for split, files in (
        ("val", {p: f"pred_{name}_{p}_val.npz" for p in ("none", "crop-mask")}),
        ("test", {"crop-mask": f"pred_{name}_crop-mask_test.npz"}),
    ):
        dataset = load(name, root, split)
        formats = dict(zip(dataset.image_keys, image_formats(dataset), strict=True))
        for preprocess, file in files.items():
            pred = np.load(results_dir / file)
            strata = [formats[k] for k in pred["keys"]]
            gap = stratification_gap(
                pred["y_true"], pred["probs"], list(pred["class_names"]), strata, list(pred["groups"]), n_boot=1000
            )
            out[f"{split}_{preprocess}"] = gap
    return out


def fit_head(X_train, y_train, seed: int = 42) -> MultiLabelHead:
    torch.manual_seed(seed)
    loader = DataLoader(
        list(zip(torch.from_numpy(X_train), torch.from_numpy(y_train), strict=True)), batch_size=16, shuffle=True
    )
    return train_head(MultiLabelHead(X_train.shape[1], y_train.shape[1]), loader, epochs=30, lr=1e-3, device="cpu")


def head_scores(X_train, y_train, X_eval, seed: int = 42) -> np.ndarray:
    head = fit_head(X_train, y_train, seed)
    with torch.no_grad():
        return torch.sigmoid(head(torch.from_numpy(X_eval))).numpy()


def transfer(rfmid_root: Path, odir_root: Path, cache_dir: Path) -> dict:
    columns = {"rfmid": list(SHARED_CLASSES), "odir": list(SHARED_CLASSES.values())}
    names = ["DR/D (rétinopathie diabétique)", "ARMD/A (DMLA)", "MYA/M (myopie)"]
    roots = {"rfmid": rfmid_root, "odir": odir_root}
    data = {}
    for name in ("rfmid", "odir"):
        for split in ("train", "test"):
            ds = load(name, roots[name], split)
            data[name, split] = (
                embeddings(cache_dir, name, "crop-mask", ds.image_keys),
                ds.df[columns[name]].to_numpy(np.float32),
                ds.groups,
            )
    out = {}
    for source, target in (("odir", "rfmid"), ("rfmid", "odir")):
        X_test, y_test, groups = data[target, "test"]
        scores = {}
        for label, train_on in (("in_domain", target), ("transfer", source)):
            X_train, y_train, _ = data[train_on, "train"]
            probs = head_scores(X_train, y_train, X_test)
            metrics = evaluate_multilabel(y_test, probs, names)
            metrics.update(bootstrap_auroc_ci(y_test, probs, names, groups, n_boot=1000))
            scores[label] = {
                "trained_on": train_on,
                "auroc_per_class": metrics["auroc_per_class"],
                "auroc_per_class_ci": metrics["auroc_per_class_ci"],
                "auroc_macro": metrics["auroc_macro"],
                "auroc_macro_ci": metrics["auroc_macro_ci"],
                "positives_per_class": metrics["positives_per_class"],
            }
        out[f"{source}_to_{target}"] = {"evaluated_on": f"{target} test", **scores}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rfmid-root", type=Path, required=True)
    parser.add_argument("--odir-root", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True, help="Dossier des caches .npz")
    parser.add_argument("--results", type=Path, default=Path("results"), help="Prédictions sauvegardées")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    roots = {"rfmid": args.rfmid_root, "odir": args.odir_root}

    report = {"device_probe": {}, "device_gap": {}}
    for name, root in roots.items():
        report["device_probe"][name] = probe = device_probe(name, root, args.embeddings)
        print(
            f"[1] {name} : appareil retrouvé à partir des embeddings ({probe['devices']} appareils, hasard "
            f"{probe['chance_balanced_accuracy']:.2f}) : none {probe['none']['balanced_accuracy']:.2f}, "
            f"crop-mask {probe['crop-mask']['balanced_accuracy']:.2f} (exactitude équilibrée, validation)"
        )
        report["device_gap"][name] = gaps = device_gap(name, root, args.results)
        for run, per_class in gaps.items():
            top = sorted(per_class.items(), key=lambda kv: -kv[1]["gap"])[:3]
            print(
                f"[2] {name} {run} : plus grands écarts AUROC globale − à appareil égal : "
                + " ; ".join(f"{c} {v['gap']:+.3f} [{v['gap_ci'][0]:+.3f} ; {v['gap_ci'][1]:+.3f}]" for c, v in top)
            )
    report["transfer"] = transfer(args.rfmid_root, args.odir_root, args.embeddings)
    for direction, scores in report["transfer"].items():
        for label in ("in_domain", "transfer"):
            s = scores[label]
            ci = s["auroc_per_class_ci"]
            print(
                f"[3] {direction} {label:9s} (entraîné sur {s['trained_on']}) : "
                + " ; ".join(
                    f"{c.split(' ')[0]} {v:.3f} [{ci[c][0]:.3f}-{ci[c][1]:.3f}]"
                    for c, v in s["auroc_per_class"].items()
                )
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"Résultats : {args.out}")


if __name__ == "__main__":
    main()

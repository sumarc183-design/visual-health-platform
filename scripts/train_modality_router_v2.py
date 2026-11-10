#!/usr/bin/env python3
"""Routeur de modalité, protocole 2 : découpage par patient et inconnus variés (CPU).

Plan fixé le 10 novembre 2026, avant tout run, après l'audit du 9 novembre. Le
protocole 1 (`train_modality_router.py`) et son critère non atteint restent tels quels.

Inchangé : embeddings DINOv2 ViT-B/14 figés, image brute (petit côté 256, recadrage
central 224), standardisation puis régression logistique (C = 1), seuil de confiance
0,9 (en dessous : « unknown »), graine 0.

Changements :
- ajustement et contrôle séparés par patient (ODIR) et par lésion (ISIC) avant de tirer
  les images ; Imagenette et MedMNIST n'ont pas d'identifiant patient (image = groupe) ;
- classe « unknown » apprise sur des photos (Imagenette) et deux types d'images médicales
  hors cible : radiographies du thorax (PneumoniaMNIST) et frottis sanguins (BloodMNIST) ;
- test sur des sources jamais vues à l'ajustement, dont trois types d'images médicales
  jamais vus : OCT de la rétine (OCTMNIST), échographie mammaire (BreastMNIST), scanner
  (OrganCMNIST) ; et une seconde source de fonds d'œil (RetinaMNIST, DeepDRiD).

Tirages : ajustement 1 000 images par classe (inconnus : 500 Imagenette, 250 thorax,
250 sang) ; contrôle 300 par classe (150, 75, 75), groupes disjoints de l'ajustement ;
test 300 par source (500 pour RFMiD et ISIC, comme au protocole 1 ; toutes les images
de test de BreastMNIST, 156).

Critères de réussite (test) : rappel >= 0,95 pour chaque source de fonds d'œil (RFMiD,
RetinaMNIST) et pour la dermoscopie ; aucune erreur croisée entre fond d'œil et
dermoscopie ; pour chaque source d'inconnus, au plus 2 % d'images envoyées à un modèle
spécialisé. La tête du protocole 1 est aussi évaluée sur ces sources, pour comparaison.

Réserves : images MedMNIST déjà redimensionnées (128 ou 224 px, carrées) ; pas de photos
de peau prises au téléphone ni de vues externes de l'œil ; ISIC limité au début des
archives. Licences : MedMNIST CC BY 4.0 (sources d'origine sous leurs propres termes).

Exemple :
    python scripts/train_modality_router_v2.py --odir-root data/raw/odir --rfmid-root data/raw/rfmid \\
        --modality-root data/raw/modality --medmnist-root data/raw/medmnist \\
        --isic-csv data/raw/isic2019 --checkpoint third_party/dinov2/dinov2_vitb14_timm.pth
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.robustness_checks import load  # noqa: E402
from scripts.train_modality_router import SPECIALISTS, exact_ci  # noqa: E402
from src.data.isic_dataset import skin_tensor  # noqa: E402
from src.eval.provenance import run_info  # noqa: E402
from src.models.embedding_cache import EmbeddingCache, file_fingerprint  # noqa: E402
from src.models.retfound_backbone import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from src.models.timm_backbones import DINOV2_IMAGE_SIZE, DINOv2Backbone  # noqa: E402
from src.pipeline.modality_classifier import (  # noqa: E402
    CONFIDENCE_THRESHOLD,
    MODALITIES,
    ModalityHead,
    linear_from_sklearn,
)

MEDMNIST = {
    "retina": "retinamnist_224",
    "pneumonia": "pneumoniamnist_224",
    "blood": "bloodmnist_128",
    "oct": "octmnist_128",
    "breast": "breastmnist_224",
    "organc": "organcmnist_128",
}
TEST_UNKNOWN = ("imagenette_val", "oct", "breast", "organc")


class Item:
    """Une image : clé de cache, groupe (patient, lésion ou image) et chargeur."""

    def __init__(self, key: str, group: str, opener: Callable[[], Image.Image]):
        self.key, self.group, self.opener = key, group, opener


def file_item(path: Path, group: str | None = None) -> Item:
    return Item(f"modality/{path.parent.name}/{path.name}", group or path.name, lambda: Image.open(path))


def medmnist_items(root: Path, name: str, split: str) -> list[Item]:
    data = np.load(root / f"{MEDMNIST[name]}.npz")
    images = data[f"{split}_images"]
    return [
        Item(f"medmnist/{name}/{split}/{i}", f"{name}/{split}/{i}", lambda a=images[i]: Image.fromarray(a))
        for i in range(len(images))
    ]


def split_by_group(items: list[Item], n_control: int, n_fit: int, rng) -> tuple[list[Item], list[Item]]:
    """Groupes tirés au hasard pour le contrôle jusqu'à n_control images, puis n_fit
    images tirées parmi les autres groupes : aucun groupe commun aux deux parties."""
    groups = sorted({it.group for it in items})
    order = [groups[i] for i in rng.permutation(len(groups))]
    by_group: dict[str, list[Item]] = {}
    for it in items:
        by_group.setdefault(it.group, []).append(it)
    control, used = [], set()
    for g in order:
        if len(control) >= n_control:
            break
        control.extend(by_group[g])
        used.add(g)
    control = control[:n_control]  # les images coupées d'un groupe ne servent nulle part
    rest = [it for it in items if it.group not in used]
    if len(rest) < n_fit:
        raise ValueError(f"{len(rest)} images hors contrôle, {n_fit} demandées")
    fit = [rest[i] for i in sorted(rng.choice(len(rest), n_fit, replace=False))]
    return fit, control


def sample(items: list[Item], n: int, rng) -> list[Item]:
    if len(items) <= n:
        return list(items)
    return [items[i] for i in sorted(rng.choice(len(items), n, replace=False))]


def build_sets(args) -> tuple[dict, dict, dict]:
    rng = np.random.default_rng(0)
    odir = load("odir", args.odir_root, "train")
    odir_items = [file_item(odir.image_path(i), f"odir/{g}") for i, g in enumerate(odir.groups)]
    meta = pd.read_csv(args.isic_csv / "ISIC_2019_Training_Metadata.csv").set_index("image")
    isic_paths = sorted((args.modality_root / "isic_train").rglob("ISIC_*.jpg"))
    isic_items = [
        file_item(p, f"isic/{meta.loc[p.stem, 'lesion_id']}" if pd.notna(meta.loc[p.stem, "lesion_id"]) else None)
        for p in isic_paths
    ]
    net = args.modality_root / "imagenette/imagenette2-320"
    fit, control = {}, {}
    fit["fundus"], control["fundus"] = split_by_group(odir_items, 300, 1000, rng)
    fit["dermoscopy"], control["dermoscopy"] = split_by_group(isic_items, 300, 1000, rng)
    imagenette = [file_item(p) for p in sorted((net / "train").rglob("*.JPEG"))]
    unknown_fit, unknown_control = [], []
    for items, n_fit, n_control in (
        (imagenette, 500, 150),
        (medmnist_items(args.medmnist_root, "pneumonia", "train"), 250, 0),
        (medmnist_items(args.medmnist_root, "blood", "train"), 250, 0),
    ):
        f, c = split_by_group(items, n_control, n_fit, rng) if n_control else (sample(items, n_fit, rng), [])
        unknown_fit += f
        unknown_control += c
    unknown_control += sample(medmnist_items(args.medmnist_root, "pneumonia", "val"), 75, rng)
    unknown_control += sample(medmnist_items(args.medmnist_root, "blood", "val"), 75, rng)
    fit["unknown"], control["unknown"] = unknown_fit, unknown_control

    rfmid = load("rfmid", args.rfmid_root, "test")
    test = {
        "fundus/rfmid": sample([file_item(rfmid.image_path(i)) for i in range(len(rfmid))], 500, rng),
        "fundus/retinamnist": sample(medmnist_items(args.medmnist_root, "retina", "test"), 300, rng),
        "dermoscopy/isic_test": sample(
            [file_item(p) for p in sorted((args.modality_root / "isic_test").rglob("ISIC_*.jpg"))], 500, rng
        ),
        "unknown/imagenette_val": sample([file_item(p) for p in sorted((net / "val").rglob("*.JPEG"))], 300, rng),
        "unknown/oct": sample(medmnist_items(args.medmnist_root, "oct", "test"), 300, rng),
        "unknown/breast": sample(medmnist_items(args.medmnist_root, "breast", "test"), 300, rng),
        "unknown/organc": sample(medmnist_items(args.medmnist_root, "organc", "test"), 300, rng),
    }
    return fit, control, test


def embed(items: list[Item], backbone, cache: EmbeddingCache, batch_size: int = 32) -> np.ndarray:
    keys = [it.key for it in items]
    missing = set(cache.missing(keys))
    todo = [it for it in items if it.key in missing]
    start = time.time()
    for b in range(0, len(todo), batch_size):
        part = todo[b : b + batch_size]
        x = torch.stack([skin_tensor(it.opener(), DINOV2_IMAGE_SIZE) for it in part])
        cache.add([it.key for it in part], backbone((x - IMAGENET_MEAN) / IMAGENET_STD).numpy())
        if (b // batch_size) % 20 == 0:
            cache.save()
            print(f"  {b + len(part)}/{len(todo)} en {time.time() - start:.0f} s", flush=True)
    cache.save()
    return cache.get(keys)


def routed(head: ModalityHead, X: np.ndarray, truth: str) -> dict:
    decided = head.predict(X)
    n = len(decided)
    k = decided.count(truth)
    wrong = sum(d in SPECIALISTS and d != truth for d in decided)
    return {
        "n": n,
        "decisions": {m: decided.count(m) for m in MODALITIES},
        "correct": {"value": k / n, "ci": exact_ci(k, n)},
        "to_wrong_specialist": {"count": wrong, "value": wrong / n, "ci": exact_ci(wrong, n)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--odir-root", type=Path, required=True)
    parser.add_argument("--rfmid-root", type=Path, required=True)
    parser.add_argument("--modality-root", type=Path, required=True)
    parser.add_argument("--medmnist-root", type=Path, required=True)
    parser.add_argument("--isic-csv", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--v1-head", type=Path, default=Path("heads/modality_router.pt"))
    parser.add_argument("--cache", type=Path, default=Path("embeddings/modality_dinov2_vitb14.npz"))
    parser.add_argument("--out", type=Path, default=Path("results/modality_router_v2.json"))
    parser.add_argument("--head", type=Path, default=Path("heads/modality_router_v2.pt"))
    args = parser.parse_args()

    fit, control, test = build_sets(args)
    shared = {m: len({it.group for it in fit[m]} & {it.group for it in control[m]}) for m in MODALITIES}
    if any(shared.values()):
        raise ValueError(f"groupes communs entre ajustement et contrôle : {shared}")
    fingerprint = file_fingerprint(args.checkpoint)
    cache = EmbeddingCache(args.cache, meta={"checkpoint_sha256": fingerprint, "image_size": DINOV2_IMAGE_SIZE})
    backbone = DINOv2Backbone(args.checkpoint)
    X = {}
    for part, sets in (("fit", fit), ("control", control), ("test", test)):
        for name, items in sets.items():
            print(f"{part} / {name} : {len(items)} images", flush=True)
            X[part, name] = embed(items, backbone, cache)

    features = np.vstack([X["fit", m] for m in MODALITIES])
    target = np.concatenate([np.full(len(fit[m]), i) for i, m in enumerate(MODALITIES)])
    scaler = StandardScaler().fit(features)
    model = LogisticRegression(C=1.0, max_iter=5000).fit(scaler.transform(features), target)
    weight, bias = linear_from_sklearn(scaler, model)
    head = ModalityHead(weight.shape[1], CONFIDENCE_THRESHOLD)
    head.linear.weight.data = torch.from_numpy(weight)
    head.linear.bias.data = torch.from_numpy(bias)
    v1, _ = ModalityHead.load(args.v1_head)

    report = {
        "run": run_info(),
        "plan": "fixé le 10 novembre 2026 avant tout run (docstring du script)",
        "backbone": {"model": "dinov2_vitb14", "checkpoint_sha256": fingerprint, "image_size": DINOV2_IMAGE_SIZE},
        "threshold": CONFIDENCE_THRESHOLD,
        "sizes": {
            part: {k: len(v) for k, v in sets.items()}
            for part, sets in (("fit", fit), ("control", control), ("test", test))
        },
        "groups_shared_fit_control": shared,
        "control": {m: routed(head, X["control", m], m) for m in MODALITIES},
        "test": {},
        "test_v1_head": {},
    }
    for name in test:
        truth = name.split("/")[0]
        report["test"][name] = routed(head, X["test", name], truth)
        report["test_v1_head"][name] = routed(v1, X["test", name], truth)
    t = report["test"]
    cross = sum(t[n]["to_wrong_specialist"]["count"] for n in t if not n.startswith("unknown"))
    report["criteria"] = {
        "fundus_rfmid_recall_ge_0.95": t["fundus/rfmid"]["correct"]["value"] >= 0.95,
        "fundus_retinamnist_recall_ge_0.95": t["fundus/retinamnist"]["correct"]["value"] >= 0.95,
        "dermoscopy_recall_ge_0.95": t["dermoscopy/isic_test"]["correct"]["value"] >= 0.95,
        "no_cross_specialist_error": cross == 0,
        **{
            f"unknown_{u}_to_specialist_le_0.02": t[f"unknown/{u}"]["to_wrong_specialist"]["value"] <= 0.02
            for u in TEST_UNKNOWN
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    head.save(args.head, {"backbone": "dinov2", "checkpoint_sha256": fingerprint, "image_size": DINOV2_IMAGE_SIZE})
    for name in test:
        print(f"{name:24s} v2 {report['test'][name]['decisions']}   v1 {report['test_v1_head'][name]['decisions']}")
    print("critères :", report["criteria"])
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Tête peau : ISIC 2020 sans les images « unknown », mesure sur MILK10k (CPU).

Plan fixé le 13 novembre 2026, avant tout calcul, après `skin_transfer_milk10k.py` : la
tête B (ISIC 2019 + toutes les images d'ISIC 2020) gagne sur le mélanome (+0,017) mais
perd sur le BCC (−0,025), les nævus (−0,039) et le macro (−0,007). Cause supposée : les
27 124 images « unknown » d'ISIC 2020, apprises comme négatives pour toutes les classes.

Tête C : même tête et réglages (graine 42), apprise sur l'apprentissage ISIC 2019 + les
seules images d'ISIC 2020 à diagnostic connu et rattaché à une classe (mélanome, nævus,
kératoses et lentigos → MEL, NV, BKL ; environ 6 000 images).

Mesure : MILK10k, mêmes 5 240 dermoscopies. **Deuxième utilisation de MILK10k** (la
première a comparé A et B) : résultat à lire comme confirmatoire de la cause, pas comme
une validation neuve.

Critère (fixé) : C − A, borne basse de l'IC de l'écart d'AUROC du mélanome > 0 ET borne
basse de l'écart macro (8 classes) > −0,005 (pas de perte notable sur les autres
classes). Secondaires : C − B (mélanome, BCC, NV, macro) ; écarts par classe ; test ISIC
2019, C − A.

Conséquence prévue : critère atteint → C devient le candidat de remplacement proposé au
tuteur (à la place de B) ; sinon, l'ajout d'ISIC 2020 reste un compromis entre classes et
la tête actuelle reste.

Exemple :
    python scripts/skin_transfer_labeled.py --milk-root data/raw/milk10k --isic2019-root data/raw/isic2019 \\
        --isic2019-embeddings embeddings_derm --isic2020-root data/raw/isic2020 \\
        --checkpoint third_party/dinov2/dinov2_vitb14_timm.pth
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
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.dermatology_analysis import ImageList  # noqa: E402
from scripts.pad_ufes20_skin import auroc_ci  # noqa: E402
from scripts.robustness_checks import fit_head  # noqa: E402
from scripts.skin_transfer_milk10k import ISIC2020_TO_ISIC, SKIN_HEAD, isic2020_targets, milk_targets, scores  # noqa: E402
from scripts.train_modality_router_v2 import Item, embed  # noqa: E402
from src.data.isic_dataset import CLASS_NAMES  # noqa: E402
from src.eval.metrics import paired_bootstrap_diff  # noqa: E402
from src.eval.provenance import run_info  # noqa: E402
from src.models.classifier_head import MultiLabelHead  # noqa: E402
from src.models.embedding_cache import EmbeddingCache, file_fingerprint, load_embeddings  # noqa: E402
from src.models.timm_backbones import DINOV2_IMAGE_SIZE, DINOv2Backbone  # noqa: E402

MACRO_MARGIN = -0.005


def labeled(diagnosis: pd.Series) -> np.ndarray:
    """Images ISIC 2020 dont le diagnostic est rattaché à une classe ISIC 2019."""
    return diagnosis.isin(list(ISIC2020_TO_ISIC)).to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--milk-root", type=Path, required=True)
    parser.add_argument("--isic2019-root", type=Path, required=True)
    parser.add_argument("--isic2019-embeddings", type=Path, required=True)
    parser.add_argument("--isic2020-root", type=Path, required=True)
    parser.add_argument("--isic2020-cache", type=Path, default=Path("embeddings/isic2020_dinov2_vitb14.npz"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("embeddings/milk10k_dinov2_vitb14.npz"))
    parser.add_argument("--out", type=Path, default=Path("results/skin_transfer_labeled.json"))
    args = parser.parse_args()

    fingerprint = file_fingerprint(args.checkpoint)
    backbone = DINOv2Backbone(args.checkpoint)
    meta = {"checkpoint_sha256": fingerprint, "image_size": DINOV2_IMAGE_SIZE}

    images = pd.read_csv(args.milk_root / "metadata.csv")
    derm = images[images["image_type"] == "dermoscopic"].drop_duplicates("lesion_id")
    gt = pd.read_csv(args.milk_root / "supplements" / "training_gt.csv").merge(
        derm[["lesion_id", "isic_id"]], on="lesion_id"
    )
    paths = {p.stem: p for p in args.milk_root.rglob("*.jpg")}
    items = [
        Item(f"milk10k/{i}", g, lambda p=paths[i]: Image.open(p))
        for i, g in zip(gt["isic_id"], gt["lesion_id"], strict=True)
    ]
    X_eval = embed(items, backbone, EmbeddingCache(args.cache, meta=meta))
    y_eval = milk_targets(gt)
    groups = list(gt["lesion_id"])

    train19, test19 = ImageList(args.isic2019_root, "train"), ImageList(args.isic2019_root, "test")
    X19 = load_embeddings(args.isic2019_embeddings, "isic2019", train19.keys, "dinov2", preprocess="none")
    X19_test = load_embeddings(args.isic2019_embeddings, "isic2019", test19.keys, "dinov2", preprocess="none")
    m20 = pd.read_csv(args.isic2020_root / "train.csv")
    paths20 = {p.stem: p for p in args.isic2020_root.rglob("*.jpg")}
    items20 = [
        Item(f"isic2020/{n}", g, lambda p=paths20[n]: Image.open(p))
        for n, g in zip(m20["image_name"], m20["patient_id"], strict=True)
    ]
    X20 = embed(items20, backbone, EmbeddingCache(args.isic2020_cache, meta=meta))
    y20 = isic2020_targets(m20["diagnosis"])
    sel = labeled(m20["diagnosis"])

    artifact = torch.load(SKIN_HEAD, map_location="cpu", weights_only=True)
    if fingerprint not in {artifact["checkpoint_sha256"], artifact.get("embeddings_checkpoint_sha256")}:
        raise ValueError("tête peau calculée avec un autre DINOv2-B")
    head_a = MultiLabelHead(X_eval.shape[1], len(CLASS_NAMES))
    head_a.load_state_dict(artifact["state_dict"])
    with contextlib.redirect_stdout(io.StringIO()):
        head_b = fit_head(np.vstack([X19, X20]), np.vstack([train19.targets, y20]))
        head_c = fit_head(np.vstack([X19, X20[sel]]), np.vstack([train19.targets, y20[sel]]))

    p = {name: scores(h, X_eval) for name, h in (("A", head_a), ("B", head_b), ("C", head_c))}
    names = list(CLASS_NAMES)
    c_minus_a = paired_bootstrap_diff(y_eval, p["A"], p["C"], names, groups)
    c_minus_b = paired_bootstrap_diff(y_eval, p["B"], p["C"], names, groups)
    t = {name: scores(h, X19_test) for name, h in (("A", head_a), ("C", head_c))}
    test19 = paired_bootstrap_diff(test19.targets, t["A"], t["C"], names, list(test19.lesions))
    mel = names.index("MEL")

    met = c_minus_a["MEL"]["ci"][0] > 0 and c_minus_a["macro"]["ci"][0] > MACRO_MARGIN
    report = {
        "run": run_info(),
        "plan": "fixé le 13 novembre 2026 avant tout calcul ; deuxième utilisation de MILK10k (docstring)",
        "training_c": {"isic2019": len(X19), "isic2020_labeled": int(sel.sum())},
        "melanoma": {k: auroc_ci(y_eval[:, mel], v[:, mel], groups) for k, v in p.items()},
        "c_minus_a": c_minus_a,
        "c_minus_b": {k: c_minus_b[k] for k in ("MEL", "BCC", "NV", "macro")},
        "isic2019_test_c_minus_a": {"MEL": test19["MEL"], "macro": test19["macro"]},
        "criterion": {
            "melanoma": c_minus_a["MEL"],
            "macro": c_minus_a["macro"],
            "macro_margin": MACRO_MARGIN,
            "met": bool(met),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print("Images ISIC 2020 gardées :", int(sel.sum()))
    print("Mélanome :", {k: round(v["auroc"], 3) for k, v in report["melanoma"].items()})
    for c in [*names, "macro"]:
        print(f"  C − A {c:5s} {c_minus_a[c]['diff']:+.3f} {[round(x, 3) for x in c_minus_a[c]['ci']]}")
    print("C − B :", report["c_minus_b"])
    print("Test ISIC 2019, C − A :", report["isic2019_test_c_minus_a"])
    print("Critère :", met)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()

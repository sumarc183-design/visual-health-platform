#!/usr/bin/env python3
"""PAD-UFES-20 : routeur et tête peau face à de vraies photos de smartphone (CPU).

Plan fixé le 12 novembre 2026, avant tout calcul d'embedding sur cette base. Seuls la
structure de l'archive et le décompte des diagnostics ont été lus.

Données : PAD-UFES-20 (Brésil, CC BY 4.0, doi:10.17632/zr7vgbcyr2.1), 2 298 photos
cliniques prises au smartphone (pas de dermoscopie), 1 373 patients ; BCC 845, ACK 730,
NEV 244, SEK 235, SCC 192 (maladie de Bowen incluse), MEL 52 ; BCC, SCC et MEL confirmés
par biopsie (58 % des images au total). Archive SHA-256
e8c7e17bac1698c97e44d4096ec20ac1b91c135285c1446b7b2e7ebbc9be933c. Jamais utilisée dans
le projet.

Embeddings : DINOv2-B local (celui du routeur et de la tête peau), petit côté 256 px,
recadrage 224 px ; un seul calcul sert aux deux tests.

Test 1, routeur (têtes des protocoles 1 et 2, inchangées, seuil 0,9) : décisions (fond
d'œil, dermoscopie, inconnu) sur l'ensemble et par diagnostic. Critère de sécurité :
borne haute de l'IC exact à 95 % du taux envoyé en fond d'œil au plus 0,5 %, pour chaque
tête. Sans critère : part « dermoscopie » contre « inconnu ».

Test 2, tête peau (`heads/isic2019_dinov2b_head.pt`, apprise sur dermoscopies ISIC 2019,
inchangée) : correspondance des classes ACK → AK, SEK → BKL, NEV → NV, BCC, SCC, MEL ;
DF et VASC absents, non évalués.
- mesure principale : AUROC macro des 6 classes, un contre tous, IC par bootstrap sur les
  patients (1 000 tirages) ;
- secondaire : AUROC « cancer » (BCC, SCC ou MEL contre le reste), score = somme des
  probabilités de ces trois classes ;
- référence : même tête (mêmes réglages, graine 42) entraînée dans PAD-UFES-20 sur les
  mêmes embeddings, validation croisée à 5 plis groupés par patient (graine 0) ; écart
  apparié transfert − référence avec IC ;
- contrôle : images confirmées par biopsie seulement.

Lecture fixée à l'avance : si l'écart transfert − référence a une borne haute < 0, la
tête apprise sur dermoscopie ne se transpose pas aux photos de téléphone, et l'outil peau
doit être réservé à la dermoscopie (type de photo déclaré) ; sinon, la perte n'est pas
démontrée.

Réserves : un seul pays, une seule source ; la référence par validation croisée est
optimiste (même population) ; classes non biopsiées diagnostiquées par consensus.

Exemple :
    python scripts/pad_ufes20_skin.py --root data/raw/pad_ufes20 \\
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.modality_router_pathology_stress import HEADS, rates  # noqa: E402
from scripts.modality_router_skin_photo_stress import to_fundus  # noqa: E402
from scripts.robustness_checks import fit_head  # noqa: E402
from scripts.train_modality_router_v2 import Item, embed  # noqa: E402
from src.data.isic_dataset import CLASS_NAMES  # noqa: E402
from src.eval.metrics import _group_resamples, bootstrap_auroc_ci, paired_bootstrap_diff  # noqa: E402
from src.eval.provenance import run_info  # noqa: E402
from src.models.classifier_head import MultiLabelHead  # noqa: E402
from src.models.embedding_cache import EmbeddingCache, file_fingerprint  # noqa: E402
from src.models.timm_backbones import DINOV2_IMAGE_SIZE, DINOv2Backbone  # noqa: E402
from src.pipeline.modality_classifier import ModalityHead  # noqa: E402

TO_ISIC = {"BCC": "BCC", "SCC": "SCC", "MEL": "MEL", "ACK": "AK", "SEK": "BKL", "NEV": "NV"}
EVALUATED = [c for c in CLASS_NAMES if c in TO_ISIC.values()]
CANCER = ("BCC", "SCC", "MEL")
SKIN_HEAD = Path("heads/isic2019_dinov2b_head.pt")


def targets(diagnostic: pd.Series) -> np.ndarray:
    """Matrice un contre tous dans l'ordre de `EVALUATED`."""
    mapped = diagnostic.map(TO_ISIC)
    if mapped.isna().any():
        raise ValueError(f"diagnostic inattendu : {sorted(set(diagnostic[mapped.isna()]))}")
    return np.stack([(mapped == c).to_numpy() for c in EVALUATED], axis=1).astype(np.float32)


def cancer_score(probs: np.ndarray) -> np.ndarray:
    return probs[:, [EVALUATED.index(c) for c in CANCER]].sum(axis=1)


def auroc_ci(y: np.ndarray, p: np.ndarray, groups: list[str], seed: int = 0) -> dict:
    draws = [roc_auc_score(y[i], p[i]) for i in _group_resamples(groups, 1000, seed) if 0 < y[i].sum() < len(i)]
    return {"auroc": float(roc_auc_score(y, p)), "ci": np.percentile(draws, [2.5, 97.5]).tolist()}


def skin_report(y: np.ndarray, transfer: np.ndarray, reference: np.ndarray, groups: list[str]) -> dict:
    out = {"n": len(y), "patients": len(set(groups))}
    for name, p in (("transfer", transfer), ("in_domain", reference)):
        per_class = {c: float(roc_auc_score(y[:, i], p[:, i])) for i, c in enumerate(EVALUATED)}
        out[name] = {
            "auroc_macro": float(np.mean(list(per_class.values()))),
            "auroc_per_class": per_class,
            **bootstrap_auroc_ci(y, p, EVALUATED, groups),
            "cancer": auroc_ci(y[:, [EVALUATED.index(c) for c in CANCER]].max(axis=1), cancer_score(p), groups),
        }
    out["transfer_minus_in_domain"] = paired_bootstrap_diff(y, reference, transfer, EVALUATED, groups)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("embeddings/pad_ufes20_dinov2_vitb14.npz"))
    parser.add_argument("--out", type=Path, default=Path("results/pad_ufes20_skin.json"))
    args = parser.parse_args()

    meta = pd.read_csv(args.root / "metadata.csv")
    paths = {p.name: p for p in args.root.rglob("*.png")}
    missing = set(meta["img_id"]) - set(paths)
    if missing:
        raise FileNotFoundError(f"{len(missing)} images absentes, ex. {sorted(missing)[:3]}")
    items = [
        Item(f"pad_ufes20/{i}", g, lambda p=paths[i]: Image.open(p))
        for i, g in zip(meta["img_id"], meta["patient_id"], strict=True)
    ]
    fingerprint = file_fingerprint(args.checkpoint)
    cache = EmbeddingCache(args.cache, meta={"checkpoint_sha256": fingerprint, "image_size": DINOV2_IMAGE_SIZE})
    X = embed(items, DINOv2Backbone(args.checkpoint), cache)
    diagnostic = meta["diagnostic"].to_numpy()
    groups = list(meta["patient_id"])

    report = {
        "run": run_info(),
        "plan": "fixé le 12 novembre 2026 avant tout calcul sur PAD-UFES-20 (docstring du script)",
        "backbone": {"model": "dinov2_vitb14", "checkpoint_sha256": fingerprint},
        "images": {
            "total": len(meta),
            "patients": len(set(groups)),
            "by_diagnostic": meta["diagnostic"].value_counts().to_dict(),
        },
        "router": {},
    }
    for name, path in HEADS.items():
        head, _ = ModalityHead.load(path)
        decided = np.array(head.decide(head.probabilities(X)))
        report["router"][name] = {
            "all": rates(decided),
            "by_diagnostic": {d: rates(decided[diagnostic == d]) for d in sorted(set(diagnostic))},
            "criterion_safe": to_fundus(decided),
        }

    artifact = torch.load(SKIN_HEAD, map_location="cpu", weights_only=True)
    if artifact["checkpoint_sha256"] != fingerprint:
        raise ValueError("tête peau calculée avec un autre DINOv2-B")
    skin = MultiLabelHead(X.shape[1], len(CLASS_NAMES))
    skin.load_state_dict(artifact["state_dict"])
    with torch.no_grad():
        transfer = torch.sigmoid(skin(torch.from_numpy(X))).numpy()[:, [CLASS_NAMES.index(c) for c in EVALUATED]]
    y = targets(meta["diagnostic"])
    reference = np.zeros_like(y)
    folds = StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, diagnostic, groups=groups)
    for fit, held_out in folds:
        with contextlib.redirect_stdout(io.StringIO()):
            fold_head = fit_head(X[fit], y[fit])
        with torch.no_grad():
            reference[held_out] = torch.sigmoid(fold_head(torch.from_numpy(X[held_out]))).numpy()

    biopsy = meta["biopsed"].astype(bool).to_numpy()
    report["skin_head"] = {
        "classes": {**TO_ISIC},
        "all": skin_report(y, transfer, reference, groups),
        "biopsy_only": skin_report(
            y[biopsy], transfer[biopsy], reference[biopsy], [g for g, b in zip(groups, biopsy, strict=True) if b]
        ),
    }
    gap = report["skin_head"]["all"]["transfer_minus_in_domain"]["macro"]
    report["reading"] = (
        "la tête apprise sur dermoscopie ne se transpose pas aux photos de téléphone"
        if gap["ci"][1] < 0
        else "perte de transfert non démontrée"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    for name, r in report["router"].items():
        print(name, r["all"]["decisions"], "fond d'œil :", r["criterion_safe"])
    a = report["skin_head"]["all"]
    for name in ("transfer", "in_domain"):
        print(
            f"{name} : AUROC macro {a[name]['auroc_macro']:.3f} {a[name]['auroc_macro_ci']}, cancer {a[name]['cancer']}"
        )
    print("transfert − référence :", gap, "|", report["reading"])
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()

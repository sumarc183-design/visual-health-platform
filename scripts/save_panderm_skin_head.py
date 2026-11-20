#!/usr/bin/env python3
"""Enregistre la tête peau servie P (PanDerm-Large figé, ISIC 2019, 8 classes) et ses seuils (CPU).

Décision du tuteur du 20 novembre 2026 : servir P (PanDerm appris sur ISIC 2019 seul),
seuil « sensibilité » d'ISIC 2019, présentée comme aide en dermoscopie (pas en dépistage).
Résultats qui fondent ce choix : DERM12345, mélanome 0,947 [0,930 ; 0,962] (critère J90
atteint, nævus intact, `results/panderm_derm12345.json`) ; seuils
(`results/panderm_p_thresholds.json`).

Aucune nouvelle mesure. Le script :
- réapprend P exactement comme dans les études (`fit_head`, graine 42, entraînement
  ISIC 2019) et vérifie que son empreinte est celle de la tête archivée
  (`panderm_derm12345.json`, `head_p.head_sha256`) ;
- vérifie que la tête redonne les scores archivés de validation et de test ISIC 2019
  (`results/panderm_snapshot_isic2019_*.npz`, écart <= 1e-5) ;
- calcule les seuils par classe comme pour la tête DINOv2-B (`skin_decision_thresholds.py` :
  F1 maximal et sensibilité >= 0,90 choisis sur la validation, rapportés une fois au test,
  liés à la tête) et vérifie que le seuil « sensibilité » du mélanome est celui archivé
  (`mel_thresholds.isic2019_val.sensitivity`).

Réserves : PanDerm sous CC BY-NC-ND 4.0, données ISIC CC BY-NC : tête non commerciale ;
dermoscopie uniquement ; le routeur de modalité reste DINOv2-B.

Exemple :
    python scripts/save_panderm_skin_head.py --isic-root data/raw/isic2019 --embeddings embeddings_derm \\
        --checkpoint third_party/panderm/panderm_large.pth
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

from scripts.decision_metrics import summarize  # noqa: E402
from scripts.dermatology_analysis import ImageList  # noqa: E402
from scripts.robustness_checks import fit_head  # noqa: E402
from scripts.skin_decision_thresholds import MAX_GAP, TARGET  # noqa: E402
from src.data.fundus_preprocess import PREPROCESS_VERSION  # noqa: E402
from src.data.isic_dataset import CLASS_NAMES, SKIN_TRANSFORM  # noqa: E402
from src.eval.decision import per_class_report  # noqa: E402
from src.eval.predictions import check_gap, require_finite  # noqa: E402
from src.eval.provenance import run_info  # noqa: E402
from src.models.classifier_head import head_contract, head_probe, head_sha256  # noqa: E402
from src.models.embedding_cache import KNOWN_CHECKPOINTS, file_fingerprint, load_embeddings  # noqa: E402
from src.models.timm_backbones import PANDERM_IMAGE_SIZE  # noqa: E402

STUDY = Path("results/panderm_derm12345.json")
THRESHOLDS_STUDY = Path("results/panderm_p_thresholds.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--isic-root", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="poids PanDerm-Large locaux")
    parser.add_argument("--out", type=Path, default=Path("heads/isic2019_panderm_head.pt"))
    parser.add_argument("--thresholds-out", type=Path, default=Path("results/decision_isic2019_panderm.json"))
    args = parser.parse_args()
    if file_fingerprint(args.checkpoint) != KNOWN_CHECKPOINTS["panderm"]:
        raise ValueError("poids PanDerm différents de ceux des études")

    sets = {s: ImageList(args.isic_root, s) for s in ("train", "val", "test")}
    X = {s: load_embeddings(args.embeddings, "isic2019", d.keys, "panderm", preprocess="none") for s, d in sets.items()}
    with contextlib.redirect_stdout(io.StringIO()):
        head = fit_head(X["train"], sets["train"].targets)
    archived_sha = json.loads(STUDY.read_text(encoding="utf-8"))["head_p"]["head_sha256"]
    if head_sha256(head.state_dict()) != archived_sha:
        raise ValueError("la tête réapprise n'est pas la tête P des études")

    data, gaps = {}, {}
    with torch.no_grad():
        for split in ("val", "test"):
            with np.load(f"results/panderm_snapshot_isic2019_{split}.npz") as ref:
                if list(ref["keys"]) != sets[split].keys:
                    raise ValueError(f"clés {split} différentes de l'instantané")
                archived = require_finite(ref["scores_P"], f"scores archivés {split}")
            scores = torch.sigmoid(head(torch.from_numpy(X[split]))).numpy()
            gaps[split] = float(np.abs(require_finite(scores, f"scores {split}") - archived).max())
            check_gap(gaps[split], MAX_GAP, f"la tête ne redonne pas les scores archivés ({split})")
            data[split] = (sets[split].targets, archived)

    state = head.cpu().state_dict()
    payload = {
        "state_dict": state,
        "class_names": list(CLASS_NAMES),
        "dataset": "isic2019",
        "modality": "dermoscopy",
        "backbone": "panderm",
        "image_size": PANDERM_IMAGE_SIZE,
        "preprocess": "skin",
        "transform": SKIN_TRANSFORM,
        "preprocess_version": PREPROCESS_VERSION,
        "checkpoint_sha256": file_fingerprint(args.checkpoint),
        "embeddings_checkpoint_sha256": KNOWN_CHECKPOINTS["panderm"],
        "head_training": {"lr": 1e-3, "epochs": 30, "batch_size": 16},
        "seed": 42,
        "git_commit": run_info()["git_commit"],
    }

    report = per_class_report(*data["val"], *data["test"], list(CLASS_NAMES), TARGET)
    mel = report["classes"]["MEL"]["threshold_sensitivity"]["threshold"]
    archived_mel = json.loads(THRESHOLDS_STUDY.read_text(encoding="utf-8"))["mel_thresholds"]["isic2019_val"]
    check_gap(abs(mel - archived_mel["sensitivity"]), 1e-9, "seuil « sensibilité » du mélanome différent de l'étude")
    report.update(
        dataset="isic2019",
        modality="dermoscopy",
        head_file=args.out.name,
        head_sha256=head_sha256(state),
        head_contract=head_contract(payload),
        head_probe=head_probe(state),
        head_check_max_abs_diff=gaps,
        run=run_info(),
        plan="décision du tuteur du 20 novembre 2026 : P, seuil « sensibilité » d'ISIC 2019 (docstring du script)",
        served_policy="sensitivity",
        target_sensitivity=TARGET,
        external_validation={"file": STUDY.name, "melanoma_auroc_derm12345": "0,947 [0,930 ; 0,962]"},
    )
    report["summary"] = summarize(report)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    args.thresholds_out.write_text(json.dumps(report, indent=1, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"Tête P vérifiée (empreinte archivée, écart max {max(gaps.values()):.1e}) -> {args.out}")
    for name, entry in report["classes"].items():
        if entry["insufficient_evidence"]:
            print(f"  {name:5s} preuve insuffisante")
            continue
        m = entry["threshold_sensitivity"]
        print(f"  {name:5s} seuil {m['threshold']:.4f} sens. {m['sensitivity']:.2f} spéc. {m['specificity']:.2f}")
    print(f"-> {args.thresholds_out}")


if __name__ == "__main__":
    main()

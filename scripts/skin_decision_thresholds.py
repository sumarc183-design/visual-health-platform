#!/usr/bin/env python3
"""Seuils de décision de la tête peau (DINOv2-B, ISIC 2019, 8 classes) (CPU).

Plan fixé le 13 novembre 2026, avant tout calcul de seuil. Jusqu'ici, `predict.py` ne
donne pour la peau que des probabilités : aucune décision n'est possible.

Méthode, la même que pour le fond d'œil (`scripts/decision_metrics.py`,
`src/eval/decision.per_class_report`) :
- scores archivés de l'analyse du 9 novembre (`results/dermatology_isic2019.npz`,
  DINOv2-B), après vérification que la tête enregistrée
  (`heads/isic2019_dinov2b_head.pt`) les redonne depuis le cache d'embeddings (écart
  maximal <= 1e-5, validation et test) ;
- seuils par classe choisis sur la validation (F1 maximal ; sensibilité >= 0,90 pour la
  politique dépistage), appliqués une seule fois au test ; « preuve insuffisante » sous
  10 positifs en validation ou en test ;
- seuils liés à la tête (`head_sha256`, contrat) : `predict.py` refuse une autre tête.

Rapportés, sans critère (étape d'usage, pas de choix de modèle) : sensibilité,
spécificité et VPP au test pour chaque classe et chaque politique ; attention particulière
au mélanome (MEL), dont l'oubli est l'erreur la plus grave.

Réserves : test ISIC 2019 de même source que l'apprentissage (pas de validation externe
en dermoscopie à ce jour) ; dermoscopie uniquement (perte de 0,21 d'AUROC sur photos de
smartphone, PAD-UFES-20) ; données CC BY-NC.

Exemple :
    python scripts/skin_decision_thresholds.py --isic-root data/raw/isic2019 --embeddings embeddings_derm
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.decision_metrics import summarize  # noqa: E402
from scripts.dermatology_analysis import ImageList  # noqa: E402
from src.data.isic_dataset import CLASS_NAMES  # noqa: E402
from src.eval.decision import per_class_report  # noqa: E402
from src.eval.provenance import run_info  # noqa: E402
from src.models.classifier_head import head_contract, head_probe, head_sha256  # noqa: E402
from src.models.embedding_cache import load_embeddings  # noqa: E402

MAX_GAP = 1e-5
TARGET = 0.90


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--isic-root", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--head", type=Path, default=Path("heads/isic2019_dinov2b_head.pt"))
    parser.add_argument("--reference", type=Path, default=Path("results/dermatology_isic2019.npz"))
    parser.add_argument("--out", type=Path, default=Path("results/decision_isic2019_dinov2b.json"))
    args = parser.parse_args()

    head = torch.load(args.head, map_location="cpu", weights_only=True)
    if list(head["class_names"]) != list(CLASS_NAMES):
        raise ValueError("classes de la tête différentes d'ISIC 2019")
    weight = head["state_dict"]["fc.weight"].double().numpy()
    bias = head["state_dict"]["fc.bias"].double().numpy()
    data, gaps = {}, {}
    with np.load(args.reference) as ref:
        for split in ("val", "test"):
            images = ImageList(args.isic_root, split)
            if list(ref[f"{split}_keys"]) != images.keys:
                raise ValueError(f"clés {split} différentes de l'archive")
            X = load_embeddings(args.embeddings, "isic2019", images.keys, "dinov2", preprocess="none")
            probs = ref[f"DINOv2-B_{split}"]
            recomputed = 1 / (1 + np.exp(-(X @ weight.T + bias)))
            gaps[split] = float(np.abs(recomputed - probs).max())
            if gaps[split] > MAX_GAP:
                raise ValueError(f"la tête ne redonne pas les scores archivés ({split}, écart {gaps[split]:.1e})")
            data[split] = (images.targets, probs)

    report = per_class_report(*data["val"], *data["test"], list(CLASS_NAMES), TARGET)
    report.update(
        dataset="isic2019",
        modality="dermoscopy",
        head_file=args.head.name,
        head_sha256=head_sha256(head["state_dict"]),
        head_contract=head_contract(head),
        head_probe=head_probe(head["state_dict"]),
        head_check_max_abs_diff=gaps,
        run=run_info(),
        plan="fixé le 13 novembre 2026 avant tout calcul de seuil (docstring du script)",
        target_sensitivity=TARGET,
    )
    report["summary"] = summarize(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"Tête vérifiée (écart max {max(gaps.values()):.1e})")
    for name, entry in report["classes"].items():
        if entry["insufficient_evidence"]:
            print(f"  {name:5s} preuve insuffisante")
            continue
        parts = []
        for key in ("threshold_f1", "threshold_sensitivity"):
            m = entry[key]
            parts.append(f"{key[10:]:11s} sens. {m['sensitivity']:.2f} spéc. {m['specificity']:.2f} VPP {m['ppv']:.2f}")
        print(f"  {name:5s} " + " | ".join(parts))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()

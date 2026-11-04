#!/usr/bin/env python3
"""Prédit les maladies sur une ou plusieurs images de fond d'œil nouvelles.

Même chaîne que l'entraînement (src/pipeline/inference.py) : contrôle
qualité, prétraitement enregistré dans la tête, RETFound figé, tête, puis
décision par maladie avec les seuils choisis sur la validation
(results/decision_<dataset>.json, produits par scripts/decision_metrics.py).

Politiques de seuil :
- `f1` (défaut) : compromis entre maladies ratées et fausses alertes ;
- `sensitivity` : dépistage, vise à rater peu de malades (sensibilité
  ≥ 0,90 en validation) au prix de nombreuses fausses alertes.

Les seuils publiés correspondent aux têtes du run publié (crop-mask,
30 epochs, graine 42) ; avec une autre tête, recalculer les seuils.

Outil de recherche, non validé cliniquement : les décisions ne sont pas un
diagnostic.

Exemple :
    python predict.py image1.jpg image2.png \\
        --head heads/odir_crop-mask_head.pt \\
        --checkpoint third_party/retfound/checkpoints/RETFound_mae_natureCFP.pth \\
        --thresholds results/decision_odir.json --policy f1 --out predictions.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pipeline.inference import DISCLAIMER, THRESHOLD_POLICIES, Predictor  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", type=Path, nargs="+", help="Images de fond d'œil")
    parser.add_argument("--head", type=Path, required=True, help="Tête sauvegardée par run_baseline.py --save-head")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint RETFound (celui de la tête)")
    parser.add_argument("--thresholds", type=Path, default=None, help="results/decision_<dataset>.json")
    parser.add_argument("--policy", choices=THRESHOLD_POLICIES, default="f1")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--trust-checkpoint", action="store_true", help="Voir run_baseline.py --trust-checkpoint")
    parser.add_argument(
        "--reject-unusable", action="store_true", help="Pas de prédiction si le contrôle qualité échoue"
    )
    parser.add_argument("--out", type=Path, default=None, help="Résultats JSON")
    args = parser.parse_args()

    predictor = Predictor(
        args.head,
        args.checkpoint,
        args.thresholds,
        policy=args.policy,
        device=args.device,
        trust_checkpoint=True if args.trust_checkpoint else None,
    )
    results = []
    for image in args.images:
        prediction = predictor.predict(image)
        if args.reject_unusable and not prediction.quality["usable"]:
            print(f"\n{image} : refusée par le contrôle qualité ({' ; '.join(prediction.quality['reasons'])})")
            results.append({"image": str(image), "rejected": True, "quality": prediction.quality})
            continue
        results.append(prediction.to_dict())
        print(f"\n{image}")
        for warning in prediction.warnings:
            print(f"  ⚠ {warning}")
        for name, finding in sorted(prediction.findings.items(), key=lambda kv: -kv[1]["probability"]):
            decision = finding["decision"] or "pas de décision (preuve insuffisante)"
            print(f"  {name:>6s} : {finding['probability']:.3f}  {decision}")
    print(f"\n{DISCLAIMER}")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"Résultats : {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prédit les maladies sur une ou plusieurs images de fond d'œil nouvelles.

Même chaîne que l'entraînement (src/pipeline/inference.py) : contrôle
qualité, prétraitement enregistré dans la tête, modèle figé, tête, puis
décision par maladie avec les seuils choisis sur la validation
(results/decision_<dataset>.json, produits par scripts/decision_metrics.py).

Politiques de seuil :
- `f1` (défaut) : compromis entre maladies ratées et fausses alertes ;
- `sensitivity` : dépistage, vise à rater peu de malades (sensibilité
  ≥ 0,90 en validation) au prix de nombreuses fausses alertes.

Les seuils publiés correspondent aux têtes du run publié (crop-mask,
30 epochs, graine 42) ; avec une autre tête, recalculer les seuils.

Le modèle se choisit avec la tête, qui indique le sien : RETFound (têtes RFMiD
et ODIR) ou RETFound-Green (tête RFMiD : meilleure généralisation externe sur
la rétinopathie diabétique, moins bon sur la DMLA ; voir README). Le
checkpoint doit être celui de la tête, sinon la prédiction est refusée.

`--modality-check` (désactivé par défaut) : le routeur de modalité vérifie que
l'image ressemble à un fond d'œil et avertit sinon, sans jamais bloquer
(`src/pipeline/modality_check.py`). Demande les poids DINOv2-B du routeur.

Outil de recherche, non validé cliniquement : les décisions ne sont pas un
diagnostic.

Exemple :
    python predict.py image1.jpg image2.png \\
        --head heads/odir_crop-mask_head.pt \\
        --checkpoint third_party/retfound/checkpoints/RETFound_mae_natureCFP.pth \\
        --thresholds results/decision_odir.json --policy f1 --out predictions.json
    python predict.py image1.jpg \\
        --head heads/rfmid_green_head.pt \\
        --checkpoint third_party/retfound_green/retfoundgreen_statedict.pth \\
        --thresholds results/decision_rfmid_green.json
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
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint du modèle (celui de la tête)")
    parser.add_argument("--thresholds", type=Path, default=None, help="results/decision_<dataset>.json")
    parser.add_argument("--policy", choices=THRESHOLD_POLICIES, default="f1")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--trust-checkpoint", action="store_true", help="Voir run_baseline.py --trust-checkpoint")
    parser.add_argument(
        "--reject-unusable", action="store_true", help="Pas de prédiction si le contrôle qualité échoue"
    )
    parser.add_argument(
        "--allow-unverified-thresholds",
        action="store_true",
        help="Accepte un fichier de seuils sans identité de tête (historique) : compatibilité non vérifiée",
    )
    parser.add_argument(
        "--modality-check", action="store_true", help="Avertit si le routeur ne voit pas un fond d'œil (non bloquant)"
    )
    parser.add_argument("--router-head", type=Path, default=Path("heads/modality_router_v2.pt"))
    parser.add_argument("--router-checkpoint", type=Path, default=Path("third_party/dinov2/dinov2_vitb14_timm.pth"))
    parser.add_argument("--out", type=Path, default=None, help="Résultats JSON")
    args = parser.parse_args()

    predictor = Predictor(
        args.head,
        args.checkpoint,
        args.thresholds,
        policy=args.policy,
        device=args.device,
        trust_checkpoint=True if args.trust_checkpoint else None,
        allow_unverified_thresholds=args.allow_unverified_thresholds,
    )
    check = None
    if args.modality_check:
        from src.pipeline.modality_check import ModalityCheck

        check = ModalityCheck(args.router_head, args.router_checkpoint)
    results = []
    for image in args.images:
        prediction = predictor.predict(image, reject_unusable=args.reject_unusable)
        if check is not None:
            seen = check(image)
            prediction.modality = seen.to_dict()
            if seen.warning:
                prediction.warnings.append(seen.warning)
        if prediction.rejected:  # contrôle qualité échoué : pas passée dans le modèle
            print(f"\n{image} : refusée par le contrôle qualité ({' ; '.join(prediction.quality['reasons'])})")
            results.append(prediction.to_dict())
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

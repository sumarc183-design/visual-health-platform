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

Peau : tête `heads/isic2019_dinov2b_head.pt` (DINOv2-B, ISIC 2019, 8 classes,
probabilités seules) avec `--checkpoint third_party/dinov2/dinov2_vitb14_timm.pth`.
Dermoscopie uniquement : sur des photos de smartphone, elle perd 0,21 d'AUROC
(PAD-UFES-20, docs/DERMATOLOGIE.md) ; le routeur ne sait pas faire ce tri.

Contrôle du type d'image (actif par défaut depuis le 12 novembre, décision du tuteur) :
le routeur de modalité vérifie que l'image ressemble au type de la tête (fond d'œil ou
peau) et avertit sinon, sans jamais bloquer (`src/pipeline/modality_check.py`). Sans les
poids DINOv2-B du routeur, la prédiction se fait avec un avertissement ;
`--no-modality-check` le désactive.

Dépistage de la rétinopathie diabétique : `--dr-screening` charge la tête Green
apprise sur RFMiD + EyePACS (`heads/dr_screening_green_head.pt`) et son seuil choisi sur
la validation EyePACS (politique `sensitivity`). Sur DDR (Chine, jamais utilisée pour
régler) : 87 % des rétinopathies repérées, mais une fausse alerte pour deux personnes
saines environ (spécificité 0,52). La tête RFMiD seule n'en repérait que 27 %.

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
    python predict.py image1.jpg --dr-screening
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pipeline.inference import DISCLAIMER, THRESHOLD_POLICIES, Predictor  # noqa: E402

# Réglage « dépistage » de la rétinopathie diabétique (scripts/dr_screening_sensitivity.py,
# critère atteint le 12 novembre 2026 : sensibilité sur DDR 0,87 [0,86 ; 0,88], spécificité 0,52).
DR_SCREENING = {
    "head": Path("heads/dr_screening_green_head.pt"),
    "checkpoint": Path("third_party/retfound_green/retfoundgreen_statedict.pth"),
    "thresholds": Path("results/decision_dr_screening.json"),
    "policy": "sensitivity",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", type=Path, nargs="+", help="Images (fond d'œil, ou peau avec une tête peau)")
    parser.add_argument("--head", type=Path, default=None, help="Tête sauvegardée par run_baseline.py --save-head")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint du modèle (celui de la tête)")
    parser.add_argument("--thresholds", type=Path, default=None, help="results/decision_<dataset>.json")
    parser.add_argument("--policy", choices=THRESHOLD_POLICIES, default=None, help="f1 par défaut")
    parser.add_argument(
        "--dr-screening",
        action="store_true",
        help="Réglage dépistage de la rétinopathie diabétique (tête Green RFMiD + EyePACS, seuil sensibilité)",
    )
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
        "--no-modality-check",
        dest="modality_check",
        action="store_false",
        help="Désactive l'avertissement du routeur (actif par défaut)",
    )
    parser.add_argument("--router-head", type=Path, default=Path("heads/modality_router_v2.pt"))
    parser.add_argument("--router-checkpoint", type=Path, default=Path("third_party/dinov2/dinov2_vitb14_timm.pth"))
    parser.add_argument("--out", type=Path, default=None, help="Résultats JSON")
    args = parser.parse_args()
    if args.dr_screening:
        if args.policy == "f1":
            parser.error("--dr-screening utilise la politique sensitivity (seule évaluée pour cette tête)")
        for name, value in DR_SCREENING.items():
            if getattr(args, name) is None:
                setattr(args, name, value)
    args.policy = args.policy or "f1"
    if args.head is None or args.checkpoint is None:
        parser.error("--head et --checkpoint sont requis (ou --dr-screening)")

    predictor = Predictor(
        args.head,
        args.checkpoint,
        args.thresholds,
        policy=args.policy,
        device=args.device,
        trust_checkpoint=True if args.trust_checkpoint else None,
        allow_unverified_thresholds=args.allow_unverified_thresholds,
    )
    check, extra = None, []
    if args.modality_check:
        from src.pipeline.modality_check import ModalityCheck

        if args.router_checkpoint.exists() and args.router_head.exists():
            check = ModalityCheck(args.router_head, args.router_checkpoint)
        else:
            extra.append(f"Type d'image non vérifié : poids du routeur absents ({args.router_checkpoint}).")
    results = []
    for image in args.images:
        prediction = predictor.predict(image, reject_unusable=args.reject_unusable)
        prediction.warnings.extend(extra)
        if check is not None:
            seen = check(image, declared=predictor.modality)
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
            missing = (
                "probabilité seule (pas de seuil)"
                if args.thresholds is None
                else "pas de décision (preuve insuffisante)"
            )
            decision = finding["decision"] or missing
            print(f"  {name:>6s} : {finding['probability']:.3f}  {decision}")
    print(f"\n{DISCLAIMER}")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"Résultats : {args.out}")


if __name__ == "__main__":
    main()

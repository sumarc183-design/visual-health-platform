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

Peau (aide en dermoscopie, pas un outil de dépistage) : `--skin` charge la tête servie
depuis le 20 novembre (décision du tuteur), PanDerm-Large appris sur ISIC 2019
(`heads/isic2019_panderm_head.pt`, 8 classes), avec le seuil « sensibilité » choisi sur
la validation ISIC 2019 (`results/decision_isic2019_panderm.json`). Validation externe
sur DERM12345 (Turquie, base vue ni par PanDerm ni par ISIC) : AUROC du mélanome 0,947
[0,930 ; 0,962] ; à ce seuil, 92 % des mélanomes repérés et 20 % de fausses alertes. En
dépistage de population (ISIC 2020, 1,8 % de mélanomes), aucun seuil ne donne de
compromis acceptable. Poids PanDerm sous CC BY-NC-ND 4.0 : non commercial. L'ancienne
tête DINOv2-B (`heads/isic2019_dinov2b_head.pt`, `results/decision_isic2019_dinov2b.json`)
reste utilisable avec `--head`.
Contrôle qualité propre à la peau (flou au centre de l'image, seuil calibré sur ISIC
2019 ; `--reject-unusable` refuse les images floues). Dermoscopie uniquement : sur des
photos de smartphone, elle perd 0,21 d'AUROC (PAD-UFES-20, docs/DERMATOLOGIE.md) ; le
routeur ne sait pas faire ce tri.

Contrôle du type d'image (actif par défaut depuis le 12 novembre, décision du tuteur) :
le routeur de modalité vérifie que l'image ressemble au type de la tête (fond d'œil ou
peau) et avertit sinon, sans jamais bloquer (`src/pipeline/modality_check.py`). Sans les
poids DINOv2-B du routeur, la prédiction se fait avec un avertissement ;
`--no-modality-check` le désactive.

Dépistage de la rétinopathie diabétique : `--dr-screening` charge la tête Green
apprise sur RFMiD + EyePACS (`heads/dr_screening_green_head.pt`) et son seuil choisi sur
la validation EyePACS (politique `sensitivity`). Sur Messidor-2 (France, vue par aucun
des modèles, 16 nov.) : 83 % des rétinopathies repérées [80 % ; 86 %], 94 % des cas à
adresser, mais spécificité 0,41 (plus d'une fausse alerte pour deux images saines) ;
IDRiD (Inde) : 96 %, spécificité 0,48. Sur DDR : 87 %, spécificité 0,52, chiffres
peut-être optimistes (DDR fait partie du pré-entraînement de Green, découvert le 13 nov.).

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
    python predict.py lesion.jpg --skin
    python predict.py image1.jpg --dr-screening --explain cartes/

Localisation (`--explain DOSSIER`) : pour chaque maladie positive (3 au plus), une image
PNG montre en rouge les zones dont le masquage fait baisser le score (carte d'occlusion,
mesurée sur IDRiD le 6 novembre pour la rétinopathie : avec Green, les cartes suivent les
lésions au-delà de leur position typique et évitent le disque optique, mais l'accord reste
modeste, AUROC au pixel 0,66 ; avec RETFound, pas mieux que la position ; pas de validation
externe, rien de mesuré pour les autres maladies ni pour la peau). Ce n'est pas une
segmentation des lésions. Durée : voir docs/GUIDE.md.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.explainability.occlusion import overlay  # noqa: E402
from src.pipeline.inference import DISCLAIMER, THRESHOLD_POLICIES, Prediction, Predictor  # noqa: E402

# Réglage « dépistage » de la rétinopathie diabétique (scripts/dr_screening_sensitivity.py,
# critère atteint le 12 novembre 2026 sur DDR, base du pré-entraînement de Green ; reconfirmé
# le 16 novembre sur Messidor-2 : sensibilité 0,832 [0,802 ; 0,859], spécificité 0,41).
DR_SCREENING = {
    "head": Path("heads/dr_screening_green_head.pt"),
    "checkpoint": Path("third_party/retfound_green/retfoundgreen_statedict.pth"),
    "thresholds": Path("results/decision_dr_screening.json"),
    "policy": "sensitivity",
}
# Tête peau servie (décision du tuteur du 20 novembre 2026, scripts/save_panderm_skin_head.py) :
# PanDerm-Large appris sur ISIC 2019 seul (« P »), seuil « sensibilité » de la validation ISIC 2019.
SKIN = {
    "head": Path("heads/isic2019_panderm_head.pt"),
    "checkpoint": Path("third_party/panderm/panderm_large.pth"),
    "thresholds": Path("results/decision_isic2019_panderm.json"),
    "policy": "sensitivity",
}


def explain(predictor: Predictor, prediction: Prediction, image: Path, out_dir: Path, limit: int) -> list[dict]:
    """Cartes d'occlusion des maladies positives (les plus probables d'abord, `limit` au plus) ;
    sans maladie positive, celle de la maladie la plus probable, signalée comme telle."""
    ranked = sorted(prediction.findings.items(), key=lambda kv: -kv[1]["probability"])
    chosen = [(name, "positif") for name, f in ranked if f["decision"] == "positif"][:limit]
    if not chosen:
        chosen = [(ranked[0][0], "aucune maladie positive : la plus probable")]
    out_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for name, why in chosen:
        result = predictor.explain(image, name)
        path = out_dir / f"{image.stem}_{name}.png"
        Image.fromarray(overlay(result["image"], result["heat"])).save(path)
        share = result.get("outside_retina_share")
        items.append(
            {
                "class": name,
                "why": why,
                "file": str(path),
                "logit": result["logit"],
                "patch": result["patch"],
                "stride": result["stride"],
                "outside_retina_share": None if share is None or math.isnan(share) else share,
            }
        )
    return items


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
    parser.add_argument(
        "--skin",
        action="store_true",
        help="Tête peau servie (PanDerm, ISIC 2019, seuil sensibilité) : aide en dermoscopie",
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
    parser.add_argument(
        "--explain",
        type=Path,
        default=None,
        metavar="DOSSIER",
        help="Enregistre la carte d'occlusion (zones qui font monter le score) des maladies positives",
    )
    parser.add_argument("--explain-max", type=int, default=3, help="Nombre maximal de cartes par image (3)")
    args = parser.parse_args()
    if args.explain_max < 1:
        parser.error("--explain-max doit valoir au moins 1")
    if args.dr_screening and args.skin:
        parser.error("--dr-screening et --skin s'excluent")
    for flag, preset in (("dr_screening", DR_SCREENING), ("skin", SKIN)):
        if not getattr(args, flag):
            continue
        if args.policy == "f1":
            parser.error(f"--{flag.replace('_', '-')} utilise la politique sensitivity (seule retenue pour cette tête)")
        for name, value in preset.items():
            if getattr(args, name) is None:
                setattr(args, name, value)
    args.policy = args.policy or "f1"
    if args.head is None or args.checkpoint is None:
        parser.error("--head et --checkpoint sont requis (ou --dr-screening, ou --skin)")

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
        if args.explain is not None:
            prediction.explanations = explain(predictor, prediction, image, args.explain, args.explain_max)
        results.append(prediction.to_dict())
        print(f"\n{image}")
        for warning in prediction.warnings:
            print(f"  ⚠ {warning}")
        for item in prediction.explanations or []:
            share = item["outside_retina_share"]
            outside = "" if share is None else f", {share:.0%} de l'effet hors de la rétine"
            print(f"  Carte {item['class']} ({item['why']}{outside}) : {item['file']}")
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
        args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        print(f"Résultats : {args.out}")


if __name__ == "__main__":
    main()

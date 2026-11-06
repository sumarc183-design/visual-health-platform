"""Prédiction sur une image nouvelle : même chaîne que l'entraînement.

contrôle qualité (fichier d'origine) -> prétraitement de la tête (`none` ou
`crop-mask`, taille d'image) -> normalisation du modèle -> modèle figé
(RETFound ou RETFound-Green, indiqué par la tête) -> tête -> probabilités ->
décision par maladie avec les seuils choisis sur la validation
(`scripts/decision_metrics.py`).

Outil de recherche : les décisions ne sont pas un diagnostic. Les maladies
marquées « preuve insuffisante » (moins de 10 cas en validation ou en test)
reçoivent une probabilité mais aucune décision.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch

from src.data.fundus_preprocess import PREPROCESS_VERSION, load_fundus_image
from src.models.classifier_head import MultiLabelHead, head_contract, head_sha256
from src.models.embedding_cache import file_fingerprint
from src.models.retfound_backbone import IMAGENET_MEAN, IMAGENET_STD, RETFoundBackbone
from src.models.retfound_green import RETFoundGreenBackbone
from src.pipeline.quality_control import DEFAULT_BLUR_THRESHOLD, check_fundus_file

THRESHOLD_POLICIES = ("f1", "sensitivity")
BACKBONES = ("retfound", "green")
DISCLAIMER = "Outil de recherche, non validé cliniquement : ces décisions ne sont pas un diagnostic."


@dataclass
class Prediction:
    image: str
    quality: dict
    findings: dict  # maladie -> {"probability", "threshold", "decision"}
    warnings: list[str] = field(default_factory=list)
    rejected: bool = False

    def to_dict(self) -> dict:
        return {
            "image": self.image,
            "rejected": self.rejected,
            "quality": self.quality,
            "findings": self.findings,
            "warnings": self.warnings,
            "disclaimer": DISCLAIMER,
        }


def load_thresholds(
    path: str | Path,
    class_names: list[str],
    policy: str,
    head: dict | None = None,
    allow_unverified: bool = False,
) -> tuple[dict, bool]:
    """Seuils par maladie d'un fichier results/decision_<dataset>.json. None =
    pas de décision (preuve insuffisante, ou aucun positif en validation).

    `head` : tête chargée (state_dict, class_names, image_size...). Le fichier
    doit porter l'identité exacte de la tête qui a produit ses prédictions
    (`head_sha256`) et le même contrat (modèle, prétraitement, taille, classes) :
    une autre tête est refusée, même de mêmes classes et de même sonde. Fichier
    sans identité (historique) : refusé, sauf `allow_unverified`.
    Renvoie (seuils, compatibilité vérifiée)."""
    if policy not in THRESHOLD_POLICIES:
        raise ValueError(f"politique de seuil {policy!r} inconnue, choisir parmi {THRESHOLD_POLICIES}")
    report = json.loads(Path(path).read_text())
    classes = report["classes"]
    if list(classes) != list(class_names):
        raise ValueError(
            f"Les seuils ({path}) ne portent pas sur les mêmes classes que la tête : "
            f"{list(classes)[:4]}... contre {list(class_names)[:4]}..."
        )
    stored = report.get("head_sha256")
    if stored is None or head is None:
        if not allow_unverified:
            raise ValueError(
                f"Les seuils ({path}) ne sont pas liés à une tête (pas de « head_sha256 ») : compatibilité "
                "invérifiable. Regénérer les seuils avec scripts/decision_metrics.py --head --embeddings, ou "
                "accepter explicitement (--allow-unverified-thresholds)."
            )
        verified = False
    elif stored != head_sha256(head["state_dict"]):
        raise ValueError(f"Les seuils ({path}) ont été calculés pour une autre tête que celle chargée.")
    elif report.get("head_contract") != head_contract(head):
        raise ValueError(
            f"Les seuils ({path}) supposent une autre chaîne : {report.get('head_contract')} "
            f"contre {head_contract(head)}."
        )
    else:
        verified = True
    thresholds = {}
    for name, entry in classes.items():
        decided = entry.get(f"threshold_{policy}")
        value = None if entry["insufficient_evidence"] or decided is None else decided["threshold"]
        if value is not None and not (math.isfinite(value) and 0.0 <= value <= 1.0):
            raise ValueError(f"Seuil invalide pour {name} : {value} (attendu entre 0 et 1)")
        thresholds[name] = value
    return thresholds, verified


class Predictor:
    def __init__(
        self,
        head_path: str | Path,
        checkpoint_path: str | Path,
        thresholds_path: str | Path | None = None,
        policy: str = "f1",
        device: str = "cpu",
        trust_checkpoint: bool | None = None,
        blur_threshold: float = DEFAULT_BLUR_THRESHOLD,
        allow_unverified_thresholds: bool = False,
    ):
        head_ckpt = torch.load(head_path, map_location="cpu", weights_only=True)
        self.backbone_name = head_ckpt.get("backbone", "retfound")  # têtes historiques : RETFound
        if self.backbone_name not in BACKBONES:
            raise ValueError(f"Modèle {self.backbone_name!r} inconnu, attendu parmi {BACKBONES}")
        self.class_names = list(head_ckpt["class_names"])
        self.image_size = head_ckpt["image_size"]
        self.preprocess = head_ckpt.get("preprocess", "none")
        self.device = device
        self.blur_threshold = blur_threshold
        self.warnings: list[str] = []

        expected_version = head_ckpt.get("preprocess_version")
        if expected_version is None:
            self.warnings.append("Tête sauvegardée sans version de prétraitement : version actuelle supposée.")
        elif expected_version != PREPROCESS_VERSION:
            raise ValueError(
                f"La tête a été entraînée avec le prétraitement v{expected_version}, le code est en "
                f"v{PREPROCESS_VERSION} : les images ne seraient pas préparées de la même façon."
            )
        expected_checkpoint = head_ckpt.get("checkpoint_sha256")
        if expected_checkpoint is None:
            self.warnings.append("Tête sauvegardée sans empreinte du checkpoint : compatibilité non vérifiée.")
        elif expected_checkpoint != file_fingerprint(checkpoint_path):
            raise ValueError(
                f"Le checkpoint ({self.backbone_name}) n'est pas celui avec lequel la tête a été entraînée."
            )

        if thresholds_path is not None:
            self.thresholds, verified = load_thresholds(
                thresholds_path,
                self.class_names,
                policy,
                head=head_ckpt,
                allow_unverified=allow_unverified_thresholds,
            )
            if not verified:
                self.warnings.append("Seuils non liés à cette tête (fichier historique) : compatibilité non vérifiée.")
        else:
            self.thresholds = dict.fromkeys(self.class_names)
            self.warnings.append("Aucun fichier de seuils : probabilités seules, sans décision.")

        if self.backbone_name == "green":
            backbone = RETFoundGreenBackbone(checkpoint_path, img_size=self.image_size)
        else:
            backbone = RETFoundBackbone(
                checkpoint_path, img_size=self.image_size, embedding_dim=1024, trust_checkpoint=trust_checkpoint
            )
        self.backbone = backbone.to(device)
        self.input_mean = getattr(backbone, "input_mean", IMAGENET_MEAN).to(device)
        self.input_std = getattr(backbone, "input_std", IMAGENET_STD).to(device)
        self.head = MultiLabelHead(embedding_dim=backbone.embedding_dim, num_classes=len(self.class_names))
        self.head.load_state_dict(head_ckpt["state_dict"])
        self.head.to(device).eval()

    @torch.no_grad()
    def probabilities(self, image_path: str | Path) -> torch.Tensor:
        image = load_fundus_image(image_path, self.image_size, self.preprocess).unsqueeze(0).to(self.device)
        image = (image - self.input_mean) / self.input_std
        return torch.sigmoid(self.head(self.backbone(image)))[0].cpu()

    def predict(self, image_path: str | Path, reject_unusable: bool = False) -> Prediction:
        """Contrôle qualité d'abord ; avec `reject_unusable`, une image qui
        échoue n'est pas passée dans le modèle (`rejected`, aucune décision)."""
        report = check_fundus_file(image_path, blur_threshold=self.blur_threshold)
        quality = {"usable": report.is_usable, "blur_score": report.blur_score, "reasons": report.reasons}
        warnings = list(self.warnings)
        if not report.is_usable:
            if reject_unusable:
                return Prediction(image=str(image_path), quality=quality, findings={}, warnings=warnings, rejected=True)
            reasons = " ; ".join(report.reasons)
            warnings.append(f"Contrôle qualité : {reasons}. Résultat à interpréter avec prudence.")
        probabilities = self.probabilities(image_path)
        findings = {}
        for name, probability in zip(self.class_names, probabilities.tolist(), strict=True):
            threshold = self.thresholds[name]
            findings[name] = {
                "probability": probability,
                "threshold": threshold,
                "decision": None if threshold is None else ("positif" if probability >= threshold else "négatif"),
            }
        return Prediction(
            image=str(image_path),
            quality=quality,
            findings=findings,
            warnings=warnings,
        )

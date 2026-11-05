"""Prédiction sur une image nouvelle : même chaîne que l'entraînement.

contrôle qualité (fichier d'origine) -> prétraitement de la tête (`none` ou
`crop-mask`, taille d'image) -> normalisation ImageNet -> RETFound figé ->
tête -> probabilités -> décision par maladie avec les seuils choisis sur la
validation (`scripts/decision_metrics.py`).

Outil de recherche : les décisions ne sont pas un diagnostic. Les maladies
marquées « preuve insuffisante » (moins de 10 cas en validation ou en test)
reçoivent une probabilité mais aucune décision.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch

from src.data.fundus_preprocess import PREPROCESS_VERSION, load_fundus_image
from src.models.classifier_head import MultiLabelHead
from src.models.embedding_cache import file_fingerprint
from src.models.retfound_backbone import IMAGENET_MEAN, IMAGENET_STD, RETFoundBackbone
from src.pipeline.quality_control import DEFAULT_BLUR_THRESHOLD, check_fundus_file

THRESHOLD_POLICIES = ("f1", "sensitivity")
DISCLAIMER = "Outil de recherche, non validé cliniquement : ces décisions ne sont pas un diagnostic."


@dataclass
class Prediction:
    image: str
    quality: dict
    findings: dict  # maladie -> {"probability", "threshold", "decision"}
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "image": self.image,
            "quality": self.quality,
            "findings": self.findings,
            "warnings": self.warnings,
            "disclaimer": DISCLAIMER,
        }


def load_thresholds(path: str | Path, class_names: list[str], policy: str) -> dict:
    """Seuils par maladie d'un fichier results/decision_<dataset>.json. None =
    pas de décision (preuve insuffisante, ou aucun positif en validation)."""
    if policy not in THRESHOLD_POLICIES:
        raise ValueError(f"politique de seuil {policy!r} inconnue, choisir parmi {THRESHOLD_POLICIES}")
    report = json.loads(Path(path).read_text())
    classes = report["classes"]
    if list(classes) != list(class_names):
        raise ValueError(
            f"Les seuils ({path}) ne portent pas sur les mêmes classes que la tête : "
            f"{list(classes)[:4]}... contre {list(class_names)[:4]}..."
        )
    thresholds = {}
    for name, entry in classes.items():
        decided = entry.get(f"threshold_{policy}")
        thresholds[name] = None if entry["insufficient_evidence"] or decided is None else decided["threshold"]
    return thresholds


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
    ):
        head_ckpt = torch.load(head_path, map_location="cpu", weights_only=True)
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
            raise ValueError("Le checkpoint RETFound n'est pas celui avec lequel la tête a été entraînée.")

        self.thresholds = (
            load_thresholds(thresholds_path, self.class_names, policy)
            if thresholds_path is not None
            else dict.fromkeys(self.class_names)
        )
        if thresholds_path is None:
            self.warnings.append("Aucun fichier de seuils : probabilités seules, sans décision.")

        self.backbone = RETFoundBackbone(
            checkpoint_path, img_size=self.image_size, embedding_dim=1024, trust_checkpoint=trust_checkpoint
        ).to(device)
        self.head = MultiLabelHead(embedding_dim=1024, num_classes=len(self.class_names))
        self.head.load_state_dict(head_ckpt["state_dict"])
        self.head.to(device).eval()

    @torch.no_grad()
    def probabilities(self, image_path: str | Path) -> torch.Tensor:
        image = load_fundus_image(image_path, self.image_size, self.preprocess).unsqueeze(0).to(self.device)
        image = (image - IMAGENET_MEAN.to(self.device)) / IMAGENET_STD.to(self.device)
        return torch.sigmoid(self.head(self.backbone(image)))[0].cpu()

    def predict(self, image_path: str | Path) -> Prediction:
        report = check_fundus_file(image_path, blur_threshold=self.blur_threshold)
        warnings = list(self.warnings)
        if not report.is_usable:
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
            quality={"usable": report.is_usable, "blur_score": report.blur_score, "reasons": report.reasons},
            findings=findings,
            warnings=warnings,
        )

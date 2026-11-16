"""Chargement strict d'une tête servie (et de ses seuils) dans les scripts d'analyse.

Même liaison que `Predictor` (audit du 15 novembre, constat 3) : poids finis, checkpoint
du modèle figé identique à celui de la tête, seuils liés à la tête par `head_sha256` et
`head_contract` (refus d'un fichier historique non lié). Renvoie aussi l'identité
effectivement lue, à enregistrer dans les résultats.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from src.models.classifier_head import MultiLabelHead, check_finite_state_dict, head_contract, head_sha256
from src.models.embedding_cache import file_fingerprint


@dataclass
class ServedHead:
    artifact: dict
    head: MultiLabelHead
    thresholds: dict | None  # rapport de seuils complet, vérifié ; None si non demandé
    identity: dict

    def probabilities(self, X) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.head(torch.as_tensor(X, dtype=torch.float32))).numpy()


def load_served_head(head_path: Path, checkpoint_fingerprint: str, thresholds_path: Path | None = None) -> ServedHead:
    """`checkpoint_fingerprint` : empreinte (`file_fingerprint`) du modèle figé qui a produit
    les embeddings ; acceptée si elle est celle de la tête (ou de ses embeddings d'origine)."""
    from src.pipeline.inference import load_thresholds  # import local : inference importe les backbones

    artifact = torch.load(head_path, map_location="cpu", weights_only=True)
    check_finite_state_dict(artifact["state_dict"], str(head_path))
    known = {artifact.get("checkpoint_sha256"), artifact.get("embeddings_checkpoint_sha256")} - {None}
    if checkpoint_fingerprint not in known:
        raise ValueError(f"{head_path} : tête calculée avec un autre modèle figé ({checkpoint_fingerprint})")
    weight = artifact["state_dict"]["fc.weight"]
    head = MultiLabelHead(weight.shape[1], len(artifact["class_names"]))
    head.load_state_dict(artifact["state_dict"])
    head.eval()
    identity = {
        "head_file": Path(head_path).name,
        "head_sha256": head_sha256(artifact["state_dict"]),
        "head_contract": head_contract(artifact),
        "checkpoint_sha256": checkpoint_fingerprint,
    }
    report = None
    if thresholds_path is not None:
        _, verified = load_thresholds(thresholds_path, list(artifact["class_names"]), "sensitivity", head=artifact)
        if not verified:
            raise ValueError(f"{thresholds_path} : seuils non liés à {head_path}")
        report = json.loads(Path(thresholds_path).read_text(encoding="utf-8"))
        identity["thresholds_file"] = Path(thresholds_path).name
        identity["thresholds_fingerprint"] = file_fingerprint(thresholds_path)
    return ServedHead(artifact, head, report, identity)

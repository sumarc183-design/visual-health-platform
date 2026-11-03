"""Vérifie le classifieur de modalité (étape 2 du pipeline, encore sans
données d'entraînement) : sortie aux bonnes dimensions et noms de modalités."""

import torch

from src.pipeline.modality_classifier import MODALITIES, ModalityClassifier


def test_predict_modality_returns_known_names():
    model = ModalityClassifier(pretrained=False).eval()
    images = torch.rand(2, 3, 224, 224)
    with torch.no_grad():
        logits = model(images)
        names = model.predict_modality(images)
    assert logits.shape == (2, len(MODALITIES))
    assert len(names) == 2 and set(names) <= set(MODALITIES)

"""Vérifie la prédiction sur image nouvelle (src/pipeline/inference.py, predict.py)
avec un checkpoint RETFound à poids aléatoires."""

import gc
import json
import sys

import numpy as np
import pytest
import timm
import torch
from PIL import Image

from run_baseline import build_random_checkpoint, collect_embeddings
from src.data.fundus_preprocess import PREPROCESS_VERSION, load_fundus_image
from src.models.classifier_head import MultiLabelHead, head_contract, head_probe, head_sha256, probes_match
from src.models.embedding_cache import file_fingerprint
from src.models.retfound_green import GREEN_EMBEDDING_DIM, RETFoundGreenBackbone
from src.pipeline.inference import Predictor, load_thresholds

CLASSES = ["DR", "MYA", "RARE"]
GREEN_SIZE = 98

# Bout en bout avec un ViT-L (~1,2 Go) : étape « lente » de la CI, dans un
# processus séparé de la suite rapide.
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    root = tmp_path_factory.mktemp("inference")
    checkpoint = root / "random.pth"
    build_random_checkpoint(checkpoint)

    torch.manual_seed(0)
    head = MultiLabelHead(1024, len(CLASSES))
    head_path = root / "head.pt"
    torch.save(
        {
            "state_dict": head.state_dict(),
            "class_names": CLASSES,
            "dataset": "rfmid",
            "image_size": 224,
            "preprocess": "crop-mask",
            "preprocess_version": PREPROCESS_VERSION,
            "checkpoint_sha256": file_fingerprint(checkpoint),
        },
        head_path,
    )

    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:600, :600]
    disc = (yy - 300) ** 2 + (xx - 300) ** 2 <= 270**2
    image = np.zeros((600, 600, 3), np.uint8)
    image[disc] = rng.integers(40, 240, (disc.sum(), 3))
    image_path = root / "fundus.png"
    Image.fromarray(image).save(image_path)
    small_path = root / "small.png"
    Image.fromarray(image[::4, ::4]).save(small_path)

    thresholds_path = root / "decision.json"
    thresholds_path.write_text(
        json.dumps(
            {
                "head_probe": head_probe(head.state_dict()),
                "head_sha256": head_sha256(head.state_dict()),
                "head_contract": head_contract(torch.load(head_path, weights_only=True)),
                "classes": {
                    "DR": {"insufficient_evidence": False, "threshold_f1": {"threshold": 0.0}},
                    "MYA": {"insufficient_evidence": False, "threshold_f1": {"threshold": 1.0}},
                    "RARE": {"insufficient_evidence": True, "threshold_f1": {"threshold": 0.5}},
                },
            }
        )
    )
    predictor = Predictor(head_path, checkpoint, thresholds_path, policy="f1")
    yield dict(
        root=root,
        checkpoint=checkpoint,
        head=head,
        head_path=head_path,
        image=image_path,
        small=small_path,
        thresholds=thresholds_path,
        predictor=predictor,
    )
    del predictor  # libère le ViT-L avant les modules suivants
    gc.collect()


def test_prediction_matches_the_training_pipeline(setup):
    """Mêmes probabilités que la chaîne de run_baseline.py (collect_embeddings
    puis tête) : l'inférence ne réimplémente rien de différent."""

    class OneImage(torch.utils.data.Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return load_fundus_image(setup["image"], 224, "crop-mask"), torch.zeros(len(CLASSES))

    # Même backbone (un seul ViT-L en mémoire : le runner de CI a ~7 Go), mais
    # chemin d'entraînement : collect_embeddings normalise et extrait comme run_baseline.py.
    X, _ = collect_embeddings(setup["predictor"].backbone, OneImage())
    with torch.no_grad():
        expected = torch.sigmoid(setup["head"](torch.from_numpy(X)))[0]
    assert torch.allclose(setup["predictor"].probabilities(setup["image"]), expected, atol=1e-5)


def test_decisions_follow_thresholds_and_rare_classes_get_none(setup):
    prediction = setup["predictor"].predict(setup["image"]).to_dict()
    findings = prediction["findings"]
    assert findings["DR"]["decision"] == "positif"  # seuil 0 : toujours positif
    assert findings["MYA"]["decision"] == "négatif"  # seuil 1 : toujours négatif
    assert findings["RARE"]["decision"] is None and findings["RARE"]["threshold"] is None
    assert prediction["quality"]["usable"] and "diagnostic" in prediction["disclaimer"]


def test_inconsistent_artifacts_are_refused(setup, tmp_path):
    head_ckpt = torch.load(setup["head_path"], weights_only=True)
    for change, message in (
        ({"checkpoint_sha256": "0" * 16}, "checkpoint"),
        ({"preprocess_version": PREPROCESS_VERSION + 1}, "prétraitement"),
    ):
        bad = tmp_path / "bad.pt"
        torch.save({**head_ckpt, **change}, bad)
        with pytest.raises(ValueError, match=message):
            Predictor(bad, setup["checkpoint"])
    with pytest.raises(ValueError, match="mêmes classes"):
        load_thresholds(setup["thresholds"], ["autre", "MYA", "RARE"], "f1")


def test_thresholds_are_bound_to_the_exact_head(setup, tmp_path):
    """Seuils d'une autre tête de mêmes classes, seuil hors [0, 1], fichier
    historique sans identité : refusés (sauf accord explicite pour le dernier)."""
    report = json.loads(setup["thresholds"].read_text())
    head = torch.load(setup["head_path"], weights_only=True)
    assert load_thresholds(setup["thresholds"], CLASSES, "f1", head)[1] is True

    other = {**head, "state_dict": {k: v + 1.0 for k, v in head["state_dict"].items()}}
    with pytest.raises(ValueError, match="autre tête"):
        load_thresholds(setup["thresholds"], CLASSES, "f1", other)

    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps(
            {
                **report,
                "classes": {
                    **report["classes"],
                    "DR": {"insufficient_evidence": False, "threshold_f1": {"threshold": 1.5}},
                },
            }
        )
    )
    with pytest.raises(ValueError, match="invalide"):
        load_thresholds(bad, CLASSES, "f1", head)

    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"classes": report["classes"], "head_probe": report["head_probe"]}))
    with pytest.raises(ValueError, match="head_sha256"):
        load_thresholds(legacy, CLASSES, "f1", head)
    assert load_thresholds(legacy, CLASSES, "f1", head, allow_unverified=True)[1] is False


def test_head_with_the_same_probe_but_other_weights_is_refused(setup):
    """Modification orthogonale au vecteur de référence : même sonde, autre tête."""
    head = torch.load(setup["head_path"], weights_only=True)
    weight = head["state_dict"]["fc.weight"].clone()
    reference = torch.from_numpy(np.random.default_rng(0).standard_normal(weight.shape[1])).float()
    change = torch.randn(weight.shape[1])
    change -= (change @ reference) / (reference @ reference) * reference
    other = {**head, "state_dict": {**head["state_dict"], "fc.weight": weight + change}}
    assert probes_match(head_probe(head["state_dict"]), head_probe(other["state_dict"]))
    with pytest.raises(ValueError, match="autre tête"):
        load_thresholds(setup["thresholds"], CLASSES, "f1", other)


def test_thresholds_of_another_preprocessing_are_refused(setup):
    head = torch.load(setup["head_path"], weights_only=True)
    with pytest.raises(ValueError, match="autre chaîne"):
        load_thresholds(setup["thresholds"], CLASSES, "f1", {**head, "preprocess": "none"})


def test_rejected_images_never_reach_the_model(setup, monkeypatch):
    predictor = setup["predictor"]

    def forbidden(*args, **kwargs):
        raise AssertionError("une image refusée ne doit pas passer dans le modèle")

    monkeypatch.setattr(predictor, "probabilities", forbidden)
    prediction = predictor.predict(setup["small"], reject_unusable=True)
    assert prediction.rejected and prediction.findings == {}


def test_cli_rejects_unusable_images_on_request(setup, monkeypatch, capsys):
    import predict

    # Le prédicteur déjà chargé : la CLI n'en recharge pas un second (mémoire de la CI).
    monkeypatch.setattr(predict, "Predictor", lambda *args, **kwargs: setup["predictor"])
    out = setup["root"] / "predictions.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "predict.py",
            str(setup["image"]),
            str(setup["small"]),
            "--head",
            str(setup["head_path"]),
            "--checkpoint",
            str(setup["checkpoint"]),
            "--thresholds",
            str(setup["thresholds"]),
            "--reject-unusable",
            "--no-modality-check",
            "--device",
            "cpu",
            "--out",
            str(out),
        ],
    )
    predict.main()
    results = json.loads(out.read_text())
    assert "findings" in results[0] and results[1]["rejected"]
    assert "résolution" in " ".join(results[1]["quality"]["reasons"])


@pytest.fixture
def green_head(tmp_path):
    """Tête RETFound-Green, sur un checkpoint aléatoire et une petite taille d'image (multiple de 14)."""
    torch.manual_seed(0)
    model = timm.create_model("vit_small_patch14_reg4_dinov2", img_size=(GREEN_SIZE, GREEN_SIZE), num_classes=0)
    checkpoint = tmp_path / "green.pth"
    torch.save(model.state_dict(), checkpoint)
    head = MultiLabelHead(GREEN_EMBEDDING_DIM, len(CLASSES))
    path = tmp_path / "green_head.pt"
    torch.save(
        {
            "state_dict": head.state_dict(),
            "class_names": CLASSES,
            "backbone": "green",
            "dataset": "rfmid",
            "image_size": GREEN_SIZE,
            "preprocess": "crop-mask",
            "preprocess_version": PREPROCESS_VERSION,
            "checkpoint_sha256": file_fingerprint(checkpoint),
        },
        path,
    )
    return path, checkpoint, head


def test_green_head_selects_its_model_and_matches_the_training_pipeline(setup, green_head):
    path, checkpoint, head = green_head
    predictor = Predictor(path, checkpoint)
    assert predictor.backbone_name == "green" and isinstance(predictor.backbone, RETFoundGreenBackbone)

    class OneImage(torch.utils.data.Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return load_fundus_image(setup["image"], GREEN_SIZE, "crop-mask"), torch.zeros(len(CLASSES))

    X, _ = collect_embeddings(predictor.backbone, OneImage())
    with torch.no_grad():
        expected = torch.sigmoid(head(torch.from_numpy(X)))[0]
    assert torch.allclose(predictor.probabilities(setup["image"]), expected, atol=1e-5)


def test_head_of_one_model_refuses_the_checkpoint_of_another(setup, green_head):
    path, _, _ = green_head
    with pytest.raises(ValueError, match="checkpoint"):
        Predictor(path, setup["checkpoint"])

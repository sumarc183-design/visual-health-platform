"""Segmentation IDRiD : alignement image/masque, mesures contre les masques."""

import numpy as np
import pytest
import torch
from PIL import Image

from scripts.localization_analysis import occlusion_scores, paired_ap, pool_tokens, segmentation_scores, upsample
from src.data.fundus_preprocess import common_mask, crop_mask, crop_mask_annotation
from src.data.idrid_dataset import (
    LESION_DIRS,
    MAY_BE_ABSENT,
    IDRiDSegmentation,
    check_official_split,
    find_segmentation_root,
    read_annotation,
)
from src.explainability.localization import cell_targets, map_auroc, resize_map, top_enrichment

H, W = 300, 400


def fundus_with_spot():
    """Rétine décentrée sur fond noir, tache blanche connue : l'image et son masque."""
    array = np.zeros((H, W, 3), np.uint8)
    yy, xx = np.mgrid[:H, :W]
    array[(yy - 150) ** 2 + (xx - 230) ** 2 <= 130**2] = (120, 60, 30)
    spot = (yy - 110) ** 2 + (xx - 270) ** 2 <= 15**2
    array[spot] = (250, 250, 200)
    return Image.fromarray(array), spot


def test_annotation_gets_the_same_geometry_as_the_image():
    image, spot = fundus_with_spot()
    transformed = crop_mask(image, 112)
    mask = crop_mask_annotation(image, spot, 112)
    bright = transformed[..., 1] > 180
    assert bright.sum() > 10
    overlap = (bright & (mask > 0)).sum() / (bright | (mask > 0)).sum()
    assert overlap > 0.7


def test_annotation_refuses_another_size():
    image, _ = fundus_with_spot()
    with pytest.raises(ValueError, match="annotation"):
        crop_mask_annotation(image, np.zeros((10, 10), bool), 112)


def test_annotation_keeps_tiny_lesions_and_stays_inside_the_mask():
    image, _ = fundus_with_spot()
    tiny = np.zeros((H, W), bool)
    tiny[150, 230] = True  # un seul pixel d'origine
    assert crop_mask_annotation(image, tiny, 56).max() > 0
    everything = crop_mask_annotation(image, np.ones((H, W), bool), 56)
    assert everything[~common_mask(56)].max() == 0


@pytest.fixture
def idrid(tmp_path):
    image, spot = fundus_with_spot()
    (tmp_path / "A" / "1. Original Images" / "a. Training Set").mkdir(parents=True)
    image.save(tmp_path / "A" / "1. Original Images" / "a. Training Set" / "IDRiD_01.jpg", quality=95)
    root = tmp_path / "A" / "2. All Segmentation Groundtruths" / "a. Training Set"
    for lesion, folder in LESION_DIRS.items():
        (root / folder).mkdir(parents=True)
        if lesion not in MAY_BE_ABSENT:
            Image.fromarray(spot.astype(np.uint8)).save(root / folder / f"IDRiD_01_{lesion}.tif")
    return tmp_path


def test_dataset_finds_its_folder_and_treats_missing_masks_as_empty(idrid):
    assert find_segmentation_root(idrid).name == "A"
    dataset = IDRiDSegmentation(idrid, split="train", image_size=112)
    assert dataset.ids == ["IDRiD_01"] and dataset.image(0).shape == (3, 112, 112)
    masks = dataset.masks(0)
    assert set(masks) == {"MA", "HE", "EX", "SE", "OD"}
    assert masks["EX"].max() > 0 and masks["HE"].max() == 0 and masks["SE"].max() == 0


def test_incomplete_annotations_are_refused(idrid):
    root = idrid / "A" / "2. All Segmentation Groundtruths" / "a. Training Set"
    (root / "1. Microaneurysms" / "IDRiD_01_MA.tif").unlink()
    with pytest.raises(FileNotFoundError, match="Masque manquant"):
        IDRiDSegmentation(idrid, split="train", image_size=112).masks(0)
    (root / "1. Microaneurysms").rmdir()
    with pytest.raises(FileNotFoundError, match="incomplètes"):
        IDRiDSegmentation(idrid, split="train", image_size=112)


def test_official_split_is_checked():
    ids = [f"IDRiD_{i:02d}" for i in range(1, 82)]
    splits = ["train"] * 54 + ["test"] * 27
    check_official_split(ids, splits)
    with pytest.raises(ValueError, match="découpage officiel"):
        check_official_split(ids, ["test"] + splits[1:])
    with pytest.raises(ValueError, match="répétés"):
        check_official_split(["IDRiD_01", *ids[1:-1], "IDRiD_01"], splits)


def test_rgba_annotation_ignores_the_full_alpha_channel(tmp_path):
    array = np.zeros((4, 4, 4), np.uint8)
    array[..., 3] = 255
    array[1, 2, 0] = 255
    path = tmp_path / "rgba.tif"
    Image.fromarray(array, mode="RGBA").save(path)
    mask = read_annotation(path)
    assert mask.shape == (4, 4) and mask.sum() == 1 and mask[1, 2]


def test_map_measures():
    retina = np.ones((8, 8), bool)
    target = np.zeros((8, 8), bool)
    target[:2] = True
    perfect = target.astype(float) + np.linspace(0, 0.1, 64).reshape(8, 8)
    assert map_auroc(perfect, target, retina) == 1.0
    assert np.isnan(map_auroc(perfect, np.zeros((8, 8), bool), retina))
    assert top_enrichment(perfect, target, retina, fraction=0.1) == pytest.approx(4.0)


def test_top_enrichment_keeps_exactly_the_announced_surface():
    """Carte constante : tous les pixels à égalité, le quantile les prendrait tous."""
    retina = np.ones((10, 10), bool)
    target = np.zeros((10, 10), bool)
    target[0, 0] = True  # premier pixel : ex æquo départagé par l'ordre, sans regarder la cible
    assert top_enrichment(np.ones((10, 10)), target, retina, fraction=0.05) == pytest.approx(20.0)
    target = np.zeros((10, 10), bool)
    target[9, 9] = True
    assert top_enrichment(np.ones((10, 10)), target, retina, fraction=0.05) == 0.0


def test_resize_map_averages():
    heat = np.array([[0, 2], [4, 6]], dtype=float)
    assert resize_map(heat, 1)[0, 0] == pytest.approx(3.0)


def test_cell_targets_detects_any_pixel_and_refuses_a_bad_grid():
    mask = np.zeros((8, 8), bool)
    mask[0, 5] = True
    cells, inside = cell_targets(mask, 2, np.ones((8, 8), bool))
    assert cells.tolist() == [[False, True], [False, False]] and inside.all()
    with pytest.raises(ValueError, match="grille"):
        cell_targets(mask, 3, np.ones((8, 8), bool))


def test_reference_map_separates_position_from_the_image():
    """Toutes les lésions à la même place : la carte des autres images suffit,
    donc l'écart à cette référence est nul malgré une AUROC parfaite."""
    target = np.zeros((16, 16), bool)
    target[:4] = True
    maps = np.stack([target.astype(float) + np.random.default_rng(i).normal(0, 0.01, (16, 16)) for i in range(4)])
    masks = [{"EX": target} for _ in range(4)]
    s = occlusion_scores(maps, masks, "EX", np.ones((16, 16), bool))
    assert s["own"].mean() > 0.99 and abs((s["own"] - s["other"]).mean()) < 0.01


def test_segmentation_learns_from_tokens_and_is_evaluated_on_the_common_grid():
    rng = np.random.default_rng(0)
    grid, images = 14, 12
    targets, tokens = [], []
    for _ in range(images):
        cells = rng.random((grid, grid)) < 0.2
        target = np.kron(cells, np.ones((16, 16), bool)).astype(bool)
        targets.append(target)
        tokens.append(np.stack([cells.ravel().astype(float), rng.normal(size=grid**2)], axis=1))
    train = np.arange(images) < 8
    scores = segmentation_scores(np.stack(tokens), targets, train, np.ones((224, 224), bool))
    assert scores.shape == (images, 56, 56)
    expected = upsample(cell_targets(targets[-1], grid, np.ones((224, 224), bool))[0].astype(float))
    assert ((scores[-1] > 0.5) == (expected > 0.5)).all()


def test_pool_tokens_averages_neighbouring_cells():
    tokens = np.arange(16, dtype=float).reshape(1, 16, 1)  # grille 4×4, lignes 0-3, 4-7, ...
    pooled = pool_tokens(tokens)
    assert pooled.shape == (1, 4, 1)
    assert pooled[0, 0, 0] == pytest.approx(np.mean([0, 1, 4, 5]))
    assert pooled[0, 3, 0] == pytest.approx(np.mean([10, 11, 14, 15]))


def test_paired_ap_is_zero_for_identical_scores():
    rng = np.random.default_rng(0)
    y = rng.random((10, 20)) < 0.3
    p = rng.random((10, 20))
    result = paired_ap(y, p, p, rng)
    assert result["value"] == 0.0 and result["ci"] == [0.0, 0.0]


@pytest.mark.slow
def test_retfound_tokens_average_to_the_embedding(tmp_path):
    from src.explainability.localization import FrozenClassifier
    from src.models.retfound_backbone import build_retfound_vit_large

    torch.manual_seed(0)
    path = tmp_path / "ckpt.pth"
    torch.save({"model": build_retfound_vit_large(img_size=224).state_dict()}, path)
    model = FrozenClassifier("RETFound", path, np.ones(1024), 0.0, trust=True)
    x = torch.rand(1, 3, 224, 224)
    tokens = model.tokens(x)
    assert tokens.shape == (1, 196, 1024)
    expected = model.vit.fc_norm(tokens.mean(dim=1)).sum(dim=1)
    assert torch.allclose(model(x)[:, 0], expected, atol=1e-3)

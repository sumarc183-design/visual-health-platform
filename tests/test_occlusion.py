"""Vérifie l'occlusion sur un modèle jouet dont on connaît la zone utile."""

import numpy as np
import torch
import torch.nn as nn

from src.explainability.occlusion import background_test, fundus_mask, occlusion_map, outside_share

SIZE = 64


class RegionModel(nn.Module):
    """Logit de la classe 0 = intensité moyenne d'une région de l'image ;
    classe 1 constante."""

    def __init__(self, region: tuple[slice, slice]):
        super().__init__()
        self.region = region
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logit = x[:, :, self.region[0], self.region[1]].mean(dim=(1, 2, 3)) * self.scale
        return torch.stack([logit, torch.zeros_like(logit)], dim=1)


def fundus_image() -> torch.Tensor:
    """Disque clair (rétine) sur fond noir, comme une image de fond d'œil."""
    yy, xx = np.mgrid[:SIZE, :SIZE]
    disc = (yy - SIZE / 2) ** 2 + (xx - SIZE / 2) ** 2 < (SIZE / 2 - 4) ** 2
    return torch.from_numpy(np.where(disc, 0.9, 0.0)).float().expand(3, SIZE, SIZE).clone()


def test_fundus_mask_separates_retina_from_margins():
    image = fundus_image()
    mask = fundus_mask(image)
    assert mask[SIZE // 2, SIZE // 2] and not mask[0, 0]

    image[:, :, :3] = 1.0  # bande blanche de remplissage
    assert not fundus_mask(image)[SIZE // 2, 0]


def test_occlusion_peaks_on_the_region_the_model_uses():
    center = (slice(24, 40), slice(24, 40))
    heat, base = occlusion_map(RegionModel(center), fundus_image(), class_idx=0, patch=16, stride=8)
    assert heat.shape == (SIZE, SIZE)
    assert base > 0
    assert heat[32, 32] > 0
    assert abs(heat[2, 2]) < 1e-6  # coin jamais lu par le modèle


def test_background_test_flags_a_model_that_reads_the_margins():
    image = fundus_image()
    reads_center = background_test(RegionModel((slice(24, 40), slice(24, 40))), image, class_idx=0)
    reads_corner = background_test(RegionModel((slice(0, 6), slice(0, 6))), image, class_idx=0)
    assert abs(reads_center["logit_fond_gris"] - reads_center["logit"]) < 1e-6
    assert abs(reads_corner["logit_fond_gris"] - reads_corner["logit"]) > 0.1
    assert 0 < reads_center["part_hors_champ"] < 0.5


def test_outside_share():
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True
    heat = np.zeros((4, 4))
    heat[0, 0], heat[1, 1] = 3.0, 1.0
    assert outside_share(heat, mask) == 0.75
    assert np.isnan(outside_share(np.zeros((4, 4)), mask))

"""Known image-space deformations for label-free correspondence training.

The deformation is defined in target-to-source sampling coordinates.  This
keeps the image operation and the correspondence target generated from the
same deterministic field, so no benchmark keypoint enters training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


@dataclass(frozen=True)
class AppearancePlan:
    brightness: float
    contrast: float
    color: float
    blur_radius: float
    invert_probability: float

    def apply(self, image: Image.Image, rng: random.Random) -> Image.Image:
        result = image.convert("RGB")
        result = ImageEnhance.Brightness(result).enhance(self.brightness)
        result = ImageEnhance.Contrast(result).enhance(self.contrast)
        result = ImageEnhance.Color(result).enhance(self.color)
        if self.blur_radius > 0.0:
            result = result.filter(ImageFilter.GaussianBlur(self.blur_radius))
        if rng.random() < self.invert_probability:
            result = ImageOps.invert(result)
        return result


@dataclass(frozen=True)
class ElasticDeformationPlan:
    amplitude_x: float
    amplitude_y: float
    cycles_x: float
    cycles_y: float
    phase_x: float
    phase_y: float

    @staticmethod
    def sample(rng: random.Random) -> "ElasticDeformationPlan":
        return ElasticDeformationPlan(
            amplitude_x=rng.uniform(0.025, 0.075),
            amplitude_y=rng.uniform(0.025, 0.075),
            cycles_x=rng.uniform(0.8, 1.8),
            cycles_y=rng.uniform(0.8, 1.8),
            phase_x=rng.uniform(0.0, 2.0 * math.pi),
            phase_y=rng.uniform(0.0, 2.0 * math.pi),
        )

    def _displacement(self, points: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Return target->source displacement for [N,2] xy pixel points."""

        xy = points.float()
        x = xy[:, 0] / float(max(1, width - 1))
        y = xy[:, 1] / float(max(1, height - 1))
        envelope = torch.sin(math.pi * x).clamp_min(0.0) * torch.sin(math.pi * y).clamp_min(0.0)
        dx = float(self.amplitude_x * max(1, width)) * envelope * torch.sin(
            2.0 * math.pi * float(self.cycles_y) * y + float(self.phase_x)
        )
        dy = float(self.amplitude_y * max(1, height)) * envelope * torch.sin(
            2.0 * math.pi * float(self.cycles_x) * x + float(self.phase_y)
        )
        return torch.stack((dx, dy), dim=1)

    def apply(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        width, height = image.size
        source = torch.from_numpy(np.array(image)).float()
        source = source.permute(2, 0, 1).unsqueeze(0) / 255.0
        yy, xx = torch.meshgrid(
            torch.arange(height, dtype=torch.float32),
            torch.arange(width, dtype=torch.float32),
            indexing="ij",
        )
        target_points = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)
        source_points = target_points + self._displacement(target_points, height, width)
        grid = torch.stack(
            (
                2.0 * source_points[:, 0] / float(max(1, width - 1)) - 1.0,
                2.0 * source_points[:, 1] / float(max(1, height - 1)) - 1.0,
            ),
            dim=1,
        ).reshape(1, height, width, 2)
        warped = F.grid_sample(
            source,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )[0].permute(1, 2, 0).clamp(0.0, 1.0)
        return Image.fromarray((warped * 255.0).round().byte().numpy(), mode="RGB")

    def source_to_target(self, points: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Invert x = y + d(y) with fixed-point iterations."""

        source = points.float()
        target = source.clone()
        for _ in range(8):
            target = source - self._displacement(target, height, width)
        maximum = torch.tensor(
                [float(max(0, width - 1)), float(max(0, height - 1))],
                dtype=target.dtype,
            )
        return target.clamp_min(0.0).minimum(maximum)

    def target_to_source(self, points: torch.Tensor, height: int, width: int) -> torch.Tensor:
        maximum = torch.tensor(
                [float(max(0, width - 1)), float(max(0, height - 1))],
                dtype=points.dtype,
            )
        return (points.float() + self._displacement(points.float(), height, width)).clamp_min(0.0).minimum(maximum)


def sample_appearance_plan(rng: random.Random) -> AppearancePlan:
    return AppearancePlan(
        brightness=rng.uniform(0.65, 1.35),
        contrast=rng.uniform(0.55, 1.45),
        color=rng.uniform(0.0, 1.8),
        blur_radius=rng.uniform(0.0, 1.1),
        invert_probability=0.03,
    )


def make_view_transform(
    appearance: AppearancePlan,
    rng: random.Random,
    deformation: ElasticDeformationPlan | None = None,
):
    def transform(image: Image.Image) -> Image.Image:
        result = appearance.apply(image, rng)
        if deformation is not None:
            result = deformation.apply(result)
        return result

    return transform

"""Canonical DINOv2 ViT-B/14 geometry helpers for SPair-71k.

The protocol is copied from the existing DINOv2 parity evaluator: an
aspect-preserving 840 square canvas, block-11 pre-final-norm tokens, and a
60-by-60 patch grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from PIL import Image


@dataclass(frozen=True)
class DINOConfig:
    model_name: str = "facebook/dinov2-base"
    hub_model: str = "dinov2_vitb14"
    layer: int = 11
    patch_size: int = 14
    image_size: int = 840

    @property
    def grid_size(self) -> int:
        if self.image_size % self.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        return self.image_size // self.patch_size


def square_canvas_geometry(
    height: int, width: int, target_res: int
) -> tuple[float, int, int, int, int]:
    """Return scale, offsets, and content size for the parity canvas."""
    if height <= 0 or width <= 0 or target_res <= 0:
        raise ValueError("image and target dimensions must be positive")
    scale = float(target_res) / max(height, width)
    resized_h = (
        max(1, int(np.around(target_res * height / width)))
        if height <= width
        else target_res
    )
    resized_w = (
        target_res
        if height <= width
        else max(1, int(np.around(target_res * width / height)))
    )
    offset_y = (target_res - resized_h) // 2
    offset_x = (target_res - resized_w) // 2
    return scale, offset_x, offset_y, resized_h, resized_w


def preprocess_square_canvas(image: Image.Image, target_res: int) -> torch.Tensor:
    """Create the normalized DINOv2 parity input tensor."""
    image = image.convert("RGB")
    _, offset_x, offset_y, resized_h, resized_w = square_canvas_geometry(
        image.height, image.width, target_res
    )
    image = image.resize((resized_w, resized_h), Image.Resampling.LANCZOS)
    canvas = np.zeros((target_res, target_res, 3), dtype=np.uint8)
    canvas[offset_y : offset_y + resized_h, offset_x : offset_x + resized_w] = np.asarray(image)
    tensor = torch.from_numpy(canvas.copy()).permute(2, 0, 1).float().div_(255.0)
    mean = tensor.new_tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
    std = tensor.new_tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
    return tensor.sub_(mean).div_(std)


def transform_points_to_canvas(
    points: Sequence[Sequence[float]], height: int, width: int, target_res: int
) -> torch.Tensor:
    """Map original-image coordinates onto the DINOv2 square canvas."""
    scale, offset_x, offset_y, _, _ = square_canvas_geometry(height, width, target_res)
    output = torch.as_tensor(points, dtype=torch.float32).clone()
    if output.ndim != 2 or output.shape[1] != 2:
        raise ValueError(f"Expected Nx2 points, got {tuple(output.shape)}")
    output[:, 0].mul_(scale).add_(offset_x)
    output[:, 1].mul_(scale).add_(offset_y)
    return output


def tokens_to_patch_map(tokens: torch.Tensor, grid_size: int) -> torch.Tensor:
    """Convert transformer tokens to B,C,H,W, dropping special tokens."""
    if tokens.ndim != 3:
        raise ValueError(f"Expected BxNxC tokens, got {tuple(tokens.shape)}")
    patch_count = grid_size * grid_size
    if tokens.shape[1] < patch_count:
        raise ValueError(f"Expected at least {patch_count} patch tokens, got {tokens.shape[1]}")
    patches = tokens[:, -patch_count:, :]
    return patches.transpose(1, 2).reshape(
        tokens.shape[0], tokens.shape[2], grid_size, grid_size
    )


def points_to_patch_indices(
    points: torch.Tensor, image_size: int, grid_size: int
) -> torch.Tensor:
    """Map canvas coordinates to flattened patch indices by floor."""
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected Nx2 points, got {tuple(points.shape)}")
    xy = torch.floor(points * (grid_size / image_size)).long().clamp_(0, grid_size - 1)
    return xy[:, 1] * grid_size + xy[:, 0]

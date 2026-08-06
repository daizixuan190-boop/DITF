"""Exact cosine featureization of a continuous forward warp."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


def _endpoint_grid(
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    ys, xs = torch.meshgrid(
        torch.linspace(0.0, 1.0, int(height), device=device, dtype=dtype),
        torch.linspace(0.0, 1.0, int(width), device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack((xs, ys), dim=-1)


def _resize_field(
    field: torch.Tensor,
    size: Sequence[int],
) -> torch.Tensor:
    channels = 1 if field.ndim == 2 else int(field.shape[-1])
    if field.ndim == 2:
        field = field.unsqueeze(-1)
    resized = F.interpolate(
        field.permute(2, 0, 1).unsqueeze(0),
        size=(int(size[0]), int(size[1])),
        mode="bilinear",
        align_corners=True,
    )
    return resized[0].permute(1, 2, 0).contiguous().reshape(
        int(size[0]), int(size[1]), channels
    )


def _asymmetric_squared_distance_embedding(
    coordinates: torch.Tensor,
    *,
    source_side: bool,
) -> torch.Tensor:
    """Return unit features whose cross-dot is ``-||source-target||^2 / 9``.

    Source and target slack occupy orthogonal channels. Every vector has norm
    one over the full normalized image square, so cosine ranking is exactly
    nearest-coordinate ranking without a learned bandwidth or Fourier alias.
    """

    if coordinates.ndim != 3 or int(coordinates.shape[-1]) != 2:
        raise ValueError("coordinates must have shape [height,width,2]")
    coordinates = coordinates.float().clamp(0.0, 1.0)
    squared_norm = coordinates.square().sum(dim=-1, keepdim=True)
    base_norm_squared = (1.0 + squared_norm).square()
    slack = torch.sqrt((9.0 - base_norm_squared).clamp_min(0.0))
    ones = torch.ones_like(squared_norm)
    zeros = torch.zeros_like(squared_norm)
    xy = float(2.0**0.5) * coordinates
    if source_side:
        values = (xy, -squared_norm, ones, slack, zeros)
    else:
        values = (xy, ones, -squared_norm, zeros, slack)
    return (torch.cat(values, dim=-1) / 3.0).permute(2, 0, 1).unsqueeze(0)


def build_exact_forward_coordinate_maps(
    warp_ab: torch.Tensor,
    reliability_a: torch.Tensor,
    reliability_b: torch.Tensor,
    *,
    source_size: Sequence[int],
    target_size: Sequence[int],
) -> dict[str, torch.Tensor]:
    """Build full-resolution exact direct-warp identity maps and gates."""

    if warp_ab.ndim != 3 or int(warp_ab.shape[-1]) != 2:
        raise ValueError("warp_ab must have shape [height,width,2]")
    if tuple(reliability_a.shape) != tuple(warp_ab.shape[:2]):
        raise ValueError("source reliability must align with warp_ab")
    if reliability_b.ndim != 2:
        raise ValueError("target reliability must be two-dimensional")
    forward = _resize_field(warp_ab.float(), source_size).clamp(0.0, 1.0)
    source_reliability = _resize_field(
        reliability_a.float(), source_size
    ).permute(2, 0, 1).unsqueeze(0).clamp(0.0, 1.0)
    target_reliability = _resize_field(
        reliability_b.float(), target_size
    ).permute(2, 0, 1).unsqueeze(0).clamp(0.0, 1.0)
    target_coordinates = _endpoint_grid(
        int(target_size[0]),
        int(target_size[1]),
        device=warp_ab.device,
        dtype=warp_ab.dtype,
    )
    source_unit = _asymmetric_squared_distance_embedding(
        forward, source_side=True
    )
    target_unit = _asymmetric_squared_distance_embedding(
        target_coordinates, source_side=False
    )
    return {
        "source_unit": source_unit,
        "target_unit": target_unit,
        "source_gated": source_unit * source_reliability,
        "target_gated": target_unit * target_reliability,
        "source_reliability": source_reliability,
        "target_reliability": target_reliability,
        "forward_full": forward,
    }

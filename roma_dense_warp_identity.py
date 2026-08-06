"""Convert a frozen symmetric RoMa output into token-grid identity fields."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _as_unbatched(
    warp: torch.Tensor,
    certainty: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if warp.ndim == 4:
        if int(warp.shape[0]) != 1:
            raise ValueError("RoMa dense identity expects one image pair")
        warp = warp[0]
    if certainty.ndim == 3:
        if int(certainty.shape[0]) != 1:
            raise ValueError("RoMa dense identity expects one certainty map")
        certainty = certainty[0]
    if warp.ndim != 3 or int(warp.shape[-1]) != 4:
        raise ValueError("RoMa warp must have shape [height,2*width,4]")
    if certainty.ndim != 2 or tuple(certainty.shape) != tuple(warp.shape[:2]):
        raise ValueError("RoMa certainty must align with the warp grid")
    if int(warp.shape[1]) % 2:
        raise ValueError("symmetric RoMa warp must contain equal directional halves")
    return warp.float(), certainty.float()


def _roma_query_grid(
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if int(height) <= 0 or int(width) <= 0:
        raise ValueError("token grid dimensions must be positive")
    ys = torch.linspace(
        -1.0 + 1.0 / float(height),
        1.0 - 1.0 / float(height),
        int(height),
        device=device,
        dtype=dtype,
    )
    xs = torch.linspace(
        -1.0 + 1.0 / float(width),
        1.0 - 1.0 / float(width),
        int(width),
        device=device,
        dtype=dtype,
    )
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((xx, yy), dim=-1)


def _sample_roma_field(field: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    if field.ndim == 2:
        field = field.unsqueeze(-1)
    if field.ndim != 3:
        raise ValueError("RoMa field must have shape [height,width,channels]")
    sampled = F.grid_sample(
        field.permute(2, 0, 1).unsqueeze(0),
        query.unsqueeze(0),
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    return sampled[0].permute(1, 2, 0).contiguous()


def _roma_coordinate_to_endpoint(
    coordinate: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Map RoMa/grid-sample center coordinates to endpoint token coordinates."""

    result = torch.empty_like(coordinate)
    if int(width) > 1:
        x_index = 0.5 * ((coordinate[..., 0] + 1.0) * float(width) - 1.0)
        result[..., 0] = x_index / float(width - 1)
    else:
        result[..., 0] = 0.0
    if int(height) > 1:
        y_index = 0.5 * ((coordinate[..., 1] + 1.0) * float(height) - 1.0)
        result[..., 1] = y_index / float(height - 1)
    else:
        result[..., 1] = 0.0
    return result.clamp(0.0, 1.0)


def normalize_roma_certainty(certainty: torch.Tensor) -> torch.Tensor:
    """Scale RoMa certainty per pair without a learned or labelled threshold.

    Cross-instance RoMa certainty is often numerically tiny even when its warp
    is useful. Dividing by ``value + positive_median`` preserves ordering and
    is invariant to an arbitrary global certainty scale.
    """

    value = torch.nan_to_num(
        certainty.float(), nan=0.0, posinf=0.0, neginf=0.0
    ).clamp_min(0.0)
    positive = value[value > 0]
    if int(positive.numel()) == 0:
        return torch.zeros_like(value)
    scale = torch.median(positive).clamp_min(1e-12)
    return (value / (value + scale)).clamp(0.0, 1.0)


def roma_warp_to_token_fields(
    warp: torch.Tensor,
    certainty: torch.Tensor,
    *,
    source_grid: tuple[int, int],
    target_grid: tuple[int, int],
) -> dict[str, torch.Tensor]:
    """Sample symmetric RoMa fields at native source/target token centers."""

    warp, certainty = _as_unbatched(warp, certainty)
    half_width = int(warp.shape[1]) // 2
    forward_field = warp[:, :half_width, 2:]
    backward_field = warp[:, half_width:, :2]
    forward_certainty = certainty[:, :half_width]
    backward_certainty = certainty[:, half_width:]
    source_height, source_width = map(int, source_grid)
    target_height, target_width = map(int, target_grid)
    source_query = _roma_query_grid(
        source_height,
        source_width,
        device=warp.device,
        dtype=warp.dtype,
    )
    target_query = _roma_query_grid(
        target_height,
        target_width,
        device=warp.device,
        dtype=warp.dtype,
    )
    forward_sample = _sample_roma_field(forward_field, source_query)
    backward_sample = _sample_roma_field(backward_field, target_query)
    certainty_a_raw = _sample_roma_field(forward_certainty, source_query)[..., 0]
    certainty_b_raw = _sample_roma_field(backward_certainty, target_query)[..., 0]
    return {
        "warp_ab": _roma_coordinate_to_endpoint(
            forward_sample, target_height, target_width
        ),
        "warp_ba": _roma_coordinate_to_endpoint(
            backward_sample, source_height, source_width
        ),
        "support_a": normalize_roma_certainty(certainty_a_raw),
        "support_b": normalize_roma_certainty(certainty_b_raw),
        "certainty_a_raw": certainty_a_raw,
        "certainty_b_raw": certainty_b_raw,
    }

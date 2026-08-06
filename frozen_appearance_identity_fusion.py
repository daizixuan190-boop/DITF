"""Frozen appearance and RoMa feature-side correspondence fusion.

The constructions are deliberately annotation-free and contain no candidate
router. FLUX is composed with either DINOv2 or a paired spectral descriptor;
the existing reliability-weighted RoMa identity field is then concatenated
before the unchanged global cosine nearest-neighbour matcher.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

from dino_v2_spair import square_canvas_geometry
from flux_dense_warp_identity import (
    DenseWarpIdentityConfig,
    build_identity_maps_from_warp_fields,
)


def sample_dino_map_on_image_grid(
    feature: torch.Tensor,
    *,
    image_size: Sequence[int],
    output_grid: Sequence[int],
    canvas_size: int,
) -> torch.Tensor:
    """Sample a square-canvas DINO map on original-image token centres.

    ``image_size`` and ``output_grid`` use ``(height, width)`` order.  Token
    centres are chosen as the inverse of PyTorch's later
    ``align_corners=False`` image upsampling, so the returned map shares the
    same original-image coordinate frame as the native FLUX descriptor.
    """

    if feature.ndim == 3:
        feature = feature.unsqueeze(0)
    if feature.ndim != 4 or int(feature.shape[0]) != 1:
        raise ValueError("DINO feature must have shape [C,H,W] or [1,C,H,W]")
    if len(image_size) != 2 or len(output_grid) != 2:
        raise ValueError("image_size and output_grid must contain height and width")
    image_height, image_width = map(int, image_size)
    output_height, output_width = map(int, output_grid)
    if min(image_height, image_width, output_height, output_width, int(canvas_size)) <= 0:
        raise ValueError("image, output-grid, and canvas dimensions must be positive")

    scale, offset_x, offset_y, _resized_h, _resized_w = square_canvas_geometry(
        image_height, image_width, int(canvas_size)
    )
    dtype = torch.float32
    device = feature.device
    source_x = (
        (torch.arange(output_width, device=device, dtype=dtype) + 0.5)
        * (float(image_width) / output_width)
        - 0.5
    )
    source_y = (
        (torch.arange(output_height, device=device, dtype=dtype) + 0.5)
        * (float(image_height) / output_height)
        - 0.5
    )
    canvas_x = source_x * float(scale) + float(offset_x)
    canvas_y = source_y * float(scale) + float(offset_y)
    normalized_x = 2.0 * (canvas_x + 0.5) / float(canvas_size) - 1.0
    normalized_y = 2.0 * (canvas_y + 0.5) / float(canvas_size) - 1.0
    yy, xx = torch.meshgrid(normalized_y, normalized_x, indexing="ij")
    grid = torch.stack((xx, yy), dim=-1).unsqueeze(0)
    return F.grid_sample(
        feature.float(),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )


def build_equal_energy_appearance_base(
    flux_feature: torch.Tensor,
    dino_feature: torch.Tensor,
) -> torch.Tensor:
    """Concatenate unit FLUX/DINO descriptors with fixed equal energy."""

    if flux_feature.ndim != 4 or dino_feature.ndim != 4:
        raise ValueError("FLUX and DINO descriptors must have shape [B,C,H,W]")
    if int(flux_feature.shape[0]) != int(dino_feature.shape[0]):
        raise ValueError("FLUX and DINO descriptor batches must agree")
    if tuple(flux_feature.shape[-2:]) != tuple(dino_feature.shape[-2:]):
        raise ValueError("FLUX and DINO descriptor spatial grids must agree")
    flux_unit = F.normalize(flux_feature.float(), dim=1, eps=1e-12)
    dino_unit = F.normalize(dino_feature.float(), dim=1, eps=1e-12)
    return torch.cat((flux_unit, dino_unit), dim=1) * float(2.0**-0.5)


def build_weighted_appearance_base(
    native_feature: torch.Tensor,
    auxiliary_feature: torch.Tensor,
    *,
    auxiliary_weight: float,
) -> torch.Tensor:
    """Fuse unit branches with one fixed auxiliary amplitude.

    The function is used for the locked attention-kernel construction.  The
    weight is an explicit method constant, not inferred from annotations or
    evaluation outcomes.
    """

    if native_feature.ndim != 4 or auxiliary_feature.ndim != 4:
        raise ValueError("native and auxiliary descriptors must have shape [B,C,H,W]")
    if int(native_feature.shape[0]) != int(auxiliary_feature.shape[0]):
        raise ValueError("native and auxiliary descriptor batches must agree")
    if tuple(native_feature.shape[-2:]) != tuple(auxiliary_feature.shape[-2:]):
        raise ValueError("native and auxiliary descriptor spatial grids must agree")
    if float(auxiliary_weight) < 0.0:
        raise ValueError("auxiliary_weight must be non-negative")
    native_unit = F.normalize(native_feature.float(), dim=1, eps=1e-12)
    auxiliary_unit = F.normalize(auxiliary_feature.float(), dim=1, eps=1e-12)
    return F.normalize(
        torch.cat(
            (native_unit, float(auxiliary_weight) * auxiliary_unit),
            dim=1,
        ),
        dim=1,
        eps=1e-12,
    )


def build_weighted_roma_augmented_appearance(
    source_native: torch.Tensor,
    target_native: torch.Tensor,
    source_auxiliary: torch.Tensor,
    target_auxiliary: torch.Tensor,
    warp_ab: torch.Tensor,
    warp_ba: torch.Tensor,
    *,
    auxiliary_weight: float,
    source_support: torch.Tensor | None = None,
    target_support: torch.Tensor | None = None,
    config: DenseWarpIdentityConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Compose a fixed weighted appearance field with frozen RoMa identity."""

    source_appearance = build_weighted_appearance_base(
        source_native,
        source_auxiliary,
        auxiliary_weight=auxiliary_weight,
    )
    target_appearance = build_weighted_appearance_base(
        target_native,
        target_auxiliary,
        auxiliary_weight=auxiliary_weight,
    )
    identity = build_identity_maps_from_warp_fields(
        source_appearance,
        target_appearance,
        warp_ab,
        warp_ba,
        source_support=source_support,
        target_support=target_support,
        config=config,
    )
    result = dict(identity)
    result.update(
        {
            "source_appearance": source_appearance,
            "target_appearance": target_appearance,
        }
    )
    return result


def build_roma_augmented_appearance(
    source_flux: torch.Tensor,
    target_flux: torch.Tensor,
    source_dino: torch.Tensor,
    target_dino: torch.Tensor,
    warp_ab: torch.Tensor,
    warp_ba: torch.Tensor,
    *,
    source_support: torch.Tensor | None = None,
    target_support: torch.Tensor | None = None,
    config: DenseWarpIdentityConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Build the fixed three-factor descriptor used by the main experiment."""

    source_appearance = build_equal_energy_appearance_base(source_flux, source_dino)
    target_appearance = build_equal_energy_appearance_base(target_flux, target_dino)
    identity = build_identity_maps_from_warp_fields(
        source_appearance,
        target_appearance,
        warp_ab,
        warp_ba,
        source_support=source_support,
        target_support=target_support,
        config=config,
    )
    result = dict(identity)
    result.update(
        {
            "source_appearance": source_appearance,
            "target_appearance": target_appearance,
            # The variable-norm form is retained deliberately: it is the only
            # RoMa fusion that produced a repeatable positive discovery20 gain.
            "source_fused": identity["source_fused_variable_norm"],
            "target_fused": identity["target_fused_variable_norm"],
        }
    )
    return result

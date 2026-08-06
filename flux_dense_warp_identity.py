"""Train-free identity features from a full bidirectional attention posterior.

The module deliberately operates after the existing exact FLUX replay.  It
does not recompute Q/K/V or alter a Transformer block.  Instead, it converts
the complete source/target posterior into two continuous local-mode warp
fields, validates those fields by bidirectional cycle and local-Jacobian
stability, and encodes the resulting shared coordinates as a small identity
branch next to the untouched native DiTF descriptor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DenseWarpIdentityConfig:
    """Fixed, annotation-free construction parameters."""

    mode_sigma_cells: float = 0.75
    mode_iterations: int = 3
    basin_radius_cells: float = 1.5
    fourier_frequencies: tuple[int, ...] = (1, 2, 4)
    identity_scale: float = 1.0
    basin_weight: float = 0.5
    cycle_weight: float = 1.0
    jacobian_weight: float = 0.5

    def validate(self) -> None:
        if self.mode_sigma_cells <= 0:
            raise ValueError("mode_sigma_cells must be positive")
        if self.mode_iterations < 1:
            raise ValueError("mode_iterations must be positive")
        if self.basin_radius_cells <= 0:
            raise ValueError("basin_radius_cells must be positive")
        if not self.fourier_frequencies or any(
            int(value) <= 0 for value in self.fourier_frequencies
        ):
            raise ValueError("fourier_frequencies must contain positive integers")
        if self.identity_scale < 0:
            raise ValueError("identity_scale must be non-negative")
        if min(self.basin_weight, self.cycle_weight, self.jacobian_weight) < 0:
            raise ValueError("reliability weights must be non-negative")


def _normalized_grid(
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if int(height) <= 0 or int(width) <= 0:
        raise ValueError("spatial grids must be positive")
    ys, xs = torch.meshgrid(
        torch.linspace(0.0, 1.0, int(height), device=device, dtype=dtype),
        torch.linspace(0.0, 1.0, int(width), device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack((xs, ys), dim=-1)


def _cell_grid(
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    ys, xs = torch.meshgrid(
        torch.arange(int(height), device=device, dtype=dtype),
        torch.arange(int(width), device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack((xs, ys), dim=-1).reshape(-1, 2)


def _row_normalize(probability: torch.Tensor) -> torch.Tensor:
    probability = torch.nan_to_num(
        probability.float(), nan=0.0, posinf=0.0, neginf=0.0
    ).clamp_min(0.0)
    mass = probability.sum(dim=1, keepdim=True)
    return torch.where(
        mass > 1e-20,
        probability / mass.clamp_min(1e-20),
        torch.zeros_like(probability),
    )


def mutual_attention_posteriors(
    p_ab: torch.Tensor,
    p_ba: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return row-conditional reciprocal posteriors in both directions."""

    if p_ab.ndim != 2 or p_ba.ndim != 2:
        raise ValueError("attention posterior must be two-dimensional")
    if tuple(p_ba.shape) != (int(p_ab.shape[1]), int(p_ab.shape[0])):
        raise ValueError(
            "attention shape mismatch: expected p_ba to transpose p_ab, "
            f"got {tuple(p_ab.shape)} and {tuple(p_ba.shape)}"
        )
    forward = torch.nan_to_num(p_ab.float(), nan=0.0, posinf=0.0, neginf=0.0)
    reverse = torch.nan_to_num(p_ba.float(), nan=0.0, posinf=0.0, neginf=0.0)
    mutual = torch.sqrt((forward.clamp_min(0.0) * reverse.t().clamp_min(0.0)).clamp_min(0.0))
    return _row_normalize(mutual), _row_normalize(mutual.t())


def local_mode_coordinates(
    probability: torch.Tensor,
    *,
    target_height: int,
    target_width: int,
    sigma_cells: float,
    iterations: int,
    basin_radius_cells: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Follow the strongest local posterior mode without averaging remote modes.

    Coordinates are returned in normalized ``[0,1]`` xy convention.  The
    associated basin mass is the posterior probability inside a fixed spatial
    radius around the converged mode.
    """

    if probability.ndim != 2:
        raise ValueError("probability must have shape [query,candidate]")
    candidate_count = int(target_height) * int(target_width)
    if int(probability.shape[1]) != candidate_count:
        raise ValueError(
            "probability candidate count does not match target grid: "
            f"{int(probability.shape[1])} != {candidate_count}"
        )
    if sigma_cells <= 0 or iterations < 1 or basin_radius_cells <= 0:
        raise ValueError("mode parameters must be positive")

    posterior = _row_normalize(probability)
    cells = _cell_grid(
        target_height,
        target_width,
        device=posterior.device,
        dtype=posterior.dtype,
    )
    peak = torch.argmax(posterior, dim=1)
    center = cells[peak]
    cutoff2 = float(3.0 * sigma_cells) ** 2
    sigma2 = float(sigma_cells) ** 2

    for _ in range(int(iterations)):
        distance2 = (cells.unsqueeze(0) - center.unsqueeze(1)).square().sum(dim=-1)
        kernel = torch.exp(-0.5 * distance2 / sigma2)
        kernel = kernel * (distance2 <= cutoff2).to(kernel.dtype)
        weights = posterior * kernel
        mass = weights.sum(dim=1, keepdim=True)
        updated = torch.matmul(weights, cells) / mass.clamp_min(1e-20)
        center = torch.where(mass > 1e-20, updated, center)

    final_distance2 = (cells.unsqueeze(0) - center.unsqueeze(1)).square().sum(dim=-1)
    basin_mass = (
        posterior
        * (final_distance2 <= float(basin_radius_cells) ** 2).to(posterior.dtype)
    ).sum(dim=1)
    scale = center.new_tensor(
        [float(max(int(target_width) - 1, 1)), float(max(int(target_height) - 1, 1))]
    )
    normalized = center / scale
    if int(target_width) == 1:
        normalized[:, 0] = 0.0
    if int(target_height) == 1:
        normalized[:, 1] = 0.0
    return normalized.clamp(0.0, 1.0), basin_mass.clamp(0.0, 1.0)


def _sample_field(field: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample an HxWxC field at normalized ``[0,1]`` xy."""

    if field.ndim != 3 or field.shape[-1] < 1:
        raise ValueError("field must have shape [height,width,channels]")
    original_shape = coordinates.shape[:-1]
    grid = coordinates.reshape(1, -1, 1, 2).to(field.device, field.dtype)
    grid = grid * 2.0 - 1.0
    sampled = F.grid_sample(
        field.permute(2, 0, 1).unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled[0, :, :, 0].t().reshape(*original_shape, int(field.shape[-1]))


def _cycle_error_cells(
    first_grid: torch.Tensor,
    forward_warp: torch.Tensor,
    reverse_warp: torch.Tensor,
    first_height: int,
    first_width: int,
) -> torch.Tensor:
    reconstructed = _sample_field(reverse_warp, forward_warp)
    scale = first_grid.new_tensor(
        [float(max(int(first_width) - 1, 1)), float(max(int(first_height) - 1, 1))]
    )
    return ((reconstructed - first_grid) * scale).square().sum(dim=-1).sqrt()


def _finite_difference_jacobian(field_cells: torch.Tensor) -> torch.Tensor:
    """Return one-sided boundary / central interior spatial derivatives."""

    if field_cells.ndim != 3 or int(field_cells.shape[-1]) != 2:
        raise ValueError("warp field must have shape [height,width,2]")
    height, width = map(int, field_cells.shape[:2])
    dx = torch.zeros_like(field_cells)
    dy = torch.zeros_like(field_cells)
    if width > 1:
        dx[:, 1:-1] = 0.5 * (field_cells[:, 2:] - field_cells[:, :-2])
        dx[:, 0] = field_cells[:, 1] - field_cells[:, 0]
        dx[:, -1] = field_cells[:, -1] - field_cells[:, -2]
    if height > 1:
        dy[1:-1] = 0.5 * (field_cells[2:] - field_cells[:-2])
        dy[0] = field_cells[1] - field_cells[0]
        dy[-1] = field_cells[-1] - field_cells[-2]
    return torch.cat((dx, dy), dim=-1)


def _jacobian_consistency(
    warp: torch.Tensor,
    target_height: int,
    target_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale = warp.new_tensor(
        [float(max(int(target_width) - 1, 1)), float(max(int(target_height) - 1, 1))]
    )
    jacobian = _finite_difference_jacobian(warp * scale)
    channels = jacobian.permute(2, 0, 1).unsqueeze(0)
    local_mean = F.avg_pool2d(
        F.pad(channels, (1, 1, 1, 1), mode="replicate"),
        kernel_size=3,
        stride=1,
    )[0].permute(1, 2, 0)
    # A central finite-difference Jacobian can miss an isolated impulse at the
    # impulse itself because it reads the two neighbors.  Include the local
    # Jacobian variation (second spatial difference) explicitly.  It is zero
    # for every affine field, including scale, shear, and reflection, while a
    # discontinuous correspondence receives low confidence at its origin.
    curvature_x = torch.zeros_like(warp)
    curvature_y = torch.zeros_like(warp)
    field_cells = warp * scale
    if int(warp.shape[1]) > 2:
        curvature_x[:, 1:-1] = (
            field_cells[:, 2:] - 2.0 * field_cells[:, 1:-1] + field_cells[:, :-2]
        )
    if int(warp.shape[0]) > 2:
        curvature_y[1:-1] = (
            field_cells[2:] - 2.0 * field_cells[1:-1] + field_cells[:-2]
        )
    residual = torch.cat(
        (jacobian - local_mean, curvature_x, curvature_y), dim=-1
    ).square().sum(dim=-1).sqrt()
    return 1.0 / (1.0 + residual), residual


def _coordinate_encoding(
    first: torch.Tensor,
    second: torch.Tensor,
    frequencies: Sequence[int],
) -> torch.Tensor:
    values = [2.0 * first - 1.0, 2.0 * second - 1.0]
    for coordinates in (first, second):
        for frequency in frequencies:
            phase = coordinates * (torch.pi * float(frequency))
            values.extend((torch.sin(phase), torch.cos(phase)))
    encoded = torch.cat(values, dim=-1)
    encoded = F.normalize(encoded, dim=-1, eps=1e-12)
    return encoded.permute(2, 0, 1).unsqueeze(0).contiguous()


def _single_coordinate_encoding(
    coordinates: torch.Tensor,
    frequencies: Sequence[int],
) -> torch.Tensor:
    """Encode one normalized xy field as a unit-norm spatial descriptor."""

    values = [2.0 * coordinates - 1.0]
    for frequency in frequencies:
        phase = coordinates * (torch.pi * float(frequency))
        values.extend((torch.sin(phase), torch.cos(phase)))
    encoded = torch.cat(values, dim=-1)
    encoded = F.normalize(encoded, dim=-1, eps=1e-12)
    return encoded.permute(2, 0, 1).unsqueeze(0).contiguous()


def _weighted_reliability(
    basin_mass: torch.Tensor,
    cycle_error: torch.Tensor,
    jacobian_confidence: torch.Tensor,
    config: DenseWarpIdentityConfig,
) -> torch.Tensor:
    result = torch.ones_like(basin_mass)
    if config.basin_weight:
        result = result * basin_mass.clamp(0.0, 1.0).pow(config.basin_weight)
    if config.cycle_weight:
        cycle_confidence = 1.0 / (1.0 + cycle_error.clamp_min(0.0))
        result = result * cycle_confidence.pow(config.cycle_weight)
    if config.jacobian_weight:
        result = result * jacobian_confidence.clamp(0.0, 1.0).pow(
            config.jacobian_weight
        )
    return result.clamp(0.0, 1.0)


def build_identity_maps_from_warp_fields(
    source_native: torch.Tensor,
    target_native: torch.Tensor,
    warp_ab: torch.Tensor,
    warp_ba: torch.Tensor,
    *,
    source_support: torch.Tensor | None = None,
    target_support: torch.Tensor | None = None,
    config: DenseWarpIdentityConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Encode already-constructed continuous warps as shared identity maps.

    ``warp_ab`` and ``warp_ba`` use normalized endpoint coordinates in
    ``[0,1]``.  The function is agnostic to the field provider: the fields may
    come from FLUX attention, RoMa, or a synthetic unit test.  No outcome label
    enters support normalization or feature fusion.
    """

    config = config or DenseWarpIdentityConfig()
    config.validate()
    if source_native.ndim != 4 or target_native.ndim != 4:
        raise ValueError("native descriptors must have shape [1,channels,height,width]")
    if int(source_native.shape[0]) != 1 or int(target_native.shape[0]) != 1:
        raise ValueError("native descriptors must have batch size one")
    if int(source_native.shape[1]) != int(target_native.shape[1]):
        raise ValueError("native descriptor channels must agree")
    source_height, source_width = map(int, source_native.shape[-2:])
    target_height, target_width = map(int, target_native.shape[-2:])
    if tuple(warp_ab.shape) != (source_height, source_width, 2):
        raise ValueError("forward warp does not match the source descriptor grid")
    if tuple(warp_ba.shape) != (target_height, target_width, 2):
        raise ValueError("backward warp does not match the target descriptor grid")
    if source_support is None:
        source_support = torch.ones(
            (source_height, source_width), device=warp_ab.device, dtype=warp_ab.dtype
        )
    if target_support is None:
        target_support = torch.ones(
            (target_height, target_width), device=warp_ba.device, dtype=warp_ba.dtype
        )
    if tuple(source_support.shape) != (source_height, source_width):
        raise ValueError("source support does not match the source descriptor grid")
    if tuple(target_support.shape) != (target_height, target_width):
        raise ValueError("target support does not match the target descriptor grid")
    source_support = torch.nan_to_num(
        source_support.float(), nan=0.0, posinf=0.0, neginf=0.0
    ).clamp(0.0, 1.0)
    target_support = torch.nan_to_num(
        target_support.float(), nan=0.0, posinf=0.0, neginf=0.0
    ).clamp(0.0, 1.0)
    warp_ab = torch.nan_to_num(
        warp_ab.float(), nan=0.5, posinf=1.0, neginf=0.0
    ).clamp(0.0, 1.0)
    warp_ba = torch.nan_to_num(
        warp_ba.float(), nan=0.5, posinf=1.0, neginf=0.0
    ).clamp(0.0, 1.0)

    source_coordinates = _normalized_grid(
        source_height,
        source_width,
        device=warp_ab.device,
        dtype=warp_ab.dtype,
    )
    target_coordinates = _normalized_grid(
        target_height,
        target_width,
        device=warp_ba.device,
        dtype=warp_ba.dtype,
    )
    cycle_error_a = _cycle_error_cells(
        source_coordinates,
        warp_ab,
        warp_ba,
        source_height,
        source_width,
    )
    cycle_error_b = _cycle_error_cells(
        target_coordinates,
        warp_ba,
        warp_ab,
        target_height,
        target_width,
    )
    jacobian_confidence_a, jacobian_residual_a = _jacobian_consistency(
        warp_ab, target_height, target_width
    )
    jacobian_confidence_b, jacobian_residual_b = _jacobian_consistency(
        warp_ba, source_height, source_width
    )
    reliability_a = _weighted_reliability(
        source_support, cycle_error_a, jacobian_confidence_a, config
    )
    reliability_b = _weighted_reliability(
        target_support, cycle_error_b, jacobian_confidence_b, config
    )
    source_identity = _coordinate_encoding(
        source_coordinates, warp_ab, config.fourier_frequencies
    )
    target_identity = _coordinate_encoding(
        warp_ba, target_coordinates, config.fourier_frequencies
    )
    source_identity = source_identity * reliability_a.unsqueeze(0).unsqueeze(0)
    target_identity = target_identity * reliability_b.unsqueeze(0).unsqueeze(0)
    source_forward_identity = _single_coordinate_encoding(
        warp_ab, config.fourier_frequencies
    )
    target_forward_identity = _single_coordinate_encoding(
        target_coordinates, config.fourier_frequencies
    )
    source_forward_identity = (
        source_forward_identity * reliability_a.unsqueeze(0).unsqueeze(0)
    )
    target_forward_identity = (
        target_forward_identity * reliability_b.unsqueeze(0).unsqueeze(0)
    )
    # Each unit branch receives equal energy. Their concatenation therefore
    # still has norm r, exactly matching the original mutual-identity branch.
    # This preserves the successful target-reliability/hubness prior while
    # averaging two distinct kernels: bidirectional shared identity and the
    # less restrictive A->B direct-warp coordinate evidence.
    dual_scale = float(2.0**-0.5)
    source_dual_identity = dual_scale * torch.cat(
        (source_identity, source_forward_identity), dim=1
    )
    target_dual_identity = dual_scale * torch.cat(
        (target_identity, target_forward_identity), dim=1
    )
    source_native_unit = F.normalize(source_native.float(), dim=1, eps=1e-12)
    target_native_unit = F.normalize(target_native.float(), dim=1, eps=1e-12)

    # Keep every augmented token at the same pre-normalization norm. Without
    # this slack, cosine([native, r * identity]) divides each target candidate
    # by sqrt(1 + scale^2 * r_target^2), which penalizes reliable candidates
    # even when the query has r_source=0 and contributes no identity evidence.
    # Source and target slack occupy orthogonal channels, so they contribute
    # norm but zero cross-image similarity. Consequently cosine ranking is
    # exactly equivalent to:
    #   native_cosine + scale^2 * r_source * r_target * identity_cosine
    # up to one query-independent positive constant.
    source_slack = torch.sqrt(
        (1.0 - reliability_a.square()).clamp_min(0.0)
    ).unsqueeze(0).unsqueeze(0)
    target_slack = torch.sqrt(
        (1.0 - reliability_b.square()).clamp_min(0.0)
    ).unsqueeze(0).unsqueeze(0)
    source_zero = torch.zeros_like(source_slack)
    target_zero = torch.zeros_like(target_slack)
    identity_scale = float(config.identity_scale)
    source_fused_variable_norm = F.normalize(
        torch.cat(
            (source_native_unit, identity_scale * source_identity), dim=1
        ),
        dim=1,
        eps=1e-12,
    )
    target_fused_variable_norm = F.normalize(
        torch.cat(
            (target_native_unit, identity_scale * target_identity), dim=1
        ),
        dim=1,
        eps=1e-12,
    )
    source_fused_forward = F.normalize(
        torch.cat(
            (source_native_unit, identity_scale * source_forward_identity), dim=1
        ),
        dim=1,
        eps=1e-12,
    )
    target_fused_forward = F.normalize(
        torch.cat(
            (target_native_unit, identity_scale * target_forward_identity), dim=1
        ),
        dim=1,
        eps=1e-12,
    )
    source_fused_dual = F.normalize(
        torch.cat(
            (source_native_unit, identity_scale * source_dual_identity), dim=1
        ),
        dim=1,
        eps=1e-12,
    )
    target_fused_dual = F.normalize(
        torch.cat(
            (target_native_unit, identity_scale * target_dual_identity), dim=1
        ),
        dim=1,
        eps=1e-12,
    )
    source_fused = F.normalize(
        torch.cat(
            (
                source_native_unit,
                identity_scale * source_identity,
                identity_scale * source_slack,
                source_zero,
            ),
            dim=1,
        ),
        dim=1,
        eps=1e-12,
    )
    target_fused = F.normalize(
        torch.cat(
            (
                target_native_unit,
                identity_scale * target_identity,
                target_zero,
                identity_scale * target_slack,
            ),
            dim=1,
        ),
        dim=1,
        eps=1e-12,
    )
    return {
        "source_fused": source_fused,
        "target_fused": target_fused,
        "source_fused_variable_norm": source_fused_variable_norm,
        "target_fused_variable_norm": target_fused_variable_norm,
        "source_fused_forward": source_fused_forward,
        "target_fused_forward": target_fused_forward,
        "source_fused_dual": source_fused_dual,
        "target_fused_dual": target_fused_dual,
        "source_identity": source_identity,
        "target_identity": target_identity,
        "source_forward_identity": source_forward_identity,
        "target_forward_identity": target_forward_identity,
        "source_dual_identity": source_dual_identity,
        "target_dual_identity": target_dual_identity,
        "warp_ab": warp_ab,
        "warp_ba": warp_ba,
        "support_a": source_support,
        "support_b": target_support,
        "cycle_error_a": cycle_error_a,
        "cycle_error_b": cycle_error_b,
        "jacobian_confidence_a": jacobian_confidence_a,
        "jacobian_confidence_b": jacobian_confidence_b,
        "jacobian_residual_a": jacobian_residual_a,
        "jacobian_residual_b": jacobian_residual_b,
        "reliability_a": reliability_a,
        "reliability_b": reliability_b,
    }


def build_dense_warp_identity_maps(
    source_native: torch.Tensor,
    target_native: torch.Tensor,
    p_ab: torch.Tensor,
    p_ba: torch.Tensor,
    *,
    source_grid: tuple[int, int],
    target_grid: tuple[int, int],
    config: DenseWarpIdentityConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Construct native-preserving, pair-conditioned identity descriptors."""

    config = config or DenseWarpIdentityConfig()
    config.validate()
    source_height, source_width = map(int, source_grid)
    target_height, target_width = map(int, target_grid)
    if source_native.ndim != 4 or target_native.ndim != 4:
        raise ValueError("native descriptors must have shape [1,channels,height,width]")
    if int(source_native.shape[0]) != 1 or int(target_native.shape[0]) != 1:
        raise ValueError("native descriptors must have batch size one")
    if int(source_native.shape[1]) != int(target_native.shape[1]):
        raise ValueError("native descriptor channels must agree")
    if tuple(source_native.shape[-2:]) != (source_height, source_width):
        raise ValueError("source native descriptor does not match source grid")
    if tuple(target_native.shape[-2:]) != (target_height, target_width):
        raise ValueError("target native descriptor does not match target grid")
    expected_ab = (source_height * source_width, target_height * target_width)
    expected_ba = (expected_ab[1], expected_ab[0])
    if tuple(p_ab.shape) != expected_ab or tuple(p_ba.shape) != expected_ba:
        raise ValueError(
            "attention shape mismatch: expected "
            f"{expected_ab}/{expected_ba}, got {tuple(p_ab.shape)}/{tuple(p_ba.shape)}"
        )

    posterior_ab, posterior_ba = mutual_attention_posteriors(p_ab, p_ba)
    warp_ab_flat, basin_a_flat = local_mode_coordinates(
        posterior_ab,
        target_height=target_height,
        target_width=target_width,
        sigma_cells=config.mode_sigma_cells,
        iterations=config.mode_iterations,
        basin_radius_cells=config.basin_radius_cells,
    )
    warp_ba_flat, basin_b_flat = local_mode_coordinates(
        posterior_ba,
        target_height=source_height,
        target_width=source_width,
        sigma_cells=config.mode_sigma_cells,
        iterations=config.mode_iterations,
        basin_radius_cells=config.basin_radius_cells,
    )
    warp_ab = warp_ab_flat.reshape(source_height, source_width, 2)
    warp_ba = warp_ba_flat.reshape(target_height, target_width, 2)
    basin_a = basin_a_flat.reshape(source_height, source_width)
    basin_b = basin_b_flat.reshape(target_height, target_width)

    source_coordinates = _normalized_grid(
        source_height,
        source_width,
        device=warp_ab.device,
        dtype=warp_ab.dtype,
    )
    target_coordinates = _normalized_grid(
        target_height,
        target_width,
        device=warp_ba.device,
        dtype=warp_ba.dtype,
    )
    cycle_error_a = _cycle_error_cells(
        source_coordinates,
        warp_ab,
        warp_ba,
        source_height,
        source_width,
    )
    cycle_error_b = _cycle_error_cells(
        target_coordinates,
        warp_ba,
        warp_ab,
        target_height,
        target_width,
    )
    jacobian_confidence_a, jacobian_residual_a = _jacobian_consistency(
        warp_ab, target_height, target_width
    )
    jacobian_confidence_b, jacobian_residual_b = _jacobian_consistency(
        warp_ba, source_height, source_width
    )
    reliability_a = _weighted_reliability(
        basin_a, cycle_error_a, jacobian_confidence_a, config
    )
    reliability_b = _weighted_reliability(
        basin_b, cycle_error_b, jacobian_confidence_b, config
    )

    source_identity = _coordinate_encoding(
        source_coordinates, warp_ab, config.fourier_frequencies
    )
    target_identity = _coordinate_encoding(
        warp_ba, target_coordinates, config.fourier_frequencies
    )
    source_identity = source_identity * reliability_a.unsqueeze(0).unsqueeze(0)
    target_identity = target_identity * reliability_b.unsqueeze(0).unsqueeze(0)
    source_native_unit = F.normalize(source_native.float(), dim=1, eps=1e-12)
    target_native_unit = F.normalize(target_native.float(), dim=1, eps=1e-12)
    source_fused = F.normalize(
        torch.cat(
            (source_native_unit, float(config.identity_scale) * source_identity),
            dim=1,
        ),
        dim=1,
        eps=1e-12,
    )
    target_fused = F.normalize(
        torch.cat(
            (target_native_unit, float(config.identity_scale) * target_identity),
            dim=1,
        ),
        dim=1,
        eps=1e-12,
    )
    return {
        "source_fused": source_fused,
        "target_fused": target_fused,
        "source_identity": source_identity,
        "target_identity": target_identity,
        "warp_ab": warp_ab,
        "warp_ba": warp_ba,
        "basin_mass_a": basin_a,
        "basin_mass_b": basin_b,
        "cycle_error_a": cycle_error_a,
        "cycle_error_b": cycle_error_b,
        "jacobian_confidence_a": jacobian_confidence_a,
        "jacobian_confidence_b": jacobian_confidence_b,
        "jacobian_residual_a": jacobian_residual_a,
        "jacobian_residual_b": jacobian_residual_b,
        "reliability_a": reliability_a,
        "reliability_b": reliability_b,
    }


def sample_dense_field(
    field: torch.Tensor,
    points_xy: Sequence[Sequence[float]],
    image_size: Sequence[int],
) -> torch.Tensor:
    """Sample a grid field at original-image xy coordinates."""

    height, width = int(image_size[0]), int(image_size[1])
    if height <= 0 or width <= 0:
        raise ValueError("image size must be positive")
    points = torch.as_tensor(points_xy, device=field.device, dtype=field.dtype)
    if points.ndim != 2 or int(points.shape[1]) != 2:
        raise ValueError("points must have shape [point,2]")
    scale = points.new_tensor(
        [float(max(width - 1, 1)), float(max(height - 1, 1))]
    )
    normalized = points / scale
    if width == 1:
        normalized[:, 0] = 0.0
    if height == 1:
        normalized[:, 1] = 0.0
    return _sample_field(field, normalized.clamp(0.0, 1.0))

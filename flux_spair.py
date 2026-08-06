"""Flux/SPair feature processing and matching helpers.

The native path mirrors ``eval_spair.py``. The diagnostic path resamples each
feature map to an aspect-preserving grid whose longest side has a fixed number
of cells, so candidate statistics are not dominated by native pixel density.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


FLUX_DISCARDED_CHANNELS = (154, 1446)
PAPER_FLUX_ALL_POINT = 0.671


def prepare_flux_feature(raw: torch.Tensor, ada: torch.Tensor, channel_discard: bool) -> torch.Tensor:
    """Apply the official channel-discard, LayerNorm and AdaLN operations."""
    if raw.ndim != 4 or raw.shape[0] != 1:
        raise ValueError(f"Expected a 1xCxHxW Flux feature, got {tuple(raw.shape)}")
    feature = raw.clone()
    channels = feature.shape[1]
    if channel_discard:
        if channels <= max(FLUX_DISCARDED_CHANNELS):
            raise ValueError("Feature has too few channels for official Flux channel discard")
        feature[:, FLUX_DISCARDED_CHANNELS, :, :] = 0.0

    feature = feature.permute(0, 2, 3, 1)
    feature = F.layer_norm(feature, (channels,), weight=None, bias=None, eps=1e-6)
    feature = feature.permute(0, 3, 1, 2)

    if ada.ndim < 3 or ada.shape[0] < 1 or ada.shape[1] < 2:
        raise ValueError(f"Unexpected cached AdaLN tensor shape: {tuple(ada.shape)}")
    # Mirror eval_spair.py exactly: the released cache stores channel-wise
    # shift and scale at ada[0][0] and ada[0][1].
    shift = ada[0][0].reshape(1, -1, 1, 1).to(device=feature.device, dtype=feature.dtype)
    scale = ada[0][1].reshape(1, -1, 1, 1).to(device=feature.device, dtype=feature.dtype)
    if shift.shape[1] != channels or scale.shape[1] != channels:
        raise ValueError(
            f"AdaLN channels do not match feature channels: "
            f"shift={shift.shape[1]}, scale={scale.shape[1]}, feature={channels}"
        )
    return (1.0 + scale) * feature + shift


def long_side_grid_shape(height: int, width: int, long_side: int) -> tuple[int, int]:
    """Return an aspect-preserving grid with exactly ``long_side`` on its long axis."""
    if min(height, width, long_side) <= 0:
        raise ValueError("Image and grid dimensions must be positive")
    if height >= width:
        return long_side, max(1, int(round(long_side * width / height)))
    return max(1, int(round(long_side * height / width))), long_side


def resize_feature_long_side(
    feature: torch.Tensor, height: int, width: int, long_side: int
) -> torch.Tensor:
    """Resize a 1xCxHxW feature to the controlled aspect-preserving grid."""
    grid_h, grid_w = long_side_grid_shape(height, width, long_side)
    return F.interpolate(feature, size=(grid_h, grid_w), mode="bilinear", align_corners=False)


def grid_candidate_points(
    height: int,
    width: int,
    grid_h: int,
    grid_w: int,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Map flattened diagnostic-grid cell centers back to original-image coordinates."""
    ys = ((torch.arange(grid_h, device=device, dtype=torch.float32) + 0.5) * height / grid_h - 0.5)
    xs = ((torch.arange(grid_w, device=device, dtype=torch.float32) + 0.5) * width / grid_w - 0.5)
    ys.clamp_(0, max(height - 1, 0))
    xs.clamp_(0, max(width - 1, 0))
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)


def points_to_grid_indices(
    points: Sequence[Sequence[float]] | torch.Tensor,
    height: int,
    width: int,
    grid_h: int,
    grid_w: int,
) -> torch.Tensor:
    """Assign original-image points to cells under half-pixel interpolation geometry."""
    points = torch.as_tensor(points, dtype=torch.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected Nx2 points, got {tuple(points.shape)}")
    x = torch.floor((points[:, 0] + 0.5) * grid_w / width).long().clamp_(0, grid_w - 1)
    y = torch.floor((points[:, 1] + 0.5) * grid_h / height).long().clamp_(0, grid_h - 1)
    return y * grid_w + x


def grid_cosine_scores(
    source_feature: torch.Tensor,
    target_feature: torch.Tensor,
    source_points: Sequence[Sequence[float]] | torch.Tensor,
    source_height: int,
    source_width: int,
) -> torch.Tensor:
    """Return source-keypoint to target-grid cosine similarities."""
    if source_feature.ndim != 4 or target_feature.ndim != 4:
        raise ValueError("Expected batched CxHxW source and target features")
    if source_feature.shape[:2] != target_feature.shape[:2] or source_feature.shape[0] != 1:
        raise ValueError("Source and target features must be 1xC spatial maps")
    source = F.normalize(source_feature[0].float().flatten(1).transpose(0, 1), dim=1, eps=1e-10)
    target = F.normalize(target_feature[0].float().flatten(1).transpose(0, 1), dim=1, eps=1e-10)
    indices = points_to_grid_indices(
        source_points,
        source_height,
        source_width,
        source_feature.shape[-2],
        source_feature.shape[-1],
    ).to(source.device)
    return source[indices] @ target.transpose(0, 1)


def native_flux_predictions(
    source_feature: torch.Tensor,
    target_feature: torch.Tensor,
    source_points: Sequence[Sequence[int]],
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> torch.Tensor:
    """Run the native-pixel cosine NN used by the official Flux evaluator."""
    source = F.interpolate(source_feature, size=source_size, mode="bilinear", align_corners=False)
    target = F.interpolate(target_feature, size=target_size, mode="bilinear", align_corners=False)
    target_vectors = F.normalize(target[0].flatten(1).transpose(0, 1), dim=1)
    predictions: list[torch.Tensor] = []
    for x, y in source_points:
        source_vector = source[0, :, int(y), int(x)].reshape(1, -1)
        source_vector = F.normalize(source_vector, dim=1)
        index = torch.mm(target_vectors, source_vector.transpose(0, 1)).argmax()
        predictions.append(torch.stack((index % target_size[1], index // target_size[1])).float())
    return torch.stack(predictions) if predictions else source.new_empty((0, 2), dtype=torch.float32)


def chunked_native_flux_predictions(
    source_feature: torch.Tensor,
    target_feature: torch.Tensor,
    source_points: Sequence[Sequence[int]],
    source_size: tuple[int, int],
    target_size: tuple[int, int],
    *,
    channel_chunk: int = 256,
) -> torch.Tensor:
    """Run native-pixel cosine NN without a full CxHxW upsample allocation.

    A 3072x960x960 tensor exceeds PyTorch's INT_MAX element limit. Bilinear
    interpolation is channel-separable, so source vectors can be gathered and
    target dot products/norms accumulated over channel chunks with unchanged
    spatial geometry and full-dimensional cosine matching.
    """
    if source_feature.ndim != 4 or target_feature.ndim != 4:
        raise ValueError("Expected batched CxHxW source and target features")
    if source_feature.shape[:2] != target_feature.shape[:2] or source_feature.shape[0] != 1:
        raise ValueError("Source and target features must be 1xC spatial maps")
    if channel_chunk < 1:
        raise ValueError("channel_chunk must be positive")
    points = torch.as_tensor(source_points, device=source_feature.device, dtype=torch.long)
    if points.numel() == 0:
        return source_feature.new_empty((0, 2), dtype=torch.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected Nx2 source points, got {tuple(points.shape)}")
    source_h, source_w = source_size
    target_h, target_w = target_size
    if (
        (points[:, 0] < 0).any()
        or (points[:, 0] >= source_w).any()
        or (points[:, 1] < 0).any()
        or (points[:, 1] >= source_h).any()
    ):
        raise ValueError("Source points fall outside the upsampled feature map")

    channels = source_feature.shape[1]
    source_parts: list[torch.Tensor] = []
    for start in range(0, channels, channel_chunk):
        end = min(start + channel_chunk, channels)
        upsampled = F.interpolate(
            source_feature[:, start:end].contiguous(),
            size=source_size,
            mode="bilinear",
            align_corners=False,
        )
        source_parts.append(upsampled[0, :, points[:, 1], points[:, 0]].transpose(0, 1).float())
        del upsampled
    source_vectors = torch.cat(source_parts, dim=1)
    del source_parts

    pixel_count = target_h * target_w
    dot_products = torch.zeros(
        (points.shape[0], pixel_count), device=source_feature.device, dtype=torch.float32
    )
    target_squared_norm = torch.zeros(
        pixel_count, device=source_feature.device, dtype=torch.float32
    )
    for start in range(0, channels, channel_chunk):
        end = min(start + channel_chunk, channels)
        upsampled = F.interpolate(
            target_feature[:, start:end].contiguous(),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        target_vectors = upsampled[0].flatten(1).transpose(0, 1).float()
        dot_products.addmm_(source_vectors[:, start:end], target_vectors.transpose(0, 1))
        target_squared_norm += target_vectors.square().sum(dim=1)
        del upsampled, target_vectors

    source_norm = source_vectors.square().sum(dim=1).sqrt().clamp_min_(1e-12)
    target_norm = target_squared_norm.sqrt().clamp_min_(1e-12)
    dot_products /= source_norm[:, None]
    dot_products /= target_norm[None, :]
    indices = dot_products.argmax(dim=1)
    return torch.stack((indices % target_w, indices // target_w), dim=1).float()


def native_flux_pck_hits(
    predictions: torch.Tensor, targets: torch.Tensor, threshold: float, alpha: float = 0.1
) -> torch.Tensor:
    """Use the inclusive PCK boundary from the released Flux evaluator."""
    if threshold <= 0:
        raise ValueError("PCK threshold must be positive")
    return torch.linalg.vector_norm(predictions.float() - targets.float(), dim=1) <= alpha * threshold

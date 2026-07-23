"""DINOv2 feature extraction and candidate-ownership diagnostics for SPair.

This module intentionally keeps DINOv2 separate from the Flux evaluator.  It
uses the Hugging Face DINOv2 implementation and exposes only tensor utilities
needed by the SPair evaluator, so the exploratory 4090 runs cannot alter the
official Flux baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DINOConfig:
    model_name: str = "facebook/dinov2-large"
    layer: int = 11
    patch_size: int = 14
    max_side: int = 840


def resize_shape(height: int, width: int, max_side: int, patch_size: int = 14) -> tuple[int, int]:
    """Return the resized content shape before square-canvas padding."""
    scale = float(max_side) / max(height, width)
    resized_h = max(1, int(round(height * scale)))
    resized_w = max(1, int(round(width * scale)))
    out_h = max(patch_size, (resized_h // patch_size) * patch_size)
    out_w = max(patch_size, (resized_w // patch_size) * patch_size)
    return out_h, out_w


def square_canvas_geometry(
    height: int, width: int, target_res: int
) -> tuple[float, int, int, int, int]:
    """Return scale, offsets, and resized content shape for official DINO input."""
    scale = float(target_res) / max(height, width)
    resized_h = max(1, int(round(height * scale)))
    resized_w = max(1, int(round(width * scale)))
    offset_y = (target_res - resized_h) // 2
    offset_x = (target_res - resized_w) // 2
    return scale, offset_x, offset_y, resized_h, resized_w


def dino_tokens_to_map(tokens: torch.Tensor, height: int, width: int, patch_size: int = 14) -> torch.Tensor:
    """Convert patch tokens to ``C,H/patch,W/patch`` without the CLS token."""
    if tokens.ndim != 3:
        raise ValueError(f"Expected [B,N,C] tokens, got {tuple(tokens.shape)}")
    grid_h, grid_w = height // patch_size, width // patch_size
    expected = grid_h * grid_w
    if tokens.shape[1] == expected + 1:
        tokens = tokens[:, 1:, :]
    if tokens.shape[1] != expected:
        raise ValueError(
            f"Token/grid mismatch: N={tokens.shape[1]}, expected {expected} "
            f"for image {(height, width)} and patch {patch_size}"
        )
    return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], grid_h, grid_w)


def normalize_feature_map(feature_map: torch.Tensor) -> torch.Tensor:
    """L2-normalize a ``C,H,W`` feature map along its channel dimension."""
    if feature_map.ndim != 3:
        raise ValueError(f"Expected [C,H,W], got {tuple(feature_map.shape)}")
    return F.normalize(feature_map.float(), dim=0, eps=1e-6)


def topk_candidate_indices(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    source_points: Sequence[Sequence[int]],
    topk: int,
) -> torch.Tensor:
    """Return target-map flat indices for every source keypoint.

    The result has shape ``[num_source_points, topk]``.  Source and target
    maps must already be resized to the same target/query image coordinates.
    """
    src = normalize_feature_map(source_features)
    trg = normalize_feature_map(target_features)
    vectors = torch.stack([src[:, int(y), int(x)] for x, y in source_points], dim=0)
    scores = torch.einsum("nc,chw->nhw", vectors, trg).flatten(1)
    return scores.topk(min(topk, scores.shape[1]), dim=1).indices


def candidate_hit(candidates: torch.Tensor, gt_xy: Sequence[float], width: int, threshold: float) -> torch.Tensor:
    """PCK hit for each row of flat target-map candidate indices."""
    y = torch.div(candidates, width, rounding_mode="floor").float()
    x = (candidates % width).float()
    gt = torch.tensor(gt_xy, device=candidates.device, dtype=torch.float32)
    distance = torch.sqrt((x - gt[0]) ** 2 + (y - gt[1]) ** 2)
    return (distance / max(float(threshold), 1e-6) <= 0.1)


def union_hit(
    candidates: torch.Tensor,
    gt_xy: Sequence[float],
    width: int,
    threshold: float,
    exclude_row: int | None = None,
) -> bool:
    """Check whether a GT point is covered by a union of candidate rows."""
    rows = candidates if exclude_row is None else torch.cat((candidates[:exclude_row], candidates[exclude_row + 1 :]))
    if rows.numel() == 0:
        return False
    return bool(candidate_hit(rows.reshape(1, -1), gt_xy, width, threshold).any().item())


def summarize_candidate_rows(
    candidates: torch.Tensor,
    gt_points: Sequence[Sequence[float]],
    threshold: float,
    width: int,
    ks: Iterable[int],
) -> list[dict[str, int]]:
    """Create per-keypoint ownership diagnostics for a candidate matrix."""
    rows: list[dict[str, int]] = []
    for point_index, gt_xy in enumerate(gt_points):
        row = {"point_index": point_index}
        for k in ks:
            k = min(int(k), candidates.shape[1])
            local = candidates[:, :k]
            owner_hit = bool(candidate_hit(local[point_index : point_index + 1], gt_xy, width, threshold).any().item())
            other_hit = union_hit(local, gt_xy, width, threshold, exclude_row=point_index)
            row[f"owner_candidate_hit@{k}"] = int(owner_hit)
            row[f"other_source_candidate_hit@{k}"] = int(other_hit)
            row[f"global_union_candidate_hit@{k}"] = int(owner_hit or other_hit)
        rows.append(row)
    return rows

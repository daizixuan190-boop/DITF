"""Shared DINOv2/SPair geometry and nearest-neighbour evaluation utilities.

The preprocessing and patch-coordinate protocol follows GeoAware-SC, which is
the DINOv2 reference implementation cited by DiTF: aspect-preserving resize to
an 840 square zero canvas, ViT-B/14 block-11 tokens, and a 60x60 patch grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


PAPER_DINO_ALL_POINT = 0.556
PAPER_DINO_ALL_IMAGE = 0.539


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


@dataclass
class CategoryMetrics:
    pair_scores: list[float] = field(default_factory=list)
    correct: int = 0
    total: int = 0

    def update(self, hits: torch.Tensor) -> None:
        hits = hits.detach().bool().cpu()
        self.pair_scores.append(float(hits.float().mean()))
        self.correct += int(hits.sum())
        self.total += int(hits.numel())

    @property
    def per_image(self) -> float:
        return float(np.mean(self.pair_scores)) if self.pair_scores else 0.0

    @property
    def per_point(self) -> float:
        return self.correct / self.total if self.total else 0.0


def square_canvas_geometry(
    height: int, width: int, target_res: int
) -> tuple[float, int, int, int, int]:
    """Return scale, offsets and content size used by GeoAware-SC ``resize``."""
    if height <= 0 or width <= 0 or target_res <= 0:
        raise ValueError("image and target dimensions must be positive")
    scale = float(target_res) / max(height, width)
    resized_h = max(1, int(np.around(target_res * height / width))) if height <= width else target_res
    resized_w = target_res if height <= width else max(1, int(np.around(target_res * width / height)))
    offset_y = (target_res - resized_h) // 2
    offset_x = (target_res - resized_w) // 2
    return scale, offset_x, offset_y, resized_h, resized_w


def preprocess_square_canvas(image: Image.Image, target_res: int) -> torch.Tensor:
    """Create the normalized square tensor used by the cited DINOv2 baseline."""
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
    """Map original-image ``(x, y)`` points onto the square input canvas."""
    scale, offset_x, offset_y, _, _ = square_canvas_geometry(height, width, target_res)
    output = torch.as_tensor(points, dtype=torch.float32).clone()
    if output.ndim != 2 or output.shape[1] != 2:
        raise ValueError(f"Expected Nx2 points, got {tuple(output.shape)}")
    output[:, 0].mul_(scale).add_(offset_x)
    output[:, 1].mul_(scale).add_(offset_y)
    return output


def tokens_to_patch_map(tokens: torch.Tensor, grid_size: int) -> torch.Tensor:
    """Convert block tokens into ``B,C,H,W``, dropping all special tokens."""
    if tokens.ndim != 3:
        raise ValueError(f"Expected BxNxC tokens, got {tuple(tokens.shape)}")
    patch_count = grid_size * grid_size
    if tokens.shape[1] < patch_count:
        raise ValueError(f"Expected at least {patch_count} patch tokens, got {tokens.shape[1]}")
    # Patch tokens are last in both standard and register-token DINOv2 models.
    patches = tokens[:, -patch_count:, :]
    return patches.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], grid_size, grid_size)


def points_to_patch_indices(points: torch.Tensor, image_size: int, grid_size: int) -> torch.Tensor:
    """Map canvas coordinates to flattened source patch indices by floor."""
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected Nx2 points, got {tuple(points.shape)}")
    xy = torch.floor(points * (grid_size / image_size)).long().clamp_(0, grid_size - 1)
    return xy[:, 1] * grid_size + xy[:, 0]


def cosine_nn_predictions(
    source_map: torch.Tensor,
    target_map: torch.Tensor,
    source_points: torch.Tensor,
    image_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match source keypoints to target patch centers with cosine NN."""
    scores = cosine_similarity_scores(source_map, target_map, source_points, image_size)
    best_scores, target_indices = scores.max(dim=1)
    predictions = patch_indices_to_points(target_indices, target_map.shape[-1], image_size / target_map.shape[-1])
    return predictions, best_scores


def cosine_similarity_scores(
    source_map: torch.Tensor,
    target_map: torch.Tensor,
    source_points: torch.Tensor,
    image_size: int,
) -> torch.Tensor:
    """Return source-keypoint to all-target-patch cosine similarities."""
    if source_map.shape != target_map.shape or source_map.ndim != 3:
        raise ValueError("source and target maps must have identical CxHxW shapes")
    _, grid_h, grid_w = source_map.shape
    if grid_h != grid_w:
        raise ValueError("the official SPair protocol expects a square patch grid")
    source = F.normalize(source_map.float().flatten(1).transpose(0, 1), dim=1, eps=1e-10)
    target = F.normalize(target_map.float().flatten(1).transpose(0, 1), dim=1, eps=1e-10)
    source_indices = points_to_patch_indices(source_points, image_size, grid_h).to(source.device)
    return source[source_indices] @ target.transpose(0, 1)


def patch_indices_to_points(indices: torch.Tensor, grid_width: int, stride: float) -> torch.Tensor:
    """Decode flattened patch indices as canvas-space patch centers."""
    x = (indices % grid_width).float() * stride + stride / 2
    y = torch.div(indices, grid_width, rounding_mode="floor").float() * stride + stride / 2
    return torch.stack((x, y), dim=-1)


def pck_hits(predictions: torch.Tensor, targets: torch.Tensor, threshold: float, alpha: float = 0.1) -> torch.Tensor:
    """Return per-point PCK hits using SPair's target bounding-box threshold."""
    if threshold <= 0:
        raise ValueError("PCK threshold must be positive")
    return torch.linalg.vector_norm(predictions.float() - targets.float(), dim=1) < alpha * threshold


# Candidate utilities are retained for the separate post-parity diagnostic.
def candidate_hit(
    candidates: torch.Tensor,
    gt_xy: Sequence[float],
    width: int,
    threshold: float,
    patch_stride: float = 1.0,
) -> torch.Tensor:
    points = patch_indices_to_points(candidates, width, patch_stride)
    gt = torch.tensor(gt_xy, device=candidates.device, dtype=torch.float32)
    return torch.linalg.vector_norm(points - gt, dim=-1) < 0.1 * float(threshold)


def summarize_candidate_rows(
    candidates: torch.Tensor,
    gt_points: Sequence[Sequence[float]],
    threshold: float,
    width: int,
    ks: Iterable[int],
    patch_stride: float = 1.0,
) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    for point_index, gt_xy in enumerate(gt_points):
        row = {"point_index": point_index}
        for requested_k in ks:
            k = min(int(requested_k), candidates.shape[1])
            local = candidates[:, :k]
            owner = bool(
                candidate_hit(local[point_index : point_index + 1], gt_xy, width, threshold, patch_stride).any()
            )
            others = torch.cat((local[:point_index], local[point_index + 1 :])).reshape(1, -1)
            other = bool(
                others.numel() and candidate_hit(others, gt_xy, width, threshold, patch_stride).any()
            )
            row[f"owner_candidate_hit@{requested_k}"] = int(owner)
            row[f"other_source_candidate_hit@{requested_k}"] = int(other)
            row[f"global_union_candidate_hit@{requested_k}"] = int(owner or other)
        rows.append(row)
    return rows


def _random_union_hit_probability(population: int, positives: int, draws: int) -> float:
    """Exact hit probability for uniformly sampling ``draws`` unique patches."""
    draws = min(max(int(draws), 0), population)
    positives = min(max(int(positives), 0), population)
    if draws == 0 or positives == 0:
        return 0.0
    if draws > population - positives:
        return 1.0
    log_miss = (
        math.lgamma(population - positives + 1)
        - math.lgamma(population - positives - draws + 1)
        - math.lgamma(population + 1)
        + math.lgamma(population - draws + 1)
    )
    return min(max(1.0 - math.exp(log_miss), 0.0), 1.0)


def controlled_candidate_rows(
    scores: torch.Tensor,
    gt_points: torch.Tensor,
    threshold: float,
    width: int,
    ks: Iterable[int],
    patch_stride: float,
) -> list[dict[str, int | float]]:
    """Compute ownership diagnostics with overlap, budget and random controls.

    Ground truth is used only to label candidate coverage after ranking. It
    never changes ``scores``, candidate proposals or baseline predictions.
    """
    if scores.ndim != 2 or scores.shape[0] != gt_points.shape[0]:
        raise ValueError("scores and gt_points must describe the same source points")
    if scores.shape[1] % width:
        raise ValueError("target patch count must be divisible by width")
    ks = tuple(sorted(set(int(k) for k in ks)))
    if not ks or min(ks) < 1:
        raise ValueError("candidate K values must be positive")

    num_sources, population = scores.shape
    max_rank = min(max(ks) * num_sources, population)
    ranked = scores.topk(max_rank, dim=1).indices
    all_indices = torch.arange(population, device=scores.device)
    all_points = patch_indices_to_points(all_indices, width, patch_stride)
    gt_points = gt_points.to(device=scores.device, dtype=torch.float32)
    radius = 0.1 * float(threshold)
    patch_hits_all_gt = torch.cdist(all_points, gt_points) < radius
    rows: list[dict[str, int | float]] = []

    for owner_index, gt_xy in enumerate(gt_points):
        row: dict[str, int | float] = {"point_index": owner_index}
        patch_hits_owner = patch_hits_all_gt[:, owner_index]
        positive_count = int(patch_hits_owner.sum())
        other_gt_columns = torch.arange(num_sources, device=scores.device) != owner_index

        for requested_k in ks:
            k = min(requested_k, ranked.shape[1])
            local = ranked[:, :k]
            owner_candidates = local[owner_index]
            owner_hit = bool(patch_hits_owner[owner_candidates].any())
            other_rows = torch.cat((local[:owner_index], local[owner_index + 1 :]), dim=0)
            other_candidates = other_rows.reshape(-1)
            other_hit_mask = patch_hits_owner[other_candidates] if other_candidates.numel() else torch.zeros(
                0, dtype=torch.bool, device=scores.device
            )
            other_hit = bool(other_hit_mask.any())
            strict_other_hit = bool(
                other_candidates.numel()
                and (other_hit_mask & ~patch_hits_all_gt[other_candidates][:, other_gt_columns].any(dim=1)).any()
            )
            global_hit = owner_hit or other_hit
            strict_global_hit = owner_hit or strict_other_hit

            global_unique = torch.unique(local)
            unique_budget = int(global_unique.numel())
            budget_owner = ranked[owner_index, :unique_budget]
            budget_owner_hit = bool(patch_hits_owner[budget_owner].any())
            source_support = int(patch_hits_owner[local].any(dim=1).sum())
            random_expected = _random_union_hit_probability(population, positive_count, unique_budget)

            row[f"owner_candidate_hit@{requested_k}"] = int(owner_hit)
            row[f"other_source_candidate_hit@{requested_k}"] = int(other_hit)
            row[f"global_union_candidate_hit@{requested_k}"] = int(global_hit)
            row[f"strict_other_source_candidate_hit@{requested_k}"] = int(strict_other_hit)
            row[f"strict_global_union_candidate_hit@{requested_k}"] = int(strict_global_hit)
            row[f"budget_matched_owner_candidate_hit@{requested_k}"] = int(budget_owner_hit)
            row[f"global_not_budget_owner_hit@{requested_k}"] = int(global_hit and not budget_owner_hit)
            row[f"strict_global_not_budget_owner_hit@{requested_k}"] = int(
                strict_global_hit and not budget_owner_hit
            )
            row[f"global_unique_candidate_count@{requested_k}"] = unique_budget
            row[f"proposal_source_count@{requested_k}"] = source_support
            row[f"random_union_expected_hit@{requested_k}"] = random_expected
        rows.append(row)
    return rows

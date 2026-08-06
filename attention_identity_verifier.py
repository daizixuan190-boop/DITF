"""Self-supervised verifier for fixed cross-attention candidate sets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_FEATURE_GROUPS = (
    "attention_aggregate",
    "qk_expert",
    "value_expert",
    "token_state",
    "channel_state_sketch",
)


@dataclass(frozen=True)
class VerifierConfig:
    feature_dims: dict[str, int]
    feature_groups: tuple[str, ...] = DEFAULT_FEATURE_GROUPS
    group_width: int = 32
    hidden_width: int = 128
    dropout: float = 0.1
    attention_prior_weight: float = 1.0
    global_query_context: bool = False


def _candidate_standardize(value: torch.Tensor) -> torch.Tensor:
    value = torch.nan_to_num(value.float(), nan=0.0, posinf=0.0, neginf=0.0)
    mean = value.mean(dim=1, keepdim=True)
    variance = (value - mean).square().mean(dim=1, keepdim=True)
    return (value - mean) / variance.add(1e-5).sqrt()


def attention_prior_scores(
    attention_scores: torch.Tensor,
    *,
    weight: float = 1.0,
) -> torch.Tensor:
    log_attention = torch.log(attention_scores.float().clamp_min(1e-30))
    standardized = _candidate_standardize(log_attention.unsqueeze(2)).squeeze(2)
    return float(weight) * standardized


class CandidateIdentityVerifier(nn.Module):
    """Permutation-equivariant residual scorer over a fixed candidate set."""

    def __init__(self, config: VerifierConfig):
        super().__init__()
        self.config = config
        missing = [name for name in config.feature_groups if name not in config.feature_dims]
        if missing:
            raise ValueError(f"verifier configuration lacks feature dimensions: {missing}")
        self.group_projections = nn.ModuleDict({
            name: nn.Sequential(
                nn.LayerNorm(int(config.feature_dims[name])),
                nn.Linear(int(config.feature_dims[name]), int(config.group_width)),
                nn.GELU(),
            )
            for name in config.feature_groups
        })
        embedding_width = int(config.group_width) * len(config.feature_groups)
        context_multiplier = 5 if bool(config.global_query_context) else 3
        self.pair_attention = None
        if bool(config.global_query_context):
            heads = 4 if embedding_width % 4 == 0 else 1
            self.pair_attention = nn.MultiheadAttention(
                embedding_width,
                num_heads=heads,
                dropout=float(config.dropout),
                batch_first=True,
            )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(embedding_width * context_multiplier, int(config.hidden_width)),
            nn.GELU(),
            nn.Dropout(float(config.dropout)),
            nn.Linear(int(config.hidden_width), int(config.hidden_width)),
            nn.GELU(),
        )
        self.residual_head = nn.Linear(int(config.hidden_width), 1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def forward(
        self,
        feature_groups: Mapping[str, torch.Tensor],
        attention_scores: torch.Tensor,
    ) -> torch.Tensor:
        projected = []
        expected_shape = None
        for name in self.config.feature_groups:
            if name not in feature_groups:
                raise ValueError(f"candidate batch lacks verifier feature group {name}")
            value = feature_groups[name]
            if value.ndim != 3:
                raise ValueError(f"feature group {name} must be [query,candidate,feature]")
            if int(value.shape[2]) != int(self.config.feature_dims[name]):
                raise ValueError(
                    f"feature dimension mismatch for {name}: {value.shape[2]} != "
                    f"{self.config.feature_dims[name]}"
                )
            if expected_shape is None:
                expected_shape = tuple(value.shape[:2])
            elif tuple(value.shape[:2]) != expected_shape:
                raise ValueError("verifier feature groups do not share candidate axes")
            projected.append(self.group_projections[name](_candidate_standardize(value)))
        embedding = torch.cat(projected, dim=2)
        if self.pair_attention is not None:
            query_count, candidate_count, width = embedding.shape
            flat = embedding.reshape(1, query_count * candidate_count, width)
            contextual, _ = self.pair_attention(flat, flat, flat, need_weights=False)
            embedding = (flat + contextual).reshape(query_count, candidate_count, width)
        mean_context = embedding.mean(dim=1, keepdim=True).expand_as(embedding)
        max_context = embedding.amax(dim=1, keepdim=True).expand_as(embedding)
        context = [embedding, mean_context, max_context]
        if bool(self.config.global_query_context):
            # A candidate can only be resolved against repeated/symmetric parts
            # when the scorer sees the other source queries in the same pair.
            global_mean = embedding.mean(dim=(0, 1), keepdim=True).expand_as(embedding)
            global_max = embedding.amax(dim=(0, 1), keepdim=True).expand_as(embedding)
            context.extend((global_mean, global_max))
        residual = self.residual_head(
            self.candidate_encoder(torch.cat(context, dim=2))
        ).squeeze(2)
        if attention_scores.shape != residual.shape:
            raise ValueError("attention scores do not align with candidate features")
        attention_prior = attention_prior_scores(
            attention_scores,
            weight=float(self.config.attention_prior_weight),
        )
        return attention_prior + residual


def verifier_config_from_batch(
    batch: Mapping[str, Any],
    *,
    feature_groups: Sequence[str] = DEFAULT_FEATURE_GROUPS,
    group_width: int = 32,
    hidden_width: int = 128,
    dropout: float = 0.1,
    attention_prior_weight: float = 1.0,
    global_query_context: bool = False,
) -> VerifierConfig:
    groups = batch.get("feature_groups")
    if not isinstance(groups, Mapping):
        raise ValueError("candidate batch lacks feature_groups")
    names = tuple(str(name) for name in feature_groups)
    dimensions = {}
    for name in names:
        value = groups.get(name)
        if not isinstance(value, torch.Tensor) or value.ndim != 3:
            raise ValueError(f"candidate batch has invalid feature group {name}")
        dimensions[name] = int(value.shape[2])
    return VerifierConfig(
        feature_dims=dimensions,
        feature_groups=names,
        group_width=int(group_width),
        hidden_width=int(hidden_width),
        dropout=float(dropout),
        attention_prior_weight=float(attention_prior_weight),
        global_query_context=bool(global_query_context),
    )


def transformed_candidate_targets(
    candidate_pixels: torch.Tensor,
    target_points: torch.Tensor,
    target_size: Sequence[int],
    *,
    sigma_pixels: float,
    max_distance_pixels: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create soft labels solely from a known image-space transformation."""

    if candidate_pixels.ndim != 2:
        raise ValueError("candidate_pixels must be [query,candidate]")
    if target_points.ndim != 2 or target_points.shape != (candidate_pixels.shape[0], 2):
        raise ValueError("target_points must be [query,2] and align with candidates")
    target_h, target_w = map(int, target_size)
    if target_h <= 0 or target_w <= 0:
        raise ValueError("target size must be positive")
    if float(sigma_pixels) <= 0 or float(max_distance_pixels) <= 0:
        raise ValueError("target sigma and maximum distance must be positive")
    pixels = candidate_pixels.long()
    candidate_x = (pixels % target_w).float()
    candidate_y = torch.div(pixels, target_w, rounding_mode="floor").float()
    distance_squared = (
        (candidate_x - target_points[:, 0, None].float()).square()
        + (candidate_y - target_points[:, 1, None].float()).square()
    )
    minimum_distance = distance_squared.amin(dim=1).sqrt()
    recoverable = minimum_distance <= float(max_distance_pixels)
    logits = -distance_squared / (2.0 * float(sigma_pixels) ** 2)
    targets = torch.softmax(logits, dim=1)
    return targets, recoverable, minimum_distance


def listwise_identity_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    recoverable: torch.Tensor,
) -> torch.Tensor:
    if scores.shape != targets.shape or recoverable.shape != scores.shape[:1]:
        raise ValueError("listwise verifier scores, targets, and mask do not align")
    if not bool(recoverable.any()):
        return scores.sum() * 0.0
    log_probabilities = F.log_softmax(scores[recoverable], dim=1)
    return -(targets[recoverable] * log_probabilities).sum(dim=1).mean()


def weighted_listwise_identity_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    query_weights: torch.Tensor,
) -> torch.Tensor:
    """Cross entropy for noisy soft targets with continuous query confidence."""

    if scores.shape != targets.shape or query_weights.shape != scores.shape[:1]:
        raise ValueError("weighted listwise scores, targets, and weights do not align")
    weights = torch.nan_to_num(
        query_weights.to(device=scores.device, dtype=torch.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).clamp_min(0.0)
    if float(weights.sum()) <= 0.0:
        return scores.sum() * 0.0
    per_query = -(
        targets.to(device=scores.device, dtype=torch.float32)
        * F.log_softmax(scores, dim=1)
    ).sum(dim=1)
    return (per_query * weights).sum() / weights.sum().clamp_min(1e-12)


def _geometry_grid(
    point_map: torch.Tensor,
    grid_size: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resize and normalize a single-view point map without fixing its frame."""

    points = point_map.float()
    if points.ndim == 4:
        if int(points.shape[0]) != 1:
            raise ValueError("geometry point map must have batch size one")
        points = points[0]
    if points.ndim != 3:
        raise ValueError("geometry point map must be [H,W,3] or [1,H,W,3]")
    if int(points.shape[-1]) == 3:
        points = points.permute(2, 0, 1).unsqueeze(0)
    elif int(points.shape[0]) == 3:
        points = points.unsqueeze(0)
    else:
        raise ValueError("geometry point map must have three coordinate channels")
    grid_h, grid_w = map(int, grid_size)
    if grid_h <= 0 or grid_w <= 0:
        raise ValueError("geometry grid size must be positive")
    resized = F.interpolate(
        points,
        size=(grid_h, grid_w),
        mode="bilinear",
        align_corners=False,
    )[0].permute(1, 2, 0).reshape(-1, 3)
    valid = torch.isfinite(resized).all(dim=1)
    clean = torch.nan_to_num(resized, nan=0.0, posinf=0.0, neginf=0.0)
    if bool(valid.any()):
        center = clean[valid].median(dim=0).values
        centered = clean - center
        scale = centered[valid].norm(dim=1).median().clamp_min(1e-6)
        clean = centered / scale
    else:
        clean = torch.zeros_like(clean)
    return clean, valid


def _normalize_cost(cost: torch.Tensor) -> torch.Tensor:
    finite = torch.nan_to_num(cost.float(), nan=0.0, posinf=0.0, neginf=0.0)
    minimum = finite.amin()
    maximum = finite.amax()
    return (finite - minimum) / (maximum - minimum).clamp_min(1e-6)


def _unbalanced_sinkhorn_plan(
    cost: torch.Tensor,
    *,
    rho: float,
    iterations: int,
) -> torch.Tensor:
    """Log-domain UOT update used by the Shape-of-You pseudo-labeler."""

    if cost.ndim != 2 or min(cost.shape) <= 0:
        raise ValueError("UOT cost must be a non-empty matrix")
    if float(rho) <= 0.0 or int(iterations) <= 0:
        raise ValueError("UOT rho and iteration count must be positive")
    rows, columns = map(int, cost.shape)
    log_kernel = -cost.float() / float(rho)
    log_source_mass = cost.new_full((rows,), -torch.log(cost.new_tensor(float(rows))))
    log_target_mass = cost.new_full(
        (columns,), -torch.log(cost.new_tensor(float(columns)))
    )
    source_dual = torch.zeros_like(log_source_mass)
    target_dual = torch.zeros_like(log_target_mass)
    relaxation = float(rho) / (float(rho) + 1.0)
    for _ in range(int(iterations)):
        source_dual = relaxation * (
            log_source_mass
            - torch.logsumexp(log_kernel + target_dual[None, :], dim=1)
        )
        target_dual = relaxation * (
            log_target_mass
            - torch.logsumexp(log_kernel + source_dual[:, None], dim=0)
        )
    log_plan = log_kernel + source_dual[:, None] + target_dual[None, :]
    return torch.exp(log_plan - log_plan.amax()).clamp_min(1e-30)


def _reciprocal_plan_anchors(
    plan: torch.Tensor,
    source_valid: torch.Tensor,
    target_valid: torch.Tensor,
    *,
    max_anchors: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if int(plan.shape[1]) < 2:
        empty = torch.empty(0, device=plan.device, dtype=torch.long)
        return empty, empty, torch.empty(0, device=plan.device)
    values, target_indices = plan.topk(k=2, dim=1)
    source_indices = torch.arange(int(plan.shape[0]), device=plan.device)
    reverse = plan.argmax(dim=0)
    reciprocal = reverse[target_indices[:, 0]].eq(source_indices)
    valid = source_valid & target_valid[target_indices[:, 0]] & reciprocal
    margin = (values[:, 0] - values[:, 1]).clamp_min(0.0)
    confidence = margin / values[:, 0].clamp_min(1e-12)
    valid = valid & confidence.gt(1e-6)
    valid_sources = source_indices[valid]
    valid_targets = target_indices[:, 0][valid]
    valid_confidence = confidence[valid]
    keep = min(max(0, int(max_anchors)), int(valid_sources.numel()))
    if keep <= 0:
        empty = torch.empty(0, device=plan.device, dtype=torch.long)
        return empty, empty, torch.empty(0, device=plan.device)
    order = torch.argsort(valid_confidence, descending=True)[:keep]
    return (
        valid_sources[order],
        valid_targets[order],
        valid_confidence[order],
    )


def geometry_fgw_pseudo_targets(
    batch: Mapping[str, Any],
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    source_point_map: torch.Tensor,
    target_point_map: torch.Tensor,
    *,
    alpha: float = 0.3,
    rho: float = 0.75,
    refinement_steps: int = 5,
    sinkhorn_iterations: int = 20,
    max_anchors: int = 64,
    minimum_anchors: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Generate soft attention-candidate labels with a training-only 3D teacher.

    FLUX native cosine defines the dense semantic UOT plan. Single-view 3D
    point maps define only intra-image distances, so their arbitrary camera
    frames, translation, and scale do not enter the supervision. The final
    plan is restricted back to the fixed cross-attention candidate set.
    """

    if source_features.ndim != 4 or target_features.ndim != 4:
        raise ValueError("geometry FGW teacher expects [B,C,H,W] feature maps")
    if int(source_features.shape[0]) != 1 or int(target_features.shape[0]) != 1:
        raise ValueError("geometry FGW teacher expects batch size one")
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("geometry fusion alpha must be in [0,1]")
    source_cells = batch.get("source_cells")
    candidate_cells = batch.get("candidate_cells")
    if not isinstance(source_cells, torch.Tensor) or not isinstance(
        candidate_cells, torch.Tensor
    ):
        raise ValueError("geometry FGW batch lacks source/candidate cells")
    if source_cells.ndim != 1 or candidate_cells.ndim != 2:
        raise ValueError("geometry FGW cells must be [query] and [query,candidate]")
    if int(source_cells.shape[0]) != int(candidate_cells.shape[0]):
        raise ValueError("geometry FGW query axes do not align")

    device = source_features.device
    source_cells_device = source_cells.to(device=device, dtype=torch.long)
    candidate_cells_device = candidate_cells.to(device=device, dtype=torch.long)
    source_flat = F.normalize(
        torch.nan_to_num(source_features[0].float()).reshape(
            int(source_features.shape[1]), -1
        ).t(),
        dim=1,
        eps=1e-12,
    )
    target_flat = F.normalize(
        torch.nan_to_num(target_features[0].float()).reshape(
            int(target_features.shape[1]), -1
        ).t(),
        dim=1,
        eps=1e-12,
    )
    if bool((source_cells_device < 0).any()) or bool(
        (source_cells_device >= int(source_flat.shape[0])).any()
    ):
        raise ValueError("geometry FGW source cell is out of bounds")
    if bool((candidate_cells_device < 0).any()) or bool(
        (candidate_cells_device >= int(target_flat.shape[0])).any()
    ):
        raise ValueError("geometry FGW candidate cell is out of bounds")

    source_geometry, source_geometry_valid = _geometry_grid(
        source_point_map.to(device), source_features.shape[-2:]
    )
    target_geometry, target_geometry_valid = _geometry_grid(
        target_point_map.to(device), target_features.shape[-2:]
    )
    query_features = source_flat.index_select(0, source_cells_device)
    semantic_cost = 1.0 - torch.mm(query_features, target_flat.t())
    semantic_cost_normalized = _normalize_cost(semantic_cost)
    plan = _unbalanced_sinkhorn_plan(
        semantic_cost_normalized,
        rho=float(rho),
        iterations=int(sinkhorn_iterations),
    )
    query_geometry = source_geometry.index_select(0, source_cells_device)
    query_valid = source_geometry_valid.index_select(0, source_cells_device)
    used_geometry = False
    anchor_count = 0
    anchor_confidence_mean = 0.0
    for _ in range(max(0, int(refinement_steps))):
        anchor_source_rows, anchor_target_cells, anchor_confidence = (
            _reciprocal_plan_anchors(
                plan,
                query_valid,
                target_geometry_valid,
                max_anchors=int(max_anchors),
            )
        )
        anchor_count = int(anchor_source_rows.numel())
        if anchor_count < int(minimum_anchors):
            break
        source_anchor_geometry = query_geometry.index_select(0, anchor_source_rows)
        target_anchor_geometry = target_geometry.index_select(0, anchor_target_cells)
        source_distances = torch.cdist(query_geometry, source_anchor_geometry)
        target_distances = torch.cdist(target_geometry, target_anchor_geometry)
        weights = anchor_confidence / anchor_confidence.sum().clamp_min(1e-12)
        geometry_cost = (
            (
                source_distances[:, None, :]
                - target_distances[None, :, :]
            ).abs()
            * weights[None, None, :]
        ).sum(dim=2)
        invalid = (~query_valid)[:, None] | (~target_geometry_valid)[None, :]
        geometry_cost = geometry_cost.masked_fill(invalid, geometry_cost.amax() + 1.0)
        total_cost = (
            (1.0 - float(alpha)) * semantic_cost_normalized
            + float(alpha) * _normalize_cost(geometry_cost)
        )
        plan = _unbalanced_sinkhorn_plan(
            total_cost,
            rho=float(rho),
            iterations=int(sinkhorn_iterations),
        )
        used_geometry = True
        anchor_confidence_mean = float(anchor_confidence.mean().detach().cpu())

    candidate_plan = plan.gather(1, candidate_cells_device).clamp_min(1e-30)
    targets = candidate_plan / candidate_plan.sum(dim=1, keepdim=True).clamp_min(1e-30)
    if int(targets.shape[1]) > 1:
        entropy = -(targets * targets.clamp_min(1e-30).log()).sum(dim=1)
        entropy_confidence = 1.0 - entropy / torch.log(
            targets.new_tensor(float(targets.shape[1]))
        )
    else:
        entropy = torch.zeros(int(targets.shape[0]), device=device)
        entropy_confidence = torch.ones_like(entropy)
    candidate_geometry_valid = target_geometry_valid[candidate_cells_device].any(dim=1)
    query_weights = (
        entropy_confidence.clamp(0.0, 1.0)
        * query_valid.float()
        * candidate_geometry_valid.float()
    )
    if not used_geometry:
        query_weights = torch.zeros_like(query_weights)
    diagnostics: dict[str, Any] = {
        "used_geometry": bool(used_geometry),
        "anchor_count": int(anchor_count),
        "anchor_confidence_mean": float(anchor_confidence_mean),
        "target_entropy_mean": float(entropy.mean().detach().cpu()),
        "query_weight_mean": float(query_weights.mean().detach().cpu()),
        "geometry_valid_query_fraction": float(query_valid.float().mean().detach().cpu()),
        "geometry_valid_target_fraction": float(
            target_geometry_valid.float().mean().detach().cpu()
        ),
    }
    return targets.detach().cpu(), query_weights.detach().cpu(), diagnostics


def native_cycle_pseudo_targets(
    batch: Mapping[str, Any],
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    *,
    cycle_radius_cells: float,
    minimum_native_margin: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Build cross-instance pseudo labels from frozen native FLUX consistency."""

    if source_features.ndim != 4 or target_features.ndim != 4:
        raise ValueError("native cycle teacher expects [B,C,H,W] feature maps")
    if source_features.shape[0] != 1 or target_features.shape[0] != 1:
        raise ValueError("native cycle teacher expects batch size one")
    groups = batch.get("feature_groups")
    if not isinstance(groups, Mapping) or "native_control" not in groups:
        raise ValueError("candidate batch lacks frozen native-control scores")
    native_scores = groups["native_control"]
    if native_scores.ndim != 3 or native_scores.shape[2] != 1:
        raise ValueError("native-control scores must be [query,candidate,1]")
    if native_scores.shape[1] < 2:
        raise ValueError("native cycle teacher requires at least two candidates")
    source_cells = batch.get("source_cells")
    candidate_cells = batch.get("candidate_cells")
    if not isinstance(source_cells, torch.Tensor) or not isinstance(candidate_cells, torch.Tensor):
        raise ValueError("candidate batch lacks source/candidate cells")
    if candidate_cells.shape[:2] != native_scores.shape[:2]:
        raise ValueError("native scores and candidate cells do not align")

    device = source_features.device
    scores = native_scores.squeeze(2).to(device=device, dtype=torch.float32)
    teacher_rank = scores.argmax(dim=1)
    selected_target_cells = candidate_cells.to(device=device, dtype=torch.long).gather(
        1, teacher_rank[:, None]
    ).squeeze(1)
    source_cells_device = source_cells.to(device=device, dtype=torch.long)
    source_flat = F.normalize(
        source_features[0].float().reshape(source_features.shape[1], -1).t(),
        dim=1,
    )
    target_flat = F.normalize(
        target_features[0].float().reshape(target_features.shape[1], -1).t(),
        dim=1,
    )
    selected_target = target_flat.index_select(0, selected_target_cells)
    reverse_source_cells = torch.mm(selected_target, source_flat.t()).argmax(dim=1)
    source_width = int(source_features.shape[-1])
    source_x = source_cells_device % source_width
    source_y = torch.div(source_cells_device, source_width, rounding_mode="floor")
    reverse_x = reverse_source_cells % source_width
    reverse_y = torch.div(reverse_source_cells, source_width, rounding_mode="floor")
    cycle_distance = torch.sqrt(
        (source_x.float() - reverse_x.float()).square()
        + (source_y.float() - reverse_y.float()).square()
    )
    top_two = scores.topk(k=2, dim=1).values
    native_margin = top_two[:, 0] - top_two[:, 1]
    confident = (
        cycle_distance <= float(cycle_radius_cells)
    ) & (
        native_margin >= float(minimum_native_margin)
    )
    targets = F.one_hot(teacher_rank, num_classes=int(scores.shape[1])).float()
    diagnostics = {
        "teacher_rank": teacher_rank.detach().cpu(),
        "reverse_source_cells": reverse_source_cells.detach().cpu(),
        "cycle_distance_cells": cycle_distance.detach().cpu(),
        "native_margin": native_margin.detach().cpu(),
        "attention_teacher_agreement": teacher_rank.eq(0).detach().cpu(),
    }
    return targets.detach().cpu(), confident.detach().cpu(), diagnostics


def triangle_cycle_pseudo_targets(
    batch: Mapping[str, Any],
    mutual_target_bridge: torch.Tensor,
    mutual_bridge_source: torch.Tensor,
    *,
    source_grid_size: Sequence[int],
    cycle_radius_cells: float,
    require_unique_best: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Select A->B candidates whose B->C->A attention path closes uniquely.

    The candidate set itself is the fixed A->B mutual-attention top-k.  No
    annotation, descriptor teacher, coordinate prior, or category target is
    consulted.  Path support is used only to break equal cycle-distance ties;
    tied minima remain unconfirmed when ``require_unique_best`` is enabled.
    """

    source_cells = batch.get("source_cells")
    candidate_cells = batch.get("candidate_cells")
    if not isinstance(source_cells, torch.Tensor) or not isinstance(
        candidate_cells, torch.Tensor
    ):
        raise ValueError("triangle-cycle batch lacks source/candidate cells")
    if source_cells.ndim != 1 or candidate_cells.ndim != 2:
        raise ValueError("triangle-cycle cells must be [query] and [query,candidate]")
    if int(candidate_cells.shape[0]) != int(source_cells.shape[0]):
        raise ValueError("triangle-cycle source and candidate queries do not align")
    if mutual_target_bridge.ndim != 2 or mutual_bridge_source.ndim != 2:
        raise ValueError("triangle-cycle mutual attention matrices must be rank two")
    if int(mutual_target_bridge.shape[1]) != int(mutual_bridge_source.shape[0]):
        raise ValueError("triangle-cycle bridge dimensions do not align")
    source_h, source_w = map(int, source_grid_size)
    if source_h <= 0 or source_w <= 0:
        raise ValueError("triangle-cycle source grid must be positive")
    if int(mutual_bridge_source.shape[1]) != source_h * source_w:
        raise ValueError("triangle-cycle return matrix does not match source grid")
    if float(cycle_radius_cells) < 0.0:
        raise ValueError("triangle-cycle radius must be non-negative")

    device = mutual_target_bridge.device
    if mutual_bridge_source.device != device:
        raise ValueError("triangle-cycle mutual matrices must share a device")
    candidates = candidate_cells.to(device=device, dtype=torch.long)
    sources = source_cells.to(device=device, dtype=torch.long)
    if bool((candidates < 0).any()) or bool(
        (candidates >= int(mutual_target_bridge.shape[0])).any()
    ):
        raise ValueError("triangle-cycle candidate cell is out of bounds")
    if bool((sources < 0).any()) or bool((sources >= source_h * source_w).any()):
        raise ValueError("triangle-cycle source cell is out of bounds")

    query_count, candidate_count = map(int, candidates.shape)
    target_bridge_rows = mutual_target_bridge.index_select(0, candidates.reshape(-1))
    bridge_cells = target_bridge_rows.argmax(dim=1)
    first_support = target_bridge_rows.gather(1, bridge_cells[:, None]).squeeze(1)
    bridge_source_rows = mutual_bridge_source.index_select(0, bridge_cells)
    return_cells = bridge_source_rows.argmax(dim=1)
    second_support = bridge_source_rows.gather(1, return_cells[:, None]).squeeze(1)

    return_cells = return_cells.reshape(query_count, candidate_count)
    bridge_cells = bridge_cells.reshape(query_count, candidate_count)
    path_support = torch.sqrt(
        (first_support * second_support).clamp_min(0.0)
    ).reshape(query_count, candidate_count)
    source_x = (sources % source_w).float()[:, None]
    source_y = torch.div(sources, source_w, rounding_mode="floor").float()[:, None]
    return_x = (return_cells % source_w).float()
    return_y = torch.div(return_cells, source_w, rounding_mode="floor").float()
    cycle_distance = torch.sqrt(
        (return_x - source_x).square() + (return_y - source_y).square()
    )
    best_distance = cycle_distance.amin(dim=1)
    minimum_mask = cycle_distance.eq(best_distance[:, None])
    unique_best = minimum_mask.sum(dim=1).eq(1)
    tie_break_support = path_support.masked_fill(~minimum_mask, float("-inf"))
    teacher_rank = tie_break_support.argmax(dim=1)
    confident = best_distance.le(float(cycle_radius_cells))
    if bool(require_unique_best):
        confident = confident & unique_best
    targets = F.one_hot(teacher_rank, num_classes=candidate_count).float()
    diagnostics = {
        "teacher_rank": teacher_rank.detach().cpu(),
        "bridge_cells": bridge_cells.detach().cpu(),
        "return_cells": return_cells.detach().cpu(),
        "cycle_distance_cells": cycle_distance.detach().cpu(),
        "best_cycle_distance_cells": best_distance.detach().cpu(),
        "unique_best": unique_best.detach().cpu(),
        "path_support": path_support.detach().cpu(),
        "attention_teacher_agreement": teacher_rank.eq(0).detach().cpu(),
    }
    return targets.detach().cpu(), confident.detach().cpu(), diagnostics


def horizontal_flip_points(points: torch.Tensor, image_width: int) -> torch.Tensor:
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must be [query,2]")
    result = points.clone().float()
    result[:, 0] = float(int(image_width) - 1) - result[:, 0]
    return result


def sample_replay_cell_centers(
    image_size: Sequence[int],
    replay_shape: Sequence[int],
    *,
    count: int,
    generator: torch.Generator,
    border_cells: int = 1,
) -> torch.Tensor:
    image_h, image_w = map(int, image_size)
    grid_h, grid_w = map(int, replay_shape)
    border = max(0, int(border_cells))
    ys = torch.arange(border, max(border, grid_h - border), dtype=torch.long)
    xs = torch.arange(border, max(border, grid_w - border), dtype=torch.long)
    if ys.numel() == 0 or xs.numel() == 0:
        raise ValueError("replay grid is too small for the requested border")
    cells = torch.cartesian_prod(ys, xs)
    selected_count = min(max(1, int(count)), int(cells.shape[0]))
    order = torch.randperm(int(cells.shape[0]), generator=generator)[:selected_count]
    selected = cells.index_select(0, order)
    x = (selected[:, 1].float() + 0.5) * float(image_w) / float(grid_w) - 0.5
    y = (selected[:, 0].float() + 0.5) * float(image_h) / float(grid_h) - 0.5
    return torch.stack((x, y), dim=1)


def select_candidate_pixels(scores: torch.Tensor, candidate_pixels: torch.Tensor) -> torch.Tensor:
    if scores.shape != candidate_pixels.shape:
        raise ValueError("scores and candidate pixels do not align")
    selected = scores.argmax(dim=1, keepdim=True)
    return candidate_pixels.long().gather(1, selected).squeeze(1)


def checkpoint_payload(
    model: CandidateIdentityVerifier,
    *,
    training_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "method": "equivariant_attention_candidate_identity_verifier",
        "config": asdict(model.config),
        "state_dict": model.state_dict(),
        "training_metadata": dict(training_metadata),
    }


def load_verifier_checkpoint(
    path: str,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[CandidateIdentityVerifier, dict[str, Any]]:
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict) or int(payload.get("format_version", -1)) != 1:
        raise ValueError("unsupported attention identity verifier checkpoint")
    raw_config = dict(payload["config"])
    raw_config["feature_groups"] = tuple(raw_config["feature_groups"])
    raw_config["feature_dims"] = {
        str(name): int(value) for name, value in raw_config["feature_dims"].items()
    }
    model = CandidateIdentityVerifier(VerifierConfig(**raw_config))
    model.load_state_dict(payload["state_dict"])
    return model, payload

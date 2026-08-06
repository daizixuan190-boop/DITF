"""Attention-focused train-free SPair matchers.

This module keeps the official cosine NN baseline plus the current FJSAR
cross-attention replay work surface.  Deprecated post-processing, fusion, CSLS,
SCOT/GWOT, and local-graph experiments were removed from the active code path.
"""


from __future__ import annotations


import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Sequence


import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching


import torch


from torch.nn import functional as F


from flux_joint_replay import (
    FluxReplayState,
    _block_qkv,
    _state_to_device,
    flux_candidate_clamped_causal_probe,
    flux_candidate_counterfactual_fingerprint_probe,
    flux_candidate_internal_state_probe,
    flux_cross_readout_probe,
    flux_persistent_candidate_slot_replay_probe,
    native_image_tokens,
    native_parity_error,
    parity_metrics,
    run_flux_balanced_transport_stack,
    run_flux_cross_only_stack,
    run_flux_geometry_consistent_stack,
    run_flux_identity_preserving_stack,
    run_flux_joint_stack,
    run_flux_native_stack,
    run_flux_qk_identity_stack,
)


@dataclass(frozen=True)
class AttentionSparsePartialGraph:
    """Sparse attention candidate graph reserved for partial assignment."""

    source_edge_index: torch.Tensor
    candidate_target_index: torch.Tensor
    candidate_unary_log_probability: torch.Tensor
    candidate_mask: torch.Tensor
    source_grid_size: tuple[int, int]
    target_grid_size: tuple[int, int]
    dustbin_target_index: int

    def contract(self) -> dict[str, Any]:
        source_count, candidate_count = self.candidate_target_index.shape
        target_height, target_width = self.target_grid_size
        return {
            "formulation": "dense_nodes_sparse_candidates_partial_assignment",
            "source_node_count": int(source_count),
            "target_node_count": int(target_height * target_width),
            "source_edge_count": int(self.source_edge_index.shape[1]),
            "candidate_topk": int(candidate_count),
            "candidate_edge_count": int(self.candidate_mask.sum().detach().cpu()),
            "source_grid_size": [int(value) for value in self.source_grid_size],
            "target_grid_size": [int(value) for value in self.target_grid_size],
            "dustbin_target_index": int(self.dustbin_target_index),
            "dustbin_reserved": True,
            "candidate_source": "mutual_cross_attention_topk_only",
            "unary_signal": "row_normalized_log_mutual_cross_attention",
            "pairwise_signal": "local_spatial_and_ditf_relation_max_product",
            "gt_used_for_scoring": False,
            "native_candidate_injected": False,
            "native_fallback_used": False,
        }


def cosine_nn_predict(
    src_features: torch.Tensor,
    trg_features: torch.Tensor,
    source_points: Iterable[Sequence[float]],
) -> list[list[int]]:
    """Official DiTF per-point cosine argmax on already-upsampled features."""
    if src_features.ndim != 4 or trg_features.ndim != 4:
        raise ValueError("features must have shape [B, C, H, W]")
    _, channels, height, width = trg_features.shape
    trg_flat = F.normalize(trg_features[0].reshape(channels, -1).t(), dim=1)
    predictions = []
    for point in source_points:
        x, y = int(point[0]), int(point[1])
        src_vec = F.normalize(src_features[0, :, y, x].reshape(1, channels), dim=1)
        index = torch.argmax(torch.mm(trg_flat, src_vec.t()).reshape(-1))
        predictions.append([int(index % width), int(index // width)])
    return predictions


def cosine_nn_predict_with_diagnostics(
    src_features: torch.Tensor,
    trg_features: torch.Tensor,
    source_points: Iterable[Sequence[float]],
    *,
    nonlocal_radius: int = 8,
) -> tuple[list[list[int]], list[dict[str, Any]]]:
    """Official cosine argmax plus train-free native reliability signals."""

    if src_features.ndim != 4 or trg_features.ndim != 4:
        raise ValueError("features must have shape [B, C, H, W]")
    if src_features.shape[0] != 1 or trg_features.shape[0] != 1:
        raise ValueError("features must have batch size one")
    if src_features.shape[1] != trg_features.shape[1]:
        raise ValueError("source and target features must share channels")
    if nonlocal_radius < 0:
        raise ValueError("nonlocal_radius must be non-negative")

    source_points = list(source_points)
    _, channels, source_h, source_w = src_features.shape
    _, _, target_h, target_w = trg_features.shape
    target_flat = F.normalize(
        trg_features[0].reshape(channels, -1).t(),
        dim=1,
    )
    source_norm = torch.linalg.vector_norm(src_features[0], dim=0).clamp_min(1e-12)
    predictions: list[list[int]] = []
    diagnostics: list[dict[str, Any]] = []

    for point in source_points:
        source_x = max(0, min(source_w - 1, int(point[0])))
        source_y = max(0, min(source_h - 1, int(point[1])))
        source_vector = F.normalize(
            src_features[0, :, source_y, source_x].reshape(1, channels),
            dim=1,
        )
        similarities = torch.mm(target_flat, source_vector.t()).reshape(-1)
        top1_index = int(torch.argmax(similarities).detach().cpu())
        top1_value = similarities[top1_index]
        top1_x = int(top1_index % target_w)
        top1_y = int(top1_index // target_w)

        second_scores = similarities.clone()
        second_scores[top1_index] = -torch.inf
        top2_index = int(torch.argmax(second_scores).detach().cpu())
        top2_value = second_scores[top2_index]
        top2_x = int(top2_index % target_w)
        top2_y = int(top2_index // target_w)

        nonlocal_scores = similarities.reshape(target_h, target_w).clone()
        y0 = max(0, top1_y - nonlocal_radius)
        y1 = min(target_h, top1_y + nonlocal_radius + 1)
        x0 = max(0, top1_x - nonlocal_radius)
        x1 = min(target_w, top1_x + nonlocal_radius + 1)
        nonlocal_scores[y0:y1, x0:x1] = -torch.inf
        nonlocal_index = int(torch.argmax(nonlocal_scores.reshape(-1)).detach().cpu())
        nonlocal_value = similarities[nonlocal_index]
        nonlocal_x = int(nonlocal_index % target_w)
        nonlocal_y = int(nonlocal_index // target_w)

        target_vector = target_flat[top1_index]
        source_dot = torch.einsum("chw,c->hw", src_features[0], target_vector)
        cycle_scores = source_dot / source_norm
        cycle_index = int(torch.argmax(cycle_scores.reshape(-1)).detach().cpu())
        cycle_x = int(cycle_index % source_w)
        cycle_y = int(cycle_index // source_w)
        cycle_distance = math.hypot(cycle_x - source_x, cycle_y - source_y)

        predictions.append([top1_x, top1_y])
        diagnostics.append(
            {
                "top1_cosine": float(top1_value.detach().float().cpu()),
                "top2_cosine": float(top2_value.detach().float().cpu()),
                "top1_top2_margin": float(
                    (top1_value - top2_value).detach().float().cpu()
                ),
                "top2_prediction": [top2_x, top2_y],
                "top1_top2_pixel_distance": float(
                    math.hypot(top2_x - top1_x, top2_y - top1_y)
                ),
                "nonlocal_radius": int(nonlocal_radius),
                "top1_nonlocal_cosine": float(
                    nonlocal_value.detach().float().cpu()
                ),
                "top1_nonlocal_margin": float(
                    (top1_value - nonlocal_value).detach().float().cpu()
                ),
                "nonlocal_prediction": [nonlocal_x, nonlocal_y],
                "cycle_prediction": [cycle_x, cycle_y],
                "cycle_source_distance": float(cycle_distance),
                "reciprocal_exact": bool(
                    cycle_x == source_x and cycle_y == source_y
                ),
            }
        )
    return predictions, diagnostics


def cosine_candidate_diagnostics(
    src_features: torch.Tensor,
    trg_features: torch.Tensor,
    source_points: Iterable[Sequence[float]],
    candidate_points: Sequence[Sequence[Sequence[float]]],
) -> list[list[dict[str, Any]]]:
    """Score fixed target candidates with the unmodified native descriptor."""

    if src_features.ndim != 4 or trg_features.ndim != 4:
        raise ValueError("features must have shape [B, C, H, W]")
    if src_features.shape[0] != 1 or trg_features.shape[0] != 1:
        raise ValueError("features must have batch size one")
    if src_features.shape[1] != trg_features.shape[1]:
        raise ValueError("source and target features must share channels")

    source_points = list(source_points)
    if len(source_points) != len(candidate_points):
        raise ValueError("source point and candidate row counts must agree")
    _, channels, source_h, source_w = src_features.shape
    _, _, target_h, target_w = trg_features.shape
    rows: list[list[dict[str, Any]]] = []

    for point, candidates in zip(source_points, candidate_points):
        source_x = max(0, min(source_w - 1, int(point[0])))
        source_y = max(0, min(source_h - 1, int(point[1])))
        source_vector = F.normalize(
            src_features[0, :, source_y, source_x].reshape(1, channels),
            dim=1,
        )
        if not candidates:
            rows.append([])
            continue
        candidate_xy = [
            (
                max(0, min(target_w - 1, int(candidate[0]))),
                max(0, min(target_h - 1, int(candidate[1]))),
            )
            for candidate in candidates
        ]
        target_vectors = torch.stack(
            [trg_features[0, :, y, x] for x, y in candidate_xy],
            dim=0,
        )
        target_vectors = F.normalize(target_vectors, dim=1)
        scores = torch.mm(target_vectors, source_vector.t()).reshape(-1)
        order = torch.argsort(scores, descending=True, stable=True)
        ranks = torch.empty_like(order)
        ranks.scatter_(
            0,
            order,
            torch.arange(len(candidate_xy), device=order.device),
        )
        best = scores[order[0]]
        row = []
        for index, ((x, y), score) in enumerate(zip(candidate_xy, scores)):
            row.append(
                {
                    "pixel": [int(x), int(y)],
                    "native_cosine": float(score.detach().float().cpu()),
                    "native_candidate_rank": int(ranks[index].detach().cpu()) + 1,
                    "native_gap_to_candidate_top1": float(
                        (best - score).detach().float().cpu()
                    ),
                }
            )
        rows.append(row)
    return rows


def _rowwise_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a.float(), b.float(), dim=1, eps=1e-12).clamp(0.0, 1.0)


def _chunked_descriptor_nn_predict(
    src_features: torch.Tensor,
    trg_features: torch.Tensor,
    source_points: Iterable[Sequence[float]],
    source_size: Sequence[int],
    target_size: Sequence[int],
    *,
    channel_chunk: int = 256,
    pixel_chunk: int = 65536,
) -> list[list[int]]:
    """Full-resolution cosine NN for large descriptors without materializing all channels."""
    source_points = list(source_points)
    if not source_points:
        return []
    if src_features.ndim != 4 or trg_features.ndim != 4:
        raise ValueError("features must have shape [1, C, H, W]")
    if src_features.shape[0] != 1 or trg_features.shape[0] != 1:
        raise ValueError("features must have a batch size of one")
    if src_features.shape[1] != trg_features.shape[1]:
        raise ValueError("source and target descriptors must share channels")
    source_h, source_w = int(source_size[0]), int(source_size[1])
    target_h, target_w = int(target_size[0]), int(target_size[1])
    src_features = torch.nan_to_num(src_features.float(), nan=0.0, posinf=0.0, neginf=0.0)
    trg_features = torch.nan_to_num(trg_features.float(), nan=0.0, posinf=0.0, neginf=0.0)
    points = torch.tensor(
        [[int(point[0]), int(point[1])] for point in source_points],
        device=src_features.device,
        dtype=torch.long,
    )
    points[:, 0].clamp_(0, source_w - 1)
    points[:, 1].clamp_(0, source_h - 1)

    channels = int(src_features.shape[1])
    source_parts: list[torch.Tensor] = []
    for start in range(0, channels, channel_chunk):
        end = min(channels, start + channel_chunk)
        upsampled = F.interpolate(
            src_features[:, start:end].contiguous(),
            size=(source_h, source_w),
            mode="bilinear",
            align_corners=False,
        )
        source_parts.append(upsampled[0, :, points[:, 1], points[:, 0]].t().float())
        del upsampled
    source_vectors = torch.cat(source_parts, dim=1)
    source_norm_raw = source_vectors.square().sum(dim=1).sqrt()
    source_norm = source_norm_raw.clamp_min(1e-12)
    del source_parts

    pixel_count = target_h * target_w
    best_scores = torch.full(
        (points.shape[0],),
        -float("inf"),
        device=src_features.device,
        dtype=torch.float32,
    )
    best_indices = torch.zeros((points.shape[0],), device=src_features.device, dtype=torch.long)
    for pixel_start in range(0, pixel_count, pixel_chunk):
        pixel_end = min(pixel_count, pixel_start + pixel_chunk)
        current_pixels = pixel_end - pixel_start
        dot_products = torch.zeros(
            (points.shape[0], current_pixels),
            device=src_features.device,
            dtype=torch.float32,
        )
        target_squared_norm = torch.zeros(
            current_pixels,
            device=src_features.device,
            dtype=torch.float32,
        )
        for channel_start in range(0, channels, channel_chunk):
            channel_end = min(channels, channel_start + channel_chunk)
            upsampled = F.interpolate(
                trg_features[:, channel_start:channel_end].contiguous(),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )
            target_vectors = upsampled[0].flatten(1).t()[pixel_start:pixel_end].float()
            target_vectors = torch.nan_to_num(target_vectors, nan=0.0, posinf=0.0, neginf=0.0)
            dot_products.addmm_(
                source_vectors[:, channel_start:channel_end],
                target_vectors.t(),
            )
            target_squared_norm += target_vectors.square().sum(dim=1)
            del upsampled, target_vectors
        dot_products /= source_norm[:, None]
        dot_products /= target_squared_norm.sqrt().clamp_min_(1e-12)[None, :]
        dot_products = torch.nan_to_num(
            dot_products, nan=-float("inf"), posinf=-float("inf"), neginf=-float("inf")
        )
        chunk_scores, chunk_indices = dot_products.max(dim=1)
        update = chunk_scores > best_scores
        best_scores[update] = chunk_scores[update]
        best_indices[update] = chunk_indices[update] + pixel_start
        del dot_products, target_squared_norm

    predictions = [
        [int(index % target_w), int(index // target_w)]
        for index in best_indices.detach().cpu().tolist()
    ]
    return predictions


def _chunked_descriptor_topk_indices(
    src_features: torch.Tensor,
    trg_features: torch.Tensor,
    source_points: Iterable[Sequence[float]],
    source_size: Sequence[int],
    target_size: Sequence[int],
    *,
    topk: int,
    channel_chunk: int = 256,
    pixel_chunk: int = 65536,
) -> torch.Tensor:
    """Return full-resolution target pixel top-k indices for descriptor maps."""

    source_points = list(source_points)
    if not source_points:
        return torch.empty((0, 0), device=src_features.device, dtype=torch.long)
    if src_features.ndim != 4 or trg_features.ndim != 4:
        raise ValueError("features must have shape [1, C, H, W]")
    if src_features.shape[0] != 1 or trg_features.shape[0] != 1:
        raise ValueError("features must have a batch size of one")
    if src_features.shape[1] != trg_features.shape[1]:
        raise ValueError("source and target descriptors must share channels")

    source_h, source_w = int(source_size[0]), int(source_size[1])
    target_h, target_w = int(target_size[0]), int(target_size[1])
    src_features = torch.nan_to_num(src_features.float(), nan=0.0, posinf=0.0, neginf=0.0)
    trg_features = torch.nan_to_num(trg_features.float(), nan=0.0, posinf=0.0, neginf=0.0)
    points = torch.tensor(
        [[int(point[0]), int(point[1])] for point in source_points],
        device=src_features.device,
        dtype=torch.long,
    )
    points[:, 0].clamp_(0, source_w - 1)
    points[:, 1].clamp_(0, source_h - 1)

    channels = int(src_features.shape[1])
    source_parts: list[torch.Tensor] = []
    for start in range(0, channels, channel_chunk):
        end = min(channels, start + channel_chunk)
        upsampled = F.interpolate(
            src_features[:, start:end].contiguous(),
            size=(source_h, source_w),
            mode="bilinear",
            align_corners=False,
        )
        source_parts.append(upsampled[0, :, points[:, 1], points[:, 0]].t().float())
        del upsampled
    source_vectors = torch.cat(source_parts, dim=1)
    source_norm = source_vectors.square().sum(dim=1).sqrt().clamp_min(1e-12)
    del source_parts

    pixel_count = target_h * target_w
    k = min(max(1, int(topk)), pixel_count)
    best_scores = torch.full(
        (points.shape[0], 0),
        -float("inf"),
        device=src_features.device,
        dtype=torch.float32,
    )
    best_indices = torch.empty((points.shape[0], 0), device=src_features.device, dtype=torch.long)
    for pixel_start in range(0, pixel_count, pixel_chunk):
        pixel_end = min(pixel_count, pixel_start + pixel_chunk)
        current_pixels = pixel_end - pixel_start
        dot_products = torch.zeros(
            (points.shape[0], current_pixels),
            device=src_features.device,
            dtype=torch.float32,
        )
        target_squared_norm = torch.zeros(
            current_pixels,
            device=src_features.device,
            dtype=torch.float32,
        )
        for channel_start in range(0, channels, channel_chunk):
            channel_end = min(channels, channel_start + channel_chunk)
            upsampled = F.interpolate(
                trg_features[:, channel_start:channel_end].contiguous(),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )
            target_vectors = upsampled[0].flatten(1).t()[pixel_start:pixel_end].float()
            target_vectors = torch.nan_to_num(target_vectors, nan=0.0, posinf=0.0, neginf=0.0)
            dot_products.addmm_(
                source_vectors[:, channel_start:channel_end],
                target_vectors.t(),
            )
            target_squared_norm += target_vectors.square().sum(dim=1)
            del upsampled, target_vectors
        dot_products /= source_norm[:, None]
        dot_products /= target_squared_norm.sqrt().clamp_min_(1e-12)[None, :]
        dot_products = torch.nan_to_num(
            dot_products, nan=-float("inf"), posinf=-float("inf"), neginf=-float("inf")
        )
        chunk_k = min(k, current_pixels)
        chunk_scores, chunk_indices = torch.topk(dot_products, k=chunk_k, dim=1, sorted=True)
        merged_scores = torch.cat((best_scores, chunk_scores), dim=1)
        merged_indices = torch.cat((best_indices, chunk_indices + pixel_start), dim=1)
        keep_scores, keep_positions = torch.topk(
            merged_scores,
            k=min(k, merged_scores.shape[1]),
            dim=1,
            sorted=True,
        )
        best_scores = keep_scores
        best_indices = torch.gather(merged_indices, 1, keep_positions)
        del dot_products, target_squared_norm, chunk_scores, chunk_indices, merged_scores, merged_indices
    return best_indices


def _chunked_descriptor_topk_scores_indices(
    src_features: torch.Tensor,
    trg_features: torch.Tensor,
    source_points: Iterable[Sequence[float]],
    source_size: Sequence[int],
    target_size: Sequence[int],
    *,
    topk: int,
    candidate_indices: torch.Tensor | None = None,
    channel_chunk: int = 256,
    pixel_chunk: int = 65536,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return descriptor cosine top-k scores and target pixel indices.

    When ``candidate_indices`` is supplied, scores are computed only for that
    per-query proposal set.  This lets cross-attention provide high-recall
    candidates while frozen descriptors perform semantic verification.
    """

    source_points = list(source_points)
    if not source_points:
        empty_scores = torch.empty((0, 0), device=src_features.device, dtype=torch.float32)
        empty_indices = torch.empty((0, 0), device=src_features.device, dtype=torch.long)
        return empty_scores, empty_indices
    if src_features.ndim != 4 or trg_features.ndim != 4:
        raise ValueError("features must have shape [1, C, H, W]")
    if src_features.shape[0] != 1 or trg_features.shape[0] != 1:
        raise ValueError("features must have a batch size of one")
    if src_features.shape[1] != trg_features.shape[1]:
        raise ValueError("source and target descriptors must share channels")

    source_h, source_w = int(source_size[0]), int(source_size[1])
    target_h, target_w = int(target_size[0]), int(target_size[1])
    src_features = torch.nan_to_num(src_features.float(), nan=0.0, posinf=0.0, neginf=0.0)
    trg_features = torch.nan_to_num(trg_features.float(), nan=0.0, posinf=0.0, neginf=0.0)
    points = torch.tensor(
        [[int(point[0]), int(point[1])] for point in source_points],
        device=src_features.device,
        dtype=torch.long,
    )
    points[:, 0].clamp_(0, source_w - 1)
    points[:, 1].clamp_(0, source_h - 1)

    channels = int(src_features.shape[1])
    source_parts: list[torch.Tensor] = []
    for start in range(0, channels, channel_chunk):
        end = min(channels, start + channel_chunk)
        upsampled = F.interpolate(
            src_features[:, start:end].contiguous(),
            size=(source_h, source_w),
            mode="bilinear",
            align_corners=False,
        )
        source_parts.append(upsampled[0, :, points[:, 1], points[:, 0]].t().float())
        del upsampled
    source_vectors = torch.cat(source_parts, dim=1)
    source_norm = source_vectors.square().sum(dim=1).sqrt().clamp_min(1e-12)
    del source_parts

    if candidate_indices is not None:
        if candidate_indices.ndim != 2 or candidate_indices.shape[0] != points.shape[0]:
            raise ValueError("candidate_indices must be [point_count, topk]")
        proposals = candidate_indices.to(device=src_features.device, dtype=torch.long)
        proposals = proposals.clamp(0, target_h * target_w - 1)
        query_count, proposal_count = proposals.shape
        scores = torch.zeros((query_count, proposal_count), device=src_features.device, dtype=torch.float32)
        target_squared_norm = torch.zeros((query_count, proposal_count), device=src_features.device, dtype=torch.float32)
        flat_positions = proposals.reshape(-1)
        for channel_start in range(0, channels, channel_chunk):
            channel_end = min(channels, channel_start + channel_chunk)
            upsampled = F.interpolate(
                trg_features[:, channel_start:channel_end].contiguous(),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )
            target_vectors = upsampled[0].flatten(1).t()[flat_positions].float().reshape(
                query_count,
                proposal_count,
                channel_end - channel_start,
            )
            target_vectors = torch.nan_to_num(target_vectors, nan=0.0, posinf=0.0, neginf=0.0)
            source_chunk = source_vectors[:, channel_start:channel_end]
            scores += (target_vectors * source_chunk[:, None, :]).sum(dim=2)
            target_squared_norm += target_vectors.square().sum(dim=2)
            del upsampled, target_vectors
        scores /= source_norm[:, None]
        scores /= target_squared_norm.sqrt().clamp_min_(1e-12)
        scores = torch.nan_to_num(scores, nan=-float("inf"), posinf=-float("inf"), neginf=-float("inf"))
        keep_k = min(max(1, int(topk)), proposal_count)
        sorted_scores, order = torch.topk(scores, k=keep_k, dim=1, sorted=True)
        sorted_indices = torch.gather(proposals, 1, order)
        return sorted_scores, sorted_indices

    pixel_count = target_h * target_w
    k = min(max(1, int(topk)), pixel_count)
    best_scores = torch.full(
        (points.shape[0], 0),
        -float("inf"),
        device=src_features.device,
        dtype=torch.float32,
    )
    best_indices = torch.empty((points.shape[0], 0), device=src_features.device, dtype=torch.long)
    for pixel_start in range(0, pixel_count, pixel_chunk):
        pixel_end = min(pixel_count, pixel_start + pixel_chunk)
        current_pixels = pixel_end - pixel_start
        dot_products = torch.zeros(
            (points.shape[0], current_pixels),
            device=src_features.device,
            dtype=torch.float32,
        )
        target_squared_norm = torch.zeros(current_pixels, device=src_features.device, dtype=torch.float32)
        for channel_start in range(0, channels, channel_chunk):
            channel_end = min(channels, channel_start + channel_chunk)
            upsampled = F.interpolate(
                trg_features[:, channel_start:channel_end].contiguous(),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )
            target_vectors = upsampled[0].flatten(1).t()[pixel_start:pixel_end].float()
            target_vectors = torch.nan_to_num(target_vectors, nan=0.0, posinf=0.0, neginf=0.0)
            dot_products.addmm_(source_vectors[:, channel_start:channel_end], target_vectors.t())
            target_squared_norm += target_vectors.square().sum(dim=1)
            del upsampled, target_vectors
        dot_products /= source_norm[:, None]
        dot_products /= target_squared_norm.sqrt().clamp_min_(1e-12)[None, :]
        dot_products = torch.nan_to_num(
            dot_products, nan=-float("inf"), posinf=-float("inf"), neginf=-float("inf")
        )
        chunk_k = min(k, current_pixels)
        chunk_scores, chunk_indices = torch.topk(dot_products, k=chunk_k, dim=1, sorted=True)
        merged_scores = torch.cat((best_scores, chunk_scores), dim=1)
        merged_indices = torch.cat((best_indices, chunk_indices + pixel_start), dim=1)
        keep_scores, keep_positions = torch.topk(
            merged_scores,
            k=min(k, merged_scores.shape[1]),
            dim=1,
            sorted=True,
        )
        best_scores = keep_scores
        best_indices = torch.gather(merged_indices, 1, keep_positions)
        del dot_products, target_squared_norm, chunk_scores, chunk_indices, merged_scores, merged_indices
    return best_scores, best_indices


def _topk_hit_counts(
    indices: torch.Tensor,
    target_points: Iterable[Sequence[float]],
    threshold: float,
    target_size: Sequence[int],
    topks: Sequence[int],
) -> dict[int, int]:
    """Count per-source GT hits inside top-k full-resolution candidate indices."""

    targets = torch.tensor(
        [[float(point[0]), float(point[1])] for point in target_points],
        device=indices.device,
        dtype=torch.float32,
    )
    target_h, target_w = int(target_size[0]), int(target_size[1])
    if indices.numel() == 0 or targets.numel() == 0:
        return {int(k): 0 for k in topks}
    x = (indices % target_w).float()
    y = torch.div(indices, target_w, rounding_mode="floor").float()
    coords = torch.stack((x, y), dim=-1)
    hits = torch.linalg.vector_norm(coords - targets[:, None, :], dim=2) <= 0.1 * float(threshold)
    return {
        int(k): int(hits[:, : min(int(k), hits.shape[1])].any(dim=1).sum().detach().cpu())
        for k in topks
    }


def _fjsar_attention_candidate_records(
    src_features: torch.Tensor,
    trg_features: torch.Tensor,
    attention: dict[str, torch.Tensor],
    source_points: Sequence[Sequence[float]],
    source_size: Sequence[int],
    target_size: Sequence[int],
    src_state: FluxReplayState,
    trg_state: FluxReplayState,
    *,
    topk: int,
    target_points: Sequence[Sequence[float]] | None = None,
    pck_threshold: float | None = None,
    candidate_descriptor_audit: bool = False,
    method_descriptor_audit_name: str | None = None,
    method_descriptor_src: torch.Tensor | None = None,
    method_descriptor_trg: torch.Tensor | None = None,
    transport_lift_branch_descriptors: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
    attention_flow_audit: bool = False,
    attention_flow_radius: int = 2,
    attention_kernel_audit: bool = False,
    attention_kernel_radius: int = 2,
    attention_kernel_topk: Sequence[int] = (1, 5, 20),
    basin_identity_audit: bool = False,
    basin_identity_topk: int = 20,
    basin_identity_radius: int = 2,
    basin_identity_rank_topk: Sequence[int] = (1, 3, 5, 10, 20),
    kernel_featureization_audit: bool = False,
    kernel_featureization_ranks: Sequence[int] = (32, 64),
    kernel_featureization_weights: Sequence[float] = (0.5, 1.0),
    kernel_featureization_radius: int = 2,
    kernel_featureization_topk: Sequence[int] = (1, 5, 20),
    residual_readout_audit: bool = False,
    residual_readout_topk: Sequence[int] = (1, 3, 5, 10, 20),
    latent_expert_audit: bool = False,
    latent_expert_topk: Sequence[int] = (1, 3, 5, 10, 20),
    candidate_clamped_causal_replay_audit: bool = False,
    candidate_clamped_causal_replay_topk: Sequence[int] = (1, 3, 5, 10, 20),
    causal_release_block: Any | None = None,
    counterfactual_fingerprint_audit: bool = False,
    counterfactual_fingerprint_topk: Sequence[int] = (1, 3, 5, 10, 20),
    counterfactual_fingerprint_scales: Sequence[float] = (0.75, 1.0, 1.25),
    persistent_candidate_slot_replay_audit: bool = False,
    persistent_candidate_slot_replay_topk: Sequence[int] = (1, 3, 5, 10, 20),
    persistent_candidate_slot_replay_chunk: int = 1,
    persistent_candidate_slot_replay_blocks: Sequence[Any] | None = None,
    local_relational_identity_audit: bool = False,
    local_relational_radius: int = 2,
    dense_candidate_edge_audit: bool = False,
    dense_candidate_edge_radius: int = 1,
    dense_transport_consistency_audit: bool = False,
    dense_transport_topk: Sequence[int] = (1, 5, 20),
    candidate_field_consistency_audit: bool = False,
    candidate_field_topm: int = 20,
    candidate_field_source: str = "native_basin",
    anchor_topology_audit: bool = False,
    multilayer_identity_audit: bool = False,
    multilayer_descriptor_maps: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
    blocks: Sequence[Any] | None = None,
    interaction_mode: str = "exact",
    use_coordinate_bias: bool = False,
    transport_factorization_audit: bool = False,
    transport_factorization_radius: int = 2,
    transport_factorization_basis_radius: int = 0,
    operator_manifold_audits: Sequence[dict[str, Any]] | None = None,
    trajectory_identity_audits: Sequence[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return lightweight per-keypoint attention proposal diagnostics."""

    points = list(source_points)
    if not points:
        return []
    device = src_features.device
    src_cells = _native_cell_indices_for_points(
        points,
        source_size,
        src_state.image_height,
        src_state.image_width,
        attention["p_ab"].device,
    )
    mutual_attention = torch.sqrt((attention["p_ab"].float() * attention["p_ba"].float().t()).clamp_min(0.0))
    mutual_attention = torch.nan_to_num(mutual_attention, nan=0.0, posinf=0.0, neginf=0.0)
    candidate_count = min(max(1, int(topk)), mutual_attention.shape[1])
    attn_values, candidate_cells = torch.topk(
        mutual_attention[src_cells],
        k=candidate_count,
        dim=1,
        sorted=True,
    )
    target_h, target_w = int(target_size[0]), int(target_size[1])
    cell_x = (candidate_cells % trg_state.image_width).float()
    cell_y = torch.div(candidate_cells, trg_state.image_width, rounding_mode="floor").float()
    proposal_x = torch.round((cell_x + 0.5) * float(target_w) / float(trg_state.image_width) - 0.5).long()
    proposal_y = torch.round((cell_y + 0.5) * float(target_h) / float(trg_state.image_height) - 0.5).long()
    proposal_x.clamp_(0, target_w - 1)
    proposal_y.clamp_(0, target_h - 1)
    proposal_pixels = (proposal_y.to(device) * target_w + proposal_x.to(device)).long()

    native_scores, native_indices = _chunked_descriptor_topk_scores_indices(
        src_features,
        trg_features,
        points,
        source_size,
        target_size,
        topk=max(2, candidate_count),
    )
    proposal_scores, proposal_indices = _chunked_descriptor_topk_scores_indices(
        src_features,
        trg_features,
        points,
        source_size,
        target_size,
        topk=candidate_count,
        candidate_indices=proposal_pixels,
    )
    gt_pixels = None
    diagnostic_scores = diagnostic_indices = None
    if target_points is not None:
        gt_tensor = torch.tensor(
            [[float(point[0]), float(point[1])] for point in target_points],
            device=device,
            dtype=torch.float32,
        )
        gt_x = torch.round(gt_tensor[:, 0]).long().clamp_(0, target_w - 1)
        gt_y = torch.round(gt_tensor[:, 1]).long().clamp_(0, target_h - 1)
        gt_pixels = (gt_y * target_w + gt_x).long()
        diagnostic_candidates = torch.stack((proposal_pixels[:, 0], gt_pixels), dim=1)
        diagnostic_scores, diagnostic_indices = _chunked_descriptor_topk_scores_indices(
            src_features,
            trg_features,
            points,
            source_size,
            target_size,
            topk=2,
            candidate_indices=diagnostic_candidates,
        )
    cond_ab, _mass_ab = _conditional_cross_distribution(attention["p_ab"])
    cond_ba, _mass_ba = _conditional_cross_distribution(attention["p_ba"])
    kernel_audits = None
    if attention_kernel_audit:
        kernel_audits = _attention_kernel_audit_for_points(
            mutual_attention,
            src_cells,
            target_size,
            src_state,
            trg_state,
            target_points=target_points,
            pck_threshold=pck_threshold,
            topks=attention_kernel_topk,
            radius=attention_kernel_radius,
        )
    basin_audits = None
    if basin_identity_audit:
        basin_audits = _basin_identity_audit_for_points(
            src_features,
            trg_features,
            mutual_attention,
            src_cells,
            points,
            source_size,
            target_size,
            src_state,
            trg_state,
            target_points=target_points,
            pck_threshold=pck_threshold,
            basin_topk=basin_identity_topk,
            radius=basin_identity_radius,
            rank_topks=basin_identity_rank_topk,
        )
    featureization_audits = None
    if kernel_featureization_audit:
        featureization_audits = _kernel_featureization_audit_for_points(
            src_features,
            trg_features,
            mutual_attention,
            src_cells,
            points,
            source_size,
            target_size,
            src_state,
            trg_state,
            target_points=target_points,
            pck_threshold=pck_threshold,
            topks=kernel_featureization_topk,
            ranks=kernel_featureization_ranks,
            weights=kernel_featureization_weights,
            radius=kernel_featureization_radius,
        )
    readout_probe = None
    if residual_readout_audit or latent_expert_audit:
        if blocks is None:
            raise ValueError("cross-readout audits require replay blocks")
        readout_probe = flux_cross_readout_probe(
            blocks,
            src_state,
            trg_state,
            src_cells,
            candidate_cells,
            mode=interaction_mode,
            use_coordinate_bias=use_coordinate_bias,
        )
    residual_readout_audits = None
    if residual_readout_audit:
        residual_readout_audits = _residual_readout_audit_for_points(
            readout_probe,
            proposal_pixels,
            target_size,
            target_points=target_points,
            pck_threshold=pck_threshold,
            topks=residual_readout_topk,
        )
    latent_expert_audits = None
    if latent_expert_audit:
        latent_expert_audits = _latent_expert_audit_for_points(
            readout_probe,
            proposal_pixels,
            target_size,
            aggregated_attention_scores=attn_values,
            target_points=target_points,
            pck_threshold=pck_threshold,
            topks=latent_expert_topk,
        )
    causal_replay_audits = None
    if candidate_clamped_causal_replay_audit:
        if blocks is None or len(blocks) != 1 or causal_release_block is None:
            raise ValueError(
                "candidate-clamped causal replay requires one clamp block and "
                "the adjacent release block"
            )
        causal_probe = flux_candidate_clamped_causal_probe(
            blocks[0],
            causal_release_block,
            src_state,
            trg_state,
            src_cells,
            candidate_cells,
        )
        causal_replay_audits = _candidate_clamped_causal_replay_audit_for_points(
            causal_probe,
            proposal_pixels,
            target_size,
            target_points=target_points,
            pck_threshold=pck_threshold,
            topks=candidate_clamped_causal_replay_topk,
        )
    fingerprint_audits = None
    if counterfactual_fingerprint_audit:
        if blocks is None or len(blocks) != 1 or causal_release_block is None:
            raise ValueError(
                "counterfactual fingerprint audit requires one clamp block and "
                "the adjacent release block"
            )
        fingerprint_probe = flux_candidate_counterfactual_fingerprint_probe(
            blocks[0],
            causal_release_block,
            src_state,
            trg_state,
            src_cells,
            candidate_cells,
            intervention_scales=counterfactual_fingerprint_scales,
        )
        fingerprint_audits = _counterfactual_fingerprint_audit_for_points(
            fingerprint_probe,
            proposal_pixels,
            target_size,
            target_points=target_points,
            pck_threshold=pck_threshold,
            topks=counterfactual_fingerprint_topk,
        )
    persistent_slot_audits = None
    if persistent_candidate_slot_replay_audit:
        persistent_blocks = persistent_candidate_slot_replay_blocks or blocks
        if persistent_blocks is None or len(persistent_blocks) not in (1, 2):
            raise ValueError("persistent candidate-slot replay requires one or two replay blocks")
        persistent_probe = flux_persistent_candidate_slot_replay_probe(
            persistent_blocks,
            src_state,
            trg_state,
            src_cells,
            candidate_cells,
            hypothesis_chunk=persistent_candidate_slot_replay_chunk,
        )
        persistent_slot_audits = _persistent_candidate_slot_replay_audit_for_points(
            persistent_probe,
            proposal_pixels,
            target_size,
            attention_scores=attn_values,
            target_points=target_points,
            pck_threshold=pck_threshold,
            topks=persistent_candidate_slot_replay_topk,
        )
    local_relational_audits = None
    if local_relational_identity_audit:
        local_relational_audits = _local_relational_identity_audit_for_points(
            src_features,
            trg_features,
            mutual_attention,
            src_cells,
            points,
            proposal_pixels,
            source_size,
            target_size,
            src_state,
            trg_state,
            target_points=target_points,
            pck_threshold=pck_threshold,
            radius=local_relational_radius,
        )
    dense_candidate_edge_audits = None
    if dense_candidate_edge_audit:
        dense_candidate_edge_audits = _dense_candidate_edge_separability_audit_for_points(
            src_features,
            trg_features,
            mutual_attention,
            src_cells,
            candidate_cells,
            proposal_pixels,
            target_size,
            src_state,
            trg_state,
            target_points=target_points,
            pck_threshold=pck_threshold,
            edge_radius=dense_candidate_edge_radius,
        )
    dense_transport_audits = None
    if dense_transport_consistency_audit:
        dense_transport_audits = _dense_transport_consistency_audit_for_points(
            mutual_attention,
            src_cells,
            proposal_pixels,
            target_size,
            src_state,
            trg_state,
            target_points=target_points,
            pck_threshold=pck_threshold,
            topks=dense_transport_topk,
        )
    candidate_field_audits = None
    if candidate_field_consistency_audit:
        candidate_field_audits = _candidate_field_consistency_audit_for_points(
            src_features,
            trg_features,
            mutual_attention,
            src_cells,
            points,
            source_size,
            target_size,
            trg_state,
            target_points=target_points,
            pck_threshold=pck_threshold,
            candidate_topk=candidate_count,
            field_topm=candidate_field_topm,
            candidate_source=candidate_field_source,
        )
    anchor_topology_audits = None
    if anchor_topology_audit:
        anchor_topology_audits = _anchor_topology_audit_for_points(
            points,
            source_size,
            target_size,
            proposal_pixels,
            native_indices,
            native_scores,
            target_points=target_points,
            pck_threshold=pck_threshold,
        )
    multilayer_identity_audits = None
    if multilayer_identity_audit:
        if not multilayer_descriptor_maps:
            raise ValueError("multilayer_identity_audit requires multilayer descriptor maps")
        multilayer_identity_audits = _multilayer_identity_audit_for_points(
            multilayer_descriptor_maps,
            points,
            proposal_pixels,
            source_size,
            target_size,
            target_points=target_points,
            pck_threshold=pck_threshold,
            gt_pixels=gt_pixels,
        )
    rows: list[dict[str, Any]] = []
    for row in range(len(points)):
        src_cell = int(src_cells[row])
        native_pixel = int(native_indices[row, 0])
        native_cell_x = int((float(native_pixel % target_w) + 0.5) * float(trg_state.image_width) / float(target_w))
        native_cell_y = int((float(native_pixel // target_w) + 0.5) * float(trg_state.image_height) / float(target_h))
        native_cell_x = max(0, min(trg_state.image_width - 1, native_cell_x))
        native_cell_y = max(0, min(trg_state.image_height - 1, native_cell_y))
        native_cell = native_cell_y * trg_state.image_width + native_cell_x
        native_best = float(native_scores[row, 0].detach().cpu())
        native_second = float(native_scores[row, 1].detach().cpu()) if native_scores.shape[1] > 1 else native_best
        native_reciprocal = float(
            torch.sqrt((cond_ab[src_cell, native_cell] * cond_ba[native_cell, src_cell]).clamp_min(0.0))
            .detach()
            .cpu()
        )
        target = target_points[row] if target_points is not None else None
        native_xy = [int(native_pixel % target_w), int(native_pixel // target_w)]
        native_hit = (
            bool(_point_hit(native_xy, target, pck_threshold))
            if target is not None and pck_threshold is not None
            else None
        )
        attention_pixels = proposal_pixels[row].detach().cpu().tolist()
        attention_scores = [float(value) for value in attn_values[row].detach().cpu().tolist()]
        attention_top1_pixel = int(attention_pixels[0])
        attention_top1_xy = [int(attention_top1_pixel % target_w), int(attention_top1_pixel // target_w)]
        attention_top1_hit = (
            bool(_point_hit(attention_top1_xy, target, pck_threshold))
            if target is not None and pck_threshold is not None
            else None
        )
        attention_top1_distance = None
        attention_top1_distance_over_threshold = None
        if target is not None:
            dx = float(attention_top1_xy[0]) - float(target[0])
            dy = float(attention_top1_xy[1]) - float(target[1])
            attention_top1_distance = float((dx * dx + dy * dy) ** 0.5)
            if pck_threshold is not None:
                attention_top1_distance_over_threshold = float(attention_top1_distance / float(pck_threshold))
        attention_margin = (
            float(attention_scores[0] - attention_scores[1])
            if len(attention_scores) > 1
            else float(attention_scores[0])
        )
        attention_distribution = mutual_attention[src_cell]
        attention_concentration = float(
            _attention_concentration_safe(attention_distribution.reshape(1, -1))[0]
            .detach()
            .cpu()
        )
        diagnostic_score_by_pixel: dict[int, float] = {}
        if diagnostic_scores is not None and diagnostic_indices is not None:
            diagnostic_score_by_pixel = {
                int(pixel): float(score)
                for pixel, score in zip(
                    diagnostic_indices[row].detach().cpu().tolist(),
                    diagnostic_scores[row].detach().cpu().tolist(),
                )
            }
        attention_top1_native_score = diagnostic_score_by_pixel.get(attention_top1_pixel)
        gt_pixel = int(gt_pixels[row].detach().cpu()) if gt_pixels is not None else None
        gt_native_score = diagnostic_score_by_pixel.get(gt_pixel) if gt_pixel is not None else None
        native_score_gap_attention_top1_minus_gt = (
            float(attention_top1_native_score - gt_native_score)
            if attention_top1_native_score is not None and gt_native_score is not None
            else None
        )
        proposals: list[dict[str, Any]] = []
        gt_ranks = {
            "attention": None,
            "descriptor": None,
            "reciprocal": None,
            "fused": None,
            "trajectory": None,
        }
        reciprocal_scores: list[float] = []
        fused_scores: list[float] = []
        attention_lookup: dict[int, tuple[int, float]] = {}
        for attn_rank, (attn_pixel, attn_score) in enumerate(
            zip(attention_pixels, attention_scores),
            start=1,
        ):
            attention_lookup.setdefault(int(attn_pixel), (int(attn_rank), float(attn_score)))
        descriptor_audit = None
        if candidate_descriptor_audit:
            descriptor_audit = _candidate_descriptor_audit_for_point(
                src_features,
                trg_features,
                attention,
                points[row],
                source_size,
                target_size,
                src_state,
                trg_state,
                src_cell,
                proposal_pixels[row],
                target=target,
                pck_threshold=pck_threshold,
                gt_pixel=gt_pixels[row] if gt_pixels is not None else None,
            )
        method_descriptor_audit = None
        if method_descriptor_src is not None and method_descriptor_trg is not None:
            method_descriptor_audit = _method_descriptor_audit_for_point(
                method_descriptor_src,
                method_descriptor_trg,
                str(method_descriptor_audit_name or "method"),
                points[row],
                source_size,
                target_size,
                proposal_pixels[row],
                target=target,
                pck_threshold=pck_threshold,
                gt_pixel=gt_pixels[row] if gt_pixels is not None else None,
            )
        transport_lift_branch_audit = None
        if transport_lift_branch_descriptors is not None:
            transport_lift_branch_audit = _transport_lift_branch_audit_for_point(
                transport_lift_branch_descriptors,
                points[row],
                source_size,
                target_size,
                proposal_pixels[row],
                target=target,
                pck_threshold=pck_threshold,
                gt_pixel=gt_pixels[row] if gt_pixels is not None else None,
            )
        flow_audit = None
        if attention_flow_audit:
            flow_audit = _attention_flow_patch_audit_for_point(
                mutual_attention,
                src_cell,
                proposal_pixels[row],
                source_size,
                target_size,
                src_state,
                trg_state,
                target=target,
                pck_threshold=pck_threshold,
                radius=attention_flow_radius,
            )
        kernel_audit = kernel_audits[row] if kernel_audits is not None else None
        basin_audit = basin_audits[row] if basin_audits is not None else None
        featureization_audit = featureization_audits[row] if featureization_audits is not None else None
        residual_readout_row = (
            residual_readout_audits[row]
            if residual_readout_audits is not None
            else None
        )
        latent_expert_row = (
            latent_expert_audits[row]
            if latent_expert_audits is not None
            else None
        )
        causal_replay_row = (
            causal_replay_audits[row]
            if causal_replay_audits is not None
            else None
        )
        fingerprint_row = (
            fingerprint_audits[row]
            if fingerprint_audits is not None
            else None
        )
        local_relational_row = (
            local_relational_audits[row]
            if local_relational_audits is not None
            else None
        )
        dense_candidate_edge_row = (
            dense_candidate_edge_audits[row]
            if dense_candidate_edge_audits is not None
            else None
        )
        dense_transport_row = (
            dense_transport_audits[row]
            if dense_transport_audits is not None
            else None
        )
        candidate_field_row = (
            candidate_field_audits[row]
            if candidate_field_audits is not None
            else None
        )
        anchor_topology_row = (
            anchor_topology_audits[row]
            if anchor_topology_audits is not None
            else None
        )
        multilayer_identity_row = (
            multilayer_identity_audits[row]
            if multilayer_identity_audits is not None
            else None
        )
        operator_manifold_audit = (
            operator_manifold_audits[row]
            if operator_manifold_audits is not None
            else None
        )
        trajectory_identity_audit = (
            trajectory_identity_audits[row]
            if trajectory_identity_audits is not None and row < len(trajectory_identity_audits)
            else None
        )
        if isinstance(trajectory_identity_audit, dict):
            trajectory_rank = trajectory_identity_audit.get("ranks", {}).get("trajectory")
            if trajectory_rank is not None:
                gt_ranks["trajectory"] = int(trajectory_rank)
        factorization_audit = None
        if transport_factorization_audit:
            factorization_audit = _transport_factorization_audit_for_point(
                mutual_attention,
                src_cell,
                proposal_pixels[row],
                target_size,
                src_state,
                trg_state,
                target=target,
                pck_threshold=pck_threshold,
                radius=transport_factorization_radius,
                basis_radius=transport_factorization_basis_radius,
            )
        for rank in range(proposal_indices.shape[1]):
            pixel = int(proposal_indices[row, rank])
            cell = int(
                torch.div(proposal_indices[row, rank], target_w, rounding_mode="floor")
                .mul(float(trg_state.image_height) / float(target_h))
                .floor()
                .clamp(0, trg_state.image_height - 1)
                .item()
            ) * trg_state.image_width + int(
                ((proposal_indices[row, rank] % target_w).float() * float(trg_state.image_width) / float(target_w))
                .floor()
                .clamp(0, trg_state.image_width - 1)
                .item()
            )
            reciprocal = float(
                torch.sqrt((cond_ab[src_cell, cell] * cond_ba[cell, src_cell]).clamp_min(0.0))
                .detach()
                .cpu()
            )
            reciprocal_scores.append(reciprocal)
        def _normalize_list(values: list[float]) -> list[float]:
            if not values:
                return []
            lo = min(values)
            hi = max(values)
            denom = max(hi - lo, 1e-6)
            return [(value - lo) / denom for value in values]
        descriptor_scores = [float(value) for value in proposal_scores[row].detach().cpu().tolist()]
        norm_desc = _normalize_list(descriptor_scores)
        norm_recip = _normalize_list(reciprocal_scores)
        for rank in range(proposal_indices.shape[1]):
            pixel = int(proposal_indices[row, rank])
            xy = [int(pixel % target_w), int(pixel // target_w)]
            descriptor_score = descriptor_scores[rank]
            reciprocal = reciprocal_scores[rank]
            semantic_gap = descriptor_score - native_best
            fused = norm_desc[rank] + norm_recip[rank] + semantic_gap
            fused_scores.append(fused)
            hit = (
                bool(_point_hit(xy, target, pck_threshold))
                if target is not None and pck_threshold is not None
                else None
            )
            proposal = {
                "rank_descriptor": int(rank + 1),
                "rank_attention": (
                    int(attention_lookup[pixel][0])
                    if pixel in attention_lookup
                    else None
                ),
                "pixel": xy,
                "pixel_index": int(pixel),
                "attention_score": (
                    float(attention_lookup[pixel][1])
                    if pixel in attention_lookup
                    else 0.0
                ),
                "descriptor_score": float(descriptor_score),
                "semantic_gap_to_native": float(semantic_gap),
                "reciprocal_attention": float(reciprocal),
                "fused_score": float(fused),
                "pck_hit": hit,
            }
            proposals.append(proposal)
        # Attention rank is based on the mutual-attention top-k order before
        # descriptor sorting; descriptor/fused/reciprocal ranks use their own scores.
        for rank, pixel in enumerate(attention_pixels):
            xy = [int(pixel % target_w), int(pixel // target_w)]
            if target is not None and pck_threshold is not None and _point_hit(xy, target, pck_threshold):
                gt_ranks["attention"] = rank + 1
                break
        for name, values in (
            ("descriptor", descriptor_scores),
            ("reciprocal", reciprocal_scores),
            ("fused", fused_scores),
        ):
            order = sorted(range(len(values)), key=lambda index: values[index], reverse=True)
            for rank, index in enumerate(order):
                hit = proposals[index]["pck_hit"]
                if hit:
                    gt_ranks[name] = rank + 1
                    break
        rows.append({
            "source_point": [float(points[row][0]), float(points[row][1])],
            "target_point": [float(target[0]), float(target[1])] if target is not None else None,
            "source_cell": int(src_cell),
            "native": {
                "pixel": native_xy,
                "pixel_index": int(native_pixel),
                "score_top1": native_best,
                "score_top2": native_second,
                "margin": float(native_best - native_second),
                "reciprocal_attention": native_reciprocal,
                "pck_hit": native_hit,
            },
            "attention_top1": {
                "pixel": attention_top1_xy,
                "pixel_index": int(attention_top1_pixel),
                "score": float(attention_scores[0]),
                "score_top2": float(attention_scores[1]) if len(attention_scores) > 1 else float(attention_scores[0]),
                "margin": attention_margin,
                "concentration": attention_concentration,
                "native_descriptor_score": attention_top1_native_score,
                "pck_hit": attention_top1_hit,
                "distance_to_gt": attention_top1_distance,
                "distance_over_threshold": attention_top1_distance_over_threshold,
            },
            "gt_pixel_index": gt_pixel,
            "gt_native_descriptor_score": gt_native_score,
            "native_score_gap_attention_top1_minus_gt": native_score_gap_attention_top1_minus_gt,
            "gt_ranks": gt_ranks,
            "candidate_descriptor_audit": descriptor_audit,
            "method_descriptor_audit": method_descriptor_audit,
            "transport_lift_branch_audit": transport_lift_branch_audit,
            "attention_flow_audit": flow_audit,
            "attention_kernel_audit": kernel_audit,
            "basin_identity_audit": basin_audit,
            "kernel_featureization_audit": featureization_audit,
            "residual_readout_audit": residual_readout_row,
            "latent_expert_audit": latent_expert_row,
            "candidate_clamped_causal_replay_audit": causal_replay_row,
            "counterfactual_fingerprint_audit": fingerprint_row,
            "persistent_candidate_slot_replay_audit": (
                persistent_slot_audits[row] if persistent_slot_audits is not None else None
            ),
            "local_relational_identity_audit": local_relational_row,
            "dense_candidate_edge_audit": dense_candidate_edge_row,
            "dense_transport_consistency_audit": dense_transport_row,
            "candidate_field_consistency_audit": candidate_field_row,
            "anchor_topology_audit": anchor_topology_row,
            "multilayer_identity_audit": multilayer_identity_row,
            "trajectory_identity_audit": trajectory_identity_audit,
            "transport_factorization_audit": factorization_audit,
            "operator_manifold_audit": operator_manifold_audit,
            "proposals": proposals,
        })
    return rows


def _sample_feature_vectors_at_pixels(
    features: torch.Tensor,
    pixels: torch.Tensor,
    image_size: Sequence[int],
) -> torch.Tensor:
    """Bilinearly sample low-resolution feature maps at full-resolution pixels."""

    if features.ndim != 4 or features.shape[0] != 1:
        raise ValueError("features must have shape [1, C, H, W]")
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError("pixels must have shape [N, 2]")
    image_h, image_w = int(image_size[0]), int(image_size[1])
    if pixels.numel() == 0:
        return torch.empty((0, features.shape[1]), device=features.device, dtype=torch.float32)
    pixels = pixels.to(device=features.device, dtype=torch.float32)
    x = pixels[:, 0].clamp(0, image_w - 1)
    y = pixels[:, 1].clamp(0, image_h - 1)
    grid_x = ((x + 0.5) * 2.0 / float(image_w)) - 1.0
    grid_y = ((y + 0.5) * 2.0 / float(image_h)) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=1).reshape(1, -1, 1, 2)
    sampled = F.grid_sample(
        torch.nan_to_num(features.float(), nan=0.0, posinf=0.0, neginf=0.0),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    return sampled[0, :, :, 0].t().contiguous()


def _local_self_similarity_descriptor(
    features: torch.Tensor,
    center_pixels: torch.Tensor,
    image_size: Sequence[int],
    cell_stride_xy: tuple[float, float],
) -> torch.Tensor:
    """Describe a point by cosine relations to fixed local offsets."""

    offsets = []
    for radius in (1.0, 2.0):
        for dx, dy in (
            (-radius, 0.0),
            (radius, 0.0),
            (0.0, -radius),
            (0.0, radius),
            (-radius, -radius),
            (radius, -radius),
            (-radius, radius),
            (radius, radius),
        ):
            offsets.append((dx * cell_stride_xy[0], dy * cell_stride_xy[1]))
    centers = center_pixels.to(device=features.device, dtype=torch.float32)
    center_vectors = F.normalize(
        _sample_feature_vectors_at_pixels(features, centers, image_size),
        dim=1,
        eps=1e-12,
    )
    scores = []
    for dx, dy in offsets:
        neighbor_pixels = centers + torch.tensor([dx, dy], device=features.device, dtype=torch.float32)
        neighbor_vectors = F.normalize(
            _sample_feature_vectors_at_pixels(features, neighbor_pixels, image_size),
            dim=1,
            eps=1e-12,
        )
        scores.append(F.cosine_similarity(center_vectors, neighbor_vectors, dim=1, eps=1e-12))
    return torch.stack(scores, dim=1)


def _pixel_indices_to_xy(indices: torch.Tensor, image_size: Sequence[int]) -> torch.Tensor:
    image_w = int(image_size[1])
    indices = indices.to(dtype=torch.long)
    x = indices % image_w
    y = torch.div(indices, image_w, rounding_mode="floor")
    return torch.stack((x, y), dim=1).float()


def _pixel_indices_to_replay_cells(
    indices: torch.Tensor,
    image_size: Sequence[int],
    state: FluxReplayState,
) -> torch.Tensor:
    image_h, image_w = int(image_size[0]), int(image_size[1])
    x = (indices % image_w).float()
    y = torch.div(indices, image_w, rounding_mode="floor").float()
    cell_x = torch.floor((x + 0.5) * float(state.image_width) / float(image_w)).long()
    cell_y = torch.floor((y + 0.5) * float(state.image_height) / float(image_h)).long()
    cell_x.clamp_(0, state.image_width - 1)
    cell_y.clamp_(0, state.image_height - 1)
    return cell_y * state.image_width + cell_x


def _attention_local_support_scores(
    mutual_attention: torch.Tensor,
    src_cell: int,
    candidate_cells: torch.Tensor,
    src_state: FluxReplayState,
    trg_state: FluxReplayState,
) -> torch.Tensor:
    """Score whether neighboring source cells support the same local target displacement."""

    if candidate_cells.numel() == 0:
        return torch.empty((0,), device=mutual_attention.device, dtype=torch.float32)
    src_x = int(src_cell % src_state.image_width)
    src_y = int(src_cell // src_state.image_width)
    candidate_cells = candidate_cells.to(device=mutual_attention.device, dtype=torch.long)
    cand_x = (candidate_cells % trg_state.image_width).long()
    cand_y = torch.div(candidate_cells, trg_state.image_width, rounding_mode="floor").long()
    offsets = (
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (1, -1),
        (-1, 1),
        (1, 1),
        (-2, 0),
        (2, 0),
        (0, -2),
        (0, 2),
    )
    scores = []
    for dx, dy in offsets:
        nsx = src_x + dx
        nsy = src_y + dy
        if nsx < 0 or nsx >= src_state.image_width or nsy < 0 or nsy >= src_state.image_height:
            continue
        ntx = cand_x + int(round(dx * float(trg_state.image_width) / float(src_state.image_width)))
        nty = cand_y + int(round(dy * float(trg_state.image_height) / float(src_state.image_height)))
        valid = (ntx >= 0) & (ntx < trg_state.image_width) & (nty >= 0) & (nty < trg_state.image_height)
        target_cells = (nty.clamp(0, trg_state.image_height - 1) * trg_state.image_width + ntx.clamp(0, trg_state.image_width - 1)).long()
        row = mutual_attention[nsy * src_state.image_width + nsx].float()
        row_scale = row.max().clamp_min(1e-12)
        values = row[target_cells] / row_scale
        scores.append(torch.where(valid, values, torch.zeros_like(values)))
    if not scores:
        return torch.zeros((candidate_cells.shape[0],), device=mutual_attention.device, dtype=torch.float32)
    return torch.stack(scores, dim=1).mean(dim=1)


def _cell_offsets(radius: int) -> list[tuple[int, int]]:
    radius = max(0, int(radius))
    return [
        (dx, dy)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
    ]


def _valid_cell(x: int, y: int, width: int, height: int) -> bool:
    return 0 <= int(x) < int(width) and 0 <= int(y) < int(height)


def _cell_index(x: int, y: int, width: int) -> int:
    return int(y) * int(width) + int(x)


def _entropy_from_values(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return 0.0
    probability = values.float().clamp_min(0.0)
    probability = probability / probability.sum().clamp_min(1e-12)
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum()
    denom = math.log(max(2, int(values.numel())))
    return float((entropy / denom).clamp(0.0, 1.0).detach().cpu())


def _attention_patch_metrics(
    mutual_attention: torch.Tensor,
    src_cell: int,
    candidate_cell: int,
    src_state: FluxReplayState,
    trg_state: FluxReplayState,
    *,
    radius: int,
) -> dict[str, float | int]:
    """Measure local transport around a source cell and target candidate."""

    src_w, src_h = int(src_state.image_width), int(src_state.image_height)
    trg_w, trg_h = int(trg_state.image_width), int(trg_state.image_height)
    src_x, src_y = int(src_cell % src_w), int(src_cell // src_w)
    cand_x, cand_y = int(candidate_cell % trg_w), int(candidate_cell // trg_w)
    offsets = _cell_offsets(radius)
    center_row = mutual_attention[int(src_cell)].float()
    center_value = center_row[int(candidate_cell)].float()
    center_row_max = center_row.max().clamp_min(1e-12)

    patch_values = []
    center_rank_values = []
    for dx, dy in offsets:
        tx, ty = cand_x + dx, cand_y + dy
        if not _valid_cell(tx, ty, trg_w, trg_h):
            continue
        value = center_row[_cell_index(tx, ty, trg_w)].float()
        patch_values.append(value)
        center_rank_values.append(value)
    if patch_values:
        patch_tensor = torch.stack(patch_values)
        patch_sorted = torch.sort(patch_tensor, descending=True).values
        center_patch_mass = float((center_value / patch_tensor.sum().clamp_min(1e-12)).detach().cpu())
        patch_peak_margin = float(
            (patch_sorted[0] - patch_sorted[1]).detach().cpu()
            if patch_sorted.numel() > 1
            else patch_sorted[0].detach().cpu()
        )
        center_rank = int((patch_tensor > center_value).sum().detach().cpu()) + 1
        patch_entropy = _entropy_from_values(patch_tensor)
    else:
        center_patch_mass = 0.0
        patch_peak_margin = 0.0
        center_rank = 0
        patch_entropy = 0.0

    aligned_values = []
    inverse_values = []
    local_peak_values = []
    displacement_errors = []
    residual_bins: Counter[tuple[int, int]] = Counter()
    valid_neighbor_count = 0
    scale_x = float(trg_w) / float(max(1, src_w))
    scale_y = float(trg_h) / float(max(1, src_h))
    for dx, dy in offsets:
        if dx == 0 and dy == 0:
            continue
        nsx, nsy = src_x + dx, src_y + dy
        if not _valid_cell(nsx, nsy, src_w, src_h):
            continue
        expected_dx = int(round(float(dx) * scale_x))
        expected_dy = int(round(float(dy) * scale_y))
        ntx, nty = cand_x + expected_dx, cand_y + expected_dy
        if not _valid_cell(ntx, nty, trg_w, trg_h):
            continue
        valid_neighbor_count += 1
        src_neighbor = _cell_index(nsx, nsy, src_w)
        target_expected = _cell_index(ntx, nty, trg_w)
        row = mutual_attention[src_neighbor].float()
        row_max = row.max().clamp_min(1e-12)
        col_max = mutual_attention[:, target_expected].float().max().clamp_min(1e-12)
        aligned = row[target_expected].float()
        aligned_values.append(aligned / row_max)
        inverse_values.append(aligned / col_max)

        local_values = []
        local_cells = []
        for pdx, pdy in offsets:
            px, py = ntx + pdx, nty + pdy
            if not _valid_cell(px, py, trg_w, trg_h):
                continue
            local_cells.append((px, py))
            local_values.append(row[_cell_index(px, py, trg_w)].float())
        if not local_values:
            continue
        local_tensor = torch.stack(local_values)
        local_argmax = int(torch.argmax(local_tensor).detach().cpu())
        peak_x, peak_y = local_cells[local_argmax]
        peak_value = local_tensor[local_argmax] / row_max
        local_peak_values.append(peak_value)
        residual_x = int(peak_x - ntx)
        residual_y = int(peak_y - nty)
        residual_bins[(residual_x, residual_y)] += 1
        displacement_errors.append(float((residual_x * residual_x + residual_y * residual_y) ** 0.5))

    def _mean_tensor(values: list[torch.Tensor]) -> float:
        if not values:
            return 0.0
        return float(torch.stack(values).mean().detach().cpu())

    residual_counts = torch.tensor(list(residual_bins.values()), dtype=torch.float32, device=mutual_attention.device)
    mean_displacement_error = float(sum(displacement_errors) / len(displacement_errors)) if displacement_errors else 0.0
    max_error = float(max(1, int(radius)) * (2.0 ** 0.5))
    shape_preservation = max(0.0, 1.0 - mean_displacement_error / max_error)
    return {
        "valid_neighbor_count": int(valid_neighbor_count),
        "transport_consistency": _mean_tensor(aligned_values),
        "inverse_transport_consistency": _mean_tensor(inverse_values),
        "local_peak_support": _mean_tensor(local_peak_values),
        "mean_displacement_error": mean_displacement_error,
        "shape_preservation": float(shape_preservation),
        "displacement_entropy": _entropy_from_values(residual_counts),
        "center_patch_mass": float(center_patch_mass),
        "center_patch_entropy": float(patch_entropy),
        "center_patch_peak_margin": float(patch_peak_margin),
        "center_rank_in_patch": int(center_rank),
        "center_score_over_row_peak": float((center_value / center_row_max).detach().cpu()),
    }


def _attention_flow_patch_audit_for_point(
    mutual_attention: torch.Tensor,
    src_cell: int,
    proposal_pixels: torch.Tensor,
    source_size: Sequence[int],
    target_size: Sequence[int],
    src_state: FluxReplayState,
    trg_state: FluxReplayState,
    *,
    target: Sequence[float] | None,
    pck_threshold: float | None,
    radius: int,
) -> dict[str, Any]:
    """Dump local attention-flow patch metrics for attention proposals."""

    del source_size
    target_h, target_w = int(target_size[0]), int(target_size[1])
    proposal_pixels = proposal_pixels.to(device=mutual_attention.device, dtype=torch.long).flatten()
    candidate_cells = _pixel_indices_to_replay_cells(proposal_pixels, target_size, trg_state)
    candidates: list[dict[str, Any]] = []
    score_names = (
        "transport_consistency",
        "inverse_transport_consistency",
        "local_peak_support",
        "shape_preservation",
        "center_patch_mass",
        "center_score_over_row_peak",
        "negative_mean_displacement_error",
        "negative_displacement_entropy",
        "negative_center_patch_entropy",
    )
    signal_scores: dict[str, list[float]] = {name: [] for name in score_names}
    hits: list[bool] = []
    seen_pixels: set[int] = set()
    for rank, (pixel_tensor, cell_tensor) in enumerate(zip(proposal_pixels, candidate_cells), start=1):
        pixel = int(pixel_tensor.detach().cpu())
        if pixel in seen_pixels:
            continue
        seen_pixels.add(pixel)
        xy = [int(pixel % target_w), int(pixel // target_w)]
        hit = bool(_point_hit(xy, target, pck_threshold)) if target is not None and pck_threshold is not None else False
        metrics = _attention_patch_metrics(
            mutual_attention,
            int(src_cell),
            int(cell_tensor.detach().cpu()),
            src_state,
            trg_state,
            radius=radius,
        )
        scores = {
            "transport_consistency": float(metrics["transport_consistency"]),
            "inverse_transport_consistency": float(metrics["inverse_transport_consistency"]),
            "local_peak_support": float(metrics["local_peak_support"]),
            "shape_preservation": float(metrics["shape_preservation"]),
            "center_patch_mass": float(metrics["center_patch_mass"]),
            "center_score_over_row_peak": float(metrics["center_score_over_row_peak"]),
            "negative_mean_displacement_error": -float(metrics["mean_displacement_error"]),
            "negative_displacement_entropy": -float(metrics["displacement_entropy"]),
            "negative_center_patch_entropy": -float(metrics["center_patch_entropy"]),
        }
        for name, value in scores.items():
            signal_scores[name].append(value)
        hits.append(hit)
        candidates.append({
            "rank_attention": int(rank),
            "pixel": xy,
            "pixel_index": int(pixel),
            "pck_hit": hit,
            "replay_cell": int(cell_tensor.detach().cpu()),
            "metrics": metrics,
            "scores": scores,
        })
    ranks = {name: _rank_first_hit(values, hits) for name, values in signal_scores.items()}
    attention_top1_scores = {
        name: values[0] if values else None
        for name, values in signal_scores.items()
    }
    score_gaps = {}
    for name, values in signal_scores.items():
        hit_values = [value for value, hit in zip(values, hits) if hit]
        best_hit = max(hit_values) if hit_values else None
        top1 = attention_top1_scores[name]
        score_gaps[f"{name}_attention_top1_minus_best_pck_hit_proposal"] = (
            float(top1 - best_hit) if top1 is not None and best_hit is not None else None
        )
    return {
        "radius": int(max(1, radius)),
        "proposal_count": len(candidates),
        "score_names": list(score_names),
        "ranks": ranks,
        "score_gaps": score_gaps,
        "candidates": candidates,
    }


def _local_transport_support_rows(
    mutual_attention: torch.Tensor,
    src_cells: torch.Tensor,
    src_state: FluxReplayState,
    trg_state: FluxReplayState,
    *,
    radius: int,
) -> torch.Tensor:
    """Vectorized local transport support for all target cells."""

    src_w, src_h = int(src_state.image_width), int(src_state.image_height)
    trg_w, trg_h = int(trg_state.image_width), int(trg_state.image_height)
    src_cells = src_cells.to(device=mutual_attention.device, dtype=torch.long).flatten()
    query_count = int(src_cells.shape[0])
    target_count = int(trg_w) * int(trg_h)
    support = torch.zeros((query_count, target_count), device=mutual_attention.device, dtype=torch.float32)
    counts = torch.zeros_like(support)
    if query_count == 0 or target_count == 0:
        return support

    src_x = src_cells % src_w
    src_y = torch.div(src_cells, src_w, rounding_mode="floor")
    target_indices = torch.arange(target_count, device=mutual_attention.device, dtype=torch.long)
    target_x = target_indices % trg_w
    target_y = torch.div(target_indices, trg_w, rounding_mode="floor")
    scale_x = float(trg_w) / float(max(1, src_w))
    scale_y = float(trg_h) / float(max(1, src_h))
    for dx, dy in _cell_offsets(radius):
        if dx == 0 and dy == 0:
            continue
        nsx = src_x + int(dx)
        nsy = src_y + int(dy)
        valid_source = (nsx >= 0) & (nsx < src_w) & (nsy >= 0) & (nsy < src_h)
        if not bool(valid_source.any()):
            continue
        trg_dx = int(round(float(dx) * scale_x))
        trg_dy = int(round(float(dy) * scale_y))
        ntx = target_x + trg_dx
        nty = target_y + trg_dy
        valid_target = (ntx >= 0) & (ntx < trg_w) & (nty >= 0) & (nty < trg_h)
        if not bool(valid_target.any()):
            continue
        expected_cells = (nty.clamp(0, trg_h - 1) * trg_w + ntx.clamp(0, trg_w - 1)).long()
        neighbor_cells = (nsy.clamp(0, src_h - 1) * src_w + nsx.clamp(0, src_w - 1)).long()
        rows = mutual_attention[neighbor_cells].float()
        aligned = torch.gather(rows, 1, expected_cells.reshape(1, -1).expand(query_count, -1))
        row_peak = rows.max(dim=1, keepdim=True).values.clamp_min(1e-12)
        col_peak = mutual_attention[:, expected_cells].float().max(dim=0, keepdim=True).values.clamp_min(1e-12)
        term = torch.sqrt((aligned / row_peak).clamp_min(0.0) * (aligned / col_peak).clamp_min(0.0))
        valid = valid_source.reshape(-1, 1) & valid_target.reshape(1, -1)
        support += torch.where(valid, term, torch.zeros_like(term))
        counts += valid.float()
    support = support / counts.clamp_min(1.0)
    return torch.nan_to_num(support, nan=0.0, posinf=0.0, neginf=0.0)


def _dense_grid_edge_index(
    height: int,
    width: int,
    radius: int,
    device: torch.device,
) -> torch.Tensor:
    """Build directed local edges for every node in a dense token grid."""

    height, width = int(height), int(width)
    radius = max(1, int(radius))
    if height <= 0 or width <= 0:
        raise ValueError("dense graph grid dimensions must be positive")
    node_grid = torch.arange(height * width, device=device, dtype=torch.long).reshape(height, width)
    edge_rows: list[torch.Tensor] = []
    edge_cols: list[torch.Tensor] = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            source_y_start = max(0, -dy)
            source_y_end = min(height, height - dy)
            source_x_start = max(0, -dx)
            source_x_end = min(width, width - dx)
            if source_y_start >= source_y_end or source_x_start >= source_x_end:
                continue
            source = node_grid[source_y_start:source_y_end, source_x_start:source_x_end].reshape(-1)
            target = node_grid[
                source_y_start + dy:source_y_end + dy,
                source_x_start + dx:source_x_end + dx,
            ].reshape(-1)
            edge_rows.append(source)
            edge_cols.append(target)
    if not edge_rows:
        return torch.empty((2, 0), device=device, dtype=torch.long)
    return torch.stack((torch.cat(edge_rows), torch.cat(edge_cols)), dim=0)


def _build_attention_sparse_partial_graph(
    mutual_attention: torch.Tensor,
    source_grid_size: Sequence[int],
    target_grid_size: Sequence[int],
    *,
    candidate_topk: int,
    edge_radius: int,
) -> AttentionSparsePartialGraph:
    """Create the GT-free sparse problem consumed by the audit and future solver."""

    source_height, source_width = map(int, source_grid_size)
    target_height, target_width = map(int, target_grid_size)
    source_count = source_height * source_width
    target_count = target_height * target_width
    if mutual_attention.ndim != 2 or tuple(mutual_attention.shape) != (source_count, target_count):
        raise ValueError("mutual attention must align with the dense source and target grids")
    candidate_count = min(max(1, int(candidate_topk)), target_count)
    attention = torch.nan_to_num(
        mutual_attention.float(), nan=0.0, posinf=0.0, neginf=0.0
    ).clamp_min(0.0)
    candidate_values, candidate_targets = torch.topk(
        attention, k=candidate_count, dim=1, sorted=True
    )
    candidate_log = candidate_values.clamp_min(1e-12).log()
    candidate_log = candidate_log - torch.logsumexp(candidate_log, dim=1, keepdim=True)
    return AttentionSparsePartialGraph(
        source_edge_index=_dense_grid_edge_index(
            source_height,
            source_width,
            edge_radius,
            attention.device,
        ),
        candidate_target_index=candidate_targets,
        candidate_unary_log_probability=candidate_log,
        candidate_mask=torch.ones_like(candidate_targets, dtype=torch.bool),
        source_grid_size=(source_height, source_width),
        target_grid_size=(target_height, target_width),
        dustbin_target_index=target_count,
    )


def _dense_candidate_edge_separability_audit_for_points(
    src_features: torch.Tensor,
    trg_features: torch.Tensor,
    mutual_attention: torch.Tensor,
    src_cells: torch.Tensor,
    candidate_cells: torch.Tensor,
    proposal_pixels: torch.Tensor,
    target_size: Sequence[int],
    src_state: FluxReplayState,
    trg_state: FluxReplayState,
    *,
    target_points: Sequence[Sequence[float]] | None,
    pck_threshold: float | None,
    edge_radius: int,
) -> list[dict[str, Any]]:
    """Test one-step dense graph messages without changing predictions."""

    if src_features.ndim != 4 or trg_features.ndim != 4:
        raise ValueError("dense candidate-edge audit requires [1,C,H,W] feature maps")
    if src_features.shape[0] != 1 or trg_features.shape[0] != 1:
        raise ValueError("dense candidate-edge audit requires batch size one")
    if src_features.shape[1] != trg_features.shape[1]:
        raise ValueError("source and target relation features must share channels")
    if tuple(src_features.shape[-2:]) != (src_state.image_height, src_state.image_width):
        raise ValueError("source relation features do not align with the replay grid")
    if tuple(trg_features.shape[-2:]) != (trg_state.image_height, trg_state.image_width):
        raise ValueError("target relation features do not align with the replay grid")
    if candidate_cells.ndim != 2 or proposal_pixels.shape != candidate_cells.shape:
        raise ValueError("candidate cells and proposal pixels must align")

    graph = _build_attention_sparse_partial_graph(
        mutual_attention,
        (src_state.image_height, src_state.image_width),
        (trg_state.image_height, trg_state.image_width),
        candidate_topk=int(candidate_cells.shape[1]),
        edge_radius=edge_radius,
    )
    src_cells = src_cells.to(device=mutual_attention.device, dtype=torch.long).flatten()
    candidate_cells = candidate_cells.to(device=mutual_attention.device, dtype=torch.long)
    proposal_pixels = proposal_pixels.to(device=mutual_attention.device, dtype=torch.long)
    if tuple(candidate_cells.shape) != (
        int(src_cells.numel()),
        int(graph.candidate_target_index.shape[1]),
    ):
        raise ValueError("query candidates do not align with the dense partial graph")
    if not torch.equal(graph.candidate_target_index[src_cells], candidate_cells):
        raise RuntimeError("query candidates diverged from the dense graph candidate contract")

    source_tokens = F.normalize(
        torch.nan_to_num(src_features[0].float()).permute(1, 2, 0).reshape(
            src_state.image_height * src_state.image_width,
            src_features.shape[1],
        ),
        dim=1,
        eps=1e-12,
    )
    target_tokens = F.normalize(
        torch.nan_to_num(trg_features[0].float()).permute(1, 2, 0).reshape(
            trg_state.image_height * trg_state.image_width,
            trg_features.shape[1],
        ),
        dim=1,
        eps=1e-12,
    )
    source_height, source_width = graph.source_grid_size
    target_height, target_width = graph.target_grid_size
    _target_h, target_w = map(int, target_size)
    score_names = [
        "attention_unary_control",
        "dense_edge_spatial_message",
        "dense_edge_relation_message",
        "dense_edge_joint_message",
        "dense_partial_graph_one_step_belief",
    ]
    graph_contract = graph.contract()
    graph_contract["edge_radius"] = int(max(1, int(edge_radius)))
    rows: list[dict[str, Any]] = []
    for row, source_cell_tensor in enumerate(src_cells):
        source_cell = int(source_cell_tensor.detach().cpu())
        outgoing = graph.source_edge_index[1, graph.source_edge_index[0] == source_cell]
        query_targets = graph.candidate_target_index[source_cell]
        query_unary = graph.candidate_unary_log_probability[source_cell]
        if int(outgoing.numel()) > 0:
            neighbor_targets = graph.candidate_target_index[outgoing]
            neighbor_unary = _standardize_finite(
                graph.candidate_unary_log_probability[outgoing], dim=1
            )

            source_x = float(source_cell % source_width)
            source_y = float(source_cell // source_width)
            neighbor_x = (outgoing % source_width).float()
            neighbor_y = torch.div(outgoing, source_width, rounding_mode="floor").float()
            source_delta = torch.stack(
                (
                    (neighbor_x - source_x) / float(max(1, source_width)),
                    (neighbor_y - source_y) / float(max(1, source_height)),
                ),
                dim=1,
            )

            query_x = (query_targets % target_width).float()
            query_y = torch.div(query_targets, target_width, rounding_mode="floor").float()
            neighbor_target_x = (neighbor_targets % target_width).float()
            neighbor_target_y = torch.div(
                neighbor_targets, target_width, rounding_mode="floor"
            ).float()
            target_delta = torch.stack(
                (
                    (neighbor_target_x[:, None, :] - query_x[None, :, None])
                    / float(max(1, target_width)),
                    (neighbor_target_y[:, None, :] - query_y[None, :, None])
                    / float(max(1, target_height)),
                ),
                dim=3,
            )
            spatial_raw = -torch.linalg.vector_norm(
                target_delta - source_delta[:, None, None, :], dim=3
            )

            source_relation = (
                source_tokens[outgoing] * source_tokens[source_cell].reshape(1, -1)
            ).sum(dim=1)
            query_descriptors = target_tokens[query_targets]
            neighbor_descriptors = target_tokens[neighbor_targets]
            target_relation = torch.einsum(
                "kc,dlc->dkl", query_descriptors, neighbor_descriptors
            )
            relation_raw = -(target_relation - source_relation[:, None, None]).abs()

            spatial_compatibility = _standardize_finite(spatial_raw, dim=2)
            relation_compatibility = _standardize_finite(relation_raw, dim=2)
            joint_compatibility = 0.5 * (
                spatial_compatibility + relation_compatibility
            )
            neighbor_evidence = neighbor_unary[:, None, :]
            spatial_message = (
                neighbor_evidence + spatial_compatibility
            ).amax(dim=2).mean(dim=0)
            relation_message = (
                neighbor_evidence + relation_compatibility
            ).amax(dim=2).mean(dim=0)
            joint_message = (
                neighbor_evidence + joint_compatibility
            ).amax(dim=2).mean(dim=0)
        else:
            spatial_message = torch.zeros_like(query_unary)
            relation_message = torch.zeros_like(query_unary)
            joint_message = torch.zeros_like(query_unary)
        one_step_belief = _standardize_finite(query_unary, dim=0) + _standardize_finite(
            joint_message, dim=0
        )
        signals = {
            "attention_unary_control": query_unary,
            "dense_edge_spatial_message": spatial_message,
            "dense_edge_relation_message": relation_message,
            "dense_edge_joint_message": joint_message,
            "dense_partial_graph_one_step_belief": one_step_belief,
        }

        target = target_points[row] if target_points is not None else None
        hits: list[bool] = []
        candidates: list[dict[str, Any]] = []
        for candidate_index, pixel_tensor in enumerate(proposal_pixels[row]):
            pixel = int(pixel_tensor.detach().cpu())
            xy = [int(pixel % target_w), int(pixel // target_w)]
            hit = bool(_point_hit(xy, target, pck_threshold)) if (
                target is not None and pck_threshold is not None
            ) else False
            hits.append(hit)
            candidates.append({
                "attention_rank": int(candidate_index + 1),
                "target_cell": int(query_targets[candidate_index].detach().cpu()),
                "pixel_index": int(pixel),
                "pixel": xy,
                "pck_hit": bool(hit),
                "scores": {
                    name: float(values[candidate_index].detach().cpu())
                    for name, values in signals.items()
                },
            })

        ranks: dict[str, int | None] = {}
        topk_hits: dict[str, bool] = {}
        score_gaps: dict[str, float | None] = {}
        selected_attention_ranks: dict[str, int] = {}
        for name, values_tensor in signals.items():
            values = [float(value) for value in values_tensor.detach().cpu().tolist()]
            ranks[name] = _rank_first_hit(values, hits)
            order = sorted(range(len(values)), key=lambda index: values[index], reverse=True)
            selected_attention_ranks[name] = int(order[0] + 1)
            for topk in (1, 3, 5, 10, 20):
                topk_hits[f"{name}@{topk}"] = bool(
                    any(hits[index] for index in order[: min(topk, len(order))])
                )
            hit_values = [value for value, hit in zip(values, hits) if hit]
            best_hit = max(hit_values) if hit_values else None
            score_gaps[f"{name}_attention_top1_minus_best_pck_hit_proposal"] = (
                float(values[0] - best_hit) if best_hit is not None else None
            )
        rows.append({
            "score_names": score_names,
            "ranks": ranks,
            "topk_hits": topk_hits,
            "score_gaps": score_gaps,
            "graph_contract": graph_contract,
            "diagnostics": {
                "source_cell": int(source_cell),
                "source_neighbor_count": int(outgoing.numel()),
                "candidate_count": int(len(candidates)),
                "pck_hit_candidate_count": int(sum(hits)),
                "pck_hit_candidate_fraction": float(sum(hits) / max(1, len(hits))),
                "selected_attention_ranks": selected_attention_ranks,
                "gt_used_for_scoring": False,
                "native_candidate_injected": False,
                "native_fallback_used": False,
            },
            "candidates": candidates,
        })
    return rows


def _dense_relation_edge_messages(
    graph: AttentionSparsePartialGraph,
    src_features: torch.Tensor,
    trg_features: torch.Tensor,
    *,
    neighbor_evidence: torch.Tensor | None = None,
    edge_chunk_size: int = 256,
) -> torch.Tensor:
    """Aggregate the audited relation factor over every dense source node."""

    if src_features.ndim != 4 or trg_features.ndim != 4:
        raise ValueError("dense relation messages require [1,C,H,W] feature maps")
    if src_features.shape[0] != 1 or trg_features.shape[0] != 1:
        raise ValueError("dense relation messages require batch size one")
    if src_features.shape[1] != trg_features.shape[1]:
        raise ValueError("source and target relation features must share channels")
    if tuple(src_features.shape[-2:]) != tuple(graph.source_grid_size):
        raise ValueError("source relation features do not align with the partial graph")
    if tuple(trg_features.shape[-2:]) != tuple(graph.target_grid_size):
        raise ValueError("target relation features do not align with the partial graph")

    source_tokens = F.normalize(
        torch.nan_to_num(
            src_features[0].float(), nan=0.0, posinf=0.0, neginf=0.0
        ).permute(1, 2, 0).reshape(-1, src_features.shape[1]),
        dim=1,
        eps=1e-12,
    )
    target_tokens = F.normalize(
        torch.nan_to_num(
            trg_features[0].float(), nan=0.0, posinf=0.0, neginf=0.0
        ).permute(1, 2, 0).reshape(-1, trg_features.shape[1]),
        dim=1,
        eps=1e-12,
    )
    target_gram = torch.nan_to_num(
        target_tokens @ target_tokens.t(), nan=0.0, posinf=0.0, neginf=0.0
    )
    evidence = (
        graph.candidate_unary_log_probability
        if neighbor_evidence is None
        else neighbor_evidence
    )
    if evidence.shape != graph.candidate_target_index.shape:
        raise ValueError("neighbor evidence must align with graph candidates")
    evidence = _standardize_finite(evidence, dim=1).masked_fill(
        ~graph.candidate_mask, -1e4
    )

    source_edge, neighbor_edge = graph.source_edge_index
    source_count, candidate_count = graph.candidate_target_index.shape
    messages = torch.zeros(
        (source_count, candidate_count),
        device=source_tokens.device,
        dtype=torch.float32,
    )
    degree = torch.zeros(source_count, device=source_tokens.device, dtype=torch.float32)
    chunk_size = max(1, int(edge_chunk_size))
    for edge_start in range(0, int(source_edge.numel()), chunk_size):
        edge_end = min(int(source_edge.numel()), edge_start + chunk_size)
        centers = source_edge[edge_start:edge_end]
        neighbors = neighbor_edge[edge_start:edge_end]
        center_candidates = graph.candidate_target_index[centers]
        neighbor_candidates = graph.candidate_target_index[neighbors]
        source_relation = (source_tokens[centers] * source_tokens[neighbors]).sum(dim=1)
        target_relation = target_gram[
            center_candidates[:, :, None], neighbor_candidates[:, None, :]
        ]
        relation_compatibility = _standardize_finite(
            -(target_relation - source_relation[:, None, None]).abs(), dim=2
        )
        edge_message = (
            evidence[neighbors, None, :] + relation_compatibility
        ).amax(dim=2)
        messages.index_add_(0, centers, edge_message)
        degree.index_add_(0, centers, torch.ones_like(centers, dtype=torch.float32))
    return torch.nan_to_num(
        messages / degree.clamp_min(1.0)[:, None],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def _solve_attention_sparse_partial_assignment(
    candidate_target_index: torch.Tensor,
    candidate_scores: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    target_count: int,
    required_source_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Solve exact sparse one-to-one assignment with one private dustbin per source."""

    if candidate_target_index.ndim != 2 or candidate_scores.shape != candidate_target_index.shape:
        raise ValueError("candidate targets and scores must share [source, candidate]")
    if candidate_mask.shape != candidate_target_index.shape:
        raise ValueError("candidate mask must align with candidate targets")
    source_count, candidate_count = candidate_target_index.shape
    target_count = int(target_count)
    if source_count <= 0 or candidate_count <= 0 or target_count <= 0:
        raise ValueError("partial assignment requires non-empty source, candidate, and target sets")
    if bool(((candidate_target_index < 0) | (candidate_target_index >= target_count))[candidate_mask].any()):
        raise ValueError("real candidate target index is outside the target partition")

    required = (
        torch.zeros(source_count, device=candidate_scores.device, dtype=torch.bool)
        if required_source_mask is None
        else required_source_mask.to(device=candidate_scores.device, dtype=torch.bool).flatten()
    )
    if required.shape != (source_count,):
        raise ValueError("required source mask must align with the source partition")
    scores = torch.nan_to_num(
        candidate_scores.float(), nan=-1e4, posinf=1e4, neginf=-1e4
    ).masked_fill(~candidate_mask, -1e4)
    valid_count = candidate_mask.sum(dim=1).clamp_min(1)
    dustbin_scores = (
        scores.masked_fill(~candidate_mask, 0.0).sum(dim=1) / valid_count.float()
    )
    row_minimum = torch.minimum(
        scores.masked_fill(~candidate_mask, float("inf")).amin(dim=1),
        dustbin_scores,
    )
    row_shift = 1.0 - row_minimum

    source_grid = torch.arange(source_count, device=scores.device)[:, None].expand_as(scores)
    real_rows = source_grid[candidate_mask].detach().cpu().numpy()
    real_cols = candidate_target_index[candidate_mask].detach().cpu().numpy()
    real_weights = (scores + row_shift[:, None])[candidate_mask].detach().cpu().double().numpy()
    dustbin_rows = torch.nonzero(~required, as_tuple=False).flatten().detach().cpu().numpy()
    dustbin_cols = target_count + dustbin_rows
    dustbin_weights = (
        dustbin_scores[~required] + row_shift[~required]
    ).detach().cpu().double().numpy()
    sparse_weights = coo_matrix(
        (
            np.concatenate((real_weights, dustbin_weights)),
            (
                np.concatenate((real_rows, dustbin_rows)),
                np.concatenate((real_cols, dustbin_cols)),
            ),
        ),
        shape=(source_count, target_count + source_count),
    ).tocsr()
    matched_rows, matched_cols = min_weight_full_bipartite_matching(
        sparse_weights, maximize=True
    )
    if len(matched_rows) != source_count:
        raise RuntimeError("sparse partial assignment did not cover every source node")
    selected_columns = np.full(source_count, -1, dtype=np.int64)
    selected_columns[matched_rows] = matched_cols
    selected_targets = torch.from_numpy(selected_columns).to(
        device=scores.device, dtype=torch.long
    )
    assignment_state = torch.full(
        (source_count,), -1, device=scores.device, dtype=torch.long
    )
    real_assignment = selected_targets < target_count
    if bool((required & ~real_assignment).any()):
        raise RuntimeError("a required query source was assigned to dustbin")
    if bool(real_assignment.any()):
        real_sources = torch.nonzero(real_assignment, as_tuple=False).flatten()
        matches = (
            candidate_target_index[real_sources]
            == selected_targets[real_sources, None]
        ) & candidate_mask[real_sources]
        if bool((matches.sum(dim=1) != 1).any()):
            raise RuntimeError("solver selected a target outside its source candidate domain")
        assignment_state[real_sources] = matches.float().argmax(dim=1)
    selected_targets = torch.where(
        real_assignment,
        selected_targets,
        selected_targets.new_full(selected_targets.shape, target_count),
    )

    unconstrained_state = scores.argmax(dim=1)
    unconstrained_targets = candidate_target_index.gather(
        1, unconstrained_state[:, None]
    ).squeeze(1)
    unconstrained_counts = torch.bincount(
        unconstrained_targets, minlength=target_count
    )
    selected_score = dustbin_scores.clone()
    if bool(real_assignment.any()):
        real_sources = torch.nonzero(real_assignment, as_tuple=False).flatten()
        selected_score[real_sources] = scores[
            real_sources, assignment_state[real_sources]
        ]
    return {
        "assignment_state": assignment_state,
        "selected_target": selected_targets,
        "dustbin_score": dustbin_scores,
        "unconstrained_state": unconstrained_state,
        "unconstrained_target": unconstrained_targets,
        "unconstrained_collision_count": int(
            (unconstrained_counts - 1).clamp_min(0).sum().detach().cpu()
        ),
        "matched_real_count": int(real_assignment.sum().detach().cpu()),
        "dustbin_count": int((~real_assignment).sum().detach().cpu()),
        "required_source_count": int(required.sum().detach().cpu()),
        "partial_assignment_objective": float(selected_score.sum().detach().cpu()),
        "unconstrained_objective": float(
            scores.gather(1, unconstrained_state[:, None]).sum().detach().cpu()
        ),
        "solver": "scipy_sparse_full_bipartite_matching_with_private_dustbins",
        "dustbin_rule": "row_mean_dense_graph_belief",
    }


def _dense_partial_graph_matching_rankings(
    src_features: torch.Tensor,
    trg_features: torch.Tensor,
    attention: dict[str, torch.Tensor],
    source_points: Sequence[Sequence[float]],
    source_size: Sequence[int],
    target_size: Sequence[int],
    src_state: FluxReplayState,
    trg_state: FluxReplayState,
    *,
    candidate_topk: int = 20,
    target_points: Sequence[Sequence[float]] | None = None,
    pck_threshold: float | None = None,
) -> tuple[torch.Tensor, list[dict[str, Any]], dict[str, Any]]:
    """Match query points by solving the audited dense partial assignment."""

    points = list(source_points)
    targets = list(target_points) if target_points is not None else None
    if not points:
        return torch.empty((0, 0), device=src_features.device, dtype=torch.long), [], {}
    if targets is not None and len(targets) != len(points):
        raise ValueError("target_points must align with source_points")
    mutual = torch.sqrt(
        (attention["p_ab"].float() * attention["p_ba"].float().t()).clamp_min(0.0)
    )
    mutual = torch.nan_to_num(mutual, nan=0.0, posinf=0.0, neginf=0.0).to(
        src_features.device
    )
    graph = _build_attention_sparse_partial_graph(
        mutual,
        (src_state.image_height, src_state.image_width),
        (trg_state.image_height, trg_state.image_width),
        candidate_topk=candidate_topk,
        edge_radius=1,
    )
    relation_message = _dense_relation_edge_messages(graph, src_features, trg_features)
    attention_unary = _standardize_finite(
        graph.candidate_unary_log_probability, dim=1
    )
    relation_unary = _standardize_finite(relation_message, dim=1)
    graph_belief = attention_unary + relation_unary
    target_count = int(trg_state.image_height * trg_state.image_width)
    src_cells = _native_cell_indices_for_points(
        points,
        source_size,
        src_state.image_height,
        src_state.image_width,
        src_features.device,
    )
    required_source_mask = torch.zeros(
        graph.candidate_target_index.shape[0],
        device=src_features.device,
        dtype=torch.bool,
    )
    required_source_mask[src_cells] = True
    solution = _solve_attention_sparse_partial_assignment(
        graph.candidate_target_index,
        graph_belief,
        graph.candidate_mask,
        target_count=target_count,
        required_source_mask=required_source_mask,
    )
    query_candidates = graph.candidate_target_index[src_cells]
    candidate_pixels = _cell_topk_to_pixel_indices(
        query_candidates, target_size, trg_state
    )
    candidate_count = int(query_candidates.shape[1])
    target_w = int(target_size[1])

    def _order(scores: torch.Tensor, selected: int | None = None) -> list[int]:
        ordered = [int(value) for value in torch.argsort(scores, descending=True).detach().cpu()]
        if selected is not None and selected >= 0:
            ordered = [int(selected)] + [value for value in ordered if value != selected]
        return ordered

    def _hits(order: Sequence[int], row: int) -> dict[str, bool]:
        if targets is None or pck_threshold is None:
            return {f"@{k}": False for k in (1, 3, 5, 10, 20)}
        candidate_hits = []
        for state in order:
            pixel = int(candidate_pixels[row, state])
            candidate_hits.append(
                _point_hit(
                    [int(pixel % target_w), int(pixel // target_w)],
                    targets[row],
                    float(pck_threshold),
                )
            )
        return {
            f"@{k}": bool(any(candidate_hits[: min(k, len(candidate_hits))]))
            for k in (1, 3, 5, 10, 20)
        }

    ranked_rows: list[list[int]] = []
    audits: list[dict[str, Any]] = []
    selected_attention_ranks: list[int] = []
    query_dustbin_count = 0
    query_changed_count = 0
    for row, source_cell_tensor in enumerate(src_cells):
        source_cell = int(source_cell_tensor.detach().cpu())
        assigned_state = int(solution["assignment_state"][source_cell])
        solver_dustbin = assigned_state < 0
        belief_order = _order(graph_belief[source_cell])
        selected_state = belief_order[0] if solver_dustbin else assigned_state
        final_order = _order(graph_belief[source_cell], selected_state)
        relation_order = _order(relation_message[source_cell])
        attention_order = list(range(candidate_count))
        ranked_rows.append(
            [int(candidate_pixels[row, state]) for state in final_order]
        )
        selected_attention_ranks.append(selected_state + 1)
        query_dustbin_count += int(solver_dustbin)
        query_changed_count += int(selected_state != 0)

        candidates: list[dict[str, Any]] = []
        final_rank = {state: rank for rank, state in enumerate(final_order, start=1)}
        for state in final_order:
            pixel = int(candidate_pixels[row, state])
            xy = [int(pixel % target_w), int(pixel // target_w)]
            candidates.append({
                "rank": int(final_rank[state]),
                "attention_rank": int(state + 1),
                "target_cell": int(query_candidates[row, state]),
                "pixel_index": pixel,
                "pixel": xy,
                "pck_hit": bool(
                    targets is not None
                    and pck_threshold is not None
                    and _point_hit(xy, targets[row], float(pck_threshold))
                ),
                "partial_assignment_selected": bool(
                    not solver_dustbin and state == assigned_state
                ),
                "scores": {
                    "attention_unary": float(attention_unary[source_cell, state]),
                    "dense_relation_message": float(relation_message[source_cell, state]),
                    "dense_graph_belief": float(graph_belief[source_cell, state]),
                },
            })
        audits.append({
            "source_cell": source_cell,
            "candidate_count": candidate_count,
            "solver_assigned_dustbin": bool(solver_dustbin),
            "dustbin_readout": "dense_graph_belief_top1" if solver_dustbin else None,
            "selected_state": int(selected_state),
            "selected_attention_rank": int(selected_state + 1),
            "final_changed_from_attention": bool(selected_state != 0),
            "native_candidate_injected": False,
            "native_fallback_used": False,
            "gt_used_for_inference": False,
            "topk_hits": {
                "attention": _hits(attention_order, row),
                "dense_relation": _hits(relation_order, row),
                "dense_graph_belief": _hits(belief_order, row),
                "dense_partial_assignment": _hits(final_order, row),
            },
            "candidates": candidates,
        })

    summary = {
        **graph.contract(),
        "pairwise_signal": "dense_local_ditf_self_similarity_relation_only",
        "solver": solution["solver"],
        "dustbin_rule": solution["dustbin_rule"],
        "pairwise_mechanism": "dense_local_ditf_self_similarity_relation",
        "spatial_edge_used": False,
        "relation_edge_used": True,
        "descriptor_unary_used": False,
        "target_capacity": 1,
        "matched_real_count": int(solution["matched_real_count"]),
        "dustbin_count": int(solution["dustbin_count"]),
        "required_source_count": int(solution["required_source_count"]),
        "required_source_dustbin_count": 0,
        "matched_real_fraction": float(solution["matched_real_count"] / max(1, graph.candidate_target_index.shape[0])),
        "unconstrained_collision_count": int(solution["unconstrained_collision_count"]),
        "partial_assignment_collision_count": 0,
        "partial_assignment_objective": float(solution["partial_assignment_objective"]),
        "unconstrained_objective": float(solution["unconstrained_objective"]),
        "query_count": int(len(points)),
        "query_dustbin_count": int(query_dustbin_count),
        "query_changed_from_attention_count": int(query_changed_count),
        "selected_attention_rank_mean": float(
            sum(selected_attention_ranks) / max(1, len(selected_attention_ranks))
        ),
        "candidate_source": "mutual_cross_attention_topk_only",
        "native_candidate_injected_count": 0,
        "native_fallback_count": 0,
        "gt_used_for_inference": False,
    }
    return torch.tensor(ranked_rows, device=src_features.device, dtype=torch.long), audits, summary


def _expert_preserving_hypothesis_scores(
    probe: dict[str, Any],
    aggregated_attention: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Route candidate-conditioned QK/V hypotheses before expert aggregation."""

    expert_scores = probe.get("expert_scores", {})
    support_name = (
        "log_exact_mutual_cross_probability"
        if isinstance(
            expert_scores.get("log_exact_mutual_cross_probability"),
            torch.Tensor,
        )
        else "bidirectional_negative_log_rank"
    )
    support = expert_scores.get(support_name)
    identity_name = "symmetric_value_residual_alignment"
    identity = expert_scores.get(identity_name)
    if not isinstance(support, torch.Tensor) or support.ndim != 4:
        raise ValueError(
            "expert-preserving replay requires [ensemble,head,point,candidate] QK support"
        )
    if not isinstance(identity, torch.Tensor) or identity.shape != support.shape:
        raise ValueError(
            "expert-preserving replay requires aligned expert-wise symmetric value identity"
        )
    if tuple(aggregated_attention.shape) != tuple(support.shape[-2:]):
        raise ValueError("aggregated attention must align with expert hypotheses")

    support_z = _standardize_finite(support, dim=-1)
    identity_z = _standardize_finite(identity, dim=-1)
    attention_z = _standardize_finite(
        aggregated_attention.to(device=support.device, dtype=torch.float32),
        dim=-1,
    )
    ensemble_count, head_count, point_count, candidate_count = map(
        int, support.shape
    )

    head_support = support_z.mean(dim=0)
    head_identity = identity_z.mean(dim=0)
    head_point_agreement = F.cosine_similarity(
        head_support,
        head_identity,
        dim=-1,
        eps=1e-12,
    )
    pair_head_agreement = head_point_agreement.mean(dim=1)
    selected_head = int(pair_head_agreement.argmax().detach().cpu())
    pair_head_support = head_support[selected_head]
    pair_head_identity = head_identity[selected_head]

    expert_agreement = F.cosine_similarity(
        support_z,
        identity_z,
        dim=-1,
        eps=1e-12,
    ).mean(dim=2)
    selected_expert_flat = int(expert_agreement.reshape(-1).argmax().detach().cpu())
    selected_member = selected_expert_flat // head_count
    selected_expert_head = selected_expert_flat % head_count
    pair_expert_support = support_z[selected_member, selected_expert_head]
    pair_expert_identity = identity_z[selected_member, selected_expert_head]

    point_head = head_point_agreement.argmax(dim=0)
    point_index = torch.arange(point_count, device=support.device)
    point_head_support = head_support[point_head, point_index]
    point_head_identity = head_identity[point_head, point_index]

    mean_support = support_z.mean(dim=(0, 1))
    mean_identity = identity_z.mean(dim=(0, 1))
    signals = {
        "attention": aggregated_attention.to(
            device=support.device, dtype=torch.float32
        ),
        "mean_expert_hypothesis": attention_z + mean_support + mean_identity,
        "pair_head_support": pair_head_support,
        "pair_head_identity": pair_head_identity,
        "pair_head_hypothesis": (
            attention_z + pair_head_support + pair_head_identity
        ),
        "pair_expert_hypothesis": (
            attention_z + pair_expert_support + pair_expert_identity
        ),
        "point_head_hypothesis": (
            attention_z + point_head_support + point_head_identity
        ),
    }
    top_head_values = torch.topk(
        pair_head_agreement,
        k=min(2, head_count),
    ).values
    route = {
        "support_signal": support_name,
        "identity_signal": identity_name,
        "ensemble_count": ensemble_count,
        "head_count": head_count,
        "point_count": point_count,
        "candidate_count": candidate_count,
        "selected_head": selected_head,
        "selected_head_agreement": float(
            pair_head_agreement[selected_head].detach().cpu()
        ),
        "selected_head_agreement_margin": float(
            (top_head_values[0] - top_head_values[1]).detach().cpu()
            if int(top_head_values.numel()) > 1
            else 0.0
        ),
        "head_agreements": [
            float(value) for value in pair_head_agreement.detach().cpu().tolist()
        ],
        "selected_expert": {
            "member": selected_member,
            "head": selected_expert_head,
            "agreement": float(
                expert_agreement[selected_member, selected_expert_head]
                .detach()
                .cpu()
            ),
        },
        "point_heads": [
            int(value) for value in point_head.detach().cpu().tolist()
        ],
        "aggregation_order": (
            "candidate_conditioned_qk_and_symmetric_value_identity_then_ensemble"
        ),
    }
    return signals, route


def _expert_preserving_attention_hypothesis_conditioned_replay_rankings(
    attention: dict[str, torch.Tensor],
    source_points: Sequence[Sequence[float]],
    source_size: Sequence[int],
    target_size: Sequence[int],
    src_state: FluxReplayState,
    trg_state: FluxReplayState,
    blocks: Sequence[Any],
    *,
    candidate_topk: int = 20,
    target_points: Sequence[Sequence[float]] | None = None,
    pck_threshold: float | None = None,
) -> tuple[torch.Tensor, list[dict[str, Any]], dict[str, Any]]:
    """Match attention candidates after pair-level latent-head identity routing."""

    points = list(source_points)
    targets = list(target_points) if target_points is not None else None
    if not points:
        device = attention["p_ab"].device
        return torch.empty((0, 0), device=device, dtype=torch.long), [], {}
    if targets is not None and len(targets) != len(points):
        raise ValueError("target_points must align with source_points")

    device = attention["p_ab"].device
    src_cells = _native_cell_indices_for_points(
        points,
        source_size,
        src_state.image_height,
        src_state.image_width,
        device,
    )
    mutual = torch.sqrt(
        (attention["p_ab"].float() * attention["p_ba"].float().t()).clamp_min(0.0)
    )
    mutual = torch.nan_to_num(mutual, nan=0.0, posinf=0.0, neginf=0.0)
    candidate_count = min(max(1, int(candidate_topk)), int(mutual.shape[1]))
    attention_values, candidate_cells = torch.topk(
        mutual[src_cells],
        k=candidate_count,
        dim=1,
        sorted=True,
    )
    probe = flux_cross_readout_probe(
        blocks,
        src_state,
        trg_state,
        src_cells,
        candidate_cells,
        mode="exact",
        use_coordinate_bias=False,
    )
    signals, route = _expert_preserving_hypothesis_scores(
        probe,
        attention_values,
    )
    candidate_pixels = _cell_topk_to_pixel_indices(
        candidate_cells,
        target_size,
        trg_state,
    ).to(device)
    target_w = int(target_size[1])
    method_name = "pair_head_hypothesis"
    signal_names = list(signals.keys())

    def _order(values: torch.Tensor) -> list[int]:
        return [
            int(value)
            for value in torch.argsort(values, descending=True).detach().cpu()
        ]

    def _hits(order: Sequence[int], row: int) -> dict[str, bool]:
        if targets is None or pck_threshold is None:
            return {f"@{k}": False for k in (1, 3, 5, 10, 20)}
        candidate_hits = []
        for candidate in order:
            pixel = int(candidate_pixels[row, candidate])
            candidate_hits.append(
                _point_hit(
                    [int(pixel % target_w), int(pixel // target_w)],
                    targets[row],
                    float(pck_threshold),
                )
            )
        return {
            f"@{k}": bool(any(candidate_hits[: min(k, len(candidate_hits))]))
            for k in (1, 3, 5, 10, 20)
        }

    orders = {
        name: [_order(values[row]) for row in range(len(points))]
        for name, values in signals.items()
    }
    final_orders = orders[method_name]
    ranked_rows = [
        [int(candidate_pixels[row, candidate]) for candidate in final_orders[row]]
        for row in range(len(points))
    ]
    audits: list[dict[str, Any]] = []
    selected_attention_ranks: list[int] = []
    changed_count = 0
    for row, final_order in enumerate(final_orders):
        selected_state = int(final_order[0])
        selected_attention_ranks.append(selected_state + 1)
        changed_count += int(selected_state != 0)
        rank_by_state = {
            state: rank for rank, state in enumerate(final_order, start=1)
        }
        candidates = []
        for state in final_order:
            pixel = int(candidate_pixels[row, state])
            xy = [int(pixel % target_w), int(pixel // target_w)]
            candidates.append({
                "rank": int(rank_by_state[state]),
                "attention_rank": int(state + 1),
                "target_cell": int(candidate_cells[row, state]),
                "pixel_index": pixel,
                "pixel": xy,
                "pck_hit": bool(
                    targets is not None
                    and pck_threshold is not None
                    and _point_hit(xy, targets[row], float(pck_threshold))
                ),
                "scores": {
                    name: float(values[row, state].detach().cpu())
                    for name, values in signals.items()
                },
            })
        audits.append({
            "source_cell": int(src_cells[row].detach().cpu()),
            "candidate_count": candidate_count,
            "selected_state": selected_state,
            "selected_attention_rank": selected_state + 1,
            "final_changed_from_attention": bool(selected_state != 0),
            "selected_head": int(route["selected_head"]),
            "selected_point_head": int(route["point_heads"][row]),
            "native_candidate_injected": False,
            "native_fallback_used": False,
            "gt_used_for_inference": False,
            "topk_hits": {
                name: _hits(orders[name][row], row)
                for name in signal_names
            },
            "candidates": candidates,
        })

    summary = {
        **route,
        "candidate_source": "exact_mutual_cross_attention_topk_only",
        "method_signal": method_name,
        "attention_control_signal": "attention",
        "expert_axes_preserved_until_candidate_scoring": True,
        "candidate_value_aggregation_used": False,
        "symmetric_candidate_identity_used": True,
        "native_candidate_injected_count": 0,
        "native_fallback_count": 0,
        "descriptor_unary_used": False,
        "gt_used_for_inference": False,
        "query_count": len(points),
        "query_changed_from_attention_count": changed_count,
        "selected_attention_rank_mean": float(
            sum(selected_attention_ranks) / max(1, len(selected_attention_ranks))
        ),
    }
    return (
        torch.tensor(ranked_rows, device=device, dtype=torch.long),
        audits,
        summary,
    )


def _dense_transport_consistency_audit_for_points(
    mutual_attention: torch.Tensor,
    src_cells: torch.Tensor,
    proposal_pixels: torch.Tensor,
    target_size: Sequence[int],
    src_state: FluxReplayState,
    trg_state: FluxReplayState,
    *,
    target_points: Sequence[Sequence[float]] | None,
    pck_threshold: float | None,
    topks: Sequence[int],
) -> list[dict[str, Any]]:
    """Audit whether all source-token attention rows support a candidate transport field."""

    target_h, target_w = int(target_size[0]), int(target_size[1])
    topks = tuple(sorted({max(1, int(k)) for k in topks}))
    max_topk = min(max(topks), int(mutual_attention.shape[1])) if topks else 1
    src_cells = src_cells.to(device=mutual_attention.device, dtype=torch.long).flatten()
    proposal_pixels = proposal_pixels.to(device=mutual_attention.device, dtype=torch.long)
    if src_cells.numel() == 0 or proposal_pixels.numel() == 0:
        return []

    rows_attn = torch.nan_to_num(mutual_attention.float(), nan=0.0, posinf=0.0, neginf=0.0)
    source_count, target_count = int(rows_attn.shape[0]), int(rows_attn.shape[1])
    src_w, src_h = int(src_state.image_width), int(src_state.image_height)
    trg_w, trg_h = int(trg_state.image_width), int(trg_state.image_height)
    query_count = int(src_cells.shape[0])
    candidate_cells = _pixel_indices_to_replay_cells(
        proposal_pixels.reshape(-1),
        target_size,
        trg_state,
    ).reshape(query_count, -1)
    candidate_count = int(candidate_cells.shape[1])

    source_indices = torch.arange(source_count, device=rows_attn.device, dtype=torch.long)
    source_x = (source_indices % src_w).float()
    source_y = torch.div(source_indices, src_w, rounding_mode="floor").float()
    query_x = (src_cells % src_w).float()
    query_y = torch.div(src_cells, src_w, rounding_mode="floor").float()
    cand_x = (candidate_cells % trg_w).float()
    cand_y = torch.div(candidate_cells, trg_w, rounding_mode="floor").float()
    scale_x = float(trg_w) / float(max(1, src_w))
    scale_y = float(trg_h) / float(max(1, src_h))
    expected_x = cand_x[:, :, None] + torch.round((source_x[None, None, :] - query_x[:, None, None]) * scale_x)
    expected_y = cand_y[:, :, None] + torch.round((source_y[None, None, :] - query_y[:, None, None]) * scale_y)
    valid = (expected_x >= 0) & (expected_x < trg_w) & (expected_y >= 0) & (expected_y < trg_h)
    expected_cells = (
        expected_y.clamp(0, trg_h - 1).long() * trg_w
        + expected_x.clamp(0, trg_w - 1).long()
    )
    gather_sources = source_indices.reshape(1, 1, source_count).expand(query_count, candidate_count, source_count)
    aligned = rows_attn[gather_sources.reshape(-1), expected_cells.reshape(-1)].reshape(
        query_count,
        candidate_count,
        source_count,
    )
    row_peak = rows_attn.max(dim=1).values.clamp_min(1e-12)
    col_peak = rows_attn.max(dim=0).values.clamp_min(1e-12)
    row_norm = aligned / row_peak.reshape(1, 1, source_count)
    col_norm = aligned / col_peak[expected_cells].clamp_min(1e-12)
    reciprocal = torch.sqrt((row_norm * col_norm).clamp_min(0.0))
    valid_float = valid.float()
    valid_count = valid_float.sum(dim=2).clamp_min(1.0)
    dense_row_support = (row_norm * valid_float).sum(dim=2) / valid_count
    dense_reciprocal_support = (reciprocal * valid_float).sum(dim=2) / valid_count

    target_indices = torch.arange(target_count, device=rows_attn.device, dtype=torch.long)
    target_x = (target_indices % trg_w).float()
    target_y = torch.div(target_indices, trg_w, rounding_mode="floor").float()
    row_mass = rows_attn.sum(dim=1).clamp_min(1e-12)
    bary_x = (rows_attn * target_x.reshape(1, -1)).sum(dim=1) / row_mass
    bary_y = (rows_attn * target_y.reshape(1, -1)).sum(dim=1) / row_mass
    target_diag = max(1.0, float((trg_w * trg_w + trg_h * trg_h) ** 0.5))
    bary_dist = torch.sqrt(
        (bary_x.reshape(1, 1, source_count) - expected_x).square()
        + (bary_y.reshape(1, 1, source_count) - expected_y).square()
    )
    dense_barycenter_consistency = ((1.0 / (1.0 + bary_dist / target_diag)) * valid_float).sum(dim=2) / valid_count

    top1_cells = rows_attn.argmax(dim=1)
    top1_x = (top1_cells % trg_w).float()
    top1_y = torch.div(top1_cells, trg_w, rounding_mode="floor").float()
    top1_dist = torch.sqrt(
        (top1_x.reshape(1, 1, source_count) - expected_x).square()
        + (top1_y.reshape(1, 1, source_count) - expected_y).square()
    )
    dense_top1_flow_consistency = ((1.0 / (1.0 + top1_dist / target_diag)) * valid_float).sum(dim=2) / valid_count

    top_cells = torch.topk(rows_attn, k=max_topk, dim=1, sorted=True).indices
    dense_topk_supports: dict[int, torch.Tensor] = {}
    for topk in topks:
        top = top_cells[:, : min(int(topk), max_topk)]
        hit = (top.reshape(1, 1, source_count, -1) == expected_cells[:, :, :, None]).any(dim=3).float()
        dense_topk_supports[int(topk)] = (hit * valid_float).sum(dim=2) / valid_count

    score_names = [
        "dense_transport_row_support",
        "dense_transport_reciprocal_support",
        "dense_barycenter_flow_consistency",
        "dense_top1_flow_consistency",
    ]
    for topk in topks:
        score_names.append(f"dense_attention_top{int(topk)}_field_support")
    score_names.append("hybrid_dense_transport_consistency")

    rows: list[dict[str, Any]] = []
    for row in range(query_count):
        signal_scores: dict[str, list[float]] = {
            "dense_transport_row_support": [
                float(value) for value in dense_row_support[row].detach().cpu().tolist()
            ],
            "dense_transport_reciprocal_support": [
                float(value) for value in dense_reciprocal_support[row].detach().cpu().tolist()
            ],
            "dense_barycenter_flow_consistency": [
                float(value) for value in dense_barycenter_consistency[row].detach().cpu().tolist()
            ],
            "dense_top1_flow_consistency": [
                float(value) for value in dense_top1_flow_consistency[row].detach().cpu().tolist()
            ],
        }
        for topk in topks:
            signal_scores[f"dense_attention_top{int(topk)}_field_support"] = [
                float(value) for value in dense_topk_supports[int(topk)][row].detach().cpu().tolist()
            ]
        rank_base_names = [
            "dense_transport_row_support",
            "dense_transport_reciprocal_support",
            "dense_barycenter_flow_consistency",
            "dense_top1_flow_consistency",
        ]
        if topks:
            rank_base_names.append(f"dense_attention_top{int(max(topks))}_field_support")
        rank_positions = {
            name: {
                index: rank
                for rank, index in enumerate(
                    sorted(range(candidate_count), key=lambda idx: signal_scores[name][idx], reverse=True),
                    start=1,
                )
            }
            for name in rank_base_names
        }
        signal_scores["hybrid_dense_transport_consistency"] = [
            -float(sum(rank_positions[name][index] for name in rank_base_names)) / float(len(rank_base_names))
            for index in range(candidate_count)
        ]

        candidates = []
        hits = []
        seen_pixels: set[int] = set()
        target = target_points[row] if target_points is not None else None
        for index in range(candidate_count):
            pixel = int(proposal_pixels[row, index].detach().cpu())
            if pixel in seen_pixels:
                continue
            seen_pixels.add(pixel)
            xy = [int(pixel % target_w), int(pixel // target_w)]
            hit = bool(_point_hit(xy, target, pck_threshold)) if target is not None and pck_threshold is not None else False
            hits.append(hit)
            valid_fraction = float(valid_float[row, index].mean().detach().cpu())
            metrics = {
                "valid_source_fraction": valid_fraction,
                "valid_source_count": int(valid_float[row, index].sum().detach().cpu()),
                "target_grid_diagonal": float(target_diag),
            }
            candidates.append({
                "rank_attention": int(index + 1),
                "pixel": xy,
                "pixel_index": int(pixel),
                "pck_hit": hit,
                "replay_cell": int(candidate_cells[row, index].detach().cpu()),
                "scores": {name: float(signal_scores[name][index]) for name in score_names},
                "metrics": metrics,
            })
        compact_scores = {
            name: [
                float(candidate["scores"][name])
                for candidate in candidates
            ]
            for name in score_names
        }
        ranks = {name: _rank_first_hit(values, hits) for name, values in compact_scores.items()}
        score_gaps = {}
        for name, values in compact_scores.items():
            top1 = values[0] if values else None
            hit_values = [value for value, hit in zip(values, hits) if hit]
            best_hit = max(hit_values) if hit_values else None
            score_gaps[f"{name}_attention_top1_minus_best_pck_hit_proposal"] = (
                float(top1 - best_hit) if top1 is not None and best_hit is not None else None
            )
        rows.append({
            "dense_source_scope": "all_tokens",
            "transport_model": "candidate_centered_scaled_translation",
            "dense_transport_topk": [int(topk) for topk in topks],
            "proposal_count": len(candidates),
            "score_names": list(score_names),
            "ranks": ranks,
            "score_gaps": score_gaps,
            "candidates": candidates,
        })
    return rows


def _attention_kernel_audit_for_points(
    mutual_attention: torch.Tensor,
    src_cells: torch.Tensor,
    target_size: Sequence[int],
    src_state: FluxReplayState,
    trg_state: FluxReplayState,
    *,
    target_points: Sequence[Sequence[float]] | None,
    pck_threshold: float | None,
    topks: Sequence[int],
    radius: int,
) -> list[dict[str, Any]]:
    """Audit raw vs locally transport-filtered attention kernels."""

    target_h, target_w = int(target_size[0]), int(target_size[1])
    topks = tuple(sorted({max(1, int(k)) for k in topks}))
    max_k = min(max(topks), int(mutual_attention.shape[1])) if topks else 1
    src_cells = src_cells.to(device=mutual_attention.device, dtype=torch.long).flatten()
    raw_rows = mutual_attention[src_cells].float()
    filtered_rows = raw_rows * _local_transport_support_rows(
        mutual_attention,
        src_cells,
        src_state,
        trg_state,
        radius=radius,
    )
    raw_top = torch.topk(raw_rows, k=max_k, dim=1, sorted=True)
    filtered_top = torch.topk(filtered_rows, k=max_k, dim=1, sorted=True)

    def _cell_to_pixel(cell: int) -> list[int]:
        x = (int(cell) % int(trg_state.image_width) + 0.5) * float(target_w) / float(trg_state.image_width) - 0.5
        y = (int(cell) // int(trg_state.image_width) + 0.5) * float(target_h) / float(trg_state.image_height) - 0.5
        return [
            int(round(max(0.0, min(float(target_w - 1), x)))),
            int(round(max(0.0, min(float(target_h - 1), y)))),
        ]

    rows: list[dict[str, Any]] = []
    for row_idx in range(src_cells.shape[0]):
        target = target_points[row_idx] if target_points is not None else None
        audit = {
            "radius": int(max(0, radius)),
            "score_names": ["raw_attention", "filtered_attention"],
            "ranks": {},
            "topk_hits": {},
            "top1": {},
        }
        for name, values, indices in (
            ("raw_attention", raw_top.values[row_idx], raw_top.indices[row_idx]),
            ("filtered_attention", filtered_top.values[row_idx], filtered_top.indices[row_idx]),
        ):
            hit_rank = None
            decoded_hits = []
            for rank_idx, (score, cell) in enumerate(
                zip(values.detach().cpu().tolist(), indices.detach().cpu().tolist()),
                start=1,
            ):
                pixel = _cell_to_pixel(int(cell))
                hit = (
                    bool(_point_hit(pixel, target, pck_threshold))
                    if target is not None and pck_threshold is not None
                    else False
                )
                if hit and hit_rank is None:
                    hit_rank = int(rank_idx)
                if rank_idx == 1:
                    audit["top1"][name] = {
                        "cell": int(cell),
                        "pixel": pixel,
                        "score": float(score),
                        "pck_hit": bool(hit),
                    }
                decoded_hits.append(hit)
            audit["ranks"][name] = hit_rank
            for k in topks:
                audit["topk_hits"][f"{name}@{k}"] = bool(any(decoded_hits[: min(int(k), len(decoded_hits))]))
        rows.append(audit)
    return rows


def _rank_normalize(values: torch.Tensor) -> torch.Tensor:
    values = torch.nan_to_num(values.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if values.numel() == 0:
        return values
    lo = values.min()
    hi = values.max()
    if float((hi - lo).detach().cpu()) <= 1e-12:
        return torch.ones_like(values)
    return (values - lo) / (hi - lo).clamp_min(1e-12)


def _weighted_affine_prediction(
    source_anchors: torch.Tensor,
    target_anchors: torch.Tensor,
    weights: torch.Tensor,
    query_source: torch.Tensor,
) -> tuple[torch.Tensor, float, bool]:
    valid = torch.isfinite(source_anchors).all(dim=1) & torch.isfinite(target_anchors).all(dim=1)
    valid = valid & torch.isfinite(weights) & (weights > 0)
    if int(valid.sum().detach().cpu()) < 3:
        return target_anchors.new_zeros(2), 0.0, False
    src = source_anchors[valid].float()
    trg = target_anchors[valid].float()
    w = weights[valid].float().clamp_min(0.0)
    sqrt_w = w.sqrt().unsqueeze(1)
    ones = torch.ones((src.shape[0], 1), device=src.device, dtype=src.dtype)
    design = torch.cat((src, ones), dim=1)
    weighted_design = design * sqrt_w
    weighted_target = trg * sqrt_w
    try:
        beta = torch.linalg.lstsq(weighted_design, weighted_target).solution
    except RuntimeError:
        return target_anchors.new_zeros(2), 0.0, False
    predicted_anchors = design @ beta
    residual = (predicted_anchors - trg).norm(dim=1)
    scale = float(((residual.square() * w).sum() / w.sum().clamp_min(1e-12)).sqrt().detach().cpu())
    query_design = torch.cat((query_source.float(), query_source.new_ones(1).float()), dim=0).reshape(1, 3)
    predicted = (query_design @ beta).reshape(2)
    return predicted, scale, True


def _pairwise_topology_scores(
    query_source: torch.Tensor,
    candidate_targets: torch.Tensor,
    source_anchors: torch.Tensor,
    target_anchors: torch.Tensor,
    weights: torch.Tensor,
    source_size: Sequence[int],
    target_size: Sequence[int],
) -> torch.Tensor:
    valid = torch.isfinite(source_anchors).all(dim=1) & torch.isfinite(target_anchors).all(dim=1)
    valid = valid & torch.isfinite(weights) & (weights > 0)
    if int(valid.sum().detach().cpu()) <= 0:
        return torch.zeros(candidate_targets.shape[0], device=candidate_targets.device)
    src_vec = source_anchors[valid].float() - query_source.float().reshape(1, 2)
    trg_anchor = target_anchors[valid].float()
    w = weights[valid].float().clamp_min(0.0)
    source_diag = float((float(source_size[0]) ** 2 + float(source_size[1]) ** 2) ** 0.5)
    target_diag = float((float(target_size[0]) ** 2 + float(target_size[1]) ** 2) ** 0.5)
    src_dist = src_vec.norm(dim=1).clamp_min(1.0) / max(source_diag, 1.0)
    src_unit = src_vec / src_vec.norm(dim=1, keepdim=True).clamp_min(1e-6)
    candidate_scores = []
    for candidate in candidate_targets.float():
        trg_vec = trg_anchor - candidate.reshape(1, 2)
        trg_dist = trg_vec.norm(dim=1).clamp_min(1.0) / max(target_diag, 1.0)
        trg_unit = trg_vec / trg_vec.norm(dim=1, keepdim=True).clamp_min(1e-6)
        distance_score = 1.0 / (1.0 + (trg_dist.clamp_min(1e-6).log() - src_dist.clamp_min(1e-6).log()).abs())
        direction_score = ((src_unit * trg_unit).sum(dim=1).clamp(-1.0, 1.0) + 1.0) * 0.5
        order_x = (torch.sign(src_vec[:, 0]) == torch.sign(trg_vec[:, 0])).float()
        order_y = (torch.sign(src_vec[:, 1]) == torch.sign(trg_vec[:, 1])).float()
        order_score = 0.5 * (order_x + order_y)
        score = (distance_score * direction_score * order_score).clamp_min(0.0)
        candidate_scores.append((score * w).sum() / w.sum().clamp_min(1e-12))
    return torch.stack(candidate_scores, dim=0) if candidate_scores else torch.zeros(0, device=candidate_targets.device)


def _anchor_topology_scores_for_candidates(
    query_source: torch.Tensor,
    candidate_targets: torch.Tensor,
    source_anchors: torch.Tensor,
    target_anchors: torch.Tensor,
    weights: torch.Tensor,
    source_size: Sequence[int],
    target_size: Sequence[int],
) -> torch.Tensor:
    """Topology score used by both the audit and the rescue matcher."""

    if candidate_targets.numel() == 0:
        return torch.zeros(0, device=candidate_targets.device)
    affine_prediction, affine_residual_scale, affine_valid = _weighted_affine_prediction(
        source_anchors,
        target_anchors,
        weights,
        query_source,
    )
    if affine_valid:
        affine_distance = (candidate_targets.float() - affine_prediction.reshape(1, 2)).norm(dim=1)
        affine_scores = 1.0 / (1.0 + affine_distance / max(float(affine_residual_scale), 1.0))
    else:
        affine_scores = torch.zeros(candidate_targets.shape[0], device=candidate_targets.device)
    pairwise_scores = _pairwise_topology_scores(
        query_source,
        candidate_targets,
        source_anchors,
        target_anchors,
        weights,
        source_size,
        target_size,
    )
    return torch.sqrt((affine_scores.clamp_min(0.0) * pairwise_scores.clamp_min(0.0)).clamp_min(0.0))


def _anchor_topology_audit_for_points(
    source_points: Sequence[Sequence[float]],
    source_size: Sequence[int],
    target_size: Sequence[int],
    proposal_pixels: torch.Tensor,
    native_indices: torch.Tensor,
    native_scores: torch.Tensor,
    *,
    target_points: Sequence[Sequence[float]] | None,
    pck_threshold: float | None,
) -> list[dict[str, Any]]:
    point_count = len(source_points)
    if point_count <= 0:
        return []
    device = proposal_pixels.device
    target_h, target_w = int(target_size[0]), int(target_size[1])
    source_xy = torch.tensor(
        [[float(point[0]), float(point[1])] for point in source_points],
        device=device,
        dtype=torch.float32,
    )
    native_top1 = native_indices[:, 0].long().to(device)
    native_xy = torch.stack(
        ((native_top1 % target_w).float(), torch.div(native_top1, target_w, rounding_mode="floor").float()),
        dim=1,
    )
    native_top1_score = native_scores[:, 0].float().to(device)
    native_top2_score = (
        native_scores[:, 1].float().to(device)
        if native_scores.shape[1] > 1
        else native_scores[:, 0].float().to(device)
    )
    margin = (native_top1_score - native_top2_score).clamp_min(0.0)
    duplicate_counts = torch.bincount(native_top1, minlength=target_h * target_w).float().to(device)
    uniqueness = 1.0 / duplicate_counts[native_top1].clamp_min(1.0)
    confidence = _rank_normalize(margin) * _rank_normalize(native_top1_score) * uniqueness
    if point_count == 1:
        confidence = torch.zeros_like(confidence)

    rows: list[dict[str, Any]] = []
    score_names = ["anchor_affine_topology", "anchor_pairwise_topology", "hybrid_anchor_topology"]
    for row in range(point_count):
        candidate_pixels = proposal_pixels[row].long().to(device)
        candidate_xy = torch.stack(
            (
                (candidate_pixels % target_w).float(),
                torch.div(candidate_pixels, target_w, rounding_mode="floor").float(),
            ),
            dim=1,
        )
        anchor_mask = torch.ones(point_count, device=device, dtype=torch.bool)
        anchor_mask[row] = False
        anchor_weights = confidence[anchor_mask]
        source_anchors = source_xy[anchor_mask]
        target_anchors = native_xy[anchor_mask]
        positive_anchor_count = int((anchor_weights > 0).sum().detach().cpu())
        weight_sum = float(anchor_weights.sum().detach().cpu())
        effective_anchor_count = float(
            (anchor_weights.sum().square() / anchor_weights.square().sum().clamp_min(1e-12)).detach().cpu()
        ) if positive_anchor_count > 0 else 0.0

        affine_prediction, affine_residual_scale, affine_valid = _weighted_affine_prediction(
            source_anchors,
            target_anchors,
            anchor_weights,
            source_xy[row],
        )
        if affine_valid:
            affine_distance = (candidate_xy.float() - affine_prediction.reshape(1, 2)).norm(dim=1)
            affine_scores = 1.0 / (1.0 + affine_distance / max(float(affine_residual_scale), 1.0))
        else:
            affine_scores = torch.zeros(candidate_pixels.shape[0], device=device)
        pairwise_scores = _pairwise_topology_scores(
            source_xy[row],
            candidate_xy,
            source_anchors,
            target_anchors,
            anchor_weights,
            source_size,
            target_size,
        )
        hybrid_scores = _anchor_topology_scores_for_candidates(
            source_xy[row],
            candidate_xy,
            source_anchors,
            target_anchors,
            anchor_weights,
            source_size,
            target_size,
        )
        score_tensors = {
            "anchor_affine_topology": affine_scores,
            "anchor_pairwise_topology": pairwise_scores,
            "hybrid_anchor_topology": hybrid_scores,
        }
        target = target_points[row] if target_points is not None else None
        hits = [
            bool(_point_hit([int(pixel % target_w), int(pixel // target_w)], target, pck_threshold))
            if target is not None and pck_threshold is not None
            else False
            for pixel in candidate_pixels.detach().cpu().tolist()
        ]
        ranks = {
            name: _rank_first_hit([float(value) for value in tensor.detach().cpu().tolist()], hits)
            for name, tensor in score_tensors.items()
        }
        score_gaps = {}
        for name, tensor in score_tensors.items():
            values = [float(value) for value in tensor.detach().cpu().tolist()]
            top1 = values[0] if values else None
            hit_values = [value for value, hit in zip(values, hits) if hit]
            best_hit = max(hit_values) if hit_values else None
            score_gaps[f"{name}_attention_top1_minus_best_pck_hit_proposal"] = (
                float(top1 - best_hit) if top1 is not None and best_hit is not None else None
            )
        candidates = []
        for rank, pixel in enumerate(candidate_pixels.detach().cpu().tolist(), start=1):
            pixel = int(pixel)
            candidates.append({
                "rank_attention": int(rank),
                "pixel": [int(pixel % target_w), int(pixel // target_w)],
                "pixel_index": int(pixel),
                "pck_hit": bool(hits[rank - 1]),
                "scores": {
                    name: float(score_tensors[name][rank - 1].detach().cpu())
                    for name in score_names
                },
            })
        anchor_conf = [float(value) for value in anchor_weights.detach().cpu().tolist()]
        anchor_conf_sorted = sorted(anchor_conf)
        median_conf = (
            float(anchor_conf_sorted[len(anchor_conf_sorted) // 2])
            if anchor_conf_sorted else 0.0
        )
        rows.append({
            "score_names": score_names,
            "ranks": ranks,
            "score_gaps": score_gaps,
            "anchor_count": int(max(0, point_count - 1)),
            "positive_anchor_count": positive_anchor_count,
            "effective_anchor_count": effective_anchor_count,
            "anchor_confidence": {
                "sum": weight_sum,
                "mean": float(weight_sum / max(1, len(anchor_conf))),
                "median": median_conf,
                "max": float(max(anchor_conf) if anchor_conf else 0.0),
            },
            "affine_valid": bool(affine_valid),
            "affine_prediction": [
                float(affine_prediction[0].detach().cpu()) if affine_valid else None,
                float(affine_prediction[1].detach().cpu()) if affine_valid else None,
            ],
            "affine_residual_scale": float(affine_residual_scale),
            "candidates": candidates,
        })
    return rows


def _native_preserving_topology_rescue_rankings(
    src_features: torch.Tensor,
    trg_features: torch.Tensor,
    attention: dict[str, torch.Tensor],
    points: Sequence[Sequence[float]],
    source_size: Sequence[int],
    target_size: Sequence[int],
    src_state: FluxReplayState,
    trg_state: FluxReplayState,
    *,
    candidate_topk: int,
    target_points: Sequence[Sequence[float]] | None = None,
    pck_threshold: float | None = None,
) -> tuple[torch.Tensor, list[dict[str, Any]], dict[str, float]]:
    """Jointly preserve native identity and rescue via anchor topology."""

    points = list(points)
    if not points:
        return torch.empty((0, 0), dtype=torch.long), [], {}
    device = src_features.device
    src_cells = _native_cell_indices_for_points(
        points,
        source_size,
        src_state.image_height,
        src_state.image_width,
        attention["p_ab"].device,
    )
    mutual_attention = torch.sqrt((attention["p_ab"].float() * attention["p_ba"].float().t()).clamp_min(0.0))
    mutual_attention = torch.nan_to_num(mutual_attention, nan=0.0, posinf=0.0, neginf=0.0)
    candidate_count = min(max(1, int(candidate_topk)), mutual_attention.shape[1])
    attn_values, candidate_cells = torch.topk(
        mutual_attention[src_cells],
        k=candidate_count,
        dim=1,
        sorted=True,
    )
    target_h, target_w = int(target_size[0]), int(target_size[1])
    cell_x = (candidate_cells % trg_state.image_width).float()
    cell_y = torch.div(candidate_cells, trg_state.image_width, rounding_mode="floor").float()
    proposal_x = torch.round((cell_x + 0.5) * float(target_w) / float(trg_state.image_width) - 0.5).long()
    proposal_y = torch.round((cell_y + 0.5) * float(target_h) / float(trg_state.image_height) - 0.5).long()
    proposal_x.clamp_(0, target_w - 1)
    proposal_y.clamp_(0, target_h - 1)
    proposal_pixels = (proposal_y.to(device) * target_w + proposal_x.to(device)).long()

    native_scores, native_indices = _chunked_descriptor_topk_scores_indices(
        src_features,
        trg_features,
        points,
        source_size,
        target_size,
        topk=max(2, candidate_count),
    )
    native_top1_pixels = native_indices[:, 0].long()
    native_top1_scores = native_scores[:, 0].float()
    native_top2_scores = native_scores[:, 1].float() if native_scores.shape[1] > 1 else native_top1_scores

    native_xy_all = torch.stack(
        (
            (native_top1_pixels % target_w).float(),
            torch.div(native_top1_pixels, target_w, rounding_mode="floor").float(),
        ),
        dim=1,
    )
    native_margin = (native_top1_scores - native_top2_scores).clamp_min(0.0)
    duplicate_counts = torch.bincount(native_top1_pixels, minlength=target_h * target_w).float().to(device)
    confidence = _rank_normalize(native_margin.to(device)) * _rank_normalize(native_top1_scores.to(device)) * (
        1.0 / duplicate_counts[native_top1_pixels.to(device)].clamp_min(1.0)
    )

    ranked_pixels: list[list[int]] = []
    audits: list[dict[str, Any]] = []
    selected_top1_support: list[float] = []
    native_candidate_support: list[float] = []
    native_keep_count = 0
    rescue_count = 0
    source_xy = torch.tensor(
        [[float(point[0]), float(point[1])] for point in points],
        device=device,
        dtype=torch.float32,
    )
    for row in range(len(points)):
        candidate_list = [int(native_top1_pixels[row].item())]
        seen = {candidate_list[0]}
        for pixel in proposal_pixels[row].detach().cpu().tolist():
            pixel = int(pixel)
            if pixel not in seen:
                candidate_list.append(pixel)
                seen.add(pixel)
        candidate_tensor = torch.tensor([candidate_list], device=device, dtype=torch.long)
        desc_scores, desc_pixels = _chunked_descriptor_topk_scores_indices(
            src_features,
            trg_features,
            [points[row]],
            source_size,
            target_size,
            topk=len(candidate_list),
            candidate_indices=candidate_tensor,
        )
        desc_pixels_row = [int(pixel) for pixel in desc_pixels[0].detach().cpu().tolist()]
        desc_scores_row = [float(score) for score in desc_scores[0].detach().cpu().tolist()]
        desc_by_pixel = {pixel: score for pixel, score in zip(desc_pixels_row, desc_scores_row)}
        anchor_mask = torch.ones(len(points), device=device, dtype=torch.bool)
        anchor_mask[row] = False
        source_anchors = source_xy[anchor_mask]
        target_anchors = native_xy_all[anchor_mask]
        anchor_weights = confidence[anchor_mask]
        candidate_xy = torch.stack(
            (
                (candidate_tensor[0] % target_w).float(),
                torch.div(candidate_tensor[0], target_w, rounding_mode="floor").float(),
            ),
            dim=1,
        )
        topology_scores = _anchor_topology_scores_for_candidates(
            source_xy[row],
            candidate_xy,
            source_anchors,
            target_anchors,
            anchor_weights,
            source_size,
            target_size,
        )
        descriptor_values = torch.tensor(
            [float(desc_by_pixel.get(pixel, float("-inf"))) for pixel in candidate_list],
            device=device,
            dtype=torch.float32,
        )
        topology_values = topology_scores.to(device=device, dtype=torch.float32)
        descriptor_norm = _rank_normalize(descriptor_values)
        topology_norm = _rank_normalize(topology_values)
        native_gate = confidence[row].to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
        combined_scores = (native_gate * descriptor_norm) + ((1.0 - native_gate) * topology_norm)
        order = sorted(
            range(len(candidate_list)),
            key=lambda index: float(combined_scores[index].detach().cpu()),
            reverse=True,
        )
        ranked_candidates = [candidate_list[index] for index in order]
        ranked_pixels.append(ranked_candidates)

        target = target_points[row] if target_points is not None else None
        candidates: list[dict[str, Any]] = []
        native_candidate_rank = None
        for rank, index in enumerate(order, start=1):
            pixel = int(candidate_list[index])
            xy = [int(pixel % target_w), int(pixel // target_w)]
            hit = (
                bool(_point_hit(xy, target, pck_threshold))
                if target is not None and pck_threshold is not None
                else False
            )
            if pixel == int(native_top1_pixels[row].item()):
                native_candidate_rank = int(rank)
            candidates.append({
                "rank": int(rank),
                "pixel": xy,
                "pixel_index": int(pixel),
                "descriptor_score": float(descriptor_values[index].detach().cpu()),
                "descriptor_score_norm": float(descriptor_norm[index].detach().cpu()),
                "topology_score": float(topology_values[index].detach().cpu()),
                "topology_score_norm": float(topology_norm[index].detach().cpu()),
                "combined_score": float(combined_scores[index].detach().cpu()),
                "native_preserve_gate": float(native_gate.detach().cpu()),
                "native_candidate": bool(pixel == int(native_top1_pixels[row].item())),
                "attention_rank": (
                    int(next((attn_rank for attn_rank, attn_pixel in enumerate(proposal_pixels[row].detach().cpu().tolist(), start=1) if int(attn_pixel) == pixel), 0))
                    if pixel in proposal_pixels[row].detach().cpu().tolist()
                    else None
                ),
                "pck_hit": hit,
            })
        selected_pixel = int(ranked_candidates[0])
        if selected_pixel == int(native_top1_pixels[row].item()):
            native_keep_count += 1
        else:
            rescue_count += 1
        selected_top1_support.append(float(combined_scores[order[0]].detach().cpu()))
        native_candidate_support.append(float(combined_scores[0].detach().cpu()))
        hits = [bool(item["pck_hit"]) for item in candidates]
        combined_scores_list = [float(value) for value in combined_scores.detach().cpu().tolist()]
        ranks = {"native_preserving_topology_rescue": _rank_first_hit(combined_scores_list, hits)}
        score_gaps = {
            "native_preserving_topology_rescue_attention_top1_minus_best_pck_hit_proposal": (
                float(combined_scores_list[0] - max(value for value, hit in zip(combined_scores_list, hits) if hit))
                if any(hits) else None
            )
        }
        audits.append({
            "candidate_count": len(candidate_list),
            "native_candidate_rank": native_candidate_rank,
            "native_keep": bool(selected_pixel == int(native_top1_pixels[row].item())),
            "native_candidate_score": float(descriptor_values[0].detach().cpu()),
            "native_candidate_topology": float(topology_values[0].detach().cpu()),
            "native_preserve_gate": float(native_gate.detach().cpu()),
            "score_names": [
                "descriptor_score",
                "topology_score",
                "combined_score",
            ],
            "ranks": ranks,
            "score_gaps": score_gaps,
            "candidates": candidates,
        })

    max_len = max((len(row) for row in ranked_pixels), default=0)
    padded = []
    for row in ranked_pixels:
        if len(row) < max_len:
            row = row + row[-1:] * (max_len - len(row)) if row else [0] * max_len
        padded.append(row)
    ranked_tensor = torch.tensor(padded, device=device, dtype=torch.long)
    summary = {
        "candidate_pool_mean": float(sum(len(row) for row in ranked_pixels) / max(1, len(ranked_pixels))),
        "native_keep_rate": float(native_keep_count / max(1, len(points))),
        "rescue_rate": float(rescue_count / max(1, len(points))),
        "native_confidence_mean": float(confidence.mean().detach().cpu()),
        "selected_top1_support_mean": float(sum(selected_top1_support) / max(1, len(selected_top1_support))),
        "native_candidate_support_mean": float(sum(native_candidate_support) / max(1, len(native_candidate_support))),
        "combined_top1_support_mean": float(sum(selected_top1_support) / max(1, len(selected_top1_support))),
    }
    return ranked_tensor, audits, summary


def _attention_basin_native_refine_rankings(
    src_features: torch.Tensor,
    trg_features: torch.Tensor,
    attention: dict[str, torch.Tensor],
    points: Sequence[Sequence[float]],
    source_size: Sequence[int],
    target_size: Sequence[int],
    src_state: FluxReplayState,
    trg_state: FluxReplayState,
    *,
    candidate_topk: int,
) -> torch.Tensor:
    """Use attention as a region proposal and native descriptors for dense point refinement."""

    points = list(points)
    if not points:
        return torch.empty((0, 0), device=src_features.device, dtype=torch.long)
    device = src_features.device
    src_cells = _native_cell_indices_for_points(
        points,
        source_size,
        src_state.image_height,
        src_state.image_width,
        attention["p_ab"].device,
    )
    mutual_attention = torch.sqrt((attention["p_ab"].float() * attention["p_ba"].float().t()).clamp_min(0.0))
    mutual_attention = torch.nan_to_num(mutual_attention, nan=0.0, posinf=0.0, neginf=0.0)
    candidate_count = min(max(1, int(candidate_topk)), int(mutual_attention.shape[1]))
    candidate_cells = torch.topk(
        mutual_attention[src_cells],
        k=candidate_count,
        dim=1,
        sorted=True,
    ).indices
    target_h, target_w = int(target_size[0]), int(target_size[1])
    trg_grid_h, trg_grid_w = int(trg_state.image_height), int(trg_state.image_width)
    candidate_rows: list[list[int]] = []
    max_count = 0
    for cells in candidate_cells.detach().cpu().tolist():
        pixels: list[int] = []
        seen: set[int] = set()
        for cell in cells:
            cell = int(cell)
            cell_x = cell % trg_grid_w
            cell_y = cell // trg_grid_w
            x0 = int(math.floor(float(cell_x) * float(target_w) / float(trg_grid_w)))
            x1 = int(math.ceil(float(cell_x + 1) * float(target_w) / float(trg_grid_w)))
            y0 = int(math.floor(float(cell_y) * float(target_h) / float(trg_grid_h)))
            y1 = int(math.ceil(float(cell_y + 1) * float(target_h) / float(trg_grid_h)))
            x0 = max(0, min(target_w - 1, x0))
            x1 = max(x0 + 1, min(target_w, x1))
            y0 = max(0, min(target_h - 1, y0))
            y1 = max(y0 + 1, min(target_h, y1))
            for y in range(y0, y1):
                base = y * target_w
                for x in range(x0, x1):
                    pixel = base + x
                    if pixel not in seen:
                        seen.add(pixel)
                        pixels.append(pixel)
        if not pixels:
            pixels = [0]
        candidate_rows.append(pixels)
        max_count = max(max_count, len(pixels))
    padded = [
        row + row[-1:] * (max_count - len(row))
        for row in candidate_rows
    ]
    candidate_pixels = torch.tensor(padded, device=device, dtype=torch.long)
    _scores, ranked_pixels = _chunked_descriptor_topk_scores_indices(
        src_features,
        trg_features,
        points,
        source_size,
        target_size,
        topk=max_count,
        candidate_indices=candidate_pixels,
    )
    return ranked_pixels


def _candidate_conditioned_verification_rankings(
    src_features: torch.Tensor,
    trg_features: torch.Tensor,
    attention: dict[str, torch.Tensor],
    points: Sequence[Sequence[float]],
    source_size: Sequence[int],
    target_size: Sequence[int],
    src_state: FluxReplayState,
    trg_state: FluxReplayState,
    *,
    candidate_topk: int,
    target_points: Sequence[Sequence[float]] | None = None,
    pck_threshold: float | None = None,
) -> tuple[torch.Tensor, list[dict[str, Any]], dict[str, float]]:
    """Verify each attention proposal as an explicit correspondence hypothesis.

    Cross-attention is treated only as a high-recall proposal generator.  Each
    target candidate is then scored by independent evidence: posterior support,
    native point identity, local relational identity, and weak object topology
    from native-attention consensus anchors.  The final order is an equal-rank
    vote over those evidence sources, so no hand-tuned score weights are needed.
    """

    points = list(points)
    if not points:
        return torch.empty((0, 0), device=src_features.device, dtype=torch.long), [], {}
    device = src_features.device
    target_h, target_w = int(target_size[0]), int(target_size[1])
    src_cells = _native_cell_indices_for_points(
        points,
        source_size,
        src_state.image_height,
        src_state.image_width,
        attention["p_ab"].device,
    )
    mutual_attention = torch.sqrt(
        (attention["p_ab"].float() * attention["p_ba"].float().t()).clamp_min(0.0)
    )
    mutual_attention = torch.nan_to_num(mutual_attention, nan=0.0, posinf=0.0, neginf=0.0)
    candidate_count = min(max(1, int(candidate_topk)), int(mutual_attention.shape[1]))
    attention_values, candidate_cells = torch.topk(
        mutual_attention[src_cells.to(mutual_attention.device)],
        k=candidate_count,
        dim=1,
        sorted=True,
    )
    attention_pixels = _cell_topk_to_pixel_indices(candidate_cells, target_size, trg_state).to(device)

    native_scores, native_indices = _chunked_descriptor_topk_scores_indices(
        src_features,
        trg_features,
        points,
        source_size,
        target_size,
        topk=max(2, candidate_count),
    )
    native_top1 = native_indices[:, 0].long().to(device)
    native_top1_scores = native_scores[:, 0].float().to(device)
    native_top2_scores = (
        native_scores[:, 1].float().to(device)
        if native_scores.shape[1] > 1
        else native_top1_scores
    )
    native_margin = (native_top1_scores - native_top2_scores).clamp_min(0.0)
    duplicate_counts = torch.bincount(native_top1, minlength=target_h * target_w).float().to(device)
    native_uniqueness = 1.0 / duplicate_counts[native_top1].clamp_min(1.0)
    native_cells = _pixel_indices_to_replay_cells(native_top1, target_size, trg_state).to(mutual_attention.device)
    row_max = mutual_attention[src_cells].amax(dim=1).clamp_min(1e-12)
    native_attention_support = (
        mutual_attention[src_cells, native_cells] / row_max
    ).to(device).clamp_min(0.0)
    anchor_confidence = (
        _rank_normalize(native_margin)
        * _rank_normalize(native_top1_scores)
        * native_uniqueness
        * native_attention_support
    )
    if len(points) <= 1:
        anchor_confidence = torch.zeros_like(anchor_confidence)

    candidate_rows: list[list[int]] = []
    attention_score_rows: list[list[float]] = []
    attention_rank_rows: list[list[int | None]] = []
    max_len = 0
    for row in range(len(points)):
        pixel_to_attention: dict[int, tuple[float, int]] = {}
        for rank, (pixel, score) in enumerate(
            zip(attention_pixels[row].detach().cpu().tolist(), attention_values[row].detach().cpu().tolist()),
            start=1,
        ):
            pixel = int(pixel)
            score = float(score)
            if pixel not in pixel_to_attention or score > pixel_to_attention[pixel][0]:
                pixel_to_attention[pixel] = (score, int(rank))
        ordered = list(pixel_to_attention.keys())
        native_pixel = int(native_top1[row].detach().cpu())
        if native_pixel not in pixel_to_attention:
            ordered.append(native_pixel)
        score_row: list[float] = []
        rank_row: list[int | None] = []
        for pixel in ordered:
            if pixel in pixel_to_attention:
                score, rank = pixel_to_attention[pixel]
            else:
                replay_cell = int(_pixel_indices_to_replay_cells(
                    torch.tensor([pixel], device=mutual_attention.device, dtype=torch.long),
                    target_size,
                    trg_state,
                )[0].detach().cpu())
                score = float(mutual_attention[int(src_cells[row].detach().cpu()), replay_cell].detach().cpu())
                rank = None
            score_row.append(score)
            rank_row.append(rank)
        candidate_rows.append(ordered)
        attention_score_rows.append(score_row)
        attention_rank_rows.append(rank_row)
        max_len = max(max_len, len(ordered))
    if max_len <= 0:
        return torch.empty((0, 0), device=device, dtype=torch.long), [], {}
    padded_candidates = [
        row + row[-1:] * (max_len - len(row))
        for row in candidate_rows
    ]
    candidate_tensor = torch.tensor(padded_candidates, device=device, dtype=torch.long)

    descriptor_sorted_scores, descriptor_sorted_pixels = _chunked_descriptor_topk_scores_indices(
        src_features,
        trg_features,
        points,
        source_size,
        target_size,
        topk=max_len,
        candidate_indices=candidate_tensor,
    )
    descriptor_by_row: list[dict[int, float]] = []
    for row in range(len(points)):
        descriptor_by_row.append({
            int(pixel): float(score)
            for pixel, score in zip(
                descriptor_sorted_pixels[row].detach().cpu().tolist(),
                descriptor_sorted_scores[row].detach().cpu().tolist(),
            )
        })

    local_rows = _local_relational_identity_audit_for_points(
        src_features,
        trg_features,
        mutual_attention,
        src_cells,
        points,
        candidate_tensor,
        source_size,
        target_size,
        src_state,
        trg_state,
        target_points=target_points,
        pck_threshold=pck_threshold,
        radius=1,
    )
    local_by_row: list[dict[int, float]] = []
    for row in local_rows:
        local_by_row.append({
            int(candidate["pixel_index"]): float(candidate["scores"]["hybrid_local_relational_identity"])
            for candidate in row.get("candidates", [])
        })
    while len(local_by_row) < len(points):
        local_by_row.append({})

    source_xy = torch.tensor(
        [[float(point[0]), float(point[1])] for point in points],
        device=device,
        dtype=torch.float32,
    )
    native_xy = torch.stack(
        (
            (native_top1 % target_w).float(),
            torch.div(native_top1, target_w, rounding_mode="floor").float(),
        ),
        dim=1,
    )
    ranked_rows: list[list[int]] = []
    audits: list[dict[str, Any]] = []
    native_rank_values: list[float] = []
    selected_attention_ranks: list[float] = []
    selected_is_native = 0
    selected_is_attention = 0
    for row, ordered in enumerate(candidate_rows):
        valid_count = len(ordered)
        candidate_pixels = torch.tensor(ordered, device=device, dtype=torch.long)
        candidate_xy = torch.stack(
            (
                (candidate_pixels % target_w).float(),
                torch.div(candidate_pixels, target_w, rounding_mode="floor").float(),
            ),
            dim=1,
        )
        anchor_mask = torch.ones(len(points), device=device, dtype=torch.bool)
        anchor_mask[row] = False
        topology = _anchor_topology_scores_for_candidates(
            source_xy[row],
            candidate_xy,
            source_xy[anchor_mask],
            native_xy[anchor_mask],
            anchor_confidence[anchor_mask],
            source_size,
            target_size,
        )
        raw_signals = {
            "attention_posterior": torch.tensor(attention_score_rows[row], device=device, dtype=torch.float32),
            "native_identity": torch.tensor(
                [descriptor_by_row[row].get(int(pixel), -float("inf")) for pixel in ordered],
                device=device,
                dtype=torch.float32,
            ),
            "local_relation": torch.tensor(
                [local_by_row[row].get(int(pixel), 0.0) for pixel in ordered],
                device=device,
                dtype=torch.float32,
            ),
            "anchor_topology": topology.to(device=device, dtype=torch.float32),
        }
        rank_positions: dict[str, dict[int, int]] = {}
        reciprocal_vote = torch.zeros(valid_count, device=device, dtype=torch.float32)
        active_signal_names: list[str] = []
        for name, values in raw_signals.items():
            values = torch.nan_to_num(values.float(), nan=-float("inf"), posinf=-float("inf"), neginf=-float("inf"))
            finite = torch.isfinite(values)
            if int(finite.sum().detach().cpu()) <= 0:
                continue
            finite_values = values[finite]
            if float((finite_values.max() - finite_values.min()).detach().cpu()) <= 1e-12:
                continue
            active_signal_names.append(name)
            order = sorted(
                range(valid_count),
                key=lambda index: float(values[index].detach().cpu()),
                reverse=True,
            )
            rank_positions[name] = {index: rank for rank, index in enumerate(order, start=1)}
            for index, rank in rank_positions[name].items():
                reciprocal_vote[index] += 1.0 / float(rank)
        if not active_signal_names:
            active_signal_names = ["attention_posterior"]
            order = list(range(valid_count))
            rank_positions["attention_posterior"] = {index: rank for rank, index in enumerate(order, start=1)}
            for index, rank in rank_positions["attention_posterior"].items():
                reciprocal_vote[index] += 1.0 / float(rank)
        median_rank = torch.tensor(
            [
                sorted(rank_positions[name][index] for name in active_signal_names)[len(active_signal_names) // 2]
                for index in range(valid_count)
            ],
            device=device,
            dtype=torch.float32,
        )
        fused_scores = reciprocal_vote / median_rank.clamp_min(1.0).sqrt()
        fused_order = sorted(
            range(valid_count),
            key=lambda index: (
                float(fused_scores[index].detach().cpu()),
                float(raw_signals["native_identity"][index].detach().cpu()),
                float(raw_signals["local_relation"][index].detach().cpu()),
                float(raw_signals["attention_posterior"][index].detach().cpu()),
            ),
            reverse=True,
        )
        ranked = [int(ordered[index]) for index in fused_order]
        ranked_rows.append(ranked)
        selected_pixel = ranked[0]
        if selected_pixel == int(native_top1[row].detach().cpu()):
            selected_is_native += 1
        if selected_pixel in set(int(pixel) for pixel in attention_pixels[row].detach().cpu().tolist()):
            selected_is_attention += 1
        attention_rank = attention_rank_rows[row][ordered.index(selected_pixel)]
        if attention_rank is not None:
            selected_attention_ranks.append(float(attention_rank))
        native_rank = next(
            (rank for rank, index in enumerate(fused_order, start=1) if int(ordered[index]) == int(native_top1[row].detach().cpu())),
            None,
        )
        if native_rank is not None:
            native_rank_values.append(float(native_rank))
        target = target_points[row] if target_points is not None else None
        candidates: list[dict[str, Any]] = []
        hits: list[bool] = []
        for rank, index in enumerate(fused_order, start=1):
            pixel = int(ordered[index])
            xy = [int(pixel % target_w), int(pixel // target_w)]
            hit = (
                bool(_point_hit(xy, target, pck_threshold))
                if target is not None and pck_threshold is not None
                else False
            )
            hits.append(hit)
            candidates.append({
                "rank": int(rank),
                "pixel": xy,
                "pixel_index": int(pixel),
                "native_candidate": bool(pixel == int(native_top1[row].detach().cpu())),
                "attention_rank": attention_rank_rows[row][index],
                "pck_hit": bool(hit),
                "scores": {
                    "candidate_conditioned_verification": float(fused_scores[index].detach().cpu()),
                    **{
                        name: float(values[index].detach().cpu())
                        for name, values in raw_signals.items()
                    },
                },
                "evidence_ranks": {
                    name: int(rank_positions[name][index])
                    for name in active_signal_names
                },
                "active_evidence": list(active_signal_names),
            })
        fused_list = [float(fused_scores[index].detach().cpu()) for index in range(valid_count)]
        score_gaps = {
            "candidate_conditioned_verification_attention_top1_minus_best_pck_hit_proposal": (
                float(fused_list[0] - max(value for value, hit in zip(fused_list, [
                    bool(_point_hit([int(pixel % target_w), int(pixel // target_w)], target, pck_threshold))
                    if target is not None and pck_threshold is not None else False
                    for pixel in ordered
                ]) if hit))
                if target is not None and pck_threshold is not None and any(
                    bool(_point_hit([int(pixel % target_w), int(pixel // target_w)], target, pck_threshold))
                    for pixel in ordered
                ) else None
            )
        }
        audits.append({
            "candidate_count": int(valid_count),
            "score_names": [
                "candidate_conditioned_verification",
                "attention_posterior",
                "native_identity",
                "local_relation",
                "anchor_topology",
            ],
            "ranks": {
                "candidate_conditioned_verification": _rank_first_hit(
                    [float(fused_scores[index].detach().cpu()) for index in fused_order],
                    hits,
                )
            },
            "score_gaps": score_gaps,
            "native_candidate_rank": native_rank,
            "anchor_confidence_sum": float(anchor_confidence[anchor_mask].sum().detach().cpu()),
            "anchor_confidence_max": float(anchor_confidence[anchor_mask].max().detach().cpu()) if int(anchor_mask.sum().detach().cpu()) else 0.0,
            "candidates": candidates,
        })

    max_ranked_len = max((len(row) for row in ranked_rows), default=0)
    padded_ranked = [
        row + row[-1:] * (max_ranked_len - len(row))
        for row in ranked_rows
    ]
    ranked_tensor = torch.tensor(padded_ranked, device=device, dtype=torch.long)
    summary = {
        "candidate_pool_mean": float(sum(len(row) for row in candidate_rows) / max(1, len(candidate_rows))),
        "native_selected_rate": float(selected_is_native / max(1, len(points))),
        "attention_selected_rate": float(selected_is_attention / max(1, len(points))),
        "native_rank_mean": float(sum(native_rank_values) / max(1, len(native_rank_values))),
        "selected_attention_rank_mean": float(sum(selected_attention_ranks) / max(1, len(selected_attention_ranks))),
        "anchor_confidence_mean": float(anchor_confidence.mean().detach().cpu()),
    }
    return ranked_tensor, audits, summary


def _candidate_local_transport_verification_rankings(
    src_features: torch.Tensor,
    trg_features: torch.Tensor,
    attention: dict[str, torch.Tensor],
    points: Sequence[Sequence[float]],
    source_size: Sequence[int],
    target_size: Sequence[int],
    src_state: FluxReplayState,
    trg_state: FluxReplayState,
    *,
    candidate_topk: int,
    target_points: Sequence[Sequence[float]] | None = None,
    pck_threshold: float | None = None,
) -> tuple[torch.Tensor, list[dict[str, Any]], dict[str, float]]:
    """Verify attention proposals by the local transport field they imply.

    A target proposal is treated as a hypothesis ``source point -> candidate``.
    The hypothesis induces a local source-to-target coordinate frame.  We then
    test whether source-neighborhood descriptors, target-neighborhood
    descriptors, and neighboring attention rows agree with that local field.
    The native NN prediction is included as an explicit hypothesis and is only
    replaced when an attention proposal dominates it on a majority of
    informative transport evidence sources.
    """

    points = list(points)
    if not points:
        return torch.empty((0, 0), device=src_features.device, dtype=torch.long), [], {}
    device = src_features.device
    source_h, source_w = int(source_size[0]), int(source_size[1])
    target_h, target_w = int(target_size[0]), int(target_size[1])
    src_cells = _native_cell_indices_for_points(
        points,
        source_size,
        src_state.image_height,
        src_state.image_width,
        attention["p_ab"].device,
    )
    mutual_attention = torch.sqrt(
        (attention["p_ab"].float() * attention["p_ba"].float().t()).clamp_min(0.0)
    )
    mutual_attention = torch.nan_to_num(mutual_attention, nan=0.0, posinf=0.0, neginf=0.0)
    candidate_count = min(max(1, int(candidate_topk)), int(mutual_attention.shape[1]))
    attention_values, candidate_cells = torch.topk(
        mutual_attention[src_cells.to(mutual_attention.device)],
        k=candidate_count,
        dim=1,
        sorted=True,
    )
    attention_pixels = _cell_topk_to_pixel_indices(candidate_cells, target_size, trg_state).to(device)

    native_scores, native_indices = _chunked_descriptor_topk_scores_indices(
        src_features,
        trg_features,
        points,
        source_size,
        target_size,
        topk=max(2, candidate_count),
    )
    native_top1 = native_indices[:, 0].long().to(device)
    native_top1_scores = native_scores[:, 0].float().to(device)
    native_top2_scores = (
        native_scores[:, 1].float().to(device)
        if native_scores.shape[1] > 1
        else native_top1_scores
    )
    native_margin = (native_top1_scores - native_top2_scores).clamp_min(0.0)
    duplicate_counts = torch.bincount(native_top1, minlength=target_h * target_w).float().to(device)
    native_uniqueness = 1.0 / duplicate_counts[native_top1].clamp_min(1.0)
    native_cells = _pixel_indices_to_replay_cells(native_top1, target_size, trg_state).to(mutual_attention.device)
    row_max = mutual_attention[src_cells].amax(dim=1).clamp_min(1e-12)
    native_attention_support = (mutual_attention[src_cells, native_cells] / row_max).to(device).clamp_min(0.0)
    anchor_confidence = (
        _rank_normalize(native_margin)
        * _rank_normalize(native_top1_scores)
        * native_uniqueness
        * native_attention_support
    )
    if len(points) <= 1:
        anchor_confidence = torch.zeros_like(anchor_confidence)

    source_xy = torch.tensor(
        [[float(point[0]), float(point[1])] for point in points],
        device=device,
        dtype=torch.float32,
    )
    native_xy = torch.stack(
        (
            (native_top1 % target_w).float(),
            torch.div(native_top1, target_w, rounding_mode="floor").float(),
        ),
        dim=1,
    )
    src_step = torch.tensor(
        [
            max(1.0, float(source_w) / max(1.0, float(src_state.image_width))),
            max(1.0, float(source_h) / max(1.0, float(src_state.image_height))),
        ],
        device=device,
        dtype=torch.float32,
    )
    trg_step = torch.tensor(
        [
            max(1.0, float(target_w) / max(1.0, float(trg_state.image_width))),
            max(1.0, float(target_h) / max(1.0, float(trg_state.image_height))),
        ],
        device=device,
        dtype=torch.float32,
    )
    offset_values = (-1.0, 0.0, 1.0)
    source_offsets = torch.tensor(
        [[dx * src_step[0], dy * src_step[1]] for dy in offset_values for dx in offset_values],
        device=device,
        dtype=torch.float32,
    )
    target_offsets = torch.tensor(
        [[dx * trg_step[0], dy * trg_step[1]] for dy in offset_values for dx in offset_values],
        device=device,
        dtype=torch.float32,
    )

    candidate_rows: list[list[int]] = []
    attention_score_rows: list[list[float]] = []
    attention_rank_rows: list[list[int | None]] = []
    max_len = 0
    for row in range(len(points)):
        pixel_to_attention: dict[int, tuple[float, int]] = {}
        for rank, (pixel, score) in enumerate(
            zip(attention_pixels[row].detach().cpu().tolist(), attention_values[row].detach().cpu().tolist()),
            start=1,
        ):
            pixel = int(pixel)
            score = float(score)
            if pixel not in pixel_to_attention or score > pixel_to_attention[pixel][0]:
                pixel_to_attention[pixel] = (score, int(rank))
        ordered = list(pixel_to_attention.keys())
        native_pixel = int(native_top1[row].detach().cpu())
        if native_pixel not in pixel_to_attention:
            ordered.append(native_pixel)
        score_row: list[float] = []
        rank_row: list[int | None] = []
        for pixel in ordered:
            if pixel in pixel_to_attention:
                score, rank = pixel_to_attention[pixel]
            else:
                replay_cell = int(_pixel_indices_to_replay_cells(
                    torch.tensor([pixel], device=mutual_attention.device, dtype=torch.long),
                    target_size,
                    trg_state,
                )[0].detach().cpu())
                score = float(mutual_attention[int(src_cells[row].detach().cpu()), replay_cell].detach().cpu())
                rank = None
            score_row.append(score)
            rank_row.append(rank)
        candidate_rows.append(ordered)
        attention_score_rows.append(score_row)
        attention_rank_rows.append(rank_row)
        max_len = max(max_len, len(ordered))
    padded_candidates = [row + row[-1:] * (max_len - len(row)) for row in candidate_rows]
    candidate_tensor = torch.tensor(padded_candidates, device=device, dtype=torch.long)
    descriptor_sorted_scores, descriptor_sorted_pixels = _chunked_descriptor_topk_scores_indices(
        src_features,
        trg_features,
        points,
        source_size,
        target_size,
        topk=max_len,
        candidate_indices=candidate_tensor,
    )
    descriptor_by_row = [
        {
            int(pixel): float(score)
            for pixel, score in zip(
                descriptor_sorted_pixels[row].detach().cpu().tolist(),
                descriptor_sorted_scores[row].detach().cpu().tolist(),
            )
        }
        for row in range(len(points))
    ]

    ranked_rows: list[list[int]] = []
    audits: list[dict[str, Any]] = []
    selected_is_native = 0
    rescued_count = 0
    abstained_count = 0
    selected_attention_ranks: list[float] = []
    native_rank_values: list[float] = []
    transport_margin_values: list[float] = []
    dominance_win_values: list[float] = []
    for row, ordered in enumerate(candidate_rows):
        valid_count = len(ordered)
        candidate_pixels = torch.tensor(ordered, device=device, dtype=torch.long)
        candidate_xy = torch.stack(
            (
                (candidate_pixels % target_w).float(),
                torch.div(candidate_pixels, target_w, rounding_mode="floor").float(),
            ),
            dim=1,
        )
        src_patch_xy = source_xy[row].reshape(1, 2) + source_offsets
        src_patch_xy[:, 0].clamp_(0, source_w - 1)
        src_patch_xy[:, 1].clamp_(0, source_h - 1)
        src_patch_vectors = F.normalize(
            _sample_feature_vectors_at_pixels(src_features, src_patch_xy.round().long(), source_size),
            dim=1,
            eps=1e-12,
        )
        src_patch_gram = src_patch_vectors @ src_patch_vectors.t()
        off_diag = ~torch.eye(src_patch_gram.shape[0], device=device, dtype=torch.bool)
        src_self = src_patch_gram[off_diag]
        trg_patch_xy_all = candidate_xy[:, None, :] + target_offsets.reshape(1, -1, 2)
        trg_patch_xy_all[:, :, 0].clamp_(0, target_w - 1)
        trg_patch_xy_all[:, :, 1].clamp_(0, target_h - 1)
        trg_patch_vectors = F.normalize(
            _sample_feature_vectors_at_pixels(
                trg_features,
                trg_patch_xy_all.round().long().reshape(-1, 2),
                target_size,
            ).reshape(valid_count, len(target_offsets), -1),
            dim=2,
            eps=1e-12,
        )
        patch_cos = (src_patch_vectors.reshape(1, len(source_offsets), -1) * trg_patch_vectors).sum(dim=2).clamp(-1.0, 1.0)
        center_weight = torch.ones((len(source_offsets),), device=device, dtype=torch.float32)
        center_weight[4] = 0.5
        patch_identity_tensor = ((patch_cos + 1.0) * 0.5 * center_weight.reshape(1, -1)).sum(dim=1) / center_weight.sum().clamp_min(1e-12)
        self_similarity_scores: list[float] = []
        for candidate_patch in trg_patch_vectors:
            trg_self = (candidate_patch @ candidate_patch.t())[off_diag]
            if src_self.numel() > 0 and trg_self.numel() > 0:
                self_similarity = F.cosine_similarity(
                    src_self.reshape(1, -1),
                    trg_self.reshape(1, -1),
                    dim=1,
                    eps=1e-12,
                )[0]
                self_similarity = (self_similarity.clamp(-1.0, 1.0) + 1.0) * 0.5
            else:
                self_similarity = torch.tensor(1.0, device=device)
            self_similarity_scores.append(float(self_similarity.detach().cpu()))
        src_neighbor_cells = _native_cell_indices_for_points(
            src_patch_xy.detach().cpu().tolist(),
            source_size,
            src_state.image_height,
            src_state.image_width,
            mutual_attention.device,
        )
        trg_flat_xy = trg_patch_xy_all.round().long().to(mutual_attention.device)
        trg_neighbor_cells = _pixel_indices_to_replay_cells(
            (trg_flat_xy[:, :, 1].clamp(0, target_h - 1) * target_w)
            + trg_flat_xy[:, :, 0].clamp(0, target_w - 1),
            target_size,
            trg_state,
        ).reshape(valid_count, len(target_offsets))
        neighbor_rows = mutual_attention[src_neighbor_cells]
        neighbor_row_max = neighbor_rows.amax(dim=1).clamp_min(1e-12)
        expanded_rows = neighbor_rows.unsqueeze(0).expand(valid_count, -1, -1)
        neighbor_support = torch.gather(expanded_rows, 2, trg_neighbor_cells.unsqueeze(2)).squeeze(2)
        neighbor_support = neighbor_support / neighbor_row_max.reshape(1, -1)
        neighbor_weight = torch.ones((len(target_offsets),), device=mutual_attention.device, dtype=torch.float32)
        neighbor_weight[4] = 0.5
        neighbor_attention_tensor = (
            neighbor_support.clamp_min(0.0) * neighbor_weight.reshape(1, -1)
        ).sum(dim=1) / neighbor_weight.sum().clamp_min(1e-12)
        patch_identity_scores = [float(value) for value in patch_identity_tensor.detach().cpu().tolist()]
        neighbor_attention_scores = [float(value) for value in neighbor_attention_tensor.detach().cpu().tolist()]

        anchor_mask = torch.ones(len(points), device=device, dtype=torch.bool)
        anchor_mask[row] = False
        topology = _anchor_topology_scores_for_candidates(
            source_xy[row],
            candidate_xy,
            source_xy[anchor_mask],
            native_xy[anchor_mask],
            anchor_confidence[anchor_mask],
            source_size,
            target_size,
        )
        raw_signals = {
            "native_identity": torch.tensor(
                [descriptor_by_row[row].get(int(pixel), -float("inf")) for pixel in ordered],
                device=device,
                dtype=torch.float32,
            ),
            "attention_posterior": torch.tensor(attention_score_rows[row], device=device, dtype=torch.float32),
            "patch_transport_identity": torch.tensor(patch_identity_scores, device=device, dtype=torch.float32),
            "local_self_similarity": torch.tensor(self_similarity_scores, device=device, dtype=torch.float32),
            "neighbor_attention_transport": torch.tensor(neighbor_attention_scores, device=device, dtype=torch.float32),
            "anchor_topology": topology.to(device=device, dtype=torch.float32),
        }
        verifier_names = [
            "patch_transport_identity",
            "local_self_similarity",
            "neighbor_attention_transport",
            "anchor_topology",
        ]
        rank_positions: dict[str, dict[int, int]] = {}
        reciprocal_vote = torch.zeros(valid_count, device=device, dtype=torch.float32)
        active_names: list[str] = []
        for name, values in raw_signals.items():
            values = torch.nan_to_num(values.float(), nan=-float("inf"), posinf=-float("inf"), neginf=-float("inf"))
            finite = torch.isfinite(values)
            if int(finite.sum().detach().cpu()) <= 0:
                continue
            finite_values = values[finite]
            if float((finite_values.max() - finite_values.min()).detach().cpu()) <= 1e-12:
                continue
            active_names.append(name)
            order = sorted(range(valid_count), key=lambda index: float(values[index].detach().cpu()), reverse=True)
            rank_positions[name] = {index: rank for rank, index in enumerate(order, start=1)}
            for index, rank in rank_positions[name].items():
                reciprocal_vote[index] += 1.0 / float(rank)
        if not active_names:
            active_names = ["native_identity"]
            rank_positions["native_identity"] = {index: index + 1 for index in range(valid_count)}
            for index in range(valid_count):
                reciprocal_vote[index] += 1.0 / float(index + 1)
        fused_order = sorted(
            range(valid_count),
            key=lambda index: (
                float(reciprocal_vote[index].detach().cpu()),
                float(raw_signals["patch_transport_identity"][index].detach().cpu()),
                float(raw_signals["neighbor_attention_transport"][index].detach().cpu()),
                float(raw_signals["native_identity"][index].detach().cpu()),
            ),
            reverse=True,
        )
        native_pixel = int(native_top1[row].detach().cpu())
        native_index = next((index for index, pixel in enumerate(ordered) if int(pixel) == native_pixel), None)
        if native_index is None:
            native_index = fused_order[0]
        best_index = fused_order[0]
        if best_index == native_index and len(fused_order) > 1:
            best_non_native = next((index for index in fused_order if index != native_index), native_index)
        else:
            best_non_native = best_index

        informative_verifiers = [
            name for name in verifier_names
            if name in rank_positions and float((raw_signals[name].max() - raw_signals[name].min()).detach().cpu()) > 1e-12
        ]
        wins = 0
        losses = 0
        ties = 0
        for name in informative_verifiers:
            candidate_value = float(raw_signals[name][best_non_native].detach().cpu())
            native_value = float(raw_signals[name][native_index].detach().cpu())
            if candidate_value > native_value:
                wins += 1
            elif candidate_value < native_value:
                losses += 1
            else:
                ties += 1
        native_identity_values = raw_signals["native_identity"]
        native_identity_order = sorted(
            range(valid_count),
            key=lambda index: float(native_identity_values[index].detach().cpu()),
            reverse=True,
        )
        native_identity_rank = next(
            (rank for rank, index in enumerate(native_identity_order, start=1) if index == best_non_native),
            valid_count,
        )
        native_identity_acceptable = native_identity_rank <= (valid_count + 1) // 2
        rescue = (
            best_non_native != native_index
            and bool(attention_rank_rows[row][best_non_native] is not None)
            and wins > losses
            and wins >= max(1, (len(informative_verifiers) + 1) // 2)
            and native_identity_acceptable
        )
        selected_index = best_non_native if rescue else native_index
        if selected_index == native_index:
            selected_is_native += 1
            abstained_count += int(best_non_native != native_index)
        else:
            rescued_count += 1
        if informative_verifiers:
            dominance_win_values.append(float(wins) / float(len(informative_verifiers)))
        transport_margin_values.append(
            float(
                (
                    raw_signals["patch_transport_identity"][best_non_native]
                    - raw_signals["patch_transport_identity"][native_index]
                ).detach().cpu()
            )
        )
        final_order = [selected_index] + [
            index for index in fused_order
            if index != selected_index
        ]
        ranked = [int(ordered[index]) for index in final_order]
        ranked_rows.append(ranked)
        selected_attention_rank = attention_rank_rows[row][selected_index]
        if selected_attention_rank is not None:
            selected_attention_ranks.append(float(selected_attention_rank))
        native_rank = next((rank for rank, index in enumerate(final_order, start=1) if index == native_index), None)
        if native_rank is not None:
            native_rank_values.append(float(native_rank))

        target = target_points[row] if target_points is not None else None
        candidates: list[dict[str, Any]] = []
        hits: list[bool] = []
        for rank, index in enumerate(final_order, start=1):
            pixel = int(ordered[index])
            xy = [int(pixel % target_w), int(pixel // target_w)]
            hit = (
                bool(_point_hit(xy, target, pck_threshold))
                if target is not None and pck_threshold is not None
                else False
            )
            hits.append(hit)
            candidates.append({
                "rank": int(rank),
                "pixel": xy,
                "pixel_index": int(pixel),
                "native_candidate": bool(index == native_index),
                "selected_by_rescue": bool(rank == 1 and rescue),
                "attention_rank": attention_rank_rows[row][index],
                "pck_hit": bool(hit),
                "scores": {
                    "candidate_local_transport_verification": float(reciprocal_vote[index].detach().cpu()),
                    **{
                        name: float(values[index].detach().cpu())
                        for name, values in raw_signals.items()
                    },
                },
                "evidence_ranks": {
                    name: int(rank_positions[name][index])
                    for name in active_names
                },
            })
        audits.append({
            "candidate_count": int(valid_count),
            "score_names": [
                "candidate_local_transport_verification",
                *list(raw_signals.keys()),
            ],
            "ranks": {
                "candidate_local_transport_verification": _rank_first_hit(
                    [float(reciprocal_vote[index].detach().cpu()) for index in final_order],
                    hits,
                )
            },
            "native_candidate_rank": native_rank,
            "rescue_applied": bool(rescue),
            "rescue_candidate_attention_rank": attention_rank_rows[row][best_non_native],
            "rescue_dominance_wins": int(wins),
            "rescue_dominance_losses": int(losses),
            "rescue_dominance_ties": int(ties),
            "native_identity_rank_of_rescue": int(native_identity_rank),
            "transport_margin_over_native": float(transport_margin_values[-1]),
            "anchor_confidence_sum": float(anchor_confidence[anchor_mask].sum().detach().cpu()),
            "candidates": candidates,
        })

    max_ranked_len = max((len(row) for row in ranked_rows), default=0)
    padded_ranked = [
        row + row[-1:] * (max_ranked_len - len(row))
        for row in ranked_rows
    ]
    ranked_tensor = torch.tensor(padded_ranked, device=device, dtype=torch.long)
    summary = {
        "candidate_pool_mean": float(sum(len(row) for row in candidate_rows) / max(1, len(candidate_rows))),
        "native_selected_rate": float(selected_is_native / max(1, len(points))),
        "rescue_rate": float(rescued_count / max(1, len(points))),
        "abstained_challenger_rate": float(abstained_count / max(1, len(points))),
        "native_rank_mean": float(sum(native_rank_values) / max(1, len(native_rank_values))),
        "selected_attention_rank_mean": float(sum(selected_attention_ranks) / max(1, len(selected_attention_ranks))),
        "anchor_confidence_mean": float(anchor_confidence.mean().detach().cpu()),
        "transport_margin_over_native_mean": float(sum(transport_margin_values) / max(1, len(transport_margin_values))),
        "dominance_win_fraction_mean": float(sum(dominance_win_values) / max(1, len(dominance_win_values))),
    }
    return ranked_tensor, audits, summary


def _candidate_graph_consensus_from_audits(
    source_points: Sequence[Sequence[float]],
    candidate_audits: Sequence[dict[str, Any]],
    source_size: Sequence[int],
    target_size: Sequence[int],
    device: torch.device,
    *,
    max_iterations: int = 2,
    target_points: Sequence[Sequence[float]] | None = None,
    pck_threshold: float | None = None,
) -> tuple[torch.Tensor, list[dict[str, Any]], dict[str, float]]:
    """Refine local unary scores with pair-level geometry consensus."""

    points = list(source_points)
    if not points:
        return torch.empty((0, 0), device=device, dtype=torch.long), [], {}
    if len(candidate_audits) != len(points):
        raise ValueError("candidate audits must align with source points")

    target_h, target_w = int(target_size[0]), int(target_size[1])
    source_scale = torch.tensor(
        [max(1.0, float(source_size[1])), max(1.0, float(source_size[0]))],
        device=device,
        dtype=torch.float32,
    )
    target_scale = torch.tensor(
        [max(1.0, float(target_w)), max(1.0, float(target_h))],
        device=device,
        dtype=torch.float32,
    )
    source_xy = torch.tensor(
        [[float(point[0]), float(point[1])] for point in points],
        device=device,
        dtype=torch.float32,
    )

    preferred_names = (
        "candidate_local_transport_verification",
        "patch_transport_identity",
        "local_self_similarity",
        "neighbor_attention_transport",
        "anchor_topology",
        "native_identity",
        "attention_posterior",
    )

    candidate_rows: list[list[int]] = []
    score_rows: list[dict[str, torch.Tensor]] = []
    native_pixels: list[int] = []
    for row in candidate_audits:
        candidates = row.get("candidates", []) if isinstance(row, dict) else []
        if not candidates:
            candidate_rows.append([0])
            score_rows.append({name: torch.zeros(1, device=device, dtype=torch.float32) for name in preferred_names})
            native_pixels.append(0)
            continue
        pixels: list[int] = []
        score_maps: dict[str, list[float]] = {name: [] for name in preferred_names}
        for candidate in candidates:
            pixel_index = int(candidate.get("pixel_index", candidate.get("pixel", [0, 0])[1] * target_w + candidate.get("pixel", [0, 0])[0]))
            pixels.append(pixel_index)
            candidate_scores = candidate.get("scores", {}) if isinstance(candidate, dict) else {}
            for name in preferred_names:
                value = candidate_scores.get(name, float("-inf"))
                try:
                    score_maps[name].append(float(value))
                except (TypeError, ValueError):
                    score_maps[name].append(float("-inf"))
        candidate_rows.append(pixels)
        score_rows.append({name: torch.tensor(values, device=device, dtype=torch.float32) for name, values in score_maps.items()})
        native_pixels.append(int(pixels[0]))

    max_len = max((len(row) for row in candidate_rows), default=0)
    padded_candidates = [row + row[-1:] * (max_len - len(row)) for row in candidate_rows]
    candidate_tensor = torch.tensor(padded_candidates, device=device, dtype=torch.long)
    candidate_xy = torch.stack(
        (
            (candidate_tensor % target_w).float(),
            torch.div(candidate_tensor, target_w, rounding_mode="floor").float(),
        ),
        dim=2,
    )

    def _confidence_from_row(values: torch.Tensor) -> float:
        if values.numel() == 0:
            return 1.0
        top2 = torch.topk(values, k=min(2, int(values.shape[0]))).values
        if top2.numel() == 1:
            return float(_rank_normalize(top2[:1])[0].detach().cpu())
        margin = (top2[0] - top2[1]).clamp_min(0.0)
        return float((_rank_normalize(top2[:1])[0] * _rank_normalize(margin[None])[0]).detach().cpu())

    selected_pixels = [int(row[0]) for row in candidate_rows]
    selected_confidence = torch.tensor(
        [_confidence_from_row(score_rows[row]["candidate_local_transport_verification"]) for row in range(len(candidate_rows))],
        device=device,
        dtype=torch.float32,
    ).clamp_min(0.0)

    def _consensus_scores(current_pixels: Sequence[int], weights: torch.Tensor) -> torch.Tensor:
        if len(points) <= 1:
            return torch.ones((len(points), max_len), device=device, dtype=torch.float32)
        current_xy = torch.tensor(
            [[float(pixel % target_w), float(pixel // target_w)] for pixel in current_pixels],
            device=device,
            dtype=torch.float32,
        )
        total_weight = weights.sum().clamp_min(1e-12)
        consensus = torch.ones((len(points), max_len), device=device, dtype=torch.float32)
        for row in range(len(points)):
            for index in range(len(candidate_rows[row])):
                candidate_point = candidate_xy[row, index]
                compat_values = []
                for other in range(len(points)):
                    if other == row:
                        continue
                    src_delta = (source_xy[other] - source_xy[row]) / source_scale
                    trg_delta = (current_xy[other] - candidate_point) / target_scale
                    geometry = 1.0 / (1.0 + torch.linalg.vector_norm(trg_delta - src_delta, ord=2))
                    compat_values.append(geometry * weights[other].clamp_min(0.0))
                if compat_values:
                    consensus[row, index] = torch.stack(compat_values).sum() / total_weight
                else:
                    consensus[row, index] = 1.0
        return consensus

    def _fuse_row(row_index: int, consensus_row: torch.Tensor) -> tuple[torch.Tensor, list[str]]:
        evidence_names = [
            name for name in preferred_names
            if name in score_rows[row_index]
            and score_rows[row_index][name].numel() > 0
            and float((score_rows[row_index][name].max() - score_rows[row_index][name].min()).detach().cpu()) > 1e-12
        ]
        if consensus_row.numel() > 0 and float((consensus_row.max() - consensus_row.min()).detach().cpu()) > 1e-12:
            evidence_names.append("graph_consensus")
        if not evidence_names:
            evidence_names = ["graph_consensus"]
        rank_positions: dict[str, dict[int, int]] = {}
        reciprocal_vote = torch.zeros(len(candidate_rows[row_index]), device=device, dtype=torch.float32)
        active_names: list[str] = []
        for name in evidence_names:
            values = consensus_row[: len(candidate_rows[row_index])] if name == "graph_consensus" else score_rows[row_index][name]
            values = torch.nan_to_num(values.float(), nan=-float("inf"), posinf=-float("inf"), neginf=-float("inf"))
            finite = torch.isfinite(values)
            if int(finite.sum().detach().cpu()) <= 0:
                continue
            finite_values = values[finite]
            if float((finite_values.max() - finite_values.min()).detach().cpu()) <= 1e-12:
                continue
            active_names.append(name)
            order = sorted(
                range(len(candidate_rows[row_index])),
                key=lambda index: float(values[index].detach().cpu()),
                reverse=True,
            )
            rank_positions[name] = {index: rank for rank, index in enumerate(order, start=1)}
            for index, rank in rank_positions[name].items():
                reciprocal_vote[index] += 1.0 / float(rank)
        if not active_names:
            active_names = ["graph_consensus"]
            rank_positions["graph_consensus"] = {index: rank for rank, index in enumerate(range(len(candidate_rows[row_index])), start=1)}
            for index, rank in rank_positions["graph_consensus"].items():
                reciprocal_vote[index] += 1.0 / float(rank)
        median_rank = torch.tensor(
            [
                sorted(rank_positions[name][index] for name in active_names)[len(active_names) // 2]
                for index in range(len(candidate_rows[row_index]))
            ],
            device=device,
            dtype=torch.float32,
        )
        fused = reciprocal_vote / median_rank.clamp_min(1.0).sqrt()
        if consensus_row.numel() > 0:
            fused = fused * consensus_row[: len(candidate_rows[row_index])].clamp_min(0.0).sqrt()
        return fused, active_names

    final_scores: list[torch.Tensor] = [torch.ones(len(row), device=device, dtype=torch.float32) for row in candidate_rows]
    iteration_count = 0
    for iteration in range(max(1, int(max_iterations))):
        consensus = _consensus_scores(selected_pixels, selected_confidence)
        new_scores: list[torch.Tensor] = []
        new_pixels: list[int] = []
        new_confidence: list[float] = []
        for row in range(len(points)):
            fused, _ = _fuse_row(row, consensus[row])
            new_scores.append(fused)
            best_index = int(torch.argmax(fused).detach().cpu())
            new_pixels.append(int(candidate_rows[row][best_index]))
            if fused.numel() > 1:
                top2 = torch.topk(fused, k=min(2, int(fused.shape[0]))).values
                if top2.numel() > 1:
                    margin = (top2[0] - top2[1]).clamp_min(0.0)
                    new_confidence.append(
                        float((_rank_normalize(top2[:1])[0] * _rank_normalize(margin[None])[0]).detach().cpu())
                    )
                else:
                    new_confidence.append(float(_rank_normalize(top2[:1])[0].detach().cpu()))
            else:
                new_confidence.append(float(fused[0].detach().cpu()))
        iteration_count = iteration + 1
        final_scores = new_scores
        if new_pixels == selected_pixels:
            selected_confidence = torch.tensor(new_confidence, device=device, dtype=torch.float32).clamp_min(0.0)
            break
        selected_pixels = new_pixels
        selected_confidence = torch.tensor(new_confidence, device=device, dtype=torch.float32).clamp_min(0.0)

    final_consensus = _consensus_scores(selected_pixels, selected_confidence)
    final_scores = [
        _fuse_row(row, final_consensus[row])[0]
        for row in range(len(points))
    ]
    ranked_rows: list[list[int]] = []
    audits: list[dict[str, Any]] = []
    selected_is_native = 0
    rescued_count = 0
    selected_confidences: list[float] = []
    consensus_margins: list[float] = []
    for row in range(len(points)):
        fused = final_scores[row]
        candidate_list = candidate_rows[row]
        order = sorted(
            range(len(candidate_list)),
            key=lambda index: (
                float(fused[index].detach().cpu()),
                float(score_rows[row]["candidate_local_transport_verification"][index].detach().cpu()) if "candidate_local_transport_verification" in score_rows[row] else float("-inf"),
                float(score_rows[row]["native_identity"][index].detach().cpu()) if "native_identity" in score_rows[row] else float("-inf"),
                float(score_rows[row]["attention_posterior"][index].detach().cpu()) if "attention_posterior" in score_rows[row] else float("-inf"),
            ),
            reverse=True,
        )
        ranked_rows.append([int(candidate_list[index]) for index in order])
        selected_pixel = int(candidate_list[order[0]])
        if selected_pixel == native_pixels[row]:
            selected_is_native += 1
        else:
            rescued_count += 1
        if fused.numel() > 1:
            top2 = torch.topk(fused, k=min(2, int(fused.shape[0]))).values
            if top2.numel() > 1:
                consensus_margins.append(float((top2[0] - top2[1]).detach().cpu()))
            selected_confidences.append(float(top2[0].detach().cpu()))
        else:
            selected_confidences.append(float(fused[0].detach().cpu()))
        target = target_points[row] if target_points is not None else None
        candidates: list[dict[str, Any]] = []
        hits: list[bool] = []
        for rank, index in enumerate(order, start=1):
            pixel = int(candidate_list[index])
            xy = [int(pixel % target_w), int(pixel // target_w)]
            hit = (
                bool(_point_hit(xy, target, pck_threshold))
                if target is not None and pck_threshold is not None
                else False
            )
            hits.append(hit)
            candidate_scores = {
                name: float(score_rows[row][name][index].detach().cpu())
                for name in score_rows[row]
            }
            candidate_scores["graph_consensus"] = float(final_consensus[row, index].detach().cpu())
            candidates.append({
                "rank": int(rank),
                "pixel": xy,
                "pixel_index": int(pixel),
                "native_candidate": bool(pixel == native_pixels[row]),
                "pck_hit": bool(hit),
                "scores": candidate_scores,
            })
        score_values = [float(fused[index].detach().cpu()) for index in order]
        audits.append({
            "candidate_count": int(len(candidate_list)),
            "iteration_count": int(iteration_count),
            "score_names": [
                "candidate_graph_consensus",
                "candidate_local_transport_verification",
                "patch_transport_identity",
                "local_self_similarity",
                "neighbor_attention_transport",
                "anchor_topology",
                "native_identity",
                "attention_posterior",
                "graph_consensus",
            ],
            "ranks": {"candidate_graph_consensus": _rank_first_hit(score_values, hits)},
            "selected_native": bool(selected_pixel == native_pixels[row]),
            "selected_confidence": float(selected_confidences[-1]),
            "graph_consensus_margin": float(consensus_margins[-1]) if consensus_margins else 0.0,
            "candidates": candidates,
        })

    max_len = max((len(row) for row in ranked_rows), default=0)
    padded = [row + row[-1:] * (max_len - len(row)) for row in ranked_rows]
    ranked_tensor = torch.tensor(padded, device=device, dtype=torch.long)
    summary = {
        "candidate_pool_mean": float(sum(len(row) for row in candidate_rows) / max(1, len(candidate_rows))),
        "native_selected_rate": float(selected_is_native / max(1, len(points))),
        "rescue_rate": float(rescued_count / max(1, len(points))),
        "iteration_mean": float(iteration_count),
        "selected_confidence_mean": float(sum(selected_confidences) / max(1, len(selected_confidences))),
        "consensus_margin_mean": float(sum(consensus_margins) / max(1, len(consensus_margins))) if consensus_margins else 0.0,
    }
    return ranked_tensor, audits, summary


def _candidate_graph_consensus_verification_rankings(
    src_features: torch.Tensor,
    trg_features: torch.Tensor,
    attention: dict[str, torch.Tensor],
    points: Sequence[Sequence[float]],
    source_size: Sequence[int],
    target_size: Sequence[int],
    src_state: FluxReplayState,
    trg_state: FluxReplayState,
    *,
    candidate_topk: int,
    target_points: Sequence[Sequence[float]] | None = None,
    pck_threshold: float | None = None,
) -> tuple[torch.Tensor, list[dict[str, Any]], dict[str, float]]:
    """Refine local transport verification with pair-level candidate consensus."""

    base_ranked, base_audits, base_summary = _candidate_local_transport_verification_rankings(
        src_features,
        trg_features,
        attention,
        points,
        source_size,
        target_size,
        src_state,
        trg_state,
        candidate_topk=candidate_topk,
        target_points=target_points,
        pck_threshold=pck_threshold,
    )
    consensus_ranked, consensus_audits, consensus_summary = _candidate_graph_consensus_from_audits(
        points,
        base_audits,
        source_size,
        target_size,
        src_features.device,
        max_iterations=2,
        target_points=target_points,
        pck_threshold=pck_threshold,
    )
    summary = {
        "candidate_pool_mean": float(consensus_summary.get("candidate_pool_mean", base_summary.get("candidate_pool_mean", 0.0))),
        "native_selected_rate": float(consensus_summary.get("native_selected_rate", base_summary.get("native_selected_rate", 0.0))),
        "rescue_rate": float(consensus_summary.get("rescue_rate", base_summary.get("rescue_rate", 0.0))),
        "iteration_mean": float(consensus_summary.get("iteration_mean", 0.0)),
        "selected_confidence_mean": float(consensus_summary.get("selected_confidence_mean", 0.0)),
        "consensus_margin_mean": float(consensus_summary.get("consensus_margin_mean", 0.0)),
        "local_transport_native_selected_rate": float(base_summary.get("native_selected_rate", 0.0)),
        "local_transport_rescue_rate": float(base_summary.get("rescue_rate", 0.0)),
        "local_transport_anchor_confidence_mean": float(base_summary.get("anchor_confidence_mean", 0.0)),
        "local_transport_selected_attention_rank_mean": float(base_summary.get("selected_attention_rank_mean", 0.0)),
    }
    return consensus_ranked, consensus_audits, summary


def _standardize_finite(values: torch.Tensor, dim: int) -> torch.Tensor:
    """Standardize score families without dataset-tuned scales or thresholds."""

    values = torch.nan_to_num(values.float(), nan=0.0, posinf=0.0, neginf=0.0)
    mean = values.mean(dim=dim, keepdim=True)
    centered = values - mean
    scale = centered.square().mean(dim=dim, keepdim=True).sqrt()
    return torch.where(scale > 1e-6, centered / scale.clamp_min(1e-6), torch.zeros_like(centered))


def _attention_graph_assignment_energy(
    assignment: torch.Tensor,
    unary: torch.Tensor,
    pairs: torch.Tensor,
    potentials: torch.Tensor,
) -> torch.Tensor:
    point_index = torch.arange(int(unary.shape[0]), device=unary.device)
    energy = unary[point_index, assignment].sum()
    if int(pairs.shape[0]) > 0:
        edge_index = torch.arange(int(pairs.shape[0]), device=unary.device)
        energy = energy + potentials[
            edge_index,
            assignment[pairs[:, 0]],
            assignment[pairs[:, 1]],
        ].sum()
    return energy


def _attention_graph_conditional_scores(
    point_index: int,
    assignment: torch.Tensor,
    unary: torch.Tensor,
    pairs: torch.Tensor,
    potentials: torch.Tensor,
) -> torch.Tensor:
    scores = unary[point_index].clone()
    first_edges = torch.nonzero(pairs[:, 0] == point_index, as_tuple=False).flatten()
    if int(first_edges.numel()) > 0:
        other = pairs[first_edges, 1]
        scores += potentials[first_edges, :, assignment[other]].sum(dim=0)
    second_edges = torch.nonzero(pairs[:, 1] == point_index, as_tuple=False).flatten()
    if int(second_edges.numel()) > 0:
        other = pairs[second_edges, 0]
        scores += potentials[second_edges, assignment[other], :].sum(dim=0)
    return scores


def _attention_graph_max_product(
    unary: torch.Tensor,
    pairs: torch.Tensor,
    potentials: torch.Tensor,
    *,
    steps: int = 8,
    damping: float = 0.5,
) -> torch.Tensor:
    num_points, candidate_count = unary.shape
    edge_count = int(pairs.shape[0])
    if edge_count == 0:
        return unary.clone()
    first, second = pairs[:, 0], pairs[:, 1]
    sender = torch.cat((first, second), dim=0)
    receiver = torch.cat((second, first), dim=0)
    directed = torch.cat((potentials, potentials.transpose(1, 2)), dim=0)
    reverse = torch.cat(
        (
            torch.arange(edge_count, 2 * edge_count, device=unary.device),
            torch.arange(edge_count, device=unary.device),
        )
    )
    messages = torch.zeros((2 * edge_count, candidate_count), device=unary.device, dtype=unary.dtype)
    keep = min(max(float(damping), 0.0), 0.99)
    for _ in range(max(1, int(steps))):
        incoming = torch.zeros_like(unary)
        incoming.index_add_(0, receiver, messages)
        cavity = unary[sender] + incoming[sender] - messages[reverse]
        updated = torch.max(cavity.unsqueeze(2) + directed, dim=1).values
        updated -= updated.amax(dim=1, keepdim=True)
        messages = keep * messages + (1.0 - keep) * updated
    incoming = torch.zeros_like(unary)
    incoming.index_add_(0, receiver, messages)
    return unary + incoming


def _attention_graph_icm(
    initial: torch.Tensor,
    unary: torch.Tensor,
    pairs: torch.Tensor,
    potentials: torch.Tensor,
    *,
    steps: int = 8,
) -> torch.Tensor:
    assignment = initial.clone()
    for _ in range(max(1, int(steps))):
        changed = 0
        for point_index in range(int(unary.shape[0])):
            scores = _attention_graph_conditional_scores(
                point_index, assignment, unary, pairs, potentials
            )
            current = int(assignment[point_index])
            selected = int(torch.argmax(scores))
            if float(scores[selected]) > float(scores[current]) + 1e-8:
                assignment[point_index] = selected
                changed += int(selected != current)
        if changed == 0:
            break
    return assignment


def _solve_attention_relational_graph(
    unary: torch.Tensor,
    candidate_cells: torch.Tensor,
    source_descriptors: torch.Tensor,
    target_descriptors: torch.Tensor,
) -> dict[str, Any]:
    """Jointly select attention candidates using descriptor-space relations."""

    if unary.ndim != 2 or candidate_cells.shape != unary.shape:
        raise ValueError("unary and candidate_cells must share [points, candidates]")
    num_points, candidate_count = unary.shape
    if source_descriptors.shape[0] != num_points:
        raise ValueError("source_descriptors must align with graph points")
    if num_points == 0 or candidate_count == 0:
        raise ValueError("attention relational graph requires non-empty candidates")

    source = F.normalize(torch.nan_to_num(source_descriptors.float()), dim=1, eps=1e-12)
    target = F.normalize(torch.nan_to_num(target_descriptors.float()), dim=1, eps=1e-12)
    candidate_descriptors = target[candidate_cells.long()]
    pairs = (
        torch.combinations(torch.arange(num_points, device=unary.device), r=2)
        if num_points > 1
        else torch.empty((0, 2), device=unary.device, dtype=torch.long)
    )
    edge_count = int(pairs.shape[0])
    if edge_count:
        first, second = pairs[:, 0], pairs[:, 1]
        relation_chunks: list[torch.Tensor] = []
        for edge_start in range(0, edge_count, 32):
            edge_end = min(edge_count, edge_start + 32)
            edge_first = first[edge_start:edge_end]
            edge_second = second[edge_start:edge_end]
            source_first = source[edge_first]
            source_second = source[edge_second]
            target_first = candidate_descriptors[edge_first]
            target_second = candidate_descriptors[edge_second]

            target_cosine = torch.bmm(target_first, target_second.transpose(1, 2))
            source_cosine = (source_first * source_second).sum(dim=1)
            cosine_preservation = -(target_cosine - source_cosine[:, None, None]).abs()

            source_difference = F.normalize(
                source_second - source_first, dim=1, eps=1e-12
            )
            first_projection = (target_first * source_difference[:, None, :]).sum(dim=2)
            second_projection = (target_second * source_difference[:, None, :]).sum(dim=2)
            target_difference_norm = (
                2.0 - 2.0 * target_cosine
            ).clamp_min(0.0).sqrt().clamp_min(1e-12)
            difference_direction = (
                second_projection[:, None, :] - first_projection[:, :, None]
            ) / target_difference_norm

            source_coactivation = source_first * source_second
            source_coactivation_norm = source_coactivation.norm(dim=1).clamp_min(1e-12)
            coactivation_numerator = torch.bmm(
                target_first * source_coactivation[:, None, :],
                target_second.transpose(1, 2),
            )
            target_coactivation_norm = torch.bmm(
                target_first.square(), target_second.square().transpose(1, 2)
            ).clamp_min(0.0).sqrt().clamp_min(1e-12)
            coactivation_preservation = coactivation_numerator / (
                source_coactivation_norm[:, None, None] * target_coactivation_norm
            )
            relation_chunks.append(
                torch.stack(
                    (cosine_preservation, difference_direction, coactivation_preservation),
                    dim=1,
                )
            )
        relation_signals = torch.cat(relation_chunks, dim=0)
        relation_signals = _standardize_finite(
            relation_signals.flatten(2), dim=2
        ).reshape(edge_count, 3, candidate_count, candidate_count)
        potentials = relation_signals.mean(dim=1)
        degree = torch.bincount(pairs.flatten(), minlength=num_points).float().clamp_min(1.0)
        edge_scale = 2.0 / (degree[first] + degree[second])
        potentials = potentials * edge_scale[:, None, None]
    else:
        potentials = torch.empty(
            (0, candidate_count, candidate_count), device=unary.device, dtype=unary.dtype
        )

    unary_assignment = torch.argmax(unary, dim=1)
    point_index = torch.arange(num_points, device=unary.device)
    unary_assignment_unary_energy = unary[point_index, unary_assignment].sum()
    unary_start_energy = _attention_graph_assignment_energy(
        unary_assignment, unary, pairs, potentials
    )
    beliefs = _attention_graph_max_product(unary, pairs, potentials)
    bp_assignment = torch.argmax(beliefs, dim=1)
    bp_refined = _attention_graph_icm(bp_assignment, unary, pairs, potentials)
    unary_refined = _attention_graph_icm(unary_assignment, unary, pairs, potentials)
    bp_energy = _attention_graph_assignment_energy(bp_refined, unary, pairs, potentials)
    unary_refined_energy = _attention_graph_assignment_energy(
        unary_refined, unary, pairs, potentials
    )
    if float(bp_energy) > float(unary_refined_energy):
        assignment, selected_energy = bp_refined, bp_energy
        selected_start = "bp"
    else:
        assignment, selected_energy = unary_refined, unary_refined_energy
        selected_start = "unary"
    selected_unary_energy = unary[point_index, assignment].sum()
    conditionals = torch.stack(
        [
            _attention_graph_conditional_scores(index, assignment, unary, pairs, potentials)
            for index in range(num_points)
        ]
    )
    return {
        "assignment": assignment,
        "unary_assignment": unary_assignment,
        "bp_assignment": bp_assignment,
        "bp_refined_assignment": bp_refined,
        "unary_refined_assignment": unary_refined,
        "beliefs": beliefs,
        "conditional_scores": conditionals,
        "pairs": pairs,
        "potentials": potentials,
        "edge_count": edge_count,
        "unary_start_energy": unary_start_energy,
        "unary_assignment_unary_energy": unary_assignment_unary_energy,
        "unary_assignment_pairwise_energy": (
            unary_start_energy - unary_assignment_unary_energy
        ),
        "selected_energy": selected_energy,
        "selected_unary_energy": selected_unary_energy,
        "selected_pairwise_energy": selected_energy - selected_unary_energy,
        "bp_energy": bp_energy,
        "unary_refined_energy": unary_refined_energy,
        "selected_start": selected_start,
    }


def _attention_relational_graph_matching_rankings(
    src_descriptor_tokens: torch.Tensor,
    trg_descriptor_tokens: torch.Tensor,
    attention: dict[str, torch.Tensor],
    source_points: Sequence[Sequence[float]],
    source_size: Sequence[int],
    target_size: Sequence[int],
    src_state: FluxReplayState,
    trg_state: FluxReplayState,
    *,
    candidate_topk: int = 20,
    target_points: Sequence[Sequence[float]] | None = None,
    pck_threshold: float | None = None,
) -> tuple[torch.Tensor, list[dict[str, Any]], dict[str, Any]]:
    """Select only mutual-attention candidates with joint feature-relation inference."""

    points = list(source_points)
    targets = list(target_points) if target_points is not None else None
    if not points:
        return torch.empty((0, 0), device=src_descriptor_tokens.device, dtype=torch.long), [], {}
    if targets is not None and len(targets) != len(points):
        raise ValueError("target_points must align with source_points")
    device = src_descriptor_tokens.device
    src_cells = _native_cell_indices_for_points(
        points, source_size, src_state.image_height, src_state.image_width, device
    )
    mutual = torch.sqrt(
        (attention["p_ab"].float() * attention["p_ba"].float().t()).clamp_min(0.0)
    )
    mutual = torch.nan_to_num(mutual, nan=0.0, posinf=0.0, neginf=0.0).to(device)
    candidate_count = min(max(1, int(candidate_topk)), int(mutual.shape[1]))
    attention_scores, candidate_cells = torch.topk(
        mutual[src_cells], k=candidate_count, dim=1, sorted=True
    )
    candidate_pixels = _cell_topk_to_pixel_indices(candidate_cells, target_size, trg_state)

    source_descriptors = F.normalize(
        torch.nan_to_num(src_descriptor_tokens.float())[src_cells], dim=1, eps=1e-12
    )
    target_descriptors = F.normalize(
        torch.nan_to_num(trg_descriptor_tokens.float()), dim=1, eps=1e-12
    )
    descriptor_scores = (
        target_descriptors[candidate_cells] * source_descriptors[:, None, :]
    ).sum(dim=2)
    attention_unary = _standardize_finite(attention_scores.clamp_min(1e-12).log(), dim=1)
    descriptor_unary = _standardize_finite(descriptor_scores, dim=1)
    fused_unary = 0.5 * (attention_unary + descriptor_unary)
    solution = _solve_attention_relational_graph(
        fused_unary,
        candidate_cells,
        source_descriptors,
        target_descriptors,
    )
    assignment = solution["assignment"]
    beliefs = solution["beliefs"]
    conditionals = solution["conditional_scores"]
    pairwise_contribution = conditionals - fused_unary

    def _ordered_states(scores: torch.Tensor, selected: int | None = None) -> list[int]:
        order = torch.argsort(scores, descending=True).detach().cpu().tolist()
        if selected is not None:
            order = [selected] + [index for index in order if index != selected]
        return [int(index) for index in order]

    def _hit_summary(order: Sequence[int], row: int) -> dict[str, bool]:
        topks = (1, 3, 5, 10, 20)
        if targets is None or pck_threshold is None:
            return {f"@{k}": False for k in topks}
        hits = []
        target_w = int(target_size[1])
        for state_index in order:
            pixel = int(candidate_pixels[row, state_index])
            hits.append(
                _point_hit(
                    [int(pixel % target_w), int(pixel // target_w)],
                    targets[row],
                    float(pck_threshold),
                )
            )
        return {f"@{k}": bool(any(hits[: min(k, len(hits))])) for k in topks}

    target_w = int(target_size[1])
    ranked_rows: list[list[int]] = []
    audits: list[dict[str, Any]] = []
    selected_ranks: list[int] = []
    selected_pairwise: list[float] = []
    for row in range(len(points)):
        selected_state = int(assignment[row])
        final_order = _ordered_states(conditionals[row], selected_state)
        attention_order = list(range(candidate_count))
        descriptor_order = _ordered_states(descriptor_scores[row])
        unary_order = _ordered_states(fused_unary[row])
        ranked_rows.append([int(candidate_pixels[row, index]) for index in final_order])
        selected_ranks.append(selected_state + 1)
        selected_pairwise.append(float(pairwise_contribution[row, selected_state]))

        candidates: list[dict[str, Any]] = []
        final_rank_by_state = {state: rank for rank, state in enumerate(final_order, start=1)}
        for state_index in final_order:
            pixel = int(candidate_pixels[row, state_index])
            xy = [int(pixel % target_w), int(pixel // target_w)]
            hit = bool(
                targets is not None
                and pck_threshold is not None
                and _point_hit(xy, targets[row], float(pck_threshold))
            )
            candidates.append(
                {
                    "rank": int(final_rank_by_state[state_index]),
                    "attention_rank": int(state_index + 1),
                    "target_cell": int(candidate_cells[row, state_index]),
                    "pixel_index": pixel,
                    "pixel": xy,
                    "pck_hit": hit,
                    "scores": {
                        "attention_posterior": float(attention_scores[row, state_index]),
                        "attention_unary": float(attention_unary[row, state_index]),
                        "descriptor_cosine": float(descriptor_scores[row, state_index]),
                        "descriptor_unary": float(descriptor_unary[row, state_index]),
                        "fused_unary": float(fused_unary[row, state_index]),
                        "graph_belief": float(beliefs[row, state_index]),
                        "graph_conditional": float(conditionals[row, state_index]),
                        "pairwise_relation_contribution": float(
                            pairwise_contribution[row, state_index]
                        ),
                    },
                }
            )
        audits.append(
            {
                "candidate_count": candidate_count,
                "native_candidate_injected": False,
                "native_fallback_used": False,
                "selected_state": selected_state,
                "selected_attention_rank": selected_state + 1,
                "unary_state": int(solution["unary_assignment"][row]),
                "bp_state": int(solution["bp_assignment"][row]),
                "bp_changed_assignment": bool(
                    int(solution["bp_assignment"][row])
                    != int(solution["unary_assignment"][row])
                ),
                "bp_icm_changed_assignment": bool(
                    int(solution["bp_refined_assignment"][row])
                    != int(solution["bp_assignment"][row])
                ),
                "unary_icm_changed_assignment": bool(
                    int(solution["unary_refined_assignment"][row])
                    != int(solution["unary_assignment"][row])
                ),
                "final_changed_from_unary": bool(
                    selected_state != int(solution["unary_assignment"][row])
                ),
                "selected_pairwise_relation_contribution": float(
                    pairwise_contribution[row, selected_state]
                ),
                "topk_hits": {
                    "attention": _hit_summary(attention_order, row),
                    "descriptor_unary": _hit_summary(descriptor_order, row),
                    "fused_unary": _hit_summary(unary_order, row),
                    "relational_graph": _hit_summary(final_order, row),
                },
                "candidates": candidates,
            }
        )

    unary_start_energy = float(solution["unary_start_energy"])
    selected_energy = float(solution["selected_energy"])
    summary: dict[str, Any] = {
        "candidate_pool_mean": float(candidate_count),
        "edge_count": int(solution["edge_count"]),
        "unary_start_energy": unary_start_energy,
        "unary_assignment_unary_energy": float(solution["unary_assignment_unary_energy"]),
        "unary_assignment_pairwise_energy": float(solution["unary_assignment_pairwise_energy"]),
        "selected_energy": selected_energy,
        "selected_unary_energy": float(solution["selected_unary_energy"]),
        "selected_pairwise_energy": float(solution["selected_pairwise_energy"]),
        "energy_gain": selected_energy - unary_start_energy,
        "bp_energy": float(solution["bp_energy"]),
        "unary_refined_energy": float(solution["unary_refined_energy"]),
        "selected_start": str(solution["selected_start"]),
        "bp_changed_count": int(
            (solution["bp_assignment"] != solution["unary_assignment"]).sum()
        ),
        "bp_icm_changed_count": int(
            (solution["bp_refined_assignment"] != solution["bp_assignment"]).sum()
        ),
        "unary_icm_changed_count": int(
            (solution["unary_refined_assignment"] != solution["unary_assignment"]).sum()
        ),
        "final_changed_from_unary_count": int(
            (assignment != solution["unary_assignment"]).sum()
        ),
        "selected_attention_rank_mean": float(sum(selected_ranks) / len(selected_ranks)),
        "selected_pairwise_relation_contribution_mean": float(
            sum(selected_pairwise) / len(selected_pairwise)
        ),
        "native_injected_candidate_count": 0,
        "native_fallback_count": 0,
        "candidate_source": "mutual_cross_attention_topk_only",
        "pairwise_mechanism": [
            "feature_cosine_preservation",
            "descriptor_difference_direction",
            "channel_coactivation_preservation",
        ],
        "coordinate_candidate_score_count": 0,
    }
    return torch.tensor(ranked_rows, device=device, dtype=torch.long), audits, summary


def _attention_basin_pixel_candidates(
    candidate_cells: torch.Tensor,
    target_size: Sequence[int],
    trg_state: FluxReplayState,
) -> torch.Tensor:
    """Expand attention token proposals to their full-resolution image cells."""

    target_h, target_w = int(target_size[0]), int(target_size[1])
    trg_grid_h, trg_grid_w = int(trg_state.image_height), int(trg_state.image_width)
    rows: list[list[int]] = []
    max_count = 0
    for cells in candidate_cells.detach().cpu().tolist():
        pixels: list[int] = []
        seen: set[int] = set()
        for cell in cells:
            cell = int(cell)
            cell_x = cell % trg_grid_w
            cell_y = cell // trg_grid_w
            x0 = int(math.floor(float(cell_x) * float(target_w) / float(trg_grid_w)))
            x1 = int(math.ceil(float(cell_x + 1) * float(target_w) / float(trg_grid_w)))
            y0 = int(math.floor(float(cell_y) * float(target_h) / float(trg_grid_h)))
            y1 = int(math.ceil(float(cell_y + 1) * float(target_h) / float(trg_grid_h)))
            x0 = max(0, min(target_w - 1, x0))
            x1 = max(x0 + 1, min(target_w, x1))
            y0 = max(0, min(target_h - 1, y0))
            y1 = max(y0 + 1, min(target_h, y1))
            for y in range(y0, y1):
                base = y * target_w
                for x in range(x0, x1):
                    pixel = base + x
                    if pixel not in seen:
                        seen.add(pixel)
                        pixels.append(pixel)
        if not pixels:
            pixels = [0]
        rows.append(pixels)
        max_count = max(max_count, len(pixels))
    padded = [row + row[-1:] * (max_count - len(row)) for row in rows]
    return torch.tensor(padded, device=candidate_cells.device, dtype=torch.long)


def _candidate_field_consistency_audit_for_points(
    src_features: torch.Tensor,
    trg_features: torch.Tensor,
    mutual_attention: torch.Tensor,
    src_cells: torch.Tensor,
    source_points: Sequence[Sequence[float]],
    source_size: Sequence[int],
    target_size: Sequence[int],
    trg_state: FluxReplayState,
    *,
    target_points: Sequence[Sequence[float]] | None,
    pck_threshold: float | None,
    candidate_topk: int,
    field_topm: int,
    candidate_source: str = "native_basin",
) -> list[dict[str, Any]]:
    """Audit whether multi-point candidate-field consistency can decode attention proposals."""

    points = list(source_points)
    if not points:
        return []
    device = src_features.device
    target_h, target_w = int(target_size[0]), int(target_size[1])
    candidate_count = min(max(1, int(candidate_topk)), int(mutual_attention.shape[1]))
    field_topm = max(1, int(field_topm))
    candidate_cells = torch.topk(
        mutual_attention[src_cells.to(mutual_attention.device)],
        k=candidate_count,
        dim=1,
        sorted=True,
    ).indices
    candidate_source = str(candidate_source)
    if candidate_source == "native_basin":
        basin_pixels = _attention_basin_pixel_candidates(candidate_cells, target_size, trg_state).to(device)
        descriptor_scores, descriptor_pixels = _chunked_descriptor_topk_scores_indices(
            src_features,
            trg_features,
            points,
            source_size,
            target_size,
            topk=min(field_topm, int(basin_pixels.shape[1])),
            candidate_indices=basin_pixels,
        )
        source_candidate_count = int(basin_pixels.shape[1])
    elif candidate_source == "attention_tokens":
        descriptor_pixels = _cell_topk_to_pixel_indices(
            candidate_cells[:, : min(field_topm, candidate_count)],
            target_size,
            trg_state,
        ).to(device)
        descriptor_scores, descriptor_pixels = _chunked_descriptor_topk_scores_indices(
            src_features,
            trg_features,
            points,
            source_size,
            target_size,
            topk=int(descriptor_pixels.shape[1]),
            candidate_indices=descriptor_pixels,
        )
        source_candidate_count = int(candidate_count)
    else:
        raise ValueError(f"unsupported candidate field source: {candidate_source}")
    point_count, candidate_m = descriptor_pixels.shape
    if candidate_m == 0:
        return []
    source_xy = torch.tensor(
        [[float(point[0]), float(point[1])] for point in points],
        device=device,
        dtype=torch.float32,
    )
    candidate_xy = torch.stack(
        (
            (descriptor_pixels % target_w).float(),
            torch.div(descriptor_pixels, target_w, rounding_mode="floor").float(),
        ),
        dim=2,
    )
    unary_norm = torch.stack([_rank_normalize(row) for row in descriptor_scores.float()], dim=0)
    field_scores = torch.zeros((point_count, candidate_m), device=device, dtype=torch.float32)
    geometry_scores = torch.zeros_like(field_scores)
    source_scale = torch.tensor(
        [max(1.0, float(source_size[1])), max(1.0, float(source_size[0]))],
        device=device,
        dtype=torch.float32,
    )
    target_scale = torch.tensor(
        [max(1.0, float(target_w)), max(1.0, float(target_h))],
        device=device,
        dtype=torch.float32,
    )
    for i in range(point_count):
        support_rows = []
        geometry_rows = []
        for j in range(point_count):
            if i == j:
                continue
            src_delta = (source_xy[j] - source_xy[i]) / source_scale
            target_delta = (candidate_xy[j][None, :, :] - candidate_xy[i][:, None, :]) / target_scale
            geometry = 1.0 / (1.0 + torch.linalg.vector_norm(target_delta - src_delta.reshape(1, 1, 2), dim=2))
            support_rows.append((geometry * unary_norm[j].reshape(1, -1)).max(dim=1).values)
            geometry_rows.append(geometry.max(dim=1).values)
        if support_rows:
            support = torch.stack(support_rows, dim=0).mean(dim=0)
            geometry_support = torch.stack(geometry_rows, dim=0).mean(dim=0)
        else:
            support = torch.ones(candidate_m, device=device)
            geometry_support = torch.ones(candidate_m, device=device)
        geometry_scores[i] = geometry_support
        field_scores[i] = torch.sqrt((unary_norm[i].clamp_min(0.0) * support.clamp_min(0.0)).clamp_min(0.0))

    rows: list[dict[str, Any]] = []
    score_tensors = {
        "candidate_field_unary": unary_norm,
        "candidate_field_geometry": geometry_scores,
        "candidate_field_consistency": field_scores,
    }
    for row in range(point_count):
        target = target_points[row] if target_points is not None else None
        hits = [
            bool(_point_hit([int(pixel % target_w), int(pixel // target_w)], target, pck_threshold))
            if target is not None and pck_threshold is not None
            else False
            for pixel in descriptor_pixels[row].detach().cpu().tolist()
        ]
        ranks = {
            name: _rank_first_hit([float(value) for value in tensor[row].detach().cpu().tolist()], hits)
            for name, tensor in score_tensors.items()
        }
        topk_hits = {
            f"{name}@{k}": bool(
                any(
                    hits[index]
                    for index in sorted(
                        range(candidate_m),
                        key=lambda idx: float(tensor[row, idx].detach().cpu()),
                        reverse=True,
                    )[: min(k, candidate_m)]
                )
            )
            for name, tensor in score_tensors.items()
            for k in (1, 3, 5, 10, 20)
        }
        candidates = []
        order = sorted(
            range(candidate_m),
            key=lambda index: float(field_scores[row, index].detach().cpu()),
            reverse=True,
        )
        for rank, index in enumerate(order, start=1):
            pixel = int(descriptor_pixels[row, index].detach().cpu())
            candidates.append({
                "rank": int(rank),
                "pixel": [int(pixel % target_w), int(pixel // target_w)],
                "pixel_index": int(pixel),
                "pck_hit": bool(hits[index]),
                "scores": {
                    name: float(tensor[row, index].detach().cpu())
                    for name, tensor in score_tensors.items()
                },
            })
        rows.append({
            "candidate_source": candidate_source,
            "candidate_topk": int(candidate_count),
            "field_topm": int(candidate_m),
            "source_candidate_count": int(source_candidate_count),
            "basin_pixel_count": int(source_candidate_count),
            "score_names": list(score_tensors.keys()),
            "ranks": ranks,
            "topk_hits": topk_hits,
            "candidates": candidates,
        })
    return rows


def _basin_identity_audit_for_points(
    src_features: torch.Tensor,
    trg_features: torch.Tensor,
    mutual_attention: torch.Tensor,
    src_cells: torch.Tensor,
    source_points: Sequence[Sequence[float]],
    source_size: Sequence[int],
    target_size: Sequence[int],
    src_state: FluxReplayState,
    trg_state: FluxReplayState,
    *,
    target_points: Sequence[Sequence[float]] | None,
    pck_threshold: float | None,
    basin_topk: int,
    radius: int,
    rank_topks: Sequence[int],
) -> list[dict[str, Any]]:
    """Audit whether native identity can disambiguate inside attention basins."""

    points = list(source_points)
    if not points:
        return []
    rank_topks = tuple(sorted({max(1, int(k)) for k in rank_topks}))
    basin_k = min(max(1, int(basin_topk)), int(mutual_attention.shape[1]))
    src_cells = src_cells.to(device=mutual_attention.device, dtype=torch.long).flatten()
    raw_rows = mutual_attention[src_cells].float()
    filtered_rows = raw_rows * _local_transport_support_rows(
        mutual_attention,
        src_cells,
        src_state,
        trg_state,
        radius=radius,
    )
    raw_cells = torch.topk(raw_rows, k=basin_k, dim=1, sorted=True).indices
    filtered_cells = torch.topk(filtered_rows, k=basin_k, dim=1, sorted=True).indices
    raw_pixels = _cell_topk_to_pixel_indices(raw_cells, target_size, trg_state).to(src_features.device)
    filtered_pixels = _cell_topk_to_pixel_indices(filtered_cells, target_size, trg_state).to(src_features.device)

    raw_scores, raw_native_sorted = _chunked_descriptor_topk_scores_indices(
        src_features,
        trg_features,
        points,
        source_size,
        target_size,
        topk=basin_k,
        candidate_indices=raw_pixels,
    )
    filtered_scores, filtered_native_sorted = _chunked_descriptor_topk_scores_indices(
        src_features,
        trg_features,
        points,
        source_size,
        target_size,
        topk=basin_k,
        candidate_indices=filtered_pixels,
    )
    native_sorted = {
        "raw_basin_native_descriptor": raw_native_sorted,
        "filtered_basin_native_descriptor": filtered_native_sorted,
    }
    native_scores = {
        "raw_basin_native_descriptor": raw_scores,
        "filtered_basin_native_descriptor": filtered_scores,
    }
    attention_pixels = {
        "raw_basin_native_descriptor": raw_pixels,
        "filtered_basin_native_descriptor": filtered_pixels,
    }
    signal_hits = {
        name: _per_point_topk_hits(indices, target_points, pck_threshold, target_size, rank_topks)
        for name, indices in native_sorted.items()
    }
    attention_hits = {
        name: _per_point_topk_hits(indices, target_points, pck_threshold, target_size, rank_topks)
        for name, indices in attention_pixels.items()
    }

    target_h, target_w = int(target_size[0]), int(target_size[1])
    rows = []
    for row in range(len(points)):
        audit = {
            "basin_topk": int(basin_k),
            "radius": int(max(0, radius)),
            "rank_topks": [int(k) for k in rank_topks],
            "score_names": list(native_sorted.keys()),
            "ranks": {},
            "topk_hits": {},
            "basins": {},
        }
        for name in native_sorted:
            indices = native_sorted[name][row]
            scores = native_scores[name][row]
            attn_indices = attention_pixels[name][row]
            pck_rank = signal_hits[name][row]["rank"]
            audit["ranks"][name] = pck_rank
            for k in rank_topks:
                audit["topk_hits"][f"{name}@{int(k)}"] = bool(signal_hits[name][row]["topk_hits"][int(k)])
            pck_flags = []
            target = target_points[row] if target_points is not None else None
            for pixel in attn_indices.detach().cpu().tolist():
                xy = [int(pixel % target_w), int(pixel // target_w)]
                pck_flags.append(
                    bool(_point_hit(xy, target, pck_threshold))
                    if target is not None and pck_threshold is not None
                    else False
                )
            top_pixel = int(indices[0].detach().cpu())
            top_xy = [int(top_pixel % target_w), int(top_pixel // target_w)]
            attention_top_pixel = int(attn_indices[0].detach().cpu())
            attention_top_xy = [int(attention_top_pixel % target_w), int(attention_top_pixel // target_w)]
            native_top_attention_rank = None
            for rank, pixel in enumerate(attn_indices.detach().cpu().tolist(), start=1):
                if int(pixel) == top_pixel:
                    native_top_attention_rank = int(rank)
                    break
            audit["basins"][name] = {
                "attention_basin_has_pck_hit": bool(any(pck_flags)),
                "attention_basin_pck_hit_count": int(sum(1 for flag in pck_flags if flag)),
                "attention_basin_pck_rank": attention_hits[name][row]["rank"],
                "attention_top1": {
                    "pixel": attention_top_xy,
                    "pixel_index": int(attention_top_pixel),
                    "pck_hit": bool(pck_flags[0]) if pck_flags else False,
                },
                "native_top1_in_basin": {
                    "pixel": top_xy,
                    "pixel_index": int(top_pixel),
                    "score": float(scores[0].detach().cpu()),
                    "pck_hit": (
                        bool(_point_hit(top_xy, target, pck_threshold))
                        if target is not None and pck_threshold is not None
                        else False
                    ),
                    "attention_rank": native_top_attention_rank,
                },
            }
        rows.append(audit)
    return rows


def _local_patch_feature_grid(
    features: torch.Tensor,
    center_pixels: torch.Tensor,
    image_size: Sequence[int],
    stride_xy: tuple[float, float],
    *,
    radius: int,
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """Sample normalized local feature grids around full-resolution centers."""

    offsets = _cell_offsets(radius)
    if not offsets:
        offsets = [(0, 0)]
    centers = center_pixels.to(device=features.device, dtype=torch.float32).reshape(-1, 2)
    offset_tensor = torch.tensor(
        [[float(dx) * float(stride_xy[0]), float(dy) * float(stride_xy[1])] for dx, dy in offsets],
        device=features.device,
        dtype=torch.float32,
    )
    sample_pixels = (centers[:, None, :] + offset_tensor[None, :, :]).reshape(-1, 2)
    vectors = _sample_feature_vectors_at_pixels(features, sample_pixels, image_size)
    vectors = F.normalize(torch.nan_to_num(vectors.float(), nan=0.0, posinf=0.0, neginf=0.0), dim=1, eps=1e-12)
    return vectors.reshape(centers.shape[0], len(offsets), -1), offsets


def _local_self_similarity_from_patch(patch: torch.Tensor, offsets: Sequence[tuple[int, int]]) -> torch.Tensor:
    center_index = 0
    for index, offset in enumerate(offsets):
        if int(offset[0]) == 0 and int(offset[1]) == 0:
            center_index = index
            break
    center = patch[:, center_index:center_index + 1, :]
    similarities = (center * patch).sum(dim=2)
    keep = [
        index
        for index, offset in enumerate(offsets)
        if not (int(offset[0]) == 0 and int(offset[1]) == 0)
    ]
    if not keep:
        return similarities[:, center_index:center_index + 1]
    return similarities[:, keep]


def _attention_row_differential_metrics(
    mutual_attention: torch.Tensor,
    src_cell: int,
    candidate_cell: int,
    src_state: FluxReplayState,
    trg_state: FluxReplayState,
    *,
    radius: int,
) -> dict[str, float | int]:
    """Compare neighboring source attention rows after aligning target neighborhoods."""

    src_w, src_h = int(src_state.image_width), int(src_state.image_height)
    trg_w, trg_h = int(trg_state.image_width), int(trg_state.image_height)
    src_x, src_y = int(src_cell % src_w), int(src_cell // src_w)
    cand_x, cand_y = int(candidate_cell % trg_w), int(candidate_cell // trg_w)
    offsets = _cell_offsets(radius)
    center_row = mutual_attention[int(src_cell)].float()
    center_patch_offsets = [
        (dx, dy)
        for dx, dy in offsets
        if _valid_cell(cand_x + dx, cand_y + dy, trg_w, trg_h)
    ]
    center_patch_cells = [
        _cell_index(cand_x + dx, cand_y + dy, trg_w)
        for dx, dy in center_patch_offsets
    ]
    if not center_patch_cells:
        return {
            "valid_neighbor_count": 0,
            "row_differential_cosine": 0.0,
            "expected_support": 0.0,
            "peak_alignment": 0.0,
            "attention_row_differential": 0.0,
        }
    center_patch = center_row[torch.tensor(center_patch_cells, device=mutual_attention.device, dtype=torch.long)]
    center_patch = center_patch - center_patch.mean()
    center_norm = center_patch.norm().clamp_min(1e-12)
    scale_x = float(trg_w) / float(max(1, src_w))
    scale_y = float(trg_h) / float(max(1, src_h))
    cosines = []
    supports = []
    peak_alignments = []
    products = []
    for dx, dy in offsets:
        if dx == 0 and dy == 0:
            continue
        nsx, nsy = src_x + dx, src_y + dy
        if not _valid_cell(nsx, nsy, src_w, src_h):
            continue
        expected_dx = int(round(float(dx) * scale_x))
        expected_dy = int(round(float(dy) * scale_y))
        ntx, nty = cand_x + expected_dx, cand_y + expected_dy
        if not _valid_cell(ntx, nty, trg_w, trg_h):
            continue
        if not all(_valid_cell(ntx + pdx, nty + pdy, trg_w, trg_h) for pdx, pdy in center_patch_offsets):
            continue
        neighbor_patch_cells = [
            _cell_index(ntx + pdx, nty + pdy, trg_w)
            for pdx, pdy in center_patch_offsets
        ]
        if len(neighbor_patch_cells) != len(center_patch_cells):
            continue
        neighbor_cell = _cell_index(nsx, nsy, src_w)
        row = mutual_attention[neighbor_cell].float()
        row_max = row.max().clamp_min(1e-12)
        expected_cell = _cell_index(ntx, nty, trg_w)
        support = (row[expected_cell] / row_max).clamp_min(0.0)
        neighbor_patch = row[torch.tensor(neighbor_patch_cells, device=mutual_attention.device, dtype=torch.long)]
        neighbor_patch = neighbor_patch - neighbor_patch.mean()
        cosine = ((center_patch * neighbor_patch).sum() / (center_norm * neighbor_patch.norm().clamp_min(1e-12))).clamp(-1.0, 1.0)
        local_argmax = int(torch.argmax(row[torch.tensor(neighbor_patch_cells, device=mutual_attention.device, dtype=torch.long)]).detach().cpu())
        peak_cell = int(neighbor_patch_cells[local_argmax])
        peak_alignment = 1.0 if peak_cell == int(expected_cell) else 0.0
        cosines.append(cosine)
        supports.append(support)
        peak_alignments.append(torch.tensor(float(peak_alignment), device=mutual_attention.device))
        products.append(cosine.clamp_min(0.0) * support)

    def _mean(values: list[torch.Tensor]) -> float:
        if not values:
            return 0.0
        return float(torch.stack(values).mean().detach().cpu())

    return {
        "valid_neighbor_count": int(len(products)),
        "row_differential_cosine": _mean(cosines),
        "expected_support": _mean(supports),
        "peak_alignment": _mean(peak_alignments),
        "attention_row_differential": _mean(products),
    }


def _local_relational_identity_audit_for_points(
    src_features: torch.Tensor,
    trg_features: torch.Tensor,
    mutual_attention: torch.Tensor,
    src_cells: torch.Tensor,
    source_points: Sequence[Sequence[float]],
    proposal_pixels: torch.Tensor,
    source_size: Sequence[int],
    target_size: Sequence[int],
    src_state: FluxReplayState,
    trg_state: FluxReplayState,
    *,
    target_points: Sequence[Sequence[float]] | None,
    pck_threshold: float | None,
    radius: int,
) -> list[dict[str, Any]]:
    """Audit local relational identity signals inside attention proposals."""

    points = list(source_points)
    if not points:
        return []
    radius = max(1, int(radius))
    target_h, target_w = int(target_size[0]), int(target_size[1])
    src_stride = (
        float(source_size[1]) / float(max(1, src_state.image_width)),
        float(source_size[0]) / float(max(1, src_state.image_height)),
    )
    trg_stride = (
        float(target_size[1]) / float(max(1, trg_state.image_width)),
        float(target_size[0]) / float(max(1, trg_state.image_height)),
    )
    source_pixel_tensor = torch.tensor(
        [[float(point[0]), float(point[1])] for point in points],
        device=src_features.device,
        dtype=torch.float32,
    )
    src_patch, offsets = _local_patch_feature_grid(
        src_features,
        source_pixel_tensor,
        source_size,
        src_stride,
        radius=radius,
    )
    src_lss = _local_self_similarity_from_patch(src_patch, offsets)
    src_lss = F.normalize(src_lss, dim=1, eps=1e-12)

    rows: list[dict[str, Any]] = []
    score_names = (
        "native_patch_correlation",
        "local_self_similarity_consistency",
        "attention_row_differential",
        "hybrid_local_relational_identity",
    )
    for row, point in enumerate(points):
        del point
        ordered_pixels: list[int] = []
        seen_pixels: set[int] = set()
        for pixel in proposal_pixels[row].detach().cpu().tolist():
            pixel = int(pixel)
            if pixel not in seen_pixels:
                ordered_pixels.append(pixel)
                seen_pixels.add(pixel)
        if not ordered_pixels:
            rows.append({
                "radius": int(radius),
                "proposal_count": 0,
                "score_names": list(score_names),
                "ranks": {name: None for name in score_names},
                "score_gaps": {},
                "candidates": [],
            })
            continue
        candidate_pixels = torch.tensor(ordered_pixels, device=trg_features.device, dtype=torch.long)
        candidate_xy = _pixel_indices_to_xy(candidate_pixels, target_size).to(device=trg_features.device)
        trg_patch, _ = _local_patch_feature_grid(
            trg_features,
            candidate_xy,
            target_size,
            trg_stride,
            radius=radius,
        )
        patch_corr = torch.matmul(src_patch[row:row + 1].expand_as(trg_patch), trg_patch.transpose(1, 2))
        patch_diag = torch.diagonal(patch_corr, dim1=1, dim2=2)
        native_patch_correlation = patch_diag.mean(dim=1)
        trg_lss = F.normalize(_local_self_similarity_from_patch(trg_patch, offsets), dim=1, eps=1e-12)
        local_self_similarity = F.cosine_similarity(
            src_lss[row:row + 1].expand_as(trg_lss),
            trg_lss,
            dim=1,
            eps=1e-12,
        )
        candidate_cells = _pixel_indices_to_replay_cells(candidate_pixels, target_size, trg_state)
        attention_metrics = [
            _attention_row_differential_metrics(
                mutual_attention,
                int(src_cells[row].detach().cpu()),
                int(cell.detach().cpu()),
                src_state,
                trg_state,
                radius=radius,
            )
            for cell in candidate_cells
        ]
        attention_row_differential = [
            float(metric["attention_row_differential"])
            for metric in attention_metrics
        ]
        base_scores = {
            "native_patch_correlation": [float(value) for value in native_patch_correlation.detach().cpu().tolist()],
            "local_self_similarity_consistency": [float(value) for value in local_self_similarity.detach().cpu().tolist()],
            "attention_row_differential": attention_row_differential,
        }
        rank_positions: dict[str, dict[int, int]] = {}
        for name, values in base_scores.items():
            order = sorted(range(len(values)), key=lambda index: values[index], reverse=True)
            rank_positions[name] = {index: rank for rank, index in enumerate(order, start=1)}
        hybrid_scores = [
            -float(
                rank_positions["native_patch_correlation"][index]
                + rank_positions["local_self_similarity_consistency"][index]
                + rank_positions["attention_row_differential"][index]
            ) / 3.0
            for index in range(len(ordered_pixels))
        ]
        signal_scores = {**base_scores, "hybrid_local_relational_identity": hybrid_scores}
        hits: list[bool] = []
        candidates: list[dict[str, Any]] = []
        target = target_points[row] if target_points is not None else None
        for index, pixel in enumerate(ordered_pixels):
            xy = [int(pixel % target_w), int(pixel // target_w)]
            hit = bool(_point_hit(xy, target, pck_threshold)) if target is not None and pck_threshold is not None else False
            hits.append(hit)
            metrics = dict(attention_metrics[index])
            metrics.update({
                "patch_correlation_diagonal_mean": float(base_scores["native_patch_correlation"][index]),
                "patch_correlation_diagonal_max": float(patch_diag[index].max().detach().cpu()),
                "patch_correlation_diagonal_min": float(patch_diag[index].min().detach().cpu()),
                "patch_correlation_matrix_mean": float(patch_corr[index].mean().detach().cpu()),
                "local_self_similarity_consistency": float(base_scores["local_self_similarity_consistency"][index]),
            })
            candidates.append({
                "rank_attention": int(index + 1),
                "pixel": xy,
                "pixel_index": int(pixel),
                "pck_hit": hit,
                "replay_cell": int(candidate_cells[index].detach().cpu()),
                "scores": {name: float(signal_scores[name][index]) for name in score_names},
                "metrics": metrics,
            })
        ranks = {name: _rank_first_hit(values, hits) for name, values in signal_scores.items()}
        score_gaps = {}
        for name, values in signal_scores.items():
            top1 = values[0] if values else None
            hit_values = [value for value, hit in zip(values, hits) if hit]
            best_hit = max(hit_values) if hit_values else None
            score_gaps[f"{name}_attention_top1_minus_best_pck_hit_proposal"] = (
                float(top1 - best_hit) if top1 is not None and best_hit is not None else None
            )
        rows.append({
            "radius": int(radius),
            "proposal_count": len(candidates),
            "score_names": list(score_names),
            "ranks": ranks,
            "score_gaps": score_gaps,
            "candidates": candidates,
        })
    return rows


def _feature_drift_row(native: torch.Tensor, joint: torch.Tensor) -> dict[str, float]:
    native = torch.nan_to_num(native.float(), nan=0.0, posinf=0.0, neginf=0.0)
    joint = torch.nan_to_num(joint.float(), nan=0.0, posinf=0.0, neginf=0.0)
    cosine = F.cosine_similarity(joint.reshape(1, -1), native.reshape(1, -1), dim=1)[0]
    delta = joint - native
    native_norm = native.norm().clamp_min(1e-12)
    return {
        "joint_native_cosine": float(cosine.detach().cpu()),
        "drift_l2": float(delta.norm().detach().cpu()),
        "native_l2": float(native_norm.detach().cpu()),
        "drift_l2_ratio": float((delta.norm() / native_norm).detach().cpu()),
    }


def _operator_manifold_audit_for_points(
    src_native_prepared: torch.Tensor,
    trg_native_prepared: torch.Tensor,
    src_joint_prepared: torch.Tensor,
    trg_joint_prepared: torch.Tensor,
    source_points: Sequence[Sequence[float]],
    source_size: Sequence[int],
    target_points: Sequence[Sequence[float]] | None,
    target_size: Sequence[int],
    src_state: FluxReplayState,
    trg_state: FluxReplayState,
) -> list[dict[str, Any]]:
    """Per-keypoint drift between native and pair-conditioned prepared tokens."""

    points = list(source_points)
    if not points:
        return []
    device = src_native_prepared.device
    src_cells = _native_cell_indices_for_points(
        points,
        source_size,
        src_state.image_height,
        src_state.image_width,
        device,
    )
    trg_cells = None
    if target_points is not None:
        trg_cells = _native_cell_indices_for_points(
            target_points,
            target_size,
            trg_state.image_height,
            trg_state.image_width,
            device,
        )
    source_cosine_all = F.cosine_similarity(src_joint_prepared, src_native_prepared, dim=1)
    target_cosine_all = F.cosine_similarity(trg_joint_prepared, trg_native_prepared, dim=1)
    rows = []
    for index, src_cell in enumerate(src_cells.detach().cpu().tolist()):
        src_cell = int(src_cell)
        source = _feature_drift_row(
            src_native_prepared[src_cell],
            src_joint_prepared[src_cell],
        )
        row = {
            "source_cell": src_cell,
            "source": source,
            "pair": {
                "source_joint_native_cosine_mean": float(source_cosine_all.mean().detach().cpu()),
                "source_joint_native_cosine_min": float(source_cosine_all.min().detach().cpu()),
                "target_joint_native_cosine_mean": float(target_cosine_all.mean().detach().cpu()),
                "target_joint_native_cosine_min": float(target_cosine_all.min().detach().cpu()),
                "source_drift_l2_ratio_mean": float(
                    ((src_joint_prepared - src_native_prepared).norm(dim=1)
                     / src_native_prepared.norm(dim=1).clamp_min(1e-12)).mean().detach().cpu()
                ),
                "target_drift_l2_ratio_mean": float(
                    ((trg_joint_prepared - trg_native_prepared).norm(dim=1)
                     / trg_native_prepared.norm(dim=1).clamp_min(1e-12)).mean().detach().cpu()
                ),
            },
        }
        if trg_cells is not None:
            trg_cell = int(trg_cells[index].detach().cpu())
            row["target_gt_cell"] = trg_cell
            row["target_gt"] = _feature_drift_row(
                trg_native_prepared[trg_cell],
                trg_joint_prepared[trg_cell],
            )
        rows.append(row)
    return rows


def _cell_topk_to_pixel_indices(
    cells: torch.Tensor,
    target_size: Sequence[int],
    trg_state: FluxReplayState,
) -> torch.Tensor:
    target_h, target_w = int(target_size[0]), int(target_size[1])
    cells = cells.long()
    cell_x = (cells % int(trg_state.image_width)).float()
    cell_y = torch.div(cells, int(trg_state.image_width), rounding_mode="floor").float()
    pixel_x = torch.round((cell_x + 0.5) * float(target_w) / float(trg_state.image_width) - 0.5).long()
    pixel_y = torch.round((cell_y + 0.5) * float(target_h) / float(trg_state.image_height) - 0.5).long()
    pixel_x.clamp_(0, target_w - 1)
    pixel_y.clamp_(0, target_h - 1)
    return (pixel_y.to(cells.device) * target_w + pixel_x.to(cells.device)).long()


def _per_point_topk_hits(
    indices: torch.Tensor,
    target_points: Sequence[Sequence[float]] | None,
    pck_threshold: float | None,
    target_size: Sequence[int],
    topks: Sequence[int],
) -> list[dict[str, Any]]:
    topks = tuple(sorted({max(1, int(k)) for k in topks}))
    if target_points is None or pck_threshold is None:
        return [
            {"rank": None, "topk_hits": {int(k): False for k in topks}}
            for _ in range(int(indices.shape[0]))
        ]
    target_h, target_w = int(target_size[0]), int(target_size[1])
    targets = torch.tensor(
        [[float(point[0]), float(point[1])] for point in target_points],
        device=indices.device,
        dtype=torch.float32,
    )
    x = (indices % target_w).float()
    y = torch.div(indices, target_w, rounding_mode="floor").float()
    coords = torch.stack((x, y), dim=-1)
    hits = torch.linalg.vector_norm(coords - targets[:, None, :], dim=2) <= 0.1 * float(pck_threshold)
    rows = []
    for row in range(indices.shape[0]):
        hit_positions = torch.nonzero(hits[row], as_tuple=False).flatten()
        rank = int(hit_positions[0].detach().cpu()) + 1 if hit_positions.numel() else None
        rows.append({
            "rank": rank,
            "topk_hits": {
                int(k): bool(hits[row, : min(int(k), hits.shape[1])].any().detach().cpu())
                for k in topks
            },
        })
    return rows


def _kernel_positive_feature_maps(
    kernel: torch.Tensor,
    src_state: FluxReplayState,
    trg_state: FluxReplayState,
) -> tuple[torch.Tensor, torch.Tensor]:
    src_count = int(src_state.image_height) * int(src_state.image_width)
    trg_count = int(trg_state.image_height) * int(trg_state.image_width)
    kernel = torch.nan_to_num(kernel.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    if kernel.shape != (src_count, trg_count):
        raise ValueError("kernel shape does not match replay grids")
    src_descriptor = F.normalize(kernel, dim=1, eps=1e-12)
    trg_descriptor = torch.eye(trg_count, device=kernel.device, dtype=torch.float32)
    return (
        src_descriptor.t().reshape(1, trg_count, int(src_state.image_height), int(src_state.image_width)).contiguous(),
        trg_descriptor.t().reshape(1, trg_count, int(trg_state.image_height), int(trg_state.image_width)).contiguous(),
    )


def _kernel_svd_feature_maps(
    kernel: torch.Tensor,
    src_state: FluxReplayState,
    trg_state: FluxReplayState,
    *,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    src_count = int(src_state.image_height) * int(src_state.image_width)
    trg_count = int(trg_state.image_height) * int(trg_state.image_width)
    kernel = torch.nan_to_num(kernel.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if kernel.shape != (src_count, trg_count):
        raise ValueError("kernel shape does not match replay grids")
    centered = kernel - kernel.mean(dim=1, keepdim=True)
    centered = centered - centered.mean(dim=0, keepdim=True)
    centered = centered + kernel.mean()
    max_rank = min(max(1, int(rank)), min(src_count, trg_count))
    if centered.square().mean() <= 1e-12:
        src_descriptor = torch.zeros((src_count, max_rank), device=kernel.device, dtype=torch.float32)
        trg_descriptor = torch.zeros((trg_count, max_rank), device=kernel.device, dtype=torch.float32)
    else:
        try:
            u, s, vh = torch.linalg.svd(centered, full_matrices=False)
        except RuntimeError:
            u, s, vh = torch.linalg.svd(centered.cpu(), full_matrices=False)
            u = u.to(kernel.device)
            s = s.to(kernel.device)
            vh = vh.to(kernel.device)
        s = torch.nan_to_num(s[:max_rank].float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        scale = torch.sqrt(s).reshape(1, -1)
        src_descriptor = u[:, :max_rank].float() * scale
        trg_descriptor = vh[:max_rank, :].t().float() * scale
    return (
        src_descriptor.t().reshape(1, max_rank, int(src_state.image_height), int(src_state.image_width)).contiguous(),
        trg_descriptor.t().reshape(1, max_rank, int(trg_state.image_height), int(trg_state.image_width)).contiguous(),
    )


def _native_plus_kernel_feature_maps(
    src_native_map: torch.Tensor,
    trg_native_map: torch.Tensor,
    src_kernel_map: torch.Tensor,
    trg_kernel_map: torch.Tensor,
    *,
    weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight = float(max(0.0, weight))
    src_native = F.normalize(torch.nan_to_num(src_native_map.float(), nan=0.0, posinf=0.0, neginf=0.0), dim=1, eps=1e-12)
    trg_native = F.normalize(torch.nan_to_num(trg_native_map.float(), nan=0.0, posinf=0.0, neginf=0.0), dim=1, eps=1e-12)
    src_kernel = F.normalize(torch.nan_to_num(src_kernel_map.float(), nan=0.0, posinf=0.0, neginf=0.0), dim=1, eps=1e-12)
    trg_kernel = F.normalize(torch.nan_to_num(trg_kernel_map.float(), nan=0.0, posinf=0.0, neginf=0.0), dim=1, eps=1e-12)
    src = F.normalize(torch.cat((src_native, weight * src_kernel), dim=1), dim=1, eps=1e-12)
    trg = F.normalize(torch.cat((trg_native, weight * trg_kernel), dim=1), dim=1, eps=1e-12)
    return src.contiguous(), trg.contiguous()


def _expert_coherence_relative_gates(
    attention: dict[str, torch.Tensor],
    src_state: FluxReplayState,
    trg_state: FluxReplayState,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Measure where early mutual mass survives within-expert reciprocity.

    The raw coherent/early mass ratio is in [0, 1] up to numerical error. The
    fixed transform ``2r / (r + mean(r))`` uses pair-average reliability as
    the unit reference while remaining bounded in [0, 2]. It changes only
    spatial injection strength and introduces no learned or validation-selected
    threshold.
    """

    p_ab = attention.get("p_ab")
    p_ba = attention.get("p_ba")
    coherent = attention.get("expert_coherent_mutual")
    if not all(isinstance(value, torch.Tensor) for value in (p_ab, p_ba, coherent)):
        raise ValueError(
            "expert coherence gating requires p_ab, p_ba, and "
            "expert_coherent_mutual tensors"
        )
    src_count = int(src_state.image_height) * int(src_state.image_width)
    trg_count = int(trg_state.image_height) * int(trg_state.image_width)
    if tuple(p_ab.shape) != (src_count, trg_count):
        raise ValueError("p_ab shape does not match replay grids")
    if tuple(p_ba.shape) != (trg_count, src_count):
        raise ValueError("p_ba shape does not match replay grids")
    if tuple(coherent.shape) != (src_count, trg_count):
        raise ValueError("expert_coherent_mutual shape does not match replay grids")

    early = torch.sqrt((p_ab.float() * p_ba.float().t()).clamp_min(0.0))
    early = torch.nan_to_num(early, nan=0.0, posinf=0.0, neginf=0.0)
    coherent = torch.nan_to_num(
        coherent.float(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).clamp(min=0.0)
    eps = 1e-12
    source_ratio = (
        coherent.sum(dim=1) / early.sum(dim=1).clamp_min(eps)
    ).clamp(0.0, 1.0)
    target_ratio = (
        coherent.sum(dim=0) / early.sum(dim=0).clamp_min(eps)
    ).clamp(0.0, 1.0)

    def relative_gate(ratio: torch.Tensor) -> torch.Tensor:
        mean = ratio.mean()
        if float(mean.detach().cpu()) <= eps:
            return torch.ones_like(ratio)
        return (2.0 * ratio / (ratio + mean).clamp_min(eps)).clamp(0.0, 2.0)

    source_gate = relative_gate(source_ratio)
    target_gate = relative_gate(target_ratio)
    diagnostics = {
        "source_mean_coherence_ratio": float(source_ratio.mean().detach().cpu()),
        "target_mean_coherence_ratio": float(target_ratio.mean().detach().cpu()),
        "source_min_relative_gate": float(source_gate.min().detach().cpu()),
        "source_max_relative_gate": float(source_gate.max().detach().cpu()),
        "target_min_relative_gate": float(target_gate.min().detach().cpu()),
        "target_max_relative_gate": float(target_gate.max().detach().cpu()),
        "gate_formula": "2r/(r+spatial_mean_r)",
        "gt_used": False,
    }
    return (
        source_gate.reshape(
            1,
            1,
            int(src_state.image_height),
            int(src_state.image_width),
        ),
        target_gate.reshape(
            1,
            1,
            int(trg_state.image_height),
            int(trg_state.image_width),
        ),
        diagnostics,
    )


def _native_plus_gated_kernel_feature_maps(
    src_native_map: torch.Tensor,
    trg_native_map: torch.Tensor,
    src_kernel_map: torch.Tensor,
    trg_kernel_map: torch.Tensor,
    src_gate: torch.Tensor,
    trg_gate: torch.Tensor,
    *,
    weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight = float(max(0.0, weight))
    src_native = F.normalize(
        torch.nan_to_num(src_native_map.float(), nan=0.0, posinf=0.0, neginf=0.0),
        dim=1,
        eps=1e-12,
    )
    trg_native = F.normalize(
        torch.nan_to_num(trg_native_map.float(), nan=0.0, posinf=0.0, neginf=0.0),
        dim=1,
        eps=1e-12,
    )
    src_kernel = F.normalize(
        torch.nan_to_num(src_kernel_map.float(), nan=0.0, posinf=0.0, neginf=0.0),
        dim=1,
        eps=1e-12,
    )
    trg_kernel = F.normalize(
        torch.nan_to_num(trg_kernel_map.float(), nan=0.0, posinf=0.0, neginf=0.0),
        dim=1,
        eps=1e-12,
    )
    src = F.normalize(
        torch.cat((src_native, weight * src_gate.to(src_kernel) * src_kernel), dim=1),
        dim=1,
        eps=1e-12,
    )
    trg = F.normalize(
        torch.cat((trg_native, weight * trg_gate.to(trg_kernel) * trg_kernel), dim=1),
        dim=1,
        eps=1e-12,
    )
    return src.contiguous(), trg.contiguous()


def _early_mutual_kernel(
    attention: dict[str, torch.Tensor],
    src_state: FluxReplayState,
    trg_state: FluxReplayState,
) -> torch.Tensor:
    """Return the reciprocal early-average cross-attention kernel."""

    src_count = int(src_state.image_height) * int(src_state.image_width)
    trg_count = int(trg_state.image_height) * int(trg_state.image_width)
    p_ab = attention.get("p_ab")
    p_ba = attention.get("p_ba")
    if not isinstance(p_ab, torch.Tensor) or not isinstance(p_ba, torch.Tensor):
        raise ValueError("attention must contain tensor p_ab and p_ba fields")
    if tuple(p_ab.shape) != (src_count, trg_count):
        raise ValueError("p_ab shape does not match replay grids")
    if tuple(p_ba.shape) != (trg_count, src_count):
        raise ValueError("p_ba shape does not match replay grids")
    return torch.nan_to_num(
        torch.sqrt((p_ab.float() * p_ba.float().t()).clamp_min(0.0)),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def _align_flip_mutual_kernel(
    mutual: torch.Tensor,
    src_state: FluxReplayState,
    trg_state: FluxReplayState,
    *,
    source_flipped: bool,
    target_flipped: bool,
) -> torch.Tensor:
    """Inverse-map a mutual kernel from flipped views to original token order."""

    src_height = int(src_state.image_height)
    src_width = int(src_state.image_width)
    trg_height = int(trg_state.image_height)
    trg_width = int(trg_state.image_width)
    if tuple(mutual.shape) != (
        src_height * src_width,
        trg_height * trg_width,
    ):
        raise ValueError("mutual kernel shape does not match replay grids")
    grid = mutual.reshape(src_height, src_width, trg_height, trg_width)
    flip_dims = []
    if bool(source_flipped):
        flip_dims.append(1)
    if bool(target_flipped):
        flip_dims.append(3)
    if flip_dims:
        grid = grid.flip(tuple(flip_dims))
    return grid.reshape(mutual.shape).contiguous()


def _filtered_spectral_kernel_maps_from_mutual(
    mutual: torch.Tensor,
    src_state: FluxReplayState,
    trg_state: FluxReplayState,
    *,
    rank: int = 64,
    radius: int = 2,
    mutual_aggregation: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Apply the locked local-support and paired-SVD construction."""

    if int(rank) < 1:
        raise ValueError("rank must be positive")
    if int(radius) < 1:
        raise ValueError("radius must be positive")
    src_count = int(src_state.image_height) * int(src_state.image_width)
    trg_count = int(trg_state.image_height) * int(trg_state.image_width)
    if tuple(mutual.shape) != (src_count, trg_count):
        raise ValueError("mutual kernel shape does not match replay grids")
    mutual = torch.nan_to_num(
        mutual.float(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    support = _local_transport_support_rows(
        mutual,
        torch.arange(src_count, device=mutual.device, dtype=torch.long),
        src_state,
        trg_state,
        radius=int(radius),
    )
    filtered_kernel = mutual * support
    src_kernel, trg_kernel = _kernel_svd_feature_maps(
        filtered_kernel,
        src_state,
        trg_state,
        rank=int(rank),
    )
    diagnostics = {
        "rank": int(rank),
        "effective_rank": int(src_kernel.shape[1]),
        "radius": int(radius),
        "mutual_aggregation": mutual_aggregation,
        "mean_mutual_attention": float(mutual.mean().detach().cpu()),
        "mean_local_support": float(support.mean().detach().cpu()),
        "mean_filtered_kernel": float(filtered_kernel.mean().detach().cpu()),
        "gt_used": False,
    }
    return src_kernel, trg_kernel, diagnostics


def _filtered_spectral_kernel_maps(
    attention: dict[str, torch.Tensor],
    src_state: FluxReplayState,
    trg_state: FluxReplayState,
    *,
    rank: int = 64,
    radius: int = 2,
    mutual_aggregation: str = "early_average",
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    src_count = int(src_state.image_height) * int(src_state.image_width)
    trg_count = int(trg_state.image_height) * int(trg_state.image_width)
    if mutual_aggregation == "early_average":
        mutual = _early_mutual_kernel(attention, src_state, trg_state)
    elif mutual_aggregation in {"head_coherent", "expert_coherent"}:
        key = f"{mutual_aggregation}_mutual"
        mutual = attention.get(key)
        if not isinstance(mutual, torch.Tensor):
            raise ValueError(f"attention must contain tensor {key} field")
        if tuple(mutual.shape) != (src_count, trg_count):
            raise ValueError(f"{key} shape does not match replay grids")
        mutual = mutual.float()
    else:
        raise ValueError(
            "mutual_aggregation must be early_average, head_coherent, or "
            "expert_coherent"
        )
    return _filtered_spectral_kernel_maps_from_mutual(
        mutual,
        src_state,
        trg_state,
        rank=int(rank),
        radius=int(radius),
        mutual_aggregation=mutual_aggregation,
    )


def _kernel_lowrank_svd_feature_maps(
    kernel: torch.Tensor,
    src_state: FluxReplayState,
    trg_state: FluxReplayState,
    *,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic low-rank paired coordinates for one head kernel."""

    src_count = int(src_state.image_height) * int(src_state.image_width)
    trg_count = int(trg_state.image_height) * int(trg_state.image_width)
    kernel = torch.nan_to_num(
        kernel.float(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if tuple(kernel.shape) != (src_count, trg_count):
        raise ValueError("kernel shape does not match replay grids")
    centered = kernel - kernel.mean(dim=1, keepdim=True)
    centered = centered - centered.mean(dim=0, keepdim=True)
    centered = centered + kernel.mean()
    max_rank = min(max(1, int(rank)), min(src_count, trg_count))
    if centered.square().mean() <= 1e-12:
        source = torch.zeros(
            (src_count, max_rank),
            device=kernel.device,
            dtype=torch.float32,
        )
        target = torch.zeros(
            (trg_count, max_rank),
            device=kernel.device,
            dtype=torch.float32,
        )
    else:
        sketch_rank = min(max_rank + 4, min(src_count, trg_count))
        cuda_devices = []
        if kernel.is_cuda:
            cuda_devices = [
                kernel.device.index
                if kernel.device.index is not None
                else torch.cuda.current_device()
            ]
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(2027)
            if kernel.is_cuda:
                torch.cuda.manual_seed(2027)
            try:
                u, s, v = torch.svd_lowrank(
                    centered,
                    q=sketch_rank,
                    niter=2,
                )
            except RuntimeError:
                u, s, v = torch.svd_lowrank(
                    centered.cpu(),
                    q=sketch_rank,
                    niter=2,
                )
                u = u.to(kernel.device)
                s = s.to(kernel.device)
                v = v.to(kernel.device)
        s = torch.nan_to_num(
            s[:max_rank].float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(0.0)
        scale = torch.sqrt(s).reshape(1, -1)
        source = u[:, :max_rank].float() * scale
        target = v[:, :max_rank].float() * scale
    return (
        source.t().reshape(
            1,
            max_rank,
            int(src_state.image_height),
            int(src_state.image_width),
        ).contiguous(),
        target.t().reshape(
            1,
            max_rank,
            int(trg_state.image_height),
            int(trg_state.image_width),
        ).contiguous(),
    )


def _head_preserving_spectral_kernel_maps(
    attention: dict[str, torch.Tensor],
    src_state: FluxReplayState,
    trg_state: FluxReplayState,
    *,
    rank: int = 64,
    radius: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Factorize each head mutual kernel before concatenating coordinates."""

    if int(rank) < 1:
        raise ValueError("rank must be positive")
    if int(radius) < 1:
        raise ValueError("radius must be positive")
    head_stack = attention.get("head_coherent_mutual_stack")
    if not isinstance(head_stack, torch.Tensor) or head_stack.ndim != 3:
        raise ValueError(
            "head-preserving spectral coordinates require "
            "head_coherent_mutual_stack [head, source, target]"
        )
    src_count = int(src_state.image_height) * int(src_state.image_width)
    trg_count = int(trg_state.image_height) * int(trg_state.image_width)
    if tuple(head_stack.shape[1:]) != (src_count, trg_count):
        raise ValueError("head mutual stack shape does not match replay grids")
    head_count = int(head_stack.shape[0])
    if head_count < 1:
        raise ValueError("head mutual stack cannot be empty")
    rank_per_head = max(1, int(math.ceil(float(rank) / float(head_count))))
    source_heads = []
    target_heads = []
    support_means = []
    for head in range(head_count):
        mutual = torch.nan_to_num(
            head_stack[head].float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        support = _local_transport_support_rows(
            mutual,
            torch.arange(src_count, device=mutual.device, dtype=torch.long),
            src_state,
            trg_state,
            radius=int(radius),
        )
        source_head, target_head = _kernel_lowrank_svd_feature_maps(
            mutual * support,
            src_state,
            trg_state,
            rank=rank_per_head,
        )
        source_heads.append(F.normalize(source_head, dim=1, eps=1e-12))
        target_heads.append(F.normalize(target_head, dim=1, eps=1e-12))
        support_means.append(float(support.mean().detach().cpu()))
    source = torch.cat(source_heads, dim=1).contiguous()
    target = torch.cat(target_heads, dim=1).contiguous()
    diagnostics = {
        "rank_budget": int(rank),
        "rank_per_head": int(rank_per_head),
        "effective_rank": int(source.shape[1]),
        "head_count": int(head_count),
        "radius": int(radius),
        "mean_local_support": float(np.mean(support_means)),
        "head_identity_preserved": True,
        "head_fusion": "equal_energy_concatenation",
        "factorization": "fixed_seed_svd_lowrank_oversample4_niter2",
        "gt_used": False,
    }
    return source, target, diagnostics


def filtered_spectral_kernel_feature_maps(
    src_native_map: torch.Tensor,
    trg_native_map: torch.Tensor,
    attention: dict[str, torch.Tensor],
    src_state: FluxReplayState,
    trg_state: FluxReplayState,
    *,
    rank: int = 64,
    radius: int = 2,
    weight: float = 0.5,
    mutual_aggregation: str = "early_average",
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Turn exact mutual cross-attention into paired spectral descriptors.

    This is the deployable form of the former kernel-featureization audit.
    It intentionally reuses the audit's local support, SVD factorization, and
    native-feature composition without consulting keypoints or annotations.
    """

    if float(weight) < 0.0:
        raise ValueError("weight must be non-negative")
    if tuple(src_native_map.shape[-2:]) != (
        int(src_state.image_height),
        int(src_state.image_width),
    ):
        raise ValueError("source native map does not match replay grid")
    if tuple(trg_native_map.shape[-2:]) != (
        int(trg_state.image_height),
        int(trg_state.image_width),
    ):
        raise ValueError("target native map does not match replay grid")
    src_kernel, trg_kernel, diagnostics = _filtered_spectral_kernel_maps(
        attention,
        src_state,
        trg_state,
        rank=int(rank),
        radius=int(radius),
        mutual_aggregation=mutual_aggregation,
    )
    src_fused, trg_fused = _native_plus_kernel_feature_maps(
        src_native_map,
        trg_native_map,
        src_kernel,
        trg_kernel,
        weight=float(weight),
    )
    diagnostics["weight"] = float(weight)
    return src_fused, trg_fused, diagnostics


def flux_fjsar_filtered_spectral_feature_maps(
    src_native_map: torch.Tensor,
    trg_native_map: torch.Tensor,
    *,
    src_replay_state: dict[str, Any] | FluxReplayState,
    trg_replay_state: dict[str, Any] | FluxReplayState,
    blocks: Sequence[Any],
    rank: int = 64,
    radius: int = 2,
    weight: float = 0.5,
    include_native: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Replay exact cross-attention and return the locked spectral maps."""

    if not blocks:
        raise ValueError("spectral replay requires at least one FLUX block")
    src_state = (
        FluxReplayState.from_dict(src_replay_state)
        if isinstance(src_replay_state, dict)
        else src_replay_state
    )
    trg_state = (
        FluxReplayState.from_dict(trg_replay_state)
        if isinstance(trg_replay_state, dict)
        else trg_replay_state
    )
    if src_state.global_block_index != trg_state.global_block_index:
        raise ValueError("source and target replay states start at different blocks")
    if src_state.ensemble_size != trg_state.ensemble_size:
        raise ValueError("source and target replay ensemble sizes differ")
    with torch.no_grad():
        _src_joint, _trg_joint, attention = run_flux_joint_stack(
            blocks,
            src_state,
            trg_state,
            mode="exact",
            use_coordinate_bias=False,
        )
        del _src_joint, _trg_joint
        if include_native:
            src_fused, trg_fused, diagnostics = (
                filtered_spectral_kernel_feature_maps(
                    src_native_map,
                    trg_native_map,
                    attention,
                    src_state,
                    trg_state,
                    rank=int(rank),
                    radius=int(radius),
                    weight=float(weight),
                )
            )
        else:
            src_fused, trg_fused, diagnostics = _filtered_spectral_kernel_maps(
                attention,
                src_state,
                trg_state,
                rank=int(rank),
                radius=int(radius),
            )
            diagnostics["weight"] = float(weight)
    diagnostics.update(
        {
            "interaction_mode": "exact",
            "coordinate_bias": False,
            "ensemble_size": int(src_state.ensemble_size),
            "includes_native": bool(include_native),
        }
    )
    return src_fused, trg_fused, diagnostics


def flux_fjsar_filtered_spectral_feature_map_variants(
    src_native_map: torch.Tensor,
    trg_native_map: torch.Tensor,
    *,
    src_replay_state: dict[str, Any] | FluxReplayState,
    trg_replay_state: dict[str, Any] | FluxReplayState,
    src_hflip_replay_state: dict[str, Any] | FluxReplayState | None = None,
    trg_hflip_replay_state: dict[str, Any] | FluxReplayState | None = None,
    blocks: Sequence[Any],
    rank: int = 64,
    radius: int = 2,
    weight: float = 0.5,
    include_native: bool = True,
) -> dict[str, tuple[torch.Tensor, torch.Tensor, dict[str, Any]]]:
    """Return spectral variants, optionally including flip-orbit consensus."""

    if not blocks:
        raise ValueError("spectral replay requires at least one FLUX block")
    src_state = (
        FluxReplayState.from_dict(src_replay_state)
        if isinstance(src_replay_state, dict)
        else src_replay_state
    )
    trg_state = (
        FluxReplayState.from_dict(trg_replay_state)
        if isinstance(trg_replay_state, dict)
        else trg_replay_state
    )
    if src_state.global_block_index != trg_state.global_block_index:
        raise ValueError("source and target replay states start at different blocks")
    if src_state.ensemble_size != trg_state.ensemble_size:
        raise ValueError("source and target replay ensemble sizes differ")
    if (src_hflip_replay_state is None) != (trg_hflip_replay_state is None):
        raise ValueError("source and target hflip replay states must be paired")
    src_hflip_state = None
    trg_hflip_state = None
    if src_hflip_replay_state is not None and trg_hflip_replay_state is not None:
        src_hflip_state = (
            FluxReplayState.from_dict(src_hflip_replay_state)
            if isinstance(src_hflip_replay_state, dict)
            else src_hflip_replay_state
        )
        trg_hflip_state = (
            FluxReplayState.from_dict(trg_hflip_replay_state)
            if isinstance(trg_hflip_replay_state, dict)
            else trg_hflip_replay_state
        )
        for name, original, flipped in (
            ("source", src_state, src_hflip_state),
            ("target", trg_state, trg_hflip_state),
        ):
            original_protocol = (
                int(original.image_height),
                int(original.image_width),
                int(original.global_block_index),
                int(original.ensemble_size),
            )
            flipped_protocol = (
                int(flipped.image_height),
                int(flipped.image_width),
                int(flipped.global_block_index),
                int(flipped.ensemble_size),
            )
            if original_protocol != flipped_protocol:
                raise ValueError(
                    f"{name} hflip replay protocol does not match original"
                )
    variants: dict[
        str,
        tuple[torch.Tensor, torch.Tensor, dict[str, Any]],
    ] = {}
    with torch.no_grad():
        _src_joint, _trg_joint, attention = run_flux_joint_stack(
            blocks,
            src_state,
            trg_state,
            mode="exact",
            use_coordinate_bias=False,
            preserve_coherent_mutual=True,
        )
        del _src_joint, _trg_joint
        early_kernel_pair = None
        early_diagnostics = None
        for mutual_aggregation in (
            "early_average",
            "head_coherent",
            "expert_coherent",
        ):
            src_kernel, trg_kernel, diagnostics = _filtered_spectral_kernel_maps(
                attention,
                src_state,
                trg_state,
                rank=int(rank),
                radius=int(radius),
                mutual_aggregation=mutual_aggregation,
            )
            if mutual_aggregation == "early_average":
                early_kernel_pair = (src_kernel, trg_kernel)
                early_diagnostics = dict(diagnostics)
            if include_native:
                src_fused, trg_fused = _native_plus_kernel_feature_maps(
                    src_native_map,
                    trg_native_map,
                    src_kernel,
                    trg_kernel,
                    weight=float(weight),
                )
            else:
                src_fused, trg_fused = src_kernel, trg_kernel
            diagnostics["weight"] = float(weight)
            diagnostics.update(
                {
                    "interaction_mode": "exact",
                    "coordinate_bias": False,
                    "ensemble_size": int(src_state.ensemble_size),
                    "includes_native": bool(include_native),
                }
            )
            variants[mutual_aggregation] = (
                src_fused,
                trg_fused,
                diagnostics,
            )
        if early_kernel_pair is None or early_diagnostics is None:
            raise RuntimeError("early-average spectral kernel was not produced")
        source_gate, target_gate, gate_diagnostics = (
            _expert_coherence_relative_gates(
                attention,
                src_state,
                trg_state,
            )
        )
        if include_native:
            gated_source, gated_target = _native_plus_gated_kernel_feature_maps(
                src_native_map,
                trg_native_map,
                early_kernel_pair[0],
                early_kernel_pair[1],
                source_gate,
                target_gate,
                weight=float(weight),
            )
        else:
            gated_source = early_kernel_pair[0] * source_gate.to(early_kernel_pair[0])
            gated_target = early_kernel_pair[1] * target_gate.to(early_kernel_pair[1])
        gated_diagnostics = {
            **early_diagnostics,
            **gate_diagnostics,
            "weight": float(weight),
            "mutual_aggregation": "early_average",
            "spectral_injection": "expert_coherence_relative_gate",
            "interaction_mode": "exact",
            "coordinate_bias": False,
            "ensemble_size": int(src_state.ensemble_size),
            "includes_native": bool(include_native),
        }
        variants["expert_coherence_gated"] = (
            gated_source,
            gated_target,
            gated_diagnostics,
        )
        head_source, head_target, head_diagnostics = (
            _head_preserving_spectral_kernel_maps(
                attention,
                src_state,
                trg_state,
                rank=int(rank),
                radius=int(radius),
            )
        )
        if include_native:
            head_source, head_target = _native_plus_kernel_feature_maps(
                src_native_map,
                trg_native_map,
                head_source,
                head_target,
                weight=float(weight),
            )
        head_diagnostics.update(
            {
                "weight": float(weight),
                "spectral_injection": "head_preserving_paired_coordinates",
                "interaction_mode": "exact",
                "coordinate_bias": False,
                "ensemble_size": int(src_state.ensemble_size),
                "includes_native": bool(include_native),
            }
        )
        variants["head_preserving"] = (
            head_source,
            head_target,
            head_diagnostics,
        )
        if src_hflip_state is not None and trg_hflip_state is not None:
            orbit_mutual = _early_mutual_kernel(
                attention,
                src_state,
                trg_state,
            )
            del attention
            for (
                view_source,
                view_target,
                source_flipped,
                target_flipped,
            ) in (
                (src_hflip_state, trg_state, True, False),
                (src_state, trg_hflip_state, False, True),
                (src_hflip_state, trg_hflip_state, True, True),
            ):
                view_source_joint, view_target_joint, view_attention = (
                    run_flux_joint_stack(
                        blocks,
                        view_source,
                        view_target,
                        mode="exact",
                        use_coordinate_bias=False,
                        preserve_coherent_mutual=False,
                    )
                )
                view_mutual = _early_mutual_kernel(
                    view_attention,
                    view_source,
                    view_target,
                )
                orbit_mutual = orbit_mutual + _align_flip_mutual_kernel(
                    view_mutual,
                    src_state,
                    trg_state,
                    source_flipped=source_flipped,
                    target_flipped=target_flipped,
                )
                del (
                    view_source_joint,
                    view_target_joint,
                    view_attention,
                    view_mutual,
                )
            orbit_mutual = orbit_mutual / 4.0
            orbit_source, orbit_target, orbit_diagnostics = (
                _filtered_spectral_kernel_maps_from_mutual(
                    orbit_mutual,
                    src_state,
                    trg_state,
                    rank=int(rank),
                    radius=int(radius),
                    mutual_aggregation="flip_orbit_early_average",
                )
            )
            del orbit_mutual
            if include_native:
                orbit_source, orbit_target = _native_plus_kernel_feature_maps(
                    src_native_map,
                    trg_native_map,
                    orbit_source,
                    orbit_target,
                    weight=float(weight),
                )
            orbit_diagnostics.update(
                {
                    "weight": float(weight),
                    "spectral_injection": "geometric_orbit_consensus",
                    "interaction_mode": "exact",
                    "coordinate_bias": False,
                    "ensemble_size": int(src_state.ensemble_size),
                    "includes_native": bool(include_native),
                    "orbit_views": 4,
                    "orbit_transforms": (
                        "original_original",
                        "hflip_original",
                        "original_hflip",
                        "hflip_hflip",
                    ),
                    "orbit_alignment": "inverse_horizontal_flip",
                    "orbit_fusion": "equal_mean_mutual_kernel",
                }
            )
            variants["flip_orbit"] = (
                orbit_source,
                orbit_target,
                orbit_diagnostics,
            )
    return variants


def _kernel_featureization_audit_for_points(
    src_features: torch.Tensor,
    trg_features: torch.Tensor,
    mutual_attention: torch.Tensor,
    src_cells: torch.Tensor,
    source_points: Sequence[Sequence[float]],
    source_size: Sequence[int],
    target_size: Sequence[int],
    src_state: FluxReplayState,
    trg_state: FluxReplayState,
    *,
    target_points: Sequence[Sequence[float]] | None,
    pck_threshold: float | None,
    topks: Sequence[int],
    ranks: Sequence[int],
    weights: Sequence[float],
    radius: int,
) -> list[dict[str, Any]]:
    topks = tuple(sorted({max(1, int(k)) for k in topks}))
    ranks = tuple(sorted({max(1, int(rank)) for rank in ranks}))
    weights = tuple(dict.fromkeys(float(weight) for weight in weights))
    max_k = min(max(topks) if topks else 1, int(target_size[0]) * int(target_size[1]))
    src_cells = src_cells.to(device=mutual_attention.device, dtype=torch.long).flatten()
    full_src_cells = torch.arange(mutual_attention.shape[0], device=mutual_attention.device, dtype=torch.long)
    support = _local_transport_support_rows(
        mutual_attention,
        full_src_cells,
        src_state,
        trg_state,
        radius=radius,
    )
    filtered_attention = mutual_attention.float() * support
    signal_indices: dict[str, torch.Tensor] = {}

    signal_indices["native_descriptor"] = _chunked_descriptor_topk_indices(
        src_features,
        trg_features,
        source_points,
        source_size,
        target_size,
        topk=max_k,
    )
    raw_cells = torch.topk(mutual_attention[src_cells].float(), k=min(max_k, mutual_attention.shape[1]), dim=1, sorted=True).indices
    filtered_cells = torch.topk(filtered_attention[src_cells].float(), k=min(max_k, filtered_attention.shape[1]), dim=1, sorted=True).indices
    signal_indices["raw_attention_kernel"] = _cell_topk_to_pixel_indices(raw_cells, target_size, trg_state).to(src_features.device)
    signal_indices["filtered_attention_kernel"] = _cell_topk_to_pixel_indices(filtered_cells, target_size, trg_state).to(src_features.device)

    positive_src, positive_trg = _kernel_positive_feature_maps(mutual_attention, src_state, trg_state)
    filtered_positive_src, filtered_positive_trg = _kernel_positive_feature_maps(filtered_attention, src_state, trg_state)
    signal_indices["positive_kernel_feature"] = _chunked_descriptor_topk_indices(
        positive_src,
        positive_trg,
        source_points,
        source_size,
        target_size,
        topk=max_k,
    )
    signal_indices["filtered_positive_kernel_feature"] = _chunked_descriptor_topk_indices(
        filtered_positive_src,
        filtered_positive_trg,
        source_points,
        source_size,
        target_size,
        topk=max_k,
    )

    max_rank = min(max(ranks) if ranks else 32, min(mutual_attention.shape))
    svd_src, svd_trg = _kernel_svd_feature_maps(mutual_attention, src_state, trg_state, rank=max_rank)
    filtered_svd_src, filtered_svd_trg = _kernel_svd_feature_maps(filtered_attention, src_state, trg_state, rank=max_rank)
    for rank in ranks:
        rank = min(int(rank), int(svd_src.shape[1]))
        if rank <= 0:
            continue
        src_rank = svd_src[:, :rank].contiguous()
        trg_rank = svd_trg[:, :rank].contiguous()
        filtered_src_rank = filtered_svd_src[:, :rank].contiguous()
        filtered_trg_rank = filtered_svd_trg[:, :rank].contiguous()
        signal_indices[f"svd_kernel_rank{rank}"] = _chunked_descriptor_topk_indices(
            src_rank,
            trg_rank,
            source_points,
            source_size,
            target_size,
            topk=max_k,
        )
        signal_indices[f"filtered_svd_kernel_rank{rank}"] = _chunked_descriptor_topk_indices(
            filtered_src_rank,
            filtered_trg_rank,
            source_points,
            source_size,
            target_size,
            topk=max_k,
        )
        for weight in weights:
            src_plus, trg_plus = _native_plus_kernel_feature_maps(
                src_features,
                trg_features,
                src_rank,
                trg_rank,
                weight=weight,
            )
            signal_indices[f"native_plus_svd_rank{rank}_w{weight:g}"] = _chunked_descriptor_topk_indices(
                src_plus,
                trg_plus,
                source_points,
                source_size,
                target_size,
                topk=max_k,
            )
            src_filtered_plus, trg_filtered_plus = _native_plus_kernel_feature_maps(
                src_features,
                trg_features,
                filtered_src_rank,
                filtered_trg_rank,
                weight=weight,
            )
            signal_indices[f"native_plus_filtered_svd_rank{rank}_w{weight:g}"] = _chunked_descriptor_topk_indices(
                src_filtered_plus,
                trg_filtered_plus,
                source_points,
                source_size,
                target_size,
                topk=max_k,
            )

    signal_hits = {
        name: _per_point_topk_hits(indices.to(src_features.device), target_points, pck_threshold, target_size, topks)
        for name, indices in signal_indices.items()
    }
    rows = []
    score_names = list(signal_indices.keys())
    for row in range(len(source_points)):
        ranks_by_signal = {
            name: signal_hits[name][row]["rank"]
            for name in score_names
        }
        topk_hits = {
            f"{name}@{int(k)}": bool(signal_hits[name][row]["topk_hits"][int(k)])
            for name in score_names
            for k in topks
        }
        top1 = {}
        target_h, target_w = int(target_size[0]), int(target_size[1])
        for name in score_names:
            pixel = int(signal_indices[name][row, 0].detach().cpu())
            xy = [int(pixel % target_w), int(pixel // target_w)]
            target = target_points[row] if target_points is not None else None
            top1[name] = {
                "pixel": xy,
                "pixel_index": int(pixel),
                "pck_hit": (
                    bool(_point_hit(xy, target, pck_threshold))
                    if target is not None and pck_threshold is not None
                    else False
                ),
            }
        rows.append({
            "topks": [int(k) for k in topks],
            "ranks_requested": [int(rank) for rank in ranks],
            "weights_requested": [float(weight) for weight in weights],
            "radius": int(max(0, radius)),
            "score_names": score_names,
            "ranks": ranks_by_signal,
            "topk_hits": topk_hits,
            "top1": top1,
        })
    return rows


def _residual_readout_audit_for_points(
    probe: dict[str, Any],
    proposal_pixels: torch.Tensor,
    target_size: Sequence[int],
    *,
    target_points: Sequence[Sequence[float]] | None,
    pck_threshold: float | None,
    topks: Sequence[int],
) -> list[dict[str, Any]]:
    """Audit candidate identity signals from common-free attention readout."""

    score_names = list(probe.get("score_names", []))
    scores = probe.get("scores", {})
    topks = tuple(sorted({max(1, int(k)) for k in topks}))
    target_h, target_w = int(target_size[0]), int(target_size[1])
    proposal_pixels = proposal_pixels.to(dtype=torch.long)
    rows: list[dict[str, Any]] = []
    for row in range(int(proposal_pixels.shape[0])):
        target = target_points[row] if target_points is not None else None
        hits: list[bool] = []
        for pixel_tensor in proposal_pixels[row].detach().cpu().tolist():
            pixel = int(pixel_tensor)
            xy = [int(pixel % target_w), int(pixel // target_w)]
            hits.append(
                bool(_point_hit(xy, target, pck_threshold))
                if target is not None and pck_threshold is not None
                else False
            )
        ranks: dict[str, int | None] = {}
        topk_hits: dict[str, bool] = {}
        score_gaps: dict[str, float | None] = {}
        candidates = []
        for score_name in score_names:
            matrix = scores.get(score_name)
            if not isinstance(matrix, torch.Tensor):
                continue
            values = [
                float(value)
                for value in matrix[row].detach().cpu().tolist()
            ]
            ranks[score_name] = _rank_first_hit(values, hits)
            order = sorted(range(len(values)), key=lambda index: values[index], reverse=True)
            for k in topks:
                topk_hits[f"{score_name}@{int(k)}"] = bool(
                    any(hits[index] for index in order[: min(int(k), len(order))])
                )
            top1 = values[0] if values else None
            hit_values = [value for value, hit in zip(values, hits) if hit]
            best_hit = max(hit_values) if hit_values else None
            score_gaps[f"{score_name}_attention_top1_minus_best_pck_hit_proposal"] = (
                float(top1 - best_hit)
                if top1 is not None and best_hit is not None
                else None
            )
        for rank, pixel_tensor in enumerate(proposal_pixels[row].detach().cpu().tolist(), start=1):
            pixel = int(pixel_tensor)
            xy = [int(pixel % target_w), int(pixel // target_w)]
            item_scores = {}
            for score_name in score_names:
                matrix = scores.get(score_name)
                if isinstance(matrix, torch.Tensor):
                    item_scores[score_name] = float(matrix[row, rank - 1].detach().cpu())
            candidates.append({
                "rank_attention": int(rank),
                "pixel": xy,
                "pixel_index": int(pixel),
                "pck_hit": bool(hits[rank - 1]),
                "scores": item_scores,
            })
        rows.append({
            "topks": [int(k) for k in topks],
            "score_names": score_names,
            "metadata": dict(probe.get("metadata", {})),
            "ranks": ranks,
            "topk_hits": topk_hits,
            "score_gaps": score_gaps,
            "candidates": candidates,
        })
    return rows


def _candidate_clamped_causal_replay_audit_for_points(
    probe: dict[str, Any],
    proposal_pixels: torch.Tensor,
    target_size: Sequence[int],
    *,
    target_points: Sequence[Sequence[float]] | None,
    pck_threshold: float | None,
    topks: Sequence[int],
) -> list[dict[str, Any]]:
    """Format causal-clamp scores without exposing GT to candidate scoring."""

    rows = _residual_readout_audit_for_points(
        probe,
        proposal_pixels,
        target_size,
        target_points=target_points,
        pck_threshold=pck_threshold,
        topks=topks,
    )
    diagnostic_matrices = probe.get("diagnostics", {})
    metadata = dict(probe.get("metadata", {}))
    primary_signal = "post_release_bidirectional_negative_log_rank"
    for row_index, row in enumerate(rows):
        candidates = row.get("candidates", [])
        for candidate_index, candidate in enumerate(candidates):
            candidate_diagnostics = {}
            for name, matrix in diagnostic_matrices.items():
                if isinstance(matrix, torch.Tensor):
                    candidate_diagnostics[name] = float(
                        matrix[row_index, candidate_index].detach().cpu()
                    )
            candidate["diagnostics"] = candidate_diagnostics
        selected_attention_ranks = {}
        for score_name in row.get("score_names", []):
            values = [
                float(candidate.get("scores", {}).get(score_name, -float("inf")))
                for candidate in candidates
            ]
            selected_attention_ranks[score_name] = (
                int(max(range(len(values)), key=lambda index: values[index])) + 1
                if values
                else None
            )
        hit_fraction = (
            float(sum(int(bool(candidate.get("pck_hit"))) for candidate in candidates))
            / max(1, len(candidates))
        )
        row["diagnostics"] = {
            "primary_signal": primary_signal,
            "selected_attention_ranks": selected_attention_ranks,
            "pck_hit_candidate_fraction": hit_fraction,
            "source_cross_mass_mean": (
                float(sum(
                    candidate.get("diagnostics", {}).get("source_cross_mass", 0.0)
                    for candidate in candidates
                ) / max(1, len(candidates)))
            ),
            "target_cross_mass_mean": (
                float(sum(
                    candidate.get("diagnostics", {}).get("target_cross_mass", 0.0)
                    for candidate in candidates
                ) / max(1, len(candidates)))
            ),
            "source_intervention_relative_l2_mean": (
                float(sum(
                    candidate.get("diagnostics", {}).get(
                        "source_intervention_relative_l2", 0.0
                    )
                    for candidate in candidates
                ) / max(1, len(candidates)))
            ),
            "target_intervention_relative_l2_mean": (
                float(sum(
                    candidate.get("diagnostics", {}).get(
                        "target_intervention_relative_l2", 0.0
                    )
                    for candidate in candidates
                ) / max(1, len(candidates)))
            ),
            "causal_rank_improvement_mean": (
                float(sum(
                    candidate.get("scores", {}).get(
                        "causal_rank_improvement", 0.0
                    )
                    for candidate in candidates
                ) / max(1, len(candidates)))
            ),
            "causal_improvement_positive_fraction_mean": (
                float(sum(
                    candidate.get("diagnostics", {}).get(
                        "causal_improvement_positive_fraction", 0.0
                    )
                    for candidate in candidates
                ) / max(1, len(candidates)))
            ),
            "post_release_score_std_mean": (
                float(sum(
                    candidate.get("diagnostics", {}).get(
                        "post_release_score_std", 0.0
                    )
                    for candidate in candidates
                ) / max(1, len(candidates)))
            ),
            "native_candidate_injected": False,
            "native_fallback_used": False,
            "gt_used_for_scoring": False,
        }
        row["causal_contract"] = {
            **metadata,
            "primary_signal": primary_signal,
            "candidate_value_conditioning": True,
            "original_total_cross_mass_preserved": True,
            "local_attention_contribution_preserved": True,
            "release_qk_is_unclamped": True,
            "prediction_changed": False,
        }
    return rows


def _counterfactual_fingerprint_audit_for_points(
    probe: dict[str, Any],
    proposal_pixels: torch.Tensor,
    target_size: Sequence[int],
    *,
    target_points: Sequence[Sequence[float]] | None,
    pck_threshold: float | None,
    topks: Sequence[int],
) -> list[dict[str, Any]]:
    """Format multi-dose causal response curves as candidate rankings."""

    scores = {
        "fingerprint_score": probe["fingerprint_score"],
        "fingerprint_mean_bidirectional": probe["fingerprint_mean_bidirectional"],
        "fingerprint_reciprocity_error": -probe["fingerprint_reciprocity_error"],
        "fingerprint_response_magnitude": probe["fingerprint_response_magnitude"],
    }
    rows = _residual_readout_audit_for_points(
        {
            "score_names": list(scores.keys()),
            "scores": scores,
            "metadata": probe.get("metadata", {}),
        },
        proposal_pixels,
        target_size,
        target_points=target_points,
        pck_threshold=pck_threshold,
        topks=topks,
    )
    scales = [float(scale) for scale in probe.get("intervention_scales", ())]
    source_by_scale = probe.get("fingerprint_source_score_by_scale")
    target_by_scale = probe.get("fingerprint_target_score_by_scale")
    for row_index, row in enumerate(rows):
        for candidate_index, candidate in enumerate(row.get("candidates", [])):
            if isinstance(source_by_scale, torch.Tensor) and isinstance(target_by_scale, torch.Tensor):
                candidate["response_curve"] = {
                    "scales": scales,
                    "source": [float(value) for value in source_by_scale[:, row_index, candidate_index].detach().cpu()],
                    "target": [float(value) for value in target_by_scale[:, row_index, candidate_index].detach().cpu()],
                }
        row["fingerprint_contract"] = {
            **dict(probe.get("metadata", {})),
            "primary_signal": "fingerprint_score",
            "candidate_value_conditioning": True,
            "original_total_cross_mass_preserved": True,
            "local_attention_contribution_preserved": True,
            "prediction_changed": False,
            "gt_used_for_scoring": False,
        }
    return rows


def _persistent_candidate_slot_replay_audit_for_points(
    probe: dict[str, Any],
    proposal_pixels: torch.Tensor,
    target_size: Sequence[int],
    *,
    attention_scores: torch.Tensor | None = None,
    target_points: Sequence[Sequence[float]] | None,
    pck_threshold: float | None,
    topks: Sequence[int],
) -> list[dict[str, Any]]:
    """Format persistent candidate-slot states as a diagnostic-only ranking."""

    scores = probe.get("pair_cosine")
    if not isinstance(scores, torch.Tensor) or scores.ndim != 2:
        raise ValueError("persistent candidate-slot probe must return [point,candidate] pair cosine")
    if tuple(proposal_pixels.shape) != tuple(scores.shape):
        raise ValueError("persistent candidate-slot scores and proposal pixels must align")
    score_names = [
        name
        for name in (
            "directional_anchor_cosine",
            "pair_cosine",
            "intervention_gain",
        )
        if isinstance(probe.get(name), torch.Tensor)
    ]
    if not score_names:
        raise ValueError("persistent candidate-slot probe returned no score matrices")
    if attention_scores is not None and tuple(attention_scores.shape) != tuple(scores.shape):
        raise ValueError("persistent attention scores and candidate scores must align")
    audit_probe = {
        "score_names": score_names,
        "scores": {name: probe[name] for name in score_names},
        "diagnostics": {
            name: probe[name]
            for name in (
                "source_cross_mass",
                "target_cross_mass",
                "source_relative_delta",
                "target_relative_delta",
            )
            if isinstance(probe.get(name), torch.Tensor)
        },
        "metadata": dict(probe.get("metadata", {})),
    }
    rows = _residual_readout_audit_for_points(
        audit_probe,
        proposal_pixels,
        target_size,
        target_points=target_points,
        pck_threshold=pck_threshold,
        topks=topks,
    )
    source_divergence = probe.get("source_slot_divergence")
    target_divergence = probe.get("target_slot_divergence")
    source_similarity = probe.get("source_slot_similarity")
    target_similarity = probe.get("target_slot_similarity")
    for row_index, row in enumerate(rows):
        values = scores[row_index].detach().float().cpu().tolist()
        attention_values = (
            attention_scores[row_index].detach().float().cpu().tolist()
            if isinstance(attention_scores, torch.Tensor)
            else None
        )
        selected_attention_rank = (
            int(max(range(len(attention_values)), key=lambda index: attention_values[index])) + 1
            if attention_values else None
        )
        candidate_diagnostics = []
        diagnostic_matrices = audit_probe["diagnostics"]
        for candidate_index in range(len(values)):
            candidate_diagnostics.append({
                name: float(matrix[row_index, candidate_index].detach().cpu())
                for name, matrix in diagnostic_matrices.items()
            })
        row["diagnostics"] = {
            "selected_attention_rank": selected_attention_rank,
            "best_pck_hit_rank": row.get("ranks", {}).get("directional_anchor_cosine"),
            "slot_score_margin": float(max(values) - sorted(values)[-2]) if len(values) > 1 else 0.0,
            "source_slot_divergence": float(source_divergence[row_index].detach().cpu())
            if isinstance(source_divergence, torch.Tensor) else None,
            "target_slot_divergence": float(target_divergence[row_index].detach().cpu())
            if isinstance(target_divergence, torch.Tensor) else None,
            "source_slot_similarity": float(source_similarity[row_index].detach().cpu())
            if isinstance(source_similarity, torch.Tensor) else None,
            "target_slot_similarity": float(target_similarity[row_index].detach().cpu())
            if isinstance(target_similarity, torch.Tensor) else None,
            "candidate_missing_gt": not any(bool(item.get("pck_hit")) for item in row.get("candidates", [])),
            "native_candidate_injected": False,
            "native_fallback_used": False,
            "gt_used_for_scoring": False,
        }
        for candidate, item in zip(row.get("candidates", []), candidate_diagnostics):
            candidate["diagnostics"] = dict(item)
        # Candidate order is the exact mutual-attention order.  Store the
        # corresponding attention value without using it as a replay score.
        if attention_values is not None:
            for candidate_index, candidate in enumerate(row.get("candidates", [])):
                candidate.setdefault("diagnostics", {})["attention_score"] = float(
                    attention_values[candidate_index]
                )
        row["persistent_slot_contract"] = {
            **dict(probe.get("metadata", {})),
            "candidate_axis_persisted_across_blocks": True,
            "local_self_attention_preserved": True,
            "original_cross_mass_used": True,
            "unit_cross_attention_forced": False,
            "native_candidate_injected": False,
            "native_fallback_used": False,
            "gt_used_for_scoring": False,
            "prediction_changed": False,
        }
    return rows


def _latent_expert_audit_for_points(
    probe: dict[str, Any],
    proposal_pixels: torch.Tensor,
    target_size: Sequence[int],
    *,
    aggregated_attention_scores: torch.Tensor | None = None,
    target_points: Sequence[Sequence[float]] | None,
    pck_threshold: float | None,
    topks: Sequence[int] = (1, 3, 5, 10, 20),
) -> list[dict[str, Any]]:
    """Test whether a pair shares stable attention heads hidden by EH averaging."""

    expert_scores = probe.get("expert_scores", {})
    support_name = (
        "log_exact_mutual_cross_probability"
        if isinstance(expert_scores.get("log_exact_mutual_cross_probability"), torch.Tensor)
        else "bidirectional_negative_log_rank"
    )
    support = expert_scores.get(support_name)
    if not isinstance(support, torch.Tensor) or support.ndim != 4:
        raise ValueError(
            "latent_expert_audit requires a per-expert support tensor [ensemble, head, point, candidate]"
        )
    ensemble_count, head_count, point_count, candidate_count = map(int, support.shape)
    if tuple(proposal_pixels.shape) != (point_count, candidate_count):
        raise ValueError("latent expert support and proposal pixels must align")
    if target_points is not None and len(target_points) != point_count:
        raise ValueError("target_points must align with latent expert points")
    support = torch.nan_to_num(support.float(), nan=-100.0, posinf=0.0, neginf=-100.0)
    proposal_pixels = proposal_pixels.to(device=support.device, dtype=torch.long)
    topks = tuple(sorted({max(1, int(k)) for k in topks}))
    point_index = torch.arange(point_count, device=support.device)

    head_support = support.mean(dim=0)
    member_support = support.mean(dim=1)
    mean_support = support.mean(dim=(0, 1))
    expert_choice = support.argmax(dim=3)
    head_choice = head_support.argmax(dim=2)
    member_choice = member_support.argmax(dim=2)

    def _mean_margin(scores: torch.Tensor) -> float:
        if candidate_count <= 1:
            return 0.0
        top2 = torch.topk(scores, k=2, dim=-1).values
        return float((top2[..., 0] - top2[..., 1]).mean().detach().cpu())

    def _uniqueness(choices: torch.Tensor) -> float:
        selected = proposal_pixels[point_index, choices.long()]
        return float(torch.unique(selected).numel() / max(1, point_count))

    head_mode = torch.mode(expert_choice, dim=0).values
    head_agreement = (expert_choice == head_mode.unsqueeze(0)).float().mean(dim=(0, 2))
    head_metrics = []
    for head in range(head_count):
        head_metrics.append({
            "head": int(head),
            "ensemble_agreement": float(head_agreement[head].detach().cpu()),
            "mean_margin": _mean_margin(head_support[head]),
            "target_uniqueness": _uniqueness(head_choice[head]),
        })
    stable_head_order = sorted(
        range(head_count),
        key=lambda head: (
            head_metrics[head]["ensemble_agreement"],
            head_metrics[head]["mean_margin"],
            head_metrics[head]["target_uniqueness"],
            -head,
        ),
        reverse=True,
    )

    member_mode = torch.mode(expert_choice, dim=1).values
    member_agreement = (expert_choice == member_mode.unsqueeze(1)).float().mean(dim=(1, 2))
    member_metrics = []
    for member in range(ensemble_count):
        member_metrics.append({
            "member": int(member),
            "head_agreement": float(member_agreement[member].detach().cpu()),
            "mean_margin": _mean_margin(member_support[member]),
            "target_uniqueness": _uniqueness(member_choice[member]),
        })
    stable_member_order = sorted(
        range(ensemble_count),
        key=lambda member: (
            member_metrics[member]["head_agreement"],
            member_metrics[member]["mean_margin"],
            member_metrics[member]["target_uniqueness"],
            -member,
        ),
        reverse=True,
    )

    expert_metrics = []
    for member in range(ensemble_count):
        for head in range(head_count):
            expert_metrics.append({
                "member": int(member),
                "head": int(head),
                "mean_margin": _mean_margin(support[member, head]),
                "target_uniqueness": _uniqueness(expert_choice[member, head]),
            })
    stable_expert_order = sorted(
        range(len(expert_metrics)),
        key=lambda index: (
            expert_metrics[index]["mean_margin"],
            expert_metrics[index]["target_uniqueness"],
            -index,
        ),
        reverse=True,
    )

    signals: dict[str, torch.Tensor] = {}
    if aggregated_attention_scores is not None:
        if tuple(aggregated_attention_scores.shape) != (point_count, candidate_count):
            raise ValueError("aggregated attention scores must align with latent expert candidates")
        signals["aggregated_attention"] = torch.nan_to_num(
            aggregated_attention_scores.to(device=support.device, dtype=torch.float32)
        )
    signals.update({
        "mean_expert_support": mean_support,
        "max_head_support": head_support.max(dim=0).values,
        "max_member_support": member_support.max(dim=0).values,
    })
    stable_head_selections: dict[str, list[int]] = {}
    for requested in (1, 2, 4):
        actual = min(requested, head_count)
        selected = stable_head_order[:actual]
        name = f"stable_head_{requested}"
        stable_head_selections[name] = [int(index) for index in selected]
        signals[name] = head_support[selected].mean(dim=0)
    stable_member = int(stable_member_order[0])
    stable_expert_flat = int(stable_expert_order[0])
    stable_expert_member = stable_expert_flat // head_count
    stable_expert_head = stable_expert_flat % head_count
    signals["stable_member_1"] = member_support[stable_member]
    signals["confident_expert_1"] = support[stable_expert_member, stable_expert_head]

    target_h, target_w = int(target_size[0]), int(target_size[1])
    hits = torch.zeros((point_count, candidate_count), device=support.device, dtype=torch.bool)
    if target_points is not None and pck_threshold is not None:
        for row in range(point_count):
            for candidate in range(candidate_count):
                pixel = int(proposal_pixels[row, candidate].detach().cpu())
                hits[row, candidate] = _point_hit(
                    [int(pixel % target_w), int(pixel // target_w)],
                    target_points[row],
                    float(pck_threshold),
                )

    def _top1_hit_count(scores: torch.Tensor) -> int:
        choices = scores.argmax(dim=1)
        return int(hits[point_index, choices].sum().detach().cpu())

    oracle_selector: dict[str, Any] = {}
    if target_points is not None and pck_threshold is not None:
        stable_head_position = {head: rank for rank, head in enumerate(stable_head_order)}
        head_hit_counts = [_top1_hit_count(head_support[head]) for head in range(head_count)]
        oracle_head = max(
            range(head_count),
            key=lambda head: (head_hit_counts[head], -stable_head_position[head]),
        )
        member_hit_counts = [_top1_hit_count(member_support[member]) for member in range(ensemble_count)]
        stable_member_position = {member: rank for rank, member in enumerate(stable_member_order)}
        oracle_member = max(
            range(ensemble_count),
            key=lambda member: (member_hit_counts[member], -stable_member_position[member]),
        )
        expert_hit_counts = [
            _top1_hit_count(support[index // head_count, index % head_count])
            for index in range(ensemble_count * head_count)
        ]
        stable_expert_position = {index: rank for rank, index in enumerate(stable_expert_order)}
        oracle_expert = max(
            range(ensemble_count * head_count),
            key=lambda index: (expert_hit_counts[index], -stable_expert_position[index]),
        )
        signals["oracle_pair_head_1"] = head_support[oracle_head]
        signals["oracle_pair_member_1"] = member_support[oracle_member]
        signals["oracle_pair_expert_1"] = support[
            oracle_expert // head_count, oracle_expert % head_count
        ]

        greedy_heads: list[int] = []
        remaining = set(range(head_count))
        for size in range(1, min(4, head_count) + 1):
            selected = max(
                remaining,
                key=lambda head: (
                    _top1_hit_count(head_support[greedy_heads + [head]].mean(dim=0)),
                    -stable_head_position[head],
                ),
            )
            greedy_heads.append(int(selected))
            remaining.remove(selected)
            if size in (2, 4):
                signals[f"oracle_pair_head_{size}"] = head_support[greedy_heads].mean(dim=0)
        if head_count < 4:
            signals["oracle_pair_head_4"] = head_support[greedy_heads].mean(dim=0)
        oracle_selector = {
            "best_head": int(oracle_head),
            "best_head_top1_hits": int(head_hit_counts[oracle_head]),
            "best_member": int(oracle_member),
            "best_member_top1_hits": int(member_hit_counts[oracle_member]),
            "best_expert": {
                "member": int(oracle_expert // head_count),
                "head": int(oracle_expert % head_count),
                "top1_hits": int(expert_hit_counts[oracle_expert]),
            },
            "greedy_head_order": greedy_heads,
        }

    pair_selector = {
        "point_count": int(point_count),
        "candidate_count": int(candidate_count),
        "stable_head_order": [int(index) for index in stable_head_order],
        "stable_head_selections": stable_head_selections,
        "stable_member_order": [int(index) for index in stable_member_order],
        "confident_expert": {
            "member": int(stable_expert_member),
            "head": int(stable_expert_head),
        },
        "head_metrics": head_metrics,
        "member_metrics": member_metrics,
        "oracle_gt_only": oracle_selector,
    }
    score_names = list(signals.keys())
    rows: list[dict[str, Any]] = []
    for row in range(point_count):
        hit_values = [bool(value) for value in hits[row].detach().cpu().tolist()]
        ranks: dict[str, int | None] = {}
        topk_hits: dict[str, bool] = {}
        score_gaps: dict[str, float | None] = {}
        for name, matrix in signals.items():
            values = [float(value) for value in matrix[row].detach().cpu().tolist()]
            ranks[name] = _rank_first_hit(values, hit_values)
            order = sorted(range(candidate_count), key=lambda index: values[index], reverse=True)
            for k in topks:
                topk_hits[f"{name}@{int(k)}"] = bool(
                    any(hit_values[index] for index in order[: min(k, candidate_count)])
                )
            best_hit = max(
                (value for value, hit in zip(values, hit_values) if hit),
                default=None,
            )
            score_gaps[f"{name}_attention_top1_minus_best_pck_hit_proposal"] = (
                float(values[0] - best_hit) if best_hit is not None else None
            )

        head_top1_hits = hits[row, head_choice[:, row]]
        member_top1_hits = hits[row, member_choice[:, row]]
        expert_top1_hits = hits[row].gather(0, expert_choice[:, :, row].reshape(-1))
        hit_mask = hits[row]
        if bool(hit_mask.any()):
            correct_head_best = head_support[:, row, hit_mask].amax(dim=1)
            correct_beats_attention = correct_head_best > head_support[:, row, 0]
            correct_member_best = member_support[:, row, hit_mask].amax(dim=1)
            correct_member_beats_attention = correct_member_best > member_support[:, row, 0]
        else:
            correct_beats_attention = torch.zeros(head_count, device=support.device, dtype=torch.bool)
            correct_member_beats_attention = torch.zeros(
                ensemble_count, device=support.device, dtype=torch.bool
            )

        candidates = []
        for candidate in range(candidate_count):
            flat_support = support[:, :, row, candidate].reshape(-1)
            top_count = min(4, int(flat_support.numel()))
            top_values, top_indices = torch.topk(flat_support, k=top_count)
            candidates.append({
                "rank_attention": int(candidate + 1),
                "pixel_index": int(proposal_pixels[row, candidate].detach().cpu()),
                "pck_hit": bool(hit_values[candidate]),
                "scores": {
                    name: float(matrix[row, candidate].detach().cpu())
                    for name, matrix in signals.items()
                },
                "head_support": [
                    float(value) for value in head_support[:, row, candidate].detach().cpu().tolist()
                ],
                "member_support": [
                    float(value) for value in member_support[:, row, candidate].detach().cpu().tolist()
                ],
                "top_expert_indices": [int(value) for value in top_indices.detach().cpu().tolist()],
                "top_expert_support": [float(value) for value in top_values.detach().cpu().tolist()],
            })
        rows.append({
            "score_names": score_names,
            "topks": [int(k) for k in topks],
            "metadata": {
                **dict(probe.get("metadata", {})),
                "expert_index_mapping": "flat_index = ensemble_member * head_count + head",
                "support_signal": support_name,
            },
            "pair_selector": pair_selector,
            "ranks": ranks,
            "topk_hits": topk_hits,
            "score_gaps": score_gaps,
            "diagnostics": {
                "any_head_top1_pck_hit": bool(head_top1_hits.any()),
                "any_member_top1_pck_hit": bool(member_top1_hits.any()),
                "any_expert_top1_pck_hit": bool(expert_top1_hits.any()),
                "head_top1_pck_hit_fraction": float(head_top1_hits.float().mean().detach().cpu()),
                "member_top1_pck_hit_fraction": float(member_top1_hits.float().mean().detach().cpu()),
                "expert_top1_pck_hit_fraction": float(expert_top1_hits.float().mean().detach().cpu()),
                "correct_beats_attention_top1_head_fraction": float(
                    correct_beats_attention.float().mean().detach().cpu()
                ),
                "correct_beats_attention_top1_member_fraction": float(
                    correct_member_beats_attention.float().mean().detach().cpu()
                ),
            },
            "candidates": candidates,
        })
    return rows


def _local_cell_indices(center_cell: int, width: int, height: int, radius: int) -> list[int]:
    cx, cy = int(center_cell % width), int(center_cell // width)
    cells = []
    for dx, dy in _cell_offsets(max(0, int(radius))):
        x, y = cx + dx, cy + dy
        if _valid_cell(x, y, width, height):
            cells.append(_cell_index(x, y, width))
    return cells


def _shifted_outgoing_transport_signature(
    mutual_attention: torch.Tensor,
    src_cell: int,
    src_state: FluxReplayState,
    trg_state: FluxReplayState,
    *,
    radius: int,
) -> torch.Tensor:
    """Source-side descriptor: neighbor-supported target positions."""

    src_w, src_h = int(src_state.image_width), int(src_state.image_height)
    trg_w, trg_h = int(trg_state.image_width), int(trg_state.image_height)
    src_x, src_y = int(src_cell % src_w), int(src_cell // src_w)
    signature = torch.zeros((trg_h, trg_w), device=mutual_attention.device, dtype=torch.float32)
    counts = torch.zeros_like(signature)
    scale_x = float(trg_w) / float(max(1, src_w))
    scale_y = float(trg_h) / float(max(1, src_h))
    for dx, dy in _cell_offsets(radius):
        nsx, nsy = src_x + dx, src_y + dy
        if not _valid_cell(nsx, nsy, src_w, src_h):
            continue
        shift_x = int(round(float(dx) * scale_x))
        shift_y = int(round(float(dy) * scale_y))
        row = mutual_attention[_cell_index(nsx, nsy, src_w)].float()
        row = row / row.max().clamp_min(1e-12)
        row_2d = row.reshape(trg_h, trg_w)
        base_x0 = max(0, -shift_x)
        base_x1 = min(trg_w, trg_w - shift_x)
        base_y0 = max(0, -shift_y)
        base_y1 = min(trg_h, trg_h - shift_y)
        if base_x0 >= base_x1 or base_y0 >= base_y1:
            continue
        signature[base_y0:base_y1, base_x0:base_x1] += row_2d[
            base_y0 + shift_y:base_y1 + shift_y,
            base_x0 + shift_x:base_x1 + shift_x,
        ]
        counts[base_y0:base_y1, base_x0:base_x1] += 1.0
    signature = signature / counts.clamp_min(1.0)
    return signature.flatten()


def _shifted_incoming_transport_signature(
    mutual_attention: torch.Tensor,
    trg_cell: int,
    src_state: FluxReplayState,
    trg_state: FluxReplayState,
    *,
    radius: int,
) -> torch.Tensor:
    """Target-side descriptor: neighbor-supported source positions."""

    src_w, src_h = int(src_state.image_width), int(src_state.image_height)
    trg_w, trg_h = int(trg_state.image_width), int(trg_state.image_height)
    trg_x, trg_y = int(trg_cell % trg_w), int(trg_cell // trg_w)
    signature = torch.zeros((src_h, src_w), device=mutual_attention.device, dtype=torch.float32)
    counts = torch.zeros_like(signature)
    scale_x = float(src_w) / float(max(1, trg_w))
    scale_y = float(src_h) / float(max(1, trg_h))
    attention_t = mutual_attention.float().t().contiguous()
    for dx, dy in _cell_offsets(radius):
        ntx, nty = trg_x + dx, trg_y + dy
        if not _valid_cell(ntx, nty, trg_w, trg_h):
            continue
        shift_x = int(round(float(dx) * scale_x))
        shift_y = int(round(float(dy) * scale_y))
        column = attention_t[_cell_index(ntx, nty, trg_w)].float()
        column = column / column.max().clamp_min(1e-12)
        column_2d = column.reshape(src_h, src_w)
        base_x0 = max(0, -shift_x)
        base_x1 = min(src_w, src_w - shift_x)
        base_y0 = max(0, -shift_y)
        base_y1 = min(src_h, src_h - shift_y)
        if base_x0 >= base_x1 or base_y0 >= base_y1:
            continue
        signature[base_y0:base_y1, base_x0:base_x1] += column_2d[
            base_y0 + shift_y:base_y1 + shift_y,
            base_x0 + shift_x:base_x1 + shift_x,
        ]
        counts[base_y0:base_y1, base_x0:base_x1] += 1.0
    signature = signature / counts.clamp_min(1.0)
    return signature.flatten()


def _unit_basis_dot(signature: torch.Tensor, cells: list[int]) -> float:
    if not cells:
        return 0.0
    vector = F.normalize(signature.reshape(1, -1), dim=1, eps=1e-12)[0]
    index = torch.tensor(cells, device=signature.device, dtype=torch.long)
    value = vector[index].sum() / float(len(cells) ** 0.5)
    return float(value.detach().cpu())


def _transport_factorization_audit_for_point(
    mutual_attention: torch.Tensor,
    src_cell: int,
    proposal_pixels: torch.Tensor,
    target_size: Sequence[int],
    src_state: FluxReplayState,
    trg_state: FluxReplayState,
    *,
    target: Sequence[float] | None,
    pck_threshold: float | None,
    radius: int,
    basis_radius: int,
) -> dict[str, Any]:
    """Audit whether local transport can be represented as two-sided descriptors."""

    target_h, target_w = int(target_size[0]), int(target_size[1])
    proposal_pixels = proposal_pixels.to(device=mutual_attention.device, dtype=torch.long).flatten()
    candidate_cells = _pixel_indices_to_replay_cells(proposal_pixels, target_size, trg_state)
    source_outgoing = _shifted_outgoing_transport_signature(
        mutual_attention,
        int(src_cell),
        src_state,
        trg_state,
        radius=radius,
    )
    source_center_cells = _local_cell_indices(
        int(src_cell),
        int(src_state.image_width),
        int(src_state.image_height),
        radius=0,
    )
    source_patch_cells = _local_cell_indices(
        int(src_cell),
        int(src_state.image_width),
        int(src_state.image_height),
        radius=basis_radius,
    )
    score_names = (
        "factorized_outgoing_center",
        "factorized_incoming_center",
        "factorized_bidirectional_center",
        "factorized_outgoing_patch",
        "factorized_incoming_patch",
        "factorized_bidirectional_patch",
    )
    signal_scores: dict[str, list[float]] = {name: [] for name in score_names}
    hits: list[bool] = []
    candidates: list[dict[str, Any]] = []
    seen_pixels: set[int] = set()
    incoming_cache: dict[int, torch.Tensor] = {}
    for rank, (pixel_tensor, cell_tensor) in enumerate(zip(proposal_pixels, candidate_cells), start=1):
        pixel = int(pixel_tensor.detach().cpu())
        if pixel in seen_pixels:
            continue
        seen_pixels.add(pixel)
        trg_cell = int(cell_tensor.detach().cpu())
        xy = [int(pixel % target_w), int(pixel // target_w)]
        hit = bool(_point_hit(xy, target, pck_threshold)) if target is not None and pck_threshold is not None else False
        target_center_cells = _local_cell_indices(
            trg_cell,
            int(trg_state.image_width),
            int(trg_state.image_height),
            radius=0,
        )
        target_patch_cells = _local_cell_indices(
            trg_cell,
            int(trg_state.image_width),
            int(trg_state.image_height),
            radius=basis_radius,
        )
        if trg_cell not in incoming_cache:
            incoming_cache[trg_cell] = _shifted_incoming_transport_signature(
                mutual_attention,
                trg_cell,
                src_state,
                trg_state,
                radius=radius,
            )
        target_incoming = incoming_cache[trg_cell]
        outgoing_center = _unit_basis_dot(source_outgoing, target_center_cells)
        incoming_center = _unit_basis_dot(target_incoming, source_center_cells)
        outgoing_patch = _unit_basis_dot(source_outgoing, target_patch_cells)
        incoming_patch = _unit_basis_dot(target_incoming, source_patch_cells)
        scores = {
            "factorized_outgoing_center": outgoing_center,
            "factorized_incoming_center": incoming_center,
            "factorized_bidirectional_center": outgoing_center + incoming_center,
            "factorized_outgoing_patch": outgoing_patch,
            "factorized_incoming_patch": incoming_patch,
            "factorized_bidirectional_patch": outgoing_patch + incoming_patch,
        }
        for name, value in scores.items():
            signal_scores[name].append(float(value))
        hits.append(hit)
        candidates.append({
            "rank_attention": int(rank),
            "pixel": xy,
            "pixel_index": int(pixel),
            "pck_hit": hit,
            "replay_cell": int(trg_cell),
            "scores": scores,
        })
    ranks = {name: _rank_first_hit(values, hits) for name, values in signal_scores.items()}
    score_gaps = {}
    for name, values in signal_scores.items():
        top1 = values[0] if values else None
        hit_values = [value for value, hit in zip(values, hits) if hit]
        best_hit = max(hit_values) if hit_values else None
        score_gaps[f"{name}_attention_top1_minus_best_pck_hit_proposal"] = (
            float(top1 - best_hit) if top1 is not None and best_hit is not None else None
        )
    return {
        "radius": int(max(1, radius)),
        "basis_radius": int(max(0, basis_radius)),
        "proposal_count": len(candidates),
        "score_names": list(score_names),
        "ranks": ranks,
        "score_gaps": score_gaps,
        "candidates": candidates,
    }


def _rank_first_hit(scores: list[float], hits: list[bool]) -> int | None:
    order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    for rank, index in enumerate(order, start=1):
        if hits[index]:
            return rank
    return None


def _candidate_descriptor_audit_for_point(
    src_features: torch.Tensor,
    trg_features: torch.Tensor,
    attention: dict[str, torch.Tensor],
    source_point: Sequence[float],
    source_size: Sequence[int],
    target_size: Sequence[int],
    src_state: FluxReplayState,
    trg_state: FluxReplayState,
    src_cell: int,
    proposal_pixels: torch.Tensor,
    *,
    target: Sequence[float] | None,
    pck_threshold: float | None,
    gt_pixel: torch.Tensor | None,
) -> dict[str, Any]:
    """Audit candidate-side identity signals without changing predictions."""

    target_h, target_w = int(target_size[0]), int(target_size[1])
    source_pixel = torch.tensor(
        [[float(source_point[0]), float(source_point[1])]],
        device=src_features.device,
        dtype=torch.float32,
    )
    candidate_pixels = proposal_pixels.to(device=trg_features.device, dtype=torch.long).flatten()
    ordered_pixels = []
    seen_pixels = set()
    for pixel in candidate_pixels.detach().cpu().tolist():
        pixel = int(pixel)
        if pixel not in seen_pixels:
            ordered_pixels.append(pixel)
            seen_pixels.add(pixel)
    proposal_pixel_set = set(ordered_pixels)
    proposal_count = len(ordered_pixels)
    if gt_pixel is not None:
        gt_pixel_int_for_order = int(gt_pixel.detach().cpu())
        if gt_pixel_int_for_order not in seen_pixels:
            ordered_pixels.append(gt_pixel_int_for_order)
    candidate_pixels = torch.tensor(ordered_pixels, device=trg_features.device, dtype=torch.long)
    candidate_xy = _pixel_indices_to_xy(candidate_pixels, target_size).to(device=trg_features.device)

    src_stride = (
        float(source_size[1]) / float(max(1, src_state.image_width)),
        float(source_size[0]) / float(max(1, src_state.image_height)),
    )
    trg_stride = (
        float(target_size[1]) / float(max(1, trg_state.image_width)),
        float(target_size[0]) / float(max(1, trg_state.image_height)),
    )
    src_lss = _local_self_similarity_descriptor(
        src_features,
        source_pixel,
        source_size,
        src_stride,
    )
    trg_lss = _local_self_similarity_descriptor(
        trg_features,
        candidate_xy,
        target_size,
        trg_stride,
    )
    local_self_similarity = F.cosine_similarity(
        F.normalize(src_lss, dim=1, eps=1e-12).expand_as(trg_lss),
        F.normalize(trg_lss, dim=1, eps=1e-12),
        dim=1,
        eps=1e-12,
    )

    mutual_attention = torch.sqrt((attention["p_ab"].float() * attention["p_ba"].float().t()).clamp_min(0.0))
    mutual_attention = torch.nan_to_num(mutual_attention, nan=0.0, posinf=0.0, neginf=0.0)
    candidate_cells = _pixel_indices_to_replay_cells(candidate_pixels, target_size, trg_state)
    attention_jacobian = _attention_local_support_scores(
        mutual_attention,
        src_cell,
        candidate_cells,
        src_state,
        trg_state,
    )

    native_scores, native_indices = _chunked_descriptor_topk_scores_indices(
        src_features,
        trg_features,
        [source_point],
        source_size,
        target_size,
        topk=int(candidate_pixels.shape[0]),
        candidate_indices=candidate_pixels.reshape(1, -1),
    )
    native_score_by_pixel = {
        int(pixel): float(score)
        for pixel, score in zip(
            native_indices[0].detach().cpu().tolist(),
            native_scores[0].detach().cpu().tolist(),
        )
    }
    attention_score_by_pixel = {
        int(pixel): float(score)
        for pixel, score in zip(
            proposal_pixels.detach().cpu().tolist(),
            mutual_attention[int(src_cell), _pixel_indices_to_replay_cells(proposal_pixels, target_size, trg_state)]
            .detach()
            .cpu()
            .tolist(),
        )
    }

    candidates: list[dict[str, Any]] = []
    hits: list[bool] = []
    signal_scores = {
        "native_descriptor": [],
        "local_self_similarity": [],
        "attention_jacobian": [],
        "attention": [],
    }
    gt_pixel_int = int(gt_pixel.detach().cpu()) if gt_pixel is not None else None
    for index, pixel_tensor in enumerate(candidate_pixels.detach().cpu().tolist()):
        pixel = int(pixel_tensor)
        xy = [int(pixel % target_w), int(pixel // target_w)]
        hit = bool(_point_hit(xy, target, pck_threshold)) if target is not None and pck_threshold is not None else False
        hits.append(hit)
        native_score = native_score_by_pixel.get(pixel, -float("inf"))
        lss_score = float(local_self_similarity[index].detach().cpu())
        jacobian_score = float(attention_jacobian[index].detach().cpu())
        attention_score = float(attention_score_by_pixel.get(pixel, 0.0))
        signal_scores["native_descriptor"].append(native_score)
        signal_scores["local_self_similarity"].append(lss_score)
        signal_scores["attention_jacobian"].append(jacobian_score)
        signal_scores["attention"].append(attention_score)
        candidates.append({
            "pixel": xy,
            "pixel_index": pixel,
            "is_attention_proposal": bool(pixel in proposal_pixel_set),
            "is_gt_pixel": bool(gt_pixel_int is not None and pixel == gt_pixel_int),
            "pck_hit": hit,
            "native_descriptor": native_score,
            "local_self_similarity": lss_score,
            "attention_jacobian": jacobian_score,
            "attention": attention_score,
        })

    proposal_mask = [candidate["is_attention_proposal"] for candidate in candidates]
    proposal_hits = [hit and keep for hit, keep in zip(hits, proposal_mask)]
    proposal_scores = {
        name: [score if keep else -float("inf") for score, keep in zip(values, proposal_mask)]
        for name, values in signal_scores.items()
    }
    proposal_only_ranks = {
        name: _rank_first_hit(values, proposal_hits)
        for name, values in proposal_scores.items()
    }
    gt_exact_augmented_ranks = {
        name: _rank_first_hit(values, hits)
        for name, values in signal_scores.items()
    }
    attention_top1_pixel = int(proposal_pixels.flatten()[0].detach().cpu()) if proposal_pixels.numel() else None
    gt_exact_gaps = {}
    proposal_hit_gaps = {}
    for name, values in signal_scores.items():
        score_by_pixel = {
            int(pixel): float(value)
            for pixel, value in zip(candidate_pixels.detach().cpu().tolist(), values)
        }
        gt_score = score_by_pixel.get(gt_pixel_int) if gt_pixel_int is not None else None
        top1_score = score_by_pixel.get(attention_top1_pixel) if attention_top1_pixel is not None else None
        gt_exact_gaps[f"{name}_attention_top1_minus_gt_exact"] = (
            float(top1_score - gt_score) if top1_score is not None and gt_score is not None else None
        )
        proposal_hit_scores = [
            float(score)
            for score, hit, keep in zip(values, hits, proposal_mask)
            if hit and keep
        ]
        best_hit_score = max(proposal_hit_scores) if proposal_hit_scores else None
        proposal_hit_gaps[f"{name}_attention_top1_minus_best_pck_hit_proposal"] = (
            float(top1_score - best_hit_score)
            if top1_score is not None and best_hit_score is not None
            else None
        )
    return {
        "candidate_count": int(candidate_pixels.shape[0]),
        "proposal_count": int(proposal_count),
        "gt_exact_in_proposals": bool(gt_pixel_int is not None and gt_pixel_int in proposal_pixel_set),
        "gt_pixel_index": gt_pixel_int,
        "ranks": proposal_only_ranks,
        "proposal_only_ranks": proposal_only_ranks,
        "gt_exact_augmented_ranks": gt_exact_augmented_ranks,
        "score_gaps": proposal_hit_gaps,
        "proposal_hit_score_gaps": proposal_hit_gaps,
        "gt_exact_score_gaps": gt_exact_gaps,
        "candidates": candidates,
    }


def _method_descriptor_audit_for_point(
    src_descriptor_map: torch.Tensor,
    trg_descriptor_map: torch.Tensor,
    descriptor_name: str,
    source_point: Sequence[float],
    source_size: Sequence[int],
    target_size: Sequence[int],
    proposal_pixels: torch.Tensor,
    *,
    target: Sequence[float] | None,
    pck_threshold: float | None,
    gt_pixel: torch.Tensor | None,
) -> dict[str, Any]:
    """Audit a matcher descriptor on the same attention proposal set."""

    target_h, target_w = int(target_size[0]), int(target_size[1])
    candidate_pixels = proposal_pixels.to(device=trg_descriptor_map.device, dtype=torch.long).flatten()
    ordered_pixels = []
    seen_pixels = set()
    for pixel in candidate_pixels.detach().cpu().tolist():
        pixel = int(pixel)
        if pixel not in seen_pixels:
            ordered_pixels.append(pixel)
            seen_pixels.add(pixel)
    proposal_pixel_set = set(ordered_pixels)
    proposal_count = len(ordered_pixels)
    if gt_pixel is not None:
        gt_pixel_int_for_order = int(gt_pixel.detach().cpu())
        if gt_pixel_int_for_order not in seen_pixels:
            ordered_pixels.append(gt_pixel_int_for_order)
    candidate_pixels = torch.tensor(ordered_pixels, device=trg_descriptor_map.device, dtype=torch.long)
    scores, indices = _chunked_descriptor_topk_scores_indices(
        src_descriptor_map,
        trg_descriptor_map,
        [source_point],
        source_size,
        target_size,
        topk=int(candidate_pixels.shape[0]),
        candidate_indices=candidate_pixels.reshape(1, -1),
    )
    score_by_pixel = {
        int(pixel): float(score)
        for pixel, score in zip(
            indices[0].detach().cpu().tolist(),
            scores[0].detach().cpu().tolist(),
        )
    }

    signal_name = "method_descriptor"
    candidates: list[dict[str, Any]] = []
    hits: list[bool] = []
    signal_scores: list[float] = []
    gt_pixel_int = int(gt_pixel.detach().cpu()) if gt_pixel is not None else None
    for pixel in candidate_pixels.detach().cpu().tolist():
        pixel = int(pixel)
        xy = [int(pixel % target_w), int(pixel // target_w)]
        hit = bool(_point_hit(xy, target, pck_threshold)) if target is not None and pck_threshold is not None else False
        score = float(score_by_pixel.get(pixel, -float("inf")))
        hits.append(hit)
        signal_scores.append(score)
        candidates.append({
            "pixel": xy,
            "pixel_index": pixel,
            "is_attention_proposal": bool(pixel in proposal_pixel_set),
            "is_gt_pixel": bool(gt_pixel_int is not None and pixel == gt_pixel_int),
            "pck_hit": hit,
            signal_name: score,
        })

    proposal_mask = [candidate["is_attention_proposal"] for candidate in candidates]
    proposal_hits = [hit and keep for hit, keep in zip(hits, proposal_mask)]
    proposal_scores = [score if keep else -float("inf") for score, keep in zip(signal_scores, proposal_mask)]
    proposal_only_ranks = {signal_name: _rank_first_hit(proposal_scores, proposal_hits)}
    gt_exact_augmented_ranks = {signal_name: _rank_first_hit(signal_scores, hits)}
    attention_top1_pixel = int(proposal_pixels.flatten()[0].detach().cpu()) if proposal_pixels.numel() else None
    gt_score = score_by_pixel.get(gt_pixel_int) if gt_pixel_int is not None else None
    top1_score = score_by_pixel.get(attention_top1_pixel) if attention_top1_pixel is not None else None
    proposal_hit_scores = [
        float(score)
        for score, hit, keep in zip(signal_scores, hits, proposal_mask)
        if hit and keep
    ]
    best_hit_score = max(proposal_hit_scores) if proposal_hit_scores else None
    return {
        "descriptor_name": str(descriptor_name),
        "candidate_count": int(candidate_pixels.shape[0]),
        "proposal_count": int(proposal_count),
        "gt_exact_in_proposals": bool(gt_pixel_int is not None and gt_pixel_int in proposal_pixel_set),
        "gt_pixel_index": gt_pixel_int,
        "score_names": [signal_name],
        "ranks": proposal_only_ranks,
        "proposal_only_ranks": proposal_only_ranks,
        "gt_exact_augmented_ranks": gt_exact_augmented_ranks,
        "score_gaps": {
            f"{signal_name}_attention_top1_minus_best_pck_hit_proposal": (
                float(top1_score - best_hit_score)
                if top1_score is not None and best_hit_score is not None
                else None
            ),
        },
        "proposal_hit_score_gaps": {
            f"{signal_name}_attention_top1_minus_best_pck_hit_proposal": (
                float(top1_score - best_hit_score)
                if top1_score is not None and best_hit_score is not None
                else None
            ),
        },
        "gt_exact_score_gaps": {
            f"{signal_name}_attention_top1_minus_gt_exact": (
                float(top1_score - gt_score)
                if top1_score is not None and gt_score is not None
                else None
            ),
        },
        "candidates": candidates,
    }


def _multilayer_identity_audit_for_point(
    descriptor_maps: dict[str, tuple[torch.Tensor, torch.Tensor]],
    source_point: Sequence[float],
    proposal_pixels: torch.Tensor,
    source_size: Sequence[int],
    target_size: Sequence[int],
    *,
    target: Sequence[float] | None,
    pck_threshold: float | None,
    gt_pixel: torch.Tensor | None,
) -> dict[str, Any]:
    """Audit whether native descriptors from multiple FLUX blocks recover the GT inside the attention basin."""

    if not descriptor_maps:
        return {
            "candidate_count": 0,
            "proposal_count": 0,
            "gt_exact_in_proposals": False,
            "gt_pixel_index": None,
            "score_names": [],
            "ranks": {},
            "proposal_only_ranks": {},
            "gt_exact_augmented_ranks": {},
            "score_gaps": {},
            "proposal_hit_score_gaps": {},
            "gt_exact_score_gaps": {},
            "candidates": [],
        }
    target_h, target_w = int(target_size[0]), int(target_size[1])
    candidate_pixels = proposal_pixels.to(device=next(iter(descriptor_maps.values()))[0].device, dtype=torch.long).flatten()
    ordered_pixels: list[int] = []
    seen_pixels: set[int] = set()
    for pixel in candidate_pixels.detach().cpu().tolist():
        pixel = int(pixel)
        if pixel not in seen_pixels:
            ordered_pixels.append(pixel)
            seen_pixels.add(pixel)
    proposal_pixel_set = set(ordered_pixels)
    proposal_count = len(ordered_pixels)
    if gt_pixel is not None:
        gt_pixel_int_for_order = int(gt_pixel.detach().cpu())
        if gt_pixel_int_for_order not in seen_pixels:
            ordered_pixels.append(gt_pixel_int_for_order)
    device = next(iter(descriptor_maps.values()))[0].device
    candidate_pixels = torch.tensor(ordered_pixels, device=device, dtype=torch.long)

    score_names = list(descriptor_maps.keys())
    score_by_name_pixel: dict[str, dict[int, float]] = {}
    for name, (src_map, trg_map) in descriptor_maps.items():
        scores, indices = _chunked_descriptor_topk_scores_indices(
            src_map,
            trg_map,
            [source_point],
            source_size,
            target_size,
            topk=int(candidate_pixels.shape[0]),
            candidate_indices=candidate_pixels.reshape(1, -1),
        )
        score_by_name_pixel[name] = {
            int(pixel): float(score)
            for pixel, score in zip(
                indices[0].detach().cpu().tolist(),
                scores[0].detach().cpu().tolist(),
            )
        }

    candidates: list[dict[str, Any]] = []
    hits: list[bool] = []
    signal_scores: dict[str, list[float]] = {name: [] for name in score_names}
    gt_pixel_int = int(gt_pixel.detach().cpu()) if gt_pixel is not None else None
    for pixel in candidate_pixels.detach().cpu().tolist():
        pixel = int(pixel)
        xy = [int(pixel % target_w), int(pixel // target_w)]
        hit = bool(_point_hit(xy, target, pck_threshold)) if target is not None and pck_threshold is not None else False
        hits.append(hit)
        candidate = {
            "pixel": xy,
            "pixel_index": pixel,
            "is_attention_proposal": bool(pixel in proposal_pixel_set),
            "is_gt_pixel": bool(gt_pixel_int is not None and pixel == gt_pixel_int),
            "pck_hit": hit,
            "scores": {},
        }
        for name in score_names:
            score = float(score_by_name_pixel[name].get(pixel, -float("inf")))
            signal_scores[name].append(score)
            candidate["scores"][name] = score
        candidates.append(candidate)

    proposal_mask = [candidate["is_attention_proposal"] for candidate in candidates]
    proposal_hits = [hit and keep for hit, keep in zip(hits, proposal_mask)]
    ranks = {}
    score_gaps = {}
    attention_top1_pixel = int(proposal_pixels.flatten()[0].detach().cpu()) if proposal_pixels.numel() else None
    for name, values in signal_scores.items():
        proposal_scores = [score if keep else -float("inf") for score, keep in zip(values, proposal_mask)]
        ranks[name] = _rank_first_hit(proposal_scores, proposal_hits)
        score_by_pixel = {
            int(pixel): float(value)
            for pixel, value in zip(candidate_pixels.detach().cpu().tolist(), values)
        }
        top1_score = score_by_pixel.get(attention_top1_pixel) if attention_top1_pixel is not None else None
        proposal_hit_scores = [
            float(score)
            for score, hit, keep in zip(values, hits, proposal_mask)
            if hit and keep
        ]
        best_hit_score = max(proposal_hit_scores) if proposal_hit_scores else None
        score_gaps[f"{name}_attention_top1_minus_best_pck_hit_proposal"] = (
            float(top1_score - best_hit_score)
            if top1_score is not None and best_hit_score is not None
            else None
        )

    return {
        "candidate_count": int(candidate_pixels.shape[0]),
        "proposal_count": int(proposal_count),
        "gt_exact_in_proposals": bool(gt_pixel_int is not None and gt_pixel_int in proposal_pixel_set),
        "gt_pixel_index": gt_pixel_int,
        "score_names": score_names,
        "ranks": ranks,
        "proposal_only_ranks": ranks,
        "gt_exact_augmented_ranks": {
            name: _rank_first_hit(values, hits) for name, values in signal_scores.items()
        },
        "score_gaps": score_gaps,
        "proposal_hit_score_gaps": score_gaps,
        "gt_exact_score_gaps": {
            f"{name}_attention_top1_minus_gt_exact": (
                float(
                    (
                        score_by_name_pixel[name].get(attention_top1_pixel)
                        if attention_top1_pixel is not None
                        else None
                    )
                    - (score_by_name_pixel[name].get(gt_pixel_int) if gt_pixel_int is not None else None)
                )
                if attention_top1_pixel is not None and gt_pixel_int is not None
                and score_by_name_pixel[name].get(attention_top1_pixel) is not None
                and score_by_name_pixel[name].get(gt_pixel_int) is not None
                else None
            )
            for name in score_names
        },
        "candidates": candidates,
    }


def _multilayer_identity_audit_for_points(
    descriptor_maps: dict[str, tuple[torch.Tensor, torch.Tensor]],
    source_points: Sequence[Sequence[float]],
    proposal_pixels: torch.Tensor,
    source_size: Sequence[int],
    target_size: Sequence[int],
    *,
    target_points: Sequence[Sequence[float]] | None,
    pck_threshold: float | None,
    gt_pixels: torch.Tensor | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row, source_point in enumerate(source_points):
        rows.append(
            _multilayer_identity_audit_for_point(
                descriptor_maps,
                source_point,
                proposal_pixels[row],
                source_size,
                target_size,
                target=target_points[row] if target_points is not None else None,
                pck_threshold=pck_threshold,
                gt_pixel=gt_pixels[row] if gt_pixels is not None else None,
            )
        )
    return rows


def _transport_lift_branch_audit_for_point(
    branch_descriptors: dict[str, tuple[torch.Tensor, torch.Tensor]],
    source_point: Sequence[float],
    source_size: Sequence[int],
    target_size: Sequence[int],
    proposal_pixels: torch.Tensor,
    *,
    target: Sequence[float] | None,
    pck_threshold: float | None,
    gt_pixel: torch.Tensor | None,
) -> dict[str, Any]:
    """Audit transport-lift branches on the same attention proposal set."""

    target_h, target_w = int(target_size[0]), int(target_size[1])
    device = next(iter(branch_descriptors.values()))[1].device
    candidate_pixels = proposal_pixels.to(device=device, dtype=torch.long).flatten()
    ordered_pixels = []
    seen_pixels = set()
    for pixel in candidate_pixels.detach().cpu().tolist():
        pixel = int(pixel)
        if pixel not in seen_pixels:
            ordered_pixels.append(pixel)
            seen_pixels.add(pixel)
    proposal_pixel_set = set(ordered_pixels)
    proposal_count = len(ordered_pixels)
    if gt_pixel is not None:
        gt_pixel_int_for_order = int(gt_pixel.detach().cpu())
        if gt_pixel_int_for_order not in seen_pixels:
            ordered_pixels.append(gt_pixel_int_for_order)
    candidate_pixels = torch.tensor(ordered_pixels, device=device, dtype=torch.long)

    score_names = list(branch_descriptors.keys())
    score_by_name_pixel: dict[str, dict[int, float]] = {}
    for name, (src_map, trg_map) in branch_descriptors.items():
        scores, indices = _chunked_descriptor_topk_scores_indices(
            src_map,
            trg_map,
            [source_point],
            source_size,
            target_size,
            topk=int(candidate_pixels.shape[0]),
            candidate_indices=candidate_pixels.reshape(1, -1),
        )
        score_by_name_pixel[name] = {
            int(pixel): float(score)
            for pixel, score in zip(
                indices[0].detach().cpu().tolist(),
                scores[0].detach().cpu().tolist(),
            )
        }

    candidates: list[dict[str, Any]] = []
    hits: list[bool] = []
    signal_scores: dict[str, list[float]] = {name: [] for name in score_names}
    gt_pixel_int = int(gt_pixel.detach().cpu()) if gt_pixel is not None else None
    for pixel in candidate_pixels.detach().cpu().tolist():
        pixel = int(pixel)
        xy = [int(pixel % target_w), int(pixel // target_w)]
        hit = bool(_point_hit(xy, target, pck_threshold)) if target is not None and pck_threshold is not None else False
        hits.append(hit)
        candidate = {
            "pixel": xy,
            "pixel_index": pixel,
            "is_attention_proposal": bool(pixel in proposal_pixel_set),
            "is_gt_pixel": bool(gt_pixel_int is not None and pixel == gt_pixel_int),
            "pck_hit": hit,
            "scores": {},
        }
        for name in score_names:
            score = float(score_by_name_pixel[name].get(pixel, -float("inf")))
            signal_scores[name].append(score)
            candidate["scores"][name] = score
        candidates.append(candidate)

    proposal_mask = [candidate["is_attention_proposal"] for candidate in candidates]
    proposal_hits = [hit and keep for hit, keep in zip(hits, proposal_mask)]
    ranks = {}
    score_gaps = {}
    attention_top1_pixel = int(proposal_pixels.flatten()[0].detach().cpu()) if proposal_pixels.numel() else None
    for name, values in signal_scores.items():
        proposal_scores = [score if keep else -float("inf") for score, keep in zip(values, proposal_mask)]
        ranks[name] = _rank_first_hit(proposal_scores, proposal_hits)
        score_by_pixel = {
            int(pixel): float(value)
            for pixel, value in zip(candidate_pixels.detach().cpu().tolist(), values)
        }
        top1_score = score_by_pixel.get(attention_top1_pixel) if attention_top1_pixel is not None else None
        proposal_hit_scores = [
            float(score)
            for score, hit, keep in zip(values, hits, proposal_mask)
            if hit and keep
        ]
        best_hit_score = max(proposal_hit_scores) if proposal_hit_scores else None
        score_gaps[f"{name}_attention_top1_minus_best_pck_hit_proposal"] = (
            float(top1_score - best_hit_score)
            if top1_score is not None and best_hit_score is not None
            else None
        )

    return {
        "candidate_count": int(candidate_pixels.shape[0]),
        "proposal_count": int(proposal_count),
        "gt_exact_in_proposals": bool(gt_pixel_int is not None and gt_pixel_int in proposal_pixel_set),
        "gt_pixel_index": gt_pixel_int,
        "score_names": score_names,
        "ranks": ranks,
        "score_gaps": score_gaps,
        "candidates": candidates,
    }


def _point_hit(point: Sequence[float], target: Sequence[float], threshold: float) -> bool:
    dx = float(point[0]) - float(target[0])
    dy = float(point[1]) - float(target[1])
    return (dx * dx + dy * dy) ** 0.5 <= float(threshold) * 0.1


def _native_cell_indices_for_points(
    points: Iterable[Sequence[float]],
    image_size: Sequence[int],
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    image_h, image_w = int(image_size[0]), int(image_size[1])
    tensor = torch.tensor(
        [[float(point[0]), float(point[1])] for point in points],
        device=device,
        dtype=torch.float32,
    )
    if tensor.numel() == 0:
        return torch.empty((0,), device=device, dtype=torch.long)
    x = torch.floor((tensor[:, 0] + 0.5) * float(width) / float(image_w)).long().clamp_(0, width - 1)
    y = torch.floor((tensor[:, 1] + 0.5) * float(height) / float(image_h)).long().clamp_(0, height - 1)
    return y * width + x


def _decode_native_cells(indices: torch.Tensor, image_size: Sequence[int], height: int, width: int) -> list[list[int]]:
    image_h, image_w = int(image_size[0]), int(image_size[1])
    output = []
    for index in indices.detach().cpu().tolist():
        index = int(index)
        x = (index % width + 0.5) * float(image_w) / float(width) - 0.5
        y = (index // width + 0.5) * float(image_h) / float(height) - 0.5
        output.append([
            int(round(max(0.0, min(float(image_w - 1), x)))),
            int(round(max(0.0, min(float(image_h - 1), y)))),
        ])
    return output


def _attention_case_records_from_attention(
    attention: dict[str, torch.Tensor],
    source_points: Iterable[Sequence[float]],
    source_size: Sequence[int],
    target_size: Sequence[int],
    src_state: FluxReplayState,
    trg_state: FluxReplayState,
    *,
    target_points: Iterable[Sequence[float]] | None = None,
    pck_threshold: float | None = None,
    candidate_topk: int = 20,
) -> list[dict[str, Any]]:
    points = list(source_points)
    gt_points = list(target_points) if target_points is not None else None
    if not points:
        return []
    if gt_points is not None and len(gt_points) != len(points):
        raise ValueError("target_points must align with source_points")
    src_cells = _native_cell_indices_for_points(
        points,
        source_size,
        src_state.image_height,
        src_state.image_width,
        attention["p_ab"].device,
    )
    mutual = torch.sqrt((attention["p_ab"].float() * attention["p_ba"].float().t()).clamp_min(0.0))
    mutual = torch.nan_to_num(mutual, nan=0.0, posinf=0.0, neginf=0.0)
    k = min(max(1, int(candidate_topk)), int(mutual.shape[1]))
    top_cells = torch.topk(mutual[src_cells], k=k, dim=1, sorted=True).indices
    decoded = _decode_native_cells(
        top_cells.reshape(-1),
        target_size,
        trg_state.image_height,
        trg_state.image_width,
    )
    rows: list[dict[str, Any]] = []
    for row_idx in range(len(points)):
        target = gt_points[row_idx] if gt_points is not None else None
        top_pixels = decoded[row_idx * k:(row_idx + 1) * k]
        hit_rank = None
        for rank, pixel in enumerate(top_pixels, start=1):
            hit = (
                bool(_point_hit(pixel, target, pck_threshold))
                if target is not None and pck_threshold is not None
                else False
            )
            if hit:
                hit_rank = int(rank)
                break
        rows.append({
            "attention_top1_pck_hit": bool(hit_rank == 1),
            "attention_topk_pck_hit": bool(hit_rank is not None),
            "attention_gt_rank": hit_rank,
            "attention_top1_pixel": top_pixels[0] if top_pixels else None,
        })
    return rows


def _trajectory_attention_layers(
    trajectory_replay_states: dict[str, Any] | None,
    trajectory_block_modules: dict[str, Any] | None,
    trajectory_blocks: Sequence[int],
    *,
    fallback_src_state: FluxReplayState,
    fallback_trg_state: FluxReplayState,
    fallback_attention: dict[str, torch.Tensor],
) -> dict[str, dict[str, Any]]:
    blocks = tuple(sorted({int(block) for block in trajectory_blocks}))
    if not blocks:
        main_block = int(fallback_src_state.global_block_index) + 1
        return {str(main_block): {"src_state": fallback_src_state, "trg_state": fallback_trg_state, "attention": fallback_attention}}
    if trajectory_replay_states is None or trajectory_block_modules is None:
        raise ValueError("cross-attention trajectory requires replay states and block modules")
    layers: dict[str, dict[str, Any]] = {}
    for block in blocks:
        key = str(int(block))
        src_state = FluxReplayState.from_dict(trajectory_replay_states["src"][key])
        trg_state = FluxReplayState.from_dict(trajectory_replay_states["trg"][key])
        module = trajectory_block_modules.get(key)
        if module is None:
            raise ValueError(f"missing trajectory block module for block {key}")
        if src_state.global_block_index != int(block) - 1 or trg_state.global_block_index != int(block) - 1:
            raise ValueError(f"trajectory state for block {key} starts from an incompatible pre-block")
        _src_out, _trg_out, attention = run_flux_cross_only_stack([module], src_state, trg_state)
        layers[key] = {"src_state": src_state, "trg_state": trg_state, "attention": attention}
    return layers


def _trajectory_mutual(attention: dict[str, torch.Tensor]) -> torch.Tensor:
    mutual = torch.sqrt((attention["p_ab"].float() * attention["p_ba"].float().t()).clamp_min(0.0))
    return torch.nan_to_num(mutual, nan=0.0, posinf=0.0, neginf=0.0)


def _cross_attention_trajectory_rankings(
    trajectory_layers: dict[str, dict[str, Any]],
    main_block: int,
    points: Sequence[Sequence[float]],
    source_size: Sequence[int],
    target_size: Sequence[int],
    *,
    candidate_topk: int,
    target_points: Sequence[Sequence[float]] | None = None,
    pck_threshold: float | None = None,
) -> tuple[torch.Tensor, list[dict[str, Any]], dict[str, float]]:
    if not points:
        return torch.empty((0, 0), dtype=torch.long), [], {}
    ordered_blocks = tuple(sorted(int(block) for block in trajectory_layers.keys()))
    main_key = str(int(main_block))
    if main_key not in trajectory_layers:
        main_key = str(min(ordered_blocks, key=lambda block: abs(int(block) - int(main_block))))
    main_layer = trajectory_layers[main_key]
    src_state: FluxReplayState = main_layer["src_state"]
    trg_state: FluxReplayState = main_layer["trg_state"]
    device = main_layer["attention"]["p_ab"].device
    src_cells = _native_cell_indices_for_points(
        points,
        source_size,
        src_state.image_height,
        src_state.image_width,
        device,
    )
    layer_mutual = {str(block): _trajectory_mutual(trajectory_layers[str(block)]["attention"]) for block in ordered_blocks}
    main_mutual = layer_mutual[main_key]
    k = min(max(1, int(candidate_topk)), int(main_mutual.shape[1]))
    main_candidates = torch.topk(main_mutual[src_cells], k=k, dim=1, sorted=True).indices
    ranked_cells: list[list[int]] = []
    audits: list[dict[str, Any]] = []
    stability_values: list[float] = []
    target_h, target_w = int(target_size[0]), int(target_size[1])

    for row_index, src_cell_value in enumerate(src_cells.detach().cpu().tolist()):
        src_cell = int(src_cell_value)
        candidate_cells = [int(cell) for cell in main_candidates[row_index].detach().cpu().tolist()]
        scored: list[tuple[tuple[float, float, float, float], int, dict[str, Any]]] = []
        for attention_rank, cell in enumerate(candidate_cells, start=1):
            per_block: dict[str, dict[str, float]] = {}
            topk_count = 0
            reciprocal_rank_sum = 0.0
            centered_score_sum = 0.0
            for block in ordered_blocks:
                matrix = layer_mutual[str(block)]
                row = matrix[src_cell]
                value_tensor = row[cell]
                value = float(value_tensor.detach().cpu())
                rank = int((row > value_tensor).sum().detach().cpu()) + 1
                topk_hit = rank <= k
                topk_count += int(topk_hit)
                reciprocal_rank = 1.0 / float(rank)
                centered_score = value - float(row.mean().detach().cpu())
                reciprocal_rank_sum += reciprocal_rank
                centered_score_sum += centered_score
                per_block[str(block)] = {
                    "rank": float(rank),
                    "score": value,
                    "topk_hit": float(topk_hit),
                    "reciprocal_rank": reciprocal_rank,
                    "centered_score": centered_score,
                }
            layer_count = float(max(1, len(ordered_blocks)))
            mean_reciprocal_rank = reciprocal_rank_sum / layer_count
            mean_centered_score = centered_score_sum / layer_count
            main_score = float(main_mutual[src_cells[row_index], cell].detach().cpu())
            sort_key = (
                float(topk_count),
                float(mean_reciprocal_rank),
                float(mean_centered_score),
                float(main_score),
            )
            pixel_index = int(
                _cell_topk_to_pixel_indices(torch.tensor([cell], device=device), target_size, trg_state)[0].detach().cpu()
            )
            xy = [int(pixel_index % target_w), int(pixel_index // target_w)]
            hit = (
                bool(_point_hit(xy, target_points[row_index], pck_threshold))
                if target_points is not None and pck_threshold is not None
                else False
            )
            scored.append((
                sort_key,
                cell,
                {
                    "rank_attention": int(attention_rank),
                    "pixel": xy,
                    "pixel_index": int(pixel_index),
                    "replay_cell": int(cell),
                    "pck_hit": hit,
                    "trajectory_topk_count": int(topk_count),
                    "trajectory_topk_rate": float(topk_count / layer_count),
                    "trajectory_mean_reciprocal_rank": float(mean_reciprocal_rank),
                    "trajectory_mean_centered_score": float(mean_centered_score),
                    "main_attention_score": float(main_score),
                    "per_block": per_block,
                },
            ))
        scored.sort(key=lambda item: item[0], reverse=True)
        ranked_cells.append([cell for _key, cell, _payload in scored])
        candidates = []
        for trajectory_rank, (_key, _cell, payload) in enumerate(scored, start=1):
            payload = dict(payload)
            payload["rank_trajectory"] = int(trajectory_rank)
            candidates.append(payload)
        hits = [bool(item["pck_hit"]) for item in candidates]
        trajectory_scores = [float(len(candidates) - index) for index in range(len(candidates))]
        ranks = {"trajectory": _rank_first_hit(trajectory_scores, hits)}
        stability_values.append(float(candidates[0]["trajectory_topk_rate"]) if candidates else 0.0)
        audits.append({
            "blocks": [int(block) for block in ordered_blocks],
            "main_block": int(main_key),
            "candidate_topk": int(k),
            "candidate_count": len(candidates),
            "score_names": [
                "trajectory_topk_count",
                "trajectory_mean_reciprocal_rank",
                "trajectory_mean_centered_score",
                "main_attention_score",
            ],
            "ranks": ranks,
            "candidates": candidates,
        })
    max_len = max((len(row) for row in ranked_cells), default=0)
    padded = []
    for row in ranked_cells:
        if len(row) < max_len:
            row = row + row[-1:] * (max_len - len(row)) if row else [0] * max_len
        padded.append(row)
    ranked_tensor = torch.tensor(padded, device=device, dtype=torch.long)
    summary = {
        "trajectory_layer_count": float(len(ordered_blocks)),
        "trajectory_mean_top1_stability": float(sum(stability_values) / max(1, len(stability_values))),
    }
    return ranked_tensor, audits, summary


def flux_fjsar_attention_case_records(
    source_points: Iterable[Sequence[float]],
    source_size: Sequence[int],
    target_size: Sequence[int],
    *,
    src_replay_state: dict[str, Any],
    trg_replay_state: dict[str, Any],
    blocks: Sequence[Any],
    interaction_mode: str = "exact",
    use_coordinate_bias: bool = False,
    target_points: Iterable[Sequence[float]] | None = None,
    pck_threshold: float | None = None,
    candidate_topk: int = 20,
    geometry_radius: int = 2,
    geometry_strength: float = 0.5,
    trajectory_replay_states: dict[str, Any] | None = None,
    trajectory_block_modules: dict[str, Any] | None = None,
    trajectory_blocks: Sequence[int] = (),
) -> list[dict[str, Any]]:
    """Lightweight attention top-k records for dump prefiltering."""

    points = list(source_points)
    gt_points = list(target_points) if target_points is not None else None
    if not points:
        return []
    if gt_points is not None and len(gt_points) != len(points):
        raise ValueError("target_points must align with source_points")
    src_state = FluxReplayState.from_dict(src_replay_state)
    trg_state = FluxReplayState.from_dict(trg_replay_state)
    if src_state.global_block_index != trg_state.global_block_index:
        raise ValueError("source and target replay caches use different starting blocks")
    if len(blocks) not in (1, 2):
        raise ValueError("FJSAR attention case records require one or two replay blocks")
    with torch.no_grad():
        if interaction_mode == "cross_only":
            _src_joint, _trg_joint, attention = run_flux_cross_only_stack(
                blocks,
                src_state,
                trg_state,
            )
        elif interaction_mode == "identity_preserving":
            _src_joint, _trg_joint, attention = run_flux_identity_preserving_stack(
                blocks,
                src_state,
                trg_state,
            )
        elif interaction_mode == "balanced_transport":
            _src_joint, _trg_joint, attention = run_flux_balanced_transport_stack(
                blocks,
                src_state,
                trg_state,
            )
        elif interaction_mode == "qk_identity":
            _src_joint, _trg_joint, attention = run_flux_qk_identity_stack(
                blocks,
                src_state,
                trg_state,
            )
        elif interaction_mode == "trajectory":
            _src_joint, _trg_joint, attention = run_flux_cross_only_stack(
                blocks,
                src_state,
                trg_state,
            )
        elif interaction_mode == "geometry_consistent":
            _src_joint, _trg_joint, attention = run_flux_geometry_consistent_stack(
                blocks,
                src_state,
                trg_state,
                radius=geometry_radius,
                strength=geometry_strength,
            )
        else:
            _src_joint, _trg_joint, attention = run_flux_joint_stack(
                blocks,
                src_state,
                trg_state,
                mode=interaction_mode,
                use_coordinate_bias=use_coordinate_bias,
            )
    return _attention_case_records_from_attention(
        attention,
        points,
        source_size,
        target_size,
        src_state,
        trg_state,
        target_points=gt_points,
        pck_threshold=pck_threshold,
        candidate_topk=candidate_topk,
    )


def _conditional_cross_distribution(probability: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    probability = torch.nan_to_num(probability.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    mass = probability.sum(dim=1).clamp(0.0, 1.0)
    conditional = probability / mass.clamp_min(1e-12).unsqueeze(1)
    conditional = torch.nan_to_num(conditional, nan=0.0, posinf=0.0, neginf=0.0)
    return conditional, mass


def _extract_ada_shift_scale(
    ada: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read the exact shift/scale layout used by the official evaluator."""

    if isinstance(ada, torch.Tensor):
        tensor = ada
        if tensor.ndim >= 3:
            shift = tensor[0, 0]
            scale = tensor[0, 1]
        elif tensor.ndim == 2 and tensor.shape[0] == 2:
            shift, scale = tensor[0], tensor[1]
        else:
            raise ValueError(f"unsupported AdaLN tensor shape: {tuple(tensor.shape)}")
    elif isinstance(ada, (list, tuple)) and len(ada) > 0:
        first = ada[0]
        if isinstance(first, torch.Tensor) and first.ndim >= 2 and first.shape[0] >= 2:
            shift, scale = first[0], first[1]
        elif isinstance(first, (list, tuple)) and len(first) >= 2:
            shift, scale = first[0], first[1]
        elif len(ada) >= 2:
            shift, scale = ada[0], ada[1]
        else:
            raise ValueError("unsupported AdaLN sequence layout")
    else:
        raise TypeError("AdaLN state must be a tensor or tensor sequence")
    shift = torch.as_tensor(shift, device=device, dtype=dtype).reshape(1, 1, -1)
    scale = torch.as_tensor(scale, device=device, dtype=dtype).reshape(1, 1, -1)
    return shift, scale


def _quantile_risk_tokens(values: torch.Tensor, quantile: float, tail: str) -> torch.Tensor:
    flat = values.flatten()
    threshold = torch.quantile(flat, float(quantile))
    if tail == "high":
        denom = (flat.max() - threshold).clamp_min(1e-6)
        return ((values - threshold) / denom).clamp(0.0, 1.0)
    if tail == "low":
        denom = (threshold - flat.min()).clamp_min(1e-6)
        return ((threshold - values) / denom).clamp(0.0, 1.0)
    raise ValueError(f"unsupported risk tail: {tail}")


def _prepare_replay_tokens(
    raw_tokens: torch.Tensor,
    ada: Any,
    *,
    discard_channels: Sequence[int] = (),
    calibration: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Apply the official DiTF postprocess to ensemble-averaged raw tokens.

    Frozen replay blocks are evaluated per ensemble member.  Their raw outputs
    are averaged only *after* the nonlinear blocks, matching the ensemble
    reduction of the cached raw feature.  Channel discard, LayerNorm, AdaLN and
    optional shift calibration are then applied identically to the evaluator.
    """

    if raw_tokens.ndim != 3:
        raise ValueError("raw replay tokens must be [E,N,C]")
    tokens = raw_tokens.float().mean(dim=0, keepdim=True)
    if discard_channels:
        tokens = tokens.clone()
        valid = [int(index) for index in discard_channels if 0 <= int(index) < tokens.shape[-1]]
        if valid:
            tokens[..., valid] = 0.0
    tokens_ln = F.layer_norm(tokens, (tokens.shape[-1],), weight=None, bias=None, eps=1e-6)
    shift, scale = _extract_ada_shift_scale(ada, tokens.device, tokens.dtype)
    content = (1.0 + scale) * tokens_ln
    output = content + shift
    config = calibration or {}
    if not config.get("shift_calibration", False) and not config.get("joint_calibration", False):
        return output

    shift_energy = shift.square().sum(dim=-1, keepdim=True)
    content_energy = content.square().sum(dim=-1, keepdim=True)
    post_energy = output.square().sum(dim=-1, keepdim=True).clamp_min(1e-6)
    shift_ratio = shift_energy / post_energy
    content_ratio = content_energy / post_energy
    if config.get("joint_calibration", False):
        high_shift = _quantile_risk_tokens(
            shift_ratio, float(config.get("joint_shift_quantile", 0.75)), "high"
        )
        low_content = _quantile_risk_tokens(
            content_ratio, float(config.get("joint_content_quantile", 0.25)), "low"
        )
        risk = high_shift * low_content
        lambda_map = (
            1.0 - float(config.get("joint_shift_strength", 0.5)) * risk
        ).clamp(min=float(config.get("joint_min_shift_lambda", 0.2)), max=1.0)
        content_gain = (
            1.0 + float(config.get("joint_content_strength", 0.25)) * risk
        ).clamp(max=float(config.get("joint_max_content_gain", 1.5)))
        return content_gain * content + lambda_map * shift
    excess = _quantile_risk_tokens(
        shift_ratio, float(config.get("shift_calibration_quantile", 0.75)), "high"
    )
    lambda_map = (
        1.0 - float(config.get("shift_calibration_strength", 0.5)) * excess
    ).clamp(min=float(config.get("shift_calibration_min_lambda", 0.2)), max=1.0)
    return content + lambda_map * shift


def _raw_feature_tokens(feature: torch.Tensor) -> torch.Tensor:
    if feature.ndim != 4 or feature.shape[0] != 1:
        raise ValueError("cached raw feature must be [1,C,H,W]")
    return feature[0].permute(1, 2, 0).reshape(1, -1, feature.shape[1]).float()


def _attention_concentration_safe(probability: torch.Tensor) -> torch.Tensor:
    if probability.shape[1] <= 1:
        return torch.ones(probability.shape[0], device=probability.device)
    entropy = -(probability.clamp_min(1e-12) * probability.clamp_min(1e-12).log()).sum(dim=1)
    return (1.0 - entropy / float(np.log(probability.shape[1]))).clamp(0.0, 1.0)


def _orthogonal_pair_residual(delta: torch.Tensor, native: torch.Tensor) -> torch.Tensor:
    delta = torch.nan_to_num(delta.float(), nan=0.0, posinf=0.0, neginf=0.0)
    native = torch.nan_to_num(native.float(), nan=0.0, posinf=0.0, neginf=0.0)
    delta = delta - delta.mean(dim=0, keepdim=True)
    residual = delta - (
        (delta * native).sum(dim=1, keepdim=True)
        / native.square().sum(dim=1, keepdim=True).clamp_min(1e-12)
    ) * native
    return torch.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0)


def _fjsar_relation_statistics(
    attention: dict[str, torch.Tensor],
    src_delta: torch.Tensor,
    trg_delta: torch.Tensor,
    src_native: torch.Tensor,
    trg_native: torch.Tensor,
) -> dict[str, torch.Tensor]:
    cond_ab, mass_a = _conditional_cross_distribution(attention["p_ab"])
    cond_ba, mass_b = _conditional_cross_distribution(attention["p_ba"])
    concentration_a = _attention_concentration_safe(cond_ab)
    concentration_b = _attention_concentration_safe(cond_ba)
    reciprocal_a = _rowwise_cosine(cond_ab, cond_ba.t())
    reciprocal_b = _rowwise_cosine(cond_ba, cond_ab.t())
    coordinate_a = attention.get("coordinate_reliability_a", torch.ones_like(mass_a)).float()
    coordinate_b = attention.get("coordinate_reliability_b", torch.ones_like(mass_b)).float()
    # All four signals must agree.  Squaring the conjunction is a fixed
    # native-preserving calibration: weak accidental agreement decays rapidly,
    # while a genuinely strong bidirectional relation remains active.
    relation_a = (mass_a * concentration_a * reciprocal_a * coordinate_a).clamp(0.0, 1.0).square()
    relation_b = (mass_b * concentration_b * reciprocal_b * coordinate_b).clamp(0.0, 1.0).square()

    src_delta_norm = src_delta.float().norm(dim=1)
    trg_delta_norm = trg_delta.float().norm(dim=1)
    src_native_norm = src_native.float().norm(dim=1)
    trg_native_norm = trg_native.float().norm(dim=1)
    magnitude_a = src_delta_norm / (src_native_norm + src_delta_norm).clamp_min(1e-12)
    magnitude_b = trg_delta_norm / (trg_native_norm + trg_delta_norm).clamp_min(1e-12)
    return {
        "cond_ab": cond_ab,
        "cond_ba": cond_ba,
        "cross_mass_a": mass_a,
        "cross_mass_b": mass_b,
        "concentration_a": concentration_a,
        "concentration_b": concentration_b,
        "reciprocal_a": reciprocal_a,
        "reciprocal_b": reciprocal_b,
        "coordinate_a": coordinate_a,
        "coordinate_b": coordinate_b,
        "relation_a": relation_a,
        "relation_b": relation_b,
        "magnitude_a": magnitude_a,
        "magnitude_b": magnitude_b,
    }


def _attention_soft_common(tokens: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    """Average tokens that share the same cross-attention distribution."""

    tokens = torch.nan_to_num(tokens.float(), nan=0.0, posinf=0.0, neginf=0.0)
    cond = torch.nan_to_num(cond.float(), nan=0.0, posinf=0.0, neginf=0.0)
    affinity = (cond @ cond.t()).clamp_min(0.0)
    if affinity.shape[0] > 1:
        affinity.fill_diagonal_(0.0)
    affinity = affinity / affinity.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return affinity @ tokens


def _cross_expected_native_context(
    src_native: torch.Tensor,
    trg_native: torch.Tensor,
    attention: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cond_ab, _mass_a = _conditional_cross_distribution(attention["p_ab"])
    cond_ba, _mass_b = _conditional_cross_distribution(attention["p_ba"])
    src_expected_trg = cond_ab.to(trg_native.device, dtype=trg_native.dtype) @ trg_native.float()
    trg_expected_src = cond_ba.to(src_native.device, dtype=src_native.dtype) @ src_native.float()
    return src_expected_trg, trg_expected_src, cond_ab, cond_ba


def _attention_signature_descriptors(
    src_native_map: torch.Tensor,
    trg_native_map: torch.Tensor,
    attention: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Distributional identity signature from cross-attention expectation.

    The source descriptor is ``[source native, expected target native]`` and the
    target descriptor is ``[expected source native, target native]``.  A correct
    pair is favored only when each side's full attention distribution predicts
    the other side's native identity, rather than when one attention row has the
    largest scalar peak.
    """

    src_height, src_width = src_native_map.shape[-2:]
    trg_height, trg_width = trg_native_map.shape[-2:]
    src_native = src_native_map[0].permute(1, 2, 0).reshape(-1, src_native_map.shape[1]).float()
    trg_native = trg_native_map[0].permute(1, 2, 0).reshape(-1, trg_native_map.shape[1]).float()
    src_expected_trg, trg_expected_src, _cond_ab, _cond_ba = _cross_expected_native_context(
        src_native,
        trg_native,
        attention,
    )
    src_descriptor = F.normalize(
        torch.cat(
            (
                F.normalize(src_native, dim=1, eps=1e-12),
                F.normalize(src_expected_trg, dim=1, eps=1e-12),
            ),
            dim=1,
        ),
        dim=1,
        eps=1e-12,
    )
    trg_descriptor = F.normalize(
        torch.cat(
            (
                F.normalize(trg_expected_src, dim=1, eps=1e-12),
                F.normalize(trg_native, dim=1, eps=1e-12),
            ),
            dim=1,
        ),
        dim=1,
        eps=1e-12,
    )
    return (
        src_descriptor.t().reshape(1, src_descriptor.shape[1], src_height, src_width).contiguous(),
        trg_descriptor.t().reshape(1, trg_descriptor.shape[1], trg_height, trg_width).contiguous(),
    )


def _part_common_sharpen_descriptors(
    src_native_map: torch.Tensor,
    trg_native_map: torch.Tensor,
    attention: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove attention-region common components before NN matching."""

    src_height, src_width = src_native_map.shape[-2:]
    trg_height, trg_width = trg_native_map.shape[-2:]
    src_native = src_native_map[0].permute(1, 2, 0).reshape(-1, src_native_map.shape[1]).float()
    trg_native = trg_native_map[0].permute(1, 2, 0).reshape(-1, trg_native_map.shape[1]).float()
    src_expected_trg, trg_expected_src, cond_ab, cond_ba = _cross_expected_native_context(
        src_native,
        trg_native,
        attention,
    )
    src_native_unique = src_native - _attention_soft_common(src_native, cond_ab.to(src_native.device))
    trg_native_unique = trg_native - _attention_soft_common(trg_native, cond_ba.to(trg_native.device))
    src_context_unique = src_expected_trg - _attention_soft_common(src_expected_trg, cond_ab.to(src_expected_trg.device))
    trg_context_unique = trg_expected_src - _attention_soft_common(trg_expected_src, cond_ba.to(trg_expected_src.device))
    src_descriptor = F.normalize(
        torch.cat(
            (
                F.normalize(src_native, dim=1, eps=1e-12),
                _weighted_unit_branch(src_native_unique, src_native),
                _weighted_unit_branch(src_context_unique, src_native),
            ),
            dim=1,
        ),
        dim=1,
        eps=1e-12,
    )
    trg_descriptor = F.normalize(
        torch.cat(
            (
                F.normalize(trg_native, dim=1, eps=1e-12),
                _weighted_unit_branch(trg_native_unique, trg_native),
                _weighted_unit_branch(trg_context_unique, trg_native),
            ),
            dim=1,
        ),
        dim=1,
        eps=1e-12,
    )
    return (
        src_descriptor.t().reshape(1, src_descriptor.shape[1], src_height, src_width).contiguous(),
        trg_descriptor.t().reshape(1, trg_descriptor.shape[1], trg_height, trg_width).contiguous(),
    )


def _orthogonal_context_descriptors(
    src_native_map: torch.Tensor,
    trg_native_map: torch.Tensor,
    src_joint_tokens: torch.Tensor,
    trg_joint_tokens: torch.Tensor,
    src_native_replay: torch.Tensor,
    trg_native_replay: torch.Tensor,
    attention: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Native identity plus orthogonal cross-context, kept as separate branches."""

    src_height, src_width = src_native_map.shape[-2:]
    trg_height, trg_width = trg_native_map.shape[-2:]
    src_native = src_native_map[0].permute(1, 2, 0).reshape(-1, src_native_map.shape[1]).float()
    trg_native = trg_native_map[0].permute(1, 2, 0).reshape(-1, trg_native_map.shape[1]).float()
    src_expected_trg, trg_expected_src, _cond_ab, _cond_ba = _cross_expected_native_context(
        src_native,
        trg_native,
        attention,
    )
    src_replay_residual = _orthogonal_pair_residual(src_joint_tokens - src_native_replay, src_native)
    trg_replay_residual = _orthogonal_pair_residual(trg_joint_tokens - trg_native_replay, trg_native)
    src_cross_residual = _orthogonal_pair_residual(src_expected_trg - src_native, src_native)
    trg_cross_residual = _orthogonal_pair_residual(trg_expected_src - trg_native, trg_native)
    src_descriptor = F.normalize(
        torch.cat(
            (
                F.normalize(src_native, dim=1, eps=1e-12),
                F.normalize(src_expected_trg, dim=1, eps=1e-12),
                _weighted_unit_branch(src_cross_residual, src_native),
                _weighted_unit_branch(src_replay_residual, src_native),
            ),
            dim=1,
        ),
        dim=1,
        eps=1e-12,
    )
    trg_descriptor = F.normalize(
        torch.cat(
            (
                F.normalize(trg_expected_src, dim=1, eps=1e-12),
                F.normalize(trg_native, dim=1, eps=1e-12),
                _weighted_unit_branch(trg_cross_residual, trg_native),
                _weighted_unit_branch(trg_replay_residual, trg_native),
            ),
            dim=1,
        ),
        dim=1,
        eps=1e-12,
    )
    return (
        src_descriptor.t().reshape(1, src_descriptor.shape[1], src_height, src_width).contiguous(),
        trg_descriptor.t().reshape(1, trg_descriptor.shape[1], trg_height, trg_width).contiguous(),
    )


def _spectral_attention_identity_descriptors(
    src_native_map: torch.Tensor,
    trg_native_map: torch.Tensor,
    attention: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Native descriptor augmented with shared spectral coordinates of attention.

    Cross-attention is a soft bipartite correspondence operator.  Its row peaks
    are often part-level rather than point-unique, so this descriptor does not
    use attention argmax.  Instead it decomposes the centered mutual operator
    and appends the aligned singular coordinates as a separate feature branch.
    """

    src_height, src_width = src_native_map.shape[-2:]
    trg_height, trg_width = trg_native_map.shape[-2:]
    src_native = src_native_map[0].permute(1, 2, 0).reshape(-1, src_native_map.shape[1]).float()
    trg_native = trg_native_map[0].permute(1, 2, 0).reshape(-1, trg_native_map.shape[1]).float()

    cond_ab, _mass_a = _conditional_cross_distribution(attention["p_ab"])
    cond_ba, _mass_b = _conditional_cross_distribution(attention["p_ba"])
    mutual = torch.sqrt((cond_ab.float() * cond_ba.float().t()).clamp_min(0.0))
    mutual = torch.nan_to_num(mutual, nan=0.0, posinf=0.0, neginf=0.0)

    # Remove the rank-1/common part so the spectral branch represents deviations
    # that distinguish points inside a shared semantic region.
    mutual = mutual - mutual.mean(dim=1, keepdim=True)
    mutual = mutual - mutual.mean(dim=0, keepdim=True)
    mutual = mutual + mutual.mean()

    min_tokens = min(mutual.shape)
    if min_tokens <= 1 or mutual.square().mean() <= 1e-12:
        spectral_src = torch.zeros((src_native.shape[0], 1), device=src_native.device, dtype=torch.float32)
        spectral_trg = torch.zeros((trg_native.shape[0], 1), device=trg_native.device, dtype=torch.float32)
    else:
        rank = min(32, max(4, int(round(float(min_tokens) ** 0.5))), min_tokens - 1)
        try:
            u, s, vh = torch.linalg.svd(mutual, full_matrices=False)
        except RuntimeError:
            u, s, vh = torch.linalg.svd(mutual.cpu(), full_matrices=False)
            u = u.to(src_native.device)
            s = s.to(src_native.device)
            vh = vh.to(src_native.device)
        s = torch.nan_to_num(s[:rank].float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        scale = torch.sqrt(s).reshape(1, -1)
        spectral_src = u[:, :rank].float() * scale
        spectral_trg = vh[:rank, :].t().float() * scale

    src_descriptor = F.normalize(
        torch.cat(
            (
                F.normalize(src_native, dim=1, eps=1e-12),
                F.normalize(spectral_src, dim=1, eps=1e-12),
            ),
            dim=1,
        ),
        dim=1,
        eps=1e-12,
    )
    trg_descriptor = F.normalize(
        torch.cat(
            (
                F.normalize(trg_native, dim=1, eps=1e-12),
                F.normalize(spectral_trg, dim=1, eps=1e-12),
            ),
            dim=1,
        ),
        dim=1,
        eps=1e-12,
    )
    return (
        src_descriptor.t().reshape(1, src_descriptor.shape[1], src_height, src_width).contiguous(),
        trg_descriptor.t().reshape(1, trg_descriptor.shape[1], trg_height, trg_width).contiguous(),
    )


def _shifted_row_gather(
    rows: torch.Tensor,
    height: int,
    width: int,
    dx: int,
    dy: int,
) -> torch.Tensor:
    """For each center cell, gather the row at center+(dx,dy)."""

    rows_2d = rows.reshape(int(height), int(width), rows.shape[1])
    shifted = torch.zeros_like(rows_2d)
    x0 = max(0, -int(dx))
    x1 = min(int(width), int(width) - int(dx))
    y0 = max(0, -int(dy))
    y1 = min(int(height), int(height) - int(dy))
    if x0 < x1 and y0 < y1:
        shifted[y0:y1, x0:x1] = rows_2d[y0 + int(dy):y1 + int(dy), x0 + int(dx):x1 + int(dx)]
    return shifted.reshape(int(height) * int(width), rows.shape[1])


def _shifted_cell_basis(
    height: int,
    width: int,
    dx: int,
    dy: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """For each center cell, one-hot encode center+(dx,dy)."""

    count = int(height) * int(width)
    basis = torch.zeros((count, count), device=device, dtype=torch.float32)
    xs = torch.arange(int(width), device=device)
    ys = torch.arange(int(height), device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    nx = grid_x + int(dx)
    ny = grid_y + int(dy)
    valid = (nx >= 0) & (nx < int(width)) & (ny >= 0) & (ny < int(height))
    centers = (grid_y[valid] * int(width) + grid_x[valid]).long()
    neighbors = (ny[valid] * int(width) + nx[valid]).long()
    if centers.numel() > 0:
        basis[centers, neighbors] = 1.0
    return basis


def _local_transport_lift_descriptors(
    src_native_map: torch.Tensor,
    trg_native_map: torch.Tensor,
    attention: dict[str, torch.Tensor],
    *,
    radius: int = 1,
    include_native: bool = True,
    include_outgoing: bool = True,
    include_incoming: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lift local attention transport into explicit source/target descriptors.

    Pairwise flow diagnostics showed that the useful signal is lost when it is
    compressed into a small separable signature.  This descriptor keeps the
    local transport witness as a high-dimensional feature map: source channels
    store shifted attention rows, while target channels are shifted spatial
    bases.  Their dot product measures whether a target candidate is supported
    by aligned cross-image transport, yet prediction still uses cosine + NN.
    """

    src_height, src_width = src_native_map.shape[-2:]
    trg_height, trg_width = trg_native_map.shape[-2:]
    src_count = int(src_height) * int(src_width)
    trg_count = int(trg_height) * int(trg_width)
    src_native = src_native_map[0].permute(1, 2, 0).reshape(src_count, src_native_map.shape[1]).float()
    trg_native = trg_native_map[0].permute(1, 2, 0).reshape(trg_count, trg_native_map.shape[1]).float()

    mutual = torch.sqrt((attention["p_ab"].float() * attention["p_ba"].float().t()).clamp_min(0.0))
    mutual = torch.nan_to_num(mutual, nan=0.0, posinf=0.0, neginf=0.0)
    outgoing = mutual / mutual.max(dim=1, keepdim=True).values.clamp_min(1e-12)
    incoming = mutual / mutual.max(dim=0, keepdim=True).values.clamp_min(1e-12)
    outgoing = torch.nan_to_num(outgoing, nan=0.0, posinf=0.0, neginf=0.0)
    incoming = torch.nan_to_num(incoming, nan=0.0, posinf=0.0, neginf=0.0)

    src_out_parts: list[torch.Tensor] = []
    trg_out_parts: list[torch.Tensor] = []
    src_in_parts: list[torch.Tensor] = []
    trg_in_parts: list[torch.Tensor] = []
    for dx, dy in _cell_offsets(radius):
        trg_dx = int(round(float(dx) * float(trg_width) / float(max(1, src_width))))
        trg_dy = int(round(float(dy) * float(trg_height) / float(max(1, src_height))))
        src_out_parts.append(_shifted_row_gather(outgoing, src_height, src_width, dx, dy))
        trg_out_parts.append(
            _shifted_cell_basis(
                trg_height,
                trg_width,
                trg_dx,
                trg_dy,
                device=mutual.device,
            )
        )
        src_in_parts.append(
            _shifted_cell_basis(
                src_height,
                src_width,
                dx,
                dy,
                device=mutual.device,
            )
        )
        incoming_by_target = incoming.t().contiguous()
        trg_in_parts.append(_shifted_row_gather(incoming_by_target, trg_height, trg_width, trg_dx, trg_dy))

    src_out = torch.cat(src_out_parts, dim=1) if src_out_parts else torch.zeros((src_count, 1), device=mutual.device)
    trg_out = torch.cat(trg_out_parts, dim=1) if trg_out_parts else torch.zeros((trg_count, 1), device=mutual.device)
    src_in = torch.cat(src_in_parts, dim=1) if src_in_parts else torch.zeros((src_count, 1), device=mutual.device)
    trg_in = torch.cat(trg_in_parts, dim=1) if trg_in_parts else torch.zeros((trg_count, 1), device=mutual.device)

    src_branches = []
    trg_branches = []
    if include_native:
        src_branches.append(F.normalize(src_native, dim=1, eps=1e-12))
        trg_branches.append(F.normalize(trg_native, dim=1, eps=1e-12))
    if include_outgoing:
        src_branches.append(F.normalize(src_out, dim=1, eps=1e-12))
        trg_branches.append(F.normalize(trg_out, dim=1, eps=1e-12))
    if include_incoming:
        src_branches.append(F.normalize(src_in, dim=1, eps=1e-12))
        trg_branches.append(F.normalize(trg_in, dim=1, eps=1e-12))
    if not src_branches or not trg_branches:
        raise ValueError("transport lift descriptor needs at least one branch")
    src_descriptor = F.normalize(torch.cat(src_branches, dim=1), dim=1, eps=1e-12)
    trg_descriptor = F.normalize(torch.cat(trg_branches, dim=1), dim=1, eps=1e-12)
    return (
        src_descriptor.t().reshape(1, src_descriptor.shape[1], src_height, src_width).contiguous(),
        trg_descriptor.t().reshape(1, trg_descriptor.shape[1], trg_height, trg_width).contiguous(),
    )


def _row_topk_weights(rows: torch.Tensor, topk: int) -> torch.Tensor:
    rows = torch.nan_to_num(rows.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    if rows.numel() == 0:
        return rows
    k = min(max(1, int(topk)), int(rows.shape[1]))
    top = torch.topk(rows, k=k, dim=1, sorted=False)
    masked = torch.zeros_like(rows)
    masked.scatter_(1, top.indices, top.values)
    return masked / masked.sum(dim=1, keepdim=True).clamp_min(1e-12)


def _basin_contrast_rows(
    query_native: torch.Tensor,
    key_native: torch.Tensor,
    kernel_rows: torch.Tensor,
    *,
    basin_topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Raw transport rows plus within-basin native residual contrast."""

    query_unit = F.normalize(
        torch.nan_to_num(query_native.float(), nan=0.0, posinf=0.0, neginf=0.0),
        dim=1,
        eps=1e-12,
    )
    key_unit = F.normalize(
        torch.nan_to_num(key_native.float(), nan=0.0, posinf=0.0, neginf=0.0),
        dim=1,
        eps=1e-12,
    )
    raw = torch.nan_to_num(kernel_rows.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    raw = raw / raw.max(dim=1, keepdim=True).values.clamp_min(1e-12)
    raw = torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

    weights = _row_topk_weights(kernel_rows, basin_topk)
    common = weights.to(key_unit.device, dtype=key_unit.dtype) @ key_unit
    native_similarity = query_unit @ key_unit.t()
    common_similarity = (query_unit * common).sum(dim=1, keepdim=True)
    centered = native_similarity - common_similarity
    contrast = centered * weights.to(centered.device, dtype=centered.dtype)
    contrast = torch.nan_to_num(contrast, nan=0.0, posinf=0.0, neginf=0.0)
    return raw, contrast


def _basin_contrastive_identity_descriptors(
    src_native_map: torch.Tensor,
    trg_native_map: torch.Tensor,
    attention: dict[str, torch.Tensor],
    *,
    basin_topk: int = 20,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pair-conditioned identity descriptor from raw attention basins.

    The descriptor keeps native identity and raw mutual-attention transport as
    explicit feature branches.  It adds a source-conditioned residual branch
    formed by removing, inside each raw attention basin, the common target
    component seen by that source token.  A symmetric target-to-source residual
    branch is included so the feature remains pair-conditioned on both images.
    Prediction still uses the unchanged cosine + nearest-neighbor path.
    """

    src_height, src_width = src_native_map.shape[-2:]
    trg_height, trg_width = trg_native_map.shape[-2:]
    src_count = int(src_height) * int(src_width)
    trg_count = int(trg_height) * int(trg_width)
    src_native = src_native_map[0].permute(1, 2, 0).reshape(src_count, src_native_map.shape[1]).float()
    trg_native = trg_native_map[0].permute(1, 2, 0).reshape(trg_count, trg_native_map.shape[1]).float()

    mutual = torch.sqrt((attention["p_ab"].float() * attention["p_ba"].float().t()).clamp_min(0.0))
    mutual = torch.nan_to_num(mutual, nan=0.0, posinf=0.0, neginf=0.0)
    raw_ab, contrast_ab = _basin_contrast_rows(
        src_native,
        trg_native,
        mutual,
        basin_topk=basin_topk,
    )
    raw_ba, contrast_ba = _basin_contrast_rows(
        trg_native,
        src_native,
        mutual.t().contiguous(),
        basin_topk=basin_topk,
    )

    src_basis = torch.eye(src_count, device=mutual.device, dtype=torch.float32)
    trg_basis = torch.eye(trg_count, device=mutual.device, dtype=torch.float32)
    src_descriptor = F.normalize(
        torch.cat(
            (
                F.normalize(src_native, dim=1, eps=1e-12),
                F.normalize(raw_ab, dim=1, eps=1e-12),
                _weighted_unit_branch(contrast_ab, raw_ab),
                F.normalize(src_basis, dim=1, eps=1e-12),
            ),
            dim=1,
        ),
        dim=1,
        eps=1e-12,
    )
    trg_descriptor = F.normalize(
        torch.cat(
            (
                F.normalize(trg_native, dim=1, eps=1e-12),
                F.normalize(trg_basis, dim=1, eps=1e-12),
                F.normalize(trg_basis, dim=1, eps=1e-12),
                _weighted_unit_branch(contrast_ba, raw_ba),
            ),
            dim=1,
        ),
        dim=1,
        eps=1e-12,
    )
    return (
        src_descriptor.t().reshape(1, src_descriptor.shape[1], src_height, src_width).contiguous(),
        trg_descriptor.t().reshape(1, trg_descriptor.shape[1], trg_height, trg_width).contiguous(),
    )


def _mutual_nearest_attention_anchors(
    mutual: torch.Tensor,
    *,
    max_anchors: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return high-confidence reciprocal attention anchors."""

    mutual = torch.nan_to_num(mutual.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    if mutual.ndim != 2 or mutual.numel() == 0:
        empty = torch.empty(0, device=mutual.device, dtype=torch.long)
        return empty, empty, torch.empty(0, device=mutual.device, dtype=torch.float32)
    row_values, row_indices = mutual.max(dim=1)
    _col_values, col_indices = mutual.max(dim=0)
    src_indices = torch.arange(mutual.shape[0], device=mutual.device, dtype=torch.long)
    reciprocal = col_indices[row_indices] == src_indices
    anchor_src = src_indices[reciprocal]
    anchor_trg = row_indices[reciprocal]
    anchor_scores = row_values[reciprocal]
    if anchor_scores.numel() == 0:
        empty = torch.empty(0, device=mutual.device, dtype=torch.long)
        return empty, empty, torch.empty(0, device=mutual.device, dtype=torch.float32)
    keep = min(max(1, int(max_anchors)), int(anchor_scores.numel()))
    order = torch.argsort(anchor_scores, descending=True)[:keep]
    return anchor_src[order], anchor_trg[order], anchor_scores[order]


def _attention_guided_isometry_descriptors(
    src_native_map: torch.Tensor,
    trg_native_map: torch.Tensor,
    attention: dict[str, torch.Tensor],
    *,
    max_anchors: int = 128,
    rank: int = 32,
    min_anchors: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align native descriptors by an attention-guided low-rank isometry.

    Cross-attention gives high-recall transport but direct value updates drift
    far from the native DiTF descriptor manifold.  This feature-side operator
    uses only reciprocal high-confidence attention anchors to estimate a
    pair-specific orthogonal transform.  It rotates source descriptors inside
    the anchor subspace and keeps the orthogonal complement unchanged, so the
    source descriptor geometry is preserved instead of overwritten.
    """

    src_height, src_width = src_native_map.shape[-2:]
    trg_height, trg_width = trg_native_map.shape[-2:]
    src_count = int(src_height) * int(src_width)
    trg_count = int(trg_height) * int(trg_width)
    src_native = src_native_map[0].permute(1, 2, 0).reshape(src_count, src_native_map.shape[1]).float()
    trg_native = trg_native_map[0].permute(1, 2, 0).reshape(trg_count, trg_native_map.shape[1]).float()
    src_unit = F.normalize(torch.nan_to_num(src_native, nan=0.0, posinf=0.0, neginf=0.0), dim=1, eps=1e-12)
    trg_unit = F.normalize(torch.nan_to_num(trg_native, nan=0.0, posinf=0.0, neginf=0.0), dim=1, eps=1e-12)

    mutual = torch.sqrt((attention["p_ab"].float() * attention["p_ba"].float().t()).clamp_min(0.0))
    anchor_src, anchor_trg, anchor_scores = _mutual_nearest_attention_anchors(
        mutual,
        max_anchors=max_anchors,
    )
    if int(anchor_src.numel()) < int(min_anchors):
        return (
            src_unit.t().reshape(1, src_unit.shape[1], src_height, src_width).contiguous(),
            trg_unit.t().reshape(1, trg_unit.shape[1], trg_height, trg_width).contiguous(),
        )

    anchor_x = src_unit[anchor_src.to(src_unit.device)]
    anchor_y = trg_unit[anchor_trg.to(trg_unit.device)]
    weights = anchor_scores.to(anchor_x.device, dtype=anchor_x.dtype).clamp_min(0.0)
    weights = weights / weights.mean().clamp_min(1e-12)
    weighted_stack = torch.cat(
        (
            anchor_x * weights.sqrt().reshape(-1, 1),
            anchor_y * weights.sqrt().reshape(-1, 1),
        ),
        dim=0,
    )
    subspace_rank = min(int(rank), int(anchor_src.numel()), int(weighted_stack.shape[0]) - 1, int(weighted_stack.shape[1]))
    if subspace_rank <= 0 or weighted_stack.square().mean() <= 1e-12:
        return (
            src_unit.t().reshape(1, src_unit.shape[1], src_height, src_width).contiguous(),
            trg_unit.t().reshape(1, trg_unit.shape[1], trg_height, trg_width).contiguous(),
        )
    try:
        _u, _s, vh = torch.linalg.svd(weighted_stack, full_matrices=False)
    except RuntimeError:
        _u, _s, vh = torch.linalg.svd(weighted_stack.cpu(), full_matrices=False)
        vh = vh.to(src_unit.device)
    basis = vh[:subspace_rank].t().contiguous()
    x_sub = anchor_x @ basis
    y_sub = anchor_y @ basis
    cross_cov = x_sub.t() @ (y_sub * weights.reshape(-1, 1))
    try:
        u, _s, vh_small = torch.linalg.svd(cross_cov, full_matrices=False)
    except RuntimeError:
        u, _s, vh_small = torch.linalg.svd(cross_cov.cpu(), full_matrices=False)
        u = u.to(src_unit.device)
        vh_small = vh_small.to(src_unit.device)
    rotation = u @ vh_small
    src_projection = src_unit @ basis
    src_aligned = src_unit + ((src_projection @ rotation) - src_projection) @ basis.t()
    src_aligned = F.normalize(torch.nan_to_num(src_aligned, nan=0.0, posinf=0.0, neginf=0.0), dim=1, eps=1e-12)
    return (
        src_aligned.t().reshape(1, src_aligned.shape[1], src_height, src_width).contiguous(),
        trg_unit.t().reshape(1, trg_unit.shape[1], trg_height, trg_width).contiguous(),
    )


def _weighted_unit_branch(branch: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Normalize a branch and scale it by its self-measured energy ratio."""

    branch_norm = branch.norm(dim=1, keepdim=True)
    reference_norm = reference.norm(dim=1, keepdim=True).clamp_min(1e-12)
    weight = torch.sqrt((branch_norm / (reference_norm + branch_norm).clamp_min(1e-12)).clamp(0.0, 1.0))
    return F.normalize(branch, dim=1, eps=1e-12) * weight


def _fjsar_candidate_oracle_counts(
    *,
    src_features: torch.Tensor,
    trg_features: torch.Tensor,
    points: Sequence[Sequence[float]],
    target_points: Sequence[Sequence[float]] | None,
    source_size: Sequence[int],
    target_size: Sequence[int],
    pck_threshold: float | None,
    oracle_topk: Sequence[int],
    src_state: FluxReplayState,
    trg_state: FluxReplayState,
    src_native_prepared: torch.Tensor,
    trg_native_prepared: torch.Tensor,
    src_joint_prepared: torch.Tensor,
    trg_joint_prepared: torch.Tensor,
    attention: dict[str, torch.Tensor],
    descriptor_modes: Sequence[str] = (),
) -> dict[str, int]:
    """GT-only FJSAR candidate coverage audit; not used for prediction."""

    if target_points is None or pck_threshold is None or not points:
        return {}
    topks = tuple(sorted({max(1, int(k)) for k in oracle_topk}))
    if not topks:
        return {}

    native_src = _token_matrix_to_map(
        F.normalize(src_native_prepared, dim=1, eps=1e-12),
        src_state.image_height,
        src_state.image_width,
    )
    native_trg = _token_matrix_to_map(
        F.normalize(trg_native_prepared, dim=1, eps=1e-12),
        trg_state.image_height,
        trg_state.image_width,
    )
    max_k = max(topks)
    branch_indices = {
        "native": _chunked_descriptor_topk_indices(
            native_src,
            native_trg,
            points,
            source_size,
            target_size,
            topk=max_k,
        ),
    }
    descriptor_modes = tuple(dict.fromkeys(str(mode) for mode in descriptor_modes))
    for descriptor_mode in descriptor_modes:
        if descriptor_mode == "attention":
            continue
        if descriptor_mode == "attention_signature":
            desc_src, desc_trg = _attention_signature_descriptors(src_features, trg_features, attention)
        elif descriptor_mode == "part_sharpen":
            desc_src, desc_trg = _part_common_sharpen_descriptors(src_features, trg_features, attention)
        elif descriptor_mode == "orthogonal_context":
            desc_src, desc_trg = _orthogonal_context_descriptors(
                src_features,
                trg_features,
                src_joint_prepared,
                trg_joint_prepared,
                src_native_prepared,
                trg_native_prepared,
                attention,
            )
        elif descriptor_mode == "spectral_identity":
            desc_src, desc_trg = _spectral_attention_identity_descriptors(
                src_features,
                trg_features,
                attention,
            )
        elif descriptor_mode == "filtered_spectral_kernel":
            desc_src, desc_trg, _spectral_diagnostics = (
                filtered_spectral_kernel_feature_maps(
                    src_features,
                    trg_features,
                    attention,
                    src_state,
                    trg_state,
                )
            )
        elif descriptor_mode == "transport_lift":
            desc_src, desc_trg = _local_transport_lift_descriptors(
                src_features,
                trg_features,
                attention,
            )
        elif descriptor_mode == "basin_contrastive_identity":
            desc_src, desc_trg = _basin_contrastive_identity_descriptors(
                src_features,
                trg_features,
                attention,
            )
        elif descriptor_mode == "attention_isometry":
            desc_src, desc_trg = _attention_guided_isometry_descriptors(
                src_features,
                trg_features,
                attention,
            )
        elif descriptor_mode == "geometry_consistent_attention":
            desc_src = _token_matrix_to_map(
                F.normalize(src_joint_prepared, dim=1, eps=1e-12),
                src_state.image_height,
                src_state.image_width,
            )
            desc_trg = _token_matrix_to_map(
                F.normalize(trg_joint_prepared, dim=1, eps=1e-12),
                trg_state.image_height,
                trg_state.image_width,
            )
        elif descriptor_mode == "identity_preserving_attention":
            desc_src = _token_matrix_to_map(
                F.normalize(src_joint_prepared, dim=1, eps=1e-12),
                src_state.image_height,
                src_state.image_width,
            )
            desc_trg = _token_matrix_to_map(
                F.normalize(trg_joint_prepared, dim=1, eps=1e-12),
                trg_state.image_height,
                trg_state.image_width,
            )
        elif descriptor_mode == "balanced_transport_attention":
            desc_src = _token_matrix_to_map(
                F.normalize(src_joint_prepared, dim=1, eps=1e-12),
                src_state.image_height,
                src_state.image_width,
            )
            desc_trg = _token_matrix_to_map(
                F.normalize(trg_joint_prepared, dim=1, eps=1e-12),
                trg_state.image_height,
                trg_state.image_width,
            )
        elif descriptor_mode == "qk_identity_attention":
            desc_src = _token_matrix_to_map(
                F.normalize(src_joint_prepared, dim=1, eps=1e-12),
                src_state.image_height,
                src_state.image_width,
            )
            desc_trg = _token_matrix_to_map(
                F.normalize(trg_joint_prepared, dim=1, eps=1e-12),
                trg_state.image_height,
                trg_state.image_width,
            )
        elif descriptor_mode == "native_preserving_topology_rescue":
            ranked_indices, _audits, _summary = _native_preserving_topology_rescue_rankings(
                src_features,
                trg_features,
                attention,
                points,
                source_size,
                target_size,
                src_state,
                trg_state,
                candidate_topk=max_k,
                target_points=target_points,
                pck_threshold=pck_threshold,
            )
            branch_indices[descriptor_mode] = ranked_indices
            continue
        elif descriptor_mode == "attention_basin_native_refine":
            branch_indices[descriptor_mode] = _attention_basin_native_refine_rankings(
                src_features,
                trg_features,
                attention,
                points,
                source_size,
                target_size,
                src_state,
                trg_state,
                candidate_topk=max_k,
            )
            continue
        elif descriptor_mode == "candidate_conditioned_verification":
            ranked_indices, _audits, _summary = _candidate_conditioned_verification_rankings(
                src_features,
                trg_features,
                attention,
                points,
                source_size,
                target_size,
                src_state,
                trg_state,
                candidate_topk=max_k,
                target_points=target_points,
                pck_threshold=pck_threshold,
            )
            branch_indices[descriptor_mode] = ranked_indices
            continue
        elif descriptor_mode == "candidate_local_transport_verification":
            ranked_indices, _audits, _summary = _candidate_local_transport_verification_rankings(
                src_features,
                trg_features,
                attention,
                points,
                source_size,
                target_size,
                src_state,
                trg_state,
                candidate_topk=max_k,
                target_points=target_points,
                pck_threshold=pck_threshold,
            )
            branch_indices[descriptor_mode] = ranked_indices
            continue
        elif descriptor_mode == "candidate_graph_consensus_verification":
            ranked_indices, _audits, _summary = _candidate_graph_consensus_verification_rankings(
                src_features,
                trg_features,
                attention,
                points,
                source_size,
                target_size,
                src_state,
                trg_state,
                candidate_topk=max_k,
                target_points=target_points,
                pck_threshold=pck_threshold,
            )
            branch_indices[descriptor_mode] = ranked_indices
            continue
        else:
            raise ValueError(f"unsupported oracle descriptor mode: {descriptor_mode}")
        branch_indices[descriptor_mode] = _chunked_descriptor_topk_indices(
            desc_src,
            desc_trg,
            points,
            source_size,
            target_size,
            topk=max_k,
        )

    src_cell_indices = _native_cell_indices_for_points(
        points,
        source_size,
        src_state.image_height,
        src_state.image_width,
        attention["p_ab"].device,
    )
    attention_mutual = torch.sqrt((attention["p_ab"].float() * attention["p_ba"].float().t()).clamp_min(0.0))
    attention_cells = torch.topk(
        attention_mutual[src_cell_indices],
        k=min(max_k, attention_mutual.shape[1]),
        dim=1,
        sorted=True,
    ).indices
    target_h, target_w = int(target_size[0]), int(target_size[1])
    cell_x = (attention_cells % trg_state.image_width).float()
    cell_y = torch.div(attention_cells, trg_state.image_width, rounding_mode="floor").float()
    pixel_x = torch.round((cell_x + 0.5) * float(target_w) / float(trg_state.image_width) - 0.5).long()
    pixel_y = torch.round((cell_y + 0.5) * float(target_h) / float(trg_state.image_height) - 0.5).long()
    pixel_x.clamp_(0, target_w - 1)
    pixel_y.clamp_(0, target_h - 1)
    branch_indices["attention"] = pixel_y.to(src_features.device) * target_w + pixel_x.to(src_features.device)

    counts: dict[str, int] = {"fjsar_oracle_total": len(points)}
    for name, indices in branch_indices.items():
        for k, count in _topk_hit_counts(
            indices,
            target_points,
            float(pck_threshold),
            target_size,
            topks,
        ).items():
            counts[f"fjsar_oracle_owner_{name}@{k}"] = count

    return counts


def _token_matrix_to_map(tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if tokens.ndim != 2 or tokens.shape[0] != int(height) * int(width):
        raise ValueError("token matrix does not match the requested spatial grid")
    tokens = torch.nan_to_num(tokens.float(), nan=0.0, posinf=0.0, neginf=0.0)
    return tokens.t().reshape(1, tokens.shape[1], int(height), int(width)).contiguous()


def _layerwise_routing_identity_rankings(
    routing_src: torch.Tensor,
    routing_trg: torch.Tensor,
    identity_maps: dict[str, tuple[torch.Tensor, torch.Tensor]],
    attention: dict[str, torch.Tensor],
    points: Sequence[Sequence[float]],
    source_size: Sequence[int],
    target_size: Sequence[int],
    src_state: FluxReplayState,
    trg_state: FluxReplayState,
    *,
    candidate_topk: int,
    target_points: Sequence[Sequence[float]] | None,
    pck_threshold: float | None,
    identity_only_primary: bool = False,
) -> tuple[torch.Tensor, list[dict[str, Any]], dict[str, Any]]:
    """Fuse deep cross-image routing with source-only layer identity features.

    The joint feature is used to define the exact mutual-attention basin.  It
    is never the sole identity descriptor: each official source-only block
    feature is normalized as an independent identity carrier and concatenated
    with the routing feature at equal branch norm.  No learned weight,
    native fallback, or annotation enters the ranking.
    """

    points = list(points)
    if not points:
        empty = torch.empty((0, 0), device=routing_src.device, dtype=torch.long)
        return empty, [], {"points": 0}
    if not identity_maps:
        raise ValueError("layerwise routing identity requires at least one layer map")
    if routing_src.ndim != 4 or routing_trg.ndim != 4:
        raise ValueError("routing features must have shape [1,C,H,W]")
    if routing_src.shape[0] != 1 or routing_trg.shape[0] != 1:
        raise ValueError("routing features must have batch size one")
    if routing_src.shape[1] != routing_trg.shape[1]:
        raise ValueError("routing source/target channels must agree")

    device = routing_src.device
    mutual = torch.sqrt(
        (attention["p_ab"].float() * attention["p_ba"].float().t()).clamp_min(0.0)
    )
    mutual = torch.nan_to_num(mutual, nan=0.0, posinf=0.0, neginf=0.0)
    src_cells = _native_cell_indices_for_points(
        points,
        source_size,
        src_state.image_height,
        src_state.image_width,
        device,
    )
    candidate_count = min(max(1, int(candidate_topk)), int(mutual.shape[1]))
    _attention_scores, candidate_cells = torch.topk(
        mutual[src_cells], k=candidate_count, dim=1, sorted=True
    )
    proposal_pixels = _cell_topk_to_pixel_indices(
        candidate_cells, target_size, trg_state
    ).to(device)

    def _unit_branch(value: torch.Tensor) -> torch.Tensor:
        value = torch.nan_to_num(value.float(), nan=0.0, posinf=0.0, neginf=0.0)
        return F.normalize(value, dim=1, eps=1e-12)

    route_src = _unit_branch(routing_src)
    route_trg = _unit_branch(routing_trg)
    branch_maps: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name, (src_map, trg_map) in identity_maps.items():
        if src_map.ndim != 4 or trg_map.ndim != 4:
            raise ValueError(f"identity map {name} must have shape [1,C,H,W]")
        if src_map.shape[0] != 1 or trg_map.shape[0] != 1:
            raise ValueError(f"identity map {name} must have batch size one")
        if src_map.shape[1] != trg_map.shape[1]:
            raise ValueError(f"identity map {name} source/target channels must agree")
        if src_map.shape[-2:] != route_src.shape[-2:] or trg_map.shape[-2:] != route_trg.shape[-2:]:
            raise ValueError(f"identity map {name} grid does not match routing grid")
        branch_maps[str(name)] = (_unit_branch(src_map), _unit_branch(trg_map))

    layer_names = list(branch_maps)
    fused_maps: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name, (src_identity, trg_identity) in branch_maps.items():
        fused_maps[f"routing_plus_{name}"] = (
            F.normalize(torch.cat((route_src, src_identity), dim=1), dim=1, eps=1e-12),
            F.normalize(torch.cat((route_trg, trg_identity), dim=1), dim=1, eps=1e-12),
        )
    all_src = [route_src, *(branch_maps[name][0] for name in layer_names)]
    all_trg = [route_trg, *(branch_maps[name][1] for name in layer_names)]
    fused_maps["routing_plus_all_layers"] = (
        F.normalize(torch.cat(all_src, dim=1), dim=1, eps=1e-12),
        F.normalize(torch.cat(all_trg, dim=1), dim=1, eps=1e-12),
    )
    if identity_only_primary:
        fused_maps["all_identity_layers"] = (
            F.normalize(
                torch.cat([branch_maps[name][0] for name in layer_names], dim=1),
                dim=1,
                eps=1e-12,
            ),
            F.normalize(
                torch.cat([branch_maps[name][1] for name in layer_names], dim=1),
                dim=1,
                eps=1e-12,
            ),
        )

    score_maps: dict[str, tuple[torch.Tensor, torch.Tensor]] = {
        **branch_maps,
        **fused_maps,
    }
    ranked_by_name: dict[str, torch.Tensor] = {}
    scores_by_name: dict[str, torch.Tensor] = {}
    for name, (src_map, trg_map) in score_maps.items():
        scores, indices = _chunked_descriptor_topk_scores_indices(
            src_map,
            trg_map,
            points,
            source_size,
            target_size,
            topk=candidate_count,
            candidate_indices=proposal_pixels,
        )
        scores_by_name[name] = scores
        ranked_by_name[name] = indices

    primary_name = (
        "all_identity_layers" if identity_only_primary else "routing_plus_all_layers"
    )
    ranked_pixels = ranked_by_name[primary_name]
    target_w = int(target_size[1])
    records: list[dict[str, Any]] = []
    for row in range(len(points)):
        target = target_points[row] if target_points is not None else None
        proposal = proposal_pixels[row].detach().cpu().tolist()
        hits = [
            bool(
                _point_hit(
                    [int(pixel % target_w), int(pixel // target_w)],
                    target,
                    pck_threshold,
                )
            )
            if target is not None and pck_threshold is not None
            else False
            for pixel in proposal
        ]
        branch_audits: dict[str, Any] = {}
        for name in score_maps:
            branch_scores = scores_by_name[name][row]
            branch_pixels = ranked_by_name[name][row]
            order_hits = []
            for pixel in branch_pixels.detach().cpu().tolist():
                order_hits.append(
                    bool(
                        _point_hit(
                            [int(pixel % target_w), int(pixel // target_w)],
                            target,
                            pck_threshold,
                        )
                    )
                    if target is not None and pck_threshold is not None
                    else False
                )
            rank = next(
                (index + 1 for index, hit in enumerate(order_hits) if hit),
                None,
            )
            branch_audits[name] = {
                "top1_pixel": int(branch_pixels[0].detach().cpu()),
                "top1_pck_hit": bool(order_hits[0]) if order_hits else False,
                "gt_rank": rank,
                "top20_pck_hit": bool(any(order_hits[:20])),
                "top1_score": float(branch_scores[0].detach().cpu()),
                "gt_in_attention_pool": bool(any(hits)),
            }
        records.append(
            {
                "source_cell": int(src_cells[row].detach().cpu()),
                "candidate_count": int(candidate_count),
                "attention_top1_pixel": int(proposal[0]),
                "attention_top1_pck_hit": bool(hits[0]) if hits else False,
                "attention_top20_pck_hit": bool(any(hits)),
                "candidate_missing_gt": bool(not any(hits)),
                "branches": branch_audits,
                "primary_branch": primary_name,
            }
        )

    summary: dict[str, Any] = {
        "hypothesis": (
            {
                "name": "Pre-Single-Stream Identity Routing",
                "routing_branch": "deep_joint_cross_attention_candidate_pool_only",
                "identity_branch": "double_stream_image_features_before_single_stream_merge",
                "fusion": "equal_norm_identity_concatenation",
                "attention_used_as_identity_score": False,
                "candidate_source": "exact_mutual_cross_attention_topk_only",
                "train_free": True,
                "native_fallback": False,
                "gt_used_for_inference": False,
            }
            if identity_only_primary
            else {
                "name": "Layerwise Routing Identity",
                "routing_branch": "deep_joint_cross_attention",
                "identity_branch": "source_only_official_single_block_features",
                "fusion": "equal_norm_concatenation",
                "candidate_source": "exact_mutual_cross_attention_topk_only",
                "train_free": True,
                "native_fallback": False,
                "gt_used_for_inference": False,
            }
        ),
        "layer_names": layer_names,
        "primary_branch": primary_name,
        "candidate_pool_mean": float(candidate_count),
        "candidate_missing_gt_count": int(sum(row["candidate_missing_gt"] for row in records)),
        "branches": {},
    }
    for name in score_maps:
        audits = [row["branches"][name] for row in records]
        ranks = [row["gt_rank"] for row in audits if row["gt_rank"] is not None]
        summary["branches"][name] = {
            "top1_pck_hits": int(sum(row["top1_pck_hit"] for row in audits)),
            "top20_pck_hits": int(sum(row["top20_pck_hit"] for row in audits)),
            "gt_rank_median": float(np.median(ranks)) if ranks else None,
            "gt_rank_mean": float(np.mean(ranks)) if ranks else None,
            "gt_rank_count": int(len(ranks)),
        }
    return ranked_pixels, records, summary


def flux_fjsar_attention_candidates(
    *,
    src_replay_state: dict[str, Any],
    trg_replay_state: dict[str, Any],
    blocks: Sequence[Any],
    points: Sequence[Sequence[float]],
    source_size: Sequence[int],
    target_size: Sequence[int],
    candidate_topk: int = 20,
    interaction_mode: str = "exact",
    use_coordinate_bias: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Return exact mutual-attention candidate pixels without identity scoring.

    This helper intentionally exposes only the routing contract used by the
    DINOv2 experiment: block28 cross-attention freezes the candidate pool, and
    every candidate identity score is computed outside this function.
    """
    src_state = FluxReplayState.from_dict(src_replay_state)
    trg_state = FluxReplayState.from_dict(trg_replay_state)
    if src_state.global_block_index != trg_state.global_block_index:
        raise ValueError("source and target replay caches use different blocks")
    if interaction_mode == "exact":
        _src_joint, _trg_joint, attention = run_flux_joint_stack(
            blocks,
            src_state,
            trg_state,
            mode="exact",
            use_coordinate_bias=use_coordinate_bias,
        )
    elif interaction_mode == "calibrated":
        _src_joint, _trg_joint, attention = run_flux_joint_stack(
            blocks,
            src_state,
            trg_state,
            mode="calibrated",
            use_coordinate_bias=use_coordinate_bias,
        )
    else:
        raise ValueError(f"unsupported attention candidate mode: {interaction_mode}")
    device = attention["p_ab"].device
    src_cells = _native_cell_indices_for_points(
        points,
        source_size,
        src_state.image_height,
        src_state.image_width,
        device,
    )
    mutual = torch.sqrt(
        (attention["p_ab"].float() * attention["p_ba"].float().t()).clamp_min(0.0)
    )
    mutual = torch.nan_to_num(mutual, nan=0.0, posinf=0.0, neginf=0.0)
    candidate_count = min(max(1, int(candidate_topk)), int(mutual.shape[1]))
    scores, candidate_cells = torch.topk(
        mutual[src_cells], k=candidate_count, dim=1, sorted=True
    )
    candidate_pixels = _cell_topk_to_pixel_indices(
        candidate_cells, target_size, trg_state
    ).to(device)
    return candidate_pixels, scores, {
        "candidate_cells": candidate_cells,
        "source_cells": src_cells,
        "attention": attention,
        "source_state": src_state,
        "target_state": trg_state,
    }


def _pre_softmax_channelwise_identity_rankings(
    src_features: torch.Tensor,
    trg_features: torch.Tensor,
    attention: dict[str, torch.Tensor],
    points: Sequence[Sequence[float]],
    source_size: Sequence[int],
    target_size: Sequence[int],
    src_state: FluxReplayState,
    trg_state: FluxReplayState,
    block: Any,
    *,
    candidate_topk: int,
    target_points: Sequence[Sequence[float]] | None,
    pck_threshold: float | None,
) -> tuple[torch.Tensor, list[dict[str, Any]], dict[str, Any]]:
    """Build a train-free candidate identity field before attention pooling.

    The routing posterior is used only to form the candidate pool.  Candidate
    scores are computed from channelwise pre-softmax Q/K interaction, local V
    residuals, and a deformation-tolerant local relation signature.  No
    attention probability is multiplied into the identity score, so the method
    cannot collapse back to attention ordering.  All tensors are frozen FLUX
    states and the existing block projections; no parameters are learned.
    """

    points = list(points)
    if not points:
        empty = torch.empty((0, 0), device=src_features.device, dtype=torch.long)
        return empty, [], {"points": 0}
    if target_points is not None and len(points) != len(target_points):
        raise ValueError("target_points must align with source points")

    weight = next(block.parameters())
    src = _state_to_device(src_state, weight.device, weight.dtype)
    trg = _state_to_device(trg_state, weight.device, weight.dtype)
    mutual = torch.sqrt(
        (attention["p_ab"].float() * attention["p_ba"].float().t()).clamp_min(0.0)
    )
    mutual = torch.nan_to_num(mutual, nan=0.0, posinf=0.0, neginf=0.0)
    src_cells = _native_cell_indices_for_points(
        points,
        source_size,
        src.image_height,
        src.image_width,
        mutual.device,
    )
    candidate_count = min(max(1, int(candidate_topk)), int(mutual.shape[1]))
    attention_scores, candidate_cells = torch.topk(
        mutual[src_cells], k=candidate_count, dim=1, sorted=True
    )

    with torch.no_grad():
        # Recompute all Q/K/V once.  Cross-image attention in FLUX uses the
        # unrotated image Q/K tensors, matching the established replay path.
        q_a, k_a, v_a, _mlp_a, _mod_a = _block_qkv(block, src.x, src.vec)
        q_b, k_b, v_b, _mlp_b, _mod_b = _block_qkv(block, trg.x, trg.vec)
        q_a = q_a[:, :, int(src.text_token_count):].float()
        k_a = k_a[:, :, int(src.text_token_count):].float()
        v_a = v_a[:, :, int(src.text_token_count):].float()
        q_b = q_b[:, :, int(trg.text_token_count):].float()
        k_b = k_b[:, :, int(trg.text_token_count):].float()
        v_b = v_b[:, :, int(trg.text_token_count):].float()

    ensemble, heads, _src_count, head_dim = q_a.shape
    point_count = int(src_cells.numel())

    def _gather(values: torch.Tensor, cells: torch.Tensor) -> torch.Tensor:
        flat = cells.reshape(-1).long()
        gathered = values[:, :, flat, :]
        return gathered.reshape(values.shape[0], values.shape[1], *cells.shape, values.shape[-1])

    source_q = _gather(q_a, src_cells)  # [E,H,P,D]
    source_k = _gather(k_a, src_cells)
    target_q = _gather(q_b, candidate_cells)  # [E,H,P,K,D]
    target_k = _gather(k_b, candidate_cells)

    # Remove the channel mean, which contains the scalar dot-product evidence
    # already used by attention.  The remaining signed channel pattern is the
    # pre-softmax identity hypothesis.
    qk_ab = source_q.unsqueeze(3) * target_k
    qk_ba = target_q * source_k.unsqueeze(3)
    qk_ab = qk_ab - qk_ab.mean(dim=-1, keepdim=True)
    qk_ba = qk_ba - qk_ba.mean(dim=-1, keepdim=True)
    qk_ab = F.normalize(qk_ab, dim=-1, eps=1e-12)
    qk_ba = F.normalize(qk_ba, dim=-1, eps=1e-12)

    def _flatten_pair(value: torch.Tensor) -> torch.Tensor:
        # [E,H,P,K,D] -> [P,K,E*H*D]
        return value.permute(2, 3, 0, 1, 4).reshape(point_count, candidate_count, -1)

    qk_source = _flatten_pair(qk_ab)
    qk_target = _flatten_pair(qk_ba)
    qk_score = F.cosine_similarity(qk_source, qk_target, dim=-1, eps=1e-12)

    def _local_residual(values: torch.Tensor, height: int, width: int) -> torch.Tensor:
        # [E,H,N,D] -> [E,H,N,D], subtracting a soft 3x3 spatial mean.
        grid = values.permute(0, 1, 3, 2).reshape(values.shape[0] * values.shape[1] * values.shape[3], height, width)
        mean = F.avg_pool2d(grid.unsqueeze(1), kernel_size=3, stride=1, padding=1, count_include_pad=False)
        residual = (grid - mean[:, 0]).reshape(values.shape[0], values.shape[1], values.shape[3], height * width)
        return residual.permute(0, 1, 3, 2).contiguous()

    source_v_residual = _local_residual(v_a, src.image_height, src.image_width)
    target_v_residual = _local_residual(v_b, trg.image_height, trg.image_width)
    source_v = _gather(source_v_residual, src_cells)
    target_v = _gather(target_v_residual, candidate_cells)
    value_source = source_v.unsqueeze(3).expand(-1, -1, -1, candidate_count, -1)
    value_target = target_v
    value_source = F.normalize(
        value_source.permute(2, 3, 0, 1, 4).reshape(point_count, candidate_count, -1),
        dim=-1,
        eps=1e-12,
    )
    value_target = F.normalize(
        value_target.permute(2, 3, 0, 1, 4).reshape(point_count, candidate_count, -1),
        dim=-1,
        eps=1e-12,
    )
    value_score = F.cosine_similarity(value_source, value_target, dim=-1, eps=1e-12)

    def _local_signature(values: torch.Tensor, height: int, width: int, cells: torch.Tensor) -> torch.Tensor:
        # Sorted local similarities avoid a hard orientation/rigid-geometry
        # constraint while retaining the local texture/structure response.
        normalized = F.normalize(values, dim=-1, eps=1e-12)
        flat_cells = cells.reshape(-1).long()
        x = flat_cells % int(width)
        y = torch.div(flat_cells, int(width), rounding_mode="floor")
        offsets = ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1))
        center = normalized[:, :, flat_cells, :]
        signatures = []
        for dx, dy in offsets:
            nx = (x + int(dx)).clamp(0, int(width) - 1)
            ny = (y + int(dy)).clamp(0, int(height) - 1)
            neighbor = normalized[:, :, ny * int(width) + nx, :]
            signatures.append((center * neighbor).sum(dim=-1))
        signature = torch.stack(signatures, dim=-1)
        signature = torch.sort(signature, dim=-1).values
        return signature

    source_local = _local_signature(v_a, src.image_height, src.image_width, src_cells)
    target_local = _local_signature(v_b, trg.image_height, trg.image_width, candidate_cells)
    source_local = source_local.unsqueeze(3).expand(-1, -1, -1, candidate_count, -1)
    local_source = F.normalize(
        source_local.permute(2, 3, 0, 1, 4).reshape(point_count, candidate_count, -1),
        dim=-1,
        eps=1e-12,
    )
    local_target = F.normalize(
        target_local.permute(2, 3, 0, 1).reshape(point_count, candidate_count, -1),
        dim=-1,
        eps=1e-12,
    )
    local_score = F.cosine_similarity(local_source, local_target, dim=-1, eps=1e-12)

    # Keep the original DiTF local descriptor as one normalized branch.  It is
    # part of the feature field, never a conditional fallback or selector.
    src_native_tokens = src_features[0].permute(1, 2, 0).reshape(-1, src_features.shape[1]).float()
    trg_native_tokens = trg_features[0].permute(1, 2, 0).reshape(-1, trg_features.shape[1]).float()
    native_source = F.normalize(src_native_tokens[src_cells], dim=-1, eps=1e-12)
    native_target = F.normalize(trg_native_tokens[candidate_cells], dim=-1, eps=1e-12)
    native_source = native_source.unsqueeze(1).expand(-1, candidate_count, -1)
    native_score = F.cosine_similarity(native_source, native_target, dim=-1, eps=1e-12)

    # Equal-norm concatenation is a fixed, parameter-free feature fusion.  It
    # makes the final score the mean of the four evidence branches, with no
    # tunable coefficient and no direct attention-weight term.
    combined_source = torch.cat((F.normalize(qk_source, dim=-1), value_source, local_source, native_source), dim=-1)
    combined_target = torch.cat((F.normalize(qk_target, dim=-1), value_target, local_target, native_target), dim=-1)
    combined_source = F.normalize(combined_source, dim=-1, eps=1e-12)
    combined_target = F.normalize(combined_target, dim=-1, eps=1e-12)
    combined_score = F.cosine_similarity(combined_source, combined_target, dim=-1, eps=1e-12)

    target_h, target_w = int(target_size[0]), int(target_size[1])
    proposal_pixels = _cell_topk_to_pixel_indices(candidate_cells, target_size, trg_state)
    native_scores, native_indices = _chunked_descriptor_topk_scores_indices(
        src_features,
        trg_features,
        points,
        source_size,
        target_size,
        topk=1,
    )
    native_pixels = native_indices[:, 0]
    records: list[dict[str, Any]] = []
    hits_all: list[bool] = []
    combined_order = torch.argsort(combined_score, dim=1, descending=True)
    ranked_pixels = torch.gather(proposal_pixels, 1, combined_order)
    method_indices = combined_order[:, 0]
    for row in range(point_count):
        target = target_points[row] if target_points is not None else None
        pixels = proposal_pixels[row].detach().cpu().tolist()
        hits = [
            bool(_point_hit([int(pixel % target_w), int(pixel // target_w)], target, pck_threshold))
            if target is not None and pck_threshold is not None else False
            for pixel in pixels
        ]
        attention_top1_hit = bool(hits[0]) if hits else False
        native_pixel = int(native_pixels[row].detach().cpu())
        native_hit = bool(
            _point_hit([int(native_pixel % target_w), int(native_pixel // target_w)], target, pck_threshold)
        ) if target is not None and pck_threshold is not None else False
        order_map = {
            "qk_channelwise": torch.argsort(qk_score[row], descending=True).detach().cpu().tolist(),
            "value_residual": torch.argsort(value_score[row], descending=True).detach().cpu().tolist(),
            "local_relation": torch.argsort(local_score[row], descending=True).detach().cpu().tolist(),
            "native_local": torch.argsort(native_score[row], descending=True).detach().cpu().tolist(),
            "combined": torch.argsort(combined_score[row], descending=True).detach().cpu().tolist(),
        }
        ranks = {}
        for name, order in order_map.items():
            ranks[name] = next((index + 1 for index, candidate in enumerate(order) if hits[candidate]), None)
        topk_hits = {
            name: {
                f"@{k}": bool(ranks[name] is not None and ranks[name] <= k)
                for k in (1, 3, 5, 10, 20)
            }
            for name in order_map
        }
        candidates = []
        for rank, candidate in enumerate(order_map["combined"], start=1):
            pixel = int(pixels[candidate])
            candidates.append({
                "rank": int(rank),
                "rank_attention": int(candidate + 1),
                "pixel": [int(pixel % target_w), int(pixel // target_w)],
                "pixel_index": pixel,
                "pck_hit": bool(hits[candidate]),
                "scores": {
                    "combined": float(combined_score[row, candidate].detach().cpu()),
                    "qk_channelwise": float(qk_score[row, candidate].detach().cpu()),
                    "value_residual": float(value_score[row, candidate].detach().cpu()),
                    "local_relation": float(local_score[row, candidate].detach().cpu()),
                    "native_local": float(native_score[row, candidate].detach().cpu()),
                    "attention": float(attention_scores[row, candidate].detach().cpu()),
                },
            })
        both_wrong = bool((not native_hit) and (not attention_top1_hit) and any(hits))
        combined_rank = torch.argsort(torch.argsort(combined_score[row])).float()
        attention_rank = torch.arange(candidate_count, device=combined_score.device, dtype=torch.float32)
        if candidate_count > 1 and float(attention_rank.std().detach().cpu()) > 1e-12 and float(combined_rank.std().detach().cpu()) > 1e-12:
            rank_corr = float(torch.corrcoef(torch.stack((attention_rank, combined_rank)))[0, 1].detach().cpu())
        else:
            rank_corr = 1.0
        records.append({
            "source_cell": int(src_cells[row].detach().cpu()),
            "native_pck_hit": native_hit,
            "attention_top1_pck_hit": attention_top1_hit,
            "candidate_missing_gt": bool(target is not None and not any(hits)),
            "both_wrong_top20_hit": both_wrong,
            "gt_ranks": ranks,
            "topk_hits": topk_hits,
            "candidate_pck_hit_fraction": float(sum(hits) / max(1, len(hits))),
            "attention_selected_rank": 1,
            "method_selected_rank": 1,
            "method_selected_attention_rank": int(torch.argmax(combined_score[row]).detach().cpu()) + 1,
            "attention_method_rank_correlation": rank_corr,
            "candidates": candidates,
        })
        hits_all.append(bool(hits[int(method_indices[row].detach().cpu())]) if hits else False)

    def _summary_for(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
        selected = [row for row in rows if name == "all" or bool(row.get(name, False))]
        if not selected:
            return {"points": 0}
        ranks = [row["gt_ranks"]["combined"] for row in selected if row["gt_ranks"].get("combined") is not None]
        hits = [row["gt_ranks"]["combined"] == 1 for row in selected if row["gt_ranks"].get("combined") is not None]
        return {
            "points": len(selected),
            "candidate_pck_hit_fraction_mean": float(np.mean([row["candidate_pck_hit_fraction"] for row in selected])),
            "combined_top1_pck_hits": int(sum(hits)),
            "combined_top1_pck_rate": float(np.mean(hits)) if hits else 0.0,
            "combined_gt_rank_median": float(np.median(ranks)) if ranks else None,
            "qk_channelwise_top1_pck_hits": int(sum(row["gt_ranks"].get("qk_channelwise") == 1 for row in selected)),
            "value_residual_top1_pck_hits": int(sum(row["gt_ranks"].get("value_residual") == 1 for row in selected)),
            "local_relation_top1_pck_hits": int(sum(row["gt_ranks"].get("local_relation") == 1 for row in selected)),
            "native_local_top1_pck_hits": int(sum(row["gt_ranks"].get("native_local") == 1 for row in selected)),
        }

    summary = {
        "hypothesis": {
            "name": "Pre-Softmax Channelwise Identity Field",
            "candidate_pool": "exact_mutual_cross_attention_topk_only",
            "attention_used_as_identity_score": False,
            "identity_components": [
                "centered_qk_channel_product",
                "local_v_residual",
                "sorted_local_relation",
                "native_local_descriptor",
            ],
            "train_free": True,
            "native_fallback_used": False,
            "gt_used_for_inference": False,
        },
        "all": _summary_for(records, "all"),
        "both_wrong_top20_hit": _summary_for(records, "both_wrong_top20_hit"),
        "candidate_missing_gt": _summary_for(records, "candidate_missing_gt"),
        "method_changed_count": int(sum(int(row["method_selected_attention_rank"] != 1) for row in records)),
        "prediction_changed": True,
    }
    return ranked_pixels, records, summary


def _mean_off_diagonal_cosine(tokens: torch.Tensor) -> float:
    tokens = F.normalize(tokens.float(), dim=1, eps=1e-12)
    if tokens.shape[0] <= 1:
        return 1.0
    gram = tokens @ tokens.t()
    value = (gram.sum() - gram.diagonal().sum()) / float(tokens.shape[0] * (tokens.shape[0] - 1))
    return float(value.detach().cpu())


def flux_fjsar_predict(
    src_features: torch.Tensor,
    trg_features: torch.Tensor,
    source_points: Iterable[Sequence[float]],
    source_size: Sequence[int],
    target_size: Sequence[int],
    *,
    src_replay_state: dict[str, Any],
    trg_replay_state: dict[str, Any],
    src_raw_feature: torch.Tensor,
    trg_raw_feature: torch.Tensor,
    src_ada: Any,
    trg_ada: Any,
    blocks: Sequence[Any],
    mode: str = "attention_signature",
    interaction_mode: str = "calibrated",
    use_coordinate_bias: bool = True,
    discard_channels: Sequence[int] = (),
    calibration: dict[str, Any] | None = None,
    target_points: Iterable[Sequence[float]] | None = None,
    pck_threshold: float | None = None,
    oracle_topk: Sequence[int] = (1, 5, 10, 20, 50),
    candidate_topk: int = 20,
    geometry_radius: int = 2,
    geometry_strength: float = 0.5,
    trajectory_replay_states: dict[str, Any] | None = None,
    trajectory_block_modules: dict[str, Any] | None = None,
    trajectory_blocks: Sequence[int] = (),
    layer_identity_maps: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
    return_diagnostics: bool = False,
) -> list[list[int]] | tuple[list[list[int]], dict[str, float]]:
    """Boundary-aligned, ensemble-safe frozen joint self-attention replay.

    Correctness invariants:

    * replay state, raw feature and AdaLN state come from the same forward;
    * every ensemble member passes through the frozen replay block(s) before
      averaging;
    * the native replay raw boundary and prepared descriptor are audited against
      the aligned cache;
    * the final descriptor keeps the official dimensionality and becomes the
      native feature exactly when pair evidence vanishes.
    """

    points = list(source_points)
    gt_points = list(target_points) if target_points is not None else None
    if not points:
        empty: list[list[int]] = []
        return (empty, {"mean_cross_mass_source": 0.0}) if return_diagnostics else empty
    if gt_points is not None and len(gt_points) != len(points):
        raise ValueError("target_points must align with source_points")
    if len(blocks) not in (1, 2):
        raise ValueError("FJSAR requires one or two complete FLUX interaction blocks")

    src_state = FluxReplayState.from_dict(src_replay_state)
    trg_state = FluxReplayState.from_dict(trg_replay_state)
    if src_state.global_block_index != trg_state.global_block_index:
        raise ValueError("source and target replay caches use different starting blocks")
    if src_state.ensemble_size != trg_state.ensemble_size:
        raise ValueError("source and target ensemble counts differ")
    if (src_state.image_height, src_state.image_width) != tuple(src_features.shape[-2:]):
        raise ValueError("source replay grid does not match the DiTF feature grid")
    if (trg_state.image_height, trg_state.image_width) != tuple(trg_features.shape[-2:]):
        raise ValueError("target replay grid does not match the DiTF feature grid")

    with torch.no_grad():
        src_native_sequence = run_flux_native_stack(blocks, src_state)
        trg_native_sequence = run_flux_native_stack(blocks, trg_state)
        if interaction_mode == "cross_only":
            src_joint_sequence, trg_joint_sequence, attention = run_flux_cross_only_stack(
                blocks,
                src_state,
                trg_state,
            )
        elif interaction_mode == "identity_preserving":
            src_joint_sequence, trg_joint_sequence, attention = run_flux_identity_preserving_stack(
                blocks,
                src_state,
                trg_state,
            )
        elif interaction_mode == "balanced_transport":
            src_joint_sequence, trg_joint_sequence, attention = run_flux_balanced_transport_stack(
                blocks,
                src_state,
                trg_state,
            )
        elif interaction_mode == "qk_identity":
            src_joint_sequence, trg_joint_sequence, attention = run_flux_qk_identity_stack(
                blocks,
                src_state,
                trg_state,
            )
        elif interaction_mode == "trajectory":
            src_joint_sequence, trg_joint_sequence, attention = run_flux_cross_only_stack(
                blocks,
                src_state,
                trg_state,
            )
        elif interaction_mode == "geometry_consistent":
            src_joint_sequence, trg_joint_sequence, attention = run_flux_geometry_consistent_stack(
                blocks,
                src_state,
                trg_state,
                radius=geometry_radius,
                strength=geometry_strength,
            )
        else:
            src_joint_sequence, trg_joint_sequence, attention = run_flux_joint_stack(
                blocks,
                src_state,
                trg_state,
                mode=interaction_mode,
                use_coordinate_bias=use_coordinate_bias,
            )
        src_native_raw_members = native_image_tokens(src_native_sequence, src_state)
        trg_native_raw_members = native_image_tokens(trg_native_sequence, trg_state)
        src_joint_raw_members = native_image_tokens(src_joint_sequence, src_state)
        trg_joint_raw_members = native_image_tokens(trg_joint_sequence, trg_state)

        src_raw_parity = parity_metrics(
            _raw_feature_tokens(src_raw_feature).to(src_native_raw_members.device),
            src_native_raw_members.float().mean(dim=0, keepdim=True),
        )
        trg_raw_parity = parity_metrics(
            _raw_feature_tokens(trg_raw_feature).to(trg_native_raw_members.device),
            trg_native_raw_members.float().mean(dim=0, keepdim=True),
        )

        src_native_prepared = _prepare_replay_tokens(
            src_native_raw_members,
            src_ada,
            discard_channels=discard_channels,
            calibration=calibration,
        )[0]
        trg_native_prepared = _prepare_replay_tokens(
            trg_native_raw_members,
            trg_ada,
            discard_channels=discard_channels,
            calibration=calibration,
        )[0]
        src_joint_prepared = _prepare_replay_tokens(
            src_joint_raw_members,
            src_ada,
            discard_channels=discard_channels,
            calibration=calibration,
        )[0]
        trg_joint_prepared = _prepare_replay_tokens(
            trg_joint_raw_members,
            trg_ada,
            discard_channels=discard_channels,
            calibration=calibration,
        )[0]
        src_official = src_features[0].permute(1, 2, 0).reshape(-1, src_features.shape[1]).float()
        trg_official = trg_features[0].permute(1, 2, 0).reshape(-1, trg_features.shape[1]).float()
        src_prepared_parity = parity_metrics(src_official, src_native_prepared)
        trg_prepared_parity = parity_metrics(trg_official, trg_native_prepared)

        # A failed boundary audit means the adapter is not replaying the feature
        # used by the official baseline.  Silently mixing those spaces would make
        # the benchmark uninterpretable, so fall back exactly.
        parity_ok = (
            src_raw_parity["cosine"] >= 0.995
            and trg_raw_parity["cosine"] >= 0.995
            and src_prepared_parity["cosine"] >= 0.995
            and trg_prepared_parity["cosine"] >= 0.995
            and src_prepared_parity["relative_l2_error"] <= 0.15
            and trg_prepared_parity["relative_l2_error"] <= 0.15
        )

        relation = _fjsar_relation_statistics(
            attention,
            src_joint_prepared - src_native_prepared,
            trg_joint_prepared - trg_native_prepared,
            src_native_prepared,
            trg_native_prepared,
        )
        trajectory_ranked_cells = None
        trajectory_audits = None
        trajectory_summary: dict[str, float] = {}
        topology_rescue_ranked_pixels = None
        topology_rescue_audits = None
        topology_rescue_summary: dict[str, float] = {}
        basin_refine_ranked_pixels = None
        candidate_verification_ranked_pixels = None
        candidate_verification_audits = None
        candidate_verification_summary: dict[str, float] = {}
        local_transport_verification_ranked_pixels = None
        local_transport_verification_audits = None
        local_transport_verification_summary: dict[str, float] = {}
        graph_consensus_verification_ranked_pixels = None
        graph_consensus_verification_audits = None
        graph_consensus_verification_summary: dict[str, float] = {}
        attention_relational_graph_ranked_pixels = None
        attention_relational_graph_audits = None
        attention_relational_graph_summary: dict[str, Any] = {}
        dense_partial_graph_ranked_pixels = None
        dense_partial_graph_audits = None
        dense_partial_graph_summary: dict[str, Any] = {}
        expert_hypothesis_ranked_pixels = None
        expert_hypothesis_audits = None
        expert_hypothesis_summary: dict[str, Any] = {}
        pre_softmax_identity_ranked_pixels = None
        pre_softmax_identity_audits = None
        pre_softmax_identity_summary: dict[str, Any] = {}
        layer_routed_identity_ranked_pixels = None
        layer_routed_identity_audits = None
        layer_routed_identity_summary: dict[str, Any] = {}
        spectral_kernel_diagnostics: dict[str, Any] = {}
        if mode == "pre_softmax_channelwise_identity":
            (
                pre_softmax_identity_ranked_pixels,
                pre_softmax_identity_audits,
                pre_softmax_identity_summary,
            ) = _pre_softmax_channelwise_identity_rankings(
                src_features,
                trg_features,
                attention,
                points,
                source_size,
                target_size,
                src_state,
                trg_state,
                blocks[0],
                candidate_topk=candidate_topk,
                target_points=gt_points,
                pck_threshold=pck_threshold,
            )
        elif mode in ("layer_routed_identity", "pre_single_stream_identity"):
            if not layer_identity_maps:
                raise ValueError(
                    f"{mode} requires source-only multilayer identity maps"
                )
            (
                layer_routed_identity_ranked_pixels,
                layer_routed_identity_audits,
                layer_routed_identity_summary,
            ) = _layerwise_routing_identity_rankings(
                _token_matrix_to_map(
                        src_joint_prepared, src_state.image_height, src_state.image_width
                    ),
                    _token_matrix_to_map(
                        trg_joint_prepared, trg_state.image_height, trg_state.image_width
                    ),
                layer_identity_maps,
                attention,
                points,
                source_size,
                target_size,
                src_state,
                trg_state,
                candidate_topk=candidate_topk,
                target_points=gt_points,
                pck_threshold=pck_threshold,
                identity_only_primary=(mode == "pre_single_stream_identity"),
            )
        elif mode == "cross_attention_trajectory":
            trajectory_layers = _trajectory_attention_layers(
                trajectory_replay_states,
                trajectory_block_modules,
                trajectory_blocks,
                fallback_src_state=src_state,
                fallback_trg_state=trg_state,
                fallback_attention=attention,
            )
            trajectory_ranked_cells, trajectory_audits, trajectory_summary = _cross_attention_trajectory_rankings(
                trajectory_layers,
                int(src_state.global_block_index) + 1,
                points,
                source_size,
                target_size,
                candidate_topk=candidate_topk,
                target_points=gt_points,
                pck_threshold=pck_threshold,
            )
        elif mode == "native_preserving_topology_rescue":
            topology_rescue_ranked_pixels, topology_rescue_audits, topology_rescue_summary = (
                _native_preserving_topology_rescue_rankings(
                    src_features,
                    trg_features,
                    attention,
                    points,
                    source_size,
                    target_size,
                    src_state,
                    trg_state,
                    candidate_topk=candidate_topk,
                    target_points=gt_points,
                    pck_threshold=pck_threshold,
                )
            )
        elif mode == "attention_basin_native_refine":
            basin_refine_ranked_pixels = _attention_basin_native_refine_rankings(
                src_features,
                trg_features,
                attention,
                points,
                source_size,
                target_size,
                src_state,
                trg_state,
                candidate_topk=candidate_topk,
            )
        elif mode == "candidate_conditioned_verification":
            candidate_verification_ranked_pixels, candidate_verification_audits, candidate_verification_summary = (
                _candidate_conditioned_verification_rankings(
                    src_features,
                    trg_features,
                    attention,
                    points,
                    source_size,
                    target_size,
                    src_state,
                    trg_state,
                    candidate_topk=candidate_topk,
                    target_points=gt_points,
                    pck_threshold=pck_threshold,
                )
            )
        elif mode == "candidate_local_transport_verification":
            local_transport_verification_ranked_pixels, local_transport_verification_audits, local_transport_verification_summary = (
                _candidate_local_transport_verification_rankings(
                    src_features,
                    trg_features,
                    attention,
                    points,
                    source_size,
                    target_size,
                    src_state,
                    trg_state,
                    candidate_topk=candidate_topk,
                    target_points=gt_points,
                    pck_threshold=pck_threshold,
                )
            )
        elif mode == "candidate_graph_consensus_verification":
            graph_consensus_verification_ranked_pixels, graph_consensus_verification_audits, graph_consensus_verification_summary = (
                _candidate_graph_consensus_verification_rankings(
                    src_features,
                    trg_features,
                    attention,
                    points,
                    source_size,
                    target_size,
                    src_state,
                    trg_state,
                    candidate_topk=candidate_topk,
                    target_points=gt_points,
                    pck_threshold=pck_threshold,
                )
            )
        elif mode == "attention_relational_graph_matching":
            (
                attention_relational_graph_ranked_pixels,
                attention_relational_graph_audits,
                attention_relational_graph_summary,
            ) = _attention_relational_graph_matching_rankings(
                src_native_prepared,
                trg_native_prepared,
                attention,
                points,
                source_size,
                target_size,
                src_state,
                trg_state,
                candidate_topk=candidate_topk,
                target_points=gt_points,
                pck_threshold=pck_threshold,
            )
        elif mode == "dense_partial_graph_matching":
            (
                dense_partial_graph_ranked_pixels,
                dense_partial_graph_audits,
                dense_partial_graph_summary,
            ) = _dense_partial_graph_matching_rankings(
                src_features,
                trg_features,
                attention,
                points,
                source_size,
                target_size,
                src_state,
                trg_state,
                candidate_topk=candidate_topk,
                target_points=gt_points,
                pck_threshold=pck_threshold,
            )
        elif mode == "expert_preserving_attention_hypothesis_conditioned_replay":
            (
                expert_hypothesis_ranked_pixels,
                expert_hypothesis_audits,
                expert_hypothesis_summary,
            ) = _expert_preserving_attention_hypothesis_conditioned_replay_rankings(
                attention,
                points,
                source_size,
                target_size,
                src_state,
                trg_state,
                blocks,
                candidate_topk=candidate_topk,
                target_points=gt_points,
                pck_threshold=pck_threshold,
            )
        oracle_counts = _fjsar_candidate_oracle_counts(
            src_features=src_features,
            trg_features=trg_features,
            points=points,
            target_points=gt_points,
            source_size=source_size,
            target_size=target_size,
            pck_threshold=pck_threshold,
            oracle_topk=oracle_topk,
            src_state=src_state,
            trg_state=trg_state,
            src_native_prepared=src_native_prepared,
            trg_native_prepared=trg_native_prepared,
            src_joint_prepared=src_joint_prepared,
            trg_joint_prepared=trg_joint_prepared,
            attention=attention,
            descriptor_modes=(
                ()
                if mode in (
                    "cross_attention_trajectory",
                    "native_preserving_topology_rescue",
                    "attention_basin_native_refine",
                    "candidate_conditioned_verification",
                    "candidate_local_transport_verification",
                    "attention_relational_graph_matching",
                    "dense_partial_graph_matching",
                    "expert_preserving_attention_hypothesis_conditioned_replay",
                    "pre_softmax_channelwise_identity",
                    "layer_routed_identity",
                    "pre_single_stream_identity",
                )
                else (mode,)
            ),
        )
        if trajectory_ranked_cells is not None and gt_points is not None and pck_threshold is not None:
            trajectory_pixels = _cell_topk_to_pixel_indices(trajectory_ranked_cells, target_size, trg_state)
            trajectory_hits = _per_point_topk_hits(
                trajectory_pixels,
                gt_points,
                pck_threshold,
                target_size,
                oracle_topk,
            )
            for k in tuple(sorted({max(1, int(value)) for value in oracle_topk})):
                oracle_counts[f"fjsar_oracle_owner_cross_attention_trajectory@{k}"] = sum(
                    int(bool(row["topk_hits"].get(int(k), False))) for row in trajectory_hits
                )
        if topology_rescue_ranked_pixels is not None and gt_points is not None and pck_threshold is not None:
            for k, count in _topk_hit_counts(
                topology_rescue_ranked_pixels,
                gt_points,
                float(pck_threshold),
                target_size,
                tuple(sorted({max(1, int(value)) for value in oracle_topk})),
            ).items():
                oracle_counts[f"fjsar_oracle_owner_native_preserving_topology_rescue@{k}"] = count
        if basin_refine_ranked_pixels is not None and gt_points is not None and pck_threshold is not None:
            for k, count in _topk_hit_counts(
                basin_refine_ranked_pixels,
                gt_points,
                float(pck_threshold),
                target_size,
                tuple(sorted({max(1, int(value)) for value in oracle_topk})),
            ).items():
                oracle_counts[f"fjsar_oracle_owner_attention_basin_native_refine@{k}"] = count
        if candidate_verification_ranked_pixels is not None and gt_points is not None and pck_threshold is not None:
            for k, count in _topk_hit_counts(
                candidate_verification_ranked_pixels,
                gt_points,
                float(pck_threshold),
                target_size,
                tuple(sorted({max(1, int(value)) for value in oracle_topk})),
            ).items():
                oracle_counts[f"fjsar_oracle_owner_candidate_conditioned_verification@{k}"] = count
        if local_transport_verification_ranked_pixels is not None and gt_points is not None and pck_threshold is not None:
            for k, count in _topk_hit_counts(
                local_transport_verification_ranked_pixels,
                gt_points,
                float(pck_threshold),
                target_size,
                tuple(sorted({max(1, int(value)) for value in oracle_topk})),
            ).items():
                oracle_counts[f"fjsar_oracle_owner_candidate_local_transport_verification@{k}"] = count
        if graph_consensus_verification_ranked_pixels is not None and gt_points is not None and pck_threshold is not None:
            for k, count in _topk_hit_counts(
                graph_consensus_verification_ranked_pixels,
                gt_points,
                float(pck_threshold),
                target_size,
                tuple(sorted({max(1, int(value)) for value in oracle_topk})),
            ).items():
                oracle_counts[f"fjsar_oracle_owner_candidate_graph_consensus_verification@{k}"] = count
        if attention_relational_graph_ranked_pixels is not None and gt_points is not None and pck_threshold is not None:
            for k, count in _topk_hit_counts(
                attention_relational_graph_ranked_pixels,
                gt_points,
                float(pck_threshold),
                target_size,
                tuple(sorted({max(1, int(value)) for value in oracle_topk})),
            ).items():
                oracle_counts[f"fjsar_oracle_owner_attention_relational_graph_matching@{k}"] = count
        if dense_partial_graph_ranked_pixels is not None and gt_points is not None and pck_threshold is not None:
            for k, count in _topk_hit_counts(
                dense_partial_graph_ranked_pixels,
                gt_points,
                float(pck_threshold),
                target_size,
                tuple(sorted({max(1, int(value)) for value in oracle_topk})),
            ).items():
                oracle_counts[f"fjsar_oracle_owner_dense_partial_graph_matching@{k}"] = count
        if expert_hypothesis_ranked_pixels is not None and gt_points is not None and pck_threshold is not None:
            for k, count in _topk_hit_counts(
                expert_hypothesis_ranked_pixels,
                gt_points,
                float(pck_threshold),
                target_size,
                tuple(sorted({max(1, int(value)) for value in oracle_topk})),
            ).items():
                oracle_counts[
                    f"fjsar_oracle_owner_expert_preserving_attention_hypothesis_conditioned_replay@{k}"
                ] = count
        if pre_softmax_identity_ranked_pixels is not None and gt_points is not None and pck_threshold is not None:
            for k, count in _topk_hit_counts(
                pre_softmax_identity_ranked_pixels,
                gt_points,
                float(pck_threshold),
                target_size,
                tuple(sorted({max(1, int(value)) for value in oracle_topk})),
            ).items():
                oracle_counts[f"fjsar_oracle_owner_pre_softmax_channelwise_identity@{k}"] = count
        if layer_routed_identity_ranked_pixels is not None and gt_points is not None and pck_threshold is not None:
            oracle_owner = (
                "pre_single_stream_identity"
                if mode == "pre_single_stream_identity"
                else "layer_routed_identity"
            )
            for k, count in _topk_hit_counts(
                layer_routed_identity_ranked_pixels,
                gt_points,
                float(pck_threshold),
                target_size,
                tuple(sorted({max(1, int(value)) for value in oracle_topk})),
            ).items():
                oracle_counts[f"fjsar_oracle_owner_{oracle_owner}@{k}"] = count
        attention_case_records = _attention_case_records_from_attention(
            attention,
            points,
            source_size,
            target_size,
            src_state,
            trg_state,
            target_points=gt_points,
            pck_threshold=pck_threshold,
            candidate_topk=candidate_topk,
        )

        if mode == "pre_softmax_channelwise_identity":
            if pre_softmax_identity_ranked_pixels is None:
                raise RuntimeError("pre_softmax_channelwise_identity did not produce ranked candidates")
            selected = pre_softmax_identity_ranked_pixels[:, 0].long()
            target_w = int(target_size[1])
            predictions = [
                [int(pixel % target_w), int(pixel // target_w)]
                for pixel in selected.detach().cpu().tolist()
            ]
        elif mode in ("layer_routed_identity", "pre_single_stream_identity"):
            if layer_routed_identity_ranked_pixels is None:
                raise RuntimeError(f"{mode} did not produce ranked candidates")
            selected = layer_routed_identity_ranked_pixels[:, 0].long()
            target_w = int(target_size[1])
            predictions = [
                [int(pixel % target_w), int(pixel // target_w)]
                for pixel in selected.detach().cpu().tolist()
            ]
        elif mode == "attention":
            src_indices = _native_cell_indices_for_points(
                points, source_size, src_state.image_height, src_state.image_width, src_features.device
            )
            mutual = torch.sqrt((attention["p_ab"] * attention["p_ba"].t()).clamp_min(0.0))
            selected = mutual[src_indices].argmax(dim=1)
            predictions = _decode_native_cells(
                selected, target_size, trg_state.image_height, trg_state.image_width
            )
        elif mode == "cross_attention_trajectory":
            if trajectory_ranked_cells is None:
                raise RuntimeError("cross_attention_trajectory did not produce ranked candidates")
            selected = trajectory_ranked_cells[:, 0]
            predictions = _decode_native_cells(
                selected, target_size, trg_state.image_height, trg_state.image_width
            )
        elif mode == "native_preserving_topology_rescue":
            if topology_rescue_ranked_pixels is None:
                raise RuntimeError("native_preserving_topology_rescue did not produce ranked candidates")
            selected = topology_rescue_ranked_pixels[:, 0].long()
            target_h, target_w = int(target_size[0]), int(target_size[1])
            predictions = [
                [int(pixel % target_w), int(pixel // target_w)]
                for pixel in selected.detach().cpu().tolist()
            ]
        elif mode == "attention_basin_native_refine":
            if basin_refine_ranked_pixels is None:
                raise RuntimeError("attention_basin_native_refine did not produce ranked candidates")
            selected = basin_refine_ranked_pixels[:, 0].long()
            target_h, target_w = int(target_size[0]), int(target_size[1])
            predictions = [
                [int(pixel % target_w), int(pixel // target_w)]
                for pixel in selected.detach().cpu().tolist()
            ]
        elif mode == "candidate_conditioned_verification":
            if candidate_verification_ranked_pixels is None:
                raise RuntimeError("candidate_conditioned_verification did not produce ranked candidates")
            selected = candidate_verification_ranked_pixels[:, 0].long()
            target_h, target_w = int(target_size[0]), int(target_size[1])
            predictions = [
                [int(pixel % target_w), int(pixel // target_w)]
                for pixel in selected.detach().cpu().tolist()
            ]
        elif mode == "candidate_local_transport_verification":
            if local_transport_verification_ranked_pixels is None:
                raise RuntimeError("candidate_local_transport_verification did not produce ranked candidates")
            selected = local_transport_verification_ranked_pixels[:, 0].long()
            target_h, target_w = int(target_size[0]), int(target_size[1])
            predictions = [
                [int(pixel % target_w), int(pixel // target_w)]
                for pixel in selected.detach().cpu().tolist()
            ]
        elif mode == "candidate_graph_consensus_verification":
            if graph_consensus_verification_ranked_pixels is None:
                raise RuntimeError("candidate_graph_consensus_verification did not produce ranked candidates")
            selected = graph_consensus_verification_ranked_pixels[:, 0].long()
            target_h, target_w = int(target_size[0]), int(target_size[1])
            predictions = [
                [int(pixel % target_w), int(pixel // target_w)]
                for pixel in selected.detach().cpu().tolist()
            ]
        elif mode == "attention_relational_graph_matching":
            if attention_relational_graph_ranked_pixels is None:
                raise RuntimeError("attention_relational_graph_matching did not produce ranked candidates")
            selected = attention_relational_graph_ranked_pixels[:, 0].long()
            target_w = int(target_size[1])
            predictions = [
                [int(pixel % target_w), int(pixel // target_w)]
                for pixel in selected.detach().cpu().tolist()
            ]
        elif mode == "dense_partial_graph_matching":
            if dense_partial_graph_ranked_pixels is None:
                raise RuntimeError("dense_partial_graph_matching did not produce ranked candidates")
            selected = dense_partial_graph_ranked_pixels[:, 0].long()
            target_w = int(target_size[1])
            predictions = [
                [int(pixel % target_w), int(pixel // target_w)]
                for pixel in selected.detach().cpu().tolist()
            ]
        elif mode == "expert_preserving_attention_hypothesis_conditioned_replay":
            if expert_hypothesis_ranked_pixels is None:
                raise RuntimeError(
                    "expert_preserving_attention_hypothesis_conditioned_replay "
                    "did not produce ranked candidates"
                )
            selected = expert_hypothesis_ranked_pixels[:, 0].long()
            target_w = int(target_size[1])
            predictions = [
                [int(pixel % target_w), int(pixel // target_w)]
                for pixel in selected.detach().cpu().tolist()
            ]
        else:
            if mode == "attention_signature":
                src_map, trg_map = _attention_signature_descriptors(
                    src_features,
                    trg_features,
                    attention,
                )
            elif mode == "part_sharpen":
                src_map, trg_map = _part_common_sharpen_descriptors(
                    src_features,
                    trg_features,
                    attention,
                )
            elif mode == "orthogonal_context":
                src_map, trg_map = _orthogonal_context_descriptors(
                    src_features,
                    trg_features,
                    src_joint_prepared,
                    trg_joint_prepared,
                    src_native_prepared,
                    trg_native_prepared,
                    attention,
                )
            elif mode == "spectral_identity":
                src_map, trg_map = _spectral_attention_identity_descriptors(
                    src_features,
                    trg_features,
                    attention,
                )
            elif mode == "filtered_spectral_kernel":
                src_map, trg_map, spectral_kernel_diagnostics = (
                    filtered_spectral_kernel_feature_maps(
                        src_features,
                        trg_features,
                        attention,
                        src_state,
                        trg_state,
                    )
                )
            elif mode == "transport_lift":
                src_map, trg_map = _local_transport_lift_descriptors(
                    src_features,
                    trg_features,
                    attention,
                )
            elif mode == "basin_contrastive_identity":
                src_map, trg_map = _basin_contrastive_identity_descriptors(
                    src_features,
                    trg_features,
                    attention,
                )
            elif mode == "attention_isometry":
                src_map, trg_map = _attention_guided_isometry_descriptors(
                    src_features,
                    trg_features,
                    attention,
                )
            elif mode == "geometry_consistent_attention":
                src_map = _token_matrix_to_map(
                    F.normalize(src_joint_prepared, dim=1, eps=1e-12),
                    src_state.image_height,
                    src_state.image_width,
                )
                trg_map = _token_matrix_to_map(
                    F.normalize(trg_joint_prepared, dim=1, eps=1e-12),
                    trg_state.image_height,
                    trg_state.image_width,
                )
            elif mode == "identity_preserving_attention":
                src_map = _token_matrix_to_map(
                    F.normalize(src_joint_prepared, dim=1, eps=1e-12),
                    src_state.image_height,
                    src_state.image_width,
                )
                trg_map = _token_matrix_to_map(
                    F.normalize(trg_joint_prepared, dim=1, eps=1e-12),
                    trg_state.image_height,
                    trg_state.image_width,
                )
            elif mode == "balanced_transport_attention":
                src_map = _token_matrix_to_map(
                    F.normalize(src_joint_prepared, dim=1, eps=1e-12),
                    src_state.image_height,
                    src_state.image_width,
                )
                trg_map = _token_matrix_to_map(
                    F.normalize(trg_joint_prepared, dim=1, eps=1e-12),
                    trg_state.image_height,
                    trg_state.image_width,
                )
            elif mode == "qk_identity_attention":
                src_map = _token_matrix_to_map(
                    F.normalize(src_joint_prepared, dim=1, eps=1e-12),
                    src_state.image_height,
                    src_state.image_width,
                )
                trg_map = _token_matrix_to_map(
                    F.normalize(trg_joint_prepared, dim=1, eps=1e-12),
                    trg_state.image_height,
                    trg_state.image_width,
                )
            else:
                raise ValueError(f"unsupported FJSAR mode: {mode}")
            descriptors_valid = bool(
                torch.isfinite(src_map).all()
                and torch.isfinite(trg_map).all()
                and src_map.float().square().mean() > 1e-12
                and trg_map.float().square().mean() > 1e-12
            )
            src_map = torch.nan_to_num(src_map.float(), nan=0.0, posinf=0.0, neginf=0.0)
            trg_map = torch.nan_to_num(trg_map.float(), nan=0.0, posinf=0.0, neginf=0.0)
            predictions = _chunked_descriptor_nn_predict(
                src_map,
                trg_map,
                points,
                source_size,
                target_size,
            )

        formula_parity = native_parity_error(blocks[0], src_state)
        diagnostics = {
            "mean_cross_mass_source": float(relation["cross_mass_a"].mean().detach().cpu()),
            "mean_cross_mass_target": float(relation["cross_mass_b"].mean().detach().cpu()),
            "mean_cross_excess_source": float(attention["cross_excess_a"].mean().detach().cpu()),
            "mean_cross_excess_target": float(attention["cross_excess_b"].mean().detach().cpu()),
            "mean_concentration_source": float(relation["concentration_a"].mean().detach().cpu()),
            "mean_concentration_target": float(relation["concentration_b"].mean().detach().cpu()),
            "mean_reciprocal_source": float(relation["reciprocal_a"].mean().detach().cpu()),
            "mean_reciprocal_target": float(relation["reciprocal_b"].mean().detach().cpu()),
            "mean_residual_confidence_source": float(relation["relation_a"].mean().detach().cpu()),
            "mean_residual_confidence_target": float(relation["relation_b"].mean().detach().cpu()),
            "mean_position_confidence_source": float(relation["coordinate_a"].mean().detach().cpu()),
            "mean_position_confidence_target": float(relation["coordinate_b"].mean().detach().cpu()),
            "mean_cycle_error_source": float(attention["cycle_error_a"].mean().detach().cpu()),
            "mean_cycle_error_target": float(attention["cycle_error_b"].mean().detach().cpu()),
            "native_parity_cosine": float(formula_parity["cosine"]),
            "native_parity_max_abs_error": float(formula_parity["max_abs_error"]),
            "raw_boundary_parity_cosine": min(src_raw_parity["cosine"], trg_raw_parity["cosine"]),
            "raw_boundary_relative_l2": max(src_raw_parity["relative_l2_error"], trg_raw_parity["relative_l2_error"]),
            "prepared_feature_parity_cosine": min(src_prepared_parity["cosine"], trg_prepared_parity["cosine"]),
            "prepared_feature_relative_l2": max(src_prepared_parity["relative_l2_error"], trg_prepared_parity["relative_l2_error"]),
            "ensemble_size": float(src_state.ensemble_size),
            "replay_depth": float(len(blocks)),
            "interaction_exact": float(interaction_mode == "exact"),
            "used_coordinate_bias": float(use_coordinate_bias),
            "spectral_kernel_rank": float(
                spectral_kernel_diagnostics.get("rank", 0)
            ),
            "spectral_kernel_effective_rank": float(
                spectral_kernel_diagnostics.get("effective_rank", 0)
            ),
            "spectral_kernel_radius": float(
                spectral_kernel_diagnostics.get("radius", 0)
            ),
            "spectral_kernel_weight": float(
                spectral_kernel_diagnostics.get("weight", 0.0)
            ),
            "spectral_kernel_mean_local_support": float(
                spectral_kernel_diagnostics.get("mean_local_support", 0.0)
            ),
            "geometry_radius": float(geometry_radius if interaction_mode == "geometry_consistent" else 0),
            "geometry_strength": float(geometry_strength if interaction_mode == "geometry_consistent" else 0.0),
            "mean_identity_residual_ratio_source": float(
                attention.get("identity_residual_ratio_a", torch.zeros(1, device=src_joint_prepared.device)).float().mean().detach().cpu()
            ),
            "mean_identity_residual_ratio_target": float(
                attention.get("identity_residual_ratio_b", torch.zeros(1, device=trg_joint_prepared.device)).float().mean().detach().cpu()
            ),
            "mean_geometry_support_source": float(
                attention.get("geometry_support_a", torch.zeros(1, device=src_joint_prepared.device)).float().mean().detach().cpu()
            ),
            "mean_geometry_support_target": float(
                attention.get("geometry_support_b", torch.zeros(1, device=trg_joint_prepared.device)).float().mean().detach().cpu()
            ),
            "mean_balanced_transport_row_error": float(
                attention.get("balanced_transport_row_error_a", torch.zeros(1, device=src_joint_prepared.device)).float().mean().detach().cpu()
            ),
            "mean_balanced_transport_col_error": float(
                attention.get("balanced_transport_col_error_b", torch.zeros(1, device=trg_joint_prepared.device)).float().mean().detach().cpu()
            ),
            "mean_qk_fisher_ratio": float(
                0.5
                * (
                    attention.get("qk_fisher_ratio_a", torch.zeros(1, device=src_joint_prepared.device)).float().mean()
                    + attention.get("qk_fisher_ratio_b", torch.zeros(1, device=trg_joint_prepared.device)).float().mean()
                ).detach().cpu()
            ),
            "mean_trajectory_layer_count": float(trajectory_summary.get("trajectory_layer_count", 0.0)),
            "mean_trajectory_top1_stability": float(trajectory_summary.get("trajectory_mean_top1_stability", 0.0)),
            "topology_rescue_candidate_pool_mean": float(topology_rescue_summary.get("candidate_pool_mean", 0.0)),
            "topology_rescue_native_keep_rate": float(topology_rescue_summary.get("native_keep_rate", 0.0)),
            "topology_rescue_rescue_rate": float(topology_rescue_summary.get("rescue_rate", 0.0)),
            "topology_rescue_native_confidence_mean": float(topology_rescue_summary.get("native_confidence_mean", 0.0)),
            "topology_rescue_selected_support_mean": float(topology_rescue_summary.get("selected_top1_support_mean", 0.0)),
            "candidate_verification_candidate_pool_mean": float(candidate_verification_summary.get("candidate_pool_mean", 0.0)),
            "candidate_verification_native_selected_rate": float(candidate_verification_summary.get("native_selected_rate", 0.0)),
            "candidate_verification_attention_selected_rate": float(candidate_verification_summary.get("attention_selected_rate", 0.0)),
            "candidate_verification_native_rank_mean": float(candidate_verification_summary.get("native_rank_mean", 0.0)),
            "candidate_verification_selected_attention_rank_mean": float(candidate_verification_summary.get("selected_attention_rank_mean", 0.0)),
            "candidate_verification_anchor_confidence_mean": float(candidate_verification_summary.get("anchor_confidence_mean", 0.0)),
            "candidate_local_transport_candidate_pool_mean": float(local_transport_verification_summary.get("candidate_pool_mean", 0.0)),
            "candidate_local_transport_native_selected_rate": float(local_transport_verification_summary.get("native_selected_rate", 0.0)),
            "candidate_local_transport_rescue_rate": float(local_transport_verification_summary.get("rescue_rate", 0.0)),
            "candidate_local_transport_abstained_challenger_rate": float(local_transport_verification_summary.get("abstained_challenger_rate", 0.0)),
            "candidate_local_transport_native_rank_mean": float(local_transport_verification_summary.get("native_rank_mean", 0.0)),
            "candidate_local_transport_selected_attention_rank_mean": float(local_transport_verification_summary.get("selected_attention_rank_mean", 0.0)),
            "candidate_local_transport_anchor_confidence_mean": float(local_transport_verification_summary.get("anchor_confidence_mean", 0.0)),
            "candidate_local_transport_margin_over_native_mean": float(local_transport_verification_summary.get("transport_margin_over_native_mean", 0.0)),
            "candidate_local_transport_dominance_win_fraction_mean": float(local_transport_verification_summary.get("dominance_win_fraction_mean", 0.0)),
            "candidate_graph_consensus_candidate_pool_mean": float(graph_consensus_verification_summary.get("candidate_pool_mean", 0.0)),
            "candidate_graph_consensus_native_selected_rate": float(graph_consensus_verification_summary.get("native_selected_rate", 0.0)),
            "candidate_graph_consensus_rescue_rate": float(graph_consensus_verification_summary.get("rescue_rate", 0.0)),
            "candidate_graph_consensus_iteration_mean": float(graph_consensus_verification_summary.get("iteration_mean", 0.0)),
            "candidate_graph_consensus_selected_confidence_mean": float(graph_consensus_verification_summary.get("selected_confidence_mean", 0.0)),
            "candidate_graph_consensus_consensus_margin_mean": float(graph_consensus_verification_summary.get("consensus_margin_mean", 0.0)),
            "candidate_graph_consensus_local_transport_native_selected_rate": float(graph_consensus_verification_summary.get("local_transport_native_selected_rate", 0.0)),
            "candidate_graph_consensus_local_transport_rescue_rate": float(graph_consensus_verification_summary.get("local_transport_rescue_rate", 0.0)),
            "candidate_graph_consensus_local_transport_anchor_confidence_mean": float(graph_consensus_verification_summary.get("local_transport_anchor_confidence_mean", 0.0)),
            "candidate_graph_consensus_local_transport_selected_attention_rank_mean": float(graph_consensus_verification_summary.get("local_transport_selected_attention_rank_mean", 0.0)),
            "attention_relational_graph_candidate_pool_mean": float(attention_relational_graph_summary.get("candidate_pool_mean", 0.0)),
            "attention_relational_graph_edge_count": float(attention_relational_graph_summary.get("edge_count", 0.0)),
            "attention_relational_graph_energy_gain": float(attention_relational_graph_summary.get("energy_gain", 0.0)),
            "attention_relational_graph_selected_attention_rank_mean": float(attention_relational_graph_summary.get("selected_attention_rank_mean", 0.0)),
            "attention_relational_graph_native_injected_candidate_count": float(attention_relational_graph_summary.get("native_injected_candidate_count", 0.0)),
            "attention_relational_graph_native_fallback_count": float(attention_relational_graph_summary.get("native_fallback_count", 0.0)),
            "dense_partial_graph_matched_real_fraction": float(dense_partial_graph_summary.get("matched_real_fraction", 0.0)),
            "dense_partial_graph_unconstrained_collision_count": float(dense_partial_graph_summary.get("unconstrained_collision_count", 0.0)),
            "dense_partial_graph_dustbin_count": float(dense_partial_graph_summary.get("dustbin_count", 0.0)),
            "dense_partial_graph_query_dustbin_count": float(dense_partial_graph_summary.get("query_dustbin_count", 0.0)),
            "dense_partial_graph_query_changed_count": float(dense_partial_graph_summary.get("query_changed_from_attention_count", 0.0)),
            "dense_partial_graph_native_injected_candidate_count": float(dense_partial_graph_summary.get("native_candidate_injected_count", 0.0)),
            "dense_partial_graph_native_fallback_count": float(dense_partial_graph_summary.get("native_fallback_count", 0.0)),
            "expert_hypothesis_selected_head": float(expert_hypothesis_summary.get("selected_head", -1)),
            "expert_hypothesis_selected_head_agreement": float(expert_hypothesis_summary.get("selected_head_agreement", 0.0)),
            "expert_hypothesis_selected_head_agreement_margin": float(expert_hypothesis_summary.get("selected_head_agreement_margin", 0.0)),
            "expert_hypothesis_query_changed_count": float(expert_hypothesis_summary.get("query_changed_from_attention_count", 0.0)),
            "expert_hypothesis_selected_attention_rank_mean": float(expert_hypothesis_summary.get("selected_attention_rank_mean", 0.0)),
            "expert_hypothesis_native_injected_candidate_count": float(expert_hypothesis_summary.get("native_candidate_injected_count", 0.0)),
            "expert_hypothesis_native_fallback_count": float(expert_hypothesis_summary.get("native_fallback_count", 0.0)),
            "parity_ok": float(parity_ok),
            "joint_native_cosine_source": float(
                F.cosine_similarity(src_joint_prepared, src_native_prepared, dim=1).mean().detach().cpu()
            ),
            "joint_native_cosine_target": float(
                F.cosine_similarity(trg_joint_prepared, trg_native_prepared, dim=1).mean().detach().cpu()
            ),
            "native_intra_cosine_source": _mean_off_diagonal_cosine(src_native_prepared),
            "joint_intra_cosine_source": _mean_off_diagonal_cosine(src_joint_prepared),
            "model_counts": oracle_counts,
            "attention_case_records": attention_case_records,
            "attention_relational_graph_audit": {
                "summary": attention_relational_graph_summary,
                "points": attention_relational_graph_audits or [],
            },
            "dense_partial_graph_audit": {
                "summary": dense_partial_graph_summary,
                "points": dense_partial_graph_audits or [],
            },
            "expert_hypothesis_audit": {
                "summary": expert_hypothesis_summary,
                "points": expert_hypothesis_audits or [],
            },
            "pre_softmax_channelwise_identity_audit": {
                "summary": pre_softmax_identity_summary,
                "points": pre_softmax_identity_audits or [],
            },
            "layer_routed_identity_audit": {
                "summary": layer_routed_identity_summary,
                "points": layer_routed_identity_audits or [],
            },
            "pre_single_stream_identity_audit": {
                "summary": layer_routed_identity_summary,
                "points": layer_routed_identity_audits or [],
            },
        }
    return (predictions, diagnostics) if return_diagnostics else predictions


def flux_fjsar_candidate_feature_batch(
    src_features: torch.Tensor,
    trg_features: torch.Tensor,
    source_points: Iterable[Sequence[float]],
    source_size: Sequence[int],
    target_size: Sequence[int],
    *,
    src_replay_state: dict[str, Any],
    trg_replay_state: dict[str, Any],
    blocks: Sequence[Any],
    interaction_mode: str = "exact",
    use_coordinate_bias: bool = False,
    candidate_topk: int = 20,
    include_identity_token_sketches: bool = False,
) -> dict[str, Any]:
    """Build annotation-free features for fixed mutual-attention candidates."""

    points = list(source_points)
    if not points:
        raise ValueError("candidate feature batch requires at least one source point")
    if interaction_mode not in {"exact", "calibrated"}:
        raise ValueError(
            "identity decodability audit currently supports exact/calibrated attention"
        )
    src_state = FluxReplayState.from_dict(src_replay_state)
    trg_state = FluxReplayState.from_dict(trg_replay_state)
    if src_state.global_block_index != trg_state.global_block_index:
        raise ValueError("identity decodability replay states use different blocks")

    with torch.no_grad():
        _src_joint, _trg_joint, attention = run_flux_joint_stack(
            blocks,
            src_state,
            trg_state,
            mode=interaction_mode,
            use_coordinate_bias=use_coordinate_bias,
        )
        src_cells = _native_cell_indices_for_points(
            points,
            source_size,
            src_state.image_height,
            src_state.image_width,
            attention["p_ab"].device,
        )
        mutual_attention = torch.sqrt(
            (attention["p_ab"].float() * attention["p_ba"].float().t()).clamp_min(0.0)
        )
        mutual_attention = torch.nan_to_num(
            mutual_attention,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        candidate_count = min(
            max(1, int(candidate_topk)),
            int(mutual_attention.shape[1]),
        )
        attention_values, candidate_cells = torch.topk(
            mutual_attention.index_select(0, src_cells),
            k=candidate_count,
            dim=1,
            sorted=True,
        )

        target_h, target_w = int(target_size[0]), int(target_size[1])
        cell_x = (candidate_cells % int(trg_state.image_width)).float()
        cell_y = torch.div(
            candidate_cells,
            int(trg_state.image_width),
            rounding_mode="floor",
        ).float()
        proposal_x = torch.round(
            (cell_x + 0.5) * float(target_w) / float(trg_state.image_width) - 0.5
        ).long().clamp_(0, target_w - 1)
        proposal_y = torch.round(
            (cell_y + 0.5) * float(target_h) / float(trg_state.image_height) - 0.5
        ).long().clamp_(0, target_h - 1)
        proposal_pixels = proposal_y * target_w + proposal_x

        internal = flux_candidate_internal_state_probe(
            blocks,
            src_state,
            trg_state,
            src_cells,
            candidate_cells,
            mode=interaction_mode,
            use_coordinate_bias=use_coordinate_bias,
            include_identity_token_sketches=include_identity_token_sketches,
        )
        native_sorted_scores, native_sorted_pixels = _chunked_descriptor_topk_scores_indices(
            src_features,
            trg_features,
            points,
            source_size,
            target_size,
            topk=candidate_count,
            candidate_indices=proposal_pixels,
        )
        matches = proposal_pixels.unsqueeze(2) == native_sorted_pixels.unsqueeze(1)
        if not bool(matches.any(dim=2).all()):
            raise RuntimeError("native control scores lost an attention candidate")
        native_scores = (
            matches.to(native_sorted_scores.dtype)
            * native_sorted_scores.unsqueeze(1)
        ).sum(dim=2)

        source_xy = torch.tensor(points, device=proposal_x.device, dtype=torch.float32)
        source_x = source_xy[:, 0].clamp(0, int(source_size[1]) - 1)
        source_y = source_xy[:, 1].clamp(0, int(source_size[0]) - 1)
        source_x_norm = source_x / float(max(1, int(source_size[1]) - 1))
        source_y_norm = source_y / float(max(1, int(source_size[0]) - 1))
        target_x_norm = proposal_x.float() / float(max(1, target_w - 1))
        target_y_norm = proposal_y.float() / float(max(1, target_h - 1))
        candidate_rank = torch.arange(
            candidate_count,
            device=proposal_x.device,
            dtype=torch.float32,
        ).reshape(1, -1).expand(len(points), -1)
        candidate_rank = candidate_rank / float(max(1, candidate_count - 1))
        proposal_attention = torch.stack(
            (
                attention_values,
                torch.log(attention_values.clamp_min(1e-30)),
                candidate_rank,
            ),
            dim=2,
        )
        geometry = torch.stack(
            (
                source_x_norm[:, None].expand(-1, candidate_count),
                source_y_norm[:, None].expand(-1, candidate_count),
                target_x_norm,
                target_y_norm,
                target_x_norm - source_x_norm[:, None],
                target_y_norm - source_y_norm[:, None],
            ),
            dim=2,
        )
    feature_groups = {
        name: value.detach().cpu().to(torch.float16)
        for name, value in internal["feature_groups"].items()
    }
    feature_groups.update({
        "proposal_attention": proposal_attention.detach().cpu().to(torch.float16),
        "native_control": native_scores.unsqueeze(2).detach().cpu().to(torch.float16),
        "geometry_control": geometry.detach().cpu().to(torch.float16),
    })
    family_names = dict(internal["feature_family_names"])
    family_names.update({
        "proposal_attention": [
            "mutual_attention",
            "log_mutual_attention",
            "attention_rank_normalized",
        ],
        "native_control": ["official_ditf_cosine_within_attention_candidates"],
        "geometry_control": [
            "source_x_normalized",
            "source_y_normalized",
            "target_x_normalized",
            "target_y_normalized",
            "normalized_dx",
            "normalized_dy",
        ],
    })
    return {
        "format_version": 1,
        "feature_groups": feature_groups,
        "feature_family_names": family_names,
        "candidate_cells": candidate_cells.detach().cpu().to(torch.int32),
        "candidate_pixels": proposal_pixels.detach().cpu().to(torch.int32),
        "attention_scores": attention_values.detach().cpu().float(),
        "source_cells": src_cells.detach().cpu().to(torch.int32),
        "metadata": {
            **internal["metadata"],
            "point_count": int(len(points)),
            "candidate_count": int(candidate_count),
            "source_size": [int(value) for value in source_size],
            "target_size": [int(value) for value in target_size],
            "gt_used_for_features": False,
            "gt_used_for_labels_only": False,
            "labels_present": False,
        },
    }


def flux_fjsar_identity_decodability_batch(
    src_features: torch.Tensor,
    trg_features: torch.Tensor,
    source_points: Iterable[Sequence[float]],
    source_size: Sequence[int],
    target_size: Sequence[int],
    *,
    src_replay_state: dict[str, Any],
    trg_replay_state: dict[str, Any],
    blocks: Sequence[Any],
    interaction_mode: str = "exact",
    use_coordinate_bias: bool = False,
    target_points: Iterable[Sequence[float]],
    pck_threshold: float,
    candidate_topk: int = 20,
) -> dict[str, Any]:
    """Add diagnostic PCK labels to an annotation-free candidate batch."""

    points = list(source_points)
    targets = list(target_points)
    if len(points) != len(targets):
        raise ValueError("identity decodability source/target points must align")
    batch = flux_fjsar_candidate_feature_batch(
        src_features,
        trg_features,
        points,
        source_size,
        target_size,
        src_replay_state=src_replay_state,
        trg_replay_state=trg_replay_state,
        blocks=blocks,
        interaction_mode=interaction_mode,
        use_coordinate_bias=use_coordinate_bias,
        candidate_topk=candidate_topk,
        include_identity_token_sketches=True,
    )
    target_h, target_w = map(int, target_size)
    proposal_pixels = batch["candidate_pixels"].long()
    proposal_x = (proposal_pixels % target_w).float()
    proposal_y = torch.div(proposal_pixels, target_w, rounding_mode="floor").float()
    target_xy = torch.tensor(targets, dtype=torch.float32)
    distances = torch.sqrt(
        (proposal_x - target_xy[:, 0, None]).square()
        + (proposal_y - target_xy[:, 1, None]).square()
    )
    batch["candidate_hits"] = distances <= 0.1 * float(pck_threshold)
    batch["metadata"].update({
        "pck_threshold": float(pck_threshold),
        "gt_used_for_labels_only": True,
        "labels_present": True,
        "probe_is_matcher": False,
    })
    return batch


def flux_fjsar_dump_candidates(
    src_features: torch.Tensor,
    trg_features: torch.Tensor,
    source_points: Iterable[Sequence[float]],
    source_size: Sequence[int],
    target_size: Sequence[int],
    *,
    src_replay_state: dict[str, Any],
    trg_replay_state: dict[str, Any],
    blocks: Sequence[Any],
    interaction_mode: str = "exact",
    use_coordinate_bias: bool = False,
    target_points: Iterable[Sequence[float]] | None = None,
    pck_threshold: float | None = None,
    candidate_topk: int = 20,
    candidate_descriptor_audit: bool = False,
    method_descriptor_audit_mode: str | None = None,
    transport_lift_branch_audit: bool = False,
    attention_flow_audit: bool = False,
    attention_flow_radius: int = 2,
    attention_kernel_audit: bool = False,
    attention_kernel_radius: int = 2,
    attention_kernel_topk: Sequence[int] = (1, 5, 20),
    basin_identity_audit: bool = False,
    basin_identity_topk: int = 20,
    basin_identity_radius: int = 2,
    basin_identity_rank_topk: Sequence[int] = (1, 3, 5, 10, 20),
    kernel_featureization_audit: bool = False,
    kernel_featureization_ranks: Sequence[int] = (32, 64),
    kernel_featureization_weights: Sequence[float] = (0.5, 1.0),
    kernel_featureization_radius: int = 2,
    kernel_featureization_topk: Sequence[int] = (1, 5, 20),
    residual_readout_audit: bool = False,
    residual_readout_topk: Sequence[int] = (1, 3, 5, 10, 20),
    latent_expert_audit: bool = False,
    latent_expert_topk: Sequence[int] = (1, 3, 5, 10, 20),
    candidate_clamped_causal_replay_audit: bool = False,
    candidate_clamped_causal_replay_topk: Sequence[int] = (1, 3, 5, 10, 20),
    causal_release_block: Any | None = None,
    counterfactual_fingerprint_audit: bool = False,
    counterfactual_fingerprint_topk: Sequence[int] = (1, 3, 5, 10, 20),
    counterfactual_fingerprint_scales: Sequence[float] = (0.75, 1.0, 1.25),
    persistent_candidate_slot_replay_audit: bool = False,
    persistent_candidate_slot_replay_topk: Sequence[int] = (1, 3, 5, 10, 20),
    persistent_candidate_slot_replay_chunk: int = 1,
    persistent_candidate_slot_replay_blocks: Sequence[Any] | None = None,
    local_relational_identity_audit: bool = False,
    local_relational_radius: int = 2,
    dense_candidate_edge_audit: bool = False,
    dense_candidate_edge_radius: int = 1,
    dense_transport_consistency_audit: bool = False,
    dense_transport_topk: Sequence[int] = (1, 5, 20),
    candidate_field_consistency_audit: bool = False,
    candidate_field_topm: int = 20,
    candidate_field_source: str = "native_basin",
    anchor_topology_audit: bool = False,
    multilayer_identity_audit: bool = False,
    multilayer_descriptor_maps: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
    geometry_radius: int = 2,
    geometry_strength: float = 0.5,
    trajectory_replay_states: dict[str, Any] | None = None,
    trajectory_block_modules: dict[str, Any] | None = None,
    trajectory_blocks: Sequence[int] = (),
    transport_factorization_audit: bool = False,
    transport_factorization_radius: int = 2,
    transport_factorization_basis_radius: int = 0,
    src_ada: Any | None = None,
    trg_ada: Any | None = None,
    discard_channels: Sequence[int] = (),
    calibration: dict[str, Any] | None = None,
    operator_manifold_audit: bool = False,
) -> list[dict[str, Any]]:
    """Dump lightweight attention proposal diagnostics."""

    points = list(source_points)
    gt_points = list(target_points) if target_points is not None else None
    if not points:
        return []
    if gt_points is not None and len(gt_points) != len(points):
        raise ValueError("target_points must align with source_points")
    src_state = FluxReplayState.from_dict(src_replay_state)
    trg_state = FluxReplayState.from_dict(trg_replay_state)
    if src_state.global_block_index != trg_state.global_block_index:
        raise ValueError("source and target replay caches use different starting blocks")
    if len(blocks) not in (1, 2):
        raise ValueError("FJSAR candidate dump requires one or two replay blocks")
    with torch.no_grad():
        if interaction_mode == "cross_only":
            _src_joint_sequence, _trg_joint_sequence, attention = run_flux_cross_only_stack(
                blocks,
                src_state,
                trg_state,
            )
        elif interaction_mode == "identity_preserving":
            _src_joint_sequence, _trg_joint_sequence, attention = run_flux_identity_preserving_stack(
                blocks,
                src_state,
                trg_state,
            )
        elif interaction_mode == "balanced_transport":
            _src_joint_sequence, _trg_joint_sequence, attention = run_flux_balanced_transport_stack(
                blocks,
                src_state,
                trg_state,
            )
        elif interaction_mode == "qk_identity":
            _src_joint_sequence, _trg_joint_sequence, attention = run_flux_qk_identity_stack(
                blocks,
                src_state,
                trg_state,
            )
        elif interaction_mode == "trajectory":
            _src_joint_sequence, _trg_joint_sequence, attention = run_flux_cross_only_stack(
                blocks,
                src_state,
                trg_state,
            )
        elif interaction_mode == "geometry_consistent":
            _src_joint_sequence, _trg_joint_sequence, attention = run_flux_geometry_consistent_stack(
                blocks,
                src_state,
                trg_state,
                radius=geometry_radius,
                strength=geometry_strength,
            )
        else:
            _src_joint_sequence, _trg_joint_sequence, attention = run_flux_joint_stack(
                blocks,
                src_state,
                trg_state,
                mode=interaction_mode,
                use_coordinate_bias=use_coordinate_bias,
            )
        method_descriptor_src = None
        method_descriptor_trg = None
        transport_lift_branch_descriptors = None
        operator_manifold_audits = None
        trajectory_audits = None
        if trajectory_blocks:
            trajectory_layers = _trajectory_attention_layers(
                trajectory_replay_states,
                trajectory_block_modules,
                trajectory_blocks,
                fallback_src_state=src_state,
                fallback_trg_state=trg_state,
                fallback_attention=attention,
            )
            _trajectory_ranked_cells, trajectory_audits, _trajectory_summary = _cross_attention_trajectory_rankings(
                trajectory_layers,
                int(src_state.global_block_index) + 1,
                points,
                source_size,
                target_size,
                candidate_topk=candidate_topk,
                target_points=gt_points,
                pck_threshold=pck_threshold,
            )
        if operator_manifold_audit:
            if src_ada is None or trg_ada is None:
                raise ValueError("operator_manifold_audit requires src_ada and trg_ada")
            src_native_sequence = run_flux_native_stack(blocks, src_state)
            trg_native_sequence = run_flux_native_stack(blocks, trg_state)
            src_native_prepared = _prepare_replay_tokens(
                native_image_tokens(src_native_sequence, src_state),
                src_ada,
                discard_channels=discard_channels,
                calibration=calibration,
            )[0]
            trg_native_prepared = _prepare_replay_tokens(
                native_image_tokens(trg_native_sequence, trg_state),
                trg_ada,
                discard_channels=discard_channels,
                calibration=calibration,
            )[0]
            src_joint_prepared = _prepare_replay_tokens(
                native_image_tokens(_src_joint_sequence, src_state),
                src_ada,
                discard_channels=discard_channels,
                calibration=calibration,
            )[0]
            trg_joint_prepared = _prepare_replay_tokens(
                native_image_tokens(_trg_joint_sequence, trg_state),
                trg_ada,
                discard_channels=discard_channels,
                calibration=calibration,
            )[0]
            operator_manifold_audits = _operator_manifold_audit_for_points(
                src_native_prepared,
                trg_native_prepared,
                src_joint_prepared,
                trg_joint_prepared,
                points,
                source_size,
                gt_points,
                target_size,
                src_state,
                trg_state,
            )
        if method_descriptor_audit_mode is not None:
            if method_descriptor_audit_mode == "attention_signature":
                method_descriptor_src, method_descriptor_trg = _attention_signature_descriptors(
                    src_features,
                    trg_features,
                    attention,
                )
            elif method_descriptor_audit_mode == "part_sharpen":
                method_descriptor_src, method_descriptor_trg = _part_common_sharpen_descriptors(
                    src_features,
                    trg_features,
                    attention,
                )
            elif method_descriptor_audit_mode == "spectral_identity":
                method_descriptor_src, method_descriptor_trg = _spectral_attention_identity_descriptors(
                    src_features,
                    trg_features,
                    attention,
                )
            elif method_descriptor_audit_mode == "filtered_spectral_kernel":
                (
                    method_descriptor_src,
                    method_descriptor_trg,
                    _spectral_diagnostics,
                ) = filtered_spectral_kernel_feature_maps(
                    src_features,
                    trg_features,
                    attention,
                    src_state,
                    trg_state,
                )
            elif method_descriptor_audit_mode == "transport_lift":
                method_descriptor_src, method_descriptor_trg = _local_transport_lift_descriptors(
                    src_features,
                    trg_features,
                    attention,
                )
            elif method_descriptor_audit_mode == "basin_contrastive_identity":
                method_descriptor_src, method_descriptor_trg = _basin_contrastive_identity_descriptors(
                    src_features,
                    trg_features,
                    attention,
                )
            elif method_descriptor_audit_mode == "attention_isometry":
                method_descriptor_src, method_descriptor_trg = _attention_guided_isometry_descriptors(
                    src_features,
                    trg_features,
                    attention,
                )
            elif method_descriptor_audit_mode == "qk_identity_attention":
                if src_ada is None or trg_ada is None:
                    raise ValueError("qk_identity_attention method descriptor audit requires src_ada and trg_ada")
                method_descriptor_src = _token_matrix_to_map(
                    F.normalize(
                        _prepare_replay_tokens(
                            native_image_tokens(_src_joint_sequence, src_state),
                            src_ada,
                            discard_channels=discard_channels,
                            calibration=calibration,
                        )[0],
                        dim=1,
                        eps=1e-12,
                    ),
                    src_state.image_height,
                    src_state.image_width,
                )
                method_descriptor_trg = _token_matrix_to_map(
                    F.normalize(
                        _prepare_replay_tokens(
                            native_image_tokens(_trg_joint_sequence, trg_state),
                            trg_ada,
                            discard_channels=discard_channels,
                            calibration=calibration,
                        )[0],
                        dim=1,
                        eps=1e-12,
                    ),
                    trg_state.image_height,
                    trg_state.image_width,
                )
            else:
                raise ValueError(
                    f"unsupported method descriptor audit mode: {method_descriptor_audit_mode}"
                )
        if transport_lift_branch_audit:
            transport_lift_branch_descriptors = {
                "native_only": _local_transport_lift_descriptors(
                    src_features,
                    trg_features,
                    attention,
                    include_native=True,
                    include_outgoing=False,
                    include_incoming=False,
                ),
                "out_only": _local_transport_lift_descriptors(
                    src_features,
                    trg_features,
                    attention,
                    include_native=False,
                    include_outgoing=True,
                    include_incoming=False,
                ),
                "in_only": _local_transport_lift_descriptors(
                    src_features,
                    trg_features,
                    attention,
                    include_native=False,
                    include_outgoing=False,
                    include_incoming=True,
                ),
                "no_native": _local_transport_lift_descriptors(
                    src_features,
                    trg_features,
                    attention,
                    include_native=False,
                    include_outgoing=True,
                    include_incoming=True,
                ),
                "full": _local_transport_lift_descriptors(
                    src_features,
                    trg_features,
                    attention,
                    include_native=True,
                    include_outgoing=True,
                    include_incoming=True,
                ),
            }
        return _fjsar_attention_candidate_records(
            src_features,
            trg_features,
            attention,
            points,
            source_size,
            target_size,
            src_state,
            trg_state,
            topk=candidate_topk,
            target_points=gt_points,
            pck_threshold=pck_threshold,
            candidate_descriptor_audit=candidate_descriptor_audit,
            method_descriptor_audit_name=method_descriptor_audit_mode,
            method_descriptor_src=method_descriptor_src,
            method_descriptor_trg=method_descriptor_trg,
            transport_lift_branch_descriptors=transport_lift_branch_descriptors,
            attention_flow_audit=attention_flow_audit,
            attention_flow_radius=attention_flow_radius,
            attention_kernel_audit=attention_kernel_audit,
            attention_kernel_radius=attention_kernel_radius,
            attention_kernel_topk=attention_kernel_topk,
            basin_identity_audit=basin_identity_audit,
            basin_identity_topk=basin_identity_topk,
            basin_identity_radius=basin_identity_radius,
            basin_identity_rank_topk=basin_identity_rank_topk,
            kernel_featureization_audit=kernel_featureization_audit,
            kernel_featureization_ranks=kernel_featureization_ranks,
            kernel_featureization_weights=kernel_featureization_weights,
            kernel_featureization_radius=kernel_featureization_radius,
            kernel_featureization_topk=kernel_featureization_topk,
            residual_readout_audit=residual_readout_audit,
            residual_readout_topk=residual_readout_topk,
            latent_expert_audit=latent_expert_audit,
            latent_expert_topk=latent_expert_topk,
            candidate_clamped_causal_replay_audit=(
                candidate_clamped_causal_replay_audit
            ),
            candidate_clamped_causal_replay_topk=(
                candidate_clamped_causal_replay_topk
            ),
            causal_release_block=causal_release_block,
            counterfactual_fingerprint_audit=(
                counterfactual_fingerprint_audit
            ),
            counterfactual_fingerprint_topk=(
                counterfactual_fingerprint_topk
            ),
            counterfactual_fingerprint_scales=(
                counterfactual_fingerprint_scales
            ),
            persistent_candidate_slot_replay_audit=(
                persistent_candidate_slot_replay_audit
            ),
            persistent_candidate_slot_replay_topk=(
                persistent_candidate_slot_replay_topk
            ),
            persistent_candidate_slot_replay_chunk=(
                persistent_candidate_slot_replay_chunk
            ),
            persistent_candidate_slot_replay_blocks=(
                persistent_candidate_slot_replay_blocks
            ),
            local_relational_identity_audit=local_relational_identity_audit,
            local_relational_radius=local_relational_radius,
            dense_candidate_edge_audit=dense_candidate_edge_audit,
            dense_candidate_edge_radius=dense_candidate_edge_radius,
            dense_transport_consistency_audit=dense_transport_consistency_audit,
            dense_transport_topk=dense_transport_topk,
            candidate_field_consistency_audit=candidate_field_consistency_audit,
            candidate_field_topm=candidate_field_topm,
            candidate_field_source=candidate_field_source,
            anchor_topology_audit=anchor_topology_audit,
            multilayer_identity_audit=multilayer_identity_audit,
            multilayer_descriptor_maps=multilayer_descriptor_maps,
            blocks=blocks,
            interaction_mode=interaction_mode,
            use_coordinate_bias=use_coordinate_bias,
            transport_factorization_audit=transport_factorization_audit,
            transport_factorization_radius=transport_factorization_radius,
            transport_factorization_basis_radius=transport_factorization_basis_radius,
            operator_manifold_audits=operator_manifold_audits,
            trajectory_identity_audits=trajectory_audits,
        )

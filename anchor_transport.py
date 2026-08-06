"""Label-free anchor certification and local candidate transport primitives.

All functions in this module consume only source points, frozen matcher
predictions, reverse frozen predictions, and a fixed candidate set.  Ground
truth belongs exclusively in the caller's post-hoc audit.
"""

from __future__ import annotations

import torch


def _points(name: str, value: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[1] != 2:
        raise ValueError(f"{name} must be [point,2]")
    return torch.nan_to_num(value.float(), nan=0.0, posinf=0.0, neginf=0.0)


def certified_anchor_mask(
    source_points: torch.Tensor,
    baseline_predictions: torch.Tensor,
    reverse_predictions: torch.Tensor,
    attention_top1_predictions: torch.Tensor,
    *,
    source_cell_diagonal: float,
    target_cell_diagonal: float,
    cycle_radius_cells: float,
    agreement_radius_cells: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Certify anchors from frozen forward/backward and cross-head agreement.

    The returned mask is intentionally independent of correspondence labels.
    Cell-normalized distances make the rule transferable across image sizes.
    """

    source = _points("source_points", source_points)
    baseline = _points("baseline_predictions", baseline_predictions)
    reverse = _points("reverse_predictions", reverse_predictions)
    attention = _points("attention_top1_predictions", attention_top1_predictions)
    if not (
        source.shape == baseline.shape == reverse.shape == attention.shape
    ):
        raise ValueError("anchor point tensors must share shape")
    if min(
        float(source_cell_diagonal),
        float(target_cell_diagonal),
        float(cycle_radius_cells),
        float(agreement_radius_cells),
    ) <= 0.0:
        raise ValueError("anchor cell scales and radii must be positive")
    cycle_error = (reverse - source).norm(dim=1) / float(source_cell_diagonal)
    agreement_error = (baseline - attention).norm(dim=1) / float(target_cell_diagonal)
    anchors = (
        (cycle_error <= float(cycle_radius_cells))
        & (agreement_error <= float(agreement_radius_cells))
    )
    return anchors, cycle_error, agreement_error


def local_affine_transport(
    source_points: torch.Tensor,
    anchor_target_predictions: torch.Tensor,
    candidate_predictions: torch.Tensor,
    anchors: torch.Tensor,
    *,
    neighbor_count: int,
    minimum_anchors: int,
    target_cell_diagonal: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rank candidates using a leave-one-out local affine prediction.

    A query never anchors itself.  The transform is estimated only from
    certified source-to-target predictions in its nearest source-point
    neighbourhood.  Invalid/undersupported queries are returned with rank
    ``-1`` and infinite support distance so callers can exactly retain a
    baseline fallback.
    """

    source = _points("source_points", source_points)
    targets = _points("anchor_target_predictions", anchor_target_predictions)
    if source.shape != targets.shape:
        raise ValueError("source and anchor target points must align")
    if not isinstance(candidate_predictions, torch.Tensor) or candidate_predictions.ndim != 3:
        raise ValueError("candidate_predictions must be [point,candidate,2]")
    candidates = torch.nan_to_num(
        candidate_predictions.float(), nan=0.0, posinf=0.0, neginf=0.0
    )
    if candidates.shape[0] != source.shape[0] or candidates.shape[2] != 2:
        raise ValueError("candidate predictions must align with source points")
    if not isinstance(anchors, torch.Tensor) or anchors.shape != source.shape[:1]:
        raise ValueError("anchors must be [point]")
    if int(neighbor_count) < int(minimum_anchors) or int(minimum_anchors) < 3:
        raise ValueError("local affine transport needs at least three anchors")
    if float(target_cell_diagonal) <= 0.0:
        raise ValueError("target_cell_diagonal must be positive")

    point_count = int(source.shape[0])
    ranks = torch.full((point_count,), -1, dtype=torch.long, device=source.device)
    valid = torch.zeros(point_count, dtype=torch.bool, device=source.device)
    support = torch.full((point_count,), float("inf"), dtype=torch.float32, device=source.device)
    anchor_mask = anchors.to(device=source.device, dtype=torch.bool)
    for query in range(point_count):
        distances = (source - source[query]).square().sum(dim=1)
        order = torch.argsort(distances)
        selected = order[(order != query) & anchor_mask.index_select(0, order)]
        selected = selected[: int(neighbor_count)]
        if int(selected.numel()) < int(minimum_anchors):
            continue
        design = torch.cat((
            source.index_select(0, selected),
            torch.ones((int(selected.numel()), 1), device=source.device),
        ), dim=1)
        if int(torch.linalg.matrix_rank(design)) < 3:
            continue
        solution = torch.linalg.lstsq(
            design,
            targets.index_select(0, selected),
        ).solution
        prediction = torch.cat((
            source[query],
            torch.ones(1, device=source.device),
        )) @ solution
        candidate_distance = (candidates[query] - prediction).square().sum(dim=1).sqrt()
        rank = candidate_distance.argmin()
        ranks[query] = rank
        valid[query] = True
        support[query] = candidate_distance[rank] / float(target_cell_diagonal)
    return ranks, valid, support


def baseline_preserving_transport_ranks(
    transport_ranks: torch.Tensor,
    transport_valid: torch.Tensor,
    transport_support_cells: torch.Tensor,
    *,
    transport_radius_cells: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return attention ranks only where local transport has support.

    ``-1`` means preserve the baseline exactly.  The function never receives
    a target label and does not decide a fallback from ground truth.
    """

    if not (
        isinstance(transport_ranks, torch.Tensor)
        and isinstance(transport_valid, torch.Tensor)
        and isinstance(transport_support_cells, torch.Tensor)
    ):
        raise TypeError("transport ranks, validity, and support must be tensors")
    if not (
        transport_ranks.shape == transport_valid.shape == transport_support_cells.shape
    ):
        raise ValueError("transport tensors must share shape")
    if float(transport_radius_cells) <= 0.0:
        raise ValueError("transport_radius_cells must be positive")
    switched = (
        transport_valid.to(dtype=torch.bool)
        & (transport_ranks >= 0)
        & torch.isfinite(transport_support_cells)
        & (transport_support_cells <= float(transport_radius_cells))
    )
    selected = torch.full_like(transport_ranks, -1, dtype=torch.long)
    selected[switched] = transport_ranks[switched].long()
    return selected, switched

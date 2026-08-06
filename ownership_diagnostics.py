"""Backbone-agnostic candidate-ownership diagnostics.

Ground truth is used only after ranking to label candidate coverage. None of
the helpers in this module change similarities, proposal ranks or predictions.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math
from typing import Iterable

import torch


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
        return sum(self.pair_scores) / len(self.pair_scores) if self.pair_scores else 0.0

    @property
    def per_point(self) -> float:
        return self.correct / self.total if self.total else 0.0


def pck_hits(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
    alpha: float = 0.1,
) -> torch.Tensor:
    if threshold <= 0:
        raise ValueError("PCK threshold must be positive")
    return torch.linalg.vector_norm(predictions.float() - targets.float(), dim=1) < alpha * threshold


def _random_union_hit_probability(population: int, positives: int, draws: int) -> float:
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
    candidate_points: torch.Tensor,
    gt_points: torch.Tensor,
    threshold: float,
    ks: Iterable[int],
    baseline_indices: torch.Tensor | None = None,
) -> list[dict[str, int | float]]:
    """Measure ownership gaps under overlap, budget and random controls."""
    if scores.ndim != 2 or scores.shape[0] != gt_points.shape[0]:
        raise ValueError("scores and gt_points must describe the same source points")
    num_sources, population = scores.shape
    candidate_points = candidate_points.to(device=scores.device, dtype=torch.float32)
    if candidate_points.shape != (population, 2):
        raise ValueError(
            f"candidate_points must have shape ({population}, 2), got {tuple(candidate_points.shape)}"
        )
    ks = tuple(sorted(set(int(k) for k in ks)))
    if not ks or min(ks) < 1:
        raise ValueError("candidate K values must be positive")

    max_rank = min(max(ks) * num_sources, population)
    ranked = scores.topk(max_rank, dim=1).indices
    if baseline_indices is not None:
        baseline_indices = baseline_indices.to(device=scores.device, dtype=torch.long).flatten()
        if baseline_indices.shape[0] != num_sources:
            raise ValueError("baseline_indices must contain one index per source point")
        ranked = ranked.clone()
        for source_index, baseline_index in enumerate(baseline_indices):
            matches = torch.nonzero(ranked[source_index] == baseline_index, as_tuple=False).flatten()
            if matches.numel():
                match_index = int(matches[0])
                ranked[source_index, 0], ranked[source_index, match_index] = (
                    ranked[source_index, match_index].clone(),
                    ranked[source_index, 0].clone(),
                )
            else:
                ranked[source_index, 1:] = ranked[source_index, :-1].clone()
                ranked[source_index, 0] = baseline_index

    gt_points = gt_points.to(device=scores.device, dtype=torch.float32)
    patch_hits_all_gt = torch.cdist(candidate_points, gt_points) < 0.1 * float(threshold)
    rows: list[dict[str, int | float]] = []
    for owner_index in range(num_sources):
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
            other_hit_mask = (
                patch_hits_owner[other_candidates]
                if other_candidates.numel()
                else torch.zeros(0, dtype=torch.bool, device=scores.device)
            )
            other_hit = bool(other_hit_mask.any())
            strict_other_hit = bool(
                other_candidates.numel()
                and (
                    other_hit_mask
                    & ~patch_hits_all_gt[other_candidates][:, other_gt_columns].any(dim=1)
                ).any()
            )
            global_hit = owner_hit or other_hit
            strict_global_hit = owner_hit or strict_other_hit
            global_unique = torch.unique(local)
            unique_budget = int(global_unique.numel())
            budget_owner_hit = bool(patch_hits_owner[ranked[owner_index, :unique_budget]].any())
            source_support = int(patch_hits_owner[local].any(dim=1).sum())

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
            row[f"random_union_expected_hit@{requested_k}"] = _random_union_hit_probability(
                population, positive_count, unique_budget
            )
        rows.append(row)
    return rows


def empty_counts(ks: tuple[int, ...]) -> dict[int, dict[str, int | float]]:
    return {k: defaultdict(int) for k in ks}


def update_counts(
    counts: dict[int, dict[str, int | float]],
    rows: list[dict[str, int | float]],
    baseline_hits: torch.Tensor,
    ks: tuple[int, ...],
) -> None:
    for row, baseline_hit in zip(rows, baseline_hits.bool().cpu().tolist()):
        for k in ks:
            values = counts[k]
            owner = row[f"owner_candidate_hit@{k}"]
            union = row[f"global_union_candidate_hit@{k}"]
            strict_other = row[f"strict_other_source_candidate_hit@{k}"]
            source_count = row[f"proposal_source_count@{k}"]
            values["points"] += 1
            values["owner"] += owner
            values["other"] += row[f"other_source_candidate_hit@{k}"]
            values["global"] += union
            values["strict_other"] += strict_other
            values["strict_global"] += row[f"strict_global_union_candidate_hit@{k}"]
            values["budget_owner"] += row[f"budget_matched_owner_candidate_hit@{k}"]
            values["global_not_budget"] += row[f"global_not_budget_owner_hit@{k}"]
            values["strict_global_not_budget"] += row[f"strict_global_not_budget_owner_hit@{k}"]
            values["unique_candidates"] += row[f"global_unique_candidate_count@{k}"]
            values["random_expected"] += row[f"random_union_expected_hit@{k}"]
            if union:
                values["global_hits"] += 1
                values["source_support_on_global"] += source_count
                values["multi_source_global"] += int(source_count >= 2)
            if not baseline_hit:
                values["failures"] += 1
                values["failure_owner"] += owner
                values["failure_global"] += union
                values["failure_transferable"] += int(union and not owner)
                values["failure_strict_transferable"] += int(strict_other and not owner)
                values["failure_budget_owner"] += row[f"budget_matched_owner_candidate_hit@{k}"]
                values["failure_global_not_budget"] += row[f"global_not_budget_owner_hit@{k}"]
                values["failure_strict_global_not_budget"] += row[
                    f"strict_global_not_budget_owner_hit@{k}"
                ]
                values["failure_random_expected"] += row[f"random_union_expected_hit@{k}"]
                if union:
                    values["failure_global_hits"] += 1
                    values["failure_source_support_on_global"] += source_count
                    values["failure_multi_source_global"] += int(source_count >= 2)


def ratios(
    counts: dict[int, dict[str, int | float]], ks: tuple[int, ...]
) -> dict[str, dict[str, float | int]]:
    output: dict[str, dict[str, float | int]] = {}
    for k in ks:
        values = counts[k]
        points = max(values["points"], 1)
        failures = max(values["failures"], 1)
        global_hits = max(values["global_hits"], 1)
        failure_global_hits = max(values["failure_global_hits"], 1)
        output[str(k)] = {
            "point_count": values["points"],
            "baseline_failure_count": values["failures"],
            "owner_candidate_recall": values["owner"] / points,
            "other_source_transfer": values["other"] / points,
            "global_union_recall": values["global"] / points,
            "global_minus_owner": (values["global"] - values["owner"]) / points,
            "strict_other_source_recall": values["strict_other"] / points,
            "strict_global_union_recall": values["strict_global"] / points,
            "budget_matched_owner_recall": values["budget_owner"] / points,
            "global_minus_budget_matched_owner": (values["global"] - values["budget_owner"]) / points,
            "global_not_budget_matched_owner_rate": values["global_not_budget"] / points,
            "strict_global_not_budget_matched_owner_rate": values["strict_global_not_budget"] / points,
            "mean_unique_global_candidates": values["unique_candidates"] / points,
            "random_union_expected_recall": values["random_expected"] / points,
            "global_excess_over_random": (values["global"] - values["random_expected"]) / points,
            "mean_proposal_source_count_on_global_hits": values["source_support_on_global"] / global_hits,
            "multi_source_rate_on_global_hits": values["multi_source_global"] / global_hits,
            "failure_owner_candidate_recall": values["failure_owner"] / failures,
            "failure_global_union_recall": values["failure_global"] / failures,
            "failure_transferable_rate": values["failure_transferable"] / failures,
            "failure_strict_transferable_rate": values["failure_strict_transferable"] / failures,
            "failure_budget_matched_owner_recall": values["failure_budget_owner"] / failures,
            "failure_global_not_budget_matched_owner_rate": values["failure_global_not_budget"] / failures,
            "failure_strict_global_not_budget_matched_owner_rate": (
                values["failure_strict_global_not_budget"] / failures
            ),
            "failure_random_union_expected_recall": values["failure_random_expected"] / failures,
            "failure_global_excess_over_random": (
                values["failure_global"] - values["failure_random_expected"]
            ) / failures,
            "failure_mean_proposal_source_count_on_global_hits": (
                values["failure_source_support_on_global"] / failure_global_hits
            ),
            "failure_multi_source_rate_on_global_hits": (
                values["failure_multi_source_global"] / failure_global_hits
            ),
        }
    return output


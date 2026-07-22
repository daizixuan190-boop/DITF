"""Evaluate top-M joint candidate ownership with training-free graph matching.

The evaluator preserves DiTF post-AdaLN + channel discard and reuses the v2
relational/geometric score only as a unary proposal generator. Candidate
ownership is selected jointly with source-graph distance compatibility and a
soft target-collision penalty. SPair target annotations remain outside all
candidate construction, scoring, and optimization code.
"""

import argparse
from typing import Any

import torch

from eval_spair_relational_ownership import main as run_evaluation
from eval_spair_relational_ownership_v2 import (
    build_v2_parser,
    geometry_aware_relational_ownership,
)


def build_topm_candidates(
    scores: torch.Tensor,
    base_columns: torch.Tensor,
    topm: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Take unary top-M candidates and force each baseline candidate to remain."""
    num_candidates = int(scores.shape[1])
    k = min(max(int(topm), 1), num_candidates)
    values, columns = torch.topk(scores, k=k, dim=1)
    contains_base = torch.any(columns == base_columns.unsqueeze(1), dim=1)
    missing = torch.nonzero(~contains_base, as_tuple=False).squeeze(1)
    if int(missing.numel()) > 0:
        columns[missing, -1] = base_columns[missing]
        values[missing, -1] = scores[missing, base_columns[missing]]
    order = torch.argsort(values, dim=1, descending=True)
    columns = torch.gather(columns, 1, order)
    values = torch.gather(values, 1, order)
    base_positions = torch.argmax((columns == base_columns.unsqueeze(1)).long(), dim=1)
    return columns, values, base_positions


def build_joint_potentials(
    src_xy: torch.Tensor,
    candidate_xy: torch.Tensor,
    candidate_columns: torch.Tensor,
    base_positions: torch.Tensor,
    src_threshold: float,
    trg_threshold: float,
    args,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build complete collision edges and kNN structural compatibility edges."""
    num_points = int(src_xy.shape[0])
    topm = int(candidate_xy.shape[1])
    device = src_xy.device
    if num_points < 2:
        return (
            torch.empty((0, 2), dtype=torch.long, device=device),
            torch.empty((0, topm, topm), dtype=torch.float32, device=device),
            torch.empty((0,), dtype=torch.bool, device=device),
        )

    pairs = torch.combinations(torch.arange(num_points, device=device), r=2)
    source_distances = torch.cdist(src_xy.float(), src_xy.float())
    source_distances = source_distances / max(float(src_threshold), 1e-6)
    neighbor_count = min(max(int(args.jgm_graph_k), 1), num_points - 1)
    knn_distances = source_distances.clone()
    knn_distances.fill_diagonal_(float("inf"))
    neighbors = torch.topk(knn_distances, k=neighbor_count, dim=1, largest=False).indices
    adjacency = torch.zeros((num_points, num_points), dtype=torch.bool, device=device)
    adjacency.scatter_(1, neighbors, True)
    adjacency = adjacency | adjacency.transpose(0, 1)
    structural = adjacency[pairs[:, 0], pairs[:, 1]]
    edge_source_distance = source_distances[pairs[:, 0], pairs[:, 1]]
    structural = structural & (edge_source_distance >= float(args.jgm_min_source_separation))

    first_xy = candidate_xy[pairs[:, 0]].unsqueeze(2)
    second_xy = candidate_xy[pairs[:, 1]].unsqueeze(1)
    target_distance = torch.linalg.norm(first_xy - second_xy, dim=3)
    target_distance = target_distance / max(float(trg_threshold), 1e-6)
    residual = torch.abs(target_distance - edge_source_distance[:, None, None])
    residual = torch.clamp(residual, max=float(args.jgm_distance_truncation))
    compatibility = torch.exp(-residual / max(float(args.jgm_distance_sigma), 1e-6))

    edge_index = torch.arange(int(pairs.shape[0]), device=device)
    base_first = base_positions[pairs[:, 0]]
    base_second = base_positions[pairs[:, 1]]
    base_compatibility = compatibility[edge_index, base_first, base_second]
    structural_delta = compatibility - base_compatibility[:, None, None]

    same_candidate = (
        candidate_columns[pairs[:, 0]].unsqueeze(2)
        == candidate_columns[pairs[:, 1]].unsqueeze(1)
    )
    near_collision = (
        target_distance < float(args.jgm_collision_radius)
    ) & (
        edge_source_distance[:, None, None]
        > float(args.jgm_collision_source_ratio) * float(args.jgm_collision_radius)
    )
    collision = (same_candidate | near_collision).float()
    base_collision = collision[edge_index, base_first, base_second]
    collision_delta = collision - base_collision[:, None, None]

    potentials = (
        float(args.jgm_pairwise_weight)
        * structural.float()[:, None, None]
        * structural_delta
        - float(args.jgm_collision_weight) * collision_delta
    )
    return pairs, potentials.float(), structural


def max_product_beliefs(
    unary: torch.Tensor,
    pairs: torch.Tensor,
    potentials: torch.Tensor,
    steps: int,
    damping: float,
) -> torch.Tensor:
    """Vectorized loopy max-product messages over pairwise candidate states."""
    num_points, topm = unary.shape
    num_edges = int(pairs.shape[0])
    if num_edges == 0:
        return unary.clone()
    first = pairs[:, 0]
    second = pairs[:, 1]
    sender = torch.cat((first, second), dim=0)
    receiver = torch.cat((second, first), dim=0)
    directed_potentials = torch.cat((potentials, potentials.transpose(1, 2)), dim=0)
    reverse = torch.cat(
        (
            torch.arange(num_edges, 2 * num_edges, device=unary.device),
            torch.arange(0, num_edges, device=unary.device),
        ),
        dim=0,
    )
    messages = torch.zeros((2 * num_edges, topm), device=unary.device, dtype=unary.dtype)
    keep = min(max(float(damping), 0.0), 0.99)
    for _ in range(max(int(steps), 1)):
        incoming = torch.zeros((num_points, topm), device=unary.device, dtype=unary.dtype)
        incoming.index_add_(0, receiver, messages)
        cavity = unary[sender] + incoming[sender] - messages[reverse]
        updated = torch.max(cavity.unsqueeze(2) + directed_potentials, dim=1).values
        updated = updated - torch.max(updated, dim=1, keepdim=True).values
        messages = keep * messages + (1.0 - keep) * updated
    incoming = torch.zeros((num_points, topm), device=unary.device, dtype=unary.dtype)
    incoming.index_add_(0, receiver, messages)
    return unary + incoming


def assignment_energy(
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


def conditional_scores(
    point_idx: int,
    assignment: torch.Tensor,
    unary: torch.Tensor,
    pairs: torch.Tensor,
    potentials: torch.Tensor,
) -> torch.Tensor:
    scores = unary[point_idx].clone()
    if int(pairs.shape[0]) == 0:
        return scores
    first_edges = torch.nonzero(pairs[:, 0] == point_idx, as_tuple=False).squeeze(1)
    if int(first_edges.numel()) > 0:
        other = pairs[first_edges, 1]
        scores = scores + potentials[
            first_edges,
            :,
            assignment[other],
        ].sum(dim=0)
    second_edges = torch.nonzero(pairs[:, 1] == point_idx, as_tuple=False).squeeze(1)
    if int(second_edges.numel()) > 0:
        other = pairs[second_edges, 0]
        scores = scores + potentials[
            second_edges,
            assignment[other],
            :,
        ].sum(dim=0)
    return scores


def icm_refine(
    initial: torch.Tensor,
    unary: torch.Tensor,
    pairs: torch.Tensor,
    potentials: torch.Tensor,
    steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sequential coordinate ascent; each update cannot reduce joint energy."""
    assignment = initial.clone()
    for _ in range(max(int(steps), 1)):
        changed = 0
        for point_idx in range(int(unary.shape[0])):
            scores = conditional_scores(point_idx, assignment, unary, pairs, potentials)
            current = int(assignment[point_idx].item())
            selected = int(torch.argmax(scores).item())
            if float(scores[selected].item()) > float(scores[current].item()) + 1e-8:
                assignment[point_idx] = selected
                changed += int(selected != current)
        if changed == 0:
            break
    return assignment, assignment_energy(assignment, unary, pairs, potentials)


def solve_joint_assignment(
    unary: torch.Tensor,
    base_positions: torch.Tensor,
    pairs: torch.Tensor,
    potentials: torch.Tensor,
    args,
) -> dict[str, torch.Tensor]:
    baseline_energy = assignment_energy(base_positions, unary, pairs, potentials)
    beliefs = max_product_beliefs(
        unary,
        pairs,
        potentials,
        args.jgm_bp_steps,
        args.jgm_bp_damping,
    )
    bp_initial = torch.argmax(beliefs, dim=1)
    bp_assignment, bp_energy = icm_refine(
        bp_initial,
        unary,
        pairs,
        potentials,
        args.jgm_icm_steps,
    )
    base_assignment, base_start_energy = icm_refine(
        base_positions,
        unary,
        pairs,
        potentials,
        args.jgm_icm_steps,
    )
    if float(bp_energy.item()) > float(base_start_energy.item()):
        assignment = bp_assignment
        selected_energy = bp_energy
        selected_start = torch.tensor(1, device=unary.device)
    else:
        assignment = base_assignment
        selected_energy = base_start_energy
        selected_start = torch.tensor(0, device=unary.device)
    final_conditionals = torch.stack(
        [
            conditional_scores(point_idx, assignment, unary, pairs, potentials)
            for point_idx in range(int(unary.shape[0]))
        ],
        dim=0,
    )
    return {
        "assignment": assignment,
        "beliefs": beliefs,
        "conditional_scores": final_conditionals,
        "baseline_energy": baseline_energy,
        "selected_energy": selected_energy,
        "bp_energy": bp_energy,
        "base_start_energy": base_start_energy,
        "selected_start": selected_start,
    }


def joint_graph_candidate_ownership(
    records: list[dict[str, Any]],
    src_ft: torch.Tensor,
    trg_ft: torch.Tensor,
    src_points: list[list[int]],
    src_threshold: float,
    trg_threshold: float,
    args,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[dict[str, Any]], dict[str, torch.Tensor]]:
    _, baseline_predictions, v2_diagnostics, bundle = geometry_aware_relational_ownership(
        records,
        src_ft,
        trg_ft,
        src_points,
        src_threshold,
        trg_threshold,
        args,
    )
    unary_full = bundle["final_scores"].float()
    base_columns = bundle["base_columns"]
    candidate_columns, candidate_values, base_positions = build_topm_candidates(
        unary_full,
        base_columns,
        args.jgm_topm,
    )
    candidate_xy = bundle["union_xy"][candidate_columns]
    base_values = unary_full[
        torch.arange(int(unary_full.shape[0]), device=unary_full.device),
        base_columns,
    ]
    unary = float(args.jgm_unary_weight) * (candidate_values - base_values.unsqueeze(1))
    if args.jgm_freeze_anchors:
        anchor_indices = bundle["anchor_indices"]
        if int(anchor_indices.numel()) > 0:
            anchor_mask = torch.zeros_like(unary, dtype=torch.bool)
            anchor_mask[anchor_indices] = True
            anchor_base_mask = torch.zeros_like(unary, dtype=torch.bool)
            anchor_base_mask[anchor_indices, base_positions[anchor_indices]] = True
            unary = torch.where(anchor_mask & ~anchor_base_mask, torch.full_like(unary, -1e4), unary)

    pairs, potentials, structural_edges = build_joint_potentials(
        bundle["src_xy"],
        candidate_xy,
        candidate_columns,
        base_positions,
        src_threshold,
        trg_threshold,
        args,
    )
    if int(structural_edges.sum().item()) < int(args.jgm_min_structural_edges):
        unary = torch.full_like(unary, -1e4)
        unary[
            torch.arange(int(unary.shape[0]), device=unary.device),
            base_positions,
        ] = 0.0

    solution = solve_joint_assignment(unary, base_positions, pairs, potentials, args)
    assignment = solution["assignment"]
    point_index = torch.arange(int(unary.shape[0]), device=unary.device)
    selected_columns = candidate_columns[point_index, assignment]
    joint_gain = float((solution["selected_energy"] - solution["baseline_energy"]).item())
    bp_gain = float((solution["bp_energy"] - solution["baseline_energy"]).item())

    dense_joint_scores = torch.full_like(unary_full, -1e4)
    dense_joint_scores.scatter_(1, candidate_columns, solution["conditional_scores"])
    predictions: list[tuple[int, int]] = []
    diagnostics: list[dict[str, Any]] = []
    for point_idx in range(int(unary.shape[0])):
        base_column = int(base_columns[point_idx].item())
        selected_column = int(selected_columns[point_idx].item())
        selected_state = int(assignment[point_idx].item())
        base_state = int(base_positions[point_idx].item())
        scores = solution["conditional_scores"][point_idx]
        selected_score = float(scores[selected_state].item())
        base_score = float(scores[base_state].item())
        predictions.append(
            (
                int(bundle["union_x"][selected_column].item()),
                int(bundle["union_y"][selected_column].item()),
            )
        )
        diagnostic = dict(v2_diagnostics[point_idx])
        diagnostic.update(
            {
                "v2_proposed_column": diagnostic.get("proposed_column"),
                "joint_candidate_state": selected_state,
                "joint_candidate_rank": selected_state + 1,
                "joint_unary_gain": float(
                    (candidate_values[point_idx, selected_state] - candidate_values[point_idx, base_state]).item()
                ),
                "joint_conditional_gain": selected_score - base_score,
                "joint_energy_gain": joint_gain,
                "joint_bp_energy_gain": bp_gain,
                "joint_selected_bp_start": int(solution["selected_start"].item()),
                "joint_pair_count": int(pairs.shape[0]),
                "joint_structural_edge_count": int(structural_edges.sum().item()),
                "joint_topm": int(candidate_columns.shape[1]),
                "proposed_pred_x": int(bundle["union_x"][selected_column].item()),
                "proposed_pred_y": int(bundle["union_y"][selected_column].item()),
                "proposed_column": selected_column,
                "final_column": selected_column,
                "changed": int(selected_column != base_column),
                "gate_passed": int(selected_column != base_column),
            }
        )
        diagnostics.append(diagnostic)

    bundle["unary_scores"] = unary_full
    bundle["final_scores"] = dense_joint_scores
    bundle["joint_candidate_columns"] = candidate_columns
    bundle["joint_assignment"] = assignment
    return predictions, baseline_predictions, diagnostics, bundle


def build_joint_parser() -> argparse.ArgumentParser:
    parser = build_v2_parser()
    parser.description = "SPair top-M joint relational graph matching evaluator"
    parser.add_argument("--jgm_topm", default=10, type=int)
    parser.add_argument("--jgm_graph_k", default=3, type=int)
    parser.add_argument("--jgm_unary_weight", default=1.0, type=float)
    parser.add_argument("--jgm_pairwise_weight", default=0.15, type=float)
    parser.add_argument("--jgm_distance_sigma", default=0.12, type=float)
    parser.add_argument("--jgm_distance_truncation", default=0.30, type=float)
    parser.add_argument("--jgm_min_source_separation", default=0.02, type=float)
    parser.add_argument("--jgm_collision_weight", default=0.10, type=float)
    parser.add_argument("--jgm_collision_radius", default=0.03, type=float)
    parser.add_argument("--jgm_collision_source_ratio", default=2.0, type=float)
    parser.add_argument("--jgm_bp_steps", default=6, type=int)
    parser.add_argument("--jgm_bp_damping", default=0.5, type=float)
    parser.add_argument("--jgm_icm_steps", default=6, type=int)
    parser.add_argument("--jgm_min_structural_edges", default=1, type=int)
    parser.add_argument(
        "--jgm_freeze_anchors",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


if __name__ == "__main__":
    args = build_joint_parser().parse_args()
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    method_tag = (
        f"jgm_topm{args.jgm_topm}_gk{args.jgm_graph_k}"
        f"_rw{args.rco_relation_weight}_gw{args.rco_v2_geometry_weight}"
        f"_pw{args.jgm_pairwise_weight}_cw{args.jgm_collision_weight}"
        f"_bp{args.jgm_bp_steps}"
    )
    run_evaluation(
        args,
        ownership_fn=joint_graph_candidate_ownership,
        method_tag=method_tag,
        method_name="joint_graph_candidate_ownership",
    )

"""Measure candidate recall, assignment ceilings, and unlabeled repairability.

This is a diagnosis-only analysis for frozen DiTF features on SPair-71k.
It preserves post-AdaLN and optional channel discard, and never uses target
keypoint labels to choose candidates or risk signals. Ground-truth labels are
used only after selection to measure recall and oracle ceilings.
"""

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from analyze_spair_token_residuals import (
    build_post_feature,
    make_grid_for_output_window,
    sample_feature_at_pixel,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Measure raw top-K candidate recall, candidate-constrained assignment "
            "oracle, and label-free repairability signals on SPair-71k."
        )
    )
    parser.add_argument("--dataset_path", required=True, help="Path to SPair-71k root.")
    parser.add_argument("--feature_path", required=True, help="Path to cached per-category features.")
    parser.add_argument("--output_dir", required=True, help="Directory for records and summary.")
    parser.add_argument("--device", default="cuda", help="Feature matching device, e.g. cuda or cpu.")
    parser.add_argument("--feature_dim", type=int, default=3072, help="Feature dimension.")
    parser.add_argument("--cd", action="store_true", help="Apply DiTF channel discard before AdaLN.")
    parser.add_argument("--discard_channels", nargs="+", type=int, default=[154, 1446])
    parser.add_argument("--tile_rows", type=int, default=32, help="Target rows processed per score-map tile.")
    parser.add_argument("--raw_topk", nargs="+", type=int, default=[1, 5, 10, 20, 50])
    parser.add_argument("--max_pairs_per_cat", type=int, default=0, help="Category-wise cap for quick runs.")
    parser.add_argument("--flush_every_pairs", type=int, default=10)
    parser.add_argument("--entropy_temperature", type=float, default=0.05)
    parser.add_argument(
        "--collision_radius_norm",
        type=float,
        default=0.05,
        help="Target-coordinate collision radius as a fraction of target bbox scale.",
    )
    parser.add_argument(
        "--assignment_fallback_penalty",
        type=float,
        default=0.05,
        help="Score penalty for the private unmatched/fallback slot in raw global assignment.",
    )
    parser.add_argument(
        "--repair_fractions",
        nargs="+",
        type=float,
        default=[0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50],
    )
    return parser.parse_args()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def write_records_csv(records: list[dict[str, Any]], path: str):
    if not records:
        return
    fields = sorted({key for record in records for key in record})
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def safe_mean(values: list[float | int]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def safe_rate(values: list[int]) -> float | None:
    return safe_mean(values)


def load_pair_lists(dataset_path: str) -> tuple[list[str], dict[str, list[str]]]:
    test_path = os.path.join(dataset_path, "PairAnnotation", "test")
    image_root = os.path.join(dataset_path, "JPEGImages")
    json_names = os.listdir(test_path)
    categories = os.listdir(image_root)
    category_pairs = {
        category: [name for name in json_names if category in name]
        for category in categories
    }
    return categories, category_pairs


def load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_post_features(
    output_dict: dict[str, Any],
    ada_dict: dict[str, Any],
    data: dict[str, Any],
    pre_norm: nn.LayerNorm,
    args,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    src_raw, src_ada = output_dict[data["src_imname"]], ada_dict[data["src_imname"]]
    trg_raw, trg_ada = output_dict[data["trg_imname"]], ada_dict[data["trg_imname"]]
    _, _, src_post, _, _ = build_post_feature(
        src_raw.float(), src_ada.float(), pre_norm, args.discard_channels, args.cd
    )
    _, _, trg_post, _, _ = build_post_feature(
        trg_raw.float(), trg_ada.float(), pre_norm, args.discard_channels, args.cd
    )
    return src_post.float().to(device), trg_post.float().to(device)


def sample_keypoint_vectors(
    feature: torch.Tensor,
    points: list[list[int]],
    eval_h: int,
    eval_w: int,
) -> torch.Tensor:
    vectors = [sample_feature_at_pixel(feature, int(point[0]), int(point[1]), eval_h, eval_w) for point in points]
    return F.normalize(torch.stack(vectors, dim=0).float(), dim=1)


def sample_flat_index_vectors(
    feature: torch.Tensor,
    flat_indices: np.ndarray,
    eval_h: int,
    eval_w: int,
) -> torch.Tensor:
    """Batch-sample high-resolution pixel locations from a low-resolution feature map."""
    if len(flat_indices) == 0:
        return torch.empty((0, feature.shape[1]), dtype=torch.float32, device=feature.device)
    indices = torch.as_tensor(flat_indices, dtype=torch.long, device=feature.device)
    x = (indices % int(eval_w)).float()
    y = torch.div(indices, int(eval_w), rounding_mode="floor").float()
    grid_x = 2.0 * ((x + 0.5) / float(eval_w)) - 1.0
    grid_y = 2.0 * ((y + 0.5) / float(eval_h)) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=1).view(1, -1, 1, 2)
    sampled = F.grid_sample(feature, grid, mode="bilinear", align_corners=False)
    vectors = sampled[0, :, :, 0].transpose(0, 1).contiguous().float()
    return F.normalize(vectors, dim=1)


def topk_score_maps(
    src_vectors: torch.Tensor,
    trg_feature: torch.Tensor,
    out_h: int,
    out_w: int,
    tile_rows: int,
    max_topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return top-k scores and flattened target indices for all source points."""
    num_points = int(src_vectors.shape[0])
    max_topk = min(max(int(max_topk), 1), int(out_h * out_w))
    running_values = torch.full(
        (num_points, max_topk),
        fill_value=-torch.inf,
        dtype=torch.float32,
        device=trg_feature.device,
    )
    running_indices = torch.full(
        (num_points, max_topk),
        fill_value=-1,
        dtype=torch.long,
        device=trg_feature.device,
    )

    for y_start in range(0, out_h, max(int(tile_rows), 1)):
        y_end = min(y_start + max(int(tile_rows), 1), out_h)
        grid = make_grid_for_output_window(0, out_w, y_start, y_end, out_h, out_w, trg_feature.device)
        tile = F.grid_sample(trg_feature, grid, mode="bilinear", align_corners=False)[0]
        tile = F.normalize(tile.float(), dim=0)
        tile_flat = tile.reshape(tile.shape[0], -1)
        tile_scores = src_vectors @ tile_flat
        tile_k = min(max_topk, int(tile_scores.shape[1]))
        tile_values, tile_local_indices = torch.topk(tile_scores, k=tile_k, dim=1)
        tile_indices = tile_local_indices + int(y_start * out_w)
        merged_values = torch.cat((running_values, tile_values), dim=1)
        merged_indices = torch.cat((running_indices, tile_indices), dim=1)
        running_values, keep = torch.topk(merged_values, k=max_topk, dim=1)
        running_indices = torch.gather(merged_indices, 1, keep)

    return running_values, running_indices


def target_coordinates(flat_indices: np.ndarray, width: int) -> np.ndarray:
    return np.stack((flat_indices % width, flat_indices // width), axis=-1).astype(np.float64)


def point_is_correct(x: float, y: float, gt: list[int], threshold: float) -> int:
    distance = math.sqrt((x - float(gt[0])) ** 2 + (y - float(gt[1])) ** 2)
    return int(distance / max(threshold, 1e-6) <= 0.1)


def candidate_hit(candidates: np.ndarray, gt: list[int], radius: float) -> int:
    if candidates.size == 0:
        return 0
    delta = candidates.astype(np.float64) - np.asarray(gt, dtype=np.float64).reshape(1, 2)
    return int(np.any(np.sum(delta * delta, axis=1) <= float(radius * radius)))


def normalized_entropy(values: np.ndarray, temperature: float) -> float:
    if len(values) <= 1:
        return 0.0
    x = values.astype(np.float64) / max(float(temperature), 1e-6)
    x = x - np.max(x)
    probabilities = np.exp(np.clip(x, -80.0, 80.0))
    probabilities /= max(float(probabilities.sum()), 1e-12)
    entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-12)))
    return float(entropy / max(math.log(len(values)), 1e-12))


def build_candidate_union(top_indices: np.ndarray, top_values: np.ndarray, width: int):
    num_points, topk = top_indices.shape
    union = sorted({int(index) for index in top_indices.reshape(-1).tolist() if int(index) >= 0})
    union_lookup = {flat_index: column for column, flat_index in enumerate(union)}
    num_candidates = len(union)
    allowed_scores = np.full((num_points, num_candidates), -1e6, dtype=np.float64)
    for point_idx in range(num_points):
        for rank in range(topk):
            flat_index = int(top_indices[point_idx, rank])
            if flat_index < 0:
                continue
            allowed_scores[point_idx, union_lookup[flat_index]] = float(top_values[point_idx, rank])
    return np.asarray(union, dtype=np.int64), allowed_scores


def solve_own_candidate_score_assignment(
    top_indices: np.ndarray,
    top_values: np.ndarray,
    gt_points: list[list[int]],
    target_width: int,
    target_threshold: float,
) -> dict[str, Any]:
    """Run a non-oracle relaxed assignment over the top-k candidate union."""
    num_points, _ = top_indices.shape
    union, allowed_scores = build_candidate_union(top_indices, top_values, target_width)
    num_candidates = len(union)
    augmented = np.full((num_points, num_candidates + num_points), -1e6, dtype=np.float64)
    augmented[:, :num_candidates] = allowed_scores
    for point_idx in range(num_points):
        augmented[point_idx, num_candidates + point_idx] = float(top_values[point_idx, 0])
    rows, columns = linear_sum_assignment(-augmented)
    chosen_indices = np.full(num_points, -1, dtype=np.int64)
    chosen_scores = np.full(num_points, np.nan, dtype=np.float64)
    for row, column in zip(rows.tolist(), columns.tolist()):
        if column < num_candidates:
            chosen_indices[row] = union[column]
            chosen_scores[row] = augmented[row, column]
        else:
            chosen_indices[row] = int(top_indices[row, 0])
            chosen_scores[row] = float(top_values[row, 0])

    chosen_xy = target_coordinates(chosen_indices, target_width)
    correct = np.asarray(
        [point_is_correct(x, y, gt_points[idx], target_threshold) for idx, (x, y) in enumerate(chosen_xy)],
        dtype=np.int64,
    )
    raw_xy = target_coordinates(top_indices[:, 0], target_width)
    raw_correct = np.asarray(
        [point_is_correct(x, y, gt_points[idx], target_threshold) for idx, (x, y) in enumerate(raw_xy)],
        dtype=np.int64,
    )
    return {
        "chosen_indices": chosen_indices,
        "chosen_scores": chosen_scores,
        "correct": correct,
        "raw_correct": raw_correct,
        "changed": (chosen_indices != top_indices[:, 0]).astype(np.int64),
    }


def solve_global_score_assignment(
    union_indices: np.ndarray,
    union_scores: np.ndarray,
    top1_indices: np.ndarray,
    top1_values: np.ndarray,
    gt_points: list[list[int]],
    target_width: int,
    target_threshold: float,
    fallback_penalty: float,
) -> dict[str, Any]:
    """Assign any candidate in the global union using only raw feature scores."""
    num_points, num_candidates = union_scores.shape
    augmented = np.full((num_points, num_candidates + num_points), -1e6, dtype=np.float64)
    augmented[:, :num_candidates] = union_scores.astype(np.float64)
    for point_idx in range(num_points):
        augmented[point_idx, num_candidates + point_idx] = (
            float(top1_values[point_idx]) - float(fallback_penalty)
        )
    rows, columns = linear_sum_assignment(-augmented)
    chosen_indices = np.asarray(top1_indices, dtype=np.int64).copy()
    chosen_scores = np.asarray(top1_values, dtype=np.float64).copy() - float(fallback_penalty)
    used_fallback = np.ones(num_points, dtype=np.int64)
    for row, column in zip(rows.tolist(), columns.tolist()):
        if column < num_candidates:
            chosen_indices[row] = int(union_indices[column])
            chosen_scores[row] = float(union_scores[row, column])
            used_fallback[row] = 0

    chosen_xy = target_coordinates(chosen_indices, target_width)
    correct = np.asarray(
        [point_is_correct(x, y, gt_points[idx], target_threshold) for idx, (x, y) in enumerate(chosen_xy)],
        dtype=np.int64,
    )
    return {
        "chosen_indices": chosen_indices,
        "chosen_scores": chosen_scores,
        "correct": correct,
        "changed": (chosen_indices != np.asarray(top1_indices)).astype(np.int64),
        "used_fallback": used_fallback,
    }


def solve_assignment_oracle(
    top_indices: np.ndarray,
    gt_points: list[list[int]],
    target_width: int,
    target_threshold: float,
    restrict_to_own_candidates: bool,
) -> dict[str, Any]:
    """Maximum PCK-correct assignment inside either own sets or their global union."""
    num_points, _ = top_indices.shape
    union = sorted({int(index) for index in top_indices.reshape(-1).tolist() if int(index) >= 0})
    union = np.asarray(union, dtype=np.int64)
    num_candidates = len(union)
    union_lookup = {int(index): column for column, index in enumerate(union.tolist())}
    allowed = np.zeros((num_points, num_candidates), dtype=bool)
    if restrict_to_own_candidates:
        for point_idx in range(num_points):
            for index in top_indices[point_idx].tolist():
                if int(index) >= 0:
                    allowed[point_idx, union_lookup[int(index)]] = True
    else:
        allowed[:, :] = True
    candidate_xy = target_coordinates(union, target_width)
    valid = np.zeros((num_points, num_candidates), dtype=np.float64)
    for point_idx, gt in enumerate(gt_points):
        delta = candidate_xy - np.asarray(gt, dtype=np.float64).reshape(1, 2)
        valid[point_idx] = (np.sum(delta * delta, axis=1) <= float((0.1 * target_threshold) ** 2)).astype(np.float64)

    augmented = np.zeros((num_points, num_candidates + num_points), dtype=np.float64)
    augmented[:, :num_candidates] = np.where((valid > 0) & allowed, 1.0, -1e6)
    rows, columns = linear_sum_assignment(-augmented)
    matched = np.zeros(num_points, dtype=np.int64)
    chosen_indices = np.full(num_points, -1, dtype=np.int64)
    for row, column in zip(rows.tolist(), columns.tolist()):
        if column < num_candidates and valid[row, column] > 0 and allowed[row, column]:
            matched[row] = 1
            chosen_indices[row] = union[column]
    return {
        "matched": matched,
        "chosen_indices": chosen_indices,
        "candidate_count": num_candidates,
    }


def compute_risk_features(
    top_indices: np.ndarray,
    top_values: np.ndarray,
    top1_cross_scores: np.ndarray,
    target_width: int,
    target_threshold: float,
    entropy_temperature: float,
    collision_radius_norm: float,
    assignment_output: dict[str, Any],
) -> list[dict[str, float]]:
    num_points, topk = top_indices.shape
    top1_xy = target_coordinates(top_indices[:, 0], target_width)
    collision_radius = float(collision_radius_norm * target_threshold)
    output: list[dict[str, float]] = []
    for point_idx in range(num_points):
        other_indices = [idx for idx in range(num_points) if idx != point_idx]
        other_top_indices = top_indices[other_indices].reshape(-1) if other_indices else np.asarray([], dtype=np.int64)
        other_xy = target_coordinates(other_top_indices, target_width) if len(other_top_indices) else np.empty((0, 2))
        delta = other_xy - top1_xy[point_idx].reshape(1, 2)
        collision_degree = int(np.sum(np.sqrt(np.sum(delta * delta, axis=1)) <= collision_radius)) if len(other_xy) else 0
        margin = float(top_values[point_idx, 0] - top_values[point_idx, min(1, topk - 1)]) if topk > 1 else 0.0
        cross_values = top1_cross_scores[point_idx]
        rival_values = np.delete(cross_values, point_idx)
        strongest_rival = float(np.max(rival_values)) if len(rival_values) else float("-inf")
        exclusivity = float(cross_values[point_idx] - strongest_rival) if len(rival_values) else float("inf")
        chosen_score = float(assignment_output["chosen_scores"][point_idx])
        top1_score = float(top_values[point_idx, 0])
        output.append(
            {
                "top1_top2_margin": margin,
                "candidate_entropy": normalized_entropy(top_values[point_idx], entropy_temperature),
                "collision_degree": float(collision_degree),
                "top1_exclusivity": exclusivity,
                "assignment_changed": float(assignment_output["changed"][point_idx]),
                "assignment_loss": float(top1_score - chosen_score),
            }
        )
    return output


def risk_curve(
    records: list[dict[str, Any]],
    signal: str,
    direction: str,
    repair_key: str,
    fractions: list[float],
) -> list[dict[str, Any]]:
    valid = [record for record in records if record.get(signal) is not None]
    reverse = direction == "high"
    ordered = sorted(valid, key=lambda record: float(record[signal]), reverse=reverse)
    total = len(records)
    total_errors = sum(1 - int(record["raw_correct"]) for record in records)
    total_repairable = sum(
        int(record[repair_key]) for record in records if int(record["raw_correct"]) == 0
    )
    output = []
    for fraction in fractions:
        count = min(max(1, int(round(total * fraction))), len(ordered)) if ordered else 0
        chosen = ordered[:count]
        selected_errors = sum(1 - int(record["raw_correct"]) for record in chosen)
        selected_repairable = sum(
            int(record[repair_key])
            for record in chosen
            if int(record["raw_correct"]) == 0
        )
        output.append(
            {
                "signal": signal,
                "direction": direction,
                "fraction": float(fraction),
                "num_selected": count,
                "selected_raw_error_rate": safe_rate([1 - int(record["raw_correct"]) for record in chosen]),
                "selected_repairable_count": selected_repairable,
                "repairable_recall": selected_repairable / max(total_repairable, 1),
                "oracle_gain_if_selected_repaired": selected_repairable / max(total, 1),
                "overall_error_rate": total_errors / max(total, 1),
                "overall_repairable_rate": total_repairable / max(total, 1),
            }
        )
    return output


def summarize_records(records: list[dict[str, Any]], topks: list[int]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "count": len(records),
        "raw_pck": safe_rate([int(record["raw_correct"]) for record in records]),
        "raw_error_rate": safe_rate([1 - int(record["raw_correct"]) for record in records]),
        "candidate_oracle_pck": {},
        "own_candidate_assignment_oracle_pck": {},
        "global_union_assignment_oracle_pck": {},
        "global_union_raw_score_assignment_pck": {},
    }
    for topk in topks:
        key = str(topk)
        summary["candidate_oracle_pck"][key] = safe_rate([int(record[f"candidate_hit@{topk}"]) for record in records])
        summary["own_candidate_assignment_oracle_pck"][key] = safe_rate(
            [int(record[f"own_assignment_oracle_hit@{topk}"]) for record in records]
        )
        summary["global_union_assignment_oracle_pck"][key] = safe_rate(
            [int(record[f"global_union_assignment_oracle_hit@{topk}"]) for record in records]
        )
        summary["global_union_raw_score_assignment_pck"][key] = safe_rate(
            [int(record[f"global_assignment_correct@{topk}"]) for record in records]
        )
    return summary


def grouped_summary(records: list[dict[str, Any]], topks: list[int], key: str) -> dict[str, Any]:
    output = {}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record.get(key))].append(record)
    for value, subset in sorted(groups.items()):
        output[value] = summarize_records(subset, topks)
    return output


def main():
    args = parse_args()
    topks = sorted({int(value) for value in args.raw_topk if int(value) > 0})
    if not topks:
        raise ValueError("--raw_topk must contain at least one positive value.")
    if max(topks) <= 0:
        raise ValueError("The largest raw top-k must be positive.")
    ensure_dir(args.output_dir)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    print(f"Using analysis device: {device}")

    categories, category_pairs = load_pair_lists(args.dataset_path)
    test_path = os.path.join(args.dataset_path, "PairAnnotation", "test")
    pre_norm = nn.LayerNorm(args.feature_dim, elementwise_affine=False, eps=1e-6)
    all_records: list[dict[str, Any]] = []
    processed_pairs = 0

    max_topk = max(topks)
    for category in categories:
        feature_file = os.path.join(args.feature_path, f"{category}.pth")
        ada_file = os.path.join(args.feature_path, f"{category}_ada.pth")
        if not os.path.exists(feature_file) or not os.path.exists(ada_file):
            print(f"Skipping {category}: missing cached feature or AdaLN file.")
            continue
        output_dict = torch.load(feature_file, map_location="cpu", weights_only=True)
        ada_dict = torch.load(ada_file, map_location="cpu", weights_only=True)
        pair_names = list(category_pairs[category])
        if args.max_pairs_per_cat > 0:
            pair_names = pair_names[: args.max_pairs_per_cat]

        for pair_name in pair_names:
            data = load_json(os.path.join(test_path, pair_name))
            src_eval_h, src_eval_w = data["src_imsize"][:2][::-1]
            trg_eval_h, trg_eval_w = data["trg_imsize"][:2][::-1]
            target_bbox = data["trg_bndbox"]
            target_threshold = max(
                target_bbox[3] - target_bbox[1],
                target_bbox[2] - target_bbox[0],
            )
            src_feature, trg_feature = build_post_features(output_dict, ada_dict, data, pre_norm, args, device)
            src_vectors = sample_keypoint_vectors(src_feature, data["src_kps"], src_eval_h, src_eval_w)
            top_values_t, top_indices_t = topk_score_maps(
                src_vectors,
                trg_feature,
                trg_eval_h,
                trg_eval_w,
                args.tile_rows,
                max_topk,
            )
            top_values = top_values_t.detach().cpu().numpy()
            top_indices = top_indices_t.detach().cpu().numpy()

            top1_indices = top_indices[:, 0]
            top1_xy = target_coordinates(top1_indices, trg_eval_w)
            raw_correct = np.asarray(
                [
                    point_is_correct(x, y, data["trg_kps"][idx], target_threshold)
                    for idx, (x, y) in enumerate(top1_xy)
                ],
                dtype=np.int64,
            )

            top1_target_vectors = sample_keypoint_vectors(
                trg_feature,
                [[int(x), int(y)] for x, y in top1_xy.tolist()],
                trg_eval_h,
                trg_eval_w,
            )
            top1_cross_scores = (top1_target_vectors @ src_vectors.transpose(0, 1)).detach().cpu().numpy()
            max_union_indices = np.asarray(
                sorted({int(index) for index in top_indices.reshape(-1).tolist() if int(index) >= 0}),
                dtype=np.int64,
            )
            max_union_vectors = sample_flat_index_vectors(
                trg_feature,
                max_union_indices,
                trg_eval_h,
                trg_eval_w,
            )
            max_union_scores = (
                src_vectors @ max_union_vectors.transpose(0, 1)
            ).detach().cpu().numpy()
            max_union_lookup = {int(index): column for column, index in enumerate(max_union_indices.tolist())}
            max_top1_scores = top_values[:, 0]
            raw_assignment = solve_global_score_assignment(
                max_union_indices,
                max_union_scores,
                top_indices[:, 0],
                max_top1_scores,
                data["trg_kps"],
                trg_eval_w,
                target_threshold,
                args.assignment_fallback_penalty,
            )
            risks = compute_risk_features(
                top_indices,
                top_values,
                top1_cross_scores,
                trg_eval_w,
                target_threshold,
                args.entropy_temperature,
                args.collision_radius_norm,
                raw_assignment,
            )

            pair_records: list[dict[str, Any]] = []
            for point_idx, (src_point, trg_point) in enumerate(zip(data["src_kps"], data["trg_kps"])):
                record: dict[str, Any] = {
                    "category": category,
                    "pair_name": pair_name,
                    "src_imname": data["src_imname"],
                    "trg_imname": data["trg_imname"],
                    "kp_idx": point_idx,
                    "src_x": int(src_point[0]),
                    "src_y": int(src_point[1]),
                    "trg_x": int(trg_point[0]),
                    "trg_y": int(trg_point[1]),
                    "target_threshold": float(target_threshold),
                    "raw_pred_x": int(top1_xy[point_idx, 0]),
                    "raw_pred_y": int(top1_xy[point_idx, 1]),
                    "raw_correct": int(raw_correct[point_idx]),
                    "raw_top1_score": float(top_values[point_idx, 0]),
                    "raw_top2_score": float(top_values[point_idx, min(1, max_topk - 1)]),
                    "top1_cross_self_score": float(top1_cross_scores[point_idx, point_idx]),
                    "raw_assignment_pred_x": int(
                        target_coordinates(raw_assignment["chosen_indices"], trg_eval_w)[point_idx, 0]
                    ),
                    "raw_assignment_pred_y": int(
                        target_coordinates(raw_assignment["chosen_indices"], trg_eval_w)[point_idx, 1]
                    ),
                    "raw_assignment_correct": int(raw_assignment["correct"][point_idx]),
                }
                record.update(risks[point_idx])
                for topk in topks:
                    top_idx = top_indices[point_idx, :topk]
                    top_val = top_values[point_idx, :topk]
                    candidates = target_coordinates(top_idx, trg_eval_w)
                    record[f"candidate_hit@{topk}"] = candidate_hit(candidates, trg_point, 0.1 * target_threshold)
                    record[f"raw_rank_in_pck_candidates@{topk}"] = (
                        int(np.argmax(
                            np.sum((candidates - np.asarray(trg_point, dtype=np.float64).reshape(1, 2)) ** 2, axis=1)
                            <= (0.1 * target_threshold) ** 2
                        ))
                        if record[f"candidate_hit@{topk}"]
                        else None
                    )
                pair_records.append(record)

            for topk in topks:
                selected_indices = top_indices[:, :topk]
                own_assignment_oracle = solve_assignment_oracle(
                    selected_indices,
                    data["trg_kps"],
                    trg_eval_w,
                    target_threshold,
                    restrict_to_own_candidates=True,
                )
                global_assignment_oracle = solve_assignment_oracle(
                    selected_indices,
                    data["trg_kps"],
                    trg_eval_w,
                    target_threshold,
                    restrict_to_own_candidates=False,
                )
                selected_union_indices = np.asarray(
                    sorted({int(index) for index in selected_indices.reshape(-1).tolist() if int(index) >= 0}),
                    dtype=np.int64,
                )
                selected_union_columns = np.asarray(
                    [max_union_lookup[int(index)] for index in selected_union_indices.tolist()],
                    dtype=np.int64,
                )
                score_assignment = solve_global_score_assignment(
                    selected_union_indices,
                    max_union_scores[:, selected_union_columns],
                    selected_indices[:, 0],
                    top_values[:, 0],
                    data["trg_kps"],
                    trg_eval_w,
                    target_threshold,
                    args.assignment_fallback_penalty,
                )
                for point_idx, record in enumerate(pair_records):
                    record[f"own_assignment_oracle_hit@{topk}"] = int(own_assignment_oracle["matched"][point_idx])
                    record[f"global_union_assignment_oracle_hit@{topk}"] = int(global_assignment_oracle["matched"][point_idx])
                    record[f"global_assignment_correct@{topk}"] = int(score_assignment["correct"][point_idx])
                    record[f"global_assignment_changed@{topk}"] = int(score_assignment["changed"][point_idx])
                    record[f"assignment_pred_x@{topk}"] = int(
                        target_coordinates(score_assignment["chosen_indices"], trg_eval_w)[point_idx, 0]
                    )
                    record[f"assignment_pred_y@{topk}"] = int(
                        target_coordinates(score_assignment["chosen_indices"], trg_eval_w)[point_idx, 1]
                    )

            all_records.extend(pair_records)
            processed_pairs += 1
            if processed_pairs % max(int(args.flush_every_pairs), 1) == 0:
                write_records_csv(
                    all_records,
                    os.path.join(args.output_dir, "topk_assignment_repairability_records.csv"),
                )
                print(f"[Flush] pairs={processed_pairs} records={len(all_records)}")
            del src_feature, trg_feature, src_vectors, top_values_t, top_indices_t
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if not all_records:
        raise RuntimeError("No records produced. Check dataset_path and feature_path.")

    signal_specs = {
        "top1_top2_margin": "low",
        "candidate_entropy": "high",
        "collision_degree": "high",
        "top1_exclusivity": "low",
        "assignment_changed": "high",
        "assignment_loss": "high",
    }
    repairability = {}
    for signal, direction in signal_specs.items():
        repairability[signal] = {}
        for topk in topks:
            repairability[signal][str(topk)] = {
                "own_candidate_repairability": risk_curve(
                    all_records,
                    signal,
                    direction,
                    f"candidate_hit@{topk}",
                    args.repair_fractions,
                ),
                "global_union_assignment_repairability": risk_curve(
                    all_records,
                    signal,
                    direction,
                    f"global_union_assignment_oracle_hit@{topk}",
                    args.repair_fractions,
                ),
            }

    summary = {
        "num_records": len(all_records),
        "num_pairs": len({(record["category"], record["pair_name"]) for record in all_records}),
        "raw_topk": topks,
        "cd": bool(args.cd),
        "device": str(device),
        "collision_radius_norm": float(args.collision_radius_norm),
        "assignment_fallback_penalty": float(args.assignment_fallback_penalty),
        "overall": summarize_records(all_records, topks),
        "failure_subset": summarize_records(
            [record for record in all_records if int(record["raw_correct"]) == 0], topks
        ),
        "by_category": grouped_summary(all_records, topks, "category"),
        "repairability": repairability,
        "interpretation": {
            "candidate_hit": "GT lies within the PCK radius of at least one raw top-K candidate; this is a candidate-recall ceiling, not an online method.",
            "own_assignment_oracle_hit": "Maximum PCK-correct assignment when each source point may use only its own top-K candidates.",
            "global_union_assignment_oracle_hit": "Maximum PCK-correct assignment when all source points share the union of top-K candidates; this is a broader candidate-pool ceiling and may exceed per-point recall.",
            "global_assignment_correct": "Non-oracle raw-feature score assignment over the shared candidate union, with a private fallback penalty.",
            "repairability": "Risk curves rank points only by label-free score-map signals, then use GT after ranking to measure own-candidate and global-union repairable errors.",
            "limitations": "Candidate and assignment ceilings do not repair GT absent from the candidate set or identity information already lost in raw features.",
        },
    }

    records_path = os.path.join(args.output_dir, "topk_assignment_repairability_records.csv")
    summary_path = os.path.join(args.output_dir, "topk_assignment_repairability_summary.json")
    write_records_csv(all_records, records_path)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"Saved records to: {records_path}")
    print(f"Saved summary to: {summary_path}")
    print("Overall:", summary["overall"])
    print("Failure subset:", summary["failure_subset"])
    print("Repairability signals:", list(repairability))


if __name__ == "__main__":
    main()

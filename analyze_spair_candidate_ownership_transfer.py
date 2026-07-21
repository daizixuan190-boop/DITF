"""Analyze source-to-candidate ownership transfer inside global-union oracle recoveries.

The analysis uses GT only to define an offline cohort: raw matching failures that
can be PCK-correctly matched by a global-union assignment oracle. All ownership
scores, candidate origins, and score ranks are computed from frozen DiTF features.
"""

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from analyze_spair_topk_assignment_repairability import (
    build_post_features,
    ensure_dir,
    load_pair_lists,
    point_is_correct,
    sample_flat_index_vectors,
    sample_keypoint_vectors,
    solve_assignment_oracle,
    target_coordinates,
    topk_score_maps,
    write_records_csv,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose which source keypoints originate target candidates that a "
            "global-union oracle assigns to other raw-failed source keypoints."
        )
    )
    parser.add_argument("--dataset_path", required=True, help="Path to SPair-71k root.")
    parser.add_argument("--feature_path", required=True, help="Path to cached per-category features.")
    parser.add_argument("--output_dir", required=True, help="Directory for ownership-transfer outputs.")
    parser.add_argument("--device", default="cuda", help="Feature matching device, e.g. cuda or cpu.")
    parser.add_argument("--feature_dim", type=int, default=3072)
    parser.add_argument("--cd", action="store_true", help="Apply DiTF channel discard before AdaLN.")
    parser.add_argument("--discard_channels", nargs="+", type=int, default=[154, 1446])
    parser.add_argument("--candidate_topk", type=int, default=50, help="Per-source top-K used to form the shared union.")
    parser.add_argument("--tile_rows", type=int, default=32)
    parser.add_argument("--max_pairs_per_cat", type=int, default=0)
    parser.add_argument("--flush_every_pairs", type=int, default=10)
    parser.add_argument("--min_pair_support", type=int, default=10, help="Minimum records for ranked ownership-pair summaries.")
    return parser.parse_args()


def pair_scalar_fields(data: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in data.items()
        if isinstance(value, (int, float, bool, str))
    }


def normalized_distance(point_a: list[int] | np.ndarray, point_b: list[int] | np.ndarray, normalizer: float) -> float:
    dx = float(point_a[0]) - float(point_b[0])
    dy = float(point_a[1]) - float(point_b[1])
    return float(math.sqrt(dx * dx + dy * dy) / max(float(normalizer), 1e-6))


def candidate_origins(
    top_indices: np.ndarray,
    top_values: np.ndarray,
    candidate_index: int,
) -> list[dict[str, Any]]:
    origins = []
    for source_idx in range(top_indices.shape[0]):
        ranks = np.flatnonzero(top_indices[source_idx] == int(candidate_index))
        if len(ranks) == 0:
            continue
        rank = int(ranks[0])
        origins.append(
            {
                "source_idx": source_idx,
                "candidate_rank": rank + 1,
                "candidate_score": float(top_values[source_idx, rank]),
            }
        )
    return origins


def choose_primary_origin(origins: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not origins:
        return None
    return max(origins, key=lambda item: (float(item["candidate_score"]), -int(item["candidate_rank"])))


def score_rank(scores: np.ndarray, index: int) -> int:
    value = float(scores[index])
    return 1 + int(np.sum(scores > value))


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"count": 0}
    return {
        "count": len(records),
        "transfer_rate": float(np.mean([int(record["is_transfer"]) for record in records])),
        "owner_is_dominant_source_rate": float(np.mean([int(record["owner_is_dominant_source"]) for record in records])),
        "owner_score_below_best_other_rate": float(
            np.mean([int(record["owner_score_below_best_other"]) for record in records])
        ),
        "candidate_multi_origin_rate": float(np.mean([int(record["num_origin_sources"]) > 1 for record in records])),
        "candidate_overlaps_other_gt_rate": float(
            np.mean([int(record["candidate_overlaps_other_gt_pck"]) for record in records])
        ),
        "mean_owner_score_rank": float(np.mean([float(record["owner_score_rank_at_candidate"]) for record in records])),
        "mean_owner_minus_best_other_score": safe_mean(
            [record.get("owner_minus_best_other_score") for record in records]
        ),
        "mean_owner_to_primary_origin_src_dist": safe_mean(
            [record.get("owner_to_primary_origin_src_norm_dist") for record in records]
        ),
        "mean_owner_to_primary_origin_trg_dist": safe_mean(
            [record.get("owner_to_primary_origin_trg_norm_dist") for record in records]
        ),
        "mean_num_origin_sources": float(np.mean([float(record["num_origin_sources"]) for record in records])),
        "mean_num_gt_within_candidate_pck": float(
            np.mean([float(record["num_gt_within_candidate_pck"]) for record in records])
        ),
    }


def safe_mean(values: list[Any]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    return None if not valid else float(np.mean(np.asarray(valid, dtype=np.float64)))


def aggregate_ownership_pairs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        primary = record.get("primary_origin_idx")
        if primary is None:
            continue
        key = (str(record["category"]), int(record["kp_idx"]), int(primary))
        groups[key].append(record)
    output = []
    for (category, owner_idx, origin_idx), subset in groups.items():
        output.append(
            {
                "category": category,
                "owner_kp_idx": owner_idx,
                "primary_origin_idx": origin_idx,
                "count": len(subset),
                "transfer_count": sum(int(record["is_transfer"]) for record in subset),
                "transfer_rate": float(np.mean([int(record["is_transfer"]) for record in subset])),
                "owner_dominant_rate": float(
                    np.mean([int(record["owner_is_dominant_source"]) for record in subset])
                ),
                "owner_below_other_rate": float(
                    np.mean([int(record["owner_score_below_best_other"]) for record in subset])
                ),
                "mean_owner_score_rank": float(
                    np.mean([float(record["owner_score_rank_at_candidate"]) for record in subset])
                ),
                "mean_owner_minus_best_other_score": safe_mean(
                    [record.get("owner_minus_best_other_score") for record in subset]
                ),
                "mean_src_norm_dist": safe_mean(
                    [record.get("owner_to_primary_origin_src_norm_dist") for record in subset]
                ),
                "mean_trg_norm_dist": safe_mean(
                    [record.get("owner_to_primary_origin_trg_norm_dist") for record in subset]
                ),
                "candidate_overlap_other_gt_rate": float(
                    np.mean([int(record["candidate_overlaps_other_gt_pck"]) for record in subset])
                ),
                "pair_count": len({(record["category"], record["pair_name"]) for record in subset}),
            }
        )
    return sorted(output, key=lambda item: (int(item["count"]), int(item["transfer_count"])), reverse=True)


def grouped_factor_summary(records: list[dict[str, Any]], fields: list[str]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for field in fields:
        values = [record.get(field) for record in records if record.get(field) is not None]
        unique = sorted({str(value) for value in values})
        if not unique or len(unique) > 10:
            continue
        output[field] = {}
        for value in unique:
            output[field][value] = summarize([record for record in records if str(record.get(field)) == value])
    return output


def main():
    args = parse_args()
    if args.candidate_topk <= 0:
        raise ValueError("--candidate_topk must be positive.")
    ensure_dir(args.output_dir)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    print(f"Using analysis device: {device}")
    print(
        "Cohort: raw failures that a GT-aware global-union assignment oracle can match. "
        "All candidate ownership scores remain label-free."
    )

    categories, category_pairs = load_pair_lists(args.dataset_path)
    test_path = os.path.join(args.dataset_path, "PairAnnotation", "test")
    pre_norm = nn.LayerNorm(args.feature_dim, elementwise_affine=False, eps=1e-6)
    records: list[dict[str, Any]] = []
    processed_pairs = 0
    total_raw_failures = 0
    total_global_oracle_recoveries = 0

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
            with open(os.path.join(test_path, pair_name), "r", encoding="utf-8") as handle:
                data = json.load(handle)
            src_eval_h, src_eval_w = data["src_imsize"][:2][::-1]
            trg_eval_h, trg_eval_w = data["trg_imsize"][:2][::-1]
            src_bbox = data["src_bndbox"]
            trg_bbox = data["trg_bndbox"]
            src_threshold = max(src_bbox[3] - src_bbox[1], src_bbox[2] - src_bbox[0])
            trg_threshold = max(trg_bbox[3] - trg_bbox[1], trg_bbox[2] - trg_bbox[0])
            pck_radius = 0.1 * float(trg_threshold)

            src_feature, trg_feature = build_post_features(output_dict, ada_dict, data, pre_norm, args, device)
            src_vectors = sample_keypoint_vectors(src_feature, data["src_kps"], src_eval_h, src_eval_w)
            top_values_t, top_indices_t = topk_score_maps(
                src_vectors,
                trg_feature,
                trg_eval_h,
                trg_eval_w,
                args.tile_rows,
                args.candidate_topk,
            )
            top_values = top_values_t.detach().cpu().numpy()
            top_indices = top_indices_t.detach().cpu().numpy()
            top1_xy = target_coordinates(top_indices[:, 0], trg_eval_w)
            raw_correct = np.asarray(
                [
                    point_is_correct(x, y, data["trg_kps"][idx], trg_threshold)
                    for idx, (x, y) in enumerate(top1_xy)
                ],
                dtype=np.int64,
            )
            total_raw_failures += int(np.sum(1 - raw_correct))

            global_oracle = solve_assignment_oracle(
                top_indices,
                data["trg_kps"],
                trg_eval_w,
                trg_threshold,
                restrict_to_own_candidates=False,
            )
            cohort_indices = np.flatnonzero((raw_correct == 0) & (global_oracle["matched"] == 1))
            total_global_oracle_recoveries += int(len(cohort_indices))
            if len(cohort_indices) == 0:
                processed_pairs += 1
                continue

            union_indices = np.asarray(
                sorted({int(index) for index in top_indices.reshape(-1).tolist() if int(index) >= 0}),
                dtype=np.int64,
            )
            union_lookup = {int(index): column for column, index in enumerate(union_indices.tolist())}
            union_vectors = sample_flat_index_vectors(trg_feature, union_indices, trg_eval_h, trg_eval_w)
            union_scores = (src_vectors @ union_vectors.transpose(0, 1)).detach().cpu().numpy()
            pair_scalars = pair_scalar_fields(data)

            for owner_idx in cohort_indices.tolist():
                candidate_index = int(global_oracle["chosen_indices"][owner_idx])
                candidate_column = union_lookup[candidate_index]
                candidate_xy = target_coordinates(np.asarray([candidate_index], dtype=np.int64), trg_eval_w)[0]
                origins = candidate_origins(top_indices, top_values, candidate_index)
                primary_origin = choose_primary_origin(origins)
                owner_origins = [item for item in origins if int(item["source_idx"]) == int(owner_idx)]
                owner_rank_in_own_candidates = None if not owner_origins else int(owner_origins[0]["candidate_rank"])
                candidate_scores = union_scores[:, candidate_column]
                best_other_scores = np.delete(candidate_scores, owner_idx)
                best_other_score = float(np.max(best_other_scores)) if len(best_other_scores) else None
                dominant_source_idx = int(np.argmax(candidate_scores))
                owner_score = float(candidate_scores[owner_idx])
                gt_distances = np.asarray(
                    [normalized_distance(candidate_xy, target_point, 1.0) for target_point in data["trg_kps"]],
                    dtype=np.float64,
                )
                gt_distances_pixels = np.asarray(
                    [
                        math.sqrt(
                            (float(candidate_xy[0]) - float(target_point[0])) ** 2
                            + (float(candidate_xy[1]) - float(target_point[1])) ** 2
                        )
                        for target_point in data["trg_kps"]
                    ],
                    dtype=np.float64,
                )
                within_pck = np.flatnonzero(gt_distances_pixels <= pck_radius)
                other_within_pck = [index for index in within_pck.tolist() if int(index) != int(owner_idx)]

                record: dict[str, Any] = {
                    "category": category,
                    "pair_name": pair_name,
                    "src_imname": data["src_imname"],
                    "trg_imname": data["trg_imname"],
                    "kp_idx": int(owner_idx),
                    "candidate_topk": int(args.candidate_topk),
                    "src_x": int(data["src_kps"][owner_idx][0]),
                    "src_y": int(data["src_kps"][owner_idx][1]),
                    "trg_x": int(data["trg_kps"][owner_idx][0]),
                    "trg_y": int(data["trg_kps"][owner_idx][1]),
                    "raw_pred_x": int(top1_xy[owner_idx, 0]),
                    "raw_pred_y": int(top1_xy[owner_idx, 1]),
                    "global_candidate_x": int(candidate_xy[0]),
                    "global_candidate_y": int(candidate_xy[1]),
                    "global_candidate_flat_idx": candidate_index,
                    "raw_correct": 0,
                    "global_union_oracle_hit": 1,
                    "owner_in_own_topk": int(bool(owner_origins)),
                    "owner_rank_in_own_topk": owner_rank_in_own_candidates,
                    "num_origin_sources": len(origins),
                    "origin_source_indices": "|".join(str(item["source_idx"]) for item in origins),
                    "dominant_source_idx": dominant_source_idx,
                    "owner_is_dominant_source": int(dominant_source_idx == owner_idx),
                    "owner_score_at_candidate": owner_score,
                    "best_other_score_at_candidate": best_other_score,
                    "owner_minus_best_other_score": (
                        None if best_other_score is None else float(owner_score - best_other_score)
                    ),
                    "owner_score_below_best_other": int(
                        best_other_score is not None and owner_score < best_other_score
                    ),
                    "owner_score_rank_at_candidate": score_rank(candidate_scores, owner_idx),
                    "num_gt_within_candidate_pck": int(len(within_pck)),
                    "candidate_overlaps_other_gt_pck": int(len(other_within_pck) > 0),
                    "nearest_other_gt_idx": (
                        None
                        if len(data["trg_kps"]) <= 1
                        else int(np.argmin(np.where(np.arange(len(data["trg_kps"])) == owner_idx, np.inf, gt_distances_pixels)))
                    ),
                    "nearest_other_gt_norm_dist": (
                        None
                        if len(data["trg_kps"]) <= 1
                        else float(
                            np.min(
                                np.where(np.arange(len(data["trg_kps"])) == owner_idx, np.inf, gt_distances_pixels)
                            )
                            / max(float(trg_threshold), 1e-6)
                        )
                    ),
                }
                if primary_origin is not None:
                    origin_idx = int(primary_origin["source_idx"])
                    record.update(
                        {
                            "primary_origin_idx": origin_idx,
                            "primary_origin_candidate_rank": int(primary_origin["candidate_rank"]),
                            "primary_origin_candidate_score": float(primary_origin["candidate_score"]),
                            "is_transfer": int(origin_idx != owner_idx),
                            "owner_to_primary_origin_src_norm_dist": normalized_distance(
                                data["src_kps"][owner_idx], data["src_kps"][origin_idx], src_threshold
                            ),
                            "owner_to_primary_origin_trg_norm_dist": normalized_distance(
                                data["trg_kps"][owner_idx], data["trg_kps"][origin_idx], trg_threshold
                            ),
                        }
                    )
                else:
                    record.update(
                        {
                            "primary_origin_idx": None,
                            "primary_origin_candidate_rank": None,
                            "primary_origin_candidate_score": None,
                            "is_transfer": 0,
                            "owner_to_primary_origin_src_norm_dist": None,
                            "owner_to_primary_origin_trg_norm_dist": None,
                        }
                    )
                record.update(pair_scalars)
                records.append(record)

            processed_pairs += 1
            if processed_pairs % max(int(args.flush_every_pairs), 1) == 0:
                write_records_csv(records, os.path.join(args.output_dir, "candidate_ownership_transfer_records.csv"))
                print(f"[Flush] pairs={processed_pairs} cohort_records={len(records)}")

            del src_feature, trg_feature, src_vectors, top_values_t, top_indices_t, union_vectors
            if device.type == "cuda":
                torch.cuda.empty_cache()

    ownership_pairs = aggregate_ownership_pairs(records)
    supported_pairs = [record for record in ownership_pairs if int(record["count"]) >= args.min_pair_support]
    non_overlap_records = [
        record for record in records if int(record["candidate_overlaps_other_gt_pck"]) == 0
    ]
    overlap_records = [
        record for record in records if int(record["candidate_overlaps_other_gt_pck"]) == 1
    ]
    non_overlap_transfer_records = [
        record for record in non_overlap_records if int(record["is_transfer"]) == 1
    ]
    summary = {
        "candidate_topk": int(args.candidate_topk),
        "num_cohort_records": len(records),
        "total_raw_failures": total_raw_failures,
        "global_oracle_failure_coverage": total_global_oracle_recoveries / max(total_raw_failures, 1),
        "overall": summarize(records),
        "transfer_only": summarize([record for record in records if int(record["is_transfer"]) == 1]),
        "owner_proposed": summarize([record for record in records if int(record["is_transfer"]) == 0]),
        "non_overlap": summarize(non_overlap_records),
        "overlap": summarize(overlap_records),
        "non_overlap_transfer": summarize(non_overlap_transfer_records),
        "non_overlap_recovery_coverage_of_raw_failures": len(non_overlap_records) / max(total_raw_failures, 1),
        "non_overlap_transfer_coverage_of_raw_failures": len(non_overlap_transfer_records) / max(total_raw_failures, 1),
        "num_ownership_pairs": len(ownership_pairs),
        "min_pair_support": int(args.min_pair_support),
        "top_ownership_pairs": ownership_pairs[:50],
        "top_supported_ownership_pairs": supported_pairs[:50],
        "factor_breakdown": grouped_factor_summary(
            records,
            ["scale_variation", "viewpoint_variation", "occlusion", "truncation"],
        ),
        "interpretation": {
            "cohort": "Raw failures whose GT falls in a candidate selected by a GT-aware global-union assignment oracle.",
            "is_transfer": "The strongest originating source query for the oracle candidate is not the owner source keypoint.",
            "owner_score_rank": "Rank of the owner source score among all source keypoint scores at the oracle candidate; rank 1 means raw features already favor the owner at that location.",
            "pck_overlap": "If a candidate lies in multiple target keypoints' PCK regions, global-oracle recovery can partly reflect evaluation tolerance rather than a distinct ownership transfer.",
            "method_constraint": "No future method may use GT candidate ownership; these records only identify which label-free ownership signals are missing.",
        },
    }

    records_path = os.path.join(args.output_dir, "candidate_ownership_transfer_records.csv")
    pairs_path = os.path.join(args.output_dir, "candidate_ownership_transfer_pairs.csv")
    summary_path = os.path.join(args.output_dir, "candidate_ownership_transfer_summary.json")
    write_records_csv(records, records_path)
    write_records_csv(ownership_pairs, pairs_path)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"Saved records to: {records_path}")
    print(f"Saved ownership pairs to: {pairs_path}")
    print(f"Saved summary to: {summary_path}")
    print(
        "Cohort:",
        {
            "raw_failures": total_raw_failures,
            "global_oracle_recoveries": total_global_oracle_recoveries,
            "coverage": summary["global_oracle_failure_coverage"],
        },
    )
    print("Overall transfer summary:", summary["overall"])


if __name__ == "__main__":
    main()

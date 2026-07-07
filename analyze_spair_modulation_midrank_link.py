import argparse
import csv
import json
import math
import os
from typing import Any

import numpy as np


MERGE_KEYS = [
    "category",
    "pair_name",
    "src_imname",
    "trg_imname",
    "kp_idx",
]

MODULATION_KEYS = [
    "shift_ratio_src",
    "content_ratio_src",
    "interaction_ratio_src",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Link post-AdaLN modulation imbalance to mid-rank local identity failure on SPair-71k. "
            "The script merges per-point residual diagnostics with mid-rank suppression records and tests "
            "whether high-shift / low-content points are more likely to fall into failure buckets where "
            "the GT remains inside a local semantic neighborhood but is still suppressed in rank."
        )
    )
    parser.add_argument("--residual_csv", type=str, required=True, help="Path to per_point_records.csv.")
    parser.add_argument("--midrank_csv", type=str, required=True, help="Path to midrank_suppression_records.csv.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs.")
    parser.add_argument(
        "--midrank_range",
        nargs=2,
        type=int,
        default=[50, 500],
        help="Inclusive oracle-rank range used to define the core mid-rank suppression bucket.",
    )
    parser.add_argument(
        "--high_shift_quantile",
        type=float,
        default=0.75,
        help="Quantile used to define high-shift tokens.",
    )
    parser.add_argument(
        "--low_content_quantile",
        type=float,
        default=0.25,
        help="Quantile used to define low-content tokens.",
    )
    parser.add_argument(
        "--negative_interaction_quantile",
        type=float,
        default=0.25,
        help="Quantile used to define strongly negative interaction tokens.",
    )
    parser.add_argument(
        "--near_frac_threshold",
        type=float,
        default=0.90,
        help="Minimum top-50 near-fraction@x1 used to define local-neighborhood failures.",
    )
    parser.add_argument(
        "--spread_quantile",
        type=float,
        default=0.50,
        help="Upper quantile used to define compact candidate clusters via top50_spread_norm.",
    )
    parser.add_argument(
        "--centroid_quantile",
        type=float,
        default=0.75,
        help="Upper quantile used to keep locally centered clusters via top50_centroid_norm_dist.",
    )
    return parser.parse_args()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def parse_scalar(value: str) -> Any:
    if value is None or value == "":
        return None
    try:
        num = float(value)
    except ValueError:
        return value
    if math.isfinite(num) and abs(num - round(num)) < 1e-12:
        return int(round(num))
    return num


def load_csv(csv_path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({key: parse_scalar(value) for key, value in row.items()})
    return records


def write_records_csv(records: list[dict[str, Any]], csv_path: str):
    if not records:
        return
    fieldnames = sorted({key for record in records for key in record.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def merge_records(
    residual_records: list[dict[str, Any]],
    midrank_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    residual_map = {
        tuple(record[key] for key in MERGE_KEYS): record
        for record in residual_records
    }
    merged: list[dict[str, Any]] = []
    for midrank in midrank_records:
        merge_id = tuple(midrank[key] for key in MERGE_KEYS)
        residual = residual_map.get(merge_id)
        if residual is None:
            continue
        merged_record = dict(midrank)
        for key in MODULATION_KEYS + [
            "correct",
            "norm_dist",
            "sim_margin",
            "sim_entropy",
            "src_boundary_margin",
            "trg_boundary_margin",
            "pair_displacement",
        ]:
            if key in residual:
                merged_record[key] = residual[key]
        merged.append(merged_record)
    return merged


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def rate(records: list[dict[str, Any]], key: str) -> float | None:
    if not records:
        return None
    return float(np.mean([float(record[key]) for record in records]))


def quantile_threshold(records: list[dict[str, Any]], key: str, q: float) -> float:
    values = np.asarray([float(record[key]) for record in records], dtype=np.float64)
    return float(np.quantile(values, q))


def build_flags(records: list[dict[str, Any]], args) -> dict[str, float]:
    spread_thr = quantile_threshold(records, "top50_spread_norm", args.spread_quantile)
    centroid_thr = quantile_threshold(records, "top50_centroid_norm_dist", args.centroid_quantile)
    shift_thr = quantile_threshold(records, "shift_ratio_src", args.high_shift_quantile)
    content_thr = quantile_threshold(records, "content_ratio_src", args.low_content_quantile)
    interaction_thr = quantile_threshold(records, "interaction_ratio_src", args.negative_interaction_quantile)

    low_rank, high_rank = int(args.midrank_range[0]), int(args.midrank_range[1])

    for record in records:
        current_error = 1 - int(record["correct"])
        oracle_rank = int(record["oracle_best_rank"])
        is_midrank = int(low_rank <= oracle_rank <= high_rank)
        is_local_cluster = int(
            float(record["top50_near_frac@x1"]) >= args.near_frac_threshold
            and float(record["top50_spread_norm"]) <= spread_thr
            and float(record["top50_centroid_norm_dist"]) <= centroid_thr
        )
        record["current_error"] = current_error
        record["midrank_bucket"] = is_midrank
        record["local_cluster_failure"] = int(current_error == 1 and is_local_cluster == 1)
        record["midrank_local_identity_failure"] = int(current_error == 1 and is_midrank == 1 and is_local_cluster == 1)
        record["high_shift"] = int(float(record["shift_ratio_src"]) >= shift_thr)
        record["low_content"] = int(float(record["content_ratio_src"]) <= content_thr)
        record["negative_interaction"] = int(float(record["interaction_ratio_src"]) <= interaction_thr)
        record["high_shift_low_content"] = int(record["high_shift"] == 1 and record["low_content"] == 1)
        record["triple_imbalance"] = int(
            record["high_shift"] == 1 and record["low_content"] == 1 and record["negative_interaction"] == 1
        )

    return {
        "shift_ratio_src_q_high": shift_thr,
        "content_ratio_src_q_low": content_thr,
        "interaction_ratio_src_q_low": interaction_thr,
        "top50_spread_norm_q": spread_thr,
        "top50_centroid_norm_dist_q": centroid_thr,
    }


def decile_summary(records: list[dict[str, Any]], score_key: str, target_key: str) -> list[dict[str, Any]]:
    if not records:
        return []
    values = np.asarray([float(record[score_key]) for record in records], dtype=np.float64)
    order = np.argsort(values)
    chunks = np.array_split(order, 10)
    output = []
    for idx, chunk in enumerate(chunks):
        if len(chunk) == 0:
            continue
        subset = [records[i] for i in chunk]
        output.append(
            {
                "decile": idx,
                "count": len(subset),
                "score_min": float(min(float(record[score_key]) for record in subset)),
                "score_max": float(max(float(record[score_key]) for record in subset)),
                target_key: rate(subset, target_key),
                "error_rate": rate(subset, "current_error"),
                "midrank_bucket_rate": rate(subset, "midrank_bucket"),
                "mean_oracle_rank": float(np.mean([float(record["oracle_best_rank"]) for record in subset])),
            }
        )
    return output


def binary_group_summary(records: list[dict[str, Any]], flag_key: str, target_key: str) -> dict[str, Any]:
    pos = [record for record in records if int(record[flag_key]) == 1]
    neg = [record for record in records if int(record[flag_key]) == 0]
    return {
        "flag_key": flag_key,
        "positive_count": len(pos),
        "negative_count": len(neg),
        "positive_target_rate": rate(pos, target_key),
        "negative_target_rate": rate(neg, target_key),
        "rate_gap": None if not pos or not neg else float(rate(pos, target_key) - rate(neg, target_key)),
        "positive_error_rate": rate(pos, "current_error"),
        "negative_error_rate": rate(neg, "current_error"),
        "positive_mean_oracle_rank": mean_or_none([float(record["oracle_best_rank"]) for record in pos]),
        "negative_mean_oracle_rank": mean_or_none([float(record["oracle_best_rank"]) for record in neg]),
    }


def two_by_two_summary(records: list[dict[str, Any]], left_key: str, right_key: str, target_key: str) -> list[dict[str, Any]]:
    cells = []
    for left_val in [0, 1]:
        for right_val in [0, 1]:
            subset = [
                record
                for record in records
                if int(record[left_key]) == left_val and int(record[right_key]) == right_val
            ]
            cells.append(
                {
                    left_key: left_val,
                    right_key: right_val,
                    "count": len(subset),
                    target_key: rate(subset, target_key),
                    "error_rate": rate(subset, "current_error"),
                    "mean_oracle_rank": mean_or_none([float(record["oracle_best_rank"]) for record in subset]),
                }
            )
    return cells


def bucket_summary(records: list[dict[str, Any]], target_key: str) -> dict[str, Any]:
    failures = [record for record in records if int(record["current_error"]) == 1]
    target_failures = [record for record in records if int(record[target_key]) == 1]
    return {
        "num_points": len(records),
        "num_failures": len(failures),
        "num_target_failures": len(target_failures),
        "error_rate": rate(records, "current_error"),
        "target_failure_rate_over_all_points": rate(records, target_key),
        "target_failure_rate_over_failures": None if not failures else float(len(target_failures) / len(failures)),
        "target_mean_oracle_rank": mean_or_none([float(record["oracle_best_rank"]) for record in target_failures]),
        "target_mean_norm_dist": mean_or_none([float(record["norm_dist"]) for record in target_failures]),
        "target_mean_sim_margin": mean_or_none([float(record["sim_margin"]) for record in target_failures]),
        "target_mean_sim_entropy": mean_or_none([float(record["sim_entropy"]) for record in target_failures]),
    }


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    residual_records = load_csv(args.residual_csv)
    midrank_records = load_csv(args.midrank_csv)
    merged = merge_records(residual_records, midrank_records)
    if not merged:
        raise RuntimeError("No merged records found. Check that residual_csv and midrank_csv use the same MERGE_KEYS.")

    thresholds = build_flags(merged, args)
    output_csv = os.path.join(args.output_dir, "modulation_midrank_link_records.csv")
    output_json = os.path.join(args.output_dir, "modulation_midrank_link_summary.json")
    write_records_csv(merged, output_csv)

    target_key = "midrank_local_identity_failure"
    summary = {
        "num_merged_points": len(merged),
        "definitions": {
            "midrank_range": [int(args.midrank_range[0]), int(args.midrank_range[1])],
            "near_frac_threshold": args.near_frac_threshold,
            "spread_quantile": args.spread_quantile,
            "centroid_quantile": args.centroid_quantile,
            "target_key": target_key,
            "target_meaning": (
                "Current prediction is wrong, oracle GT rank falls in the specified mid-rank range, "
                "and top-50 candidates remain highly local and compact around the GT neighborhood."
            ),
        },
        "thresholds": thresholds,
        "overall": bucket_summary(merged, target_key),
        "deciles": {
            "shift_ratio_src": decile_summary(merged, "shift_ratio_src", target_key),
            "content_ratio_src": decile_summary(merged, "content_ratio_src", target_key),
            "interaction_ratio_src": decile_summary(merged, "interaction_ratio_src", target_key),
        },
        "binary_flags": {
            "high_shift": binary_group_summary(merged, "high_shift", target_key),
            "low_content": binary_group_summary(merged, "low_content", target_key),
            "negative_interaction": binary_group_summary(merged, "negative_interaction", target_key),
            "high_shift_low_content": binary_group_summary(merged, "high_shift_low_content", target_key),
            "triple_imbalance": binary_group_summary(merged, "triple_imbalance", target_key),
        },
        "two_by_two": {
            "high_shift_x_low_content": two_by_two_summary(merged, "high_shift", "low_content", target_key),
            "high_shift_x_negative_interaction": two_by_two_summary(merged, "high_shift", "negative_interaction", target_key),
            "low_content_x_negative_interaction": two_by_two_summary(merged, "low_content", "negative_interaction", target_key),
        },
        "notes": {
            "interpretation": (
                "If high-shift / low-content groups show clearly higher target failure rates, then modulation imbalance "
                "is not merely correlated with generic failure, but is specifically linked to the local-identity mid-rank suppression pattern."
            ),
            "caution": (
                "This script establishes a stronger behavioral link, not full causal proof. "
                "Small but consistent gaps support an upstream-factor story; large gaps would support a more central mechanism role."
            ),
        },
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved records to: {output_csv}")
    print(f"Saved summary to: {output_json}")
    print(f"Num merged points: {summary['num_merged_points']}")
    print(f"Target failures: {summary['overall']['num_target_failures']}")
    print(f"Target failure rate over all points: {summary['overall']['target_failure_rate_over_all_points']}")
    print("Binary target rates:")
    for key, value in summary["binary_flags"].items():
        print(
            f"  {key}: pos={value['positive_target_rate']} neg={value['negative_target_rate']} gap={value['rate_gap']}"
        )


if __name__ == "__main__":
    main()

import argparse
import csv
import json
import math
import os
from typing import Any

import numpy as np


RANK_BUCKETS = [
    ("rank_2_10", 2, 10),
    ("rank_11_50", 11, 50),
    ("rank_51_100", 51, 100),
    ("rank_101_500", 101, 500),
    ("rank_gt_500", 501, None),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze local identity barriers inside SPair-71k failures. "
            "This script works on midrank_suppression_records.csv and identifies which concrete barrier "
            "most strongly characterizes the failures where the GT stays in the correct local neighborhood "
            "but cannot reach the front ranks."
        )
    )
    parser.add_argument("--midrank_csv", type=str, required=True, help="Path to midrank_suppression_records.csv.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs.")
    parser.add_argument(
        "--local_near_frac_threshold",
        type=float,
        default=0.90,
        help="Minimum top50_near_frac@x1 used to define a locally correct candidate cluster.",
    )
    parser.add_argument(
        "--local_near_frac_x2_threshold",
        type=float,
        default=0.98,
        help="Optional second local-neighborhood constraint using top50_near_frac@x2.",
    )
    parser.add_argument(
        "--target_rank_min",
        type=int,
        default=11,
        help="Minimum oracle rank for the target local-neighborhood failure definition.",
    )
    parser.add_argument(
        "--target_rank_max",
        type=int,
        default=500,
        help="Maximum oracle rank for the target local-neighborhood failure definition.",
    )
    parser.add_argument(
        "--low_margin_quantile",
        type=float,
        default=0.25,
        help="Lower quantile for top1_top50_score_gap or sim_margin to define margin suppression.",
    )
    parser.add_argument(
        "--high_entropy_quantile",
        type=float,
        default=0.75,
        help="Upper quantile for sim_entropy to define diffuse ambiguity.",
    )
    parser.add_argument(
        "--high_spread_quantile",
        type=float,
        default=0.75,
        help="Upper quantile for top50_spread_norm to define cluster diffusion.",
    )
    parser.add_argument(
        "--high_centroid_quantile",
        type=float,
        default=0.75,
        help="Upper quantile for top50_centroid_norm_dist to define centroid drift.",
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


def assign_rank_bucket(oracle_rank: int) -> str:
    for bucket_name, low, high in RANK_BUCKETS:
        if high is None:
            if oracle_rank >= low:
                return bucket_name
        elif low <= oracle_rank <= high:
            return bucket_name
    return "rank_unknown"


def build_barrier_flags(records: list[dict[str, Any]], args) -> dict[str, float]:
    low_gap_thr = quantile_threshold(records, "top1_top50_score_gap", args.low_margin_quantile)
    low_margin_thr = quantile_threshold(records, "sim_margin", args.low_margin_quantile)
    high_entropy_thr = quantile_threshold(records, "sim_entropy", args.high_entropy_quantile)
    high_spread_thr = quantile_threshold(records, "top50_spread_norm", args.high_spread_quantile)
    high_centroid_thr = quantile_threshold(records, "top50_centroid_norm_dist", args.high_centroid_quantile)

    for record in records:
        current_error = 1 - int(record["correct"])
        oracle_rank = int(record["oracle_best_rank"])
        rank_bucket = assign_rank_bucket(oracle_rank)
        local_neighborhood = int(
            float(record["top50_near_frac@x1"]) >= args.local_near_frac_threshold
            and float(record["top50_near_frac@x2"]) >= args.local_near_frac_x2_threshold
        )
        local_identity_failure = int(
            current_error == 1
            and local_neighborhood == 1
            and args.target_rank_min <= oracle_rank <= args.target_rank_max
        )

        barrier_low_gap = int(float(record["top1_top50_score_gap"]) <= low_gap_thr)
        barrier_low_margin = int(float(record["sim_margin"]) <= low_margin_thr)
        barrier_high_entropy = int(float(record["sim_entropy"]) >= high_entropy_thr)
        barrier_high_spread = int(float(record["top50_spread_norm"]) >= high_spread_thr)
        barrier_centroid_drift = int(float(record["top50_centroid_norm_dist"]) >= high_centroid_thr)

        record["current_error"] = current_error
        record["rank_bucket"] = rank_bucket
        record["local_neighborhood"] = local_neighborhood
        record["local_identity_failure"] = local_identity_failure
        record["barrier_low_gap"] = barrier_low_gap
        record["barrier_low_margin"] = barrier_low_margin
        record["barrier_high_entropy"] = barrier_high_entropy
        record["barrier_high_spread"] = barrier_high_spread
        record["barrier_centroid_drift"] = barrier_centroid_drift
        record["barrier_margin_suppression"] = int(barrier_low_gap == 1 and barrier_low_margin == 1)
        record["barrier_diffuse_ambiguity"] = int(barrier_high_entropy == 1 and barrier_high_spread == 1)
        record["barrier_local_drift"] = int(barrier_centroid_drift == 1)
        record["barrier_all_three"] = int(
            record["barrier_margin_suppression"] == 1
            and barrier_high_entropy == 1
            and barrier_centroid_drift == 1
        )

    return {
        "low_gap_threshold": low_gap_thr,
        "low_margin_threshold": low_margin_thr,
        "high_entropy_threshold": high_entropy_thr,
        "high_spread_threshold": high_spread_thr,
        "high_centroid_threshold": high_centroid_thr,
    }


def summarize_target(records: list[dict[str, Any]], target_key: str) -> dict[str, Any]:
    target_records = [record for record in records if int(record[target_key]) == 1]
    return {
        "count": len(target_records),
        "rate_over_all_points": rate(records, target_key),
        "rate_over_failures": None
        if not any(int(record["current_error"]) == 1 for record in records)
        else float(
            len(target_records) / max(sum(int(record["current_error"]) == 1 for record in records), 1)
        ),
        "mean_oracle_rank": mean_or_none([float(record["oracle_best_rank"]) for record in target_records]),
        "mean_norm_dist": mean_or_none([float(record["norm_dist"]) for record in target_records]),
        "mean_sim_margin": mean_or_none([float(record["sim_margin"]) for record in target_records]),
        "mean_sim_entropy": mean_or_none([float(record["sim_entropy"]) for record in target_records]),
    }


def barrier_summary(records: list[dict[str, Any]], target_key: str, barrier_key: str) -> dict[str, Any]:
    target_records = [record for record in records if int(record[target_key]) == 1]
    pos = [record for record in target_records if int(record[barrier_key]) == 1]
    neg = [record for record in target_records if int(record[barrier_key]) == 0]
    return {
        "target_key": target_key,
        "barrier_key": barrier_key,
        "target_count": len(target_records),
        "barrier_positive_count": len(pos),
        "barrier_negative_count": len(neg),
        "barrier_positive_rate_within_target": None if not target_records else float(len(pos) / len(target_records)),
        "positive_mean_oracle_rank": mean_or_none([float(record["oracle_best_rank"]) for record in pos]),
        "negative_mean_oracle_rank": mean_or_none([float(record["oracle_best_rank"]) for record in neg]),
        "positive_mean_sim_margin": mean_or_none([float(record["sim_margin"]) for record in pos]),
        "positive_mean_top1_top50_gap": mean_or_none([float(record["top1_top50_score_gap"]) for record in pos]),
        "positive_mean_entropy": mean_or_none([float(record["sim_entropy"]) for record in pos]),
    }


def rank_bucket_summary(records: list[dict[str, Any]], barrier_keys: list[str]) -> dict[str, Any]:
    failures = [record for record in records if int(record["current_error"]) == 1]
    output: dict[str, Any] = {}
    for bucket_name, _, _ in RANK_BUCKETS:
        subset = [record for record in failures if record["rank_bucket"] == bucket_name]
        if not subset:
            output[bucket_name] = {"count": 0}
            continue
        bucket = {
            "count": len(subset),
            "fraction_of_failures": float(len(subset) / len(failures)),
            "local_neighborhood_rate": rate(subset, "local_neighborhood"),
            "mean_norm_dist": mean_or_none([float(record["norm_dist"]) for record in subset]),
            "mean_sim_margin": mean_or_none([float(record["sim_margin"]) for record in subset]),
            "mean_top1_top50_gap": mean_or_none([float(record["top1_top50_score_gap"]) for record in subset]),
            "mean_sim_entropy": mean_or_none([float(record["sim_entropy"]) for record in subset]),
            "mean_top50_spread_norm": mean_or_none([float(record["top50_spread_norm"]) for record in subset]),
            "mean_top50_centroid_norm_dist": mean_or_none(
                [float(record["top50_centroid_norm_dist"]) for record in subset]
            ),
        }
        for barrier_key in barrier_keys:
            bucket[barrier_key] = rate(subset, barrier_key)
        output[bucket_name] = bucket
    return output


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    records = load_csv(args.midrank_csv)
    if not records:
        raise RuntimeError("No records found in midrank_csv.")

    thresholds = build_barrier_flags(records, args)
    barrier_keys = [
        "barrier_low_gap",
        "barrier_low_margin",
        "barrier_high_entropy",
        "barrier_high_spread",
        "barrier_centroid_drift",
        "barrier_margin_suppression",
        "barrier_diffuse_ambiguity",
        "barrier_local_drift",
        "barrier_all_three",
    ]

    output_csv = os.path.join(args.output_dir, "local_identity_barrier_records.csv")
    output_json = os.path.join(args.output_dir, "local_identity_barrier_summary.json")
    write_records_csv(records, output_csv)

    summary = {
        "num_points": len(records),
        "num_failures": int(sum(1 - int(record["correct"]) for record in records)),
        "thresholds": thresholds,
        "targets": {
            "local_identity_failure": summarize_target(records, "local_identity_failure"),
            "local_neighborhood": summarize_target(records, "local_neighborhood"),
        },
        "barriers_within_local_identity_failure": {
            barrier_key: barrier_summary(records, "local_identity_failure", barrier_key)
            for barrier_key in barrier_keys
        },
        "rank_buckets": rank_bucket_summary(records, barrier_keys),
        "notes": {
            "goal": (
                "Identify which barrier most characterizes failures where the model already focuses on the right local "
                "semantic neighborhood but still cannot lift the GT to the front ranks."
            ),
            "local_identity_failure": (
                "Target is now decoupled from spread and centroid statistics: wrong prediction, oracle GT rank lies in the "
                "configured range, and top-50 candidates remain predominantly local around the GT neighborhood."
            ),
            "barrier_margin_suppression": (
                "Low top1-vs-top50 gap together with low similarity margin: the GT is not separated sharply enough from nearby rivals."
            ),
            "barrier_diffuse_ambiguity": (
                "High entropy with high local spread: many nearby candidates share similar scores, so local identity is diffuse."
            ),
            "barrier_local_drift": (
                "Even when candidates are somewhat local, the candidate cluster center is still biased away from the GT neighborhood."
            ),
        },
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved records to: {output_csv}")
    print(f"Saved summary to: {output_json}")
    print(f"Num points: {summary['num_points']}")
    print(f"Num failures: {summary['num_failures']}")
    print(
        "Local identity failure count:",
        summary["targets"]["local_identity_failure"]["count"],
    )
    print("Barrier rates within local identity failure:")
    for barrier_key, barrier_stats in summary["barriers_within_local_identity_failure"].items():
        print(f"  {barrier_key}: {barrier_stats['barrier_positive_rate_within_target']}")


if __name__ == "__main__":
    main()

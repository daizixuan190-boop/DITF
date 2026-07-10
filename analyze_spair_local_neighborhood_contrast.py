import argparse
import csv
import json
import math
import os
from typing import Any

import numpy as np


LOCAL_GROUPS = [
    ("local_success", None),
    ("local_fail_rank_2_10", (2, 10)),
    ("local_fail_rank_11_50", (11, 50)),
    ("local_fail_rank_51_100", (51, 100)),
    ("local_fail_rank_101_500", (101, 500)),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Contrast samples inside the same local-neighborhood regime on SPair-71k. "
            "This script focuses only on points where the candidate cloud stays local around the GT neighborhood, "
            "then compares local successes against local failures at different oracle-rank severities."
        )
    )
    parser.add_argument(
        "--records_csv",
        type=str,
        required=True,
        help="Path to local_identity_barrier_records.csv.",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs.")
    parser.add_argument(
        "--local_key",
        type=str,
        default="local_neighborhood",
        help="Binary column defining the local-neighborhood regime.",
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


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def rate(records: list[dict[str, Any]], key: str) -> float | None:
    if not records:
        return None
    return float(np.mean([float(record[key]) for record in records]))


def assign_local_group(record: dict[str, Any]) -> str | None:
    if int(record["local_neighborhood"]) != 1:
        return None
    current_error = int(record["current_error"])
    oracle_rank = int(record["oracle_best_rank"])
    if current_error == 0:
        return "local_success"
    for group_name, rank_range in LOCAL_GROUPS[1:]:
        low, high = rank_range
        if low <= oracle_rank <= high:
            return group_name
    return None


def add_family_flags(records: list[dict[str, Any]]):
    for record in records:
        low_gap = int(record["barrier_low_gap"])
        low_margin = int(record["barrier_low_margin"])
        high_entropy = int(record["barrier_high_entropy"])
        high_spread = int(record["barrier_high_spread"])
        centroid_drift = int(record["barrier_centroid_drift"])

        record["cause_margin_family"] = int(low_gap == 1 or low_margin == 1)
        record["cause_diffusion_family"] = int(high_entropy == 1 or high_spread == 1)
        record["cause_drift_family"] = int(centroid_drift == 1)
        record["cause_drift_and_diffusion"] = int(record["cause_drift_family"] == 1 and record["cause_diffusion_family"] == 1)
        record["cause_margin_and_diffusion"] = int(record["cause_margin_family"] == 1 and record["cause_diffusion_family"] == 1)


def group_subset(records: list[dict[str, Any]], group_name: str) -> list[dict[str, Any]]:
    return [record for record in records if record.get("local_group") == group_name]


def summarize_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(records),
        "mean_oracle_rank": mean_or_none([float(record["oracle_best_rank"]) for record in records]),
        "mean_norm_dist": mean_or_none([float(record["norm_dist"]) for record in records]),
        "mean_sim_margin": mean_or_none([float(record["sim_margin"]) for record in records]),
        "mean_top1_top50_gap": mean_or_none([float(record["top1_top50_score_gap"]) for record in records]),
        "mean_sim_entropy": mean_or_none([float(record["sim_entropy"]) for record in records]),
        "mean_top50_spread_norm": mean_or_none([float(record["top50_spread_norm"]) for record in records]),
        "mean_top50_centroid_norm_dist": mean_or_none([float(record["top50_centroid_norm_dist"]) for record in records]),
        "barrier_low_gap": rate(records, "barrier_low_gap"),
        "barrier_low_margin": rate(records, "barrier_low_margin"),
        "barrier_high_entropy": rate(records, "barrier_high_entropy"),
        "barrier_high_spread": rate(records, "barrier_high_spread"),
        "barrier_centroid_drift": rate(records, "barrier_centroid_drift"),
        "cause_margin_family": rate(records, "cause_margin_family"),
        "cause_diffusion_family": rate(records, "cause_diffusion_family"),
        "cause_drift_family": rate(records, "cause_drift_family"),
        "cause_drift_and_diffusion": rate(records, "cause_drift_and_diffusion"),
        "cause_margin_and_diffusion": rate(records, "cause_margin_and_diffusion"),
    }


def compare_to_success(local_success: list[dict[str, Any]], target_group: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [
        "sim_margin",
        "top1_top50_score_gap",
        "sim_entropy",
        "top50_spread_norm",
        "top50_centroid_norm_dist",
        "barrier_low_gap",
        "barrier_low_margin",
        "barrier_high_entropy",
        "barrier_high_spread",
        "barrier_centroid_drift",
        "cause_margin_family",
        "cause_diffusion_family",
        "cause_drift_family",
        "cause_drift_and_diffusion",
        "cause_margin_and_diffusion",
    ]
    comparisons = {}
    for metric in metrics:
        success_values = [float(record[metric]) for record in local_success]
        target_values = [float(record[metric]) for record in target_group]
        comparisons[metric] = {
            "success_mean": mean_or_none(success_values),
            "target_mean": mean_or_none(target_values),
            "target_minus_success": None
            if not success_values or not target_values
            else float(np.mean(target_values) - np.mean(success_values)),
        }
    return comparisons


def monotonic_group_trend(group_summaries: dict[str, dict[str, Any]], metric: str) -> dict[str, Any]:
    ordered_groups = [
        "local_fail_rank_2_10",
        "local_fail_rank_11_50",
        "local_fail_rank_51_100",
        "local_fail_rank_101_500",
    ]
    values = [group_summaries[group][metric] for group in ordered_groups if group in group_summaries]
    valid = [value for value in values if value is not None]
    monotonic_non_decreasing = None
    gain = None
    if len(valid) >= 2:
        monotonic_non_decreasing = bool(np.all(np.diff(np.asarray(valid, dtype=np.float64)) >= -1e-12))
        gain = float(valid[-1] - valid[0])
    return {
        "ordered_groups": ordered_groups,
        "values": {
            group: group_summaries[group][metric] if group in group_summaries else None
            for group in ordered_groups
        },
        "monotonic_non_decreasing": monotonic_non_decreasing,
        "gain_101_500_minus_2_10": gain,
    }


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    records = load_csv(args.records_csv)
    if not records:
        raise RuntimeError("No records found in records_csv.")
    if args.local_key not in records[0]:
        raise RuntimeError(f"Local key '{args.local_key}' not found in records.")

    add_family_flags(records)
    for record in records:
        record["local_group"] = assign_local_group(record)

    local_records = [record for record in records if record["local_group"] is not None]
    group_summaries = {}
    group_records = {}
    for group_name, _ in LOCAL_GROUPS:
        subset = group_subset(local_records, group_name)
        group_records[group_name] = subset
        group_summaries[group_name] = summarize_group(subset)

    local_success = group_records["local_success"]
    contrasts = {
        group_name: compare_to_success(local_success, group_records[group_name])
        for group_name, _ in LOCAL_GROUPS[1:]
    }

    trend_metrics = [
        "barrier_centroid_drift",
        "barrier_high_spread",
        "barrier_high_entropy",
        "barrier_low_margin",
        "barrier_low_gap",
        "cause_drift_family",
        "cause_diffusion_family",
        "cause_margin_family",
        "cause_drift_and_diffusion",
        "cause_margin_and_diffusion",
        "mean_top50_centroid_norm_dist",
        "mean_top50_spread_norm",
        "mean_sim_entropy",
        "mean_sim_margin",
        "mean_top1_top50_gap",
    ]
    trends = {
        metric: monotonic_group_trend(group_summaries, metric)
        for metric in trend_metrics
    }

    summary = {
        "num_points": len(records),
        "num_local_records": len(local_records),
        "local_group_summaries": group_summaries,
        "contrasts_vs_local_success": contrasts,
        "severity_trends_within_local_failures": trends,
        "notes": {
            "goal": (
                "Separate target-specific local-identity mechanisms from broad hard-sample signals by comparing only "
                "samples that already reside in the same local-neighborhood regime."
            ),
            "reading_guide": (
                "A strong target-specific factor should already differ between local_success and local_fail_rank_11_50, "
                "and typically keep worsening as rank severity increases toward local_fail_rank_101_500."
            ),
        },
    }

    output_json = os.path.join(args.output_dir, "local_neighborhood_contrast_summary.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved summary to: {output_json}")
    print(f"Num local records: {summary['num_local_records']}")
    print("Group counts:")
    for group_name, group_summary in group_summaries.items():
        print(f"  {group_name}: {group_summary['count']}")
    print("Key contrasts vs local_success:")
    for group_name in ["local_fail_rank_11_50", "local_fail_rank_51_100", "local_fail_rank_101_500"]:
        comp = contrasts[group_name]
        print(
            f"  {group_name}: "
            f"centroid={comp['top50_centroid_norm_dist']['target_minus_success']}, "
            f"spread={comp['top50_spread_norm']['target_minus_success']}, "
            f"entropy={comp['sim_entropy']['target_minus_success']}, "
            f"margin={comp['sim_margin']['target_minus_success']}"
        )


if __name__ == "__main__":
    main()

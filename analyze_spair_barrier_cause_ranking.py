import argparse
import csv
import json
import math
import os
from typing import Any

import numpy as np


TARGET_BUCKETS = [
    ("rank_11_50", 11, 50),
    ("rank_51_100", 51, 100),
    ("rank_101_500", 101, 500),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Rank candidate barrier causes for SPair local-identity failures. "
            "This script scores each barrier by coverage within the target failures, "
            "enrichment over other failures, and monotonic strengthening across target rank buckets."
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
        "--target_key",
        type=str,
        default="local_identity_failure",
        help="Binary target column to explain.",
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


def rate(records: list[dict[str, Any]], key: str) -> float | None:
    if not records:
        return None
    return float(np.mean([float(record[key]) for record in records]))


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


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
        record["cause_margin_and_diffusion"] = int(record["cause_margin_family"] == 1 and record["cause_diffusion_family"] == 1)
        record["cause_drift_and_diffusion"] = int(record["cause_drift_family"] == 1 and record["cause_diffusion_family"] == 1)


def bucket_subset(records: list[dict[str, Any]], low: int, high: int) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if low <= int(record["oracle_best_rank"]) <= high
    ]


def monotonic_stats(target_records: list[dict[str, Any]], barrier_key: str) -> dict[str, Any]:
    bucket_rates = []
    bucket_counts = []
    for bucket_name, low, high in TARGET_BUCKETS:
        subset = bucket_subset(target_records, low, high)
        bucket_counts.append({"bucket": bucket_name, "count": len(subset)})
        bucket_rates.append(rate(subset, barrier_key) if subset else None)

    valid = [(idx, val) for idx, val in enumerate(bucket_rates) if val is not None]
    monotonic_non_decreasing = None
    slope = None
    gain = None
    if len(valid) >= 2:
        xs = np.asarray([x for x, _ in valid], dtype=np.float64)
        ys = np.asarray([y for _, y in valid], dtype=np.float64)
        monotonic_non_decreasing = bool(np.all(np.diff(ys) >= -1e-12))
        slope = float(np.polyfit(xs, ys, deg=1)[0])
        gain = float(ys[-1] - ys[0])

    return {
        "bucket_rates": {
            bucket_name: bucket_rates[idx]
            for idx, (bucket_name, _, _) in enumerate(TARGET_BUCKETS)
        },
        "bucket_counts": bucket_counts,
        "monotonic_non_decreasing": monotonic_non_decreasing,
        "slope": slope,
        "gain_101_500_minus_11_50": gain,
    }


def analyze_barrier(
    all_records: list[dict[str, Any]],
    target_records: list[dict[str, Any]],
    other_failures: list[dict[str, Any]],
    barrier_key: str,
) -> dict[str, Any]:
    target_rate = rate(target_records, barrier_key)
    other_failure_rate = rate(other_failures, barrier_key)
    enrichment_gap = None if target_rate is None or other_failure_rate is None else float(target_rate - other_failure_rate)
    odds_ratio = None
    if target_rate is not None and other_failure_rate is not None:
        eps = 1e-6
        odds_ratio = float((target_rate + eps) / (other_failure_rate + eps))

    trend = monotonic_stats(target_records, barrier_key)
    coverage = 0.0 if target_rate is None else target_rate
    specificity = 0.0 if enrichment_gap is None else max(enrichment_gap, 0.0)
    trend_gain = 0.0 if trend["gain_101_500_minus_11_50"] is None else max(trend["gain_101_500_minus_11_50"], 0.0)
    evidence_score = float(0.5 * coverage + 0.3 * specificity + 0.2 * trend_gain)

    return {
        "barrier_key": barrier_key,
        "target_rate": target_rate,
        "other_failure_rate": other_failure_rate,
        "enrichment_gap": enrichment_gap,
        "odds_ratio_target_vs_other_failures": odds_ratio,
        "count_in_target": int(sum(int(record[barrier_key]) == 1 for record in target_records)),
        "count_in_other_failures": int(sum(int(record[barrier_key]) == 1 for record in other_failures)),
        "count_all": int(sum(int(record[barrier_key]) == 1 for record in all_records)),
        "trend": trend,
        "evidence_score": evidence_score,
    }


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    records = load_csv(args.records_csv)
    if not records:
        raise RuntimeError("No records found in records_csv.")
    if args.target_key not in records[0]:
        raise RuntimeError(f"Target key '{args.target_key}' not found in records.")

    add_family_flags(records)

    target_records = [record for record in records if int(record[args.target_key]) == 1]
    all_failures = [record for record in records if int(record["current_error"]) == 1]
    other_failures = [record for record in all_failures if int(record[args.target_key]) == 0]

    barrier_keys = [
        "cause_drift_family",
        "cause_diffusion_family",
        "cause_margin_family",
        "cause_drift_and_diffusion",
        "cause_margin_and_diffusion",
        "barrier_centroid_drift",
        "barrier_high_spread",
        "barrier_high_entropy",
        "barrier_low_margin",
        "barrier_low_gap",
        "barrier_local_drift",
        "barrier_diffuse_ambiguity",
        "barrier_margin_suppression",
    ]

    analyses = [
        analyze_barrier(records, target_records, other_failures, barrier_key)
        for barrier_key in barrier_keys
    ]
    analyses.sort(key=lambda item: item["evidence_score"], reverse=True)

    summary = {
        "num_points": len(records),
        "num_failures": len(all_failures),
        "num_target_failures": len(target_records),
        "target_key": args.target_key,
        "target_rate_over_failures": None if not all_failures else float(len(target_records) / len(all_failures)),
        "barrier_rankings": analyses,
        "top_family": analyses[0]["barrier_key"] if analyses else None,
        "notes": {
            "evidence_score": (
                "Transparent ranking score = 0.5 * target coverage + 0.3 * positive enrichment over other failures "
                "+ 0.2 * positive rank-severity gain. It is not a causal proof, but a compact stability-oriented ranking."
            ),
            "interpretation": (
                "A stronger candidate cause should cover many target failures, be more specific to target failures than to "
                "other failures, and become more frequent as oracle rank worsens from 11-50 to 101-500."
            ),
        },
    }

    output_json = os.path.join(args.output_dir, "barrier_cause_ranking_summary.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved summary to: {output_json}")
    print(f"Num target failures: {summary['num_target_failures']}")
    print("Top ranked barriers:")
    for item in analyses[:5]:
        print(
            f"  {item['barrier_key']}: score={item['evidence_score']:.4f} "
            f"target={item['target_rate']:.4f} other={item['other_failure_rate']:.4f} "
            f"gap={item['enrichment_gap']:.4f} gain={item['trend']['gain_101_500_minus_11_50']}"
        )


if __name__ == "__main__":
    main()

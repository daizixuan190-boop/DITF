import argparse
import csv
import json
import os
import re
from typing import Any

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Summarize real-part competition records by success/failure and by rival distance buckets. "
            "This is a cheap post-processing step over real_part_competition_records.csv."
        )
    )
    parser.add_argument("--records_csv", type=str, required=True, help="Path to real_part_competition_records.csv.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs.")
    parser.add_argument(
        "--near_thresholds",
        nargs="+",
        type=float,
        default=[1.0, 2.0, 4.0],
        help="Thresholds for best_other_trg_norm_dist buckets.",
    )
    parser.add_argument("--tag_column", type=str, default="", help="Optional column already present in records_csv to summarize by.")
    parser.add_argument("--tag_values", nargs="*", default=[], help="Optional subset values to summarize; default uses all observed values.")
    return parser.parse_args()


def parse_scalar(value: str) -> Any:
    if value is None or value == "":
        return None
    try:
        num = float(value)
    except ValueError:
        return value
    if abs(num - round(num)) < 1e-12:
        return int(round(num))
    return num


def load_csv(csv_path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({key: parse_scalar(value) for key, value in row.items()})
    return records


def safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def safe_rate(values: list[int]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def detect_context_radii(records: list[dict[str, Any]]) -> list[int]:
    radii = set()
    pattern = re.compile(r"^margin_r(\d+)$")
    for record in records:
        for key in record.keys():
            match = pattern.match(key)
            if match:
                radii.add(int(match.group(1)))
    return sorted(radii)


def assign_rival_bucket(record: dict[str, Any], thresholds: list[float]) -> str:
    dist = record.get("best_other_trg_norm_dist")
    if dist is None:
        return "unknown"
    dist = float(dist)
    prev = 0.0
    for thr in thresholds:
        if dist <= thr:
            return f"({prev:g},{thr:g}]"
        prev = thr
    return f">{thresholds[-1]:g}"


def summarize_subset(records: list[dict[str, Any]], radii: list[int]) -> dict[str, Any]:
    if not records:
        return {"count": 0}
    summary = {
        "count": len(records),
        "failure_rate": safe_rate([1 - int(record["correct"]) for record in records]),
        "gt_beaten_by_other_center_rate": safe_rate([int(record["gt_beaten_by_other_center"]) for record in records]),
        "mean_center_margin": safe_mean([float(record["center_margin"]) for record in records]),
        "mean_gt_rank_among_real_parts_center": safe_mean([float(record["gt_rank_among_real_parts_center"]) for record in records]),
        "mean_best_other_trg_norm_dist": safe_mean([float(record["best_other_trg_norm_dist"]) for record in records if record.get("best_other_trg_norm_dist") is not None]),
        "mean_best_other_src_norm_dist": safe_mean([float(record["best_other_src_norm_dist"]) for record in records if record.get("best_other_src_norm_dist") is not None]),
        "pred_nearest_matches_best_other_rate": safe_rate([int(record["pred_nearest_matches_best_other"]) for record in records if record.get("pred_nearest_matches_best_other") is not None]),
        "context": {},
    }
    for radius in radii:
        summary["context"][f"r{radius}"] = {
            "gt_beaten_rate": safe_rate([int(record[f"gt_beaten_by_other_r{radius}"]) for record in records]),
            "mean_margin": safe_mean([float(record[f"margin_r{radius}"]) for record in records]),
            "mean_margin_gain_vs_center": safe_mean([float(record[f"margin_gain_r{radius}"]) for record in records]),
            "rank_improve_rate_vs_center": safe_rate([int(record[f"rank_improve_r{radius}"]) for record in records]),
            "rank_worsen_rate_vs_center": safe_rate([int(record[f"rank_worsen_r{radius}"]) for record in records]),
        }
    return summary


def summary_diff(a: dict[str, Any], b: dict[str, Any], radii: list[int]) -> dict[str, Any]:
    if a.get("count", 0) == 0 or b.get("count", 0) == 0:
        return {"count_a": a.get("count", 0), "count_b": b.get("count", 0)}

    diff = {
        "count_a": a["count"],
        "count_b": b["count"],
        "gt_beaten_by_other_center_rate_gap": (
            a["gt_beaten_by_other_center_rate"] - b["gt_beaten_by_other_center_rate"]
            if a["gt_beaten_by_other_center_rate"] is not None and b["gt_beaten_by_other_center_rate"] is not None
            else None
        ),
        "mean_center_margin_gap": (
            a["mean_center_margin"] - b["mean_center_margin"]
            if a["mean_center_margin"] is not None and b["mean_center_margin"] is not None
            else None
        ),
        "mean_gt_rank_gap": (
            a["mean_gt_rank_among_real_parts_center"] - b["mean_gt_rank_among_real_parts_center"]
            if a["mean_gt_rank_among_real_parts_center"] is not None and b["mean_gt_rank_among_real_parts_center"] is not None
            else None
        ),
        "mean_best_other_trg_norm_dist_gap": (
            a["mean_best_other_trg_norm_dist"] - b["mean_best_other_trg_norm_dist"]
            if a["mean_best_other_trg_norm_dist"] is not None and b["mean_best_other_trg_norm_dist"] is not None
            else None
        ),
        "pred_nearest_matches_best_other_rate_gap": (
            a["pred_nearest_matches_best_other_rate"] - b["pred_nearest_matches_best_other_rate"]
            if a["pred_nearest_matches_best_other_rate"] is not None and b["pred_nearest_matches_best_other_rate"] is not None
            else None
        ),
        "context": {},
    }
    for radius in radii:
        key = f"r{radius}"
        a_ctx = a["context"].get(key, {})
        b_ctx = b["context"].get(key, {})
        diff["context"][key] = {
            "gt_beaten_rate_gap": (
                a_ctx.get("gt_beaten_rate") - b_ctx.get("gt_beaten_rate")
                if a_ctx.get("gt_beaten_rate") is not None and b_ctx.get("gt_beaten_rate") is not None
                else None
            ),
            "mean_margin_gain_vs_center_gap": (
                a_ctx.get("mean_margin_gain_vs_center") - b_ctx.get("mean_margin_gain_vs_center")
                if a_ctx.get("mean_margin_gain_vs_center") is not None and b_ctx.get("mean_margin_gain_vs_center") is not None
                else None
            ),
            "rank_improve_rate_gap": (
                a_ctx.get("rank_improve_rate_vs_center") - b_ctx.get("rank_improve_rate_vs_center")
                if a_ctx.get("rank_improve_rate_vs_center") is not None and b_ctx.get("rank_improve_rate_vs_center") is not None
                else None
            ),
        }
    return diff


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    records = load_csv(args.records_csv)
    if not records:
        raise RuntimeError("No records found in records_csv.")

    radii = detect_context_radii(records)
    thresholds = sorted(float(x) for x in args.near_thresholds)

    for record in records:
        record["result_group"] = "success" if int(record["correct"]) == 1 else "failure"
        record["rival_bucket"] = assign_rival_bucket(record, thresholds)

    overall = summarize_subset(records, radii)
    successes = summarize_subset([record for record in records if int(record["correct"]) == 1], radii)
    failures = summarize_subset([record for record in records if int(record["correct"]) == 0], radii)

    rival_bucket_values = []
    seen_buckets = []
    for record in records:
        bucket = str(record["rival_bucket"])
        if bucket not in seen_buckets:
            seen_buckets.append(bucket)
    rival_bucket_values = seen_buckets

    by_rival_bucket = {
        bucket: summarize_subset([record for record in records if str(record["rival_bucket"]) == bucket], radii)
        for bucket in rival_bucket_values
    }
    by_rival_bucket_and_result = {
        bucket: {
            "success": summarize_subset(
                [record for record in records if str(record["rival_bucket"]) == bucket and int(record["correct"]) == 1],
                radii,
            ),
            "failure": summarize_subset(
                [record for record in records if str(record["rival_bucket"]) == bucket and int(record["correct"]) == 0],
                radii,
            ),
        }
        for bucket in rival_bucket_values
    }

    tag_summary = {}
    if args.tag_column:
        observed_tag_values = sorted(
            {
                str(record.get(args.tag_column))
                for record in records
                if record.get(args.tag_column) is not None and record.get(args.tag_column) != ""
            }
        )
        tag_values = args.tag_values if args.tag_values else observed_tag_values
        for value in tag_values:
            subset = [record for record in records if str(record.get(args.tag_column)) == str(value)]
            tag_summary[str(value)] = {
                "overall": summarize_subset(subset, radii),
                "success": summarize_subset([record for record in subset if int(record["correct"]) == 1], radii),
                "failure": summarize_subset([record for record in subset if int(record["correct"]) == 0], radii),
            }

    summary = {
        "num_records": len(records),
        "context_radii": radii,
        "near_thresholds": thresholds,
        "overall": overall,
        "success": successes,
        "failure": failures,
        "failure_minus_success": summary_diff(failures, successes, radii),
        "by_rival_bucket": by_rival_bucket,
        "by_rival_bucket_and_result": by_rival_bucket_and_result,
        "tag_column": args.tag_column if args.tag_column else None,
        "by_tag": tag_summary,
        "notes": {
            "interpretation": (
                "If failure samples show a much higher gt_beaten_by_other_center_rate and more negative center_margin than success samples, "
                "raw feature competition against other real parts is failure-specific. "
                "Rival distance buckets then reveal whether that competition is mainly local-neighbor confusion or longer-range repeated-part confusion."
            )
        },
    }

    summary_path = os.path.join(args.output_dir, "real_part_competition_splits_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved summary to: {summary_path}")
    print(
        "Failure vs Success:",
        {
            "failure_count": failures["count"],
            "success_count": successes["count"],
            "gt_beaten_rate_gap": summary["failure_minus_success"].get("gt_beaten_by_other_center_rate_gap"),
            "mean_center_margin_gap": summary["failure_minus_success"].get("mean_center_margin_gap"),
        },
    )
    print("Rival buckets:")
    for bucket, stats in by_rival_bucket.items():
        print(
            f"  {bucket}: count={stats['count']} failure_rate={stats.get('failure_rate')} "
            f"gt_beaten_rate={stats.get('gt_beaten_by_other_center_rate')} "
            f"mean_center_margin={stats.get('mean_center_margin')}"
        )


if __name__ == "__main__":
    main()

import argparse
import csv
import json
import os
from typing import Any

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Post-hoc subgroup analysis for cross-view identity anchoring weakness. "
            "Consumes identity_side_diagnostics_records.csv without recomputing features."
        )
    )
    parser.add_argument("--records_csv", type=str, required=True, help="Path to identity_side_diagnostics_records.csv.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs.")
    parser.add_argument("--radii", nargs="+", type=int, default=[0, 1, 2, 4], help="Radii to summarize.")
    parser.add_argument("--near_thresholds", nargs="+", type=float, default=[1.0, 2.0, 4.0], help="Thresholds for rival distance buckets.")
    parser.add_argument("--difficulty_quantiles", nargs="+", type=float, default=[0.25, 0.5, 0.75], help="Quantiles used for geometry proxy buckets.")
    parser.add_argument("--extra_group_columns", nargs="*", default=["category"], help="Additional categorical columns already present in records_csv to summarize.")
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


def summarize_subset(records: list[dict[str, Any]], radii: list[int]) -> dict[str, Any]:
    if not records:
        return {"count": 0}
    summary = {
        "count": len(records),
        "failure_rate": safe_rate([1 - int(record["correct"]) for record in records]),
        "mean_best_other_trg_norm_dist": safe_mean([float(record["best_other_trg_norm_dist"]) for record in records if record.get("best_other_trg_norm_dist") is not None]),
        "mean_center_margin": safe_mean([float(record["center_margin"]) for record in records if record.get("center_margin") is not None]),
        "mean_gt_rank_center": safe_mean([float(record["gt_rank_among_real_parts_center"]) for record in records if record.get("gt_rank_among_real_parts_center") is not None]),
        "radii": {},
    }
    for radius in radii:
        key = f"r{radius}"
        summary["radii"][key] = {
            "mean_cross_margin": safe_mean([float(record[f"cross_margin_r{radius}"]) for record in records if record.get(f"cross_margin_r{radius}") is not None]),
            "mean_cross_gt_score": safe_mean([float(record[f"cross_gt_score_r{radius}"]) for record in records if record.get(f"cross_gt_score_r{radius}") is not None]),
            "mean_cross_rival_score": safe_mean([float(record[f"cross_rival_score_r{radius}"]) for record in records if record.get(f"cross_rival_score_r{radius}") is not None]),
            "mean_src_rival_sim": safe_mean([float(record[f"src_rival_sim_r{radius}"]) for record in records if record.get(f"src_rival_sim_r{radius}") is not None]),
            "mean_trg_rival_sim": safe_mean([float(record[f"trg_rival_sim_r{radius}"]) for record in records if record.get(f"trg_rival_sim_r{radius}") is not None]),
            "mean_trg_minus_src": safe_mean([float(record[f"trg_minus_src_rival_sim_r{radius}"]) for record in records if record.get(f"trg_minus_src_rival_sim_r{radius}") is not None]),
            "target_more_collapsed_rate": safe_rate([int(record[f"target_more_collapsed_r{radius}"]) for record in records if record.get(f"target_more_collapsed_r{radius}") is not None]),
            "mean_cross_margin_gain_vs_center": safe_mean([float(record[f"cross_margin_gain_r{radius}"]) for record in records if record.get(f"cross_margin_gain_r{radius}") is not None]),
        }
    return summary


def diff_summary(a: dict[str, Any], b: dict[str, Any], radii: list[int]) -> dict[str, Any]:
    if a.get("count", 0) == 0 or b.get("count", 0) == 0:
        return {"count_a": a.get("count", 0), "count_b": b.get("count", 0)}
    diff = {
        "count_a": a["count"],
        "count_b": b["count"],
        "failure_rate_gap": (
            a["failure_rate"] - b["failure_rate"]
            if a.get("failure_rate") is not None and b.get("failure_rate") is not None
            else None
        ),
        "mean_center_margin_gap": (
            a["mean_center_margin"] - b["mean_center_margin"]
            if a.get("mean_center_margin") is not None and b.get("mean_center_margin") is not None
            else None
        ),
        "mean_best_other_trg_norm_dist_gap": (
            a["mean_best_other_trg_norm_dist"] - b["mean_best_other_trg_norm_dist"]
            if a.get("mean_best_other_trg_norm_dist") is not None and b.get("mean_best_other_trg_norm_dist") is not None
            else None
        ),
        "radii": {},
    }
    for radius in radii:
        key = f"r{radius}"
        a_r = a["radii"].get(key, {})
        b_r = b["radii"].get(key, {})
        diff["radii"][key] = {
            "cross_margin_gap": (
                a_r.get("mean_cross_margin") - b_r.get("mean_cross_margin")
                if a_r.get("mean_cross_margin") is not None and b_r.get("mean_cross_margin") is not None
                else None
            ),
            "cross_gt_score_gap": (
                a_r.get("mean_cross_gt_score") - b_r.get("mean_cross_gt_score")
                if a_r.get("mean_cross_gt_score") is not None and b_r.get("mean_cross_gt_score") is not None
                else None
            ),
            "cross_rival_score_gap": (
                a_r.get("mean_cross_rival_score") - b_r.get("mean_cross_rival_score")
                if a_r.get("mean_cross_rival_score") is not None and b_r.get("mean_cross_rival_score") is not None
                else None
            ),
            "src_rival_sim_gap": (
                a_r.get("mean_src_rival_sim") - b_r.get("mean_src_rival_sim")
                if a_r.get("mean_src_rival_sim") is not None and b_r.get("mean_src_rival_sim") is not None
                else None
            ),
            "trg_rival_sim_gap": (
                a_r.get("mean_trg_rival_sim") - b_r.get("mean_trg_rival_sim")
                if a_r.get("mean_trg_rival_sim") is not None and b_r.get("mean_trg_rival_sim") is not None
                else None
            ),
            "trg_minus_src_gap": (
                a_r.get("mean_trg_minus_src") - b_r.get("mean_trg_minus_src")
                if a_r.get("mean_trg_minus_src") is not None and b_r.get("mean_trg_minus_src") is not None
                else None
            ),
            "target_more_collapsed_rate_gap": (
                a_r.get("target_more_collapsed_rate") - b_r.get("target_more_collapsed_rate")
                if a_r.get("target_more_collapsed_rate") is not None and b_r.get("target_more_collapsed_rate") is not None
                else None
            ),
            "cross_margin_gain_gap": (
                a_r.get("mean_cross_margin_gain_vs_center") - b_r.get("mean_cross_margin_gain_vs_center")
                if a_r.get("mean_cross_margin_gain_vs_center") is not None and b_r.get("mean_cross_margin_gain_vs_center") is not None
                else None
            ),
        }
    return diff


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


def assign_quantile_bucket(values: np.ndarray, boundaries: list[float], value: float) -> str:
    if len(values) == 0:
        return "unknown"
    quantiles = [float(np.quantile(values, q)) for q in boundaries]
    prev_label = "min"
    for q, thr in zip(boundaries, quantiles):
        if value <= thr:
            return f"<={q:g}"
        prev_label = f">{q:g}"
    return prev_label


def collect_numeric_field(records: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(record[key]) for record in records if record.get(key) is not None], dtype=np.float64)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    records = load_csv(args.records_csv)
    if not records:
        raise RuntimeError("No records found in records_csv.")

    radii = sorted(int(r) for r in args.radii)
    thresholds = sorted(float(x) for x in args.near_thresholds)

    geometry_keys = [
        "best_other_trg_norm_dist",
        "best_other_src_norm_dist",
    ]
    available_geometry_keys = [key for key in geometry_keys if any(record.get(key) is not None for record in records)]
    geometry_arrays = {key: collect_numeric_field(records, key) for key in available_geometry_keys}

    for record in records:
        record["result_group"] = "success" if int(record["correct"]) == 1 else "failure"
        record["rival_bucket"] = assign_rival_bucket(record, thresholds)
        for key, values in geometry_arrays.items():
            if record.get(key) is None:
                record[f"{key}_bucket"] = "unknown"
            else:
                record[f"{key}_bucket"] = assign_quantile_bucket(values, args.difficulty_quantiles, float(record[key]))

    overall = summarize_subset(records, radii)
    success = summarize_subset([record for record in records if int(record["correct"]) == 1], radii)
    failure = summarize_subset([record for record in records if int(record["correct"]) == 0], radii)

    by_category = {}
    for category in sorted({str(record["category"]) for record in records}):
        subset = [record for record in records if str(record["category"]) == category]
        by_category[category] = {
            "overall": summarize_subset(subset, radii),
            "success": summarize_subset([record for record in subset if int(record["correct"]) == 1], radii),
            "failure": summarize_subset([record for record in subset if int(record["correct"]) == 0], radii),
        }

    rival_buckets = []
    seen = set()
    for record in records:
        bucket = str(record["rival_bucket"])
        if bucket not in seen:
            seen.add(bucket)
            rival_buckets.append(bucket)
    by_rival_bucket = {
        bucket: {
            "overall": summarize_subset([record for record in records if str(record["rival_bucket"]) == bucket], radii),
            "success": summarize_subset([record for record in records if str(record["rival_bucket"]) == bucket and int(record["correct"]) == 1], radii),
            "failure": summarize_subset([record for record in records if str(record["rival_bucket"]) == bucket and int(record["correct"]) == 0], radii),
        }
        for bucket in rival_buckets
    }

    by_geometry_bucket = {}
    for key in available_geometry_keys:
        bucket_key = f"{key}_bucket"
        values = sorted({str(record[bucket_key]) for record in records})
        by_geometry_bucket[key] = {
            value: {
                "overall": summarize_subset([record for record in records if str(record[bucket_key]) == value], radii),
                "success": summarize_subset([record for record in records if str(record[bucket_key]) == value and int(record["correct"]) == 1], radii),
                "failure": summarize_subset([record for record in records if str(record[bucket_key]) == value and int(record["correct"]) == 0], radii),
            }
            for value in values
        }

    by_extra_group = {}
    for column in args.extra_group_columns:
        if not any(column in record for record in records):
            continue
        values = sorted({str(record.get(column)) for record in records if record.get(column) is not None and record.get(column) != ""})
        by_extra_group[column] = {
            value: {
                "overall": summarize_subset([record for record in records if str(record.get(column)) == value], radii),
                "success": summarize_subset([record for record in records if str(record.get(column)) == value and int(record["correct"]) == 1], radii),
                "failure": summarize_subset([record for record in records if str(record.get(column)) == value and int(record["correct"]) == 0], radii),
            }
            for value in values
        }

    summary = {
        "num_records": len(records),
        "context_radii": radii,
        "near_thresholds": thresholds,
        "difficulty_quantiles": args.difficulty_quantiles,
        "overall": overall,
        "success": success,
        "failure": failure,
        "failure_minus_success": diff_summary(failure, success, radii),
        "by_category": by_category,
        "by_rival_bucket": by_rival_bucket,
        "by_geometry_bucket": by_geometry_bucket,
        "by_extra_group": by_extra_group,
        "notes": {
            "interpretation": (
                "Use failure-minus-success gaps in cross_margin to identify which categories, rival-distance buckets, "
                "or geometry-proxy buckets concentrate cross-view identity anchoring weakness. "
                "If the strongest gaps are local-rival buckets, the problem is mainly local-part identity separation; "
                "if the strongest gaps are harder geometry buckets, view/shape change is likely a major driver."
            )
        },
    }

    summary_path = os.path.join(args.output_dir, "identity_anchoring_subgroups_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved summary to: {summary_path}")
    print(
        "Global failure vs success:",
        {
            "failure_count": failure["count"],
            "success_count": success["count"],
            "cross_margin_gap_r0": summary["failure_minus_success"]["radii"].get("r0", {}).get("cross_margin_gap"),
            "src_rival_sim_gap_r0": summary["failure_minus_success"]["radii"].get("r0", {}).get("src_rival_sim_gap"),
            "trg_rival_sim_gap_r0": summary["failure_minus_success"]["radii"].get("r0", {}).get("trg_rival_sim_gap"),
        },
    )
    print("Rival buckets:")
    for bucket, stats in by_rival_bucket.items():
        failure_stats = stats["failure"]
        print(
            f"  {bucket}: failure_count={failure_stats['count']} "
            f"cross_margin_r0={failure_stats['radii'].get('r0', {}).get('mean_cross_margin')}"
        )


if __name__ == "__main__":
    main()

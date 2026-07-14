import argparse
import csv
import json
import math
import os
from collections import defaultdict
from typing import Any

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze whether scale-linked GT damage mainly triggers on near-boundary confusion pairs. "
            "Consumes same_confusion_pair_scale_transitions.csv from the prior scale-sensitivity analysis. "
            "CPU-only post-hoc analysis."
        )
    )
    parser.add_argument("--transitions_csv", type=str, required=True, help="Path to same_confusion_pair_scale_transitions.csv.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs.")
    parser.add_argument(
        "--boundary_abs_thresholds",
        nargs="+",
        type=float,
        default=[0.01, 0.02, 0.05, 0.1],
        help="Absolute baseline-margin thresholds used to define near-boundary buckets.",
    )
    parser.add_argument(
        "--primary_boundary_threshold",
        type=float,
        default=0.05,
        help="Primary |margin| threshold used for the near-boundary vs non-boundary comparison.",
    )
    parser.add_argument(
        "--strong_margin_drop",
        type=float,
        default=-0.01,
        help="Threshold on delta_cross_margin used to define a strong trigger.",
    )
    parser.add_argument(
        "--strong_failure_gain",
        type=float,
        default=0.02,
        help="Threshold on delta_failure_rate used to define a strong trigger.",
    )
    parser.add_argument("--topk", type=int, default=30, help="Number of top trigger pairs to keep in summaries.")
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


def safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def safe_rate(values: list[int]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def require_columns(records: list[dict[str, Any]], columns: list[str]):
    if not records:
        raise RuntimeError("transitions_csv is empty.")
    missing = [column for column in columns if column not in records[0]]
    if missing:
        raise RuntimeError(f"transitions_csv is missing required columns: {missing}")


def sort_records(records: list[dict[str, Any]], key: str, reverse: bool = True) -> list[dict[str, Any]]:
    def sort_value(record: dict[str, Any]):
        value = record.get(key)
        if value is None:
            return -math.inf if reverse else math.inf
        return float(value)

    return sorted(records, key=sort_value, reverse=reverse)


def assign_abs_margin_bucket(abs_margin: float | None, thresholds: list[float]) -> str:
    if abs_margin is None:
        return "unknown"
    previous = 0.0
    for threshold in thresholds:
        if abs_margin <= threshold:
            return f"({previous:g},{threshold:g}]"
        previous = threshold
    return f">{thresholds[-1]:g}"


def assign_margin_regime(margin: float | None, primary_boundary_threshold: float) -> str:
    if margin is None:
        return "unknown"
    if margin < -primary_boundary_threshold:
        return "already_flipped"
    if abs(margin) <= primary_boundary_threshold:
        return "near_boundary"
    return "anchored"


def support_weight(record: dict[str, Any]) -> int:
    return min(int(record["count_a"]), int(record["count_b"]))


def weighted_mean(records: list[dict[str, Any]], value_key: str) -> float | None:
    numerator = 0.0
    denominator = 0.0
    for record in records:
        value = record.get(value_key)
        if value is None:
            continue
        weight = support_weight(record)
        numerator += float(value) * float(weight)
        denominator += float(weight)
    if denominator <= 0.0:
        return None
    return numerator / denominator


def summarize_subset(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"count": 0}
    return {
        "count": len(records),
        "weighted_support_sum": int(sum(support_weight(record) for record in records)),
        "mean_margin_a": safe_mean([float(record["mean_cross_margin_a"]) for record in records if record.get("mean_cross_margin_a") is not None]),
        "mean_failure_rate_a": safe_mean([float(record["failure_rate_a"]) for record in records if record.get("failure_rate_a") is not None]),
        "mean_delta_cross_gt_score": safe_mean([float(record["delta_cross_gt_score"]) for record in records if record.get("delta_cross_gt_score") is not None]),
        "mean_delta_cross_rival_score": safe_mean([float(record["delta_cross_rival_score"]) for record in records if record.get("delta_cross_rival_score") is not None]),
        "mean_delta_cross_margin": safe_mean([float(record["delta_cross_margin"]) for record in records if record.get("delta_cross_margin") is not None]),
        "mean_delta_failure_rate": safe_mean([float(record["delta_failure_rate"]) for record in records if record.get("delta_failure_rate") is not None]),
        "weighted_mean_delta_cross_gt_score": weighted_mean(records, "delta_cross_gt_score"),
        "weighted_mean_delta_cross_rival_score": weighted_mean(records, "delta_cross_rival_score"),
        "weighted_mean_delta_cross_margin": weighted_mean(records, "delta_cross_margin"),
        "weighted_mean_delta_failure_rate": weighted_mean(records, "delta_failure_rate"),
        "preferential_gt_damage_rate": safe_rate(
            [int(record["preferential_gt_damage"]) for record in records if record.get("preferential_gt_damage") is not None]
        ),
        "gt_drop_more_than_rival_rate": safe_rate(
            [int(record["gt_drop_more_than_rival"]) for record in records if record.get("gt_drop_more_than_rival") is not None]
        ),
        "gt_drop_and_rival_not_drop_rate": safe_rate(
            [int(record["gt_drop_and_rival_not_drop"]) for record in records if record.get("gt_drop_and_rival_not_drop") is not None]
        ),
        "failure_gt_drop_more_than_rival_rate": safe_rate(
            [int(record["failure_gt_drop_more_than_rival"]) for record in records if record.get("failure_gt_drop_more_than_rival") is not None]
        ),
        "strong_trigger_rate": safe_rate(
            [int(record["strong_trigger"]) for record in records if record.get("strong_trigger") is not None]
        ),
        "selective_trigger_rate": safe_rate(
            [int(record["selective_trigger"]) for record in records if record.get("selective_trigger") is not None]
        ),
        "failure_selective_trigger_rate": safe_rate(
            [int(record["failure_selective_trigger"]) for record in records if record.get("failure_selective_trigger") is not None]
        ),
    }


def compare_groups(a_records: list[dict[str, Any]], b_records: list[dict[str, Any]]) -> dict[str, Any]:
    a = summarize_subset(a_records)
    b = summarize_subset(b_records)

    def gap(key: str) -> float | None:
        av = a.get(key)
        bv = b.get(key)
        if av is None or bv is None:
            return None
        return float(av - bv)

    return {
        "group_a": a,
        "group_b": b,
        "a_minus_b": {
            "mean_delta_cross_gt_score_gap": gap("mean_delta_cross_gt_score"),
            "mean_delta_cross_rival_score_gap": gap("mean_delta_cross_rival_score"),
            "mean_delta_cross_margin_gap": gap("mean_delta_cross_margin"),
            "mean_delta_failure_rate_gap": gap("mean_delta_failure_rate"),
            "weighted_mean_delta_cross_margin_gap": gap("weighted_mean_delta_cross_margin"),
            "weighted_mean_delta_failure_rate_gap": gap("weighted_mean_delta_failure_rate"),
            "preferential_gt_damage_rate_gap": gap("preferential_gt_damage_rate"),
            "gt_drop_more_than_rival_rate_gap": gap("gt_drop_more_than_rival_rate"),
            "selective_trigger_rate_gap": gap("selective_trigger_rate"),
            "failure_selective_trigger_rate_gap": gap("failure_selective_trigger_rate"),
        },
    }


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    records = load_csv(args.transitions_csv)
    require_columns(
        records,
        [
            "category",
            "kp_idx",
            "best_other_idx_center",
            "scale_a",
            "scale_b",
            "count_a",
            "count_b",
            "failure_rate_a",
            "mean_cross_margin_a",
            "delta_cross_gt_score",
            "delta_cross_rival_score",
            "delta_cross_margin",
            "delta_failure_rate",
            "preferential_gt_damage",
            "gt_drop_more_than_rival",
            "gt_drop_and_rival_not_drop",
            "failure_gt_drop_more_than_rival",
        ],
    )

    thresholds = sorted(float(value) for value in args.boundary_abs_thresholds)
    annotated: list[dict[str, Any]] = []
    for record in records:
        out = dict(record)
        margin_a = None if record.get("mean_cross_margin_a") is None else float(record["mean_cross_margin_a"])
        abs_margin_a = None if margin_a is None else abs(margin_a)
        out["abs_margin_a"] = abs_margin_a
        out["boundary_abs_bucket"] = assign_abs_margin_bucket(abs_margin_a, thresholds)
        out["margin_regime"] = assign_margin_regime(margin_a, args.primary_boundary_threshold)
        out["near_boundary_primary"] = None if abs_margin_a is None else int(abs_margin_a <= args.primary_boundary_threshold)
        out["strong_trigger"] = (
            None
            if record.get("delta_cross_margin") is None or record.get("delta_failure_rate") is None
            else int(float(record["delta_cross_margin"]) <= args.strong_margin_drop and float(record["delta_failure_rate"]) >= args.strong_failure_gain)
        )
        out["selective_trigger"] = (
            None
            if out.get("strong_trigger") is None or record.get("gt_drop_more_than_rival") is None
            else int(int(out["strong_trigger"]) == 1 and int(record["gt_drop_more_than_rival"]) == 1)
        )
        out["failure_selective_trigger"] = (
            None
            if out.get("strong_trigger") is None or record.get("failure_gt_drop_more_than_rival") is None
            else int(int(out["strong_trigger"]) == 1 and int(record["failure_gt_drop_more_than_rival"]) == 1)
        )
        annotated.append(out)

    near_boundary = [record for record in annotated if record.get("near_boundary_primary") == 1]
    non_boundary = [record for record in annotated if record.get("near_boundary_primary") == 0]

    by_transition: dict[str, Any] = {}
    transition_keys = sorted({f"{record['scale_a']}->{record['scale_b']}" for record in annotated})
    for transition_key in transition_keys:
        subset = [
            record
            for record in annotated
            if f"{record['scale_a']}->{record['scale_b']}" == transition_key
        ]
        transition_near = [record for record in subset if record.get("near_boundary_primary") == 1]
        transition_non = [record for record in subset if record.get("near_boundary_primary") == 0]

        by_bucket = {}
        for bucket in sorted({str(record["boundary_abs_bucket"]) for record in subset}):
            bucket_subset = [record for record in subset if str(record["boundary_abs_bucket"]) == bucket]
            by_bucket[bucket] = summarize_subset(bucket_subset)

        by_regime = {}
        for regime in ["anchored", "near_boundary", "already_flipped"]:
            regime_subset = [record for record in subset if str(record["margin_regime"]) == regime]
            by_regime[regime] = summarize_subset(regime_subset)

        by_transition[transition_key] = {
            "overall": summarize_subset(subset),
            "near_vs_non_boundary": compare_groups(transition_near, transition_non),
            "by_abs_bucket": by_bucket,
            "by_regime": by_regime,
            "top_selective_triggers": sort_records(
                [record for record in subset if record.get("selective_trigger") == 1],
                "delta_cross_margin",
                reverse=False,
            )[: args.topk],
            "top_strong_triggers": sort_records(
                [record for record in subset if record.get("strong_trigger") == 1],
                "delta_cross_margin",
                reverse=False,
            )[: args.topk],
        }

    summary = {
        "num_records": len(annotated),
        "primary_boundary_threshold": args.primary_boundary_threshold,
        "boundary_abs_thresholds": thresholds,
        "strong_margin_drop": args.strong_margin_drop,
        "strong_failure_gain": args.strong_failure_gain,
        "overall": summarize_subset(annotated),
        "near_vs_non_boundary": compare_groups(near_boundary, non_boundary),
        "by_transition": by_transition,
        "notes": {
            "interpretation": (
                "If near-boundary pairs show larger negative delta_cross_margin and higher selective-trigger rate "
                "than non-boundary pairs, then scale-linked GT damage is acting mainly as a trigger that pushes "
                "borderline local competitions across the decision boundary."
            )
        },
    }

    annotated_csv = os.path.join(args.output_dir, "near_boundary_trigger_records.csv")
    summary_json = os.path.join(args.output_dir, "near_boundary_trigger_summary.json")
    write_records_csv(annotated, annotated_csv)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved records to: {annotated_csv}")
    print(f"Saved summary to: {summary_json}")
    print(
        "Overall near-vs-non-boundary weighted gap:",
        summary["near_vs_non_boundary"]["a_minus_b"],
    )


if __name__ == "__main__":
    main()

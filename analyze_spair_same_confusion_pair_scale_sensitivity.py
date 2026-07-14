import argparse
import csv
import json
import math
import os
from collections import defaultdict
from typing import Any

import numpy as np


CONFUSION_KEY_FIELDS = ["category", "kp_idx", "best_other_idx_center"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze scale sensitivity on the same GT->rival confusion pair across SPair records. "
            "This is a CPU-only post-hoc analysis over identity_side_diagnostics_records.csv."
        )
    )
    parser.add_argument("--records_csv", type=str, required=True, help="Path to identity_side_diagnostics_records.csv.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs.")
    parser.add_argument("--scale_field", type=str, default="scale_variation", help="Pair-level scale variation field.")
    parser.add_argument("--min_count_per_scale", type=int, default=10, help="Minimum support for a confusion pair within a scale group.")
    parser.add_argument("--topk", type=int, default=30, help="Number of top pair transitions to keep in summary outputs.")
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
        raise RuntimeError("records_csv is empty.")
    missing = [column for column in columns if column not in records[0]]
    if missing:
        raise RuntimeError(f"records_csv is missing required columns: {missing}")


def normalize_scalar(value: Any) -> Any:
    if isinstance(value, float) and abs(value - round(value)) < 1e-12:
        return int(round(value))
    return value


def sort_scale_values(values: set[Any]) -> list[Any]:
    normalized = [normalize_scalar(value) for value in values]
    if all(isinstance(value, (int, float, bool)) for value in normalized):
        return sorted(normalized, key=float)
    return sorted(normalized, key=str)


def record_failure(record: dict[str, Any]) -> int:
    return 1 - int(record["correct"])


def aggregate_pair_scale_stats(records: list[dict[str, Any]], scale_field: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("best_other_idx_center") is None or record.get(scale_field) is None:
            continue
        if int(record["best_other_idx_center"]) == int(record["kp_idx"]):
            continue
        key = tuple(record[field] for field in CONFUSION_KEY_FIELDS) + (normalize_scalar(record[scale_field]),)
        grouped[key].append(record)

    output: list[dict[str, Any]] = []
    for key, subset in grouped.items():
        failure_subset = [record for record in subset if record_failure(record) == 1]
        success_subset = [record for record in subset if record_failure(record) == 0]
        out = {
            "category": key[0],
            "kp_idx": int(key[1]),
            "best_other_idx_center": int(key[2]),
            scale_field: key[3],
            "count": len(subset),
            "failure_count": len(failure_subset),
            "success_count": len(success_subset),
            "failure_rate": safe_rate([record_failure(record) for record in subset]),
            "mean_cross_gt_score_r0": safe_mean([float(record["cross_gt_score_r0"]) for record in subset if record.get("cross_gt_score_r0") is not None]),
            "mean_cross_rival_score_r0": safe_mean([float(record["cross_rival_score_r0"]) for record in subset if record.get("cross_rival_score_r0") is not None]),
            "mean_cross_margin_r0": safe_mean([float(record["cross_margin_r0"]) for record in subset if record.get("cross_margin_r0") is not None]),
            "mean_center_margin": safe_mean([float(record["center_margin"]) for record in subset if record.get("center_margin") is not None]),
            "mean_best_other_trg_norm_dist": safe_mean(
                [float(record["best_other_trg_norm_dist"]) for record in subset if record.get("best_other_trg_norm_dist") is not None]
            ),
            "failure_mean_cross_gt_score_r0": safe_mean(
                [float(record["cross_gt_score_r0"]) for record in failure_subset if record.get("cross_gt_score_r0") is not None]
            ),
            "failure_mean_cross_rival_score_r0": safe_mean(
                [float(record["cross_rival_score_r0"]) for record in failure_subset if record.get("cross_rival_score_r0") is not None]
            ),
            "failure_mean_cross_margin_r0": safe_mean(
                [float(record["cross_margin_r0"]) for record in failure_subset if record.get("cross_margin_r0") is not None]
            ),
            "success_mean_cross_gt_score_r0": safe_mean(
                [float(record["cross_gt_score_r0"]) for record in success_subset if record.get("cross_gt_score_r0") is not None]
            ),
            "success_mean_cross_rival_score_r0": safe_mean(
                [float(record["cross_rival_score_r0"]) for record in success_subset if record.get("cross_rival_score_r0") is not None]
            ),
            "success_mean_cross_margin_r0": safe_mean(
                [float(record["cross_margin_r0"]) for record in success_subset if record.get("cross_margin_r0") is not None]
            ),
        }
        output.append(out)
    return output


def build_transition_records(
    pair_scale_records: list[dict[str, Any]],
    scale_field: str,
    min_count_per_scale: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in pair_scale_records:
        grouped[tuple(record[field] for field in CONFUSION_KEY_FIELDS)].append(record)

    transitions: list[dict[str, Any]] = []
    for key, subset in grouped.items():
        subset = sorted(subset, key=lambda record: str(record[scale_field]))
        by_scale = {normalize_scalar(record[scale_field]): record for record in subset}
        scales = sorted(by_scale.keys(), key=lambda value: float(value) if isinstance(value, (int, float, bool)) else str(value))
        for idx_a in range(len(scales)):
            for idx_b in range(idx_a + 1, len(scales)):
                scale_a = scales[idx_a]
                scale_b = scales[idx_b]
                record_a = by_scale[scale_a]
                record_b = by_scale[scale_b]
                if int(record_a["count"]) < min_count_per_scale or int(record_b["count"]) < min_count_per_scale:
                    continue

                delta_gt = subtract_or_none(record_b.get("mean_cross_gt_score_r0"), record_a.get("mean_cross_gt_score_r0"))
                delta_rival = subtract_or_none(record_b.get("mean_cross_rival_score_r0"), record_a.get("mean_cross_rival_score_r0"))
                delta_margin = subtract_or_none(record_b.get("mean_cross_margin_r0"), record_a.get("mean_cross_margin_r0"))
                delta_failure_rate = subtract_or_none(record_b.get("failure_rate"), record_a.get("failure_rate"))
                delta_best_other_dist = subtract_or_none(
                    record_b.get("mean_best_other_trg_norm_dist"),
                    record_a.get("mean_best_other_trg_norm_dist"),
                )
                delta_center_margin = subtract_or_none(record_b.get("mean_center_margin"), record_a.get("mean_center_margin"))

                out = {
                    "category": key[0],
                    "kp_idx": int(key[1]),
                    "best_other_idx_center": int(key[2]),
                    "scale_a": scale_a,
                    "scale_b": scale_b,
                    "count_a": int(record_a["count"]),
                    "count_b": int(record_b["count"]),
                    "failure_count_a": int(record_a["failure_count"]),
                    "failure_count_b": int(record_b["failure_count"]),
                    "failure_rate_a": record_a.get("failure_rate"),
                    "failure_rate_b": record_b.get("failure_rate"),
                    "mean_cross_gt_score_a": record_a.get("mean_cross_gt_score_r0"),
                    "mean_cross_gt_score_b": record_b.get("mean_cross_gt_score_r0"),
                    "mean_cross_rival_score_a": record_a.get("mean_cross_rival_score_r0"),
                    "mean_cross_rival_score_b": record_b.get("mean_cross_rival_score_r0"),
                    "mean_cross_margin_a": record_a.get("mean_cross_margin_r0"),
                    "mean_cross_margin_b": record_b.get("mean_cross_margin_r0"),
                    "delta_cross_gt_score": delta_gt,
                    "delta_cross_rival_score": delta_rival,
                    "delta_cross_margin": delta_margin,
                    "delta_failure_rate": delta_failure_rate,
                    "delta_best_other_trg_norm_dist": delta_best_other_dist,
                    "delta_center_margin": delta_center_margin,
                    "preferential_gt_damage": (
                        None
                        if delta_margin is None
                        else int(delta_margin < 0.0)
                    ),
                    "gt_drop_more_than_rival": (
                        None
                        if delta_gt is None or delta_rival is None
                        else int(delta_gt < delta_rival)
                    ),
                    "gt_drop_and_rival_not_drop": (
                        None
                        if delta_gt is None or delta_rival is None
                        else int(delta_gt < 0.0 and delta_rival >= 0.0)
                    ),
                    "margin_drop_per_failure_gain": (
                        None
                        if delta_margin is None or delta_failure_rate is None or abs(float(delta_failure_rate)) < 1e-12
                        else float(delta_margin) / float(delta_failure_rate)
                    ),
                }

                # Failure-only paired sensitivity to see whether the same pattern remains within hard points.
                out["failure_delta_cross_gt_score"] = subtract_or_none(
                    record_b.get("failure_mean_cross_gt_score_r0"),
                    record_a.get("failure_mean_cross_gt_score_r0"),
                )
                out["failure_delta_cross_rival_score"] = subtract_or_none(
                    record_b.get("failure_mean_cross_rival_score_r0"),
                    record_a.get("failure_mean_cross_rival_score_r0"),
                )
                out["failure_delta_cross_margin"] = subtract_or_none(
                    record_b.get("failure_mean_cross_margin_r0"),
                    record_a.get("failure_mean_cross_margin_r0"),
                )
                out["failure_gt_drop_more_than_rival"] = (
                    None
                    if out["failure_delta_cross_gt_score"] is None or out["failure_delta_cross_rival_score"] is None
                    else int(out["failure_delta_cross_gt_score"] < out["failure_delta_cross_rival_score"])
                )
                transitions.append(out)
    return transitions


def subtract_or_none(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    return float(a) - float(b)


def summarize_transition_subset(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"count": 0}

    weights = [min(int(record["count_a"]), int(record["count_b"])) for record in records]
    return {
        "count": len(records),
        "weighted_support_sum": int(sum(weights)),
        "mean_delta_cross_gt_score": safe_mean([float(record["delta_cross_gt_score"]) for record in records if record.get("delta_cross_gt_score") is not None]),
        "mean_delta_cross_rival_score": safe_mean([float(record["delta_cross_rival_score"]) for record in records if record.get("delta_cross_rival_score") is not None]),
        "mean_delta_cross_margin": safe_mean([float(record["delta_cross_margin"]) for record in records if record.get("delta_cross_margin") is not None]),
        "mean_delta_failure_rate": safe_mean([float(record["delta_failure_rate"]) for record in records if record.get("delta_failure_rate") is not None]),
        "preferential_gt_damage_rate": safe_rate(
            [int(record["preferential_gt_damage"]) for record in records if record.get("preferential_gt_damage") is not None]
        ),
        "gt_drop_more_than_rival_rate": safe_rate(
            [int(record["gt_drop_more_than_rival"]) for record in records if record.get("gt_drop_more_than_rival") is not None]
        ),
        "gt_drop_and_rival_not_drop_rate": safe_rate(
            [int(record["gt_drop_and_rival_not_drop"]) for record in records if record.get("gt_drop_and_rival_not_drop") is not None]
        ),
        "failure_mean_delta_cross_gt_score": safe_mean(
            [float(record["failure_delta_cross_gt_score"]) for record in records if record.get("failure_delta_cross_gt_score") is not None]
        ),
        "failure_mean_delta_cross_rival_score": safe_mean(
            [float(record["failure_delta_cross_rival_score"]) for record in records if record.get("failure_delta_cross_rival_score") is not None]
        ),
        "failure_mean_delta_cross_margin": safe_mean(
            [float(record["failure_delta_cross_margin"]) for record in records if record.get("failure_delta_cross_margin") is not None]
        ),
        "failure_gt_drop_more_than_rival_rate": safe_rate(
            [int(record["failure_gt_drop_more_than_rival"]) for record in records if record.get("failure_gt_drop_more_than_rival") is not None]
        ),
        "mean_delta_best_other_trg_norm_dist": safe_mean(
            [float(record["delta_best_other_trg_norm_dist"]) for record in records if record.get("delta_best_other_trg_norm_dist") is not None]
        ),
    }


def weighted_mean(records: list[dict[str, Any]], value_key: str, weight_key: str) -> float | None:
    numerator = 0.0
    denominator = 0.0
    for record in records:
        value = record.get(value_key)
        weight = record.get(weight_key)
        if value is None or weight is None:
            continue
        numerator += float(value) * float(weight)
        denominator += float(weight)
    if denominator <= 0.0:
        return None
    return numerator / denominator


def summarize_transition_subset_weighted(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"count": 0}
    weighted_records = []
    for record in records:
        item = dict(record)
        item["transition_weight"] = min(int(record["count_a"]), int(record["count_b"]))
        weighted_records.append(item)
    return {
        "count": len(weighted_records),
        "weighted_mean_delta_cross_gt_score": weighted_mean(weighted_records, "delta_cross_gt_score", "transition_weight"),
        "weighted_mean_delta_cross_rival_score": weighted_mean(weighted_records, "delta_cross_rival_score", "transition_weight"),
        "weighted_mean_delta_cross_margin": weighted_mean(weighted_records, "delta_cross_margin", "transition_weight"),
        "weighted_mean_delta_failure_rate": weighted_mean(weighted_records, "delta_failure_rate", "transition_weight"),
        "weighted_failure_mean_delta_cross_margin": weighted_mean(weighted_records, "failure_delta_cross_margin", "transition_weight"),
    }


def sort_records(records: list[dict[str, Any]], key: str, reverse: bool = True) -> list[dict[str, Any]]:
    def sort_value(record: dict[str, Any]):
        value = record.get(key)
        if value is None:
            return -math.inf if reverse else math.inf
        return float(value)

    return sorted(records, key=sort_value, reverse=reverse)


def top_categories(records: list[dict[str, Any]], key: str, topk: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["category"])].append(record)
    output = []
    for category, subset in grouped.items():
        output.append(
            {
                "category": category,
                "count": len(subset),
                "mean_delta_cross_gt_score": safe_mean([float(record["delta_cross_gt_score"]) for record in subset if record.get("delta_cross_gt_score") is not None]),
                "mean_delta_cross_rival_score": safe_mean([float(record["delta_cross_rival_score"]) for record in subset if record.get("delta_cross_rival_score") is not None]),
                "mean_delta_cross_margin": safe_mean([float(record["delta_cross_margin"]) for record in subset if record.get("delta_cross_margin") is not None]),
                "preferential_gt_damage_rate": safe_rate(
                    [int(record["preferential_gt_damage"]) for record in subset if record.get("preferential_gt_damage") is not None]
                ),
                "mean_delta_failure_rate": safe_mean([float(record["delta_failure_rate"]) for record in subset if record.get("delta_failure_rate") is not None]),
            }
        )
    return sort_records(output, key, reverse=False)[:topk]


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    records = load_csv(args.records_csv)
    require_columns(
        records,
        [
            "category",
            "kp_idx",
            "best_other_idx_center",
            args.scale_field,
            "correct",
            "cross_gt_score_r0",
            "cross_rival_score_r0",
            "cross_margin_r0",
            "center_margin",
            "best_other_trg_norm_dist",
        ],
    )

    scale_values = sort_scale_values({record[args.scale_field] for record in records if record.get(args.scale_field) is not None})
    if len(scale_values) < 2:
        raise RuntimeError(f"Need at least two values in scale_field={args.scale_field}, got {scale_values}")

    pair_scale_records = aggregate_pair_scale_stats(records, args.scale_field)
    transitions = build_transition_records(pair_scale_records, args.scale_field, args.min_count_per_scale)
    if not transitions:
        raise RuntimeError("No valid same-confusion-pair transitions found. Lower --min_count_per_scale or inspect records_csv.")

    transition_csv = os.path.join(args.output_dir, "same_confusion_pair_scale_transitions.csv")
    pair_scale_csv = os.path.join(args.output_dir, "same_confusion_pair_scale_stats.csv")
    summary_json = os.path.join(args.output_dir, "same_confusion_pair_scale_sensitivity_summary.json")

    write_records_csv(pair_scale_records, pair_scale_csv)
    write_records_csv(transitions, transition_csv)

    by_transition = {}
    for idx_a in range(len(scale_values)):
        for idx_b in range(idx_a + 1, len(scale_values)):
            scale_a = scale_values[idx_a]
            scale_b = scale_values[idx_b]
            subset = [
                record
                for record in transitions
                if normalize_scalar(record["scale_a"]) == scale_a and normalize_scalar(record["scale_b"]) == scale_b
            ]
            key = f"{scale_a}->{scale_b}"
            by_transition[key] = {
                "unweighted": summarize_transition_subset(subset),
                "weighted": summarize_transition_subset_weighted(subset),
                "top_margin_drops": sort_records(
                    [record for record in subset if record.get("delta_cross_margin") is not None],
                    "delta_cross_margin",
                    reverse=False,
                )[: args.topk],
                "top_gt_drops": sort_records(
                    [record for record in subset if record.get("delta_cross_gt_score") is not None],
                    "delta_cross_gt_score",
                    reverse=False,
                )[: args.topk],
                "top_category_margin_drops": top_categories(subset, "mean_delta_cross_margin", args.topk),
            }

    summary = {
        "num_records": len(records),
        "num_pair_scale_records": len(pair_scale_records),
        "num_transitions": len(transitions),
        "scale_field": args.scale_field,
        "scale_values": scale_values,
        "min_count_per_scale": args.min_count_per_scale,
        "overall": {
            "unweighted": summarize_transition_subset(transitions),
            "weighted": summarize_transition_subset_weighted(transitions),
        },
        "by_transition": by_transition,
        "binding_flags": {
            "overall_preferential_gt_damage_majority": (
                None
                if summary_placeholder_rate(transitions, "preferential_gt_damage") is None
                else summary_placeholder_rate(transitions, "preferential_gt_damage") > 0.5
            ),
            "overall_gt_drop_more_than_rival_majority": (
                None
                if summary_placeholder_rate(transitions, "gt_drop_more_than_rival") is None
                else summary_placeholder_rate(transitions, "gt_drop_more_than_rival") > 0.5
            ),
            "overall_failure_gt_drop_more_than_rival_majority": (
                None
                if summary_placeholder_rate(transitions, "failure_gt_drop_more_than_rival") is None
                else summary_placeholder_rate(transitions, "failure_gt_drop_more_than_rival") > 0.5
            ),
        },
        "notes": {
            "interpretation": (
                "If the same confusion pair shows more negative delta on cross_gt_score than on cross_rival_score "
                "when moving to higher scale-variation groups, then scale is selectively damaging GT identity anchoring "
                "rather than uniformly degrading all local candidates."
            )
        },
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved pair-scale stats to: {pair_scale_csv}")
    print(f"Saved transition records to: {transition_csv}")
    print(f"Saved summary to: {summary_json}")
    print(
        "Overall weighted deltas:",
        {
            "delta_cross_gt_score": summary["overall"]["weighted"]["weighted_mean_delta_cross_gt_score"],
            "delta_cross_rival_score": summary["overall"]["weighted"]["weighted_mean_delta_cross_rival_score"],
            "delta_cross_margin": summary["overall"]["weighted"]["weighted_mean_delta_cross_margin"],
            "delta_failure_rate": summary["overall"]["weighted"]["weighted_mean_delta_failure_rate"],
        },
    )


def summary_placeholder_rate(records: list[dict[str, Any]], key: str) -> float | None:
    values = [int(record[key]) for record in records if record.get(key) is not None]
    return safe_rate(values)


if __name__ == "__main__":
    main()

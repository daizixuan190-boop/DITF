import argparse
import csv
import json
import math
import os
from typing import Any

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze whether scale-linked anchor erosion grows with baseline GT-vs-rival margin strength. "
            "Consumes same_confusion_pair_scale_transitions.csv from the prior scale-sensitivity analysis. "
            "CPU-only post-hoc analysis."
        )
    )
    parser.add_argument("--transitions_csv", type=str, required=True, help="Path to same_confusion_pair_scale_transitions.csv.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs.")
    parser.add_argument(
        "--boundary_margin",
        type=float,
        default=0.05,
        help="Threshold used to separate anchored / near-boundary / already-flipped baseline regimes.",
    )
    parser.add_argument(
        "--num_bins",
        type=int,
        default=5,
        help="Requested number of quantile bins for continuous baseline-margin-strength analysis.",
    )
    parser.add_argument(
        "--min_bin_count",
        type=int,
        default=10,
        help="Minimum number of records required for a valid quantile bin.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=30,
        help="Number of top erosion transitions to keep in the summary.",
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
    with open(csv_path, "r", encoding="utf-8-sig") as f:
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


def require_columns(records: list[dict[str, Any]], columns: list[str]):
    if not records:
        raise RuntimeError("transitions_csv is empty.")
    missing = [column for column in columns if column not in records[0]]
    if missing:
        raise RuntimeError(f"transitions_csv is missing required columns: {missing}")


def safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def safe_rate(values: list[int]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


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


def weighted_corr(records: list[dict[str, Any]], x_key: str, y_key: str) -> float | None:
    xs = []
    ys = []
    ws = []
    for record in records:
        x = record.get(x_key)
        y = record.get(y_key)
        if x is None or y is None:
            continue
        xs.append(float(x))
        ys.append(float(y))
        ws.append(float(support_weight(record)))
    if len(xs) < 2:
        return None

    x_arr = np.asarray(xs, dtype=np.float64)
    y_arr = np.asarray(ys, dtype=np.float64)
    w_arr = np.asarray(ws, dtype=np.float64)
    weight_sum = float(np.sum(w_arr))
    if weight_sum <= 0.0:
        return None

    x_mean = float(np.sum(w_arr * x_arr) / weight_sum)
    y_mean = float(np.sum(w_arr * y_arr) / weight_sum)
    x_center = x_arr - x_mean
    y_center = y_arr - y_mean
    cov = float(np.sum(w_arr * x_center * y_center) / weight_sum)
    var_x = float(np.sum(w_arr * x_center * x_center) / weight_sum)
    var_y = float(np.sum(w_arr * y_center * y_center) / weight_sum)
    if var_x <= 0.0 or var_y <= 0.0:
        return None
    return cov / math.sqrt(var_x * var_y)


def assign_margin_regime(margin: float | None, boundary_margin: float) -> str:
    if margin is None:
        return "unknown"
    if margin < -boundary_margin:
        return "already_flipped"
    if abs(margin) <= boundary_margin:
        return "near_boundary"
    return "anchored"


def format_interval(low: float, high: float) -> str:
    return f"({low:.6g},{high:.6g}]"


def unique_sorted(values: list[float]) -> list[float]:
    return sorted({float(value) for value in values})


def build_quantile_edges(values: list[float], num_bins: int) -> list[float]:
    if not values:
        return []
    if num_bins < 1:
        raise ValueError("num_bins must be >= 1")

    array = np.asarray(values, dtype=np.float64)
    raw_edges = np.quantile(array, np.linspace(0.0, 1.0, num_bins + 1))
    edges = unique_sorted(raw_edges.tolist())
    if len(edges) < 2:
        edges = [float(np.min(array)), float(np.max(array))]
    return edges


def assign_quantile_bucket(value: float | None, edges: list[float], prefix: str) -> str:
    if value is None:
        return "unknown"
    if len(edges) < 2:
        return f"{prefix}_Q1{format_interval(float(value), float(value))}"

    for idx in range(1, len(edges)):
        low = edges[idx - 1]
        high = edges[idx]
        if idx == len(edges) - 1:
            if float(value) <= high:
                return f"{prefix}_Q{idx}{format_interval(low, high)}"
        if float(value) <= high:
            return f"{prefix}_Q{idx}{format_interval(low, high)}"
    return f"{prefix}_Q{len(edges) - 1}{format_interval(edges[-2], edges[-1])}"


def sort_records(records: list[dict[str, Any]], key: str, reverse: bool = True) -> list[dict[str, Any]]:
    def sort_value(record: dict[str, Any]):
        value = record.get(key)
        if value is None:
            return -math.inf if reverse else math.inf
        return float(value)

    return sorted(records, key=sort_value, reverse=reverse)


def quantile_bucket_sort_key(name: str) -> tuple[int, str]:
    if "_Q" not in name:
        return (10**9, name)
    suffix = name.split("_Q", 1)[1]
    digits = []
    for ch in suffix:
        if ch.isdigit():
            digits.append(ch)
        else:
            break
    if not digits:
        return (10**9, name)
    return (int("".join(digits)), name)


def summarize_subset(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"count": 0}

    return {
        "count": len(records),
        "weighted_support_sum": int(sum(support_weight(record) for record in records)),
        "mean_margin_a": safe_mean([float(record["mean_cross_margin_a"]) for record in records if record.get("mean_cross_margin_a") is not None]),
        "mean_abs_margin_a": safe_mean([float(record["abs_margin_a"]) for record in records if record.get("abs_margin_a") is not None]),
        "mean_positive_margin_strength": safe_mean(
            [float(record["positive_margin_strength"]) for record in records if record.get("positive_margin_strength") is not None]
        ),
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
        "failure_gt_drop_more_than_rival_rate": safe_rate(
            [int(record["failure_gt_drop_more_than_rival"]) for record in records if record.get("failure_gt_drop_more_than_rival") is not None]
        ),
        "weighted_corr_abs_margin_vs_delta_margin": weighted_corr(records, "abs_margin_a", "delta_cross_margin"),
        "weighted_corr_abs_margin_vs_delta_failure_rate": weighted_corr(records, "abs_margin_a", "delta_failure_rate"),
        "weighted_corr_positive_strength_vs_delta_margin": weighted_corr(records, "positive_margin_strength", "delta_cross_margin"),
        "weighted_corr_positive_strength_vs_delta_failure_rate": weighted_corr(records, "positive_margin_strength", "delta_failure_rate"),
    }


def summarize_bins(records: list[dict[str, Any]], bucket_key: str, min_bin_count: int) -> dict[str, Any]:
    bucket_names = sorted(
        {str(record[bucket_key]) for record in records if record.get(bucket_key) is not None},
        key=quantile_bucket_sort_key,
    )
    output: dict[str, Any] = {}
    for bucket_name in bucket_names:
        subset = [record for record in records if str(record.get(bucket_key)) == bucket_name]
        summary = summarize_subset(subset)
        summary["eligible"] = int(len(subset) >= min_bin_count)
        output[bucket_name] = summary
    return output


def compare_weakest_vs_strongest(bin_summary: dict[str, Any]) -> dict[str, Any] | None:
    eligible_names = [name for name, stats in bin_summary.items() if int(stats.get("eligible", 0)) == 1]
    if len(eligible_names) < 2:
        return None

    weakest = bin_summary[eligible_names[0]]
    strongest = bin_summary[eligible_names[-1]]

    def diff(key: str) -> float | None:
        a = strongest.get(key)
        b = weakest.get(key)
        if a is None or b is None:
            return None
        return float(a - b)

    return {
        "weakest_bin": eligible_names[0],
        "strongest_bin": eligible_names[-1],
        "strongest_minus_weakest": {
            "weighted_mean_delta_cross_gt_score_gap": diff("weighted_mean_delta_cross_gt_score"),
            "weighted_mean_delta_cross_rival_score_gap": diff("weighted_mean_delta_cross_rival_score"),
            "weighted_mean_delta_cross_margin_gap": diff("weighted_mean_delta_cross_margin"),
            "weighted_mean_delta_failure_rate_gap": diff("weighted_mean_delta_failure_rate"),
            "preferential_gt_damage_rate_gap": diff("preferential_gt_damage_rate"),
            "gt_drop_more_than_rival_rate_gap": diff("gt_drop_more_than_rival_rate"),
        },
    }


def annotate_records(records: list[dict[str, Any]], boundary_margin: float, num_bins: int) -> tuple[list[dict[str, Any]], dict[str, list[float]]]:
    annotated: list[dict[str, Any]] = []
    abs_margins: list[float] = []
    positive_strengths: list[float] = []
    anchored_positive_strengths: list[float] = []

    for record in records:
        out = dict(record)
        margin_a = None if record.get("mean_cross_margin_a") is None else float(record["mean_cross_margin_a"])
        abs_margin_a = None if margin_a is None else abs(margin_a)
        positive_strength = None if margin_a is None or margin_a <= 0.0 else float(margin_a)
        margin_regime = assign_margin_regime(margin_a, boundary_margin)

        out["margin_regime"] = margin_regime
        out["abs_margin_a"] = abs_margin_a
        out["positive_margin_strength"] = positive_strength
        out["is_anchored"] = None if margin_regime == "unknown" else int(margin_regime == "anchored")

        if abs_margin_a is not None:
            abs_margins.append(abs_margin_a)
        if positive_strength is not None:
            positive_strengths.append(positive_strength)
            if margin_regime == "anchored":
                anchored_positive_strengths.append(positive_strength)
        annotated.append(out)

    abs_edges = build_quantile_edges(abs_margins, num_bins)
    pos_edges = build_quantile_edges(positive_strengths, num_bins)
    anchored_edges = build_quantile_edges(anchored_positive_strengths, num_bins)

    for out in annotated:
        out["abs_margin_quantile"] = assign_quantile_bucket(out.get("abs_margin_a"), abs_edges, "abs")
        out["positive_margin_quantile"] = assign_quantile_bucket(out.get("positive_margin_strength"), pos_edges, "pos")
        if out.get("is_anchored") == 1:
            out["anchored_strength_quantile"] = assign_quantile_bucket(
                out.get("positive_margin_strength"),
                anchored_edges,
                "anchored",
            )
        else:
            out["anchored_strength_quantile"] = "not_anchored"

    return annotated, {
        "abs_margin_edges": abs_edges,
        "positive_margin_edges": pos_edges,
        "anchored_positive_margin_edges": anchored_edges,
    }


def summarize_transition_groups(records: list[dict[str, Any]], min_bin_count: int, topk: int) -> dict[str, Any]:
    by_transition: dict[str, Any] = {}
    transition_keys = sorted({f"{record['scale_a']}->{record['scale_b']}" for record in records})
    for transition_key in transition_keys:
        subset = [record for record in records if f"{record['scale_a']}->{record['scale_b']}" == transition_key]
        anchored_subset = [record for record in subset if record.get("is_anchored") == 1]
        abs_bins = summarize_bins(subset, "abs_margin_quantile", min_bin_count)
        anchored_bins = summarize_bins(anchored_subset, "anchored_strength_quantile", min_bin_count)

        by_transition[transition_key] = {
            "overall": summarize_subset(subset),
            "anchored_only": summarize_subset(anchored_subset),
            "by_abs_margin_quantile": abs_bins,
            "by_anchored_strength_quantile": anchored_bins,
            "anchored_strongest_vs_weakest": compare_weakest_vs_strongest(anchored_bins),
            "top_anchor_erosion_cases": sort_records(
                [record for record in subset if record.get("delta_cross_margin") is not None],
                "delta_cross_margin",
                reverse=False,
            )[:topk],
        }
    return by_transition


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
            "mean_cross_margin_a",
            "delta_cross_gt_score",
            "delta_cross_rival_score",
            "delta_cross_margin",
            "delta_failure_rate",
            "preferential_gt_damage",
            "gt_drop_more_than_rival",
            "failure_gt_drop_more_than_rival",
        ],
    )

    annotated, edge_summary = annotate_records(records, args.boundary_margin, args.num_bins)

    anchored_subset = [record for record in annotated if record.get("is_anchored") == 1]
    near_boundary_subset = [record for record in annotated if str(record.get("margin_regime")) == "near_boundary"]
    already_flipped_subset = [record for record in annotated if str(record.get("margin_regime")) == "already_flipped"]

    overall_abs_bins = summarize_bins(annotated, "abs_margin_quantile", args.min_bin_count)
    overall_pos_bins = summarize_bins(
        [record for record in annotated if record.get("positive_margin_strength") is not None],
        "positive_margin_quantile",
        args.min_bin_count,
    )
    anchored_bins = summarize_bins(anchored_subset, "anchored_strength_quantile", args.min_bin_count)

    summary = {
        "num_records": len(annotated),
        "boundary_margin": args.boundary_margin,
        "num_bins_requested": args.num_bins,
        "min_bin_count": args.min_bin_count,
        "bin_edges": edge_summary,
        "overall": summarize_subset(annotated),
        "regimes": {
            "anchored": summarize_subset(anchored_subset),
            "near_boundary": summarize_subset(near_boundary_subset),
            "already_flipped": summarize_subset(already_flipped_subset),
        },
        "by_abs_margin_quantile": overall_abs_bins,
        "by_positive_margin_quantile": overall_pos_bins,
        "by_anchored_strength_quantile": anchored_bins,
        "anchored_strongest_vs_weakest": compare_weakest_vs_strongest(anchored_bins),
        "by_transition": summarize_transition_groups(annotated, args.min_bin_count, args.topk),
        "notes": {
            "interpretation": (
                "If stronger baseline positive margins show more negative delta_cross_margin and more positive "
                "delta_failure_rate under larger scale transitions, then scale is eroding previously anchored "
                "GT-vs-rival identity separation rather than only flipping already borderline cases."
            )
        },
    }

    records_csv = os.path.join(args.output_dir, "anchor_erosion_margin_strength_records.csv")
    summary_json = os.path.join(args.output_dir, "anchor_erosion_margin_strength_summary.json")

    write_records_csv(annotated, records_csv)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved records to: {records_csv}")
    print(f"Saved summary to: {summary_json}")
    print(
        "Anchored strongest-vs-weakest weighted gap:",
        None if summary["anchored_strongest_vs_weakest"] is None else summary["anchored_strongest_vs_weakest"]["strongest_minus_weakest"],
    )


if __name__ == "__main__":
    main()

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from typing import Any

import numpy as np


PAIR_KEY_FIELDS = ["category", "pair_name", "src_imname", "trg_imname"]
CONFUSION_KEY_FIELDS = ["category", "kp_idx", "best_other_idx_center"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Bind scale variation to the current SPair residual-failure mechanism chain using only "
            "existing identity-side diagnostics records. "
            "This is a CPU-only post-hoc analysis."
        )
    )
    parser.add_argument("--records_csv", type=str, required=True, help="Path to identity_side_diagnostics_records.csv.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs.")
    parser.add_argument("--scale_field", type=str, default="scale_variation", help="Pair-level scale variation field in records_csv.")
    parser.add_argument("--local_thresholds", nargs="+", type=float, default=[1.0, 2.0, 4.0], help="Thresholds for local rival-distance buckets.")
    parser.add_argument("--min_pair_count", type=int, default=10, help="Minimum support for confusion-pair structure summaries.")
    parser.add_argument("--topk", type=int, default=20, help="Number of top confusion pairs per scale to keep.")
    parser.add_argument("--bidirectional_thresholds", nargs="+", type=float, default=[0.5, 0.8], help="Thresholds for strong bidirectional confusion coverage.")
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


def normalized_scalar(value: Any) -> Any:
    if isinstance(value, float) and abs(value - round(value)) < 1e-12:
        return int(round(value))
    return value


def sorted_group_values(values: set[Any]) -> list[Any]:
    normalized = [normalized_scalar(value) for value in values]
    if all(isinstance(value, (int, float, bool)) for value in normalized):
        return sorted(normalized, key=float)
    return sorted(normalized, key=str)


def record_failure(record: dict[str, Any]) -> int:
    return 1 - int(record["correct"])


def assign_rival_bucket(dist: float | None, thresholds: list[float]) -> str:
    if dist is None:
        return "unknown"
    previous = 0.0
    for threshold in thresholds:
        if float(dist) <= threshold:
            return f"({previous:g},{threshold:g}]"
        previous = threshold
    return f">{thresholds[-1]:g}"


def extract_float(record: dict[str, Any], key: str) -> float | None:
    value = record.get(key)
    if value is None:
        return None
    return float(value)


def summarize_records(records: list[dict[str, Any]], thresholds: list[float]) -> dict[str, Any]:
    if not records:
        return {"count": 0}

    summary = {
        "count": len(records),
        "failure_rate": safe_rate([record_failure(record) for record in records]),
        "mean_center_margin": safe_mean([float(record["center_margin"]) for record in records if record.get("center_margin") is not None]),
        "mean_cross_margin_r0": safe_mean([float(record["cross_margin_r0"]) for record in records if record.get("cross_margin_r0") is not None]),
        "mean_src_rival_sim_r0": safe_mean([float(record["src_rival_sim_r0"]) for record in records if record.get("src_rival_sim_r0") is not None]),
        "mean_trg_rival_sim_r0": safe_mean([float(record["trg_rival_sim_r0"]) for record in records if record.get("trg_rival_sim_r0") is not None]),
        "mean_trg_minus_src_r0": safe_mean(
            [float(record["trg_minus_src_rival_sim_r0"]) for record in records if record.get("trg_minus_src_rival_sim_r0") is not None]
        ),
        "target_more_collapsed_rate_r0": safe_rate(
            [int(record["target_more_collapsed_r0"]) for record in records if record.get("target_more_collapsed_r0") is not None]
        ),
        "mean_best_other_trg_norm_dist": safe_mean(
            [float(record["best_other_trg_norm_dist"]) for record in records if record.get("best_other_trg_norm_dist") is not None]
        ),
        "rival_bucket_rates": {},
        "gt_beaten_by_other_center_rate": safe_rate(
            [int(record["gt_beaten_by_other_center"]) for record in records if record.get("gt_beaten_by_other_center") is not None]
        ),
    }

    for threshold in thresholds:
        key = f"<=x{threshold:g}"
        summary["rival_bucket_rates"][key] = safe_rate(
            [
                int(float(record["best_other_trg_norm_dist"]) <= threshold)
                for record in records
                if record.get("best_other_trg_norm_dist") is not None
            ]
        )
    return summary


def diff_success_failure(records: list[dict[str, Any]], thresholds: list[float]) -> dict[str, Any]:
    success = summarize_records([record for record in records if int(record["correct"]) == 1], thresholds)
    failure = summarize_records([record for record in records if int(record["correct"]) == 0], thresholds)

    def gap(key: str) -> float | None:
        a = failure.get(key)
        b = success.get(key)
        if a is None or b is None:
            return None
        return float(a - b)

    out = {
        "success": success,
        "failure": failure,
        "failure_minus_success": {
            "failure_rate_gap": gap("failure_rate"),
            "center_margin_gap": gap("mean_center_margin"),
            "cross_margin_r0_gap": gap("mean_cross_margin_r0"),
            "src_rival_sim_r0_gap": gap("mean_src_rival_sim_r0"),
            "trg_rival_sim_r0_gap": gap("mean_trg_rival_sim_r0"),
            "trg_minus_src_r0_gap": gap("mean_trg_minus_src_r0"),
            "target_more_collapsed_rate_r0_gap": gap("target_more_collapsed_rate_r0"),
            "best_other_trg_norm_dist_gap": gap("mean_best_other_trg_norm_dist"),
            "gt_beaten_by_other_center_rate_gap": gap("gt_beaten_by_other_center_rate"),
            "rival_bucket_rate_gaps": {},
        },
    }
    for bucket_key in success.get("rival_bucket_rates", {}).keys():
        a = failure.get("rival_bucket_rates", {}).get(bucket_key)
        b = success.get("rival_bucket_rates", {}).get(bucket_key)
        out["failure_minus_success"]["rival_bucket_rate_gaps"][bucket_key] = None if a is None or b is None else float(a - b)
    return out


def aggregate_confusion_pairs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("best_other_idx_center") is None:
            continue
        if int(record["best_other_idx_center"]) == int(record["kp_idx"]):
            continue
        grouped[tuple(record[field] for field in CONFUSION_KEY_FIELDS)].append(record)

    total_failures = sum(record_failure(record) for record in records)
    total_count = len(records)
    global_failure_rate = total_failures / max(total_count, 1)

    aggregated: list[dict[str, Any]] = []
    for key, subset in grouped.items():
        failures = [record for record in subset if record_failure(record) == 1]
        failure_count = len(failures)
        pair_record = {
            "category": key[0],
            "kp_idx": int(key[1]),
            "best_other_idx_center": int(key[2]),
            "count": len(subset),
            "failure_count": failure_count,
            "success_count": len(subset) - failure_count,
            "failure_rate": failure_count / max(len(subset), 1),
            "failure_rate_gap_vs_global": failure_count / max(len(subset), 1) - global_failure_rate,
            "failure_share": failure_count / max(total_failures, 1),
            "mean_cross_margin_r0": safe_mean([float(record["cross_margin_r0"]) for record in subset if record.get("cross_margin_r0") is not None]),
            "mean_center_margin": safe_mean([float(record["center_margin"]) for record in subset if record.get("center_margin") is not None]),
            "mean_best_other_trg_norm_dist": safe_mean(
                [float(record["best_other_trg_norm_dist"]) for record in subset if record.get("best_other_trg_norm_dist") is not None]
            ),
        }
        aggregated.append(pair_record)
    return aggregated


def attach_reverse_pair_stats(pair_records: list[dict[str, Any]]):
    lookup = {
        (str(record["category"]), int(record["kp_idx"]), int(record["best_other_idx_center"])): record
        for record in pair_records
    }
    for record in pair_records:
        reverse = lookup.get((str(record["category"]), int(record["best_other_idx_center"]), int(record["kp_idx"])))
        reverse_failure_count = None if reverse is None else int(reverse["failure_count"])
        record["reverse_exists"] = int(reverse is not None)
        record["reverse_failure_count"] = reverse_failure_count
        if reverse_failure_count is None:
            record["bidirectional_failure_min_share"] = None
        else:
            denom = max(int(record["failure_count"]), reverse_failure_count, 1)
            record["bidirectional_failure_min_share"] = min(int(record["failure_count"]), reverse_failure_count) / denom


def sort_records(records: list[dict[str, Any]], key: str, reverse: bool = True) -> list[dict[str, Any]]:
    def sort_value(record: dict[str, Any]):
        value = record.get(key)
        if value is None:
            return -math.inf if reverse else math.inf
        return float(value)

    return sorted(records, key=sort_value, reverse=reverse)


def coverage_curve(pair_records: list[dict[str, Any]], cutoffs: list[int]) -> dict[str, float]:
    ordered = sort_records(pair_records, "failure_count", reverse=True)
    total_failures = sum(int(record["failure_count"]) for record in ordered)
    covered = 0
    cursor = 0
    output: dict[str, float] = {}
    for cutoff in sorted(cutoffs):
        while cursor < min(cutoff, len(ordered)):
            covered += int(ordered[cursor]["failure_count"])
            cursor += 1
        output[str(cutoff)] = covered / max(total_failures, 1)
    return output


def summarize_confusion_structure(
    records: list[dict[str, Any]],
    min_pair_count: int,
    topk: int,
    bidirectional_thresholds: list[float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pair_records = aggregate_confusion_pairs(records)
    attach_reverse_pair_stats(pair_records)
    pair_records = sort_records(pair_records, "failure_count", reverse=True)
    supported = [record for record in pair_records if int(record["count"]) >= min_pair_count]
    supported_reverse = [record for record in supported if record.get("reverse_exists") == 1 and record.get("bidirectional_failure_min_share") is not None]
    total_failures = sum(int(record["failure_count"]) for record in pair_records)

    summary = {
        "num_confusion_pairs": len(pair_records),
        "min_pair_count": min_pair_count,
        "coverage_by_top_pairs": coverage_curve(pair_records, [1, 5, 10, 20, 50, 100]),
        "coverage_by_top_supported_pairs": coverage_curve(supported, [1, 5, 10, 20, 50]) if supported else {},
        "reverse_exists_failure_coverage_supported": (
            sum(int(record["failure_count"]) for record in supported if int(record.get("reverse_exists", 0)) == 1) / max(total_failures, 1)
            if supported
            else None
        ),
        "mean_bidirectional_min_share_supported": safe_mean(
            [float(record["bidirectional_failure_min_share"]) for record in supported_reverse]
        ),
        "strong_bidirectional_failure_coverage_supported": {},
        "mean_pair_local_dist_supported": safe_mean(
            [float(record["mean_best_other_trg_norm_dist"]) for record in supported if record.get("mean_best_other_trg_norm_dist") is not None]
        ),
        "top_pairs_by_failure_count": pair_records[:topk],
    }

    for threshold in bidirectional_thresholds:
        key = f">={threshold:g}"
        strong = [
            record
            for record in supported_reverse
            if float(record["bidirectional_failure_min_share"]) >= threshold
        ]
        summary["strong_bidirectional_failure_coverage_supported"][key] = (
            sum(int(record["failure_count"]) for record in strong) / max(total_failures, 1)
            if strong
            else 0.0
        )
    return summary, pair_records


def summarize_rival_buckets(records: list[dict[str, Any]], thresholds: list[float]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        bucket = assign_rival_bucket(extract_float(record, "best_other_trg_norm_dist"), thresholds)
        groups[bucket].append(record)

    output = {}
    for bucket, subset in groups.items():
        output[bucket] = {
            "count": len(subset),
            "failure_rate": safe_rate([record_failure(record) for record in subset]),
            "mean_cross_margin_r0": safe_mean([float(record["cross_margin_r0"]) for record in subset if record.get("cross_margin_r0") is not None]),
            "mean_best_other_trg_norm_dist": safe_mean(
                [float(record["best_other_trg_norm_dist"]) for record in subset if record.get("best_other_trg_norm_dist") is not None]
            ),
        }
    return output


def monotonic_non_decreasing(values: list[float | None]) -> bool | None:
    clean = [value for value in values if value is not None]
    if len(clean) < 2:
        return None
    return all(clean[idx] <= clean[idx + 1] + 1e-12 for idx in range(len(clean) - 1))


def monotonic_non_increasing(values: list[float | None]) -> bool | None:
    clean = [value for value in values if value is not None]
    if len(clean) < 2:
        return None
    return all(clean[idx] >= clean[idx + 1] - 1e-12 for idx in range(len(clean) - 1))


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    records = load_csv(args.records_csv)
    require_columns(
        records,
        [
            "category",
            "pair_name",
            "src_imname",
            "trg_imname",
            "kp_idx",
            "best_other_idx_center",
            "correct",
            args.scale_field,
            "center_margin",
            "cross_margin_r0",
            "src_rival_sim_r0",
            "trg_rival_sim_r0",
            "trg_minus_src_rival_sim_r0",
            "target_more_collapsed_r0",
            "best_other_trg_norm_dist",
        ],
    )

    scale_values = sorted_group_values({record[args.scale_field] for record in records if record.get(args.scale_field) is not None})
    if not scale_values:
        raise RuntimeError(f"No valid values found for scale_field={args.scale_field}")

    overall_summary = summarize_records(records, args.local_thresholds)
    overall_split = diff_success_failure(records, args.local_thresholds)
    overall_confusion, overall_pair_records = summarize_confusion_structure(
        records,
        args.min_pair_count,
        args.topk,
        args.bidirectional_thresholds,
    )

    scale_group_summary = {}
    pair_output_records: list[dict[str, Any]] = []
    for scale_value in scale_values:
        subset = [record for record in records if normalized_scalar(record.get(args.scale_field)) == scale_value]
        scale_summary = summarize_records(subset, args.local_thresholds)
        split_summary = diff_success_failure(subset, args.local_thresholds)
        confusion_summary, pair_records = summarize_confusion_structure(
            subset,
            args.min_pair_count,
            args.topk,
            args.bidirectional_thresholds,
        )
        rival_bucket_summary = summarize_rival_buckets(subset, args.local_thresholds)

        for pair_record in pair_records:
            out_record = dict(pair_record)
            out_record[args.scale_field] = scale_value
            pair_output_records.append(out_record)

        scale_group_summary[str(scale_value)] = {
            "overall": scale_summary,
            "success_failure_split": split_summary,
            "confusion_structure": confusion_summary,
            "rival_buckets": rival_bucket_summary,
        }

    failure_rate_series = [scale_group_summary[str(scale)]["overall"].get("failure_rate") for scale in scale_values]
    cross_margin_series = [scale_group_summary[str(scale)]["overall"].get("mean_cross_margin_r0") for scale in scale_values]
    local_rival_series = [
        scale_group_summary[str(scale)]["overall"].get("rival_bucket_rates", {}).get("<=x1")
        for scale in scale_values
    ]
    reverse_coverage_series = [
        scale_group_summary[str(scale)]["confusion_structure"].get("reverse_exists_failure_coverage_supported")
        for scale in scale_values
    ]
    bidirectional_series = [
        scale_group_summary[str(scale)]["confusion_structure"].get("mean_bidirectional_min_share_supported")
        for scale in scale_values
    ]

    summary = {
        "num_records": len(records),
        "scale_field": args.scale_field,
        "scale_values": scale_values,
        "overall": overall_summary,
        "overall_success_failure_split": overall_split,
        "overall_confusion_structure": overall_confusion,
        "by_scale": scale_group_summary,
        "binding_tests": {
            "failure_rate_monotonic_non_decreasing": monotonic_non_decreasing(failure_rate_series),
            "cross_margin_r0_monotonic_non_increasing": monotonic_non_increasing(cross_margin_series),
            "local_rival_rate_x1_monotonic_non_decreasing": monotonic_non_decreasing(local_rival_series),
            "reverse_exists_failure_coverage_supported_monotonic_non_decreasing": monotonic_non_decreasing(reverse_coverage_series),
            "mean_bidirectional_min_share_supported_monotonic_non_decreasing": monotonic_non_decreasing(bidirectional_series),
            "series": {
                "failure_rate": dict(zip([str(value) for value in scale_values], failure_rate_series)),
                "mean_cross_margin_r0": dict(zip([str(value) for value in scale_values], cross_margin_series)),
                "local_rival_rate_x1": dict(zip([str(value) for value in scale_values], local_rival_series)),
                "reverse_exists_failure_coverage_supported": dict(zip([str(value) for value in scale_values], reverse_coverage_series)),
                "mean_bidirectional_min_share_supported": dict(zip([str(value) for value in scale_values], bidirectional_series)),
            },
        },
        "notes": {
            "interpretation": (
                "If higher scale variation simultaneously raises failure rate, lowers cross-view GT-vs-rival margin, "
                "and preserves or strengthens local / bidirectional real-part confusion structure, "
                "then scale is not just a generic difficulty factor but a mechanism-linked amplifier of weak identity anchoring."
            )
        },
    }

    pair_csv = os.path.join(args.output_dir, "scale_mechanism_binding_pair_records.csv")
    summary_json = os.path.join(args.output_dir, "scale_mechanism_binding_summary.json")
    write_records_csv(pair_output_records, pair_csv)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved pair records to: {pair_csv}")
    print(f"Saved summary to: {summary_json}")
    print(f"Scale field: {args.scale_field}")
    print("Binding series:")
    print("  failure_rate:", summary["binding_tests"]["series"]["failure_rate"])
    print("  mean_cross_margin_r0:", summary["binding_tests"]["series"]["mean_cross_margin_r0"])
    print("  reverse_exists_failure_coverage_supported:", summary["binding_tests"]["series"]["reverse_exists_failure_coverage_supported"])
    print("  mean_bidirectional_min_share_supported:", summary["binding_tests"]["series"]["mean_bidirectional_min_share_supported"])


if __name__ == "__main__":
    main()

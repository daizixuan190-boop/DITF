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
            "Aggregate recurring GT->rival keypoint confusion patterns from SPair identity/competition records. "
            "This is a CPU-only post-hoc analysis over existing CSV records."
        )
    )
    parser.add_argument("--records_csv", type=str, required=True, help="Path to identity_side_diagnostics_records.csv or compatible records CSV.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs.")
    parser.add_argument("--min_pair_count", type=int, default=10, help="Minimum support for a confusion pair to enter ranked summaries.")
    parser.add_argument("--topk", type=int, default=20, help="Number of top patterns to keep in the JSON summary.")
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


def safe_rate(values: list[int]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def require_columns(records: list[dict[str, Any]], columns: list[str]):
    if not records:
        raise RuntimeError("records_csv is empty.")
    missing = [column for column in columns if column not in records[0]]
    if missing:
        raise RuntimeError(f"records_csv is missing required columns: {missing}")


def record_failure(record: dict[str, Any]) -> int:
    return 1 - int(record["correct"])


def safe_float(record: dict[str, Any], key: str) -> float | None:
    value = record.get(key)
    if value is None:
        return None
    return float(value)


def sort_dict_records(records: list[dict[str, Any]], key: str, reverse: bool = True) -> list[dict[str, Any]]:
    def sort_value(item: dict[str, Any]):
        value = item.get(key)
        if value is None:
            return -math.inf if reverse else math.inf
        return float(value)

    return sorted(records, key=sort_value, reverse=reverse)


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
        successes = [record for record in subset if record_failure(record) == 0]
        failure_count = len(failures)
        success_count = len(successes)
        pair_record = {
            "category": key[0],
            "kp_idx": int(key[1]),
            "best_other_idx_center": int(key[2]),
            "count": len(subset),
            "failure_count": failure_count,
            "success_count": success_count,
            "failure_rate": failure_count / max(len(subset), 1),
            "failure_rate_gap_vs_global": failure_count / max(len(subset), 1) - global_failure_rate,
            "failure_share": failure_count / max(total_failures, 1),
            "mean_center_margin": safe_mean([float(record["center_margin"]) for record in subset if record.get("center_margin") is not None]),
            "mean_cross_margin_r0": safe_mean([float(record["cross_margin_r0"]) for record in subset if record.get("cross_margin_r0") is not None]),
            "mean_best_other_trg_norm_dist": safe_mean(
                [float(record["best_other_trg_norm_dist"]) for record in subset if record.get("best_other_trg_norm_dist") is not None]
            ),
            "mean_best_other_src_norm_dist": safe_mean(
                [float(record["best_other_src_norm_dist"]) for record in subset if record.get("best_other_src_norm_dist") is not None]
            ),
            "mean_gt_rank_center": safe_mean(
                [float(record["gt_rank_among_real_parts_center"]) for record in subset if record.get("gt_rank_among_real_parts_center") is not None]
            ),
            "num_pairs": len({tuple(record.get(field) for field in PAIR_KEY_FIELDS) for record in subset}),
            "local_group_mode": modal_value([record.get("local_group") for record in subset if record.get("local_group") not in (None, "")]),
        }
        aggregated.append(pair_record)
    return aggregated


def modal_value(values: list[Any]) -> Any:
    if not values:
        return None
    counts: dict[str, int] = defaultdict(int)
    original: dict[str, Any] = {}
    for value in values:
        key = str(value)
        counts[key] += 1
        original[key] = value
    best_key = max(counts.items(), key=lambda item: item[1])[0]
    return original[best_key]


def attach_reverse_pair_stats(pair_records: list[dict[str, Any]]):
    lookup = {
        (str(record["category"]), int(record["kp_idx"]), int(record["best_other_idx_center"])): record
        for record in pair_records
    }
    for record in pair_records:
        reverse = lookup.get((str(record["category"]), int(record["best_other_idx_center"]), int(record["kp_idx"])))
        reverse_failure_count = None if reverse is None else int(reverse["failure_count"])
        reverse_count = None if reverse is None else int(reverse["count"])
        record["reverse_exists"] = int(reverse is not None)
        record["reverse_failure_count"] = reverse_failure_count
        record["reverse_count"] = reverse_count
        if reverse_failure_count is None:
            record["bidirectional_failure_min_share"] = None
        else:
            denom = max(int(record["failure_count"]), reverse_failure_count, 1)
            record["bidirectional_failure_min_share"] = min(int(record["failure_count"]), reverse_failure_count) / denom


def aggregate_part_roles(pair_records: list[dict[str, Any]], role: str) -> list[dict[str, Any]]:
    if role not in {"source", "rival"}:
        raise ValueError(f"Unsupported role: {role}")

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in pair_records:
        kp = int(record["kp_idx"]) if role == "source" else int(record["best_other_idx_center"])
        grouped[(str(record["category"]), kp)].append(record)

    output: list[dict[str, Any]] = []
    for (category, kp), subset in grouped.items():
        counts = np.asarray([int(record["failure_count"]) for record in subset], dtype=np.int64)
        total_failures = int(counts.sum())
        dominant = max(subset, key=lambda record: int(record["failure_count"]))
        output.append(
            {
                "category": category,
                "kp_idx": kp,
                "num_confusion_pairs": len(subset),
                "failure_count_sum": total_failures,
                "count_sum": int(sum(int(record["count"]) for record in subset)),
                "mean_pair_failure_rate": safe_mean([float(record["failure_rate"]) for record in subset]),
                "mean_pair_cross_margin_r0": safe_mean(
                    [float(record["mean_cross_margin_r0"]) for record in subset if record.get("mean_cross_margin_r0") is not None]
                ),
                "dominant_partner_idx": (
                    int(dominant["best_other_idx_center"]) if role == "source" else int(dominant["kp_idx"])
                ),
                "dominant_partner_failure_count": int(dominant["failure_count"]),
                "dominant_partner_failure_share": int(dominant["failure_count"]) / max(total_failures, 1),
            }
        )
    return output


def coverage_curve(pair_records: list[dict[str, Any]], cutoffs: list[int]) -> dict[str, float]:
    ordered = sort_dict_records(pair_records, "failure_count", reverse=True)
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


def top_pairs_by_category(pair_records: list[dict[str, Any]], topk: int) -> dict[str, list[dict[str, Any]]]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in pair_records:
        by_category[str(record["category"])].append(record)
    output = {}
    for category, subset in by_category.items():
        output[category] = sort_dict_records(subset, "failure_count", reverse=True)[:topk]
    return output


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
        ],
    )

    pair_records = aggregate_confusion_pairs(records)
    attach_reverse_pair_stats(pair_records)
    pair_records = sort_dict_records(pair_records, "failure_count", reverse=True)

    source_records = sort_dict_records(aggregate_part_roles(pair_records, "source"), "failure_count_sum", reverse=True)
    rival_records = sort_dict_records(aggregate_part_roles(pair_records, "rival"), "failure_count_sum", reverse=True)

    pair_csv = os.path.join(args.output_dir, "confusion_pair_patterns.csv")
    source_csv = os.path.join(args.output_dir, "source_part_vulnerability.csv")
    rival_csv = os.path.join(args.output_dir, "rival_part_attractors.csv")
    summary_json = os.path.join(args.output_dir, "confusion_pair_patterns_summary.json")

    write_records_csv(pair_records, pair_csv)
    write_records_csv(source_records, source_csv)
    write_records_csv(rival_records, rival_csv)

    supported = [record for record in pair_records if int(record["count"]) >= args.min_pair_count]
    summary = {
        "num_records": len(records),
        "num_confusion_pairs": len(pair_records),
        "global_failure_rate": safe_rate([record_failure(record) for record in records]),
        "min_pair_count": args.min_pair_count,
        "coverage_by_top_pairs": coverage_curve(pair_records, [1, 5, 10, 20, 50, 100]),
        "coverage_by_top_supported_pairs": coverage_curve(supported, [1, 5, 10, 20, 50]) if supported else {},
        "top_pairs_by_failure_count": pair_records[: args.topk],
        "top_pairs_by_failure_rate_gap": sort_dict_records(supported, "failure_rate_gap_vs_global", reverse=True)[: args.topk],
        "worst_pairs_by_cross_margin_r0": sort_dict_records(
            [record for record in supported if record.get("mean_cross_margin_r0") is not None],
            "mean_cross_margin_r0",
            reverse=False,
        )[: args.topk],
        "top_source_parts": source_records[: args.topk],
        "top_rival_parts": rival_records[: args.topk],
        "top_pairs_by_category": top_pairs_by_category(supported, min(args.topk, 10)),
        "notes": {
            "interpretation": (
                "If a small number of GT->rival keypoint pairs cover a large fraction of failures, "
                "the residual bottleneck is concentrated in specific part-identity competitions rather than diffuse noise. "
                "If many high-failure pairs are bidirectional, the confusion is more symmetry/repetition-like; "
                "if they are strongly one-directional, some source parts are intrinsically less anchorable."
            )
        },
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved pair records to: {pair_csv}")
    print(f"Saved source-part summary to: {source_csv}")
    print(f"Saved rival-part summary to: {rival_csv}")
    print(f"Saved summary to: {summary_json}")
    print(f"Num confusion pairs: {len(pair_records)}")
    print("Failure coverage by top pairs:", summary["coverage_by_top_pairs"])
    if pair_records:
        top = pair_records[0]
        print(
            "Top pair:",
            {
                "category": top["category"],
                "kp_idx": top["kp_idx"],
                "best_other_idx_center": top["best_other_idx_center"],
                "failure_count": top["failure_count"],
                "failure_rate": top["failure_rate"],
                "mean_cross_margin_r0": top.get("mean_cross_margin_r0"),
            },
        )


if __name__ == "__main__":
    main()

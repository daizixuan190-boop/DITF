import argparse
import csv
import json
import math
import os
from collections import defaultdict
from typing import Any

import numpy as np

PAIR_KEY_FIELDS = ["category", "pair_name", "src_imname", "trg_imname"]
BASE_EXCLUDE_FIELDS = {
    "category",
    "pair_name",
    "src_imname",
    "trg_imname",
    "filename",
    "kp_idx",
    "correct",
    "pred_x",
    "pred_y",
    "threshold",
    "num_points",
}
PREFIX_EXCLUDES = (
    "pair_",
    "gt_",
    "best_other_",
    "margin_",
    "cross_",
    "src_",
    "trg_",
    "pred_",
    "rank_",
    "local_",
    "post_",
    "ln_",
)
SUFFIX_EXCLUDES = ("_score", "_idx", "_rate", "_gain", "_dist", "_dx", "_dy")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Discover which pair-level annotation factors are associated with residual failure "
            "and cross-view identity anchoring weakness. "
            "This is a CPU-only post-hoc analysis over existing records CSV."
        )
    )
    parser.add_argument("--records_csv", type=str, required=True, help="Path to identity_side_diagnostics_records.csv or compatible records CSV.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs.")
    parser.add_argument("--field_allowlist", nargs="*", default=[], help="Optional explicit pair-level fields to analyze.")
    parser.add_argument("--max_exact_groups", type=int, default=8, help="Max unique values for a field to be treated as exact categorical.")
    parser.add_argument("--quantiles", nargs="+", type=float, default=[0.25, 0.5, 0.75], help="Quantiles used for numeric binning.")
    parser.add_argument("--min_group_pairs", type=int, default=20, help="Minimum number of pairs per group when ranking fields.")
    parser.add_argument("--topk", type=int, default=20, help="Number of top fields to keep in ranked summaries.")
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


def pair_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(record.get(field) for field in PAIR_KEY_FIELDS)


def is_scalar_candidate(value: Any) -> bool:
    return isinstance(value, (int, float, bool, str))


def should_exclude_field(field: str) -> bool:
    if field in BASE_EXCLUDE_FIELDS:
        return True
    if any(field.startswith(prefix) for prefix in PREFIX_EXCLUDES):
        return True
    if any(field.endswith(suffix) for suffix in SUFFIX_EXCLUDES):
        return True
    return False


def record_failure(record: dict[str, Any]) -> int:
    return 1 - int(record["correct"])


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def safe_pair_metric(pair_record: dict[str, Any], key: str) -> float | None:
    value = pair_record.get(key)
    if value is None:
        return None
    return float(value)


def build_pair_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[pair_key(record)].append(record)

    pair_records: list[dict[str, Any]] = []
    for key, subset in grouped.items():
        pair_record: dict[str, Any] = {field: key[idx] for idx, field in enumerate(PAIR_KEY_FIELDS)}
        pair_record["num_points"] = len(subset)
        pair_record["pair_failure_rate"] = safe_rate([record_failure(record) for record in subset])
        pair_record["pair_mean_center_margin"] = safe_mean([float(record["center_margin"]) for record in subset if record.get("center_margin") is not None])
        pair_record["pair_mean_cross_margin_r0"] = safe_mean([float(record["cross_margin_r0"]) for record in subset if record.get("cross_margin_r0") is not None])
        pair_record["pair_mean_best_other_trg_norm_dist"] = safe_mean(
            [float(record["best_other_trg_norm_dist"]) for record in subset if record.get("best_other_trg_norm_dist") is not None]
        )

        for field in subset[0].keys():
            if field in pair_record or not is_scalar_candidate(subset[0].get(field)):
                continue
            values = [record.get(field) for record in subset if record.get(field) not in (None, "")]
            unique_values = {normalize_scalar(value) for value in values}
            pair_record[field] = values[0] if len(unique_values) == 1 and values else None
            pair_record[f"__field_consistent__{field}"] = int(len(unique_values) <= 1)
        pair_records.append(pair_record)
    return pair_records


def normalize_scalar(value: Any) -> Any:
    if isinstance(value, float) and abs(value - round(value)) < 1e-12:
        return int(round(value))
    return value


def detect_pair_level_fields(pair_records: list[dict[str, Any]], field_allowlist: list[str]) -> list[str]:
    if field_allowlist:
        return [field for field in field_allowlist if any(field in pair_record for pair_record in pair_records)]

    discovered: list[str] = []
    fields = sorted({field for record in pair_records for field in record.keys() if not field.startswith("__field_consistent__")})
    for field in fields:
        if should_exclude_field(field):
            continue
        if field in PAIR_KEY_FIELDS:
            continue
        consistency_key = f"__field_consistent__{field}"
        consistency = [int(record.get(consistency_key, 1)) for record in pair_records if consistency_key in record]
        values = [record.get(field) for record in pair_records if record.get(field) not in (None, "")]
        if not values or not consistency:
            continue
        if safe_rate(consistency) < 0.99:
            continue
        unique_values = {normalize_scalar(value) for value in values}
        if len(unique_values) <= 1:
            continue
        discovered.append(field)
    return discovered


def build_numeric_bins(values: np.ndarray, quantiles: list[float]) -> list[tuple[str, float, float]]:
    thresholds = [float(np.quantile(values, q)) for q in quantiles]
    bins: list[tuple[str, float, float]] = []
    previous = -np.inf
    for q, threshold in zip(quantiles, thresholds):
        bins.append((f"<=q{q:g}", previous, threshold))
        previous = threshold
    bins.append((f">q{quantiles[-1]:g}", previous, np.inf))
    return bins


def assign_numeric_bin(value: float, bins: list[tuple[str, float, float]]) -> str:
    for label, lower, upper in bins:
        if value <= upper and value > lower:
            return label
    return bins[-1][0]


def group_pair_records(pair_records: list[dict[str, Any]], field: str, args) -> tuple[str, dict[str, list[dict[str, Any]]], dict[str, Any]]:
    values = [record.get(field) for record in pair_records if record.get(field) not in (None, "")]
    normalized_values = [normalize_scalar(value) for value in values]
    if not normalized_values:
        return "empty", {}, {}

    numeric = all(isinstance(value, (int, float, bool)) for value in normalized_values)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    metadata: dict[str, Any] = {}

    if numeric:
        numeric_values = np.asarray([float(value) for value in normalized_values], dtype=np.float64)
        unique_count = len(set(float(value) for value in numeric_values.tolist()))
        if unique_count <= args.max_exact_groups:
            field_type = "discrete_numeric"
            for record in pair_records:
                value = record.get(field)
                if value in (None, ""):
                    continue
                groups[str(normalize_scalar(value))].append(record)
        else:
            field_type = "binned_numeric"
            bins = build_numeric_bins(numeric_values, args.quantiles)
            metadata["bins"] = [{"label": label, "lower": lower, "upper": upper} for label, lower, upper in bins]
            for record in pair_records:
                value = record.get(field)
                if value in (None, ""):
                    continue
                groups[assign_numeric_bin(float(value), bins)].append(record)
    else:
        field_type = "categorical"
        unique_count = len(set(str(value) for value in normalized_values))
        if unique_count > args.max_exact_groups:
            top_values = [value for value, _ in top_counts([str(value) for value in normalized_values], args.max_exact_groups - 1)]
            keep = set(top_values)
            for record in pair_records:
                value = record.get(field)
                if value in (None, ""):
                    continue
                label = str(value)
                groups[label if label in keep else "__OTHER__"].append(record)
        else:
            for record in pair_records:
                value = record.get(field)
                if value in (None, ""):
                    continue
                groups[str(value)].append(record)
    return field_type, groups, metadata


def top_counts(values: list[str], topk: int) -> list[tuple[str, int]]:
    counts: dict[str, int] = defaultdict(int)
    for value in values:
        counts[value] += 1
    return sorted(counts.items(), key=lambda item: item[1], reverse=True)[:topk]


def summarize_group(pair_records: list[dict[str, Any]]) -> dict[str, Any]:
    if not pair_records:
        return {"pair_count": 0, "record_count": 0}
    return {
        "pair_count": len(pair_records),
        "record_count": int(sum(int(record["num_points"]) for record in pair_records)),
        "pair_mean_failure_rate": safe_mean([float(record["pair_failure_rate"]) for record in pair_records if record.get("pair_failure_rate") is not None]),
        "pair_mean_cross_margin_r0": safe_mean(
            [float(record["pair_mean_cross_margin_r0"]) for record in pair_records if record.get("pair_mean_cross_margin_r0") is not None]
        ),
        "pair_mean_center_margin": safe_mean(
            [float(record["pair_mean_center_margin"]) for record in pair_records if record.get("pair_mean_center_margin") is not None]
        ),
        "pair_mean_best_other_trg_norm_dist": safe_mean(
            [float(record["pair_mean_best_other_trg_norm_dist"]) for record in pair_records if record.get("pair_mean_best_other_trg_norm_dist") is not None]
        ),
        "record_weighted_failure_rate": record_weighted_mean(pair_records, "pair_failure_rate"),
        "record_weighted_cross_margin_r0": record_weighted_mean(pair_records, "pair_mean_cross_margin_r0"),
    }


def record_weighted_mean(pair_records: list[dict[str, Any]], key: str) -> float | None:
    numerator = 0.0
    denominator = 0.0
    for record in pair_records:
        value = safe_pair_metric(record, key)
        if value is None:
            continue
        weight = float(record["num_points"])
        numerator += weight * value
        denominator += weight
    if denominator <= 0:
        return None
    return numerator / denominator


def field_effect_summary(group_summaries: dict[str, dict[str, Any]], min_group_pairs: int) -> dict[str, Any]:
    eligible = [summary for summary in group_summaries.values() if int(summary["pair_count"]) >= min_group_pairs]
    if len(eligible) < 2:
        return {
            "num_eligible_groups": len(eligible),
            "failure_rate_range": None,
            "cross_margin_range": None,
            "hardest_group": None,
            "easiest_group": None,
            "worst_margin_group": None,
            "best_margin_group": None,
        }

    def failure_rate(summary: dict[str, Any]) -> float:
        value = summary.get("pair_mean_failure_rate")
        return -1.0 if value is None else float(value)

    def cross_margin(summary: dict[str, Any]) -> float:
        value = summary.get("pair_mean_cross_margin_r0")
        return -np.inf if value is None else float(value)

    hardest = max(eligible, key=failure_rate)
    easiest = min(eligible, key=failure_rate)
    worst_margin = min(eligible, key=cross_margin)
    best_margin = max(eligible, key=cross_margin)
    return {
        "num_eligible_groups": len(eligible),
        "failure_rate_range": float(hardest["pair_mean_failure_rate"] - easiest["pair_mean_failure_rate"]),
        "cross_margin_range": float(best_margin["pair_mean_cross_margin_r0"] - worst_margin["pair_mean_cross_margin_r0"]),
        "hardest_group": hardest["group_name"],
        "easiest_group": easiest["group_name"],
        "worst_margin_group": worst_margin["group_name"],
        "best_margin_group": best_margin["group_name"],
    }


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    records = load_csv(args.records_csv)
    if not records:
        raise RuntimeError("records_csv is empty.")
    required = [field for field in PAIR_KEY_FIELDS if field not in records[0]]
    if required:
        raise RuntimeError(f"records_csv is missing required pair key fields: {required}")

    pair_records = build_pair_records(records)
    discovered_fields = detect_pair_level_fields(pair_records, args.field_allowlist)
    if not discovered_fields:
        raise RuntimeError("No pair-level scalar fields were discovered. Check the input CSV or pass --field_allowlist.")

    field_rows: list[dict[str, Any]] = []
    summary_fields: dict[str, Any] = {}
    for field in discovered_fields:
        field_type, groups, metadata = group_pair_records(pair_records, field, args)
        group_summaries = {}
        for group_name, subset in groups.items():
            stats = summarize_group(subset)
            stats["group_name"] = group_name
            group_summaries[group_name] = stats

        effect = field_effect_summary(group_summaries, args.min_group_pairs)
        summary_fields[field] = {
            "field_type": field_type,
            "metadata": metadata,
            "effect": effect,
            "groups": group_summaries,
        }
        field_rows.append(
            {
                "field": field,
                "field_type": field_type,
                "num_groups": len(group_summaries),
                "num_eligible_groups": effect["num_eligible_groups"],
                "failure_rate_range": effect["failure_rate_range"],
                "cross_margin_range": effect["cross_margin_range"],
                "hardest_group": effect["hardest_group"],
                "worst_margin_group": effect["worst_margin_group"],
            }
        )

    field_rows_sorted_failure = sorted(
        field_rows,
        key=lambda record: -np.inf if record["failure_rate_range"] is None else float(record["failure_rate_range"]),
        reverse=True,
    )
    field_rows_sorted_margin = sorted(
        field_rows,
        key=lambda record: -np.inf if record["cross_margin_range"] is None else float(record["cross_margin_range"]),
        reverse=True,
    )

    csv_path = os.path.join(args.output_dir, "pair_factor_association_ranking.csv")
    json_path = os.path.join(args.output_dir, "pair_factor_association_summary.json")
    write_records_csv(field_rows, csv_path)

    summary = {
        "num_records": len(records),
        "num_pairs": len(pair_records),
        "discovered_fields": discovered_fields,
        "top_fields_by_failure_rate_range": field_rows_sorted_failure[: args.topk],
        "top_fields_by_cross_margin_range": field_rows_sorted_margin[: args.topk],
        "fields": summary_fields,
        "notes": {
            "interpretation": (
                "This analysis only trusts fields that are pair-level consistent within the CSV. "
                "If a field shows a large failure-rate range and cross-margin range across groups, "
                "it is a credible candidate driver of cross-view identity anchoring weakness."
            )
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved ranking CSV to: {csv_path}")
    print(f"Saved summary JSON to: {json_path}")
    print(f"Discovered pair-level fields: {discovered_fields}")
    print("Top fields by failure-rate range:", summary["top_fields_by_failure_rate_range"][: min(args.topk, 5)])
    print("Top fields by cross-margin range:", summary["top_fields_by_cross_margin_range"][: min(args.topk, 5)])


if __name__ == "__main__":
    main()

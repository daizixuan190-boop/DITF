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
            "Compare the dominant anchor-erosion cohorts, especially source-dominant versus "
            "cross-view-only erosion, to test whether they are distinct failure populations. "
            "Consumes anchor_erosion_side_transitions.csv and optionally identity_side_diagnostics_records.csv. "
            "CPU-only post-hoc analysis."
        )
    )
    parser.add_argument("--transitions_csv", type=str, required=True, help="Path to anchor_erosion_side_transitions.csv.")
    parser.add_argument(
        "--records_csv",
        type=str,
        default="",
        help="Optional path to identity_side_diagnostics_records.csv for factor enrichment.",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs.")
    parser.add_argument("--scale_field", type=str, default="scale_variation", help="Scale field used in records_csv.")
    parser.add_argument(
        "--cohort_keys",
        nargs="+",
        default=[
            "source_dominant_erosion",
            "cross_view_only_erosion",
            "target_dominant_erosion",
            "bilateral_erosion",
        ],
        help="Mechanism cohort columns from transitions_csv to compare.",
    )
    parser.add_argument(
        "--factor_fields",
        nargs="+",
        default=["occlusion", "truncation", "viewpoint_variation"],
        help="Optional numeric factor fields to enrich from records_csv if present.",
    )
    parser.add_argument("--topk", type=int, default=20, help="Number of top categories/confusion keys to keep.")
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
        raise RuntimeError("input CSV is empty.")
    missing = [column for column in columns if column not in records[0]]
    if missing:
        raise RuntimeError(f"CSV is missing required columns: {missing}")


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


def weighted_mean(records: list[dict[str, Any]], key: str) -> float | None:
    numerator = 0.0
    denominator = 0.0
    for record in records:
        value = record.get(key)
        if value is None:
            continue
        weight = support_weight(record)
        numerator += float(value) * float(weight)
        denominator += float(weight)
    if denominator <= 0.0:
        return None
    return numerator / denominator


def weighted_rate(records: list[dict[str, Any]], key: str) -> float | None:
    numerator = 0.0
    denominator = 0.0
    for record in records:
        value = record.get(key)
        if value is None:
            continue
        weight = support_weight(record)
        numerator += float(value) * float(weight)
        denominator += float(weight)
    if denominator <= 0.0:
        return None
    return numerator / denominator


def normalized_scalar(value: Any) -> Any:
    if isinstance(value, float) and abs(value - round(value)) < 1e-12:
        return int(round(value))
    return value


def subtract_or_none(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    return float(a) - float(b)


def transition_key(record: dict[str, Any]) -> str:
    return f"{record['scale_a']}->{record['scale_b']}"


def confusion_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(record[field] for field in CONFUSION_KEY_FIELDS)


def build_factor_lookup(records: list[dict[str, Any]], scale_field: str, factor_fields: list[str]) -> tuple[dict[tuple[Any, ...], dict[str, Any]], list[str]]:
    available_factor_fields = [field for field in factor_fields if field in records[0]]
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("best_other_idx_center") is None or record.get(scale_field) is None:
            continue
        if int(record["best_other_idx_center"]) == int(record["kp_idx"]):
            continue
        key = tuple(record[field] for field in CONFUSION_KEY_FIELDS) + (normalized_scalar(record[scale_field]),)
        grouped[key].append(record)

    lookup: dict[tuple[Any, ...], dict[str, Any]] = {}
    for key, subset in grouped.items():
        item: dict[str, Any] = {}
        for field in available_factor_fields:
            values = [float(record[field]) for record in subset if record.get(field) is not None]
            item[f"mean_{field}"] = safe_mean(values)
        lookup[key] = item
    return lookup, available_factor_fields


def enrich_with_factors(transitions: list[dict[str, Any]], factor_lookup: dict[tuple[Any, ...], dict[str, Any]], factor_fields: list[str]):
    for record in transitions:
        key_a = tuple(record[field] for field in CONFUSION_KEY_FIELDS) + (normalized_scalar(record["scale_a"]),)
        key_b = tuple(record[field] for field in CONFUSION_KEY_FIELDS) + (normalized_scalar(record["scale_b"]),)
        stats_a = factor_lookup.get(key_a, {})
        stats_b = factor_lookup.get(key_b, {})
        for field in factor_fields:
            mean_a = stats_a.get(f"mean_{field}")
            mean_b = stats_b.get(f"mean_{field}")
            record[f"mean_{field}_a"] = mean_a
            record[f"mean_{field}_b"] = mean_b
            record[f"delta_{field}"] = subtract_or_none(mean_b, mean_a)


def summarize_distribution(records: list[dict[str, Any]], key: str) -> dict[str, float]:
    grouped: dict[str, int] = defaultdict(int)
    for record in records:
        value = record.get(key)
        grouped[str(value)] += 1
    total = float(sum(grouped.values()))
    if total <= 0.0:
        return {}
    return {name: count / total for name, count in sorted(grouped.items())}


def summarize_distribution_weighted(records: list[dict[str, Any]], key: str) -> dict[str, float]:
    grouped: dict[str, float] = defaultdict(float)
    total = 0.0
    for record in records:
        value = record.get(key)
        weight = float(support_weight(record))
        grouped[str(value)] += weight
        total += weight
    if total <= 0.0:
        return {}
    return {name: value / total for name, value in sorted(grouped.items())}


def top_categories(records: list[dict[str, Any]], topk: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["category"])].append(record)
    output = []
    total_weight = float(sum(support_weight(record) for record in records))
    for category, subset in grouped.items():
        weight = float(sum(support_weight(record) for record in subset))
        output.append(
            {
                "category": category,
                "count": len(subset),
                "weighted_support_sum": int(weight),
                "weighted_share": None if total_weight <= 0.0 else weight / total_weight,
                "weighted_mean_delta_cross_margin": weighted_mean(subset, "delta_cross_margin"),
                "weighted_mean_delta_failure_rate": weighted_mean(subset, "delta_failure_rate"),
            }
        )
    return sorted(output, key=lambda item: (-float(item["weighted_support_sum"]), item["category"]))[:topk]


def top_confusion_keys(records: list[dict[str, Any]], topk: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[confusion_key(record)].append(record)
    output = []
    total_weight = float(sum(support_weight(record) for record in records))
    for key, subset in grouped.items():
        weight = float(sum(support_weight(record) for record in subset))
        output.append(
            {
                "category": str(key[0]),
                "kp_idx": int(key[1]),
                "best_other_idx_center": int(key[2]),
                "count": len(subset),
                "weighted_support_sum": int(weight),
                "weighted_share": None if total_weight <= 0.0 else weight / total_weight,
                "weighted_mean_delta_cross_margin": weighted_mean(subset, "delta_cross_margin"),
                "transition_distribution": summarize_distribution(subset, "__transition_key__"),
            }
        )
    return sorted(output, key=lambda item: (-float(item["weighted_support_sum"]), item["category"], item["kp_idx"]))[:topk]


def key_overlap_summary(records_a: list[dict[str, Any]], records_b: list[dict[str, Any]]) -> dict[str, Any]:
    keys_a = {confusion_key(record) for record in records_a}
    keys_b = {confusion_key(record) for record in records_b}
    inter = keys_a & keys_b
    union = keys_a | keys_b

    weight_map_a: dict[tuple[Any, ...], float] = defaultdict(float)
    for record in records_a:
        weight_map_a[confusion_key(record)] += float(support_weight(record))
    weight_map_b: dict[tuple[Any, ...], float] = defaultdict(float)
    for record in records_b:
        weight_map_b[confusion_key(record)] += float(support_weight(record))

    all_keys = union
    intersection_weight = float(sum(min(weight_map_a.get(key, 0.0), weight_map_b.get(key, 0.0)) for key in all_keys))
    union_weight = float(sum(max(weight_map_a.get(key, 0.0), weight_map_b.get(key, 0.0)) for key in all_keys))

    return {
        "num_keys_a": len(keys_a),
        "num_keys_b": len(keys_b),
        "num_intersection_keys": len(inter),
        "jaccard": None if not union else len(inter) / len(union),
        "weighted_jaccard": None if union_weight <= 0.0 else intersection_weight / union_weight,
    }


def summarize_cohort(records: list[dict[str, Any]], factor_fields: list[str], topk: int) -> dict[str, Any]:
    if not records:
        return {"count": 0}

    summary = {
        "count": len(records),
        "weighted_support_sum": int(sum(support_weight(record) for record in records)),
        "transition_distribution": summarize_distribution_weighted(records, "__transition_key__"),
        "margin_regime_distribution": summarize_distribution_weighted(records, "margin_regime"),
        "weighted_mean_margin_a": weighted_mean(records, "mean_cross_margin_a"),
        "weighted_mean_delta_src_rival_sim": weighted_mean(records, "delta_src_rival_sim"),
        "weighted_mean_delta_trg_rival_sim": weighted_mean(records, "delta_trg_rival_sim"),
        "weighted_mean_delta_trg_minus_src_rival_sim": weighted_mean(records, "delta_trg_minus_src_rival_sim"),
        "weighted_mean_delta_cross_gt_score": weighted_mean(records, "delta_cross_gt_score"),
        "weighted_mean_delta_cross_rival_score": weighted_mean(records, "delta_cross_rival_score"),
        "weighted_mean_delta_cross_margin": weighted_mean(records, "delta_cross_margin"),
        "weighted_mean_delta_failure_rate": weighted_mean(records, "delta_failure_rate"),
        "weighted_mean_delta_target_more_collapsed_rate": weighted_mean(records, "delta_target_more_collapsed_rate"),
        "gt_drop_more_than_rival_rate": weighted_rate(records, "gt_drop_more_than_rival"),
        "top_categories": top_categories(records, topk),
        "top_confusion_keys": top_confusion_keys(records, topk),
        "factor_signatures": {},
    }

    for field in factor_fields:
        summary["factor_signatures"][field] = {
            "weighted_mean_a": weighted_mean(records, f"mean_{field}_a"),
            "weighted_mean_b": weighted_mean(records, f"mean_{field}_b"),
            "weighted_mean_delta": weighted_mean(records, f"delta_{field}"),
        }
    return summary


def compare_cohorts(records_a: list[dict[str, Any]], records_b: list[dict[str, Any]], factor_fields: list[str]) -> dict[str, Any]:
    summary_a = summarize_cohort(records_a, factor_fields, topk=20)
    summary_b = summarize_cohort(records_b, factor_fields, topk=20)

    def diff_from_dict(a: dict[str, Any], b: dict[str, Any], key: str) -> float | None:
        av = a.get(key)
        bv = b.get(key)
        if av is None or bv is None:
            return None
        return float(av - bv)

    out = {
        "cohort_a": summary_a,
        "cohort_b": summary_b,
        "a_minus_b": {
            "weighted_mean_margin_a_gap": diff_from_dict(summary_a, summary_b, "weighted_mean_margin_a"),
            "weighted_mean_delta_src_rival_sim_gap": diff_from_dict(summary_a, summary_b, "weighted_mean_delta_src_rival_sim"),
            "weighted_mean_delta_trg_rival_sim_gap": diff_from_dict(summary_a, summary_b, "weighted_mean_delta_trg_rival_sim"),
            "weighted_mean_delta_trg_minus_src_rival_sim_gap": diff_from_dict(summary_a, summary_b, "weighted_mean_delta_trg_minus_src_rival_sim"),
            "weighted_mean_delta_cross_gt_score_gap": diff_from_dict(summary_a, summary_b, "weighted_mean_delta_cross_gt_score"),
            "weighted_mean_delta_cross_rival_score_gap": diff_from_dict(summary_a, summary_b, "weighted_mean_delta_cross_rival_score"),
            "weighted_mean_delta_cross_margin_gap": diff_from_dict(summary_a, summary_b, "weighted_mean_delta_cross_margin"),
            "weighted_mean_delta_failure_rate_gap": diff_from_dict(summary_a, summary_b, "weighted_mean_delta_failure_rate"),
            "weighted_mean_delta_target_more_collapsed_rate_gap": diff_from_dict(summary_a, summary_b, "weighted_mean_delta_target_more_collapsed_rate"),
            "gt_drop_more_than_rival_rate_gap": diff_from_dict(summary_a, summary_b, "gt_drop_more_than_rival_rate"),
            "factor_gaps": {},
        },
        "confusion_key_overlap": key_overlap_summary(records_a, records_b),
    }
    for field in factor_fields:
        a_factor = summary_a["factor_signatures"].get(field, {})
        b_factor = summary_b["factor_signatures"].get(field, {})
        out["a_minus_b"]["factor_gaps"][field] = {
            "weighted_mean_a_gap": diff_from_dict(a_factor, b_factor, "weighted_mean_a"),
            "weighted_mean_b_gap": diff_from_dict(a_factor, b_factor, "weighted_mean_b"),
            "weighted_mean_delta_gap": diff_from_dict(a_factor, b_factor, "weighted_mean_delta"),
        }
    return out


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    transitions = load_csv(args.transitions_csv)
    required_columns = [
        "category",
        "kp_idx",
        "best_other_idx_center",
        "scale_a",
        "scale_b",
        "count_a",
        "count_b",
        "margin_regime",
        "delta_src_rival_sim",
        "delta_trg_rival_sim",
        "delta_trg_minus_src_rival_sim",
        "delta_cross_gt_score",
        "delta_cross_rival_score",
        "delta_cross_margin",
        "delta_failure_rate",
        "delta_target_more_collapsed_rate",
        "erosion_case",
        "gt_drop_more_than_rival",
    ] + args.cohort_keys
    require_columns(transitions, required_columns)

    transitions = [dict(record, __transition_key__=transition_key(record)) for record in transitions]
    erosion_records = [record for record in transitions if int(record.get("erosion_case", 0)) == 1]

    factor_fields_used: list[str] = []
    if args.records_csv:
        raw_records = load_csv(args.records_csv)
        require_columns(
            raw_records,
            [
                "category",
                "kp_idx",
                "best_other_idx_center",
                args.scale_field,
            ],
        )
        factor_lookup, factor_fields_used = build_factor_lookup(raw_records, args.scale_field, args.factor_fields)
        enrich_with_factors(transitions, factor_lookup, factor_fields_used)
        erosion_records = [record for record in transitions if int(record.get("erosion_case", 0)) == 1]

    annotated_records: list[dict[str, Any]] = []
    cohort_summary: dict[str, Any] = {}
    for cohort_key in args.cohort_keys:
        subset = [record for record in erosion_records if int(record.get(cohort_key, 0)) == 1]
        cohort_summary[cohort_key] = summarize_cohort(subset, factor_fields_used, args.topk)
        for record in subset:
            item = dict(record)
            item["cohort_name"] = cohort_key
            annotated_records.append(item)

    pairwise_comparisons: dict[str, Any] = {}
    for idx_a in range(len(args.cohort_keys)):
        for idx_b in range(idx_a + 1, len(args.cohort_keys)):
            cohort_a = args.cohort_keys[idx_a]
            cohort_b = args.cohort_keys[idx_b]
            subset_a = [record for record in erosion_records if int(record.get(cohort_a, 0)) == 1]
            subset_b = [record for record in erosion_records if int(record.get(cohort_b, 0)) == 1]
            pairwise_comparisons[f"{cohort_a}__vs__{cohort_b}"] = compare_cohorts(subset_a, subset_b, factor_fields_used)

    summary = {
        "num_transitions": len(transitions),
        "num_erosion_records": len(erosion_records),
        "cohort_keys": args.cohort_keys,
        "factor_fields_used": factor_fields_used,
        "cohorts": cohort_summary,
        "pairwise_comparisons": pairwise_comparisons,
        "notes": {
            "interpretation": (
                "If source_dominant_erosion and cross_view_only_erosion show different transition distributions, "
                "category concentrations, confusion-key overlap, or external factor signatures, then they should "
                "be treated as distinct mechanism cohorts rather than one blurred erosion phenomenon."
            )
        },
    }

    records_csv = os.path.join(args.output_dir, "erosion_cohort_signature_records.csv")
    summary_json = os.path.join(args.output_dir, "erosion_cohort_signature_summary.json")
    write_records_csv(annotated_records, records_csv)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    source_vs_cross = pairwise_comparisons.get("source_dominant_erosion__vs__cross_view_only_erosion")
    print(f"Saved records to: {records_csv}")
    print(f"Saved summary to: {summary_json}")
    print(
        "Source-vs-cross_view gap:",
        None if source_vs_cross is None else source_vs_cross["a_minus_b"],
    )


if __name__ == "__main__":
    main()

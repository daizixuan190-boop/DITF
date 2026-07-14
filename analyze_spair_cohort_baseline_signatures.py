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
            "Test whether downstream erosion cohorts are already distinguishable from baseline "
            "single-scale signatures available before erosion. "
            "Consumes anchor_erosion_side_transitions.csv and identity_side_diagnostics_records.csv. "
            "CPU-only post-hoc analysis."
        )
    )
    parser.add_argument("--transitions_csv", type=str, required=True, help="Path to anchor_erosion_side_transitions.csv.")
    parser.add_argument("--records_csv", type=str, required=True, help="Path to identity_side_diagnostics_records.csv.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs.")
    parser.add_argument("--scale_field", type=str, default="scale_variation", help="Scale field in records_csv.")
    parser.add_argument("--context_radius", type=int, default=0, help="Context radius used for baseline identity-side features.")
    parser.add_argument(
        "--cohort_keys",
        nargs="+",
        default=["source_dominant_erosion", "cross_view_only_erosion"],
        help="Erosion cohort columns from transitions_csv to compare.",
    )
    parser.add_argument(
        "--factor_fields",
        nargs="+",
        default=["occlusion", "truncation", "viewpoint_variation"],
        help="Optional scalar factor fields to aggregate from records_csv if present.",
    )
    parser.add_argument("--topk", type=int, default=20, help="Top separative features to keep in summaries.")
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


def weighted_var(records: list[dict[str, Any]], key: str) -> float | None:
    values = []
    weights = []
    for record in records:
        value = record.get(key)
        if value is None:
            continue
        values.append(float(value))
        weights.append(float(support_weight(record)))
    if len(values) < 2:
        return None
    x = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    w_sum = float(np.sum(w))
    if w_sum <= 0.0:
        return None
    mean = float(np.sum(w * x) / w_sum)
    var = float(np.sum(w * (x - mean) ** 2) / w_sum)
    return var


def pooled_std(records_a: list[dict[str, Any]], records_b: list[dict[str, Any]], key: str) -> float | None:
    var_a = weighted_var(records_a, key)
    var_b = weighted_var(records_b, key)
    if var_a is None or var_b is None:
        return None
    pooled = max((var_a + var_b) / 2.0, 0.0)
    return math.sqrt(pooled)


def normalized_scalar(value: Any) -> Any:
    if isinstance(value, float) and abs(value - round(value)) < 1e-12:
        return int(round(value))
    return value


def aggregate_baseline_groups(records: list[dict[str, Any]], scale_field: str, radius: int, factor_fields: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    src_key = f"src_rival_sim_r{radius}"
    trg_key = f"trg_rival_sim_r{radius}"
    cross_gt_key = f"cross_gt_score_r{radius}"
    cross_rival_key = f"cross_rival_score_r{radius}"
    cross_margin_key = f"cross_margin_r{radius}"
    target_more_key = f"target_more_collapsed_r{radius}"

    available_factors = [field for field in factor_fields if field in records[0]]
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("best_other_idx_center") is None or record.get(scale_field) is None:
            continue
        if int(record["best_other_idx_center"]) == int(record["kp_idx"]):
            continue
        key = tuple(record[field] for field in CONFUSION_KEY_FIELDS) + (normalized_scalar(record[scale_field]),)
        grouped[key].append(record)

    output = []
    for key, subset in grouped.items():
        item = {
            "category": key[0],
            "kp_idx": int(key[1]),
            "best_other_idx_center": int(key[2]),
            "scale_value": key[3],
            "count": len(subset),
            "failure_rate": safe_rate([1 - int(record["correct"]) for record in subset]),
            "mean_center_margin": safe_mean([float(record["center_margin"]) for record in subset if record.get("center_margin") is not None]),
            "mean_best_other_trg_norm_dist": safe_mean(
                [float(record["best_other_trg_norm_dist"]) for record in subset if record.get("best_other_trg_norm_dist") is not None]
            ),
            "mean_src_rival_sim": safe_mean([float(record[src_key]) for record in subset if record.get(src_key) is not None]),
            "mean_trg_rival_sim": safe_mean([float(record[trg_key]) for record in subset if record.get(trg_key) is not None]),
            "mean_cross_gt_score": safe_mean([float(record[cross_gt_key]) for record in subset if record.get(cross_gt_key) is not None]),
            "mean_cross_rival_score": safe_mean([float(record[cross_rival_key]) for record in subset if record.get(cross_rival_key) is not None]),
            "mean_cross_margin": safe_mean([float(record[cross_margin_key]) for record in subset if record.get(cross_margin_key) is not None]),
            "target_more_collapsed_rate": safe_rate([int(record[target_more_key]) for record in subset if record.get(target_more_key) is not None]),
        }
        for field in available_factors:
            item[f"mean_{field}"] = safe_mean([float(record[field]) for record in subset if record.get(field) is not None])
        output.append(item)
    return output, available_factors


def attach_baseline_features(
    transitions: list[dict[str, Any]],
    baseline_records: list[dict[str, Any]],
    cohort_keys: list[str],
) -> list[dict[str, Any]]:
    lookup = {
        (record["category"], int(record["kp_idx"]), int(record["best_other_idx_center"]), record["scale_value"]): record
        for record in baseline_records
    }

    annotated: list[dict[str, Any]] = []
    for transition in transitions:
        if int(transition.get("erosion_case", 0)) != 1:
            continue
        cohort_names = [key for key in cohort_keys if int(transition.get(key, 0)) == 1]
        if not cohort_names:
            continue
        lookup_key = (
            transition["category"],
            int(transition["kp_idx"]),
            int(transition["best_other_idx_center"]),
            normalized_scalar(transition["scale_a"]),
        )
        baseline = lookup.get(lookup_key)
        if baseline is None:
            continue
        for cohort_name in cohort_names:
            item = dict(transition)
            item["cohort_name"] = cohort_name
            for key, value in baseline.items():
                if key not in item:
                    item[key] = value
            annotated.append(item)
    return annotated


def summarize_cohort(records: list[dict[str, Any]], feature_keys: list[str]) -> dict[str, Any]:
    if not records:
        return {"count": 0}
    return {
        "count": len(records),
        "weighted_support_sum": int(sum(support_weight(record) for record in records)),
        "transition_distribution": summarize_distribution_weighted(records, "__transition_key__"),
        "margin_regime_distribution": summarize_distribution_weighted(records, "margin_regime"),
        "baseline_features": {
            key: weighted_mean(records, key)
            for key in feature_keys
        },
    }


def summarize_distribution_weighted(records: list[dict[str, Any]], key: str) -> dict[str, float]:
    grouped: dict[str, float] = defaultdict(float)
    total = 0.0
    for record in records:
        name = str(record.get(key))
        weight = float(support_weight(record))
        grouped[name] += weight
        total += weight
    if total <= 0.0:
        return {}
    return {name: value / total for name, value in sorted(grouped.items())}


def compare_feature_sets(records_a: list[dict[str, Any]], records_b: list[dict[str, Any]], feature_keys: list[str], topk: int) -> dict[str, Any]:
    summary_a = summarize_cohort(records_a, feature_keys)
    summary_b = summarize_cohort(records_b, feature_keys)

    feature_gaps = []
    for key in feature_keys:
        mean_a = weighted_mean(records_a, key)
        mean_b = weighted_mean(records_b, key)
        gap = None if mean_a is None or mean_b is None else float(mean_a - mean_b)
        ps = pooled_std(records_a, records_b, key)
        standardized_gap = None if gap is None or ps is None or ps <= 1e-12 else float(gap / ps)
        feature_gaps.append(
            {
                "feature": key,
                "weighted_mean_a": mean_a,
                "weighted_mean_b": mean_b,
                "gap_a_minus_b": gap,
                "pooled_std": ps,
                "standardized_gap": standardized_gap,
            }
        )

    sortable = [
        item for item in feature_gaps if item["standardized_gap"] is not None
    ]
    sortable.sort(key=lambda item: abs(float(item["standardized_gap"])), reverse=True)

    return {
        "cohort_a": summary_a,
        "cohort_b": summary_b,
        "all_feature_gaps": feature_gaps,
        "top_separative_features": sortable[:topk],
    }


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    transitions = load_csv(args.transitions_csv)
    require_columns(
        transitions,
        [
            "category",
            "kp_idx",
            "best_other_idx_center",
            "scale_a",
            "count_a",
            "count_b",
            "margin_regime",
            "erosion_case",
        ] + args.cohort_keys,
    )
    transitions = [dict(record, __transition_key__=f"{record['scale_a']}->{record['scale_b']}") for record in transitions]

    records = load_csv(args.records_csv)
    src_key = f"src_rival_sim_r{args.context_radius}"
    trg_key = f"trg_rival_sim_r{args.context_radius}"
    cross_gt_key = f"cross_gt_score_r{args.context_radius}"
    cross_rival_key = f"cross_rival_score_r{args.context_radius}"
    cross_margin_key = f"cross_margin_r{args.context_radius}"
    target_more_key = f"target_more_collapsed_r{args.context_radius}"
    require_columns(
        records,
        [
            "category",
            "kp_idx",
            "best_other_idx_center",
            args.scale_field,
            "correct",
            "center_margin",
            src_key,
            trg_key,
            cross_gt_key,
            cross_rival_key,
            cross_margin_key,
            target_more_key,
        ],
    )

    baseline_groups, factor_fields_used = aggregate_baseline_groups(records, args.scale_field, args.context_radius, args.factor_fields)
    annotated_records = attach_baseline_features(transitions, baseline_groups, args.cohort_keys)
    if not annotated_records:
        raise RuntimeError("No annotated cohort baseline records found. Check transitions_csv, records_csv, and scale_field.")

    baseline_feature_keys = [
        "failure_rate",
        "mean_center_margin",
        "mean_best_other_trg_norm_dist",
        "mean_src_rival_sim",
        "mean_trg_rival_sim",
        "mean_cross_gt_score",
        "mean_cross_rival_score",
        "mean_cross_margin",
        "target_more_collapsed_rate",
    ] + [f"mean_{field}" for field in factor_fields_used]

    cohort_summary = {}
    for cohort_name in args.cohort_keys:
        subset = [record for record in annotated_records if str(record["cohort_name"]) == cohort_name]
        cohort_summary[cohort_name] = summarize_cohort(subset, baseline_feature_keys)

    pairwise = {}
    for idx_a in range(len(args.cohort_keys)):
        for idx_b in range(idx_a + 1, len(args.cohort_keys)):
            name_a = args.cohort_keys[idx_a]
            name_b = args.cohort_keys[idx_b]
            subset_a = [record for record in annotated_records if str(record["cohort_name"]) == name_a]
            subset_b = [record for record in annotated_records if str(record["cohort_name"]) == name_b]
            pairwise[f"{name_a}__vs__{name_b}"] = compare_feature_sets(subset_a, subset_b, baseline_feature_keys, args.topk)

    summary = {
        "num_transitions": len(transitions),
        "num_baseline_groups": len(baseline_groups),
        "num_annotated_records": len(annotated_records),
        "cohort_keys": args.cohort_keys,
        "context_radius": args.context_radius,
        "scale_field": args.scale_field,
        "factor_fields_used": factor_fields_used,
        "cohorts": cohort_summary,
        "pairwise_comparisons": pairwise,
        "notes": {
            "interpretation": (
                "If downstream erosion cohorts already separate on baseline single-scale signatures, then "
                "cohort-aware intervention is in principle feasible at inference time. If they do not separate, "
                "the cohorts may still be analytically distinct but hard to route with explicit gating."
            )
        },
    }

    records_csv = os.path.join(args.output_dir, "cohort_baseline_signature_records.csv")
    summary_json = os.path.join(args.output_dir, "cohort_baseline_signature_summary.json")
    write_records_csv(annotated_records, records_csv)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    focus_key = None
    if len(args.cohort_keys) >= 2:
        focus_key = f"{args.cohort_keys[0]}__vs__{args.cohort_keys[1]}"

    print(f"Saved records to: {records_csv}")
    print(f"Saved summary to: {summary_json}")
    print(
        "Focus comparison top features:",
        None if focus_key is None else pairwise.get(focus_key, {}).get("top_separative_features"),
    )


if __name__ == "__main__":
    main()

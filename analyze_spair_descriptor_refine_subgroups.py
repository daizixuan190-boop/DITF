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
            "Analyze whether descriptor-refinement gains are concentrated in a stable subgroup. "
            "Consumes per_point_records from eval_spair_descriptor_refine.py and estimates "
            "how much selective gating could realistically recover."
        )
    )
    parser.add_argument("--records_csv", type=str, required=True, help="Path to per_point_records_sdr_*.csv.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs.")
    parser.add_argument(
        "--num_bins",
        type=int,
        default=5,
        help="Requested number of quantile bins for continuous fields.",
    )
    parser.add_argument(
        "--min_group_count",
        type=int,
        default=50,
        help="Minimum group size required for a valid subgroup summary.",
    )
    parser.add_argument(
        "--candidate_fields",
        nargs="+",
        default=[
            "source_ambiguity",
            "risk_weight",
            "margin",
            "descriptor_delta_norm",
            "rival_count",
            "scale_variation",
            "raw_gt_rank",
            "raw_gt_margin",
            "raw_gt_score",
        ],
        help="Fields used to search for beneficial selective subgroups.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=20,
        help="Number of top subgroups / rules to keep in the summary.",
    )
    parser.add_argument(
        "--max_rule_candidates",
        type=int,
        default=21,
        help="Maximum number of quantile-based threshold candidates scanned per continuous field.",
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
        raise RuntimeError("records_csv is empty.")
    missing = [column for column in columns if column not in records[0]]
    if missing:
        raise RuntimeError(f"records_csv is missing required columns: {missing}")


def safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def safe_rate(values: list[int]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


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


def format_interval(low: float, high: float) -> str:
    return f"({low:.6g},{high:.6g}]"


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


def normalize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for record in records:
        raw_correct = int(record["raw_correct"])
        final_correct = int(record["correct"])
        changed = int(
            int(record["raw_pred_x"]) != int(record["pred_x"])
            or int(record["raw_pred_y"]) != int(record["pred_y"])
        )
        improvement = int(raw_correct == 0 and final_correct == 1)
        harm = int(raw_correct == 1 and final_correct == 0)
        delta = int(final_correct - raw_correct)
        margin_gain = float(record["gt_margin_gain"])
        rank_gain = int(record["gt_rank_gain"])
        normalized.append(
            {
                **record,
                "changed_prediction": changed,
                "improvement": improvement,
                "harm": harm,
                "net_delta": delta,
                "positive_mechanism": int(margin_gain > 0.0 or rank_gain > 0),
                "strong_positive_mechanism": int(margin_gain > 0.0 and rank_gain > 0),
                "negative_mechanism": int(margin_gain < 0.0 or rank_gain < 0),
            }
        )
    return normalized


def summarize_subset(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(records),
        "raw_acc": safe_rate([int(record["raw_correct"]) for record in records]),
        "final_acc": safe_rate([int(record["correct"]) for record in records]),
        "net_delta_rate": safe_mean([float(record["net_delta"]) for record in records]),
        "changed_rate": safe_rate([int(record["changed_prediction"]) for record in records]),
        "improvement_rate": safe_rate([int(record["improvement"]) for record in records]),
        "harm_rate": safe_rate([int(record["harm"]) for record in records]),
        "positive_mechanism_rate": safe_rate([int(record["positive_mechanism"]) for record in records]),
        "strong_positive_mechanism_rate": safe_rate([int(record["strong_positive_mechanism"]) for record in records]),
        "mean_gt_rank_gain": safe_mean([float(record["gt_rank_gain"]) for record in records]),
        "mean_gt_margin_gain": safe_mean([float(record["gt_margin_gain"]) for record in records]),
        "mean_source_ambiguity": safe_mean([float(record["source_ambiguity"]) for record in records if record.get("source_ambiguity") is not None]),
        "mean_risk_weight": safe_mean([float(record["risk_weight"]) for record in records if record.get("risk_weight") is not None]),
        "mean_margin": safe_mean([float(record["margin"]) for record in records if record.get("margin") is not None]),
        "mean_descriptor_delta_norm": safe_mean([float(record["descriptor_delta_norm"]) for record in records if record.get("descriptor_delta_norm") is not None]),
        "mean_rival_count": safe_mean([float(record["rival_count"]) for record in records if record.get("rival_count") is not None]),
    }


def oracle_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    raw_correct = [int(record["raw_correct"]) for record in records]
    final_correct = [int(record["correct"]) for record in records]
    improvements = sum(int(record["improvement"]) for record in records)
    harms = sum(int(record["harm"]) for record in records)
    raw_total = int(sum(raw_correct))
    final_total = int(sum(final_correct))
    oracle_total = raw_total + improvements
    return {
        "count": len(records),
        "raw_acc": raw_total / max(len(records), 1),
        "final_acc": final_total / max(len(records), 1),
        "oracle_acc_if_perfect_gate": oracle_total / max(len(records), 1),
        "actual_net_gain": (final_total - raw_total) / max(len(records), 1),
        "oracle_net_gain": improvements / max(len(records), 1),
        "improvement_count": improvements,
        "harm_count": harms,
    }


def discrete_group_summary(records: list[dict[str, Any]], field: str, min_group_count: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        value = record.get(field)
        key = "unknown" if value is None else str(value)
        grouped.setdefault(key, []).append(record)

    output = []
    for key, subset in grouped.items():
        if len(subset) < min_group_count:
            continue
        item = {
            "field": field,
            "group": key,
            **summarize_subset(subset),
        }
        output.append(item)
    output.sort(key=lambda item: (float(item["net_delta_rate"] or -1e9), int(item["count"])), reverse=True)
    return output


def quantile_group_summary(records: list[dict[str, Any]], field: str, num_bins: int, min_group_count: int) -> list[dict[str, Any]]:
    valid_values = [float(record[field]) for record in records if record.get(field) is not None]
    edges = build_quantile_edges(valid_values, num_bins)
    if len(edges) < 2:
        return []

    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        value = record.get(field)
        if value is None:
            continue
        bucket = assign_quantile_bucket(float(value), edges, field)
        grouped.setdefault(bucket, []).append(record)

    output = []
    for bucket, subset in grouped.items():
        if len(subset) < min_group_count:
            continue
        item = {
            "field": field,
            "group": bucket,
            **summarize_subset(subset),
        }
        output.append(item)
    output.sort(key=lambda item: quantile_bucket_sort_key(str(item["group"])))
    return output


def build_threshold_rules(
    records: list[dict[str, Any]],
    field: str,
    min_group_count: int,
    max_rule_candidates: int,
) -> list[dict[str, Any]]:
    values = [float(record[field]) for record in records if record.get(field) is not None]
    if len(values) < 2:
        return []

    unique_values = unique_sorted(values)
    if len(unique_values) < 2:
        return []

    candidate_count = max(3, int(max_rule_candidates))
    if len(unique_values) <= min(32, candidate_count):
        thresholds = unique_values[1:-1]
    else:
        num_candidates = max(3, candidate_count)
        raw_thresholds = np.quantile(
            np.asarray(values, dtype=np.float64),
            np.linspace(0.0, 1.0, num_candidates + 2)[1:-1],
        )
        thresholds = unique_sorted(raw_thresholds.tolist())

    rules = []
    for threshold in thresholds:
        for direction in ("le", "ge"):
            if direction == "le":
                subset = [record for record in records if record.get(field) is not None and float(record[field]) <= threshold]
                rule_name = f"{field} <= {threshold:.6g}"
            else:
                subset = [record for record in records if record.get(field) is not None and float(record[field]) >= threshold]
                rule_name = f"{field} >= {threshold:.6g}"

            if len(subset) < min_group_count:
                continue
            summary = summarize_subset(subset)
            item = {
                "rule": rule_name,
                "field": field,
                "threshold": float(threshold),
                "direction": direction,
                **summary,
            }
            rules.append(item)
    rules.sort(key=lambda item: (float(item["net_delta_rate"] or -1e9), float(item["strong_positive_mechanism_rate"] or -1e9), int(item["count"])), reverse=True)
    return rules


def apply_gate(records: list[dict[str, Any]], selected: list[dict[str, Any]]) -> dict[str, Any]:
    selected_ids = {
        (record["category"], record["pair_name"], int(record["kp_idx"]))
        for record in selected
    }
    raw_total = 0
    gated_total = 0
    selected_count = 0
    for record in records:
        raw_total += int(record["raw_correct"])
        key = (record["category"], record["pair_name"], int(record["kp_idx"]))
        if key in selected_ids:
            selected_count += 1
            gated_total += int(record["correct"])
        else:
            gated_total += int(record["raw_correct"])
    return {
        "selected_count": selected_count,
        "selected_rate": selected_count / max(len(records), 1),
        "gated_acc": gated_total / max(len(records), 1),
        "gated_net_gain": (gated_total - raw_total) / max(len(records), 1),
    }


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    records = normalize_records(load_csv(args.records_csv))
    require_columns(
        records,
        [
            "category",
            "pair_name",
            "kp_idx",
            "raw_correct",
            "correct",
            "gt_rank_gain",
            "gt_margin_gain",
            "source_ambiguity",
            "risk_weight",
            "margin",
            "descriptor_delta_norm",
            "rival_count",
        ],
    )

    summary = {
        "num_records": len(records),
        "overall": summarize_subset(records),
        "oracle": oracle_summary(records),
        "subgroups": {},
        "top_rules": [],
    }

    annotated_records = []
    for field in args.candidate_fields:
        if field not in records[0]:
            continue
        example_value = next((record[field] for record in records if record.get(field) is not None), None)
        if example_value is None:
            continue

        if isinstance(example_value, str):
            discrete = discrete_group_summary(records, field, args.min_group_count)
            summary["subgroups"][field] = {"type": "categorical", "groups": discrete}
            annotated_records.extend(discrete)
            continue

        numeric_values = [float(record[field]) for record in records if record.get(field) is not None]
        unique_count = len(set(numeric_values))
        should_build_rules = unique_count > max(8, args.num_bins)
        if unique_count <= max(8, args.num_bins):
            discrete = discrete_group_summary(records, field, args.min_group_count)
            summary["subgroups"][field] = {"type": "discrete_numeric", "groups": discrete}
            annotated_records.extend(discrete)
        else:
            quantile_groups = quantile_group_summary(records, field, args.num_bins, args.min_group_count)
            summary["subgroups"][field] = {"type": "continuous_quantile", "groups": quantile_groups}
            annotated_records.extend(quantile_groups)

        if should_build_rules:
            rules = build_threshold_rules(records, field, args.min_group_count, args.max_rule_candidates)
        else:
            rules = []
        if rules:
            best_rule = rules[0]
            gated = apply_gate(
                records,
                [
                    record
                    for record in records
                    if record.get(field) is not None
                    and (
                        float(record[field]) <= float(best_rule["threshold"])
                        if best_rule["direction"] == "le"
                        else float(record[field]) >= float(best_rule["threshold"])
                    )
                ],
            )
            summary["top_rules"].append({**best_rule, **gated})

    summary["top_rules"].sort(
        key=lambda item: (
            float(item.get("gated_net_gain", -1e9)),
            float(item.get("net_delta_rate", -1e9)),
            int(item.get("count", 0)),
        ),
        reverse=True,
    )
    summary["top_rules"] = summary["top_rules"][: args.topk]

    summary_path = os.path.join(args.output_dir, "descriptor_refine_subgroups_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    records_path = os.path.join(args.output_dir, "descriptor_refine_subgroups_records.csv")
    write_records_csv(annotated_records, records_path)

    print(f"Saved summary to: {summary_path}")
    print(f"Saved subgroup records to: {records_path}")
    print(f"Overall: {summary['overall']}")
    print(f"Oracle: {summary['oracle']}")
    if summary["top_rules"]:
        print("Top selective rules:")
        for item in summary["top_rules"][: min(5, len(summary["top_rules"]))]:
            print(
                f"  {item['rule']}: gated_net_gain={item['gated_net_gain']} "
                f"net_delta_rate={item['net_delta_rate']} count={item['count']} "
                f"strong_positive_mechanism_rate={item['strong_positive_mechanism_rate']}"
            )


if __name__ == "__main__":
    main()

import argparse
import csv
import json
import math
import os
from typing import Any

import numpy as np


CORE_KEYS = [
    "shift_ratio_src",
    "content_ratio_src",
    "interaction_ratio_src",
]

CONTROL_KEYS = [
    "src_boundary_margin",
    "trg_boundary_margin",
    "pair_displacement",
]

MERGE_KEYS = [
    "category",
    "pair_name",
    "src_imname",
    "trg_imname",
    "kp_idx",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Merge residual-analysis and matching-ceiling records to identify which post-AdaLN factor "
            "most strongly drives current error and rank degradation on SPair-71k."
        )
    )
    parser.add_argument("--residual_csv", type=str, required=True, help="Path to per_point_records.csv.")
    parser.add_argument("--ceiling_csv", type=str, required=True, help="Path to matching_ceiling_records.csv.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save rank-driver analysis.")
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
            parsed = {key: parse_scalar(value) for key, value in row.items()}
            records.append(parsed)
    return records


def merge_records(
    residual_records: list[dict[str, Any]],
    ceiling_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ceiling_map = {
        tuple(record[key] for key in MERGE_KEYS): record
        for record in ceiling_records
    }
    merged: list[dict[str, Any]] = []
    for residual in residual_records:
        merge_id = tuple(residual[key] for key in MERGE_KEYS)
        ceiling = ceiling_map.get(merge_id)
        if ceiling is None:
            continue
        record = dict(residual)
        record["oracle_best_rank"] = float(ceiling["oracle_best_rank"])
        record["oracle_best_rank_frac"] = float(ceiling["oracle_best_rank_frac"])
        record["rank_gt_1"] = int(float(ceiling["oracle_best_rank"]) > 1)
        record["rank_gt_10"] = int(float(ceiling["oracle_best_rank"]) > 10)
        record["rank_gt_50"] = int(float(ceiling["oracle_best_rank"]) > 50)
        record["rank_gt_100"] = int(float(ceiling["oracle_best_rank"]) > 100)
        record["current_error"] = 1 - int(record["correct"])
        record["log_oracle_rank"] = float(math.log1p(float(ceiling["oracle_best_rank"])))
        merged.append(record)
    return merged


def decile_target_summary(records: list[dict[str, Any]], score_key: str) -> list[dict[str, Any]]:
    if not records:
        return []
    values = np.array([float(r[score_key]) for r in records], dtype=np.float64)
    order = np.argsort(values)
    chunks = np.array_split(order, 10)
    summary = []
    for idx, chunk in enumerate(chunks):
        if len(chunk) == 0:
            continue
        subset = [records[i] for i in chunk]
        summary.append(
            {
                "decile": idx,
                "count": len(subset),
                "score_min": float(min(float(r[score_key]) for r in subset)),
                "score_max": float(max(float(r[score_key]) for r in subset)),
                "current_error_rate": float(np.mean([float(r["current_error"]) for r in subset])),
                "mean_oracle_rank": float(np.mean([float(r["oracle_best_rank"]) for r in subset])),
                "rank_gt_10_rate": float(np.mean([float(r["rank_gt_10"]) for r in subset])),
                "rank_gt_50_rate": float(np.mean([float(r["rank_gt_50"]) for r in subset])),
                "rank_gt_100_rate": float(np.mean([float(r["rank_gt_100"]) for r in subset])),
            }
        )
    return summary


def conditional_decile_summary(
    records: list[dict[str, Any]],
    outer_key: str,
    inner_key: str,
) -> list[dict[str, Any]]:
    if not records:
        return []
    outer_values = np.array([float(r[outer_key]) for r in records], dtype=np.float64)
    outer_order = np.argsort(outer_values)
    outer_chunks = np.array_split(outer_order, 4)
    results = []
    for quartile_idx, chunk in enumerate(outer_chunks):
        subset = [records[i] for i in chunk]
        if len(subset) < 20:
            continue
        deciles = decile_target_summary(subset, inner_key)
        results.append(
            {
                "outer_key": outer_key,
                "inner_key": inner_key,
                "outer_quartile": quartile_idx,
                "outer_min": float(min(float(r[outer_key]) for r in subset)),
                "outer_max": float(max(float(r[outer_key]) for r in subset)),
                "inner_bottom_current_error": deciles[0]["current_error_rate"],
                "inner_top_current_error": deciles[-1]["current_error_rate"],
                "inner_bottom_rank_gt_10": deciles[0]["rank_gt_10_rate"],
                "inner_top_rank_gt_10": deciles[-1]["rank_gt_10_rate"],
                "current_error_gap": deciles[-1]["current_error_rate"] - deciles[0]["current_error_rate"],
                "rank_gt_10_gap": deciles[-1]["rank_gt_10_rate"] - deciles[0]["rank_gt_10_rate"],
                "count": len(subset),
            }
        )
    return results


def standardized_ols(records: list[dict[str, Any]], target_key: str, feature_keys: list[str]) -> dict[str, Any]:
    if not records:
        return {}
    X = np.array([[float(r[key]) for key in feature_keys] for r in records], dtype=np.float64)
    y = np.array([float(r[target_key]) for r in records], dtype=np.float64)

    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)
    y_mean = y.mean()
    y_std = y.std()

    valid = (x_std > 1e-12)
    X = X[:, valid]
    valid_features = [key for key, keep in zip(feature_keys, valid) if keep]
    x_mean = x_mean[valid]
    x_std = x_std[valid]
    if y_std <= 1e-12 or X.shape[1] == 0:
        return {"target": target_key, "num_samples": len(records), "coefficients": {}}

    Xz = (X - x_mean) / x_std
    yz = (y - y_mean) / y_std
    X_design = np.concatenate([np.ones((Xz.shape[0], 1), dtype=np.float64), Xz], axis=1)
    coef, *_ = np.linalg.lstsq(X_design, yz, rcond=None)
    preds = X_design @ coef
    ss_res = float(np.sum((yz - preds) ** 2))
    ss_tot = float(np.sum((yz - yz.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

    return {
        "target": target_key,
        "num_samples": len(records),
        "r2": float(r2),
        "coefficients": {
            key: float(weight)
            for key, weight in zip(valid_features, coef[1:])
        },
    }


def write_records_csv(records: list[dict[str, Any]], csv_path: str):
    if not records:
        return
    fieldnames = sorted({key for record in records for key in record.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    residual_records = load_csv(args.residual_csv)
    ceiling_records = load_csv(args.ceiling_csv)
    merged_records = merge_records(residual_records, ceiling_records)
    failure_records = [record for record in merged_records if int(record["current_error"]) == 1]

    feature_keys = CORE_KEYS + CONTROL_KEYS

    summary = {
        "num_points": len(merged_records),
        "num_failures": len(failure_records),
        "base_current_error_rate": float(np.mean([float(r["current_error"]) for r in merged_records])) if merged_records else None,
        "base_rank_gt_10_rate": float(np.mean([float(r["rank_gt_10"]) for r in merged_records])) if merged_records else None,
        "base_rank_gt_50_rate": float(np.mean([float(r["rank_gt_50"]) for r in merged_records])) if merged_records else None,
        "base_rank_gt_100_rate": float(np.mean([float(r["rank_gt_100"]) for r in merged_records])) if merged_records else None,
        "deciles_all": {
            key: decile_target_summary(merged_records, key)
            for key in CORE_KEYS
        },
        "deciles_failures_only": {
            key: decile_target_summary(failure_records, key)
            for key in CORE_KEYS
        },
        "conditional_controls": {
            "shift_within_content_quartiles": conditional_decile_summary(merged_records, "content_ratio_src", "shift_ratio_src"),
            "shift_within_interaction_quartiles": conditional_decile_summary(merged_records, "interaction_ratio_src", "shift_ratio_src"),
            "interaction_within_shift_quartiles": conditional_decile_summary(merged_records, "shift_ratio_src", "interaction_ratio_src"),
        },
        "standardized_ols": {
            "current_error": standardized_ols(merged_records, "current_error", feature_keys),
            "log_oracle_rank": standardized_ols(merged_records, "log_oracle_rank", feature_keys),
            "rank_gt_10": standardized_ols(merged_records, "rank_gt_10", feature_keys),
        },
        "notes": {
            "scope": "Frozen DiTF features after post-AdaLN, no intervention applied.",
            "interpretation": (
                "Deciles show monotonic association structure. Standardized OLS is for driver prioritization, "
                "not a causal proof."
            ),
        },
    }

    merged_csv = os.path.join(args.output_dir, "rank_driver_records.csv")
    summary_json = os.path.join(args.output_dir, "rank_driver_summary.json")
    write_records_csv(merged_records, merged_csv)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved merged records to: {merged_csv}")
    print(f"Saved summary to: {summary_json}")
    print(f"Num points: {summary['num_points']}")
    print(f"Num failures: {summary['num_failures']}")
    print(f"Base current error rate: {summary['base_current_error_rate']}")
    print(f"Base rank_gt_10 rate: {summary['base_rank_gt_10_rate']}")


if __name__ == "__main__":
    main()

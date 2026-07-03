import argparse
import csv
import json
import math
import os
from typing import Any

import numpy as np


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
            "Diagnose whether remaining SPair-71k failures are better explained by "
            "selection-like mistakes, local ambiguity, or deep representation failure."
        )
    )
    parser.add_argument("--residual_csv", type=str, required=True, help="Path to per_point_records.csv.")
    parser.add_argument("--ceiling_csv", type=str, required=True, help="Path to matching_ceiling_records.csv.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save analysis outputs.")
    parser.add_argument("--selection_topk", nargs="+", type=int, default=[5, 10, 50], help="Top-k cutoffs used to define selection-like failures.")
    parser.add_argument("--near_miss_multipliers", nargs="+", type=float, default=[2.0, 3.0], help="Multipliers over the PCK threshold 0.1 for near-miss diagnostics.")
    parser.add_argument("--ambiguity_rank_cutoff", type=int, default=10, help="Top-k rank cutoff for ambiguity-like failures.")
    parser.add_argument("--ambiguity_margin_quantile", type=float, default=0.25, help="Low-margin quantile used in the local-ambiguity proxy.")
    parser.add_argument("--ambiguity_entropy_quantile", type=float, default=0.75, help="High-entropy quantile used in the local-ambiguity proxy.")
    parser.add_argument("--ambiguity_near_miss_multiplier", type=float, default=3.0, help="Normalized distance multiplier for the ambiguity proxy.")
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
        record["oracle_best_rank"] = int(ceiling["oracle_best_rank"])
        record["oracle_best_rank_frac"] = float(ceiling["oracle_best_rank_frac"])
        record["current_error"] = 1 - int(record["correct"])
        merged.append(record)
    return merged


def rate(records: list[dict[str, Any]], key: str) -> float | None:
    if not records:
        return None
    return float(np.mean([float(record[key]) for record in records]))


def decile_summary(records: list[dict[str, Any]], score_key: str) -> list[dict[str, Any]]:
    if not records:
        return []
    values = np.array([float(record[score_key]) for record in records], dtype=np.float64)
    order = np.argsort(values)
    chunks = np.array_split(order, 10)
    output = []
    for idx, chunk in enumerate(chunks):
        if len(chunk) == 0:
            continue
        subset = [records[i] for i in chunk]
        output.append(
            {
                "decile": idx,
                "count": len(subset),
                "score_min": float(min(float(record[score_key]) for record in subset)),
                "score_max": float(max(float(record[score_key]) for record in subset)),
                "error_rate": rate(subset, "current_error"),
                "mean_oracle_rank": float(np.mean([float(record["oracle_best_rank"]) for record in subset])),
            }
        )
    return output


def tag_failure_modes(records: list[dict[str, Any]], args) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    failures = [record for record in records if int(record["current_error"]) == 1]
    if not failures:
        return records, {"num_failures": 0}

    error_margins = np.array([float(record["sim_margin"]) for record in failures], dtype=np.float64)
    error_entropies = np.array([float(record["sim_entropy"]) for record in failures], dtype=np.float64)
    margin_threshold = float(np.quantile(error_margins, args.ambiguity_margin_quantile))
    entropy_threshold = float(np.quantile(error_entropies, args.ambiguity_entropy_quantile))
    near_miss_threshold = 0.1 * args.ambiguity_near_miss_multiplier

    for record in records:
        oracle_rank = int(record["oracle_best_rank"])
        current_error = int(record["current_error"])
        norm_dist = float(record["norm_dist"])
        sim_margin = float(record["sim_margin"])
        sim_entropy = float(record["sim_entropy"])

        for topk in args.selection_topk:
            record[f"selection_like@{topk}"] = int(current_error == 1 and oracle_rank <= topk)
            record[f"representation_like@{topk}"] = int(current_error == 1 and oracle_rank > topk)

        for mult in args.near_miss_multipliers:
            record[f"near_miss@x{mult:g}"] = int(current_error == 1 and norm_dist <= 0.1 * mult)

        record["local_ambiguity_proxy"] = int(
            current_error == 1
            and oracle_rank <= args.ambiguity_rank_cutoff
            and norm_dist <= near_miss_threshold
            and sim_margin <= margin_threshold
            and sim_entropy >= entropy_threshold
        )

        record["deep_representation_proxy"] = int(
            current_error == 1
            and oracle_rank > max(args.selection_topk)
        )

    summary = {
        "num_points": len(records),
        "num_failures": len(failures),
        "error_rate": rate(records, "current_error"),
        "margin_threshold_q": {
            "quantile": args.ambiguity_margin_quantile,
            "value": margin_threshold,
        },
        "entropy_threshold_q": {
            "quantile": args.ambiguity_entropy_quantile,
            "value": entropy_threshold,
        },
        "selection_like_rates": {
            str(topk): rate(failures, f"selection_like@{topk}")
            for topk in args.selection_topk
        },
        "representation_like_rates": {
            str(topk): rate(failures, f"representation_like@{topk}")
            for topk in args.selection_topk
        },
        "near_miss_rates": {
            f"x{mult:g}": rate(failures, f"near_miss@x{mult:g}")
            for mult in args.near_miss_multipliers
        },
        "local_ambiguity_proxy_rate": rate(failures, "local_ambiguity_proxy"),
        "deep_representation_proxy_rate": rate(failures, "deep_representation_proxy"),
        "error_deciles": {
            "sim_margin": decile_summary(failures, "sim_margin"),
            "sim_entropy": decile_summary(failures, "sim_entropy"),
            "norm_dist": decile_summary(failures, "norm_dist"),
            "oracle_best_rank": decile_summary(failures, "oracle_best_rank"),
        },
        "notes": {
            "selection_like": "Current top-1 is wrong, but GT already appears within oracle top-k on the frozen similarity map.",
            "representation_like": "Current top-1 is wrong and GT still ranks beyond top-k on the frozen similarity map.",
            "local_ambiguity_proxy": (
                "A stricter proxy: wrong prediction, GT rank still within a small top-k, error is spatially near the GT, "
                "margin is low, and similarity-map entropy is high."
            ),
            "deep_representation_proxy": "A coarse proxy for genuinely poor token discriminability on the frozen feature map.",
        },
    }
    return records, summary


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    residual_records = load_csv(args.residual_csv)
    ceiling_records = load_csv(args.ceiling_csv)
    merged_records = merge_records(residual_records, ceiling_records)
    tagged_records, summary = tag_failure_modes(merged_records, args)

    records_csv = os.path.join(args.output_dir, "failure_mode_records.csv")
    summary_json = os.path.join(args.output_dir, "failure_mode_summary.json")
    write_records_csv(tagged_records, records_csv)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved records to: {records_csv}")
    print(f"Saved summary to: {summary_json}")
    print(f"Num points: {summary['num_points']}")
    print(f"Num failures: {summary['num_failures']}")
    print(f"Error rate: {summary['error_rate']}")
    print(f"Selection-like failure rates: {summary['selection_like_rates']}")
    print(f"Representation-like failure rates: {summary['representation_like_rates']}")
    print(f"Near-miss failure rates: {summary['near_miss_rates']}")
    print(f"Local ambiguity proxy rate: {summary['local_ambiguity_proxy_rate']}")
    print(f"Deep representation proxy rate: {summary['deep_representation_proxy_rate']}")


if __name__ == "__main__":
    main()

import argparse
import csv
import json
import os
from typing import Any

import numpy as np


MERGE_KEYS = ["category", "pair_name", "src_imname", "trg_imname", "kp_idx"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare SPair per-point predictions from a baseline and a method run."
    )
    parser.add_argument("--baseline_csv", type=str, required=True, help="Baseline per-point CSV.")
    parser.add_argument("--method_csv", type=str, required=True, help="Method per-point CSV.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save summary JSON.")
    parser.add_argument("--tag_csv", type=str, default="", help="Optional extra CSV to tag subsets.")
    parser.add_argument("--tag_column", type=str, default="", help="Column from tag_csv used to define subset values.")
    parser.add_argument("--tag_values", nargs="*", default=[], help="Optional subset values to summarize; default uses all observed non-empty values.")
    return parser.parse_args()


def parse_scalar(value: str) -> Any:
    if value is None or value == "":
        return None
    try:
        num = float(value)
    except ValueError:
        return value
    if abs(num - round(num)) < 1e-12:
        return int(round(num))
    return num


def load_csv(path: str):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [{k: parse_scalar(v) for k, v in row.items()} for row in reader]


def merge_records(baseline_records, method_records):
    method_map = {tuple(record[key] for key in MERGE_KEYS): record for record in method_records}
    merged = []
    for base in baseline_records:
        key = tuple(base[key] for key in MERGE_KEYS)
        method = method_map.get(key)
        if method is None:
            continue
        merged.append(
            {
                **{key_name: base[key_name] for key_name in MERGE_KEYS},
                "baseline_correct": int(base["correct"]),
                "method_correct": int(method["correct"]),
                "baseline_norm_dist": float(base["norm_dist"]),
                "method_norm_dist": float(method["norm_dist"]),
                "delta_correct": int(method["correct"]) - int(base["correct"]),
                "delta_norm_dist": float(method["norm_dist"]) - float(base["norm_dist"]),
            }
        )
    return merged


def attach_tags(records, tag_records, tag_column):
    if not tag_records or not tag_column:
        return records, []
    tag_map = {tuple(record[key] for key in MERGE_KEYS): record for record in tag_records}
    observed_values = set()
    for record in records:
        tag_record = tag_map.get(tuple(record[key] for key in MERGE_KEYS))
        tag_value = None if tag_record is None else tag_record.get(tag_column)
        record[tag_column] = tag_value
        if tag_value is not None and tag_value != "":
            observed_values.add(str(tag_value))
    return records, sorted(observed_values)


def summarize(records):
    if not records:
        return {
            "count": 0,
            "baseline_pck": None,
            "method_pck": None,
            "delta_pck": None,
            "rescue_rate_among_baseline_failures": None,
            "flip_rate_among_baseline_successes": None,
            "mean_delta_norm_dist": None,
        }
    baseline_correct = np.array([int(record["baseline_correct"]) for record in records], dtype=np.float64)
    method_correct = np.array([int(record["method_correct"]) for record in records], dtype=np.float64)
    delta_norm = np.array([float(record["delta_norm_dist"]) for record in records], dtype=np.float64)

    fail_mask = baseline_correct == 0
    success_mask = baseline_correct == 1
    rescue_rate = None if fail_mask.sum() == 0 else float(np.mean(method_correct[fail_mask]))
    flip_rate = None if success_mask.sum() == 0 else float(np.mean(1.0 - method_correct[success_mask]))

    return {
        "count": int(len(records)),
        "baseline_pck": float(np.mean(baseline_correct)),
        "method_pck": float(np.mean(method_correct)),
        "delta_pck": float(np.mean(method_correct - baseline_correct)),
        "rescue_rate_among_baseline_failures": rescue_rate,
        "flip_rate_among_baseline_successes": flip_rate,
        "mean_delta_norm_dist": float(np.mean(delta_norm)),
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    baseline_records = load_csv(args.baseline_csv)
    method_records = load_csv(args.method_csv)
    merged_records = merge_records(baseline_records, method_records)

    observed_values = []
    if args.tag_csv:
        tag_records = load_csv(args.tag_csv)
        merged_records, observed_values = attach_tags(merged_records, tag_records, args.tag_column)

    subset_values = args.tag_values if args.tag_values else observed_values
    summary = {
        "overall": summarize(merged_records),
        "tag_column": args.tag_column if args.tag_csv else None,
        "subset_summaries": {},
    }
    if args.tag_csv and args.tag_column:
        for value in subset_values:
            subset = [record for record in merged_records if str(record.get(args.tag_column)) == str(value)]
            summary["subset_summaries"][str(value)] = summarize(subset)

    summary_path = os.path.join(args.output_dir, "method_delta_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved summary to: {summary_path}")
    print("Overall:", summary["overall"])
    if summary["subset_summaries"]:
        print("Subset summaries:")
        for key, value in summary["subset_summaries"].items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

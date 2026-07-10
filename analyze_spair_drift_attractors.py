import argparse
import csv
import json
import math
import os
import random
from collections import defaultdict
from typing import Any

import numpy as np


FAIL_GROUPS = [
    "local_fail_rank_11_50",
    "local_fail_rank_51_100",
    "local_fail_rank_101_500",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze whether local candidate-cloud drift on SPair-71k follows stable attractor-like offset modes "
            "rather than random scatter. The script tests whether drift directions are recurrent within the same "
            "category and keypoint identity, compared against a within-category shuffled baseline."
        )
    )
    parser.add_argument(
        "--records_csv",
        type=str,
        required=True,
        help="Path to centroid_alignment_records.csv.",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs.")
    parser.add_argument(
        "--exclude_topn",
        type=int,
        default=5,
        help="Use records computed with this exclude_topn setting from centroid_alignment_records.csv.",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=FAIL_GROUPS,
        help="Failure groups to analyze.",
    )
    parser.add_argument(
        "--min_samples_per_kp",
        type=int,
        default=20,
        help="Minimum number of samples required for a category+kp_idx attractor analysis.",
    )
    parser.add_argument(
        "--num_shuffles",
        type=int,
        default=200,
        help="Number of within-category randomization trials for the shuffled baseline.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for shuffled baseline.")
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


def unit_vectors(records: list[dict[str, Any]], key_x: str, key_y: str) -> np.ndarray:
    vecs = []
    for record in records:
        dx = float(record[key_x])
        dy = float(record[key_y])
        norm = math.sqrt(dx * dx + dy * dy)
        if norm <= 1e-9:
            continue
        vecs.append([dx / norm, dy / norm])
    if not vecs:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(vecs, dtype=np.float64)


def mean_resultant_length(vecs: np.ndarray) -> float | None:
    if vecs.size == 0:
        return None
    mean_vec = np.mean(vecs, axis=0)
    return float(np.linalg.norm(mean_vec))


def mean_direction(vecs: np.ndarray) -> list[float] | None:
    if vecs.size == 0:
        return None
    mean_vec = np.mean(vecs, axis=0)
    norm = np.linalg.norm(mean_vec)
    if norm <= 1e-9:
        return [0.0, 0.0]
    mean_vec = mean_vec / norm
    return [float(mean_vec[0]), float(mean_vec[1])]


def mean_cos_to_direction(vecs: np.ndarray, direction: list[float] | None) -> float | None:
    if vecs.size == 0 or direction is None:
        return None
    direction_arr = np.asarray(direction, dtype=np.float64)
    direction_norm = np.linalg.norm(direction_arr)
    if direction_norm <= 1e-9:
        return None
    direction_arr = direction_arr / direction_norm
    return float(np.mean(vecs @ direction_arr))


def record_subset(records: list[dict[str, Any]], group_name: str) -> list[dict[str, Any]]:
    return [record for record in records if str(record["local_group"]) == group_name]


def build_kp_groups(records: list[dict[str, Any]], min_samples: int) -> dict[tuple[str, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        group_key = (str(record["category"]), int(record["kp_idx"]))
        grouped[group_key].append(record)
    return {
        key: subset
        for key, subset in grouped.items()
        if len(subset) >= min_samples
    }


def within_category_shuffle_baseline(
    records: list[dict[str, Any]],
    value_key_x: str,
    value_key_y: str,
    min_samples: int,
    num_shuffles: int,
    seed: int,
) -> dict[tuple[str, int], dict[str, float | None]]:
    rng = random.Random(seed)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_category[str(record["category"])].append(record)

    real_groups = build_kp_groups(records, min_samples)
    baseline: dict[tuple[str, int], list[float]] = {key: [] for key in real_groups.keys()}

    for category, cat_records in by_category.items():
        if len(cat_records) < min_samples:
            continue
        kp_values = [int(record["kp_idx"]) for record in cat_records]
        base_records = [dict(record) for record in cat_records]
        for _ in range(num_shuffles):
            shuffled_kps = kp_values[:]
            rng.shuffle(shuffled_kps)
            shuffled_records = []
            for record, shuffled_kp in zip(base_records, shuffled_kps):
                rec = dict(record)
                rec["kp_idx"] = shuffled_kp
                shuffled_records.append(rec)
            shuffled_groups = build_kp_groups(shuffled_records, min_samples)
            for key, subset in shuffled_groups.items():
                if key not in baseline:
                    continue
                vecs = unit_vectors(subset, value_key_x, value_key_y)
                mrl = mean_resultant_length(vecs)
                if mrl is not None:
                    baseline[key].append(mrl)

    summary = {}
    for key, values in baseline.items():
        summary[key] = {
            "shuffle_mean_mrl": float(np.mean(values)) if values else None,
            "shuffle_std_mrl": float(np.std(values)) if values else None,
            "num_effective_shuffles": len(values),
        }
    return summary


def analyze_groups(records: list[dict[str, Any]], args) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    per_group_records = []
    summary = {"groups": {}}

    for local_group in args.groups:
        group_records = record_subset(records, local_group)
        kp_groups = build_kp_groups(group_records, args.min_samples_per_kp)
        centroid_baseline = within_category_shuffle_baseline(
            group_records,
            "centroid_dx",
            "centroid_dy",
            args.min_samples_per_kp,
            args.num_shuffles,
            args.seed,
        )
        error_baseline = within_category_shuffle_baseline(
            group_records,
            "error_dx",
            "error_dy",
            args.min_samples_per_kp,
            args.num_shuffles,
            args.seed + 1,
        )

        for (category, kp_idx), subset in kp_groups.items():
            centroid_vecs = unit_vectors(subset, "centroid_dx", "centroid_dy")
            error_vecs = unit_vectors(subset, "error_dx", "error_dy")
            centroid_mrl = mean_resultant_length(centroid_vecs)
            error_mrl = mean_resultant_length(error_vecs)
            centroid_dir = mean_direction(centroid_vecs)
            error_dir = mean_direction(error_vecs)
            centroid_to_error = mean_cos_to_direction(error_vecs, centroid_dir)
            error_to_centroid = mean_cos_to_direction(centroid_vecs, error_dir)
            centroid_shuffle = centroid_baseline.get((category, kp_idx), {})
            error_shuffle = error_baseline.get((category, kp_idx), {})
            per_group_records.append(
                {
                    "local_group": local_group,
                    "category": category,
                    "kp_idx": kp_idx,
                    "count": len(subset),
                    "centroid_mrl": centroid_mrl,
                    "error_mrl": error_mrl,
                    "centroid_mean_dir_x": None if centroid_dir is None else centroid_dir[0],
                    "centroid_mean_dir_y": None if centroid_dir is None else centroid_dir[1],
                    "error_mean_dir_x": None if error_dir is None else error_dir[0],
                    "error_mean_dir_y": None if error_dir is None else error_dir[1],
                    "mean_error_cos_to_centroid_mean_dir": centroid_to_error,
                    "mean_centroid_cos_to_error_mean_dir": error_to_centroid,
                    "centroid_shuffle_mean_mrl": centroid_shuffle.get("shuffle_mean_mrl"),
                    "centroid_shuffle_std_mrl": centroid_shuffle.get("shuffle_std_mrl"),
                    "error_shuffle_mean_mrl": error_shuffle.get("shuffle_mean_mrl"),
                    "error_shuffle_std_mrl": error_shuffle.get("shuffle_std_mrl"),
                    "centroid_mrl_minus_shuffle": None
                    if centroid_mrl is None or centroid_shuffle.get("shuffle_mean_mrl") is None
                    else float(centroid_mrl - centroid_shuffle["shuffle_mean_mrl"]),
                    "error_mrl_minus_shuffle": None
                    if error_mrl is None or error_shuffle.get("shuffle_mean_mrl") is None
                    else float(error_mrl - error_shuffle["shuffle_mean_mrl"]),
                    "mean_centroid_norm_dist": float(np.mean([float(record["centroid_norm_dist"]) for record in subset])),
                    "mean_error_norm_dist": float(np.mean([float(record["error_norm_dist"]) for record in subset])),
                }
            )

        group_rows = [row for row in per_group_records if row["local_group"] == local_group]
        summary["groups"][local_group] = {
            "num_kp_groups": len(group_rows),
            "mean_centroid_mrl": float(np.mean([row["centroid_mrl"] for row in group_rows])) if group_rows else None,
            "mean_error_mrl": float(np.mean([row["error_mrl"] for row in group_rows])) if group_rows else None,
            "mean_centroid_mrl_minus_shuffle": float(np.mean([row["centroid_mrl_minus_shuffle"] for row in group_rows if row["centroid_mrl_minus_shuffle"] is not None])) if group_rows else None,
            "mean_error_mrl_minus_shuffle": float(np.mean([row["error_mrl_minus_shuffle"] for row in group_rows if row["error_mrl_minus_shuffle"] is not None])) if group_rows else None,
            "mean_error_cos_to_centroid_mean_dir": float(np.mean([row["mean_error_cos_to_centroid_mean_dir"] for row in group_rows if row["mean_error_cos_to_centroid_mean_dir"] is not None])) if group_rows else None,
            "mean_centroid_cos_to_error_mean_dir": float(np.mean([row["mean_centroid_cos_to_error_mean_dir"] for row in group_rows if row["mean_centroid_cos_to_error_mean_dir"] is not None])) if group_rows else None,
            "frac_centroid_mrl_above_shuffle": float(np.mean([row["centroid_mrl_minus_shuffle"] > 0.0 for row in group_rows if row["centroid_mrl_minus_shuffle"] is not None])) if group_rows else None,
            "frac_error_mrl_above_shuffle": float(np.mean([row["error_mrl_minus_shuffle"] > 0.0 for row in group_rows if row["error_mrl_minus_shuffle"] is not None])) if group_rows else None,
        }

    return per_group_records, summary


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    records = load_csv(args.records_csv)
    records = [
        record
        for record in records
        if int(record.get("exclude_topn", -1)) == args.exclude_topn
        and str(record.get("local_group")) in args.groups
    ]
    if not records:
        raise RuntimeError("No matching records found. Check records_csv, exclude_topn, and groups.")

    per_group_records, summary = analyze_groups(records, args)

    output_csv = os.path.join(args.output_dir, "drift_attractor_records.csv")
    output_json = os.path.join(args.output_dir, "drift_attractor_summary.json")
    write_records_csv(per_group_records, output_csv)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved records to: {output_csv}")
    print(f"Saved summary to: {output_json}")
    for group_name, group_summary in summary["groups"].items():
        print(
            f"{group_name}: num_kp_groups={group_summary['num_kp_groups']} "
            f"mean_centroid_mrl={group_summary['mean_centroid_mrl']} "
            f"mean_centroid_mrl_minus_shuffle={group_summary['mean_centroid_mrl_minus_shuffle']} "
            f"mean_error_cos_to_centroid_mean_dir={group_summary['mean_error_cos_to_centroid_mean_dir']}"
        )


if __name__ == "__main__":
    main()

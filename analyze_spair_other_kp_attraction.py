import argparse
import csv
import json
import math
import os
from collections import defaultdict
from typing import Any

import numpy as np


DEFAULT_GROUPS = [
    "local_success",
    "local_fail_rank_11_50",
    "local_fail_rank_51_100",
    "local_fail_rank_101_500",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze whether local candidate-cloud drift and prediction errors are attracted toward other annotated "
            "target keypoints on the same object. This tests a stronger mechanism hypothesis: the residual error is "
            "not arbitrary drift, but part-to-part identity confusion."
        )
    )
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to SPair-71k root.")
    parser.add_argument("--records_csv", type=str, required=True, help="Path to centroid_alignment_records.csv.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs.")
    parser.add_argument("--exclude_topn", type=int, default=5, help="Use records with this exclude_topn setting.")
    parser.add_argument("--groups", nargs="+", default=DEFAULT_GROUPS, help="Local groups to analyze.")
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


def safe_cosine(dx1: float, dy1: float, dx2: float, dy2: float) -> float | None:
    norm1 = math.sqrt(dx1 * dx1 + dy1 * dy1)
    norm2 = math.sqrt(dx2 * dx2 + dy2 * dy2)
    if norm1 <= 1e-9 or norm2 <= 1e-9:
        return None
    return float((dx1 * dx2 + dy1 * dy2) / (norm1 * norm2))


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def rate(records: list[dict[str, Any]], key: str) -> float | None:
    if not records:
        return None
    return float(np.mean([float(record[key]) for record in records]))


def summarize_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(records),
        "mean_best_other_error_cosine": mean_or_none([float(record["best_other_error_cosine"]) for record in records]),
        "mean_best_other_centroid_cosine": mean_or_none([float(record["best_other_centroid_cosine"]) for record in records]),
        "mean_nearest_other_to_pred_norm_dist": mean_or_none([float(record["nearest_other_to_pred_norm_dist"]) for record in records]),
        "mean_nearest_other_to_centroid_norm_dist": mean_or_none([float(record["nearest_other_to_centroid_norm_dist"]) for record in records]),
        "pred_closer_to_other_than_gt_rate": rate(records, "pred_closer_to_other_than_gt"),
        "centroid_closer_to_other_than_gt_rate": rate(records, "centroid_closer_to_other_than_gt"),
        "best_other_error_cosine_gt_0_9_rate": rate(records, "best_other_error_cosine_gt_0_9"),
        "best_other_centroid_cosine_gt_0_9_rate": rate(records, "best_other_centroid_cosine_gt_0_9"),
        "joint_attraction_signature_rate": rate(records, "joint_attraction_signature"),
    }


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

    by_pair: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_pair[(str(record["category"]), str(record["pair_name"]))].append(record)

    test_path = os.path.join(args.dataset_path, "PairAnnotation", "test")
    output_records: list[dict[str, Any]] = []

    for (_, pair_name), pair_records in by_pair.items():
        with open(os.path.join(test_path, pair_name), "r", encoding="utf-8") as f:
            data = json.load(f)

        trg_kps = data["trg_kps"]
        for record in pair_records:
            kp_idx = int(record["kp_idx"])
            trg_x = float(record["trg_x"])
            trg_y = float(record["trg_y"])
            pred_x = float(record["pred_x"])
            pred_y = float(record["pred_y"])
            centroid_x = float(record["centroid_x_topk"])
            centroid_y = float(record["centroid_y_topk"])
            error_dx = pred_x - trg_x
            error_dy = pred_y - trg_y
            centroid_dx = centroid_x - trg_x
            centroid_dy = centroid_y - trg_y
            error_norm_dist = float(record["error_norm_dist"])
            centroid_norm_dist = float(record["centroid_norm_dist"])

            best_other_error_cosine = None
            best_other_centroid_cosine = None
            nearest_other_to_pred_norm_dist = None
            nearest_other_to_centroid_norm_dist = None

            for other_idx, other_point in enumerate(trg_kps):
                if other_idx == kp_idx:
                    continue
                other_x = float(other_point[0])
                other_y = float(other_point[1])
                vec_dx = other_x - trg_x
                vec_dy = other_y - trg_y

                err_cos = safe_cosine(error_dx, error_dy, vec_dx, vec_dy)
                cen_cos = safe_cosine(centroid_dx, centroid_dy, vec_dx, vec_dy)
                if err_cos is not None:
                    best_other_error_cosine = err_cos if best_other_error_cosine is None else max(best_other_error_cosine, err_cos)
                if cen_cos is not None:
                    best_other_centroid_cosine = cen_cos if best_other_centroid_cosine is None else max(best_other_centroid_cosine, cen_cos)

                pred_dist = math.sqrt((pred_x - other_x) ** 2 + (pred_y - other_y) ** 2)
                cen_dist = math.sqrt((centroid_x - other_x) ** 2 + (centroid_y - other_y) ** 2)
                nearest_other_to_pred_norm_dist = pred_dist if nearest_other_to_pred_norm_dist is None else min(nearest_other_to_pred_norm_dist, pred_dist)
                nearest_other_to_centroid_norm_dist = cen_dist if nearest_other_to_centroid_norm_dist is None else min(nearest_other_to_centroid_norm_dist, cen_dist)

            threshold = max(float(record["error_norm_dist"]) * 0.0 + 1.0, 1.0)
            # Recover the true PCK normalizer from normalized distances already stored in the record.
            # Use the GT-pred distance whenever available; if error is exactly zero, fall back to centroid distance.
            raw_error_dist = math.sqrt(error_dx * error_dx + error_dy * error_dy)
            if error_norm_dist > 1e-9 and raw_error_dist > 1e-9:
                threshold = raw_error_dist / error_norm_dist
            elif centroid_norm_dist > 1e-9:
                raw_centroid_dist = math.sqrt(centroid_dx * centroid_dx + centroid_dy * centroid_dy)
                if raw_centroid_dist > 1e-9:
                    threshold = raw_centroid_dist / centroid_norm_dist

            nearest_other_to_pred_norm_dist = None if nearest_other_to_pred_norm_dist is None else float(nearest_other_to_pred_norm_dist / max(threshold, 1e-6))
            nearest_other_to_centroid_norm_dist = None if nearest_other_to_centroid_norm_dist is None else float(nearest_other_to_centroid_norm_dist / max(threshold, 1e-6))

            pred_closer_to_other_than_gt = None
            if nearest_other_to_pred_norm_dist is not None:
                pred_closer_to_other_than_gt = int(nearest_other_to_pred_norm_dist < error_norm_dist)
            centroid_closer_to_other_than_gt = None
            if nearest_other_to_centroid_norm_dist is not None:
                centroid_closer_to_other_than_gt = int(nearest_other_to_centroid_norm_dist < centroid_norm_dist)

            out_record = dict(record)
            out_record.update(
                {
                    "best_other_error_cosine": best_other_error_cosine,
                    "best_other_centroid_cosine": best_other_centroid_cosine,
                    "nearest_other_to_pred_norm_dist": nearest_other_to_pred_norm_dist,
                    "nearest_other_to_centroid_norm_dist": nearest_other_to_centroid_norm_dist,
                    "pred_closer_to_other_than_gt": pred_closer_to_other_than_gt,
                    "centroid_closer_to_other_than_gt": centroid_closer_to_other_than_gt,
                    "best_other_error_cosine_gt_0_9": None if best_other_error_cosine is None else int(best_other_error_cosine > 0.9),
                    "best_other_centroid_cosine_gt_0_9": None if best_other_centroid_cosine is None else int(best_other_centroid_cosine > 0.9),
                    "joint_attraction_signature": int(
                        best_other_error_cosine is not None
                        and best_other_centroid_cosine is not None
                        and best_other_error_cosine > 0.9
                        and best_other_centroid_cosine > 0.9
                        and pred_closer_to_other_than_gt == 1
                    ),
                }
            )
            output_records.append(out_record)

    group_summary = {}
    for group_name in args.groups:
        group_summary[group_name] = summarize_group([record for record in output_records if str(record["local_group"]) == group_name])

    summary = {
        "num_records": len(output_records),
        "exclude_topn": args.exclude_topn,
        "groups": group_summary,
        "notes": {
            "interpretation": (
                "If local failure groups are much more aligned with some other annotated target keypoint than local success, "
                "and the prediction/centroid are often closer to that other keypoint than to the GT, then the drift is not "
                "generic noise but part-to-part identity confusion."
            )
        },
    }

    output_csv = os.path.join(args.output_dir, "other_kp_attraction_records.csv")
    output_json = os.path.join(args.output_dir, "other_kp_attraction_summary.json")
    write_records_csv(output_records, output_csv)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved records to: {output_csv}")
    print(f"Saved summary to: {output_json}")
    for group_name, stats in group_summary.items():
        print(
            f"{group_name}: count={stats['count']} "
            f"mean_best_other_error_cosine={stats['mean_best_other_error_cosine']} "
            f"pred_closer_to_other_than_gt_rate={stats['pred_closer_to_other_than_gt_rate']} "
            f"joint_attraction_signature_rate={stats['joint_attraction_signature_rate']}"
        )


if __name__ == "__main__":
    main()

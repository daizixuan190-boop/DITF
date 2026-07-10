import argparse
import csv
import json
import math
import os
import random
from collections import defaultdict
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from analyze_spair_midrank_suppression import extract_topk_candidates_hr
from analyze_spair_token_residuals import build_post_feature, ensure_dir, sample_feature_at_pixel
from estimate_spair_matching_ceiling import maybe_apply_post_calibration


LOCAL_GROUPS = [
    "local_success",
    "local_fail_rank_11_50",
    "local_fail_rank_51_100",
    "local_fail_rank_101_500",
]

EXCLUDE_TOPN_CHOICES = [0, 1, 5]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Validate whether local candidate-cloud mis-centering directionally drives prediction errors on SPair-71k. "
            "For each selected local-neighborhood point, the script recomputes the top-k candidate cloud, estimates its centroid, "
            "and measures whether the top-1 prediction error points in the same direction as the centroid shift away from GT."
        )
    )
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to SPair-71k root.")
    parser.add_argument("--feature_path", type=str, required=True, help="Path to saved per-category features.")
    parser.add_argument("--records_csv", type=str, required=True, help="Path to local_identity_barrier_records.csv.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs.")
    parser.add_argument("--device", type=str, default="cuda", help="Analysis device, e.g. cuda or cpu.")
    parser.add_argument("--feature_dim", type=int, default=3072, help="Feature dimension.")
    parser.add_argument("--discard_channels", nargs="+", type=int, default=[154, 1446], help="Channels to discard for Flux.")
    parser.add_argument("--cd", action="store_true", default=False, help="Apply channel discard as in eval_spair.py.")
    parser.add_argument("--tile_rows", type=int, default=128, help="Row chunk size for GPU candidate extraction.")
    parser.add_argument("--topk_candidates", type=int, default=50, help="Number of local candidates used to define the cloud centroid.")
    parser.add_argument(
        "--exclude_topn_list",
        nargs="+",
        type=int,
        default=EXCLUDE_TOPN_CHOICES,
        help="Evaluate centroid alignment after excluding the first N ranked candidates from the centroid computation.",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=LOCAL_GROUPS,
        help="Local groups to analyze.",
    )
    parser.add_argument(
        "--max_records_per_group",
        type=int,
        default=0,
        help="Optional deterministic cap per group for faster runs. 0 means full group.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed used when sub-sampling records.")
    parser.add_argument("--flush_every_pairs", type=int, default=10, help="Write partial outputs every N pairs.")
    parser.add_argument("--shift_calibration", action="store_true", default=False, help="Apply shift-only post-AdaLN calibration before analysis.")
    parser.add_argument("--shift_calibration_quantile", default=0.75, type=float, help="Only suppress tokens above this shift-ratio quantile.")
    parser.add_argument("--shift_calibration_strength", default=0.5, type=float, help="Suppression strength for high-shift tokens.")
    parser.add_argument("--shift_calibration_min_lambda", default=0.2, type=float, help="Minimum retained shift scaling after calibration.")
    parser.add_argument("--joint_calibration", action="store_true", default=False, help="Apply joint high-shift low-content calibration before analysis.")
    parser.add_argument("--joint_shift_quantile", default=0.75, type=float, help="High-shift token threshold for joint calibration.")
    parser.add_argument("--joint_content_quantile", default=0.25, type=float, help="Low-content token threshold for joint calibration.")
    parser.add_argument("--joint_shift_strength", default=0.5, type=float, help="Shift suppression strength under joint calibration.")
    parser.add_argument("--joint_min_shift_lambda", default=0.2, type=float, help="Minimum retained shift scaling under joint calibration.")
    parser.add_argument("--joint_content_strength", default=0.25, type=float, help="Content amplification strength under joint calibration.")
    parser.add_argument("--joint_max_content_gain", default=1.5, type=float, help="Upper bound for content amplification under joint calibration.")
    return parser.parse_args()


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


def assign_local_group(record: dict[str, Any]) -> str | None:
    if int(record["local_neighborhood"]) != 1:
        return None
    current_error = int(record["current_error"])
    oracle_rank = int(record["oracle_best_rank"])
    if current_error == 0:
        return "local_success"
    if 11 <= oracle_rank <= 50:
        return "local_fail_rank_11_50"
    if 51 <= oracle_rank <= 100:
        return "local_fail_rank_51_100"
    if 101 <= oracle_rank <= 500:
        return "local_fail_rank_101_500"
    return None


def select_records(records: list[dict[str, Any]], groups: list[str], max_per_group: int, seed: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        group = assign_local_group(record)
        if group is None or group not in groups:
            continue
        record = dict(record)
        record["local_group"] = group
        grouped[group].append(record)

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for group in groups:
        subset = grouped.get(group, [])
        if max_per_group > 0 and len(subset) > max_per_group:
            subset = rng.sample(subset, max_per_group)
        selected.extend(subset)
    return selected


def safe_cosine(dx1: float, dy1: float, dx2: float, dy2: float) -> float | None:
    norm1 = math.sqrt(dx1 * dx1 + dy1 * dy1)
    norm2 = math.sqrt(dx2 * dx2 + dy2 * dy2)
    if norm1 <= 1e-9 or norm2 <= 1e-9:
        return None
    return float((dx1 * dx2 + dy1 * dy2) / (norm1 * norm2))


def summarize_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {"count": len(records), "exclude_topn": {}}
    exclude_values = sorted({int(record["exclude_topn"]) for record in records}) if records else []
    for exclude_topn in exclude_values:
        subset = [record for record in records if int(record["exclude_topn"]) == exclude_topn]
        cosines = [float(record["error_centroid_cosine"]) for record in subset if record["error_centroid_cosine"] is not None]
        summary["exclude_topn"][str(exclude_topn)] = {
            "count": len(subset),
            "mean_centroid_norm_dist": float(np.mean([float(record["centroid_norm_dist"]) for record in subset])) if subset else None,
            "mean_error_norm_dist": float(np.mean([float(record["error_norm_dist"]) for record in subset])) if subset else None,
            "mean_pred_centroid_norm_dist": float(np.mean([float(record["pred_centroid_norm_dist"]) for record in subset])) if subset else None,
            "mean_error_centroid_cosine": float(np.mean(cosines)) if cosines else None,
            "frac_error_centroid_cosine_gt_0": float(np.mean([c > 0.0 for c in cosines])) if cosines else None,
            "frac_error_centroid_cosine_gt_0_5": float(np.mean([c > 0.5 for c in cosines])) if cosines else None,
            "frac_error_centroid_cosine_gt_0_8": float(np.mean([c > 0.8 for c in cosines])) if cosines else None,
        }
    return summary


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    print(f"Using analysis device: {device}")

    input_records = load_csv(args.records_csv)
    selected_records = select_records(input_records, args.groups, args.max_records_per_group, args.seed)
    if not selected_records:
        raise RuntimeError("No selected records found. Check groups and records_csv.")

    records_by_pair: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in selected_records:
        records_by_pair[(str(record["category"]), str(record["pair_name"]))].append(record)

    pre_norm = torch.nn.LayerNorm(args.feature_dim, elementwise_affine=False, eps=1e-6)
    test_path = os.path.join(args.dataset_path, "PairAnnotation", "test")
    output_records: list[dict[str, Any]] = []
    processed_pairs = 0

    output_csv = os.path.join(args.output_dir, "centroid_alignment_records.csv")
    output_json = os.path.join(args.output_dir, "centroid_alignment_summary.json")

    for (category, pair_name), pair_records in records_by_pair.items():
        feat_file = os.path.join(args.feature_path, f"{category}.pth")
        ada_file = os.path.join(args.feature_path, f"{category}_ada.pth")
        if not os.path.exists(feat_file) or not os.path.exists(ada_file):
            continue

        output_dict = torch.load(feat_file, map_location="cpu", weights_only=True)
        ada_dict = torch.load(ada_file, map_location="cpu", weights_only=True)

        with open(os.path.join(test_path, pair_name), "r", encoding="utf-8") as f:
            data = json.load(f)

        src_imname = data["src_imname"]
        trg_imname = data["trg_imname"]

        _, src_ft_ln, _, src_shift, src_scale = build_post_feature(
            output_dict[src_imname].float(),
            ada_dict[src_imname].float(),
            pre_norm,
            args.discard_channels,
            args.cd,
        )
        _, trg_ft_ln, _, trg_shift, trg_scale = build_post_feature(
            output_dict[trg_imname].float(),
            ada_dict[trg_imname].float(),
            pre_norm,
            args.discard_channels,
            args.cd,
        )
        src_ft_post = maybe_apply_post_calibration(src_ft_ln, src_shift, src_scale, args).float().to(device)
        trg_ft_post = maybe_apply_post_calibration(trg_ft_ln, trg_shift, trg_scale, args).float().to(device)

        src_eval_h, src_eval_w = data["src_imsize"][:2][::-1]
        trg_eval_h, trg_eval_w = data["trg_imsize"][:2][::-1]
        trg_bndbox = data["trg_bndbox"]
        threshold = max(trg_bndbox[3] - trg_bndbox[1], trg_bndbox[2] - trg_bndbox[0])

        for record in pair_records:
            src_x = int(record["src_x"])
            src_y = int(record["src_y"])
            trg_x = int(record["trg_x"])
            trg_y = int(record["trg_y"])
            pred_x = int(record["pred_x"])
            pred_y = int(record["pred_y"])

            src_vec = sample_feature_at_pixel(src_ft_post, src_x, src_y, src_eval_h, src_eval_w)
            candidate_vals, candidate_idx = extract_topk_candidates_hr(
                src_vec,
                trg_ft_post,
                trg_eval_h,
                trg_eval_w,
                args.tile_rows,
                args.topk_candidates,
            )

            valid = candidate_idx >= 0
            candidate_idx = candidate_idx[valid].long().cpu().numpy()
            if candidate_idx.size == 0:
                continue

            error_dx = float(pred_x - trg_x)
            error_dy = float(pred_y - trg_y)
            error_norm_dist = float(math.sqrt(error_dx * error_dx + error_dy * error_dy) / max(float(threshold), 1e-6))

            for exclude_topn in args.exclude_topn_list:
                exclude_topn = int(exclude_topn)
                if exclude_topn < 0 or exclude_topn >= candidate_idx.size:
                    continue
                kept_idx = candidate_idx[exclude_topn:]
                if kept_idx.size == 0:
                    continue

                xs = (kept_idx % trg_eval_w).astype(np.float64)
                ys = (kept_idx // trg_eval_w).astype(np.float64)
                centroid_x = float(np.mean(xs))
                centroid_y = float(np.mean(ys))

                centroid_dx = centroid_x - float(trg_x)
                centroid_dy = centroid_y - float(trg_y)
                pred_centroid_dx = float(pred_x) - centroid_x
                pred_centroid_dy = float(pred_y) - centroid_y

                centroid_norm_dist = float(
                    math.sqrt(centroid_dx * centroid_dx + centroid_dy * centroid_dy) / max(float(threshold), 1e-6)
                )
                pred_centroid_norm_dist = float(
                    math.sqrt(pred_centroid_dx * pred_centroid_dx + pred_centroid_dy * pred_centroid_dy)
                    / max(float(threshold), 1e-6)
                )
                error_centroid_cosine = safe_cosine(error_dx, error_dy, centroid_dx, centroid_dy)

                out_record = dict(record)
                out_record.update(
                    {
                        "exclude_topn": exclude_topn,
                        "centroid_x_topk": centroid_x,
                        "centroid_y_topk": centroid_y,
                        "centroid_dx": centroid_dx,
                        "centroid_dy": centroid_dy,
                        "centroid_norm_dist": centroid_norm_dist,
                        "error_dx": error_dx,
                        "error_dy": error_dy,
                        "error_norm_dist": error_norm_dist,
                        "pred_centroid_norm_dist": pred_centroid_norm_dist,
                        "error_centroid_cosine": error_centroid_cosine,
                        "same_half_plane": None if error_centroid_cosine is None else int(error_centroid_cosine > 0.0),
                    }
                )
                output_records.append(out_record)

        processed_pairs += 1
        if processed_pairs % args.flush_every_pairs == 0:
            write_records_csv(output_records, output_csv)
            print(f"[Flush] processed_pairs={processed_pairs} processed_points={len(output_records)}")

    write_records_csv(output_records, output_csv)

    group_summary = {}
    for group in args.groups:
        subset = [record for record in output_records if record["local_group"] == group]
        group_summary[group] = summarize_group(subset)

    summary = {
        "num_selected_records": len(selected_records),
        "num_analyzed_records": len(output_records),
        "groups": group_summary,
        "notes": {
            "interpretation": (
                "If local-failure groups show a much larger centroid shift than local_success, and the prediction error vector "
                "is strongly aligned with the centroid shift vector, then candidate-cloud mis-centering is not just correlated "
                "with failure; it is directionally consistent with the observed top-1 error."
            )
        },
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved records to: {output_csv}")
    print(f"Saved summary to: {output_json}")
    print(f"Num analyzed records: {summary['num_analyzed_records']}")
    for group, stats in group_summary.items():
        print(f"{group}: count={stats['count']}")
        for exclude_topn, substats in stats["exclude_topn"].items():
            print(
                f"  exclude_topn={exclude_topn} "
                f"mean_centroid_norm_dist={substats['mean_centroid_norm_dist']} "
                f"mean_error_centroid_cosine={substats['mean_error_centroid_cosine']} "
                f"frac_cos_gt_0={substats['frac_error_centroid_cosine_gt_0']}"
            )


if __name__ == "__main__":
    main()

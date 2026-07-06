import argparse
import csv
import json
import math
import os
from collections import defaultdict
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from analyze_spair_token_residuals import build_post_feature, ensure_dir, sample_feature_at_pixel
from estimate_spair_matching_ceiling import make_grid_for_output_window, maybe_apply_post_calibration


MERGE_KEYS = [
    "category",
    "pair_name",
    "src_imname",
    "trg_imname",
    "kp_idx",
]

RANK_BUCKETS = [
    ("rank_2_10", 2, 10),
    ("rank_11_50", 11, 50),
    ("rank_51_100", 51, 100),
    ("rank_101_500", 101, 500),
    ("rank_gt_500", 501, None),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze mid-rank suppression on SPair-71k: why GT often stays in top-50~500 rather than entering the front ranks."
        )
    )
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to SPair-71k root.")
    parser.add_argument("--feature_path", type=str, required=True, help="Path to saved per-category features.")
    parser.add_argument("--residual_csv", type=str, required=True, help="Path to per_point_records.csv.")
    parser.add_argument("--ceiling_csv", type=str, required=True, help="Path to matching_ceiling_records.csv.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs.")
    parser.add_argument("--device", type=str, default="cuda", help="Device for candidate extraction, e.g. cuda or cpu.")
    parser.add_argument("--feature_dim", type=int, default=3072, help="Feature dimension.")
    parser.add_argument("--discard_channels", nargs="+", type=int, default=[154, 1446], help="Channels to discard for Flux.")
    parser.add_argument("--cd", action="store_true", default=False, help="Apply channel discard as in eval_spair.py.")
    parser.add_argument("--tile_rows", type=int, default=128, help="Row chunk size for GPU candidate extraction.")
    parser.add_argument("--topk_candidates", type=int, default=500, help="Number of top candidates to keep per point.")
    parser.add_argument("--near_radius_multipliers", nargs="+", type=float, default=[1.0, 2.0, 4.0], help="PCK-threshold multiples used to measure local vs remote competition.")
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


def merge_support_maps(
    residual_records: list[dict[str, Any]],
    ceiling_records: list[dict[str, Any]],
) -> tuple[dict[tuple[Any, ...], dict[str, Any]], dict[tuple[str, str, str], list[int]]]:
    residual_map = {
        tuple(record[key] for key in MERGE_KEYS): record
        for record in residual_records
    }
    ceiling_map = {
        tuple(record[key] for key in MERGE_KEYS): record
        for record in ceiling_records
    }
    merged_map: dict[tuple[Any, ...], dict[str, Any]] = {}
    pair_to_kps: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for merge_id, ceiling in ceiling_map.items():
        residual = residual_map.get(merge_id)
        if residual is None:
            continue
        record = dict(residual)
        record["oracle_best_rank"] = int(ceiling["oracle_best_rank"])
        record["oracle_best_rank_frac"] = float(ceiling["oracle_best_rank_frac"])
        record["current_error"] = 1 - int(record["correct"])
        merged_map[merge_id] = record
        pair_to_kps[(record["category"], record["pair_name"], record["src_imname"])] .append(int(record["kp_idx"]))
    return merged_map, pair_to_kps


def extract_topk_candidates_hr(
    src_vec: torch.Tensor,
    trg_feat: torch.Tensor,
    out_h: int,
    out_w: int,
    tile_rows: int,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    src_vec = F.normalize(src_vec.view(1, -1).float(), dim=1)[0]
    global_vals = torch.full((topk,), -1e9, dtype=torch.float32, device=trg_feat.device)
    global_idx = torch.full((topk,), -1, dtype=torch.long, device=trg_feat.device)
    trg_feat = trg_feat.float()

    for y_start in range(0, out_h, tile_rows):
        y_end = min(y_start + tile_rows, out_h)
        grid = make_grid_for_output_window(0, out_w, y_start, y_end, out_h, out_w, trg_feat.device)
        tile_feat = F.grid_sample(trg_feat, grid, mode="bilinear", align_corners=False)
        tile_feat = F.normalize(tile_feat, dim=1)[0]  # C, tile_h, out_w
        tile_scores = torch.einsum("c,chw->hw", src_vec, tile_feat).reshape(-1)
        local_k = min(topk, int(tile_scores.numel()))
        local_vals, local_idx = torch.topk(tile_scores, k=local_k)
        local_idx = local_idx + y_start * out_w

        merged_vals = torch.cat([global_vals, local_vals], dim=0)
        merged_idx = torch.cat([global_idx, local_idx], dim=0)
        keep_vals, keep_pos = torch.topk(merged_vals, k=topk)
        global_vals = keep_vals
        global_idx = merged_idx[keep_pos]

    return global_vals, global_idx


def candidate_metrics(
    candidate_vals: torch.Tensor,
    candidate_idx: torch.Tensor,
    trg_x: int,
    trg_y: int,
    out_w: int,
    threshold: float,
    near_radius_multipliers: list[float],
) -> dict[str, float]:
    valid = candidate_idx >= 0
    candidate_vals = candidate_vals[valid].float().cpu().numpy()
    candidate_idx = candidate_idx[valid].long().cpu().numpy()
    if candidate_idx.size == 0:
        return {}

    xs = (candidate_idx % out_w).astype(np.int64)
    ys = (candidate_idx // out_w).astype(np.int64)
    dists = np.sqrt((xs - int(trg_x)) ** 2 + (ys - int(trg_y)) ** 2)
    norm_dists = dists / max(float(threshold), 1e-6)

    metrics: dict[str, float] = {
        "best_wrong_norm_dist": float(norm_dists[0]),
        "top10_mean_norm_dist": float(np.mean(norm_dists[: min(10, len(norm_dists))])),
        "top50_mean_norm_dist": float(np.mean(norm_dists[: min(50, len(norm_dists))])),
        "top100_mean_norm_dist": float(np.mean(norm_dists[: min(100, len(norm_dists))])),
    }

    for topk in [10, 50, 100, 500]:
        k = min(topk, len(norm_dists))
        subset = norm_dists[:k]
        cand_x = xs[:k].astype(np.float64)
        cand_y = ys[:k].astype(np.float64)
        centroid_x = float(np.mean(cand_x))
        centroid_y = float(np.mean(cand_y))
        centroid_dist = math.sqrt((centroid_x - float(trg_x)) ** 2 + (centroid_y - float(trg_y)) ** 2)
        spread = float(np.mean(np.sqrt((cand_x - centroid_x) ** 2 + (cand_y - centroid_y) ** 2)))
        metrics[f"top{topk}_centroid_norm_dist"] = float(centroid_dist / max(float(threshold), 1e-6))
        metrics[f"top{topk}_spread_norm"] = float(spread / max(float(threshold), 1e-6))
        for mult in near_radius_multipliers:
            metrics[f"top{topk}_near_frac@x{mult:g}"] = float(np.mean(subset <= mult))

    if len(candidate_vals) > 1:
        metrics["top1_top5_score_gap"] = float(candidate_vals[0] - np.mean(candidate_vals[: min(5, len(candidate_vals))]))
        metrics["top1_top50_score_gap"] = float(candidate_vals[0] - np.mean(candidate_vals[: min(50, len(candidate_vals))]))
    else:
        metrics["top1_top5_score_gap"] = 0.0
        metrics["top1_top50_score_gap"] = 0.0
    return metrics


def summarize_rank_buckets(records: list[dict[str, Any]]) -> dict[str, Any]:
    failures = [record for record in records if int(record["current_error"]) == 1]
    summary: dict[str, Any] = {
        "num_points": len(records),
        "num_failures": len(failures),
        "failure_rate": float(np.mean([float(record["current_error"]) for record in records])) if records else None,
        "rank_buckets": {},
        "notes": {
            "purpose": (
                "Middle-rank suppression analysis: determine whether GT is mainly held back by local near competition, "
                "remote structured competition, or diffuse weak discrimination."
            )
        },
    }

    metric_keys = [
        "norm_dist",
        "sim_margin",
        "sim_entropy",
        "best_wrong_norm_dist",
        "top10_mean_norm_dist",
        "top50_mean_norm_dist",
        "top50_centroid_norm_dist",
        "top50_spread_norm",
        "top100_centroid_norm_dist",
        "top100_spread_norm",
        "top500_centroid_norm_dist",
        "top500_spread_norm",
        "top10_near_frac@x1",
        "top10_near_frac@x2",
        "top50_near_frac@x1",
        "top50_near_frac@x2",
        "top50_near_frac@x4",
        "top100_near_frac@x2",
        "top500_near_frac@x2",
        "top1_top5_score_gap",
        "top1_top50_score_gap",
    ]

    for bucket_name, low, high in RANK_BUCKETS:
        if high is None:
            subset = [record for record in failures if int(record["oracle_best_rank"]) >= low]
        else:
            subset = [record for record in failures if low <= int(record["oracle_best_rank"]) <= high]
        if not subset:
            summary["rank_buckets"][bucket_name] = {"count": 0}
            continue
        bucket_summary = {
            "count": len(subset),
            "fraction_of_failures": float(len(subset) / max(len(failures), 1)),
            "mean_oracle_rank": float(np.mean([float(record["oracle_best_rank"]) for record in subset])),
        }
        for key in metric_keys:
            values = [float(record[key]) for record in subset if key in record and record[key] is not None]
            bucket_summary[key] = float(np.mean(values)) if values else None
        summary["rank_buckets"][bucket_name] = bucket_summary
    return summary


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    print(f"Using analysis device: {device}")

    residual_records = load_csv(args.residual_csv)
    ceiling_records = load_csv(args.ceiling_csv)
    merged_map, pair_to_kps = merge_support_maps(residual_records, ceiling_records)

    test_path = os.path.join(args.dataset_path, "PairAnnotation", "test")
    pre_norm = torch.nn.LayerNorm(args.feature_dim, elementwise_affine=False, eps=1e-6)
    all_records: list[dict[str, Any]] = []
    processed_pairs = 0

    records_csv = os.path.join(args.output_dir, "midrank_suppression_records.csv")
    summary_json = os.path.join(args.output_dir, "midrank_suppression_summary.json")
    progress_log = os.path.join(args.output_dir, "progress.log")

    pairs_by_category: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for key in pair_to_kps.keys():
        category, pair_name, src_imname = key
        pairs_by_category[category].append((category, pair_name, src_imname))

    for category, pair_items in pairs_by_category.items():
        feat_file = os.path.join(args.feature_path, f"{category}.pth")
        ada_file = os.path.join(args.feature_path, f"{category}_ada.pth")
        if not os.path.exists(feat_file) or not os.path.exists(ada_file):
            continue

        output_dict = torch.load(feat_file, map_location="cpu", weights_only=True)
        ada_dict = torch.load(ada_file, map_location="cpu", weights_only=True)

        print(f"[Category] {category}: {len(pair_items)} pairs")
        for _, pair_name, _ in pair_items:
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

            kp_indices = sorted(pair_to_kps[(category, pair_name, src_imname)])
            for kp_idx in kp_indices:
                src_point = data["src_kps"][kp_idx]
                trg_point = data["trg_kps"][kp_idx]
                src_x, src_y = int(src_point[0]), int(src_point[1])
                trg_x, trg_y = int(trg_point[0]), int(trg_point[1])

                merge_id = (category, pair_name, src_imname, trg_imname, kp_idx)
                base_record = dict(merged_map[merge_id])

                src_vec = sample_feature_at_pixel(src_ft_post, src_x, src_y, src_eval_h, src_eval_w)
                candidate_vals, candidate_idx = extract_topk_candidates_hr(
                    src_vec,
                    trg_ft_post,
                    trg_eval_h,
                    trg_eval_w,
                    args.tile_rows,
                    args.topk_candidates,
                )
                metrics = candidate_metrics(
                    candidate_vals,
                    candidate_idx,
                    trg_x,
                    trg_y,
                    trg_eval_w,
                    threshold,
                    args.near_radius_multipliers,
                )
                base_record.update(metrics)
                all_records.append(base_record)

            processed_pairs += 1
            if processed_pairs % args.flush_every_pairs == 0:
                write_records_csv(all_records, records_csv)
                summary = summarize_rank_buckets(all_records)
                with open(summary_json, "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)
                with open(progress_log, "a", encoding="utf-8") as f:
                    f.write(
                        f"processed_pairs={processed_pairs}, num_points={summary['num_points']}, num_failures={summary['num_failures']}\n"
                    )
                print(
                    f"[Flush] pairs={processed_pairs} points={summary['num_points']} failures={summary['num_failures']}"
                )

    write_records_csv(all_records, records_csv)
    summary = summarize_rank_buckets(all_records)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved records to: {records_csv}")
    print(f"Saved summary to: {summary_json}")
    print(f"Num points: {summary['num_points']}")
    print(f"Num failures: {summary['num_failures']}")
    for bucket_name, bucket_summary in summary["rank_buckets"].items():
        print(f"{bucket_name}: {bucket_summary.get('count', 0)}")


if __name__ == "__main__":
    main()

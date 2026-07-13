import argparse
import csv
import json
import math
import os
from collections import defaultdict
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from analyze_spair_token_residuals import build_post_feature, ensure_dir
from estimate_spair_matching_ceiling import maybe_apply_post_calibration


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
            "Analyze whether SPair residual failures are already caused by raw feature competition "
            "between the GT part and other annotated real parts, and how local context changes that competition."
        )
    )
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to SPair-71k root.")
    parser.add_argument("--feature_path", type=str, required=True, help="Path to saved per-category features.")
    parser.add_argument("--residual_csv", type=str, required=True, help="Path to per-point records CSV.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs.")
    parser.add_argument("--device", type=str, default="cuda", help="Analysis device, e.g. cuda or cpu.")
    parser.add_argument("--feature_dim", type=int, default=3072, help="Feature dimension.")
    parser.add_argument("--discard_channels", nargs="+", type=int, default=[154, 1446], help="Channels to discard for Flux.")
    parser.add_argument("--cd", action="store_true", default=False, help="Apply channel discard as in eval_spair.py.")
    parser.add_argument("--max_pairs_per_cat", type=int, default=0, help="Optional cap per category based on residual_csv order.")
    parser.add_argument("--context_radii", nargs="+", type=int, default=[0, 1, 2, 4], help="Local pooling radii used to measure context effects.")
    parser.add_argument("--compare_only_failures", action="store_true", default=False, help="Only analyze current failure points from residual_csv.")
    parser.add_argument("--tag_csv", type=str, default="", help="Optional tag CSV for subgroup analysis.")
    parser.add_argument("--tag_column", type=str, default="", help="Column from tag_csv to summarize.")
    parser.add_argument("--tag_values", nargs="*", default=[], help="Optional tag values to keep/summarize.")
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


def get_pair_scalar_fields(data: dict[str, Any]) -> dict[str, Any]:
    scalar_fields = {}
    for key, value in data.items():
        if isinstance(value, (int, float, bool, str)):
            scalar_fields[key] = value
    return scalar_fields


def resolve_pair_json_path(test_path: str, pair_name: str) -> str:
    direct_path = os.path.join(test_path, pair_name)
    if os.path.exists(direct_path):
        return direct_path
    json_path = os.path.join(test_path, f"{pair_name}.json")
    if os.path.exists(json_path):
        return json_path
    raise FileNotFoundError(f"Missing pair annotation for pair_name={pair_name}")


def merge_tags(records: list[dict[str, Any]], tag_records: list[dict[str, Any]], tag_column: str):
    tag_map = {tuple(record[key] for key in MERGE_KEYS): record for record in tag_records}
    observed_values = set()
    for record in records:
        merge_id = tuple(record[key] for key in MERGE_KEYS)
        tag_record = tag_map.get(merge_id)
        tag_value = None if tag_record is None else tag_record.get(tag_column)
        record[tag_column] = tag_value
        if tag_value is not None and tag_value != "":
            observed_values.add(str(tag_value))
    return sorted(observed_values)


def filter_records(records: list[dict[str, Any]], args) -> list[dict[str, Any]]:
    if args.compare_only_failures:
        records = [record for record in records if int(record["correct"]) == 0]
    if args.max_pairs_per_cat > 0:
        by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            by_cat[str(record["category"])].append(record)
        limited = []
        for category, subset in by_cat.items():
            pair_seen = []
            kept_pairs = set()
            for record in subset:
                pair_name = str(record["pair_name"])
                if pair_name not in kept_pairs:
                    if len(pair_seen) >= args.max_pairs_per_cat:
                        continue
                    pair_seen.append(pair_name)
                    kept_pairs.add(pair_name)
                limited.append(record)
        records = limited
    return records


def normalize_rows(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=1)


def patch_descriptor(feat: torch.Tensor, x: int, y: int, radius: int) -> torch.Tensor:
    _, channels, height, width = feat.shape
    x0 = max(int(x) - radius, 0)
    x1 = min(int(x) + radius + 1, width)
    y0 = max(int(y) - radius, 0)
    y1 = min(int(y) + radius + 1, height)
    patch = feat[0, :, y0:y1, x0:x1].reshape(channels, -1)
    desc = patch.mean(dim=1, keepdim=False).view(1, channels)
    return normalize_rows(desc)[0]


def batch_patch_descriptors(feat: torch.Tensor, points: list[list[int]], radius: int) -> torch.Tensor:
    descriptors = [patch_descriptor(feat, int(point[0]), int(point[1]), radius) for point in points]
    return torch.stack(descriptors, dim=0)


def rank_of_index(scores: torch.Tensor, index: int) -> int:
    gt_score = float(scores[index].item())
    return 1 + int(torch.sum(scores > gt_score).item())


def best_other_index(scores: torch.Tensor, gt_index: int) -> int:
    mask = torch.ones(scores.numel(), dtype=torch.bool, device=scores.device)
    mask[gt_index] = False
    other_scores = scores.masked_fill(~mask, -1e9)
    return int(torch.argmax(other_scores).item())


def safe_rate(values: list[int]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def summarize_subset(records: list[dict[str, Any]], radii: list[int]) -> dict[str, Any]:
    if not records:
        return {"count": 0}

    summary = {
        "count": len(records),
        "error_rate": safe_rate([1 - int(record["correct"]) for record in records]),
        "gt_beaten_by_other_center_rate": safe_rate([int(record["gt_beaten_by_other_center"]) for record in records]),
        "mean_gt_rank_among_real_parts_center": safe_mean([float(record["gt_rank_among_real_parts_center"]) for record in records]),
        "mean_center_margin": safe_mean([float(record["center_margin"]) for record in records]),
        "mean_best_other_trg_norm_dist": safe_mean([float(record["best_other_trg_norm_dist"]) for record in records if record["best_other_trg_norm_dist"] is not None]),
        "mean_best_other_src_norm_dist": safe_mean([float(record["best_other_src_norm_dist"]) for record in records if record["best_other_src_norm_dist"] is not None]),
        "best_other_trg_close@x1_rate": safe_rate([int(record["best_other_trg_close@x1"]) for record in records if record["best_other_trg_close@x1"] is not None]),
        "best_other_trg_close@x2_rate": safe_rate([int(record["best_other_trg_close@x2"]) for record in records if record["best_other_trg_close@x2"] is not None]),
        "best_other_trg_close@x4_rate": safe_rate([int(record["best_other_trg_close@x4"]) for record in records if record["best_other_trg_close@x4"] is not None]),
        "pred_nearest_matches_best_other_rate": safe_rate([int(record["pred_nearest_matches_best_other"]) for record in records if record["pred_nearest_matches_best_other"] is not None]),
        "context": {},
    }

    for radius in radii:
        key = f"r{radius}"
        summary["context"][key] = {
            "gt_beaten_rate": safe_rate([int(record[f"gt_beaten_by_other_r{radius}"]) for record in records]),
            "mean_margin": safe_mean([float(record[f"margin_r{radius}"]) for record in records]),
            "mean_margin_gain_vs_center": safe_mean([float(record[f"margin_gain_r{radius}"]) for record in records]),
            "rank_improve_rate_vs_center": safe_rate([int(record[f"rank_improve_r{radius}"]) for record in records]),
            "rank_worsen_rate_vs_center": safe_rate([int(record[f"rank_worsen_r{radius}"]) for record in records]),
        }
    return summary


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    print(f"Using analysis device: {device}")

    residual_records = filter_records(load_csv(args.residual_csv), args)
    if not residual_records:
        raise RuntimeError("No residual records selected. Check residual_csv and filters.")

    observed_tag_values = []
    if args.tag_csv and args.tag_column:
        observed_tag_values = merge_tags(residual_records, load_csv(args.tag_csv), args.tag_column)

    records_by_pair: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in residual_records:
        records_by_pair[(str(record["category"]), str(record["pair_name"]))].append(record)

    pre_norm = nn.LayerNorm(args.feature_dim, elementwise_affine=False, eps=1e-6)
    test_path = os.path.join(args.dataset_path, "PairAnnotation", "test")
    output_records: list[dict[str, Any]] = []

    output_csv = os.path.join(args.output_dir, "real_part_competition_records.csv")
    output_json = os.path.join(args.output_dir, "real_part_competition_summary.json")

    processed_pairs = 0
    for (category, pair_name), pair_records in records_by_pair.items():
        feat_file = os.path.join(args.feature_path, f"{category}.pth")
        ada_file = os.path.join(args.feature_path, f"{category}_ada.pth")
        if not os.path.exists(feat_file) or not os.path.exists(ada_file):
            continue

        output_dict = torch.load(feat_file, map_location="cpu", weights_only=True)
        ada_dict = torch.load(ada_file, map_location="cpu", weights_only=True)
        pair_json_path = resolve_pair_json_path(test_path, pair_name)
        with open(pair_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        src_imname = data["src_imname"]
        trg_imname = data["trg_imname"]
        src_img_size = data["src_imsize"][:2][::-1]
        trg_img_size = data["trg_imsize"][:2][::-1]

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

        src_ft = nn.Upsample(size=src_img_size, mode="bilinear")(src_ft_post)
        trg_ft = nn.Upsample(size=trg_img_size, mode="bilinear")(trg_ft_post)

        src_kps = data["src_kps"]
        trg_kps = data["trg_kps"]
        if len(src_kps) != len(trg_kps):
            continue

        src_bndbox = data["src_bndbox"]
        trg_bndbox = data["trg_bndbox"]
        src_threshold = max(src_bndbox[3] - src_bndbox[1], src_bndbox[2] - src_bndbox[0])
        trg_threshold = max(trg_bndbox[3] - trg_bndbox[1], trg_bndbox[2] - trg_bndbox[0])
        pair_scalars = get_pair_scalar_fields(data)

        target_desc_by_radius = {
            radius: batch_patch_descriptors(trg_ft, trg_kps, radius)
            for radius in args.context_radii
        }

        for record in pair_records:
            kp_idx = int(record["kp_idx"])
            src_point = src_kps[kp_idx]
            trg_point = trg_kps[kp_idx]

            source_desc_by_radius = {
                radius: patch_descriptor(src_ft, int(src_point[0]), int(src_point[1]), radius)
                for radius in args.context_radii
            }

            center_scores = torch.mv(target_desc_by_radius[0], source_desc_by_radius[0])
            gt_center_score = float(center_scores[kp_idx].item())
            rival_idx_center = best_other_index(center_scores, kp_idx)
            rival_center_score = float(center_scores[rival_idx_center].item())
            gt_rank_center = rank_of_index(center_scores, kp_idx)

            pred_x = int(record["pred_x"]) if "pred_x" in record and record["pred_x"] is not None else None
            pred_y = int(record["pred_y"]) if "pred_y" in record and record["pred_y"] is not None else None
            pred_nearest_idx = None
            if pred_x is not None and pred_y is not None:
                pred_dists = [
                    (pred_x - int(point[0])) ** 2 + (pred_y - int(point[1])) ** 2
                    for point in trg_kps
                ]
                pred_nearest_idx = int(np.argmin(np.asarray(pred_dists, dtype=np.float64)))

            best_other_trg_dx = float(trg_kps[rival_idx_center][0] - trg_point[0])
            best_other_trg_dy = float(trg_kps[rival_idx_center][1] - trg_point[1])
            best_other_src_dx = float(src_kps[rival_idx_center][0] - src_point[0])
            best_other_src_dy = float(src_kps[rival_idx_center][1] - src_point[1])
            best_other_trg_norm_dist = math.sqrt(best_other_trg_dx * best_other_trg_dx + best_other_trg_dy * best_other_trg_dy) / max(float(trg_threshold), 1e-6)
            best_other_src_norm_dist = math.sqrt(best_other_src_dx * best_other_src_dx + best_other_src_dy * best_other_src_dy) / max(float(src_threshold), 1e-6)

            out_record = dict(record)
            out_record.update(
                {
                    "gt_center_score": gt_center_score,
                    "best_other_idx_center": rival_idx_center,
                    "best_other_center_score": rival_center_score,
                    "center_margin": gt_center_score - rival_center_score,
                    "gt_rank_among_real_parts_center": gt_rank_center,
                    "gt_beaten_by_other_center": int(rival_center_score > gt_center_score),
                    "best_other_trg_norm_dist": best_other_trg_norm_dist,
                    "best_other_src_norm_dist": best_other_src_norm_dist,
                    "best_other_trg_close@x1": int(best_other_trg_norm_dist <= 1.0),
                    "best_other_trg_close@x2": int(best_other_trg_norm_dist <= 2.0),
                    "best_other_trg_close@x4": int(best_other_trg_norm_dist <= 4.0),
                    "pred_nearest_real_idx": pred_nearest_idx,
                    "pred_nearest_matches_best_other": None if pred_nearest_idx is None else int(pred_nearest_idx == rival_idx_center),
                }
            )

            for radius in args.context_radii:
                scores_r = torch.mv(target_desc_by_radius[radius], source_desc_by_radius[radius])
                gt_score_r = float(scores_r[kp_idx].item())
                rival_idx_r = best_other_index(scores_r, kp_idx)
                rival_score_r = float(scores_r[rival_idx_r].item())
                gt_rank_r = rank_of_index(scores_r, kp_idx)
                margin_r = gt_score_r - rival_score_r
                out_record[f"gt_score_r{radius}"] = gt_score_r
                out_record[f"best_other_idx_r{radius}"] = rival_idx_r
                out_record[f"best_other_score_r{radius}"] = rival_score_r
                out_record[f"margin_r{radius}"] = margin_r
                out_record[f"margin_gain_r{radius}"] = margin_r - (gt_center_score - rival_center_score)
                out_record[f"gt_rank_among_real_parts_r{radius}"] = gt_rank_r
                out_record[f"gt_beaten_by_other_r{radius}"] = int(rival_score_r > gt_score_r)
                out_record[f"rank_improve_r{radius}"] = int(gt_rank_r < gt_rank_center)
                out_record[f"rank_worsen_r{radius}"] = int(gt_rank_r > gt_rank_center)

            out_record.update(pair_scalars)
            output_records.append(out_record)

        processed_pairs += 1
        if processed_pairs % args.flush_every_pairs == 0:
            write_records_csv(output_records, output_csv)
            print(f"[Flush] pairs={processed_pairs} records={len(output_records)}")

    write_records_csv(output_records, output_csv)

    overall = summarize_subset(output_records, args.context_radii)
    failures = summarize_subset([record for record in output_records if int(record["correct"]) == 0], args.context_radii)
    successes = summarize_subset([record for record in output_records if int(record["correct"]) == 1], args.context_radii)
    by_tag = {}
    if args.tag_csv and args.tag_column:
        tag_values = args.tag_values if args.tag_values else observed_tag_values
        for value in tag_values:
            subset = [record for record in output_records if str(record.get(args.tag_column)) == str(value)]
            by_tag[str(value)] = summarize_subset(subset, args.context_radii)

    summary = {
        "num_records": len(output_records),
        "context_radii": args.context_radii,
        "overall": overall,
        "failures": failures,
        "successes": successes,
        "tag_column": args.tag_column if args.tag_csv else None,
        "by_tag": by_tag,
        "notes": {
            "interpretation": (
                "If GT is already frequently beaten by another annotated real part under raw frozen feature similarity, "
                "then the bottleneck is not mainly downstream selection but part-level feature discriminability. "
                "If larger local pooling systematically worsens GT-vs-rival margin, context is likely polluting identity; "
                "if it improves margin, local context may help resolve identity."
            )
        },
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved records to: {output_csv}")
    print(f"Saved summary to: {output_json}")
    print(
        f"Overall: count={overall['count']} "
        f"gt_beaten_by_other_center_rate={overall['gt_beaten_by_other_center_rate']} "
        f"mean_center_margin={overall['mean_center_margin']}"
    )
    print(
        f"Failures: count={failures['count']} "
        f"gt_beaten_by_other_center_rate={failures['gt_beaten_by_other_center_rate']} "
        f"mean_center_margin={failures['mean_center_margin']}"
    )
    for radius in args.context_radii:
        key = f"r{radius}"
        stats = failures["context"].get(key, {})
        print(
            f"Failure context {key}: "
            f"gt_beaten_rate={stats.get('gt_beaten_rate')} "
            f"mean_margin_gain_vs_center={stats.get('mean_margin_gain_vs_center')} "
            f"rank_improve_rate_vs_center={stats.get('rank_improve_rate_vs_center')}"
        )


if __name__ == "__main__":
    main()

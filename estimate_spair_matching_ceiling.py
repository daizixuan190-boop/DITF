import argparse
import csv
import json
import math
import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from analyze_spair_token_residuals import (
    build_post_feature,
    compute_exact_cos_map_hr,
    ensure_dir,
    sample_feature_at_pixel,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Estimate matching-stage ceiling on SPair-71k for frozen DiTF features. "
            "This gives an exact oracle upper bound for better selection/reranking on the current similarity maps, "
            "not the ultimate ceiling of all possible training-free methods."
        )
    )
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to SPair-71k root.")
    parser.add_argument("--feature_path", type=str, required=True, help="Path to saved per-category features.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs.")
    parser.add_argument("--cd", action="store_true", default=False, help="Apply channel discard as in eval_spair.py.")
    parser.add_argument("--discard_channels", nargs="+", type=int, default=[154, 1446], help="Channels to discard for Flux.")
    parser.add_argument("--feature_dim", type=int, default=3072, help="Feature dimension.")
    parser.add_argument("--tile_rows", type=int, default=32, help="Row chunk size for exact high-resolution cosine map evaluation.")
    parser.add_argument("--flush_every_pairs", type=int, default=10, help="Write partial outputs every N pairs.")
    parser.add_argument("--device", type=str, default="cuda", help="Device for exact chunked matching, e.g. cuda or cpu.")
    parser.add_argument("--max_pairs_per_cat", type=int, default=0, help="Optional category-wise cap for quick diagnosis.")
    parser.add_argument(
        "--oracle_topk",
        nargs="+",
        type=int,
        default=[1, 5, 10, 50, 100, 500],
        help="Top-k cutoffs for oracle PCK on the current similarity map.",
    )
    return parser.parse_args()


def get_pair_scalar_fields(data: dict[str, Any]) -> dict[str, Any]:
    scalar_fields = {}
    for key, value in data.items():
        if isinstance(value, (int, float, bool, str)):
            scalar_fields[key] = value
    return scalar_fields


def write_records_csv(records: list[dict[str, Any]], csv_path: str):
    if not records:
        return
    fieldnames = sorted({key for record in records for key in record.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def best_valid_rank_and_score(
    cos_map: torch.Tensor,
    trg_x: int,
    trg_y: int,
    pck_radius: float,
) -> tuple[int, float]:
    out_h, out_w = cos_map.shape
    radius_int = max(1, int(math.ceil(pck_radius)))
    x0 = max(trg_x - radius_int, 0)
    x1 = min(trg_x + radius_int + 1, out_w)
    y0 = max(trg_y - radius_int, 0)
    y1 = min(trg_y + radius_int + 1, out_h)

    patch = cos_map[y0:y1, x0:x1]
    ys = torch.arange(y0, y1, dtype=torch.float32)
    xs = torch.arange(x0, x1, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    mask = ((xx - float(trg_x)) ** 2 + (yy - float(trg_y)) ** 2) <= (float(pck_radius) ** 2)
    valid_scores = patch[mask]
    if valid_scores.numel() == 0:
        gt_score = float(cos_map[trg_y, trg_x].item())
        rank = 1 + int((cos_map > gt_score).sum().item())
        return rank, gt_score

    best_score = float(valid_scores.max().item())
    rank = 1 + int((cos_map > best_score).sum().item())
    return rank, best_score


def build_summary(records: list[dict[str, Any]], oracle_topk: list[int]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "num_points": len(records),
        "current_pck@0.1": None,
        "oracle_pck@0.1": {},
        "headroom_gain": {},
        "rank_stats": {},
    }
    if not records:
        return summary

    current = float(np.mean([r["current_correct"] for r in records]))
    summary["current_pck@0.1"] = current

    ranks = np.array([r["oracle_best_rank"] for r in records], dtype=np.int64)
    summary["rank_stats"] = {
        "median_oracle_rank": float(np.median(ranks)),
        "mean_oracle_rank": float(np.mean(ranks)),
        "oracle_rank_le_10_frac": float(np.mean(ranks <= 10)),
        "oracle_rank_le_50_frac": float(np.mean(ranks <= 50)),
        "oracle_rank_le_100_frac": float(np.mean(ranks <= 100)),
        "oracle_rank_le_500_frac": float(np.mean(ranks <= 500)),
    }

    for k in oracle_topk:
        oracle_pck = float(np.mean([r[f"oracle_hit@{k}"] for r in records]))
        summary["oracle_pck@0.1"][str(k)] = oracle_pck
        summary["headroom_gain"][str(k)] = float(oracle_pck - current)
    return summary


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    print(f"Using matching device: {device}")
    print(
        "Ceiling type: exact oracle for the current frozen similarity maps; "
        "not an exact ceiling for all training-free feature modifications or model fusion."
    )

    test_path = os.path.join(args.dataset_path, "PairAnnotation", "test")
    image_root = os.path.join(args.dataset_path, "JPEGImages")
    json_list = os.listdir(test_path)
    all_cats = os.listdir(image_root)

    cat2json = {}
    for cat in all_cats:
        cat2json[cat] = [name for name in json_list if cat in name]

    pre_norm = nn.LayerNorm(args.feature_dim, elementwise_affine=False, eps=1e-6)
    records: list[dict[str, Any]] = []
    csv_path = os.path.join(args.output_dir, "matching_ceiling_records.csv")
    summary_path = os.path.join(args.output_dir, "matching_ceiling_summary.json")
    progress_path = os.path.join(args.output_dir, "matching_ceiling_progress.log")
    processed_pairs = 0

    for cat in all_cats:
        feat_file = os.path.join(args.feature_path, f"{cat}.pth")
        ada_file = os.path.join(args.feature_path, f"{cat}_ada.pth")
        if not os.path.exists(feat_file) or not os.path.exists(ada_file):
            continue

        output_dict = torch.load(feat_file, map_location="cpu", weights_only=True)
        ada_dict = torch.load(ada_file, map_location="cpu", weights_only=True)

        pair_names = cat2json[cat]
        if args.max_pairs_per_cat > 0:
            pair_names = pair_names[: args.max_pairs_per_cat]

        print(f"[Category] {cat}: {len(pair_names)} pairs")
        for pair_name in pair_names:
            with open(os.path.join(test_path, pair_name), "r", encoding="utf-8") as f:
                data = json.load(f)

            src_imname = data["src_imname"]
            trg_imname = data["trg_imname"]

            _, _, src_ft_post, _, _ = build_post_feature(
                output_dict[src_imname].float(),
                ada_dict[src_imname].float(),
                pre_norm,
                args.discard_channels,
                args.cd,
            )
            _, _, trg_ft_post, _, _ = build_post_feature(
                output_dict[trg_imname].float(),
                ada_dict[trg_imname].float(),
                pre_norm,
                args.discard_channels,
                args.cd,
            )
            src_ft_post_dev = src_ft_post.float().to(device)
            trg_ft_post_dev = trg_ft_post.float().to(device)

            src_eval_h, src_eval_w = data["src_imsize"][:2][::-1]
            trg_eval_h, trg_eval_w = data["trg_imsize"][:2][::-1]
            trg_bndbox = data["trg_bndbox"]
            threshold = max(trg_bndbox[3] - trg_bndbox[1], trg_bndbox[2] - trg_bndbox[0])
            pck_radius = 0.1 * threshold
            pair_scalars = get_pair_scalar_fields(data)

            for kp_idx, (src_point, trg_point) in enumerate(zip(data["src_kps"], data["trg_kps"])):
                src_x, src_y = int(src_point[0]), int(src_point[1])
                trg_x, trg_y = int(trg_point[0]), int(trg_point[1])

                src_vec = sample_feature_at_pixel(src_ft_post_dev, src_x, src_y, src_eval_h, src_eval_w)
                cos_map_hr = compute_exact_cos_map_hr(src_vec, trg_ft_post_dev, trg_eval_h, trg_eval_w, args.tile_rows)
                pred_y, pred_x = np.unravel_index(int(cos_map_hr.view(-1).argmax().item()), cos_map_hr.shape)

                dist = math.sqrt((pred_x - trg_x) ** 2 + (pred_y - trg_y) ** 2)
                current_correct = int((dist / max(threshold, 1e-6)) <= 0.1)

                oracle_best_rank, oracle_best_score = best_valid_rank_and_score(cos_map_hr, trg_x, trg_y, pck_radius)
                num_pixels = int(cos_map_hr.numel())

                record = {
                    "category": cat,
                    "pair_name": pair_name,
                    "src_imname": src_imname,
                    "trg_imname": trg_imname,
                    "kp_idx": kp_idx,
                    "src_x": src_x,
                    "src_y": src_y,
                    "trg_x": trg_x,
                    "trg_y": trg_y,
                    "pred_x": int(pred_x),
                    "pred_y": int(pred_y),
                    "current_correct": current_correct,
                    "oracle_best_rank": int(oracle_best_rank),
                    "oracle_best_rank_frac": float(oracle_best_rank / max(num_pixels, 1)),
                    "oracle_best_score": float(oracle_best_score),
                    "num_pixels": num_pixels,
                    "pck_radius": float(pck_radius),
                }
                for k in args.oracle_topk:
                    record[f"oracle_hit@{k}"] = int(oracle_best_rank <= k)
                record.update(pair_scalars)
                records.append(record)

            processed_pairs += 1
            if processed_pairs % args.flush_every_pairs == 0:
                write_records_csv(records, csv_path)
                summary = build_summary(records, args.oracle_topk)
                with open(summary_path, "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)
                with open(progress_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"processed_pairs={processed_pairs}, num_points={summary['num_points']}, "
                        f"current_pck@0.1={summary['current_pck@0.1']}, oracle_pck@0.1={summary['oracle_pck@0.1']}\n"
                    )
                print(
                    f"[Flush] pairs={processed_pairs} points={summary['num_points']} "
                    f"current_pck@0.1={summary['current_pck@0.1']:.4f}"
                )

    write_records_csv(records, csv_path)
    summary = build_summary(records, args.oracle_topk)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved records to: {csv_path}")
    print(f"Saved summary to: {summary_path}")
    print(f"Num points: {summary['num_points']}")
    print(f"Current PCK@0.1: {summary['current_pck@0.1']}")
    print(f"Oracle PCK@0.1 by top-k: {summary['oracle_pck@0.1']}")
    print(f"Headroom gain by top-k: {summary['headroom_gain']}")


if __name__ == "__main__":
    main()

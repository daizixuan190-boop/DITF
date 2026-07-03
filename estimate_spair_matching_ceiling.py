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


def quantile_risk_map(values: torch.Tensor, quantile: float, tail: str = "high") -> torch.Tensor:
    flat = values.flatten()
    threshold = torch.quantile(flat, quantile)

    if tail == "high":
        ref = flat.max()
        denom = (ref - threshold).clamp_min(1e-6)
        return ((values - threshold) / denom).clamp(min=0.0, max=1.0)

    if tail == "low":
        ref = flat.min()
        denom = (threshold - ref).clamp_min(1e-6)
        return ((threshold - values) / denom).clamp(min=0.0, max=1.0)

    raise ValueError(f"Unsupported tail: {tail}")


def maybe_apply_post_calibration(
    ft_ln: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    args,
) -> torch.Tensor:
    content = (1 + scale) * ft_ln
    ft_post = content + shift
    if not args.shift_calibration and not args.joint_calibration:
        return ft_post

    shift_energy = torch.sum(shift.float() ** 2, dim=1, keepdim=True)
    content_energy = torch.sum(content.float() ** 2, dim=1, keepdim=True)
    post_energy = torch.sum(ft_post.float() ** 2, dim=1, keepdim=True).clamp_min(1e-6)
    shift_ratio = shift_energy / post_energy
    content_ratio = content_energy / post_energy

    if args.joint_calibration:
        high_shift = quantile_risk_map(shift_ratio, args.joint_shift_quantile, tail="high")
        low_content = quantile_risk_map(content_ratio, args.joint_content_quantile, tail="low")
        joint_risk = high_shift * low_content

        lambda_map = 1.0 - args.joint_shift_strength * joint_risk
        lambda_map = lambda_map.clamp(min=args.joint_min_shift_lambda, max=1.0)

        content_gain = 1.0 + args.joint_content_strength * joint_risk
        content_gain = content_gain.clamp(max=args.joint_max_content_gain)
        return content_gain.to(content.dtype) * content + lambda_map.to(content.dtype) * shift

    excess = quantile_risk_map(shift_ratio, args.shift_calibration_quantile, tail="high")
    lambda_map = 1.0 - args.shift_calibration_strength * excess
    lambda_map = lambda_map.clamp(min=args.shift_calibration_min_lambda, max=1.0)
    return content + lambda_map.to(content.dtype) * shift


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
    parser.add_argument("--shift_calibration", action="store_true", default=False, help="Apply shift-only post-AdaLN calibration before matching.")
    parser.add_argument("--shift_calibration_quantile", default=0.75, type=float, help="Only suppress tokens above this shift-ratio quantile.")
    parser.add_argument("--shift_calibration_strength", default=0.5, type=float, help="Suppression strength for high-shift tokens.")
    parser.add_argument("--shift_calibration_min_lambda", default=0.2, type=float, help="Minimum retained shift scaling after calibration.")
    parser.add_argument("--joint_calibration", action="store_true", default=False, help="Apply joint high-shift low-content calibration before matching.")
    parser.add_argument("--joint_shift_quantile", default=0.75, type=float, help="High-shift token threshold for joint calibration.")
    parser.add_argument("--joint_content_quantile", default=0.25, type=float, help="Low-content token threshold for joint calibration.")
    parser.add_argument("--joint_shift_strength", default=0.5, type=float, help="Shift suppression strength under joint calibration.")
    parser.add_argument("--joint_min_shift_lambda", default=0.2, type=float, help="Minimum retained shift scaling under joint calibration.")
    parser.add_argument("--joint_content_strength", default=0.25, type=float, help="Content amplification strength under joint calibration.")
    parser.add_argument("--joint_max_content_gain", default=1.5, type=float, help="Upper bound for content amplification under joint calibration.")
    parser.add_argument("--local_rerank", action="store_true", default=False, help="Rerank top-k forward candidates by local neighborhood feature consensus.")
    parser.add_argument("--local_rerank_topk", default=5, type=int, help="Number of forward candidates considered for local reranking.")
    parser.add_argument("--local_rerank_radius", default=1, type=int, help="Neighborhood radius in pixels for local consensus reranking.")
    parser.add_argument("--local_rerank_consensus_weight", default=0.5, type=float, help="Weight of neighborhood consensus added to center similarity.")
    parser.add_argument("--support_match", action="store_true", default=False, help="Build a support-aware score map by aggregating offset-aligned local source evidence.")
    parser.add_argument("--support_radius", default=1, type=int, help="Neighborhood radius for support-aware matching.")
    parser.add_argument("--support_sigma", default=1.0, type=float, help="Gaussian weight sigma for support offsets.")
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


def topk_candidate_points(cos_map: torch.Tensor, topk: int) -> list[tuple[int, int, float]]:
    flat = cos_map.view(-1)
    k = min(max(int(topk), 1), int(flat.numel()))
    values, indices = torch.topk(flat, k=k)
    out_w = cos_map.shape[1]
    candidates = []
    for score, flat_idx in zip(values.tolist(), indices.tolist()):
        y = int(flat_idx // out_w)
        x = int(flat_idx % out_w)
        candidates.append((x, y, float(score)))
    return candidates


def local_consensus_score(
    src_feat: torch.Tensor,
    trg_feat: torch.Tensor,
    src_x: int,
    src_y: int,
    trg_x: int,
    trg_y: int,
    src_eval_h: int,
    src_eval_w: int,
    trg_eval_h: int,
    trg_eval_w: int,
    radius: int,
) -> float:
    scores: list[float] = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            src_x_off = src_x + dx
            src_y_off = src_y + dy
            trg_x_off = trg_x + dx
            trg_y_off = trg_y + dy
            if not (0 <= src_x_off < src_eval_w and 0 <= src_y_off < src_eval_h):
                continue
            if not (0 <= trg_x_off < trg_eval_w and 0 <= trg_y_off < trg_eval_h):
                continue
            src_vec = sample_feature_at_pixel(src_feat, src_x_off, src_y_off, src_eval_h, src_eval_w)
            trg_vec = sample_feature_at_pixel(trg_feat, trg_x_off, trg_y_off, trg_eval_h, trg_eval_w)
            src_vec = torch.nn.functional.normalize(src_vec.view(1, -1), dim=1)
            trg_vec = torch.nn.functional.normalize(trg_vec.view(1, -1), dim=1)
            scores.append(float((src_vec * trg_vec).sum().item()))
    if not scores:
        return float("-inf")
    return float(np.mean(scores))


def rerank_prediction_with_local_consensus(
    cos_map: torch.Tensor,
    src_feat: torch.Tensor,
    trg_feat: torch.Tensor,
    src_x: int,
    src_y: int,
    src_eval_h: int,
    src_eval_w: int,
    trg_eval_h: int,
    trg_eval_w: int,
    args,
) -> tuple[int, int]:
    candidates = topk_candidate_points(cos_map, args.local_rerank_topk)
    best_score = float("-inf")
    best_xy = None
    for cand_x, cand_y, center_score in candidates:
        consensus = local_consensus_score(
            src_feat,
            trg_feat,
            src_x,
            src_y,
            cand_x,
            cand_y,
            src_eval_h,
            src_eval_w,
            trg_eval_h,
            trg_eval_w,
            args.local_rerank_radius,
        )
        combined = float(center_score + args.local_rerank_consensus_weight * consensus)
        if combined > best_score:
            best_score = combined
            best_xy = (cand_x, cand_y)
    if best_xy is None:
        pred_y, pred_x = np.unravel_index(int(cos_map.view(-1).argmax().item()), cos_map.shape)
        return int(pred_x), int(pred_y)
    return best_xy


def offset_weight(dx: int, dy: int, sigma: float) -> float:
    if sigma <= 0:
        return 1.0
    return float(math.exp(-float(dx * dx + dy * dy) / (2.0 * sigma * sigma)))


def accumulate_shifted_map(
    accum: torch.Tensor,
    weight_accum: torch.Tensor,
    cos_map: torch.Tensor,
    dx: int,
    dy: int,
    weight: float,
):
    out_h, out_w = cos_map.shape

    if dy >= 0:
        src_y0, src_y1 = dy, out_h
        dst_y0, dst_y1 = 0, out_h - dy
    else:
        src_y0, src_y1 = 0, out_h + dy
        dst_y0, dst_y1 = -dy, out_h

    if dx >= 0:
        src_x0, src_x1 = dx, out_w
        dst_x0, dst_x1 = 0, out_w - dx
    else:
        src_x0, src_x1 = 0, out_w + dx
        dst_x0, dst_x1 = -dx, out_w

    if src_y0 >= src_y1 or src_x0 >= src_x1:
        return

    accum[dst_y0:dst_y1, dst_x0:dst_x1] += weight * cos_map[src_y0:src_y1, src_x0:src_x1]
    weight_accum[dst_y0:dst_y1, dst_x0:dst_x1] += weight


def build_support_score_map(
    src_feat: torch.Tensor,
    trg_feat: torch.Tensor,
    src_x: int,
    src_y: int,
    src_eval_h: int,
    src_eval_w: int,
    trg_eval_h: int,
    trg_eval_w: int,
    tile_rows: int,
    support_radius: int,
    support_sigma: float,
) -> torch.Tensor:
    support_map = torch.zeros((trg_eval_h, trg_eval_w), dtype=torch.float32)
    support_weight = torch.zeros((trg_eval_h, trg_eval_w), dtype=torch.float32)

    for dy in range(-support_radius, support_radius + 1):
        for dx in range(-support_radius, support_radius + 1):
            src_x_off = src_x + dx
            src_y_off = src_y + dy
            if not (0 <= src_x_off < src_eval_w and 0 <= src_y_off < src_eval_h):
                continue

            src_vec = sample_feature_at_pixel(src_feat, src_x_off, src_y_off, src_eval_h, src_eval_w)
            cos_map = compute_exact_cos_map_hr(src_vec, trg_feat, trg_eval_h, trg_eval_w, tile_rows)
            weight = offset_weight(dx, dy, support_sigma)
            accumulate_shifted_map(support_map, support_weight, cos_map, dx, dy, weight)

    valid = support_weight > 0
    score_map = torch.full_like(support_map, fill_value=-1e9)
    score_map[valid] = support_map[valid] / support_weight[valid]
    return score_map


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
        "oracle_rank_le_1_frac": float(np.mean(ranks <= 1)),
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
            src_ft_post = maybe_apply_post_calibration(src_ft_ln, src_shift, src_scale, args)
            trg_ft_post = maybe_apply_post_calibration(trg_ft_ln, trg_shift, trg_scale, args)
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

                if args.support_match:
                    cos_map_hr = build_support_score_map(
                        src_ft_post_dev,
                        trg_ft_post_dev,
                        src_x,
                        src_y,
                        src_eval_h,
                        src_eval_w,
                        trg_eval_h,
                        trg_eval_w,
                        args.tile_rows,
                        args.support_radius,
                        args.support_sigma,
                    )
                else:
                    src_vec = sample_feature_at_pixel(src_ft_post_dev, src_x, src_y, src_eval_h, src_eval_w)
                    cos_map_hr = compute_exact_cos_map_hr(src_vec, trg_ft_post_dev, trg_eval_h, trg_eval_w, args.tile_rows)
                if args.local_rerank:
                    pred_x, pred_y = rerank_prediction_with_local_consensus(
                        cos_map_hr,
                        src_ft_post_dev,
                        trg_ft_post_dev,
                        src_x,
                        src_y,
                        src_eval_h,
                        src_eval_w,
                        trg_eval_h,
                        trg_eval_w,
                        args,
                    )
                else:
                    pred_y, pred_x = np.unravel_index(int(cos_map_hr.view(-1).argmax().item()), cos_map_hr.shape)
                    pred_x, pred_y = int(pred_x), int(pred_y)

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

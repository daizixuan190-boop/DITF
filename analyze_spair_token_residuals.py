import argparse
import csv
import json
import math
import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze token-level residual discriminability after AdaLN + channel discard on SPair-71k."
    )
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to SPair-71k root.")
    parser.add_argument("--feature_path", type=str, required=True, help="Path to saved per-category features.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save analysis outputs.")
    parser.add_argument("--img_size", nargs="+", type=int, default=[640, 640], help="Unused for analysis, kept for bookkeeping.")
    parser.add_argument("--cd", action="store_true", default=False, help="Apply channel discard as in eval_spair.py.")
    parser.add_argument("--discard_channels", nargs="+", type=int, default=[154, 1446], help="Channels to discard for Flux.")
    parser.add_argument("--feature_dim", type=int, default=3072, help="Feature dimension.")
    parser.add_argument("--top_scale_k", type=int, default=64, help="Top-|scale| channels used to define residual scale load.")
    parser.add_argument("--top_energy_k", type=int, default=32, help="Top-energy channels used to define feature concentration.")
    parser.add_argument("--max_pairs_per_cat", type=int, default=0, help="Optional category-wise cap for quick diagnosis.")
    parser.add_argument("--tile_rows", type=int, default=32, help="Row chunk size for exact high-resolution cosine map evaluation.")
    parser.add_argument("--flush_every_pairs", type=int, default=10, help="Write partial outputs every N pairs.")
    return parser.parse_args()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def apply_channel_discard(ft_raw: torch.Tensor, discard_channels: list[int]) -> torch.Tensor:
    ft = ft_raw.clone()
    for ch in discard_channels:
        if 0 <= ch < ft.shape[1]:
            ft[:, ch, :, :] = 0.0
    return ft


def build_post_feature(
    ft_raw: torch.Tensor,
    ada: torch.Tensor,
    pre_norm: nn.LayerNorm,
    discard_channels: list[int],
    use_cd: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ft_input = apply_channel_discard(ft_raw, discard_channels) if use_cd else ft_raw.clone()
    _, _, h, w = ft_input.shape

    ft_ln = rearrange(ft_input, "b c h w -> b (h w) c")
    ft_ln = pre_norm(ft_ln)
    ft_ln = rearrange(ft_ln, "b (h w) c -> b c h w", h=h, w=w)

    shift = ada[0][0].unsqueeze(0).unsqueeze(2).unsqueeze(3)
    scale = ada[0][1].unsqueeze(0).unsqueeze(2).unsqueeze(3)
    ft_post = (1 + scale) * ft_ln + shift
    return ft_raw, ft_ln, ft_post, shift, scale


def safe_energy_ratio(vec: torch.Tensor, channel_idx: list[int]) -> float:
    if vec.numel() == 0:
        return 0.0
    idx = [i for i in channel_idx if 0 <= i < vec.shape[0]]
    if not idx:
        return 0.0
    energy = torch.sum(vec.float() ** 2).item()
    if energy <= 0:
        return 0.0
    selected = torch.sum(vec[idx].float() ** 2).item()
    return float(selected / energy)


def topk_energy_ratio(vec: torch.Tensor, k: int) -> float:
    if vec.numel() == 0:
        return 0.0
    energy = vec.float() ** 2
    total = torch.sum(energy).item()
    if total <= 0:
        return 0.0
    k = min(k, energy.numel())
    topk = torch.topk(energy, k=k).values.sum().item()
    return float(topk / total)


def entropy_from_scores(scores: np.ndarray, temperature: float = 0.05) -> float:
    x = scores.astype(np.float64) / temperature
    x = x - x.max()
    prob = np.exp(x)
    prob = prob / max(prob.sum(), 1e-12)
    entropy = -np.sum(prob * np.log(prob + 1e-12))
    return float(entropy / np.log(len(scores) + 1e-12))


def bbox_margin(point_xy: list[int], bbox_xyxy: list[float], normalizer: float) -> float:
    x, y = point_xy
    x1, y1, x2, y2 = bbox_xyxy
    margin = min(x - x1, x2 - x, y - y1, y2 - y)
    return float(margin / max(normalizer, 1e-6))


def normalized_displacement(src_point: list[int], trg_point: list[int], normalizer: float) -> float:
    dx = trg_point[0] - src_point[0]
    dy = trg_point[1] - src_point[1]
    return float(math.sqrt(dx * dx + dy * dy) / max(normalizer, 1e-6))


def map_pixel_to_feature_index(pixel_x: int, pixel_y: int, eval_h: int, eval_w: int, feat_h: int, feat_w: int) -> tuple[int, int]:
    feat_x = int(round(pixel_x * (feat_w - 1) / max(eval_w - 1, 1)))
    feat_y = int(round(pixel_y * (feat_h - 1) / max(eval_h - 1, 1)))
    feat_x = min(max(feat_x, 0), feat_w - 1)
    feat_y = min(max(feat_y, 0), feat_h - 1)
    return feat_x, feat_y


def make_grid_for_output_window(
    x_start: int,
    x_end: int,
    y_start: int,
    y_end: int,
    out_h: int,
    out_w: int,
) -> torch.Tensor:
    xs = torch.arange(x_start, x_end, dtype=torch.float32)
    ys = torch.arange(y_start, y_end, dtype=torch.float32)
    grid_x = 2.0 * ((xs + 0.5) / out_w) - 1.0
    grid_y = 2.0 * ((ys + 0.5) / out_h) - 1.0
    mesh_y, mesh_x = torch.meshgrid(grid_y, grid_x, indexing="ij")
    return torch.stack((mesh_x, mesh_y), dim=-1).unsqueeze(0)


def sample_feature_at_pixel(
    feat: torch.Tensor,
    pixel_x: int,
    pixel_y: int,
    out_h: int,
    out_w: int,
) -> torch.Tensor:
    grid = make_grid_for_output_window(pixel_x, pixel_x + 1, pixel_y, pixel_y + 1, out_h, out_w)
    sampled = F.grid_sample(feat, grid, mode="bilinear", align_corners=False)
    return sampled[0, :, 0, 0]


def compute_exact_cos_map_hr(
    src_vec: torch.Tensor,
    trg_feat: torch.Tensor,
    out_h: int,
    out_w: int,
    tile_rows: int,
) -> torch.Tensor:
    src_vec = F.normalize(src_vec.view(1, -1), dim=1).view(1, -1, 1, 1)
    cos_map = torch.empty((out_h, out_w), dtype=torch.float32)
    trg_feat = trg_feat.float()
    for y_start in range(0, out_h, tile_rows):
        y_end = min(y_start + tile_rows, out_h)
        grid = make_grid_for_output_window(0, out_w, y_start, y_end, out_h, out_w)
        tile_feat = F.grid_sample(trg_feat, grid, mode="bilinear", align_corners=False)
        tile_feat = F.normalize(tile_feat, dim=1)
        tile_sim = (tile_feat * src_vec).sum(dim=1)[0]
        cos_map[y_start:y_end, :] = tile_sim.cpu()
    return cos_map


def get_pair_scalar_fields(data: dict[str, Any]) -> dict[str, Any]:
    scalar_fields = {}
    for key, value in data.items():
        if isinstance(value, (int, float, bool, str)):
            scalar_fields[key] = value
    return scalar_fields


def decile_summary(records: list[dict[str, Any]], score_key: str) -> list[dict[str, Any]]:
    values = np.array([r[score_key] for r in records], dtype=np.float64)
    if len(values) == 0:
        return []
    order = np.argsort(values)
    chunks = np.array_split(order, 10)
    summary = []
    for i, chunk in enumerate(chunks):
        if len(chunk) == 0:
            continue
        subset = [records[j] for j in chunk]
        error_rate = 1.0 - float(np.mean([r["correct"] for r in subset]))
        mean_dist = float(np.mean([r["norm_dist"] for r in subset]))
        summary.append(
            {
                "decile": i,
                "count": len(subset),
                "score_min": float(np.min([r[score_key] for r in subset])),
                "score_max": float(np.max([r[score_key] for r in subset])),
                "error_rate": error_rate,
                "mean_norm_dist": mean_dist,
            }
        )
    return summary


def quartile_control_summary(records: list[dict[str, Any]], control_key: str, score_key: str) -> list[dict[str, Any]]:
    control_values = np.array([r[control_key] for r in records], dtype=np.float64)
    if len(control_values) == 0:
        return []
    control_order = np.argsort(control_values)
    control_chunks = np.array_split(control_order, 4)
    results = []
    for q_idx, chunk in enumerate(control_chunks):
        subset = [records[j] for j in chunk]
        if len(subset) < 20:
            continue
        deciles = decile_summary(subset, score_key)
        if not deciles:
            continue
        results.append(
            {
                "control_key": control_key,
                "quartile": q_idx,
                "control_min": float(np.min([r[control_key] for r in subset])),
                "control_max": float(np.max([r[control_key] for r in subset])),
                "bottom_decile_error": deciles[0]["error_rate"],
                "top_decile_error": deciles[-1]["error_rate"],
                "error_gap": deciles[-1]["error_rate"] - deciles[0]["error_rate"],
                "count": len(subset),
            }
        )
    return results


def write_records_csv(records: list[dict[str, Any]], csv_path: str):
    if not records:
        return
    fieldnames = sorted({key for record in records for key in record.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_summary_json(records: list[dict[str, Any]], summary_path: str):
    score_keys = [
        "post_highscale_ratio_src",
        "ln_highscale_ratio_src",
        "post_topk_ratio_src",
        "shift_ratio_src",
        "pre_ma_ratio",
    ]
    summary = {
        "num_points": len(records),
        "overall_error_rate": float(1.0 - np.mean([r["correct"] for r in records])) if records else None,
        "deciles": {},
        "controls": {},
    }
    for score_key in score_keys:
        summary["deciles"][score_key] = decile_summary(records, score_key)
        summary["controls"][score_key] = {
            "by_src_boundary_margin": quartile_control_summary(records, "src_boundary_margin", score_key),
            "by_trg_boundary_margin": quartile_control_summary(records, "trg_boundary_margin", score_key),
            "by_pair_displacement": quartile_control_summary(records, "pair_displacement", score_key),
        }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    test_path = os.path.join(args.dataset_path, "PairAnnotation", "test")
    image_root = os.path.join(args.dataset_path, "JPEGImages")
    json_list = os.listdir(test_path)
    all_cats = os.listdir(image_root)

    cat2json = {}
    for cat in all_cats:
        cat2json[cat] = [name for name in json_list if cat in name]

    pre_norm = nn.LayerNorm(args.feature_dim, elementwise_affine=False, eps=1e-6)
    all_records: list[dict[str, Any]] = []
    csv_path = os.path.join(args.output_dir, "per_point_records.csv")
    summary_path = os.path.join(args.output_dir, "summary.json")
    progress_path = os.path.join(args.output_dir, "progress.log")
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
        for pair_name in tqdm(pair_names, desc=f"{cat}", leave=False):
            with open(os.path.join(test_path, pair_name), "r", encoding="utf-8") as f:
                data = json.load(f)

            src_imname = data["src_imname"]
            trg_imname = data["trg_imname"]

            src_ft_raw_orig, src_ft_ln, src_ft_post, src_shift, src_scale = build_post_feature(
                output_dict[src_imname].float(),
                ada_dict[src_imname].float(),
                pre_norm,
                args.discard_channels,
                args.cd,
            )
            trg_ft_raw_orig, trg_ft_ln, trg_ft_post, trg_shift, trg_scale = build_post_feature(
                output_dict[trg_imname].float(),
                ada_dict[trg_imname].float(),
                pre_norm,
                args.discard_channels,
                args.cd,
            )

            src_img_size = data["src_imsize"][:2][::-1]
            trg_img_size = data["trg_imsize"][:2][::-1]

            src_eval_h, src_eval_w = src_img_size
            trg_eval_h, trg_eval_w = trg_img_size

            trg_bndbox = data["trg_bndbox"]
            src_bndbox = data["src_bndbox"]
            threshold = max(trg_bndbox[3] - trg_bndbox[1], trg_bndbox[2] - trg_bndbox[0])
            pair_scalars = get_pair_scalar_fields(data)

            scale_vec_src = src_scale[0, :, 0, 0].abs()
            scale_vec_trg = trg_scale[0, :, 0, 0].abs()
            top_scale_src = torch.topk(scale_vec_src, k=min(args.top_scale_k, scale_vec_src.numel())).indices.tolist()
            top_scale_trg = torch.topk(scale_vec_trg, k=min(args.top_scale_k, scale_vec_trg.numel())).indices.tolist()

            for kp_idx, (src_point, trg_point) in enumerate(zip(data["src_kps"], data["trg_kps"])):
                src_x, src_y = int(src_point[0]), int(src_point[1])
                trg_x, trg_y = int(trg_point[0]), int(trg_point[1])

                src_vec = sample_feature_at_pixel(src_ft_post.float(), src_x, src_y, src_eval_h, src_eval_w)
                cos_map_hr = compute_exact_cos_map_hr(src_vec, trg_ft_post, trg_eval_h, trg_eval_w, args.tile_rows)
                flat_scores = cos_map_hr.view(-1).numpy()
                topk = min(2, flat_scores.shape[0])
                top2_idx = np.argpartition(flat_scores, -topk)[-topk:]
                top2_scores = np.sort(flat_scores[top2_idx])[::-1]
                top1_score = float(top2_scores[0])
                top2_score = float(top2_scores[1]) if len(top2_scores) > 1 else float(top2_scores[0])
                pred_y, pred_x = np.unravel_index(int(flat_scores.argmax()), cos_map_hr.shape)

                dist = math.sqrt((pred_x - trg_x) ** 2 + (pred_y - trg_y) ** 2)
                norm_dist = float(dist / max(threshold, 1e-6))
                correct = int(norm_dist <= 0.1)

                src_vec_raw = sample_feature_at_pixel(src_ft_raw_orig.float(), src_x, src_y, src_eval_h, src_eval_w)
                src_vec_ln = sample_feature_at_pixel(src_ft_ln.float(), src_x, src_y, src_eval_h, src_eval_w)
                src_vec_post = src_vec
                trg_vec_post = sample_feature_at_pixel(trg_ft_post.float(), trg_x, trg_y, trg_eval_h, trg_eval_w)

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
                    "correct": correct,
                    "norm_dist": norm_dist,
                    "top1_score": top1_score,
                    "top2_score": top2_score,
                    "sim_margin": float(top1_score - top2_score),
                    "sim_entropy": entropy_from_scores(flat_scores),
                    "src_boundary_margin": bbox_margin([src_x, src_y], src_bndbox, threshold),
                    "trg_boundary_margin": bbox_margin([trg_x, trg_y], trg_bndbox, threshold),
                    "pair_displacement": normalized_displacement([src_x, src_y], [trg_x, trg_y], threshold),
                    "pre_ma_ratio": safe_energy_ratio(src_vec_raw, args.discard_channels),
                    "post_highscale_ratio_src": safe_energy_ratio(src_vec_post, top_scale_src),
                    "post_highscale_ratio_trg": safe_energy_ratio(trg_vec_post, top_scale_trg),
                    "ln_highscale_ratio_src": safe_energy_ratio(src_vec_ln, top_scale_src),
                    "post_topk_ratio_src": topk_energy_ratio(src_vec_post, args.top_energy_k),
                    "post_topk_ratio_trg": topk_energy_ratio(trg_vec_post, args.top_energy_k),
                    "shift_ratio_src": float(
                        torch.sum(src_shift[0, :, 0, 0].float() ** 2).item()
                        / max(torch.sum(src_vec_post.float() ** 2).item(), 1e-6)
                    ),
                    "shift_ratio_trg": float(
                        torch.sum(trg_shift[0, :, 0, 0].float() ** 2).item()
                        / max(torch.sum(trg_vec_post.float() ** 2).item(), 1e-6)
                    ),
                }
                record.update(pair_scalars)
                all_records.append(record)

            processed_pairs += 1
            if processed_pairs % args.flush_every_pairs == 0:
                write_records_csv(all_records, csv_path)
                summary = write_summary_json(all_records, summary_path)
                with open(progress_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"processed_pairs={processed_pairs}, num_points={summary['num_points']}, overall_error_rate={summary['overall_error_rate']}\n"
                    )
                print(
                    f"[Flush] pairs={processed_pairs} points={summary['num_points']} overall_error_rate={summary['overall_error_rate']}"
                )

    write_records_csv(all_records, csv_path)
    summary = write_summary_json(all_records, summary_path)

    print(f"Saved per-point records to: {csv_path}")
    print(f"Saved summary to: {summary_path}")
    print(f"Num points: {summary['num_points']}")
    print(f"Overall error rate: {summary['overall_error_rate']}")


if __name__ == "__main__":
    main()

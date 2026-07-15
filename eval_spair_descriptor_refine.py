import argparse
import csv
import json
import os
import warnings

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F
from tqdm import tqdm

from eval_spair_confusion_rerank import maybe_apply_shift_calibration, quantile_risk_map
from eval_spair_identity_cycle import collect_spair_lists, maybe_extract_features

warnings.filterwarnings("ignore")


def build_post_feature(ft_raw: torch.Tensor, ada: torch.Tensor, pre_norm: nn.LayerNorm, args) -> torch.Tensor:
    ft = ft_raw.clone()
    if args.cd:
        ft[:, 154, :, :] = 0.0
        ft[:, 1446, :, :] = 0.0
    _, _, height, width = ft.shape
    ft = rearrange(ft, "b c h w -> b (h w) c")
    ft = pre_norm(ft)
    ft = rearrange(ft, "b (h w) c -> b c h w", h=height, w=width)
    shift = ada[0][0].unsqueeze(0).unsqueeze(2).unsqueeze(3)
    scale = ada[0][1].unsqueeze(0).unsqueeze(2).unsqueeze(3)
    return maybe_apply_shift_calibration(ft, shift, scale, args)


def pair_scale_risk(pair_data: dict) -> float:
    scale_value = pair_data.get("scale_variation")
    if scale_value is None:
        return 0.0
    try:
        scale_value = float(scale_value)
    except (TypeError, ValueError):
        return 0.0
    return float(np.clip(scale_value / 2.0, 0.0, 1.0))


def build_offset_list(radius: int) -> list[tuple[int, int]]:
    offsets = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            offsets.append((dx, dy))
    return offsets


def compute_source_context_vectors(
    src_ft: torch.Tensor,
    src_xy: torch.Tensor,
    radius: int,
    min_neighbors: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_points = int(src_xy.shape[0])
    channels = int(src_ft.shape[1])
    device = src_ft.device
    context_vecs = torch.zeros(num_points, channels, device=device, dtype=torch.float32)
    context_valid = torch.zeros(num_points, device=device, dtype=torch.bool)
    if radius <= 0:
        return context_vecs, context_valid

    _, _, src_h, src_w = src_ft.shape
    offsets = build_offset_list(radius)
    if not offsets:
        return context_vecs, context_valid

    for point_idx in range(num_points):
        src_x = int(src_xy[point_idx, 0].item())
        src_y = int(src_xy[point_idx, 1].item())
        local_vecs = []
        for dx, dy in offsets:
            x = src_x + dx
            y = src_y + dy
            if x < 0 or x >= src_w or y < 0 or y >= src_h:
                continue
            local_vecs.append(src_ft[0, :, y, x].float())
        if len(local_vecs) < int(min_neighbors):
            continue
        stacked = torch.stack(local_vecs, dim=0)
        context_vec = F.normalize(stacked, dim=1).mean(dim=0)
        context_vecs[point_idx] = F.normalize(context_vec.view(1, -1), dim=1).squeeze(0)
        context_valid[point_idx] = True
    return context_vecs, context_valid


def prepare_source_bundle(
    src_ft: torch.Tensor,
    src_points: list[list[int]],
    src_threshold: float,
    args,
) -> dict:
    device = src_ft.device
    channels = int(src_ft.shape[1])
    if not src_points:
        empty_xy = torch.empty(0, 2, device=device, dtype=torch.float32)
        empty_vec = torch.empty(0, channels, device=device, dtype=torch.float32)
        empty_mat = torch.empty(0, 0, device=device, dtype=torch.float32)
        empty_mask = torch.empty(0, device=device, dtype=torch.bool)
        return {
            "src_xy": empty_xy,
            "src_vecs": empty_vec,
            "src_ambiguity": torch.empty(0, device=device, dtype=torch.float32),
            "rival_weights": empty_mat,
            "context_vecs": empty_vec,
            "context_valid": empty_mask,
        }

    src_xy = torch.tensor(
        [(int(point[0]), int(point[1])) for point in src_points],
        device=device,
        dtype=torch.long,
    )
    src_vecs = src_ft[0, :, src_xy[:, 1], src_xy[:, 0]].transpose(0, 1).contiguous().float()
    src_vecs = F.normalize(src_vecs, dim=1)
    num_points = int(src_vecs.shape[0])

    context_vecs, context_valid = compute_source_context_vectors(
        src_ft,
        src_xy,
        int(args.sdr_context_radius),
        int(args.sdr_context_min_neighbors),
    )

    if num_points <= 1:
        return {
            "src_xy": src_xy.float(),
            "src_vecs": src_vecs,
            "src_ambiguity": torch.zeros(num_points, device=device, dtype=torch.float32),
            "rival_weights": torch.zeros(num_points, num_points, device=device, dtype=torch.float32),
            "context_vecs": context_vecs,
            "context_valid": context_valid,
        }

    src_sim = torch.mm(src_vecs, src_vecs.transpose(0, 1))
    src_sim.fill_diagonal_(-1e4)
    src_ambiguity, _ = torch.max(src_sim, dim=1)

    off_diag_mask = ~torch.eye(num_points, device=device, dtype=torch.bool)
    off_diag_vals = src_sim[off_diag_mask]
    sim_threshold = torch.quantile(off_diag_vals, float(args.sdr_source_quantile))
    sim_ref = off_diag_vals.max()
    sim_denom = (sim_ref - sim_threshold).clamp_min(1e-6)
    sim_risk = ((src_sim - sim_threshold) / sim_denom).clamp(min=0.0, max=1.0)
    sim_risk.fill_diagonal_(0.0)

    src_pair_dists = torch.cdist(src_xy.float(), src_xy.float()) / max(float(src_threshold), 1e-6)
    dist_gate = torch.exp(-src_pair_dists / max(float(args.sdr_src_distance_sigma), 1e-6))
    dist_gate.fill_diagonal_(0.0)

    rival_weights = sim_risk * dist_gate
    if int(args.sdr_max_rivals) > 0 and int(args.sdr_max_rivals) < num_points:
        top_vals, top_idx = torch.topk(rival_weights, k=int(args.sdr_max_rivals), dim=1)
        masked_weights = torch.zeros_like(rival_weights)
        masked_weights.scatter_(1, top_idx, top_vals)
        rival_weights = masked_weights

    if float(args.sdr_rival_floor) > 0.0:
        rival_weights = torch.where(
            rival_weights >= float(args.sdr_rival_floor),
            rival_weights,
            torch.zeros_like(rival_weights),
        )

    return {
        "src_xy": src_xy.float(),
        "src_vecs": src_vecs,
        "src_ambiguity": src_ambiguity,
        "rival_weights": rival_weights,
        "context_vecs": context_vecs,
        "context_valid": context_valid,
    }


def refine_source_descriptors(source_bundle: dict, args) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    src_vecs = source_bundle["src_vecs"]
    rival_weights = source_bundle["rival_weights"]
    context_vecs = source_bundle["context_vecs"]

    if int(src_vecs.shape[0]) == 0:
        return src_vecs, torch.empty(0, device=src_vecs.device), torch.empty(0, device=src_vecs.device)

    rival_proto = torch.mm(rival_weights, src_vecs)
    rival_norm = rival_proto.norm(dim=1)
    rival_valid = rival_norm > 0
    if int(rival_valid.sum().item()) > 0:
        rival_proto[rival_valid] = F.normalize(rival_proto[rival_valid], dim=1)

    delta = torch.zeros_like(src_vecs)
    if float(args.sdr_context_weight) != 0.0:
        delta = delta + float(args.sdr_context_weight) * context_vecs
    if float(args.sdr_rival_weight) != 0.0:
        delta = delta - float(args.sdr_rival_weight) * rival_proto

    refined = src_vecs + delta
    refined = F.normalize(refined, dim=1)
    delta_norm = torch.linalg.norm(delta, dim=1)
    rival_count = (rival_weights > 0).sum(dim=1).to(torch.float32)
    return refined, delta_norm, rival_count


def build_risk_weights(raw_score_mat: torch.Tensor, src_ambiguity: torch.Tensor, pair_risk: float, args) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    top_vals = torch.topk(raw_score_mat, k=min(2, int(raw_score_mat.shape[0])), dim=0).values
    top1_scores = top_vals[0]
    if int(top_vals.shape[0]) > 1:
        margins = top_vals[0] - top_vals[1]
    else:
        margins = top_vals[0]

    margin_risk = quantile_risk_map(margins, args.sdr_margin_quantile, tail="low")
    score_risk = quantile_risk_map(top1_scores, args.sdr_score_quantile, tail="low")
    ambiguity_risk = quantile_risk_map(src_ambiguity, args.sdr_ambiguity_quantile, tail="high")
    base_risk = (margin_risk + score_risk + ambiguity_risk) / 3.0
    risk_weight = (base_risk + float(args.sdr_scale_weight) * float(pair_risk)).clamp(min=0.0, max=1.0)
    return margins, top1_scores, risk_weight


def evaluate_descriptor_refine(
    trg_ft: torch.Tensor,
    trg_points: list[list[int]],
    source_bundle: dict,
    pair_risk: float,
    args,
) -> tuple[list[tuple[int, int]], list[dict]]:
    src_vecs = source_bundle["src_vecs"]
    if int(src_vecs.shape[0]) == 0:
        return [], []

    refined_src_vecs, delta_norm, rival_count = refine_source_descriptors(source_bundle, args)
    src_ambiguity = source_bundle["src_ambiguity"]
    rival_weights = source_bundle["rival_weights"]

    _, channels, trg_h, trg_w = trg_ft.shape
    trg_matrix = trg_ft.view(channels, -1).transpose(0, 1).float()
    trg_matrix = F.normalize(trg_matrix, dim=1)

    raw_score_mat = torch.mm(trg_matrix, src_vecs.transpose(0, 1))
    alt_score_mat = torch.mm(trg_matrix, refined_src_vecs.transpose(0, 1))
    margins, top1_scores, risk_weight = build_risk_weights(raw_score_mat, src_ambiguity, pair_risk, args)
    final_score_mat = raw_score_mat + risk_weight.unsqueeze(0) * (alt_score_mat - raw_score_mat)

    raw_best_idx = torch.argmax(raw_score_mat, dim=0)
    final_best_idx = torch.argmax(final_score_mat, dim=0)

    trg_gt_xy = torch.tensor(
        [(int(point[0]), int(point[1])) for point in trg_points],
        device=trg_ft.device,
        dtype=torch.long,
    )
    gt_flat_idx = trg_gt_xy[:, 1] * trg_w + trg_gt_xy[:, 0]
    point_idx = torch.arange(int(src_vecs.shape[0]), device=trg_ft.device, dtype=torch.long)

    raw_gt_score = raw_score_mat[gt_flat_idx, point_idx]
    alt_gt_score = alt_score_mat[gt_flat_idx, point_idx]
    final_gt_score = final_score_mat[gt_flat_idx, point_idx]
    raw_gt_rank = 1 + torch.sum(raw_score_mat > raw_gt_score.unsqueeze(0), dim=0)
    final_gt_rank = 1 + torch.sum(final_score_mat > final_gt_score.unsqueeze(0), dim=0)

    gt_trg_vecs = trg_ft[0, :, trg_gt_xy[:, 1], trg_gt_xy[:, 0]].transpose(0, 1).contiguous().float()
    gt_trg_vecs = F.normalize(gt_trg_vecs, dim=1)
    gt_cross_scores = torch.mm(gt_trg_vecs, src_vecs.transpose(0, 1))
    weighted_gt_rivals = gt_cross_scores * rival_weights
    gt_rival_score = torch.max(weighted_gt_rivals, dim=1).values
    raw_gt_margin = raw_gt_score - gt_rival_score
    final_gt_margin = final_gt_score - gt_rival_score

    predictions = []
    diagnostics = []
    for idx in range(int(src_vecs.shape[0])):
        raw_idx = int(raw_best_idx[idx].item())
        final_idx = int(final_best_idx[idx].item())
        raw_pred_x = raw_idx % trg_w
        raw_pred_y = raw_idx // trg_w
        final_pred_x = final_idx % trg_w
        final_pred_y = final_idx // trg_w
        predictions.append((int(final_pred_x), int(final_pred_y)))
        diagnostics.append(
            {
                "point_idx": idx,
                "risk_weight": float(risk_weight[idx].item()),
                "source_ambiguity": float(src_ambiguity[idx].item()),
                "rival_count": int(rival_count[idx].item()),
                "descriptor_delta_norm": float(delta_norm[idx].item()),
                "margin": float(margins[idx].item()),
                "base_top1_score": float(top1_scores[idx].item()),
                "raw_pred_x": int(raw_pred_x),
                "raw_pred_y": int(raw_pred_y),
                "final_pred_x": int(final_pred_x),
                "final_pred_y": int(final_pred_y),
                "raw_top1_score": float(raw_score_mat[raw_idx, idx].item()),
                "final_top1_score": float(final_score_mat[final_idx, idx].item()),
                "raw_gt_score": float(raw_gt_score[idx].item()),
                "alt_gt_score": float(alt_gt_score[idx].item()),
                "final_gt_score": float(final_gt_score[idx].item()),
                "raw_gt_rank": int(raw_gt_rank[idx].item()),
                "final_gt_rank": int(final_gt_rank[idx].item()),
                "gt_rank_gain": int(raw_gt_rank[idx].item() - final_gt_rank[idx].item()),
                "raw_gt_margin": float(raw_gt_margin[idx].item()),
                "final_gt_margin": float(final_gt_margin[idx].item()),
                "gt_margin_gain": float((final_gt_margin[idx] - raw_gt_margin[idx]).item()),
            }
        )
    return predictions, diagnostics


def save_point_records(records_path: str, point_records: list[dict]):
    if not point_records:
        return
    fieldnames = sorted({key for record in point_records for key in record.keys()})
    with open(records_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(point_records)


def main(args):
    for arg in vars(args):
        value = getattr(args, arg)
        if value is not None:
            print(f"{arg}: {value}")

    torch.cuda.set_device(0)
    all_cats, cat2json, cat2img = collect_spair_lists(args.dataset_path)
    maybe_extract_features(args, all_cats, cat2img)

    print("saving all test images' features...")
    os.makedirs(args.save_path, exist_ok=True)
    if args.reuse_saved_features:
        print("Reusing existing saved features; skip feature extraction.")

    pre_norm = nn.LayerNorm(3072, elementwise_affine=False, eps=1e-6)
    total_pck = []
    all_correct = 0
    all_total = 0
    mean_image_sum = 0.0
    mean_point_sum = 0.0
    result = {"image": {}, "point": {}}
    point_records = []
    test_path = os.path.join(args.dataset_path, "PairAnnotation", "test")

    print(f"Category numbers: {len(all_cats)}")
    for cat in all_cats:
        cat_list = cat2json[cat]
        if args.max_pairs_per_cat > 0:
            cat_list = cat_list[: args.max_pairs_per_cat]

        output_dict = torch.load(os.path.join(args.save_path, f"{cat}.pth"))
        ada_dict = torch.load(os.path.join(args.save_path, f"{cat}_ada.pth"))

        cat_pck = []
        cat_correct = 0
        cat_total = 0

        for json_name in tqdm(cat_list):
            with open(os.path.join(test_path, json_name), "r", encoding="utf-8") as f:
                data = json.load(f)

            src_img_size = data["src_imsize"][:2][::-1]
            trg_img_size = data["trg_imsize"][:2][::-1]

            src_ft_post = build_post_feature(
                output_dict[data["src_imname"]].cuda(),
                ada_dict[data["src_imname"]].cuda(),
                pre_norm,
                args,
            )
            trg_ft_post = build_post_feature(
                output_dict[data["trg_imname"]].cuda(),
                ada_dict[data["trg_imname"]].cuda(),
                pre_norm,
                args,
            )

            src_ft = nn.Upsample(size=src_img_size, mode="bilinear")(src_ft_post).to(torch.float16)
            trg_ft = nn.Upsample(size=trg_img_size, mode="bilinear")(trg_ft_post).to(torch.float16)

            src_points = data["src_kps"]
            trg_points = data["trg_kps"]
            trg_bndbox = data["trg_bndbox"]
            src_bndbox = data["src_bndbox"]
            src_threshold = max(src_bndbox[3] - src_bndbox[1], src_bndbox[2] - src_bndbox[0])
            trg_threshold = max(trg_bndbox[3] - trg_bndbox[1], trg_bndbox[2] - trg_bndbox[0])

            source_bundle = prepare_source_bundle(src_ft, src_points, src_threshold, args)
            predictions, diagnostics = evaluate_descriptor_refine(
                trg_ft,
                trg_points,
                source_bundle,
                pair_scale_risk(data),
                args,
            )

            total = 0
            correct = 0
            pair_name = os.path.splitext(json_name)[0]
            for idx, (pred_x, pred_y) in enumerate(predictions):
                total += 1
                cat_total += 1
                all_total += 1
                trg_point = trg_points[idx]
                dist = ((pred_x - trg_point[0]) ** 2 + (pred_y - trg_point[1]) ** 2) ** 0.5
                norm_dist = float(dist / max(float(trg_threshold), 1e-6))
                is_correct = int(norm_dist <= 0.1)
                if is_correct == 1:
                    correct += 1
                    cat_correct += 1
                    all_correct += 1

                if args.save_point_records:
                    diag = diagnostics[idx]
                    raw_dist = ((diag["raw_pred_x"] - trg_point[0]) ** 2 + (diag["raw_pred_y"] - trg_point[1]) ** 2) ** 0.5
                    point_records.append(
                        {
                            "category": cat,
                            "pair_name": pair_name,
                            "src_imname": data["src_imname"],
                            "trg_imname": data["trg_imname"],
                            "kp_idx": idx,
                            "src_x": int(src_points[idx][0]),
                            "src_y": int(src_points[idx][1]),
                            "trg_x": int(trg_point[0]),
                            "trg_y": int(trg_point[1]),
                            "pred_x": int(pred_x),
                            "pred_y": int(pred_y),
                            "dist": float(dist),
                            "norm_dist": norm_dist,
                            "correct": is_correct,
                            "raw_pred_x": diag["raw_pred_x"],
                            "raw_pred_y": diag["raw_pred_y"],
                            "raw_dist": float(raw_dist),
                            "raw_correct": int((raw_dist / max(float(trg_threshold), 1e-6)) <= 0.1),
                            "src_threshold": float(src_threshold),
                            "trg_threshold": float(trg_threshold),
                            "scale_variation": data.get("scale_variation"),
                            "risk_weight": diag["risk_weight"],
                            "source_ambiguity": diag["source_ambiguity"],
                            "rival_count": diag["rival_count"],
                            "descriptor_delta_norm": diag["descriptor_delta_norm"],
                            "margin": diag["margin"],
                            "base_top1_score": diag["base_top1_score"],
                            "raw_top1_score": diag["raw_top1_score"],
                            "final_top1_score": diag["final_top1_score"],
                            "raw_gt_score": diag["raw_gt_score"],
                            "alt_gt_score": diag["alt_gt_score"],
                            "final_gt_score": diag["final_gt_score"],
                            "raw_gt_rank": diag["raw_gt_rank"],
                            "final_gt_rank": diag["final_gt_rank"],
                            "gt_rank_gain": diag["gt_rank_gain"],
                            "raw_gt_margin": diag["raw_gt_margin"],
                            "final_gt_margin": diag["final_gt_margin"],
                            "gt_margin_gain": diag["gt_margin_gain"],
                            "method_tag": "descriptor_refine",
                        }
                    )

            cat_pck.append(correct / max(total, 1))
            torch.cuda.empty_cache()

        total_pck.extend(cat_pck)
        mean_image_sum += np.mean(cat_pck) * 100
        mean_point_sum += cat_correct / max(cat_total, 1) * 100

        print(f"{cat} per image PCK@0.1: {np.mean(cat_pck) * 100:.2f}")
        print(f"{cat} per point PCK@0.1: {cat_correct / max(cat_total, 1) * 100:.2f}")
        result["image"][cat] = round(np.mean(cat_pck) * 100, 2)
        result["point"][cat] = round(cat_correct / max(cat_total, 1) * 100, 2)

    print(f"All per image PCK@0.1: {np.mean(total_pck) * 100:.2f}")
    print(f"All per point PCK@0.1: {all_correct / max(all_total, 1) * 100:.2f}")
    print(f"Mean per image PCK@0.1: {mean_image_sum / len(all_cats):.2f}")
    print(f"Mean per point PCK@0.1: {mean_point_sum / len(all_cats):.2f}")

    result["image"]["All"] = round(np.mean(total_pck) * 100, 2)
    result["point"]["All"] = round(all_correct / max(all_total, 1) * 100, 2)
    result["image"]["Mean"] = round(mean_image_sum / len(all_cats), 2)
    result["point"]["Mean"] = round(mean_point_sum / len(all_cats), 2)

    save_dir = os.path.join("layers_cat", args.dit_model)
    os.makedirs(save_dir, exist_ok=True)
    method_tag = (
        f"sdr_sq{args.sdr_source_quantile}_cw{args.sdr_context_weight}_rw{args.sdr_rival_weight}"
        f"_cr{args.sdr_context_radius}_mr{args.sdr_max_rivals}"
    )
    result_path = os.path.join(save_dir, f"t{args.t}_b{args.k}_e{args.ensemble_size}_{method_tag}.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    print(f"Saved summary to: {result_path}")

    if args.save_point_records:
        records_dir = args.point_records_dir if args.point_records_dir else save_dir
        os.makedirs(records_dir, exist_ok=True)
        records_path = os.path.join(records_dir, f"per_point_records_{method_tag}.csv")
        save_point_records(records_path, point_records)
        print(f"Saved point records to: {records_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPair-71k Evaluation Script with Source Descriptor Refinement")
    parser.add_argument("--dataset_path", type=str, default="/dataset/SPair-71k", help="path to spair dataset")
    parser.add_argument("--dataset", type=str, default="SPair", help="dataset name")
    parser.add_argument("--save_path", type=str, default="/scratch/lt453/spair_ft/", help="path to save features")
    parser.add_argument("--dit_model", choices=["flux"], default="flux", help="which dit version to use")
    parser.add_argument("--img_size", nargs="+", type=int, default=[768, 768], help="input resize [w, h]")
    parser.add_argument("--t", default=260, type=int, help="diffusion timestep")
    parser.add_argument("--k", nargs="+", type=int, default=[28], help="which dit block to extract the ft map")
    parser.add_argument("--ensemble_size", default=8, type=int, help="ensemble size for getting an image ft map")
    parser.add_argument("--cd", action="store_true", default=False, help="whether adopt channel discard")
    parser.add_argument("--reuse_saved_features", action="store_true", default=False, help="skip feature extraction and reuse saved .pth features")
    parser.add_argument("--max_pairs_per_cat", default=0, type=int, help="optional category-wise cap for quick validation")

    parser.add_argument("--shift_calibration", action="store_true", default=False, help="apply training-free post-AdaLN shift calibration before matching")
    parser.add_argument("--shift_calibration_quantile", default=0.75, type=float, help="only suppress tokens above this shift-ratio quantile")
    parser.add_argument("--shift_calibration_strength", default=0.5, type=float, help="suppression strength for high-shift tokens")
    parser.add_argument("--shift_calibration_min_lambda", default=0.2, type=float, help="minimum retained shift scaling after calibration")
    parser.add_argument("--joint_calibration", action="store_true", default=False, help="apply joint high-shift low-content calibration before matching")
    parser.add_argument("--joint_shift_quantile", default=0.75, type=float, help="high-shift token threshold for joint calibration")
    parser.add_argument("--joint_content_quantile", default=0.25, type=float, help="low-content token threshold for joint calibration")
    parser.add_argument("--joint_shift_strength", default=0.5, type=float, help="shift suppression strength under joint calibration")
    parser.add_argument("--joint_min_shift_lambda", default=0.2, type=float, help="minimum retained shift scaling under joint calibration")
    parser.add_argument("--joint_content_strength", default=0.25, type=float, help="content amplification strength under joint calibration")
    parser.add_argument("--joint_max_content_gain", default=1.5, type=float, help="upper bound for content amplification under joint calibration")

    parser.add_argument("--sdr_source_quantile", default=0.7, type=float, help="source-side similarity quantile for forming rival weights")
    parser.add_argument("--sdr_max_rivals", default=4, type=int, help="maximum number of source rivals retained per point")
    parser.add_argument("--sdr_rival_floor", default=0.05, type=float, help="minimum rival weight retained after source-side pruning")
    parser.add_argument("--sdr_src_distance_sigma", default=0.6, type=float, help="distance decay for source-side rival weighting")
    parser.add_argument("--sdr_context_radius", default=2, type=int, help="source-side local context radius used in descriptor refinement")
    parser.add_argument("--sdr_context_min_neighbors", default=4, type=int, help="minimum valid local neighbors required to build a context descriptor")
    parser.add_argument("--sdr_context_weight", default=0.2, type=float, help="weight of the source-side local context descriptor")
    parser.add_argument("--sdr_rival_weight", default=0.35, type=float, help="weight used to subtract the rival prototype from the source descriptor")
    parser.add_argument("--sdr_margin_quantile", default=0.4, type=float, help="low-margin quantile used to define local ambiguity risk")
    parser.add_argument("--sdr_score_quantile", default=0.4, type=float, help="low top-1 score quantile used in the point risk")
    parser.add_argument("--sdr_ambiguity_quantile", default=0.7, type=float, help="high source-ambiguity quantile used in the point risk")
    parser.add_argument("--sdr_scale_weight", default=0.3, type=float, help="extra global risk contribution from pair-level scale variation")

    parser.add_argument("--save_point_records", action="store_true", default=False, help="save per-point records for downstream mechanism checking")
    parser.add_argument("--point_records_dir", type=str, default="", help="optional output directory for per-point records")

    args = parser.parse_args()
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main(args)

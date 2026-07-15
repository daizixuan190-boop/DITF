import argparse
import csv
import gc
import json
import os
import warnings

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F
from tqdm import tqdm

from eval_spair_confusion_rerank import (
    build_pair_candidate_records,
    maybe_apply_shift_calibration,
    quantile_risk_map,
)
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


def prepare_source_identity_stats(src_ft: torch.Tensor, src_points: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    if not src_points:
        return torch.empty(0, 2, device=src_ft.device), torch.empty(0, 0, device=src_ft.device)

    channels = src_ft.shape[1]
    src_xy = torch.tensor(
        [(int(point[0]), int(point[1])) for point in src_points],
        device=src_ft.device,
        dtype=torch.long,
    )
    src_vecs = src_ft[0, :, src_xy[:, 1], src_xy[:, 0]].transpose(0, 1).contiguous().float()
    src_vecs = F.normalize(src_vecs, dim=1)
    if int(src_vecs.shape[0]) <= 1:
        return src_xy.float(), torch.zeros(src_vecs.shape[0], device=src_ft.device, dtype=torch.float32)
    src_sim = torch.mm(src_vecs, src_vecs.transpose(0, 1))
    src_sim.fill_diagonal_(-1e4)
    max_other_sim, _ = torch.max(src_sim, dim=1)
    return src_xy.float(), max_other_sim


def normalize_candidate_record_order(records: list[dict]) -> list[dict]:
    normalized = []
    for record in records:
        order = torch.argsort(record["base_scores"], descending=True)
        sorted_scores = record["base_scores"][order]
        sorted_x = record["cand_x"][order]
        sorted_y = record["cand_y"][order]
        if int(sorted_scores.numel()) > 1:
            margin = float((sorted_scores[0] - sorted_scores[1]).item())
        elif int(sorted_scores.numel()) == 1:
            margin = float(sorted_scores[0].item())
        else:
            margin = 0.0
        normalized.append(
            {
                "src_xy": record["src_xy"],
                "cand_x": sorted_x,
                "cand_y": sorted_y,
                "base_scores": sorted_scores,
                "anchor_idx": 0 if int(sorted_scores.numel()) > 0 else -1,
                "margin": margin,
            }
        )
    return normalized


def select_anchor_indices(
    margins: torch.Tensor,
    top1_scores: torch.Tensor,
    src_ambiguity: torch.Tensor,
    args,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    margin_risk = quantile_risk_map(margins, args.asr_margin_quantile, tail="low")
    score_risk = quantile_risk_map(top1_scores, args.asr_score_quantile, tail="low")
    ambiguity_risk = quantile_risk_map(src_ambiguity, args.asr_ambiguity_quantile, tail="high")

    base_risk = (margin_risk + score_risk + ambiguity_risk) / 3.0
    confidence = 1.0 - base_risk
    confidence = confidence.clamp(min=0.0, max=1.0)

    threshold = torch.quantile(confidence, float(args.asr_anchor_quantile))
    anchor_mask = (confidence >= threshold) & (margins > 0.0)

    if int(anchor_mask.sum().item()) < int(args.asr_min_anchors):
        k = min(int(args.asr_min_anchors), int(confidence.numel()))
        top_idx = torch.topk(confidence, k=k, dim=0).indices
        anchor_mask = torch.zeros_like(anchor_mask, dtype=torch.bool)
        anchor_mask[top_idx] = True

    anchor_indices = torch.nonzero(anchor_mask, as_tuple=False).squeeze(1)
    if int(anchor_indices.numel()) > int(args.asr_max_anchors):
        keep_order = torch.argsort(confidence[anchor_indices], descending=True)[: int(args.asr_max_anchors)]
        anchor_indices = anchor_indices[keep_order]
    return anchor_indices, confidence, base_risk


def anchor_support_rerank(
    records: list[dict],
    src_points: list[list[int]],
    src_threshold: float,
    trg_threshold: float,
    src_xy: torch.Tensor,
    src_ambiguity: torch.Tensor,
    pair_risk: float,
    args,
) -> tuple[list[tuple[int, int]], list[dict]]:
    num_points = len(records)
    if num_points == 0:
        return [], []

    device = records[0]["base_scores"].device
    margins = torch.tensor([float(record["margin"]) for record in records], device=device, dtype=torch.float32)
    top1_scores = torch.stack(
        [record["base_scores"][record["anchor_idx"]].float() for record in records],
        dim=0,
    )

    anchor_indices, anchor_confidence, base_risk = select_anchor_indices(
        margins,
        top1_scores,
        src_ambiguity,
        args,
    )
    pair_risk_tensor = torch.full_like(base_risk, float(pair_risk))
    risk_weight = (base_risk + float(args.asr_scale_weight) * pair_risk_tensor).clamp(min=0.0, max=1.0)

    anchor_xy = torch.stack(
        [
            torch.tensor(
                [
                    float(records[int(anchor_idx)]["cand_x"][records[int(anchor_idx)]["anchor_idx"]].item()),
                    float(records[int(anchor_idx)]["cand_y"][records[int(anchor_idx)]["anchor_idx"]].item()),
                ],
                device=device,
                dtype=torch.float32,
            )
            for anchor_idx in anchor_indices
        ],
        dim=0,
    )
    anchor_weights = anchor_confidence[anchor_indices].clamp_min(1e-6)

    src_pair_dists = torch.cdist(src_xy, src_xy) / max(float(src_threshold), 1e-6)
    predictions: list[tuple[int, int]] = []
    diagnostics: list[dict] = []

    for point_idx, record in enumerate(records):
        cand_x = record["cand_x"].float()
        cand_y = record["cand_y"].float()
        cand_xy = torch.stack([cand_x, cand_y], dim=1)
        base_scores = record["base_scores"].float()

        other_anchor_mask = anchor_indices != point_idx
        point_anchor_indices = anchor_indices[other_anchor_mask]
        if int(point_anchor_indices.numel()) == 0:
            best_idx = int(torch.argmax(base_scores).item())
            predictions.append((int(record["cand_x"][best_idx].item()), int(record["cand_y"][best_idx].item())))
            diagnostics.append(
                {
                    "point_idx": point_idx,
                    "risk_weight": float(risk_weight[point_idx].item()),
                    "base_risk": float(base_risk[point_idx].item()),
                    "anchor_count": 0,
                    "source_ambiguity": float(src_ambiguity[point_idx].item()),
                    "margin": float(margins[point_idx].item()),
                    "base_top1_score": float(top1_scores[point_idx].item()),
                    "best_support_bonus": 0.0,
                    "best_collision_penalty": 0.0,
                    "best_final_score": float(base_scores[best_idx].item()),
                    "base_choice_idx": int(record["anchor_idx"]),
                    "final_choice_idx": best_idx,
                }
            )
            continue

        current_anchor_xy = anchor_xy[other_anchor_mask]
        current_anchor_weights = anchor_weights[other_anchor_mask]
        src_rel = src_pair_dists[point_idx, point_anchor_indices]
        valid_anchor_mask = src_rel >= float(args.asr_min_anchor_separation)
        if int(valid_anchor_mask.sum().item()) == 0:
            valid_anchor_mask = torch.ones_like(src_rel, dtype=torch.bool)

        current_anchor_xy = current_anchor_xy[valid_anchor_mask]
        current_anchor_weights = current_anchor_weights[valid_anchor_mask]
        src_rel = src_rel[valid_anchor_mask]

        trg_rel = torch.cdist(cand_xy, current_anchor_xy) / max(float(trg_threshold), 1e-6)
        delta = torch.abs(trg_rel - src_rel.unsqueeze(0))

        geom_support = torch.exp(-delta / max(float(args.asr_distance_sigma), 1e-6))
        support_bonus = torch.sum(
            geom_support * current_anchor_weights.unsqueeze(0),
            dim=1,
        ) / current_anchor_weights.sum().clamp_min(1e-6)
        support_bonus = support_bonus - support_bonus.mean()

        support_residual = torch.sum(
            delta * current_anchor_weights.unsqueeze(0),
            dim=1,
        ) / current_anchor_weights.sum().clamp_min(1e-6)
        residual_advantage = support_residual.mean() - support_residual

        overlap = (1.0 - trg_rel / float(args.asr_collision_radius)).clamp(min=0.0)
        collision_penalty = torch.sum(
            overlap * current_anchor_weights.unsqueeze(0),
            dim=1,
        ) / current_anchor_weights.sum().clamp_min(1e-6)
        collision_penalty = collision_penalty - collision_penalty.mean()

        final_scores = base_scores + risk_weight[point_idx] * (
            float(args.asr_support_weight) * support_bonus
            + float(args.asr_contrast_weight) * residual_advantage
            - float(args.asr_collision_weight) * collision_penalty
        )

        best_idx = int(torch.argmax(final_scores).item())
        predictions.append((int(record["cand_x"][best_idx].item()), int(record["cand_y"][best_idx].item())))
        diagnostics.append(
            {
                "point_idx": point_idx,
                "risk_weight": float(risk_weight[point_idx].item()),
                "base_risk": float(base_risk[point_idx].item()),
                "anchor_count": int(current_anchor_weights.numel()),
                "source_ambiguity": float(src_ambiguity[point_idx].item()),
                "margin": float(margins[point_idx].item()),
                "base_top1_score": float(top1_scores[point_idx].item()),
                "best_support_bonus": float(support_bonus[best_idx].item()),
                "best_collision_penalty": float(collision_penalty[best_idx].item()),
                "best_final_score": float(final_scores[best_idx].item()),
                "base_choice_idx": int(record["anchor_idx"]),
                "final_choice_idx": best_idx,
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
            src_bndbox = data["src_bndbox"]
            trg_bndbox = data["trg_bndbox"]
            src_threshold = max(src_bndbox[3] - src_bndbox[1], src_bndbox[2] - src_bndbox[0])
            trg_threshold = max(trg_bndbox[3] - trg_bndbox[1], trg_bndbox[2] - trg_bndbox[0])

            pair_records = build_pair_candidate_records(
                src_ft,
                trg_ft,
                src_points,
                src_threshold,
                args.asr_topk,
                args.asr_base_reverse_weight,
            )
            pair_records = normalize_candidate_record_order(pair_records)
            src_xy, src_ambiguity = prepare_source_identity_stats(src_ft, src_points)
            predictions, diagnostics = anchor_support_rerank(
                pair_records,
                src_points,
                src_threshold,
                trg_threshold,
                src_xy,
                src_ambiguity,
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
                            "src_threshold": float(src_threshold),
                            "trg_threshold": float(trg_threshold),
                            "scale_variation": data.get("scale_variation"),
                            "risk_weight": diag["risk_weight"],
                            "base_risk": diag["base_risk"],
                            "anchor_count": diag["anchor_count"],
                            "source_ambiguity": diag["source_ambiguity"],
                            "margin": diag["margin"],
                            "base_top1_score": diag["base_top1_score"],
                            "best_support_bonus": diag["best_support_bonus"],
                            "best_collision_penalty": diag["best_collision_penalty"],
                            "best_final_score": diag["best_final_score"],
                            "base_choice_idx": diag["base_choice_idx"],
                            "final_choice_idx": diag["final_choice_idx"],
                            "method_tag": "anchor_support",
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
        f"asr_topk{args.asr_topk}_aq{args.asr_anchor_quantile}_sw{args.asr_support_weight}"
        f"_cw{args.asr_collision_weight}_ctw{args.asr_contrast_weight}_ma{args.asr_max_anchors}"
        f"_rw{args.asr_base_reverse_weight}"
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
    parser = argparse.ArgumentParser(description="SPair-71k Evaluation Script with Anchor-Support Rerank")
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

    parser.add_argument("--asr_topk", default=5, type=int, help="number of local target candidates retained per source point")
    parser.add_argument("--asr_base_reverse_weight", default=0.1, type=float, help="reverse-cycle penalty weight used in baseline candidate scoring")
    parser.add_argument("--asr_anchor_quantile", default=0.7, type=float, help="confidence quantile for selecting support anchors")
    parser.add_argument("--asr_min_anchors", default=2, type=int, help="minimum number of support anchors kept per pair")
    parser.add_argument("--asr_max_anchors", default=6, type=int, help="maximum number of support anchors used per pair")
    parser.add_argument("--asr_margin_quantile", default=0.4, type=float, help="low-margin quantile used to define local ambiguity risk")
    parser.add_argument("--asr_score_quantile", default=0.4, type=float, help="low top-1 score quantile used in the point risk")
    parser.add_argument("--asr_ambiguity_quantile", default=0.7, type=float, help="high source-ambiguity quantile used in the point risk")
    parser.add_argument("--asr_scale_weight", default=0.25, type=float, help="extra global risk contribution from pair-level scale variation")
    parser.add_argument("--asr_support_weight", default=0.2, type=float, help="weight of the anchor-support bonus")
    parser.add_argument("--asr_contrast_weight", default=0.2, type=float, help="weight of the contrastive residual advantage between top-k candidates")
    parser.add_argument("--asr_collision_weight", default=0.1, type=float, help="weight of the rival-basin collision penalty")
    parser.add_argument("--asr_distance_sigma", default=0.5, type=float, help="bandwidth for normalized source-target distance agreement")
    parser.add_argument("--asr_collision_radius", default=0.6, type=float, help="target-space collision radius normalized by target threshold")
    parser.add_argument("--asr_min_anchor_separation", default=0.6, type=float, help="minimum source-space separation for an anchor to support another point")

    parser.add_argument("--save_point_records", action="store_true", default=False, help="save per-point records for downstream mechanism checking")
    parser.add_argument("--point_records_dir", type=str, default="", help="optional output directory for per-point records")

    args = parser.parse_args()
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main(args)

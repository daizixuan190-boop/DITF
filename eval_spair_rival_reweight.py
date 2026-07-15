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


def prepare_source_bundle(
    src_ft: torch.Tensor,
    src_points: list[list[int]],
    src_threshold: float,
    args,
) -> dict:
    device = src_ft.device
    if not src_points:
        empty_xy = torch.empty(0, 2, device=device, dtype=torch.float32)
        empty_vec = torch.empty(0, 0, device=device, dtype=torch.float32)
        return {
            "src_xy": empty_xy,
            "src_vecs": empty_vec,
            "src_ambiguity": torch.empty(0, device=device, dtype=torch.float32),
            "rival_weights": empty_vec,
        }

    src_xy = torch.tensor(
        [(int(point[0]), int(point[1])) for point in src_points],
        device=device,
        dtype=torch.long,
    )
    src_vecs = src_ft[0, :, src_xy[:, 1], src_xy[:, 0]].transpose(0, 1).contiguous().float()
    src_vecs = F.normalize(src_vecs, dim=1)
    num_points = int(src_vecs.shape[0])
    if num_points <= 1:
        return {
            "src_xy": src_xy.float(),
            "src_vecs": src_vecs,
            "src_ambiguity": torch.zeros(num_points, device=device, dtype=torch.float32),
            "rival_weights": torch.zeros(num_points, num_points, device=device, dtype=torch.float32),
        }

    src_sim = torch.mm(src_vecs, src_vecs.transpose(0, 1))
    src_sim.fill_diagonal_(-1e4)
    src_ambiguity, _ = torch.max(src_sim, dim=1)

    off_diag_mask = ~torch.eye(num_points, device=device, dtype=torch.bool)
    off_diag_vals = src_sim[off_diag_mask]
    sim_threshold = torch.quantile(off_diag_vals, float(args.rsr_source_quantile))
    sim_ref = off_diag_vals.max()
    sim_denom = (sim_ref - sim_threshold).clamp_min(1e-6)
    sim_risk = ((src_sim - sim_threshold) / sim_denom).clamp(min=0.0, max=1.0)
    sim_risk.fill_diagonal_(0.0)

    src_pair_dists = torch.cdist(src_xy.float(), src_xy.float()) / max(float(src_threshold), 1e-6)
    dist_gate = torch.exp(-src_pair_dists / max(float(args.rsr_src_distance_sigma), 1e-6))
    dist_gate.fill_diagonal_(0.0)

    rival_weights = sim_risk * dist_gate
    if int(args.rsr_max_rivals) > 0 and int(args.rsr_max_rivals) < num_points:
        top_vals, top_idx = torch.topk(
            rival_weights,
            k=int(args.rsr_max_rivals),
            dim=1,
        )
        masked_weights = torch.zeros_like(rival_weights)
        masked_weights.scatter_(1, top_idx, top_vals)
        rival_weights = masked_weights

    if float(args.rsr_rival_floor) > 0.0:
        rival_weights = torch.where(
            rival_weights >= float(args.rsr_rival_floor),
            rival_weights,
            torch.zeros_like(rival_weights),
        )

    return {
        "src_xy": src_xy.float(),
        "src_vecs": src_vecs,
        "src_ambiguity": src_ambiguity,
        "rival_weights": rival_weights,
    }


def build_risk_weights(records: list[dict], src_ambiguity: torch.Tensor, pair_risk: float, args) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = src_ambiguity.device
    margins = torch.tensor([float(record["margin"]) for record in records], device=device, dtype=torch.float32)
    top1_scores = torch.stack(
        [record["base_scores"][record["anchor_idx"]].float() for record in records],
        dim=0,
    )

    margin_risk = quantile_risk_map(margins, args.rsr_margin_quantile, tail="low")
    score_risk = quantile_risk_map(top1_scores, args.rsr_score_quantile, tail="low")
    ambiguity_risk = quantile_risk_map(src_ambiguity, args.rsr_ambiguity_quantile, tail="high")
    base_risk = (margin_risk + score_risk + ambiguity_risk) / 3.0

    risk_weight = base_risk + float(args.rsr_scale_weight) * float(pair_risk)
    risk_weight = risk_weight.clamp(min=0.0, max=1.0)
    return margins, top1_scores, risk_weight


def rival_score_reconstruct(
    records: list[dict],
    trg_ft: torch.Tensor,
    source_bundle: dict,
    pair_risk: float,
    args,
) -> tuple[list[tuple[int, int]], list[dict]]:
    num_points = len(records)
    if num_points == 0:
        return [], []

    src_vecs = source_bundle["src_vecs"]
    src_ambiguity = source_bundle["src_ambiguity"]
    rival_weights = source_bundle["rival_weights"]
    device = src_vecs.device

    margins, top1_scores, risk_weight = build_risk_weights(records, src_ambiguity, pair_risk, args)

    cand_x = torch.stack([record["cand_x"].long() for record in records], dim=1)
    cand_y = torch.stack([record["cand_y"].long() for record in records], dim=1)
    num_candidates = int(cand_x.shape[0])
    trg_vecs = trg_ft[0, :, cand_y.reshape(-1), cand_x.reshape(-1)].transpose(0, 1).contiguous().float()
    trg_vecs = F.normalize(trg_vecs, dim=1)
    cross_scores = torch.mm(trg_vecs, src_vecs.transpose(0, 1)).view(num_candidates, num_points, num_points)
    cross_scores = cross_scores.permute(1, 0, 2).contiguous()

    predictions: list[tuple[int, int]] = []
    diagnostics: list[dict] = []
    point_indices = torch.arange(num_points, device=device)

    for point_idx, record in enumerate(records):
        base_scores = record["base_scores"].float()
        point_cross_scores = cross_scores[point_idx]
        self_scores = point_cross_scores[:, point_idx]
        rival_weight_vec = rival_weights[point_idx]
        rival_count = int((rival_weight_vec > 0).sum().item())

        if rival_count <= 0:
            best_idx = int(torch.argmax(base_scores).item())
            predictions.append((int(record["cand_x"][best_idx].item()), int(record["cand_y"][best_idx].item())))
            diagnostics.append(
                {
                    "point_idx": point_idx,
                    "risk_weight": float(risk_weight[point_idx].item()),
                    "source_ambiguity": float(src_ambiguity[point_idx].item()),
                    "margin": float(margins[point_idx].item()),
                    "base_top1_score": float(top1_scores[point_idx].item()),
                    "rival_count": 0,
                    "best_self_score": float(self_scores[best_idx].item()),
                    "best_rival_score": float("-inf"),
                    "best_identity_margin": float("inf"),
                    "best_rival_idx": -1,
                    "best_final_score": float(base_scores[best_idx].item()),
                    "base_choice_idx": int(record["anchor_idx"]),
                    "final_choice_idx": best_idx,
                }
            )
            continue

        weighted_rival_scores = point_cross_scores * rival_weight_vec.unsqueeze(0)
        rival_scores, rival_idx = torch.max(weighted_rival_scores, dim=1)
        identity_margin = self_scores - rival_scores
        identity_margin_centered = identity_margin - identity_margin.mean()
        self_centered = self_scores - self_scores.mean()
        rival_penalty = F.relu(rival_scores - self_scores)
        rival_penalty_centered = rival_penalty - rival_penalty.mean()

        final_scores = base_scores + risk_weight[point_idx] * (
            float(args.rsr_self_weight) * self_centered
            + float(args.rsr_margin_weight) * identity_margin_centered
            - float(args.rsr_rival_weight) * rival_penalty_centered
        )
        best_idx = int(torch.argmax(final_scores).item())
        best_rival_src_idx = int(rival_idx[best_idx].item())
        if best_rival_src_idx == point_idx:
            masked_weights = rival_weight_vec.clone()
            masked_weights[point_indices == point_idx] = 0.0
            if float(masked_weights.max().item()) > 0.0:
                masked_scores = point_cross_scores[best_idx] * masked_weights
                best_rival_src_idx = int(torch.argmax(masked_scores).item())
            else:
                best_rival_src_idx = -1

        predictions.append((int(record["cand_x"][best_idx].item()), int(record["cand_y"][best_idx].item())))
        diagnostics.append(
            {
                "point_idx": point_idx,
                "risk_weight": float(risk_weight[point_idx].item()),
                "source_ambiguity": float(src_ambiguity[point_idx].item()),
                "margin": float(margins[point_idx].item()),
                "base_top1_score": float(top1_scores[point_idx].item()),
                "rival_count": rival_count,
                "best_self_score": float(self_scores[best_idx].item()),
                "best_rival_score": float(rival_scores[best_idx].item()),
                "best_identity_margin": float(identity_margin[best_idx].item()),
                "best_rival_idx": best_rival_src_idx,
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
                args.rsr_topk,
                args.rsr_base_reverse_weight,
            )
            pair_records = normalize_candidate_record_order(pair_records)
            source_bundle = prepare_source_bundle(src_ft, src_points, src_threshold, args)
            predictions, diagnostics = rival_score_reconstruct(
                pair_records,
                trg_ft,
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
                            "source_ambiguity": diag["source_ambiguity"],
                            "margin": diag["margin"],
                            "base_top1_score": diag["base_top1_score"],
                            "rival_count": diag["rival_count"],
                            "best_self_score": diag["best_self_score"],
                            "best_rival_score": diag["best_rival_score"],
                            "best_identity_margin": diag["best_identity_margin"],
                            "best_rival_idx": diag["best_rival_idx"],
                            "best_final_score": diag["best_final_score"],
                            "base_choice_idx": diag["base_choice_idx"],
                            "final_choice_idx": diag["final_choice_idx"],
                            "method_tag": "rival_score_reconstruct",
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
        f"rsr_topk{args.rsr_topk}_sq{args.rsr_source_quantile}_mw{args.rsr_margin_weight}"
        f"_sw{args.rsr_self_weight}_rw{args.rsr_rival_weight}_mr{args.rsr_max_rivals}"
        f"_brw{args.rsr_base_reverse_weight}"
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
    parser = argparse.ArgumentParser(description="SPair-71k Evaluation Script with Rival-Aware Score Reconstruction")
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

    parser.add_argument("--rsr_topk", default=5, type=int, help="number of local target candidates retained per source point")
    parser.add_argument("--rsr_base_reverse_weight", default=0.1, type=float, help="reverse-cycle penalty weight used in baseline candidate scoring")
    parser.add_argument("--rsr_source_quantile", default=0.7, type=float, help="source-side similarity quantile for forming rival weights")
    parser.add_argument("--rsr_max_rivals", default=4, type=int, help="maximum number of source rivals retained per point")
    parser.add_argument("--rsr_rival_floor", default=0.05, type=float, help="minimum rival weight retained after source-side pruning")
    parser.add_argument("--rsr_src_distance_sigma", default=0.6, type=float, help="distance decay for source-side rival weighting")
    parser.add_argument("--rsr_margin_quantile", default=0.4, type=float, help="low-margin quantile used to define local ambiguity risk")
    parser.add_argument("--rsr_score_quantile", default=0.4, type=float, help="low top-1 score quantile used in the point risk")
    parser.add_argument("--rsr_ambiguity_quantile", default=0.7, type=float, help="high source-ambiguity quantile used in the point risk")
    parser.add_argument("--rsr_scale_weight", default=0.3, type=float, help="extra global risk contribution from pair-level scale variation")
    parser.add_argument("--rsr_self_weight", default=0.15, type=float, help="reward for candidates that preserve stronger self identity score")
    parser.add_argument("--rsr_margin_weight", default=0.35, type=float, help="reward for candidates with larger self-vs-rival identity margin")
    parser.add_argument("--rsr_rival_weight", default=0.45, type=float, help="penalty for candidates more aligned with a rival source identity")

    parser.add_argument("--save_point_records", action="store_true", default=False, help="save per-point records for downstream mechanism checking")
    parser.add_argument("--point_records_dir", type=str, default="", help="optional output directory for per-point records")

    args = parser.parse_args()
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main(args)

"""Evaluate training-free relational candidate ownership on SPair-71k.

The method preserves the DiTF post-AdaLN + optional channel discard baseline.
It builds a shared target candidate union across source keypoints and uses only
label-free high-confidence matches as anchors. Target GT is used exclusively in
the outer evaluation loop and never enters method scoring or gating.
"""

import argparse
import csv
import json
import os
import warnings
from typing import Any

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
    feature = ft_raw.clone()
    if args.cd:
        feature[:, 154, :, :] = 0.0
        feature[:, 1446, :, :] = 0.0
    _, _, height, width = feature.shape
    feature = rearrange(feature, "b c h w -> b (h w) c")
    feature = pre_norm(feature)
    feature = rearrange(feature, "b (h w) c -> b c h w", h=height, w=width)
    shift = ada[0][0].unsqueeze(0).unsqueeze(2).unsqueeze(3)
    scale = ada[0][1].unsqueeze(0).unsqueeze(2).unsqueeze(3)
    return maybe_apply_shift_calibration(feature, shift, scale, args)


def normalize_candidate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for record in records:
        order = torch.argsort(record["base_scores"], descending=True)
        scores = record["base_scores"][order].float()
        cand_x = record["cand_x"][order].long()
        cand_y = record["cand_y"][order].long()
        margin = float((scores[0] - scores[1]).item()) if int(scores.numel()) > 1 else 0.0
        normalized.append(
            {
                "src_xy": record["src_xy"],
                "cand_x": cand_x,
                "cand_y": cand_y,
                "base_scores": scores,
                "anchor_idx": 0,
                "margin": margin,
            }
        )
    return normalized


def build_forward_candidate_records(
    src_ft: torch.Tensor,
    trg_ft: torch.Tensor,
    src_points: list[list[int]],
    topk: int,
) -> list[dict[str, Any]]:
    """Build forward-only candidates without the unused reverse-cycle matrix."""
    channels = int(src_ft.shape[1])
    target_width = int(trg_ft.shape[-1])
    target_matrix = trg_ft.view(channels, -1).transpose(0, 1).contiguous().float()
    target_matrix = F.normalize(target_matrix, dim=1)
    src_xy = torch.tensor(
        [(int(point[0]), int(point[1])) for point in src_points],
        device=src_ft.device,
        dtype=torch.long,
    )
    src_vectors = src_ft[0, :, src_xy[:, 1], src_xy[:, 0]].transpose(0, 1).contiguous().float()
    src_vectors = F.normalize(src_vectors, dim=1)
    score_matrix = target_matrix @ src_vectors.transpose(0, 1)
    k = min(max(int(topk), 1), int(score_matrix.shape[0]))
    top_values, top_indices = torch.topk(score_matrix, k=k, dim=0)
    candidate_y = torch.div(top_indices, target_width, rounding_mode="floor")
    candidate_x = top_indices % target_width
    records = []
    for point_idx, point in enumerate(src_points):
        scores = top_values[:, point_idx].float()
        margin = float((scores[0] - scores[1]).item()) if k > 1 else 0.0
        records.append(
            {
                "src_xy": (int(point[0]), int(point[1])),
                "cand_x": candidate_x[:, point_idx],
                "cand_y": candidate_y[:, point_idx],
                "base_scores": scores,
                "anchor_idx": 0,
                "margin": margin,
            }
        )
    return records


def build_shared_candidate_union(
    records: list[dict[str, Any]],
    src_ft: torch.Tensor,
    trg_ft: torch.Tensor,
    src_points: list[list[int]],
) -> dict[str, torch.Tensor]:
    device = src_ft.device
    target_width = int(trg_ft.shape[-1])
    flat_candidates = []
    for record in records:
        flat_candidates.append(record["cand_y"].long() * target_width + record["cand_x"].long())
    union_flat = torch.unique(torch.cat(flat_candidates, dim=0), sorted=True)
    union_y = torch.div(union_flat, target_width, rounding_mode="floor")
    union_x = union_flat % target_width

    src_xy = torch.tensor(
        [(int(point[0]), int(point[1])) for point in src_points],
        dtype=torch.long,
        device=device,
    )
    src_vectors = src_ft[0, :, src_xy[:, 1], src_xy[:, 0]].transpose(0, 1).contiguous().float()
    src_vectors = F.normalize(src_vectors, dim=1)
    trg_vectors = trg_ft[0, :, union_y, union_x].transpose(0, 1).contiguous().float()
    trg_vectors = F.normalize(trg_vectors, dim=1)
    raw_scores = src_vectors @ trg_vectors.transpose(0, 1)

    flat_to_union = {int(value): index for index, value in enumerate(union_flat.detach().cpu().tolist())}
    base_columns = []
    for record in records:
        flat_index = int(
            record["cand_y"][record["anchor_idx"]].item() * target_width
            + record["cand_x"][record["anchor_idx"]].item()
        )
        base_columns.append(flat_to_union[flat_index])

    return {
        "src_xy": src_xy.float(),
        "src_vectors": src_vectors,
        "union_flat": union_flat,
        "union_x": union_x.float(),
        "union_y": union_y.float(),
        "union_xy": torch.stack((union_x.float(), union_y.float()), dim=1),
        "raw_scores": raw_scores,
        "base_columns": torch.tensor(base_columns, dtype=torch.long, device=device),
    }


def strongest_other_scores(score_matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    num_points, num_candidates = score_matrix.shape
    if num_points <= 1:
        return (
            torch.full((num_points, num_candidates), -1.0, device=score_matrix.device),
            torch.full((num_points, num_candidates), -1, dtype=torch.long, device=score_matrix.device),
        )
    expanded = score_matrix.unsqueeze(0).expand(num_points, -1, -1).clone()
    point_idx = torch.arange(num_points, device=score_matrix.device)
    expanded[point_idx, point_idx, :] = -1e4
    return torch.max(expanded, dim=1)


def minmax_tail(values: torch.Tensor, quantile: float, tail: str) -> torch.Tensor:
    if int(values.numel()) <= 1 or float(values.max().item() - values.min().item()) < 1e-8:
        return torch.zeros_like(values)
    return quantile_risk_map(values, quantile, tail=tail)


def select_anchors_and_risk(
    margins: torch.Tensor,
    top1_exclusivity: torch.Tensor,
    args,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    margin_confidence = minmax_tail(margins, args.rco_anchor_margin_quantile, tail="high")
    exclusivity_confidence = minmax_tail(
        top1_exclusivity,
        args.rco_anchor_exclusivity_quantile,
        tail="high",
    )
    confidence = 0.5 * (margin_confidence + exclusivity_confidence)
    valid_anchor = (margins > 0.0) & (top1_exclusivity > 0.0)
    if int(valid_anchor.sum().item()) < int(args.rco_min_anchors):
        empty = torch.empty(0, dtype=torch.long, device=margins.device)
        risk = minmax_tail(top1_exclusivity, args.rco_risk_quantile, tail="low")
        return empty, confidence, risk

    valid_confidence = confidence[valid_anchor]
    threshold = torch.quantile(valid_confidence, float(args.rco_anchor_quantile))
    anchor_mask = valid_anchor & (confidence >= threshold)
    anchor_indices = torch.nonzero(anchor_mask, as_tuple=False).squeeze(1)
    if int(anchor_indices.numel()) < int(args.rco_min_anchors):
        valid_indices = torch.nonzero(valid_anchor, as_tuple=False).squeeze(1)
        order = torch.argsort(confidence[valid_indices], descending=True)
        anchor_indices = valid_indices[order[: int(args.rco_min_anchors)]]
    if int(anchor_indices.numel()) > int(args.rco_max_anchors):
        order = torch.argsort(confidence[anchor_indices], descending=True)
        anchor_indices = anchor_indices[order[: int(args.rco_max_anchors)]]

    risk = minmax_tail(top1_exclusivity, args.rco_risk_quantile, tail="low")
    return anchor_indices, confidence, risk


def relational_scores(
    src_xy: torch.Tensor,
    union_xy: torch.Tensor,
    base_columns: torch.Tensor,
    anchor_indices: torch.Tensor,
    anchor_confidence: torch.Tensor,
    src_threshold: float,
    trg_threshold: float,
    args,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_points = int(src_xy.shape[0])
    num_candidates = int(union_xy.shape[0])
    if int(anchor_indices.numel()) == 0:
        return (
            torch.zeros((num_points, num_candidates), device=src_xy.device),
            torch.zeros((num_points,), device=src_xy.device),
        )

    anchor_src_xy = src_xy[anchor_indices]
    anchor_trg_xy = union_xy[base_columns[anchor_indices]]
    src_rel = torch.cdist(src_xy, anchor_src_xy) / max(float(src_threshold), 1e-6)
    trg_rel = torch.cdist(union_xy, anchor_trg_xy) / max(float(trg_threshold), 1e-6)
    delta = torch.abs(src_rel.unsqueeze(1) - trg_rel.unsqueeze(0))
    support = torch.exp(-delta / max(float(args.rco_distance_sigma), 1e-6))

    weights = anchor_confidence[anchor_indices].clamp_min(1e-4)
    mask = torch.ones((num_points, int(anchor_indices.numel())), device=src_xy.device)
    for anchor_position, anchor_idx in enumerate(anchor_indices.tolist()):
        mask[int(anchor_idx), anchor_position] = 0.0
    separation_mask = (src_rel >= float(args.rco_min_anchor_separation)).float()
    effective = mask * torch.where(
        separation_mask.sum(dim=1, keepdim=True) > 0,
        separation_mask,
        torch.ones_like(separation_mask),
    )
    effective_weights = effective * weights.unsqueeze(0)
    denominator = effective_weights.sum(dim=1).clamp_min(1e-6)
    relation = torch.sum(support * effective_weights.unsqueeze(1), dim=2) / denominator.unsqueeze(1)
    anchor_counts = (effective_weights > 0).sum(dim=1).float()
    return relation, anchor_counts


def ownership_margins(relation: torch.Tensor) -> torch.Tensor:
    strongest_other, _ = strongest_other_scores(relation)
    return relation - strongest_other


def relational_candidate_ownership(
    records: list[dict[str, Any]],
    src_ft: torch.Tensor,
    trg_ft: torch.Tensor,
    src_points: list[list[int]],
    src_threshold: float,
    trg_threshold: float,
    args,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[dict[str, Any]], dict[str, torch.Tensor]]:
    bundle = build_shared_candidate_union(records, src_ft, trg_ft, src_points)
    raw_scores = bundle["raw_scores"].float()
    base_columns = bundle["base_columns"]
    num_points = int(raw_scores.shape[0])
    point_indices = torch.arange(num_points, device=raw_scores.device)

    margins = torch.tensor([float(record["margin"]) for record in records], device=raw_scores.device)
    strongest_raw_other, strongest_raw_other_idx = strongest_other_scores(raw_scores)
    top1_self = raw_scores[point_indices, base_columns]
    top1_other = strongest_raw_other[point_indices, base_columns]
    top1_exclusivity = top1_self - top1_other
    anchor_indices, anchor_confidence, risk = select_anchors_and_risk(margins, top1_exclusivity, args)
    relation, anchor_counts = relational_scores(
        bundle["src_xy"],
        bundle["union_xy"],
        base_columns,
        anchor_indices,
        anchor_confidence,
        src_threshold,
        trg_threshold,
        args,
    )
    relation_margin = ownership_margins(relation)
    base_relation = relation[point_indices, base_columns]
    base_relation_margin = relation_margin[point_indices, base_columns]
    relation_delta = relation - base_relation.unsqueeze(1)
    relation_margin_delta = relation_margin - base_relation_margin.unsqueeze(1)
    structural_delta = (
        float(args.rco_relation_weight) * relation_delta
        + float(args.rco_owner_margin_weight) * relation_margin_delta
    )
    final_scores = raw_scores + risk.unsqueeze(1) * structural_delta

    proposed_columns = torch.argmax(final_scores, dim=1)
    predictions: list[tuple[int, int]] = []
    baseline_predictions: list[tuple[int, int]] = []
    diagnostics: list[dict[str, Any]] = []
    anchor_set = {int(index) for index in anchor_indices.detach().cpu().tolist()}

    for point_idx in range(num_points):
        base_column = int(base_columns[point_idx].item())
        proposed_column = int(proposed_columns[point_idx].item())
        raw_gap = float((raw_scores[point_idx, base_column] - raw_scores[point_idx, proposed_column]).item())
        relation_gain = float(relation_delta[point_idx, proposed_column].item())
        owner_margin_gain = float(relation_margin_delta[point_idx, proposed_column].item())
        structural_gain = float(structural_delta[point_idx, proposed_column].item())
        gate = (
            proposed_column != base_column
            and int(anchor_counts[point_idx].item()) >= int(args.rco_min_point_anchors)
            and float(risk[point_idx].item()) >= float(args.rco_min_risk)
            and relation_gain >= float(args.rco_min_relation_gain)
            and structural_gain >= float(args.rco_min_structural_gain)
            and raw_gap <= float(args.rco_max_raw_gap)
            and (args.rco_modify_anchors or point_idx not in anchor_set)
        )
        final_column = proposed_column if gate else base_column
        base_xy = (
            int(bundle["union_x"][base_column].item()),
            int(bundle["union_y"][base_column].item()),
        )
        final_xy = (
            int(bundle["union_x"][final_column].item()),
            int(bundle["union_y"][final_column].item()),
        )
        baseline_predictions.append(base_xy)
        predictions.append(final_xy)
        diagnostics.append(
            {
                "point_idx": point_idx,
                "anchor_count": int(anchor_counts[point_idx].item()),
                "pair_anchor_count": int(anchor_indices.numel()),
                "is_anchor": int(point_idx in anchor_set),
                "anchor_confidence": float(anchor_confidence[point_idx].item()),
                "risk_weight": float(risk[point_idx].item()),
                "raw_margin": float(margins[point_idx].item()),
                "base_exclusivity": float(top1_exclusivity[point_idx].item()),
                "base_strongest_other_idx": int(strongest_raw_other_idx[point_idx, base_column].item()),
                "base_raw_score": float(raw_scores[point_idx, base_column].item()),
                "proposed_raw_score": float(raw_scores[point_idx, proposed_column].item()),
                "raw_gap": raw_gap,
                "base_relation_score": float(base_relation[point_idx].item()),
                "proposed_relation_score": float(relation[point_idx, proposed_column].item()),
                "relation_gain": relation_gain,
                "base_relation_margin": float(base_relation_margin[point_idx].item()),
                "proposed_relation_margin": float(relation_margin[point_idx, proposed_column].item()),
                "owner_margin_gain": owner_margin_gain,
                "structural_gain": structural_gain,
                "base_final_score": float(final_scores[point_idx, base_column].item()),
                "proposed_final_score": float(final_scores[point_idx, proposed_column].item()),
                "base_column": base_column,
                "proposed_column": proposed_column,
                "final_column": final_column,
                "changed": int(final_column != base_column),
                "gate_passed": int(gate),
            }
        )

    bundle.update(
        {
            "final_scores": final_scores,
            "relation_scores": relation,
            "relation_margins": relation_margin,
            "risk": risk,
            "anchor_indices": anchor_indices,
        }
    )
    return predictions, baseline_predictions, diagnostics, bundle


def best_valid_union_candidate(
    union_xy: torch.Tensor,
    scores: torch.Tensor,
    gt_point: list[int],
    pck_radius: float,
) -> tuple[int | None, int | None, float | None]:
    gt = torch.tensor(gt_point, device=union_xy.device, dtype=torch.float32)
    distances = torch.linalg.norm(union_xy - gt.unsqueeze(0), dim=1)
    valid = distances <= float(pck_radius)
    if int(valid.sum().item()) == 0:
        return None, None, None
    valid_indices = torch.nonzero(valid, as_tuple=False).squeeze(1)
    valid_scores = scores[valid_indices]
    best_local = int(torch.argmax(valid_scores).item())
    candidate_index = int(valid_indices[best_local].item())
    candidate_score = float(scores[candidate_index].item())
    rank = 1 + int((scores > candidate_score).sum().item())
    return candidate_index, rank, candidate_score


def save_records(path: str, records: list[dict[str, Any]]):
    if not records:
        return
    fields = sorted({key for record in records for key in record})
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main(args):
    for name, value in vars(args).items():
        if value is not None:
            print(f"{name}: {value}")

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    all_cats, cat2json, cat2img = collect_spair_lists(args.dataset_path)
    maybe_extract_features(args, all_cats, cat2img)
    os.makedirs(args.save_path, exist_ok=True)
    if args.reuse_saved_features:
        print("Reusing existing saved features; skip feature extraction.")

    pre_norm = nn.LayerNorm(3072, elementwise_affine=False, eps=1e-6)
    test_path = os.path.join(args.dataset_path, "PairAnnotation", "test")
    total_baseline_correct = 0
    total_method_correct = 0
    total_points = 0
    total_improvements = 0
    total_harms = 0
    total_changed = 0
    baseline_image_pck: list[float] = []
    method_image_pck: list[float] = []
    mean_baseline_image_sum = 0.0
    mean_method_image_sum = 0.0
    mean_baseline_point_sum = 0.0
    mean_method_point_sum = 0.0
    point_records: list[dict[str, Any]] = []
    result = {"baseline_image": {}, "baseline_point": {}, "method_image": {}, "method_point": {}}

    print(f"Category numbers: {len(all_cats)}")
    for category in all_cats:
        pair_names = list(cat2json[category])
        if args.max_pairs_per_cat > 0:
            pair_names = pair_names[: args.max_pairs_per_cat]
        output_dict = torch.load(os.path.join(args.save_path, f"{category}.pth"))
        ada_dict = torch.load(os.path.join(args.save_path, f"{category}_ada.pth"))
        category_baseline_correct = 0
        category_method_correct = 0
        category_total = 0
        category_baseline_image: list[float] = []
        category_method_image: list[float] = []

        for json_name in tqdm(pair_names):
            with open(os.path.join(test_path, json_name), "r", encoding="utf-8") as handle:
                data = json.load(handle)
            src_size = data["src_imsize"][:2][::-1]
            trg_size = data["trg_imsize"][:2][::-1]
            src_post = build_post_feature(
                output_dict[data["src_imname"]].cuda(),
                ada_dict[data["src_imname"]].cuda(),
                pre_norm,
                args,
            )
            trg_post = build_post_feature(
                output_dict[data["trg_imname"]].cuda(),
                ada_dict[data["trg_imname"]].cuda(),
                pre_norm,
                args,
            )
            src_ft = nn.Upsample(size=src_size, mode="bilinear")(src_post).to(torch.float16)
            trg_ft = nn.Upsample(size=trg_size, mode="bilinear")(trg_post).to(torch.float16)
            src_points = data["src_kps"]
            trg_points = data["trg_kps"]
            src_bbox = data["src_bndbox"]
            trg_bbox = data["trg_bndbox"]
            src_threshold = max(src_bbox[3] - src_bbox[1], src_bbox[2] - src_bbox[0])
            trg_threshold = max(trg_bbox[3] - trg_bbox[1], trg_bbox[2] - trg_bbox[0])

            if float(args.rco_base_reverse_weight) == 0.0:
                candidate_records = build_forward_candidate_records(
                    src_ft,
                    trg_ft,
                    src_points,
                    args.rco_topk,
                )
            else:
                candidate_records = build_pair_candidate_records(
                    src_ft,
                    trg_ft,
                    src_points,
                    src_threshold,
                    args.rco_topk,
                    args.rco_base_reverse_weight,
                )
            candidate_records = normalize_candidate_records(candidate_records)
            predictions, baseline_predictions, diagnostics, bundle = relational_candidate_ownership(
                candidate_records,
                src_ft,
                trg_ft,
                src_points,
                src_threshold,
                trg_threshold,
                args,
            )

            pair_baseline_correct = 0
            pair_method_correct = 0
            pck_radius = 0.1 * float(trg_threshold)
            pair_name = os.path.splitext(json_name)[0]
            for point_idx, (method_xy, baseline_xy, diag) in enumerate(
                zip(predictions, baseline_predictions, diagnostics)
            ):
                gt_point = trg_points[point_idx]
                baseline_dist = float(
                    np.hypot(baseline_xy[0] - gt_point[0], baseline_xy[1] - gt_point[1])
                )
                method_dist = float(np.hypot(method_xy[0] - gt_point[0], method_xy[1] - gt_point[1]))
                baseline_correct = int(baseline_dist / max(float(trg_threshold), 1e-6) <= 0.1)
                method_correct = int(method_dist / max(float(trg_threshold), 1e-6) <= 0.1)
                improvement = int(baseline_correct == 0 and method_correct == 1)
                harm = int(baseline_correct == 1 and method_correct == 0)
                pair_baseline_correct += baseline_correct
                pair_method_correct += method_correct
                category_baseline_correct += baseline_correct
                category_method_correct += method_correct
                category_total += 1
                total_baseline_correct += baseline_correct
                total_method_correct += method_correct
                total_points += 1
                total_improvements += improvement
                total_harms += harm
                total_changed += int(diag["changed"])

                raw_gt_index, raw_gt_rank, raw_gt_score = best_valid_union_candidate(
                    bundle["union_xy"],
                    bundle["raw_scores"][point_idx],
                    gt_point,
                    pck_radius,
                )
                final_gt_index, final_gt_rank, final_gt_score = best_valid_union_candidate(
                    bundle["union_xy"],
                    bundle["final_scores"][point_idx],
                    gt_point,
                    pck_radius,
                )
                gt_relation = None
                gt_relation_margin = None
                gt_overlaps_other = None
                if raw_gt_index is not None:
                    gt_relation = float(bundle["relation_scores"][point_idx, raw_gt_index].item())
                    gt_relation_margin = float(bundle["relation_margins"][point_idx, raw_gt_index].item())
                    candidate_xy = bundle["union_xy"][raw_gt_index]
                    other_gt = [
                        target_point
                        for target_idx, target_point in enumerate(trg_points)
                        if target_idx != point_idx
                    ]
                    if other_gt:
                        other_tensor = torch.tensor(other_gt, device=candidate_xy.device, dtype=torch.float32)
                        gt_overlaps_other = int(
                            bool(torch.any(torch.linalg.norm(other_tensor - candidate_xy.unsqueeze(0), dim=1) <= pck_radius))
                        )
                    else:
                        gt_overlaps_other = 0

                if args.save_point_records:
                    point_records.append(
                        {
                            "category": category,
                            "pair_name": pair_name,
                            "src_imname": data["src_imname"],
                            "trg_imname": data["trg_imname"],
                            "kp_idx": point_idx,
                            "src_x": int(src_points[point_idx][0]),
                            "src_y": int(src_points[point_idx][1]),
                            "trg_x": int(gt_point[0]),
                            "trg_y": int(gt_point[1]),
                            "baseline_pred_x": int(baseline_xy[0]),
                            "baseline_pred_y": int(baseline_xy[1]),
                            "method_pred_x": int(method_xy[0]),
                            "method_pred_y": int(method_xy[1]),
                            "baseline_norm_dist": baseline_dist / max(float(trg_threshold), 1e-6),
                            "method_norm_dist": method_dist / max(float(trg_threshold), 1e-6),
                            "baseline_correct": baseline_correct,
                            "method_correct": method_correct,
                            "improvement": improvement,
                            "harm": harm,
                            "gt_union_available": int(raw_gt_index is not None),
                            "raw_gt_union_rank": raw_gt_rank,
                            "final_gt_union_rank": final_gt_rank,
                            "raw_gt_union_score": raw_gt_score,
                            "final_gt_union_score": final_gt_score,
                            "gt_relation_score": gt_relation,
                            "gt_relation_margin": gt_relation_margin,
                            "gt_candidate_overlaps_other_pck": gt_overlaps_other,
                            "scale_variation": data.get("scale_variation"),
                            "viewpoint_variation": data.get("viewpoint_variation"),
                            "occlusion": data.get("occlusion"),
                            "truncation": data.get("truncation"),
                            **diag,
                            "method_tag": "relational_candidate_ownership",
                        }
                    )

            category_baseline_image.append(pair_baseline_correct / max(len(src_points), 1))
            category_method_image.append(pair_method_correct / max(len(src_points), 1))
            torch.cuda.empty_cache()

        baseline_image_pck.extend(category_baseline_image)
        method_image_pck.extend(category_method_image)
        baseline_image_score = float(np.mean(category_baseline_image)) if category_baseline_image else 0.0
        method_image_score = float(np.mean(category_method_image)) if category_method_image else 0.0
        baseline_point_score = category_baseline_correct / max(category_total, 1)
        method_point_score = category_method_correct / max(category_total, 1)
        mean_baseline_image_sum += baseline_image_score * 100
        mean_method_image_sum += method_image_score * 100
        mean_baseline_point_sum += baseline_point_score * 100
        mean_method_point_sum += method_point_score * 100
        result["baseline_image"][category] = round(baseline_image_score * 100, 2)
        result["method_image"][category] = round(method_image_score * 100, 2)
        result["baseline_point"][category] = round(baseline_point_score * 100, 2)
        result["method_point"][category] = round(method_point_score * 100, 2)
        print(
            f"{category}: baseline image={baseline_image_score * 100:.2f} method image={method_image_score * 100:.2f} "
            f"baseline point={baseline_point_score * 100:.2f} method point={method_point_score * 100:.2f}"
        )

    baseline_all_image = float(np.mean(baseline_image_pck)) if baseline_image_pck else 0.0
    method_all_image = float(np.mean(method_image_pck)) if method_image_pck else 0.0
    baseline_all_point = total_baseline_correct / max(total_points, 1)
    method_all_point = total_method_correct / max(total_points, 1)
    print(f"Baseline All per image PCK@0.1: {baseline_all_image * 100:.2f}")
    print(f"Method All per image PCK@0.1: {method_all_image * 100:.2f}")
    print(f"Baseline All per point PCK@0.1: {baseline_all_point * 100:.2f}")
    print(f"Method All per point PCK@0.1: {method_all_point * 100:.2f}")
    print(f"Point delta: {(method_all_point - baseline_all_point) * 100:.4f}")
    print(
        "Mechanism outcomes:",
        {
            "changed_rate": total_changed / max(total_points, 1),
            "improvement_rate": total_improvements / max(total_points, 1),
            "harm_rate": total_harms / max(total_points, 1),
            "net_gain": (total_improvements - total_harms) / max(total_points, 1),
        },
    )

    result["summary"] = {
        "baseline_all_image": baseline_all_image * 100,
        "method_all_image": method_all_image * 100,
        "baseline_all_point": baseline_all_point * 100,
        "method_all_point": method_all_point * 100,
        "point_delta": (method_all_point - baseline_all_point) * 100,
        "changed_rate": total_changed / max(total_points, 1),
        "improvement_rate": total_improvements / max(total_points, 1),
        "harm_rate": total_harms / max(total_points, 1),
        "net_gain": (total_improvements - total_harms) / max(total_points, 1),
        "mean_baseline_image": mean_baseline_image_sum / max(len(all_cats), 1),
        "mean_method_image": mean_method_image_sum / max(len(all_cats), 1),
        "mean_baseline_point": mean_baseline_point_sum / max(len(all_cats), 1),
        "mean_method_point": mean_method_point_sum / max(len(all_cats), 1),
    }

    output_dir = args.point_records_dir if args.point_records_dir else os.path.join("layers_cat", args.dit_model)
    os.makedirs(output_dir, exist_ok=True)
    method_tag = (
        f"rco_topk{args.rco_topk}_aq{args.rco_anchor_quantile}_rw{args.rco_relation_weight}"
        f"_ow{args.rco_owner_margin_weight}_sg{args.rco_min_structural_gain}"
    )
    result_path = os.path.join(output_dir, f"t{args.t}_b{args.k}_e{args.ensemble_size}_{method_tag}.json")
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(f"Saved summary to: {result_path}")
    if args.save_point_records:
        records_path = os.path.join(output_dir, f"per_point_records_{method_tag}.csv")
        save_records(records_path, point_records)
        print(f"Saved point records to: {records_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPair relational candidate ownership evaluator")
    parser.add_argument("--dataset_path", type=str, default="/dataset/SPair-71k")
    parser.add_argument("--dataset", type=str, default="spair")
    parser.add_argument("--save_path", type=str, default="/scratch/lt453/spair_ft/")
    parser.add_argument("--dit_model", choices=["flux"], default="flux")
    parser.add_argument("--img_size", nargs="+", type=int, default=[640, 640])
    parser.add_argument("--t", default=260, type=int)
    parser.add_argument("--k", nargs="+", type=int, default=[28])
    parser.add_argument("--ensemble_size", default=8, type=int)
    parser.add_argument("--cd", action="store_true", default=False)
    parser.add_argument("--reuse_saved_features", action="store_true", default=False)
    parser.add_argument("--max_pairs_per_cat", default=0, type=int)
    parser.add_argument("--shift_calibration", action="store_true", default=False)
    parser.add_argument("--shift_calibration_quantile", default=0.75, type=float)
    parser.add_argument("--shift_calibration_strength", default=0.5, type=float)
    parser.add_argument("--shift_calibration_min_lambda", default=0.2, type=float)
    parser.add_argument("--joint_calibration", action="store_true", default=False)
    parser.add_argument("--joint_shift_quantile", default=0.75, type=float)
    parser.add_argument("--joint_content_quantile", default=0.25, type=float)
    parser.add_argument("--joint_shift_strength", default=0.5, type=float)
    parser.add_argument("--joint_min_shift_lambda", default=0.2, type=float)
    parser.add_argument("--joint_content_strength", default=0.25, type=float)
    parser.add_argument("--joint_max_content_gain", default=1.5, type=float)
    parser.add_argument("--rco_topk", default=50, type=int)
    parser.add_argument("--rco_base_reverse_weight", default=0.0, type=float)
    parser.add_argument("--rco_anchor_quantile", default=0.6, type=float)
    parser.add_argument("--rco_anchor_margin_quantile", default=0.6, type=float)
    parser.add_argument("--rco_anchor_exclusivity_quantile", default=0.6, type=float)
    parser.add_argument("--rco_min_anchors", default=3, type=int)
    parser.add_argument("--rco_max_anchors", default=8, type=int)
    parser.add_argument("--rco_min_point_anchors", default=2, type=int)
    parser.add_argument("--rco_min_anchor_separation", default=0.05, type=float)
    parser.add_argument("--rco_distance_sigma", default=0.15, type=float)
    parser.add_argument("--rco_relation_weight", default=0.15, type=float)
    parser.add_argument("--rco_owner_margin_weight", default=0.10, type=float)
    parser.add_argument("--rco_risk_quantile", default=0.5, type=float)
    parser.add_argument("--rco_min_risk", default=0.25, type=float)
    parser.add_argument("--rco_min_relation_gain", default=0.05, type=float)
    parser.add_argument("--rco_min_structural_gain", default=0.02, type=float)
    parser.add_argument("--rco_max_raw_gap", default=0.20, type=float)
    parser.add_argument("--rco_modify_anchors", action="store_true", default=False)
    parser.add_argument("--save_point_records", action="store_true", default=False)
    parser.add_argument("--point_records_dir", type=str, default="")
    args = parser.parse_args()
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main(args)

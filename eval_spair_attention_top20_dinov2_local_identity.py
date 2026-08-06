"""Candidate-conditioned local DINOv2 identity inside FLUX attention top20.

Exact block28 mutual cross-attention supplies candidate coordinates only.  Each
source point and target candidate is observed through three fixed octave crops.
The matcher compares local contrast residuals and their coherent cross-scale
responses.  Attention weights, FLUX/native descriptors, geometry, fallback and
GT never enter candidate scoring.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from dino_v2_extractor import DINOExtractor, build_dino_extractor
from dino_v2_spair import DINOConfig
from eval_spair_attention_top20_dinov2_identity import (
    _category_metrics,
    _load_category_pairs,
    _topk_hits,
)
from eval_spair_matcher_ablation import (
    _FjsarMemoryCache,
    _get_flux_fjsar_entry,
    _load_flux_fjsar_runtime,
    _load_pairs,
    _make_fjsar_capture,
    _pck,
    _prepare_feature_tensors,
)
from spair_matchers import cosine_nn_predict, flux_fjsar_attention_candidates


LOCAL_INPUT_SIZE = 224
LOCAL_GRID_SIZE = 16
CROP_RATIOS = (0.125, 0.25, 0.5)
SCALE_SHIFTS = (-1, 0, 1)
SCORE_BRANCHES = ("local_identity", "center", "residual", "response")


METHOD_HYPOTHESIS = {
    "name": "Attention Top20 Candidate-Conditioned Local DINOv2 Identity",
    "mechanism_hypothesis": (
        "Global frozen tokens share a semantic basin and cannot identify repeated parts. "
        "Candidate-centered octave crops expose higher-resolution local appearance; "
        "local contrast and coherent cross-scale response provide new identity evidence."
    ),
    "routing": "exact_block28_mutual_cross_attention_top20_coordinates_only",
    "identity": "dinov2_vitb14_block11_local_contrast_plus_cross_scale_response",
    "crop_ratios": list(CROP_RATIOS),
    "attention_used_as_identity_score": False,
    "flux_descriptor_used_as_identity_score": False,
    "native_fallback": False,
    "geometry_score": False,
    "gt_used_for_inference": False,
    "train_free": True,
}


def _point_key(image_name: str, point: Sequence[float]) -> tuple[str, int, int]:
    return image_name, int(round(float(point[0]))), int(round(float(point[1])))


def _crop_square(
    image: Image.Image,
    point: Sequence[float],
    ratio: float,
) -> tuple[Image.Image, bool]:
    side = max(14, int(round(max(image.width, image.height) * float(ratio))))
    left = int(round(float(point[0]) - (side - 1) / 2.0))
    top = int(round(float(point[1]) - (side - 1) / 2.0))
    right = left + side
    bottom = top + side
    padded = left < 0 or top < 0 or right > image.width or bottom > image.height
    return image.crop((left, top, right, bottom)), padded


def _local_view_descriptors(
    patch_maps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if patch_maps.ndim != 4 or tuple(patch_maps.shape[1:]) != (768, 16, 16):
        raise ValueError(
            "Expected local DINOv2 patch maps with shape Bx768x16x16, "
            f"got {tuple(patch_maps.shape)}"
        )
    maps = torch.nan_to_num(
        patch_maps.float(), nan=0.0, posinf=0.0, neginf=0.0
    )
    center_raw = maps[:, :, 7:9, 7:9].mean(dim=(2, 3))
    context = (
        maps.sum(dim=(2, 3)) - 4.0 * center_raw
    ) / float(LOCAL_GRID_SIZE * LOCAL_GRID_SIZE - 4)
    center = F.normalize(center_raw, dim=1, eps=1e-10)
    residual = F.normalize(center_raw - context, dim=1, eps=1e-10)
    return center, residual


def _ensure_local_signatures(
    *,
    extractor: DINOExtractor,
    image: Image.Image,
    image_name: str,
    points: Sequence[Sequence[float]],
    cache: dict[tuple[str, int, int], tuple[torch.Tensor, torch.Tensor]],
    batch_size: int,
    stats: Counter,
) -> list[tuple[str, int, int]]:
    keys = [_point_key(image_name, point) for point in points]
    point_by_key: dict[tuple[str, int, int], tuple[float, float]] = {}
    for point, key in zip(points, keys):
        point_by_key.setdefault(key, (float(point[0]), float(point[1])))
    missing = [key for key in point_by_key if key not in cache]
    stats["point_requests"] += len(keys)
    stats["point_cache_hits"] += len(keys) - len(missing)
    if not missing:
        return keys

    pending: dict[tuple[str, int, int], dict[str, list[torch.Tensor | None]]] = {
        key: {
            "center": [None] * len(CROP_RATIOS),
            "residual": [None] * len(CROP_RATIOS),
        }
        for key in missing
    }
    requests = [
        (key, scale_index, point_by_key[key], ratio)
        for key in missing
        for scale_index, ratio in enumerate(CROP_RATIOS)
    ]
    for start in range(0, len(requests), int(batch_size)):
        chunk = requests[start : start + int(batch_size)]
        crops = []
        for _key, _scale_index, point, ratio in chunk:
            crop, padded = _crop_square(image, point, ratio)
            crops.append(crop)
            stats["crop_requests"] += 1
            stats["boundary_padded_crops"] += int(padded)
        patch_maps = extractor.extract_batch(crops, LOCAL_INPUT_SIZE)
        centers, residuals = _local_view_descriptors(patch_maps)
        for offset, (key, scale_index, _point, _ratio) in enumerate(chunk):
            pending[key]["center"][scale_index] = centers[offset].half().cpu()
            pending[key]["residual"][scale_index] = residuals[offset].half().cpu()
        for crop in crops:
            crop.close()

    for key, values in pending.items():
        if any(value is None for value in values["center"] + values["residual"]):
            raise RuntimeError(f"Incomplete local DINOv2 signature for {key}")
        cache[key] = (
            torch.stack([value for value in values["center"] if value is not None]),
            torch.stack([value for value in values["residual"] if value is not None]),
        )
    stats["unique_points_extracted"] += len(missing)
    return keys


def _coherent_scores(
    src_center: torch.Tensor,
    src_residual: torch.Tensor,
    trg_center: torch.Tensor,
    trg_residual: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Return candidate scores under one coherent octave shift per hypothesis."""
    if src_center.ndim != 3 or trg_center.ndim != 4:
        raise ValueError("Local source/target signatures have incompatible ranks")
    if src_center.shape != src_residual.shape:
        raise ValueError("Source center and residual signatures must align")
    if trg_center.shape != trg_residual.shape:
        raise ValueError("Target center and residual signatures must align")
    if src_center.shape[0] != trg_center.shape[0] or src_center.shape[1:] != trg_center.shape[2:]:
        raise ValueError("Source and target local signature shapes do not align")

    per_shift: dict[str, list[torch.Tensor]] = {
        branch: [] for branch in SCORE_BRANCHES
    }
    for shift in SCALE_SHIFTS:
        source_indices = [
            index
            for index in range(len(CROP_RATIOS))
            if 0 <= index + int(shift) < len(CROP_RATIOS)
        ]
        target_indices = [index + int(shift) for index in source_indices]
        src_c = src_center[:, source_indices]
        src_r = src_residual[:, source_indices]
        trg_c = trg_center[:, :, target_indices]
        trg_r = trg_residual[:, :, target_indices]
        center_score = (src_c[:, None] * trg_c).sum(dim=-1).mean(dim=-1)
        residual_score = (src_r[:, None] * trg_r).sum(dim=-1).mean(dim=-1)

        src_response = F.normalize(
            src_r[:, 1:] - src_r[:, :-1], dim=-1, eps=1e-10
        )
        trg_response = F.normalize(
            trg_r[:, :, 1:] - trg_r[:, :, :-1], dim=-1, eps=1e-10
        )
        response_score = (
            src_response[:, None] * trg_response
        ).sum(dim=-1).mean(dim=-1)
        identity_score = 0.5 * (residual_score + response_score)
        per_shift["center"].append(center_score)
        per_shift["residual"].append(residual_score)
        per_shift["response"].append(response_score)
        per_shift["local_identity"].append(identity_score)

    scores: dict[str, torch.Tensor] = {}
    shifts: dict[str, torch.Tensor] = {}
    shift_values = torch.tensor(SCALE_SHIFTS, device=src_center.device, dtype=torch.long)
    for branch in SCORE_BRANCHES:
        stacked = torch.stack(per_shift[branch], dim=-1)
        scores[branch], shift_indices = stacked.max(dim=-1)
        shifts[branch] = shift_values[shift_indices]
    return scores, shifts


def _rank_local_candidates(
    *,
    source_keys: Sequence[tuple[str, int, int]],
    target_keys: Sequence[Sequence[tuple[str, int, int]]],
    candidate_pixels: torch.Tensor,
    cache: dict[tuple[str, int, int], tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> dict[str, Any]:
    src_center = torch.stack([cache[key][0] for key in source_keys]).float().to(device)
    src_residual = torch.stack([cache[key][1] for key in source_keys]).float().to(device)
    trg_center = torch.stack(
        [torch.stack([cache[key][0] for key in row]) for row in target_keys]
    ).float().to(device)
    trg_residual = torch.stack(
        [torch.stack([cache[key][1] for key in row]) for row in target_keys]
    ).float().to(device)
    scores, shifts = _coherent_scores(
        src_center, src_residual, trg_center, trg_residual
    )
    pixels = candidate_pixels.detach().cpu().long()
    orders: dict[str, torch.Tensor] = {}
    rank_positions: dict[str, torch.Tensor] = {}
    for branch in SCORE_BRANCHES:
        branch_orders = []
        branch_ranks = []
        for row in range(pixels.shape[0]):
            row_pixels = pixels[row].tolist()
            row_scores = scores[branch][row].detach().cpu().tolist()
            order = sorted(
                range(len(row_pixels)),
                key=lambda index: (-float(row_scores[index]), int(row_pixels[index])),
            )
            order_tensor = torch.tensor(order, dtype=torch.long)
            rank_tensor = torch.empty_like(order_tensor)
            rank_tensor[order_tensor] = torch.arange(1, len(order) + 1)
            branch_orders.append(order_tensor)
            branch_ranks.append(rank_tensor)
        orders[branch] = torch.stack(branch_orders)
        rank_positions[branch] = torch.stack(branch_ranks)
    return {
        "pixels": pixels,
        "scores": {key: value.detach().float().cpu() for key, value in scores.items()},
        "shifts": {key: value.detach().long().cpu() for key, value in shifts.items()},
        "orders": orders,
        "rank_positions": rank_positions,
    }


def _ranked_branch_pixels(ranking: dict[str, Any], branch: str) -> torch.Tensor:
    return torch.gather(ranking["pixels"], 1, ranking["orders"][branch])


def _branch_summary(
    rows: Sequence[dict[str, Any]], branch: str, topks: Sequence[int]
) -> dict[str, Any]:
    ranks = [
        int(row["branch_gt_ranks"][branch])
        for row in rows
        if row["branch_gt_ranks"][branch] is not None
    ]
    return {
        "points": len(rows),
        "topk_hits": {
            str(int(k)): int(
                sum(bool(row["branch_topk_hits"][branch][str(int(k))]) for row in rows)
            )
            for k in topks
        },
        "topk_rates": {
            str(int(k)): float(
                sum(bool(row["branch_topk_hits"][branch][str(int(k))]) for row in rows)
                / max(1, len(rows))
            )
            for k in topks
        },
        "gt_rank_mean": float(np.mean(ranks)) if ranks else None,
        "gt_rank_median": float(np.median(ranks)) if ranks else None,
    }


def _summarize_points(
    records: Sequence[dict[str, Any]], topks: Sequence[int]
) -> dict[str, Any]:
    points = [point for pair in records for point in pair["points"]]
    hard = [point for point in points if point["both_wrong_top20_hit"]]

    def partition(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        return {
            "points": len(rows),
            "uniform_top1_expectation": float(
                np.mean([row["uniform_candidate_hit_probability"] for row in rows])
            ) if rows else None,
            "branches": {
                branch: _branch_summary(rows, branch, topks)
                for branch in SCORE_BRANCHES
            },
        }

    deep_partitions = {
        "attention_rank_2_3": [
            row for row in hard if row["attention_gt_rank"] is not None and row["attention_gt_rank"] <= 3
        ],
        "attention_rank_4_5": [
            row for row in hard if row["attention_gt_rank"] is not None and 4 <= row["attention_gt_rank"] <= 5
        ],
        "attention_rank_6_10": [
            row for row in hard if row["attention_gt_rank"] is not None and 6 <= row["attention_gt_rank"] <= 10
        ],
        "attention_rank_11_20": [
            row for row in hard if row["attention_gt_rank"] is not None and 11 <= row["attention_gt_rank"] <= 20
        ],
    }
    scale_shifts_all = Counter(str(int(row["method_scale_shift"])) for row in points)
    scale_shifts_hard = Counter(str(int(row["method_scale_shift"])) for row in hard)
    return {
        "pairs": len(records),
        "points": len(points),
        "baseline_correct": int(sum(bool(row["baseline_pck_hit"]) for row in points)),
        "attention_top1_correct": int(sum(bool(row["attention_top1_pck_hit"]) for row in points)),
        "method_correct": int(sum(bool(row["method_pck_hit"]) for row in points)),
        "candidate_missing_gt": int(sum(not bool(row["attention_top20_pck_hit"]) for row in points)),
        "rescued_vs_baseline": int(sum(bool(row["rescued_vs_baseline"]) for row in points)),
        "harmed_vs_baseline": int(sum(bool(row["harmed_vs_baseline"]) for row in points)),
        "all_points": partition(points),
        "both_wrong_top20_hit": partition(hard),
        "hard_by_attention_depth": {
            name: partition(rows) for name, rows in deep_partitions.items()
        },
        "method_scale_shift_counts": {
            "all_points": dict(scale_shifts_all),
            "both_wrong_top20_hit": dict(scale_shifts_hard),
        },
    }


def _cache_stats(cache: dict[Any, tuple[torch.Tensor, torch.Tensor]], stats: Counter) -> dict[str, Any]:
    tensor_bytes = sum(
        int(center.numel() * center.element_size() + residual.numel() * residual.element_size())
        for center, residual in cache.values()
    )
    crop_requests = int(stats["crop_requests"])
    return {
        **{key: int(value) for key, value in stats.items()},
        "cached_point_signatures": len(cache),
        "cache_tensor_bytes": tensor_bytes,
        "boundary_padded_crop_rate": (
            float(stats["boundary_padded_crops"]) / crop_requests if crop_requests else 0.0
        ),
    }


def evaluate(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    config = DINOConfig()
    if int(args.fjsar_candidate_topk) != 20:
        raise ValueError("This strict experiment requires --fjsar_candidate_topk 20")
    if not args.fjsar_require_disk_cache or not args.fjsar_disk_cache_path:
        raise ValueError("The strict experiment requires the canonical FLUX replay cache")
    if int(args.dino_local_batch_size) < 1:
        raise ValueError("--dino_local_batch_size must be positive")

    test_path, all_cats, cat2json, _cat2img = _load_pairs(args.dataset_path)
    with open("spair_detailed_captions.json") as handle:
        captions = json.load(handle)
    fjsar_featurizer, fjsar_model, fjsar_blocks = _load_flux_fjsar_runtime(args, all_cats)
    fjsar_capture = _make_fjsar_capture(args, fjsar_model)
    fjsar_memory_cache = _FjsarMemoryCache(
        int(float(args.fjsar_memory_cache_gb) * (1024 ** 3))
    )
    dino_extractor = build_dino_extractor(args, config)
    pre_norm = nn.LayerNorm(3072, elementwise_affine=False, eps=1e-6).to(device)
    topks = (1, 3, 5, 10, 20)
    pair_records: list[dict[str, Any]] = []
    category_results: dict[str, Any] = {}
    category_local_cache_stats: dict[str, Any] = {}
    all_pair_baseline: list[float] = []
    all_pair_method: list[float] = []
    all_baseline_correct = all_method_correct = all_total = 0
    all_improved = all_harmed = 0
    open_images: list[Image.Image] = []

    try:
        for category in all_cats:
            pair_data = _load_category_pairs(args, test_path, category, cat2json[category])
            image_root = os.path.join(args.dataset_path, "JPEGImages", category)
            image_names = sorted({
                image_name
                for _pair_name, data in pair_data
                for image_name in (data["src_imname"], data["trg_imname"])
            })
            image_cache: dict[str, Image.Image] = {}
            for image_name in image_names:
                image = Image.open(os.path.join(image_root, image_name)).convert("RGB")
                image_cache[image_name] = image
                open_images.append(image)
            signature_cache: dict[
                tuple[str, int, int], tuple[torch.Tensor, torch.Tensor]
            ] = {}
            signature_stats: Counter = Counter()

            cat_pair_baseline: list[float] = []
            cat_pair_method: list[float] = []
            cat_baseline_correct = cat_method_correct = cat_total = 0
            cat_improved = cat_harmed = 0
            for pair_name, data in tqdm(pair_data, desc=f"evaluate {category}"):
                source_size = data["src_imsize"][:2][::-1]
                target_size = data["trg_imsize"][:2][::-1]
                src_entry = _get_flux_fjsar_entry(
                    args.dataset_path,
                    category,
                    data["src_imname"],
                    captions[category + data["src_imname"]],
                    args,
                    fjsar_featurizer,
                    fjsar_capture,
                    fjsar_memory_cache,
                )
                trg_entry = _get_flux_fjsar_entry(
                    args.dataset_path,
                    category,
                    data["trg_imname"],
                    captions[category + data["trg_imname"]],
                    args,
                    fjsar_featurizer,
                    fjsar_capture,
                    fjsar_memory_cache,
                )
                src_ft = _prepare_feature_tensors(
                    src_entry["feature"], src_entry["ada"], args, pre_norm, device
                )
                trg_ft = _prepare_feature_tensors(
                    trg_entry["feature"], trg_entry["ada"], args, pre_norm, device
                )
                src_full = F.interpolate(
                    src_ft.to(torch.float16), size=source_size, mode="bilinear"
                )
                trg_full = F.interpolate(
                    trg_ft.to(torch.float16), size=target_size, mode="bilinear"
                )
                baseline_predictions = cosine_nn_predict(
                    src_full, trg_full, data["src_kps"]
                )
                threshold = max(
                    data["trg_bndbox"][3] - data["trg_bndbox"][1],
                    data["trg_bndbox"][2] - data["trg_bndbox"][0],
                )
                with torch.no_grad():
                    candidate_pixels, _attention_scores, _attention_state = (
                        flux_fjsar_attention_candidates(
                            src_replay_state=src_entry["replay_state"],
                            trg_replay_state=trg_entry["replay_state"],
                            blocks=fjsar_blocks,
                            points=data["src_kps"],
                            source_size=source_size,
                            target_size=target_size,
                            candidate_topk=args.fjsar_candidate_topk,
                            interaction_mode="exact",
                            use_coordinate_bias=False,
                        )
                    )

                target_w = int(target_size[1])
                candidate_cpu = candidate_pixels.detach().cpu().long()
                candidate_points = [
                    [
                        [int(pixel % target_w), int(pixel // target_w)]
                        for pixel in row.tolist()
                    ]
                    for row in candidate_cpu
                ]
                source_keys = _ensure_local_signatures(
                    extractor=dino_extractor,
                    image=image_cache[data["src_imname"]],
                    image_name=data["src_imname"],
                    points=data["src_kps"],
                    cache=signature_cache,
                    batch_size=args.dino_local_batch_size,
                    stats=signature_stats,
                )
                flat_target_points = [point for row in candidate_points for point in row]
                flat_target_keys = _ensure_local_signatures(
                    extractor=dino_extractor,
                    image=image_cache[data["trg_imname"]],
                    image_name=data["trg_imname"],
                    points=flat_target_points,
                    cache=signature_cache,
                    batch_size=args.dino_local_batch_size,
                    stats=signature_stats,
                )
                candidate_count = int(candidate_cpu.shape[1])
                target_keys = [
                    flat_target_keys[index * candidate_count : (index + 1) * candidate_count]
                    for index in range(len(source_keys))
                ]
                with torch.no_grad():
                    ranking = _rank_local_candidates(
                        source_keys=source_keys,
                        target_keys=target_keys,
                        candidate_pixels=candidate_cpu,
                        cache=signature_cache,
                        device=device,
                    )

                ranked_by_branch = {
                    branch: _ranked_branch_pixels(ranking, branch)
                    for branch in SCORE_BRANCHES
                }
                main_order = ranking["orders"]["local_identity"]
                main_ranked = ranked_by_branch["local_identity"]
                selected = main_ranked[:, 0].tolist()
                method_predictions = [
                    [int(pixel % target_w), int(pixel // target_w)] for pixel in selected
                ]
                point_rows = []
                pair_baseline_hits = []
                pair_method_hits = []
                for index, target_point in enumerate(data["trg_kps"]):
                    baseline_hit = bool(
                        _pck(baseline_predictions[index], target_point, threshold)
                    )
                    method_hit = bool(
                        _pck(method_predictions[index], target_point, threshold)
                    )
                    attention_pixels = candidate_cpu[index].tolist()
                    attention_hit_flags = [
                        bool(
                            _pck(
                                [int(pixel % target_w), int(pixel // target_w)],
                                target_point,
                                threshold,
                            )
                        )
                        for pixel in attention_pixels
                    ]
                    attention_gt_rank = next(
                        (rank + 1 for rank, hit in enumerate(attention_hit_flags) if hit),
                        None,
                    )
                    branch_topk_hits: dict[str, dict[str, bool]] = {}
                    branch_gt_ranks: dict[str, int | None] = {}
                    branch_hit_flags: dict[str, list[bool]] = {}
                    for branch in SCORE_BRANCHES:
                        ranked_pixels = ranked_by_branch[branch][index].tolist()
                        hits, flags = _topk_hits(
                            ranked_pixels, target_point, threshold, target_w, topks
                        )
                        branch_topk_hits[branch] = hits
                        branch_hit_flags[branch] = flags
                        branch_gt_ranks[branch] = next(
                            (rank + 1 for rank, hit in enumerate(flags) if hit), None
                        )

                    candidates = []
                    for local_rank, original_index in enumerate(
                        main_order[index].tolist(), start=1
                    ):
                        pixel = int(candidate_cpu[index, original_index])
                        candidates.append({
                            "local_identity_rank": int(local_rank),
                            "attention_rank": int(original_index + 1),
                            "pixel": [int(pixel % target_w), int(pixel // target_w)],
                            "pixel_index": pixel,
                            "pck_hit": bool(
                                branch_hit_flags["local_identity"][local_rank - 1]
                            ),
                            "scores": {
                                branch: float(ranking["scores"][branch][index, original_index])
                                for branch in SCORE_BRANCHES
                            },
                            "ranks": {
                                branch: int(ranking["rank_positions"][branch][index, original_index])
                                for branch in SCORE_BRANCHES
                            },
                            "scale_shifts": {
                                branch: int(ranking["shifts"][branch][index, original_index])
                                for branch in SCORE_BRANCHES
                            },
                        })

                    attention_top20_hit = bool(any(attention_hit_flags))
                    top_original_index = int(main_order[index, 0])
                    point_rows.append({
                        "keypoint_index": int(index),
                        "source_point": list(data["src_kps"][index]),
                        "target_point": list(target_point),
                        "baseline_prediction": list(baseline_predictions[index]),
                        "method_prediction": list(method_predictions[index]),
                        "baseline_pck_hit": baseline_hit,
                        "method_pck_hit": method_hit,
                        "attention_top1_pck_hit": bool(attention_hit_flags[0]),
                        "attention_top20_pck_hit": attention_top20_hit,
                        "attention_gt_rank": attention_gt_rank,
                        "both_wrong_top20_hit": bool(
                            not baseline_hit and not attention_hit_flags[0] and attention_top20_hit
                        ),
                        "rescued_vs_baseline": bool(method_hit and not baseline_hit),
                        "harmed_vs_baseline": bool(baseline_hit and not method_hit),
                        "branch_gt_ranks": branch_gt_ranks,
                        "branch_topk_hits": branch_topk_hits,
                        "candidate_pck_hit_count": int(sum(attention_hit_flags)),
                        "uniform_candidate_hit_probability": float(
                            sum(attention_hit_flags) / max(1, len(attention_hit_flags))
                        ),
                        "method_scale_shift": int(
                            ranking["shifts"]["local_identity"][index, top_original_index]
                        ),
                        "method_score_margin": float(
                            ranking["scores"]["local_identity"][index, int(main_order[index, 0])]
                            - ranking["scores"]["local_identity"][index, int(main_order[index, 1])]
                        ),
                        "candidates": candidates,
                    })
                    pair_baseline_hits.append(baseline_hit)
                    pair_method_hits.append(method_hit)
                    cat_baseline_correct += int(baseline_hit)
                    cat_method_correct += int(method_hit)
                    cat_total += 1
                    cat_improved += int(method_hit and not baseline_hit)
                    cat_harmed += int(baseline_hit and not method_hit)

                baseline_pair_pck = float(np.mean(pair_baseline_hits))
                method_pair_pck = float(np.mean(pair_method_hits))
                cat_pair_baseline.append(baseline_pair_pck)
                cat_pair_method.append(method_pair_pck)
                pair_records.append({
                    "category": category,
                    "pair_json": pair_name,
                    "src_image": data["src_imname"],
                    "trg_image": data["trg_imname"],
                    "keypoint_count": len(point_rows),
                    "points": point_rows,
                })

            category_results[category] = _category_metrics(
                cat_pair_baseline,
                cat_pair_method,
                cat_baseline_correct,
                cat_method_correct,
                cat_total,
                cat_improved,
                cat_harmed,
            )
            category_local_cache_stats[category] = _cache_stats(
                signature_cache, signature_stats
            )
            print(
                f"{category}: baseline image/point="
                f"{category_results[category]['baseline_image']:.2f}/"
                f"{category_results[category]['baseline_point']:.2f}, "
                f"method image/point={category_results[category]['method_image']:.2f}/"
                f"{category_results[category]['method_point']:.2f}"
            )
            all_pair_baseline.extend(cat_pair_baseline)
            all_pair_method.extend(cat_pair_method)
            all_baseline_correct += cat_baseline_correct
            all_method_correct += cat_method_correct
            all_total += cat_total
            all_improved += cat_improved
            all_harmed += cat_harmed
            for image in image_cache.values():
                image.close()
                open_images.remove(image)
            del signature_cache, image_cache
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        for image in open_images:
            image.close()
        dino_extractor.close()
        fjsar_capture.close()
        del fjsar_model, fjsar_featurizer, fjsar_blocks
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    all_result = _category_metrics(
        all_pair_baseline,
        all_pair_method,
        all_baseline_correct,
        all_method_correct,
        all_total,
        all_improved,
        all_harmed,
    )
    summary = _summarize_points(pair_records, topks)
    protocol = {
        "flux_router": "block28 exact mutual cross-attention, no coordinate bias",
        "candidate_topk": int(args.fjsar_candidate_topk),
        "dino_model": "dinov2_vitb14",
        "dino_feature": "block11 pre-final-norm local crop tokens",
        "local_input": f"{LOCAL_INPUT_SIZE}x{LOCAL_INPUT_SIZE}",
        "local_grid": f"{LOCAL_GRID_SIZE}x{LOCAL_GRID_SIZE}",
        "crop_ratios": list(CROP_RATIOS),
        "scale_shifts": list(SCALE_SHIFTS),
        "center_readout": "mean central 2x2 patch tokens",
        "local_residual": "center minus non-center crop context mean",
        "main_score": "equal residual and cross-scale-response under one coherent octave shift",
        "audit_only_branches": ["center", "residual", "response"],
    }
    result = {
        "matcher": "attention_top20_dinov2_local_identity",
        "method_hypothesis": METHOD_HYPOTHESIS,
        "protocol": protocol,
        "categories": category_results,
        "all": all_result,
        "mechanism_summary": summary,
        "local_cache_stats": category_local_cache_stats,
        "fjsar_memory_cache": fjsar_memory_cache.stats(),
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    root = output_path.with_suffix("")
    audit_path = Path(f"{root}_attention_top20_dinov2_local_identity_audit.json")
    summary_path = Path(f"{root}_attention_top20_dinov2_local_identity_summary.json")
    audit_payload = {
        "matcher": result["matcher"],
        "method_hypothesis": METHOD_HYPOTHESIS,
        "protocol": protocol,
        "summary": summary,
        "pair_records": pair_records,
    }
    summary_payload = {
        "matcher": result["matcher"],
        "method_hypothesis": METHOD_HYPOTHESIS,
        "protocol": protocol,
        "summary": summary,
        "local_cache_stats": category_local_cache_stats,
    }
    result["audit_path"] = str(audit_path)
    result["summary_path"] = str(summary_path)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    audit_path.write_text(json.dumps(audit_payload, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(
        "Matcher: attention_top20_dinov2_local_identity\n"
        f"Baseline All per image/point: {all_result['baseline_image']:.2f} / "
        f"{all_result['baseline_point']:.2f}\n"
        f"Method All per image/point: {all_result['method_image']:.2f} / "
        f"{all_result['method_point']:.2f}; point gain={all_result['point_gain']:.2f}"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--img_size", nargs="+", type=int, default=[640, 640])
    parser.add_argument("--t", type=int, default=260)
    parser.add_argument("--k", nargs="+", type=int, default=[28])
    parser.add_argument("--ensemble_size", type=int, default=8)
    parser.add_argument("--cd", action="store_true", default=False)
    parser.add_argument("--subset", choices=["all", "discovery", "heldout"], default="discovery")
    parser.add_argument("--pairs_per_cat", type=int, default=20)
    parser.add_argument("--split_seed", type=int, default=2027)
    parser.add_argument("--max_pairs_per_cat", type=int, default=0)
    parser.add_argument("--fjsar_candidate_topk", type=int, default=20)
    parser.add_argument("--fjsar_memory_cache_gb", type=float, default=4.0)
    parser.add_argument("--fjsar_disk_cache_path", required=True)
    parser.add_argument("--fjsar_require_disk_cache", action="store_true", default=False)
    parser.add_argument("--fjsar_disk_cache_min_free_gb", type=float, default=0.0)
    parser.add_argument("--fjsar_shared_noise", action="store_true", default=False)
    parser.add_argument("--dino_backend", choices=("transformers", "torch_hub"), default="transformers")
    parser.add_argument("--dino_precision", choices=("fp32", "fp16", "bf16"), default="fp16")
    parser.add_argument("--dino_local_files_only", action="store_true", default=False)
    parser.add_argument("--dino_model_repo", default=None)
    parser.add_argument("--dino_local_batch_size", type=int, default=32)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.set_defaults(
        matcher="fjsar_attn",
        fjsar_multilayer_identity_audit=False,
        fjsar_multilayer_blocks=(),
        fjsar_trajectory_blocks=(),
    )
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    run_device = torch.device(
        parsed.device if parsed.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    evaluate(parsed, run_device)

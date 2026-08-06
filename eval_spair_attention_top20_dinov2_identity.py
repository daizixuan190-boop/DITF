"""Strict FLUX-attention routing with DINOv2-only candidate identity.

Exact block28 mutual cross-attention supplies the only top-K candidate pool.
DINOv2 ViT-B/14 block-11 cosine similarity is the only candidate score.  No
attention weight, FLUX descriptor, native candidate, fallback, geometry, or GT
enters prediction.
"""

from __future__ import annotations

import argparse
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

from dino_v2_extractor import build_dino_extractor
from dino_v2_spair import (
    DINOConfig,
    points_to_patch_indices,
    transform_points_to_canvas,
)
from eval_spair_matcher_ablation import (
    _FjsarMemoryCache,
    _get_flux_fjsar_entry,
    _load_flux_fjsar_runtime,
    _load_pairs,
    _make_fjsar_capture,
    _pck,
    _prepare_feature_tensors,
    _select_category_pairs,
)
from ephemeral_category_cache import (
    category_cache_snapshot,
    delete_new_category_cache_files,
)
from spair_matchers import (
    cosine_candidate_diagnostics,
    cosine_nn_predict_with_diagnostics,
    flux_fjsar_attention_candidates,
)


METHOD_HYPOTHESIS = {
    "name": "Attention Top20 DINOv2 Identity",
    "mechanism_hypothesis": (
        "FLUX cross-attention has high semantic-region recall but lacks candidate-internal "
        "identity separability.  A frozen discriminative DINOv2 representation may provide "
        "independent point identity inside the fixed attention pool."
    ),
    "routing": "exact_block28_mutual_cross_attention_top20_only",
    "identity": "dinov2_vitb14_block11_cosine_only",
    "attention_used_as_identity_score": False,
    "flux_descriptor_used_as_identity_score": False,
    "native_fallback": False,
    "geometry_score": False,
    "gt_used_for_inference": False,
    "train_free": True,
}


def _category_image_names(pair_data: Sequence[tuple[str, dict[str, Any]]]) -> list[str]:
    names = {
        image_name
        for _pair_name, data in pair_data
        for image_name in (data["src_imname"], data["trg_imname"])
    }
    return sorted(names)


def _load_category_pairs(
    args: argparse.Namespace,
    test_path: str,
    category: str,
    pair_names: Sequence[str],
) -> list[tuple[str, dict[str, Any]]]:
    selected = _select_category_pairs(pair_names, args)
    if args.max_pairs_per_cat > 0:
        selected = selected[: args.max_pairs_per_cat]
    output = []
    for pair_name in selected:
        with open(os.path.join(args.dataset_path, test_path, pair_name)) as handle:
            output.append((pair_name, json.load(handle)))
    return output


def _dino_feature_tokens(feature: torch.Tensor, device: torch.device) -> torch.Tensor:
    if feature.ndim != 3 or tuple(feature.shape) != (768, 60, 60):
        raise ValueError(f"Expected DINOv2 feature shape (768,60,60), got {tuple(feature.shape)}")
    return F.normalize(
        feature.to(device=device, dtype=torch.float32).flatten(1).transpose(0, 1),
        dim=1,
        eps=1e-10,
    )


def _rank_dino_candidates(
    src_feature: torch.Tensor,
    trg_feature: torch.Tensor,
    source_points: Sequence[Sequence[float]],
    candidate_pixels: torch.Tensor,
    source_size: Sequence[int],
    target_size: Sequence[int],
    config: DINOConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rank fixed original-image candidate pixels by DINO cosine only."""
    source_h, source_w = int(source_size[0]), int(source_size[1])
    target_h, target_w = int(target_size[0]), int(target_size[1])
    source_canvas = transform_points_to_canvas(
        source_points, source_h, source_w, config.image_size
    )
    source_indices = points_to_patch_indices(
        source_canvas, config.image_size, config.grid_size
    ).to(device)

    candidate_pixels = candidate_pixels.to(device=device, dtype=torch.long)
    candidate_x = candidate_pixels % target_w
    candidate_y = torch.div(candidate_pixels, target_w, rounding_mode="floor")
    candidate_xy = torch.stack((candidate_x, candidate_y), dim=2).reshape(-1, 2).float().cpu()
    candidate_canvas = transform_points_to_canvas(
        candidate_xy.tolist(), target_h, target_w, config.image_size
    )
    target_indices = points_to_patch_indices(
        candidate_canvas, config.image_size, config.grid_size
    ).reshape_as(candidate_pixels).to(device)

    source_tokens = _dino_feature_tokens(src_feature, device)
    target_tokens = _dino_feature_tokens(trg_feature, device)
    scores = (
        source_tokens[source_indices].unsqueeze(1)
        * target_tokens[target_indices]
    ).sum(dim=2)

    ranked_rows = []
    score_rows = []
    token_rows = []
    attention_rank_rows = []
    for row in range(candidate_pixels.shape[0]):
        pixels = candidate_pixels[row].detach().cpu().tolist()
        values = scores[row].detach().cpu().tolist()
        # Exact score ties are resolved by pixel index, never attention rank.
        order = sorted(range(len(pixels)), key=lambda index: (-float(values[index]), int(pixels[index])))
        order_tensor = torch.tensor(order, device=device, dtype=torch.long)
        ranked_rows.append(candidate_pixels[row, order_tensor])
        score_rows.append(scores[row, order_tensor])
        token_rows.append(target_indices[row, order_tensor])
        attention_rank_rows.append(order_tensor + 1)
    return (
        torch.stack(ranked_rows),
        torch.stack(score_rows),
        torch.stack(token_rows),
        torch.stack(attention_rank_rows),
    )


def _topk_hits(
    ranked_pixels: Sequence[int],
    target_point: Sequence[float],
    threshold: float,
    target_width: int,
    topks: Sequence[int],
) -> tuple[dict[str, bool], list[bool]]:
    hits = [
        bool(
            _pck(
                [int(pixel % target_width), int(pixel // target_width)],
                target_point,
                threshold,
            )
        )
        for pixel in ranked_pixels
    ]
    return {
        str(int(k)): bool(any(hits[: min(int(k), len(hits))]))
        for k in topks
    }, hits


def _summarize_points(records: Sequence[dict[str, Any]], topks: Sequence[int]) -> dict[str, Any]:
    points = [point for pair in records for point in pair["points"]]
    hard = [point for point in points if point["both_wrong_top20_hit"]]

    def _branch(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        ranks = [int(row["dino_gt_rank"]) for row in rows if row["dino_gt_rank"] is not None]
        return {
            "points": len(rows),
            "topk_hits": {
                str(int(k)): int(sum(bool(row["dino_topk_hits"][str(int(k))]) for row in rows))
                for k in topks
            },
            "topk_rates": {
                str(int(k)): float(
                    sum(bool(row["dino_topk_hits"][str(int(k))]) for row in rows)
                    / max(1, len(rows))
                )
                for k in topks
            },
            "gt_rank_mean": float(np.mean(ranks)) if ranks else None,
            "gt_rank_median": float(np.median(ranks)) if ranks else None,
            "uniform_top1_expectation": float(
                np.mean([row["uniform_candidate_hit_probability"] for row in rows])
            ) if rows else None,
            "mean_unique_dino_tokens": float(
                np.mean([row["unique_dino_token_count"] for row in rows])
            ) if rows else None,
        }

    return {
        "pairs": len(records),
        "points": len(points),
        "baseline_correct": int(sum(bool(point["baseline_pck_hit"]) for point in points)),
        "attention_top1_correct": int(sum(bool(point["attention_top1_pck_hit"]) for point in points)),
        "method_correct": int(sum(bool(point["method_pck_hit"]) for point in points)),
        "candidate_missing_gt": int(sum(not bool(point["attention_top20_pck_hit"]) for point in points)),
        "rescued_vs_baseline": int(sum(bool(point["rescued_vs_baseline"]) for point in points)),
        "harmed_vs_baseline": int(sum(bool(point["harmed_vs_baseline"]) for point in points)),
        "all_points": _branch(points),
        "both_wrong_top20_hit": _branch(hard),
    }


def _category_metrics(
    pair_baseline: Sequence[float],
    pair_method: Sequence[float],
    baseline_correct: int,
    method_correct: int,
    total: int,
    improved: int,
    harmed: int,
) -> dict[str, Any]:
    return {
        "baseline_image": 100.0 * float(np.mean(pair_baseline)) if pair_baseline else 0.0,
        "method_image": 100.0 * float(np.mean(pair_method)) if pair_method else 0.0,
        "baseline_point": 100.0 * baseline_correct / max(1, total),
        "method_point": 100.0 * method_correct / max(1, total),
        "point_gain": 100.0 * (method_correct - baseline_correct) / max(1, total),
        "points": int(total),
        "improved_count": int(improved),
        "harmed_count": int(harmed),
    }


def evaluate(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    config = DINOConfig()
    if int(args.fjsar_candidate_topk) != 20:
        raise ValueError("This strict experiment requires --fjsar_candidate_topk 20")
    ephemeral_cache = bool(args.fjsar_ephemeral_category_cache)
    if not args.fjsar_disk_cache_path:
        raise ValueError("The strict experiment requires a FLUX replay cache path")
    if args.fjsar_require_disk_cache and ephemeral_cache:
        raise ValueError(
            "--fjsar_require_disk_cache and --fjsar_ephemeral_category_cache are mutually exclusive"
        )
    if not args.fjsar_require_disk_cache and not ephemeral_cache:
        raise ValueError(
            "Use either a precomputed --fjsar_require_disk_cache or the explicit "
            "--fjsar_ephemeral_category_cache low-disk protocol"
        )

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
    all_pair_baseline: list[float] = []
    all_pair_method: list[float] = []
    all_baseline_correct = all_method_correct = all_total = 0
    all_improved = all_harmed = 0
    ephemeral_deleted_files = 0
    active_ephemeral_category: str | None = None
    active_ephemeral_snapshot: set[Path] = set()

    try:
        for category in all_cats:
            category_cache_before = (
                category_cache_snapshot(args.fjsar_disk_cache_path, category)
                if ephemeral_cache
                else set()
            )
            if ephemeral_cache:
                active_ephemeral_category = category
                active_ephemeral_snapshot = category_cache_before
            pair_data = _load_category_pairs(
                args, test_path, category, cat2json[category]
            )
            dino_features: dict[str, torch.Tensor] = {}
            image_root = os.path.join(args.dataset_path, "JPEGImages", category)
            for image_name in tqdm(
                _category_image_names(pair_data),
                desc=f"DINO features {category}",
                leave=False,
            ):
                with Image.open(os.path.join(image_root, image_name)) as image:
                    dino_features[image_name] = dino_extractor(image)

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
                src_full = F.interpolate(src_ft.to(torch.float16), size=source_size, mode="bilinear")
                trg_full = F.interpolate(trg_ft.to(torch.float16), size=target_size, mode="bilinear")
                baseline_predictions, native_nn_diagnostics = (
                    cosine_nn_predict_with_diagnostics(
                        src_full,
                        trg_full,
                        data["src_kps"],
                        nonlocal_radius=args.native_nonlocal_radius,
                    )
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
                    ranked_pixels, ranked_scores, ranked_dino_tokens, ranked_attention_ranks = (
                        _rank_dino_candidates(
                            dino_features[data["src_imname"]],
                            dino_features[data["trg_imname"]],
                            data["src_kps"],
                            candidate_pixels,
                            source_size,
                            target_size,
                            config,
                            device,
                        )
                    )

                target_w = int(target_size[1])
                ranked_candidate_xy = [
                    [
                        [int(pixel % target_w), int(pixel // target_w)]
                        for pixel in pixel_row
                    ]
                    for pixel_row in ranked_pixels.detach().cpu().tolist()
                ]
                native_candidate_rows = cosine_candidate_diagnostics(
                    src_full,
                    trg_full,
                    data["src_kps"],
                    ranked_candidate_xy,
                )

                selected = ranked_pixels[:, 0].detach().cpu().tolist()
                method_predictions = [
                    [int(pixel % target_w), int(pixel // target_w)]
                    for pixel in selected
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
                    attention_pixels = candidate_pixels[index].detach().cpu().tolist()
                    attention_top1_xy = [
                        int(attention_pixels[0] % target_w),
                        int(attention_pixels[0] // target_w),
                    ]
                    attention_top1_hit = bool(
                        _pck(attention_top1_xy, target_point, threshold)
                    )
                    ranked = ranked_pixels[index].detach().cpu().tolist()
                    topk_hits, ranked_hit_flags = _topk_hits(
                        ranked, target_point, threshold, target_w, topks
                    )
                    gt_rank = next(
                        (rank + 1 for rank, hit in enumerate(ranked_hit_flags) if hit),
                        None,
                    )
                    unique_tokens = len(
                        set(ranked_dino_tokens[index].detach().cpu().tolist())
                    )
                    candidates = []
                    for rank, (
                        pixel,
                        score,
                        token,
                        attention_rank,
                        pck_hit,
                        native_candidate,
                    ) in enumerate(
                        zip(
                            ranked,
                            ranked_scores[index].detach().cpu().tolist(),
                            ranked_dino_tokens[index].detach().cpu().tolist(),
                            ranked_attention_ranks[index].detach().cpu().tolist(),
                            ranked_hit_flags,
                            native_candidate_rows[index],
                        ),
                        start=1,
                    ):
                        candidates.append({
                            "dino_rank": int(rank),
                            "attention_rank": int(attention_rank),
                            "pixel": [int(pixel % target_w), int(pixel // target_w)],
                            "pixel_index": int(pixel),
                            "dino_token_index": int(token),
                            "dino_cosine": float(score),
                            "native_cosine": float(native_candidate["native_cosine"]),
                            "native_candidate_rank": int(
                                native_candidate["native_candidate_rank"]
                            ),
                            "native_gap_to_candidate_top1": float(
                                native_candidate["native_gap_to_candidate_top1"]
                            ),
                            "native_gap_to_full_top1": float(
                                native_nn_diagnostics[index]["top1_cosine"]
                                - native_candidate["native_cosine"]
                            ),
                            "pck_hit": bool(pck_hit),
                        })
                    attention_top20_hit = bool(any(ranked_hit_flags))
                    point_rows.append({
                        "keypoint_index": int(index),
                        "source_point": list(data["src_kps"][index]),
                        "target_point": list(target_point),
                        "baseline_prediction": list(baseline_predictions[index]),
                        "native_nn_diagnostics": native_nn_diagnostics[index],
                        "method_prediction": list(method_predictions[index]),
                        "baseline_pck_hit": baseline_hit,
                        "method_pck_hit": method_hit,
                        "attention_top1_pck_hit": attention_top1_hit,
                        "attention_top20_pck_hit": attention_top20_hit,
                        "both_wrong_top20_hit": bool(
                            not baseline_hit and not attention_top1_hit and attention_top20_hit
                        ),
                        "rescued_vs_baseline": bool(method_hit and not baseline_hit),
                        "harmed_vs_baseline": bool(baseline_hit and not method_hit),
                        "dino_gt_rank": gt_rank,
                        "dino_topk_hits": topk_hits,
                        "candidate_pck_hit_count": int(sum(ranked_hit_flags)),
                        "uniform_candidate_hit_probability": float(
                            sum(ranked_hit_flags) / max(1, len(ranked_hit_flags))
                        ),
                        "unique_dino_token_count": int(unique_tokens),
                        "dino_token_collision_count": int(len(ranked) - unique_tokens),
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
            del dino_features
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if ephemeral_cache:
                deleted = delete_new_category_cache_files(
                    args.fjsar_disk_cache_path,
                    category,
                    category_cache_before,
                )
                ephemeral_deleted_files += deleted
                print(
                    f"{category}: removed {deleted} newly-created ephemeral FLUX cache files"
                )
                active_ephemeral_category = None
                active_ephemeral_snapshot = set()
    finally:
        if ephemeral_cache and active_ephemeral_category is not None:
            deleted = delete_new_category_cache_files(
                args.fjsar_disk_cache_path,
                active_ephemeral_category,
                active_ephemeral_snapshot,
            )
            ephemeral_deleted_files += deleted
            print(
                f"{active_ephemeral_category}: removed {deleted} newly-created "
                "ephemeral FLUX cache files during cleanup"
            )
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
    result = {
        "matcher": "attention_top20_dinov2_identity",
        "method_hypothesis": METHOD_HYPOTHESIS,
        "protocol": {
            "flux_router": "block28 exact mutual cross-attention, no coordinate bias",
            "candidate_topk": int(args.fjsar_candidate_topk),
            "subset": str(args.subset),
            "pairs_per_category": int(args.pairs_per_cat),
            "split_seed": int(args.split_seed),
            "dino_model": "dinov2_vitb14",
            "dino_feature": "block11 pre-final-norm token",
            "dino_input": "840x840 aspect-preserving zero canvas",
            "dino_grid": "60x60",
            "native_nn_diagnostics": {
                "top1_top2_margin": True,
                "top1_nonlocal_margin": True,
                "cycle_back": True,
                "candidate_native_cosine": True,
                "candidate_native_rank": True,
                "nonlocal_radius": int(args.native_nonlocal_radius),
            },
            "ephemeral_category_cache": ephemeral_cache,
            "ephemeral_deleted_files": int(ephemeral_deleted_files),
        },
        "categories": category_results,
        "all": all_result,
        "mechanism_summary": summary,
        "fjsar_memory_cache": fjsar_memory_cache.stats(),
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    root = output_path.with_suffix("")
    audit_path = Path(f"{root}_attention_top20_dinov2_identity_audit.json")
    summary_path = Path(f"{root}_attention_top20_dinov2_identity_summary.json")
    audit_payload = {
        "matcher": result["matcher"],
        "method_hypothesis": METHOD_HYPOTHESIS,
        "protocol": result["protocol"],
        "summary": summary,
        "pair_records": pair_records,
    }
    summary_payload = {
        "matcher": result["matcher"],
        "method_hypothesis": METHOD_HYPOTHESIS,
        "protocol": result["protocol"],
        "summary": summary,
    }
    result["audit_path"] = str(audit_path)
    result["summary_path"] = str(summary_path)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    audit_path.write_text(json.dumps(audit_payload, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(
        "Matcher: attention_top20_dinov2_identity\n"
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
    parser.add_argument("--native_nonlocal_radius", type=int, default=8)
    parser.add_argument("--fjsar_memory_cache_gb", type=float, default=4.0)
    parser.add_argument("--fjsar_disk_cache_path", required=True)
    parser.add_argument("--fjsar_require_disk_cache", action="store_true", default=False)
    parser.add_argument(
        "--fjsar_ephemeral_category_cache",
        action="store_true",
        default=False,
        help=(
            "Allow cache misses, process one category, then delete only cache files "
            "created during that completed category. Existing cache files are preserved."
        ),
    )
    parser.add_argument("--fjsar_disk_cache_min_free_gb", type=float, default=0.0)
    parser.add_argument("--fjsar_shared_noise", action="store_true", default=False)
    parser.add_argument("--dino_backend", choices=("transformers", "torch_hub"), default="transformers")
    parser.add_argument("--dino_precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--dino_local_files_only", action="store_true", default=False)
    parser.add_argument("--dino_model_repo", default=None)
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

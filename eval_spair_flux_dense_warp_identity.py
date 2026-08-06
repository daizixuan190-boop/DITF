"""Evaluate a train-free FLUX dense-warp identity descriptor on SPair-71k.

The existing exact block-28 joint replay supplies the complete bidirectional
cross-attention posterior.  A local-mode continuous warp is built in each
direction, validated by cycle and local-Jacobian consistency, and encoded as a
small pair-conditioned identity branch next to the untouched native DiTF
descriptor.  Final prediction remains the official global cosine NN.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ephemeral_category_cache import (
    category_cache_snapshot,
    delete_new_category_cache_files,
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
from flux_dense_warp_identity import (
    DenseWarpIdentityConfig,
    build_dense_warp_identity_maps,
    sample_dense_field,
)
from spair_matchers import cosine_nn_predict, flux_fjsar_attention_candidates


METHOD_HYPOTHESIS = {
    "name": "FLUX Full-Attention Bidirectional Dense-Warp Identity",
    "mechanism_hypothesis": (
        "Exact FLUX cross-attention provides a high-recall semantic posterior, but A@V "
        "collapses candidate identity. Retaining the complete candidate axis as a local-mode "
        "bidirectional continuous warp creates pair-conditioned identity coordinates while "
        "the original DiTF descriptor preserves native appearance evidence."
    ),
    "attention_operator": "exact_block28_full_bidirectional_mutual_posterior",
    "posterior_readout": "local_mode_mean_shift_not_global_barycenter",
    "field_validation": "bidirectional_cycle_and_local_jacobian_consistency",
    "identity_feature": "symmetric_shared_coordinate_fourier_encoding",
    "native_feature": "untouched_official_DiTF_descriptor",
    "prediction": "official_global_cosine_NN_on_pair_conditioned_descriptor",
    "attention_topk_used_for_prediction": False,
    "native_fallback": False,
    "external_correspondence_model": False,
    "training": False,
    "gt_used_for_inference": False,
}


def _load_category_pairs(
    args: argparse.Namespace,
    test_path: str,
    category: str,
    pair_names: Sequence[str],
) -> list[tuple[str, dict[str, Any]]]:
    selected = _select_category_pairs(pair_names, args)
    if int(args.max_pairs_per_cat) > 0:
        selected = selected[: int(args.max_pairs_per_cat)]
    output = []
    for pair_name in selected:
        with open(os.path.join(args.dataset_path, test_path, pair_name)) as handle:
            output.append((pair_name, json.load(handle)))
    return output


def _candidate_pixels_xy(
    candidate_pixels: torch.Tensor,
    target_width: int,
) -> list[list[list[int]]]:
    return [
        [
            [int(pixel % target_width), int(pixel // target_width)]
            for pixel in row
        ]
        for row in candidate_pixels.detach().cpu().tolist()
    ]


def _normalized_warp_to_pixels(
    normalized: torch.Tensor,
    target_size: Sequence[int],
) -> list[list[int]]:
    target_height, target_width = int(target_size[0]), int(target_size[1])
    scale = normalized.new_tensor(
        [float(max(target_width - 1, 1)), float(max(target_height - 1, 1))]
    )
    pixels = torch.round(normalized.clamp(0.0, 1.0) * scale).long()
    pixels[:, 0].clamp_(0, target_width - 1)
    pixels[:, 1].clamp_(0, target_height - 1)
    return [[int(row[0]), int(row[1])] for row in pixels.detach().cpu().tolist()]


def _sample_scalar(
    field: torch.Tensor,
    points: Sequence[Sequence[float]],
    image_size: Sequence[int],
) -> list[float]:
    sampled = sample_dense_field(field.unsqueeze(-1), points, image_size)[:, 0]
    return [float(value) for value in sampled.detach().float().cpu().tolist()]


def _point_summary(pair_records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    points = [point for pair in pair_records for point in pair["points"]]
    hard = [point for point in points if bool(point["both_wrong_top20_hit"])]

    def branch(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        total = max(1, len(rows))
        return {
            "points": len(rows),
            "baseline_top1": float(sum(row["baseline_pck_hit"] for row in rows) / total),
            "attention_top1": float(
                sum(row["attention_top1_pck_hit"] for row in rows) / total
            ),
            "attention_top20": float(
                sum(row["attention_top20_pck_hit"] for row in rows) / total
            ),
            "direct_warp_top1": float(
                sum(row["direct_warp_pck_hit"] for row in rows) / total
            ),
            "identity_only_top1": float(
                sum(row["identity_only_pck_hit"] for row in rows) / total
            ),
            "fused_top1": float(sum(row["method_pck_hit"] for row in rows) / total),
            "mean_basin_mass": float(
                np.mean([row["warp_state"]["basin_mass"] for row in rows])
            )
            if rows
            else None,
            "mean_cycle_error_cells": float(
                np.mean([row["warp_state"]["cycle_error_cells"] for row in rows])
            )
            if rows
            else None,
            "mean_jacobian_confidence": float(
                np.mean([row["warp_state"]["jacobian_confidence"] for row in rows])
            )
            if rows
            else None,
            "mean_reliability": float(
                np.mean([row["warp_state"]["reliability"] for row in rows])
            )
            if rows
            else None,
        }

    return {
        "pairs": len(pair_records),
        "points": len(points),
        "baseline_correct": int(sum(row["baseline_pck_hit"] for row in points)),
        "method_correct": int(sum(row["method_pck_hit"] for row in points)),
        "rescued_vs_baseline": int(sum(row["rescued_vs_baseline"] for row in points)),
        "harmed_vs_baseline": int(sum(row["harmed_vs_baseline"] for row in points)),
        "candidate_missing_gt": int(
            sum(not row["attention_top20_pck_hit"] for row in points)
        ),
        "all_points": branch(points),
        "both_wrong_top20_hit": branch(hard),
    }


def _metric_block(pair_records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    points = [point for pair in pair_records for point in pair["points"]]
    pair_baseline = [
        np.mean([point["baseline_pck_hit"] for point in pair["points"]])
        for pair in pair_records
    ]
    pair_method = [
        np.mean([point["method_pck_hit"] for point in pair["points"]])
        for pair in pair_records
    ]
    baseline_correct = int(sum(point["baseline_pck_hit"] for point in points))
    method_correct = int(sum(point["method_pck_hit"] for point in points))
    return {
        "pairs": len(pair_records),
        "points": len(points),
        "baseline_image": 100.0 * float(np.mean(pair_baseline)) if pair_records else 0.0,
        "method_image": 100.0 * float(np.mean(pair_method)) if pair_records else 0.0,
        "baseline_point": 100.0 * baseline_correct / max(1, len(points)),
        "method_point": 100.0 * method_correct / max(1, len(points)),
        "point_gain": 100.0 * (method_correct - baseline_correct) / max(1, len(points)),
        "improved_count": int(sum(point["rescued_vs_baseline"] for point in points)),
        "harmed_count": int(sum(point["harmed_vs_baseline"] for point in points)),
    }


def evaluate(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    ephemeral_cache = bool(args.fjsar_ephemeral_category_cache)
    if not args.fjsar_disk_cache_path:
        raise ValueError("dense-warp evaluation requires a FLUX replay cache path")
    if args.fjsar_require_disk_cache and ephemeral_cache:
        raise ValueError(
            "--fjsar_require_disk_cache and --fjsar_ephemeral_category_cache are mutually exclusive"
        )
    if not args.fjsar_require_disk_cache and not ephemeral_cache:
        raise ValueError(
            "Use either a precomputed --fjsar_require_disk_cache or the explicit "
            "--fjsar_ephemeral_category_cache low-disk protocol"
        )

    config = DenseWarpIdentityConfig()
    test_path, all_categories, cat2json, _cat2img = _load_pairs(args.dataset_path)
    with open("spair_detailed_captions.json") as handle:
        captions = json.load(handle)
    featurizer, model, blocks = _load_flux_fjsar_runtime(args, all_categories)
    capture = _make_fjsar_capture(args, model)
    memory_cache = _FjsarMemoryCache(
        int(float(args.fjsar_memory_cache_gb) * (1024**3))
    )
    pre_norm = nn.LayerNorm(3072, elementwise_affine=False, eps=1e-6).to(device)
    pair_records: list[dict[str, Any]] = []
    category_results: dict[str, Any] = {}
    ephemeral_deleted_files = 0
    active_category: str | None = None
    active_snapshot: set[Path] = set()

    try:
        for category in all_categories:
            snapshot = (
                category_cache_snapshot(args.fjsar_disk_cache_path, category)
                if ephemeral_cache
                else set()
            )
            if ephemeral_cache:
                active_category = category
                active_snapshot = snapshot
            category_pairs = _load_category_pairs(
                args, test_path, category, cat2json[category]
            )
            category_records: list[dict[str, Any]] = []

            for pair_name, data in tqdm(category_pairs, desc=f"evaluate {category}"):
                source_size = data["src_imsize"][:2][::-1]
                target_size = data["trg_imsize"][:2][::-1]
                source_entry = _get_flux_fjsar_entry(
                    args.dataset_path,
                    category,
                    data["src_imname"],
                    captions[category + data["src_imname"]],
                    args,
                    featurizer,
                    capture,
                    memory_cache,
                )
                target_entry = _get_flux_fjsar_entry(
                    args.dataset_path,
                    category,
                    data["trg_imname"],
                    captions[category + data["trg_imname"]],
                    args,
                    featurizer,
                    capture,
                    memory_cache,
                )
                source_native = _prepare_feature_tensors(
                    source_entry["feature"], source_entry["ada"], args, pre_norm, device
                )
                target_native = _prepare_feature_tensors(
                    target_entry["feature"], target_entry["ada"], args, pre_norm, device
                )

                source_native_full = F.interpolate(
                    source_native.to(torch.float16), size=source_size, mode="bilinear"
                )
                target_native_full = F.interpolate(
                    target_native.to(torch.float16), size=target_size, mode="bilinear"
                )
                baseline_predictions = cosine_nn_predict(
                    source_native_full, target_native_full, data["src_kps"]
                )
                del source_native_full, target_native_full

                with torch.no_grad():
                    candidate_pixels, _candidate_scores, attention_state = (
                        flux_fjsar_attention_candidates(
                            src_replay_state=source_entry["replay_state"],
                            trg_replay_state=target_entry["replay_state"],
                            blocks=blocks,
                            points=data["src_kps"],
                            source_size=source_size,
                            target_size=target_size,
                            candidate_topk=20,
                            interaction_mode="exact",
                            use_coordinate_bias=False,
                        )
                    )
                    attention = attention_state["attention"]
                    source_state = attention_state["source_state"]
                    target_state = attention_state["target_state"]
                    dense = build_dense_warp_identity_maps(
                        source_native,
                        target_native,
                        attention["p_ab"],
                        attention["p_ba"],
                        source_grid=(
                            int(source_state.image_height),
                            int(source_state.image_width),
                        ),
                        target_grid=(
                            int(target_state.image_height),
                            int(target_state.image_width),
                        ),
                        config=config,
                    )

                source_fused_full = F.interpolate(
                    dense["source_fused"].to(torch.float16),
                    size=source_size,
                    mode="bilinear",
                )
                target_fused_full = F.interpolate(
                    dense["target_fused"].to(torch.float16),
                    size=target_size,
                    mode="bilinear",
                )
                method_predictions = cosine_nn_predict(
                    source_fused_full, target_fused_full, data["src_kps"]
                )
                del source_fused_full, target_fused_full

                source_identity_full = F.interpolate(
                    dense["source_identity"].to(torch.float16),
                    size=source_size,
                    mode="bilinear",
                )
                target_identity_full = F.interpolate(
                    dense["target_identity"].to(torch.float16),
                    size=target_size,
                    mode="bilinear",
                )
                identity_predictions = cosine_nn_predict(
                    source_identity_full, target_identity_full, data["src_kps"]
                )
                del source_identity_full, target_identity_full

                mapped = sample_dense_field(
                    dense["warp_ab"], data["src_kps"], source_size
                )
                direct_predictions = _normalized_warp_to_pixels(mapped, target_size)
                basin_values = _sample_scalar(
                    dense["basin_mass_a"], data["src_kps"], source_size
                )
                cycle_values = _sample_scalar(
                    dense["cycle_error_a"], data["src_kps"], source_size
                )
                jacobian_values = _sample_scalar(
                    dense["jacobian_confidence_a"], data["src_kps"], source_size
                )
                reliability_values = _sample_scalar(
                    dense["reliability_a"], data["src_kps"], source_size
                )
                candidate_xy = _candidate_pixels_xy(
                    candidate_pixels, int(target_size[1])
                )
                threshold = max(
                    data["trg_bndbox"][3] - data["trg_bndbox"][1],
                    data["trg_bndbox"][2] - data["trg_bndbox"][0],
                )
                points = []
                for index, target_point in enumerate(data["trg_kps"]):
                    baseline_hit = bool(
                        _pck(baseline_predictions[index], target_point, threshold)
                    )
                    method_hit = bool(
                        _pck(method_predictions[index], target_point, threshold)
                    )
                    direct_hit = bool(
                        _pck(direct_predictions[index], target_point, threshold)
                    )
                    identity_hit = bool(
                        _pck(identity_predictions[index], target_point, threshold)
                    )
                    candidate_hits = [
                        bool(_pck(candidate, target_point, threshold))
                        for candidate in candidate_xy[index]
                    ]
                    attention_top1_hit = bool(candidate_hits[0])
                    attention_top20_hit = bool(any(candidate_hits))
                    both_wrong_top20_hit = bool(
                        not baseline_hit
                        and not attention_top1_hit
                        and attention_top20_hit
                    )
                    points.append(
                        {
                            "keypoint_index": int(index),
                            "source_point": list(data["src_kps"][index]),
                            "target_point": list(target_point),
                            "baseline_prediction": baseline_predictions[index],
                            "method_prediction": method_predictions[index],
                            "identity_only_prediction": identity_predictions[index],
                            "direct_warp_prediction": direct_predictions[index],
                            "baseline_pck_hit": baseline_hit,
                            "method_pck_hit": method_hit,
                            "identity_only_pck_hit": identity_hit,
                            "direct_warp_pck_hit": direct_hit,
                            "attention_top1_pck_hit": attention_top1_hit,
                            "attention_top20_pck_hit": attention_top20_hit,
                            "both_wrong_top20_hit": both_wrong_top20_hit,
                            "rescued_vs_baseline": bool(method_hit and not baseline_hit),
                            "harmed_vs_baseline": bool(baseline_hit and not method_hit),
                            "attention_candidates": candidate_xy[index],
                            "warp_state": {
                                "predicted_target_normalized": [
                                    float(value) for value in mapped[index].detach().cpu()
                                ],
                                "basin_mass": basin_values[index],
                                "cycle_error_cells": cycle_values[index],
                                "jacobian_confidence": jacobian_values[index],
                                "reliability": reliability_values[index],
                            },
                        }
                    )

                record = {
                    "category": category,
                    "pair_json": pair_name,
                    "src_image": data["src_imname"],
                    "trg_image": data["trg_imname"],
                    "keypoint_count": len(points),
                    "points": points,
                }
                category_records.append(record)
                pair_records.append(record)
                del dense, attention_state, attention, source_native, target_native

            category_results[category] = _metric_block(category_records)
            print(
                f"{category}: baseline image/point="
                f"{category_results[category]['baseline_image']:.2f}/"
                f"{category_results[category]['baseline_point']:.2f}, "
                f"method image/point={category_results[category]['method_image']:.2f}/"
                f"{category_results[category]['method_point']:.2f}"
            )
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if ephemeral_cache:
                deleted = delete_new_category_cache_files(
                    args.fjsar_disk_cache_path, category, snapshot
                )
                ephemeral_deleted_files += deleted
                print(
                    f"{category}: removed {deleted} newly-created ephemeral FLUX cache files"
                )
                active_category = None
                active_snapshot = set()
    finally:
        if ephemeral_cache and active_category is not None:
            deleted = delete_new_category_cache_files(
                args.fjsar_disk_cache_path, active_category, active_snapshot
            )
            ephemeral_deleted_files += deleted
            print(
                f"{active_category}: removed {deleted} newly-created ephemeral FLUX "
                "cache files during cleanup"
            )
        capture.close()
        del model, featurizer, blocks
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    all_metrics = _metric_block(pair_records)
    mechanism_summary = _point_summary(pair_records)
    result = {
        "matcher": "flux_dense_warp_identity",
        "method_hypothesis": METHOD_HYPOTHESIS,
        "protocol": {
            "subset": str(args.subset),
            "pairs_per_category": int(args.pairs_per_cat),
            "split_seed": int(args.split_seed),
            "flux_block": int(args.k[0]),
            "timestep": int(args.t),
            "ensemble_size": int(args.ensemble_size),
            "cross_attention": "exact full bidirectional posterior; no coordinate bias",
            "dense_warp_config": asdict(config),
            "native_descriptor": "official block feature + benchmark AdaLN/CD protocol",
            "final_matcher": "official global cosine NN",
            "ephemeral_category_cache": ephemeral_cache,
            "ephemeral_deleted_files": int(ephemeral_deleted_files),
        },
        "categories": category_results,
        "all": all_metrics,
        "mechanism_summary": mechanism_summary,
        "fjsar_memory_cache": memory_cache.stats(),
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    root = output_path.with_suffix("")
    audit_path = Path(f"{root}_flux_dense_warp_identity_audit.json")
    summary_path = Path(f"{root}_flux_dense_warp_identity_summary.json")
    result["audit_path"] = str(audit_path)
    result["summary_path"] = str(summary_path)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    audit_path.write_text(
        json.dumps(
            {
                "matcher": result["matcher"],
                "method_hypothesis": METHOD_HYPOTHESIS,
                "protocol": result["protocol"],
                "summary": mechanism_summary,
                "pair_records": pair_records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(
            {
                "matcher": result["matcher"],
                "method_hypothesis": METHOD_HYPOTHESIS,
                "protocol": result["protocol"],
                "summary": mechanism_summary,
                "metrics": all_metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    hard = mechanism_summary["both_wrong_top20_hit"]
    print("Matcher: flux_dense_warp_identity")
    print(
        f"Baseline All per image/point: {all_metrics['baseline_image']:.2f} / "
        f"{all_metrics['baseline_point']:.2f}"
    )
    print(
        f"Method All per image/point: {all_metrics['method_image']:.2f} / "
        f"{all_metrics['method_point']:.2f}; point gain={all_metrics['point_gain']:.2f}"
    )
    print(
        "Hard both-wrong direct/identity/fused: "
        f"{100.0 * hard['direct_warp_top1']:.2f} / "
        f"{100.0 * hard['identity_only_top1']:.2f} / "
        f"{100.0 * hard['fused_top1']:.2f}"
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
    parser.add_argument(
        "--subset", choices=("all", "discovery", "heldout"), default="discovery"
    )
    parser.add_argument("--pairs_per_cat", type=int, default=20)
    parser.add_argument("--split_seed", type=int, default=2027)
    parser.add_argument("--max_pairs_per_cat", type=int, default=0)
    parser.add_argument("--fjsar_memory_cache_gb", type=float, default=4.0)
    parser.add_argument("--fjsar_disk_cache_path", required=True)
    parser.add_argument("--fjsar_require_disk_cache", action="store_true", default=False)
    parser.add_argument(
        "--fjsar_ephemeral_category_cache", action="store_true", default=False
    )
    parser.add_argument("--fjsar_disk_cache_min_free_gb", type=float, default=0.0)
    parser.add_argument("--fjsar_shared_noise", action="store_true", default=False)
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

"""Audit a training-only VGGT-FGW teacher before expensive verifier training."""

from __future__ import annotations

import argparse
import gc
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
from tqdm import tqdm

from attention_identity_verifier import geometry_fgw_pseudo_targets
from eval_spair_attention_identity_verifier import (
    _load_category_pairs,
    _pck_hit,
    _pixel_to_xy,
    _predict_baseline,
)
from vggt_geometry_teacher import extract_vggt_geometry_maps


def _summary(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    points = [point for record in records for point in record["points"]]
    count = max(1, len(points))
    names = (
        "baseline",
        "attention_top1",
        "native_in_attention",
        "geometry_teacher",
        "attention_top20_oracle",
    )
    correct = {
        name: sum(bool(point["pck_hits"][name]) for point in points)
        for name in names
    }
    baseline_correct = [point for point in points if point["pck_hits"]["baseline"]]
    baseline_wrong = [point for point in points if not point["pck_hits"]["baseline"]]
    return {
        "pairs": len(records),
        "points": len(points),
        "point_pck": {name: 100.0 * value / count for name, value in correct.items()},
        "geometry_teacher_vs_baseline": {
            "rescued": sum(
                point["pck_hits"]["geometry_teacher"] for point in baseline_wrong
            ),
            "harmed": sum(
                not point["pck_hits"]["geometry_teacher"] for point in baseline_correct
            ),
        },
        "mean_query_weight": float(
            sum(float(point["geometry_teacher"]["query_weight"]) for point in points)
            / count
        ),
        "weighted_query_coverage": float(
            sum(float(point["geometry_teacher"]["query_weight"]) > 0.05 for point in points)
            / count
        ),
        "mean_anchor_count": float(
            sum(float(record["geometry_teacher_diagnostics"]["anchor_count"]) for record in records)
            / max(1, len(records))
        ),
        "geometry_used_pair_fraction": float(
            sum(bool(record["geometry_teacher_diagnostics"]["used_geometry"]) for record in records)
            / max(1, len(records))
        ),
    }


def evaluate(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    from eval_spair_matcher_ablation import (
        _extract_flux_fjsar_entry,
        _get_flux_fjsar_entry,
        _load_flux_fjsar_runtime,
        _load_pairs,
        _make_fjsar_capture,
        _prepare_feature_tensors,
    )
    from spair_matchers import flux_fjsar_candidate_feature_batch

    if tuple(map(int, args.k)) != (28,) or int(args.t) != 260 or int(args.ensemble_size) != 8:
        raise ValueError("locked geometry audit requires --k 28 --t 260 --ensemble_size 8")
    if not bool(args.fjsar_shared_noise):
        raise ValueError("locked geometry audit requires --fjsar_shared_noise")
    use_disk_cache = bool(args.fjsar_disk_cache_path)
    if not use_disk_cache and not bool(args.extract_native_in_memory):
        raise ValueError("use either --fjsar_disk_cache_path or --extract_native_in_memory")
    if bool(args.fjsar_require_disk_cache) and not use_disk_cache:
        raise ValueError("--fjsar_require_disk_cache needs --fjsar_disk_cache_path")

    test_path, categories, cat2json, _cat2img = _load_pairs(args.dataset_path)
    category_pairs = OrderedDict(
        (
            category,
            _load_category_pairs(args, test_path, cat2json[category]),
        )
        for category in categories
    )
    image_paths = {
        (category, data[field]): Path(args.dataset_path)
        / "JPEGImages"
        / category
        / data[field]
        for category, pairs in category_pairs.items()
        for _pair_name, data in pairs
        for field in ("src_imname", "trg_imname")
    }
    geometry_maps = extract_vggt_geometry_maps(
        image_paths,
        device=device,
        model_name=str(args.vggt_model),
        input_size=int(args.vggt_input_size),
    )

    captions = json.loads(Path(args.captions_json).read_text(encoding="utf-8"))
    args.device = str(device)
    featurizer, flux_model, blocks = _load_flux_fjsar_runtime(args, categories)
    capture = _make_fjsar_capture(args, flux_model)
    pre_norm = nn.LayerNorm(3072, elementwise_affine=False, eps=1e-6).to(device)
    all_records: list[dict[str, Any]] = []
    category_results: dict[str, Any] = OrderedDict()
    try:
        for category, pairs in category_pairs.items():
            required_images = sorted(
                {
                    data[field]
                    for _pair_name, data in pairs
                    for field in ("src_imname", "trg_imname")
                }
            )
            entries = {}
            for image_name in tqdm(required_images, desc=f"load replay {category}"):
                if use_disk_cache:
                    entries[image_name] = _get_flux_fjsar_entry(
                        args.dataset_path,
                        category,
                        image_name,
                        captions[category + image_name],
                        args,
                        featurizer,
                        capture,
                        None,
                    )
                else:
                    entries[image_name] = _extract_flux_fjsar_entry(
                        args.dataset_path,
                        category,
                        image_name,
                        captions[category + image_name],
                        args,
                        featurizer,
                        capture,
                    )
            category_records = []
            for pair_name, data in tqdm(pairs, desc=f"audit geometry {category}"):
                source_entry = entries[data["src_imname"]]
                target_entry = entries[data["trg_imname"]]
                source_native = _prepare_feature_tensors(
                    source_entry["feature"], source_entry["ada"], args, pre_norm, device
                )
                target_native = _prepare_feature_tensors(
                    target_entry["feature"], target_entry["ada"], args, pre_norm, device
                )
                source_size = data["src_imsize"][:2][::-1]
                target_size = data["trg_imsize"][:2][::-1]
                baseline = _predict_baseline(
                    source_native,
                    target_native,
                    source_size,
                    target_size,
                    data["src_kps"],
                )
                batch = flux_fjsar_candidate_feature_batch(
                    source_native,
                    target_native,
                    data["src_kps"],
                    source_size,
                    target_size,
                    src_replay_state=source_entry["replay_state"],
                    trg_replay_state=target_entry["replay_state"],
                    blocks=blocks,
                    interaction_mode="exact",
                    use_coordinate_bias=False,
                    candidate_topk=int(args.candidate_topk),
                )
                teacher_targets, query_weights, teacher_diagnostics = (
                    geometry_fgw_pseudo_targets(
                        batch,
                        source_native,
                        target_native,
                        geometry_maps[(category, data["src_imname"])],
                        geometry_maps[(category, data["trg_imname"])],
                        alpha=float(args.geometry_alpha),
                        rho=float(args.uot_rho),
                        refinement_steps=int(args.refinement_steps),
                        sinkhorn_iterations=int(args.sinkhorn_iterations),
                        max_anchors=int(args.max_anchors),
                        minimum_anchors=int(args.minimum_anchors),
                    )
                )
                candidate_pixels = batch["candidate_pixels"].long()
                geometry_ranks = teacher_targets.argmax(dim=1)
                native_ranks = batch["feature_groups"]["native_control"].squeeze(2).argmax(dim=1)
                point_rows = []
                target_width = int(target_size[1])
                threshold = max(
                    data["trg_bndbox"][3] - data["trg_bndbox"][1],
                    data["trg_bndbox"][2] - data["trg_bndbox"][0],
                )
                for index, target_point in enumerate(data["trg_kps"]):
                    predictions = {
                        "baseline": baseline[index],
                        "attention_top1": _pixel_to_xy(
                            int(candidate_pixels[index, 0]), target_width
                        ),
                        "native_in_attention": _pixel_to_xy(
                            int(candidate_pixels[index, native_ranks[index]]), target_width
                        ),
                        "geometry_teacher": _pixel_to_xy(
                            int(candidate_pixels[index, geometry_ranks[index]]), target_width
                        ),
                    }
                    hits = {
                        name: _pck_hit(prediction, target_point, threshold)
                        for name, prediction in predictions.items()
                    }
                    hits["attention_top20_oracle"] = any(
                        _pck_hit(
                            _pixel_to_xy(int(pixel), target_width),
                            target_point,
                            threshold,
                        )
                        for pixel in candidate_pixels[index]
                    )
                    point_rows.append({
                        "index": int(index),
                        "pck_hits": hits,
                        "geometry_teacher": {
                            "selected_rank": int(geometry_ranks[index]),
                            "query_weight": float(query_weights[index]),
                        },
                    })
                record = {
                    "category": category,
                    "pair_name": pair_name,
                    "points": point_rows,
                    "geometry_teacher_diagnostics": teacher_diagnostics,
                }
                category_records.append(record)
                all_records.append(record)
                del source_native, target_native, batch, teacher_targets
            category_results[category] = _summary(category_records)
            del entries
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        capture.close()

    summary = _summary(all_records)
    result = {
        "matcher": "attention_top20_vggt_fgw_teacher_audit",
        "supervision": "training-only pretrained_3d_foundation_teacher",
        "spair_keypoints_used_for_teacher": False,
        "spair_keypoints_used_for_metrics_only": True,
        "vggt_used_for_final_inference": False,
        "categories": category_results,
        "summary": summary,
        "protocol": {
            "vggt_model": str(args.vggt_model),
            "geometry_alpha": float(args.geometry_alpha),
            "uot_rho": float(args.uot_rho),
            "refinement_steps": int(args.refinement_steps),
            "max_anchors": int(args.max_anchors),
            "minimum_anchors": int(args.minimum_anchors),
            "candidate_topk": int(args.candidate_topk),
        },
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    pck = summary["point_pck"]
    comparison = summary["geometry_teacher_vs_baseline"]
    print(
        "Baseline/attention/native/geometry/oracle All point: "
        f"{pck['baseline']:.2f} / {pck['attention_top1']:.2f} / "
        f"{pck['native_in_attention']:.2f} / {pck['geometry_teacher']:.2f} / "
        f"{pck['attention_top20_oracle']:.2f}"
    )
    print(
        "Geometry teacher vs baseline: "
        f"rescued={comparison['rescued']}, harmed={comparison['harmed']}, "
        f"coverage={100.0 * summary['weighted_query_coverage']:.2f}, "
        f"anchors={summary['mean_anchor_count']:.2f}"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--captions_json", default="spair_detailed_captions.json")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--img_size", nargs="+", type=int, default=[640, 640])
    parser.add_argument("--t", type=int, default=260)
    parser.add_argument("--k", nargs="+", type=int, default=[28])
    parser.add_argument("--ensemble_size", type=int, default=8)
    parser.add_argument("--cd", action="store_true", default=False)
    parser.add_argument("--subset", choices=("all", "discovery", "heldout"), default="discovery")
    parser.add_argument("--pairs_per_cat", type=int, default=20)
    parser.add_argument("--split_seed", type=int, default=2027)
    parser.add_argument("--max_pairs_per_cat", type=int, default=0)
    parser.add_argument("--candidate_topk", type=int, default=20)
    parser.add_argument("--extract_native_in_memory", action="store_true", default=False)
    parser.add_argument("--fjsar_shared_noise", action="store_true", default=True)
    parser.add_argument("--fjsar_disk_cache_path", default="")
    parser.add_argument("--fjsar_require_disk_cache", action="store_true", default=False)
    parser.add_argument("--vggt_model", default="facebook/VGGT-1B")
    parser.add_argument("--vggt_input_size", type=int, default=518)
    parser.add_argument("--geometry_alpha", type=float, default=0.3)
    parser.add_argument("--uot_rho", type=float, default=0.75)
    parser.add_argument("--refinement_steps", type=int, default=5)
    parser.add_argument("--sinkhorn_iterations", type=int, default=20)
    parser.add_argument("--max_anchors", type=int, default=64)
    parser.add_argument("--minimum_anchors", type=int, default=4)
    parser.set_defaults(
        matcher="attention_geometry_teacher_audit",
        fjsar_disk_cache_min_free_gb=2.0,
        fjsar_multilayer_identity_audit=False,
        fjsar_multilayer_blocks=(),
        fjsar_trajectory_blocks=(),
        fjsar_multi_timestep_attention_identity_audit=False,
    )
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    run_device = torch.device(
        parsed.device if parsed.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    evaluate(parsed, run_device)

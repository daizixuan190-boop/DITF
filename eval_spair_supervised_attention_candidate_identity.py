"""Evaluate the supervised FLUX attention candidate capacity diagnostic."""

from __future__ import annotations

import argparse
import gc
import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from attention_identity_verifier import load_verifier_checkpoint, select_candidate_pixels
from eval_spair_attention_identity_verifier import (
    _load_category_pairs,
    _pck_hit,
    _pixel_to_xy,
    _predict_baseline,
)
from train_flux_attention_candidate_identity_supervised import (
    _candidate_batch_with_baseline,
)


BRANCHES = (
    "baseline",
    "attention_top1",
    "resolver",
    "attention_top20_oracle",
    "resolver_pool_oracle",
)


def candidate_kind(selected_rank: int, *, attention_candidate_count: int) -> str:
    rank = int(selected_rank)
    attention_count = int(attention_candidate_count)
    if 0 <= rank < attention_count:
        return "attention"
    if rank == attention_count:
        return "baseline_fallback"
    raise ValueError(
        f"candidate rank {rank} is outside attention top-{attention_count} plus one fallback"
    )


def _branch_metrics(records: Sequence[Mapping[str, Any]], branch: str) -> dict[str, Any]:
    pair_scores = []
    hits = []
    for pair in records:
        pair_hits = [bool(point["pck_hits"][branch]) for point in pair["points"]]
        pair_scores.append(float(np.mean(pair_hits)) if pair_hits else 0.0)
        hits.extend(pair_hits)
    return {
        "pairs": len(records),
        "points": len(hits),
        "image": 100.0 * float(np.mean(pair_scores)) if pair_scores else 0.0,
        "point": 100.0 * float(sum(hits)) / max(1, len(hits)),
        "correct": int(sum(hits)),
    }


def _comparison(
    points: Sequence[Mapping[str, Any]],
    *,
    reference: str,
    selected: str,
    oracle: str,
) -> dict[str, Any]:
    reference_correct = sum(bool(row["pck_hits"][reference]) for row in points)
    selected_correct = sum(bool(row["pck_hits"][selected]) for row in points)
    oracle_correct = sum(bool(row["pck_hits"][oracle]) for row in points)
    rescued = sum(
        bool(row["pck_hits"][selected]) and not bool(row["pck_hits"][reference])
        for row in points
    )
    harmed = sum(
        bool(row["pck_hits"][reference]) and not bool(row["pck_hits"][selected])
        for row in points
    )
    retained = sum(
        bool(row["pck_hits"][reference]) and bool(row["pck_hits"][selected])
        for row in points
    )
    gap = max(0, oracle_correct - reference_correct)
    return {
        "rescued": int(rescued),
        "harmed": int(harmed),
        "net_correct": int(selected_correct - reference_correct),
        "rescue_harm_ratio": float(rescued / harmed) if harmed else None,
        f"{reference}_correct_retained": int(retained),
        f"{reference}_correct_retention_rate": (
            float(retained / reference_correct) if reference_correct else 0.0
        ),
        "pool_oracle_gap_points": int(gap),
        "pool_oracle_gap_recovered_fraction": (
            float((selected_correct - reference_correct) / gap) if gap else 0.0
        ),
    }


def summarize_supervised_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    metrics = {branch: _branch_metrics(records, branch) for branch in BRANCHES}
    points = [point for pair in records for point in pair["points"]]
    baseline_selection = sum(
        str(point["selected_kind"]) == "baseline_fallback" for point in points
    )
    return {
        "metrics": metrics,
        "resolver_vs_baseline": _comparison(
            points,
            reference="baseline",
            selected="resolver",
            oracle="resolver_pool_oracle",
        ),
        "resolver_vs_attention": _comparison(
            points,
            reference="attention_top1",
            selected="resolver",
            oracle="attention_top20_oracle",
        ),
        "selection": {
            "points": len(points),
            "baseline_fallback_selected": int(baseline_selection),
            "attention_candidate_selected": int(len(points) - baseline_selection),
            "baseline_fallback_rate": float(baseline_selection / max(1, len(points))),
            "attention_candidate_rate": float(
                (len(points) - baseline_selection) / max(1, len(points))
            ),
        },
    }


def validate_supervised_checkpoint_metadata(
    payload: Mapping[str, Any],
    args: argparse.Namespace,
) -> None:
    metadata = payload.get("training_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("checkpoint lacks training_metadata")
    if str(metadata.get("supervision")) != "spair_train_keypoints":
        raise ValueError("checkpoint supervision is not spair_train_keypoints")
    if not bool(metadata.get("spair_keypoints_used", False)):
        raise ValueError("supervised checkpoint must disclose spair_keypoints_used")
    forbidden = (
        "test_keypoints_used_for_training",
        "test_images_used_for_training",
        "category_labels_used_for_targets",
        "keypoint_ids_used",
        "external_matcher_used",
        "dino_used",
        "roma_used",
    )
    enabled = [name for name in forbidden if bool(metadata.get(name, False))]
    if enabled:
        raise ValueError(f"supervised checkpoint violates diagnostic protocol: {enabled}")
    protocol = metadata.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("checkpoint lacks training protocol metadata")
    expected = {
        "feature_block": int(args.k[0]),
        "timestep": int(args.t),
        "ensemble_size": int(args.ensemble_size),
        "candidate_topk": int(args.candidate_topk),
        "extra_candidate_count": 1,
    }
    mismatches = {
        name: {"checkpoint": protocol.get(name), "evaluation": value}
        for name, value in expected.items()
        if int(protocol.get(name, -1)) != int(value)
    }
    image_size = [int(value) for value in protocol.get("image_size", [])]
    if image_size != [int(value) for value in args.img_size]:
        mismatches["image_size"] = {
            "checkpoint": protocol.get("image_size"),
            "evaluation": [int(value) for value in args.img_size],
        }
    if bool(protocol.get("channel_discard", False)) != bool(args.cd):
        mismatches["channel_discard"] = {
            "checkpoint": protocol.get("channel_discard"),
            "evaluation": bool(args.cd),
        }
    if mismatches:
        raise ValueError(f"checkpoint/evaluation protocol mismatch: {mismatches}")


def _score_batch(model, batch: Mapping[str, Any], device: torch.device) -> torch.Tensor:
    metadata = batch.get("metadata", {})
    if bool(metadata.get("gt_used_for_features", True)):
        raise RuntimeError("evaluation ground truth leaked into candidate features")
    groups = {
        name: batch["feature_groups"][name].to(device=device, dtype=torch.float32)
        for name in model.config.feature_groups
    }
    attention = batch["attention_scores"].to(device=device, dtype=torch.float32)
    with torch.no_grad():
        return model(groups, attention).detach().cpu()


def evaluate(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    from eval_spair_matcher_ablation import (
        _extract_flux_fjsar_entry,
        _get_flux_fjsar_entry,
        _load_flux_fjsar_runtime,
        _load_pairs,
        _make_fjsar_capture,
        _prepare_feature_tensors,
    )

    if tuple(map(int, args.k)) != (28,) or int(args.t) != 260 or int(args.ensemble_size) != 8:
        raise ValueError("locked diagnostic protocol requires --k 28 --t 260 --ensemble_size 8")
    if int(args.candidate_topk) != 20:
        raise ValueError("capacity diagnostic is locked to attention top-20")
    if not bool(args.fjsar_shared_noise):
        raise ValueError("locked diagnostic protocol requires --fjsar_shared_noise")
    use_disk_cache = bool(args.fjsar_disk_cache_path)
    if not use_disk_cache and not bool(args.extract_native_in_memory):
        raise ValueError("use either --fjsar_disk_cache_path or --extract_native_in_memory")
    if bool(args.fjsar_require_disk_cache) and not use_disk_cache:
        raise ValueError("--fjsar_require_disk_cache needs --fjsar_disk_cache_path")

    model, checkpoint = load_verifier_checkpoint(args.verifier_checkpoint, map_location="cpu")
    validate_supervised_checkpoint_metadata(checkpoint, args)
    model = model.to(device).eval()
    args.device = str(device)
    test_path, categories, cat2json, _cat2img = _load_pairs(args.dataset_path)
    captions = json.loads(Path(args.captions_json).read_text(encoding="utf-8"))
    featurizer, flux_model, blocks = _load_flux_fjsar_runtime(args, categories)
    capture = _make_fjsar_capture(args, flux_model)
    pre_norm = nn.LayerNorm(3072, elementwise_affine=False, eps=1e-6).to(device)
    all_records: list[dict[str, Any]] = []
    category_results: dict[str, Any] = OrderedDict()

    try:
        for category in categories:
            category_pairs = _load_category_pairs(args, test_path, cat2json[category])
            required_images = sorted({
                data[field]
                for _pair_name, data in category_pairs
                for field in ("src_imname", "trg_imname")
            })
            entries = {}
            for image_name in tqdm(required_images, desc=f"load replay {category}"):
                caption_key = category + image_name
                if caption_key not in captions:
                    raise KeyError(f"missing detailed caption for {category}/{image_name}")
                if use_disk_cache:
                    entries[image_name] = _get_flux_fjsar_entry(
                        args.dataset_path,
                        category,
                        image_name,
                        captions[caption_key],
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
                        captions[caption_key],
                        args,
                        featurizer,
                        capture,
                    )

            category_records = []
            for pair_name, data in tqdm(category_pairs, desc=f"evaluate {category}"):
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
                batch, baseline = _candidate_batch_with_baseline(
                    source_entry,
                    target_entry,
                    source_native,
                    target_native,
                    data["src_kps"],
                    source_size,
                    target_size,
                    blocks,
                    int(args.candidate_topk),
                )
                scores = _score_batch(model, batch, device)
                candidate_pixels = batch["candidate_pixels"].long()
                resolver_pixels = select_candidate_pixels(scores, candidate_pixels)
                resolver_ranks = scores.argmax(dim=1)
                attention_pixels = candidate_pixels[:, 0]
                target_width = int(target_size[1])
                threshold = max(
                    data["trg_bndbox"][3] - data["trg_bndbox"][1],
                    data["trg_bndbox"][2] - data["trg_bndbox"][0],
                )
                point_records = []
                for index, target_point in enumerate(data["trg_kps"]):
                    attention_prediction = _pixel_to_xy(
                        int(attention_pixels[index]), target_width
                    )
                    resolver_prediction = _pixel_to_xy(
                        int(resolver_pixels[index]), target_width
                    )
                    candidate_predictions = [
                        _pixel_to_xy(int(pixel), target_width)
                        for pixel in candidate_pixels[index]
                    ]
                    candidate_hits = [
                        _pck_hit(prediction, target_point, threshold)
                        for prediction in candidate_predictions
                    ]
                    predictions = {
                        "baseline": baseline[index],
                        "attention_top1": attention_prediction,
                        "resolver": resolver_prediction,
                    }
                    hits = {
                        name: _pck_hit(prediction, target_point, threshold)
                        for name, prediction in predictions.items()
                    }
                    hits["attention_top20_oracle"] = any(
                        candidate_hits[: int(args.candidate_topk)]
                    )
                    hits["resolver_pool_oracle"] = any(candidate_hits)
                    rank = int(resolver_ranks[index])
                    score_row = scores[index].float()
                    top_two = score_row.topk(k=min(2, int(score_row.numel()))).values
                    margin = (
                        float(top_two[0] - top_two[1])
                        if int(top_two.numel()) > 1
                        else float("inf")
                    )
                    point_records.append({
                        "index": int(index),
                        "source_point": list(data["src_kps"][index]),
                        "target_point": list(target_point),
                        "predictions": predictions,
                        "pck_hits": hits,
                        "selected_rank": rank,
                        "selected_kind": candidate_kind(
                            rank,
                            attention_candidate_count=int(args.candidate_topk),
                        ),
                        "score_top1_top2_margin": margin,
                        "candidate_predictions": candidate_predictions,
                        "candidate_scores": [float(value) for value in score_row.tolist()],
                        "candidate_pck_hits": candidate_hits,
                        "candidate_kinds": (
                            ["attention"] * int(args.candidate_topk)
                            + ["baseline_fallback"]
                        ),
                        "gt_used_for_inference": False,
                    })
                record = {
                    "category": category,
                    "pair_name": pair_name,
                    "source_image": data["src_imname"],
                    "target_image": data["trg_imname"],
                    "pck_threshold": float(threshold),
                    "points": point_records,
                }
                category_records.append(record)
                all_records.append(record)
                del source_native, target_native, batch, scores
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            category_results[category] = summarize_supervised_records(category_records)
            metrics = category_results[category]["metrics"]
            print(
                f"{category}: baseline/attention/resolver/pool-oracle point="
                f"{metrics['baseline']['point']:.2f}/"
                f"{metrics['attention_top1']['point']:.2f}/"
                f"{metrics['resolver']['point']:.2f}/"
                f"{metrics['resolver_pool_oracle']['point']:.2f}"
            )
            del entries
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        capture.close()

    summary = summarize_supervised_records(all_records)
    result = {
        "matcher": "supervised_flux_attention_candidate_identity_capacity_diagnostic",
        "claim_scope": "supervised_capacity_diagnostic_not_label_free_method",
        "protocol": {
            "subset": str(args.subset),
            "pairs_per_category": int(args.pairs_per_cat),
            "split_seed": int(args.split_seed),
            "feature_block": int(args.k[0]),
            "timestep": int(args.t),
            "ensemble_size": int(args.ensemble_size),
            "channel_discard": bool(args.cd),
            "candidate_topk": int(args.candidate_topk),
            "extra_candidate_count": 1,
            "candidate_pool": "attention_top20_plus_frozen_ditf_top1",
            "shared_noise": bool(args.fjsar_shared_noise),
            "replay_source": "canonical_disk_cache" if use_disk_cache else "fresh_in_memory",
            "disk_cache_path": str(args.fjsar_disk_cache_path) if use_disk_cache else "",
            "require_disk_cache": bool(args.fjsar_require_disk_cache),
            "gt_used_for_inference": False,
            "ground_truth_used_for_metrics_and_posthoc_audit_only": True,
            "checkpoint": str(args.verifier_checkpoint),
        },
        "checkpoint_training_metadata": checkpoint["training_metadata"],
        "summary": summary,
        "categories": category_results,
        "pair_records": all_records,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    summary_path.write_text(
        json.dumps(
            {
                "matcher": result["matcher"],
                "claim_scope": result["claim_scope"],
                "protocol": result["protocol"],
                "summary": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    metrics = summary["metrics"]
    comparison = summary["resolver_vs_baseline"]
    print("Matcher: supervised FLUX attention candidate identity capacity diagnostic")
    print(
        "Baseline/attention/resolver/attention-oracle/pool-oracle All point: "
        f"{metrics['baseline']['point']:.2f} / "
        f"{metrics['attention_top1']['point']:.2f} / "
        f"{metrics['resolver']['point']:.2f} / "
        f"{metrics['attention_top20_oracle']['point']:.2f} / "
        f"{metrics['resolver_pool_oracle']['point']:.2f}"
    )
    print(
        f"Resolver vs baseline: rescued={comparison['rescued']}, "
        f"harmed={comparison['harmed']}, net={comparison['net_correct']}, "
        f"retention={100.0 * comparison['baseline_correct_retention_rate']:.2f}, "
        f"pool-gap recovered={100.0 * comparison['pool_oracle_gap_recovered_fraction']:.2f}, "
        f"fallback selected={100.0 * summary['selection']['baseline_fallback_rate']:.2f}"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--captions_json", default="spair_detailed_captions.json")
    parser.add_argument("--verifier_checkpoint", required=True)
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
    parser.set_defaults(
        matcher="supervised_attention_candidate_identity_evaluation",
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

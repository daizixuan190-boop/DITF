"""Evaluate a self-supervised FLUX attention identity verifier on SPair-71k."""

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
import torch.nn.functional as F
from tqdm import tqdm

from attention_identity_verifier import (
    load_verifier_checkpoint,
    native_cycle_pseudo_targets,
    select_candidate_pixels,
)
BRANCHES = ("baseline", "attention_top1", "verifier", "attention_top20_oracle")


def _load_category_pairs(
    args: argparse.Namespace,
    test_path: str,
    pair_names: Sequence[str],
) -> list[tuple[str, dict[str, Any]]]:
    from eval_spair_matcher_ablation import _select_category_pairs

    selected = _select_category_pairs(pair_names, args)
    if int(args.max_pairs_per_cat) > 0:
        selected = selected[: int(args.max_pairs_per_cat)]
    rows = []
    for pair_name in selected:
        with open(os.path.join(args.dataset_path, test_path, pair_name), encoding="utf-8") as handle:
            rows.append((pair_name, json.load(handle)))
    return rows


def _predict_baseline(
    source_feature: torch.Tensor,
    target_feature: torch.Tensor,
    source_size: Sequence[int],
    target_size: Sequence[int],
    source_points: Sequence[Sequence[float]],
) -> list[list[int]]:
    from spair_matchers import cosine_nn_predict

    source_full = F.interpolate(
        source_feature.to(torch.float16), size=tuple(map(int, source_size)), mode="bilinear"
    )
    target_full = F.interpolate(
        target_feature.to(torch.float16), size=tuple(map(int, target_size)), mode="bilinear"
    )
    predictions = cosine_nn_predict(source_full, target_full, source_points)
    del source_full, target_full
    return predictions


def _pixel_to_xy(pixel: int, width: int) -> list[int]:
    return [int(pixel % int(width)), int(pixel // int(width))]


def _pck_hit(prediction: Sequence[float], target: Sequence[float], threshold: float) -> bool:
    distance = (
        (float(prediction[0]) - float(target[0])) ** 2
        + (float(prediction[1]) - float(target[1])) ** 2
    ) ** 0.5
    return bool(distance / float(threshold) <= 0.1)


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


def summarize_verifier_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = {branch: _branch_metrics(records, branch) for branch in BRANCHES}
    points = [point for pair in records for point in pair["points"]]
    attention_correct = sum(bool(row["pck_hits"]["attention_top1"]) for row in points)
    verifier_correct = sum(bool(row["pck_hits"]["verifier"]) for row in points)
    oracle_correct = sum(bool(row["pck_hits"]["attention_top20_oracle"]) for row in points)
    rescued = sum(
        bool(row["pck_hits"]["verifier"])
        and not bool(row["pck_hits"]["attention_top1"])
        for row in points
    )
    harmed = sum(
        bool(row["pck_hits"]["attention_top1"])
        and not bool(row["pck_hits"]["verifier"])
        for row in points
    )
    retained = sum(
        bool(row["pck_hits"]["attention_top1"])
        and bool(row["pck_hits"]["verifier"])
        for row in points
    )
    oracle_gap = max(0, oracle_correct - attention_correct)
    return {
        "metrics": metrics,
        "verifier_vs_attention": {
            "rescued": int(rescued),
            "harmed": int(harmed),
            "net_correct": int(verifier_correct - attention_correct),
            "rescue_harm_ratio": float(rescued / harmed) if harmed else None,
            "attention_correct_retained": int(retained),
            "attention_correct_retention_rate": (
                float(retained / attention_correct) if attention_correct else 0.0
            ),
            "attention_oracle_gap_points": int(oracle_gap),
            "oracle_gap_recovered_fraction": (
                float((verifier_correct - attention_correct) / oracle_gap)
                if oracle_gap else 0.0
            ),
        },
    }


def summarize_native_cycle_diagnostics(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Audit the training teacher and fixed confidence gate without using GT to route."""

    points = [point for pair in records for point in pair["points"]]
    audited = [point for point in points if isinstance(point.get("native_cycle"), Mapping)]
    if not audited:
        return {}
    confident = [point for point in audited if bool(point["native_cycle"]["confident"])]

    def _correct(point: Mapping[str, Any], branch: str) -> bool:
        if branch == "teacher":
            return bool(point["native_cycle"]["teacher_pck_hit"])
        return bool(point["pck_hits"][branch])

    def _point_score(rows: Sequence[Mapping[str, Any]], branch: str) -> float:
        return 100.0 * sum(_correct(point, branch) for point in rows) / max(1, len(rows))

    def _baseline_comparison(branch: str) -> dict[str, int]:
        rescued = sum(
            _correct(point, branch) and not _correct(point, "baseline")
            for point in confident
        )
        harmed = sum(
            _correct(point, "baseline") and not _correct(point, branch)
            for point in confident
        )
        return {
            "rescued": int(rescued),
            "harmed": int(harmed),
            "net_correct": int(rescued - harmed),
        }

    verifier_gate_correct = sum(
        _correct(point, "verifier")
        if bool(point["native_cycle"]["confident"])
        else _correct(point, "baseline")
        for point in audited
    )
    teacher_gate_correct = sum(
        _correct(point, "teacher")
        if bool(point["native_cycle"]["confident"])
        else _correct(point, "baseline")
        for point in audited
    )
    model_teacher_agreement = sum(
        int(point["native_cycle"]["verifier_rank"])
        == int(point["native_cycle"]["teacher_rank"])
        for point in confident
    )
    return {
        "points": int(len(audited)),
        "confident_points": int(len(confident)),
        "coverage": float(len(confident) / max(1, len(audited))),
        "confident_subset_pck": {
            branch: _point_score(confident, branch)
            for branch in ("baseline", "attention_top1", "verifier", "teacher")
        },
        "baseline_preserving_gates": {
            "verifier_point": 100.0 * verifier_gate_correct / max(1, len(audited)),
            "teacher_point": 100.0 * teacher_gate_correct / max(1, len(audited)),
        },
        "confident_verifier_vs_baseline": _baseline_comparison("verifier"),
        "confident_teacher_vs_baseline": _baseline_comparison("teacher"),
        "model_teacher_agreement_on_confident": float(
            model_teacher_agreement / max(1, len(confident))
        ),
    }


def _validate_checkpoint_metadata(
    payload: Mapping[str, Any],
    args: argparse.Namespace | None = None,
) -> None:
    metadata = payload.get("training_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("verifier checkpoint lacks training_metadata")
    forbidden_flags = (
        "spair_keypoints_used",
        "spair_bounding_boxes_used",
        "segmentation_masks_used",
        "pose_labels_used",
        "category_labels_used_for_targets",
        "caption_labels_used",
        "external_matcher_used",
        "dino_used",
        "roma_used",
    )
    allow_external_teacher = bool(
        args is not None and getattr(args, "allow_external_teacher_checkpoint", False)
    )
    enabled = [
        name
        for name in forbidden_flags
        if bool(metadata.get(name, True))
        and not (allow_external_teacher and name in {"external_matcher_used", "dino_used", "roma_used"})
    ]
    if enabled:
        raise ValueError(f"checkpoint violates label-free verifier protocol: {enabled}")
    if args is None:
        return
    protocol = metadata.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("verifier checkpoint lacks training protocol metadata")
    expected = {
        "feature_block": int(args.k[0]),
        "timestep": int(args.t),
        "ensemble_size": int(args.ensemble_size),
        "candidate_topk": int(args.candidate_topk),
    }
    mismatches = {
        name: {"checkpoint": protocol.get(name), "evaluation": value}
        for name, value in expected.items()
        if int(protocol.get(name, -1)) != int(value)
    }
    if [int(value) for value in protocol.get("image_size", [])] != [
        int(value) for value in args.img_size
    ]:
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
    if (
        bool(metadata.get("gt_used_for_features", True))
        or bool(metadata.get("gt_used_for_labels_only", True))
        or bool(metadata.get("labels_present", True))
    ):
        raise RuntimeError("evaluation candidate batch unexpectedly contains labels")
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
    from spair_matchers import flux_fjsar_candidate_feature_batch

    if tuple(map(int, args.k)) != (28,) or int(args.t) != 260 or int(args.ensemble_size) != 8:
        raise ValueError("locked verifier protocol requires --k 28 --t 260 --ensemble_size 8")
    use_disk_cache = bool(args.fjsar_disk_cache_path)
    if not use_disk_cache and not bool(args.extract_native_in_memory):
        raise ValueError(
            "use either --fjsar_disk_cache_path or --extract_native_in_memory"
        )
    if bool(args.fjsar_require_disk_cache) and not use_disk_cache:
        raise ValueError("--fjsar_require_disk_cache needs --fjsar_disk_cache_path")
    if not bool(args.fjsar_shared_noise):
        raise ValueError("locked verifier protocol requires --fjsar_shared_noise")

    model, checkpoint = load_verifier_checkpoint(args.verifier_checkpoint, map_location="cpu")
    _validate_checkpoint_metadata(checkpoint, args)
    checkpoint_protocol = checkpoint["training_metadata"]["protocol"]
    cycle_radius_cells = float(checkpoint_protocol.get("cycle_radius_cells", 1.0))
    minimum_native_margin = float(checkpoint_protocol.get("minimum_native_margin", 0.01))
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
            required_images = sorted(
                {
                    data[field]
                    for _pair_name, data in category_pairs
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
                scores = _score_batch(model, batch, device)
                candidate_pixels = batch["candidate_pixels"].long()
                verifier_pixels = select_candidate_pixels(scores, candidate_pixels)
                attention_pixels = candidate_pixels[:, 0]
                _cycle_targets, cycle_confident, cycle_teacher = native_cycle_pseudo_targets(
                    batch,
                    source_native,
                    target_native,
                    cycle_radius_cells=cycle_radius_cells,
                    minimum_native_margin=minimum_native_margin,
                )
                teacher_ranks = cycle_teacher["teacher_rank"].long()
                teacher_pixels = candidate_pixels.gather(
                    1, teacher_ranks[:, None]
                ).squeeze(1)
                verifier_ranks = scores.argmax(dim=1)
                target_width = int(target_size[1])
                threshold = max(
                    data["trg_bndbox"][3] - data["trg_bndbox"][1],
                    data["trg_bndbox"][2] - data["trg_bndbox"][0],
                )
                points = []
                for index, target_point in enumerate(data["trg_kps"]):
                    attention_prediction = _pixel_to_xy(
                        int(attention_pixels[index]), target_width
                    )
                    verifier_prediction = _pixel_to_xy(
                        int(verifier_pixels[index]), target_width
                    )
                    teacher_prediction = _pixel_to_xy(
                        int(teacher_pixels[index]), target_width
                    )
                    candidate_predictions = [
                        _pixel_to_xy(int(pixel), target_width)
                        for pixel in candidate_pixels[index]
                    ]
                    predictions = {
                        "baseline": baseline[index],
                        "attention_top1": attention_prediction,
                        "verifier": verifier_prediction,
                    }
                    hits = {
                        name: _pck_hit(prediction, target_point, threshold)
                        for name, prediction in predictions.items()
                    }
                    hits["attention_top20_oracle"] = any(
                        _pck_hit(prediction, target_point, threshold)
                        for prediction in candidate_predictions
                    )
                    points.append({
                        "index": int(index),
                        "source_point": list(data["src_kps"][index]),
                        "target_point": list(target_point),
                        "predictions": predictions,
                        "pck_hits": hits,
                        "attention_selected_rank": 0,
                        "verifier_selected_rank": int(verifier_ranks[index]),
                        "native_cycle": {
                            "confident": bool(cycle_confident[index]),
                            "teacher_rank": int(teacher_ranks[index]),
                            "verifier_rank": int(verifier_ranks[index]),
                            "teacher_prediction": teacher_prediction,
                            "teacher_pck_hit": _pck_hit(
                                teacher_prediction, target_point, threshold
                            ),
                            "cycle_distance_cells": float(
                                cycle_teacher["cycle_distance_cells"][index]
                            ),
                            "native_margin": float(cycle_teacher["native_margin"][index]),
                            "attention_teacher_agreement": bool(
                                cycle_teacher["attention_teacher_agreement"][index]
                            ),
                        },
                        "gt_used_for_inference": False,
                    })
                record = {
                    "category": category,
                    "pair_name": pair_name,
                    "source_image": data["src_imname"],
                    "target_image": data["trg_imname"],
                    "pck_threshold": float(threshold),
                    "points": points,
                }
                category_records.append(record)
                all_records.append(record)
                del source_native, target_native, batch, scores
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            category_results[category] = summarize_verifier_records(category_records)
            category_results[category]["native_cycle_diagnostics"] = (
                summarize_native_cycle_diagnostics(category_records)
            )
            category_metrics = category_results[category]["metrics"]
            print(
                f"{category}: baseline/attention/verifier/oracle point="
                f"{category_metrics['baseline']['point']:.2f}/"
                f"{category_metrics['attention_top1']['point']:.2f}/"
                f"{category_metrics['verifier']['point']:.2f}/"
                f"{category_metrics['attention_top20_oracle']['point']:.2f}"
            )
            del entries
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        capture.close()

    summary = summarize_verifier_records(all_records)
    summary["native_cycle_diagnostics"] = summarize_native_cycle_diagnostics(all_records)
    result = {
        "matcher": "self_supervised_flux_attention_identity_verifier",
        "protocol": {
            "subset": str(args.subset),
            "pairs_per_category": int(args.pairs_per_cat),
            "split_seed": int(args.split_seed),
            "feature_block": int(args.k[0]),
            "timestep": int(args.t),
            "ensemble_size": int(args.ensemble_size),
            "channel_discard": bool(args.cd),
            "candidate_topk": int(args.candidate_topk),
            "shared_noise": bool(args.fjsar_shared_noise),
            "persistent_feature_cache_written": False,
            "replay_source": "canonical_disk_cache" if use_disk_cache else "fresh_in_memory",
            "disk_cache_path": str(args.fjsar_disk_cache_path) if use_disk_cache else "",
            "require_disk_cache": bool(args.fjsar_require_disk_cache),
            "gt_used_for_inference": False,
            "native_cycle_diagnostic": {
                "cycle_radius_cells": cycle_radius_cells,
                "minimum_native_margin": minimum_native_margin,
                "ground_truth_used_for_routing": False,
            },
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
                "protocol": result["protocol"],
                "summary": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    metrics = summary["metrics"]
    comparison = summary["verifier_vs_attention"]
    print("Matcher: self_supervised_flux_attention_identity_verifier")
    print(
        "Baseline/attention/verifier/oracle All point: "
        f"{metrics['baseline']['point']:.2f} / {metrics['attention_top1']['point']:.2f} / "
        f"{metrics['verifier']['point']:.2f} / {metrics['attention_top20_oracle']['point']:.2f}"
    )
    print(
        f"Verifier vs attention: rescued={comparison['rescued']}, "
        f"harmed={comparison['harmed']}, "
        f"retention={100.0 * comparison['attention_correct_retention_rate']:.2f}, "
        f"oracle gap recovered={100.0 * comparison['oracle_gap_recovered_fraction']:.2f}"
    )
    cycle_summary = summary["native_cycle_diagnostics"]
    print(
        "Native-cycle diagnostic: "
        f"coverage={100.0 * cycle_summary['coverage']:.2f}, "
        f"confident baseline/attention/verifier/teacher="
        f"{cycle_summary['confident_subset_pck']['baseline']:.2f}/"
        f"{cycle_summary['confident_subset_pck']['attention_top1']:.2f}/"
        f"{cycle_summary['confident_subset_pck']['verifier']:.2f}/"
        f"{cycle_summary['confident_subset_pck']['teacher']:.2f}, "
        f"baseline-gated verifier/teacher="
        f"{cycle_summary['baseline_preserving_gates']['verifier_point']:.2f}/"
        f"{cycle_summary['baseline_preserving_gates']['teacher_point']:.2f}"
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
    parser.add_argument(
        "--allow_external_teacher_checkpoint",
        action="store_true",
        default=False,
        help="Diagnostic-only: permit a checkpoint trained with an external RoMa/DINO teacher.",
    )
    parser.add_argument("--fjsar_shared_noise", action="store_true", default=True)
    parser.add_argument("--fjsar_disk_cache_path", default="")
    parser.add_argument("--fjsar_require_disk_cache", action="store_true", default=False)
    parser.set_defaults(
        matcher="attention_identity_verifier_evaluation",
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

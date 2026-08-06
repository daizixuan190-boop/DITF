"""Audit geometric-orbit mutual FLUX attention on SPair-71k.

The evaluator compares the existing spectral controls against an equal-mean
consensus over original and horizontal-flip source/target interactions. It
extracts replay states into CPU RAM one category at a time, writes no feature
cache, uses no external matcher, and keeps the official global cosine nearest-
neighbour prediction unchanged.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from flux_native_in_memory import nested_tensor_nbytes
from spair_matchers import (
    cosine_nn_predict,
    flux_fjsar_filtered_spectral_feature_map_variants,
)


BRANCHES = (
    "baseline",
    "early_average",
    "head_coherent",
    "expert_coherent",
    "expert_coherence_gated",
    "head_preserving",
    "flip_orbit",
)


METHOD_HYPOTHESIS = {
    "name": "Geometric-Orbit Attention Consensus",
    "mechanism_hypothesis": (
        "A correspondence supported by frozen FLUX interaction in several "
        "reversible coordinate systems is more likely to preserve spatial "
        "identity than a view-specific attention mode. Four reciprocal mutual "
        "kernels from original and horizontal-flip source/target views are "
        "inverse-mapped to the original grids and averaged before the locked "
        "local-support paired spectral construction."
    ),
    "controls": {
        "early_average": "mean_members_and_heads_then_mutual",
        "head_coherent": "mean_members_per_head_then_mutual_then_mean_heads",
        "expert_coherent": "mutual_per_member_head_then_mean_experts",
        "expert_coherence_gated": (
            "early_spectral_geometry_with_parameter_free_tokenwise_expert_"
            "coherence_injection"
        ),
        "head_preserving": (
            "factorize_each_head_mutual_kernel_before_equal_energy_coordinate_"
            "concatenation"
        ),
        "flip_orbit": (
            "equal_mean_of_four_inverse_aligned_original_hflip_mutual_kernels"
        ),
    },
    "native_branch": "official_DiTF_FLUX_block28_descriptor",
    "spectral_branch": "local_support_radius2_SVD_rank64_weight0.5",
    "attention_replay": "exact_cross_image_without_coordinate_bias",
    "prediction": "official_global_cosine_NN",
    "external_correspondence_model": False,
    "task_training": False,
    "category_routing": False,
    "gt_used_for_inference": False,
}


def _load_category_pairs(
    args: argparse.Namespace,
    test_path: str,
    category: str,
    pair_names: Sequence[str],
) -> list[tuple[str, dict[str, Any]]]:
    from eval_spair_matcher_ablation import _select_category_pairs

    selected = _select_category_pairs(pair_names, args)
    if int(args.max_pairs_per_cat) > 0:
        selected = selected[: int(args.max_pairs_per_cat)]
    rows = []
    for pair_name in selected:
        path = os.path.join(args.dataset_path, test_path, pair_name)
        with open(path, encoding="utf-8") as handle:
            rows.append((pair_name, json.load(handle)))
    return rows


def _predict_feature_pair(
    source_feature: torch.Tensor,
    target_feature: torch.Tensor,
    source_size: Sequence[int],
    target_size: Sequence[int],
    source_points: Sequence[Sequence[float]],
) -> list[list[int]]:
    source_full = F.interpolate(
        source_feature.to(torch.float16),
        size=tuple(map(int, source_size)),
        mode="bilinear",
    )
    target_full = F.interpolate(
        target_feature.to(torch.float16),
        size=tuple(map(int, target_size)),
        mode="bilinear",
    )
    predictions = cosine_nn_predict(source_full, target_full, source_points)
    del source_full, target_full
    return predictions


def _branch_metrics(
    records: Sequence[dict[str, Any]],
    branch: str,
) -> dict[str, Any]:
    if branch not in BRANCHES:
        raise ValueError(f"unknown branch: {branch}")
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


def _comparison_counts(
    records: Sequence[dict[str, Any]],
    candidate: str,
    reference: str,
) -> dict[str, Any]:
    points = [point for pair in records for point in pair["points"]]
    rescued = sum(
        bool(point["pck_hits"][candidate])
        and not bool(point["pck_hits"][reference])
        for point in points
    )
    harmed = sum(
        bool(point["pck_hits"][reference])
        and not bool(point["pck_hits"][candidate])
        for point in points
    )
    changed = sum(
        point["predictions"][candidate] != point["predictions"][reference]
        for point in points
    )
    return {
        "reference": reference,
        "candidate": candidate,
        "rescued": int(rescued),
        "harmed": int(harmed),
        "net_correct": int(rescued - harmed),
        "changed_predictions": int(changed),
    }


def _summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    metrics = {branch: _branch_metrics(records, branch) for branch in BRANCHES}
    baseline_point = metrics["baseline"]["point"]
    for branch in BRANCHES[1:]:
        metrics[branch]["point_gain_vs_baseline"] = (
            metrics[branch]["point"] - baseline_point
        )
    comparisons = {}
    for branch in BRANCHES[1:]:
        comparisons[f"{branch}_vs_baseline"] = _comparison_counts(
            records,
            branch,
            "baseline",
        )
    for branch in (
        "head_coherent",
        "expert_coherent",
        "expert_coherence_gated",
        "head_preserving",
        "flip_orbit",
    ):
        comparisons[f"{branch}_vs_early_average"] = _comparison_counts(
            records,
            branch,
            "early_average",
        )
    return {"metrics": metrics, "comparisons": comparisons}


def evaluate(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    from eval_spair_matcher_ablation import (
        _extract_flux_fjsar_entry,
        _load_flux_fjsar_runtime,
        _load_pairs,
        _make_fjsar_capture,
        _pck,
        _prepare_feature_tensors,
    )

    if not bool(args.extract_native_in_memory):
        raise ValueError(
            "this evaluator requires --extract_native_in_memory and never reads "
            "or writes a persistent feature cache"
        )
    if tuple(map(int, args.k)) != (28,):
        raise ValueError("the locked experiment requires --k 28")
    if int(args.t) != 260 or int(args.ensemble_size) != 8:
        raise ValueError("the locked experiment requires --t 260 --ensemble_size 8")
    if not bool(args.fjsar_shared_noise):
        raise ValueError(
            "flip-orbit consensus requires --fjsar_shared_noise so all views "
            "use the same frozen noise ensemble"
        )

    args.device = str(device)
    test_path, categories, cat2json, _cat2img = _load_pairs(args.dataset_path)
    with open("spair_detailed_captions.json", encoding="utf-8") as handle:
        captions = json.load(handle)
    pre_norm = nn.LayerNorm(3072, elementwise_affine=False, eps=1e-6).to(device)
    featurizer, flux_model, blocks = _load_flux_fjsar_runtime(args, categories)
    capture = _make_fjsar_capture(args, flux_model)
    pair_records: list[dict[str, Any]] = []
    category_results: dict[str, Any] = OrderedDict()
    extracted_images = 0
    extracted_bytes = 0

    try:
        for category in categories:
            category_pairs = _load_category_pairs(
                args,
                test_path,
                category,
                cat2json[category],
            )
            required_images = sorted(
                {
                    data[field]
                    for _pair_name, data in category_pairs
                    for field in ("src_imname", "trg_imname")
                }
            )
            category_entries: dict[str, dict[str, dict[str, Any]]] = {}
            for image_name in tqdm(
                required_images,
                desc=f"extract replay {category}",
            ):
                original_entry = _extract_flux_fjsar_entry(
                    args.dataset_path,
                    category,
                    image_name,
                    captions[category + image_name],
                    args,
                    featurizer,
                    capture,
                )
                hflip_entry = _extract_flux_fjsar_entry(
                    args.dataset_path,
                    category,
                    image_name,
                    captions[category + image_name],
                    args,
                    featurizer,
                    capture,
                    horizontal_flip=True,
                )
                category_entries[image_name] = {
                    "original": original_entry,
                    "hflip": {"replay_state": hflip_entry["replay_state"]},
                }
                extracted_images += 2
                extracted_bytes += nested_tensor_nbytes(original_entry)
                extracted_bytes += nested_tensor_nbytes(
                    category_entries[image_name]["hflip"]
                )
                del hflip_entry

            category_records = []
            for pair_name, data in tqdm(
                category_pairs,
                desc=f"evaluate {category}",
            ):
                source_views = category_entries[data["src_imname"]]
                target_views = category_entries[data["trg_imname"]]
                source_entry = source_views["original"]
                target_entry = target_views["original"]
                source_native = _prepare_feature_tensors(
                    source_entry["feature"],
                    source_entry["ada"],
                    args,
                    pre_norm,
                    device,
                )
                target_native = _prepare_feature_tensors(
                    target_entry["feature"],
                    target_entry["ada"],
                    args,
                    pre_norm,
                    device,
                )
                source_size = data["src_imsize"][:2][::-1]
                target_size = data["trg_imsize"][:2][::-1]
                predictions = {
                    "baseline": _predict_feature_pair(
                        source_native,
                        target_native,
                        source_size,
                        target_size,
                        data["src_kps"],
                    )
                }
                variants = flux_fjsar_filtered_spectral_feature_map_variants(
                    source_native,
                    target_native,
                    src_replay_state=source_entry["replay_state"],
                    trg_replay_state=target_entry["replay_state"],
                    src_hflip_replay_state=source_views["hflip"]["replay_state"],
                    trg_hflip_replay_state=target_views["hflip"]["replay_state"],
                    blocks=blocks,
                    rank=64,
                    radius=2,
                    weight=0.5,
                    include_native=True,
                )
                pair_diagnostics = {}
                for branch in BRANCHES[1:]:
                    source_map, target_map, diagnostics = variants[branch]
                    if bool(diagnostics.get("gt_used", True)):
                        raise RuntimeError(
                            f"{branch} spectral extraction unexpectedly used GT"
                        )
                    predictions[branch] = _predict_feature_pair(
                        source_map,
                        target_map,
                        source_size,
                        target_size,
                        data["src_kps"],
                    )
                    pair_diagnostics[branch] = diagnostics

                threshold = max(
                    data["trg_bndbox"][3] - data["trg_bndbox"][1],
                    data["trg_bndbox"][2] - data["trg_bndbox"][0],
                )
                points = []
                for index, target_point in enumerate(data["trg_kps"]):
                    point_predictions = {
                        branch: predictions[branch][index] for branch in BRANCHES
                    }
                    point_hits = {
                        branch: bool(
                            _pck(
                                point_predictions[branch],
                                target_point,
                                threshold,
                            )
                        )
                        for branch in BRANCHES
                    }
                    points.append(
                        {
                            "index": int(index),
                            "source_point": list(data["src_kps"][index]),
                            "target_point": list(target_point),
                            "predictions": point_predictions,
                            "pck_hits": point_hits,
                            "gt_used_for_inference": False,
                        }
                    )
                pair_record = {
                    "category": category,
                    "pair_name": pair_name,
                    "source_image": data["src_imname"],
                    "target_image": data["trg_imname"],
                    "pck_threshold": float(threshold),
                    "diagnostics": pair_diagnostics,
                    "points": points,
                }
                pair_records.append(pair_record)
                category_records.append(pair_record)
                del source_native, target_native, variants
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            category_results[category] = _summary(category_records)
            category_points = category_results[category]["metrics"]
            print(
                f"{category}: baseline/early/head/expert/gated/preserved/orbit point="
                f"{category_points['baseline']['point']:.2f}/"
                f"{category_points['early_average']['point']:.2f}/"
                f"{category_points['head_coherent']['point']:.2f}/"
                f"{category_points['expert_coherent']['point']:.2f}/"
                f"{category_points['expert_coherence_gated']['point']:.2f}"
                f"/{category_points['head_preserving']['point']:.2f}"
                f"/{category_points['flip_orbit']['point']:.2f}"
            )
            del category_entries
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        capture.close()
        del capture, blocks, flux_model, featurizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = _summary(pair_records)
    result = {
        "matcher": "attention_flip_orbit_spectral",
        "method_hypothesis": METHOD_HYPOTHESIS,
        "protocol": {
            "subset": str(args.subset),
            "pairs_per_category": int(args.pairs_per_cat),
            "split_seed": int(args.split_seed),
            "native_flux_block": int(args.k[0]),
            "native_timestep": int(args.t),
            "native_ensemble_size": int(args.ensemble_size),
            "channel_discard": bool(args.cd),
            "shared_noise": bool(args.fjsar_shared_noise),
            "spectral_local_support_radius": 2,
            "spectral_rank": 64,
            "spectral_weight": 0.5,
            "coherence_gate": "2r/(r+spatial_mean_r)",
            "coherence_ratio": "expert_coherent_mass/early_average_mutual_mass",
            "head_preserving_rank_budget": 64,
            "head_preserving_rank_allocation": "ceil(64/head_count)_per_head",
            "flip_orbit_views": 4,
            "flip_orbit_transform": "horizontal_flip",
            "flip_orbit_fusion": "inverse_align_then_equal_mean_mutual_kernel",
            "interaction_mode": "exact",
            "coordinate_bias": False,
            "final_matcher": "official global cosine NN",
            "native_feature_source": "runtime_category_scoped_CPU_RAM_only",
            "persistent_feature_cache_read": False,
            "persistent_feature_cache_written": False,
            "runtime_extracted_images": int(extracted_images),
            "runtime_transient_entry_bytes_total": int(extracted_bytes),
            "gt_used_for_inference": False,
        },
        "categories": category_results,
        "all": summary,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    root = output_path.with_suffix("")
    audit_path = Path(f"{root}_attention_flip_orbit_spectral_audit.json")
    summary_path = Path(f"{root}_attention_flip_orbit_spectral_summary.json")
    result["audit_path"] = str(audit_path)
    result["summary_path"] = str(summary_path)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    audit_path.write_text(
        json.dumps(
            {
                "matcher": result["matcher"],
                "method_hypothesis": METHOD_HYPOTHESIS,
                "protocol": result["protocol"],
                "summary": summary,
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
                "metrics": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    metrics = summary["metrics"]
    print("Matcher: attention_flip_orbit_spectral")
    for branch in BRANCHES:
        row = metrics[branch]
        suffix = (
            ""
            if branch == "baseline"
            else f"; gain={row['point_gain_vs_baseline']:.2f}"
        )
        print(
            f"{branch} All per image/point: "
            f"{row['image']:.2f} / {row['point']:.2f}{suffix}"
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
        "--subset",
        choices=("all", "discovery", "heldout"),
        default="discovery",
    )
    parser.add_argument("--pairs_per_cat", type=int, default=20)
    parser.add_argument("--split_seed", type=int, default=2027)
    parser.add_argument("--max_pairs_per_cat", type=int, default=0)
    parser.add_argument(
        "--extract_native_in_memory",
        action="store_true",
        default=False,
        help=(
            "Extract current-category FLUX replay states into CPU RAM and never "
            "read or write a feature cache."
        ),
    )
    parser.add_argument("--fjsar_shared_noise", action="store_true", default=False)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.set_defaults(
        matcher="attention_flip_orbit_spectral",
        fjsar_disk_cache_path="",
        fjsar_multilayer_identity_audit=False,
        fjsar_multilayer_blocks=(),
        fjsar_trajectory_blocks=(),
    )
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    run_device = torch.device(
        parsed.device
        if parsed.device == "cpu" or torch.cuda.is_available()
        else "cpu"
    )
    evaluate(parsed, run_device)

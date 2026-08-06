"""Audit label-free certified anchors and local transport on SPair-71k.

This is a mechanism audit, not a PCK-tuned matcher.  It creates anchors only
from frozen DiTF forward/backward NN consistency and frozen attention-top1
agreement.  Ground truth is consulted strictly after prediction to report
anchor precision, candidate-oracle coverage, rescued/harmed points, and the
ceiling available to local pair-conditioned transport.
"""

from __future__ import annotations

import argparse
import gc
import json
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from anchor_transport import (
    baseline_preserving_transport_ranks,
    certified_anchor_mask,
    local_affine_transport,
)
from eval_spair_attention_identity_verifier import (
    _load_category_pairs,
    _pck_hit,
    _pixel_to_xy,
    _predict_baseline,
)


def _cell_diagonal(image_size: Sequence[int], grid_size: Sequence[int]) -> float:
    image_h, image_w = map(int, image_size)
    grid_h, grid_w = map(int, grid_size)
    if min(image_h, image_w, grid_h, grid_w) <= 0:
        raise ValueError("image and feature grid sizes must be positive")
    return float(((image_h / grid_h) ** 2 + (image_w / grid_w) ** 2) ** 0.5)


def _point_tensor(points: Sequence[Sequence[float]]) -> torch.Tensor:
    return torch.tensor([[float(value[0]), float(value[1])] for value in points])


def _candidate_xy(candidate_pixels: torch.Tensor, target_width: int) -> torch.Tensor:
    pixels = candidate_pixels.long()
    return torch.stack((
        (pixels % int(target_width)).float(),
        torch.div(pixels, int(target_width), rounding_mode="floor").float(),
    ), dim=2)


def _pck_candidate_hits(
    candidates: torch.Tensor,
    target_points: Sequence[Sequence[float]],
    threshold: float,
) -> torch.Tensor:
    target = _point_tensor(target_points).to(candidates.device)
    distance = (candidates - target[:, None]).square().sum(dim=2).sqrt()
    return distance <= 0.1 * float(threshold)


def _config_key(cycle: float, agreement: float, transport: float) -> str:
    return (
        f"cycle={float(cycle):g}|agreement={float(agreement):g}|"
        f"transport={float(transport):g}"
    )


def _new_counts() -> dict[str, int]:
    return defaultdict(int)


def _update_counts(
    counts: dict[str, int],
    *,
    baseline_hits: torch.Tensor,
    candidate_hits: torch.Tensor,
    anchors: torch.Tensor,
    transport_ranks: torch.Tensor,
    transport_valid: torch.Tensor,
    transport_support: torch.Tensor,
    selected_ranks: torch.Tensor,
    switched: torch.Tensor,
) -> None:
    point_count = int(baseline_hits.numel())
    selected_hits = baseline_hits.clone()
    if bool(switched.any()):
        selected_hits[switched] = candidate_hits[
            torch.arange(point_count)[switched], selected_ranks[switched]
        ]
    attention_oracle = candidate_hits.any(dim=1)
    routeable = (~baseline_hits) & attention_oracle
    anchor_hits = baseline_hits[anchors]
    valid_transport_hits = candidate_hits[
        torch.arange(point_count)[transport_valid],
        transport_ranks[transport_valid],
    ] if bool(transport_valid.any()) else torch.empty(0, dtype=torch.bool)
    counts["points"] += point_count
    counts["baseline_correct"] += int(baseline_hits.sum())
    counts["attention_oracle_correct"] += int(attention_oracle.sum())
    counts["anchors"] += int(anchors.sum())
    counts["anchor_baseline_correct"] += int(anchor_hits.sum())
    counts["transport_valid"] += int(transport_valid.sum())
    counts["transport_valid_candidate_correct"] += int(valid_transport_hits.sum())
    counts["transport_switched"] += int(switched.sum())
    counts["transport_correct"] += int(selected_hits.sum())
    counts["rescued"] += int(((~baseline_hits) & selected_hits).sum())
    counts["harmed"] += int((baseline_hits & (~selected_hits)).sum())
    counts["baseline_retained"] += int((baseline_hits & selected_hits).sum())
    counts["routeable"] += int(routeable.sum())
    counts["routeable_switched"] += int((routeable & switched).sum())
    counts["routeable_rescued"] += int((routeable & selected_hits).sum())


def _summarize_counts(counts: Mapping[str, int]) -> dict[str, Any]:
    points = max(1, int(counts["points"]))
    baseline_correct = int(counts["baseline_correct"])
    transport_correct = int(counts["transport_correct"])
    attention_oracle = int(counts["attention_oracle_correct"])
    gap = max(0, attention_oracle - baseline_correct)
    anchors = int(counts["anchors"])
    valid = int(counts["transport_valid"])
    routeable = int(counts["routeable"])
    return {
        "points": int(counts["points"]),
        "baseline_point": 100.0 * baseline_correct / points,
        "transport_point": 100.0 * transport_correct / points,
        "attention_top20_oracle_point": 100.0 * attention_oracle / points,
        "anchor": {
            "count": anchors,
            "coverage": anchors / points,
            "baseline_precision_posthoc": (
                int(counts["anchor_baseline_correct"]) / anchors if anchors else 0.0
            ),
        },
        "transport": {
            "valid_count": valid,
            "valid_coverage": valid / points,
            "valid_candidate_precision_posthoc": (
                int(counts["transport_valid_candidate_correct"]) / valid if valid else 0.0
            ),
            "switched_count": int(counts["transport_switched"]),
            "switch_rate": int(counts["transport_switched"]) / points,
        },
        "vs_baseline": {
            "rescued": int(counts["rescued"]),
            "harmed": int(counts["harmed"]),
            "net_correct": transport_correct - baseline_correct,
            "baseline_correct_retention_rate": (
                int(counts["baseline_retained"]) / baseline_correct
                if baseline_correct else 0.0
            ),
            "oracle_gap_points": gap,
            "oracle_gap_recovered_fraction": (
                (transport_correct - baseline_correct) / gap if gap else 0.0
            ),
        },
        "routeable_baseline_errors": {
            "count": routeable,
            "switch_rate": int(counts["routeable_switched"]) / routeable if routeable else 0.0,
            "rescued_fraction": int(counts["routeable_rescued"]) / routeable if routeable else 0.0,
        },
    }


def _run_transport_config(
    source_points: torch.Tensor,
    baseline_predictions: torch.Tensor,
    reverse_predictions: torch.Tensor,
    attention_predictions: torch.Tensor,
    candidates: torch.Tensor,
    *,
    source_cell_diagonal: float,
    target_cell_diagonal: float,
    cycle_radius_cells: float,
    agreement_radius_cells: float,
    transport_radius_cells: float,
    neighbor_count: int,
    minimum_anchors: int,
) -> dict[str, torch.Tensor]:
    anchors, cycle_error, agreement_error = certified_anchor_mask(
        source_points,
        baseline_predictions,
        reverse_predictions,
        attention_predictions,
        source_cell_diagonal=source_cell_diagonal,
        target_cell_diagonal=target_cell_diagonal,
        cycle_radius_cells=cycle_radius_cells,
        agreement_radius_cells=agreement_radius_cells,
    )
    ranks, valid, support = local_affine_transport(
        source_points,
        baseline_predictions,
        candidates,
        anchors,
        neighbor_count=neighbor_count,
        minimum_anchors=minimum_anchors,
        target_cell_diagonal=target_cell_diagonal,
    )
    selected, switched = baseline_preserving_transport_ranks(
        ranks,
        valid,
        support,
        transport_radius_cells=transport_radius_cells,
    )
    return {
        "anchors": anchors,
        "cycle_error_cells": cycle_error,
        "agreement_error_cells": agreement_error,
        "transport_ranks": ranks,
        "transport_valid": valid,
        "transport_support_cells": support,
        "selected_ranks": selected,
        "switched": switched,
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
        raise ValueError("locked anchor audit requires --k 28 --t 260 --ensemble_size 8")
    if int(args.candidate_topk) != 20:
        raise ValueError("anchor audit is locked to attention top-20")
    if not bool(args.fjsar_shared_noise):
        raise ValueError("locked anchor audit requires --fjsar_shared_noise")
    use_disk_cache = bool(args.fjsar_disk_cache_path)
    if not use_disk_cache and not bool(args.extract_native_in_memory):
        raise ValueError("use either --fjsar_disk_cache_path or --extract_native_in_memory")
    if bool(args.fjsar_require_disk_cache) and not use_disk_cache:
        raise ValueError("--fjsar_require_disk_cache needs --fjsar_disk_cache_path")

    cycle_values = sorted({float(value) for value in args.cycle_radius_cells} | {float(args.primary_cycle_radius_cells)})
    agreement_values = sorted({float(value) for value in args.agreement_radius_cells} | {float(args.primary_agreement_radius_cells)})
    transport_values = sorted({float(value) for value in args.transport_radius_cells} | {float(args.primary_transport_radius_cells)})
    if min(*cycle_values, *agreement_values, *transport_values) <= 0.0:
        raise ValueError("anchor audit radii must be positive")
    configs = [
        (cycle, agreement, transport)
        for cycle in cycle_values
        for agreement in agreement_values
        for transport in transport_values
    ]
    primary = (
        float(args.primary_cycle_radius_cells),
        float(args.primary_agreement_radius_cells),
        float(args.primary_transport_radius_cells),
    )
    primary_key = _config_key(*primary)
    config_counts = {_config_key(*config): _new_counts() for config in configs}
    category_counts: dict[str, dict[str, dict[str, int]]] = {}
    oracle_anchor_primary_counts = _new_counts()

    args.device = str(device)
    test_path, categories, cat2json, _cat2img = _load_pairs(args.dataset_path)
    captions = json.loads(Path(args.captions_json).read_text(encoding="utf-8"))
    featurizer, flux_model, blocks = _load_flux_fjsar_runtime(args, categories)
    capture = _make_fjsar_capture(args, flux_model)
    pre_norm = nn.LayerNorm(3072, elementwise_affine=False, eps=1e-6).to(device)
    all_records: list[dict[str, Any]] = []

    try:
        for category in categories:
            category_counts[category] = {
                _config_key(*config): _new_counts() for config in configs
            }
            category_records = []
            category_pairs = _load_category_pairs(args, test_path, cat2json[category])
            required_images = sorted({
                row[field]
                for _name, row in category_pairs
                for field in ("src_imname", "trg_imname")
            })
            entries = {}
            for image_name in tqdm(required_images, desc=f"load replay {category}"):
                caption_key = category + image_name
                if caption_key not in captions:
                    raise KeyError(f"missing detailed caption for {category}/{image_name}")
                if use_disk_cache:
                    entries[image_name] = _get_flux_fjsar_entry(
                        args.dataset_path, category, image_name, captions[caption_key], args,
                        featurizer, capture, None,
                    )
                else:
                    entries[image_name] = _extract_flux_fjsar_entry(
                        args.dataset_path, category, image_name, captions[caption_key], args,
                        featurizer, capture,
                    )

            for pair_name, row in tqdm(category_pairs, desc=f"audit {category}"):
                source_entry = entries[row["src_imname"]]
                target_entry = entries[row["trg_imname"]]
                source_native = _prepare_feature_tensors(
                    source_entry["feature"], source_entry["ada"], args, pre_norm, device
                )
                target_native = _prepare_feature_tensors(
                    target_entry["feature"], target_entry["ada"], args, pre_norm, device
                )
                source_size = row["src_imsize"][:2][::-1]
                target_size = row["trg_imsize"][:2][::-1]
                baseline = _predict_baseline(
                    source_native, target_native, source_size, target_size, row["src_kps"]
                )
                reverse = _predict_baseline(
                    target_native, source_native, target_size, source_size, baseline
                )
                batch = flux_fjsar_candidate_feature_batch(
                    source_native,
                    target_native,
                    row["src_kps"],
                    source_size,
                    target_size,
                    src_replay_state=source_entry["replay_state"],
                    trg_replay_state=target_entry["replay_state"],
                    blocks=blocks,
                    interaction_mode="exact",
                    use_coordinate_bias=False,
                    candidate_topk=int(args.candidate_topk),
                )
                if bool(batch["metadata"].get("gt_used_for_features", True)):
                    raise RuntimeError("ground truth leaked into anchor-audit features")
                target_width = int(target_size[1])
                candidates = _candidate_xy(batch["candidate_pixels"], target_width)
                source_tensor = _point_tensor(row["src_kps"])
                baseline_tensor = _point_tensor(baseline)
                reverse_tensor = _point_tensor(reverse)
                source_diagonal = _cell_diagonal(
                    source_size, source_entry["feature"].shape[-2:]
                )
                target_diagonal = _cell_diagonal(
                    target_size, target_entry["feature"].shape[-2:]
                )
                threshold = max(
                    row["trg_bndbox"][3] - row["trg_bndbox"][1],
                    row["trg_bndbox"][2] - row["trg_bndbox"][0],
                )
                baseline_hits = torch.tensor([
                    _pck_hit(prediction, target, threshold)
                    for prediction, target in zip(baseline, row["trg_kps"])
                ], dtype=torch.bool)
                candidate_hits = _pck_candidate_hits(candidates, row["trg_kps"], threshold)
                primary_output: dict[str, torch.Tensor] | None = None
                for config in configs:
                    key = _config_key(*config)
                    output = _run_transport_config(
                        source_tensor,
                        baseline_tensor,
                        reverse_tensor,
                        candidates[:, 0],
                        candidates,
                        source_cell_diagonal=source_diagonal,
                        target_cell_diagonal=target_diagonal,
                        cycle_radius_cells=config[0],
                        agreement_radius_cells=config[1],
                        transport_radius_cells=config[2],
                        neighbor_count=int(args.neighbor_count),
                        minimum_anchors=int(args.minimum_anchors),
                    )
                    _update_counts(
                        config_counts[key],
                        baseline_hits=baseline_hits,
                        candidate_hits=candidate_hits,
                        anchors=output["anchors"],
                        transport_ranks=output["transport_ranks"],
                        transport_valid=output["transport_valid"],
                        transport_support=output["transport_support_cells"],
                        selected_ranks=output["selected_ranks"],
                        switched=output["switched"],
                    )
                    _update_counts(
                        category_counts[category][key],
                        baseline_hits=baseline_hits,
                        candidate_hits=candidate_hits,
                        anchors=output["anchors"],
                        transport_ranks=output["transport_ranks"],
                        transport_valid=output["transport_valid"],
                        transport_support=output["transport_support_cells"],
                        selected_ranks=output["selected_ranks"],
                        switched=output["switched"],
                    )
                    if key == primary_key:
                        primary_output = output
                assert primary_output is not None
                # Diagnostic ceiling only: it reveals how much local transport
                # could recover if an oracle told us which baseline predictions
                # are trustworthy.  Labels never influence primary inference.
                oracle_ranks, oracle_valid, oracle_support = local_affine_transport(
                    source_tensor,
                    baseline_tensor,
                    candidates,
                    baseline_hits,
                    neighbor_count=int(args.neighbor_count),
                    minimum_anchors=int(args.minimum_anchors),
                    target_cell_diagonal=target_diagonal,
                )
                oracle_selected, oracle_switched = baseline_preserving_transport_ranks(
                    oracle_ranks,
                    oracle_valid,
                    oracle_support,
                    transport_radius_cells=float(args.primary_transport_radius_cells),
                )
                _update_counts(
                    oracle_anchor_primary_counts,
                    baseline_hits=baseline_hits,
                    candidate_hits=candidate_hits,
                    anchors=baseline_hits,
                    transport_ranks=oracle_ranks,
                    transport_valid=oracle_valid,
                    transport_support=oracle_support,
                    selected_ranks=oracle_selected,
                    switched=oracle_switched,
                )
                points = []
                for index, target in enumerate(row["trg_kps"]):
                    selected_rank = int(primary_output["selected_ranks"][index])
                    switched = bool(primary_output["switched"][index])
                    transport_prediction = (
                        _pixel_to_xy(int(batch["candidate_pixels"][index, selected_rank]), target_width)
                        if switched else baseline[index]
                    )
                    points.append({
                        "index": int(index),
                        "source_point": list(row["src_kps"][index]),
                        "target_point": list(target),
                        "baseline_prediction": baseline[index],
                        "reverse_baseline_prediction": reverse[index],
                        "attention_candidates": [
                            _pixel_to_xy(int(pixel), target_width)
                            for pixel in batch["candidate_pixels"][index]
                        ],
                        "cycle_error_cells": float(primary_output["cycle_error_cells"][index]),
                        "attention_agreement_cells": float(primary_output["agreement_error_cells"][index]),
                        "primary_anchor": bool(primary_output["anchors"][index]),
                        "primary_transport_valid": bool(primary_output["transport_valid"][index]),
                        "primary_transport_support_cells": float(
                            primary_output["transport_support_cells"][index]
                        ),
                        "primary_selected_attention_rank": selected_rank,
                        "primary_switched_from_baseline": switched,
                        "primary_transport_prediction": transport_prediction,
                        "pck_hits": {
                            "baseline": bool(baseline_hits[index]),
                            "attention_top1": bool(candidate_hits[index, 0]),
                            "attention_top20_oracle": bool(candidate_hits[index].any()),
                            "primary_transport": _pck_hit(
                                transport_prediction, target, threshold
                            ),
                        },
                        "candidate_pck_hits": [
                            bool(value) for value in candidate_hits[index].tolist()
                        ],
                        "gt_used_for_anchor_or_transport_inference": False,
                    })
                record = {
                    "category": category,
                    "pair_name": pair_name,
                    "source_image": row["src_imname"],
                    "target_image": row["trg_imname"],
                    "source_cell_diagonal": source_diagonal,
                    "target_cell_diagonal": target_diagonal,
                    "pck_threshold": float(threshold),
                    "points": points,
                }
                category_records.append(record)
                all_records.append(record)
                del source_native, target_native, batch
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            primary_summary = _summarize_counts(category_counts[category][primary_key])
            print(
                f"{category}: anchor coverage/precision, transport/baseline point="
                f"{100.0 * primary_summary['anchor']['coverage']:.2f}/"
                f"{100.0 * primary_summary['anchor']['baseline_precision_posthoc']:.2f}, "
                f"{primary_summary['transport_point']:.2f}/"
                f"{primary_summary['baseline_point']:.2f}"
            )
            del entries, category_records
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        capture.close()

    config_summary = {
        key: _summarize_counts(counts) for key, counts in config_counts.items()
    }
    category_summary = OrderedDict((
        category,
        _summarize_counts(category_counts[category][primary_key]),
    ) for category in categories)
    result = {
        "audit": "label_free_certified_anchor_local_transport",
        "claim_scope": "mechanism_audit_not_pck_tuned_final_matcher",
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
            "replay_source": "canonical_disk_cache" if use_disk_cache else "fresh_in_memory",
            "anchor_evidence": [
                "frozen_ditf_forward_backward_nn_cycle",
                "frozen_ditf_baseline_attention_top1_agreement",
            ],
            "anchor_labels_used": False,
            "transport_labels_used": False,
            "ground_truth_used_for_metrics_and_posthoc_audit_only": True,
            "neighbor_count": int(args.neighbor_count),
            "minimum_anchors": int(args.minimum_anchors),
            "grid_cycle_radius_cells": cycle_values,
            "grid_agreement_radius_cells": agreement_values,
            "grid_transport_radius_cells": transport_values,
            "primary_config": {
                "cycle_radius_cells": primary[0],
                "agreement_radius_cells": primary[1],
                "transport_radius_cells": primary[2],
            },
        },
        "primary_summary": config_summary[primary_key],
        "oracle_baseline_anchor_primary_ceiling": _summarize_counts(
            oracle_anchor_primary_counts
        ),
        "grid_summary": config_summary,
        "primary_category_summary": category_summary,
        "pair_records": all_records,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    summary_path.write_text(json.dumps({
        "audit": result["audit"],
        "claim_scope": result["claim_scope"],
        "protocol": result["protocol"],
        "primary_summary": result["primary_summary"],
        "oracle_baseline_anchor_primary_ceiling": result[
            "oracle_baseline_anchor_primary_ceiling"
        ],
        "grid_summary": result["grid_summary"],
        "primary_category_summary": result["primary_category_summary"],
    }, indent=2), encoding="utf-8")
    primary_metrics = result["primary_summary"]
    comparison = primary_metrics["vs_baseline"]
    print("Audit: label-free certified anchor local transport")
    print(
        "Primary anchor coverage/precision, transport/baseline/oracle point: "
        f"{100.0 * primary_metrics['anchor']['coverage']:.2f} / "
        f"{100.0 * primary_metrics['anchor']['baseline_precision_posthoc']:.2f} / "
        f"{primary_metrics['transport_point']:.2f} / "
        f"{primary_metrics['baseline_point']:.2f} / "
        f"{primary_metrics['attention_top20_oracle_point']:.2f}"
    )
    print(
        f"Transport vs baseline: rescued={comparison['rescued']}, "
        f"harmed={comparison['harmed']}, net={comparison['net_correct']}, "
        f"retention={100.0 * comparison['baseline_correct_retention_rate']:.2f}, "
        f"oracle-gap recovered={100.0 * comparison['oracle_gap_recovered_fraction']:.2f}"
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
    parser.add_argument("--neighbor_count", type=int, default=4)
    parser.add_argument("--minimum_anchors", type=int, default=3)
    parser.add_argument("--cycle_radius_cells", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    parser.add_argument("--agreement_radius_cells", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    parser.add_argument("--transport_radius_cells", nargs="+", type=float, default=[0.5, 1.0, 2.0, 4.0])
    parser.add_argument("--primary_cycle_radius_cells", type=float, default=1.0)
    parser.add_argument("--primary_agreement_radius_cells", type=float, default=1.0)
    parser.add_argument("--primary_transport_radius_cells", type=float, default=2.0)
    parser.set_defaults(
        matcher="anchor_transport_audit",
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

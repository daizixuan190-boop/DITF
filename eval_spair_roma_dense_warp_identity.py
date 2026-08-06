"""Frozen FLUX appearance evidence and RoMa identity with global cosine NN.

The evaluator supports the original FLUX+DINOv2 control and a locked spectral
branch that factorizes filtered mutual FLUX cross-attention into paired source
and target descriptors. Frozen RoMa supplies a pair-conditioned bidirectional
warp converted into cycle/Jacobian-validated identity features. No attention
candidate, expert router, task training, validation threshold, or GT field
enters prediction.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from dino_v2_extractor import build_dino_extractor
from dino_v2_spair import DINOConfig
from eval_spair_attention_top20_roma_identity import _build_roma, _run_roma_pair
from eval_spair_matcher_ablation import (
    _FjsarMemoryCache,
    _extract_flux_fjsar_entry,
    _fjsar_disk_cache_file,
    _load_flux_fjsar_runtime,
    _load_fjsar_disk_entry,
    _load_pairs,
    _make_fjsar_capture,
    _pck,
    _prepare_feature_tensors,
    _select_category_pairs,
)
from exact_warp_coordinate_identity import build_exact_forward_coordinate_maps
from frozen_appearance_identity_fusion import (
    build_roma_augmented_appearance,
    build_weighted_roma_augmented_appearance,
    sample_dino_map_on_image_grid,
)
from flux_native_in_memory import (
    extract_flux_native_entry,
    nested_tensor_nbytes,
    strip_flux_replay_entry,
)
from flux_dense_warp_identity import (
    DenseWarpIdentityConfig,
    build_identity_maps_from_warp_fields,
    sample_dense_field,
)
from roma_dense_warp_identity import roma_warp_to_token_fields
from spair_matchers import (
    cosine_nn_predict,
    flux_fjsar_filtered_spectral_feature_maps,
)


METHOD_HYPOTHESIS = {
    "name": "Frozen Tri-Factor Appearance and Transport Identity",
    "mechanism_hypothesis": (
        "FLUX supplies diffusion semantics, standalone DINOv2 supplies local discriminative "
        "appearance, and frozen RoMa supplies pair-conditioned transport identity. Equal-energy "
        "feature-side composition lets the unchanged global cosine matcher use all three."
    ),
    "native_branch": "official_DiTF_FLUX_descriptor_cached_or_runtime_extracted",
    "appearance_branch": "equal_energy_unit_FLUX_and_DINOv2_block11_descriptors",
    "identity_branch": "mutual_shared_coordinate_RoMa_identity",
    "reliability": "scale_invariant_RoMa_certainty_times_cycle_and_local_Jacobian",
    "fusion": "fixed_equal_energy_appearance_then_variable_norm_identity_concatenation",
    "target_prior": "retained_variable_norm_target_reliability_prior",
    "prediction": "official_global_cosine_NN",
    "attention_candidate_used_for_prediction": False,
    "expert_router": False,
    "native_fallback": False,
    "task_training": False,
    "gt_used_for_inference": False,
    "external_pretraining": "DINOv2 ViT-B/14 and RoMa outdoor frozen weights",
}


SPECTRAL_METHOD_HYPOTHESIS = {
    "name": "Filtered Cross-Attention Spectral Transport Identity",
    "mechanism_hypothesis": (
        "Exact mutual FLUX cross-attention is a high-recall multimodal transport kernel. "
        "Local transport support suppresses inconsistent mass, and paired SVD factors "
        "preserve its modes as source/target identity descriptors instead of averaging them. "
        "The spectral descriptor is composed with native FLUX appearance and frozen RoMa "
        "continuous transport identity before unchanged global cosine matching."
    ),
    "native_branch": "official_DiTF_FLUX_descriptor_from_read_only_cache",
    "appearance_branch": "native_plus_filtered_mutual_attention_SVD_rank64_weight0.5",
    "identity_branch": "RoMa_mutual_dual_coordinate_identity",
    "attention_replay": "exact_cross_image_without_coordinate_bias",
    "prediction": "official_global_cosine_NN",
    "attention_candidate_used_for_prediction": False,
    "expert_router": False,
    "native_fallback": False,
    "task_training": False,
    "gt_used_for_inference": False,
    "external_pretraining": "RoMa outdoor frozen weights only",
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
    rows = []
    for pair_name in selected:
        with open(os.path.join(args.dataset_path, test_path, pair_name)) as handle:
            rows.append((pair_name, json.load(handle)))
    return rows


def _cache_key(
    args: argparse.Namespace,
    category: str,
    image_name: str,
    caption: str,
) -> tuple[Any, ...]:
    feature_block = int(args.k[0])
    return (
        category,
        image_name,
        int(args.img_size[0]),
        int(args.t),
        feature_block,
        int(args.ensemble_size),
        bool(args.cd),
        bool(args.fjsar_shared_noise),
        hashlib.sha256(caption.encode("utf-8")).hexdigest(),
    )


def _load_cached_entry(
    args: argparse.Namespace,
    category: str,
    image_name: str,
    caption: str,
    memory_cache: _FjsarMemoryCache,
) -> dict[str, Any]:
    key = _cache_key(args, category, image_name, caption)
    cached = memory_cache.get(key)
    if cached is not None:
        return cached
    path = _fjsar_disk_cache_file(
        args.fjsar_disk_cache_path, category, image_name, key
    )
    entry = _load_fjsar_disk_entry(path, int(args.k[0]), caption, args)
    if entry is None:
        raise FileNotFoundError(
            "RoMa dense-warp identity requires the existing native FLUX cache; "
            f"missing or stale entry for {category}/{image_name}: {path}"
        )
    memory_cache.put(key, entry)
    return entry


def _audit_index(path: str) -> dict[tuple[str, str, int], dict[str, Any]]:
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = payload.get("pair_records")
    if not isinstance(records, list):
        raise ValueError("attention audit must contain pair_records")
    indexed = {}
    for pair in records:
        pair_id = pair.get("pair_json")
        if not pair_id:
            raise ValueError("attention audit pair must contain pair_json")
        for point in pair.get("points", []):
            key = (str(pair["category"]), str(pair_id), int(point["keypoint_index"]))
            if key in indexed:
                raise ValueError(f"duplicate attention audit key {key}")
            indexed[key] = point
    return indexed


def _normalized_to_pixels(
    normalized: torch.Tensor,
    target_size: Sequence[int],
) -> list[list[int]]:
    height, width = int(target_size[0]), int(target_size[1])
    scale = normalized.new_tensor(
        [float(max(width - 1, 1)), float(max(height - 1, 1))]
    )
    pixels = torch.round(normalized.clamp(0.0, 1.0) * scale).long()
    pixels[:, 0].clamp_(0, width - 1)
    pixels[:, 1].clamp_(0, height - 1)
    return [[int(x), int(y)] for x, y in pixels.detach().cpu().tolist()]


def _sample_scalar(
    field: torch.Tensor,
    points: Sequence[Sequence[float]],
    image_size: Sequence[int],
) -> list[float]:
    values = sample_dense_field(field.unsqueeze(-1), points, image_size)[:, 0]
    return [float(value) for value in values.detach().float().cpu().tolist()]


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


def _metric_block(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    points = [point for pair in records for point in pair["points"]]
    baseline_pairs = [
        np.mean([point["baseline_pck_hit"] for point in pair["points"]])
        for pair in records
    ]
    method_pairs = [
        np.mean([point["method_pck_hit"] for point in pair["points"]])
        for pair in records
    ]
    baseline = int(sum(point["baseline_pck_hit"] for point in points))
    method = int(sum(point["method_pck_hit"] for point in points))
    return {
        "pairs": len(records),
        "points": len(points),
        "baseline_image": 100.0 * float(np.mean(baseline_pairs)) if records else 0.0,
        "method_image": 100.0 * float(np.mean(method_pairs)) if records else 0.0,
        "baseline_point": 100.0 * baseline / max(1, len(points)),
        "method_point": 100.0 * method / max(1, len(points)),
        "point_gain": 100.0 * (method - baseline) / max(1, len(points)),
        "improved_count": int(sum(point["rescued_vs_baseline"] for point in points)),
        "harmed_count": int(sum(point["harmed_vs_baseline"] for point in points)),
    }


def _mechanism_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    points = [point for pair in records for point in pair["points"]]
    baseline_wrong = [point for point in points if not point["baseline_pck_hit"]]
    hard = [point for point in points if point.get("both_wrong_top20_hit") is True]

    def branch(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        total = max(1, len(rows))
        return {
            "points": len(rows),
            "appearance_fusion_top1": float(
                sum(point["appearance_fusion_pck_hit"] for point in rows) / total
            ),
            "appearance_roma_variable_top1": float(
                sum(point["appearance_roma_variable_pck_hit"] for point in rows)
                / total
            ),
            "appearance_roma_forward_top1": float(
                sum(point["appearance_roma_forward_pck_hit"] for point in rows)
                / total
            ),
            "appearance_roma_dual_top1": float(
                sum(point["appearance_roma_dual_pck_hit"] for point in rows)
                / total
            ),
            "direct_warp_top1": float(
                sum(point["direct_warp_pck_hit"] for point in rows) / total
            ),
            "identity_only_top1": float(
                sum(point["identity_only_pck_hit"] for point in rows) / total
            ),
            "variable_norm_fusion_top1": float(
                sum(point["variable_norm_fusion_pck_hit"] for point in rows) / total
            ),
            "constant_norm_fusion_top1": float(
                sum(point["constant_norm_fusion_pck_hit"] for point in rows) / total
            ),
            "forward_only_fusion_top1": float(
                sum(point["forward_only_fusion_pck_hit"] for point in rows) / total
            ),
            "forward_identity_only_top1": float(
                sum(point["forward_identity_only_pck_hit"] for point in rows) / total
            ),
            "dual_fusion_top1": float(
                sum(point["dual_fusion_pck_hit"] for point in rows) / total
            ),
            "exact_forward_identity_top1": float(
                sum(point["exact_forward_identity_pck_hit"] for point in rows) / total
            ),
            "exact_forward_fusion_top1": float(
                sum(point["exact_forward_fusion_pck_hit"] for point in rows) / total
            ),
            "fused_top1": float(sum(point["method_pck_hit"] for point in rows) / total),
            "mean_raw_certainty": float(
                np.mean([point["warp_state"]["raw_certainty"] for point in rows])
            )
            if rows
            else None,
            "mean_support": float(
                np.mean([point["warp_state"]["support"] for point in rows])
            )
            if rows
            else None,
            "mean_cycle_error_cells": float(
                np.mean([point["warp_state"]["cycle_error_cells"] for point in rows])
            )
            if rows
            else None,
            "mean_jacobian_confidence": float(
                np.mean([point["warp_state"]["jacobian_confidence"] for point in rows])
            )
            if rows
            else None,
            "mean_reliability": float(
                np.mean([point["warp_state"]["reliability"] for point in rows])
            )
            if rows
            else None,
        }

    return {
        "pairs": len(records),
        "points": len(points),
        "baseline_correct": int(sum(point["baseline_pck_hit"] for point in points)),
        "method_correct": int(sum(point["method_pck_hit"] for point in points)),
        "appearance_fusion_correct": int(
            sum(point["appearance_fusion_pck_hit"] for point in points)
        ),
        "appearance_roma_variable_correct": int(
            sum(point["appearance_roma_variable_pck_hit"] for point in points)
        ),
        "appearance_roma_forward_correct": int(
            sum(point["appearance_roma_forward_pck_hit"] for point in points)
        ),
        "appearance_roma_dual_correct": int(
            sum(point["appearance_roma_dual_pck_hit"] for point in points)
        ),
        "variable_norm_fusion_correct": int(
            sum(point["variable_norm_fusion_pck_hit"] for point in points)
        ),
        "constant_norm_fusion_correct": int(
            sum(point["constant_norm_fusion_pck_hit"] for point in points)
        ),
        "forward_only_fusion_correct": int(
            sum(point["forward_only_fusion_pck_hit"] for point in points)
        ),
        "forward_identity_only_correct": int(
            sum(point["forward_identity_only_pck_hit"] for point in points)
        ),
        "dual_fusion_correct": int(
            sum(point["dual_fusion_pck_hit"] for point in points)
        ),
        "exact_forward_identity_correct": int(
            sum(point["exact_forward_identity_pck_hit"] for point in points)
        ),
        "exact_forward_fusion_correct": int(
            sum(point["exact_forward_fusion_pck_hit"] for point in points)
        ),
        "exact_forward_direct_pixel_agreement": int(
            sum(point["exact_forward_direct_pixel_agreement"] for point in points)
        ),
        "rescued_vs_baseline": int(sum(point["rescued_vs_baseline"] for point in points)),
        "harmed_vs_baseline": int(sum(point["harmed_vs_baseline"] for point in points)),
        "all_points": branch(points),
        "baseline_wrong": branch(baseline_wrong),
        "both_wrong_top20_hit": branch(hard),
    }


def evaluate(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    extract_in_memory = bool(args.extract_native_in_memory)
    if extract_in_memory == bool(args.fjsar_disk_cache_path):
        raise ValueError(
            "choose exactly one native feature source: "
            "--extract_native_in_memory or --fjsar_disk_cache_path"
        )
    appearance_source = str(args.appearance_source)
    # Fail before an expensive cache-free FLUX extraction when the new machine
    # is missing any transitive RoMa runtime dependency (for example kornia).
    try:
        from romatch import roma_outdoor as _roma_import_preflight  # noqa: F401
    except ImportError as exc:
        missing = getattr(exc, "name", None) or "an unknown RoMa dependency"
        raise RuntimeError(
            "RoMa import preflight failed before FLUX extraction: "
            f"missing {missing!r}. Repair the RoMa environment and rerun."
        ) from exc
    config = DenseWarpIdentityConfig()
    dino_config = DINOConfig() if appearance_source == "dinov2" else None
    method_hypothesis = (
        METHOD_HYPOTHESIS
        if appearance_source == "dinov2"
        else SPECTRAL_METHOD_HYPOTHESIS
    )
    test_path, categories, cat2json, _cat2img = _load_pairs(args.dataset_path)
    with open("spair_detailed_captions.json") as handle:
        captions = json.load(handle)
    attention_audit = _audit_index(args.attention_audit_json)
    memory_cache = _FjsarMemoryCache(
        int(float(args.fjsar_memory_cache_gb) * (1024**3))
    )
    pre_norm = nn.LayerNorm(3072, elementwise_affine=False, eps=1e-6).to(device)
    pairs_by_category = {
        category: _load_category_pairs(
            args, test_path, category, cat2json[category]
        )
        for category in categories
    }
    required_images = {
        category: sorted(
            {
                data[field]
                for _pair_name, data in pairs_by_category[category]
                for field in ("src_imname", "trg_imname")
            }
        )
        for category in categories
    }
    total_images = sum(len(names) for names in required_images.values())

    dino_entries: dict[tuple[str, str], torch.Tensor] = {}
    dino_feature_bytes = 0
    dino_extracted_images = 0
    args.device = str(device)
    if appearance_source == "dinov2":
        # Load each large frozen model in isolation. DINO is extracted first
        # so a missing local checkpoint fails before cache-free FLUX work.
        print(
            f"Extracting {total_images} unique DINOv2 features into CPU RAM; "
            "persistent disk caching is disabled."
        )
        dino_extractor = build_dino_extractor(args, dino_config)
        try:
            for category in categories:
                image_root = Path(args.dataset_path) / "JPEGImages" / category
                for image_name in tqdm(
                    required_images[category],
                    desc=f"extract DINO {category}",
                ):
                    with Image.open(image_root / image_name) as image:
                        feature = dino_extractor(image).to(
                            dtype=torch.float16,
                            device="cpu",
                        )
                    dino_entries[(category, image_name)] = feature
                    dino_feature_bytes += int(
                        feature.numel() * feature.element_size()
                    )
        finally:
            dino_extractor.close()
            del dino_extractor
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        print(
            "DINOv2 extraction complete: "
            f"{len(dino_entries)} images, "
            f"{dino_feature_bytes / (1024 ** 3):.2f} GiB CPU RAM."
        )
        dino_extracted_images = len(dino_entries)

    native_entries: dict[tuple[str, str], dict[str, Any]] = {}
    extracted_native_bytes = 0
    extracted_native_images = 0
    if extract_in_memory and appearance_source != "filtered_attention_svd":
        print(
            f"Extracting {total_images} unique native FLUX features into CPU RAM; "
            "persistent disk caching is disabled."
        )
        featurizer, flux_model, _blocks = _load_flux_fjsar_runtime(args, categories)
        try:
            for category in categories:
                for image_name in tqdm(
                    required_images[category],
                    desc=f"extract native {category}",
                ):
                    entry = extract_flux_native_entry(
                        featurizer,
                        args,
                        dataset_path=args.dataset_path,
                        category=category,
                        image_name=image_name,
                        caption=captions[category + image_name],
                    )
                    native_entries[(category, image_name)] = entry
                    extracted_native_images += 1
                    extracted_native_bytes += nested_tensor_nbytes(entry)
        finally:
            del _blocks, flux_model, featurizer
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        print(
            "Native extraction complete: "
            f"{extracted_native_images} images, "
            f"{extracted_native_bytes / (1024 ** 3):.2f} GiB CPU RAM. "
            "Loading RoMa after releasing FLUX."
        )

    spectral_entries: dict[tuple[str, str], dict[str, Any]] = {}
    spectral_feature_bytes = 0
    spectral_diagnostics: list[dict[str, Any]] = []
    if appearance_source == "filtered_attention_svd":
        total_pairs = sum(len(rows) for rows in pairs_by_category.values())
        if extract_in_memory:
            print(
                f"Streaming {total_images} unique native FLUX entries category by "
                f"category and extracting locked spectral features for {total_pairs} "
                "pairs; replay states are released after each category and no "
                "persistent cache will be written."
            )
        else:
            print(
                f"Extracting locked filtered-attention SVD features for {total_pairs} "
                "pairs into CPU RAM; no persistent cache will be written."
            )
        featurizer, flux_model, spectral_blocks = _load_flux_fjsar_runtime(
            args,
            categories,
        )
        spectral_capture = (
            _make_fjsar_capture(args, flux_model) if extract_in_memory else None
        )
        try:
            for category in categories:
                category_replay_entries: dict[str, dict[str, Any]] = {}
                if extract_in_memory:
                    for image_name in tqdm(
                        required_images[category],
                        desc=f"extract replay {category}",
                    ):
                        entry = _extract_flux_fjsar_entry(
                            args.dataset_path,
                            category,
                            image_name,
                            captions[category + image_name],
                            args,
                            featurizer,
                            spectral_capture,
                        )
                        category_replay_entries[image_name] = entry
                        native_entry = strip_flux_replay_entry(entry)
                        native_entries[(category, image_name)] = native_entry
                        extracted_native_images += 1
                        extracted_native_bytes += nested_tensor_nbytes(native_entry)
                for pair_name, data in tqdm(
                    pairs_by_category[category],
                    desc=f"extract spectral {category}",
                ):
                    if extract_in_memory:
                        source_entry = category_replay_entries[data["src_imname"]]
                        target_entry = category_replay_entries[data["trg_imname"]]
                    else:
                        source_entry = _load_cached_entry(
                            args,
                            category,
                            data["src_imname"],
                            captions[category + data["src_imname"]],
                            memory_cache,
                        )
                        target_entry = _load_cached_entry(
                            args,
                            category,
                            data["trg_imname"],
                            captions[category + data["trg_imname"]],
                            memory_cache,
                        )
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
                    source_spectral, target_spectral, diagnostics = (
                        flux_fjsar_filtered_spectral_feature_maps(
                            source_native,
                            target_native,
                            src_replay_state=source_entry["replay_state"],
                            trg_replay_state=target_entry["replay_state"],
                            blocks=spectral_blocks,
                            rank=64,
                            radius=2,
                            weight=0.5,
                            include_native=False,
                        )
                    )
                    if bool(diagnostics.get("gt_used", True)):
                        raise RuntimeError("spectral extraction unexpectedly used GT")
                    source_cpu = source_spectral.to(
                        dtype=torch.float16,
                        device="cpu",
                    )
                    target_cpu = target_spectral.to(
                        dtype=torch.float16,
                        device="cpu",
                    )
                    spectral_entries[(category, pair_name)] = {
                        "source": source_cpu,
                        "target": target_cpu,
                        "diagnostics": diagnostics,
                    }
                    spectral_diagnostics.append(diagnostics)
                    spectral_feature_bytes += int(
                        source_cpu.numel() * source_cpu.element_size()
                        + target_cpu.numel() * target_cpu.element_size()
                    )
                    del (
                        source_native,
                        target_native,
                        source_spectral,
                        target_spectral,
                    )
                if extract_in_memory:
                    del category_replay_entries
                    gc.collect()
        finally:
            if spectral_capture is not None:
                spectral_capture.close()
            del spectral_capture, spectral_blocks, flux_model, featurizer
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if len(spectral_entries) != total_pairs:
            raise RuntimeError(
                f"spectral extraction produced {len(spectral_entries)} entries; "
                f"expected {total_pairs}"
            )
        if extract_in_memory and len(native_entries) != total_images:
            raise RuntimeError(
                f"streaming extraction retained {len(native_entries)} native entries; "
                f"expected {total_images}"
            )
        print(
            "Spectral extraction complete: "
            f"{len(spectral_entries)} pairs, "
            f"{spectral_feature_bytes / (1024 ** 3):.2f} GiB CPU RAM. "
            "Loading RoMa after releasing FLUX."
        )

    roma = _build_roma(args, device)
    pair_records: list[dict[str, Any]] = []
    category_results: dict[str, Any] = OrderedDict()

    try:
        for category in categories:
            category_records = []
            category_pairs = pairs_by_category[category]
            for pair_name, data in tqdm(category_pairs, desc=f"evaluate {category}"):
                source_size = data["src_imsize"][:2][::-1]
                target_size = data["trg_imsize"][:2][::-1]
                if extract_in_memory:
                    source_entry = native_entries[(category, data["src_imname"])]
                    target_entry = native_entries[(category, data["trg_imname"])]
                else:
                    source_entry = _load_cached_entry(
                        args,
                        category,
                        data["src_imname"],
                        captions[category + data["src_imname"]],
                        memory_cache,
                    )
                    target_entry = _load_cached_entry(
                        args,
                        category,
                        data["trg_imname"],
                        captions[category + data["trg_imname"]],
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

                source_path = (
                    Path(args.dataset_path)
                    / "JPEGImages"
                    / category
                    / data["src_imname"]
                )
                target_path = (
                    Path(args.dataset_path)
                    / "JPEGImages"
                    / category
                    / data["trg_imname"]
                )
                warp, certainty = _run_roma_pair(
                    roma, source_path, target_path, device
                )
                fields = roma_warp_to_token_fields(
                    warp,
                    certainty,
                    source_grid=tuple(map(int, source_native.shape[-2:])),
                    target_grid=tuple(map(int, target_native.shape[-2:])),
                )
                identity = build_identity_maps_from_warp_fields(
                    source_native,
                    target_native,
                    fields["warp_ab"],
                    fields["warp_ba"],
                    source_support=fields["support_a"],
                    target_support=fields["support_b"],
                    config=config,
                )

                if appearance_source == "dinov2":
                    source_auxiliary = sample_dino_map_on_image_grid(
                        dino_entries[(category, data["src_imname"])].to(device),
                        image_size=source_size,
                        output_grid=source_native.shape[-2:],
                        canvas_size=dino_config.image_size,
                    )
                    target_auxiliary = sample_dino_map_on_image_grid(
                        dino_entries[(category, data["trg_imname"])].to(device),
                        image_size=target_size,
                        output_grid=target_native.shape[-2:],
                        canvas_size=dino_config.image_size,
                    )
                    tri_factor = build_roma_augmented_appearance(
                        source_native,
                        target_native,
                        source_auxiliary,
                        target_auxiliary,
                        fields["warp_ab"],
                        fields["warp_ba"],
                        source_support=fields["support_a"],
                        target_support=fields["support_b"],
                        config=config,
                    )
                    appearance_identity = tri_factor
                    main_source_key = "source_fused"
                    main_target_key = "target_fused"
                else:
                    spectral_entry = spectral_entries[(category, pair_name)]
                    source_auxiliary = spectral_entry["source"].to(device)
                    target_auxiliary = spectral_entry["target"].to(device)
                    appearance_identity = build_weighted_roma_augmented_appearance(
                        source_native,
                        target_native,
                        source_auxiliary,
                        target_auxiliary,
                        fields["warp_ab"],
                        fields["warp_ba"],
                        auxiliary_weight=0.5,
                        source_support=fields["support_a"],
                        target_support=fields["support_b"],
                        config=config,
                    )
                    tri_factor = appearance_identity
                    main_source_key = "source_fused_dual"
                    main_target_key = "target_fused_dual"

                source_appearance_full = F.interpolate(
                    tri_factor["source_appearance"].to(torch.float16),
                    size=source_size,
                    mode="bilinear",
                )
                target_appearance_full = F.interpolate(
                    tri_factor["target_appearance"].to(torch.float16),
                    size=target_size,
                    mode="bilinear",
                )
                appearance_predictions = cosine_nn_predict(
                    source_appearance_full,
                    target_appearance_full,
                    data["src_kps"],
                )
                del source_appearance_full, target_appearance_full

                source_tri_full = F.interpolate(
                    tri_factor[main_source_key].to(torch.float16),
                    size=source_size,
                    mode="bilinear",
                )
                target_tri_full = F.interpolate(
                    tri_factor[main_target_key].to(torch.float16),
                    size=target_size,
                    mode="bilinear",
                )
                method_predictions = cosine_nn_predict(
                    source_tri_full,
                    target_tri_full,
                    data["src_kps"],
                )
                del source_tri_full, target_tri_full

                appearance_variable_predictions = _predict_feature_pair(
                    appearance_identity["source_fused_variable_norm"],
                    appearance_identity["target_fused_variable_norm"],
                    source_size,
                    target_size,
                    data["src_kps"],
                )
                appearance_forward_predictions = _predict_feature_pair(
                    appearance_identity["source_fused_forward"],
                    appearance_identity["target_fused_forward"],
                    source_size,
                    target_size,
                    data["src_kps"],
                )
                appearance_dual_predictions = _predict_feature_pair(
                    appearance_identity["source_fused_dual"],
                    appearance_identity["target_fused_dual"],
                    source_size,
                    target_size,
                    data["src_kps"],
                )

                source_fused_full = F.interpolate(
                    identity["source_fused_dual"].to(torch.float16),
                    size=source_size,
                    mode="bilinear",
                )
                target_fused_full = F.interpolate(
                    identity["target_fused_dual"].to(torch.float16),
                    size=target_size,
                    mode="bilinear",
                )
                dual_predictions = cosine_nn_predict(
                    source_fused_full, target_fused_full, data["src_kps"]
                )
                del source_fused_full, target_fused_full

                source_constant_norm_full = F.interpolate(
                    identity["source_fused"].to(torch.float16),
                    size=source_size,
                    mode="bilinear",
                )
                target_constant_norm_full = F.interpolate(
                    identity["target_fused"].to(torch.float16),
                    size=target_size,
                    mode="bilinear",
                )
                constant_norm_predictions = cosine_nn_predict(
                    source_constant_norm_full,
                    target_constant_norm_full,
                    data["src_kps"],
                )
                del source_constant_norm_full, target_constant_norm_full

                source_variable_norm_full = F.interpolate(
                    identity["source_fused_variable_norm"].to(torch.float16),
                    size=source_size,
                    mode="bilinear",
                )
                target_variable_norm_full = F.interpolate(
                    identity["target_fused_variable_norm"].to(torch.float16),
                    size=target_size,
                    mode="bilinear",
                )
                variable_norm_predictions = cosine_nn_predict(
                    source_variable_norm_full,
                    target_variable_norm_full,
                    data["src_kps"],
                )
                del source_variable_norm_full, target_variable_norm_full

                source_forward_full = F.interpolate(
                    identity["source_fused_forward"].to(torch.float16),
                    size=source_size,
                    mode="bilinear",
                )
                target_forward_full = F.interpolate(
                    identity["target_fused_forward"].to(torch.float16),
                    size=target_size,
                    mode="bilinear",
                )
                forward_predictions = cosine_nn_predict(
                    source_forward_full,
                    target_forward_full,
                    data["src_kps"],
                )
                del source_forward_full, target_forward_full

                source_identity_full = F.interpolate(
                    identity["source_identity"].to(torch.float16),
                    size=source_size,
                    mode="bilinear",
                )
                target_identity_full = F.interpolate(
                    identity["target_identity"].to(torch.float16),
                    size=target_size,
                    mode="bilinear",
                )
                identity_predictions = cosine_nn_predict(
                    source_identity_full, target_identity_full, data["src_kps"]
                )
                del source_identity_full, target_identity_full


                source_forward_identity_full = F.interpolate(
                    identity["source_forward_identity"].to(torch.float16),
                    size=source_size,
                    mode="bilinear",
                )
                target_forward_identity_full = F.interpolate(
                    identity["target_forward_identity"].to(torch.float16),
                    size=target_size,
                    mode="bilinear",
                )
                forward_identity_predictions = cosine_nn_predict(
                    source_forward_identity_full,
                    target_forward_identity_full,
                    data["src_kps"],
                )
                del source_forward_identity_full, target_forward_identity_full

                exact_forward = build_exact_forward_coordinate_maps(
                    identity["warp_ab"],
                    identity["reliability_a"],
                    identity["reliability_b"],
                    source_size=source_size,
                    target_size=target_size,
                )
                exact_forward_identity_predictions = cosine_nn_predict(
                    exact_forward["source_unit"],
                    exact_forward["target_unit"],
                    data["src_kps"],
                )
                source_native_exact_full = F.interpolate(
                    F.normalize(source_native.float(), dim=1, eps=1e-12).to(
                        torch.float16
                    ),
                    size=source_size,
                    mode="bilinear",
                )
                target_native_exact_full = F.interpolate(
                    F.normalize(target_native.float(), dim=1, eps=1e-12).to(
                        torch.float16
                    ),
                    size=target_size,
                    mode="bilinear",
                )
                source_exact_fused = F.normalize(
                    torch.cat(
                        (
                            source_native_exact_full,
                            exact_forward["source_gated"].to(torch.float16),
                        ),
                        dim=1,
                    ),
                    dim=1,
                    eps=1e-6,
                )
                target_exact_fused = F.normalize(
                    torch.cat(
                        (
                            target_native_exact_full,
                            exact_forward["target_gated"].to(torch.float16),
                        ),
                        dim=1,
                    ),
                    dim=1,
                    eps=1e-6,
                )
                exact_forward_fusion_predictions = cosine_nn_predict(
                    source_exact_fused,
                    target_exact_fused,
                    data["src_kps"],
                )
                del (
                    source_native_exact_full,
                    target_native_exact_full,
                    source_exact_fused,
                    target_exact_fused,
                    exact_forward,
                )

                mapped = sample_dense_field(
                    identity["warp_ab"], data["src_kps"], source_size
                )
                direct_predictions = _normalized_to_pixels(mapped, target_size)
                raw_certainty = _sample_scalar(
                    fields["certainty_a_raw"], data["src_kps"], source_size
                )
                support = _sample_scalar(
                    identity["support_a"], data["src_kps"], source_size
                )
                cycle = _sample_scalar(
                    identity["cycle_error_a"], data["src_kps"], source_size
                )
                jacobian = _sample_scalar(
                    identity["jacobian_confidence_a"], data["src_kps"], source_size
                )
                reliability = _sample_scalar(
                    identity["reliability_a"], data["src_kps"], source_size
                )
                threshold = max(
                    data["trg_bndbox"][3] - data["trg_bndbox"][1],
                    data["trg_bndbox"][2] - data["trg_bndbox"][0],
                )
                point_records = []
                for index, target_point in enumerate(data["trg_kps"]):
                    baseline_hit = bool(
                        _pck(baseline_predictions[index], target_point, threshold)
                    )
                    method_hit = bool(
                        _pck(method_predictions[index], target_point, threshold)
                    )
                    appearance_hit = bool(
                        _pck(appearance_predictions[index], target_point, threshold)
                    )
                    appearance_roma_variable_hit = bool(
                        _pck(
                            appearance_variable_predictions[index],
                            target_point,
                            threshold,
                        )
                    )
                    appearance_roma_forward_hit = bool(
                        _pck(
                            appearance_forward_predictions[index],
                            target_point,
                            threshold,
                        )
                    )
                    appearance_roma_dual_hit = bool(
                        _pck(
                            appearance_dual_predictions[index],
                            target_point,
                            threshold,
                        )
                    )
                    exact_forward_fusion_hit = bool(
                        _pck(
                            exact_forward_fusion_predictions[index],
                            target_point,
                            threshold,
                        )
                    )
                    dual_hit = bool(
                        _pck(dual_predictions[index], target_point, threshold)
                    )
                    identity_hit = bool(
                        _pck(identity_predictions[index], target_point, threshold)
                    )
                    variable_norm_hit = bool(
                        _pck(
                            variable_norm_predictions[index],
                            target_point,
                            threshold,
                        )
                    )
                    constant_norm_hit = bool(
                        _pck(
                            constant_norm_predictions[index],
                            target_point,
                            threshold,
                        )
                    )
                    forward_hit = bool(
                        _pck(forward_predictions[index], target_point, threshold)
                    )
                    forward_identity_hit = bool(
                        _pck(
                            forward_identity_predictions[index],
                            target_point,
                            threshold,
                        )
                    )
                    exact_forward_identity_hit = bool(
                        _pck(
                            exact_forward_identity_predictions[index],
                            target_point,
                            threshold,
                        )
                    )
                    direct_hit = bool(
                        _pck(direct_predictions[index], target_point, threshold)
                    )
                    annotation = attention_audit.get((category, pair_name, index), {})
                    if attention_audit and not annotation:
                        raise ValueError(
                            "attention audit is missing the evaluated point "
                            f"{category}/{pair_name}/{index}"
                        )
                    if annotation and bool(annotation.get("baseline_pck_hit")) != baseline_hit:
                        raise ValueError(
                            f"attention audit baseline mismatch at {category}/{pair_name}/{index}"
                        )
                    point_records.append(
                        {
                            "keypoint_index": int(index),
                            "source_point": list(data["src_kps"][index]),
                            "target_point": list(target_point),
                            "baseline_prediction": baseline_predictions[index],
                            "method_prediction": method_predictions[index],
                            "appearance_fusion_prediction": appearance_predictions[index],
                            "appearance_roma_variable_prediction": (
                                appearance_variable_predictions[index]
                            ),
                            "appearance_roma_forward_prediction": (
                                appearance_forward_predictions[index]
                            ),
                            "appearance_roma_dual_prediction": (
                                appearance_dual_predictions[index]
                            ),
                            "exact_forward_fusion_prediction": (
                                exact_forward_fusion_predictions[index]
                            ),
                            "dual_fusion_prediction": dual_predictions[index],
                            "variable_norm_fusion_prediction": (
                                variable_norm_predictions[index]
                            ),
                            "constant_norm_fusion_prediction": (
                                constant_norm_predictions[index]
                            ),
                            "forward_only_fusion_prediction": (
                                forward_predictions[index]
                            ),
                            "forward_identity_only_prediction": (
                                forward_identity_predictions[index]
                            ),
                            "exact_forward_identity_prediction": (
                                exact_forward_identity_predictions[index]
                            ),
                            "identity_only_prediction": identity_predictions[index],
                            "direct_warp_prediction": direct_predictions[index],
                            "baseline_pck_hit": baseline_hit,
                            "method_pck_hit": method_hit,
                            "appearance_fusion_pck_hit": appearance_hit,
                            "appearance_roma_variable_pck_hit": (
                                appearance_roma_variable_hit
                            ),
                            "appearance_roma_forward_pck_hit": (
                                appearance_roma_forward_hit
                            ),
                            "appearance_roma_dual_pck_hit": appearance_roma_dual_hit,
                            "exact_forward_fusion_pck_hit": exact_forward_fusion_hit,
                            "dual_fusion_pck_hit": dual_hit,
                            "variable_norm_fusion_pck_hit": variable_norm_hit,
                            "constant_norm_fusion_pck_hit": constant_norm_hit,
                            "forward_only_fusion_pck_hit": forward_hit,
                            "forward_identity_only_pck_hit": forward_identity_hit,
                            "exact_forward_identity_pck_hit": (
                                exact_forward_identity_hit
                            ),
                            "exact_forward_direct_pixel_agreement": bool(
                                exact_forward_identity_predictions[index]
                                == direct_predictions[index]
                            ),
                            "identity_only_pck_hit": identity_hit,
                            "direct_warp_pck_hit": direct_hit,
                            "rescued_vs_baseline": bool(method_hit and not baseline_hit),
                            "harmed_vs_baseline": bool(baseline_hit and not method_hit),
                            "attention_top1_pck_hit": annotation.get(
                                "attention_top1_pck_hit"
                            ),
                            "attention_top20_pck_hit": annotation.get(
                                "attention_top20_pck_hit"
                            ),
                            "both_wrong_top20_hit": annotation.get(
                                "both_wrong_top20_hit"
                            ),
                            "warp_state": {
                                "predicted_target_normalized": [
                                    float(value)
                                    for value in mapped[index].detach().cpu()
                                ],
                                "raw_certainty": raw_certainty[index],
                                "support": support[index],
                                "cycle_error_cells": cycle[index],
                                "jacobian_confidence": jacobian[index],
                                "reliability": reliability[index],
                            },
                        }
                    )
                record = {
                    "category": category,
                    "pair_json": pair_name,
                    "src_image": data["src_imname"],
                    "trg_image": data["trg_imname"],
                    "keypoint_count": len(point_records),
                    "points": point_records,
                }
                category_records.append(record)
                pair_records.append(record)
                del (
                    warp,
                    certainty,
                    fields,
                    identity,
                    tri_factor,
                    appearance_identity,
                    source_auxiliary,
                    target_auxiliary,
                    source_native,
                    target_native,
                )
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
            if extract_in_memory:
                for key in [key for key in native_entries if key[0] == category]:
                    del native_entries[key]
            for key in [key for key in dino_entries if key[0] == category]:
                del dino_entries[key]
            for key in [key for key in spectral_entries if key[0] == category]:
                del spectral_entries[key]
    finally:
        del roma
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metrics = _metric_block(pair_records)
    mechanism = _mechanism_summary(pair_records)
    matcher_name = (
        "frozen_flux_dinov2_roma_identity"
        if appearance_source == "dinov2"
        else "frozen_flux_spectral_roma_identity"
    )
    result = {
        "matcher": matcher_name,
        "method_hypothesis": method_hypothesis,
        "protocol": {
            "subset": str(args.subset),
            "pairs_per_category": int(args.pairs_per_cat),
            "split_seed": int(args.split_seed),
            "native_flux_block": int(args.k[0]),
            "native_timestep": int(args.t),
            "native_ensemble_size": int(args.ensemble_size),
            "appearance_source": appearance_source,
            "dino_model": (
                "DINOv2 ViT-B/14" if appearance_source == "dinov2" else ""
            ),
            "dino_backend": (
                str(args.dino_backend) if appearance_source == "dinov2" else ""
            ),
            "dino_model_repo": (
                str(args.dino_model_repo or dino_config.model_name)
                if appearance_source == "dinov2"
                else ""
            ),
            "dino_feature": (
                "block11 pre-final-norm token"
                if appearance_source == "dinov2"
                else ""
            ),
            "dino_input": (
                "840x840 aspect-preserving zero canvas"
                if appearance_source == "dinov2"
                else ""
            ),
            "dino_grid": (
                "60x60 sampled on native token centres"
                if appearance_source == "dinov2"
                else ""
            ),
            "spectral_attention_replay": (
                "exact_mutual_without_coordinate_bias"
                if appearance_source == "filtered_attention_svd"
                else ""
            ),
            "spectral_local_support_radius": (
                2 if appearance_source == "filtered_attention_svd" else 0
            ),
            "spectral_rank": (
                64 if appearance_source == "filtered_attention_svd" else 0
            ),
            "spectral_weight": (
                0.5 if appearance_source == "filtered_attention_svd" else 0.0
            ),
            "appearance_fusion": (
                "unit_FLUX_plus_unit_DINO_divided_by_sqrt2"
                if appearance_source == "dinov2"
                else "unit_FLUX_plus_weight0.5_filtered_attention_SVD"
            ),
            "identity_fusion": (
                "RoMa_mutual_variable_norm"
                if appearance_source == "dinov2"
                else "RoMa_dual_identity_on_spectral_appearance"
            ),
            "roma_coarse_res": int(args.roma_coarse_res),
            "roma_upsample_res": int(args.roma_upsample_res),
            "roma_precision": str(args.roma_precision),
            "dense_warp_config": asdict(config),
            "certainty_normalization": "c/(c+positive_pair_median)",
            "attention_audit_json": str(args.attention_audit_json or ""),
            "attention_used_for_prediction": bool(
                appearance_source == "filtered_attention_svd"
            ),
            "final_matcher": "official global cosine NN",
            "native_feature_source": (
                "runtime_FLUX_category_scoped_replay_CPU_RAM_only"
                if extract_in_memory
                and appearance_source == "filtered_attention_svd"
                else "runtime_FLUX_CPU_RAM_only"
                if extract_in_memory
                else "read_only_FLUX_disk_cache"
            ),
            "runtime_replay_retention": (
                "current_category_only"
                if extract_in_memory
                and appearance_source == "filtered_attention_svd"
                else "not_applicable"
            ),
            "persistent_feature_cache_written": False,
            "runtime_extracted_native_images": int(extracted_native_images),
            "runtime_extracted_native_bytes": int(extracted_native_bytes),
            "runtime_extracted_dino_images": int(dino_extracted_images),
            "runtime_extracted_dino_bytes": int(dino_feature_bytes),
            "runtime_extracted_spectral_pairs": (
                int(len(pair_records))
                if appearance_source == "filtered_attention_svd"
                else 0
            ),
            "runtime_extracted_spectral_bytes": int(spectral_feature_bytes),
            "spectral_gt_used": False,
            "spectral_mean_local_support": (
                float(
                    np.mean(
                        [
                            row["mean_local_support"]
                            for row in spectral_diagnostics
                        ]
                    )
                )
                if spectral_diagnostics
                else 0.0
            ),
        },
        "categories": category_results,
        "all": metrics,
        "mechanism_summary": mechanism,
        "fjsar_memory_cache": memory_cache.stats(),
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    root = output_path.with_suffix("")
    audit_path = Path(f"{root}_{matcher_name}_audit.json")
    summary_path = Path(f"{root}_{matcher_name}_summary.json")
    result["audit_path"] = str(audit_path)
    result["summary_path"] = str(summary_path)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    audit_path.write_text(
        json.dumps(
            {
                "matcher": result["matcher"],
                "method_hypothesis": method_hypothesis,
                "protocol": result["protocol"],
                "summary": mechanism,
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
                "method_hypothesis": method_hypothesis,
                "protocol": result["protocol"],
                "metrics": metrics,
                "summary": mechanism,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    hard = mechanism["both_wrong_top20_hit"]
    print(f"Matcher: {matcher_name}")
    print(
        f"Baseline All per image/point: {metrics['baseline_image']:.2f} / "
        f"{metrics['baseline_point']:.2f}"
    )
    print(
        f"Method All per image/point: {metrics['method_image']:.2f} / "
        f"{metrics['method_point']:.2f}; point gain={metrics['point_gain']:.2f}"
    )
    if appearance_source == "dinov2":
        print(
            "Same-run FLUX+DINO / FLUX+RoMa / tri-factor point: "
            f"{100.0 * mechanism['appearance_fusion_correct'] / max(1, mechanism['points']):.2f} / "
            f"{100.0 * mechanism['variable_norm_fusion_correct'] / max(1, mechanism['points']):.2f} / "
            f"{100.0 * mechanism['method_correct'] / max(1, mechanism['points']):.2f}"
        )
    else:
        print(
            "Same-run spectral / native+RoMa dual / "
            "spectral+RoMa variable/forward/dual point: "
            f"{100.0 * mechanism['appearance_fusion_correct'] / max(1, mechanism['points']):.2f} / "
            f"{100.0 * mechanism['dual_fusion_correct'] / max(1, mechanism['points']):.2f} / "
            f"{100.0 * mechanism['appearance_roma_variable_correct'] / max(1, mechanism['points']):.2f} / "
            f"{100.0 * mechanism['appearance_roma_forward_correct'] / max(1, mechanism['points']):.2f} / "
            f"{100.0 * mechanism['appearance_roma_dual_correct'] / max(1, mechanism['points']):.2f}"
        )
    print(
        "Same-run exact-forward fusion point: "
        f"{100.0 * mechanism['exact_forward_fusion_correct'] / max(1, mechanism['points']):.2f}"
    )
    print(
        "Same-run constant-norm/forward-only fusion point: "
        f"{100.0 * mechanism['constant_norm_fusion_correct'] / max(1, mechanism['points']):.2f} / "
        f"{100.0 * mechanism['forward_only_fusion_correct'] / max(1, mechanism['points']):.2f}"
    )
    print(
        "Same-run mutual/forward identity-only point: "
        f"{100.0 * sum(point['identity_only_pck_hit'] for pair in pair_records for point in pair['points']) / max(1, mechanism['points']):.2f} / "
        f"{100.0 * mechanism['forward_identity_only_correct'] / max(1, mechanism['points']):.2f}"
    )
    print(
        "Same-run native+RoMa dual / exact-forward identity-only point: "
        f"{100.0 * mechanism['dual_fusion_correct'] / max(1, mechanism['points']):.2f} / "
        f"{100.0 * mechanism['exact_forward_identity_correct'] / max(1, mechanism['points']):.2f}"
    )
    print(
        "Exact-forward/direct pixel agreement: "
        f"{100.0 * mechanism['exact_forward_direct_pixel_agreement'] / max(1, mechanism['points']):.2f}"
    )
    if hard["points"]:
        print(
            "Hard both-wrong direct/identity/fused: "
            f"{100.0 * hard['direct_warp_top1']:.2f} / "
            f"{100.0 * hard['identity_only_top1']:.2f} / "
            f"{100.0 * hard['fused_top1']:.2f}"
        )
    else:
        print(
            "Hard both-wrong diagnostics unavailable: "
            "no matching --attention_audit_json was supplied."
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--attention_audit_json", default="")
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
    parser.add_argument("--fjsar_disk_cache_path", default="")
    parser.add_argument(
        "--extract_native_in_memory",
        action="store_true",
        default=False,
        help=(
            "Extract only the selected pairs' FLUX evidence into CPU RAM, release "
            "FLUX before loading RoMa, and never write a feature cache. The spectral "
            "path retains full replay states for the current category only."
        ),
    )
    parser.add_argument("--fjsar_memory_cache_gb", type=float, default=4.0)
    parser.add_argument("--fjsar_shared_noise", action="store_true", default=False)
    parser.add_argument(
        "--appearance_source",
        choices=("dinov2", "filtered_attention_svd"),
        default="dinov2",
        help=(
            "Auxiliary appearance evidence. filtered_attention_svd uses the "
            "locked exact-attention radius2/rank64/weight0.5 protocol."
        ),
    )
    parser.add_argument(
        "--dino_backend",
        choices=("transformers", "torch_hub"),
        default="transformers",
    )
    parser.add_argument(
        "--dino_precision",
        choices=("fp32", "fp16", "bf16"),
        default="fp32",
    )
    parser.add_argument("--dino_local_files_only", action="store_true", default=False)
    parser.add_argument("--dino_model_repo", default=None)
    parser.add_argument("--roma_coarse_res", type=int, default=560)
    parser.add_argument("--roma_upsample_res", type=int, default=864)
    parser.add_argument(
        "--roma_precision",
        choices=("fp16", "bf16", "fp32"),
        default="fp16",
    )
    parser.add_argument("--roma_weights", required=True)
    parser.add_argument("--roma_dinov2_weights", required=True)
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
    if run_device.type == "cpu" and parsed.roma_precision != "fp32":
        parsed.roma_precision = "fp32"
    evaluate(parsed, run_device)

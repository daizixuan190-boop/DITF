"""Attach native DiTF NN reliability signals to an existing SPair audit.

This utility reuses canonical FLUX replay-cache feature tensors. It does not
load FLUX weights, run cross-attention, or load DINO. Existing predictions and
candidate rankings are preserved apart from the added protocol and per-point
native_nn_diagnostics fields.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from eval_spair_matcher_ablation import (
    _FjsarMemoryCache,
    _get_flux_fjsar_entry,
    _prepare_feature_tensors,
)
from spair_matchers import (
    cosine_candidate_diagnostics,
    cosine_nn_predict_with_diagnostics,
)


def _image_size(
    dataset_path: str,
    category: str,
    image_name: str,
) -> tuple[int, int]:
    path = Path(dataset_path) / "JPEGImages" / category / image_name
    with Image.open(path) as image:
        return int(image.height), int(image.width)


def _validate_input(payload: dict[str, Any]) -> list[dict[str, Any]]:
    pair_records = payload.get("pair_records")
    if not isinstance(pair_records, list) or not pair_records:
        raise ValueError("input audit must contain non-empty pair_records")
    for pair in pair_records:
        for key in ("category", "src_image", "trg_image", "points"):
            if key not in pair:
                raise ValueError(f"pair record is missing {key}")
        for point in pair["points"]:
            if "source_point" not in point or "baseline_prediction" not in point:
                raise ValueError(
                    "point records must contain source_point and baseline_prediction"
                )
    return pair_records


@torch.inference_mode()
def augment_audit(
    payload: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    if not args.fjsar_require_disk_cache or not args.fjsar_disk_cache_path:
        raise ValueError("canonical FLUX replay cache is required")

    result = copy.deepcopy(payload)
    pair_records = _validate_input(result)
    captions = json.loads(Path("spair_detailed_captions.json").read_text())
    memory_cache = _FjsarMemoryCache(
        int(float(args.fjsar_memory_cache_gb) * (1024 ** 3))
    )
    pre_norm = nn.LayerNorm(
        3072,
        elementwise_affine=False,
        eps=1e-6,
    ).to(device)

    for pair in tqdm(pair_records, desc="native NN diagnostics"):
        category = str(pair["category"])
        src_name = str(pair["src_image"])
        trg_name = str(pair["trg_image"])
        source_size = _image_size(args.dataset_path, category, src_name)
        target_size = _image_size(args.dataset_path, category, trg_name)

        src_entry = _get_flux_fjsar_entry(
            args.dataset_path,
            category,
            src_name,
            captions[category + src_name],
            args,
            None,
            None,
            memory_cache,
        )
        trg_entry = _get_flux_fjsar_entry(
            args.dataset_path,
            category,
            trg_name,
            captions[category + trg_name],
            args,
            None,
            None,
            memory_cache,
        )
        src_features = _prepare_feature_tensors(
            src_entry["feature"],
            src_entry["ada"],
            args,
            pre_norm,
            device,
        )
        trg_features = _prepare_feature_tensors(
            trg_entry["feature"],
            trg_entry["ada"],
            args,
            pre_norm,
            device,
        )
        src_full = F.interpolate(
            src_features.to(torch.float16),
            size=source_size,
            mode="bilinear",
        )
        trg_full = F.interpolate(
            trg_features.to(torch.float16),
            size=target_size,
            mode="bilinear",
        )
        source_points = [point["source_point"] for point in pair["points"]]
        predictions, diagnostics = cosine_nn_predict_with_diagnostics(
            src_full,
            trg_full,
            source_points,
            nonlocal_radius=args.native_nonlocal_radius,
        )
        candidate_points = [
            [candidate["pixel"] for candidate in point.get("candidates", [])]
            for point in pair["points"]
        ]
        candidate_diagnostics = cosine_candidate_diagnostics(
            src_full,
            trg_full,
            source_points,
            candidate_points,
        )
        if len(predictions) != len(pair["points"]):
            raise RuntimeError("native diagnostic point count mismatch")

        for point, prediction, diagnostic, candidate_row in zip(
            pair["points"],
            predictions,
            diagnostics,
            candidate_diagnostics,
        ):
            expected = [int(value) for value in point["baseline_prediction"]]
            actual = [int(value) for value in prediction]
            if actual != expected:
                raise RuntimeError(
                    "native prediction mismatch; cache/protocol does not match "
                    f"the input audit: category={category}, "
                    f"pair={pair.get('pair_json')}, "
                    f"keypoint={point.get('keypoint_index')}, "
                    f"expected={expected}, actual={actual}"
                )
            point["native_nn_diagnostics"] = diagnostic
            candidates = point.get("candidates", [])
            if len(candidates) != len(candidate_row):
                raise RuntimeError("native candidate diagnostic count mismatch")
            for candidate, candidate_diagnostic in zip(
                candidates,
                candidate_row,
            ):
                expected_pixel = [int(value) for value in candidate["pixel"]]
                if candidate_diagnostic["pixel"] != expected_pixel:
                    raise RuntimeError("native candidate pixel alignment mismatch")
                candidate["native_cosine"] = candidate_diagnostic["native_cosine"]
                candidate["native_candidate_rank"] = candidate_diagnostic[
                    "native_candidate_rank"
                ]
                candidate["native_gap_to_candidate_top1"] = candidate_diagnostic[
                    "native_gap_to_candidate_top1"
                ]
                candidate["native_gap_to_full_top1"] = float(
                    diagnostic["top1_cosine"]
                    - candidate_diagnostic["native_cosine"]
                )

        del src_entry, trg_entry
        del src_features, trg_features, src_full, trg_full
        if device.type == "cuda":
            torch.cuda.empty_cache()

    protocol = dict(result.get("protocol") or {})
    protocol["native_nn_diagnostics"] = {
        "source": "canonical_native_DiTF_descriptor",
        "top1_top2_margin": True,
        "top1_nonlocal_margin": True,
        "cycle_back": True,
        "candidate_native_cosine": True,
        "candidate_native_rank": True,
        "nonlocal_radius": int(args.native_nonlocal_radius),
        "gt_used": False,
    }
    protocol["native_diagnostics_cache"] = os.path.abspath(
        args.fjsar_disk_cache_path
    )
    result["protocol"] = protocol
    result["native_diagnostics_memory_cache"] = memory_cache.stats()
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_audit_json", required=True)
    parser.add_argument("--output_audit_json", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--img_size", nargs="+", type=int, default=[640, 640])
    parser.add_argument("--t", type=int, default=260)
    parser.add_argument("--k", nargs="+", type=int, default=[28])
    parser.add_argument("--ensemble_size", type=int, default=8)
    parser.add_argument("--cd", action="store_true", default=False)
    parser.add_argument("--fjsar_shared_noise", action="store_true", default=False)
    parser.add_argument("--fjsar_memory_cache_gb", type=float, default=4.0)
    parser.add_argument("--fjsar_disk_cache_path", required=True)
    parser.add_argument("--fjsar_require_disk_cache", action="store_true")
    parser.add_argument("--fjsar_disk_cache_min_free_gb", type=float, default=0.0)
    parser.add_argument("--native_nonlocal_radius", type=int, default=8)
    parser.set_defaults(
        fjsar_multilayer_identity_audit=False,
        fjsar_multilayer_blocks=(),
        fjsar_trajectory_blocks=(),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input_audit_json)
    output_path = Path(args.output_audit_json)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("output audit must not overwrite the input audit")
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    device = torch.device(
        args.device
        if args.device == "cpu" or torch.cuda.is_available()
        else "cpu"
    )
    augmented = augment_audit(payload, args, device)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(augmented, indent=2), encoding="utf-8")
    print(f"Saved native diagnostics audit: {output_path}")
    print(
        "Pairs/points: "
        f"{len(augmented['pair_records'])}/"
        f"{sum(len(pair['points']) for pair in augmented['pair_records'])}"
    )
    gc.collect()


if __name__ == "__main__":
    main()

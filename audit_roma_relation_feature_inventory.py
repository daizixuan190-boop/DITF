"""Minimal, read-only RoMa relation-feature and cost inventory.

This is a *gate*, not a matcher or a training run.  It establishes whether the
pinned, frozen RoMa runtime exposes projected fields that can retain the
``[source query, attention candidate, feature]`` axis required by a future
candidate-conditioned relation encoder.  It neither ranks candidates nor
reads PCK/GT fields after loading the pre-existing proposal coordinates.

The prior internal audit already rejected projected-feature cosine and GP
coordinate-agreement *scalar scores*.  This inventory intentionally does not
reintroduce either score: it records only official interface shapes, finite
values, timing, and peak memory for one or two natural cross-instance pairs.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from PIL import Image
from tqdm import tqdm

from eval_spair_attention_top20_roma_identity import (
    _build_roma,
    _normalize_points,
    _sample_field,
    _validate_audit,
)


METHOD_CONTRACT = {
    "name": "RoMa relation feature interface and cost gate",
    "purpose": (
        "Verify that frozen official RoMa projected fields preserve the joint "
        "source-query/candidate axis required for a future relation capacity audit."
    ),
    "is_matcher": False,
    "is_training": False,
    "candidate_pool_modified": False,
    "candidate_ranking": "none",
    "gt_used_for_feature_construction": False,
    "pck_used_for_decision": False,
    "forbidden_reuse": [
        "projected-feature cosine ranking",
        "GP coordinate-agreement ranking",
        "warp-error reranking",
        "confidence threshold routing",
        "DINO feature input",
    ],
    "feature_source": "official frozen RoMa extract_backbone_features then decoder.proj",
}


def select_projection_scales(
    pyramid: Mapping[int, torch.Tensor], projection_keys: Sequence[str]
) -> tuple[int, ...]:
    """Return only real numeric scale intersections; never invent a RoMa layer."""

    scales: list[int] = []
    for key in projection_keys:
        try:
            scale = int(key)
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"RoMa decoder projection key must be numeric, got {key!r}") from error
        if str(scale) != str(key):
            raise RuntimeError(f"RoMa decoder projection key must be canonical numeric text, got {key!r}")
        if scale in pyramid:
            field = pyramid[scale]
            if field.ndim != 4 or int(field.shape[0]) != 2:
                raise RuntimeError(
                    f"RoMa pyramid scale {scale} must be two-image [2,C,H,W], got {tuple(field.shape)}"
                )
            scales.append(scale)
    if not scales:
        raise RuntimeError(
            "no real RoMa decoder.proj key intersects extract_backbone_features output"
        )
    return tuple(sorted(scales))


def _sample_projected(field: torch.Tensor, points: torch.Tensor, image_size: Sequence[int]) -> torch.Tensor:
    if field.ndim != 4 or tuple(field.shape[:1]) != (1,):
        raise ValueError("projected RoMa field must be [1,C,H,W]")
    normalized = _normalize_points(points, image_size)
    return _sample_field(field[0].permute(1, 2, 0).contiguous(), normalized)


def inventory_projected_relation_fields(
    fields: Mapping[int, tuple[torch.Tensor, torch.Tensor]],
    *,
    source_points: torch.Tensor,
    candidate_points: torch.Tensor,
    source_size: Sequence[int],
    target_size: Sequence[int],
) -> dict[str, dict[str, Any]]:
    """Validate and describe candidate-axis tensors without calculating a score."""

    if source_points.ndim != 2 or tuple(source_points.shape[-1:]) != (2,):
        raise ValueError("source_points must be [P,2]")
    if candidate_points.ndim != 3 or tuple(candidate_points.shape[-1:]) != (2,):
        raise ValueError("candidate_points must be [P,K,2]")
    if int(source_points.shape[0]) != int(candidate_points.shape[0]):
        raise ValueError("source query and candidate rows must align")
    result: dict[str, dict[str, Any]] = {}
    for scale, (source_field, target_field) in sorted(fields.items()):
        if tuple(source_field.shape) != tuple(target_field.shape):
            raise ValueError(f"RoMa scale {scale} source/target projected shapes must match")
        source_descriptor = _sample_projected(source_field, source_points, source_size)
        candidate_descriptor = _sample_projected(target_field, candidate_points, target_size)
        if tuple(source_descriptor.shape[:1]) != tuple(candidate_descriptor.shape[:1]):
            raise RuntimeError("sampled source and candidate query axes diverged")
        result[str(scale)] = {
            "projected_field_shape": [int(value) for value in source_field.shape],
            "source_descriptor_shape": [int(value) for value in source_descriptor.shape],
            "candidate_descriptor_shape": [int(value) for value in candidate_descriptor.shape],
            "all_finite": bool(torch.isfinite(source_descriptor).all() and torch.isfinite(candidate_descriptor).all()),
        }
    return result


def _load_projected_fields(
    model: Any,
    source_path: Path,
    target_path: Path,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Use only documented objects already exercised by the prior RoMa audit."""

    try:
        from romatch.utils import get_tuple_transform_ops
    except ImportError as error:  # pragma: no cover - AutoDL runtime only.
        raise RuntimeError("RoMa preprocessing utilities are unavailable") from error
    with Image.open(source_path) as source_image, Image.open(target_path) as target_image:
        transform = get_tuple_transform_ops(
            resize=(int(model.h_resized), int(model.w_resized)), normalize=True, clahe=False
        )
        source_tensor, target_tensor = transform(
            (source_image.convert("RGB"), target_image.convert("RGB"))
        )
    batch = {"im_A": source_tensor.unsqueeze(0).to(device), "im_B": target_tensor.unsqueeze(0).to(device)}
    with torch.inference_mode():
        pyramid = model.extract_backbone_features(batch, batched=True)
        projection_keys = tuple(model.decoder.proj.keys())
        scales = select_projection_scales(pyramid, projection_keys)
        fields: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=device.type == "cuda" and amp_dtype != torch.float32,
        ):
            for scale in scales:
                projected = model.decoder.proj[str(scale)](pyramid[scale]).float()
                if projected.ndim != 4 or int(projected.shape[0]) != 2:
                    raise RuntimeError(
                        f"RoMa projected scale {scale} must be [2,C,H,W], got {tuple(projected.shape)}"
                    )
                fields[scale] = (projected[:1].detach(), projected[1:].detach())
    return fields


def _pair_points(pair: Mapping[str, Any], device: torch.device, candidate_limit: int) -> tuple[torch.Tensor, torch.Tensor]:
    rows = list(pair["points"])
    if not rows:
        raise ValueError("audit pair has no source queries")
    candidates = [sorted(row["candidates"], key=lambda item: int(item["attention_rank"])) for row in rows]
    count = min(int(candidate_limit), len(candidates[0]))
    if count <= 0 or any(len(row) < count for row in candidates):
        raise ValueError("candidate_limit does not fit the fixed attention candidate contract")
    return (
        torch.tensor([row["source_point"] for row in rows], dtype=torch.float32, device=device),
        torch.tensor(
            [[candidate["pixel"] for candidate in row[:count]] for row in candidates],
            dtype=torch.float32,
            device=device,
        ),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--attention_audit_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--roma_weights", required=True)
    parser.add_argument("--roma_dinov2_weights", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--roma_coarse_res", type=int, default=560)
    parser.add_argument("--roma_upsample_res", type=int, default=864)
    parser.add_argument("--roma_precision", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--max_pairs", type=int, default=2)
    parser.add_argument("--candidate_limit", type=int, default=20)
    parser.add_argument("--cost_projection_pairs", default="64,256,512")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if int(args.max_pairs) <= 0:
        raise ValueError("max_pairs must be positive: this gate is deliberately tiny")
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and args.roma_precision != "fp32":
        raise ValueError("CPU inventory requires --roma_precision fp32")
    precision = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.roma_precision]
    payload = json.loads(Path(args.attention_audit_json).read_text(encoding="utf-8"))
    records = _validate_audit(payload)[: int(args.max_pairs)]
    model = _build_roma(args, device)
    pair_records: list[dict[str, Any]] = []
    for pair in tqdm(records, desc="RoMa relation feature inventory"):
        source_path = Path(args.dataset_path) / "JPEGImages" / str(pair["category"]) / str(pair["src_image"])
        target_path = Path(args.dataset_path) / "JPEGImages" / str(pair["category"]) / str(pair["trg_image"])
        if not source_path.is_file() or not target_path.is_file():
            raise FileNotFoundError(f"missing SPair image: {source_path} or {target_path}")
        with Image.open(source_path) as source_image, Image.open(target_path) as target_image:
            source_size = (int(source_image.height), int(source_image.width))
            target_size = (int(target_image.height), int(target_image.width))
        source_points, candidate_points = _pair_points(pair, device, int(args.candidate_limit))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        fields = _load_projected_fields(model, source_path, target_path, device, precision)
        inventory = inventory_projected_relation_fields(
            fields,
            source_points=source_points,
            candidate_points=candidate_points,
            source_size=source_size,
            target_size=target_size,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        pair_records.append({
            "category": pair["category"], "src_image": pair["src_image"], "trg_image": pair["trg_image"],
            "query_count": int(source_points.shape[0]), "candidate_count": int(candidate_points.shape[1]),
            "elapsed_seconds": float(elapsed),
            "peak_memory_mib": float(torch.cuda.max_memory_allocated(device) / 2**20) if device.type == "cuda" else None,
            "scales": inventory,
        })
        del fields
        if device.type == "cuda":
            torch.cuda.empty_cache()
    mean_seconds = sum(row["elapsed_seconds"] for row in pair_records) / len(pair_records)
    projection_counts = [int(value.strip()) for value in str(args.cost_projection_pairs).split(",") if value.strip()]
    if not projection_counts or any(value <= 0 for value in projection_counts):
        raise ValueError("cost_projection_pairs must be a comma-separated list of positive pair counts")
    output = {
        "method_contract": METHOD_CONTRACT,
        "protocol": {
            "attention_audit_json": str(Path(args.attention_audit_json).resolve()),
            "max_pairs": int(args.max_pairs), "candidate_limit": int(args.candidate_limit),
            "roma_coarse_res": int(args.roma_coarse_res), "roma_upsample_res": int(args.roma_upsample_res),
            "roma_precision": args.roma_precision, "device": str(device),
        },
        "pair_records": pair_records,
        "cost_projection": {
            "observed_mean_seconds_per_pair": float(mean_seconds),
            "estimated": {str(count): {"pairs": count, "seconds": float(mean_seconds * count), "hours": float(mean_seconds * count / 3600.0)} for count in projection_counts},
        },
        "go_no_go": {
            "passed_interface": bool(all(row["scales"] and all(value["all_finite"] for value in row["scales"].values()) for row in pair_records)),
            "next_step_only_if": "actual estimated 512-pair frozen-feature extraction is within 6 A800 hours; then run a pre-registered small capacity probe, never full SPair training",
        },
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(
        f"RoMa relation feature gate: pairs={len(pair_records)}, "
        f"mean={mean_seconds:.2f}s/pair, interface_passed={output['go_no_go']['passed_interface']}, "
        f"estimate512={output['cost_projection']['estimated'].get('512', {}).get('hours', 'n/a')}h"
    )


if __name__ == "__main__":
    main()

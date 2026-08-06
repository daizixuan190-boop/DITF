"""Offline attention-top20 identity audit with frozen RoMa correspondence.

The input is an existing attention/DINO audit JSON.  FLUX is never loaded:
the exact block-28 attention candidate coordinates stored in that audit are
kept unchanged.  RoMa supplies a pair-conditioned bidirectional dense warp.
Candidates are ranked only by their forward/backward warp disagreement, with
no attention score, DiTF/DINO descriptor, geometry prior, or native fallback.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


TOPKS = (1, 3, 5, 10, 20)
METHOD_HYPOTHESIS = {
    "name": "Attention Top20 RoMa Pairwise Identity",
    "mechanism_hypothesis": (
        "Exact FLUX attention supplies semantic-region coverage but not candidate identity. "
        "A frozen correspondence model supplies new pair-conditioned evidence without "
        "collapsing the candidate axis into an A@V descriptor."
    ),
    "candidate_contract": "existing_exact_block28_mutual_attention_top20_coordinates_only",
    "identity_evidence": "frozen_RoMa_bidirectional_dense_warp",
    "candidate_score": "negative_mean_forward_backward_normalized_warp_error",
    "attention_used_as_identity_score": False,
    "ditf_descriptor_used": False,
    "dino_descriptor_used": False,
    "native_fallback": False,
    "gt_used_for_scoring": False,
    "method_training": False,
    "external_pretraining": "RoMa outdoor dense correspondence weights",
}

CONSENSUS_METHOD_HYPOTHESIS = {
    **METHOD_HYPOTHESIS,
    "name": "Attention Top20 RoMa Multi-View Consensus",
    "mechanism_hypothesis": (
        "Exact FLUX attention supplies semantic-region coverage. Frozen RoMa is run at "
        "multiple resolutions, and candidate ownership is assigned by rank consensus "
        "across bidirectional pair-conditioned warps rather than one warp or a fixed "
        "score fusion."
    ),
    "identity_evidence": "frozen_RoMa_multiresolution_bidirectional_rank_consensus",
    "candidate_score": "mean_candidate_rank_across_RoMa_resolutions",
}


def _as_unbatched_warp(
    warp: torch.Tensor,
    certainty: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if warp.ndim == 4:
        if int(warp.shape[0]) != 1:
            raise ValueError(f"expected one RoMa pair, got warp shape {tuple(warp.shape)}")
        warp = warp[0]
    if certainty.ndim == 3:
        if int(certainty.shape[0]) != 1:
            raise ValueError(
                f"expected one RoMa pair, got certainty shape {tuple(certainty.shape)}"
            )
        certainty = certainty[0]
    if warp.ndim != 3 or int(warp.shape[-1]) != 4:
        raise ValueError(f"expected RoMa warp [H,2W,4], got {tuple(warp.shape)}")
    if certainty.ndim != 2 or tuple(certainty.shape) != tuple(warp.shape[:2]):
        raise ValueError(
            "RoMa certainty must align with warp spatial dimensions; "
            f"got {tuple(certainty.shape)} and {tuple(warp.shape)}"
        )
    if int(warp.shape[1]) % 2:
        raise ValueError("symmetric RoMa warp width must contain equal A->B and B->A halves")
    return warp.float(), certainty.float()


def _normalize_points(
    points_xy: torch.Tensor,
    image_size: Sequence[int],
) -> torch.Tensor:
    height, width = int(image_size[0]), int(image_size[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"invalid image size {(height, width)}")
    points = points_xy.float()
    return torch.stack(
        (
            2.0 * points[..., 0] / float(width) - 1.0,
            2.0 * points[..., 1] / float(height) - 1.0,
        ),
        dim=-1,
    )


def _sample_field(field: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample an HxWxC field at normalized xy coordinates."""

    if field.ndim == 2:
        field = field.unsqueeze(-1)
    if field.ndim != 3:
        raise ValueError(f"field must be HxWxC, got {tuple(field.shape)}")
    original_shape = points.shape[:-1]
    grid = points.reshape(1, -1, 1, 2).to(field.device, field.dtype)
    sampled = F.grid_sample(
        field.permute(2, 0, 1).unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    return sampled[0, :, :, 0].transpose(0, 1).reshape(*original_shape, field.shape[-1])


def rank_attention_candidates_with_roma(
    source_points: torch.Tensor,
    candidate_points: torch.Tensor,
    source_size: Sequence[int],
    target_size: Sequence[int],
    warp: torch.Tensor,
    certainty: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Rank fixed candidates by frozen RoMa bidirectional correspondence error."""

    if source_points.ndim != 2 or int(source_points.shape[-1]) != 2:
        raise ValueError("source_points must be [P,2]")
    if candidate_points.ndim != 3 or int(candidate_points.shape[-1]) != 2:
        raise ValueError("candidate_points must be [P,K,2]")
    if int(candidate_points.shape[0]) != int(source_points.shape[0]):
        raise ValueError("source points and candidate rows must align")

    warp, certainty = _as_unbatched_warp(warp, certainty)
    half_width = int(warp.shape[1]) // 2
    forward_field = warp[:, :half_width, 2:]
    backward_field = warp[:, half_width:, :2]
    forward_certainty = certainty[:, :half_width]
    backward_certainty = certainty[:, half_width:]

    src_norm = _normalize_points(source_points.to(warp.device), source_size)
    candidate_norm = _normalize_points(candidate_points.to(warp.device), target_size)
    predicted_target = _sample_field(forward_field, src_norm)
    predicted_source = _sample_field(backward_field, candidate_norm)

    forward_error = torch.linalg.vector_norm(
        candidate_norm - predicted_target[:, None, :], dim=-1
    )
    backward_error = torch.linalg.vector_norm(
        predicted_source - src_norm[:, None, :], dim=-1
    )
    bidirectional_error = 0.5 * (forward_error + backward_error)
    source_certainty = _sample_field(forward_certainty, src_norm)[..., 0]
    candidate_certainty = _sample_field(backward_certainty, candidate_norm)[..., 0]
    mutual_certainty = torch.sqrt(
        (source_certainty[:, None].clamp_min(0.0) * candidate_certainty.clamp_min(0.0))
        .clamp_min(0.0)
    )

    # Certainty is deliberately audit-only: ranking measures whether RoMa's
    # pair-conditioned warp itself identifies the correct attention candidate.
    order = torch.argsort(bidirectional_error, dim=1, stable=True)
    return {
        "order": order,
        "score": -bidirectional_error,
        "bidirectional_error": bidirectional_error,
        "forward_error": forward_error,
        "backward_error": backward_error,
        "mutual_certainty": mutual_certainty,
        "source_certainty": source_certainty,
        "predicted_target_normalized": predicted_target,
    }


def rank_attention_candidates_with_roma_consensus(
    rankings: Sequence[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Aggregate fixed-candidate RoMa rankings without tuned score weights.

    Each view contributes only its candidate rank. Borda-style mean rank is
    invariant to the resolution-dependent scale of RoMa's warp error. The
    mean error and rank dispersion remain diagnostics and deterministic tie
    breakers, not separately tuned identity scores.
    """

    if not rankings:
        raise ValueError("at least one RoMa ranking is required")
    required = {"order", "bidirectional_error", "forward_error", "backward_error", "mutual_certainty"}
    missing = required.difference(rankings[0])
    if missing:
        raise ValueError(f"RoMa ranking is missing {sorted(missing)}")
    shape = tuple(rankings[0]["bidirectional_error"].shape)
    if len(shape) != 2:
        raise ValueError("RoMa candidate errors must have shape [P,K]")
    if any(tuple(row["bidirectional_error"].shape) != shape for row in rankings):
        raise ValueError("all RoMa views must rank the same candidate grid")

    errors = torch.stack([row["bidirectional_error"] for row in rankings], dim=0)
    forward = torch.stack([row["forward_error"] for row in rankings], dim=0)
    backward = torch.stack([row["backward_error"] for row in rankings], dim=0)
    certainty = torch.stack([row["mutual_certainty"] for row in rankings], dim=0)
    rank_positions = []
    for row in rankings:
        positions = torch.empty_like(row["order"], dtype=torch.long)
        rank_values = torch.arange(
            shape[1],
            device=positions.device,
            dtype=positions.dtype,
        )[None].expand_as(row["order"])
        positions.scatter_(1, row["order"], rank_values)
        rank_positions.append(positions)
    ranks = torch.stack(rank_positions, dim=0).float()
    mean_rank = ranks.mean(dim=0)
    mean_error = errors.mean(dim=0)
    rank_std = ranks.std(dim=0, unbiased=False)
    top1_votes = (ranks == 0).sum(dim=0)

    # Stable sorting preserves the original attention order for exact ties.
    order = torch.argsort(mean_rank, dim=1, stable=True)
    return {
        "order": order,
        "score": -mean_rank,
        "consensus_rank": mean_rank,
        "consensus_rank_std": rank_std,
        "consensus_top1_votes": top1_votes,
        "bidirectional_error": errors.mean(dim=0),
        "forward_error": forward.mean(dim=0),
        "backward_error": backward.mean(dim=0),
        "mutual_certainty": certainty.mean(dim=0),
        "source_certainty": torch.stack(
            [row["source_certainty"] for row in rankings], dim=0
        ).mean(dim=0),
        "predicted_target_normalized": torch.stack(
            [row["predicted_target_normalized"] for row in rankings], dim=0
        ).mean(dim=0),
    }


def _load_checkpoint(path: str | None, device: torch.device) -> Any:
    if not path:
        return None
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    return torch.load(checkpoint_path, map_location=device)


def _build_roma(args: argparse.Namespace, device: torch.device) -> Any:
    try:
        from romatch import roma_outdoor
    except ImportError as exc:
        missing = getattr(exc, "name", None) or "an unknown RoMa dependency"
        raise RuntimeError(
            f"RoMa runtime import failed because {missing!r} is unavailable. "
            "Install the missing dependency without replacing the DiTF torch build, "
            "then verify `from romatch import roma_outdoor` before evaluation. "
            "The pinned RoMa API itself is installed with: pip install --no-deps "
            "git+https://github.com/Parskatt/RoMa.git@905ff76a3d8ac589e85cb2f04124bf75ab3ce1b9"
        ) from exc

    torch.set_float32_matmul_precision("highest")
    precision = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[args.roma_precision]
    signature = inspect.signature(roma_outdoor)
    kwargs: dict[str, Any] = {
        "device": device,
        "weights": _load_checkpoint(args.roma_weights, device),
        "dinov2_weights": _load_checkpoint(args.roma_dinov2_weights, device),
        "coarse_res": int(args.roma_coarse_res),
        "upsample_res": int(args.roma_upsample_res),
        "amp_dtype": precision,
    }
    if "symmetric" in signature.parameters:
        kwargs["symmetric"] = True
    if "use_custom_corr" in signature.parameters:
        kwargs["use_custom_corr"] = False
    model = roma_outdoor(**kwargs)
    model.eval().requires_grad_(False)
    return model


def _set_roma_resolution(model: Any, coarse_res: int, upsample_res: int) -> None:
    """Reuse one frozen RoMa instance while changing its inference resolution."""

    for attribute in ("w_resized", "h_resized", "upsample_res"):
        if not hasattr(model, attribute):
            raise RuntimeError(
                "RoMa model does not expose the documented resolution controls: "
                f"missing {attribute}"
            )
    model.w_resized = int(coarse_res)
    model.h_resized = int(coarse_res)
    model.upsample_res = (int(upsample_res), int(upsample_res))


def _parse_resolution_specs(
    coarse_values: str | None,
    upsample_values: str | None,
) -> list[tuple[int, int]]:
    if not coarse_values and not upsample_values:
        return []
    if not coarse_values or not upsample_values:
        raise ValueError("both consensus coarse and upsample resolutions are required")
    coarse = [int(value.strip()) for value in coarse_values.split(",") if value.strip()]
    upsample = [int(value.strip()) for value in upsample_values.split(",") if value.strip()]
    if len(coarse) != len(upsample) or len(coarse) < 2:
        raise ValueError("consensus resolution lists must contain at least two paired values")
    if any(value <= 0 for pair in zip(coarse, upsample) for value in pair):
        raise ValueError("consensus resolutions must be positive")
    return list(zip(coarse, upsample))


def _run_roma_pair(
    model: Any,
    source_path: Path,
    target_path: Path,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    match_signature = inspect.signature(model.match)
    batched_default = match_signature.parameters["batched"].default
    kwargs: dict[str, Any] = {"device": device}
    if batched_default is True:
        kwargs["batched"] = True
    with torch.inference_mode():
        warp, certainty = model.match(str(source_path), str(target_path), **kwargs)
    return _as_unbatched_warp(warp, certainty)


def _validate_audit(payload: dict[str, Any]) -> list[dict[str, Any]]:
    records = payload.get("pair_records")
    if not isinstance(records, list) or not records:
        raise ValueError("attention audit must contain a non-empty pair_records list")
    for pair in records:
        for key in ("category", "src_image", "trg_image", "points"):
            if key not in pair:
                raise ValueError(f"attention audit pair is missing {key}")
        candidate_counts = set()
        for point in pair["points"]:
            candidates = point.get("candidates")
            if not isinstance(candidates, list) or not candidates:
                raise ValueError("every point must contain fixed attention candidates")
            if any("attention_rank" not in candidate for candidate in candidates):
                raise ValueError("candidate records must preserve attention_rank")
            candidate_counts.add(len(candidates))
        if len(candidate_counts) != 1:
            raise ValueError("all points in a pair must have the same candidate count")
    return records


def _summarize_points(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    points = [point for pair in records for point in pair["points"]]
    hard = [point for point in points if bool(point["both_wrong_top20_hit"])]

    def branch(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        ranks = [int(row["roma_gt_rank"]) for row in rows if row["roma_gt_rank"] is not None]
        return {
            "points": len(rows),
            "topk_hits": {
                str(k): int(sum(bool(row["roma_topk_hits"][str(k)]) for row in rows))
                for k in TOPKS
            },
            "topk_rates": {
                str(k): float(
                    sum(bool(row["roma_topk_hits"][str(k)]) for row in rows)
                    / max(1, len(rows))
                )
                for k in TOPKS
            },
            "gt_rank_mean": float(np.mean(ranks)) if ranks else None,
            "gt_rank_median": float(np.median(ranks)) if ranks else None,
            "uniform_top1_expectation": float(
                np.mean([row["uniform_candidate_hit_probability"] for row in rows])
            )
            if rows
            else None,
            "mean_selected_mutual_certainty": float(
                np.mean([row["selected_mutual_certainty"] for row in rows])
            )
            if rows
            else None,
        }

    return {
        "pairs": len(records),
        "points": len(points),
        "baseline_correct": int(sum(bool(row["baseline_pck_hit"]) for row in points)),
        "attention_top1_correct": int(
            sum(bool(row["attention_top1_pck_hit"]) for row in points)
        ),
        "method_correct": int(sum(bool(row["method_pck_hit"]) for row in points)),
        "candidate_missing_gt": int(
            sum(not bool(row["attention_top20_pck_hit"]) for row in points)
        ),
        "rescued_vs_baseline": int(sum(bool(row["rescued_vs_baseline"]) for row in points)),
        "harmed_vs_baseline": int(sum(bool(row["harmed_vs_baseline"]) for row in points)),
        "all_points": branch(points),
        "both_wrong_top20_hit": branch(hard),
    }


def _metric_block(pair_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    points = [point for pair in pair_rows for point in pair["points"]]
    baseline_pair = [np.mean([point["baseline_pck_hit"] for point in pair["points"]]) for pair in pair_rows]
    method_pair = [np.mean([point["method_pck_hit"] for point in pair["points"]]) for pair in pair_rows]
    baseline_correct = sum(bool(point["baseline_pck_hit"]) for point in points)
    method_correct = sum(bool(point["method_pck_hit"]) for point in points)
    return {
        "pairs": len(pair_rows),
        "points": len(points),
        "baseline_image": 100.0 * float(np.mean(baseline_pair)) if pair_rows else 0.0,
        "method_image": 100.0 * float(np.mean(method_pair)) if pair_rows else 0.0,
        "baseline_point": 100.0 * baseline_correct / max(1, len(points)),
        "method_point": 100.0 * method_correct / max(1, len(points)),
        "point_gain": 100.0 * (method_correct - baseline_correct) / max(1, len(points)),
        "improved_count": int(sum(bool(point["rescued_vs_baseline"]) for point in points)),
        "harmed_count": int(sum(bool(point["harmed_vs_baseline"]) for point in points)),
    }


def evaluate(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    audit_path = Path(args.attention_audit_json)
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    source_records = _validate_audit(payload)
    consensus_specs = _parse_resolution_specs(
        args.consensus_coarse_resolutions,
        args.consensus_upsample_resolutions,
    )
    method_hypothesis = (
        CONSENSUS_METHOD_HYPOTHESIS if consensus_specs else METHOD_HYPOTHESIS
    )
    matcher_name = (
        "attention_top20_roma_multiview_consensus"
        if consensus_specs
        else "attention_top20_roma_pairwise_identity"
    )
    model = _build_roma(args, device)
    selected_records: list[dict[str, Any]] = []
    category_counts: defaultdict[str, int] = defaultdict(int)

    for pair in tqdm(source_records, desc="RoMa pairwise identity"):
        category = str(pair["category"])
        if args.max_pairs_per_cat and category_counts[category] >= args.max_pairs_per_cat:
            continue
        category_counts[category] += 1
        source_path = Path(args.dataset_path) / "JPEGImages" / category / pair["src_image"]
        target_path = Path(args.dataset_path) / "JPEGImages" / category / pair["trg_image"]
        if not source_path.is_file() or not target_path.is_file():
            raise FileNotFoundError(f"missing SPair image: {source_path} or {target_path}")
        with Image.open(source_path) as source_image, Image.open(target_path) as target_image:
            source_size = (int(source_image.height), int(source_image.width))
            target_size = (int(target_image.height), int(target_image.width))

        source_points = torch.tensor(
            [point["source_point"] for point in pair["points"]],
            device=device,
            dtype=torch.float32,
        )
        attention_candidates = [
            sorted(point["candidates"], key=lambda row: int(row["attention_rank"]))
            for point in pair["points"]
        ]
        candidate_points = torch.tensor(
            [[candidate["pixel"] for candidate in rows] for rows in attention_candidates],
            device=device,
            dtype=torch.float32,
        )
        if consensus_specs:
            ranked_views = []
            for coarse_res, upsample_res in consensus_specs:
                _set_roma_resolution(model, coarse_res, upsample_res)
                warp, certainty = _run_roma_pair(model, source_path, target_path, device)
                ranked_views.append(
                    rank_attention_candidates_with_roma(
                        source_points,
                        candidate_points,
                        source_size,
                        target_size,
                        warp,
                        certainty,
                    )
                )
                del warp, certainty
            ranked = rank_attention_candidates_with_roma_consensus(ranked_views)
        else:
            warp, certainty = _run_roma_pair(model, source_path, target_path, device)
            ranked = rank_attention_candidates_with_roma(
                source_points,
                candidate_points,
                source_size,
                target_size,
                warp,
                certainty,
            )
        point_records = []
        for point_index, source_point in enumerate(pair["points"]):
            order = ranked["order"][point_index].detach().cpu().tolist()
            ranked_candidates = []
            hit_flags = []
            for roma_rank, candidate_index in enumerate(order, start=1):
                original = attention_candidates[point_index][candidate_index]
                pck_hit = bool(original["pck_hit"])
                hit_flags.append(pck_hit)
                ranked_candidates.append(
                    {
                        "roma_rank": int(roma_rank),
                        "attention_rank": int(original["attention_rank"]),
                        "pixel": [int(value) for value in original["pixel"]],
                        "pck_hit": pck_hit,
                        "roma_score": float(ranked["score"][point_index, candidate_index].cpu()),
                        "bidirectional_error": float(
                            ranked["bidirectional_error"][point_index, candidate_index].cpu()
                        ),
                        "forward_error": float(
                            ranked["forward_error"][point_index, candidate_index].cpu()
                        ),
                        "backward_error": float(
                            ranked["backward_error"][point_index, candidate_index].cpu()
                        ),
                        "mutual_certainty": float(
                            ranked["mutual_certainty"][point_index, candidate_index].cpu()
                        ),
                    }
                )
                if consensus_specs:
                    ranked_candidates[-1].update(
                        {
                            "consensus_rank": float(
                                ranked["consensus_rank"][point_index, candidate_index].cpu()
                            ),
                            "consensus_rank_std": float(
                                ranked["consensus_rank_std"][point_index, candidate_index].cpu()
                            ),
                            "consensus_top1_votes": int(
                                ranked["consensus_top1_votes"][point_index, candidate_index].cpu()
                            ),
                        }
                    )
            method_hit = bool(hit_flags[0])
            gt_rank = next((rank + 1 for rank, hit in enumerate(hit_flags) if hit), None)
            topk_hits = {
                str(k): bool(any(hit_flags[: min(k, len(hit_flags))])) for k in TOPKS
            }
            baseline_hit = bool(source_point["baseline_pck_hit"])
            point_records.append(
                {
                    "keypoint_index": int(source_point["keypoint_index"]),
                    "source_point": source_point["source_point"],
                    "target_point": source_point["target_point"],
                    "baseline_prediction": source_point.get("baseline_prediction"),
                    "method_prediction": ranked_candidates[0]["pixel"],
                    "baseline_pck_hit": baseline_hit,
                    "method_pck_hit": method_hit,
                    "attention_top1_pck_hit": bool(source_point["attention_top1_pck_hit"]),
                    "attention_top20_pck_hit": bool(source_point["attention_top20_pck_hit"]),
                    "both_wrong_top20_hit": bool(source_point["both_wrong_top20_hit"]),
                    "rescued_vs_baseline": bool(method_hit and not baseline_hit),
                    "harmed_vs_baseline": bool(baseline_hit and not method_hit),
                    "roma_gt_rank": gt_rank,
                    "roma_topk_hits": topk_hits,
                    "candidate_pck_hit_count": int(sum(hit_flags)),
                    "uniform_candidate_hit_probability": float(sum(hit_flags) / len(hit_flags)),
                    "source_certainty": float(ranked["source_certainty"][point_index].cpu()),
                    "selected_mutual_certainty": ranked_candidates[0]["mutual_certainty"],
                    "roma_predicted_target_normalized": [
                        float(value)
                        for value in ranked["predicted_target_normalized"][point_index].cpu()
                    ],
                    "candidates": ranked_candidates,
                }
            )
        selected_records.append(
            {
                "category": category,
                "pair_json": pair.get("pair_json"),
                "src_image": pair["src_image"],
                "trg_image": pair["trg_image"],
                "keypoint_count": len(point_records),
                "points": point_records,
            }
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    by_category: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in selected_records:
        by_category[pair["category"]].append(pair)
    categories = {category: _metric_block(rows) for category, rows in by_category.items()}
    all_metrics = _metric_block(selected_records)
    summary = _summarize_points(selected_records)
    protocol = {
        "router": "precomputed exact block28 mutual attention top20 coordinates",
        "router_audit_json": str(audit_path),
        "identity_model": "RoMa outdoor",
        "roma_coarse_res": int(args.roma_coarse_res),
        "roma_upsample_res": int(args.roma_upsample_res),
        "roma_precision": args.roma_precision,
        "ranking": (
            "mean Borda rank across frozen RoMa resolutions; errors and certainty are diagnostics"
            if consensus_specs
            else "bidirectional warp error only; certainty is audit-only"
        ),
        "consensus_resolutions": [list(pair) for pair in consensus_specs],
    }
    result = {
        "matcher": matcher_name,
        "method_hypothesis": method_hypothesis,
        "protocol": protocol,
        "categories": categories,
        "all": all_metrics,
        "mechanism_summary": summary,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    root = output_path.with_suffix("")
    suffix = (
        "attention_top20_roma_multiview_consensus"
        if consensus_specs
        else "attention_top20_roma_pairwise_identity"
    )
    audit_output = Path(f"{root}_{suffix}_audit.json")
    summary_output = Path(f"{root}_{suffix}_summary.json")
    audit_output.write_text(
        json.dumps(
            {
                "matcher": result["matcher"],
                "method_hypothesis": method_hypothesis,
                "protocol": protocol,
                "summary": summary,
                "pair_records": selected_records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    summary_output.write_text(
        json.dumps(
            {
                "matcher": result["matcher"],
                "method_hypothesis": method_hypothesis,
                "protocol": protocol,
                "summary": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    result["audit_path"] = str(audit_output)
    result["summary_path"] = str(summary_output)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    hard_summary = summary["both_wrong_top20_hit"]
    hard_random = hard_summary["uniform_top1_expectation"]
    hard_random_text = f"{100.0 * hard_random:.2f}" if hard_random is not None else "n/a"
    print(
        f"Matcher: {matcher_name}\n"
        f"Baseline All per image/point: {all_metrics['baseline_image']:.2f} / "
        f"{all_metrics['baseline_point']:.2f}\n"
        f"Method All per image/point: {all_metrics['method_image']:.2f} / "
        f"{all_metrics['method_point']:.2f}; point gain={all_metrics['point_gain']:.2f}\n"
        f"Hard both-wrong top1/random: "
        f"{100.0 * hard_summary['topk_rates']['1']:.2f} / {hard_random_text}"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--attention_audit_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--max_pairs_per_cat", type=int, default=0)
    parser.add_argument("--roma_coarse_res", type=int, default=560)
    parser.add_argument("--roma_upsample_res", type=int, default=864)
    parser.add_argument("--roma_precision", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--roma_weights", default=None)
    parser.add_argument("--roma_dinov2_weights", default=None)
    parser.add_argument(
        "--consensus_coarse_resolutions",
        default=None,
        help="comma-separated RoMa coarse resolutions; requires at least two values",
    )
    parser.add_argument(
        "--consensus_upsample_resolutions",
        default=None,
        help="comma-separated RoMa upsample resolutions paired with coarse resolutions",
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

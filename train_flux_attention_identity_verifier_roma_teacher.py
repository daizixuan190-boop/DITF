"""Distill pair-conditioned RoMa identity into the FLUX candidate verifier.

RoMa is used only while training to create soft candidate targets from its
bidirectional warp.  The saved verifier has no RoMa dependency at inference.
This is an explicit external-teacher diagnostic, not strict self-supervision.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from attention_identity_verifier import (
    CandidateIdentityVerifier,
    attention_prior_scores,
    checkpoint_payload,
    listwise_identity_loss,
    load_verifier_checkpoint,
    sample_replay_cell_centers,
    verifier_config_from_batch,
)
from eval_spair_attention_top20_roma_identity import (
    _build_roma,
    _run_roma_pair,
    rank_attention_candidates_with_roma,
)
from train_flux_attention_identity_verifier import (
    _batch_to_device,
    _candidate_batch,
    build_strict_training_pair_manifest,
)


def roma_soft_targets(
    bidirectional_error: torch.Tensor,
    mutual_certainty: torch.Tensor,
    *,
    temperature: float,
    minimum_probability: float,
    minimum_certainty: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Convert RoMa candidate errors into soft labels and a label-confidence mask.

    Error determines the ranking. Certainty is used only to reject unreliable
    rows, so this does not silently turn RoMa confidence into a tuned score.
    """

    if bidirectional_error.ndim != 2 or mutual_certainty.shape != bidirectional_error.shape:
        raise ValueError("RoMa teacher tensors must both be [query,candidate]")
    if float(temperature) <= 0.0 or not 0.0 <= float(minimum_probability) <= 1.0:
        raise ValueError("invalid RoMa teacher temperature/probability")
    if float(minimum_certainty) < 0.0:
        raise ValueError("minimum certainty must be non-negative")
    error = torch.nan_to_num(bidirectional_error.float(), nan=1e6, posinf=1e6, neginf=1e6)
    certainty = torch.nan_to_num(mutual_certainty.float(), nan=0.0, posinf=0.0, neginf=0.0)
    scale = error.amax(dim=1, keepdim=True).sub(error.amin(dim=1, keepdim=True)).clamp_min(1e-6)
    probabilities = torch.softmax(-(error - error.amin(dim=1, keepdim=True)) / scale / float(temperature), dim=1)
    values, ranks = probabilities.topk(k=min(2, int(probabilities.shape[1])), dim=1)
    top_probability = values[:, 0]
    top_certainty = certainty.gather(1, ranks[:, :1]).squeeze(1)
    confident = top_probability.ge(float(minimum_probability)) & top_certainty.ge(float(minimum_certainty))
    diagnostics = {
        "teacher_rank": ranks[:, 0].detach().cpu(),
        "teacher_probability": top_probability.detach().cpu(),
        "teacher_certainty": top_certainty.detach().cpu(),
        "teacher_margin": (values[:, 0] - values[:, 1]).detach().cpu() if values.shape[1] > 1 else top_probability.detach().cpu(),
    }
    return probabilities.detach().cpu(), confident.detach().cpu(), diagnostics


def _reset_residual_head(model: CandidateIdentityVerifier) -> None:
    nn.init.zeros_(model.residual_head.weight)
    nn.init.zeros_(model.residual_head.bias)


def _direction_loss(
    model: CandidateIdentityVerifier,
    batch: Mapping[str, Any],
    source_points: torch.Tensor,
    source_size: Sequence[int],
    target_size: Sequence[int],
    warp: torch.Tensor,
    certainty: torch.Tensor,
    *,
    temperature: float,
    minimum_probability: float,
    minimum_certainty: float,
    retention_weight: float,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    from eval_spair_attention_top20_roma_identity import _as_unbatched_warp

    warp, certainty = _as_unbatched_warp(warp, certainty)
    candidates = batch["candidate_pixels"]
    ranking = rank_attention_candidates_with_roma(
        source_points,
        candidates,
        source_size,
        target_size,
        warp,
        certainty,
    )
    targets, confident, teacher = roma_soft_targets(
        ranking["bidirectional_error"],
        ranking["mutual_certainty"],
        temperature=float(temperature),
        minimum_probability=float(minimum_probability),
        minimum_certainty=float(minimum_certainty),
    )
    groups, attention = _batch_to_device(batch, device, model.config.feature_groups)
    scores = model(groups, attention)
    confident_device = confident.to(device)
    loss = listwise_identity_loss(scores, targets.to(device), confident_device)
    prior = attention_prior_scores(attention, weight=float(model.config.attention_prior_weight))
    residual = scores - prior
    unconfirmed = ~confident_device
    retention = residual[unconfirmed].square().mean() if bool(unconfirmed.any()) else residual.sum() * 0.0
    loss = loss + float(retention_weight) * retention
    selected = scores.detach().argmax(dim=1).cpu()
    teacher_rank = teacher["teacher_rank"]
    return loss, {
        "queries": float(confident.numel()),
        "confident": float(confident.sum()),
        "attention_teacher_agreement": float((teacher_rank == 0).logical_and(confident).sum()),
        "model_teacher_agreement": float((selected == teacher_rank).logical_and(confident).sum()),
        "teacher_changes_attention": float((teacher_rank != 0).logical_and(confident).sum()),
        "retention_loss": float(retention.detach().cpu()),
    }


def _merge(total: dict[str, float], update: Mapping[str, float]) -> None:
    for key, value in update.items():
        total[key] = total.get(key, 0.0) + float(value)


def train(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    from eval_spair_matcher_ablation import (
        _extract_flux_fjsar_entry,
        _load_flux_fjsar_runtime,
        _make_fjsar_capture,
        _prepare_feature_tensors,
    )

    if tuple(map(int, args.k)) != (28,) or int(args.t) != 260 or int(args.ensemble_size) != 8:
        raise ValueError("locked verifier protocol requires --k 28 --t 260 --ensemble_size 8")
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    pairs, manifest_metadata = build_strict_training_pair_manifest(
        args.dataset_path, seed=int(args.seed), max_pairs=int(args.max_train_pairs)
    )
    model, base_checkpoint = load_verifier_checkpoint(args.base_checkpoint, map_location="cpu")
    if bool(args.reset_residual_head):
        _reset_residual_head(model)
    model = model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    categories = sorted({row.category for row in pairs})
    args.device = str(device)
    featurizer, flux_model, blocks = _load_flux_fjsar_runtime(args, categories)
    capture = _make_fjsar_capture(args, flux_model)
    roma = _build_roma(args, device)
    pre_norm = nn.LayerNorm(3072, elementwise_affine=False, eps=1e-6).to(device)
    generator = torch.Generator().manual_seed(int(args.seed))
    counts: dict[str, float] = {}
    losses: list[float] = []
    dimensions_checked = False
    try:
        for pair in tqdm(pairs, desc="train RoMa teacher distillation"):
            source_entry = _extract_flux_fjsar_entry(args.dataset_path, pair.category, pair.source_name, args.training_caption, args, featurizer, capture)
            target_entry = _extract_flux_fjsar_entry(args.dataset_path, pair.category, pair.target_name, args.training_caption, args, featurizer, capture)
            source_features = _prepare_feature_tensors(source_entry["feature"], source_entry["ada"], args, pre_norm, device)
            target_features = _prepare_feature_tensors(target_entry["feature"], target_entry["ada"], args, pre_norm, device)
            source_grid = tuple(map(int, source_entry["feature"].shape[-2:]))
            target_grid = tuple(map(int, target_entry["feature"].shape[-2:]))
            source_size = (source_grid[0] * 16, source_grid[1] * 16)
            target_size = (target_grid[0] * 16, target_grid[1] * 16)
            source_points = sample_replay_cell_centers(source_size, source_grid, count=int(args.queries_per_image), generator=generator, border_cells=int(args.border_cells))
            target_points = sample_replay_cell_centers(target_size, target_grid, count=int(args.queries_per_image), generator=generator, border_cells=int(args.border_cells))
            forward_batch = _candidate_batch(source_entry, target_entry, source_features, target_features, source_points, source_size, blocks, args.candidate_topk, target_size=target_size)
            reverse_batch = _candidate_batch(target_entry, source_entry, target_features, source_features, target_points, target_size, blocks, args.candidate_topk, target_size=source_size)
            if not dimensions_checked:
                observed = verifier_config_from_batch(forward_batch, feature_groups=model.config.feature_groups, group_width=model.config.group_width, hidden_width=model.config.hidden_width, dropout=model.config.dropout, attention_prior_weight=model.config.attention_prior_weight, global_query_context=model.config.global_query_context)
                if observed.feature_dims != model.config.feature_dims:
                    raise ValueError("base checkpoint feature dimensions do not match current candidate batch")
                dimensions_checked = True
            source_path = Path(args.dataset_path) / "JPEGImages" / pair.category / pair.source_name
            target_path = Path(args.dataset_path) / "JPEGImages" / pair.category / pair.target_name
            forward_warp, forward_certainty = _run_roma_pair(roma, source_path, target_path, device)
            reverse_warp, reverse_certainty = _run_roma_pair(roma, target_path, source_path, device)
            optimizer.zero_grad(set_to_none=True)
            forward_loss, forward_counts = _direction_loss(model, forward_batch, source_points, source_size, target_size, forward_warp, forward_certainty, temperature=args.teacher_temperature, minimum_probability=args.minimum_teacher_probability, minimum_certainty=args.minimum_certainty, retention_weight=args.retention_weight, device=device)
            reverse_loss, reverse_counts = _direction_loss(model, reverse_batch, target_points, target_size, source_size, reverse_warp, reverse_certainty, temperature=args.teacher_temperature, minimum_probability=args.minimum_teacher_probability, minimum_certainty=args.minimum_certainty, retention_weight=args.retention_weight, device=device)
            loss = 0.5 * (forward_loss + reverse_loss)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"non-finite RoMa distillation loss for {pair.pair_name}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            _merge(counts, forward_counts)
            _merge(counts, reverse_counts)
            del source_entry, target_entry, source_features, target_features, forward_batch, reverse_batch, loss
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        capture.close()
        del roma
    queries = max(1.0, counts.get("queries", 0.0))
    confident = max(1.0, counts.get("confident", 0.0))
    summary = {
        "optimization_steps": len(losses),
        "mean_loss": float(sum(losses) / max(1, len(losses))),
        "queries": int(counts.get("queries", 0.0)),
        "confident_pseudo_labels": int(counts.get("confident", 0.0)),
        "pseudo_label_coverage": float(counts.get("confident", 0.0) / queries),
        "attention_teacher_agreement": float(counts.get("attention_teacher_agreement", 0.0) / confident),
        "model_teacher_agreement": float(counts.get("model_teacher_agreement", 0.0) / confident),
        "teacher_change_fraction": float(counts.get("teacher_changes_attention", 0.0) / confident),
        "mean_retention_loss": float(counts.get("retention_loss", 0.0) / max(1, len(losses) * 2)),
    }
    metadata = {
        "supervision": "external_matcher_teacher_distillation",
        "strict_self_supervised": False,
        "teacher": "frozen_RoMa_bidirectional_warp_soft_candidate_targets",
        "teacher_used_for_inference": False,
        "roma_used": True,
        "spair_keypoints_used": False,
        "spair_bounding_boxes_used": False,
        "segmentation_masks_used": False,
        "pose_labels_used": False,
        "category_labels_used_for_targets": False,
        "training_pair_membership_used": True,
        "base_checkpoint": str(args.base_checkpoint),
        "base_checkpoint_sha256": hashlib.sha256(Path(args.base_checkpoint).read_bytes()).hexdigest(),
        "manifest": manifest_metadata,
        "protocol": {
            "candidate_topk": int(args.candidate_topk),
            "queries_per_image": int(args.queries_per_image),
            "teacher_temperature": float(args.teacher_temperature),
            "minimum_teacher_probability": float(args.minimum_teacher_probability),
            "minimum_certainty": float(args.minimum_certainty),
            "bidirectional": True,
            "reset_residual_head": bool(args.reset_residual_head),
        },
        "summary": summary,
    }
    output = Path(args.output_checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(model.cpu(), training_metadata=metadata), output)
    summary_path = Path(args.output_summary or output.with_suffix(".json"))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({"checkpoint": str(output), "training_metadata": metadata}, indent=2), encoding="utf-8")
    print("RoMa teacher distillation complete: " + f"coverage={100.0 * summary['pseudo_label_coverage']:.2f}, " + f"attention/teacher={100.0 * summary['attention_teacher_agreement']:.2f}, " + f"model/teacher={100.0 * summary['model_teacher_agreement']:.2f}")
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--base_checkpoint", required=True)
    parser.add_argument("--output_checkpoint", required=True)
    parser.add_argument("--output_summary", default="")
    parser.add_argument("--training_caption", default="a photo")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--img_size", nargs="+", type=int, default=[640, 640])
    parser.add_argument("--t", type=int, default=260)
    parser.add_argument("--k", nargs="+", type=int, default=[28])
    parser.add_argument("--ensemble_size", type=int, default=8)
    parser.add_argument("--cd", action="store_true")
    parser.add_argument("--fjsar_shared_noise", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--max_train_pairs", type=int, default=16)
    parser.add_argument("--queries_per_image", type=int, default=32)
    parser.add_argument("--border_cells", type=int, default=1)
    parser.add_argument("--candidate_topk", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--retention_weight", type=float, default=0.25)
    parser.add_argument("--teacher_temperature", type=float, default=0.25)
    parser.add_argument("--minimum_teacher_probability", type=float, default=0.25)
    parser.add_argument("--minimum_certainty", type=float, default=0.0)
    parser.add_argument("--reset_residual_head", action="store_true")
    parser.add_argument("--roma_coarse_res", type=int, default=560)
    parser.add_argument("--roma_upsample_res", type=int, default=864)
    parser.add_argument("--roma_precision", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--roma_weights", required=True)
    parser.add_argument("--roma_dinov2_weights", required=True)
    parser.set_defaults(
        matcher="attention_identity_verifier_roma_teacher_distillation",
        fjsar_disk_cache_path="",
        fjsar_require_disk_cache=False,
        fjsar_multilayer_identity_audit=False,
        fjsar_multilayer_blocks=(),
        fjsar_trajectory_blocks=(),
        fjsar_multi_timestep_attention_identity_audit=False,
        extract_native_in_memory=True,
    )
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    run_device = torch.device(parsed.device if parsed.device == "cpu" or torch.cuda.is_available() else "cpu")
    if run_device.type == "cpu" and parsed.roma_precision != "fp32":
        parsed.roma_precision = "fp32"
    train(parsed, run_device)

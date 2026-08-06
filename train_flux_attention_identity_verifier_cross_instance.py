"""Stage-2 cross-instance self-training for the FLUX candidate verifier.

No SPair keypoint, box, mask, pose, or category target is read. Candidate
labels are generated from frozen native-FLUX descriptor agreement and a
forward/backward cycle check inside fixed mutual-attention top-20 proposals.
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
    native_cycle_pseudo_targets,
    sample_replay_cell_centers,
    verifier_config_from_batch,
)
from train_flux_attention_identity_verifier import (
    _batch_to_device,
    _candidate_batch,
    build_strict_training_pair_manifest,
)


def _reset_residual_head(model: CandidateIdentityVerifier) -> None:
    nn.init.zeros_(model.residual_head.weight)
    nn.init.zeros_(model.residual_head.bias)


def _direction_loss(
    model: CandidateIdentityVerifier,
    batch: Mapping[str, Any],
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    *,
    cycle_radius_cells: float,
    minimum_native_margin: float,
    retention_weight: float,
    residual_l2_weight: float,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    targets, confident, teacher = native_cycle_pseudo_targets(
        batch,
        source_features,
        target_features,
        cycle_radius_cells=float(cycle_radius_cells),
        minimum_native_margin=float(minimum_native_margin),
    )
    groups, attention = _batch_to_device(batch, device, model.config.feature_groups)
    scores = model(groups, attention)
    confident_device = confident.to(device)
    pseudo_loss = listwise_identity_loss(scores, targets.to(device), confident_device)
    prior = attention_prior_scores(
        attention,
        weight=float(model.config.attention_prior_weight),
    )
    residual = scores - prior
    unconfirmed = ~confident_device
    retention_loss = (
        residual[unconfirmed].square().mean()
        if bool(unconfirmed.any())
        else residual.sum() * 0.0
    )
    residual_l2 = residual.square().mean()
    loss = (
        pseudo_loss
        + float(retention_weight) * retention_loss
        + float(residual_l2_weight) * residual_l2
    )
    teacher_rank = teacher["teacher_rank"]
    selected_rank = scores.detach().argmax(dim=1).cpu()
    confirmed_count = int(confident.sum())
    metrics = {
        "queries": float(confident.numel()),
        "confident": float(confirmed_count),
        "attention_teacher_agreement": float(
            (teacher["attention_teacher_agreement"] & confident).sum()
        ),
        "model_teacher_agreement": float(
            ((selected_rank == teacher_rank) & confident).sum()
        ),
        "teacher_changes_attention": float(
            ((teacher_rank != 0) & confident).sum()
        ),
        "cycle_distance_sum": float(teacher["cycle_distance_cells"].sum()),
        "native_margin_sum": float(teacher["native_margin"].sum()),
        "pseudo_loss": float(pseudo_loss.detach().cpu()),
        "retention_loss": float(retention_loss.detach().cpu()),
    }
    return loss, metrics


def _merge_counts(total: dict[str, float], update: Mapping[str, float]) -> None:
    for key, value in update.items():
        total[key] = total.get(key, 0.0) + float(value)


def _summary(
    counts: Mapping[str, float],
    losses: Sequence[float],
) -> dict[str, Any]:
    queries = max(1.0, counts.get("queries", 0.0))
    confident = max(1.0, counts.get("confident", 0.0))
    directions = max(1, len(losses) * 2)
    return {
        "optimization_steps": len(losses),
        "mean_total_loss": float(sum(losses) / max(1, len(losses))),
        "queries": int(counts.get("queries", 0.0)),
        "confident_pseudo_labels": int(counts.get("confident", 0.0)),
        "pseudo_label_coverage": float(counts.get("confident", 0.0) / queries),
        "attention_teacher_agreement_on_confident": float(
            counts.get("attention_teacher_agreement", 0.0) / confident
        ),
        "model_teacher_agreement_on_confident": float(
            counts.get("model_teacher_agreement", 0.0) / confident
        ),
        "teacher_change_fraction_on_confident": float(
            counts.get("teacher_changes_attention", 0.0) / confident
        ),
        "mean_cycle_distance_cells": float(
            counts.get("cycle_distance_sum", 0.0) / queries
        ),
        "mean_native_margin": float(counts.get("native_margin_sum", 0.0) / queries),
        "mean_pseudo_loss_per_direction": float(
            counts.get("pseudo_loss", 0.0) / directions
        ),
        "mean_retention_loss_per_direction": float(
            counts.get("retention_loss", 0.0) / directions
        ),
    }


def train(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    from eval_spair_matcher_ablation import (
        _extract_flux_fjsar_entry,
        _load_flux_fjsar_runtime,
        _make_fjsar_capture,
        _prepare_feature_tensors,
    )

    if tuple(map(int, args.k)) != (28,) or int(args.t) != 260 or int(args.ensemble_size) != 8:
        raise ValueError("locked verifier protocol requires --k 28 --t 260 --ensemble_size 8")
    if not bool(args.fjsar_shared_noise):
        raise ValueError("cross-instance cycle training requires --fjsar_shared_noise")
    if float(args.cycle_radius_cells) < 0.0 or float(args.minimum_native_margin) < 0.0:
        raise ValueError("cycle radius and native margin must be non-negative")

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    pairs, manifest_metadata = build_strict_training_pair_manifest(
        args.dataset_path,
        seed=int(args.seed),
        max_pairs=int(args.max_train_pairs),
    )
    model, base_checkpoint = load_verifier_checkpoint(
        args.base_checkpoint,
        map_location="cpu",
    )
    base_metadata = base_checkpoint.get("training_metadata", {})
    training_caption = str(args.training_caption).strip()
    if not training_caption:
        raise ValueError("--training_caption must be non-empty")
    if base_metadata.get("training_caption") not in (None, training_caption):
        raise ValueError("stage-1 and stage-2 training captions differ")
    _reset_residual_head(model)
    model = model.to(device).train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )

    categories = sorted({row.category for row in pairs})
    args.device = str(device)
    featurizer, flux_model, blocks = _load_flux_fjsar_runtime(args, categories)
    capture = _make_fjsar_capture(args, flux_model)
    pre_norm = nn.LayerNorm(3072, elementwise_affine=False, eps=1e-6).to(device)
    generator = torch.Generator().manual_seed(int(args.seed))
    counts: dict[str, float] = {}
    losses: list[float] = []
    dimensions_checked = False

    try:
        for pair in tqdm(pairs, desc="train cross-instance cycle verifier"):
            source_entry = _extract_flux_fjsar_entry(
                args.dataset_path,
                pair.category,
                pair.source_name,
                training_caption,
                args,
                featurizer,
                capture,
            )
            target_entry = _extract_flux_fjsar_entry(
                args.dataset_path,
                pair.category,
                pair.target_name,
                training_caption,
                args,
                featurizer,
                capture,
            )
            source_features = _prepare_feature_tensors(
                source_entry["feature"], source_entry["ada"], args, pre_norm, device
            )
            target_features = _prepare_feature_tensors(
                target_entry["feature"], target_entry["ada"], args, pre_norm, device
            )
            source_grid = tuple(map(int, source_entry["feature"].shape[-2:]))
            target_grid = tuple(map(int, target_entry["feature"].shape[-2:]))
            source_size = (source_grid[0] * 16, source_grid[1] * 16)
            target_size = (target_grid[0] * 16, target_grid[1] * 16)
            source_points = sample_replay_cell_centers(
                source_size,
                source_grid,
                count=int(args.queries_per_image),
                generator=generator,
                border_cells=int(args.border_cells),
            )
            target_points = sample_replay_cell_centers(
                target_size,
                target_grid,
                count=int(args.queries_per_image),
                generator=generator,
                border_cells=int(args.border_cells),
            )
            forward_batch = _candidate_batch(
                source_entry,
                target_entry,
                source_features,
                target_features,
                source_points,
                source_size,
                blocks,
                args.candidate_topk,
                target_size=target_size,
            )
            reverse_batch = _candidate_batch(
                target_entry,
                source_entry,
                target_features,
                source_features,
                target_points,
                target_size,
                blocks,
                args.candidate_topk,
                target_size=source_size,
            )
            if not dimensions_checked:
                observed = verifier_config_from_batch(
                    forward_batch,
                    feature_groups=model.config.feature_groups,
                    group_width=model.config.group_width,
                    hidden_width=model.config.hidden_width,
                    dropout=model.config.dropout,
                    attention_prior_weight=model.config.attention_prior_weight,
                )
                if observed.feature_dims != model.config.feature_dims:
                    raise ValueError("stage-1 checkpoint feature dimensions do not match stage-2")
                dimensions_checked = True

            optimizer.zero_grad(set_to_none=True)
            directional_losses = []
            for batch, src_features, trg_features in (
                (forward_batch, source_features, target_features),
                (reverse_batch, target_features, source_features),
            ):
                direction_loss, direction_counts = _direction_loss(
                    model,
                    batch,
                    src_features,
                    trg_features,
                    cycle_radius_cells=float(args.cycle_radius_cells),
                    minimum_native_margin=float(args.minimum_native_margin),
                    retention_weight=float(args.retention_weight),
                    residual_l2_weight=float(args.residual_l2_weight),
                    device=device,
                )
                directional_losses.append(direction_loss)
                _merge_counts(counts, direction_counts)
            loss = torch.stack(directional_losses).mean()
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"non-finite stage-2 loss for {pair.pair_name}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

            del (
                source_entry,
                target_entry,
                source_features,
                target_features,
                forward_batch,
                reverse_batch,
                loss,
            )
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        capture.close()

    summary = _summary(counts, losses)
    metadata = {
        "supervision": "unlabeled_cross_instance_native_cycle_self_distillation",
        "spair_keypoints_used": False,
        "spair_bounding_boxes_used": False,
        "segmentation_masks_used": False,
        "pose_labels_used": False,
        "category_labels_used_for_targets": False,
        "category_names_used_for_file_routing_only": True,
        "caption_labels_used": False,
        "training_pair_membership_used": True,
        "teacher": "frozen_native_flux_candidate_cosine_with_reverse_cycle",
        "teacher_used_for_inference": False,
        "external_matcher_used": False,
        "dino_used": False,
        "roma_used": False,
        "persistent_feature_cache_written": False,
        "base_checkpoint": str(args.base_checkpoint),
        "base_checkpoint_sha256": hashlib.sha256(
            Path(args.base_checkpoint).read_bytes()
        ).hexdigest(),
        "training_caption_policy": "fixed_neutral_prompt_for_all_images",
        "training_caption": training_caption,
        "manifest": manifest_metadata,
        "protocol": {
            "image_size": [int(value) for value in args.img_size],
            "feature_block": int(args.k[0]),
            "timestep": int(args.t),
            "ensemble_size": int(args.ensemble_size),
            "channel_discard": bool(args.cd),
            "shared_noise": bool(args.fjsar_shared_noise),
            "candidate_topk": int(args.candidate_topk),
            "queries_per_image": int(args.queries_per_image),
            "bidirectional": True,
            "cycle_radius_cells": float(args.cycle_radius_cells),
            "minimum_native_margin": float(args.minimum_native_margin),
            "retention_weight": float(args.retention_weight),
            "residual_l2_weight": float(args.residual_l2_weight),
            "residual_head_reset_before_stage2": True,
        },
        "summary": summary,
    }
    output = Path(args.output_checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(model.cpu(), training_metadata=metadata), output)
    summary_path = Path(args.output_summary or output.with_suffix(".json"))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps({"checkpoint": str(output), "training_metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    print(
        "Cross-instance training complete: "
        f"coverage={100.0 * summary['pseudo_label_coverage']:.2f}, "
        f"teacher changes={100.0 * summary['teacher_change_fraction_on_confident']:.2f}, "
        f"attention/teacher={100.0 * summary['attention_teacher_agreement_on_confident']:.2f}, "
        f"model/teacher={100.0 * summary['model_teacher_agreement_on_confident']:.2f}, "
        f"retention loss={summary['mean_retention_loss_per_direction']:.4f}"
    )
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
    parser.add_argument("--cd", action="store_true", default=False)
    parser.add_argument("--fjsar_shared_noise", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--max_train_pairs", type=int, default=400)
    parser.add_argument("--queries_per_image", type=int, default=64)
    parser.add_argument("--border_cells", type=int, default=1)
    parser.add_argument("--candidate_topk", type=int, default=20)
    parser.add_argument("--cycle_radius_cells", type=float, default=1.0)
    parser.add_argument("--minimum_native_margin", type=float, default=0.01)
    parser.add_argument("--retention_weight", type=float, default=0.25)
    parser.add_argument("--residual_l2_weight", type=float, default=0.01)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.set_defaults(
        matcher="attention_identity_verifier_cross_instance_training",
        fjsar_disk_cache_path="",
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
    train(parsed, run_device)

"""Stage-3 three-image attention-cycle training for the FLUX verifier.

Training triplets are unlabeled two-edge paths A-B-C from the SPair training
pair graph.  For an A->B query, a candidate in B is confirmed only when the
frozen B->C->A mutual-attention path returns uniquely to the source cell.  No
keypoint, box, mask, pose, category target, external matcher, or test image is
used.  At inference the third image and triangle teacher are absent.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from attention_identity_verifier import (
    CandidateIdentityVerifier,
    checkpoint_payload,
    listwise_identity_loss,
    load_verifier_checkpoint,
    sample_replay_cell_centers,
    triangle_cycle_pseudo_targets,
    verifier_config_from_batch,
)
from flux_joint_replay import FluxReplayState, run_flux_joint_stack
from train_flux_attention_identity_verifier import (
    _batch_to_device,
    _candidate_batch,
    build_strict_training_pair_manifest,
)


@dataclass(frozen=True)
class TrainingTriplet:
    category: str
    source_name: str
    target_name: str
    bridge_name: str
    source_target_pair: str
    target_bridge_pair: str


def build_strict_training_triplet_manifest(
    dataset_path: str | Path,
    *,
    seed: int,
    max_triplets: int,
) -> tuple[list[TrainingTriplet], dict[str, Any]]:
    """Build deterministic A-B-C paths from name-only, test-excluded pairs."""

    pairs, pair_metadata = build_strict_training_pair_manifest(
        dataset_path,
        seed=int(seed),
        max_pairs=0,
    )
    adjacency: dict[tuple[str, str], dict[str, str]] = {}
    for pair in pairs:
        adjacency.setdefault((pair.category, pair.source_name), {})[
            pair.target_name
        ] = pair.pair_name
        adjacency.setdefault((pair.category, pair.target_name), {})[
            pair.source_name
        ] = pair.pair_name

    triplets: list[TrainingTriplet] = []
    for (category, target_name), neighbor_pairs in sorted(adjacency.items()):
        neighbors = sorted(neighbor_pairs)
        for source_name in neighbors:
            for bridge_name in neighbors:
                if source_name == bridge_name:
                    continue
                triplets.append(TrainingTriplet(
                    category=category,
                    source_name=source_name,
                    target_name=target_name,
                    bridge_name=bridge_name,
                    source_target_pair=neighbor_pairs[source_name],
                    target_bridge_pair=neighbor_pairs[bridge_name],
                ))
    triplets.sort(key=lambda row: hashlib.sha256(
        (
            f"{int(seed)}:{row.category}:{row.source_name}:"
            f"{row.target_name}:{row.bridge_name}"
        ).encode("utf-8")
    ).digest())
    available_triplets = len(triplets)
    if int(max_triplets) > 0:
        triplets = triplets[: int(max_triplets)]
    if not triplets:
        raise RuntimeError("strict SPair training graph contains no two-edge triplets")
    metadata = {
        "split_source": "PairAnnotation/trn image-pair names only",
        "test_exclusion_source": "PairAnnotation/test image names only",
        "training_triplets": len(triplets),
        "available_triplets": int(available_triplets),
        "training_pair_membership_used": True,
        "triplet_labels_used": False,
        "keypoint_fields_read": False,
        "category_used_for_targets": False,
        "annotation_fields_read": ["src_imname", "trg_imname"],
        "selection_seed": int(seed),
        "max_triplets": int(max_triplets),
        "pair_manifest_sha256": pair_metadata["selected_manifest_sha256"],
        "selected_manifest_sha256": hashlib.sha256(
            "\n".join(
                f"{row.category}/{row.source_name}/{row.target_name}/{row.bridge_name}"
                for row in triplets
            ).encode("utf-8")
        ).hexdigest(),
    }
    return triplets, metadata


def _mutual_attention(
    source_entry: Mapping[str, Any],
    target_entry: Mapping[str, Any],
    blocks: Sequence[Any],
) -> torch.Tensor:
    source_state = FluxReplayState.from_dict(dict(source_entry["replay_state"]))
    target_state = FluxReplayState.from_dict(dict(target_entry["replay_state"]))
    with torch.no_grad():
        source_joint, target_joint, attention = run_flux_joint_stack(
            blocks,
            source_state,
            target_state,
            mode="exact",
            use_coordinate_bias=False,
        )
        mutual = torch.sqrt(
            (attention["p_ab"].float() * attention["p_ba"].float().t()).clamp_min(0.0)
        )
        mutual = torch.nan_to_num(mutual, nan=0.0, posinf=0.0, neginf=0.0)
        del source_joint, target_joint, attention
    return mutual


def _direction_loss(
    model: CandidateIdentityVerifier,
    frozen_base: CandidateIdentityVerifier,
    batch: Mapping[str, Any],
    mutual_target_bridge: torch.Tensor,
    mutual_bridge_source: torch.Tensor,
    source_grid: Sequence[int],
    *,
    triangle_cycle_radius_cells: float,
    require_unique_best: bool,
    retention_weight: float,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    targets, confident, teacher = triangle_cycle_pseudo_targets(
        batch,
        mutual_target_bridge,
        mutual_bridge_source,
        source_grid_size=source_grid,
        cycle_radius_cells=float(triangle_cycle_radius_cells),
        require_unique_best=bool(require_unique_best),
    )
    groups, attention = _batch_to_device(batch, device, model.config.feature_groups)
    scores = model(groups, attention)
    with torch.no_grad():
        base_scores = frozen_base(groups, attention)
    confident_device = confident.to(device)
    pseudo_loss = listwise_identity_loss(scores, targets.to(device), confident_device)
    unconfirmed = ~confident_device
    retention_loss = (
        F.kl_div(
            F.log_softmax(scores[unconfirmed], dim=1),
            F.softmax(base_scores[unconfirmed], dim=1),
            reduction="batchmean",
        )
        if bool(unconfirmed.any())
        else scores.sum() * 0.0
    )
    loss = pseudo_loss + float(retention_weight) * retention_loss
    teacher_rank = teacher["teacher_rank"]
    selected_rank = scores.detach().argmax(dim=1).cpu()
    base_rank = base_scores.detach().argmax(dim=1).cpu()
    return loss, {
        "queries": float(confident.numel()),
        "confident": float(confident.sum()),
        "unique_best": float(teacher["unique_best"].sum()),
        "attention_teacher_agreement": float(
            (teacher["attention_teacher_agreement"] & confident).sum()
        ),
        "base_teacher_agreement": float(((base_rank == teacher_rank) & confident).sum()),
        "model_teacher_agreement": float(
            ((selected_rank == teacher_rank) & confident).sum()
        ),
        "teacher_changes_base": float(((teacher_rank != base_rank) & confident).sum()),
        "best_cycle_distance_sum": float(
            teacher["best_cycle_distance_cells"].sum()
        ),
        "pseudo_loss": float(pseudo_loss.detach().cpu()),
        "retention_loss": float(retention_loss.detach().cpu()),
    }


def _merge_counts(total: dict[str, float], update: Mapping[str, float]) -> None:
    for key, value in update.items():
        total[key] = total.get(key, 0.0) + float(value)


def _summary(counts: Mapping[str, float], losses: Sequence[float]) -> dict[str, Any]:
    queries = max(1.0, counts.get("queries", 0.0))
    confident = max(1.0, counts.get("confident", 0.0))
    directions = max(1, len(losses) * 2)
    return {
        "optimization_steps": len(losses),
        "mean_total_loss": float(sum(losses) / max(1, len(losses))),
        "queries": int(counts.get("queries", 0.0)),
        "confident_pseudo_labels": int(counts.get("confident", 0.0)),
        "pseudo_label_coverage": float(counts.get("confident", 0.0) / queries),
        "unique_best_fraction": float(counts.get("unique_best", 0.0) / queries),
        "attention_teacher_agreement_on_confident": float(
            counts.get("attention_teacher_agreement", 0.0) / confident
        ),
        "base_teacher_agreement_on_confident": float(
            counts.get("base_teacher_agreement", 0.0) / confident
        ),
        "model_teacher_agreement_on_confident": float(
            counts.get("model_teacher_agreement", 0.0) / confident
        ),
        "teacher_change_fraction_from_base": float(
            counts.get("teacher_changes_base", 0.0) / confident
        ),
        "mean_best_cycle_distance_cells": float(
            counts.get("best_cycle_distance_sum", 0.0) / queries
        ),
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
        raise ValueError("triangle-cycle training requires --fjsar_shared_noise")
    if float(args.triangle_cycle_radius_cells) < 0.0:
        raise ValueError("triangle-cycle radius must be non-negative")

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    triplets, manifest_metadata = build_strict_training_triplet_manifest(
        args.dataset_path,
        seed=int(args.seed),
        max_triplets=int(args.max_train_triplets),
    )
    model, base_checkpoint = load_verifier_checkpoint(args.base_checkpoint, map_location="cpu")
    base_metadata = base_checkpoint.get("training_metadata", {})
    training_caption = str(args.training_caption).strip()
    if not training_caption:
        raise ValueError("--training_caption must be non-empty")
    if base_metadata.get("training_caption") not in (None, training_caption):
        raise ValueError("base checkpoint and triangle training captions differ")
    frozen_base = copy.deepcopy(model).to(device).eval()
    for parameter in frozen_base.parameters():
        parameter.requires_grad_(False)
    model = model.to(device).train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )

    categories = sorted({row.category for row in triplets})
    args.device = str(device)
    featurizer, flux_model, blocks = _load_flux_fjsar_runtime(args, categories)
    capture = _make_fjsar_capture(args, flux_model)
    pre_norm = nn.LayerNorm(3072, elementwise_affine=False, eps=1e-6).to(device)
    generator = torch.Generator().manual_seed(int(args.seed))
    counts: dict[str, float] = {}
    losses: list[float] = []
    dimensions_checked = False

    try:
        for triplet in tqdm(triplets, desc="train triangle-cycle verifier"):
            entries = {}
            for role, image_name in (
                ("source", triplet.source_name),
                ("target", triplet.target_name),
                ("bridge", triplet.bridge_name),
            ):
                entries[role] = _extract_flux_fjsar_entry(
                    args.dataset_path,
                    triplet.category,
                    image_name,
                    training_caption,
                    args,
                    featurizer,
                    capture,
                )
            source_features = _prepare_feature_tensors(
                entries["source"]["feature"], entries["source"]["ada"], args, pre_norm, device
            )
            target_features = _prepare_feature_tensors(
                entries["target"]["feature"], entries["target"]["ada"], args, pre_norm, device
            )
            source_grid = tuple(map(int, entries["source"]["feature"].shape[-2:]))
            target_grid = tuple(map(int, entries["target"]["feature"].shape[-2:]))
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
                entries["source"], entries["target"],
                source_features, target_features, source_points, source_size,
                blocks, args.candidate_topk, target_size=target_size,
            )
            reverse_batch = _candidate_batch(
                entries["target"], entries["source"],
                target_features, source_features, target_points, target_size,
                blocks, args.candidate_topk, target_size=source_size,
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
                    raise ValueError("base checkpoint feature dimensions do not match stage-3")
                dimensions_checked = True

            mutual_target_bridge = _mutual_attention(
                entries["target"], entries["bridge"], blocks
            )
            mutual_bridge_source = _mutual_attention(
                entries["bridge"], entries["source"], blocks
            )
            optimizer.zero_grad(set_to_none=True)
            directional_losses = []
            for batch, first_mutual, second_mutual, grid in (
                (forward_batch, mutual_target_bridge, mutual_bridge_source, source_grid),
                (
                    reverse_batch,
                    mutual_bridge_source.t().contiguous(),
                    mutual_target_bridge.t().contiguous(),
                    target_grid,
                ),
            ):
                direction_loss, direction_counts = _direction_loss(
                    model,
                    frozen_base,
                    batch,
                    first_mutual,
                    second_mutual,
                    grid,
                    triangle_cycle_radius_cells=float(args.triangle_cycle_radius_cells),
                    require_unique_best=bool(args.require_unique_best),
                    retention_weight=float(args.retention_weight),
                    device=device,
                )
                directional_losses.append(direction_loss)
                _merge_counts(counts, direction_counts)
            loss = torch.stack(directional_losses).mean()
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(
                    "non-finite triangle loss for "
                    f"{triplet.source_name}->{triplet.target_name}->{triplet.bridge_name}"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

            del (
                entries, source_features, target_features, forward_batch, reverse_batch,
                mutual_target_bridge, mutual_bridge_source, loss,
            )
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        capture.close()

    summary = _summary(counts, losses)
    base_protocol = dict(base_metadata.get("protocol", {}))
    metadata = {
        "supervision": "unlabeled_three_image_attention_triangle_cycle_distillation",
        "spair_keypoints_used": False,
        "spair_bounding_boxes_used": False,
        "segmentation_masks_used": False,
        "pose_labels_used": False,
        "category_labels_used_for_targets": False,
        "category_names_used_for_file_routing_only": True,
        "caption_labels_used": False,
        "training_pair_membership_used": True,
        "teacher": "frozen_flux_mutual_attention_B_to_C_to_A_unique_cycle",
        "teacher_used_for_inference": False,
        "external_matcher_used": False,
        "dino_used": False,
        "roma_used": False,
        "persistent_feature_cache_written": False,
        "base_checkpoint": str(args.base_checkpoint),
        "base_checkpoint_sha256": hashlib.sha256(
            Path(args.base_checkpoint).read_bytes()
        ).hexdigest(),
        "base_model_preserved_on_unconfirmed_queries": True,
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
            "cycle_radius_cells": float(base_protocol.get("cycle_radius_cells", 1.0)),
            "minimum_native_margin": float(base_protocol.get("minimum_native_margin", 0.01)),
            "triangle_cycle_radius_cells": float(args.triangle_cycle_radius_cells),
            "triangle_require_unique_best": bool(args.require_unique_best),
            "retention_weight": float(args.retention_weight),
            "residual_head_reset_before_stage3": False,
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
        "Triangle-cycle training complete: "
        f"coverage={100.0 * summary['pseudo_label_coverage']:.2f}, "
        f"unique={100.0 * summary['unique_best_fraction']:.2f}, "
        f"attention/teacher={100.0 * summary['attention_teacher_agreement_on_confident']:.2f}, "
        f"base/teacher={100.0 * summary['base_teacher_agreement_on_confident']:.2f}, "
        f"model/teacher={100.0 * summary['model_teacher_agreement_on_confident']:.2f}, "
        f"teacher changes base={100.0 * summary['teacher_change_fraction_from_base']:.2f}, "
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
    parser.add_argument("--max_train_triplets", type=int, default=400)
    parser.add_argument("--queries_per_image", type=int, default=64)
    parser.add_argument("--border_cells", type=int, default=1)
    parser.add_argument("--candidate_topk", type=int, default=20)
    parser.add_argument("--triangle_cycle_radius_cells", type=float, default=1.0)
    parser.add_argument("--require_unique_best", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retention_weight", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.set_defaults(
        matcher="attention_identity_verifier_triangle_cycle_training",
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

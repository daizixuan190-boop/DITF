"""Train a global candidate resolver from known elastic image deformations.

Training uses only individual images and a deterministic image-space warp.  The
current SPair smoke manifest uses train image names only; no pair relation or
SPair keypoint enters target construction.  The resolver still ranks only the frozen mutual-attention
top-20 pool used by evaluation.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from attention_identity_verifier import (
    CandidateIdentityVerifier,
    checkpoint_payload,
    sample_replay_cell_centers,
    verifier_config_from_batch,
)
from deformation_views import ElasticDeformationPlan, make_view_transform, sample_appearance_plan
from train_flux_attention_identity_verifier import (
    _batch_to_device,
    _candidate_batch,
    _direction_loss,
    _merge_counts,
    _training_summary,
    build_strict_training_manifest,
)


FEATURE_GROUPS = (
    "attention_aggregate",
    "qk_expert",
    "value_expert",
    "token_state",
    "channel_state_sketch",
    "proposal_attention",
    "native_control",
    "geometry_control",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_checkpoint", required=True)
    parser.add_argument("--output_summary", default="")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--img_size", nargs="+", type=int, default=[640, 640])
    parser.add_argument("--t", type=int, default=260)
    parser.add_argument("--k", nargs="+", type=int, default=[28])
    parser.add_argument("--ensemble_size", type=int, default=8)
    parser.add_argument("--cd", action="store_true")
    parser.add_argument("--fjsar_shared_noise", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--max_train_images", type=int, default=400)
    parser.add_argument("--queries_per_image", type=int, default=64)
    parser.add_argument("--border_cells", type=int, default=2)
    parser.add_argument("--candidate_topk", type=int, default=20)
    parser.add_argument("--group_width", type=int, default=32)
    parser.add_argument("--hidden_width", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--attention_prior_weight", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--target_sigma_cell_diagonals", type=float, default=0.5)
    parser.add_argument("--recoverability_cell_diagonals", type=float, default=0.75)
    return parser


def train(args: argparse.Namespace, device: torch.device) -> dict:
    from eval_spair_matcher_ablation import (
        _extract_flux_fjsar_entry,
        _load_flux_fjsar_runtime,
        _make_fjsar_capture,
        _prepare_feature_tensors,
    )

    if tuple(map(int, args.k)) != (28,) or int(args.t) != 260 or int(args.ensemble_size) != 8:
        raise ValueError("locked verifier protocol requires --k 28 --t 260 --ensemble_size 8")
    if not bool(args.fjsar_shared_noise):
        raise ValueError("deformation supervision requires --fjsar_shared_noise")
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    manifest, manifest_metadata = build_strict_training_manifest(
        args.dataset_path, seed=int(args.seed), max_images=int(args.max_train_images)
    )
    categories = sorted({row.category for row in manifest})
    args.device = str(device)
    featurizer, flux_model, blocks = _load_flux_fjsar_runtime(args, categories)
    capture = _make_fjsar_capture(args, flux_model)
    pre_norm = nn.LayerNorm(3072, elementwise_affine=False, eps=1e-6).to(device)
    generator = torch.Generator().manual_seed(int(args.seed))
    model = None
    optimizer = None
    losses = []
    counts = {}
    processed = 0

    try:
        for item in tqdm(manifest, desc="train deformation resolver"):
            image_key = f"{item.category}/{item.image_name}"
            plan_seed = int.from_bytes(
                hashlib.sha256(f"{int(args.seed)}:{image_key}".encode("utf-8")).digest()[:8],
                "little",
            )
            plan_rng = random.Random(plan_seed)
            source_appearance = sample_appearance_plan(plan_rng)
            target_appearance = sample_appearance_plan(plan_rng)
            deformation = ElasticDeformationPlan.sample(plan_rng)
            source_transform = make_view_transform(source_appearance, plan_rng)
            target_transform = make_view_transform(target_appearance, plan_rng, deformation)
            original = _extract_flux_fjsar_entry(
                args.dataset_path, item.category, item.image_name, "a photo", args,
                featurizer, capture, image_transform=source_transform,
            )
            warped = _extract_flux_fjsar_entry(
                args.dataset_path, item.category, item.image_name, "a photo", args,
                featurizer, capture, image_transform=target_transform,
            )
            grid_h, grid_w = map(int, original["feature"].shape[-2:])
            image_size = (grid_h * 16, grid_w * 16)
            source_features = _prepare_feature_tensors(
                original["feature"], original["ada"], args, pre_norm, device
            )
            warped_features = _prepare_feature_tensors(
                warped["feature"], warped["ada"], args, pre_norm, device
            )
            points = sample_replay_cell_centers(
                image_size, (grid_h, grid_w), count=int(args.queries_per_image),
                generator=generator, border_cells=int(args.border_cells),
            )
            target_points = deformation.source_to_target(points, image_size[0], image_size[1])
            reverse_points = deformation.target_to_source(target_points, image_size[0], image_size[1])
            batches = (
                (_candidate_batch(
                    original, warped, source_features, warped_features, points,
                    image_size, blocks, args.candidate_topk,
                ), target_points),
                (_candidate_batch(
                    warped, original, warped_features, source_features, target_points,
                    image_size, blocks, args.candidate_topk,
                ), reverse_points),
            )
            if model is None:
                config = verifier_config_from_batch(
                    batches[0][0], feature_groups=FEATURE_GROUPS,
                    group_width=int(args.group_width), hidden_width=int(args.hidden_width),
                    dropout=float(args.dropout), attention_prior_weight=float(args.attention_prior_weight),
                    global_query_context=True,
                )
                model = CandidateIdentityVerifier(config).to(device)
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay)
                )
            cell_diagonal = ((float(image_size[0]) / float(grid_h)) ** 2 +
                             (float(image_size[1]) / float(grid_w)) ** 2) ** 0.5
            sigma = float(args.target_sigma_cell_diagonals) * cell_diagonal
            maximum = float(args.recoverability_cell_diagonals) * cell_diagonal
            optimizer.zero_grad(set_to_none=True)
            directional_losses = []
            for batch, targets in batches:
                loss, metrics = _direction_loss(
                    model, batch, targets, image_size, sigma_pixels=sigma,
                    max_distance_pixels=maximum, device=device,
                )
                directional_losses.append(loss)
                _merge_counts(counts, metrics)
            loss = torch.stack(directional_losses).mean()
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"non-finite deformation loss for {image_key}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            processed += 1
            del original, warped, source_features, warped_features, batches, loss
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        capture.close()

    if model is None or processed == 0:
        raise RuntimeError("deformation training produced no optimization step")
    summary = _training_summary(counts, losses)
    metadata = {
        "supervision": "known_single_image_elastic_deformation",
        "spair_keypoints_used": False,
        "spair_bounding_boxes_used": False,
        "segmentation_masks_used": False,
        "pose_labels_used": False,
        "category_labels_used_for_targets": False,
        "category_names_used_for_file_routing_only": True,
        "caption_labels_used": False,
        "training_caption_policy": "fixed_neutral_prompt_for_all_images",
        "external_matcher_used": False,
        "dino_used": False,
        "roma_used": False,
        "pair_membership_used": True,
        "pair_membership_used_for_targets": False,
        "training_image_manifest_policy": "SPair train image names only; no pair relation used in targets",
        "persistent_feature_cache_written": False,
        "manifest": manifest_metadata,
        "protocol": {
            "image_size": [int(value) for value in args.img_size],
            "feature_block": int(args.k[0]), "timestep": int(args.t),
            "ensemble_size": int(args.ensemble_size), "channel_discard": bool(args.cd),
            "shared_noise": bool(args.fjsar_shared_noise), "candidate_topk": int(args.candidate_topk),
            "queries_per_image": int(args.queries_per_image), "bidirectional": True,
            "global_query_context": True, "feature_groups": list(FEATURE_GROUPS),
            "appearance_intervention": "independent_brightness_contrast_color_blur_invert",
            "deformation": "known_sinusoidal_elastic_field",
            "target_sigma_cell_diagonals": float(args.target_sigma_cell_diagonals),
            "recoverability_cell_diagonals": float(args.recoverability_cell_diagonals),
        },
        "summary": summary,
    }
    output_path = Path(args.output_checkpoint)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(model.cpu(), training_metadata=metadata), output_path)
    summary_path = Path(args.output_summary or output_path.with_suffix(".json"))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({"checkpoint": str(output_path), "training_metadata": metadata}, indent=2), encoding="utf-8")
    print(
        "Training complete: "
        f"recoverability={100.0 * summary['recoverability']:.2f}, "
        f"attention top1={100.0 * summary['attention_top1_on_recoverable']:.2f}, "
        f"resolver top1={100.0 * summary['verifier_top1_on_recoverable']:.2f}"
    )
    return metadata


if __name__ == "__main__":
    parsed = _build_parser().parse_args()
    run_device = torch.device(parsed.device if parsed.device == "cpu" or torch.cuda.is_available() else "cpu")
    train(parsed, run_device)

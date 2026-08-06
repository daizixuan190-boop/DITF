"""Supervised capacity diagnostic for frozen FLUX attention candidates.

This is intentionally not a label-free method.  It reads keypoint
correspondences from the SPair-71k *training* split and asks whether a learned
candidate-set decoder can convert the frozen cross-image attention top-k into
PCK.  Test images and test annotations are excluded from training.  DINO,
RoMa, category targets, and keypoint identity labels are not used.
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
from tqdm import tqdm

from attention_identity_verifier import (
    CandidateIdentityVerifier,
    checkpoint_payload,
    listwise_identity_loss,
    verifier_config_from_batch,
)
from eval_spair_attention_identity_verifier import _predict_baseline
from train_flux_attention_identity_verifier import (
    TrainingPair,
    _batch_to_device,
    _merge_counts,
    build_strict_training_pair_manifest,
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


def _pair_annotation(dataset_path: str, pair: TrainingPair) -> dict[str, Any]:
    path = Path(dataset_path) / "PairAnnotation" / "trn" / pair.pair_name
    with path.open(encoding="utf-8") as handle:
        row = json.load(handle)
    if str(row.get("src_imname")) != pair.source_name or str(row.get("trg_imname")) != pair.target_name:
        raise ValueError(f"training manifest and annotation disagree for {pair.pair_name}")
    required = (
        "src_kps",
        "trg_kps",
        "src_imsize",
        "trg_imsize",
        "src_bndbox",
        "trg_bndbox",
    )
    missing = [name for name in required if name not in row]
    if missing:
        raise ValueError(f"training pair {pair.pair_name} lacks fields: {missing}")
    if len(row["src_kps"]) != len(row["trg_kps"]) or not row["src_kps"]:
        raise ValueError(f"training keypoints do not align for {pair.pair_name}")
    return row


def _select_aligned_points(
    source_points: Sequence[Sequence[float]],
    target_points: Sequence[Sequence[float]],
    *,
    maximum: int,
    seed: int,
) -> tuple[list[list[float]], list[list[float]]]:
    if len(source_points) != len(target_points):
        raise ValueError("source and target keypoints must align")
    indices = list(range(len(source_points)))
    if int(maximum) > 0 and len(indices) > int(maximum):
        generator = torch.Generator().manual_seed(int(seed))
        indices = torch.randperm(len(indices), generator=generator)[: int(maximum)].tolist()
        indices.sort()
    source = [[float(value) for value in source_points[index][:2]] for index in indices]
    target = [[float(value) for value in target_points[index][:2]] for index in indices]
    return source, target


def _predictions_to_pixels(
    predictions: Sequence[Sequence[float]],
    target_size: Sequence[int],
) -> torch.Tensor:
    target_h, target_w = map(int, target_size)
    pixels = []
    for prediction in predictions:
        x = max(0, min(target_w - 1, int(round(float(prediction[0])))))
        y = max(0, min(target_h - 1, int(round(float(prediction[1])))))
        pixels.append(y * target_w + x)
    return torch.tensor(pixels, dtype=torch.long).reshape(-1, 1)


def _candidate_batch_with_baseline(
    source_entry: Mapping[str, Any],
    target_entry: Mapping[str, Any],
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    source_points: Sequence[Sequence[float]],
    source_size: Sequence[int],
    target_size: Sequence[int],
    blocks: Sequence[Any],
    candidate_topk: int,
) -> tuple[dict[str, Any], list[list[int]]]:
    from spair_matchers import flux_fjsar_candidate_feature_batch

    baseline = _predict_baseline(
        source_features,
        target_features,
        source_size,
        target_size,
        source_points,
    )
    baseline_pixels = _predictions_to_pixels(baseline, target_size)
    batch = flux_fjsar_candidate_feature_batch(
        source_features,
        target_features,
        source_points,
        source_size,
        target_size,
        src_replay_state=source_entry["replay_state"],
        trg_replay_state=target_entry["replay_state"],
        blocks=blocks,
        interaction_mode="exact",
        use_coordinate_bias=False,
        candidate_topk=int(candidate_topk),
        extra_candidate_pixels=baseline_pixels,
    )
    metadata = batch["metadata"]
    if int(metadata.get("extra_candidate_count", -1)) != 1:
        raise RuntimeError("supervised candidate pool must append exactly one baseline candidate")
    if bool(metadata.get("gt_used_for_features", True)):
        raise RuntimeError("ground truth leaked into candidate feature construction")
    return batch, baseline


def _pck_targets(
    candidate_pixels: torch.Tensor,
    target_points: torch.Tensor,
    target_size: Sequence[int],
    *,
    pck_reference: float,
    sigma_pixels: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if candidate_pixels.ndim != 2:
        raise ValueError("candidate pixels must be [query,candidate]")
    if target_points.shape != (candidate_pixels.shape[0], 2):
        raise ValueError("target points must align with candidate queries")
    if float(pck_reference) <= 0.0 or float(sigma_pixels) <= 0.0:
        raise ValueError("PCK reference and target sigma must be positive")
    target_h, target_w = map(int, target_size)
    pixels = candidate_pixels.long()
    candidate_x = (pixels % target_w).float()
    candidate_y = torch.div(pixels, target_w, rounding_mode="floor").float()
    squared = (
        (candidate_x - target_points[:, 0, None].float()).square()
        + (candidate_y - target_points[:, 1, None].float()).square()
    )
    distances = squared.sqrt()
    hits = distances <= 0.1 * float(pck_reference)
    recoverable = hits.any(dim=1)
    weights = torch.exp(-squared / (2.0 * float(sigma_pixels) ** 2)) * hits.float()
    targets = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return targets, recoverable, distances.amin(dim=1), hits


def _direction_loss(
    model: CandidateIdentityVerifier,
    batch: Mapping[str, Any],
    target_points: Sequence[Sequence[float]],
    target_size: Sequence[int],
    *,
    pck_reference: float,
    sigma_pixels: float,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    targets, recoverable, minimum_distance, hits = _pck_targets(
        batch["candidate_pixels"],
        torch.tensor(target_points, dtype=torch.float32),
        target_size,
        pck_reference=float(pck_reference),
        sigma_pixels=float(sigma_pixels),
    )
    groups, attention = _batch_to_device(batch, device, model.config.feature_groups)
    scores = model(groups, attention)
    loss = listwise_identity_loss(scores, targets.to(device), recoverable.to(device))
    selected = scores.detach().argmax(dim=1).cpu()
    selected_hit = hits.gather(1, selected[:, None]).squeeze(1)
    baseline_hit = hits[:, -1]
    attention_hit = hits[:, 0]
    counts = {
        "queries": float(hits.shape[0]),
        "recoverable": float(recoverable.sum()),
        "baseline_correct": float(baseline_hit.sum()),
        "attention_correct": float(attention_hit.sum()),
        "resolver_correct": float(selected_hit.sum()),
        "baseline_retained": float((baseline_hit & selected_hit).sum()),
        "pool_oracle_correct": float(hits.any(dim=1).sum()),
        "minimum_distance_sum": float(minimum_distance.sum()),
    }
    return loss, counts


def _training_summary(
    counts: Mapping[str, float],
    losses: Sequence[float],
    gradient_norms: Sequence[float],
) -> dict[str, Any]:
    queries = max(1.0, float(counts.get("queries", 0.0)))
    baseline_correct = max(1.0, float(counts.get("baseline_correct", 0.0)))
    baseline = float(counts.get("baseline_correct", 0.0))
    resolver = float(counts.get("resolver_correct", 0.0))
    oracle = float(counts.get("pool_oracle_correct", 0.0))
    gap = max(0.0, oracle - baseline)
    return {
        "measurement_timing": "online_pre_update_predictions_aggregated_during_single_pass",
        "optimization_steps": len(losses),
        "mean_loss": float(sum(losses) / max(1, len(losses))),
        "mean_gradient_norm": float(sum(gradient_norms) / max(1, len(gradient_norms))),
        "queries": int(counts.get("queries", 0.0)),
        "pool_recoverability": float(counts.get("recoverable", 0.0) / queries),
        "baseline_top1": baseline / queries,
        "attention_top1": float(counts.get("attention_correct", 0.0) / queries),
        "resolver_top1": resolver / queries,
        "pool_oracle": oracle / queries,
        "resolver_vs_baseline_net": int(resolver - baseline),
        "baseline_correct_retention_rate": float(
            counts.get("baseline_retained", 0.0) / baseline_correct
        ),
        "pool_oracle_gap_recovered_fraction": float((resolver - baseline) / gap) if gap else 0.0,
        "mean_nearest_candidate_distance_pixels": float(
            counts.get("minimum_distance_sum", 0.0) / queries
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
        raise ValueError("locked diagnostic protocol requires --k 28 --t 260 --ensemble_size 8")
    if not bool(args.fjsar_shared_noise):
        raise ValueError("locked diagnostic protocol requires --fjsar_shared_noise")
    if int(args.candidate_topk) != 20:
        raise ValueError("capacity diagnostic is locked to attention top-20")

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    pairs, manifest_metadata = build_strict_training_pair_manifest(
        args.dataset_path,
        seed=int(args.seed),
        max_pairs=int(args.max_train_pairs),
    )
    captions = json.loads(Path(args.captions_json).read_text(encoding="utf-8"))
    categories = sorted({pair.category for pair in pairs})
    args.device = str(device)
    featurizer, flux_model, blocks = _load_flux_fjsar_runtime(args, categories)
    capture = _make_fjsar_capture(args, flux_model)
    pre_norm = nn.LayerNorm(3072, elementwise_affine=False, eps=1e-6).to(device)
    model: CandidateIdentityVerifier | None = None
    optimizer: torch.optim.Optimizer | None = None
    counts: dict[str, float] = {}
    losses: list[float] = []
    gradient_norms: list[float] = []

    try:
        for pair in tqdm(pairs, desc="train supervised candidate identity decoder"):
            annotation = _pair_annotation(args.dataset_path, pair)
            point_seed = int.from_bytes(
                hashlib.sha256(f"{int(args.seed)}:{pair.pair_name}".encode("utf-8")).digest()[:8],
                "little",
            )
            source_points, target_points = _select_aligned_points(
                annotation["src_kps"],
                annotation["trg_kps"],
                maximum=int(args.max_keypoints_per_pair),
                seed=point_seed,
            )
            source_caption_key = pair.category + pair.source_name
            target_caption_key = pair.category + pair.target_name
            if source_caption_key not in captions or target_caption_key not in captions:
                raise KeyError(f"missing detailed caption for {pair.pair_name}")
            source_entry = _extract_flux_fjsar_entry(
                args.dataset_path,
                pair.category,
                pair.source_name,
                captions[source_caption_key],
                args,
                featurizer,
                capture,
            )
            target_entry = _extract_flux_fjsar_entry(
                args.dataset_path,
                pair.category,
                pair.target_name,
                captions[target_caption_key],
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
            source_size = annotation["src_imsize"][:2][::-1]
            target_size = annotation["trg_imsize"][:2][::-1]
            forward_batch, _forward_baseline = _candidate_batch_with_baseline(
                source_entry,
                target_entry,
                source_features,
                target_features,
                source_points,
                source_size,
                target_size,
                blocks,
                int(args.candidate_topk),
            )
            reverse_batch, _reverse_baseline = _candidate_batch_with_baseline(
                target_entry,
                source_entry,
                target_features,
                source_features,
                target_points,
                target_size,
                source_size,
                blocks,
                int(args.candidate_topk),
            )
            if model is None:
                config = verifier_config_from_batch(
                    forward_batch,
                    feature_groups=FEATURE_GROUPS,
                    group_width=int(args.group_width),
                    hidden_width=int(args.hidden_width),
                    dropout=float(args.dropout),
                    attention_prior_weight=0.0,
                    global_query_context=True,
                )
                model = CandidateIdentityVerifier(config).to(device)
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=float(args.learning_rate),
                    weight_decay=float(args.weight_decay),
                )

            source_grid = tuple(map(int, source_entry["feature"].shape[-2:]))
            target_grid = tuple(map(int, target_entry["feature"].shape[-2:]))
            source_cell_diagonal = (
                (float(source_size[0]) / source_grid[0]) ** 2
                + (float(source_size[1]) / source_grid[1]) ** 2
            ) ** 0.5
            target_cell_diagonal = (
                (float(target_size[0]) / target_grid[0]) ** 2
                + (float(target_size[1]) / target_grid[1]) ** 2
            ) ** 0.5
            source_reference = max(
                float(annotation["src_bndbox"][3] - annotation["src_bndbox"][1]),
                float(annotation["src_bndbox"][2] - annotation["src_bndbox"][0]),
            )
            target_reference = max(
                float(annotation["trg_bndbox"][3] - annotation["trg_bndbox"][1]),
                float(annotation["trg_bndbox"][2] - annotation["trg_bndbox"][0]),
            )
            assert model is not None and optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            directional_losses = []
            for batch, targets, size, reference, cell_diagonal in (
                (forward_batch, target_points, target_size, target_reference, target_cell_diagonal),
                (reverse_batch, source_points, source_size, source_reference, source_cell_diagonal),
            ):
                direction_loss, direction_counts = _direction_loss(
                    model,
                    batch,
                    targets,
                    size,
                    pck_reference=reference,
                    sigma_pixels=float(args.target_sigma_cell_diagonals) * cell_diagonal,
                    device=device,
                )
                directional_losses.append(direction_loss)
                _merge_counts(counts, direction_counts)
            loss = torch.stack(directional_losses).mean()
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"non-finite supervised loss for {pair.pair_name}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(args.max_grad_norm)
            )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            gradient_norms.append(float(gradient_norm.detach().cpu()))

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

    if model is None or not losses:
        raise RuntimeError("supervised capacity diagnostic produced no optimization step")
    summary = _training_summary(counts, losses, gradient_norms)
    metadata = {
        "diagnostic_scope": "supervised_capacity_only_not_final_method",
        "supervision": "spair_train_keypoints",
        "spair_keypoints_used": True,
        "spair_bounding_boxes_used": True,
        "test_keypoints_used_for_training": False,
        "test_images_used_for_training": False,
        "segmentation_masks_used": False,
        "pose_labels_used": False,
        "category_labels_used_for_targets": False,
        "keypoint_ids_used": False,
        "category_names_used_for_file_routing_and_frozen_flux_caption_lookup_only": True,
        "caption_labels_used": True,
        "caption_policy": "existing_per_image_detailed_captions_matching_evaluation",
        "external_matcher_used": False,
        "dino_used": False,
        "roma_used": False,
        "ground_truth_used_for_candidate_features": False,
        "ground_truth_used_for_training_targets": True,
        "persistent_feature_cache_written": False,
        "manifest": manifest_metadata,
        "annotation_fields_read": [
            "src_imname",
            "trg_imname",
            "src_kps",
            "trg_kps",
            "src_imsize",
            "trg_imsize",
            "src_bndbox",
            "trg_bndbox",
        ],
        "protocol": {
            "image_size": [int(value) for value in args.img_size],
            "feature_block": int(args.k[0]),
            "timestep": int(args.t),
            "ensemble_size": int(args.ensemble_size),
            "channel_discard": bool(args.cd),
            "shared_noise": bool(args.fjsar_shared_noise),
            "candidate_topk": int(args.candidate_topk),
            "extra_candidate_count": 1,
            "candidate_pool": "attention_top20_plus_frozen_ditf_top1",
            "max_keypoints_per_pair": int(args.max_keypoints_per_pair),
            "bidirectional": True,
            "global_query_context": True,
            "feature_groups": list(FEATURE_GROUPS),
            "attention_prior_weight": 0.0,
            "target_sigma_cell_diagonals": float(args.target_sigma_cell_diagonals),
            "pck_threshold": "0.1 * max(target_bbox_width, target_bbox_height)",
        },
        "summary": summary,
    }
    output_path = Path(args.output_checkpoint)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(model.cpu(), training_metadata=metadata), output_path)
    summary_path = Path(args.output_summary or output_path.with_suffix(".json"))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps({"checkpoint": str(output_path), "training_metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    print(
        "Supervised capacity training complete: "
        f"baseline/attention/resolver/pool-oracle="
        f"{100.0 * summary['baseline_top1']:.2f}/"
        f"{100.0 * summary['attention_top1']:.2f}/"
        f"{100.0 * summary['resolver_top1']:.2f}/"
        f"{100.0 * summary['pool_oracle']:.2f}, "
        f"baseline retention={100.0 * summary['baseline_correct_retention_rate']:.2f}, "
        f"pool-gap recovered={100.0 * summary['pool_oracle_gap_recovered_fraction']:.2f}"
    )
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--captions_json", default="spair_detailed_captions.json")
    parser.add_argument("--output_checkpoint", required=True)
    parser.add_argument("--output_summary", default="")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--img_size", nargs="+", type=int, default=[640, 640])
    parser.add_argument("--t", type=int, default=260)
    parser.add_argument("--k", nargs="+", type=int, default=[28])
    parser.add_argument("--ensemble_size", type=int, default=8)
    parser.add_argument("--cd", action="store_true", default=False)
    parser.add_argument("--fjsar_shared_noise", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--max_train_pairs", type=int, default=64)
    parser.add_argument("--max_keypoints_per_pair", type=int, default=32)
    parser.add_argument("--candidate_topk", type=int, default=20)
    parser.add_argument("--group_width", type=int, default=32)
    parser.add_argument("--hidden_width", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--target_sigma_cell_diagonals", type=float, default=0.5)
    parser.set_defaults(
        matcher="supervised_attention_candidate_identity_training",
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

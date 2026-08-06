"""Train a label-free verifier for frozen FLUX attention candidates.

The only supervision is the known correspondence between an image and its
horizontal flip. SPair pair annotations are used solely to enumerate training
images and to exclude benchmark test images; keypoints, boxes, masks, poses,
and category labels are never read for target construction.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn as nn
from tqdm import tqdm

from attention_identity_verifier import (
    CandidateIdentityVerifier,
    checkpoint_payload,
    horizontal_flip_points,
    listwise_identity_loss,
    sample_replay_cell_centers,
    transformed_candidate_targets,
    verifier_config_from_batch,
)
@dataclass(frozen=True)
class TrainingImage:
    category: str
    image_name: str


@dataclass(frozen=True)
class TrainingPair:
    category: str
    pair_name: str
    source_name: str
    target_name: str


def _pair_json_paths(dataset_path: str | os.PathLike[str], split: str) -> list[Path]:
    split_path = Path(dataset_path) / "PairAnnotation" / split
    if not split_path.is_dir():
        raise FileNotFoundError(f"SPair split directory does not exist: {split_path}")
    paths = sorted(split_path.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"SPair split contains no JSON files: {split_path}")
    return paths


def _image_category_index(dataset_path: str | os.PathLike[str]) -> dict[str, set[str]]:
    image_root = Path(dataset_path) / "JPEGImages"
    if not image_root.is_dir():
        raise FileNotFoundError(f"SPair image directory does not exist: {image_root}")
    result: dict[str, set[str]] = {}
    for category_path in sorted(path for path in image_root.iterdir() if path.is_dir()):
        for image_path in category_path.iterdir():
            if image_path.is_file():
                result.setdefault(image_path.name, set()).add(category_path.name)
    return result


def _pair_images(path: Path) -> tuple[str, str]:
    with path.open(encoding="utf-8") as handle:
        row = json.load(handle)
    # Deliberately whitelist only image names. No annotation field is exposed.
    return str(row["src_imname"]), str(row["trg_imname"])


def _resolve_pair_category(
    source_name: str,
    target_name: str,
    image_categories: Mapping[str, set[str]],
    *,
    pair_name: str,
) -> str:
    shared = image_categories.get(source_name, set()) & image_categories.get(target_name, set())
    if len(shared) == 1:
        return next(iter(shared))
    # SPair images may occur in several semantic-category directories. The
    # benchmark's own evaluator assigns pair JSONs to categories by filename.
    # Reuse that routing convention; it is not a training target.
    filename_matches = [category for category in shared if category in str(pair_name)]
    if len(filename_matches) == 1:
        return filename_matches[0]
    raise ValueError(
        "could not resolve one category directory for pair "
        f"{pair_name}: {source_name} -> {target_name}; "
        f"shared categories={sorted(shared)}, filename matches={sorted(filename_matches)}"
    )


def build_strict_training_manifest(
    dataset_path: str | os.PathLike[str],
    *,
    seed: int,
    max_images: int,
) -> tuple[list[TrainingImage], dict[str, Any]]:
    """Enumerate train images while excluding every image used by test pairs."""

    image_categories = _image_category_index(dataset_path)
    test_images: set[tuple[str, str]] = set()
    for path in _pair_json_paths(dataset_path, "test"):
        source_name, target_name = _pair_images(path)
        category = _resolve_pair_category(
            source_name,
            target_name,
            image_categories,
            pair_name=path.name,
        )
        test_images.add((category, source_name))
        test_images.add((category, target_name))

    training_images: set[tuple[str, str]] = set()
    overlap_rejections = 0
    for path in _pair_json_paths(dataset_path, "trn"):
        source_name, target_name = _pair_images(path)
        category = _resolve_pair_category(
            source_name,
            target_name,
            image_categories,
            pair_name=path.name,
        )
        for image_name in (source_name, target_name):
            key = (category, image_name)
            if key in test_images:
                overlap_rejections += 1
            else:
                training_images.add(key)

    ordered = sorted(
        training_images,
        key=lambda item: hashlib.sha256(
            f"{int(seed)}:{item[0]}:{item[1]}".encode("utf-8")
        ).digest(),
    )
    if int(max_images) > 0:
        ordered = ordered[: int(max_images)]
    if not ordered:
        raise RuntimeError("strict SPair training manifest is empty")
    manifest = [TrainingImage(category=category, image_name=name) for category, name in ordered]
    metadata = {
        "split_source": "PairAnnotation/trn image names only",
        "test_exclusion_source": "PairAnnotation/test image names only",
        "training_images": len(manifest),
        "unique_test_images_excluded": len(test_images),
        "train_occurrences_rejected_for_test_overlap": int(overlap_rejections),
        "selection_seed": int(seed),
        "max_images": int(max_images),
        "annotation_fields_read": ["src_imname", "trg_imname"],
        "selected_manifest_sha256": hashlib.sha256(
            "\n".join(f"{category}/{name}" for category, name in ordered).encode("utf-8")
        ).hexdigest(),
    }
    return manifest, metadata


def build_strict_training_pair_manifest(
    dataset_path: str | os.PathLike[str],
    *,
    seed: int,
    max_pairs: int,
) -> tuple[list[TrainingPair], dict[str, Any]]:
    """Enumerate unlabeled train pairs while excluding benchmark test images."""

    image_categories = _image_category_index(dataset_path)
    test_images: set[tuple[str, str]] = set()
    for path in _pair_json_paths(dataset_path, "test"):
        source_name, target_name = _pair_images(path)
        category = _resolve_pair_category(
            source_name,
            target_name,
            image_categories,
            pair_name=path.name,
        )
        test_images.update(((category, source_name), (category, target_name)))

    pairs = []
    overlap_rejections = 0
    for path in _pair_json_paths(dataset_path, "trn"):
        source_name, target_name = _pair_images(path)
        category = _resolve_pair_category(
            source_name,
            target_name,
            image_categories,
            pair_name=path.name,
        )
        if (category, source_name) in test_images or (category, target_name) in test_images:
            overlap_rejections += 1
            continue
        pairs.append(TrainingPair(
            category=category,
            pair_name=path.name,
            source_name=source_name,
            target_name=target_name,
        ))
    pairs.sort(key=lambda row: hashlib.sha256(
        f"{int(seed)}:{row.pair_name}".encode("utf-8")
    ).digest())
    if int(max_pairs) > 0:
        pairs = pairs[: int(max_pairs)]
    if not pairs:
        raise RuntimeError("strict SPair unlabeled training-pair manifest is empty")
    metadata = {
        "split_source": "PairAnnotation/trn image-pair names only",
        "test_exclusion_source": "PairAnnotation/test image names only",
        "training_pairs": len(pairs),
        "pair_membership_used": True,
        "keypoint_fields_read": False,
        "category_used_for_targets": False,
        "pairs_rejected_for_test_overlap": int(overlap_rejections),
        "selection_seed": int(seed),
        "max_pairs": int(max_pairs),
        "annotation_fields_read": ["src_imname", "trg_imname"],
        "selected_manifest_sha256": hashlib.sha256(
            "\n".join(row.pair_name for row in pairs).encode("utf-8")
        ).hexdigest(),
    }
    return pairs, metadata


def _batch_to_device(
    batch: Mapping[str, Any],
    device: torch.device,
    feature_groups: Iterable[str],
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    groups = {
        name: batch["feature_groups"][name].to(device=device, dtype=torch.float32)
        for name in feature_groups
    }
    attention = batch["attention_scores"].to(device=device, dtype=torch.float32)
    return groups, attention


def _assert_label_free_batch(batch: Mapping[str, Any]) -> None:
    metadata = batch.get("metadata", {})
    forbidden = {"candidate_hits", "target_points", "labels"} & set(batch)
    if forbidden:
        raise RuntimeError(f"self-supervised batch contains forbidden labels: {sorted(forbidden)}")
    if (
        bool(metadata.get("gt_used_for_features", True))
        or bool(metadata.get("gt_used_for_labels_only", True))
        or bool(metadata.get("labels_present", True))
    ):
        raise RuntimeError("candidate feature batch is not annotation-free")


def _candidate_batch(
    source_entry: Mapping[str, Any],
    target_entry: Mapping[str, Any],
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    source_points: torch.Tensor,
    image_size: Sequence[int],
    blocks: Sequence[Any],
    candidate_topk: int,
    *,
    target_size: Sequence[int] | None = None,
) -> dict[str, Any]:
    from spair_matchers import flux_fjsar_candidate_feature_batch

    batch = flux_fjsar_candidate_feature_batch(
        source_features,
        target_features,
        source_points.tolist(),
        image_size,
        target_size if target_size is not None else image_size,
        src_replay_state=source_entry["replay_state"],
        trg_replay_state=target_entry["replay_state"],
        blocks=blocks,
        interaction_mode="exact",
        use_coordinate_bias=False,
        candidate_topk=int(candidate_topk),
    )
    _assert_label_free_batch(batch)
    return batch


def _direction_loss(
    model: CandidateIdentityVerifier,
    batch: Mapping[str, Any],
    target_points: torch.Tensor,
    image_size: Sequence[int],
    *,
    sigma_pixels: float,
    max_distance_pixels: float,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    targets, recoverable, minimum_distance = transformed_candidate_targets(
        batch["candidate_pixels"],
        target_points.cpu(),
        image_size,
        sigma_pixels=float(sigma_pixels),
        max_distance_pixels=float(max_distance_pixels),
    )
    groups, attention = _batch_to_device(batch, device, model.config.feature_groups)
    scores = model(groups, attention)
    recoverable_device = recoverable.to(device)
    loss = listwise_identity_loss(scores, targets.to(device), recoverable_device)
    nearest = targets.argmax(dim=1)
    attention_correct = attention.argmax(dim=1).cpu() == nearest
    verifier_correct = scores.detach().argmax(dim=1).cpu() == nearest
    mask = recoverable.cpu()
    count = int(mask.sum())
    metrics = {
        "queries": float(mask.numel()),
        "recoverable": float(count),
        "attention_correct": float((attention_correct & mask).sum()),
        "verifier_correct": float((verifier_correct & mask).sum()),
        "minimum_distance_sum": float(minimum_distance.sum()),
    }
    return loss, metrics


def _merge_counts(total: dict[str, float], update: Mapping[str, float]) -> None:
    for key, value in update.items():
        total[key] = total.get(key, 0.0) + float(value)


def _training_summary(counts: Mapping[str, float], losses: Sequence[float]) -> dict[str, Any]:
    queries = max(1.0, float(counts.get("queries", 0.0)))
    recoverable = max(1.0, float(counts.get("recoverable", 0.0)))
    return {
        "optimization_steps": len(losses),
        "mean_loss": float(sum(losses) / max(1, len(losses))),
        "queries": int(counts.get("queries", 0.0)),
        "recoverable_queries": int(counts.get("recoverable", 0.0)),
        "recoverability": float(counts.get("recoverable", 0.0) / queries),
        "attention_top1_on_recoverable": float(
            counts.get("attention_correct", 0.0) / recoverable
        ),
        "verifier_top1_on_recoverable": float(
            counts.get("verifier_correct", 0.0) / recoverable
        ),
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
        raise ValueError("locked verifier protocol requires --k 28 --t 260 --ensemble_size 8")
    if not bool(args.fjsar_shared_noise):
        raise ValueError("equivariant supervision requires --fjsar_shared_noise")

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    manifest, manifest_metadata = build_strict_training_manifest(
        args.dataset_path,
        seed=int(args.seed),
        max_images=int(args.max_train_images),
    )
    training_caption = str(args.training_caption).strip()
    if not training_caption:
        raise ValueError("--training_caption must be non-empty")
    categories = sorted({row.category for row in manifest})
    args.device = str(device)
    featurizer, flux_model, blocks = _load_flux_fjsar_runtime(args, categories)
    capture = _make_fjsar_capture(args, flux_model)
    pre_norm = nn.LayerNorm(3072, elementwise_affine=False, eps=1e-6).to(device)
    generator = torch.Generator().manual_seed(int(args.seed))
    model: CandidateIdentityVerifier | None = None
    optimizer: torch.optim.Optimizer | None = None
    losses: list[float] = []
    counts: dict[str, float] = {}
    processed_images = 0

    try:
        for item in tqdm(manifest, desc="train equivariant verifier"):
            image_key = item.category + "/" + item.image_name
            original = _extract_flux_fjsar_entry(
                args.dataset_path,
                item.category,
                item.image_name,
                training_caption,
                args,
                featurizer,
                capture,
            )
            flipped = _extract_flux_fjsar_entry(
                args.dataset_path,
                item.category,
                item.image_name,
                training_caption,
                args,
                featurizer,
                capture,
                horizontal_flip=True,
            )
            grid_h, grid_w = map(int, original["feature"].shape[-2:])
            image_size = (grid_h * 16, grid_w * 16)
            original_features = _prepare_feature_tensors(
                original["feature"], original["ada"], args, pre_norm, device
            )
            flipped_features = _prepare_feature_tensors(
                flipped["feature"], flipped["ada"], args, pre_norm, device
            )
            points = sample_replay_cell_centers(
                image_size,
                (grid_h, grid_w),
                count=int(args.queries_per_image),
                generator=generator,
                border_cells=int(args.border_cells),
            )
            transformed = horizontal_flip_points(points, image_size[1])
            batches = (
                (
                    _candidate_batch(
                        original,
                        flipped,
                        original_features,
                        flipped_features,
                        points,
                        image_size,
                        blocks,
                        args.candidate_topk,
                    ),
                    transformed,
                ),
                (
                    _candidate_batch(
                        flipped,
                        original,
                        flipped_features,
                        original_features,
                        transformed,
                        image_size,
                        blocks,
                        args.candidate_topk,
                    ),
                    points,
                ),
            )
            if model is None:
                config = verifier_config_from_batch(
                    batches[0][0],
                    group_width=int(args.group_width),
                    hidden_width=int(args.hidden_width),
                    dropout=float(args.dropout),
                    attention_prior_weight=float(args.attention_prior_weight),
                )
                model = CandidateIdentityVerifier(config).to(device)
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=float(args.learning_rate),
                    weight_decay=float(args.weight_decay),
                )

            cell_diagonal = (
                (float(image_size[0]) / float(grid_h)) ** 2
                + (float(image_size[1]) / float(grid_w)) ** 2
            ) ** 0.5
            sigma = float(args.target_sigma_cell_diagonals) * cell_diagonal
            maximum = float(args.recoverability_cell_diagonals) * cell_diagonal
            assert model is not None and optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            directional_losses = []
            for batch, targets in batches:
                direction_loss, direction_counts = _direction_loss(
                    model,
                    batch,
                    targets,
                    image_size,
                    sigma_pixels=sigma,
                    max_distance_pixels=maximum,
                    device=device,
                )
                directional_losses.append(direction_loss)
                _merge_counts(counts, direction_counts)
            loss = torch.stack(directional_losses).mean()
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"non-finite verifier loss for {image_key}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            processed_images += 1

            del original, flipped, original_features, flipped_features, batches, loss
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        capture.close()

    if model is None or processed_images == 0:
        raise RuntimeError("verifier training produced no optimization step")
    summary = _training_summary(counts, losses)
    training_metadata = {
        "supervision": "known_same_image_horizontal_flip_correspondence",
        "spair_keypoints_used": False,
        "spair_bounding_boxes_used": False,
        "segmentation_masks_used": False,
        "pose_labels_used": False,
        "category_labels_used_for_targets": False,
        "category_names_used_for_file_routing_only": True,
        "caption_labels_used": False,
        "training_caption_policy": "fixed_neutral_prompt_for_all_images",
        "training_caption": training_caption,
        "training_caption_sha256": hashlib.sha256(
            training_caption.encode("utf-8")
        ).hexdigest(),
        "external_matcher_used": False,
        "dino_used": False,
        "roma_used": False,
        "persistent_feature_cache_written": False,
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
            "target_sigma_cell_diagonals": float(args.target_sigma_cell_diagonals),
            "recoverability_cell_diagonals": float(args.recoverability_cell_diagonals),
        },
        "summary": summary,
    }
    output_path = Path(args.output_checkpoint)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(model.cpu(), training_metadata=training_metadata), output_path)
    summary_path = Path(args.output_summary or output_path.with_suffix(".json"))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps({"checkpoint": str(output_path), "training_metadata": training_metadata}, indent=2),
        encoding="utf-8",
    )
    print(
        "Training complete: "
        f"recoverability={100.0 * summary['recoverability']:.2f}, "
        f"attention top1={100.0 * summary['attention_top1_on_recoverable']:.2f}, "
        f"verifier top1={100.0 * summary['verifier_top1_on_recoverable']:.2f}"
    )
    return training_metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument(
        "--training_caption",
        default="a photo",
        help="One fixed label-free prompt used for every self-supervised training image.",
    )
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
    parser.add_argument("--max_train_images", type=int, default=400)
    parser.add_argument("--queries_per_image", type=int, default=64)
    parser.add_argument("--border_cells", type=int, default=1)
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
    parser.set_defaults(
        matcher="attention_identity_verifier_training",
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

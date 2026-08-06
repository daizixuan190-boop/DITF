"""Streaming baseline-first Flux evaluator for the GeoAware AP-10K benchmark.

Features are held in CPU memory for one species/family group and discarded
after evaluation. No persistent 960px Flux feature cache is written.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from tqdm import tqdm

from ap10k_flux import (
    PAPER_FLUX_PER_IMAGE,
    discover_pairs,
    empty_metric_counts,
    load_square_padded_tensor,
    merge_metric_counts,
    metric_ratios,
    pair_hits,
    prepare_annotation,
    update_metric_counts,
)
from flux_spair import chunked_native_flux_predictions, prepare_flux_feature
from flux_spair import (
    grid_candidate_points,
    grid_cosine_scores,
    resize_feature_long_side,
)
from ownership_diagnostics import (
    controlled_candidate_rows,
    empty_counts as empty_ownership_counts,
    ratios as ownership_ratios,
    update_counts as update_ownership_counts,
)


def _stable_image_seed(base_seed: int, annotation_path: str) -> int:
    digest = hashlib.sha256(annotation_path.encode("utf-8")).digest()
    return (base_seed + int.from_bytes(digest[:4], "little")) % (2**31)


def _caption(mode: str, species: str) -> str:
    if mode == "species":
        # Match the category prompt template released in Featurizer4Eval.
        return f"a photo of a {species}"
    if mode == "generic":
        return "a photo of an animal"
    if mode == "empty":
        return ""
    raise ValueError(f"Unknown prompt mode: {mode}")


def _plain_ownership_counts(
    counts: dict[int, dict[str, int | float]],
) -> dict[str, dict[str, int | float]]:
    return {str(k): dict(values) for k, values in counts.items()}


def _restore_ownership_counts(
    payload: dict[str, dict[str, int | float]], ks: tuple[int, ...]
) -> dict[int, dict[str, int | float]]:
    counts = empty_ownership_counts(ks)
    for k in ks:
        counts[k].update(payload.get(str(k), {}))
    return counts


def _merge_ownership_counts(
    target: dict[int, dict[str, int | float]],
    source: dict[int, dict[str, int | float]],
) -> None:
    for k, values in source.items():
        for name, value in values.items():
            target[k][name] += value


def _save_summary(
    args: argparse.Namespace,
    native_group_counts: dict[str, dict[str, Any]],
    grid_group_counts: dict[str, dict[str, Any]],
    ownership_group_counts: dict[str, dict[str, dict[str, int | float]]],
    total_groups: int,
) -> dict[str, Any]:
    ks = tuple(sorted(set(args.topk)))
    native_total = empty_metric_counts()
    for counts in native_group_counts.values():
        merge_metric_counts(native_total, counts)
    metrics = metric_ratios(native_total) if native_total["pair_count"] else None

    grid_total = empty_metric_counts()
    for counts in grid_group_counts.values():
        merge_metric_counts(grid_total, counts)
    grid_metrics = metric_ratios(grid_total) if grid_total["pair_count"] else None

    ownership_total = empty_ownership_counts(ks)
    for counts in ownership_group_counts.values():
        _merge_ownership_counts(ownership_total, _restore_ownership_counts(counts, ks))
    ownership = (
        ownership_ratios(ownership_total, ks)
        if any(ownership_total[k]["points"] for k in ks)
        else None
    )
    validation = None
    if ownership is not None and grid_metrics is not None:
        validation = {
            "diagnostic_owner_at_1_equals_grid_baseline": (
                abs(
                    ownership["1"]["owner_candidate_recall"]
                    - grid_metrics["per_point_pck"]["0.10"]
                )
                < 1e-12
                if 1 in ks
                else None
            ),
            "failure_owner_at_1_is_zero": (
                ownership["1"]["failure_owner_candidate_recall"] == 0.0
                if 1 in ks
                else None
            ),
            "strict_is_bounded_by_global": all(
                ownership[str(k)]["strict_global_union_recall"]
                <= ownership[str(k)]["global_union_recall"]
                for k in ks
            ),
            "budget_owner_contains_owner_topk": all(
                ownership[str(k)]["budget_matched_owner_recall"]
                >= ownership[str(k)]["owner_candidate_recall"]
                for k in ks
            ),
        }

    native_complete_groups = len(native_group_counts)
    diagnostic_groups = set(grid_group_counts) & set(ownership_group_counts)
    diagnostic_complete_groups = len(diagnostic_groups)
    complete_groups = (
        diagnostic_complete_groups if args.diagnostics else native_complete_groups
    )
    paper = PAPER_FLUX_PER_IMAGE[args.setting]
    parity = None
    if metrics is not None:
        parity = {
            alpha: {
                "paper": expected,
                "observed": metrics["per_image_pck"][alpha],
                "error": metrics["per_image_pck"][alpha] - expected,
            }
            for alpha, expected in paper.items()
        }
    summary = {
        "protocol": {
            "name": "ditf-flux-ap10k-baseline-v1",
            "metric_primary": "per-image PCK",
            "alphas": [0.01, 0.05, 0.10],
            "image_preprocess": "aspect-preserving resize and zero-pad to square",
            "keypoint_preprocess": "GeoAware round-size, integer-coordinate protocol",
            "target_threshold": "max(target bbox width, height) after resize",
            "feature_postprocess": "official Flux channel discard, LayerNorm, AdaLN",
            "matching": "full-channel native-pixel cosine NN with channel-chunked accumulation",
            "diagnostic_matching": (
                f"square {args.diagnostic_grid}x{args.diagnostic_grid} feature-grid cosine NN"
                if args.diagnostics
                else None
            ),
            "ground_truth_changes_ranking": False,
            "diagnostic_controls": (
                [
                    "strict target-GT-region non-overlap",
                    "exact unique-candidate budget matched owner ranking",
                    "target-region proposal-source hubness",
                    "uniform random union expectation",
                ]
                if args.diagnostics
                else []
            ),
            "persistent_feature_cache": False,
            "paper_target": paper,
            "validation": validation,
        },
        "config": vars(args),
        "progress": {
            "complete_groups": complete_groups,
            "native_complete_groups": native_complete_groups,
            "diagnostic_complete_groups": diagnostic_complete_groups,
            "total_groups": total_groups,
            "complete": complete_groups == total_groups,
        },
        "metrics": metrics,
        "grid_metrics": grid_metrics,
        "ownership": ownership,
        "paper_parity": parity,
        "group_counts": native_group_counts,
        "grid_group_counts": grid_group_counts,
        "ownership_group_counts": ownership_group_counts,
        "group_diagnostics": {
            group: {
                "native_metrics": metric_ratios(native_group_counts[group]),
                "grid_metrics": (
                    metric_ratios(grid_group_counts[group])
                    if group in grid_group_counts
                    else None
                ),
                "ownership": (
                    ownership_ratios(
                        _restore_ownership_counts(ownership_group_counts[group], ks), ks
                    )
                    if group in ownership_group_counts
                    else None
                ),
            }
            for group in native_group_counts
        },
        "runtime": {
            "cuda_peak_memory_gib": (
                torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else None
            ),
        },
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(output)
    return summary


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.img_size % 16:
        raise ValueError("--img_size must be divisible by 16 for Flux")
    ks = tuple(sorted(set(args.topk)))
    if args.diagnostics and (not ks or min(ks) < 1):
        raise ValueError("--topk values must be positive")
    if args.diagnostics and args.diagnostic_grid < 1:
        raise ValueError("--diagnostic_grid must be positive")
    benchmark_root = Path(args.dataset_path)
    groups = discover_pairs(
        benchmark_root,
        args.setting,
        max_groups=args.max_groups,
        max_pairs_per_group=args.max_pairs_per_group,
        pair_sample_seed=args.pair_sample_seed,
    )
    output_path = Path(args.output_json)
    native_group_counts: dict[str, dict[str, Any]] = {}
    grid_group_counts: dict[str, dict[str, Any]] = {}
    ownership_group_counts: dict[str, dict[str, dict[str, int | float]]] = {}
    if args.resume and output_path.is_file():
        previous = json.loads(output_path.read_text(encoding="utf-8"))
        previous_config = previous["config"]
        invariant_keys = (
            "dataset_path", "setting", "img_size", "t", "k", "ensemble_size",
            "cd", "prompt_mode", "seed", "match_channel_chunk", "max_groups",
            "max_pairs_per_group", "diagnostics", "diagnostic_grid", "topk",
            "pair_sample_seed",
        )
        mismatch = [key for key in invariant_keys if previous_config.get(key) != getattr(args, key)]
        if mismatch:
            raise RuntimeError(f"Cannot resume with changed configuration: {mismatch}")
        native_group_counts = previous.get("group_counts", {})
        grid_group_counts = previous.get("grid_group_counts", {})
        ownership_group_counts = previous.get("ownership_group_counts", {})

    pending = [
        group
        for group in groups
        if group not in native_group_counts
        or (
            args.diagnostics
            and (group not in grid_group_counts or group not in ownership_group_counts)
        )
    ]
    if not pending:
        return _save_summary(
            args,
            native_group_counts,
            grid_group_counts,
            ownership_group_counts,
            len(groups),
        )

    from src.flux.feat_flux import Featurizer4Eval

    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.reset_peak_memory_stats(device)
    model = Featurizer4Eval(cat_list=[], ensemble_size=args.ensemble_size)
    forward_args = SimpleNamespace()

    for group in pending:
        pairs = groups[group]
        annotation_paths = sorted(
            {path for pair in pairs for path in (pair.source_annotation, pair.target_annotation)}
        )
        annotations = {
            path: prepare_annotation(benchmark_root, path, args.img_size)
            for path in annotation_paths
        }
        features: dict[str, torch.Tensor] = {}
        for path in tqdm(annotation_paths, desc=f"extract {group}"):
            annotation = annotations[path]
            seed = _stable_image_seed(args.seed, path)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            image = load_square_padded_tensor(annotation.image_path, args.img_size)
            raw, adaln = model.forward(
                forward_args,
                image,
                caption=_caption(args.prompt_mode, annotation.species),
                category=annotation.species,
                timestep=args.t,
                # FluxModel.forward_feat expects an iterable of feature indices.
                block_idx=[args.k],
                ensemble_size=args.ensemble_size,
            )
            feature = prepare_flux_feature(raw, adaln, args.cd)
            features[path] = feature.to(device="cpu", dtype=torch.bfloat16)
            del image, raw, adaln, feature
            torch.cuda.empty_cache()

        native_counts = empty_metric_counts()
        grid_counts = empty_metric_counts()
        diagnostic_counts = empty_ownership_counts(ks)
        for pair in tqdm(pairs, desc=f"evaluate {group}"):
            source = annotations[pair.source_annotation]
            target = annotations[pair.target_annotation]
            mutual = (source.visibility * target.visibility) > 0
            source_points = source.points[mutual]
            target_points = target.points[mutual].to(device)
            source_feature = features[pair.source_annotation].to(device)
            target_feature = features[pair.target_annotation].to(device)
            predictions = chunked_native_flux_predictions(
                source_feature,
                target_feature,
                source_points.tolist(),
                (args.img_size, args.img_size),
                (args.img_size, args.img_size),
                channel_chunk=args.match_channel_chunk,
            )
            update = pair_hits(predictions, target_points, target.threshold)
            update_metric_counts(native_counts, update)

            if args.diagnostics:
                source_grid = resize_feature_long_side(
                    source_feature,
                    args.img_size,
                    args.img_size,
                    args.diagnostic_grid,
                )
                target_grid = resize_feature_long_side(
                    target_feature,
                    args.img_size,
                    args.img_size,
                    args.diagnostic_grid,
                )
                scores = grid_cosine_scores(
                    source_grid,
                    target_grid,
                    source_points,
                    args.img_size,
                    args.img_size,
                )
                candidate_points = grid_candidate_points(
                    args.img_size,
                    args.img_size,
                    target_grid.shape[-2],
                    target_grid.shape[-1],
                    device=device,
                )
                baseline_indices = scores.argmax(dim=1)
                grid_predictions = candidate_points[baseline_indices]
                grid_update = pair_hits(grid_predictions, target_points, target.threshold)
                update_metric_counts(grid_counts, grid_update)
                rows = controlled_candidate_rows(
                    scores.detach().float().cpu(),
                    candidate_points.detach().cpu(),
                    target_points.detach().cpu(),
                    target.threshold,
                    ks=ks,
                    baseline_indices=baseline_indices.detach().cpu(),
                )
                update_ownership_counts(
                    diagnostic_counts,
                    rows,
                    grid_update["0.10"],
                    ks,
                )
                del (
                    source_grid,
                    target_grid,
                    scores,
                    candidate_points,
                    baseline_indices,
                    grid_predictions,
                    grid_update,
                    rows,
                )

            del source_feature, target_feature, predictions, target_points, update

        native_group_counts[group] = native_counts
        if args.diagnostics:
            grid_group_counts[group] = grid_counts
            ownership_group_counts[group] = _plain_ownership_counts(diagnostic_counts)
        _save_summary(
            args,
            native_group_counts,
            grid_group_counts,
            ownership_group_counts,
            len(groups),
        )
        del features, annotations
        gc.collect()
        torch.cuda.empty_cache()

    summary = _save_summary(
        args,
        native_group_counts,
        grid_group_counts,
        ownership_group_counts,
        len(groups),
    )
    validation = summary["protocol"]["validation"]
    if validation and any(value is False for value in validation.values()):
        raise RuntimeError(f"Controlled ownership invariant failed: {validation}")
    metrics = summary["metrics"]
    print(f"AP-10K {args.setting} pairs: {metrics['pair_count']}")
    for alpha in ("0.01", "0.05", "0.10"):
        print(
            f"Per-image PCK@{alpha}: {metrics['per_image_pck'][alpha] * 100:.2f} "
            f"(paper {PAPER_FLUX_PER_IMAGE[args.setting][alpha] * 100:.2f})"
        )
    if args.diagnostics:
        print(f"Grid {args.diagnostic_grid} baseline:")
        for alpha in ("0.01", "0.05", "0.10"):
            print(
                f"Per-image PCK@{alpha}: "
                f"{summary['grid_metrics']['per_image_pck'][alpha] * 100:.2f}"
            )
        print(f"Controlled validation: {validation}")
        for k, values in summary["ownership"].items():
            print(
                f"K={k}: owner={values['owner_candidate_recall']:.4f}, "
                f"global={values['global_union_recall']:.4f}, "
                f"failure_transferable={values['failure_transferable_rate']:.4f}, "
                f"strict={values['failure_strict_transferable_rate']:.4f}, "
                f"budget_control={values['failure_global_not_budget_matched_owner_rate']:.4f}, "
                f"combined={values['failure_strict_global_not_budget_matched_owner_rate']:.4f}"
            )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument(
        "--setting",
        choices=("intra-species", "cross-species", "cross-family"),
        required=True,
    )
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--img_size", type=int, default=960)
    parser.add_argument("--t", type=int, default=260)
    parser.add_argument("--k", type=int, default=28)
    parser.add_argument("--ensemble_size", type=int, default=8)
    parser.add_argument("--match_channel_chunk", type=int, default=256)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt_mode", choices=("species", "generic", "empty"), default="species")
    parser.add_argument("--cd", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--diagnostics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--diagnostic_grid", type=int, default=60)
    parser.add_argument("--topk", type=int, nargs="+", default=[1, 5, 10, 20, 50])
    parser.add_argument("--max_groups", type=int, default=0)
    parser.add_argument("--max_pairs_per_group", type=int, default=0)
    parser.add_argument(
        "--pair_sample_seed",
        type=int,
        default=2027,
        help="deterministic within-group hash sampling seed when a pair limit is used",
    )
    parser.add_argument("--resume", action="store_true")
    return parser


if __name__ == "__main__":
    evaluate(build_parser().parse_args())

"""Candidate-ownership diagnostic on a parity-verified DINOv2 SPair baseline.

This is an analysis evaluator, not a correspondence method. It never uses
ground truth to change the nearest-neighbour prediction or candidate ranking.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm

from dino_v2_spair import (
    PAPER_DINO_ALL_IMAGE,
    PAPER_DINO_ALL_POINT,
    CategoryMetrics,
    DINOConfig,
    controlled_candidate_rows,
    cosine_similarity_scores,
    patch_indices_to_points,
    pck_hits,
    square_canvas_geometry,
    transform_points_to_canvas,
)
from eval_spair_dinov2 import Extractor, build_extractor, category_image_names, discover_pairs


def empty_counts(ks: tuple[int, ...]) -> dict[int, dict[str, int | float]]:
    return {
        k: defaultdict(int)
        for k in ks
    }


def update_counts(
    counts: dict[int, dict[str, int | float]],
    rows: list[dict[str, int | float]],
    baseline_hits: torch.Tensor,
    ks: tuple[int, ...],
) -> None:
    for row, baseline_hit in zip(rows, baseline_hits.bool().cpu().tolist()):
        for k in ks:
            values = counts[k]
            owner = row[f"owner_candidate_hit@{k}"]
            other = row[f"other_source_candidate_hit@{k}"]
            union = row[f"global_union_candidate_hit@{k}"]
            strict_other = row[f"strict_other_source_candidate_hit@{k}"]
            strict_union = row[f"strict_global_union_candidate_hit@{k}"]
            budget_owner = row[f"budget_matched_owner_candidate_hit@{k}"]
            global_not_budget = row[f"global_not_budget_owner_hit@{k}"]
            strict_global_not_budget = row[f"strict_global_not_budget_owner_hit@{k}"]
            source_count = row[f"proposal_source_count@{k}"]
            values["points"] += 1
            values["owner"] += owner
            values["other"] += other
            values["global"] += union
            values["strict_other"] += strict_other
            values["strict_global"] += strict_union
            values["budget_owner"] += budget_owner
            values["global_not_budget"] += global_not_budget
            values["strict_global_not_budget"] += strict_global_not_budget
            values["unique_candidates"] += row[f"global_unique_candidate_count@{k}"]
            values["random_expected"] += row[f"random_union_expected_hit@{k}"]
            if union:
                values["global_hits"] += 1
                values["source_support_on_global"] += source_count
                values["multi_source_global"] += int(source_count >= 2)
            if not baseline_hit:
                values["failures"] += 1
                values["failure_owner"] += owner
                values["failure_global"] += union
                values["failure_transferable"] += int(union and not owner)
                values["failure_strict_transferable"] += int(strict_other and not owner)
                values["failure_budget_owner"] += budget_owner
                values["failure_global_not_budget"] += global_not_budget
                values["failure_strict_global_not_budget"] += strict_global_not_budget
                values["failure_random_expected"] += row[f"random_union_expected_hit@{k}"]
                if union:
                    values["failure_global_hits"] += 1
                    values["failure_source_support_on_global"] += source_count
                    values["failure_multi_source_global"] += int(source_count >= 2)


def ratios(counts: dict[int, dict[str, int | float]], ks: tuple[int, ...]) -> dict[str, dict[str, float | int]]:
    output: dict[str, dict[str, float | int]] = {}
    for k in ks:
        values = counts[k]
        points = max(values["points"], 1)
        failures = max(values["failures"], 1)
        global_hits = max(values["global_hits"], 1)
        failure_global_hits = max(values["failure_global_hits"], 1)
        output[str(k)] = {
            "point_count": values["points"],
            "baseline_failure_count": values["failures"],
            "owner_candidate_recall": values["owner"] / points,
            "other_source_transfer": values["other"] / points,
            "global_union_recall": values["global"] / points,
            "global_minus_owner": (values["global"] - values["owner"]) / points,
            "strict_other_source_recall": values["strict_other"] / points,
            "strict_global_union_recall": values["strict_global"] / points,
            "budget_matched_owner_recall": values["budget_owner"] / points,
            "global_minus_budget_matched_owner": (values["global"] - values["budget_owner"]) / points,
            "global_not_budget_matched_owner_rate": values["global_not_budget"] / points,
            "strict_global_not_budget_matched_owner_rate": values["strict_global_not_budget"] / points,
            "mean_unique_global_candidates": values["unique_candidates"] / points,
            "random_union_expected_recall": values["random_expected"] / points,
            "global_excess_over_random": (values["global"] - values["random_expected"]) / points,
            "mean_proposal_source_count_on_global_hits": values["source_support_on_global"] / global_hits,
            "multi_source_rate_on_global_hits": values["multi_source_global"] / global_hits,
            "failure_owner_candidate_recall": values["failure_owner"] / failures,
            "failure_global_union_recall": values["failure_global"] / failures,
            "failure_transferable_rate": values["failure_transferable"] / failures,
            "failure_strict_transferable_rate": values["failure_strict_transferable"] / failures,
            "failure_budget_matched_owner_recall": values["failure_budget_owner"] / failures,
            "failure_global_not_budget_matched_owner_rate": values["failure_global_not_budget"] / failures,
            "failure_strict_global_not_budget_matched_owner_rate": (
                values["failure_strict_global_not_budget"] / failures
            ),
            "failure_random_union_expected_recall": values["failure_random_expected"] / failures,
            "failure_global_excess_over_random": (
                values["failure_global"] - values["failure_random_expected"]
            ) / failures,
            "failure_mean_proposal_source_count_on_global_hits": (
                values["failure_source_support_on_global"] / failure_global_hits
            ),
            "failure_multi_source_rate_on_global_hits": (
                values["failure_multi_source_global"] / failure_global_hits
            ),
        }
    return output


def evaluate(args: argparse.Namespace, extractor: Extractor | None = None) -> dict[str, Any]:
    ks = tuple(sorted(set(args.topk)))
    if not ks or min(ks) < 1:
        raise ValueError("--topk values must be positive")
    config = DINOConfig(args.model_name, args.hub_model, 11, 14, 840)
    if config.hub_model != "dinov2_vitb14" or "large" in config.model_name.lower():
        raise ValueError("The verified diagnostic requires DINOv2 ViT-B/14")

    dataset_path = Path(args.dataset_path)
    categories, category_pairs = discover_pairs(dataset_path, args.max_pairs_per_cat)
    missing = [category for category in categories if not category_pairs[category]]
    if missing:
        raise RuntimeError(f"No SPair test pairs found for categories: {missing}")
    own_extractor = extractor is None
    extractor = extractor or build_extractor(args, config)
    device = torch.device(args.device)
    baseline = {category: CategoryMetrics() for category in categories}
    total_counts = empty_counts(ks)
    category_counts = {category: empty_counts(ks) for category in categories}
    point_records: list[dict[str, Any]] = []

    try:
        for category in categories:
            pair_paths = category_pairs[category]
            image_root = dataset_path / "JPEGImages" / category
            features: dict[str, torch.Tensor] = {}
            for image_name in tqdm(category_image_names(pair_paths), desc=f"features {category}", leave=False):
                with Image.open(image_root / image_name) as image:
                    features[image_name] = extractor(image)

            for pair_path in tqdm(pair_paths, desc=f"diagnose {category}"):
                data = json.loads(pair_path.read_text(encoding="utf-8"))
                src_h, src_w = int(data["src_imsize"][1]), int(data["src_imsize"][0])
                trg_h, trg_w = int(data["trg_imsize"][1]), int(data["trg_imsize"][0])
                src_points = transform_points_to_canvas(data["src_kps"], src_h, src_w, 840).to(device)
                trg_points = transform_points_to_canvas(data["trg_kps"], trg_h, trg_w, 840).to(device)
                scores = cosine_similarity_scores(
                    features[data["src_imname"]].to(device),
                    features[data["trg_imname"]].to(device),
                    src_points,
                    840,
                )
                stride = 14.0
                predictions = patch_indices_to_points(scores.argmax(dim=1), 60, stride)
                bbox = data["trg_bndbox"]
                threshold = max(bbox[3] - bbox[1], bbox[2] - bbox[0]) * square_canvas_geometry(
                    trg_h, trg_w, 840
                )[0]
                hits = pck_hits(predictions, trg_points, threshold)
                baseline[category].update(hits)
                rows = controlled_candidate_rows(
                    scores.detach().float().cpu(),
                    trg_points.detach().cpu(),
                    threshold,
                    60,
                    ks,
                    stride,
                )
                update_counts(total_counts, rows, hits, ks)
                update_counts(category_counts[category], rows, hits, ks)
                if args.output_csv:
                    for index, (row, hit) in enumerate(zip(rows, hits.bool().cpu().tolist())):
                        point_records.append(
                            {"category": category, "pair": pair_path.name, "point_index": index,
                             "baseline_hit": int(hit), **{key: value for key, value in row.items() if key != "point_index"}}
                        )

            del features
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        if own_extractor:
            extractor.close()

    all_pair_scores = [score for category in categories for score in baseline[category].pair_scores]
    total_correct = sum(value.correct for value in baseline.values())
    total_points = sum(value.total for value in baseline.values())
    all_per_image = sum(all_pair_scores) / len(all_pair_scores)
    all_per_point = total_correct / total_points
    ownership_summary = ratios(total_counts, ks)
    validation = {
        "owner_at_1_equals_baseline": abs(ownership_summary["1"]["owner_candidate_recall"] - all_per_point) < 1e-12
        if 1 in ks else None,
        "failure_owner_at_1_is_zero": ownership_summary["1"]["failure_owner_candidate_recall"] == 0.0
        if 1 in ks else None,
        "strict_is_bounded_by_global": all(
            ownership_summary[str(k)]["strict_global_union_recall"]
            <= ownership_summary[str(k)]["global_union_recall"]
            for k in ks
        ),
        "budget_owner_contains_owner_topk": all(
            ownership_summary[str(k)]["budget_matched_owner_recall"]
            >= ownership_summary[str(k)]["owner_candidate_recall"]
            for k in ks
        ),
    }
    if any(value is False for value in validation.values()):
        raise RuntimeError(f"Controlled ownership invariant failed: {validation}")
    summary: dict[str, Any] = {
        "protocol": {
            "diagnostic_version": "controlled-v2",
            "baseline": "parity-verified DINOv2 ViT-B/14 block-11 token NN",
            "feature_cache": "category-scoped memory only",
            "ground_truth_changes_ranking": False,
            "interpretation": "diagnostic only; not a matching method",
            "controls": [
                "strict target-GT-region non-overlap",
                "exact unique-candidate budget matched owner ranking",
                "target-region proposal-source hubness",
                "uniform random union expectation",
            ],
            "validation": validation,
        },
        "config": vars(args),
        "baseline": {
            "all_per_image_pck@0.1": all_per_image,
            "all_per_point_pck@0.1": all_per_point,
            "paper_all_per_image": PAPER_DINO_ALL_IMAGE,
            "paper_all_per_point": PAPER_DINO_ALL_POINT,
        },
        "ownership": ownership_summary,
        "categories": {
            category: {
                "baseline_per_image_pck@0.1": baseline[category].per_image,
                "baseline_per_point_pck@0.1": baseline[category].per_point,
                "ownership": ratios(category_counts[category], ks),
            }
            for category in categories
        },
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.output_csv:
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(point_records[0]) if point_records else ["category"]
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(point_records)

    print(f"Baseline All per image PCK@0.1: {100 * all_per_image:.2f}")
    print(f"Baseline All per point PCK@0.1: {100 * all_per_point:.2f}")
    for k, values in summary["ownership"].items():
        print(
            f"K={k}: owner={values['owner_candidate_recall']:.4f}, "
            f"global={values['global_union_recall']:.4f}, "
            f"failure_transferable={values['failure_transferable_rate']:.4f}, "
            f"strict={values['failure_strict_transferable_rate']:.4f}, "
            f"budget_control={values['failure_global_not_budget_matched_owner_rate']:.4f}"
        )
    print(f"Saved ownership summary to: {output_json}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--backend", choices=("transformers", "torch_hub"), default="transformers")
    parser.add_argument("--model_name", default="facebook/dinov2-base")
    parser.add_argument("--hub_model", default="dinov2_vitb14")
    parser.add_argument("--model_repo", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--max_pairs_per_cat", type=int, default=0)
    parser.add_argument("--topk", type=int, nargs="+", default=[1, 5, 10, 20, 50])
    return parser


if __name__ == "__main__":
    evaluate(build_parser().parse_args())

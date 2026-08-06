"""Audit label-free shared prototypes over FLUX candidate identity token sketches.

The audit fits cosine prototypes without PCK/keypoint identity labels on outer
training categories, then ranks fixed attention top-20 candidates on held-out
categories.  PCK is read only after scoring.  This is a mechanism diagnostic,
not a benchmark method claim.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from analyze_identity_decodability import (
    _load_torch,
    _validate_shard,
    category_folds,
    rank_metrics,
)


SOURCE_GROUP = "source_identity_token_sketch"
CANDIDATE_GROUP = "candidate_identity_token_sketch"


def _read_manifest(path: str) -> list[str]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    shards = payload.get("shards") if isinstance(payload, dict) else None
    if not isinstance(shards, list) or not shards:
        raise ValueError("prototype audit manifest must contain non-empty shards")
    root = os.path.dirname(os.path.abspath(path))
    return [value if os.path.isabs(value) else os.path.join(root, value) for value in shards]


def load_prototype_dataset(shard_paths: Sequence[str]) -> dict[str, Any]:
    sources = []
    candidates = []
    hits = []
    baseline_hits = []
    categories = []
    pairs = []
    dimensions = set()
    candidate_counts = set()
    for path in shard_paths:
        shard = _load_torch(path)
        _validate_shard(shard, path)
        groups = shard["feature_groups"]
        missing = [name for name in (SOURCE_GROUP, CANDIDATE_GROUP) if name not in groups]
        if missing:
            raise ValueError(
                f"shard {path} lacks {missing}; rerun identity decodability extraction "
                "with a revision that exports identity token sketches"
            )
        source = groups[SOURCE_GROUP].float()
        candidate = groups[CANDIDATE_GROUP].float()
        if source.shape != candidate.shape or source.ndim != 3:
            raise ValueError(f"source/candidate prototype sketches do not align: {path}")
        if not torch.allclose(source[:, :1], source[:, -1:], atol=2.0e-3, rtol=2.0e-3):
            raise ValueError(f"source prototype sketch is not candidate-invariant: {path}")
        query_count, candidate_count, dimension = map(int, candidate.shape)
        dimensions.add(dimension)
        candidate_counts.add(candidate_count)
        sources.append(source[:, 0].cpu())
        candidates.append(candidate.cpu())
        hits.append(shard["candidate_hits"].bool().cpu())
        baseline = shard.get("baseline_hits")
        if not isinstance(baseline, torch.Tensor) or baseline.numel() != query_count:
            raise ValueError(f"baseline labels do not align: {path}")
        baseline_hits.append(baseline.bool().reshape(-1).cpu())
        category = str(shard.get("category", ""))
        pair = str(shard.get("pair_id", path))
        categories.extend([category] * query_count)
        pairs.extend([pair] * query_count)
    if len(dimensions) != 1 or len(candidate_counts) != 1:
        raise ValueError("prototype sketch dimensions drift across shards")
    return {
        "sources": torch.cat(sources, dim=0),
        "candidates": torch.cat(candidates, dim=0),
        "hits": torch.cat(hits, dim=0).numpy(),
        "baseline_hits": torch.cat(baseline_hits, dim=0).numpy(),
        "categories": np.asarray(categories, dtype=object),
        "pairs": np.asarray(pairs, dtype=object),
        "dimension": dimensions.pop(),
        "candidate_count": candidate_counts.pop(),
    }


def _deterministic_sample(values: torch.Tensor, maximum: int, seed: int) -> torch.Tensor:
    if values.shape[0] <= maximum:
        return values
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    indices = torch.randperm(values.shape[0], generator=generator)[:maximum]
    return values.index_select(0, indices)


def fit_cosine_prototypes(
    values: torch.Tensor,
    *,
    prototype_count: int,
    iterations: int,
    seed: int,
    device: str,
    max_fit_tokens: int,
) -> torch.Tensor:
    """Fit balanced-agnostic cosine k-means; no correspondence labels enter."""

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("prototype audit requested cuda but CUDA is unavailable")
    values = _deterministic_sample(values.float().cpu(), max_fit_tokens, seed)
    values = F.normalize(values, dim=1, eps=1.0e-6).to(device)
    if values.shape[0] < prototype_count:
        raise ValueError("prototype count exceeds available fit tokens")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    initial = torch.randperm(values.shape[0], generator=generator)[:prototype_count]
    centers = values.index_select(0, initial.to(values.device)).clone()
    for iteration in range(max(1, int(iterations))):
        assignments = torch.empty(values.shape[0], dtype=torch.long, device=values.device)
        for start in range(0, values.shape[0], 8192):
            chunk = values[start:start + 8192]
            assignments[start:start + chunk.shape[0]] = (chunk @ centers.t()).argmax(dim=1)
        sums = torch.zeros_like(centers)
        sums.index_add_(0, assignments, values)
        counts = torch.bincount(assignments, minlength=prototype_count).to(values.dtype)
        empty = counts == 0
        if bool(empty.any()):
            replacement = torch.arange(int(empty.sum()), device=values.device)
            replacement = (replacement * 104729 + seed + iteration).remainder(values.shape[0])
            sums[empty] = values.index_select(0, replacement)
            counts[empty] = 1.0
        centers = F.normalize(sums / counts.unsqueeze(1), dim=1, eps=1.0e-6)
    return centers.cpu()


def prototype_agreement_scores(
    sources: torch.Tensor,
    candidates: torch.Tensor,
    centers: torch.Tensor,
    *,
    temperature: float,
    device: str,
) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("prototype temperature must be positive")
    centers = F.normalize(centers.float(), dim=1, eps=1.0e-6).to(device)
    output = []
    for start in range(0, sources.shape[0], 512):
        source = F.normalize(sources[start:start + 512].float(), dim=1, eps=1.0e-6).to(device)
        candidate = F.normalize(
            candidates[start:start + 512].float(), dim=2, eps=1.0e-6
        ).to(device)
        source_assignment = torch.softmax((source @ centers.t()) / temperature, dim=1)
        candidate_assignment = torch.softmax(
            torch.einsum("qkd,pd->qkp", candidate, centers) / temperature,
            dim=2,
        )
        score = torch.einsum("qp,qkp->qk", source_assignment, candidate_assignment)
        output.append(score.cpu())
    return torch.cat(output, dim=0).numpy().astype(np.float64, copy=False)


def direct_cosine_scores(sources: torch.Tensor, candidates: torch.Tensor) -> np.ndarray:
    source = F.normalize(sources.float(), dim=1, eps=1.0e-6)
    candidate = F.normalize(candidates.float(), dim=2, eps=1.0e-6)
    return torch.einsum("qd,qkd->qk", source, candidate).numpy().astype(np.float64, copy=False)


def _baseline_metrics(scores: np.ndarray, dataset: dict[str, Any]) -> dict[str, Any]:
    hits = dataset["hits"]
    selected = np.take_along_axis(hits, np.argsort(-scores, axis=1)[:, :1], axis=1)[:, 0]
    baseline = dataset["baseline_hits"]
    rescued = (~baseline) & selected
    harmed = baseline & (~selected)
    return {
        "baseline_top1": float(baseline.mean()),
        "selected_top1": float(selected.mean()),
        "rescued_vs_baseline": int(rescued.sum()),
        "harmed_vs_baseline": int(harmed.sum()),
        "net_correct_vs_baseline": int(rescued.sum() - harmed.sum()),
        "net_pck_vs_baseline": float(selected.mean() - baseline.mean()),
    }


def analyze_shared_identity_prototypes(
    shard_paths: Sequence[str],
    *,
    output_path: str,
    fold_count: int = 3,
    seed: int = 2027,
    prototype_count: int = 64,
    iterations: int = 20,
    temperature: float = 0.07,
    max_fit_tokens: int = 50000,
    device: str = "cpu",
) -> dict[str, Any]:
    dataset = load_prototype_dataset(shard_paths)
    categories = dataset["categories"]
    folds = category_folds(categories.tolist(), fold_count, seed)
    score_names = ("direct_token_cosine", "source_prototypes", "source_candidate_prototypes")
    predictions = {
        name: np.full(dataset["hits"].shape, np.nan, dtype=np.float64)
        for name in score_names
    }
    predictions["direct_token_cosine"] = direct_cosine_scores(
        dataset["sources"], dataset["candidates"]
    )
    fold_records = []
    for fold_index, test_categories in enumerate(folds):
        test_mask = np.isin(categories, np.asarray(test_categories, dtype=object))
        train_mask = ~test_mask
        train_sources = dataset["sources"][torch.from_numpy(train_mask)]
        train_candidates = dataset["candidates"][torch.from_numpy(train_mask)]
        fold_seed = seed + 1009 * fold_index
        source_centers = fit_cosine_prototypes(
            train_sources,
            prototype_count=prototype_count,
            iterations=iterations,
            seed=fold_seed,
            device=device,
            max_fit_tokens=max_fit_tokens,
        )
        both_values = torch.cat((train_sources, train_candidates.reshape(-1, train_candidates.shape[2])), dim=0)
        both_centers = fit_cosine_prototypes(
            both_values,
            prototype_count=prototype_count,
            iterations=iterations,
            seed=fold_seed + 313,
            device=device,
            max_fit_tokens=max_fit_tokens,
        )
        test_indices = np.flatnonzero(test_mask)
        test_sources = dataset["sources"][torch.from_numpy(test_mask)]
        test_candidates = dataset["candidates"][torch.from_numpy(test_mask)]
        predictions["source_prototypes"][test_mask] = prototype_agreement_scores(
            test_sources,
            test_candidates,
            source_centers,
            temperature=temperature,
            device=device,
        )
        predictions["source_candidate_prototypes"][test_mask] = prototype_agreement_scores(
            test_sources,
            test_candidates,
            both_centers,
            temperature=temperature,
            device=device,
        )
        fold_records.append({
            "fold": fold_index,
            "train_categories": sorted({str(value) for value in categories[train_mask]}),
            "test_categories": sorted(str(value) for value in test_categories),
            "test_points": int(test_indices.size),
        })
    results = {}
    for name, scores in predictions.items():
        if not bool(np.isfinite(scores).all()):
            raise RuntimeError(f"prototype scores do not cover all queries: {name}")
        results[name] = {
            "attention_metrics": rank_metrics(
                scores,
                dataset["hits"],
                dataset["baseline_hits"],
                categories,
                dataset["pairs"],
            ),
            "baseline_metrics": _baseline_metrics(scores, dataset),
        }
    best_name = max(results, key=lambda name: results[name]["baseline_metrics"]["selected_top1"])
    best_top1 = float(results[best_name]["baseline_metrics"]["selected_top1"])
    payload = {
        "audit": "shared_identity_prototype_identifiability",
        "protocol": {
            "fit_labels": "none",
            "pck_access": "metrics_only_after_candidate_scoring",
            "outer_split": "category_held_out",
            "candidate_source": "fixed_mutual_cross_attention_top20",
            "source_query_locations": "benchmark query inputs; not identity labels",
            "method_pck_claim": False,
            "prototype_count": int(prototype_count),
            "iterations": int(iterations),
            "temperature": float(temperature),
            "seed": int(seed),
            "device": str(device),
        },
        "data_contract": {
            "points": int(dataset["hits"].shape[0]),
            "candidate_count": int(dataset["candidate_count"]),
            "sketch_dimension": int(dataset["dimension"]),
            "categories": sorted({str(value) for value in categories.tolist()}),
        },
        "folds": fold_records,
        "scores": results,
        "mechanism_decision": {
            "best_score": best_name,
            "best_category_heldout_top1": best_top1,
            "reaches_pair20_target_75": bool(best_top1 >= 0.75),
            "interpretation": (
                "A positive result means pair-conditioned absolute FLUX token states admit a shared, label-free "
                "identity partition across held-out categories. A negative result rejects shared cosine prototypes "
                "and must not be repaired by PCK-tuned routing."
            ),
        },
    }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload["mechanism_decision"], indent=2))
    for name, result in results.items():
        print(name, json.dumps(result["baseline_metrics"], indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--fold_count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--prototype_count", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--max_fit_tokens", type=int, default=50000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    analyze_shared_identity_prototypes(
        _read_manifest(args.manifest),
        output_path=args.output_json,
        fold_count=args.fold_count,
        seed=args.seed,
        prototype_count=args.prototype_count,
        iterations=args.iterations,
        temperature=args.temperature,
        max_fit_tokens=args.max_fit_tokens,
        device=args.device,
    )


if __name__ == "__main__":
    main()

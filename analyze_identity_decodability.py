"""Category-held-out probes for FJSAR candidate identity diagnostics.

The probes in this module are deliberately not matchers.  Ground-truth labels
are consumed only on outer-training categories; every reported prediction is
made on categories that were absent from fitting and preprocessing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from typing import Any, Iterable, Sequence

import numpy as np
import torch


FORMAT_VERSION = 1
PROBE_FEATURE_GROUPS = {
    "attention_only": ("proposal_attention", "attention_aggregate"),
    "qk_only": ("proposal_attention", "qk_expert"),
    "value_only": ("value_expert",),
    "token_state_only": ("token_state",),
    "channel_state_only": ("channel_state_sketch",),
    "all_internal": (
        "proposal_attention",
        "attention_aggregate",
        "qk_expert",
        "value_expert",
        "token_state",
        "channel_state_sketch",
    ),
    "stable_internal": (
        "proposal_attention",
        "attention_aggregate",
        "qk_expert",
        "value_expert",
        "token_state",
    ),
    "native_plus_stable_internal": (
        "proposal_attention",
        "attention_aggregate",
        "qk_expert",
        "value_expert",
        "token_state",
        "native_control",
    ),
    "native_control": ("native_control",),
    "geometry_control": ("geometry_control",),
}


def _load_torch(path: str) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise ValueError(f"identity decodability shard is not a dict: {path}")
    if int(value.get("format_version", -1)) != FORMAT_VERSION:
        raise ValueError(f"unsupported identity decodability shard version: {path}")
    return value


def _hash_order(value: str, seed: int) -> str:
    return hashlib.sha256(f"{int(seed)}|{value}".encode("utf-8")).hexdigest()


def category_folds(categories: Iterable[str], fold_count: int, seed: int) -> list[list[str]]:
    ordered = sorted({str(value) for value in categories}, key=lambda value: _hash_order(value, seed))
    if len(ordered) < 2:
        raise ValueError("category-held-out probing requires at least two categories")
    fold_count = max(2, min(int(fold_count), len(ordered)))
    folds = [[] for _ in range(fold_count)]
    for index, category in enumerate(ordered):
        folds[index % fold_count].append(category)
    return folds


def _validate_shard(shard: dict[str, Any], path: str) -> None:
    feature_groups = shard.get("feature_groups")
    hits = shard.get("candidate_hits")
    if not isinstance(feature_groups, dict) or not isinstance(hits, torch.Tensor):
        raise ValueError(f"identity decodability shard lacks features/labels: {path}")
    if hits.ndim != 2 or hits.shape[0] == 0 or hits.shape[1] == 0:
        raise ValueError(f"invalid candidate hit tensor in {path}")
    for name, value in feature_groups.items():
        if not isinstance(value, torch.Tensor) or value.ndim != 3:
            raise ValueError(f"feature group {name} is not [point,candidate,feature] in {path}")
        if tuple(value.shape[:2]) != tuple(hits.shape):
            raise ValueError(f"feature group {name} does not align with labels in {path}")
        if not bool(torch.isfinite(value.float()).all()):
            if name != "channel_state_sketch":
                raise ValueError(f"feature group {name} contains non-finite values in {path}")
    metadata = shard.get("metadata", {})
    if bool(metadata.get("gt_used_for_features", True)):
        raise ValueError(f"shard does not satisfy annotation-free feature contract: {path}")


def _load_dataset(
    shard_paths: Sequence[str],
    feature_groups: Sequence[str],
) -> dict[str, Any]:
    features = []
    hits = []
    categories = []
    pairs = []
    baseline_hits = []
    family_dimensions: dict[str, int] = {}
    nonfinite_counts: dict[str, int] = defaultdict(int)
    candidate_count = None
    for path in shard_paths:
        shard = _load_torch(path)
        _validate_shard(shard, path)
        groups = shard["feature_groups"]
        missing = [name for name in feature_groups if name not in groups]
        if missing:
            raise ValueError(f"shard {path} lacks feature groups {missing}")
        current = []
        for name in feature_groups:
            raw_value = groups[name].float()
            nonfinite_counts[name] += int((~torch.isfinite(raw_value)).sum().item())
            value = torch.nan_to_num(
                raw_value,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).numpy()
            current.append(value)
            dimension = int(value.shape[2])
            if name in family_dimensions and family_dimensions[name] != dimension:
                raise ValueError(f"feature dimension drift for {name}")
            family_dimensions[name] = dimension
        current_features = np.concatenate(current, axis=2)
        current_hits = shard["candidate_hits"].bool().numpy()
        current_candidate_count = int(current_hits.shape[1])
        if candidate_count is None:
            candidate_count = current_candidate_count
        elif candidate_count != current_candidate_count:
            raise ValueError("candidate count differs across identity decodability shards")
        query_count = int(current_hits.shape[0])
        category = str(shard.get("category", ""))
        pair_id = str(shard.get("pair_id", path))
        if not category:
            raise ValueError(f"shard category is empty: {path}")
        current_baseline = shard.get("baseline_hits")
        if not isinstance(current_baseline, torch.Tensor) or current_baseline.numel() != query_count:
            raise ValueError(f"shard baseline hits do not align: {path}")
        features.append(current_features)
        hits.append(current_hits)
        categories.extend([category] * query_count)
        pairs.extend([pair_id] * query_count)
        baseline_hits.append(current_baseline.bool().numpy().reshape(-1))
    if not features:
        raise ValueError("identity decodability manifest contains no shards")
    return {
        "features": np.concatenate(features, axis=0),
        "hits": np.concatenate(hits, axis=0),
        "categories": np.asarray(categories, dtype=object),
        "pairs": np.asarray(pairs, dtype=object),
        "baseline_hits": np.concatenate(baseline_hits, axis=0),
        "family_dimensions": family_dimensions,
        "nonfinite_counts": dict(nonfinite_counts),
        "candidate_count": int(candidate_count or 0),
    }


def _candidate_training_rows(hits: np.ndarray, query_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    recoverable = hits.any(axis=1)
    selected_queries = np.flatnonzero(query_mask & recoverable)
    if selected_queries.size == 0:
        raise ValueError("training split has no recoverable attention candidate groups")
    candidate_count = int(hits.shape[1])
    rows = (
        selected_queries[:, None] * candidate_count
        + np.arange(candidate_count, dtype=np.int64)[None, :]
    ).reshape(-1)
    labels = hits[selected_queries].reshape(-1).astype(np.int64)
    if np.unique(labels).size != 2:
        raise ValueError("training split does not contain both positive and negative candidates")
    return rows, labels


def _query_balanced_weights(hits: np.ndarray, selected_queries: np.ndarray) -> np.ndarray:
    selected_hits = hits[selected_queries]
    positive_count = selected_hits.sum(axis=1).clip(min=1)
    negative_count = (~selected_hits).sum(axis=1).clip(min=1)
    weights = np.where(
        selected_hits,
        0.5 / positive_count[:, None],
        0.5 / negative_count[:, None],
    )
    return weights.reshape(-1).astype(np.float64)


def _fit_linear_scores(
    features: np.ndarray,
    hits: np.ndarray,
    train_query_mask: np.ndarray,
    test_query_mask: np.ndarray,
    seed: int,
) -> np.ndarray:
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler

    train_rows, train_labels = _candidate_training_rows(hits, train_query_mask)
    candidate_count = int(hits.shape[1])
    selected_queries = np.unique(train_rows // candidate_count)
    train_weights = _query_balanced_weights(hits, selected_queries)
    flat = features.reshape(-1, features.shape[2])
    scaler = StandardScaler(copy=True)
    train_x = scaler.fit_transform(flat[train_rows]).astype(np.float32, copy=False)
    classifier = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        max_iter=2000,
        tol=1e-4,
        average=True,
        random_state=int(seed),
    )
    classifier.fit(train_x, train_labels, sample_weight=train_weights)
    test_queries = np.flatnonzero(test_query_mask)
    test_rows = (
        test_queries[:, None] * candidate_count
        + np.arange(candidate_count, dtype=np.int64)[None, :]
    ).reshape(-1)
    test_x = scaler.transform(flat[test_rows]).astype(np.float32, copy=False)
    scores = classifier.decision_function(test_x)
    return np.asarray(scores, dtype=np.float64).reshape(test_queries.size, candidate_count)


def _fit_mlp_scores(
    features: np.ndarray,
    hits: np.ndarray,
    train_query_mask: np.ndarray,
    test_query_mask: np.ndarray,
    seed: int,
) -> np.ndarray:
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    train_rows, train_labels = _candidate_training_rows(hits, train_query_mask)
    candidate_count = int(hits.shape[1])
    selected_queries = np.unique(train_rows // candidate_count)
    train_weights = _query_balanced_weights(hits, selected_queries)
    # MLPClassifier does not support sample_weight on older remote sklearn
    # versions.  Deterministic per-query resampling preserves equal query mass.
    rng = np.random.default_rng(int(seed))
    positive_rows = train_rows[train_labels == 1]
    negative_rows = train_rows[train_labels == 0]
    positive_weights = train_weights[train_labels == 1]
    negative_weights = train_weights[train_labels == 0]
    sample_count = max(len(positive_rows), len(negative_rows))
    sampled_positive = rng.choice(
        positive_rows,
        size=sample_count,
        replace=True,
        p=positive_weights / positive_weights.sum(),
    )
    sampled_negative = rng.choice(
        negative_rows,
        size=sample_count,
        replace=True,
        p=negative_weights / negative_weights.sum(),
    )
    sampled_rows = np.concatenate((sampled_positive, sampled_negative))
    sampled_labels = np.concatenate((
        np.ones(sample_count, dtype=np.int64),
        np.zeros(sample_count, dtype=np.int64),
    ))
    order = rng.permutation(sampled_rows.size)
    sampled_rows = sampled_rows[order]
    sampled_labels = sampled_labels[order]
    flat = features.reshape(-1, features.shape[2])
    scaler = StandardScaler(copy=True)
    train_x = scaler.fit_transform(flat[sampled_rows]).astype(np.float32, copy=False)
    classifier = MLPClassifier(
        hidden_layer_sizes=(128,),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=512,
        learning_rate_init=1e-3,
        max_iter=50,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        random_state=int(seed),
    )
    classifier.fit(train_x, sampled_labels)
    test_queries = np.flatnonzero(test_query_mask)
    test_rows = (
        test_queries[:, None] * candidate_count
        + np.arange(candidate_count, dtype=np.int64)[None, :]
    ).reshape(-1)
    test_x = scaler.transform(flat[test_rows]).astype(np.float32, copy=False)
    scores = classifier.predict_proba(test_x)[:, 1]
    return np.asarray(scores, dtype=np.float64).reshape(test_queries.size, candidate_count)


def _fit_torch_mlp_scores(
    features: np.ndarray,
    hits: np.ndarray,
    train_query_mask: np.ndarray,
    test_query_mask: np.ndarray,
    seed: int,
    device: str,
) -> np.ndarray:
    """Fit the diagnostic one-hidden-layer probe on an explicitly selected device."""

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("torch MLP probe requested cuda but CUDA is unavailable")
    train_rows, train_labels = _candidate_training_rows(hits, train_query_mask)
    candidate_count = int(hits.shape[1])
    selected_queries = np.unique(train_rows // candidate_count)
    train_weights = _query_balanced_weights(hits, selected_queries)
    rng = np.random.default_rng(int(seed))
    positive_rows = train_rows[train_labels == 1]
    negative_rows = train_rows[train_labels == 0]
    positive_weights = train_weights[train_labels == 1]
    negative_weights = train_weights[train_labels == 0]
    sample_count = max(len(positive_rows), len(negative_rows))
    sampled_positive = rng.choice(
        positive_rows,
        size=sample_count,
        replace=True,
        p=positive_weights / positive_weights.sum(),
    )
    sampled_negative = rng.choice(
        negative_rows,
        size=sample_count,
        replace=True,
        p=negative_weights / negative_weights.sum(),
    )
    sampled_rows = np.concatenate((sampled_positive, sampled_negative))
    sampled_labels = np.concatenate((
        np.ones(sample_count, dtype=np.float32),
        np.zeros(sample_count, dtype=np.float32),
    ))
    order = rng.permutation(sampled_rows.size)
    sampled_rows = sampled_rows[order]
    sampled_labels = sampled_labels[order]

    flat = features.reshape(-1, features.shape[2]).astype(np.float32, copy=False)
    train_raw = flat[sampled_rows]
    mean = train_raw.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = train_raw.std(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(scale, 1.0e-6)
    train_np = ((train_raw - mean) / scale).astype(np.float32, copy=False)
    val_count = max(1, int(round(0.1 * train_np.shape[0])))
    val_np = train_np[:val_count]
    val_labels_np = sampled_labels[:val_count]
    fit_np = train_np[val_count:]
    fit_labels_np = sampled_labels[val_count:]
    if fit_np.shape[0] == 0:
        fit_np, fit_labels_np = train_np, sampled_labels

    torch.manual_seed(int(seed))
    if device == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    run_device = torch.device(device)
    model = torch.nn.Sequential(
        torch.nn.Linear(int(features.shape[2]), 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 1),
    ).to(run_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    criterion = torch.nn.BCEWithLogitsLoss()
    fit_x = torch.from_numpy(fit_np).to(run_device)
    fit_y = torch.from_numpy(fit_labels_np).to(run_device)
    val_x = torch.from_numpy(val_np).to(run_device)
    val_y = torch.from_numpy(val_labels_np).to(run_device)
    best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    best_loss = float("inf")
    stale_epochs = 0
    batch_size = 512
    for epoch in range(50):
        model.train()
        permutation = torch.randperm(fit_x.shape[0], device=run_device)
        for start in range(0, fit_x.shape[0], batch_size):
            index = permutation[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(fit_x.index_select(0, index)).squeeze(1), fit_y.index_select(0, index))
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(criterion(model(val_x).squeeze(1), val_y).cpu())
        if validation_loss + 1.0e-5 < best_loss:
            best_loss = validation_loss
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= 5:
                break
    model.load_state_dict(best_state)

    test_queries = np.flatnonzero(test_query_mask)
    test_rows = (
        test_queries[:, None] * candidate_count
        + np.arange(candidate_count, dtype=np.int64)[None, :]
    ).reshape(-1)
    test_np = ((flat[test_rows] - mean) / scale).astype(np.float32, copy=False)
    test_x = torch.from_numpy(test_np).to(run_device)
    with torch.no_grad():
        scores = model(test_x).squeeze(1).cpu().numpy()
    return np.asarray(scores, dtype=np.float64).reshape(test_queries.size, candidate_count)


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _sample_se(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1) / math.sqrt(len(values)))


def rank_metrics(
    scores: np.ndarray,
    hits: np.ndarray,
    baseline_hits: np.ndarray,
    categories: np.ndarray,
    pairs: np.ndarray,
) -> dict[str, Any]:
    if scores.shape != hits.shape:
        raise ValueError("probe scores and candidate labels do not align")
    order = np.argsort(-scores, axis=1, kind="stable")
    ranked_hits = np.take_along_axis(hits, order, axis=1)
    selected = ranked_hits[:, 0]
    attention = hits[:, 0]
    oracle = hits.any(axis=1)
    attention_error_recoverable = (~attention) & oracle
    both_wrong = (~baseline_hits) & attention_error_recoverable
    recovered = (~attention) & selected
    harmed = attention & (~selected)
    pair_lifts: dict[str, list[float]] = defaultdict(list)
    for pair, lift in zip(pairs.tolist(), (selected.astype(float) - attention.astype(float)).tolist()):
        pair_lifts[str(pair)].append(float(lift))
    pair_equal_lifts = [_mean(values) for values in pair_lifts.values()]
    topk = {}
    for value in (1, 3, 5, 10, 20):
        width = min(int(value), int(hits.shape[1]))
        topk[str(value)] = float(ranked_hits[:, :width].any(axis=1).mean())
    category_breakdown = {}
    for category in sorted({str(value) for value in categories.tolist()}):
        mask = categories == category
        category_breakdown[category] = {
            "points": int(mask.sum()),
            "attention_top1": float(attention[mask].mean()),
            "probe_top1": float(selected[mask].mean()),
            "net_vs_attention": float((selected[mask].astype(float) - attention[mask]).mean()),
            "attention_top20_oracle": float(oracle[mask].mean()),
        }
    return {
        "points": int(hits.shape[0]),
        "attention_top1": float(attention.mean()),
        "attention_top20_oracle": float(oracle.mean()),
        "probe_top1": float(selected.mean()),
        "probe_topk": topk,
        "recovers_attention_top1_errors": int(recovered.sum()),
        "harms_attention_top1_correct": int(harmed.sum()),
        "net_correct_vs_attention_top1": int(recovered.sum() - harmed.sum()),
        "net_pck_vs_attention_top1": float(selected.mean() - attention.mean()),
        "attention_error_top20_hit_points": int(attention_error_recoverable.sum()),
        "attention_error_top20_hit_recovery_rate": (
            float(selected[attention_error_recoverable].mean())
            if bool(attention_error_recoverable.any()) else 0.0
        ),
        "both_wrong_top20_hit_points": int(both_wrong.sum()),
        "both_wrong_top20_hit_recovery_rate": (
            float(selected[both_wrong].mean()) if bool(both_wrong.any()) else 0.0
        ),
        "attention_correct_retention_rate": (
            float(selected[attention].mean()) if bool(attention.any()) else 0.0
        ),
        "pair_equal_net_lift_mean": _mean(pair_equal_lifts),
        "pair_clustered_net_lift_standard_error": _sample_se(pair_equal_lifts),
        "category_breakdown": category_breakdown,
    }


def _cross_validated_probe(
    dataset: dict[str, Any],
    folds: Sequence[Sequence[str]],
    model: str,
    seed: int,
    device: str = "cpu",
) -> dict[str, Any]:
    features = dataset["features"]
    hits = dataset["hits"]
    categories = dataset["categories"]
    predictions = np.full(hits.shape, np.nan, dtype=np.float64)
    fold_records = []
    for fold_index, test_categories in enumerate(folds):
        test_mask = np.isin(categories, np.asarray(test_categories, dtype=object))
        train_mask = ~test_mask
        if not bool(test_mask.any()) or not bool(train_mask.any()):
            raise ValueError("category fold produced an empty train or test split")
        fold_seed = int(seed) + 1009 * int(fold_index)
        if model == "linear":
            fold_scores = _fit_linear_scores(
                features,
                hits,
                train_mask,
                test_mask,
                fold_seed,
            )
        elif model == "mlp":
            fold_scores = _fit_mlp_scores(
                features,
                hits,
                train_mask,
                test_mask,
                fold_seed,
            )
        elif model == "torch_mlp":
            fold_scores = _fit_torch_mlp_scores(
                features,
                hits,
                train_mask,
                test_mask,
                fold_seed,
                device,
            )
        else:
            raise ValueError(f"unsupported identity probe model: {model}")
        predictions[test_mask] = fold_scores
        fold_metrics = rank_metrics(
            fold_scores,
            hits[test_mask],
            dataset["baseline_hits"][test_mask],
            categories[test_mask],
            dataset["pairs"][test_mask],
        )
        fold_records.append({
            "fold": int(fold_index),
            "train_categories": sorted({str(value) for value in categories[train_mask].tolist()}),
            "test_categories": sorted(str(value) for value in test_categories),
            "metrics": fold_metrics,
        })
    if not bool(np.isfinite(predictions).all()):
        raise RuntimeError("category-held-out predictions do not cover every query")
    return {
        "model": str(model),
        "metrics": rank_metrics(
            predictions,
            hits,
            dataset["baseline_hits"],
            categories,
            dataset["pairs"],
        ),
        "folds": fold_records,
    }


def analyze_identity_decodability(
    shard_paths: Sequence[str],
    *,
    output_path: str,
    fold_count: int = 3,
    seed: int = 2027,
    run_mlp: bool = True,
    probe_names: Sequence[str] | None = None,
    run_linear: bool = True,
    mlp_backend: str = "sklearn",
    device: str = "cpu",
) -> dict[str, Any]:
    if not shard_paths:
        raise ValueError("identity decodability analysis requires shard paths")
    categories = []
    contracts = []
    for path in shard_paths:
        shard = _load_torch(path)
        _validate_shard(shard, path)
        categories.append(str(shard.get("category", "")))
        metadata = shard.get("metadata", {})
        contracts.append({
            "path": str(path),
            "gt_used_for_features": bool(metadata.get("gt_used_for_features", True)),
            "gt_used_for_labels_only": bool(metadata.get("gt_used_for_labels_only", False)),
            "probe_is_matcher": bool(metadata.get("probe_is_matcher", True)),
            "native_fallback_used": bool(metadata.get("native_fallback_used", True)),
        })
        del shard
    folds = category_folds(categories, fold_count, seed)
    probes = {}

    dataset_contract = None
    selected_probe_names = list(probe_names or PROBE_FEATURE_GROUPS)
    unknown_probe_names = [name for name in selected_probe_names if name not in PROBE_FEATURE_GROUPS]
    if unknown_probe_names:
        raise ValueError(f"unknown identity probe names: {unknown_probe_names}")
    if not run_linear and not run_mlp:
        raise ValueError("identity decodability analysis needs linear or MLP probes")
    if mlp_backend not in {"sklearn", "torch"}:
        raise ValueError(f"unsupported MLP backend: {mlp_backend}")
    for probe_name in selected_probe_names:
        groups = PROBE_FEATURE_GROUPS[probe_name]
        dataset = _load_dataset(shard_paths, groups)
        if dataset_contract is None:
            dataset_contract = {
                "points": int(dataset["hits"].shape[0]),
                "candidate_count": int(dataset["candidate_count"]),
                "categories": sorted({str(value) for value in dataset["categories"].tolist()}),
            }
        else:
            for name, count in dataset["nonfinite_counts"].items():
                dataset_contract.setdefault("nonfinite_feature_values", {})[name] = (
                    int(dataset_contract.get("nonfinite_feature_values", {}).get(name, 0))
                    + int(count)
                )
        if dataset_contract is not None and "nonfinite_feature_values" not in dataset_contract:
            dataset_contract["nonfinite_feature_values"] = {
                name: int(count) for name, count in dataset["nonfinite_counts"].items()
            }
        if run_linear:
            result = _cross_validated_probe(dataset, folds, "linear", seed, device="cpu")
            result["feature_groups"] = list(groups)
            result["feature_dimensions"] = dataset["family_dimensions"]
            probes[f"linear_{probe_name}"] = result
        if run_mlp and (probe_names is not None or probe_name == "all_internal"):
            mlp_model = "torch_mlp" if mlp_backend == "torch" else "mlp"
            nonlinear = _cross_validated_probe(
                dataset,
                folds,
                mlp_model,
                seed + 7919,
                device=device,
            )
            nonlinear["feature_groups"] = list(groups)
            nonlinear["feature_dimensions"] = dataset["family_dimensions"]
            probes[f"{mlp_model}_{probe_name}"] = nonlinear
        del dataset

    internal_names = [
        name for name in probes
        if ("all_internal" in name or "stable_internal" in name)
        and "geometry" not in name
    ]
    if not internal_names:
        raise ValueError("identity decodability analysis produced no internal probe")
    best_name = max(
        internal_names,
        key=lambda name: float(probes[name]["metrics"]["probe_top1"]),
    )
    best_rate = float(probes[best_name]["metrics"]["probe_top1"])
    payload = {
        "audit": "candidate_identity_decodability",
        "protocol": {
            "outer_split": "category_held_out",
            "fold_count": int(len(folds)),
            "folds": folds,
            "seed": int(seed),
            "candidate_source": "fixed_mutual_cross_attention_top20",
            "training_labels": "candidate_pck_hit_on_outer_training_categories_only",
            "test_label_access_during_fit": False,
            "native_and_geometry_are_controls_only": True,
            "method_pck_claim": False,
            "linear_regularization": "SGDClassifier documented alpha=1e-4; no test tuning",
            "nonlinear_probe": "one-hidden-layer MLP(128), training-category early stopping",
            "probe_names": selected_probe_names,
            "run_linear": bool(run_linear),
            "mlp_backend": str(mlp_backend),
            "device": str(device),
        },
        "data_contract": dataset_contract,
        "contract_violations": {
            "gt_used_for_features": sum(int(item["gt_used_for_features"]) for item in contracts),
            "gt_not_restricted_to_labels": sum(int(not item["gt_used_for_labels_only"]) for item in contracts),
            "probe_marked_as_matcher": sum(int(item["probe_is_matcher"]) for item in contracts),
            "native_fallback_used": sum(int(item["native_fallback_used"]) for item in contracts),
        },
        "probes": probes,
        "mechanism_decision": {
            "best_internal_probe": best_name,
            "best_internal_category_heldout_top1": best_rate,
            "supervised_probe_reaches_80": bool(best_rate >= 0.80),
            "supervised_probe_reaches_90": bool(best_rate >= 0.90),
            "unsupervised_80_established_by_this_audit": False,
            "interpretation": (
                "Positive decodability proves candidate identity is present in the audited FLUX state families. "
                "Failure does not prove information-theoretic absence; it bounds linear and shallow nonlinear access "
                "under category-held-out evaluation. Reaching 80 with a supervised diagnostic establishes capacity, "
                "not an unsupervised method."
            ),
        },
    }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return payload


def _read_manifest(path: str) -> list[str]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    paths = payload.get("shards") if isinstance(payload, dict) else None
    if not isinstance(paths, list):
        raise ValueError("identity decodability manifest must contain a shards list")
    root = os.path.dirname(os.path.abspath(path))
    return [value if os.path.isabs(value) else os.path.join(root, value) for value in paths]


def main() -> None:
    parser = argparse.ArgumentParser(description="Category-held-out FJSAR identity probes")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--fold_count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--skip_mlp", action="store_true", default=False)
    parser.add_argument(
        "--mlp_only",
        action="store_true",
        default=False,
        help="Skip linear probes and run the selected MLP probe families only.",
    )
    parser.add_argument(
        "--torch_mlp",
        action="store_true",
        default=False,
        help="Use the explicit PyTorch MLP backend; combine with --device cuda for GPU execution.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Device for the optional PyTorch MLP backend; sklearn probes remain CPU-only.",
    )
    parser.add_argument(
        "--probe_names",
        nargs="+",
        choices=tuple(PROBE_FEATURE_GROUPS),
        default=None,
        help="Optional probe families; omitted runs the original complete linear audit.",
    )
    args = parser.parse_args()
    analyze_identity_decodability(
        _read_manifest(args.manifest),
        output_path=args.output_json,
        fold_count=args.fold_count,
        seed=args.seed,
        run_mlp=not args.skip_mlp,
        probe_names=args.probe_names,
        run_linear=not args.mlp_only,
        mlp_backend="torch" if args.torch_mlp else "sklearn",
        device=args.device,
    )


if __name__ == "__main__":
    main()

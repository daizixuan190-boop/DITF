"""Measure an optimistic, supervised upper bound for existing candidate evidence.

This is an offline diagnostic, not a proposed matcher.  It uses PCK only as an
analysis label and evaluates grouped out-of-fold candidate ranking.  The point
is to test whether the currently dumped candidate evidence could plausibly
reach the requested lift before adding an unsupervised objective.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold


def _read(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _record_key(record: dict[str, Any]) -> tuple[str, int]:
    return str(record.get("pair_json", "")), int(record.get("keypoint_index", -1))


def _pixel_key(value: Any) -> tuple[int, int] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return int(round(float(value[0]))), int(round(float(value[1])))
        except (TypeError, ValueError):
            return None
    return None


def _finite(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _candidate_features(candidate: dict[str, Any], prefix: str) -> dict[str, float]:
    features: dict[str, float] = {}
    for container_name in ("scores", "metrics"):
        container = candidate.get(container_name)
        if not isinstance(container, dict):
            continue
        for name, value in container.items():
            number = _finite(value)
            if number is not None:
                features[f"{prefix}:{name}"] = number
    return features


def _auxiliary_maps(paths: list[str]) -> dict[tuple[str, int], dict[tuple[int, int], dict[str, float]]]:
    merged: dict[tuple[str, int], dict[tuple[int, int], dict[str, float]]] = {}
    for path in paths:
        payload = _read(path)
        for record in payload.get("records", []):
            if not isinstance(record, dict):
                continue
            key = _record_key(record)
            by_pixel = merged.setdefault(key, {})
            for field, value in record.items():
                if not isinstance(value, dict) or not isinstance(value.get("candidates"), list):
                    continue
                prefix = field.removesuffix("_audit")
                for candidate in value["candidates"]:
                    if not isinstance(candidate, dict):
                        continue
                    pixel = _pixel_key(candidate.get("pixel"))
                    if pixel is None:
                        continue
                    row = by_pixel.setdefault(pixel, {})
                    row.update(_candidate_features(candidate, prefix))
    return merged


def _base_features(record: dict[str, Any], candidate: dict[str, Any]) -> dict[str, float]:
    features: dict[str, float] = {}
    for name in ("attention_score", "descriptor_score", "semantic_gap_to_native", "reciprocal_attention"):
        value = _finite(candidate.get(name))
        if value is not None:
            features[name] = value
    for name in ("rank_attention", "rank_descriptor"):
        value = _finite(candidate.get(name))
        if value is not None:
            features[name] = value
    # Coordinates are deliberately excluded: the diagnostic tests appearance/
    # correspondence evidence, not an object-specific spatial shortcut.
    return features


def _matrix(payload: dict[str, Any], auxiliary: dict[tuple[str, int], dict[tuple[int, int], dict[str, float]]]):
    rows: list[dict[str, Any]] = []
    groups: list[str] = []
    point_keys: list[tuple[str, int]] = []
    feature_names: set[str] = set()
    for record in payload.get("records", []):
        if not isinstance(record, dict):
            continue
        point_key = _record_key(record)
        aux_by_pixel = auxiliary.get(point_key, {})
        for candidate in record.get("proposals", []):
            if not isinstance(candidate, dict):
                continue
            pixel = _pixel_key(candidate.get("pixel"))
            if pixel is None:
                continue
            features = _base_features(record, candidate)
            features.update(aux_by_pixel.get(pixel, {}))
            if not features:
                continue
            feature_names.update(features)
            rows.append({"record": record, "candidate": candidate, "features": features})
            groups.append(str(record.get("pair_json", point_key[0])))
            point_keys.append(point_key)
    names = sorted(feature_names)
    x = np.asarray([[row["features"].get(name, 0.0) for name in names] for row in rows], dtype=np.float32)
    y = np.asarray([1 if row["candidate"].get("pck_hit") is True else 0 for row in rows], dtype=np.int64)
    return rows, x, y, np.asarray(groups, dtype=object), names, point_keys


def _rank_predictions(rows, probabilities: np.ndarray) -> dict[tuple[str, int], bool]:
    by_point: dict[tuple[str, int], list[tuple[float, bool]]] = defaultdict(list)
    for row, probability in zip(rows, probabilities):
        key = _record_key(row["record"])
        by_point[key].append((float(probability), bool(row["candidate"].get("pck_hit"))))
    return {key: max(items, key=lambda item: item[0])[1] for key, items in by_point.items()}


def _evaluate_fold(rows, probabilities: np.ndarray) -> dict[str, float]:
    selected = _rank_predictions(rows, probabilities)
    points = {key: row["record"] for row in rows for key in [_record_key(row["record"])]}
    total = len(points)
    baseline_correct = sum(bool(record.get("baseline_pck_hit")) for record in points.values())
    selected_correct = sum(selected.values())
    oracle_gap = sum(
        not bool(record.get("baseline_pck_hit")) and bool(record.get("attention_topk_pck_hit"))
        for record in points.values()
    )
    rescued = sum(
        not bool(record.get("baseline_pck_hit")) and selected.get(key, False)
        for key, record in points.items()
    )
    harmed = sum(
        bool(record.get("baseline_pck_hit")) and not selected.get(key, False)
        for key, record in points.items()
    )
    return {
        "points": float(total),
        "baseline_correct": float(baseline_correct),
        "baseline_rate": baseline_correct / max(1, total),
        "selected_correct": float(selected_correct),
        "selected_rate": selected_correct / max(1, total),
        "oracle_gap_points": float(oracle_gap),
        "rescued": float(rescued),
        "harmed": float(harmed),
        "net_delta": float(rescued - harmed),
        "oracle_gap_recall": rescued / max(1, oracle_gap),
    }


def analyze(
    payload: dict[str, Any],
    auxiliary_paths: list[str],
    folds: int,
    seed: int,
    group_by: str,
) -> dict[str, Any]:
    auxiliary = _auxiliary_maps(auxiliary_paths)
    rows, x, y, pair_groups, names, _ = _matrix(payload, auxiliary)
    if group_by == "category":
        groups = np.asarray([str(row["record"].get("category", "")) for row in rows], dtype=object)
    else:
        groups = pair_groups
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("at least two pair groups are required")
    splitter = GroupKFold(n_splits=min(folds, len(unique_groups)))
    oof = np.zeros(len(rows), dtype=np.float32)
    fold_summaries = []
    for fold, (train, test) in enumerate(splitter.split(x, y, groups), start=1):
        model = HistGradientBoostingClassifier(
            max_iter=160,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=seed + fold,
        )
        model.fit(x[train], y[train])
        oof[test] = model.predict_proba(x[test])[:, 1]
        fold_summaries.append({"fold": fold, **_evaluate_fold([rows[i] for i in test], oof[test])})
    overall = _evaluate_fold(rows, oof)
    return {
        "diagnostic": "grouped supervised upper bound over existing candidate evidence",
        "label_contract": "PCK labels are used only for offline falsification; this is not a proposed training method",
        "records": len(payload.get("records", [])),
        "candidate_rows": len(rows),
        "feature_count": len(names),
        "features": names,
        "auxiliary_files": auxiliary_paths,
        "group_by": group_by,
        "folds": fold_summaries,
        "oof": overall,
        "go_no_go": {
            "required_point_gain_to_75": 75.0 - float(payload.get("all", {}).get("point", 68.75704093128051)),
            "interpretation": "A weak OOF oracle-gap recall or substantial harm rejects the current evidence as a basis for an unsupervised pair head.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all_candidates_json", required=True)
    parser.add_argument("--auxiliary_json", action="append", default=[])
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--group_by", choices=("pair", "category"), default="pair")
    args = parser.parse_args()
    result = analyze(_read(args.all_candidates_json), args.auxiliary_json, args.folds, args.seed, args.group_by)
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result["oof"], indent=2))
    for fold in result["folds"]:
        print(json.dumps(fold, indent=2))


if __name__ == "__main__":
    main()

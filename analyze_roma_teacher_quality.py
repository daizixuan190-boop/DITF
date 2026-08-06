"""Audit whether frozen RoMa yields stable pseudo-targets for distillation.

This is a read-only, label-after-the-fact diagnostic.  It never chooses a
training threshold from PCK: confidence cutoffs are fixed by discovery feature
quantiles and transferred unchanged to heldout.  The purpose is to decide
whether a cached, larger RoMa-distillation experiment is justified at all.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np


QUANTILES = (0.50, 0.75, 0.90)
# Bidirectional warp error is in normalized endpoint coordinates.  These are
# semantic displacement tolerances (5%, 10%, 20%), not PCK-selected cutoffs.
FIXED_MAX_BIDIRECTIONAL_ERRORS = (0.05, 0.10, 0.20)


def _read(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _points(payload: dict[str, Any]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for pair in payload.get("pair_records", []):
        if not isinstance(pair, dict):
            continue
        for point in pair.get("points", []):
            if isinstance(point, dict):
                points.append(point)
    if not points:
        raise ValueError("RoMa candidate audit has no pair_records/points")
    return points


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _record(point: dict[str, Any]) -> dict[str, Any] | None:
    candidates = point.get("candidates")
    if not isinstance(candidates, list) or len(candidates) < 2:
        return None
    ordered = sorted(
        (candidate for candidate in candidates if isinstance(candidate, dict)),
        key=lambda candidate: int(candidate.get("roma_rank", 10**9)),
    )
    if len(ordered) < 2:
        return None
    first, second = ordered[0], ordered[1]
    error_first = _finite(first.get("bidirectional_error"), default=float("inf"))
    error_second = _finite(second.get("bidirectional_error"), default=float("inf"))
    if not math.isfinite(error_first) or not math.isfinite(error_second):
        return None
    denom = max(1e-6, abs(error_first))
    return {
        "teacher_correct": bool(first.get("pck_hit")),
        "baseline_correct": bool(point.get("baseline_pck_hit")),
        "both_wrong_top20_hit": bool(point.get("both_wrong_top20_hit")),
        "relative_error_margin": (error_second - error_first) / denom,
        "negative_bidirectional_error": -error_first,
        "mutual_certainty": _finite(first.get("mutual_certainty")),
        "source_certainty": _finite(point.get("source_certainty")),
        "negative_forward_backward_disagreement": -abs(
            _finite(first.get("forward_error")) - _finite(first.get("backward_error"))
        ),
    }


def _metric(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    count = len(rows)
    if not count:
        return {
            "count": 0,
            "coverage": 0.0,
            "teacher_point": None,
            "baseline_point": None,
            "rescued": 0,
            "harmed": 0,
            "net": 0,
            "both_wrong_teacher_point": None,
        }
    teacher_correct = sum(bool(row["teacher_correct"]) for row in rows)
    baseline_correct = sum(bool(row["baseline_correct"]) for row in rows)
    rescued = sum(not row["baseline_correct"] and row["teacher_correct"] for row in rows)
    harmed = sum(row["baseline_correct"] and not row["teacher_correct"] for row in rows)
    hard = [row for row in rows if row["both_wrong_top20_hit"]]
    return {
        "count": count,
        "coverage": None,
        "teacher_point": teacher_correct / count,
        "baseline_point": baseline_correct / count,
        "rescued": rescued,
        "harmed": harmed,
        "net": rescued - harmed,
        "both_wrong_teacher_point": (
            sum(bool(row["teacher_correct"]) for row in hard) / len(hard)
            if hard else None
        ),
    }


def _threshold(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot compute confidence quantile from no values")
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def _signal_audit(
    discovery: list[dict[str, Any]],
    heldout: list[dict[str, Any]],
    name: str,
) -> dict[str, Any]:
    cutoffs = {}
    for quantile in QUANTILES:
        threshold = _threshold([row[name] for row in discovery], quantile)
        discovery_selected = [row for row in discovery if row[name] >= threshold]
        heldout_selected = [row for row in heldout if row[name] >= threshold]
        discovery_metric = _metric(discovery_selected)
        heldout_metric = _metric(heldout_selected)
        discovery_metric["coverage"] = len(discovery_selected) / len(discovery)
        heldout_metric["coverage"] = len(heldout_selected) / len(heldout)
        cutoffs[str(quantile)] = {
            "threshold_from_discovery": threshold,
            "discovery": discovery_metric,
            "heldout": heldout_metric,
        }
    return cutoffs


def analyze(discovery_payload: dict[str, Any], heldout_payload: dict[str, Any]) -> dict[str, Any]:
    discovery = [row for point in _points(discovery_payload) if (row := _record(point))]
    heldout = [row for point in _points(heldout_payload) if (row := _record(point))]
    if not discovery or not heldout:
        raise ValueError("both splits require at least two ranked RoMa candidates per point")
    names = (
        "relative_error_margin",
        "negative_bidirectional_error",
        "mutual_certainty",
        "source_certainty",
        "negative_forward_backward_disagreement",
    )
    result = {
        "audit": "frozen_roma_pseudo_target_quality",
        "claim_scope": "offline teacher-quality audit; PCK is never used to set thresholds",
        "teacher_contract": "rank-1 frozen RoMa bidirectional-warp candidate inside fixed FLUX attention top-20",
        "selection_contract": "cutoffs are confidence quantiles from discovery, transferred unchanged to heldout",
        "all_points": {
            "discovery": _metric(discovery),
            "heldout": _metric(heldout),
        },
        "signals": {name: _signal_audit(discovery, heldout, name) for name in names},
        "fixed_normalized_error_tolerances": {},
    }
    result["all_points"]["discovery"]["coverage"] = 1.0
    result["all_points"]["heldout"]["coverage"] = 1.0
    for tolerance in FIXED_MAX_BIDIRECTIONAL_ERRORS:
        # ``negative_bidirectional_error >= -tolerance`` is equivalent to
        # accepting a frozen RoMa endpoint disagreement no greater than this
        # normalized geometric displacement.
        discovery_selected = [
            row for row in discovery
            if row["negative_bidirectional_error"] >= -float(tolerance)
        ]
        heldout_selected = [
            row for row in heldout
            if row["negative_bidirectional_error"] >= -float(tolerance)
        ]
        discovery_metric = _metric(discovery_selected)
        heldout_metric = _metric(heldout_selected)
        discovery_metric["coverage"] = len(discovery_selected) / len(discovery)
        heldout_metric["coverage"] = len(heldout_selected) / len(heldout)
        result["fixed_normalized_error_tolerances"][str(tolerance)] = {
            "discovery": discovery_metric,
            "heldout": heldout_metric,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery_json", required=True)
    parser.add_argument("--heldout_json", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()
    result = analyze(_read(args.discovery_json), _read(args.heldout_json))
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result["all_points"], indent=2))
    for signal, rows in result["signals"].items():
        print(f"{signal}: heldout top-decile", json.dumps(rows["0.9"]["heldout"], sort_keys=True))


if __name__ == "__main__":
    main()

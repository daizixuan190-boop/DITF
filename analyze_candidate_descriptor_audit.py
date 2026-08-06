"""Analyze FJSAR candidate-descriptor audit dumps.

This is an offline diagnostic script.  It reads JSON produced by
``eval_spair_matcher_ablation.py --fjsar_candidate_descriptor_audit`` and
summarizes whether feature-side identity signals can recover the high-recall
attention oracle.  It does not implement, tune, or evaluate a matcher.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from statistics import mean, median
from typing import Any, Callable


SIGNALS = (
    "attention",
    "native_descriptor",
    "local_self_similarity",
    "attention_jacobian",
)
RANK_KS = (1, 2, 3, 5, 10, 20)
GROUPS: dict[str, Callable[[dict[str, Any]], bool]] = {
    "all": lambda row: True,
    "oracle_gap": lambda row: bool(row.get("oracle_gap_case")),
    "attention_harms_native": lambda row: bool(row.get("attention_harms_native_case")),
    "attention_rescues_native": lambda row: bool(row.get("attention_rescues_native_case")),
    "baseline_wrong": lambda row: not bool(row.get("baseline_pck_hit")),
    "baseline_correct": lambda row: bool(row.get("baseline_pck_hit")),
}


def _safe_float(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def _safe_median(values: list[float]) -> float | None:
    return float(median(values)) if values else None


def _safe_mean(values: list[float]) -> float | None:
    return float(mean(values)) if values else None


def _rate(count: int, total: int) -> float:
    return float(count) / float(total) if total else 0.0


def _audit(row: dict[str, Any]) -> dict[str, Any]:
    audit = row.get("candidate_descriptor_audit")
    return audit if isinstance(audit, dict) else {}


def _flow_audit(row: dict[str, Any]) -> dict[str, Any]:
    audit = row.get("attention_flow_audit")
    return audit if isinstance(audit, dict) else {}


def _factorization_audit(row: dict[str, Any]) -> dict[str, Any]:
    audit = row.get("transport_factorization_audit")
    return audit if isinstance(audit, dict) else {}


def _proposal_rank(row: dict[str, Any], signal: str) -> int | None:
    rank = _audit(row).get("proposal_only_ranks", _audit(row).get("ranks", {})).get(signal)
    return int(rank) if isinstance(rank, int) else None


def _gt_exact_rank(row: dict[str, Any], signal: str) -> int | None:
    rank = _audit(row).get("gt_exact_augmented_ranks", {}).get(signal)
    return int(rank) if isinstance(rank, int) else None


def _proposal_gap(row: dict[str, Any], signal: str) -> float | None:
    gaps = _audit(row).get("proposal_hit_score_gaps", _audit(row).get("score_gaps", {}))
    return _safe_float(gaps.get(f"{signal}_attention_top1_minus_best_pck_hit_proposal"))


def _gt_exact_gap(row: dict[str, Any], signal: str) -> float | None:
    gaps = _audit(row).get("gt_exact_score_gaps", {})
    return _safe_float(gaps.get(f"{signal}_attention_top1_minus_gt_exact"))


def _rank_hit(row: dict[str, Any], signal: str, k: int, *, exact: bool = False) -> bool:
    rank = _gt_exact_rank(row, signal) if exact else _proposal_rank(row, signal)
    return rank is not None and rank <= int(k)


def _flow_rank(row: dict[str, Any], signal: str) -> int | None:
    rank = _flow_audit(row).get("ranks", {}).get(signal)
    return int(rank) if isinstance(rank, int) else None


def _flow_gap(row: dict[str, Any], signal: str) -> float | None:
    gaps = _flow_audit(row).get("score_gaps", {})
    return _safe_float(gaps.get(f"{signal}_attention_top1_minus_best_pck_hit_proposal"))


def _flow_rank_hit(row: dict[str, Any], signal: str, k: int) -> bool:
    rank = _flow_rank(row, signal)
    return rank is not None and rank <= int(k)


def _factorization_rank(row: dict[str, Any], signal: str) -> int | None:
    rank = _factorization_audit(row).get("ranks", {}).get(signal)
    return int(rank) if isinstance(rank, int) else None


def _factorization_gap(row: dict[str, Any], signal: str) -> float | None:
    gaps = _factorization_audit(row).get("score_gaps", {})
    return _safe_float(gaps.get(f"{signal}_attention_top1_minus_best_pck_hit_proposal"))


def _factorization_rank_hit(row: dict[str, Any], signal: str, k: int) -> bool:
    rank = _factorization_rank(row, signal)
    return rank is not None and rank <= int(k)


def _group_records(records: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    return [row for row in records if GROUPS[name](row) and _audit(row)]


def _group_flow_records(records: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    return [row for row in records if GROUPS[name](row) and _flow_audit(row)]


def _group_factorization_records(records: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    return [row for row in records if GROUPS[name](row) and _factorization_audit(row)]


def _flow_score_names(records: list[dict[str, Any]]) -> list[str]:
    for row in records:
        audit = _flow_audit(row)
        names = audit.get("score_names")
        if isinstance(names, list) and names:
            return [str(name) for name in names]
    return []


def _factorization_score_names(records: list[dict[str, Any]]) -> list[str]:
    for row in records:
        audit = _factorization_audit(row)
        names = audit.get("score_names")
        if isinstance(names, list) and names:
            return [str(name) for name in names]
    return []


def _candidate_distance(candidate: dict[str, Any], target: list[float] | None) -> float | None:
    if not target or "pixel" not in candidate:
        return None
    try:
        x, y = candidate["pixel"]
        return math.hypot(float(x) - float(target[0]), float(y) - float(target[1]))
    except (TypeError, ValueError):
        return None


def _point_distance(a: list[float] | None, b: list[float] | None) -> float | None:
    if not a or not b:
        return None
    try:
        return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))
    except (TypeError, ValueError):
        return None


def _top_candidate(row: dict[str, Any], signal: str) -> dict[str, Any] | None:
    candidates = [
        candidate
        for candidate in _audit(row).get("candidates", [])
        if candidate.get("is_attention_proposal", True)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: float(candidate.get(signal, -1e30)))


def _nearest_hit_candidate(row: dict[str, Any]) -> dict[str, Any] | None:
    target = row.get("target_point")
    candidates = [
        candidate
        for candidate in _audit(row).get("candidates", [])
        if candidate.get("is_attention_proposal", True) and candidate.get("pck_hit") is True
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda candidate: _candidate_distance(candidate, target) or float("inf"),
    )


def _best_hit_candidate(row: dict[str, Any], signal: str) -> dict[str, Any] | None:
    candidates = [
        candidate
        for candidate in _audit(row).get("candidates", [])
        if candidate.get("is_attention_proposal", True) and candidate.get("pck_hit") is True
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: float(candidate.get(signal, -1e30)))


def _top1_basin_key(row: dict[str, Any]) -> tuple[str, int] | None:
    pair_json = row.get("pair_json")
    pixel_index = row.get("attention_top1", {}).get("pixel_index")
    if pair_json is None or pixel_index is None:
        return None
    return str(pair_json), int(pixel_index)


def _basin_sizes(records: list[dict[str, Any]]) -> dict[tuple[str, int], int]:
    counts: Counter[tuple[str, int]] = Counter()
    for row in records:
        key = _top1_basin_key(row)
        if key is not None:
            counts[key] += 1
    return dict(counts)


def _bin_edges(values: list[float], bins: int) -> list[float]:
    if not values:
        return []
    values = sorted(values)
    edges = []
    for index in range(1, bins):
        pos = min(len(values) - 1, max(0, round(index * (len(values) - 1) / bins)))
        edges.append(float(values[pos]))
    return edges


def _bucket(value: float | None, edges: list[float]) -> str:
    if value is None:
        return "missing"
    lower = "-inf"
    for edge in edges:
        if value <= edge:
            return f"({lower}, {edge:.6g}]"
        lower = f"{edge:.6g}"
    return f"({lower}, inf)"


def _summarize_signal(records: list[dict[str, Any]], signal: str) -> dict[str, Any]:
    proposal_ranks = [_proposal_rank(row, signal) for row in records]
    proposal_ranks = [rank for rank in proposal_ranks if rank is not None]
    gt_exact_ranks = [_gt_exact_rank(row, signal) for row in records]
    gt_exact_ranks = [rank for rank in gt_exact_ranks if rank is not None]
    proposal_gaps = [_proposal_gap(row, signal) for row in records]
    proposal_gaps = [gap for gap in proposal_gaps if gap is not None]
    gt_exact_gaps = [_gt_exact_gap(row, signal) for row in records]
    gt_exact_gaps = [gap for gap in gt_exact_gaps if gap is not None]
    return {
        "proposal_ranked_points": len(proposal_ranks),
        "proposal_pck_hit_rank_median": _safe_median([float(rank) for rank in proposal_ranks]),
        "proposal_pck_hit_at": {
            f"@{k}": sum(1 for rank in proposal_ranks if rank <= k)
            for k in RANK_KS
        },
        "proposal_pck_hit_rate": {
            f"@{k}": _rate(sum(1 for rank in proposal_ranks if rank <= k), len(proposal_ranks))
            for k in RANK_KS
        },
        "gt_exact_ranked_points": len(gt_exact_ranks),
        "gt_exact_rank_median": _safe_median([float(rank) for rank in gt_exact_ranks]),
        "attention_top1_minus_best_hit_proposal_gap_median": _safe_median(proposal_gaps),
        "best_hit_proposal_beats_attention_top1": sum(1 for gap in proposal_gaps if gap < 0.0),
        "attention_top1_beats_best_hit_proposal": sum(1 for gap in proposal_gaps if gap > 0.0),
        "attention_top1_minus_gt_exact_gap_median": _safe_median(gt_exact_gaps),
        "gt_exact_beats_attention_top1": sum(1 for gap in gt_exact_gaps if gap < 0.0),
    }


def _summarize_flow_signal(records: list[dict[str, Any]], signal: str) -> dict[str, Any]:
    ranks = [_flow_rank(row, signal) for row in records]
    ranks = [rank for rank in ranks if rank is not None]
    gaps = [_flow_gap(row, signal) for row in records]
    gaps = [gap for gap in gaps if gap is not None]
    return {
        "ranked_points": len(ranks),
        "proposal_pck_hit_rank_median": _safe_median([float(rank) for rank in ranks]),
        "proposal_pck_hit_at": {
            f"@{k}": sum(1 for rank in ranks if rank <= k)
            for k in RANK_KS
        },
        "proposal_pck_hit_rate": {
            f"@{k}": _rate(sum(1 for rank in ranks if rank <= k), len(ranks))
            for k in RANK_KS
        },
        "attention_top1_minus_best_hit_proposal_gap_median": _safe_median(gaps),
        "best_hit_proposal_beats_attention_top1": sum(1 for gap in gaps if gap < 0.0),
        "attention_top1_beats_best_hit_proposal": sum(1 for gap in gaps if gap > 0.0),
    }


def _summarize_factorization_signal(records: list[dict[str, Any]], signal: str) -> dict[str, Any]:
    ranks = [_factorization_rank(row, signal) for row in records]
    ranks = [rank for rank in ranks if rank is not None]
    gaps = [_factorization_gap(row, signal) for row in records]
    gaps = [gap for gap in gaps if gap is not None]
    return {
        "ranked_points": len(ranks),
        "proposal_pck_hit_rank_median": _safe_median([float(rank) for rank in ranks]),
        "proposal_pck_hit_at": {
            f"@{k}": sum(1 for rank in ranks if rank <= k)
            for k in RANK_KS
        },
        "proposal_pck_hit_rate": {
            f"@{k}": _rate(sum(1 for rank in ranks if rank <= k), len(ranks))
            for k in RANK_KS
        },
        "attention_top1_minus_best_hit_proposal_gap_median": _safe_median(gaps),
        "best_hit_proposal_beats_attention_top1": sum(1 for gap in gaps if gap < 0.0),
        "attention_top1_beats_best_hit_proposal": sum(1 for gap in gaps if gap > 0.0),
    }


def _summarize_flow_group(records: list[dict[str, Any]], score_names: list[str]) -> dict[str, Any]:
    total = len(records)
    top1_counts = {
        signal: sum(1 for row in records if _flow_rank_hit(row, signal, 1))
        for signal in score_names
    }


def _summarize_factorization_group(records: list[dict[str, Any]], score_names: list[str]) -> dict[str, Any]:
    total = len(records)
    top1_counts = {
        signal: sum(1 for row in records if _factorization_rank_hit(row, signal, 1))
        for signal in score_names
    }
    best_signal = max(top1_counts, key=top1_counts.get) if top1_counts else None
    top1_union = {
        index
        for index, row in enumerate(records)
        if any(_factorization_rank_hit(row, signal, 1) for signal in score_names)
    }
    return {
        "points": total,
        "score_names": score_names,
        "signals": {
            signal: _summarize_factorization_signal(records, signal)
            for signal in score_names
        },
        "top1_union": len(top1_union),
        "top1_union_rate": _rate(len(top1_union), total),
        "best_top1_signal": best_signal,
        "best_top1": top1_counts.get(best_signal, 0) if best_signal else 0,
    }
    best_signal = max(top1_counts, key=top1_counts.get) if top1_counts else None
    top1_union = {
        index
        for index, row in enumerate(records)
        if any(_flow_rank_hit(row, signal, 1) for signal in score_names)
    }
    return {
        "points": total,
        "score_names": score_names,
        "signals": {
            signal: _summarize_flow_signal(records, signal)
            for signal in score_names
        },
        "top1_union": len(top1_union),
        "top1_union_rate": _rate(len(top1_union), total),
        "best_top1_signal": best_signal,
        "best_top1": top1_counts.get(best_signal, 0) if best_signal else 0,
    }


def _summarize_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    summary: dict[str, Any] = {
        "points": total,
        "baseline_correct": sum(1 for row in records if row.get("baseline_pck_hit")),
        "method_correct": sum(1 for row in records if row.get("method_pck_hit")),
        "attention_top1_correct": sum(1 for row in records if row.get("attention_top1_pck_hit")),
        "attention_topk_correct": sum(1 for row in records if row.get("attention_topk_pck_hit")),
        "gt_exact_in_attention_proposals": sum(
            1 for row in records if _audit(row).get("gt_exact_in_proposals")
        ),
        "signals": {signal: _summarize_signal(records, signal) for signal in SIGNALS},
    }
    top1_sets = {
        signal: {
            index for index, row in enumerate(records)
            if _rank_hit(row, signal, 1)
        }
        for signal in SIGNALS
        if signal != "attention"
    }
    if top1_sets:
        union = set().union(*top1_sets.values())
        summary["top1_complementarity"] = {
            "union": len(union),
            "union_rate": _rate(len(union), total),
            "by_signal": {signal: len(indices) for signal, indices in top1_sets.items()},
            "native_union_local_self_similarity": len(
                top1_sets.get("native_descriptor", set())
                | top1_sets.get("local_self_similarity", set())
            ),
            "native_union_attention_jacobian": len(
                top1_sets.get("native_descriptor", set())
                | top1_sets.get("attention_jacobian", set())
            ),
            "all_signal_intersection": len(set.intersection(*top1_sets.values())) if top1_sets else 0,
        }
    return summary


def _category_summary(records: list[dict[str, Any]], group_name: str) -> dict[str, Any]:
    rows = _group_records(records, group_name)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[str(row.get("category", "unknown"))].append(row)
    output = {}
    for category, cat_rows in sorted(by_category.items()):
        output[category] = {
            "points": len(cat_rows),
            "attention_rank_median": _safe_median([
                float(row.get("gt_ranks", {}).get("attention"))
                for row in cat_rows
                if isinstance(row.get("gt_ranks", {}).get("attention"), int)
            ]),
            "signals_top1": {
                signal: sum(1 for row in cat_rows if _rank_hit(row, signal, 1))
                for signal in SIGNALS
            },
            "signals_top5": {
                signal: sum(1 for row in cat_rows if _rank_hit(row, signal, 5))
                for signal in SIGNALS
            },
        }
    return output


def _rank_delta_summary(records: list[dict[str, Any]], group_name: str) -> dict[str, Any]:
    rows = _group_records(records, group_name)
    output = {}
    for signal in ("local_self_similarity", "attention_jacobian"):
        better = same = worse = comparable = 0
        deltas = []
        for row in rows:
            native_rank = _proposal_rank(row, "native_descriptor")
            signal_rank = _proposal_rank(row, signal)
            if native_rank is None or signal_rank is None:
                continue
            comparable += 1
            delta = signal_rank - native_rank
            deltas.append(float(delta))
            if delta < 0:
                better += 1
            elif delta == 0:
                same += 1
            else:
                worse += 1
        output[signal] = {
            "comparable_points": comparable,
            "better_than_native": better,
            "same_as_native": same,
            "worse_than_native": worse,
            "median_rank_delta_signal_minus_native": _safe_median(deltas),
            "mean_rank_delta_signal_minus_native": _safe_mean(deltas),
        }
    return output


def _attention_rank_distribution(records: list[dict[str, Any]], group_name: str) -> dict[str, Any]:
    rows = _group_records(records, group_name)
    ranks = [
        int(row.get("gt_ranks", {}).get("attention"))
        for row in rows
        if isinstance(row.get("gt_ranks", {}).get("attention"), int)
    ]
    return {
        "ranked_points": len(ranks),
        "median": _safe_median([float(rank) for rank in ranks]),
        "hit_at": {f"@{k}": sum(1 for rank in ranks if rank <= k) for k in RANK_KS},
        "histogram": dict(sorted(Counter(ranks).items())),
    }


def _bin_success(records: list[dict[str, Any]], group_name: str, field: str, bins: int) -> dict[str, Any]:
    rows = _group_records(records, group_name)
    field_getters = {
        "attention_margin": lambda row: _safe_float(row.get("attention_top1", {}).get("margin")),
        "attention_concentration": lambda row: _safe_float(row.get("attention_top1", {}).get("concentration")),
        "attention_distance_over_threshold": lambda row: _safe_float(
            row.get("attention_top1", {}).get("distance_over_threshold")
        ),
        "native_margin": lambda row: _safe_float(row.get("native", {}).get("margin")),
    }
    values = [field_getters[field](row) for row in rows]
    numeric_values = [value for value in values if value is not None]
    edges = _bin_edges(numeric_values, bins)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row, value in zip(rows, values):
        buckets[_bucket(value, edges)].append(row)
    output = {}
    for bucket, bucket_rows in sorted(buckets.items()):
        output[bucket] = {
            "points": len(bucket_rows),
            "signals_top1_rate": {
                signal: _rate(sum(1 for row in bucket_rows if _rank_hit(row, signal, 1)), len(bucket_rows))
                for signal in SIGNALS
            },
            "signals_top5_rate": {
                signal: _rate(sum(1 for row in bucket_rows if _rank_hit(row, signal, 5)), len(bucket_rows))
                for signal in SIGNALS
            },
        }
    return {
        "field": field,
        "edges": edges,
        "buckets": output,
    }


def _top1_distance_summary(records: list[dict[str, Any]], group_name: str) -> dict[str, Any]:
    rows = _group_records(records, group_name)
    output = {}
    for signal in SIGNALS:
        top_distances = []
        best_hit_distances = []
        for row in rows:
            target = row.get("target_point")
            top = _top_candidate(row, signal)
            best_hit = _best_hit_candidate(row, signal)
            top_distance = _candidate_distance(top, target) if top else None
            best_hit_distance = _candidate_distance(best_hit, target) if best_hit else None
            if top_distance is not None:
                top_distances.append(top_distance)
            if best_hit_distance is not None:
                best_hit_distances.append(best_hit_distance)
        output[signal] = {
            "top_candidate_distance_to_gt_median": _safe_median(top_distances),
            "best_hit_candidate_distance_to_gt_median": _safe_median(best_hit_distances),
        }
    return output


def _basin_collapse_summary(records: list[dict[str, Any]], group_name: str) -> dict[str, Any]:
    all_basin_sizes = _basin_sizes(records)
    rows = _group_records(records, group_name)
    bucket_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sizes = []
    for row in rows:
        key = _top1_basin_key(row)
        size = all_basin_sizes.get(key, 1) if key is not None else 1
        sizes.append(float(size))
        if size <= 1:
            bucket = "1"
        elif size == 2:
            bucket = "2"
        elif size <= 4:
            bucket = "3-4"
        else:
            bucket = "5+"
        bucket_rows[bucket].append(row)
    repeated_rows = sum(1 for size in sizes if size > 1)
    output = {
        "points": len(rows),
        "attention_top1_reused_points": repeated_rows,
        "attention_top1_reused_rate": _rate(repeated_rows, len(rows)),
        "attention_top1_basin_size_median": _safe_median(sizes),
        "buckets": {},
    }
    for bucket in ("1", "2", "3-4", "5+"):
        bucket_items = bucket_rows.get(bucket, [])
        output["buckets"][bucket] = {
            "points": len(bucket_items),
            "baseline_correct": sum(1 for row in bucket_items if row.get("baseline_pck_hit")),
            "attention_top1_correct": sum(1 for row in bucket_items if row.get("attention_top1_pck_hit")),
            "signals_top1_rate": {
                signal: _rate(sum(1 for row in bucket_items if _rank_hit(row, signal, 1)), len(bucket_items))
                for signal in SIGNALS
            },
        }
    return output


def _signal_success_mask(row: dict[str, Any], k: int = 1) -> str:
    active = [
        signal
        for signal in ("native_descriptor", "local_self_similarity", "attention_jacobian")
        if _rank_hit(row, signal, k)
    ]
    return "+".join(active) if active else "none"


def _failure_taxonomy(records: list[dict[str, Any]], group_name: str) -> dict[str, Any]:
    rows = _group_records(records, group_name)
    top1_masks = Counter(_signal_success_mask(row, 1) for row in rows)
    top5_masks = Counter(_signal_success_mask(row, 5) for row in rows)
    attention_rank_buckets: dict[str, Counter[str]] = defaultdict(Counter)
    distance_buckets: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        rank = row.get("gt_ranks", {}).get("attention")
        if not isinstance(rank, int):
            rank_bucket = "missing"
        elif rank <= 2:
            rank_bucket = "<=2"
        elif rank <= 5:
            rank_bucket = "3-5"
        elif rank <= 10:
            rank_bucket = "6-10"
        else:
            rank_bucket = "11-20"
        attention_rank_buckets[rank_bucket][_signal_success_mask(row, 1)] += 1

        distance = _safe_float(row.get("attention_top1", {}).get("distance_over_threshold"))
        if distance is None:
            distance_bucket = "missing"
        elif distance <= 0.15:
            distance_bucket = "<=0.15"
        elif distance <= 0.30:
            distance_bucket = "0.15-0.30"
        elif distance <= 0.50:
            distance_bucket = "0.30-0.50"
        else:
            distance_bucket = ">0.50"
        distance_buckets[distance_bucket][_signal_success_mask(row, 1)] += 1
    return {
        "top1_signal_masks": dict(sorted(top1_masks.items())),
        "top5_signal_masks": dict(sorted(top5_masks.items())),
        "by_attention_rank_top1_masks": {
            bucket: dict(sorted(counter.items()))
            for bucket, counter in sorted(attention_rank_buckets.items())
        },
        "by_attention_top1_distance_over_threshold_top1_masks": {
            bucket: dict(sorted(counter.items()))
            for bucket, counter in sorted(distance_buckets.items())
        },
    }


def _metric_snapshot(row: dict[str, Any]) -> dict[str, float | None]:
    attention_rank = row.get("gt_ranks", {}).get("attention")
    attention_rank_value = float(attention_rank) if isinstance(attention_rank, int) else None
    nearest_hit = _nearest_hit_candidate(row)
    attention_top1 = row.get("attention_top1", {}).get("pixel")
    nearest_hit_distance = _point_distance(attention_top1, nearest_hit.get("pixel") if nearest_hit else None)
    threshold = _safe_float(row.get("threshold"))
    return {
        "attention_rank": attention_rank_value,
        "attention_margin": _safe_float(row.get("attention_top1", {}).get("margin")),
        "attention_concentration": _safe_float(row.get("attention_top1", {}).get("concentration")),
        "attention_distance_over_threshold": _safe_float(
            row.get("attention_top1", {}).get("distance_over_threshold")
        ),
        "native_margin": _safe_float(row.get("native", {}).get("margin")),
        "top1_to_nearest_hit_distance_over_threshold": (
            nearest_hit_distance / threshold
            if nearest_hit_distance is not None and threshold not in (None, 0.0)
            else None
        ),
    }


def _success_contrast(records: list[dict[str, Any]], group_name: str, signal: str) -> dict[str, Any]:
    rows = _group_records(records, group_name)
    split = {
        "top1_success": [row for row in rows if _rank_hit(row, signal, 1)],
        "top1_failure": [row for row in rows if not _rank_hit(row, signal, 1)],
    }
    output = {}
    for split_name, split_rows in split.items():
        metrics: dict[str, list[float]] = defaultdict(list)
        for row in split_rows:
            snapshot = _metric_snapshot(row)
            for key, value in snapshot.items():
                if value is not None:
                    metrics[key].append(float(value))
            gap = _proposal_gap(row, signal)
            if gap is not None:
                metrics[f"{signal}_proposal_gap"].append(gap)
            native_rank = _proposal_rank(row, "native_descriptor")
            signal_rank = _proposal_rank(row, signal)
            if native_rank is not None:
                metrics["native_rank"].append(float(native_rank))
            if signal_rank is not None:
                metrics[f"{signal}_rank"].append(float(signal_rank))
        output[split_name] = {
            "points": len(split_rows),
            "metrics_median": {
                key: _safe_median(values)
                for key, values in sorted(metrics.items())
            },
            "metrics_mean": {
                key: _safe_mean(values)
                for key, values in sorted(metrics.items())
            },
        }
    return output


def _pair_basin_examples(records: list[dict[str, Any]], group_name: str, limit: int) -> list[dict[str, Any]]:
    rows = _group_records(records, group_name)
    all_sizes = _basin_sizes(records)
    by_key: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = _top1_basin_key(row)
        if key is not None:
            by_key[key].append(row)
    examples = []
    for (pair_json, pixel_index), items in by_key.items():
        global_size = all_sizes.get((pair_json, pixel_index), len(items))
        if global_size <= 1:
            continue
        examples.append({
            "pair_json": pair_json,
            "attention_top1_pixel_index": pixel_index,
            "attention_top1_pixel": items[0].get("attention_top1", {}).get("pixel"),
            "global_basin_size": global_size,
            "group_points": len(items),
            "categories": sorted({str(row.get("category")) for row in items}),
            "keypoints": [
                {
                    "keypoint_index": row.get("keypoint_index"),
                    "source_point": row.get("source_point"),
                    "target_point": row.get("target_point"),
                    "attention_rank": row.get("gt_ranks", {}).get("attention"),
                    "native_rank": _proposal_rank(row, "native_descriptor"),
                    "jacobian_rank": _proposal_rank(row, "attention_jacobian"),
                    "local_self_similarity_rank": _proposal_rank(row, "local_self_similarity"),
                    "attention_distance_over_threshold": row.get("attention_top1", {}).get(
                        "distance_over_threshold"
                    ),
                }
                for row in sorted(items, key=lambda item: int(item.get("keypoint_index") or 0))
            ],
        })
    examples.sort(key=lambda item: (-int(item["global_basin_size"]), str(item["pair_json"])))
    return examples[:limit]


def _representative_cases(
    records: list[dict[str, Any]],
    group_name: str,
    signal: str,
    *,
    success: bool,
    limit: int,
) -> list[dict[str, Any]]:
    rows = _group_records(records, group_name)
    selected = []
    for row in rows:
        rank = _proposal_rank(row, signal)
        is_success = rank == 1
        if is_success != success:
            continue
        selected.append({
            "category": row.get("category"),
            "pair_json": row.get("pair_json"),
            "keypoint_index": row.get("keypoint_index"),
            "source_point": row.get("source_point"),
            "target_point": row.get("target_point"),
            "attention_rank": row.get("gt_ranks", {}).get("attention"),
            "signal_rank": rank,
            "native_rank": _proposal_rank(row, "native_descriptor"),
            "attention_margin": row.get("attention_top1", {}).get("margin"),
            "attention_distance_over_threshold": row.get("attention_top1", {}).get("distance_over_threshold"),
            "proposal_gap": _proposal_gap(row, signal),
            "attention_top1": row.get("attention_top1", {}).get("pixel"),
            "signal_top1": (_top_candidate(row, signal) or {}).get("pixel"),
            "best_hit_by_signal": (_best_hit_candidate(row, signal) or {}).get("pixel"),
        })
    selected.sort(key=lambda item: (
        item["category"] or "",
        str(item["pair_json"] or ""),
        int(item["keypoint_index"] or 0),
    ))
    return selected[:limit]


def _method_readiness(summary: dict[str, Any]) -> dict[str, Any]:
    oracle = summary["groups"].get("oracle_gap", {})
    harms = summary["groups"].get("attention_harms_native", {})
    output = {"notes": []}
    if not oracle or not harms:
        output["notes"].append("Missing oracle_gap or attention_harms_native group.")
        return output
    native_oracle = oracle["signals"]["native_descriptor"]["proposal_pck_hit_at"]["@1"]
    best_signal = max(
        ("native_descriptor", "local_self_similarity", "attention_jacobian"),
        key=lambda signal: oracle["signals"][signal]["proposal_pck_hit_at"]["@1"],
    )
    best_oracle = oracle["signals"][best_signal]["proposal_pck_hit_at"]["@1"]
    native_harms = harms["signals"]["native_descriptor"]["proposal_pck_hit_at"]["@1"]
    best_harms = harms["signals"][best_signal]["proposal_pck_hit_at"]["@1"]
    output.update({
        "best_oracle_gap_top1_signal": best_signal,
        "best_oracle_gap_top1": best_oracle,
        "native_oracle_gap_top1": native_oracle,
        "best_signal_harms_native_top1": best_harms,
        "native_harms_native_top1": native_harms,
    })
    if best_oracle <= native_oracle + 10:
        output["notes"].append(
            "No strong standalone identity signal: oracle_gap top1 is not far above native."
        )
    if best_harms < native_harms:
        output["notes"].append(
            "The best oracle-gap signal protects fewer native-correct harms than native descriptor."
        )
    output["notes"].append(
        "Treat top1 union values as diagnostic upper bounds only; they are not a train-free method."
    )
    return output


def build_summary(payload: dict[str, Any]) -> dict[str, Any]:
    records = payload.get("records", [])
    if not isinstance(records, list):
        raise ValueError("audit JSON must contain a records list")
    summary: dict[str, Any] = {
        "metadata": {
            "matcher": payload.get("matcher"),
            "candidate_topk": payload.get("candidate_topk"),
            "subset": payload.get("subset"),
            "pairs_per_cat": payload.get("pairs_per_cat"),
            "split_seed": payload.get("split_seed"),
            "records": len(records),
        },
        "groups": {},
        "category": {},
        "rank_delta_vs_native": {},
        "attention_rank_distribution": {},
        "binned_success": {},
        "top1_distance_to_gt": {},
        "basin_collapse": {},
        "failure_taxonomy": {},
        "success_contrast": {},
        "attention_flow": {},
        "transport_factorization": {},
    }
    flow_score_names = _flow_score_names(records)
    factorization_score_names = _factorization_score_names(records)
    for group_name in GROUPS:
        rows = _group_records(records, group_name)
        summary["groups"][group_name] = _summarize_group(rows)
        summary["rank_delta_vs_native"][group_name] = _rank_delta_summary(records, group_name)
        summary["attention_rank_distribution"][group_name] = _attention_rank_distribution(records, group_name)
        summary["top1_distance_to_gt"][group_name] = _top1_distance_summary(records, group_name)
        summary["basin_collapse"][group_name] = _basin_collapse_summary(records, group_name)
        summary["failure_taxonomy"][group_name] = _failure_taxonomy(records, group_name)
        summary["success_contrast"][group_name] = {
            signal: _success_contrast(records, group_name, signal)
            for signal in ("local_self_similarity", "attention_jacobian")
        }
        if flow_score_names:
            flow_rows = _group_flow_records(records, group_name)
            summary["attention_flow"][group_name] = _summarize_flow_group(flow_rows, flow_score_names)
        if factorization_score_names:
            factor_rows = _group_factorization_records(records, group_name)
            summary["transport_factorization"][group_name] = _summarize_factorization_group(
                factor_rows,
                factorization_score_names,
            )
    for group_name in ("oracle_gap", "attention_harms_native", "attention_rescues_native"):
        summary["category"][group_name] = _category_summary(records, group_name)
    for group_name in ("oracle_gap", "attention_harms_native"):
        summary["binned_success"][group_name] = {
            field: _bin_success(records, group_name, field, bins=4)
            for field in (
                "attention_margin",
                "attention_concentration",
                "attention_distance_over_threshold",
                "native_margin",
            )
        }
    summary["representative_cases"] = {
        "oracle_gap_attention_jacobian_top1_success": _representative_cases(
            records, "oracle_gap", "attention_jacobian", success=True, limit=20
        ),
        "oracle_gap_attention_jacobian_top1_failure": _representative_cases(
            records, "oracle_gap", "attention_jacobian", success=False, limit=20
        ),
        "oracle_gap_local_self_similarity_top1_success": _representative_cases(
            records, "oracle_gap", "local_self_similarity", success=True, limit=20
        ),
        "oracle_gap_reused_attention_basin_examples": _pair_basin_examples(
            records, "oracle_gap", limit=20
        ),
        "attention_harms_native_reused_attention_basin_examples": _pair_basin_examples(
            records, "attention_harms_native", limit=20
        ),
    }
    summary["method_readiness"] = _method_readiness(summary)
    return summary


def _print_compact(summary: dict[str, Any]) -> None:
    print(json.dumps(summary["metadata"], indent=2))
    for group_name in ("all", "oracle_gap", "attention_harms_native", "attention_rescues_native"):
        group = summary["groups"][group_name]
        print(f"\n[{group_name}] points={group['points']}")
        for signal in SIGNALS:
            signal_summary = group["signals"][signal]
            hits = signal_summary["proposal_pck_hit_at"]
            rates = signal_summary["proposal_pck_hit_rate"]
            print(
                f"  {signal:22s} "
                f"@1/@3/@5/@10="
                f"{hits['@1']:3d}/{hits['@3']:3d}/{hits['@5']:3d}/{hits['@10']:3d} "
                f"rates="
                f"{rates['@1']:.3f}/{rates['@3']:.3f}/{rates['@5']:.3f}/{rates['@10']:.3f} "
                f"median_rank={signal_summary['proposal_pck_hit_rank_median']}"
            )
        print(f"  top1_complementarity={group.get('top1_complementarity', {})}")
        basin = summary["basin_collapse"].get(group_name, {})
        print(
            "  basin_reuse="
            f"{basin.get('attention_top1_reused_points', 0)}/{basin.get('points', 0)} "
            f"rate={basin.get('attention_top1_reused_rate', 0.0):.3f} "
            f"median_size={basin.get('attention_top1_basin_size_median')}"
        )
        if group_name == "oracle_gap":
            print(
                "  top1_failure_taxonomy="
                f"{summary['failure_taxonomy'][group_name]['top1_signal_masks']}"
            )
        flow_group = summary.get("attention_flow", {}).get(group_name)
        if flow_group:
            print(
                "  attention_flow="
                f"best={flow_group.get('best_top1_signal')} "
                f"top1={flow_group.get('best_top1')} "
                f"union={flow_group.get('top1_union')} "
                f"union_rate={flow_group.get('top1_union_rate', 0.0):.3f}"
            )
            for signal, signal_summary in flow_group.get("signals", {}).items():
                hits = signal_summary["proposal_pck_hit_at"]
                print(
                    f"    {signal:38s} "
                    f"@1/@3/@5/@10="
                    f"{hits['@1']:3d}/{hits['@3']:3d}/{hits['@5']:3d}/{hits['@10']:3d} "
                    f"median_rank={signal_summary['proposal_pck_hit_rank_median']}"
                )
        factor_group = summary.get("transport_factorization", {}).get(group_name)
        if factor_group:
            print(
                "  transport_factorization="
                f"best={factor_group.get('best_top1_signal')} "
                f"top1={factor_group.get('best_top1')} "
                f"union={factor_group.get('top1_union')} "
                f"union_rate={factor_group.get('top1_union_rate', 0.0):.3f}"
            )
            for signal, signal_summary in factor_group.get("signals", {}).items():
                hits = signal_summary["proposal_pck_hit_at"]
                print(
                    f"    {signal:38s} "
                    f"@1/@3/@5/@10="
                    f"{hits['@1']:3d}/{hits['@3']:3d}/{hits['@5']:3d}/{hits['@10']:3d} "
                    f"median_rank={signal_summary['proposal_pck_hit_rank_median']}"
                )
    print("\n[method_readiness]")
    print(json.dumps(summary["method_readiness"], indent=2))


def _write_category_csv(summary: dict[str, Any], path: str) -> None:
    rows = []
    for group_name, by_category in summary["category"].items():
        for category, item in by_category.items():
            row = {
                "group": group_name,
                "category": category,
                "points": item["points"],
                "attention_rank_median": item["attention_rank_median"],
            }
            for signal in SIGNALS:
                row[f"{signal}_top1"] = item["signals_top1"][signal]
                row[f"{signal}_top5"] = item["signals_top5"][signal]
            rows.append(row)
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze candidate descriptor audit JSON.")
    parser.add_argument("audit_json")
    parser.add_argument("--output_json", default="")
    parser.add_argument("--category_csv", default="")
    args = parser.parse_args()
    with open(args.audit_json, encoding="utf-8") as handle:
        payload = json.load(handle)
    summary = build_summary(payload)
    _print_compact(summary)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
    if args.category_csv:
        _write_category_csv(summary, args.category_csv)


if __name__ == "__main__":
    main()

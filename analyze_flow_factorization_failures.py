"""Diagnose attention-flow signal lost by separable factorization.

This script is offline only.  It reads a JSON dump produced with
``--fjsar_attention_flow_audit`` and ``--fjsar_transport_factorization_audit``
enabled, then compares cases where pairwise flow can put a PCK-hit candidate at
rank 1 but the factorized source/target descriptor cannot.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from statistics import mean, median
from typing import Any


GROUPS = {
    "all": lambda row: True,
    "oracle_gap": lambda row: bool(row.get("oracle_gap_case")),
    "attention_harms_native": lambda row: bool(row.get("attention_harms_native_case")),
    "attention_rescues_native": lambda row: bool(row.get("attention_rescues_native_case")),
    "both_correct": lambda row: bool(row.get("baseline_pck_hit")) and bool(row.get("method_pck_hit")),
}


def _safe_float(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _safe_mean(values: list[float]) -> float | None:
    return float(mean(values)) if values else None


def _safe_median(values: list[float]) -> float | None:
    return float(median(values)) if values else None


def _rate(count: int, total: int) -> float:
    return float(count) / float(total) if total else 0.0


def _rank_hit(rank: Any, k: int = 1) -> bool:
    return isinstance(rank, int) and rank <= k


def _audit(row: dict[str, Any], key: str) -> dict[str, Any]:
    value = row.get(key)
    return value if isinstance(value, dict) else {}


def _score_names(records: list[dict[str, Any]], audit_key: str) -> list[str]:
    for row in records:
        names = _audit(row, audit_key).get("score_names")
        if isinstance(names, list) and names:
            return [str(name) for name in names]
    return []


def _rank(row: dict[str, Any], audit_key: str, signal: str) -> int | None:
    rank = _audit(row, audit_key).get("ranks", {}).get(signal)
    return int(rank) if isinstance(rank, int) else None


def _union_hit(row: dict[str, Any], audit_key: str, signals: list[str], k: int = 1) -> bool:
    return any(_rank_hit(_rank(row, audit_key, signal), k) for signal in signals)


def _candidate_list(row: dict[str, Any], audit_key: str) -> list[dict[str, Any]]:
    candidates = _audit(row, audit_key).get("candidates", [])
    return candidates if isinstance(candidates, list) else []


def _candidate_by_pixel(row: dict[str, Any], audit_key: str) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for candidate in _candidate_list(row, audit_key):
        pixel_index = candidate.get("pixel_index")
        if isinstance(pixel_index, int):
            out[pixel_index] = candidate
    return out


def _top_candidate(
    row: dict[str, Any],
    audit_key: str,
    signal: str,
    *,
    score_field: str = "scores",
) -> dict[str, Any] | None:
    candidates = _candidate_list(row, audit_key)
    if not candidates:
        return None
    scored = []
    for candidate in candidates:
        score = _safe_float(candidate.get(score_field, {}).get(signal))
        if score is not None:
            scored.append((score, candidate))
    if not scored:
        return None
    return max(scored, key=lambda item: item[0])[1]


def _rank_candidate_by_signal(
    row: dict[str, Any],
    audit_key: str,
    signal: str,
    pixel_index: int,
) -> int | None:
    candidates = _candidate_list(row, audit_key)
    scored = []
    for candidate in candidates:
        score = _safe_float(candidate.get("scores", {}).get(signal))
        candidate_pixel = candidate.get("pixel_index")
        if score is not None and isinstance(candidate_pixel, int):
            scored.append((score, candidate_pixel))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    for idx, (_, candidate_pixel) in enumerate(scored, start=1):
        if candidate_pixel == pixel_index:
            return idx
    return None


def _best_hit_signal_and_candidate(
    row: dict[str, Any],
    audit_key: str,
    signals: list[str],
) -> tuple[str | None, dict[str, Any] | None]:
    best_signal = None
    best_candidate = None
    best_score = -float("inf")
    for signal in signals:
        candidate = _top_candidate(row, audit_key, signal)
        if not candidate or candidate.get("pck_hit") is not True:
            continue
        score = _safe_float(candidate.get("scores", {}).get(signal))
        if score is not None and score > best_score:
            best_signal = signal
            best_candidate = candidate
            best_score = score
    return best_signal, best_candidate


def _point_distance(a: Any, b: Any) -> float | None:
    if not isinstance(a, list) or not isinstance(b, list) or len(a) < 2 or len(b) < 2:
        return None
    ax = _safe_float(a[0])
    ay = _safe_float(a[1])
    bx = _safe_float(b[0])
    by = _safe_float(b[1])
    if ax is None or ay is None or bx is None or by is None:
        return None
    return math.hypot(ax - bx, ay - by)


def _candidate_distance_to_gt(row: dict[str, Any], candidate: dict[str, Any] | None) -> float | None:
    if not candidate:
        return None
    return _point_distance(candidate.get("pixel"), row.get("target_point"))


def _attention_top1_distance(row: dict[str, Any]) -> float | None:
    top1 = row.get("attention_top1")
    if isinstance(top1, dict):
        distance = _safe_float(top1.get("distance_to_gt"))
        if distance is not None:
            return distance
        return _point_distance(top1.get("pixel"), row.get("target_point"))
    return None


def _factor_ranks_for_pixel(
    row: dict[str, Any],
    factor_signals: list[str],
    pixel_index: int,
) -> dict[str, int | None]:
    return {
        signal: _rank_candidate_by_signal(
            row,
            "transport_factorization_audit",
            signal,
            pixel_index,
        )
        for signal in factor_signals
    }


def _native_descriptor_rank(row: dict[str, Any], pixel_index: int) -> int | None:
    audit = _audit(row, "candidate_descriptor_audit")
    candidates = audit.get("candidates", [])
    scored = []
    for candidate in candidates if isinstance(candidates, list) else []:
        if candidate.get("is_attention_proposal") is False:
            continue
        candidate_pixel = candidate.get("pixel_index")
        score = _safe_float(candidate.get("native_descriptor"))
        if isinstance(candidate_pixel, int) and score is not None:
            scored.append((score, candidate_pixel))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    for idx, (_, candidate_pixel) in enumerate(scored, start=1):
        if candidate_pixel == pixel_index:
            return idx
    return None


def _collect_case_features(
    row: dict[str, Any],
    flow_signals: list[str],
    factor_signals: list[str],
) -> dict[str, Any]:
    flow_signal, flow_candidate = _best_hit_signal_and_candidate(
        row,
        "attention_flow_audit",
        flow_signals,
    )
    factor_signal, factor_candidate = _best_hit_signal_and_candidate(
        row,
        "transport_factorization_audit",
        factor_signals,
    )
    top_factor_signal = None
    top_factor_candidate = None
    top_factor_score = -float("inf")
    for signal in factor_signals:
        candidate = _top_candidate(row, "transport_factorization_audit", signal)
        score = _safe_float(candidate.get("scores", {}).get(signal)) if candidate else None
        if score is not None and score > top_factor_score:
            top_factor_signal = signal
            top_factor_candidate = candidate
            top_factor_score = score

    flow_pixel = flow_candidate.get("pixel_index") if flow_candidate else None
    factor_ranks = (
        _factor_ranks_for_pixel(row, factor_signals, flow_pixel)
        if isinstance(flow_pixel, int)
        else {}
    )
    factor_rank_values = [rank for rank in factor_ranks.values() if isinstance(rank, int)]
    flow_metrics = flow_candidate.get("metrics", {}) if flow_candidate else {}

    return {
        "pair_json": row.get("pair_json"),
        "category": row.get("category"),
        "keypoint_index": row.get("keypoint_index"),
        "threshold": row.get("threshold"),
        "baseline_pck_hit": bool(row.get("baseline_pck_hit")),
        "method_pck_hit": bool(row.get("method_pck_hit")),
        "oracle_gap_case": bool(row.get("oracle_gap_case")),
        "attention_harms_native_case": bool(row.get("attention_harms_native_case")),
        "attention_rescues_native_case": bool(row.get("attention_rescues_native_case")),
        "flow_signal": flow_signal,
        "flow_pixel_index": flow_pixel,
        "flow_pixel": flow_candidate.get("pixel") if flow_candidate else None,
        "flow_attention_rank": flow_candidate.get("rank_attention") if flow_candidate else None,
        "flow_distance_to_gt": _candidate_distance_to_gt(row, flow_candidate),
        "flow_native_descriptor_rank": _native_descriptor_rank(row, flow_pixel)
        if isinstance(flow_pixel, int)
        else None,
        "flow_best_factor_rank": min(factor_rank_values) if factor_rank_values else None,
        "flow_median_factor_rank": _safe_median([float(rank) for rank in factor_rank_values]),
        "flow_transport_consistency": _safe_float(flow_metrics.get("transport_consistency")),
        "flow_inverse_transport_consistency": _safe_float(flow_metrics.get("inverse_transport_consistency")),
        "flow_local_peak_support": _safe_float(flow_metrics.get("local_peak_support")),
        "flow_shape_preservation": _safe_float(flow_metrics.get("shape_preservation")),
        "flow_center_patch_mass": _safe_float(flow_metrics.get("center_patch_mass")),
        "flow_center_score_over_row_peak": _safe_float(flow_metrics.get("center_score_over_row_peak")),
        "flow_mean_displacement_error": _safe_float(flow_metrics.get("mean_displacement_error")),
        "flow_displacement_entropy": _safe_float(flow_metrics.get("displacement_entropy")),
        "flow_center_patch_entropy": _safe_float(flow_metrics.get("center_patch_entropy")),
        "top_factor_signal": top_factor_signal,
        "top_factor_pixel_index": top_factor_candidate.get("pixel_index") if top_factor_candidate else None,
        "top_factor_pck_hit": top_factor_candidate.get("pck_hit") if top_factor_candidate else None,
        "top_factor_attention_rank": top_factor_candidate.get("rank_attention") if top_factor_candidate else None,
        "top_factor_distance_to_gt": _candidate_distance_to_gt(row, top_factor_candidate),
        "attention_top1_distance_to_gt": _attention_top1_distance(row),
    }


def _summarize_features(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_keys = [
        "flow_attention_rank",
        "flow_distance_to_gt",
        "flow_native_descriptor_rank",
        "flow_best_factor_rank",
        "flow_median_factor_rank",
        "flow_transport_consistency",
        "flow_inverse_transport_consistency",
        "flow_local_peak_support",
        "flow_shape_preservation",
        "flow_center_patch_mass",
        "flow_center_score_over_row_peak",
        "flow_mean_displacement_error",
        "flow_displacement_entropy",
        "flow_center_patch_entropy",
        "top_factor_attention_rank",
        "top_factor_distance_to_gt",
        "attention_top1_distance_to_gt",
    ]
    summary: dict[str, Any] = {"points": len(rows)}
    for key in numeric_keys:
        values = [_safe_float(row.get(key)) for row in rows]
        values = [value for value in values if value is not None]
        summary[key] = {
            "count": len(values),
            "mean": _safe_mean(values),
            "median": _safe_median(values),
        }
    summary["categories"] = dict(Counter(str(row.get("category")) for row in rows))
    summary["flow_signals"] = dict(Counter(str(row.get("flow_signal")) for row in rows))
    summary["top_factor_signals"] = dict(Counter(str(row.get("top_factor_signal")) for row in rows))
    return summary


def analyze(records: list[dict[str, Any]]) -> dict[str, Any]:
    flow_signals = _score_names(records, "attention_flow_audit")
    factor_signals = _score_names(records, "transport_factorization_audit")
    if not flow_signals:
        raise ValueError("No attention_flow_audit score_names found.")
    if not factor_signals:
        raise ValueError("No transport_factorization_audit score_names found.")

    out: dict[str, Any] = {
        "records": len(records),
        "flow_signals": flow_signals,
        "factor_signals": factor_signals,
        "groups": {},
    }
    for group_name, predicate in GROUPS.items():
        group_rows = [
            row
            for row in records
            if predicate(row)
            and _audit(row, "attention_flow_audit")
            and _audit(row, "transport_factorization_audit")
        ]
        cohorts: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in group_rows:
            flow_hit = _union_hit(row, "attention_flow_audit", flow_signals)
            factor_hit = _union_hit(row, "transport_factorization_audit", factor_signals)
            if flow_hit and factor_hit:
                cohort_name = "flow_hit_factor_hit"
            elif flow_hit and not factor_hit:
                cohort_name = "flow_hit_factor_miss"
            elif not flow_hit and factor_hit:
                cohort_name = "flow_miss_factor_hit"
            else:
                cohort_name = "flow_miss_factor_miss"
            cohorts[cohort_name].append(_collect_case_features(row, flow_signals, factor_signals))

        total = len(group_rows)
        group_summary: dict[str, Any] = {
            "points": total,
            "flow_union_at1": sum(_union_hit(row, "attention_flow_audit", flow_signals) for row in group_rows),
            "factor_union_at1": sum(
                _union_hit(row, "transport_factorization_audit", factor_signals)
                for row in group_rows
            ),
            "cohorts": {},
        }
        group_summary["flow_union_rate"] = _rate(group_summary["flow_union_at1"], total)
        group_summary["factor_union_rate"] = _rate(group_summary["factor_union_at1"], total)
        for cohort_name in [
            "flow_hit_factor_hit",
            "flow_hit_factor_miss",
            "flow_miss_factor_hit",
            "flow_miss_factor_miss",
        ]:
            features = cohorts.get(cohort_name, [])
            cohort_summary = _summarize_features(features)
            cohort_summary["rate_in_group"] = _rate(len(features), total)
            group_summary["cohorts"][cohort_name] = cohort_summary
        out["groups"][group_name] = group_summary
    return out


def _write_cases_csv(path: str, records: list[dict[str, Any]], group_name: str) -> None:
    flow_signals = _score_names(records, "attention_flow_audit")
    factor_signals = _score_names(records, "transport_factorization_audit")
    predicate = GROUPS[group_name]
    rows = []
    for row in records:
        if not predicate(row):
            continue
        if not _union_hit(row, "attention_flow_audit", flow_signals):
            continue
        if _union_hit(row, "transport_factorization_audit", factor_signals):
            continue
        rows.append(_collect_case_features(row, flow_signals, factor_signals))
    rows.sort(
        key=lambda item: (
            item.get("category") or "",
            item.get("pair_json") or "",
            int(item.get("keypoint_index") or -1),
        )
    )
    fieldnames = [
        "category",
        "pair_json",
        "keypoint_index",
        "flow_signal",
        "flow_attention_rank",
        "flow_distance_to_gt",
        "flow_native_descriptor_rank",
        "flow_best_factor_rank",
        "flow_median_factor_rank",
        "flow_transport_consistency",
        "flow_inverse_transport_consistency",
        "flow_local_peak_support",
        "flow_shape_preservation",
        "flow_center_patch_mass",
        "flow_center_score_over_row_peak",
        "flow_mean_displacement_error",
        "flow_displacement_entropy",
        "top_factor_signal",
        "top_factor_pck_hit",
        "top_factor_attention_rank",
        "top_factor_distance_to_gt",
        "attention_top1_distance_to_gt",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_json", help="JSON dump containing flow and factorization audits.")
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--cases_csv", default=None)
    parser.add_argument(
        "--cases_group",
        default="oracle_gap",
        choices=sorted(GROUPS),
        help="Group used for the flow-hit/factor-miss CSV.",
    )
    args = parser.parse_args()

    with open(args.audit_json, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    records = payload.get("records", payload if isinstance(payload, list) else [])
    if not isinstance(records, list):
        raise ValueError("Input JSON must be a list or contain a 'records' list.")

    summary = analyze(records)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
    if args.cases_csv:
        _write_cases_csv(args.cases_csv, records, args.cases_group)

    for group_name in ["oracle_gap", "attention_harms_native", "attention_rescues_native", "all"]:
        group = summary["groups"][group_name]
        print(
            f"{group_name}: points={group['points']} "
            f"flow={group['flow_union_at1']} ({group['flow_union_rate']:.3f}) "
            f"factor={group['factor_union_at1']} ({group['factor_union_rate']:.3f})"
        )
        for cohort_name in [
            "flow_hit_factor_hit",
            "flow_hit_factor_miss",
            "flow_miss_factor_hit",
            "flow_miss_factor_miss",
        ]:
            cohort = group["cohorts"][cohort_name]
            print(
                f"  {cohort_name}: n={cohort['points']} "
                f"rate={cohort['rate_in_group']:.3f} "
                f"flow_attn_rank_med={cohort['flow_attention_rank']['median']} "
                f"flow_best_factor_rank_med={cohort['flow_best_factor_rank']['median']} "
                f"flow_dist_med={cohort['flow_distance_to_gt']['median']} "
                f"top_factor_dist_med={cohort['top_factor_distance_to_gt']['median']}"
            )


if __name__ == "__main__":
    main()

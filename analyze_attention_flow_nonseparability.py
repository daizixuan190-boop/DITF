"""Offline non-separability diagnostics for attention-flow evidence.

The goal is not to implement a matcher.  This script tests whether the useful
pairwise attention-flow signal is preserved by the current separable
transport-factorization scores.  It reads existing audit dumps only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from statistics import mean, median
from typing import Any

import numpy as np


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


def _audit(row: dict[str, Any], key: str) -> dict[str, Any]:
    audit = row.get(key)
    return audit if isinstance(audit, dict) else {}


def _score_names(records: list[dict[str, Any]], audit_key: str) -> list[str]:
    for row in records:
        names = _audit(row, audit_key).get("score_names")
        if isinstance(names, list) and names:
            return [str(name) for name in names]
    return []


def _candidates(row: dict[str, Any], audit_key: str) -> list[dict[str, Any]]:
    candidates = _audit(row, audit_key).get("candidates", [])
    return candidates if isinstance(candidates, list) else []


def _candidate_map(row: dict[str, Any], audit_key: str) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for candidate in _candidates(row, audit_key):
        pixel_index = candidate.get("pixel_index")
        if isinstance(pixel_index, int):
            out[pixel_index] = candidate
    return out


def _rank(row: dict[str, Any], audit_key: str, signal: str) -> int | None:
    rank = _audit(row, audit_key).get("ranks", {}).get(signal)
    return int(rank) if isinstance(rank, int) else None


def _rank_hit(row: dict[str, Any], audit_key: str, signal: str, k: int = 1) -> bool:
    rank = _rank(row, audit_key, signal)
    return rank is not None and rank <= int(k)


def _union_hit(row: dict[str, Any], audit_key: str, signals: list[str], k: int = 1) -> bool:
    return any(_rank_hit(row, audit_key, signal, k) for signal in signals)


def _score(candidate: dict[str, Any], signal: str) -> float | None:
    return _safe_float(candidate.get("scores", {}).get(signal))


def _rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = rank
        i = j
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    mx = mean(xs)
    my = mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0.0 or vy <= 0.0:
        return None
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / math.sqrt(vx * vy)


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    return _pearson(_rankdata(xs), _rankdata(ys))


def _row_signal_values(
    row: dict[str, Any],
    flow_signal: str,
    factor_signal: str,
) -> tuple[list[float], list[float]]:
    flow_by_pixel = _candidate_map(row, "attention_flow_audit")
    factor_by_pixel = _candidate_map(row, "transport_factorization_audit")
    xs: list[float] = []
    ys: list[float] = []
    for pixel_index, flow_candidate in flow_by_pixel.items():
        factor_candidate = factor_by_pixel.get(pixel_index)
        if not factor_candidate:
            continue
        flow_score = _score(flow_candidate, flow_signal)
        factor_score = _score(factor_candidate, factor_signal)
        if flow_score is not None and factor_score is not None:
            xs.append(flow_score)
            ys.append(factor_score)
    return xs, ys


def _top_two_margin(candidates: list[dict[str, Any]], signal: str) -> float | None:
    scores = sorted(
        [score for candidate in candidates if (score := _score(candidate, signal)) is not None],
        reverse=True,
    )
    if len(scores) < 2:
        return None
    return float(scores[0] - scores[1])


def _top_candidate(row: dict[str, Any], audit_key: str, signal: str) -> dict[str, Any] | None:
    scored = []
    for candidate in _candidates(row, audit_key):
        score = _score(candidate, signal)
        if score is not None:
            scored.append((score, candidate))
    if not scored:
        return None
    return max(scored, key=lambda item: item[0])[1]


def _flow_witness(
    row: dict[str, Any],
    flow_signals: list[str],
) -> tuple[str | None, dict[str, Any] | None, float | None]:
    """Return a flow signal whose rank-1 candidate is a PCK hit.

    If several flow signals work, keep the one with the largest top-2 margin.
    This avoids mixing raw scores across different signal scales.
    """

    best: tuple[str | None, dict[str, Any] | None, float | None] = (None, None, None)
    best_margin = -float("inf")
    for signal in flow_signals:
        if not _rank_hit(row, "attention_flow_audit", signal):
            continue
        candidate = _top_candidate(row, "attention_flow_audit", signal)
        if not candidate or candidate.get("pck_hit") is not True:
            continue
        margin = _top_two_margin(_candidates(row, "attention_flow_audit"), signal)
        margin_value = margin if margin is not None else 0.0
        if margin_value > best_margin:
            best = (signal, candidate, margin)
            best_margin = margin_value
    return best


def _factor_rank_for_pixel(
    row: dict[str, Any],
    factor_signal: str,
    pixel_index: int,
) -> int | None:
    scored = []
    for candidate in _candidates(row, "transport_factorization_audit"):
        candidate_pixel = candidate.get("pixel_index")
        score = _score(candidate, factor_signal)
        if isinstance(candidate_pixel, int) and score is not None:
            scored.append((score, candidate_pixel))
    scored.sort(key=lambda item: item[0], reverse=True)
    for idx, (_, candidate_pixel) in enumerate(scored, start=1):
        if candidate_pixel == pixel_index:
            return idx
    return None


def _factor_best_rank_and_margin(
    row: dict[str, Any],
    factor_signals: list[str],
    pixel_index: int,
) -> tuple[int | None, str | None, float | None]:
    factor_by_pixel = _candidate_map(row, "transport_factorization_audit")
    witness = factor_by_pixel.get(pixel_index)
    if not witness:
        return None, None, None
    best_rank = None
    best_signal = None
    best_margin = None
    for signal in factor_signals:
        rank = _factor_rank_for_pixel(row, signal, pixel_index)
        witness_score = _score(witness, signal)
        top = _top_candidate(row, "transport_factorization_audit", signal)
        top_score = _score(top, signal) if top else None
        if rank is None or witness_score is None or top_score is None:
            continue
        margin_to_top = float(witness_score - top_score)
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best_signal = signal
            best_margin = margin_to_top
    return best_rank, best_signal, best_margin


def _candidate_distance_to_gt(row: dict[str, Any], candidate: dict[str, Any] | None) -> float | None:
    if not candidate:
        return None
    pixel = candidate.get("pixel")
    target = row.get("target_point")
    if not isinstance(pixel, list) or not isinstance(target, list) or len(pixel) < 2 or len(target) < 2:
        return None
    px = _safe_float(pixel[0])
    py = _safe_float(pixel[1])
    tx = _safe_float(target[0])
    ty = _safe_float(target[1])
    if px is None or py is None or tx is None or ty is None:
        return None
    return math.hypot(px - tx, py - ty)


def _best_correlations(
    row: dict[str, Any],
    flow_signals: list[str],
    factor_signals: list[str],
) -> dict[str, Any]:
    best = None
    best_pair = None
    witness_signal, _, _ = _flow_witness(row, flow_signals)
    witness_best = None
    witness_best_pair = None
    for flow_signal in flow_signals:
        for factor_signal in factor_signals:
            xs, ys = _row_signal_values(row, flow_signal, factor_signal)
            corr = _spearman(xs, ys)
            if corr is None:
                continue
            if best is None or corr > best:
                best = corr
                best_pair = (flow_signal, factor_signal)
            if witness_signal == flow_signal and (witness_best is None or corr > witness_best):
                witness_best = corr
                witness_best_pair = (flow_signal, factor_signal)
    return {
        "best_spearman": best,
        "best_spearman_pair": best_pair,
        "witness_spearman": witness_best,
        "witness_spearman_pair": witness_best_pair,
    }


def _row_summary(
    row: dict[str, Any],
    flow_signals: list[str],
    factor_signals: list[str],
) -> dict[str, Any]:
    flow_signal, flow_candidate, flow_margin = _flow_witness(row, flow_signals)
    flow_pixel = flow_candidate.get("pixel_index") if flow_candidate else None
    best_factor_rank = None
    best_factor_signal = None
    factor_margin_to_top = None
    if isinstance(flow_pixel, int):
        best_factor_rank, best_factor_signal, factor_margin_to_top = _factor_best_rank_and_margin(
            row,
            factor_signals,
            flow_pixel,
        )
    correlations = _best_correlations(row, flow_signals, factor_signals)
    flow_metrics = flow_candidate.get("metrics", {}) if flow_candidate else {}
    return {
        "pair_json": row.get("pair_json"),
        "category": row.get("category"),
        "keypoint_index": row.get("keypoint_index"),
        "oracle_gap_case": bool(row.get("oracle_gap_case")),
        "attention_harms_native_case": bool(row.get("attention_harms_native_case")),
        "attention_rescues_native_case": bool(row.get("attention_rescues_native_case")),
        "flow_hit": _union_hit(row, "attention_flow_audit", flow_signals),
        "factor_hit": _union_hit(row, "transport_factorization_audit", factor_signals),
        "flow_witness_signal": flow_signal,
        "flow_witness_pixel_index": flow_pixel,
        "flow_witness_attention_rank": flow_candidate.get("rank_attention") if flow_candidate else None,
        "flow_witness_distance_to_gt": _candidate_distance_to_gt(row, flow_candidate),
        "flow_witness_top2_margin": flow_margin,
        "factor_best_rank_for_flow_witness": best_factor_rank,
        "factor_best_signal_for_flow_witness": best_factor_signal,
        "factor_margin_to_top_for_flow_witness": factor_margin_to_top,
        "flow_transport_consistency": _safe_float(flow_metrics.get("transport_consistency")),
        "flow_inverse_transport_consistency": _safe_float(flow_metrics.get("inverse_transport_consistency")),
        "flow_local_peak_support": _safe_float(flow_metrics.get("local_peak_support")),
        "flow_shape_preservation": _safe_float(flow_metrics.get("shape_preservation")),
        "flow_center_patch_mass": _safe_float(flow_metrics.get("center_patch_mass")),
        "flow_center_score_over_row_peak": _safe_float(flow_metrics.get("center_score_over_row_peak")),
        "flow_mean_displacement_error": _safe_float(flow_metrics.get("mean_displacement_error")),
        "flow_displacement_entropy": _safe_float(flow_metrics.get("displacement_entropy")),
        **correlations,
    }


def _summarize_numeric(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [_safe_float(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    return {
        "count": len(values),
        "mean": _safe_mean(values),
        "median": _safe_median(values),
    }


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_keys = [
        "flow_witness_attention_rank",
        "flow_witness_distance_to_gt",
        "flow_witness_top2_margin",
        "factor_best_rank_for_flow_witness",
        "factor_margin_to_top_for_flow_witness",
        "flow_transport_consistency",
        "flow_inverse_transport_consistency",
        "flow_local_peak_support",
        "flow_shape_preservation",
        "flow_center_patch_mass",
        "flow_center_score_over_row_peak",
        "flow_mean_displacement_error",
        "flow_displacement_entropy",
        "best_spearman",
        "witness_spearman",
    ]
    return {
        "points": len(rows),
        "numeric": {key: _summarize_numeric(rows, key) for key in numeric_keys},
        "categories": dict(Counter(str(row.get("category")) for row in rows)),
        "flow_witness_signals": dict(Counter(str(row.get("flow_witness_signal")) for row in rows)),
        "factor_best_signals": dict(Counter(str(row.get("factor_best_signal_for_flow_witness")) for row in rows)),
    }


def _zscore_observed(row: np.ndarray) -> np.ndarray:
    observed = np.isfinite(row)
    if not observed.any():
        return np.zeros_like(row, dtype=np.float64)
    values = row[observed]
    mu = float(values.mean())
    sigma = float(values.std())
    out = np.zeros_like(row, dtype=np.float64)
    if sigma <= 1e-12:
        out[observed] = 0.0
    else:
        out[observed] = (values - mu) / sigma
    out[~observed] = 0.0
    return out


def _matrix_for_pair(rows: list[dict[str, Any]], audit_key: str, signal: str) -> np.ndarray | None:
    pixels = sorted({
        int(candidate["pixel_index"])
        for row in rows
        for candidate in _candidates(row, audit_key)
        if isinstance(candidate.get("pixel_index"), int) and _score(candidate, signal) is not None
    })
    if len(rows) < 3 or len(pixels) < 5:
        return None
    pixel_to_col = {pixel: idx for idx, pixel in enumerate(pixels)}
    matrix = np.full((len(rows), len(pixels)), np.nan, dtype=np.float64)
    for row_idx, row in enumerate(rows):
        for candidate in _candidates(row, audit_key):
            pixel = candidate.get("pixel_index")
            if not isinstance(pixel, int):
                continue
            score = _score(candidate, signal)
            if score is not None:
                matrix[row_idx, pixel_to_col[pixel]] = score
    observed_per_row = np.isfinite(matrix).sum(axis=1)
    if int((observed_per_row >= 3).sum()) < 3:
        return None
    normalized = np.vstack([_zscore_observed(matrix[idx]) for idx in range(matrix.shape[0])])
    if not np.isfinite(normalized).all():
        return None
    return normalized


def _spectrum(matrix: np.ndarray) -> dict[str, float] | None:
    if matrix.shape[0] < 3 or matrix.shape[1] < 5:
        return None
    singular = np.linalg.svd(matrix, full_matrices=False, compute_uv=False)
    energy = singular ** 2
    total = float(energy.sum())
    if total <= 1e-12:
        return None
    probs = energy / total
    entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
    out = {
        "rows": float(matrix.shape[0]),
        "cols": float(matrix.shape[1]),
        "effective_rank": float(math.exp(entropy)),
        "rank1_energy": float(probs[:1].sum()),
        "rank2_energy": float(probs[:2].sum()),
        "rank3_energy": float(probs[:3].sum()),
    }
    return out


def _summarize_spectra(
    records: list[dict[str, Any]],
    audit_key: str,
    signals: list[str],
) -> dict[str, Any]:
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        pair = row.get("pair_json")
        if pair is not None:
            by_pair[str(pair)].append(row)
    per_signal: dict[str, list[dict[str, float]]] = {signal: [] for signal in signals}
    for rows in by_pair.values():
        if len(rows) < 3:
            continue
        for signal in signals:
            matrix = _matrix_for_pair(rows, audit_key, signal)
            if matrix is None:
                continue
            spectrum = _spectrum(matrix)
            if spectrum is not None:
                per_signal[signal].append(spectrum)

    summary: dict[str, Any] = {}
    for signal, spectra in per_signal.items():
        summary[signal] = {
            "matrices": len(spectra),
            "effective_rank_median": _safe_median([item["effective_rank"] for item in spectra]),
            "rank1_energy_median": _safe_median([item["rank1_energy"] for item in spectra]),
            "rank2_energy_median": _safe_median([item["rank2_energy"] for item in spectra]),
            "rank3_energy_median": _safe_median([item["rank3_energy"] for item in spectra]),
            "rows_median": _safe_median([item["rows"] for item in spectra]),
            "cols_median": _safe_median([item["cols"] for item in spectra]),
        }
    pooled = [item for spectra in per_signal.values() for item in spectra]
    summary["_pooled"] = {
        "matrices": len(pooled),
        "effective_rank_median": _safe_median([item["effective_rank"] for item in pooled]),
        "rank1_energy_median": _safe_median([item["rank1_energy"] for item in pooled]),
        "rank2_energy_median": _safe_median([item["rank2_energy"] for item in pooled]),
        "rank3_energy_median": _safe_median([item["rank3_energy"] for item in pooled]),
    }
    return summary


def analyze(records: list[dict[str, Any]]) -> dict[str, Any]:
    flow_signals = _score_names(records, "attention_flow_audit")
    factor_signals = _score_names(records, "transport_factorization_audit")
    if not flow_signals:
        raise ValueError("No attention_flow_audit score_names found.")
    if not factor_signals:
        raise ValueError("No transport_factorization_audit score_names found.")

    summary: dict[str, Any] = {
        "records": len(records),
        "flow_signals": flow_signals,
        "factor_signals": factor_signals,
        "groups": {},
        "pair_matrix_spectrum": {
            "attention_flow": _summarize_spectra(records, "attention_flow_audit", flow_signals),
            "transport_factorization": _summarize_spectra(
                records,
                "transport_factorization_audit",
                factor_signals,
            ),
        },
    }
    for group_name, predicate in GROUPS.items():
        rows = [
            _row_summary(row, flow_signals, factor_signals)
            for row in records
            if predicate(row)
            and _audit(row, "attention_flow_audit")
            and _audit(row, "transport_factorization_audit")
        ]
        cohorts = {
            "flow_hit_factor_hit": [
                row for row in rows if row["flow_hit"] and row["factor_hit"]
            ],
            "flow_hit_factor_miss": [
                row for row in rows if row["flow_hit"] and not row["factor_hit"]
            ],
            "flow_miss_factor_hit": [
                row for row in rows if not row["flow_hit"] and row["factor_hit"]
            ],
            "flow_miss_factor_miss": [
                row for row in rows if not row["flow_hit"] and not row["factor_hit"]
            ],
        }
        summary["groups"][group_name] = {
            "points": len(rows),
            "flow_union_at1": sum(bool(row["flow_hit"]) for row in rows),
            "factor_union_at1": sum(bool(row["factor_hit"]) for row in rows),
            "cohorts": {
                name: {
                    **_summarize_rows(cohort_rows),
                    "rate_in_group": _rate(len(cohort_rows), len(rows)),
                }
                for name, cohort_rows in cohorts.items()
            },
        }
        summary["groups"][group_name]["flow_union_rate"] = _rate(
            summary["groups"][group_name]["flow_union_at1"],
            len(rows),
        )
        summary["groups"][group_name]["factor_union_rate"] = _rate(
            summary["groups"][group_name]["factor_union_at1"],
            len(rows),
        )
    return summary


def _write_cases_csv(
    path: str,
    records: list[dict[str, Any]],
    group_name: str,
    cohort_name: str,
) -> None:
    flow_signals = _score_names(records, "attention_flow_audit")
    factor_signals = _score_names(records, "transport_factorization_audit")
    predicate = GROUPS[group_name]
    rows = [
        _row_summary(row, flow_signals, factor_signals)
        for row in records
        if predicate(row)
        and _audit(row, "attention_flow_audit")
        and _audit(row, "transport_factorization_audit")
    ]
    if cohort_name == "flow_hit_factor_hit":
        rows = [row for row in rows if row["flow_hit"] and row["factor_hit"]]
    elif cohort_name == "flow_hit_factor_miss":
        rows = [row for row in rows if row["flow_hit"] and not row["factor_hit"]]
    elif cohort_name == "flow_miss_factor_hit":
        rows = [row for row in rows if not row["flow_hit"] and row["factor_hit"]]
    elif cohort_name == "flow_miss_factor_miss":
        rows = [row for row in rows if not row["flow_hit"] and not row["factor_hit"]]
    else:
        raise ValueError(f"Unknown cohort: {cohort_name}")
    rows.sort(
        key=lambda row: (
            str(row.get("category")),
            str(row.get("pair_json")),
            int(row.get("keypoint_index") or -1),
        )
    )
    fieldnames = [
        "category",
        "pair_json",
        "keypoint_index",
        "flow_witness_signal",
        "flow_witness_attention_rank",
        "flow_witness_distance_to_gt",
        "flow_witness_top2_margin",
        "factor_best_rank_for_flow_witness",
        "factor_best_signal_for_flow_witness",
        "factor_margin_to_top_for_flow_witness",
        "flow_transport_consistency",
        "flow_inverse_transport_consistency",
        "flow_local_peak_support",
        "flow_shape_preservation",
        "flow_center_patch_mass",
        "flow_center_score_over_row_peak",
        "flow_mean_displacement_error",
        "flow_displacement_entropy",
        "best_spearman",
        "witness_spearman",
        "best_spearman_pair",
        "witness_spearman_pair",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_json", help="Audit JSON containing flow and factorization records.")
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--cases_csv", default=None)
    parser.add_argument("--cases_group", default="oracle_gap", choices=sorted(GROUPS))
    parser.add_argument(
        "--cases_cohort",
        default="flow_hit_factor_miss",
        choices=[
            "flow_hit_factor_hit",
            "flow_hit_factor_miss",
            "flow_miss_factor_hit",
            "flow_miss_factor_miss",
        ],
    )
    args = parser.parse_args()

    with open(args.audit_json, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    records = payload.get("records", payload if isinstance(payload, list) else None)
    if not isinstance(records, list):
        raise ValueError("Input JSON must be a list or contain a 'records' list.")

    summary = analyze(records)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
    if args.cases_csv:
        _write_cases_csv(args.cases_csv, records, args.cases_group, args.cases_cohort)

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
            numeric = cohort["numeric"]
            print(
                f"  {cohort_name}: n={cohort['points']} "
                f"rate={cohort['rate_in_group']:.3f} "
                f"factor_rank_med={numeric['factor_best_rank_for_flow_witness']['median']} "
                f"factor_margin_med={numeric['factor_margin_to_top_for_flow_witness']['median']} "
                f"witness_spearman_med={numeric['witness_spearman']['median']} "
                f"best_spearman_med={numeric['best_spearman']['median']}"
            )
    print("pair_matrix_spectrum:")
    for audit_name, spectra in summary["pair_matrix_spectrum"].items():
        pooled = spectra["_pooled"]
        print(
            f"  {audit_name}: matrices={pooled['matrices']} "
            f"eff_rank_med={pooled['effective_rank_median']} "
            f"rank1_energy_med={pooled['rank1_energy_median']} "
            f"rank3_energy_med={pooled['rank3_energy_median']}"
        )


if __name__ == "__main__":
    main()

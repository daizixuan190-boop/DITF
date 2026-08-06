"""Analyze active-method descriptor audits inside attention proposals.

This is an offline diagnostic script for JSON dumps produced with
``eval_spair_matcher_ablation.py --fjsar_method_descriptor_audit``.  It answers
whether the active matcher descriptor ranks GT/PCK-hit attention proposals high
enough to consume the attention oracle.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from statistics import mean, median
from typing import Any, Callable


GROUPS: dict[str, Callable[[dict[str, Any]], bool]] = {
    "all": lambda row: True,
    "oracle_gap": lambda row: bool(row.get("oracle_gap_case")),
    "attention_harms_native": lambda row: bool(row.get("attention_harms_native_case")),
    "attention_rescues_native": lambda row: bool(row.get("attention_rescues_native_case")),
    "native_correct_method_wrong": lambda row: bool(row.get("baseline_pck_hit"))
    and not bool(row.get("method_pck_hit")),
    "native_wrong_method_correct": lambda row: not bool(row.get("baseline_pck_hit"))
    and bool(row.get("method_pck_hit")),
    "native_wrong_method_wrong": lambda row: not bool(row.get("baseline_pck_hit"))
    and not bool(row.get("method_pck_hit")),
}


def _median(values: list[float]) -> float | None:
    return float(median(values)) if values else None


def _mean(values: list[float]) -> float | None:
    return float(mean(values)) if values else None


def _rate(count: int, total: int) -> float:
    return float(count) / float(total) if total else 0.0


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _audit(row: dict[str, Any]) -> dict[str, Any]:
    audit = row.get("method_descriptor_audit")
    return audit if isinstance(audit, dict) else {}


def _rank(row: dict[str, Any], signal: str = "method_descriptor") -> int | None:
    rank = _audit(row).get("proposal_only_ranks", _audit(row).get("ranks", {})).get(signal)
    return int(rank) if isinstance(rank, int) else None


def _gap(row: dict[str, Any], signal: str = "method_descriptor") -> float | None:
    value = _audit(row).get("proposal_hit_score_gaps", _audit(row).get("score_gaps", {})).get(
        f"{signal}_attention_top1_minus_best_pck_hit_proposal"
    )
    return _safe_float(value)


def _gt_exact_gap(row: dict[str, Any], signal: str = "method_descriptor") -> float | None:
    value = _audit(row).get("gt_exact_score_gaps", {}).get(f"{signal}_attention_top1_minus_gt_exact")
    return _safe_float(value)


def _proposal_hit_count(row: dict[str, Any]) -> int:
    return sum(
        1
        for candidate in _audit(row).get("candidates", [])
        if candidate.get("is_attention_proposal") and candidate.get("pck_hit")
    )


def _top_method_candidate(row: dict[str, Any]) -> dict[str, Any] | None:
    candidates = [
        candidate
        for candidate in _audit(row).get("candidates", [])
        if candidate.get("is_attention_proposal")
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: float(candidate.get("method_descriptor", -1e30)))


def _summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranks = [_rank(row) for row in rows]
    ranks = [rank for rank in ranks if rank is not None]
    gaps = [_gap(row) for row in rows]
    gaps = [gap for gap in gaps if gap is not None]
    exact_gaps = [_gt_exact_gap(row) for row in rows]
    exact_gaps = [gap for gap in exact_gaps if gap is not None]
    method_in_attention = [
        bool(row.get("method_prediction_in_attention_proposals"))
        for row in rows
        if "method_prediction_in_attention_proposals" in row
    ]
    top_candidates = [_top_method_candidate(row) for row in rows]
    top_candidates = [candidate for candidate in top_candidates if candidate is not None]
    top_hits = [bool(candidate.get("pck_hit")) for candidate in top_candidates]
    return {
        "points": len(rows),
        "gt_exact_in_attention_proposals": sum(
            1 for row in rows if _audit(row).get("gt_exact_in_proposals")
        ),
        "proposal_with_pck_hit": sum(1 for row in rows if _proposal_hit_count(row) > 0),
        "proposal_pck_hit_count_median": _median([float(_proposal_hit_count(row)) for row in rows]),
        "method_prediction_in_attention_proposals": sum(method_in_attention),
        "method_prediction_in_attention_proposals_rate": _rate(
            sum(method_in_attention), len(method_in_attention)
        ),
        "top_method_proposal_pck_hit": sum(top_hits),
        "top_method_proposal_pck_hit_rate": _rate(sum(top_hits), len(top_hits)),
        "ranked_points": len(ranks),
        "proposal_pck_hit_rank_median": _median([float(rank) for rank in ranks]),
        "proposal_pck_hit_at_1": sum(1 for rank in ranks if rank <= 1),
        "proposal_pck_hit_at_3": sum(1 for rank in ranks if rank <= 3),
        "proposal_pck_hit_at_5": sum(1 for rank in ranks if rank <= 5),
        "proposal_pck_hit_at_10": sum(1 for rank in ranks if rank <= 10),
        "attention_top1_minus_best_pck_hit_proposal_gap_median": _median(gaps),
        "best_pck_hit_proposal_beats_attention_top1": sum(1 for gap in gaps if gap < 0.0),
        "attention_top1_beats_best_pck_hit_proposal": sum(1 for gap in gaps if gap > 0.0),
        "attention_top1_minus_gt_exact_gap_median": _median(exact_gaps),
        "gt_exact_beats_attention_top1": sum(1 for gap in exact_gaps if gap < 0.0),
        "categories": dict(Counter(str(row.get("category")) for row in rows)),
    }


def analyze(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "records": len(records),
        "groups": {
            name: _summarize_group([row for row in records if predicate(row) and _audit(row)])
            for name, predicate in GROUPS.items()
        },
    }


def _write_category_csv(path: str, records: list[dict[str, Any]]) -> None:
    rows = []
    for category in sorted({str(row.get("category")) for row in records}):
        cat_rows = [row for row in records if str(row.get("category")) == category and _audit(row)]
        summary = _summarize_group(cat_rows)
        rows.append({
            "category": category,
            "points": summary["points"],
            "method_prediction_in_attention_proposals_rate": summary[
                "method_prediction_in_attention_proposals_rate"
            ],
            "top_method_proposal_pck_hit_rate": summary["top_method_proposal_pck_hit_rate"],
            "proposal_pck_hit_rank_median": summary["proposal_pck_hit_rank_median"],
            "proposal_pck_hit_at_1": summary["proposal_pck_hit_at_1"],
            "proposal_pck_hit_at_3": summary["proposal_pck_hit_at_3"],
            "proposal_pck_hit_at_5": summary["proposal_pck_hit_at_5"],
            "proposal_pck_hit_at_10": summary["proposal_pck_hit_at_10"],
        })
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["category"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_json")
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--category_csv", default=None)
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
    if args.category_csv:
        _write_category_csv(args.category_csv, records)

    for group_name in [
        "all",
        "oracle_gap",
        "native_correct_method_wrong",
        "native_wrong_method_correct",
    ]:
        group = summary["groups"][group_name]
        print(
            f"{group_name}: points={group['points']} "
            f"ranked={group['ranked_points']} "
            f"at1={group['proposal_pck_hit_at_1']} "
            f"at3={group['proposal_pck_hit_at_3']} "
            f"rank_med={group['proposal_pck_hit_rank_median']} "
            f"method_in_attn_rate={group['method_prediction_in_attention_proposals_rate']:.3f} "
            f"top_method_hit_rate={group['top_method_proposal_pck_hit_rate']:.3f}"
        )


if __name__ == "__main__":
    main()

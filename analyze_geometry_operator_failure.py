"""Offline diagnosis for geometry-consistent attention failures.

This script consumes existing candidate/kernel-audit JSON files.  It does not
rerun FLUX.  Therefore it can analyze raw/filtered attention-kernel hits,
method top-1 outcomes, and whether method predictions stay inside attention
top-k proposals.  It cannot reconstruct per-keypoint method-descriptor @5/@20
unless the original run included ``--fjsar_method_descriptor_audit``.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
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
    "attention_top20_method_wrong": lambda row: bool(row.get("attention_topk_pck_hit"))
    and not bool(row.get("method_pck_hit")),
    "filtered_top1_method_wrong": lambda row: _kernel_hit(row, "filtered_attention", 1)
    and not bool(row.get("method_pck_hit")),
}


def _load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _records(payload: Any) -> list[dict[str, Any]]:
    records = payload.get("records", payload if isinstance(payload, list) else None)
    if not isinstance(records, list):
        raise ValueError("Input JSON must be a list or contain a 'records' list.")
    return records


def _rate(count: int, total: int) -> float:
    return float(count) / float(total) if total else 0.0


def _kernel_hit(row: dict[str, Any], signal: str, k: int) -> bool:
    audit = row.get("attention_kernel_audit")
    if not isinstance(audit, dict):
        return False
    return bool(audit.get("topk_hits", {}).get(f"{signal}@{int(k)}"))


def _method_rank(row: dict[str, Any]) -> int | None:
    rank = row.get("method_prediction_attention_rank")
    return int(rank) if isinstance(rank, int) else None


def _transition_counts(rows: list[dict[str, Any]], k: int) -> dict[str, int]:
    raw_key = ("raw_attention", int(k))
    filtered_key = ("filtered_attention", int(k))
    both = raw_only = filtered_only = both_miss = 0
    for row in rows:
        raw = _kernel_hit(row, *raw_key)
        filtered = _kernel_hit(row, *filtered_key)
        if raw and filtered:
            both += 1
        elif raw:
            raw_only += 1
        elif filtered:
            filtered_only += 1
        else:
            both_miss += 1
    return {
        "both_hit": both,
        "raw_only": raw_only,
        "filtered_only": filtered_only,
        "both_miss": both_miss,
    }


def _summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    method_in_attention = sum(1 for row in rows if row.get("method_prediction_in_attention_proposals"))
    method_rank_le20 = sum(1 for row in rows if (_method_rank(row) or 10**9) <= 20)
    method_rank_le5 = sum(1 for row in rows if (_method_rank(row) or 10**9) <= 5)
    summary: dict[str, Any] = {
        "points": total,
        "baseline_correct": sum(1 for row in rows if row.get("baseline_pck_hit")),
        "method_correct": sum(1 for row in rows if row.get("method_pck_hit")),
        "improved": sum(
            1 for row in rows
            if not bool(row.get("baseline_pck_hit")) and bool(row.get("method_pck_hit"))
        ),
        "harmed": sum(
            1 for row in rows
            if bool(row.get("baseline_pck_hit")) and not bool(row.get("method_pck_hit"))
        ),
        "attention_top1_correct": sum(1 for row in rows if row.get("attention_top1_pck_hit")),
        "attention_top20_correct": sum(1 for row in rows if row.get("attention_topk_pck_hit")),
        "method_prediction_in_attention_topk": method_in_attention,
        "method_prediction_in_attention_topk_rate": _rate(method_in_attention, total),
        "method_prediction_attention_rank_at5": method_rank_le5,
        "method_prediction_attention_rank_at20": method_rank_le20,
        "method_prediction_attention_rank_at20_rate": _rate(method_rank_le20, total),
        "kernel": {},
        "transitions": {},
        "categories": dict(Counter(str(row.get("category")) for row in rows)),
    }
    for signal in ("raw_attention", "filtered_attention"):
        signal_summary = {}
        for k in (1, 5, 20):
            hits = sum(1 for row in rows if _kernel_hit(row, signal, k))
            method_hits_inside = sum(
                1 for row in rows
                if _kernel_hit(row, signal, k) and bool(row.get("method_pck_hit"))
            )
            signal_summary[f"hit_at_{k}"] = hits
            signal_summary[f"hit_rate_at_{k}"] = _rate(hits, total)
            signal_summary[f"method_correct_when_hit_at_{k}"] = method_hits_inside
            signal_summary[f"method_correct_when_hit_rate_at_{k}"] = _rate(method_hits_inside, hits)
        summary["kernel"][signal] = signal_summary
    for k in (1, 5, 20):
        summary["transitions"][f"@{k}"] = _transition_counts(rows, k)
    return summary


def _oracle_counts(main_payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(main_payload, dict):
        return {}
    counts = main_payload.get("all", {}).get("matcher_diagnostics", {}).get("model_counts", {})
    total = int(counts.get("fjsar_oracle_total", 0) or 0)
    rates = {
        key: _rate(int(value), total)
        for key, value in counts.items()
        if key != "fjsar_oracle_total"
    }
    return {"total": total, "counts": counts, "rates": rates}


def _category_table(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_category[str(record.get("category"))].append(record)
    for category in sorted(by_category):
        cat_rows = by_category[category]
        summary = _summarize_group(cat_rows)
        rows.append({
            "category": category,
            "points": summary["points"],
            "baseline_correct": summary["baseline_correct"],
            "method_correct": summary["method_correct"],
            "improved": summary["improved"],
            "harmed": summary["harmed"],
            "raw_attention_at20": summary["kernel"]["raw_attention"]["hit_at_20"],
            "filtered_attention_at20": summary["kernel"]["filtered_attention"]["hit_at_20"],
            "method_prediction_attention_rank_at20": summary["method_prediction_attention_rank_at20"],
            "method_prediction_in_attention_topk": summary["method_prediction_in_attention_topk"],
        })
    return rows


def _hard_cases(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases = []
    for row in records:
        if not row.get("method_pck_hit") and (
            _kernel_hit(row, "filtered_attention", 1)
            or row.get("oracle_gap_case")
            or row.get("attention_topk_pck_hit")
        ):
            cases.append({
                "category": row.get("category"),
                "pair_json": row.get("pair_json"),
                "keypoint_index": row.get("keypoint_index"),
                "baseline_hit": bool(row.get("baseline_pck_hit")),
                "method_hit": bool(row.get("method_pck_hit")),
                "attention_top1_hit": bool(row.get("attention_top1_pck_hit")),
                "attention_top20_hit": bool(row.get("attention_topk_pck_hit")),
                "raw_attention_at1": _kernel_hit(row, "raw_attention", 1),
                "raw_attention_at20": _kernel_hit(row, "raw_attention", 20),
                "filtered_attention_at1": _kernel_hit(row, "filtered_attention", 1),
                "filtered_attention_at20": _kernel_hit(row, "filtered_attention", 20),
                "method_prediction_in_attention_topk": bool(row.get("method_prediction_in_attention_proposals")),
                "method_prediction_attention_rank": row.get("method_prediction_attention_rank"),
                "source_point": row.get("source_point"),
                "target_point": row.get("target_point"),
                "baseline_prediction": row.get("baseline_prediction"),
                "method_prediction": row.get("method_prediction"),
            })
    return cases


def analyze(records: list[dict[str, Any]], main_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    has_method_descriptor_audit = any(isinstance(row.get("method_descriptor_audit"), dict) for row in records)
    return {
        "records": len(records),
        "matcher": main_payload.get("matcher") if isinstance(main_payload, dict) else None,
        "existing_json_limitations": {
            "has_method_descriptor_audit": has_method_descriptor_audit,
            "method_descriptor_per_keypoint_top5_top20_recoverable": has_method_descriptor_audit,
            "note": (
                "Per-keypoint geometry descriptor @5/@20 cannot be recovered from this run "
                "unless --fjsar_method_descriptor_audit was enabled."
            ),
        },
        "oracle_counts": _oracle_counts(main_payload),
        "groups": {
            name: _summarize_group([row for row in records if predicate(row)])
            for name, predicate in GROUPS.items()
        },
        "category_table": _category_table(records),
        "hard_case_count": len(_hard_cases(records)),
    }


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else ["empty"]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel_audit_json")
    parser.add_argument("--main_json", default=None)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--category_csv", default=None)
    parser.add_argument("--hard_cases_csv", default=None)
    args = parser.parse_args()

    kernel_payload = _load_json(args.kernel_audit_json)
    records = _records(kernel_payload)
    main_payload = _load_json(args.main_json) if args.main_json else None
    summary = analyze(records, main_payload if isinstance(main_payload, dict) else None)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
    if args.category_csv:
        _write_csv(args.category_csv, summary["category_table"])
    if args.hard_cases_csv:
        _write_csv(args.hard_cases_csv, _hard_cases(records))

    for group_name in ("all", "oracle_gap", "attention_harms_native", "attention_rescues_native"):
        group = summary["groups"][group_name]
        raw = group["kernel"]["raw_attention"]
        filtered = group["kernel"]["filtered_attention"]
        print(
            f"{group_name}: points={group['points']} "
            f"base={group['baseline_correct']} method={group['method_correct']} "
            f"improved/harmed={group['improved']}/{group['harmed']} "
            f"raw@1/@5/@20={raw['hit_at_1']}/{raw['hit_at_5']}/{raw['hit_at_20']} "
            f"filtered@1/@5/@20={filtered['hit_at_1']}/{filtered['hit_at_5']}/{filtered['hit_at_20']} "
            f"method_in_attn20={group['method_prediction_attention_rank_at20']}"
        )
    print("limitations:", summary["existing_json_limitations"]["note"])


if __name__ == "__main__":
    main()

"""Analyze transport-lift branch audit dumps.

Reads JSON produced by ``--fjsar_transport_lift_branch_audit`` and summarizes
whether native/outgoing/incoming/no-native/full branches can rank PCK-hit
attention proposals at top positions.
"""

from __future__ import annotations

import argparse
import json
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
}


def _audit(row: dict[str, Any]) -> dict[str, Any]:
    audit = row.get("transport_lift_branch_audit")
    return audit if isinstance(audit, dict) else {}


def _score_names(records: list[dict[str, Any]]) -> list[str]:
    for row in records:
        names = _audit(row).get("score_names")
        if isinstance(names, list) and names:
            return [str(name) for name in names]
    return []


def _rank(row: dict[str, Any], name: str) -> int | None:
    rank = _audit(row).get("ranks", {}).get(name)
    return int(rank) if isinstance(rank, int) else None


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    values = sorted(values)
    middle = len(values) // 2
    if len(values) % 2:
        return float(values[middle])
    return float((values[middle - 1] + values[middle]) / 2.0)


def _summarize_group(rows: list[dict[str, Any]], names: list[str]) -> dict[str, Any]:
    signals = {}
    for name in names:
        ranks = [_rank(row, name) for row in rows]
        ranks = [rank for rank in ranks if rank is not None]
        signals[name] = {
            "ranked_points": len(ranks),
            "rank_median": _median([float(rank) for rank in ranks]),
            "at1": sum(1 for rank in ranks if rank <= 1),
            "at3": sum(1 for rank in ranks if rank <= 3),
            "at5": sum(1 for rank in ranks if rank <= 5),
            "at10": sum(1 for rank in ranks if rank <= 10),
        }
    return {
        "points": len(rows),
        "baseline_correct": sum(1 for row in rows if row.get("baseline_pck_hit")),
        "method_correct": sum(1 for row in rows if row.get("method_pck_hit")),
        "attention_top1_correct": sum(1 for row in rows if row.get("attention_top1_pck_hit")),
        "attention_topk_correct": sum(1 for row in rows if row.get("attention_topk_pck_hit")),
        "signals": signals,
    }


def analyze(records: list[dict[str, Any]]) -> dict[str, Any]:
    names = _score_names(records)
    return {
        "records": len(records),
        "score_names": names,
        "groups": {
            group_name: _summarize_group(
                [row for row in records if predicate(row) and _audit(row)],
                names,
            )
            for group_name, predicate in GROUPS.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_json")
    parser.add_argument("--output_json", default=None)
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

    for group_name in [
        "oracle_gap",
        "native_correct_method_wrong",
        "native_wrong_method_correct",
        "attention_top20_method_wrong",
        "all",
    ]:
        group = summary["groups"][group_name]
        print(
            f"{group_name}: points={group['points']} "
            f"base={group['baseline_correct']} method={group['method_correct']} "
            f"attn1={group['attention_top1_correct']} attnK={group['attention_topk_correct']}"
        )
        for name, signal in group["signals"].items():
            print(
                f"  {name:12s} @1/@3/@5/@10="
                f"{signal['at1']}/{signal['at3']}/{signal['at5']}/{signal['at10']} "
                f"median={signal['rank_median']}"
            )


if __name__ == "__main__":
    main()

"""Offline conflict analysis for attention/kernel/native-in-basin signals.

This script consumes existing basin-identity audit JSON files. It does not run
FLUX or SPair evaluation. The goal is to quantify whether raw attention,
geometry-filtered attention, and native descriptors inside attention basins are
complementary or conflicting before designing another feature-generation method.
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
    "oracle_not_eaten": lambda row: bool(row.get("attention_topk_pck_hit"))
    and not bool(row.get("method_pck_hit")),
    "basin_identity_collapse": lambda row: bool(row.get("attention_topk_pck_hit"))
    and not _basin_native_hit(row, "raw", 1)
    and not _basin_native_hit(row, "filtered", 1),
    "raw_rescue_filter_harm": lambda row: _kernel_hit(row, "raw_attention", 1)
    and not _kernel_hit(row, "filtered_attention", 1),
    "filter_rescue_raw_miss": lambda row: not _kernel_hit(row, "raw_attention", 1)
    and _kernel_hit(row, "filtered_attention", 1),
    "native_basin_rescue_raw_miss": lambda row: not _kernel_hit(row, "raw_attention", 1)
    and _basin_native_hit(row, "raw", 1),
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


def _median(values: list[float]) -> float | None:
    values = sorted(float(value) for value in values if value is not None)
    if not values:
        return None
    middle = len(values) // 2
    if len(values) % 2:
        return float(values[middle])
    return float((values[middle - 1] + values[middle]) / 2.0)


def _mean(values: list[float]) -> float | None:
    values = [float(value) for value in values if value is not None]
    return float(sum(values) / len(values)) if values else None


def _kernel_hit(row: dict[str, Any], signal: str, k: int) -> bool:
    audit = row.get("attention_kernel_audit")
    if not isinstance(audit, dict):
        return False
    return bool(audit.get("topk_hits", {}).get(f"{signal}@{int(k)}"))


def _kernel_rank(row: dict[str, Any], signal: str) -> int | None:
    audit = row.get("attention_kernel_audit")
    if not isinstance(audit, dict):
        return None
    rank = audit.get("ranks", {}).get(signal)
    return int(rank) if rank is not None else None


def _kernel_top1(row: dict[str, Any], signal: str) -> dict[str, Any]:
    audit = row.get("attention_kernel_audit")
    if not isinstance(audit, dict):
        return {}
    top1 = audit.get("top1", {}).get(signal, {})
    return top1 if isinstance(top1, dict) else {}


def _basin_name(which: str) -> str:
    if which == "raw":
        return "raw_basin_native_descriptor"
    if which == "filtered":
        return "filtered_basin_native_descriptor"
    raise ValueError(f"Unknown basin signal: {which}")


def _basin_hit(row: dict[str, Any], which: str) -> bool:
    audit = row.get("basin_identity_audit")
    if not isinstance(audit, dict):
        return False
    basin = audit.get("basins", {}).get(_basin_name(which), {})
    return bool(basin.get("attention_basin_has_pck_hit"))


def _basin_native_hit(row: dict[str, Any], which: str, k: int) -> bool:
    audit = row.get("basin_identity_audit")
    if not isinstance(audit, dict):
        return False
    return bool(audit.get("topk_hits", {}).get(f"{_basin_name(which)}@{int(k)}"))


def _basin_native_rank(row: dict[str, Any], which: str) -> int | None:
    audit = row.get("basin_identity_audit")
    if not isinstance(audit, dict):
        return None
    rank = audit.get("ranks", {}).get(_basin_name(which))
    return int(rank) if rank is not None else None


def _basin(row: dict[str, Any], which: str) -> dict[str, Any]:
    audit = row.get("basin_identity_audit")
    if not isinstance(audit, dict):
        return {}
    basin = audit.get("basins", {}).get(_basin_name(which), {})
    return basin if isinstance(basin, dict) else {}


def _numeric(row: dict[str, Any], path: tuple[str, ...]) -> float | None:
    value: Any = row
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _method_attention_rank(row: dict[str, Any]) -> int | None:
    rank = row.get("method_prediction_attention_rank")
    return int(rank) if isinstance(rank, int) else None


def _signal_bits(row: dict[str, Any]) -> dict[str, bool]:
    return {
        "baseline": bool(row.get("baseline_pck_hit")),
        "method": bool(row.get("method_pck_hit")),
        "raw1": _kernel_hit(row, "raw_attention", 1),
        "raw5": _kernel_hit(row, "raw_attention", 5),
        "raw20": _kernel_hit(row, "raw_attention", 20),
        "filtered1": _kernel_hit(row, "filtered_attention", 1),
        "filtered5": _kernel_hit(row, "filtered_attention", 5),
        "filtered20": _kernel_hit(row, "filtered_attention", 20),
        "raw_basin": _basin_hit(row, "raw"),
        "filtered_basin": _basin_hit(row, "filtered"),
        "native_raw_basin1": _basin_native_hit(row, "raw", 1),
        "native_raw_basin3": _basin_native_hit(row, "raw", 3),
        "native_raw_basin5": _basin_native_hit(row, "raw", 5),
        "native_raw_basin10": _basin_native_hit(row, "raw", 10),
        "native_filtered_basin1": _basin_native_hit(row, "filtered", 1),
        "native_filtered_basin3": _basin_native_hit(row, "filtered", 3),
        "native_filtered_basin5": _basin_native_hit(row, "filtered", 5),
        "native_filtered_basin10": _basin_native_hit(row, "filtered", 10),
    }


def _pattern(row: dict[str, Any]) -> str:
    bits = _signal_bits(row)
    fields = ["baseline", "method", "raw1", "filtered1", "native_raw_basin1", "raw20"]
    return "|".join(f"{name}={int(bits[name])}" for name in fields)


def _boolean_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    keys = [
        "baseline",
        "method",
        "raw1",
        "raw5",
        "raw20",
        "filtered1",
        "filtered5",
        "filtered20",
        "raw_basin",
        "filtered_basin",
        "native_raw_basin1",
        "native_raw_basin3",
        "native_raw_basin5",
        "native_raw_basin10",
        "native_filtered_basin1",
        "native_filtered_basin3",
        "native_filtered_basin5",
        "native_filtered_basin10",
    ]
    counts = Counter()
    for row in rows:
        bits = _signal_bits(row)
        for key in keys:
            counts[key] += int(bits[key])
    return {
        key: {"count": int(counts[key]), "rate": _rate(int(counts[key]), total)}
        for key in keys
    }


def _conflict_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    raw1 = [row for row in rows if _kernel_hit(row, "raw_attention", 1)]
    filtered1 = [row for row in rows if _kernel_hit(row, "filtered_attention", 1)]
    native_basin1 = [row for row in rows if _basin_native_hit(row, "raw", 1)]
    raw_or_filtered = [
        row for row in rows
        if _kernel_hit(row, "raw_attention", 1) or _kernel_hit(row, "filtered_attention", 1)
    ]
    raw_or_native_basin = [
        row for row in rows
        if _kernel_hit(row, "raw_attention", 1) or _basin_native_hit(row, "raw", 1)
    ]
    any_three = [
        row for row in rows
        if _kernel_hit(row, "raw_attention", 1)
        or _kernel_hit(row, "filtered_attention", 1)
        or _basin_native_hit(row, "raw", 1)
    ]
    all_three_miss_but_oracle = [
        row for row in rows
        if _kernel_hit(row, "raw_attention", 20)
        and not _kernel_hit(row, "raw_attention", 1)
        and not _kernel_hit(row, "filtered_attention", 1)
        and not _basin_native_hit(row, "raw", 1)
    ]
    return {
        "raw1_or_filtered1": {"count": len(raw_or_filtered), "rate": _rate(len(raw_or_filtered), total)},
        "raw1_or_native_raw_basin1": {
            "count": len(raw_or_native_basin),
            "rate": _rate(len(raw_or_native_basin), total),
        },
        "raw1_or_filtered1_or_native_raw_basin1": {
            "count": len(any_three),
            "rate": _rate(len(any_three), total),
        },
        "raw1_and_filtered1": {
            "count": sum(1 for row in raw1 if _kernel_hit(row, "filtered_attention", 1)),
            "rate": _rate(sum(1 for row in raw1 if _kernel_hit(row, "filtered_attention", 1)), total),
        },
        "raw1_only_vs_filtered1": {
            "count": sum(1 for row in raw1 if not _kernel_hit(row, "filtered_attention", 1)),
            "rate": _rate(sum(1 for row in raw1 if not _kernel_hit(row, "filtered_attention", 1)), total),
        },
        "filtered1_only_vs_raw1": {
            "count": sum(1 for row in filtered1 if not _kernel_hit(row, "raw_attention", 1)),
            "rate": _rate(sum(1 for row in filtered1 if not _kernel_hit(row, "raw_attention", 1)), total),
        },
        "native_raw_basin1_only_vs_raw1": {
            "count": sum(1 for row in native_basin1 if not _kernel_hit(row, "raw_attention", 1)),
            "rate": _rate(sum(1 for row in native_basin1 if not _kernel_hit(row, "raw_attention", 1)), total),
        },
        "raw20_but_all_three_top1_miss": {
            "count": len(all_three_miss_but_oracle),
            "rate": _rate(len(all_three_miss_but_oracle), total),
        },
    }


def _numeric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = {
        "raw_attention_margin": [row.get("attention_top1", {}).get("margin") for row in rows],
        "raw_attention_concentration": [row.get("attention_top1", {}).get("concentration") for row in rows],
        "native_global_margin": [row.get("native", {}).get("margin") for row in rows],
        "raw_kernel_rank": [_kernel_rank(row, "raw_attention") for row in rows],
        "filtered_kernel_rank": [_kernel_rank(row, "filtered_attention") for row in rows],
        "native_raw_basin_rank": [_basin_native_rank(row, "raw") for row in rows],
        "native_filtered_basin_rank": [_basin_native_rank(row, "filtered") for row in rows],
        "method_attention_rank": [_method_attention_rank(row) for row in rows],
        "raw_basin_pck_hit_count": [
            _basin(row, "raw").get("attention_basin_pck_hit_count") for row in rows
        ],
        "filtered_basin_pck_hit_count": [
            _basin(row, "filtered").get("attention_basin_pck_hit_count") for row in rows
        ],
        "raw_basin_native_top_attention_rank": [
            _basin(row, "raw").get("native_top1_in_basin", {}).get("attention_rank") for row in rows
        ],
        "filtered_basin_native_top_attention_rank": [
            _basin(row, "filtered").get("native_top1_in_basin", {}).get("attention_rank") for row in rows
        ],
        "raw_kernel_top1_score": [_kernel_top1(row, "raw_attention").get("score") for row in rows],
        "filtered_kernel_top1_score": [_kernel_top1(row, "filtered_attention").get("score") for row in rows],
    }
    return {
        name: {
            "count": len([value for value in values if value is not None]),
            "mean": _mean([value for value in values if value is not None]),
            "median": _median([value for value in values if value is not None]),
        }
        for name, values in metrics.items()
    }


def _summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    patterns = Counter(_pattern(row) for row in rows)
    return {
        "points": len(rows),
        "signals": _boolean_summary(rows),
        "conflicts": _conflict_summary(rows),
        "numeric": _numeric_summary(rows),
        "top_patterns": [
            {"pattern": pattern, "count": int(count), "rate": _rate(int(count), len(rows))}
            for pattern, count in patterns.most_common(12)
        ],
        "categories": dict(Counter(str(row.get("category")) for row in rows)),
    }


def _category_table(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_category[str(row.get("category"))].append(row)
    table = []
    for category in sorted(by_category):
        rows = by_category[category]
        signals = _boolean_summary(rows)
        conflicts = _conflict_summary(rows)
        table.append({
            "category": category,
            "points": len(rows),
            "baseline": signals["baseline"]["count"],
            "method": signals["method"]["count"],
            "raw1": signals["raw1"]["count"],
            "filtered1": signals["filtered1"]["count"],
            "raw20": signals["raw20"]["count"],
            "native_raw_basin1": signals["native_raw_basin1"]["count"],
            "raw1_or_filtered1_or_native_raw_basin1": conflicts[
                "raw1_or_filtered1_or_native_raw_basin1"
            ]["count"],
            "raw20_but_all_three_top1_miss": conflicts["raw20_but_all_three_top1_miss"]["count"],
        })
    return table


def _hard_cases(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases = []
    for row in records:
        if not _kernel_hit(row, "raw_attention", 20):
            continue
        hard = (
            not _kernel_hit(row, "raw_attention", 1)
            and not _kernel_hit(row, "filtered_attention", 1)
            and not _basin_native_hit(row, "raw", 1)
        )
        if not hard:
            continue
        cases.append({
            "category": row.get("category"),
            "pair_json": row.get("pair_json"),
            "keypoint_index": row.get("keypoint_index"),
            "baseline_hit": bool(row.get("baseline_pck_hit")),
            "method_hit": bool(row.get("method_pck_hit")),
            "raw_rank": _kernel_rank(row, "raw_attention"),
            "filtered_rank": _kernel_rank(row, "filtered_attention"),
            "native_raw_basin_rank": _basin_native_rank(row, "raw"),
            "native_filtered_basin_rank": _basin_native_rank(row, "filtered"),
            "raw_basin_pck_hit_count": _basin(row, "raw").get("attention_basin_pck_hit_count"),
            "raw_attention_margin": row.get("attention_top1", {}).get("margin"),
            "raw_attention_concentration": row.get("attention_top1", {}).get("concentration"),
            "native_global_margin": row.get("native", {}).get("margin"),
            "method_attention_rank": _method_attention_rank(row),
            "source_point": row.get("source_point"),
            "target_point": row.get("target_point"),
            "baseline_prediction": row.get("baseline_prediction"),
            "method_prediction": row.get("method_prediction"),
        })
    return cases


def analyze(records: list[dict[str, Any]], main_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    has_kernel = any(isinstance(row.get("attention_kernel_audit"), dict) for row in records)
    has_basin = any(isinstance(row.get("basin_identity_audit"), dict) for row in records)
    return {
        "records": len(records),
        "matcher": main_payload.get("matcher") if isinstance(main_payload, dict) else None,
        "input_capabilities": {
            "has_attention_kernel_audit": has_kernel,
            "has_basin_identity_audit": has_basin,
            "can_run_offline_without_feature_recompute": has_kernel and has_basin,
        },
        "main_all": main_payload.get("all") if isinstance(main_payload, dict) else None,
        "groups": {
            name: _summarize_group([row for row in records if predicate(row)])
            for name, predicate in GROUPS.items()
        },
        "category_table": _category_table(records),
        "hard_cases_count": len(_hard_cases(records)),
        "interpretation_keys": {
            "raw20_but_all_three_top1_miss": (
                "Attention sees a correct top20 candidate, but raw top1, filtered top1, "
                "and native-within-raw-basin top1 all miss. This is the strongest evidence "
                "for within-basin identity collapse."
            ),
            "raw1_or_filtered1_or_native_raw_basin1": (
                "Upper bound of three diagnostic signals if one were allowed to choose "
                "per point. This is not a proposed method; it measures signal complementarity."
            ),
        },
    }


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else ["empty"]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("basin_identity_audit_json")
    parser.add_argument("--main_json", default=None)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--category_csv", default=None)
    parser.add_argument("--hard_cases_csv", default=None)
    args = parser.parse_args()

    payload = _load_json(args.basin_identity_audit_json)
    records = _records(payload)
    main_payload = _load_json(args.main_json) if args.main_json else None
    summary = analyze(records, main_payload if isinstance(main_payload, dict) else None)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
    if args.category_csv:
        _write_csv(args.category_csv, summary["category_table"])
    if args.hard_cases_csv:
        _write_csv(args.hard_cases_csv, _hard_cases(records))

    for group_name in (
        "all",
        "oracle_gap",
        "attention_harms_native",
        "attention_rescues_native",
        "basin_identity_collapse",
    ):
        group = summary["groups"][group_name]
        signals = group["signals"]
        conflicts = group["conflicts"]
        print(
            f"{group_name}: points={group['points']} "
            f"base/method={signals['baseline']['count']}/{signals['method']['count']} "
            f"raw1/filter1/nativeB1/raw20="
            f"{signals['raw1']['count']}/{signals['filtered1']['count']}/"
            f"{signals['native_raw_basin1']['count']}/{signals['raw20']['count']} "
            f"any3={conflicts['raw1_or_filtered1_or_native_raw_basin1']['count']} "
            f"raw20_all3miss={conflicts['raw20_but_all_three_top1_miss']['count']}"
        )
    print("hard_cases_count:", summary["hard_cases_count"])
    print("offline:", summary["input_capabilities"]["can_run_offline_without_feature_recompute"])


if __name__ == "__main__":
    main()

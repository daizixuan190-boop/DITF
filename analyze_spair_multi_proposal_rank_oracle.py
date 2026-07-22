"""Measure top-M candidate rank ceilings for relational ownership methods.

This is an annotation-only diagnostic. It asks whether at least one candidate
inside the top-M method scores is PCK-correct while retaining the baseline for
all other points. The resulting perfect-selector PCK is an upper bound, not a
valid training-free result.
"""

import argparse
import csv
import json
import os
from typing import Any


def parse_int(row: dict[str, str], field: str, default: int | None = None) -> int | None:
    value = row.get(field)
    if value in (None, "", "None"):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def load_rows(path: str) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def row_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row.get("category", ""), row.get("pair_name", ""), row.get("kp_idx", ""))


def select_subset(rows: list[dict[str, str]], subset: str) -> list[dict[str, str]]:
    failures = [row for row in rows if parse_int(row, "baseline_correct", 0) == 0]
    if subset == "all_failures":
        return failures
    if subset == "global_union_failures":
        return [row for row in failures if parse_int(row, "gt_union_available", 0) == 1]
    if subset == "strict_non_overlap_union_failures":
        return [
            row
            for row in failures
            if parse_int(row, "gt_union_available", 0) == 1
            and parse_int(row, "gt_candidate_overlaps_other_pck", 0) == 0
        ]
    raise ValueError(f"Unknown subset: {subset}")


def summarize_rank_source(
    rows: list[dict[str, str]],
    subset_rows: list[dict[str, str]],
    rank_field: str,
    topk: list[int],
) -> dict[str, Any]:
    total = len(rows)
    subset_count = len(subset_rows)
    baseline_correct = sum(parse_int(row, "baseline_correct", 0) or 0 for row in rows)
    baseline_pck = baseline_correct / total if total else 0.0
    result: dict[str, Any] = {
        "rank_field": rank_field,
        "subset_count": subset_count,
        "baseline_pck": baseline_pck,
        "topk": {},
    }
    for k in topk:
        recoverable = sum(
            1
            for row in subset_rows
            if (rank := parse_int(row, rank_field)) is not None and rank <= k
        )
        global_delta = recoverable / total if total else 0.0
        result["topk"][str(k)] = {
            "recoverable_count": recoverable,
            "subset_recall": recoverable / subset_count if subset_count else 0.0,
            "global_delta_upper_bound": global_delta,
            "perfect_selector_pck_upper_bound": baseline_pck + global_delta,
        }
    return result


def summarize_method(rows: list[dict[str, str]], topk: list[int]) -> dict[str, Any]:
    subsets = (
        "all_failures",
        "global_union_failures",
        "strict_non_overlap_union_failures",
    )
    return {
        "count": len(rows),
        "subsets": {
            subset: {
                "raw_scores": summarize_rank_source(
                    rows,
                    select_subset(rows, subset),
                    "raw_gt_union_rank",
                    topk,
                ),
                "method_scores": summarize_rank_source(
                    rows,
                    select_subset(rows, subset),
                    "final_gt_union_rank",
                    topk,
                ),
            }
            for subset in subsets
        },
    }


def paired_rank_comparison(
    v1_rows: list[dict[str, str]],
    v2_rows: list[dict[str, str]],
    topk: list[int],
) -> dict[str, Any]:
    v1 = {row_key(row): row for row in v1_rows}
    v2 = {row_key(row): row for row in v2_rows}
    shared_keys = sorted(set(v1) & set(v2))
    result: dict[str, Any] = {
        "shared_key_count": len(shared_keys),
        "baseline_disagreement_count": 0,
        "subsets": {},
    }
    for key in shared_keys:
        if parse_int(v1[key], "baseline_correct", 0) != parse_int(v2[key], "baseline_correct", 0):
            result["baseline_disagreement_count"] += 1

    subset_names = (
        "all_failures",
        "global_union_failures",
        "strict_non_overlap_union_failures",
    )
    for subset in subset_names:
        v1_subset_keys = {row_key(row) for row in select_subset(v1_rows, subset)}
        v2_subset_keys = {row_key(row) for row in select_subset(v2_rows, subset)}
        keys = sorted(set(shared_keys) & v1_subset_keys & v2_subset_keys)
        subset_result: dict[str, Any] = {"count": len(keys), "topk": {}}
        for k in topk:
            v1_recovered = {
                key
                for key in keys
                if (rank := parse_int(v1[key], "final_gt_union_rank")) is not None and rank <= k
            }
            v2_recovered = {
                key
                for key in keys
                if (rank := parse_int(v2[key], "final_gt_union_rank")) is not None and rank <= k
            }
            subset_result["topk"][str(k)] = {
                "v1_recoverable_count": len(v1_recovered),
                "v2_recoverable_count": len(v2_recovered),
                "shared_recoverable_count": len(v1_recovered & v2_recovered),
                "v1_only_count": len(v1_recovered - v2_recovered),
                "v2_only_count": len(v2_recovered - v1_recovered),
                "v2_minus_v1_count": len(v2_recovered) - len(v1_recovered),
            }
        result["subsets"][subset] = subset_result
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze top-M relational proposal rank ceilings")
    parser.add_argument("--v1_csv", required=True)
    parser.add_argument("--v2_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--topk", nargs="+", type=int, default=[1, 3, 5, 10, 20, 50])
    args = parser.parse_args()
    topk = sorted({max(int(value), 1) for value in args.topk})

    v1_rows = load_rows(args.v1_csv)
    v2_rows = load_rows(args.v2_csv)
    summary = {
        "topk": topk,
        "v1": {"source_csv": args.v1_csv, **summarize_method(v1_rows, topk)},
        "v2": {"source_csv": args.v2_csv, **summarize_method(v2_rows, topk)},
        "paired_v1_v2": paired_rank_comparison(v1_rows, v2_rows, topk),
    }

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "multi_proposal_rank_oracle_summary.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    for method in ("v1", "v2"):
        print(f"{method} method-score perfect-selector upper bounds:")
        for subset, values in summary[method]["subsets"].items():
            compact = {
                k: {
                    "subset_recall": round(item["subset_recall"], 6),
                    "global_delta": round(item["global_delta_upper_bound"], 6),
                }
                for k, item in values["method_scores"]["topk"].items()
            }
            print(f"  {subset}: {compact}")
    print("Paired v1/v2:", summary["paired_v1_v2"])
    print(f"Saved summary to: {output_path}")


if __name__ == "__main__":
    main()

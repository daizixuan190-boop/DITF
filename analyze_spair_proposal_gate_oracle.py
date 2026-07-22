"""Decompose proposal quality from gate quality for relational ownership methods.

The proposal oracle is an evaluation-only diagnostic: it scores the ungated
argmax proposal with SPair annotations, but it never changes inference. This
separates a bad ownership proposal from a correct proposal rejected by the
method gate.
"""

import argparse
import csv
import json
import math
import os
from collections import Counter
from typing import Any


def parse_int(row: dict[str, str], field: str, default: int = 0) -> int:
    value = row.get(field)
    if value in (None, "", "None"):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_float(row: dict[str, str], field: str) -> float | None:
    value = row.get(field)
    if value in (None, "", "None"):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def load_rows(path: str) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def row_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row.get("category", ""), row.get("pair_name", ""), row.get("kp_idx", ""))


def subset_rows(rows: list[dict[str, str]], subset: str) -> list[dict[str, str]]:
    if subset == "all":
        return rows
    if subset == "global_union_failure":
        return [
            row
            for row in rows
            if parse_int(row, "baseline_correct") == 0
            and parse_int(row, "gt_union_available") == 1
        ]
    if subset == "strict_non_overlap_union_failure":
        return [
            row
            for row in rows
            if parse_int(row, "baseline_correct") == 0
            and parse_int(row, "gt_union_available") == 1
            and parse_int(row, "gt_candidate_overlaps_other_pck") == 0
        ]
    if subset == "transform_valid":
        return [row for row in rows if parse_int(row, "transform_valid") == 1]
    if subset == "gate_blocked":
        return [row for row in rows if parse_int(row, "gate_blocked") == 1]
    raise ValueError(f"Unknown subset: {subset}")


def summarize(rows: list[dict[str, str]]) -> dict[str, Any]:
    count = len(rows)
    baseline_correct = [parse_int(row, "baseline_correct") for row in rows]
    final_correct = [parse_int(row, "method_correct") for row in rows]
    proposal_values = [parse_int(row, "proposal_correct") for row in rows]
    proposal_available = [
        row for row in rows if parse_float(row, "proposal_norm_dist") is not None
    ]
    proposal_available_values = [parse_int(row, "proposal_correct") for row in proposal_available]

    def mean(values: list[int]) -> float | None:
        return sum(values) / len(values) if values else None

    def rate(values: list[int]) -> float:
        return sum(values) / count if count else 0.0

    proposal_correct = [
        parse_int(row, "proposal_correct")
        for row in rows
        if parse_float(row, "proposal_norm_dist") is not None
    ]
    proposal_improvements = [
        int(parse_int(row, "baseline_correct") == 0 and parse_int(row, "proposal_correct") == 1)
        for row in rows
        if parse_float(row, "proposal_norm_dist") is not None
    ]
    proposal_harms = [
        int(parse_int(row, "baseline_correct") == 1 and parse_int(row, "proposal_correct") == 0)
        for row in rows
        if parse_float(row, "proposal_norm_dist") is not None
    ]
    oracle_gate_correct = [
        max(parse_int(row, "baseline_correct"), parse_int(row, "proposal_correct"))
        for row in rows
        if parse_float(row, "proposal_norm_dist") is not None
    ]
    gate_missed_recovery = [
        int(
            parse_int(row, "baseline_correct") == 0
            and parse_int(row, "proposal_correct") == 1
            and parse_int(row, "method_correct") == 0
        )
        for row in rows
        if parse_float(row, "proposal_norm_dist") is not None
    ]
    gate_accepted_recovery = [
        int(
            parse_int(row, "baseline_correct") == 0
            and parse_int(row, "proposal_correct") == 1
            and parse_int(row, "method_correct") == 1
        )
        for row in rows
        if parse_float(row, "proposal_norm_dist") is not None
    ]
    gate_avoided_harm = [
        int(
            parse_int(row, "baseline_correct") == 1
            and parse_int(row, "proposal_correct") == 0
            and parse_int(row, "method_correct") == 1
        )
        for row in rows
        if parse_float(row, "proposal_norm_dist") is not None
    ]
    gate_accepted_harm = [
        int(
            parse_int(row, "baseline_correct") == 1
            and parse_int(row, "proposal_correct") == 0
            and parse_int(row, "method_correct") == 0
        )
        for row in rows
        if parse_float(row, "proposal_norm_dist") is not None
    ]
    gate_blocked = [parse_int(row, "gate_blocked") for row in rows]
    proposal_changed = [
        int(row.get("base_column") != row.get("proposed_column"))
        for row in rows
        if row.get("proposed_column") not in (None, "")
    ]

    result = {
        "count": count,
        "proposal_available_count": len(proposal_available),
        "baseline_pck": mean(baseline_correct),
        "final_pck": mean(final_correct),
        "final_delta_vs_baseline": (mean(final_correct) - mean(baseline_correct))
        if count
        else 0.0,
        "final_improvement_rate": rate([
            int(b == 0 and f == 1) for b, f in zip(baseline_correct, final_correct)
        ]),
        "final_harm_rate": rate([
            int(b == 1 and f == 0) for b, f in zip(baseline_correct, final_correct)
        ]),
        "proposal_pck": mean(proposal_correct) if proposal_available else None,
        "proposal_delta_vs_baseline": (
            mean(proposal_correct) - mean(baseline_correct)
            if proposal_available
            else None
        ),
        "proposal_improvement_rate": sum(proposal_improvements) / count if count else 0.0,
        "proposal_harm_rate": sum(proposal_harms) / count if count else 0.0,
        "perfect_gate_pck_upper_bound": (
            sum(oracle_gate_correct) / count if oracle_gate_correct else None
        ),
        "perfect_gate_delta_vs_baseline": (
            sum(oracle_gate_correct) / count - mean(baseline_correct)
            if oracle_gate_correct and count
            else None
        ),
        "proposal_recovery_rate_upper_bound": (
            sum(proposal_improvements) / count if proposal_available else None
        ),
        "accepted_recovery_efficiency": (
            sum(gate_accepted_recovery) / sum(proposal_improvements)
            if proposal_improvements and sum(proposal_improvements)
            else None
        ),
        "proposal_harm_filter_rate": (
            sum(gate_avoided_harm) / sum(proposal_harms)
            if proposal_harms and sum(proposal_harms)
            else None
        ),
        "proposal_changed_rate": sum(proposal_changed) / count if count else 0.0,
        "gate_blocked_rate": sum(gate_blocked) / count if count else 0.0,
        "gate_missed_recovery_rate": sum(gate_missed_recovery) / count if count else 0.0,
        "gate_accepted_recovery_rate": sum(gate_accepted_recovery) / count if count else 0.0,
        "gate_avoided_harm_rate": sum(gate_avoided_harm) / count if count else 0.0,
        "gate_accepted_harm_rate": sum(gate_accepted_harm) / count if count else 0.0,
    }
    if proposal_available:
        proposal_change_count = sum(proposal_changed)
        result["proposal_changed_count"] = proposal_change_count
        result["proposal_changed_precision"] = (
            sum(proposal_improvements) / proposal_change_count
            if proposal_change_count
            else 0.0
        )
    return result


def compare_final_rows(v1_rows: list[dict[str, str]], v2_rows: list[dict[str, str]]) -> dict[str, Any]:
    v1 = {row_key(row): row for row in v1_rows}
    v2 = {row_key(row): row for row in v2_rows}
    shared = sorted(set(v1) & set(v2))
    v2_better = 0
    v2_worse = 0
    same = 0
    baseline_disagreement = 0
    for key in shared:
        a = v1[key]
        b = v2[key]
        a_base = parse_int(a, "baseline_correct")
        b_base = parse_int(b, "baseline_correct")
        if a_base != b_base:
            baseline_disagreement += 1
        a_final = parse_int(a, "method_correct")
        b_final = parse_int(b, "method_correct")
        if b_final > a_final:
            v2_better += 1
        elif b_final < a_final:
            v2_worse += 1
        else:
            same += 1
    return {
        "v1_count": len(v1_rows),
        "v2_count": len(v2_rows),
        "shared_key_count": len(shared),
        "shared_key_fraction_v1": len(shared) / len(v1_rows) if v1_rows else 0.0,
        "shared_key_fraction_v2": len(shared) / len(v2_rows) if v2_rows else 0.0,
        "baseline_disagreement_count": baseline_disagreement,
        "v2_better_final_count": v2_better,
        "v2_worse_final_count": v2_worse,
        "same_final_count": same,
        "v2_net_point_advantage_over_v1": (v2_better - v2_worse) / len(shared)
        if shared
        else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze proposal versus gate oracle outcomes")
    parser.add_argument("--v2_csv", required=True)
    parser.add_argument("--v1_csv", default="")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    v2_rows = load_rows(args.v2_csv)
    subsets = ["all", "global_union_failure", "strict_non_overlap_union_failure", "transform_valid", "gate_blocked"]
    summary: dict[str, Any] = {
        "v2": {
            "source_csv": args.v2_csv,
            "subsets": {name: summarize(subset_rows(v2_rows, name)) for name in subsets},
        }
    }
    if args.v1_csv:
        v1_rows = load_rows(args.v1_csv)
        summary["v1"] = {
            "source_csv": args.v1_csv,
            "subsets": {name: summarize(subset_rows(v1_rows, name)) for name in ["all", "global_union_failure", "strict_non_overlap_union_failure"]},
        }
        summary["v1_vs_v2_final"] = compare_final_rows(v1_rows, v2_rows)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "proposal_gate_oracle_summary.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    for method_name, method_summary in summary.items():
        if method_name not in ("v1", "v2"):
            continue
        print(f"{method_name}:")
        for subset_name, values in method_summary["subsets"].items():
            print(f"  {subset_name}: {values}")
    if "v1_vs_v2_final" in summary:
        print("v1 vs v2 final:", summary["v1_vs_v2_final"])
    print(f"Saved summary to: {output_path}")


if __name__ == "__main__":
    main()

"""Synthesize offline FJSAR attention-mechanism audit evidence.

This script reads existing JSON dumps only.  It does not implement a matcher
and cannot recover dense features or full attention rows that were not dumped.
Its purpose is to quantify which mechanistic hypotheses remain supported by
the current evidence.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
from typing import Any


def _load(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_float(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _pct(count: float, total: float) -> float:
    return 100.0 * float(count) / float(total) if total else 0.0


def _fmt(value: Any, digits: int = 2) -> str:
    number = _safe_float(value)
    return "n/a" if number is None else f"{number:.{digits}f}"


def _signal(summary: dict[str, Any], group: str, name: str) -> dict[str, Any]:
    return (
        summary.get(group, {})
        .get("signals", {})
        .get(name, {})
        if isinstance(summary, dict)
        else {}
    )


def _rank_counts(signal: dict[str, Any]) -> dict[str, Any]:
    ranked = int(signal.get("ranked_points", 0) or 0)
    return {
        "ranked": ranked,
        "at1": int(signal.get("proposal_pck_hit_at_1", signal.get("hit_at_1", 0)) or 0),
        "at3": int(signal.get("proposal_pck_hit_at_3", signal.get("hit_at_3", 0)) or 0),
        "at5": int(signal.get("proposal_pck_hit_at_5", signal.get("hit_at_5", 0)) or 0),
        "at10": int(signal.get("proposal_pck_hit_at_10", signal.get("hit_at_10", 0)) or 0),
        "median_rank": signal.get("proposal_pck_hit_rank_median"),
        "gap_median": signal.get("attention_top1_minus_best_pck_hit_proposal_gap_median"),
    }


def _records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    records = payload.get("records", [])
    return records if isinstance(records, list) else []


def _candidate_method_in_attention(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups = {
        "all": lambda row: True,
        "oracle_gap": lambda row: bool(row.get("oracle_gap_case")),
        "attention_harms_native": lambda row: bool(row.get("attention_harms_native_case")),
    }
    out: dict[str, Any] = {}
    for name, pred in groups.items():
        rows = [row for row in records if pred(row)]
        in_attention = sum(1 for row in rows if bool(row.get("method_prediction_in_attention_proposals")))
        out[name] = {
            "records": len(rows),
            "method_prediction_in_attention_proposals": in_attention,
            "rate": in_attention / len(rows) if rows else 0.0,
        }
    return out


def _single_signal_lift_bounds(
    main: dict[str, Any],
    residual_summary: dict[str, Any],
    flow_summary: dict[str, Any],
    basin_summary: dict[str, Any],
    kernel_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    total = int(main.get("all", {}).get("matcher_diagnostics", {}).get("model_counts", {}).get("fjsar_oracle_total", 0) or 0)
    items: list[tuple[str, dict[str, Any], str]] = []
    for name in (
        "raw_head_probability",
        "logit_gap_to_head_peak",
        "value_residual_alignment",
        "value_alignment_without_common_removal",
        "value_residual_energy",
        "residual_alignment_times_energy",
        "negative_common_similarity",
    ):
        items.append((f"residual:{name}", _signal(residual_summary, "oracle_gap", name), "proposal_pck_hit_at_1"))
    for name in (
        "transport_consistency",
        "inverse_transport_consistency",
        "shape_preservation",
        "negative_mean_displacement_error",
        "negative_displacement_entropy",
    ):
        items.append((f"flow:{name}", _signal(flow_summary, "oracle_gap", name), "proposal_pck_hit_at_1"))
    for name in ("raw_basin_native_descriptor", "filtered_basin_native_descriptor"):
        items.append((f"basin:{name}", _signal(basin_summary, "oracle_gap", name), "native_in_basin_hit_at_1"))
    for name in ("raw_attention", "filtered_attention"):
        items.append((f"kernel:{name}", _signal(kernel_summary, "oracle_gap", name), "hit_at_1"))

    rows = []
    for name, signal, key in items:
        count = int(signal.get(key, 0) or 0)
        ranked = int(signal.get("ranked_points", 0) or signal.get("points", 0) or 0)
        rows.append({
            "signal": name,
            "oracle_gap_top1_hits": count,
            "oracle_gap_ranked_points": ranked,
            "best_case_point_gain_if_no_harm": (100.0 * count / total) if total else None,
        })
    rows.sort(key=lambda row: row["oracle_gap_top1_hits"], reverse=True)
    return rows


def _common_removal_delta(residual_summary: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for group in ("oracle_gap", "attention_harms_native", "all"):
        residual = _rank_counts(_signal(residual_summary, group, "value_residual_alignment"))
        raw_value = _rank_counts(_signal(residual_summary, group, "value_alignment_without_common_removal"))
        out[group] = {
            "value_residual_alignment_at1": residual["at1"],
            "value_alignment_without_common_removal_at1": raw_value["at1"],
            "at1_delta_common_free_minus_raw_value": residual["at1"] - raw_value["at1"],
            "value_residual_alignment_at3": residual["at3"],
            "value_alignment_without_common_removal_at3": raw_value["at3"],
            "at3_delta_common_free_minus_raw_value": residual["at3"] - raw_value["at3"],
        }
    return out


def _best_residual_by_group(residual_summary: dict[str, Any], group: str) -> list[dict[str, Any]]:
    rows = []
    for name, signal in residual_summary.get(group, {}).get("signals", {}).items():
        counts = _rank_counts(signal)
        rows.append({
            "signal": name,
            **counts,
            "at1_rate": counts["at1"] / counts["ranked"] if counts["ranked"] else 0.0,
            "at3_rate": counts["at3"] / counts["ranked"] if counts["ranked"] else 0.0,
        })
    rows.sort(key=lambda row: (row["at1"], row["at3"]), reverse=True)
    return rows


def _method_damage_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    selected = len(records)
    baseline_correct = sum(1 for row in records if bool(row.get("baseline_pck_hit")))
    method_correct = sum(1 for row in records if bool(row.get("method_pck_hit")))
    native_correct_method_wrong = sum(
        1 for row in records
        if bool(row.get("baseline_pck_hit")) and not bool(row.get("method_pck_hit"))
    )
    native_wrong_method_correct = sum(
        1 for row in records
        if not bool(row.get("baseline_pck_hit")) and bool(row.get("method_pck_hit"))
    )
    return {
        "selected_records": selected,
        "baseline_correct_selected": baseline_correct,
        "method_correct_selected": method_correct,
        "native_correct_method_wrong": native_correct_method_wrong,
        "native_wrong_method_correct": native_wrong_method_correct,
        "net_selected_delta": method_correct - baseline_correct,
    }


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = []
    main = report["main"]
    lines.append("# Attention Mechanism Synthesis")
    lines.append("")
    lines.append("## Main Scores")
    lines.append(f"- baseline point: {_fmt(main.get('baseline_point'))}")
    lines.append(f"- method point: {_fmt(main.get('method_point'))}")
    lines.append(f"- point gain: {_fmt(main.get('point_gain'))}")
    lines.append("")
    lines.append("## Oracle")
    for row in report["oracle"]:
        lines.append(f"- {row['name']}: {row['count']} / {row['total']} = {_fmt(row['rate'])}%")
    lines.append("")
    lines.append("## Common Removal Check")
    for group, item in report["common_removal_delta"].items():
        lines.append(
            f"- {group}: common-free @1 {item['value_residual_alignment_at1']} vs raw-value @1 "
            f"{item['value_alignment_without_common_removal_at1']} "
            f"(delta {item['at1_delta_common_free_minus_raw_value']})"
        )
    lines.append("")
    lines.append("## Best Residual Signals")
    for group, rows in report["best_residual_by_group"].items():
        lines.append(f"### {group}")
        for row in rows[:5]:
            lines.append(
                f"- {row['signal']}: @1 {row['at1']}/{row['ranked']} "
                f"({_fmt(100 * row['at1_rate'])}%), @3 {row['at3']} "
                f"median-rank {row['median_rank']}"
            )
    lines.append("")
    lines.append("## Single-Signal Best-Case Lift Bounds")
    for row in report["single_signal_lift_bounds"][:10]:
        lines.append(
            f"- {row['signal']}: oracle-gap @1 {row['oracle_gap_top1_hits']}, "
            f"best-case no-harm gain {_fmt(row['best_case_point_gain_if_no_harm'])}"
        )
    lines.append("")
    lines.append("## Method Damage")
    for key, value in report["method_damage"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Interpretation")
    for item in report["interpretation"]:
        lines.append(f"- {item}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    main_payload = _load(args.main)
    residual_payload = _load(args.residual)
    flow_payload = _load(args.flow)
    basin_payload = _load(args.basin)
    kernel_payload = _load(args.kernel)
    candidate_payload = _load(args.candidates or args.residual)

    main_all = main_payload.get("all", {})
    model_counts = main_all.get("matcher_diagnostics", {}).get("model_counts", {})
    total = int(model_counts.get("fjsar_oracle_total", 0) or 0)
    oracle = []
    for owner in ("native", "attention", "attention_isometry"):
        for k in (1, 5, 10, 20, 50):
            key = f"fjsar_oracle_owner_{owner}@{k}"
            if key in model_counts:
                count = int(model_counts[key])
                oracle.append({
                    "name": f"{owner}@{k}",
                    "count": count,
                    "total": total,
                    "rate": _pct(count, total),
                })

    residual_summary = residual_payload.get("summary", main_payload.get("residual_readout_audit_summary", {}))
    flow_summary = flow_payload.get("summary", {})
    basin_summary = basin_payload.get("summary", {})
    kernel_summary = kernel_payload.get("summary", {})
    candidate_records = _records(candidate_payload)

    interpretation = [
        "Cross-attention high recall is supported when attention@20 is much larger than native@20.",
        "Single-token value residual is not a sufficient root fix if common-free value alignment is not better than raw value alignment.",
        "A signal whose oracle-gap @1 gives only a small best-case no-harm gain cannot plausibly deliver +5 without additional structure.",
        "If method predictions rarely stay inside attention proposals, the current descriptor method is not converting the high-recall basin into top1 matches.",
        "Local relational or coarse-to-fine evidence cannot be fully tested from these JSON files because dense patch features and full attention rows were not dumped.",
    ]

    return {
        "main": {
            "baseline_point": main_all.get("baseline_point"),
            "method_point": main_all.get("method_point"),
            "point_gain": main_all.get("point_gain"),
            "changed_count": main_all.get("changed_count"),
            "improved_count": main_all.get("improved_count"),
            "harmed_count": main_all.get("harmed_count"),
        },
        "oracle": oracle,
        "candidate_summary": candidate_payload.get("summary", main_payload.get("candidate_dump_summary", {})),
        "method_in_attention_proposals": _candidate_method_in_attention(candidate_records),
        "common_removal_delta": _common_removal_delta(residual_summary),
        "best_residual_by_group": {
            group: _best_residual_by_group(residual_summary, group)
            for group in ("oracle_gap", "attention_harms_native", "all")
        },
        "single_signal_lift_bounds": _single_signal_lift_bounds(
            main_payload,
            residual_summary,
            flow_summary,
            basin_summary,
            kernel_summary,
        ),
        "method_damage": _method_damage_summary(candidate_records),
        "interpretation": interpretation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", required=True, help="Main eval JSON.")
    parser.add_argument("--residual", required=True, help="Residual-readout audit JSON.")
    parser.add_argument("--candidates", default="", help="Candidate dump JSON. Defaults to --residual records.")
    parser.add_argument("--flow", default="", help="Optional attention-flow audit JSON from a comparable run.")
    parser.add_argument("--basin", default="", help="Optional basin-identity audit JSON from a comparable run.")
    parser.add_argument("--kernel", default="", help="Optional attention-kernel audit JSON from a comparable run.")
    parser.add_argument("--output_json", default="", help="Optional synthesized report JSON path.")
    parser.add_argument("--output_md", default="", help="Optional synthesized report Markdown path.")
    args = parser.parse_args()

    report = build_report(args)
    text = json.dumps(report, indent=2)
    print(text)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    if args.output_md:
        path = Path(args.output_md)
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(report, path)


if __name__ == "__main__":
    main()

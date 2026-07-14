import argparse
import csv
import json
import math
import os
from collections import defaultdict
from typing import Any

import numpy as np


CONFUSION_KEY_FIELDS = ["category", "kp_idx", "best_other_idx_center"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Decompose scale-induced anchor erosion into source-side collapse, target-side collapse, "
            "bilateral collapse, or cross-view-only erosion. "
            "Consumes identity_side_diagnostics_records.csv and performs CPU-only post-hoc analysis."
        )
    )
    parser.add_argument("--records_csv", type=str, required=True, help="Path to identity_side_diagnostics_records.csv.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs.")
    parser.add_argument("--scale_field", type=str, default="scale_variation", help="Pair-level scale field.")
    parser.add_argument("--context_radius", type=int, default=0, help="Context radius r used for side diagnostics.")
    parser.add_argument("--min_count_per_scale", type=int, default=10, help="Minimum support per scale group.")
    parser.add_argument("--boundary_margin", type=float, default=0.05, help="Boundary threshold for baseline margin regime.")
    parser.add_argument(
        "--dominance_gap",
        type=float,
        default=0.01,
        help="Minimum gap between target-side and source-side rival-sim increase to call a side dominant.",
    )
    parser.add_argument(
        "--erosion_margin_threshold",
        type=float,
        default=0.0,
        help="delta_cross_margin must be below this value to count as an erosion case.",
    )
    parser.add_argument(
        "--side_rise_threshold",
        type=float,
        default=0.0,
        help="Minimum rival-sim increase to count as a side getting worse.",
    )
    parser.add_argument("--topk", type=int, default=30, help="Number of top erosion cases to keep.")
    return parser.parse_args()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def parse_scalar(value: str) -> Any:
    if value is None or value == "":
        return None
    try:
        num = float(value)
    except ValueError:
        return value
    if math.isfinite(num) and abs(num - round(num)) < 1e-12:
        return int(round(num))
    return num


def load_csv(csv_path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({key: parse_scalar(value) for key, value in row.items()})
    return records


def write_records_csv(records: list[dict[str, Any]], csv_path: str):
    if not records:
        return
    fieldnames = sorted({key for record in records for key in record.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def safe_rate(values: list[int]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def require_columns(records: list[dict[str, Any]], columns: list[str]):
    if not records:
        raise RuntimeError("records_csv is empty.")
    missing = [column for column in columns if column not in records[0]]
    if missing:
        raise RuntimeError(f"records_csv is missing required columns: {missing}")


def normalize_scalar(value: Any) -> Any:
    if isinstance(value, float) and abs(value - round(value)) < 1e-12:
        return int(round(value))
    return value


def sort_scale_values(values: set[Any]) -> list[Any]:
    normalized = [normalize_scalar(value) for value in values]
    if all(isinstance(value, (int, float, bool)) for value in normalized):
        return sorted(normalized, key=float)
    return sorted(normalized, key=str)


def subtract_or_none(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    return float(a) - float(b)


def support_weight(record: dict[str, Any]) -> int:
    return min(int(record["count_a"]), int(record["count_b"]))


def weighted_mean(records: list[dict[str, Any]], value_key: str) -> float | None:
    numerator = 0.0
    denominator = 0.0
    for record in records:
        value = record.get(value_key)
        if value is None:
            continue
        weight = support_weight(record)
        numerator += float(value) * float(weight)
        denominator += float(weight)
    if denominator <= 0.0:
        return None
    return numerator / denominator


def record_failure(record: dict[str, Any]) -> int:
    return 1 - int(record["correct"])


def assign_margin_regime(margin: float | None, boundary_margin: float) -> str:
    if margin is None:
        return "unknown"
    if margin < -boundary_margin:
        return "already_flipped"
    if abs(margin) <= boundary_margin:
        return "near_boundary"
    return "anchored"


def sort_records(records: list[dict[str, Any]], key: str, reverse: bool = True) -> list[dict[str, Any]]:
    def sort_value(record: dict[str, Any]):
        value = record.get(key)
        if value is None:
            return -math.inf if reverse else math.inf
        return float(value)

    return sorted(records, key=sort_value, reverse=reverse)


def aggregate_pair_scale_stats(records: list[dict[str, Any]], scale_field: str, radius: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    src_key = f"src_rival_sim_r{radius}"
    trg_key = f"trg_rival_sim_r{radius}"
    cross_gt_key = f"cross_gt_score_r{radius}"
    cross_rival_key = f"cross_rival_score_r{radius}"
    cross_margin_key = f"cross_margin_r{radius}"
    target_more_key = f"target_more_collapsed_r{radius}"

    for record in records:
        if record.get("best_other_idx_center") is None or record.get(scale_field) is None:
            continue
        if int(record["best_other_idx_center"]) == int(record["kp_idx"]):
            continue
        key = tuple(record[field] for field in CONFUSION_KEY_FIELDS) + (normalize_scalar(record[scale_field]),)
        grouped[key].append(record)

    output: list[dict[str, Any]] = []
    for key, subset in grouped.items():
        failures = [record for record in subset if record_failure(record) == 1]
        out = {
            "category": key[0],
            "kp_idx": int(key[1]),
            "best_other_idx_center": int(key[2]),
            scale_field: key[3],
            "count": len(subset),
            "failure_count": len(failures),
            "failure_rate": safe_rate([record_failure(record) for record in subset]),
            "mean_center_margin": safe_mean([float(record["center_margin"]) for record in subset if record.get("center_margin") is not None]),
            "mean_src_rival_sim": safe_mean([float(record[src_key]) for record in subset if record.get(src_key) is not None]),
            "mean_trg_rival_sim": safe_mean([float(record[trg_key]) for record in subset if record.get(trg_key) is not None]),
            "mean_trg_minus_src_rival_sim": safe_mean(
                [float(record[trg_key]) - float(record[src_key]) for record in subset if record.get(src_key) is not None and record.get(trg_key) is not None]
            ),
            "target_more_collapsed_rate": safe_rate(
                [int(record[target_more_key]) for record in subset if record.get(target_more_key) is not None]
            ),
            "mean_cross_gt_score": safe_mean([float(record[cross_gt_key]) for record in subset if record.get(cross_gt_key) is not None]),
            "mean_cross_rival_score": safe_mean([float(record[cross_rival_key]) for record in subset if record.get(cross_rival_key) is not None]),
            "mean_cross_margin": safe_mean([float(record[cross_margin_key]) for record in subset if record.get(cross_margin_key) is not None]),
            "failure_mean_cross_margin": safe_mean([float(record[cross_margin_key]) for record in failures if record.get(cross_margin_key) is not None]),
        }
        output.append(out)
    return output


def build_transition_records(
    pair_scale_records: list[dict[str, Any]],
    scale_field: str,
    min_count_per_scale: int,
    boundary_margin: float,
    dominance_gap: float,
    erosion_margin_threshold: float,
    side_rise_threshold: float,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in pair_scale_records:
        grouped[tuple(record[field] for field in CONFUSION_KEY_FIELDS)].append(record)

    transitions: list[dict[str, Any]] = []
    for key, subset in grouped.items():
        subset = sorted(subset, key=lambda record: str(record[scale_field]))
        by_scale = {normalize_scalar(record[scale_field]): record for record in subset}
        scales = sorted(by_scale.keys(), key=lambda value: float(value) if isinstance(value, (int, float, bool)) else str(value))

        for idx_a in range(len(scales)):
            for idx_b in range(idx_a + 1, len(scales)):
                scale_a = scales[idx_a]
                scale_b = scales[idx_b]
                record_a = by_scale[scale_a]
                record_b = by_scale[scale_b]
                if int(record_a["count"]) < min_count_per_scale or int(record_b["count"]) < min_count_per_scale:
                    continue

                delta_src = subtract_or_none(record_b.get("mean_src_rival_sim"), record_a.get("mean_src_rival_sim"))
                delta_trg = subtract_or_none(record_b.get("mean_trg_rival_sim"), record_a.get("mean_trg_rival_sim"))
                delta_trg_minus_src = subtract_or_none(
                    record_b.get("mean_trg_minus_src_rival_sim"),
                    record_a.get("mean_trg_minus_src_rival_sim"),
                )
                delta_target_more = subtract_or_none(
                    record_b.get("target_more_collapsed_rate"),
                    record_a.get("target_more_collapsed_rate"),
                )
                delta_cross_gt = subtract_or_none(record_b.get("mean_cross_gt_score"), record_a.get("mean_cross_gt_score"))
                delta_cross_rival = subtract_or_none(record_b.get("mean_cross_rival_score"), record_a.get("mean_cross_rival_score"))
                delta_cross_margin = subtract_or_none(record_b.get("mean_cross_margin"), record_a.get("mean_cross_margin"))
                delta_failure_rate = subtract_or_none(record_b.get("failure_rate"), record_a.get("failure_rate"))
                delta_center_margin = subtract_or_none(record_b.get("mean_center_margin"), record_a.get("mean_center_margin"))

                margin_a = None if record_a.get("mean_cross_margin") is None else float(record_a["mean_cross_margin"])
                margin_regime = assign_margin_regime(margin_a, boundary_margin)
                erosion_case = None if delta_cross_margin is None else int(float(delta_cross_margin) < erosion_margin_threshold)

                src_side_rise = None if delta_src is None else int(float(delta_src) > side_rise_threshold)
                trg_side_rise = None if delta_trg is None else int(float(delta_trg) > side_rise_threshold)

                target_dominant = (
                    None
                    if delta_src is None or delta_trg is None or erosion_case is None or int(erosion_case) == 0
                    else int((float(delta_trg) - float(delta_src)) >= dominance_gap and float(delta_trg) > side_rise_threshold)
                )
                source_dominant = (
                    None
                    if delta_src is None or delta_trg is None or erosion_case is None or int(erosion_case) == 0
                    else int((float(delta_src) - float(delta_trg)) >= dominance_gap and float(delta_src) > side_rise_threshold)
                )
                bilateral = (
                    None
                    if src_side_rise is None or trg_side_rise is None or erosion_case is None or int(erosion_case) == 0
                    else int(int(src_side_rise) == 1 and int(trg_side_rise) == 1 and int(target_dominant or 0) == 0 and int(source_dominant or 0) == 0)
                )
                cross_view_only = (
                    None
                    if src_side_rise is None or trg_side_rise is None or erosion_case is None or int(erosion_case) == 0
                    else int(int(src_side_rise) == 0 and int(trg_side_rise) == 0)
                )
                mixed_residual = (
                    None
                    if erosion_case is None or int(erosion_case) == 0
                    else int(
                        int(target_dominant or 0) == 0
                        and int(source_dominant or 0) == 0
                        and int(bilateral or 0) == 0
                        and int(cross_view_only or 0) == 0
                    )
                )

                out = {
                    "category": key[0],
                    "kp_idx": int(key[1]),
                    "best_other_idx_center": int(key[2]),
                    "scale_a": scale_a,
                    "scale_b": scale_b,
                    "count_a": int(record_a["count"]),
                    "count_b": int(record_b["count"]),
                    "failure_count_a": int(record_a["failure_count"]),
                    "failure_count_b": int(record_b["failure_count"]),
                    "failure_rate_a": record_a.get("failure_rate"),
                    "failure_rate_b": record_b.get("failure_rate"),
                    "mean_cross_margin_a": record_a.get("mean_cross_margin"),
                    "mean_cross_margin_b": record_b.get("mean_cross_margin"),
                    "margin_regime": margin_regime,
                    "delta_src_rival_sim": delta_src,
                    "delta_trg_rival_sim": delta_trg,
                    "delta_trg_minus_src_rival_sim": delta_trg_minus_src,
                    "delta_target_more_collapsed_rate": delta_target_more,
                    "delta_cross_gt_score": delta_cross_gt,
                    "delta_cross_rival_score": delta_cross_rival,
                    "delta_cross_margin": delta_cross_margin,
                    "delta_failure_rate": delta_failure_rate,
                    "delta_center_margin": delta_center_margin,
                    "erosion_case": erosion_case,
                    "src_side_rise": src_side_rise,
                    "trg_side_rise": trg_side_rise,
                    "target_dominant_erosion": target_dominant,
                    "source_dominant_erosion": source_dominant,
                    "bilateral_erosion": bilateral,
                    "cross_view_only_erosion": cross_view_only,
                    "mixed_residual_erosion": mixed_residual,
                    "gt_drop_more_than_rival": (
                        None
                        if delta_cross_gt is None or delta_cross_rival is None
                        else int(float(delta_cross_gt) < float(delta_cross_rival))
                    ),
                }
                transitions.append(out)

    return transitions


def summarize_subset(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"count": 0}

    erosion_records = [record for record in records if record.get("erosion_case") == 1]
    return {
        "count": len(records),
        "weighted_support_sum": int(sum(support_weight(record) for record in records)),
        "erosion_count": len(erosion_records),
        "erosion_rate": safe_rate([int(record["erosion_case"]) for record in records if record.get("erosion_case") is not None]),
        "mean_margin_a": safe_mean([float(record["mean_cross_margin_a"]) for record in records if record.get("mean_cross_margin_a") is not None]),
        "weighted_mean_delta_src_rival_sim": weighted_mean(records, "delta_src_rival_sim"),
        "weighted_mean_delta_trg_rival_sim": weighted_mean(records, "delta_trg_rival_sim"),
        "weighted_mean_delta_trg_minus_src_rival_sim": weighted_mean(records, "delta_trg_minus_src_rival_sim"),
        "weighted_mean_delta_cross_gt_score": weighted_mean(records, "delta_cross_gt_score"),
        "weighted_mean_delta_cross_rival_score": weighted_mean(records, "delta_cross_rival_score"),
        "weighted_mean_delta_cross_margin": weighted_mean(records, "delta_cross_margin"),
        "weighted_mean_delta_failure_rate": weighted_mean(records, "delta_failure_rate"),
        "weighted_mean_delta_target_more_collapsed_rate": weighted_mean(records, "delta_target_more_collapsed_rate"),
        "target_dominant_erosion_rate": safe_rate(
            [int(record["target_dominant_erosion"]) for record in erosion_records if record.get("target_dominant_erosion") is not None]
        ),
        "source_dominant_erosion_rate": safe_rate(
            [int(record["source_dominant_erosion"]) for record in erosion_records if record.get("source_dominant_erosion") is not None]
        ),
        "bilateral_erosion_rate": safe_rate(
            [int(record["bilateral_erosion"]) for record in erosion_records if record.get("bilateral_erosion") is not None]
        ),
        "cross_view_only_erosion_rate": safe_rate(
            [int(record["cross_view_only_erosion"]) for record in erosion_records if record.get("cross_view_only_erosion") is not None]
        ),
        "mixed_residual_erosion_rate": safe_rate(
            [int(record["mixed_residual_erosion"]) for record in erosion_records if record.get("mixed_residual_erosion") is not None]
        ),
        "gt_drop_more_than_rival_rate": safe_rate(
            [int(record["gt_drop_more_than_rival"]) for record in erosion_records if record.get("gt_drop_more_than_rival") is not None]
        ),
    }


def mechanism_breakdown(records: list[dict[str, Any]]) -> dict[str, Any]:
    erosion_records = [record for record in records if record.get("erosion_case") == 1]
    if not erosion_records:
        return {"count": 0}

    def weighted_rate(key: str) -> float | None:
        numerator = 0.0
        denominator = 0.0
        for record in erosion_records:
            value = record.get(key)
            if value is None:
                continue
            weight = support_weight(record)
            numerator += float(value) * float(weight)
            denominator += float(weight)
        if denominator <= 0.0:
            return None
        return numerator / denominator

    return {
        "count": len(erosion_records),
        "weighted_support_sum": int(sum(support_weight(record) for record in erosion_records)),
        "weighted_target_dominant_rate": weighted_rate("target_dominant_erosion"),
        "weighted_source_dominant_rate": weighted_rate("source_dominant_erosion"),
        "weighted_bilateral_rate": weighted_rate("bilateral_erosion"),
        "weighted_cross_view_only_rate": weighted_rate("cross_view_only_erosion"),
        "weighted_mixed_residual_rate": weighted_rate("mixed_residual_erosion"),
    }


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    src_key = f"src_rival_sim_r{args.context_radius}"
    trg_key = f"trg_rival_sim_r{args.context_radius}"
    cross_gt_key = f"cross_gt_score_r{args.context_radius}"
    cross_rival_key = f"cross_rival_score_r{args.context_radius}"
    cross_margin_key = f"cross_margin_r{args.context_radius}"
    target_more_key = f"target_more_collapsed_r{args.context_radius}"

    records = load_csv(args.records_csv)
    require_columns(
        records,
        [
            "category",
            "kp_idx",
            "best_other_idx_center",
            args.scale_field,
            "correct",
            "center_margin",
            src_key,
            trg_key,
            cross_gt_key,
            cross_rival_key,
            cross_margin_key,
            target_more_key,
        ],
    )

    scale_values = sort_scale_values({record[args.scale_field] for record in records if record.get(args.scale_field) is not None})
    if len(scale_values) < 2:
        raise RuntimeError(f"Need at least two values for scale_field={args.scale_field}, got {scale_values}")

    pair_scale_records = aggregate_pair_scale_stats(records, args.scale_field, args.context_radius)
    transitions = build_transition_records(
        pair_scale_records,
        args.scale_field,
        args.min_count_per_scale,
        args.boundary_margin,
        args.dominance_gap,
        args.erosion_margin_threshold,
        args.side_rise_threshold,
    )
    if not transitions:
        raise RuntimeError("No valid transitions found. Lower --min_count_per_scale or inspect records_csv.")

    overall = summarize_subset(transitions)
    anchored = [record for record in transitions if str(record.get("margin_regime")) == "anchored"]
    near_boundary = [record for record in transitions if str(record.get("margin_regime")) == "near_boundary"]

    by_transition: dict[str, Any] = {}
    transition_keys = sorted({f"{record['scale_a']}->{record['scale_b']}" for record in transitions})
    for transition_key in transition_keys:
        subset = [record for record in transitions if f"{record['scale_a']}->{record['scale_b']}" == transition_key]
        anchored_subset = [record for record in subset if str(record.get("margin_regime")) == "anchored"]
        near_subset = [record for record in subset if str(record.get("margin_regime")) == "near_boundary"]

        by_transition[transition_key] = {
            "overall": summarize_subset(subset),
            "overall_mechanism_breakdown": mechanism_breakdown(subset),
            "anchored_only": summarize_subset(anchored_subset),
            "anchored_mechanism_breakdown": mechanism_breakdown(anchored_subset),
            "near_boundary_only": summarize_subset(near_subset),
            "near_boundary_mechanism_breakdown": mechanism_breakdown(near_subset),
            "top_margin_erosion_cases": sort_records(
                [record for record in subset if record.get("delta_cross_margin") is not None],
                "delta_cross_margin",
                reverse=False,
            )[: args.topk],
        }

    summary = {
        "num_records": len(records),
        "num_pair_scale_records": len(pair_scale_records),
        "num_transitions": len(transitions),
        "scale_field": args.scale_field,
        "context_radius": args.context_radius,
        "scale_values": scale_values,
        "min_count_per_scale": args.min_count_per_scale,
        "boundary_margin": args.boundary_margin,
        "dominance_gap": args.dominance_gap,
        "erosion_margin_threshold": args.erosion_margin_threshold,
        "side_rise_threshold": args.side_rise_threshold,
        "overall": overall,
        "overall_mechanism_breakdown": mechanism_breakdown(transitions),
        "anchored_only": summarize_subset(anchored),
        "anchored_mechanism_breakdown": mechanism_breakdown(anchored),
        "near_boundary_only": summarize_subset(near_boundary),
        "near_boundary_mechanism_breakdown": mechanism_breakdown(near_boundary),
        "by_transition": by_transition,
        "notes": {
            "interpretation": (
                "If target_dominant_erosion dominates, scale mainly drives target-side local collapse. "
                "If source_dominant_erosion dominates, the source template itself becomes less identity-discriminative. "
                "If bilateral_erosion dominates, both sides lose part identity together. "
                "If cross_view_only_erosion stays large, then scale is hurting cross-view anchoring even when within-image "
                "source/target rival structure does not get visibly worse."
            )
        },
    }

    pair_scale_csv = os.path.join(args.output_dir, "anchor_erosion_side_pair_scale_stats.csv")
    transition_csv = os.path.join(args.output_dir, "anchor_erosion_side_transitions.csv")
    summary_json = os.path.join(args.output_dir, "anchor_erosion_side_summary.json")

    write_records_csv(pair_scale_records, pair_scale_csv)
    write_records_csv(transitions, transition_csv)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved pair-scale stats to: {pair_scale_csv}")
    print(f"Saved transition records to: {transition_csv}")
    print(f"Saved summary to: {summary_json}")
    print("Overall mechanism breakdown:", summary["overall_mechanism_breakdown"])


if __name__ == "__main__":
    main()

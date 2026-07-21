"""Test whether GT->rival confusion pairs repeat across independent pair splits.

This is a diagnosis-only analysis. It does not use the output to alter matching
predictions. Splits are made at image-pair level to avoid leaking keypoints from
the same SPair image pair between discovery and held-out subsets.
"""

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from typing import Any

import numpy as np


DEFAULT_GROUP_FIELDS = ["category", "pair_name", "src_imname", "trg_imname"]
REQUIRED_FIELDS = [
    "category",
    "kp_idx",
    "best_other_idx_center",
    "correct",
    "cross_margin_r0",
]
DEFAULT_CONTEXT_FIELDS = ["scale_variation", "viewpoint_variation", "occlusion", "truncation"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Measure repeatability of SPair GT->rival confusion pairs across "
            "independent image-pair splits. This is CPU-only diagnosis."
        )
    )
    parser.add_argument("--records_csv", required=True, help="Existing identity/competition records CSV.")
    parser.add_argument("--output_dir", required=True, help="Directory for repeatability outputs.")
    parser.add_argument(
        "--group_fields",
        nargs="+",
        default=DEFAULT_GROUP_FIELDS,
        help="Fields defining an independent image-pair group.",
    )
    parser.add_argument(
        "--context_fields",
        nargs="*",
        default=DEFAULT_CONTEXT_FIELDS,
        help="Optional annotation fields for condition-specific held-out coverage.",
    )
    parser.add_argument("--repeats", type=int, default=100, help="Number of independent stratified splits.")
    parser.add_argument("--holdout_fraction", type=float, default=0.5, help="Fraction of groups assigned to held-out data.")
    parser.add_argument("--min_pair_count", type=int, default=10, help="Minimum discovery support for rate-ranked pairs.")
    parser.add_argument("--topk", nargs="+", type=int, default=[10, 50, 100], help="Top-K values to evaluate.")
    parser.add_argument("--min_context_failures", type=int, default=20, help="Minimum held-out failures for a context value to be reported.")
    parser.add_argument("--seed", type=int, default=20260721, help="Random seed for reproducible splits.")
    return parser.parse_args()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def parse_scalar(value: str) -> Any:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except ValueError:
        return value
    if not math.isfinite(number):
        return None
    if abs(number - round(number)) < 1e-12:
        return int(round(number))
    return number


def load_csv(path: str) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        for row in reader:
            records.append({key: parse_scalar(value) for key, value in row.items()})
    return records, fields


def write_csv(records: list[dict[str, Any]], path: str):
    if not records:
        return
    fields = sorted({key for record in records for key in record})
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def require_fields(records: list[dict[str, Any]], fields: list[str]):
    if not records:
        raise RuntimeError("records_csv is empty.")
    missing = [field for field in fields if field not in records[0]]
    if missing:
        raise RuntimeError(f"records_csv is missing required fields: {missing}")


def as_pair_key(record: dict[str, Any]) -> tuple[str, ...]:
    return (
        str(record["category"]),
        str(record["kp_idx"]),
        str(record["best_other_idx_center"]),
    )


def as_group_key(record: dict[str, Any], group_fields: list[str]) -> tuple[str, ...]:
    return tuple(str(record[field]) for field in group_fields)


def failure(record: dict[str, Any]) -> int:
    return 1 - int(record["correct"])


def safe_mean(values: list[float]) -> float | None:
    return None if not values else float(np.mean(np.asarray(values, dtype=np.float64)))


def safe_rate(values: list[int]) -> float | None:
    return None if not values else float(np.mean(np.asarray(values, dtype=np.float64)))


def safe_ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator <= 0 else float(numerator / denominator)


def pair_stats(records: list[dict[str, Any]]) -> dict[tuple[str, ...], dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        rival = record.get("best_other_idx_center")
        if rival is None or int(rival) == int(record["kp_idx"]):
            continue
        grouped[as_pair_key(record)].append(record)

    output: dict[tuple[str, ...], dict[str, Any]] = {}
    for key, subset in grouped.items():
        failures = sum(failure(record) for record in subset)
        margins = [float(record["cross_margin_r0"]) for record in subset if record.get("cross_margin_r0") is not None]
        output[key] = {
            "category": key[0],
            "kp_idx": int(key[1]),
            "rival_idx": int(key[2]),
            "count": len(subset),
            "failure_count": failures,
            "failure_rate": failures / max(len(subset), 1),
            "mean_cross_margin": safe_mean(margins),
        }
    return output


def total_failures(records: list[dict[str, Any]]) -> int:
    return sum(failure(record) for record in records)


def rank_pairs(stats: dict[tuple[str, ...], dict[str, Any]], mode: str, min_pair_count: int) -> list[tuple[str, ...]]:
    candidates = []
    for key, item in stats.items():
        if mode == "rate" and int(item["count"]) < min_pair_count:
            continue
        candidates.append((key, item))
    if mode == "count":
        candidates.sort(key=lambda pair: (int(pair[1]["failure_count"]), int(pair[1]["count"])), reverse=True)
    elif mode == "rate":
        candidates.sort(key=lambda pair: (float(pair[1]["failure_rate"]), int(pair[1]["failure_count"])), reverse=True)
    else:
        raise ValueError(f"Unknown ranking mode: {mode}")
    return [key for key, _ in candidates]


def selected_metrics(
    records: list[dict[str, Any]],
    stats: dict[tuple[str, ...], dict[str, Any]],
    selected: set[tuple[str, ...]],
    context_field: str | None = None,
    context_value: Any = None,
) -> dict[str, Any]:
    if context_field is None:
        subset = records
    else:
        subset = [record for record in records if str(record.get(context_field)) == str(context_value)]
    selected_rows = [record for record in subset if as_pair_key(record) in selected]
    failures = total_failures(subset)
    selected_failures = sum(failure(record) for record in selected_rows)
    selected_count = len(selected_rows)
    selected_failure_rate = safe_rate([failure(record) for record in selected_rows])
    global_failure_rate = safe_rate([failure(record) for record in subset])
    return {
        "records": len(subset),
        "failures": failures,
        "selected_records": selected_count,
        "selected_failures": selected_failures,
        "failure_coverage": safe_ratio(selected_failures, failures),
        "selected_failure_rate": selected_failure_rate,
        "global_failure_rate": global_failure_rate,
        "failure_rate_gap": (
            None
            if selected_failure_rate is None or global_failure_rate is None
            else float(selected_failure_rate - global_failure_rate)
        ),
        "failure_enrichment": (
            None
            if selected_failure_rate is None or global_failure_rate is None
            else safe_ratio(selected_failure_rate, global_failure_rate)
        ),
        "selected_pair_count": len({as_pair_key(record) for record in selected_rows}),
    }


def rank_dict(stats: dict[tuple[str, ...], dict[str, Any]], keys: list[tuple[str, ...]]) -> dict[tuple[str, ...], int]:
    return {key: index for index, key in enumerate(keys)}


def rank_correlation(
    left: dict[tuple[str, ...], float], right: dict[tuple[str, ...], float]
) -> float | None:
    common = [key for key in left if key in right]
    if len(common) < 3:
        return None
    left_values = np.asarray([left[key] for key in common], dtype=np.float64)
    right_values = np.asarray([right[key] for key in common], dtype=np.float64)
    left_ranks = rankdata(left_values)
    right_ranks = rankdata(right_values)
    left_centered = left_ranks - left_ranks.mean()
    right_centered = right_ranks - right_ranks.mean()
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    return None if denominator <= 0 else float(np.dot(left_centered, right_centered) / denominator)


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    cursor = 0
    while cursor < len(values):
        end = cursor + 1
        while end < len(values) and values[order[end]] == values[order[cursor]]:
            end += 1
        average_rank = 0.5 * (cursor + 1 + end)
        ranks[order[cursor:end]] = average_rank
        cursor = end
    return ranks


def jaccard(left: set[tuple[str, ...]], right: set[tuple[str, ...]]) -> float | None:
    union = left | right
    return None if not union else float(len(left & right) / len(union))


def sign(value: float | None) -> int | None:
    if value is None or abs(value) < 1e-12:
        return 0 if value is not None else None
    return 1 if value > 0 else -1


def mean_or_none(values: list[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    return safe_mean(valid)


def split_groups(
    group_keys: list[tuple[str, ...]],
    records_by_group: dict[tuple[str, ...], list[dict[str, Any]]],
    holdout_fraction: float,
    rng: np.random.Generator,
) -> tuple[set[tuple[str, ...]], set[tuple[str, ...]], dict[str, Any]]:
    by_category: dict[str, list[tuple[str, ...]]] = defaultdict(list)
    for key in group_keys:
        category = str(records_by_group[key][0]["category"])
        by_category[category].append(key)

    discovery: set[tuple[str, ...]] = set()
    heldout: set[tuple[str, ...]] = set()
    excluded_categories: list[str] = []
    for category, category_groups in sorted(by_category.items()):
        shuffled = list(category_groups)
        rng.shuffle(shuffled)
        if len(shuffled) < 2:
            excluded_categories.append(category)
            continue
        holdout_count = int(round(len(shuffled) * holdout_fraction))
        holdout_count = min(max(1, holdout_count), len(shuffled) - 1)
        heldout.update(shuffled[:holdout_count])
        discovery.update(shuffled[holdout_count:])
    return discovery, heldout, {
        "num_categories": len(by_category),
        "excluded_categories": excluded_categories,
        "discovery_groups": len(discovery),
        "heldout_groups": len(heldout),
    }


def flatten_groups(
    records_by_group: dict[tuple[str, ...], list[dict[str, Any]]], groups: set[tuple[str, ...]]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for group in groups:
        records.extend(records_by_group[group])
    return records


def top_set(keys: list[tuple[str, ...]], topk: int) -> set[tuple[str, ...]]:
    return set(keys[:topk])


def build_pair_rows(
    full_stats: dict[tuple[str, ...], dict[str, Any]],
    discovery_counts: Counter,
    heldout_counts: Counter,
    discovery_topk_hits: dict[tuple[str, int], Counter],
    heldout_topk_hits: dict[tuple[str, int], Counter],
    topks: list[int],
    repeats: int,
) -> list[dict[str, Any]]:
    keys = set(full_stats) | set(discovery_counts) | set(heldout_counts)
    rows = []
    for key in keys:
        full = full_stats.get(key, {})
        rows.append(
            {
                "category": key[0],
                "kp_idx": int(key[1]),
                "rival_idx": int(key[2]),
                "full_count": full.get("count", 0),
                "full_failure_count": full.get("failure_count", 0),
                "full_failure_rate": full.get("failure_rate"),
                "full_mean_cross_margin": full.get("mean_cross_margin"),
                "discovery_presence_rate": discovery_counts[key] / max(repeats, 1),
                "heldout_presence_rate": heldout_counts[key] / max(repeats, 1),
            }
        )
        for mode in ("count", "rate"):
            for topk in topks:
                discovery_hits = discovery_topk_hits[(mode, topk)][key]
                heldout_hits = heldout_topk_hits[(mode, topk)][key]
                rows[-1][f"{mode}_top{topk}_discovery_hits"] = discovery_hits
                rows[-1][f"{mode}_top{topk}_heldout_hits"] = heldout_hits
                rows[-1][f"{mode}_top{topk}_discovery_hit_rate"] = discovery_hits / max(repeats, 1)
                rows[-1][f"{mode}_top{topk}_heldout_hit_rate"] = heldout_hits / max(repeats, 1)
    largest_topk = max(topks)
    return sorted(
        rows,
        key=lambda row: (
            int(row.get(f"count_top{largest_topk}_heldout_hits", 0)),
            int(row["full_failure_count"]),
        ),
        reverse=True,
    )


def summarize_numeric(values: list[float | None]) -> dict[str, Any]:
    valid = np.asarray([float(value) for value in values if value is not None], dtype=np.float64)
    if len(valid) == 0:
        return {"count": 0, "mean": None, "std": None, "q05": None, "q50": None, "q95": None}
    return {
        "count": int(len(valid)),
        "mean": float(valid.mean()),
        "std": float(valid.std(ddof=1)) if len(valid) > 1 else 0.0,
        "q05": float(np.quantile(valid, 0.05)),
        "q50": float(np.quantile(valid, 0.50)),
        "q95": float(np.quantile(valid, 0.95)),
    }


def main():
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be positive.")
    if not 0.0 < args.holdout_fraction < 1.0:
        raise ValueError("--holdout_fraction must be in (0, 1).")
    topks = sorted({int(topk) for topk in args.topk if int(topk) > 0})
    if not topks:
        raise ValueError("--topk must contain a positive value.")
    ensure_dir(args.output_dir)

    records, fields = load_csv(args.records_csv)
    require_fields(records, REQUIRED_FIELDS + args.group_fields)
    available_context = [field for field in args.context_fields if field in fields]
    missing_context = [field for field in args.context_fields if field not in fields]

    records_by_group: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        records_by_group[as_group_key(record, args.group_fields)].append(record)
    group_keys = sorted(records_by_group)
    if len(group_keys) < 4:
        raise RuntimeError(f"Need at least four independent groups; found {len(group_keys)}.")

    full_stats = pair_stats(records)
    rng = np.random.default_rng(args.seed)
    repeat_rows: list[dict[str, Any]] = []
    discovery_presence = Counter()
    heldout_presence = Counter()
    discovery_topk_hits = {(mode, topk): Counter() for mode in ("count", "rate") for topk in topks}
    heldout_topk_hits = {(mode, topk): Counter() for mode in ("count", "rate") for topk in topks}
    context_accumulator: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)

    for repeat_index in range(args.repeats):
        discovery_groups, heldout_groups, split_meta = split_groups(
            group_keys, records_by_group, args.holdout_fraction, rng
        )
        discovery_records = flatten_groups(records_by_group, discovery_groups)
        heldout_records = flatten_groups(records_by_group, heldout_groups)
        discovery_stats = pair_stats(discovery_records)
        heldout_stats = pair_stats(heldout_records)
        discovery_presence.update(discovery_stats.keys())
        heldout_presence.update(heldout_stats.keys())

        row: dict[str, Any] = {
            "repeat": repeat_index,
            "discovery_groups": len(discovery_groups),
            "heldout_groups": len(heldout_groups),
            "discovery_records": len(discovery_records),
            "heldout_records": len(heldout_records),
            "discovery_failures": total_failures(discovery_records),
            "heldout_failures": total_failures(heldout_records),
            "excluded_categories": "|".join(split_meta["excluded_categories"]),
        }

        for mode in ("count", "rate"):
            discovery_order = rank_pairs(discovery_stats, mode, args.min_pair_count)
            heldout_order = rank_pairs(heldout_stats, mode, args.min_pair_count)
            discovery_values = {
                key: float(discovery_stats[key]["failure_rate"] if mode == "rate" else discovery_stats[key]["failure_count"])
                for key in discovery_stats
            }
            heldout_values = {
                key: float(heldout_stats[key]["failure_rate"] if mode == "rate" else heldout_stats[key]["failure_count"])
                for key in heldout_stats
            }
            row[f"{mode}_rank_correlation"] = rank_correlation(discovery_values, heldout_values)
            for topk in topks:
                discovery_selected = top_set(discovery_order, topk)
                heldout_selected = top_set(heldout_order, topk)
                discovery_topk_hits[(mode, topk)].update(discovery_selected)
                heldout_topk_hits[(mode, topk)].update(heldout_selected)
                heldout_metric = selected_metrics(heldout_records, heldout_stats, discovery_selected)
                discovery_metric = selected_metrics(discovery_records, discovery_stats, discovery_selected)
                row[f"{mode}_top{topk}_jaccard"] = jaccard(discovery_selected, heldout_selected)
                row[f"{mode}_top{topk}_discovery_coverage"] = discovery_metric["failure_coverage"]
                row[f"{mode}_top{topk}_heldout_coverage"] = heldout_metric["failure_coverage"]
                row[f"{mode}_top{topk}_heldout_enrichment"] = heldout_metric["failure_enrichment"]
                row[f"{mode}_top{topk}_heldout_margin_sign_agreement"] = margin_sign_agreement(
                    discovery_stats, heldout_stats, discovery_selected
                )

                for field in available_context:
                    values = sorted({str(record.get(field)) for record in heldout_records if record.get(field) is not None})
                    for value in values:
                        context_rows = [record for record in heldout_records if str(record.get(field)) == value]
                        if total_failures(context_rows) < args.min_context_failures:
                            continue
                        metric = selected_metrics(
                            context_rows,
                            heldout_stats,
                            discovery_selected,
                            context_field=None,
                        )
                        context_accumulator[(field, value, topk, mode)].append(
                            {
                                "repeat": repeat_index,
                                **metric,
                            }
                        )
        repeat_rows.append(row)

    pair_rows = build_pair_rows(
        full_stats,
        discovery_presence,
        heldout_presence,
        discovery_topk_hits,
        heldout_topk_hits,
        topks,
        args.repeats,
    )
    context_rows = []
    for (field, value, topk, mode), metrics in sorted(context_accumulator.items()):
        context_rows.append(
            {
                "context_field": field,
                "context_value": value,
                "ranking_mode": mode,
                "topk": topk,
                "num_repeats": len(metrics),
                "mean_heldout_failure_coverage": mean_or_none([item["failure_coverage"] for item in metrics]),
                "mean_heldout_failure_enrichment": mean_or_none([item["failure_enrichment"] for item in metrics]),
                "mean_heldout_failure_rate_gap": mean_or_none([item["failure_rate_gap"] for item in metrics]),
                "mean_selected_failures": mean_or_none([float(item["selected_failures"]) for item in metrics]),
                "mean_context_failures": mean_or_none([float(item["failures"]) for item in metrics]),
            }
        )

    aggregate: dict[str, Any] = {}
    for mode in ("count", "rate"):
        aggregate[mode] = {}
        for topk in topks:
            aggregate[mode][str(topk)] = {
                "jaccard": summarize_numeric([row.get(f"{mode}_top{topk}_jaccard") for row in repeat_rows]),
                "discovery_coverage": summarize_numeric([row.get(f"{mode}_top{topk}_discovery_coverage") for row in repeat_rows]),
                "heldout_coverage": summarize_numeric([row.get(f"{mode}_top{topk}_heldout_coverage") for row in repeat_rows]),
                "heldout_enrichment": summarize_numeric([row.get(f"{mode}_top{topk}_heldout_enrichment") for row in repeat_rows]),
                "heldout_margin_sign_agreement": summarize_numeric(
                    [row.get(f"{mode}_top{topk}_heldout_margin_sign_agreement") for row in repeat_rows]
                ),
            }
        aggregate[mode]["rank_correlation"] = summarize_numeric(
            [row.get(f"{mode}_rank_correlation") for row in repeat_rows]
        )

    verdict = make_verdict(aggregate, topks)
    summary = {
        "num_records": len(records),
        "num_groups": len(group_keys),
        "num_full_confusion_pairs": len(full_stats),
        "global_failure_rate": safe_rate([failure(record) for record in records]),
        "group_fields": args.group_fields,
        "available_context_fields": available_context,
        "missing_context_fields": missing_context,
        "repeats": args.repeats,
        "holdout_fraction": args.holdout_fraction,
        "min_pair_count": args.min_pair_count,
        "topk": topks,
        "ranking_metrics": aggregate,
        "context_metrics": context_rows,
        "verdict": verdict,
        "interpretation": {
            "count_ranking": "Ranks pairs by failure count and estimates how much failure mass a repeated high-support set can cover.",
            "rate_ranking": "Ranks pairs by failure rate after minimum discovery support and tests whether high-risk pairs repeat beyond support alone.",
            "heldout_rule": "Only pairs selected from discovery groups are evaluated on held-out groups.",
            "leakage_control": "All keypoints from one image-pair group stay in one split.",
            "method_constraint": "These post-hoc pair statistics are diagnostic only; a future train-free method must infer correction from features without GT pair labels.",
        },
    }

    write_csv(repeat_rows, os.path.join(args.output_dir, "confusion_pair_repeatability_repeats.csv"))
    write_csv(pair_rows, os.path.join(args.output_dir, "confusion_pair_repeatability_pairs.csv"))
    write_csv(context_rows, os.path.join(args.output_dir, "confusion_pair_repeatability_context.csv"))
    summary_path = os.path.join(args.output_dir, "confusion_pair_repeatability_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"Saved repeat records to: {os.path.join(args.output_dir, 'confusion_pair_repeatability_repeats.csv')}")
    print(f"Saved pair stability records to: {os.path.join(args.output_dir, 'confusion_pair_repeatability_pairs.csv')}")
    print(f"Saved context records to: {os.path.join(args.output_dir, 'confusion_pair_repeatability_context.csv')}")
    print(f"Saved summary to: {summary_path}")
    print("Verdict:", verdict["label"])
    print("Reason:", verdict["reason"])
    for mode in ("count", "rate"):
        print(f"{mode} rank correlation:", aggregate[mode]["rank_correlation"])
        for topk in topks:
            print(
                f"{mode} top{topk}:",
                {
                    "heldout_coverage": aggregate[mode][str(topk)]["heldout_coverage"],
                    "jaccard": aggregate[mode][str(topk)]["jaccard"],
                    "enrichment": aggregate[mode][str(topk)]["heldout_enrichment"],
                },
            )


def margin_sign_agreement(
    discovery_stats: dict[tuple[str, ...], dict[str, Any]],
    heldout_stats: dict[tuple[str, ...], dict[str, Any]],
    selected: set[tuple[str, ...]],
) -> float | None:
    agreements = []
    for key in selected:
        left = discovery_stats.get(key, {}).get("mean_cross_margin")
        right = heldout_stats.get(key, {}).get("mean_cross_margin")
        if left is None or right is None:
            continue
        left_sign = sign(float(left))
        right_sign = sign(float(right))
        agreements.append(int(left_sign == right_sign))
    return safe_rate(agreements)


def make_verdict(aggregate: dict[str, Any], topks: list[int]) -> dict[str, Any]:
    key = str(max(topks))
    count_metrics = aggregate["count"][key]
    rate_metrics = aggregate["rate"][key]
    count_cov = count_metrics["heldout_coverage"]["mean"]
    rate_cov = rate_metrics["heldout_coverage"]["mean"]
    count_jaccard = count_metrics["jaccard"]["mean"]
    rate_jaccard = rate_metrics["jaccard"]["mean"]
    count_enrichment = count_metrics["heldout_enrichment"]["mean"]
    rate_enrichment = rate_metrics["heldout_enrichment"]["mean"]
    count_rank_corr = aggregate["count"]["rank_correlation"]["mean"]
    rate_rank_corr = aggregate["rate"]["rank_correlation"]["mean"]

    stable_count = (
        count_cov is not None
        and count_jaccard is not None
        and count_enrichment is not None
        and count_rank_corr is not None
        and count_cov >= 0.5 * max(count_metrics["discovery_coverage"]["mean"] or 0.0, 1e-12)
        and count_enrichment >= 1.10
        and count_jaccard >= 0.10
        and count_rank_corr >= 0.20
    )
    stable_rate = (
        rate_cov is not None
        and rate_jaccard is not None
        and rate_enrichment is not None
        and rate_rank_corr is not None
        and rate_cov >= 0.5 * max(rate_metrics["discovery_coverage"]["mean"] or 0.0, 1e-12)
        and rate_enrichment >= 1.10
        and rate_jaccard >= 0.10
        and rate_rank_corr >= 0.20
    )
    if stable_count and stable_rate:
        label = "stable_pair_structure"
        reason = "Both support-ranked and rate-ranked pair sets retain held-out coverage and enrichment."
    elif stable_count or stable_rate:
        label = "partial_pair_structure"
        reason = "Only one ranking view is reproducible; pair-aware correction needs a conservative design."
    else:
        label = "unstable_pair_structure"
        reason = "Top pairs do not retain sufficient held-out coverage/enrichment under image-pair splits."
    return {
        "label": label,
        "reason": reason,
        "largest_topk": max(topks),
        "count_rank_correlation_mean": count_rank_corr,
        "rate_rank_correlation_mean": rate_rank_corr,
        "count_topk_heldout_coverage_mean": count_cov,
        "rate_topk_heldout_coverage_mean": rate_cov,
        "count_topk_jaccard_mean": count_jaccard,
        "rate_topk_jaccard_mean": rate_jaccard,
        "count_topk_enrichment_mean": count_enrichment,
        "rate_topk_enrichment_mean": rate_enrichment,
        "thresholds_are_screening_heuristics": True,
        "screening_thresholds": {
            "heldout_coverage_vs_discovery": 0.50,
            "heldout_enrichment": 1.10,
            "topk_jaccard": 0.10,
            "rank_correlation": 0.20,
        },
        "decision": {
            "stable_pair_structure": "Proceed to an unlabeled feature-driven structured competition method, not a test-set pair lookup.",
            "partial_pair_structure": "Use pair statistics only to define a conservative condition and validate a small method prototype.",
            "unstable_pair_structure": "Do not build a fixed pair-aware method; redirect to condition-aware or lower-level feature analysis.",
        }[label],
    }


if __name__ == "__main__":
    main()

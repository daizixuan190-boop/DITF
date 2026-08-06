"""Offline pair-level global transport consistency audit for FJSAR dumps.

The script reads an existing candidate dump JSON only.  It does not run a
model, does not use labels for scoring, and uses PCK labels only for reporting
candidate rank/hit statistics.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any


SCORE_NAMES = (
    "attention_rank",
    "candidate_graph_distance",
    "candidate_graph_vector",
    "candidate_graph_vector_hflip",
    "attention_weighted_graph_1step",
    "attention_weighted_graph_2step",
    "unlabeled_native_anchor_vector",
    "native_reciprocal_anchor_vector",
    "oracle_native_anchor_vector",
    "hybrid_graph_anchor_rank",
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _point(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, list) or len(value) < 2:
        return None
    return (_safe_float(value[0]), _safe_float(value[1]))


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    return math.sqrt(dx * dx + dy * dy)


def _sub(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return (float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _bbox_diag(points: list[tuple[float, float]]) -> float:
    if not points:
        return 1.0
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    diag = math.sqrt((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2)
    if diag > 1e-6:
        return diag
    pair_dists = [
        _dist(points[i], points[j])
        for i in range(len(points))
        for j in range(i + 1, len(points))
    ]
    return median(pair_dists) if pair_dists else 1.0


def _vector_score(
    src_delta: tuple[float, float],
    trg_delta: tuple[float, float],
    source_scale: float,
    target_scale: float,
    *,
    hflip: bool = False,
) -> float:
    sx = src_delta[0] / max(source_scale, 1e-6)
    sy = src_delta[1] / max(source_scale, 1e-6)
    if hflip:
        sx = -sx
    tx = trg_delta[0] / max(target_scale, 1e-6)
    ty = trg_delta[1] / max(target_scale, 1e-6)
    err = math.sqrt((tx - sx) ** 2 + (ty - sy) ** 2)
    return 1.0 / (1.0 + err)


def _distance_score(
    src_delta: tuple[float, float],
    trg_delta: tuple[float, float],
    source_scale: float,
    target_scale: float,
) -> float:
    ds = math.sqrt(src_delta[0] ** 2 + src_delta[1] ** 2) / max(source_scale, 1e-6)
    dt = math.sqrt(trg_delta[0] ** 2 + trg_delta[1] ** 2) / max(target_scale, 1e-6)
    return 1.0 / (1.0 + abs(math.log((dt + 1e-6) / (ds + 1e-6))))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _median(values: list[float]) -> float | None:
    return float(median(values)) if values else None


def _rank_first_hit(scores: list[float], hits: list[bool]) -> int | None:
    order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    for rank, index in enumerate(order, start=1):
        if hits[index]:
            return int(rank)
    return None


def _rank_positions(values: list[float]) -> dict[int, int]:
    order = sorted(range(len(values)), key=lambda index: values[index], reverse=True)
    return {index: rank for rank, index in enumerate(order, start=1)}


def _normalize_weights(values: list[float]) -> list[float]:
    values = [max(0.0, _safe_float(value)) for value in values]
    total = sum(values)
    if total <= 1e-12:
        return [1.0 / len(values) for _ in values] if values else []
    return [value / total for value in values]


def _candidate_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    proposal_by_pixel: dict[int, dict[str, Any]] = {}
    for proposal in record.get("proposals", []) or []:
        pixel_index = proposal.get("pixel_index")
        if pixel_index is not None:
            proposal_by_pixel[int(pixel_index)] = proposal

    local_candidates = (
        (record.get("local_relational_identity_audit") or {}).get("candidates")
        if isinstance(record.get("local_relational_identity_audit"), dict)
        else None
    )
    source = local_candidates if local_candidates else (record.get("proposals", []) or [])
    candidates: list[dict[str, Any]] = []
    seen: set[int] = set()
    for item in source:
        pixel = _point(item.get("pixel"))
        pixel_index = item.get("pixel_index")
        if pixel is None or pixel_index is None:
            continue
        pixel_index = int(pixel_index)
        if pixel_index in seen:
            continue
        seen.add(pixel_index)
        proposal = proposal_by_pixel.get(pixel_index, {})
        rank_attention = item.get("rank_attention", proposal.get("rank_attention"))
        if rank_attention is None:
            rank_attention = len(candidates) + 1
        attention_score = proposal.get("attention_score")
        if attention_score is None:
            attention_score = 1.0 / max(1.0, float(rank_attention))
        candidates.append({
            "pixel": pixel,
            "pixel_index": pixel_index,
            "rank_attention": int(rank_attention),
            "attention_score": _safe_float(attention_score),
            "pck_hit": bool(item.get("pck_hit", proposal.get("pck_hit", False))),
        })
    candidates.sort(key=lambda row: row["rank_attention"])
    return candidates


def _prepare_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for index, record in enumerate(records):
        source = _point(record.get("source_point"))
        target = _point(record.get("target_point"))
        native = _point((record.get("native") or {}).get("pixel"))
        candidates = _candidate_rows(record)
        if source is None or not candidates:
            continue
        attention_weights = _normalize_weights([candidate["attention_score"] for candidate in candidates])
        for candidate, weight in zip(candidates, attention_weights):
            candidate["attention_weight"] = weight
        out.append({
            "index": index,
            "category": record.get("category"),
            "pair_json": record.get("pair_json"),
            "source": source,
            "target": target,
            "native": native,
            "native_reciprocal": _safe_float((record.get("native") or {}).get("reciprocal_attention")),
            "baseline_pck_hit": bool(record.get("baseline_pck_hit")),
            "method_pck_hit": bool(record.get("method_pck_hit")),
            "oracle_gap_case": bool(record.get("oracle_gap_case")),
            "attention_harms_native_case": bool(record.get("attention_harms_native_case")),
            "attention_rescues_native_case": bool(record.get("attention_rescues_native_case")),
            "attention_topk_pck_hit": bool(record.get("attention_topk_pck_hit")),
            "candidates": candidates,
        })
    return out


def _pair_context(rows: list[dict[str, Any]]) -> dict[str, Any]:
    source_points = [row["source"] for row in rows]
    target_points = []
    for row in rows:
        target_points.extend([candidate["pixel"] for candidate in row["candidates"]])
        if row["native"] is not None:
            target_points.append(row["native"])
    source_scale = _bbox_diag(source_points)
    target_scale = _bbox_diag(target_points)
    native_recips = [row["native_reciprocal"] for row in rows]
    max_native_recip = max(native_recips) if native_recips else 0.0
    return {
        "source_scale": source_scale,
        "target_scale": target_scale,
        "max_native_reciprocal": max_native_recip,
    }


def _candidate_graph_scores(
    row_index: int,
    cand_index: int,
    rows: list[dict[str, Any]],
    context: dict[str, Any],
    weights_by_row: list[list[float]] | None = None,
) -> dict[str, float]:
    row = rows[row_index]
    candidate = row["candidates"][cand_index]
    source_scale = context["source_scale"]
    target_scale = context["target_scale"]
    distance_terms = []
    vector_terms = []
    vector_hflip_terms = []
    weighted_terms = []
    for other_index, other in enumerate(rows):
        if other_index == row_index or not other["candidates"]:
            continue
        src_delta = _sub(other["source"], row["source"])
        best_distance = 0.0
        best_vector = 0.0
        best_vector_hflip = 0.0
        best_weighted = 0.0
        for other_cand_index, other_candidate in enumerate(other["candidates"]):
            trg_delta = _sub(other_candidate["pixel"], candidate["pixel"])
            dist_score = _distance_score(src_delta, trg_delta, source_scale, target_scale)
            vec_score = _vector_score(src_delta, trg_delta, source_scale, target_scale, hflip=False)
            hflip_score = max(
                vec_score,
                _vector_score(src_delta, trg_delta, source_scale, target_scale, hflip=True),
            )
            if weights_by_row is None:
                weight = other_candidate.get("attention_weight", 0.0)
            else:
                weight = weights_by_row[other_index][other_cand_index]
            best_distance = max(best_distance, dist_score)
            best_vector = max(best_vector, vec_score)
            best_vector_hflip = max(best_vector_hflip, hflip_score)
            best_weighted = max(best_weighted, hflip_score * weight)
        distance_terms.append(best_distance)
        vector_terms.append(best_vector)
        vector_hflip_terms.append(best_vector_hflip)
        weighted_terms.append(best_weighted)
    return {
        "candidate_graph_distance": _mean(distance_terms),
        "candidate_graph_vector": _mean(vector_terms),
        "candidate_graph_vector_hflip": _mean(vector_hflip_terms),
        "attention_weighted_graph": _mean(weighted_terms),
    }


def _native_anchor_score(
    row_index: int,
    cand_index: int,
    rows: list[dict[str, Any]],
    context: dict[str, Any],
    *,
    reciprocal_weighted: bool,
    oracle_native_only: bool,
) -> float:
    row = rows[row_index]
    candidate = row["candidates"][cand_index]
    source_scale = context["source_scale"]
    target_scale = context["target_scale"]
    max_recip = max(context["max_native_reciprocal"], 1e-12)
    terms = []
    weights = []
    for other_index, other in enumerate(rows):
        if other_index == row_index or other["native"] is None:
            continue
        if oracle_native_only and not other["baseline_pck_hit"]:
            continue
        src_delta = _sub(other["source"], row["source"])
        trg_delta = _sub(other["native"], candidate["pixel"])
        score = max(
            _vector_score(src_delta, trg_delta, source_scale, target_scale, hflip=False),
            _vector_score(src_delta, trg_delta, source_scale, target_scale, hflip=True),
        )
        weight = 1.0
        if reciprocal_weighted:
            weight = max(0.0, other["native_reciprocal"]) / max_recip
        terms.append(score * weight)
        weights.append(weight)
    if not terms:
        return 0.0
    denom = sum(weights) if reciprocal_weighted else len(terms)
    return sum(terms) / max(denom, 1e-12)


def _audit_pair(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    context = _pair_context(rows)
    base_scores: list[list[dict[str, float]]] = []
    for row_index, row in enumerate(rows):
        row_scores = []
        for cand_index, candidate in enumerate(row["candidates"]):
            graph = _candidate_graph_scores(row_index, cand_index, rows, context)
            row_scores.append({
                "attention_rank": -float(candidate["rank_attention"]),
                "candidate_graph_distance": graph["candidate_graph_distance"],
                "candidate_graph_vector": graph["candidate_graph_vector"],
                "candidate_graph_vector_hflip": graph["candidate_graph_vector_hflip"],
                "attention_weighted_graph_1step": graph["attention_weighted_graph"],
                "unlabeled_native_anchor_vector": _native_anchor_score(
                    row_index,
                    cand_index,
                    rows,
                    context,
                    reciprocal_weighted=False,
                    oracle_native_only=False,
                ),
                "native_reciprocal_anchor_vector": _native_anchor_score(
                    row_index,
                    cand_index,
                    rows,
                    context,
                    reciprocal_weighted=True,
                    oracle_native_only=False,
                ),
                "oracle_native_anchor_vector": _native_anchor_score(
                    row_index,
                    cand_index,
                    rows,
                    context,
                    reciprocal_weighted=False,
                    oracle_native_only=True,
                ),
            })
        base_scores.append(row_scores)

    weights = [
        _normalize_weights([scores["attention_weighted_graph_1step"] for scores in row_scores])
        for row_scores in base_scores
    ]
    for row_index, row in enumerate(rows):
        for cand_index, scores in enumerate(base_scores[row_index]):
            graph = _candidate_graph_scores(row_index, cand_index, rows, context, weights_by_row=weights)
            scores["attention_weighted_graph_2step"] = graph["attention_weighted_graph"]

    record_audits = []
    for row_index, row in enumerate(rows):
        row_scores = base_scores[row_index]
        rank_maps = {
            name: _rank_positions([scores[name] for scores in row_scores])
            for name in (
                "candidate_graph_vector_hflip",
                "attention_weighted_graph_2step",
                "native_reciprocal_anchor_vector",
            )
        }
        for cand_index, scores in enumerate(row_scores):
            scores["hybrid_graph_anchor_rank"] = -(
                rank_maps["candidate_graph_vector_hflip"][cand_index]
                + rank_maps["attention_weighted_graph_2step"][cand_index]
                + rank_maps["native_reciprocal_anchor_vector"][cand_index]
            ) / 3.0

        hits = [bool(candidate["pck_hit"]) for candidate in row["candidates"]]
        ranks = {
            name: _rank_first_hit([scores[name] for scores in row_scores], hits)
            for name in SCORE_NAMES
        }
        score_gaps = {}
        for name in SCORE_NAMES:
            values = [scores[name] for scores in row_scores]
            hit_values = [value for value, hit in zip(values, hits) if hit]
            best_hit = max(hit_values) if hit_values else None
            top1 = values[0] if values else None
            score_gaps[f"{name}_attention_top1_minus_best_pck_hit_proposal"] = (
                float(top1 - best_hit) if top1 is not None and best_hit is not None else None
            )
        top1 = {}
        for name in SCORE_NAMES:
            values = [scores[name] for scores in row_scores]
            if not values:
                continue
            best_index = max(range(len(values)), key=lambda index: values[index])
            top1[name] = {
                "rank_attention": int(row["candidates"][best_index]["rank_attention"]),
                "pixel": [int(row["candidates"][best_index]["pixel"][0]), int(row["candidates"][best_index]["pixel"][1])],
                "pck_hit": bool(row["candidates"][best_index]["pck_hit"]),
                "score": float(values[best_index]),
            }
        record_audits.append({
            "category": row["category"],
            "pair_json": row["pair_json"],
            "source_point": [float(row["source"][0]), float(row["source"][1])],
            "target_point": (
                [float(row["target"][0]), float(row["target"][1])]
                if row["target"] is not None
                else None
            ),
            "baseline_pck_hit": bool(row["baseline_pck_hit"]),
            "method_pck_hit": bool(row["method_pck_hit"]),
            "oracle_gap_case": bool(row["oracle_gap_case"]),
            "attention_harms_native_case": bool(row["attention_harms_native_case"]),
            "attention_topk_pck_hit": bool(row["attention_topk_pck_hit"]),
            "candidate_count": len(row["candidates"]),
            "candidate_pck_hit_count": sum(1 for hit in hits if hit),
            "score_names": list(SCORE_NAMES),
            "ranks": ranks,
            "score_gaps": score_gaps,
            "top1": top1,
        })

    pair_summary = {
        "pair_json": rows[0]["pair_json"],
        "category": rows[0]["category"],
        "record_count": len(rows),
        "source_scale": float(context["source_scale"]),
        "target_candidate_scale": float(context["target_scale"]),
        "baseline_correct_records": sum(1 for row in rows if row["baseline_pck_hit"]),
        "oracle_gap_records": sum(1 for row in rows if row["oracle_gap_case"]),
        "attention_harms_native_records": sum(1 for row in rows if row["attention_harms_native_case"]),
        "attention_topk_hit_records": sum(1 for row in rows if row["attention_topk_pck_hit"]),
    }
    for name in SCORE_NAMES:
        ranks = [audit["ranks"][name] for audit in record_audits if audit["ranks"][name] is not None]
        pair_summary[f"{name}_hit_at_1"] = sum(1 for rank in ranks if rank <= 1)
        pair_summary[f"{name}_ranked"] = len(ranks)
    return record_audits, pair_summary


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups = {
        "all": lambda row: True,
        "oracle_gap": lambda row: bool(row.get("oracle_gap_case")),
        "attention_harms_native": lambda row: bool(row.get("attention_harms_native_case")),
        "attention_topk_hit": lambda row: bool(row.get("attention_topk_pck_hit")),
    }
    summary: dict[str, Any] = {}
    for group_name, predicate in groups.items():
        rows = [row for row in records if predicate(row)]
        group: dict[str, Any] = {
            "points": len(rows),
            "candidate_pck_hit_rows": sum(1 for row in rows if row.get("candidate_pck_hit_count", 0) > 0),
            "mean_candidate_pck_hit_count": _mean([float(row.get("candidate_pck_hit_count", 0)) for row in rows]),
            "signals": {},
        }
        random_rates = [
            float(row.get("candidate_pck_hit_count", 0)) / float(max(1, row.get("candidate_count", 1)))
            for row in rows
            if row.get("candidate_pck_hit_count", 0) > 0
        ]
        group["random_candidate_top1_rate"] = _mean(random_rates)
        for name in SCORE_NAMES:
            ranks = []
            gaps = []
            top_scores_correct = []
            top_scores_wrong = []
            top_attention_ranks_correct = []
            top_attention_ranks_wrong = []
            for row in rows:
                rank = (row.get("ranks") or {}).get(name)
                if rank is not None:
                    ranks.append(int(rank))
                gap = (row.get("score_gaps") or {}).get(
                    f"{name}_attention_top1_minus_best_pck_hit_proposal"
                )
                if gap is not None:
                    gaps.append(float(gap))
                top = (row.get("top1") or {}).get(name)
                if isinstance(top, dict):
                    if top.get("pck_hit"):
                        top_scores_correct.append(_safe_float(top.get("score")))
                        top_attention_ranks_correct.append(int(top.get("rank_attention") or 0))
                    else:
                        top_scores_wrong.append(_safe_float(top.get("score")))
                        top_attention_ranks_wrong.append(int(top.get("rank_attention") or 0))
            group["signals"][name] = {
                "ranked_points": len(ranks),
                "proposal_pck_hit_rank_median": _median([float(rank) for rank in ranks]),
                "proposal_pck_hit_at_1": sum(1 for rank in ranks if rank <= 1),
                "proposal_pck_hit_at_3": sum(1 for rank in ranks if rank <= 3),
                "proposal_pck_hit_at_5": sum(1 for rank in ranks if rank <= 5),
                "proposal_pck_hit_at_10": sum(1 for rank in ranks if rank <= 10),
                "attention_top1_minus_best_pck_hit_proposal_gap_median": _median(gaps),
                "best_pck_hit_proposal_beats_attention_top1": sum(1 for gap in gaps if gap < 0.0),
                "attention_top1_beats_best_pck_hit_proposal": sum(1 for gap in gaps if gap > 0.0),
                "top1_correct_score_median": _median(top_scores_correct),
                "top1_wrong_score_median": _median(top_scores_wrong),
                "top1_correct_attention_rank_median": _median([float(v) for v in top_attention_ranks_correct]),
                "top1_wrong_attention_rank_median": _median([float(v) for v in top_attention_ranks_wrong]),
            }
        summary[group_name] = group
    return summary


def _category_summary(records: list[dict[str, Any]], signal_name: str) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        buckets[str(row.get("category"))].append(row)
    out = []
    for category, rows in sorted(buckets.items()):
        ranks = [
            int((row.get("ranks") or {}).get(signal_name))
            for row in rows
            if (row.get("ranks") or {}).get(signal_name) is not None
        ]
        out.append({
            "category": category,
            "points": len(rows),
            "ranked_points": len(ranks),
            "hit_at_1": sum(1 for rank in ranks if rank <= 1),
            "hit_at_3": sum(1 for rank in ranks if rank <= 3),
            "hit_at_5": sum(1 for rank in ranks if rank <= 5),
        })
    return out


def _anchor_stratified_summary(records: list[dict[str, Any]], thresholds: tuple[int, ...] = (0, 1, 2, 3, 5)) -> list[dict[str, Any]]:
    baseline_by_pair: dict[str, int] = defaultdict(int)
    for row in records:
        if row.get("baseline_pck_hit"):
            baseline_by_pair[str(row.get("pair_json"))] += 1
    out = []
    oracle_rows = [row for row in records if row.get("oracle_gap_case")]
    for threshold in thresholds:
        rows = [
            row for row in oracle_rows
            if baseline_by_pair[str(row.get("pair_json"))] >= int(threshold)
        ]
        item: dict[str, Any] = {
            "min_baseline_correct_anchors_in_dumped_pair": int(threshold),
            "oracle_gap_points": len(rows),
            "pair_count": len({str(row.get("pair_json")) for row in rows}),
            "signals": {},
        }
        for name in (
            "candidate_graph_vector",
            "unlabeled_native_anchor_vector",
            "native_reciprocal_anchor_vector",
            "oracle_native_anchor_vector",
            "hybrid_graph_anchor_rank",
        ):
            ranks = [
                int((row.get("ranks") or {}).get(name))
                for row in rows
                if (row.get("ranks") or {}).get(name) is not None
            ]
            item["signals"][name] = {
                "ranked_points": len(ranks),
                "proposal_pck_hit_at_1": sum(1 for rank in ranks if rank <= 1),
                "proposal_pck_hit_at_3": sum(1 for rank in ranks if rank <= 3),
                "proposal_pck_hit_at_5": sum(1 for rank in ranks if rank <= 5),
                "proposal_pck_hit_rank_median": _median([float(rank) for rank in ranks]),
            }
        out.append(item)
    return out


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Global Transport Consistency Audit",
        "",
        f"- input: `{payload['input_path']}`",
        f"- records: {payload['record_count']}",
        f"- pairs: {payload['pair_count']}",
        f"- usable pairs with >=2 dumped keypoints: {payload['usable_pair_count']}",
        "",
        "## Summary",
        "",
    ]
    for group_name in ("oracle_gap", "attention_harms_native", "all"):
        group = payload["summary"].get(group_name, {})
        lines.append(f"### {group_name}")
        lines.append("")
        lines.append(
            f"points={group.get('points', 0)}, "
            f"candidate_hit_rows={group.get('candidate_pck_hit_rows', 0)}, "
            f"random@1={100.0 * group.get('random_candidate_top1_rate', 0.0):.2f}%"
        )
        lines.append("")
        lines.append("| signal | ranked | @1 | @3 | @5 | @10 | median rank |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for name in SCORE_NAMES:
            signal = group.get("signals", {}).get(name, {})
            ranked = int(signal.get("ranked_points", 0) or 0)
            def pct(key: str) -> str:
                count = int(signal.get(key, 0) or 0)
                return f"{count} ({100.0 * count / ranked:.2f}%)" if ranked else "0"
            lines.append(
                f"| {name} | {ranked} | {pct('proposal_pck_hit_at_1')} | "
                f"{pct('proposal_pck_hit_at_3')} | {pct('proposal_pck_hit_at_5')} | "
                f"{pct('proposal_pck_hit_at_10')} | {signal.get('proposal_pck_hit_rank_median')} |"
            )
        lines.append("")
    lines.append("## Anchor Stratification")
    lines.append("")
    lines.append("Oracle-gap rows grouped by how many baseline-correct anchors are available in the dumped records for that pair.")
    lines.append("")
    lines.append("| min anchors | oracle-gap points | signal | @1 | @3 | @5 | median rank |")
    lines.append("|---:|---:|---|---:|---:|---:|---:|")
    for item in payload.get("anchor_stratified_oracle_gap", []):
        points = int(item.get("oracle_gap_points", 0) or 0)
        for name, signal in item.get("signals", {}).items():
            ranked = int(signal.get("ranked_points", 0) or 0)
            def pct(key: str) -> str:
                count = int(signal.get(key, 0) or 0)
                return f"{count} ({100.0 * count / ranked:.2f}%)" if ranked else "0"
            lines.append(
                f"| {item.get('min_baseline_correct_anchors_in_dumped_pair')} | {points} | {name} | "
                f"{pct('proposal_pck_hit_at_1')} | {pct('proposal_pck_hit_at_3')} | "
                f"{pct('proposal_pck_hit_at_5')} | {signal.get('proposal_pck_hit_rank_median')} |"
            )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze(input_path: Path) -> dict[str, Any]:
    with input_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    records = _prepare_records(data.get("records", []) or [])
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_pair[str(record.get("pair_json"))].append(record)

    all_audits: list[dict[str, Any]] = []
    pair_summaries: list[dict[str, Any]] = []
    skipped_singleton_pairs = 0
    for pair_json, rows in sorted(by_pair.items()):
        if len(rows) < 2:
            skipped_singleton_pairs += 1
            continue
        record_audits, pair_summary = _audit_pair(rows)
        all_audits.extend(record_audits)
        pair_summaries.append(pair_summary)

    summary = _summarize(all_audits)
    category_oracle_gap = _category_summary(
        [row for row in all_audits if row.get("oracle_gap_case")],
        "hybrid_graph_anchor_rank",
    )
    category_attention_harms = _category_summary(
        [row for row in all_audits if row.get("attention_harms_native_case")],
        "hybrid_graph_anchor_rank",
    )
    return {
        "input_path": str(input_path),
        "record_count": len(records),
        "audited_record_count": len(all_audits),
        "pair_count": len(by_pair),
        "usable_pair_count": len(pair_summaries),
        "skipped_singleton_pairs": skipped_singleton_pairs,
        "limitations": [
            "Input dump contains only oracle_gap_or_harm records, so anchors from easy keypoints that were not dumped are unavailable.",
            "Scores do not use GT labels; pck_hit and baseline_pck_hit are used only for reporting, except oracle_native_anchor_vector, which is an explicit label-assisted upper bound.",
            "This is a pair-level diagnostic over candidate top-k, not a matcher implementation.",
        ],
        "score_names": list(SCORE_NAMES),
        "summary": summary,
        "category_oracle_gap_hybrid_graph_anchor_rank": category_oracle_gap,
        "category_attention_harms_native_hybrid_graph_anchor_rank": category_attention_harms,
        "anchor_stratified_oracle_gap": _anchor_stratified_summary(all_audits),
        "pair_summaries": pair_summaries,
        "records": all_audits,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="Existing FJSAR candidate dump JSON.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output JSON path. Defaults to INPUT with _global_transport_audit.json suffix.",
    )
    parser.add_argument(
        "--markdown",
        default="",
        help="Optional markdown summary path. Defaults next to output JSON.",
    )
    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_name(input_path.stem + "_global_transport_audit.json")
    markdown_path = Path(args.markdown) if args.markdown else output_path.with_suffix(".md")

    payload = analyze(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    _write_markdown(payload, markdown_path)
    print(f"wrote {output_path}")
    print(f"wrote {markdown_path}")


if __name__ == "__main__":
    main()

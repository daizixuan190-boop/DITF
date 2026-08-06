"""Baseline-preserving router over cached attention-top20 expert audits.

The router is deliberately conservative:

* native DiTF/cosine prediction is the default;
* external experts are only allowed to rescue when deployable confidence
  signals pass fixed thresholds;
* GT/PCK fields are used only for evaluation summaries, never for routing.

This script is intended as a fast pair20 screening step after expensive
attention/DINO/RoMa audits have already been produced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


METHOD_HYPOTHESIS = {
    "name": "Attention Top20 Baseline-Preserving Expert Rescue Router",
    "mechanism_hypothesis": (
        "Exact FLUX attention supplies high-recall candidate coverage, while "
        "native DiTF remains the safest default descriptor. Frozen external "
        "experts are used only as high-confidence rescue proposals, so the "
        "method tests whether added identity/geometric evidence can improve "
        "PCK without paying the full harm of replacing native NN everywhere."
    ),
    "routing_contract": (
        "No GT, PCK hit flag, GT rank, candidate hit count, or oracle field is "
        "used for routing. These fields are only read after selection for audit."
    ),
}


LOCKED_PROFILES: dict[str, dict[str, Any]] = {
    "discovery20_seed2027_rank_consensus_v1": {
        "tuning_source": "discovery20 split_seed=2027 only",
        "candidate_rule": "equal-rank sum of native DiTF, DINOv2, and RoMa pairwise",
        "parameters": {
            "preferred_expert": "candidate_rank_consensus",
            "roma_bidir_max": -1.0,
            "roma_mutual_min": -1.0,
            "baseline_attention_min_distance": 0.0,
            "dino_roma_agreement_px": -1.0,
            "max_selected_attention_rank": 20,
            "native_top1_cosine_max": 0.8,
            "native_margin_max": -1.0,
            "native_nonlocal_margin_max": -1.0,
            "native_cycle_distance_min": 16.0,
            "require_native_nonreciprocal": False,
        },
        "frozen_validation_result": {
            "pairs": 360,
            "points": 2663,
            "baseline_point": 68.9447990987608,
            "method_point": 70.82238077356365,
        },
    }
}


PointKey = tuple[str, str, int]


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "pair_records" not in payload:
        raise ValueError(f"{path} is not an audit JSON with pair_records")
    return payload


def _pair_id(pair: dict[str, Any]) -> str:
    pair_json = pair.get("pair_json")
    if pair_json:
        return str(pair_json)
    src = pair.get("src_image")
    trg = pair.get("trg_image")
    if src and trg:
        return f"{src}|{trg}"
    raise ValueError("pair record must contain pair_json or src_image/trg_image")


def _index_points(audit: dict[str, Any]) -> tuple[dict[PointKey, dict[str, Any]], list[PointKey]]:
    indexed: dict[PointKey, dict[str, Any]] = {}
    order: list[PointKey] = []
    for pair in audit["pair_records"]:
        category = str(pair.get("category"))
        pair_id = _pair_id(pair)
        for point in pair.get("points", []):
            key = (category, pair_id, int(point["keypoint_index"]))
            if key in indexed:
                raise ValueError(f"duplicate point key {key}")
            indexed[key] = point
            order.append(key)
    return indexed, order


def _validate_alignment(
    reference_order: list[PointKey],
    expert_points: dict[str, dict[PointKey, dict[str, Any]]],
) -> None:
    reference = set(reference_order)
    for name, points in expert_points.items():
        missing = reference.difference(points)
        extra = set(points).difference(reference)
        if missing or extra:
            raise ValueError(
                f"{name} audit does not align: missing={len(missing)} extra={len(extra)}"
            )


def _as_xy(value: Any) -> tuple[float, float] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return float(value[0]), float(value[1])
    return None


def _distance(a: tuple[float, float] | None, b: tuple[float, float] | None) -> float:
    if a is None or b is None:
        return float("inf")
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _top_candidate(point: dict[str, Any]) -> dict[str, Any]:
    candidates = point.get("candidates") or []
    return candidates[0] if candidates else {}


def _candidate_pixel(candidate: dict[str, Any]) -> tuple[int, int]:
    pixel = candidate.get("pixel")
    if not isinstance(pixel, (list, tuple)) or len(pixel) < 2:
        raise ValueError("candidate must contain a two-dimensional pixel")
    return int(pixel[0]), int(pixel[1])


def _candidate_rank_consensus_point(
    dino_point: dict[str, Any],
    roma_point: dict[str, Any],
) -> dict[str, Any]:
    """Build an equal-rank candidate consensus without reading outcome labels."""

    dino_candidates = dino_point.get("candidates") or []
    roma_candidates = roma_point.get("candidates") or []
    if not dino_candidates or not roma_candidates:
        raise ValueError("candidate rank consensus requires DINO and RoMa candidates")
    roma_by_pixel = {_candidate_pixel(candidate): candidate for candidate in roma_candidates}
    if len(roma_by_pixel) != len(roma_candidates):
        raise ValueError("RoMa candidates contain duplicate pixels")

    ranked: list[dict[str, Any]] = []
    for dino_candidate in dino_candidates:
        pixel = _candidate_pixel(dino_candidate)
        roma_candidate = roma_by_pixel.get(pixel)
        if roma_candidate is None:
            raise ValueError(f"DINO/RoMa candidate grids do not align at {pixel}")
        component_ranks = {
            "native": int(dino_candidate["native_candidate_rank"]),
            "dino": int(dino_candidate["dino_rank"]),
            "roma": int(roma_candidate["roma_rank"]),
        }
        candidate = dict(roma_candidate)
        candidate.update(
            {
                "pixel": [pixel[0], pixel[1]],
                "attention_rank": int(dino_candidate["attention_rank"]),
                "pck_hit": bool(dino_candidate.get("pck_hit", False)),
                "rank_consensus_score": int(sum(component_ranks.values())),
                "rank_consensus_components": component_ranks,
            }
        )
        ranked.append(candidate)

    if len(ranked) != len(dino_candidates):
        raise ValueError("DINO/RoMa candidate counts do not align")
    ranked.sort(
        key=lambda candidate: (
            int(candidate["rank_consensus_score"]),
            int(candidate["attention_rank"]),
            tuple(candidate["pixel"]),
        )
    )
    selected = ranked[0]
    return {
        "method_prediction": selected["pixel"],
        "method_pck_hit": bool(selected.get("pck_hit", False)),
        "candidates": ranked,
    }


def _supports_candidate_rank_consensus(point: dict[str, Any]) -> bool:
    candidates = point.get("candidates") or []
    return bool(candidates) and all(
        "native_candidate_rank" in candidate and "dino_rank" in candidate
        for candidate in candidates
    )


def _apply_locked_profile(args: argparse.Namespace) -> dict[str, Any] | None:
    profile_name = str(getattr(args, "locked_profile", "") or "")
    if not profile_name:
        return None
    if profile_name not in LOCKED_PROFILES:
        raise ValueError(f"unknown locked profile {profile_name!r}")
    profile = LOCKED_PROFILES[profile_name]
    for name, value in profile["parameters"].items():
        setattr(args, name, value)
    canonical = json.dumps(profile, sort_keys=True, separators=(",", ":"))
    return {
        "name": profile_name,
        "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        **profile,
    }


def _float_field(row: dict[str, Any], name: str, default: float = float("inf")) -> float:
    value = row.get(name)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _baseline_attention_distance(point: dict[str, Any]) -> float:
    baseline = _as_xy(point.get("baseline_prediction"))
    distances = [
        _distance(baseline, _as_xy(candidate.get("pixel")))
        for candidate in point.get("candidates", [])
    ]
    return min(distances) if distances else float("inf")


def _prediction_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    return _distance(
        _as_xy(left.get("method_prediction")),
        _as_xy(right.get("method_prediction")),
    )


def _proposal_passes(
    *,
    point: dict[str, Any],
    dino_point: dict[str, Any],
    roma_point: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[bool, dict[str, Any]]:
    candidate = _top_candidate(point)
    native = dino_point.get("native_nn_diagnostics") or {}
    signals = {
        "roma_bidirectional_error": _float_field(candidate, "bidirectional_error"),
        "roma_mutual_certainty": _float_field(candidate, "mutual_certainty", 0.0),
        "selected_attention_rank": int(candidate.get("attention_rank") or 999),
        "baseline_attention_distance": _baseline_attention_distance(dino_point),
        "dino_roma_prediction_distance": _prediction_distance(dino_point, roma_point),
        "native_top1_top2_margin": _float_field(
            native, "top1_top2_margin"
        ),
        "native_top1_cosine": _float_field(native, "top1_cosine"),
        "native_top1_nonlocal_margin": _float_field(
            native, "top1_nonlocal_margin"
        ),
        "native_cycle_source_distance": _float_field(
            native, "cycle_source_distance"
        ),
        "native_reciprocal_exact": bool(native.get("reciprocal_exact", False)),
    }
    checks = [
        signals["selected_attention_rank"] <= args.max_selected_attention_rank,
        signals["baseline_attention_distance"] >= args.baseline_attention_min_distance,
    ]
    if args.roma_bidir_max >= 0:
        checks.append(signals["roma_bidirectional_error"] <= args.roma_bidir_max)
    if args.roma_mutual_min >= 0:
        checks.append(signals["roma_mutual_certainty"] >= args.roma_mutual_min)
    if args.dino_roma_agreement_px >= 0:
        checks.append(
            signals["dino_roma_prediction_distance"] <= args.dino_roma_agreement_px
        )
    if args.native_margin_max >= 0:
        checks.append(
            math.isfinite(signals["native_top1_top2_margin"])
            and signals["native_top1_top2_margin"] <= args.native_margin_max
        )
    if args.native_top1_cosine_max >= 0:
        checks.append(
            math.isfinite(signals["native_top1_cosine"])
            and signals["native_top1_cosine"] <= args.native_top1_cosine_max
        )
    if args.native_nonlocal_margin_max >= 0:
        checks.append(
            math.isfinite(signals["native_top1_nonlocal_margin"])
            and signals["native_top1_nonlocal_margin"]
            <= args.native_nonlocal_margin_max
        )
    if args.native_cycle_distance_min >= 0:
        checks.append(
            math.isfinite(signals["native_cycle_source_distance"])
            and signals["native_cycle_source_distance"]
            >= args.native_cycle_distance_min
        )
    if args.require_native_nonreciprocal:
        checks.append(not signals["native_reciprocal_exact"])
    return all(checks), signals


def select_expert(
    dino_point: dict[str, Any],
    roma_point: dict[str, Any],
    consensus_point: dict[str, Any] | None,
    args: argparse.Namespace,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Select one prediction without reading GT/PCK outcome fields."""

    if args.preferred_expert == "candidate_rank_consensus":
        rank_consensus_point = _candidate_rank_consensus_point(dino_point, roma_point)
        passes, signals = _proposal_passes(
            point=rank_consensus_point,
            dino_point=dino_point,
            roma_point=roma_point,
            args=args,
        )
        if passes:
            return "candidate_rank_consensus", rank_consensus_point, signals
        return "baseline", dino_point, signals

    if args.preferred_expert == "roma_consensus" and consensus_point is not None:
        passes, signals = _proposal_passes(
            point=consensus_point,
            dino_point=dino_point,
            roma_point=roma_point,
            args=args,
        )
        if passes:
            return "roma_consensus", consensus_point, signals

    passes, signals = _proposal_passes(
        point=roma_point,
        dino_point=dino_point,
        roma_point=roma_point,
        args=args,
    )
    if passes:
        return "roma_pairwise", roma_point, signals

    return "baseline", dino_point, signals


def _summarize_points(points: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(points)
    baseline_correct = sum(bool(point["baseline_pck_hit"]) for point in points)
    method_correct = sum(bool(point["method_pck_hit"]) for point in points)
    rescue = sum(bool(point["rescued_vs_baseline"]) for point in points)
    harm = sum(bool(point["harmed_vs_baseline"]) for point in points)
    selected_counts = Counter(str(point["selected_expert"]) for point in points)
    oracle_any = sum(bool(point.get("expert_oracle_pck_hit")) for point in points)
    return {
        "points": total,
        "baseline_correct": int(baseline_correct),
        "method_correct": int(method_correct),
        "baseline_point": 100.0 * baseline_correct / max(1, total),
        "method_point": 100.0 * method_correct / max(1, total),
        "point_gain": 100.0 * (method_correct - baseline_correct) / max(1, total),
        "rescued_vs_baseline": int(rescue),
        "harmed_vs_baseline": int(harm),
        "selected_expert_counts": dict(selected_counts),
        "expert_oracle_point": 100.0 * oracle_any / max(1, total),
    }


def _category_results(
    pair_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pair_rows:
        by_category[str(pair["category"])].append(pair)

    categories: dict[str, Any] = {}
    all_points: list[dict[str, Any]] = []
    all_pair_baseline: list[float] = []
    all_pair_method: list[float] = []

    for category, pairs in sorted(by_category.items()):
        points = [point for pair in pairs for point in pair["points"]]
        all_points.extend(points)
        pair_baseline = [
            sum(bool(point["baseline_pck_hit"]) for point in pair["points"])
            / max(1, len(pair["points"]))
            for pair in pairs
        ]
        pair_method = [
            sum(bool(point["method_pck_hit"]) for point in pair["points"])
            / max(1, len(pair["points"]))
            for pair in pairs
        ]
        all_pair_baseline.extend(pair_baseline)
        all_pair_method.extend(pair_method)
        summary = _summarize_points(points)
        summary.update(
            {
                "pairs": len(pairs),
                "baseline_image": 100.0 * sum(pair_baseline) / max(1, len(pair_baseline)),
                "method_image": 100.0 * sum(pair_method) / max(1, len(pair_method)),
            }
        )
        categories[category] = summary

    all_summary = _summarize_points(all_points)
    all_summary.update(
        {
            "pairs": len(pair_rows),
            "baseline_image": 100.0
            * sum(all_pair_baseline)
            / max(1, len(all_pair_baseline)),
            "method_image": 100.0
            * sum(all_pair_method)
            / max(1, len(all_pair_method)),
        }
    )
    return categories, all_summary


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    protocol_lock = _apply_locked_profile(args)
    dino_audit = _read_json(args.dino_audit_json)
    roma_audit = _read_json(args.roma_audit_json)
    dino_points, order = _index_points(dino_audit)
    expert_points = {"roma_pairwise": _index_points(roma_audit)[0]}

    consensus_points: dict[PointKey, dict[str, Any]] | None = None
    if args.roma_consensus_audit_json:
        consensus_audit = _read_json(args.roma_consensus_audit_json)
        consensus_points = _index_points(consensus_audit)[0]
        expert_points["roma_consensus"] = consensus_points

    _validate_alignment(order, expert_points)

    pair_rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for key in order:
        category, pair_id, keypoint_index = key
        dino_point = dino_points[key]
        roma_point = expert_points["roma_pairwise"][key]
        consensus_point = consensus_points[key] if consensus_points is not None else None
        selected_expert, selected_point, signals = select_expert(
            dino_point,
            roma_point,
            consensus_point,
            args,
        )

        baseline_hit = bool(dino_point["baseline_pck_hit"])
        method_hit = (
            baseline_hit
            if selected_expert == "baseline"
            else bool(selected_point["method_pck_hit"])
        )
        prediction = (
            dino_point.get("baseline_prediction")
            if selected_expert == "baseline"
            else selected_point.get("method_prediction")
        )
        expert_hits = {
            "baseline": baseline_hit,
            "dino": bool(dino_point.get("method_pck_hit")),
            "roma_pairwise": bool(roma_point.get("method_pck_hit")),
        }
        if _supports_candidate_rank_consensus(dino_point):
            rank_consensus_point = _candidate_rank_consensus_point(dino_point, roma_point)
            expert_hits["candidate_rank_consensus"] = bool(
                rank_consensus_point.get("method_pck_hit")
            )
        if consensus_point is not None:
            expert_hits["roma_consensus"] = bool(consensus_point.get("method_pck_hit"))

        point_record = {
            "keypoint_index": int(keypoint_index),
            "source_point": dino_point.get("source_point"),
            "target_point": dino_point.get("target_point"),
            "baseline_prediction": dino_point.get("baseline_prediction"),
            "method_prediction": prediction,
            "selected_expert": selected_expert,
            "baseline_pck_hit": baseline_hit,
            "method_pck_hit": bool(method_hit),
            "rescued_vs_baseline": bool(method_hit and not baseline_hit),
            "harmed_vs_baseline": bool(baseline_hit and not method_hit),
            "expert_oracle_pck_hit": bool(any(expert_hits.values())),
            "expert_hits": expert_hits,
            "router_signals": signals,
        }
        pair_key = (category, pair_id)
        if pair_key not in pair_rows_by_key:
            pair_rows_by_key[pair_key] = {
                "category": category,
                "pair_json": pair_id,
                "points": [],
            }
        pair_rows_by_key[pair_key]["points"].append(point_record)

    pair_rows = list(pair_rows_by_key.values())
    categories, all_summary = _category_results(pair_rows)
    result = {
        "matcher": "attention_top20_baseline_preserving_expert_rescue_router",
        "method_hypothesis": METHOD_HYPOTHESIS,
        "protocol": {
            "dino_audit_json": str(args.dino_audit_json),
            "roma_audit_json": str(args.roma_audit_json),
            "roma_consensus_audit_json": str(args.roma_consensus_audit_json or ""),
            "preferred_expert": args.preferred_expert,
            "roma_bidir_max": args.roma_bidir_max,
            "roma_mutual_min": args.roma_mutual_min,
            "baseline_attention_min_distance": args.baseline_attention_min_distance,
            "dino_roma_agreement_px": args.dino_roma_agreement_px,
            "max_selected_attention_rank": args.max_selected_attention_rank,
            "native_margin_max": args.native_margin_max,
            "native_nonlocal_margin_max": args.native_nonlocal_margin_max,
            "native_cycle_distance_min": args.native_cycle_distance_min,
            "require_native_nonreciprocal": args.require_native_nonreciprocal,
            "native_top1_cosine_max": args.native_top1_cosine_max,
            "protocol_lock": protocol_lock,
        },
        "categories": categories,
        "all": all_summary,
    }
    audit = {
        **result,
        "pair_records": pair_rows,
    }
    return {"result": result, "audit": audit}


def write_outputs(payload: dict[str, Any], output_json: str | Path) -> None:
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    root = output_path.with_suffix("")
    summary_path = Path(f"{root}_expert_rescue_router_summary.json")
    audit_path = Path(f"{root}_expert_rescue_router_audit.json")

    result = dict(payload["result"])
    result["summary_path"] = str(summary_path)
    result["audit_path"] = str(audit_path)

    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    summary_path.write_text(
        json.dumps(
            {
                "matcher": result["matcher"],
                "method_hypothesis": result["method_hypothesis"],
                "protocol": result["protocol"],
                "summary": result["all"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    audit = dict(payload["audit"])
    audit["summary_path"] = str(summary_path)
    audit["audit_path"] = str(audit_path)
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dino_audit_json", required=True)
    parser.add_argument("--roma_audit_json", required=True)
    parser.add_argument("--roma_consensus_audit_json", default="")
    parser.add_argument("--output_json", required=True)
    parser.add_argument(
        "--preferred_expert",
        choices=("roma_pairwise", "roma_consensus", "candidate_rank_consensus"),
        default="roma_pairwise",
    )
    parser.add_argument(
        "--locked_profile",
        choices=("", *LOCKED_PROFILES),
        default="",
        help="Override all routing parameters with a pre-registered immutable profile.",
    )
    parser.add_argument(
        "--roma_bidir_max",
        type=float,
        default=0.05,
        help="Allow RoMa rescue only when selected candidate bidirectional warp error is this low.",
    )
    parser.add_argument("--roma_mutual_min", type=float, default=0.0)
    parser.add_argument("--baseline_attention_min_distance", type=float, default=0.0)
    parser.add_argument(
        "--dino_roma_agreement_px",
        type=float,
        default=-1.0,
        help="Require DINO and RoMa predictions to agree within this many pixels; negative disables it.",
    )
    parser.add_argument("--max_selected_attention_rank", type=int, default=20)
    parser.add_argument(
        "--native_top1_cosine_max",
        type=float,
        default=-1.0,
        help="Require native full-NN top1 cosine at or below this value; negative disables.",
    )
    parser.add_argument(
        "--native_margin_max",
        type=float,
        default=-1.0,
        help="Require native top1-top2 margin at or below this value; negative disables.",
    )
    parser.add_argument(
        "--native_nonlocal_margin_max",
        type=float,
        default=-1.0,
        help="Require native top1-nonlocal margin at or below this value; negative disables.",
    )
    parser.add_argument(
        "--native_cycle_distance_min",
        type=float,
        default=-1.0,
        help="Require native cycle-back distance at or above this value; negative disables.",
    )
    parser.add_argument("--require_native_nonreciprocal", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    payload = evaluate(args)
    write_outputs(payload, args.output_json)
    all_result = payload["result"]["all"]
    print("Matcher: attention_top20_baseline_preserving_expert_rescue_router")
    print(
        "Baseline All per image/point: "
        f"{all_result['baseline_image']:.2f} / {all_result['baseline_point']:.2f}"
    )
    print(
        "Method All per image/point: "
        f"{all_result['method_image']:.2f} / {all_result['method_point']:.2f}; "
        f"point gain={all_result['point_gain']:.2f}"
    )
    print(
        "Selected experts: "
        + ", ".join(
            f"{name}={count}"
            for name, count in sorted(all_result["selected_expert_counts"].items())
        )
    )
    print(f"Expert oracle point: {all_result['expert_oracle_point']:.2f}")


if __name__ == "__main__":
    main()

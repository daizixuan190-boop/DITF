"""Decompose what frozen RoMa contributes inside exact-attention candidates.

The analysis is deliberately offline.  It consumes aligned DINO/native and
RoMa audit JSON files and never loads FLUX, DINO, or RoMa.  Candidate choices
are computed only from deployable ranks/errors; GT/PCK fields are read after
selection to measure each evidence source.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from eval_spair_attention_top20_expert_rescue_router import (
    PointKey,
    _index_points,
    _read_json,
    _validate_alignment,
)


TOPKS = (1, 3, 5, 10, 20)


def _pixel(candidate: dict[str, Any]) -> tuple[int, int]:
    value = candidate.get("pixel")
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise ValueError("candidate pixel must contain x and y")
    return int(value[0]), int(value[1])


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"candidate field {name} must be finite")
    return result


def _ranked_candidates(
    dino_point: dict[str, Any],
    roma_point: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    dino_candidates = dino_point.get("candidates") or []
    roma_candidates = roma_point.get("candidates") or []
    if not dino_candidates or not roma_candidates:
        raise ValueError("both audits must contain candidate rows")
    dino_by_pixel = {_pixel(candidate): candidate for candidate in dino_candidates}
    roma_by_pixel = {_pixel(candidate): candidate for candidate in roma_candidates}
    if set(dino_by_pixel) != set(roma_by_pixel):
        raise ValueError("DINO/native and RoMa candidate grids do not align")

    joined = []
    for pixel, dino_candidate in dino_by_pixel.items():
        roma_candidate = roma_by_pixel[pixel]
        joined.append(
            {
                "pixel": pixel,
                "pck_hit": bool(dino_candidate.get("pck_hit", False)),
                "attention_rank": int(dino_candidate["attention_rank"]),
                "native_rank": int(dino_candidate["native_candidate_rank"]),
                "dino_rank": int(dino_candidate["dino_rank"]),
                "roma_rank": int(roma_candidate["roma_rank"]),
                "forward_error": _finite(roma_candidate["forward_error"], "forward_error"),
                "backward_error": _finite(roma_candidate["backward_error"], "backward_error"),
                "bidirectional_error": _finite(
                    roma_candidate["bidirectional_error"], "bidirectional_error"
                ),
                "mutual_certainty": _finite(
                    roma_candidate.get("mutual_certainty", 0.0), "mutual_certainty"
                ),
            }
        )

    specifications = {
        "attention": lambda row: (row["attention_rank"],),
        "native": lambda row: (row["native_rank"], row["attention_rank"]),
        "dino": lambda row: (row["dino_rank"], row["attention_rank"]),
        "flux_rank_sum": lambda row: (
            row["attention_rank"] + row["native_rank"],
            row["attention_rank"],
        ),
        "flux_dino_rank_sum": lambda row: (
            row["attention_rank"] + row["native_rank"] + row["dino_rank"],
            row["attention_rank"],
        ),
        "roma_forward": lambda row: (row["forward_error"], row["attention_rank"]),
        "roma_backward": lambda row: (row["backward_error"], row["attention_rank"]),
        "roma_bidirectional": lambda row: (
            row["bidirectional_error"],
            row["attention_rank"],
        ),
        "roma_certainty": lambda row: (-row["mutual_certainty"], row["attention_rank"]),
        "native_dino_roma_rank_sum": lambda row: (
            row["native_rank"] + row["dino_rank"] + row["roma_rank"],
            row["attention_rank"],
        ),
    }
    return {
        name: sorted(joined, key=lambda row, key=key: (*key(row), row["pixel"]))
        for name, key in specifications.items()
    }


def _spearman(left: Iterable[int], right: Iterable[int]) -> float:
    left_array = np.asarray(list(left), dtype=np.float64)
    right_array = np.asarray(list(right), dtype=np.float64)
    if left_array.size < 2 or left_array.std() == 0.0 or right_array.std() == 0.0:
        return 0.0
    return float(np.corrcoef(left_array, right_array)[0, 1])


def _point_record(
    key: PointKey,
    dino_point: dict[str, Any],
    roma_point: dict[str, Any],
) -> dict[str, Any]:
    rankings = _ranked_candidates(dino_point, roma_point)
    top1 = {name: rows[0] for name, rows in rankings.items()}
    topk = {
        name: {
            str(k): bool(any(row["pck_hit"] for row in rows[: min(k, len(rows))]))
            for k in TOPKS
        }
        for name, rows in rankings.items()
    }

    by_pixel = {row["pixel"]: row for row in rankings["attention"]}
    pixel_order = sorted(by_pixel)
    roma_ranks = [by_pixel[pixel]["roma_rank"] for pixel in pixel_order]
    rank_correlations = {
        field: _spearman(
            roma_ranks,
            [by_pixel[pixel][field] for pixel in pixel_order],
        )
        for field in ("attention_rank", "native_rank", "dino_rank")
    }

    bidirectional = rankings["roma_bidirectional"]
    error_margin = (
        bidirectional[1]["bidirectional_error"] - bidirectional[0]["bidirectional_error"]
        if len(bidirectional) > 1
        else 0.0
    )
    baseline_hit = bool(dino_point.get("baseline_pck_hit"))
    top20_hit = bool(dino_point.get("attention_top20_pck_hit"))
    scorer_hits = {name: bool(row["pck_hit"]) for name, row in top1.items()}
    return {
        "key": key,
        "baseline_hit": baseline_hit,
        "attention_top20_hit": top20_hit,
        "both_wrong_top20_hit": bool(dino_point.get("both_wrong_top20_hit")),
        "uniform_top1_expectation": float(
            dino_point.get("uniform_candidate_hit_probability", 0.0)
        ),
        "scorer_hits": scorer_hits,
        "topk_hits": topk,
        "top1_pixels": {name: list(row["pixel"]) for name, row in top1.items()},
        "rank_correlations": rank_correlations,
        "forward_backward_top1_agree": (
            top1["roma_forward"]["pixel"] == top1["roma_backward"]["pixel"]
        ),
        "roma_selected_bidirectional_error": top1["roma_bidirectional"][
            "bidirectional_error"
        ],
        "roma_error_margin": float(error_margin),
        "roma_selected_mutual_certainty": top1["roma_bidirectional"][
            "mutual_certainty"
        ],
        "roma_unique_beyond_flux": bool(
            scorer_hits["roma_bidirectional"]
            and not baseline_hit
            and not scorer_hits["attention"]
            and not scorer_hits["native"]
            and not scorer_hits["flux_rank_sum"]
        ),
        "roma_unique_beyond_flux_and_dino": bool(
            scorer_hits["roma_bidirectional"]
            and not baseline_hit
            and not scorer_hits["attention"]
            and not scorer_hits["native"]
            and not scorer_hits["dino"]
            and not scorer_hits["flux_dino_rank_sum"]
        ),
    }


def _calibration_bins(
    records: list[dict[str, Any]],
    field: str,
    bin_count: int = 5,
) -> list[dict[str, Any]]:
    if not records:
        return []
    ordered = sorted(records, key=lambda row: float(row[field]))
    bins = []
    for indices in np.array_split(np.arange(len(ordered)), min(bin_count, len(ordered))):
        rows = [ordered[int(index)] for index in indices]
        bins.append(
            {
                "points": len(rows),
                "value_min": float(rows[0][field]),
                "value_max": float(rows[-1][field]),
                "roma_top1_rate": float(
                    np.mean(
                        [row["scorer_hits"]["roma_bidirectional"] for row in rows]
                    )
                ),
            }
        )
    return bins


def _summarize_branch(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"points": 0}
    scorer_names = list(records[0]["scorer_hits"])
    scorer_top1 = {
        name: float(np.mean([row["scorer_hits"][name] for row in records]))
        for name in scorer_names
    }
    scorer_topk = {
        name: {
            str(k): float(np.mean([row["topk_hits"][name][str(k)] for row in records]))
            for k in TOPKS
        }
        for name in scorer_names
    }
    return {
        "points": len(records),
        "baseline_rate": float(np.mean([row["baseline_hit"] for row in records])),
        "uniform_top1_expectation": float(
            np.mean([row["uniform_top1_expectation"] for row in records])
        ),
        "scorer_top1_rates": scorer_top1,
        "scorer_topk_rates": scorer_topk,
        "forward_backward_top1_agreement": float(
            np.mean([row["forward_backward_top1_agree"] for row in records])
        ),
        "roma_unique_beyond_flux": int(
            sum(row["roma_unique_beyond_flux"] for row in records)
        ),
        "roma_unique_beyond_flux_and_dino": int(
            sum(row["roma_unique_beyond_flux_and_dino"] for row in records)
        ),
        "roma_rank_correlation_mean": {
            name: float(np.mean([row["rank_correlations"][name] for row in records]))
            for name in ("attention_rank", "native_rank", "dino_rank")
        },
        "roma_top1_agreement": {
            name: float(
                np.mean(
                    [
                        row["top1_pixels"]["roma_bidirectional"]
                        == row["top1_pixels"][name]
                        for row in records
                    ]
                )
            )
            for name in ("attention", "native", "dino", "flux_rank_sum")
        },
        "roma_error_margin_bins_ascending": _calibration_bins(
            records, "roma_error_margin"
        ),
        "roma_mutual_certainty_bins_ascending": _calibration_bins(
            records, "roma_selected_mutual_certainty"
        ),
    }


def analyze(dino_audit: dict[str, Any], roma_audit: dict[str, Any]) -> dict[str, Any]:
    dino_points, order = _index_points(dino_audit)
    roma_points, _ = _index_points(roma_audit)
    _validate_alignment(order, {"roma_pairwise": roma_points})
    records = [_point_record(key, dino_points[key], roma_points[key]) for key in order]

    branches = {
        "all_points": records,
        "attention_top20_hit": [row for row in records if row["attention_top20_hit"]],
        "baseline_wrong_top20_hit": [
            row
            for row in records
            if not row["baseline_hit"] and row["attention_top20_hit"]
        ],
        "both_wrong_top20_hit": [
            row for row in records if row["both_wrong_top20_hit"]
        ],
    }
    summaries = {name: _summarize_branch(rows) for name, rows in branches.items()}
    hard = summaries["both_wrong_top20_hit"]
    hard_rates = hard.get("scorer_top1_rates", {})
    return {
        "analysis": "attention_top20_roma_evidence_decomposition",
        "protocol": {
            "candidate_set": "exact FLUX attention top20 from the input audits",
            "inference_fields": (
                "attention/native/DINO ranks and RoMa forward/backward warp errors only"
            ),
            "gt_contract": "PCK/GT fields are used only after each scorer selects candidates",
            "train_free": True,
        },
        "summary": summaries,
        "mechanism_checks": {
            "hard_roma_gain_over_forward": float(
                hard_rates.get("roma_bidirectional", 0.0)
                - hard_rates.get("roma_forward", 0.0)
            ),
            "hard_roma_gain_over_backward": float(
                hard_rates.get("roma_bidirectional", 0.0)
                - hard_rates.get("roma_backward", 0.0)
            ),
            "hard_roma_gain_over_flux_rank_sum": float(
                hard_rates.get("roma_bidirectional", 0.0)
                - hard_rates.get("flux_rank_sum", 0.0)
            ),
            "hard_roma_gain_over_random": float(
                hard_rates.get("roma_bidirectional", 0.0)
                - hard.get("uniform_top1_expectation", 0.0)
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dino_audit_json", required=True)
    parser.add_argument("--roma_audit_json", required=True)
    parser.add_argument("--output_json", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = analyze(_read_json(args.dino_audit_json), _read_json(args.roma_audit_json))
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    all_points = payload["summary"]["all_points"]
    hard = payload["summary"]["both_wrong_top20_hit"]
    print(f"Points: {all_points['points']}; hard both-wrong/top20-hit: {hard['points']}")
    print("All top1 rates:", all_points["scorer_top1_rates"])
    print("Hard top1 rates:", hard["scorer_top1_rates"])
    print("Mechanism checks:", payload["mechanism_checks"])


if __name__ == "__main__":
    main()

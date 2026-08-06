"""Analyze lightweight FJSAR candidate dumps.

The dump is produced by eval_spair_matcher_ablation.py with
--fjsar_dump_candidates.  This script is CPU-only and intentionally small: it
summarizes whether cross-attention proposals contain recoverable GT candidates
and whether descriptor / reciprocal / fused scores rank them near the top.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from statistics import quantiles


def _rank_hit(record: dict, key: str, k: int) -> bool:
    rank = record.get("gt_ranks", {}).get(key)
    return isinstance(rank, int) and rank <= k


def _summarize(records: list[dict]) -> dict:
    total = len(records)
    native_correct = sum(1 for row in records if row.get("native", {}).get("pck_hit") is True)
    summary = {
        "total_keypoints": total,
        "native_pck": native_correct / total if total else 0.0,
        "rank_hits": {},
        "native_wrong_rank_hits": {},
        "category": {},
    }
    rank_names = ("attention", "descriptor", "reciprocal", "fused")
    for name in rank_names:
        summary["rank_hits"][name] = {
            f"@{k}": sum(1 for row in records if _rank_hit(row, name, k)) / total if total else 0.0
            for k in (1, 2, 3, 5, 10, 20)
        }
    native_wrong = [row for row in records if row.get("native", {}).get("pck_hit") is False]
    for name in rank_names:
        denom = len(native_wrong)
        summary["native_wrong_rank_hits"][name] = {
            f"@{k}": sum(1 for row in native_wrong if _rank_hit(row, name, k)) / denom if denom else 0.0
            for k in (1, 2, 3, 5, 10, 20)
        }

    by_cat: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        by_cat[str(row.get("category", "unknown"))].append(row)
    for category, rows in sorted(by_cat.items()):
        cat_total = len(rows)
        cat_native = sum(1 for row in rows if row.get("native", {}).get("pck_hit") is True)
        summary["category"][category] = {
            "total": cat_total,
            "native_pck": cat_native / cat_total if cat_total else 0.0,
            "attention@20": sum(1 for row in rows if _rank_hit(row, "attention", 20)) / cat_total if cat_total else 0.0,
            "descriptor@1": sum(1 for row in rows if _rank_hit(row, "descriptor", 1)) / cat_total if cat_total else 0.0,
            "fused@1": sum(1 for row in rows if _rank_hit(row, "fused", 1)) / cat_total if cat_total else 0.0,
        }
    return summary


def _best_proposal(record: dict, score_key: str) -> dict | None:
    proposals = record.get("proposals", [])
    if not proposals:
        return None
    return max(proposals, key=lambda item: float(item.get(score_key, -1e9)))


def _selector_sweep(records: list[dict]) -> list[dict]:
    """Audit simple native-anchored selectors using dumped GT labels.

    This is diagnostic only: it estimates whether the dumped unsupervised
    signals are separable enough to justify implementing a real selector.
    """

    if not records:
        return []
    base_hits = sum(1 for row in records if row.get("native", {}).get("pck_hit") is True)
    margins = [float(row.get("native", {}).get("margin", 0.0)) for row in records]
    margin_grid = {0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05}
    if len(margins) >= 10:
        margin_grid.update(float(value) for value in quantiles(margins, n=10))
    sem_grid = {-0.2, -0.1, -0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05}
    reciprocal_grid = {-0.05, 0.0, 0.0001, 0.001, 0.005, 0.01, 0.02, 0.05}
    results = []
    for score_key in ("descriptor_score", "reciprocal_attention", "fused_score"):
        for margin_max in sorted(margin_grid):
            for semantic_min in sorted(sem_grid):
                for reciprocal_delta in sorted(reciprocal_grid):
                    hits = changed = improved = harmed = 0
                    for row in records:
                        native = row.get("native", {})
                        native_hit = native.get("pck_hit") is True
                        proposal = _best_proposal(row, score_key)
                        choose = False
                        if proposal is not None:
                            choose = (
                                float(native.get("margin", 0.0)) <= margin_max
                                and float(proposal.get("semantic_gap_to_native", -1e9)) >= semantic_min
                                and float(proposal.get("reciprocal_attention", -1e9))
                                >= float(native.get("reciprocal_attention", 0.0)) + reciprocal_delta
                            )
                        if choose:
                            changed += 1
                            hit = proposal.get("pck_hit") is True
                        else:
                            hit = native_hit
                        hits += int(hit)
                        improved += int(hit and not native_hit)
                        harmed += int(native_hit and not hit)
                    results.append({
                        "gain": (hits - base_hits) / len(records),
                        "pck": hits / len(records),
                        "hits": hits,
                        "changed": changed,
                        "improved": improved,
                        "harmed": harmed,
                        "score_key": score_key,
                        "native_margin_max": margin_max,
                        "semantic_gap_min": semantic_min,
                        "reciprocal_delta_min": reciprocal_delta,
                    })
    return sorted(results, key=lambda item: (item["gain"], item["hits"]), reverse=True)[:20]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze FJSAR candidate dump JSON.")
    parser.add_argument("dump_json")
    parser.add_argument("--output_json", default="")
    args = parser.parse_args()
    with open(args.dump_json, encoding="utf-8") as handle:
        payload = json.load(handle)
    records = payload.get("records", [])
    if not isinstance(records, list):
        raise ValueError("candidate dump must contain a records list")
    summary = _summarize(records)
    summary["selector_sweep_top20"] = _selector_sweep(records)
    print(json.dumps(summary, indent=2))
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()

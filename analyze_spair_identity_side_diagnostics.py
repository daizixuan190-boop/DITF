import argparse
import csv
import json
import math
import os
from collections import defaultdict
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from analyze_spair_real_part_competition import (
    batch_patch_descriptors,
    load_csv,
    resolve_pair_json_path,
    safe_mean,
    safe_rate,
    write_records_csv,
)
from analyze_spair_token_residuals import build_post_feature, ensure_dir
from estimate_spair_matching_ceiling import maybe_apply_post_calibration


MERGE_KEYS = [
    "category",
    "pair_name",
    "src_imname",
    "trg_imname",
    "kp_idx",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose whether GT-vs-rival confusion is driven more by source-side identity weakness "
            "or target-side part collapse."
        )
    )
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to SPair-71k root.")
    parser.add_argument("--feature_path", type=str, required=True, help="Path to saved per-category features.")
    parser.add_argument("--competition_csv", type=str, required=True, help="Path to real_part_competition_records.csv.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs.")
    parser.add_argument("--device", type=str, default="cuda", help="Analysis device, e.g. cuda or cpu.")
    parser.add_argument("--feature_dim", type=int, default=3072, help="Feature dimension.")
    parser.add_argument("--discard_channels", nargs="+", type=int, default=[154, 1446], help="Channels to discard for Flux.")
    parser.add_argument("--cd", action="store_true", default=False, help="Apply channel discard as in eval_spair.py.")
    parser.add_argument("--context_radii", nargs="+", type=int, default=[0, 1, 2, 4], help="Radii used to build local descriptors.")
    parser.add_argument("--max_pairs_per_cat", type=int, default=0, help="Optional category-wise cap for faster diagnosis.")
    parser.add_argument("--tag_column", type=str, default="", help="Optional existing column in competition_csv for subgroup summaries.")
    parser.add_argument("--tag_values", nargs="*", default=[], help="Optional tag values to summarize; default uses all observed values.")
    parser.add_argument("--flush_every_pairs", type=int, default=10, help="Write partial outputs every N pairs.")
    parser.add_argument("--shift_calibration", action="store_true", default=False, help="Apply shift-only post-AdaLN calibration before analysis.")
    parser.add_argument("--shift_calibration_quantile", default=0.75, type=float, help="Only suppress tokens above this shift-ratio quantile.")
    parser.add_argument("--shift_calibration_strength", default=0.5, type=float, help="Suppression strength for high-shift tokens.")
    parser.add_argument("--shift_calibration_min_lambda", default=0.2, type=float, help="Minimum retained shift scaling after calibration.")
    parser.add_argument("--joint_calibration", action="store_true", default=False, help="Apply joint high-shift low-content calibration before analysis.")
    parser.add_argument("--joint_shift_quantile", default=0.75, type=float, help="High-shift token threshold for joint calibration.")
    parser.add_argument("--joint_content_quantile", default=0.25, type=float, help="Low-content token threshold for joint calibration.")
    parser.add_argument("--joint_shift_strength", default=0.5, type=float, help="Shift suppression strength under joint calibration.")
    parser.add_argument("--joint_min_shift_lambda", default=0.2, type=float, help="Minimum retained shift scaling under joint calibration.")
    parser.add_argument("--joint_content_strength", default=0.25, type=float, help="Content amplification strength under joint calibration.")
    parser.add_argument("--joint_max_content_gain", default=1.5, type=float, help="Upper bound for content amplification under joint calibration.")
    return parser.parse_args()


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


def filter_competition_records(records: list[dict[str, Any]], max_pairs_per_cat: int) -> list[dict[str, Any]]:
    if max_pairs_per_cat <= 0:
        return records
    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_cat[str(record["category"])].append(record)

    limited: list[dict[str, Any]] = []
    for _, subset in by_cat.items():
        pair_seen = []
        kept_pairs = set()
        for record in subset:
            pair_name = str(record["pair_name"])
            if pair_name not in kept_pairs:
                if len(pair_seen) >= max_pairs_per_cat:
                    continue
                pair_seen.append(pair_name)
                kept_pairs.add(pair_name)
            limited.append(record)
    return limited


def summarize_subset(records: list[dict[str, Any]], radii: list[int]) -> dict[str, Any]:
    if not records:
        return {"count": 0}
    summary = {
        "count": len(records),
        "failure_rate": safe_rate([1 - int(record["correct"]) for record in records]),
        "mean_best_other_trg_norm_dist": safe_mean([float(record["best_other_trg_norm_dist"]) for record in records if record.get("best_other_trg_norm_dist") is not None]),
        "radii": {},
    }
    for radius in radii:
        key = f"r{radius}"
        summary["radii"][key] = {
            "mean_src_rival_sim": safe_mean([float(record[f"src_rival_sim_r{radius}"]) for record in records]),
            "mean_trg_rival_sim": safe_mean([float(record[f"trg_rival_sim_r{radius}"]) for record in records]),
            "mean_trg_minus_src_rival_sim": safe_mean([float(record[f"trg_minus_src_rival_sim_r{radius}"]) for record in records]),
            "target_more_collapsed_rate": safe_rate([int(record[f"target_more_collapsed_r{radius}"]) for record in records]),
            "mean_cross_gt_score": safe_mean([float(record[f"cross_gt_score_r{radius}"]) for record in records]),
            "mean_cross_rival_score": safe_mean([float(record[f"cross_rival_score_r{radius}"]) for record in records]),
            "mean_cross_margin": safe_mean([float(record[f"cross_margin_r{radius}"]) for record in records]),
            "mean_cross_margin_gain_vs_center": safe_mean([float(record[f"cross_margin_gain_r{radius}"]) for record in records]),
        }
    return summary


def diff_summary(a: dict[str, Any], b: dict[str, Any], radii: list[int]) -> dict[str, Any]:
    if a.get("count", 0) == 0 or b.get("count", 0) == 0:
        return {"count_a": a.get("count", 0), "count_b": b.get("count", 0)}
    out = {
        "count_a": a["count"],
        "count_b": b["count"],
        "mean_best_other_trg_norm_dist_gap": (
            a["mean_best_other_trg_norm_dist"] - b["mean_best_other_trg_norm_dist"]
            if a["mean_best_other_trg_norm_dist"] is not None and b["mean_best_other_trg_norm_dist"] is not None
            else None
        ),
        "radii": {},
    }
    for radius in radii:
        key = f"r{radius}"
        a_r = a["radii"].get(key, {})
        b_r = b["radii"].get(key, {})
        out["radii"][key] = {
            "src_rival_sim_gap": (
                a_r.get("mean_src_rival_sim") - b_r.get("mean_src_rival_sim")
                if a_r.get("mean_src_rival_sim") is not None and b_r.get("mean_src_rival_sim") is not None
                else None
            ),
            "trg_rival_sim_gap": (
                a_r.get("mean_trg_rival_sim") - b_r.get("mean_trg_rival_sim")
                if a_r.get("mean_trg_rival_sim") is not None and b_r.get("mean_trg_rival_sim") is not None
                else None
            ),
            "trg_minus_src_gap": (
                a_r.get("mean_trg_minus_src_rival_sim") - b_r.get("mean_trg_minus_src_rival_sim")
                if a_r.get("mean_trg_minus_src_rival_sim") is not None and b_r.get("mean_trg_minus_src_rival_sim") is not None
                else None
            ),
            "target_more_collapsed_rate_gap": (
                a_r.get("target_more_collapsed_rate") - b_r.get("target_more_collapsed_rate")
                if a_r.get("target_more_collapsed_rate") is not None and b_r.get("target_more_collapsed_rate") is not None
                else None
            ),
            "cross_margin_gap": (
                a_r.get("mean_cross_margin") - b_r.get("mean_cross_margin")
                if a_r.get("mean_cross_margin") is not None and b_r.get("mean_cross_margin") is not None
                else None
            ),
            "cross_margin_gain_gap": (
                a_r.get("mean_cross_margin_gain_vs_center") - b_r.get("mean_cross_margin_gain_vs_center")
                if a_r.get("mean_cross_margin_gain_vs_center") is not None and b_r.get("mean_cross_margin_gain_vs_center") is not None
                else None
            ),
        }
    return out


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    print(f"Using analysis device: {device}")

    competition_records = filter_competition_records(load_csv(args.competition_csv), args.max_pairs_per_cat)
    if not competition_records:
        raise RuntimeError("No competition records selected. Check competition_csv and filters.")

    records_by_pair: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in competition_records:
        records_by_pair[(str(record["category"]), str(record["pair_name"]))].append(record)

    pre_norm = nn.LayerNorm(args.feature_dim, elementwise_affine=False, eps=1e-6)
    test_path = os.path.join(args.dataset_path, "PairAnnotation", "test")
    output_records: list[dict[str, Any]] = []

    output_csv = os.path.join(args.output_dir, "identity_side_diagnostics_records.csv")
    output_json = os.path.join(args.output_dir, "identity_side_diagnostics_summary.json")

    processed_pairs = 0
    for (category, pair_name), pair_records in records_by_pair.items():
        feat_file = os.path.join(args.feature_path, f"{category}.pth")
        ada_file = os.path.join(args.feature_path, f"{category}_ada.pth")
        if not os.path.exists(feat_file) or not os.path.exists(ada_file):
            continue

        output_dict = torch.load(feat_file, map_location="cpu", weights_only=True)
        ada_dict = torch.load(ada_file, map_location="cpu", weights_only=True)
        pair_json_path = resolve_pair_json_path(test_path, pair_name)
        with open(pair_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        src_imname = data["src_imname"]
        trg_imname = data["trg_imname"]
        src_img_size = data["src_imsize"][:2][::-1]
        trg_img_size = data["trg_imsize"][:2][::-1]

        _, src_ft_ln, _, src_shift, src_scale = build_post_feature(
            output_dict[src_imname].float(),
            ada_dict[src_imname].float(),
            pre_norm,
            args.discard_channels,
            args.cd,
        )
        _, trg_ft_ln, _, trg_shift, trg_scale = build_post_feature(
            output_dict[trg_imname].float(),
            ada_dict[trg_imname].float(),
            pre_norm,
            args.discard_channels,
            args.cd,
        )
        src_ft = nn.Upsample(size=src_img_size, mode="bilinear")(
            maybe_apply_post_calibration(src_ft_ln, src_shift, src_scale, args).float().to(device)
        )
        trg_ft = nn.Upsample(size=trg_img_size, mode="bilinear")(
            maybe_apply_post_calibration(trg_ft_ln, trg_shift, trg_scale, args).float().to(device)
        )

        src_kps = data["src_kps"]
        trg_kps = data["trg_kps"]
        src_desc_by_radius = {
            radius: batch_patch_descriptors(src_ft, src_kps, radius)
            for radius in args.context_radii
        }
        trg_desc_by_radius = {
            radius: batch_patch_descriptors(trg_ft, trg_kps, radius)
            for radius in args.context_radii
        }

        for record in pair_records:
            kp_idx = int(record["kp_idx"])
            rival_idx = int(record["best_other_idx_center"])
            if rival_idx == kp_idx or rival_idx < 0 or rival_idx >= len(trg_kps):
                continue

            out_record = dict(record)
            for radius in args.context_radii:
                src_desc = src_desc_by_radius[radius]
                trg_desc = trg_desc_by_radius[radius]

                src_gt = src_desc[kp_idx].view(1, -1)
                src_rival = src_desc[rival_idx].view(1, -1)
                trg_gt = trg_desc[kp_idx].view(1, -1)
                trg_rival = trg_desc[rival_idx].view(1, -1)

                src_rival_sim = float(torch.sum(src_gt * src_rival).item())
                trg_rival_sim = float(torch.sum(trg_gt * trg_rival).item())
                cross_gt_score = float(torch.sum(src_gt * trg_gt).item())
                cross_rival_score = float(torch.sum(src_gt * trg_rival).item())
                cross_margin = cross_gt_score - cross_rival_score

                out_record[f"src_rival_sim_r{radius}"] = src_rival_sim
                out_record[f"trg_rival_sim_r{radius}"] = trg_rival_sim
                out_record[f"trg_minus_src_rival_sim_r{radius}"] = trg_rival_sim - src_rival_sim
                out_record[f"target_more_collapsed_r{radius}"] = int(trg_rival_sim > src_rival_sim)
                out_record[f"cross_gt_score_r{radius}"] = cross_gt_score
                out_record[f"cross_rival_score_r{radius}"] = cross_rival_score
                out_record[f"cross_margin_r{radius}"] = cross_margin
                out_record[f"cross_margin_gain_r{radius}"] = cross_margin - float(record[f"margin_r{radius}"])

            output_records.append(out_record)

        processed_pairs += 1
        if processed_pairs % args.flush_every_pairs == 0:
            write_records_csv(output_records, output_csv)
            print(f"[Flush] pairs={processed_pairs} records={len(output_records)}")

    write_records_csv(output_records, output_csv)

    overall = summarize_subset(output_records, args.context_radii)
    success = summarize_subset([record for record in output_records if int(record["correct"]) == 1], args.context_radii)
    failure = summarize_subset([record for record in output_records if int(record["correct"]) == 0], args.context_radii)

    tag_summary = {}
    if args.tag_column:
        observed_tag_values = sorted(
            {
                str(record.get(args.tag_column))
                for record in output_records
                if record.get(args.tag_column) is not None and record.get(args.tag_column) != ""
            }
        )
        tag_values = args.tag_values if args.tag_values else observed_tag_values
        for value in tag_values:
            subset = [record for record in output_records if str(record.get(args.tag_column)) == str(value)]
            tag_summary[str(value)] = {
                "overall": summarize_subset(subset, args.context_radii),
                "success": summarize_subset([record for record in subset if int(record["correct"]) == 1], args.context_radii),
                "failure": summarize_subset([record for record in subset if int(record["correct"]) == 0], args.context_radii),
            }

    summary = {
        "num_records": len(output_records),
        "context_radii": args.context_radii,
        "overall": overall,
        "success": success,
        "failure": failure,
        "failure_minus_success": diff_summary(failure, success, args.context_radii),
        "tag_column": args.tag_column if args.tag_column else None,
        "by_tag": tag_summary,
        "notes": {
            "interpretation": (
                "If failure samples show much larger target-side GT-vs-rival similarity than source-side, "
                "the rival confusion is more target-collapse-like. "
                "If source-side and target-side rival similarity both rise, the issue is more symmetric identity weakness. "
                "If larger local radii improve cross_margin, local context helps identity anchoring; "
                "if they hurt it, context is hurting discrimination."
            )
        },
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved records to: {output_csv}")
    print(f"Saved summary to: {output_json}")
    print(
        "Failure vs Success:",
        {
            "failure_count": failure["count"],
            "success_count": success["count"],
            "center_trg_minus_src_gap_r0": summary["failure_minus_success"]["radii"].get("r0", {}).get("trg_minus_src_gap"),
            "cross_margin_gap_r0": summary["failure_minus_success"]["radii"].get("r0", {}).get("cross_margin_gap"),
        },
    )


if __name__ == "__main__":
    main()

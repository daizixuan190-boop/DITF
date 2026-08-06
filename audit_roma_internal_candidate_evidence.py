"""Audit whether RoMa's pre-warp pair representation resolves FLUX proposals.

This is deliberately an *offline evidence audit*, not a matcher.  It consumes
the frozen, exact FLUX mutual-attention top-k coordinates already recorded by
an attention candidate audit.  For the same source query and the same target
candidate pixels, it records two RoMa-internal scores:

* encoder cosine: a single-image RoMa encoder control;
* GP coordinate agreement: RoMa's learned pair-conditioned Gaussian-process
  posterior before its coordinate decoder and dense warp refiners.

The latter is the untested information source.  It compares the source GP
posterior with the target positional basis at every FLUX proposal and vice
versa.  It is not a warp error, a confidence gate, a native fallback, or a
hand-tuned fusion.  Ground-truth PCK is read only after ranking, exclusively
to evaluate candidate separability.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from eval_spair_attention_top20_roma_identity import (
    TOPKS,
    _build_roma,
    _normalize_points,
    _sample_field,
    _validate_audit,
)


METHOD_HYPOTHESIS = {
    "name": "RoMa Internal Candidate Evidence Audit",
    "question": (
        "Does the frozen RoMa pair-conditioned representation before its warp "
        "decoder contain candidate identity evidence that is absent after "
        "reducing RoMa to dense bidirectional warp error?"
    ),
    "candidate_contract": "existing_exact_block28_mutual_attention_top20_coordinates_only",
    "candidate_pool_modified": False,
    "scores": {
        "roma_encoder_cosine": "single-image encoder control at RoMa scale16",
        "roma_gp_coordinate_agreement": (
            "bidirectional agreement between the learned RoMa GP posterior "
            "and the opposite image's learned Fourier positional basis"
        ),
    },
    "warp_error_used": False,
    "certainty_used": False,
    "attention_used_as_identity_score": False,
    "ditf_descriptor_used": False,
    "dino_descriptor_used": False,
    "native_fallback": False,
    "training": False,
    "gt_used_for_scoring": False,
    "external_pretraining": "RoMa outdoor dense correspondence weights",
}


def _as_feature_map(field: torch.Tensor) -> torch.Tensor:
    if field.ndim != 4 or int(field.shape[0]) != 1:
        raise ValueError("RoMa feature field must be [1,C,H,W]")
    return field[0].permute(1, 2, 0).contiguous()


def _sample_descriptor(
    field: torch.Tensor,
    points: torch.Tensor,
    image_size: Sequence[int],
) -> torch.Tensor:
    if points.ndim < 2 or int(points.shape[-1]) != 2:
        raise ValueError("points must end with xy coordinates")
    normalized = _normalize_points(points, image_size)
    return _sample_field(_as_feature_map(field), normalized)


def rank_roma_internal_candidates(
    source_points: torch.Tensor,
    candidate_points: torch.Tensor,
    source_size: Sequence[int],
    target_size: Sequence[int],
    source_projected: torch.Tensor,
    target_projected: torch.Tensor,
    source_gp: torch.Tensor,
    target_gp: torch.Tensor,
    source_position_basis: torch.Tensor,
    target_position_basis: torch.Tensor,
) -> dict[str, dict[str, torch.Tensor]]:
    """Rank a fixed proposal pool with pre-warp RoMa representations.

    ``source_gp`` predicts the target Fourier position basis conditioned on
    both images; ``target_gp`` symmetrically predicts the source basis.  The
    score is their equal, bidirectional cosine agreement at a candidate pair.
    All tensors are frozen RoMa activations from the scale-16 decoder input.
    """

    if source_points.ndim != 2 or tuple(source_points.shape[-1:]) != (2,):
        raise ValueError("source_points must be [P,2]")
    if candidate_points.ndim != 3 or tuple(candidate_points.shape[-1:]) != (2,):
        raise ValueError("candidate_points must be [P,K,2]")
    if int(candidate_points.shape[0]) != int(source_points.shape[0]):
        raise ValueError("source points and candidate rows must align")
    fields = (
        source_projected,
        target_projected,
        source_gp,
        target_gp,
        source_position_basis,
        target_position_basis,
    )
    if any(field.ndim != 4 or int(field.shape[0]) != 1 for field in fields):
        raise ValueError("all RoMa fields must be [1,C,H,W]")
    if int(source_gp.shape[1]) != int(target_position_basis.shape[1]):
        raise ValueError("source GP and target position basis channels must agree")
    if int(target_gp.shape[1]) != int(source_position_basis.shape[1]):
        raise ValueError("target GP and source position basis channels must agree")

    source_projection = _sample_descriptor(source_projected, source_points, source_size)
    candidate_projection = _sample_descriptor(target_projected, candidate_points, target_size)
    encoder_cosine = F.cosine_similarity(
        source_projection[:, None, :], candidate_projection, dim=-1, eps=1e-12
    )

    source_gp_values = _sample_descriptor(source_gp, source_points, source_size)
    target_position_values = _sample_descriptor(
        target_position_basis, candidate_points, target_size
    )
    target_gp_values = _sample_descriptor(target_gp, candidate_points, target_size)
    source_position_values = _sample_descriptor(
        source_position_basis, source_points, source_size
    )
    forward = F.cosine_similarity(
        source_gp_values[:, None, :], target_position_values, dim=-1, eps=1e-12
    )
    backward = F.cosine_similarity(
        target_gp_values, source_position_values[:, None, :], dim=-1, eps=1e-12
    )
    gp_coordinate_agreement = 0.5 * (forward + backward)

    def ranked(score: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"score": score, "order": torch.argsort(score, dim=1, descending=True, stable=True)}

    return {
        "roma_encoder_cosine": ranked(encoder_cosine),
        "roma_gp_coordinate_agreement": {
            **ranked(gp_coordinate_agreement),
            "forward_score": forward,
            "backward_score": backward,
        },
    }


def _load_roma_scale16_fields(
    model: Any,
    source_path: Path,
    target_path: Path,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Run the official RoMa encoder and scale-16 GP exactly once per pair."""

    try:
        from romatch.utils import get_tuple_transform_ops
    except ImportError as error:  # pragma: no cover - exercised on AutoDL.
        raise RuntimeError("RoMa preprocessing utilities are unavailable") from error
    with Image.open(source_path) as source_image, Image.open(target_path) as target_image:
        transform = get_tuple_transform_ops(
            resize=(int(model.h_resized), int(model.w_resized)), normalize=True, clahe=False
        )
        source_tensor, target_tensor = transform(
            (source_image.convert("RGB"), target_image.convert("RGB"))
        )
    batch = {
        "im_A": source_tensor.unsqueeze(0).to(device),
        "im_B": target_tensor.unsqueeze(0).to(device),
    }
    with torch.inference_mode():
        pyramid = model.extract_backbone_features(batch, batched=True)
        # RoMa's public Decoder indexes the encoder pyramid by integer scale
        # (``f1[int(new_scale)]``), while its projection ModuleDict uses the
        # corresponding string key (``proj["16"]``).
        if 16 not in pyramid or int(pyramid[16].shape[0]) != 2:
            raise RuntimeError("official RoMa encoder did not return a two-image scale-16 pyramid")
        projected = model.decoder.proj["16"](pyramid[16])
        reverse_projected = torch.cat(
            (projected.chunk(2)[1], projected.chunk(2)[0]), dim=0
        )
        gp = model.decoder.gps["16"](projected, reverse_projected)
        position = model.decoder.gps["16"].get_pos_enc(reverse_projected)
    return {
        "source_projected": projected[:1].detach(),
        "target_projected": projected[1:].detach(),
        "source_gp": gp[:1].detach(),
        "target_gp": gp[1:].detach(),
        "source_position_basis": position[1:].detach(),
        "target_position_basis": position[:1].detach(),
    }


def _cohort_name(point: dict[str, Any]) -> str:
    if not bool(point["attention_top20_pck_hit"]):
        return "attention_top20_miss"
    if bool(point["both_wrong_top20_hit"]):
        return "both_wrong_top20_hit"
    if not bool(point["baseline_pck_hit"]) and not bool(point["attention_top1_pck_hit"]):
        return "oracle_gap"
    if bool(point["baseline_pck_hit"]) and not bool(point["attention_top1_pck_hit"]):
        return "attention_harms_native"
    return "other_top20_hit"


def _summarize(records: list[dict[str, Any]], score_name: str) -> dict[str, Any]:
    all_rows = [point for pair in records for point in pair["points"]]
    cohorts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    cohorts["all"].extend(all_rows)
    for point in all_rows:
        cohorts[_cohort_name(point)].append(point)
    summary: dict[str, Any] = {}
    for name, rows in cohorts.items():
        ranked_rows = [row for row in rows if row["attention_top20_pck_hit"]]
        ranks = [int(row["scores"][score_name]["gt_rank"]) for row in ranked_rows if row["scores"][score_name]["gt_rank"] is not None]
        top1 = sum(bool(row["scores"][score_name]["method_pck_hit"]) for row in rows)
        summary[name] = {
            "points": len(rows),
            "attention_top20_hit": len(ranked_rows),
            "top1_pck": float(100.0 * top1 / max(1, len(rows))),
            "top1_among_top20_hit": float(
                100.0
                * sum(bool(row["scores"][score_name]["method_pck_hit"]) for row in ranked_rows)
                / max(1, len(ranked_rows))
            ),
            "gt_rank_mean": float(sum(ranks) / len(ranks)) if ranks else None,
            "gt_rank_median": float(torch.tensor(ranks, dtype=torch.float32).median()) if ranks else None,
            **{
                f"top{k}_among_top20_hit": float(
                    100.0
                    * sum(bool(row["scores"][score_name]["topk_hits"][str(k)]) for row in ranked_rows)
                    / max(1, len(ranked_rows))
                )
                for k in TOPKS
            },
        }
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--attention_audit_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--roma_weights", required=True)
    parser.add_argument("--roma_dinov2_weights", required=True)
    parser.add_argument("--roma_coarse_res", type=int, default=560)
    parser.add_argument("--roma_upsample_res", type=int, default=864)
    parser.add_argument("--roma_precision", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--max_pairs_per_cat", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    payload = json.loads(Path(args.attention_audit_json).read_text(encoding="utf-8"))
    source_records = _validate_audit(payload)
    model = _build_roma(args, device)
    selected: list[dict[str, Any]] = []
    category_counts: defaultdict[str, int] = defaultdict(int)

    for pair in tqdm(source_records, desc="RoMa internal candidate audit"):
        category = str(pair["category"])
        if int(args.max_pairs_per_cat) and category_counts[category] >= int(args.max_pairs_per_cat):
            continue
        category_counts[category] += 1
        source_path = Path(args.dataset_path) / "JPEGImages" / category / pair["src_image"]
        target_path = Path(args.dataset_path) / "JPEGImages" / category / pair["trg_image"]
        if not source_path.is_file() or not target_path.is_file():
            raise FileNotFoundError(f"missing SPair image: {source_path} or {target_path}")
        with Image.open(source_path) as source_image, Image.open(target_path) as target_image:
            source_size = (int(source_image.height), int(source_image.width))
            target_size = (int(target_image.height), int(target_image.width))
        source_points = torch.tensor(
            [point["source_point"] for point in pair["points"]], device=device, dtype=torch.float32
        )
        attention_candidates = [
            sorted(point["candidates"], key=lambda row: int(row["attention_rank"]))
            for point in pair["points"]
        ]
        candidate_points = torch.tensor(
            [[candidate["pixel"] for candidate in row] for row in attention_candidates],
            device=device,
            dtype=torch.float32,
        )
        fields = _load_roma_scale16_fields(model, source_path, target_path, device)
        scores = rank_roma_internal_candidates(
            source_points, candidate_points, source_size, target_size, **fields
        )
        point_rows: list[dict[str, Any]] = []
        for point_index, original in enumerate(pair["points"]):
            result: dict[str, Any] = {}
            for score_name, ranking in scores.items():
                order = ranking["order"][point_index].detach().cpu().tolist()
                hit_flags = [bool(attention_candidates[point_index][index]["pck_hit"]) for index in order]
                candidate_rows = []
                for rank, index in enumerate(order, start=1):
                    candidate = attention_candidates[point_index][index]
                    row = {
                        "rank": int(rank),
                        "attention_rank": int(candidate["attention_rank"]),
                        "pixel": [int(value) for value in candidate["pixel"]],
                        "pck_hit": bool(candidate["pck_hit"]),
                        "score": float(ranking["score"][point_index, index].detach().cpu()),
                    }
                    if "forward_score" in ranking:
                        row["forward_score"] = float(ranking["forward_score"][point_index, index].detach().cpu())
                        row["backward_score"] = float(ranking["backward_score"][point_index, index].detach().cpu())
                    candidate_rows.append(row)
                result[score_name] = {
                    "method_prediction": candidate_rows[0]["pixel"],
                    "method_pck_hit": bool(hit_flags[0]),
                    "gt_rank": next((rank + 1 for rank, hit in enumerate(hit_flags) if hit), None),
                    "topk_hits": {str(k): bool(any(hit_flags[: min(k, len(hit_flags))])) for k in TOPKS},
                    "candidates": candidate_rows,
                }
            point_rows.append(
                {
                    "keypoint_index": int(original["keypoint_index"]),
                    "source_point": original["source_point"],
                    "target_point": original["target_point"],
                    "baseline_pck_hit": bool(original["baseline_pck_hit"]),
                    "attention_top1_pck_hit": bool(original["attention_top1_pck_hit"]),
                    "attention_top20_pck_hit": bool(original["attention_top20_pck_hit"]),
                    "both_wrong_top20_hit": bool(original["both_wrong_top20_hit"]),
                    "scores": result,
                }
            )
        selected.append({
            "category": category,
            "src_image": pair["src_image"],
            "trg_image": pair["trg_image"],
            "points": point_rows,
        })
        del fields, scores
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output = {
        "method_hypothesis": METHOD_HYPOTHESIS,
        "protocol": {
            "attention_audit_json": str(Path(args.attention_audit_json).resolve()),
            "roma_coarse_res": int(args.roma_coarse_res),
            "roma_upsample_res": int(args.roma_upsample_res),
            "max_pairs_per_cat": int(args.max_pairs_per_cat),
            "gt_used_for_scoring": False,
            "gt_used_for_audit_only": True,
        },
        "summaries": {},
        "pair_records": selected,
    }
    for score_name in METHOD_HYPOTHESIS["scores"]:
        output["summaries"][score_name] = _summarize(selected, score_name)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(output, indent=2), encoding="utf-8")
    for score_name, summary in output["summaries"].items():
        all_rows = summary["all"]
        hard_rows = summary.get("both_wrong_top20_hit", {})
        print(
            f"{score_name}: all top1={all_rows['top1_pck']:.2f}, "
            f"top20-hit top1={all_rows['top1_among_top20_hit']:.2f}, "
            f"both-wrong top1={hard_rows.get('top1_among_top20_hit', 0.0):.2f}"
        )


if __name__ == "__main__":
    main()

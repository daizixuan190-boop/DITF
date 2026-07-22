"""Evaluate geometry-aware relational candidate ownership on SPair-71k.

This second ownership evaluator keeps DiTF post-AdaLN and optional channel
discard unchanged. It estimates a robust, label-free source-to-target
similarity transform from high-confidence baseline anchors, then combines
transform consistency with the v1 relational distance profile. Ground-truth
target keypoints are never used by transform estimation, scoring, or gating.
"""

from typing import Any

import torch

from eval_spair_relational_ownership import (
    build_argument_parser,
    build_shared_candidate_union,
    main as run_evaluation,
    ownership_margins,
    relational_scores,
    select_anchors_and_risk,
    strongest_other_scores,
)


def _invalid_transform(device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    return {
        "valid": False,
        "rotation": torch.eye(2, device=device, dtype=dtype),
        "scale": torch.tensor(1.0, device=device, dtype=dtype),
        "translation": torch.zeros(2, device=device, dtype=dtype),
        "inlier_mask": torch.zeros(0, device=device, dtype=torch.bool),
        "inlier_count": 0,
        "inlier_ratio": 0.0,
        "median_residual": float("inf"),
        "mean_residual": float("inf"),
        "quality": 0.0,
        "is_reflection": False,
    }


def apply_similarity(points: torch.Tensor, estimate: dict[str, Any]) -> torch.Tensor:
    return (
        float(estimate["scale"].item())
        * points.float()
        @ estimate["rotation"].transpose(0, 1)
        + estimate["translation"]
    )


def fit_weighted_similarity(
    source: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    determinant: int,
) -> dict[str, Any]:
    """Fit y = scale * R x + translation with det(R) fixed to +/-1."""
    device = source.device
    dtype = torch.float32
    source = source.to(dtype)
    target = target.to(dtype)
    weights = weights.to(dtype).clamp_min(0.0)
    if int(source.shape[0]) < 2 or float(weights.sum().item()) <= 1e-8:
        return _invalid_transform(device, dtype)

    weights = weights / weights.sum()
    source_mean = torch.sum(source * weights.unsqueeze(1), dim=0)
    target_mean = torch.sum(target * weights.unsqueeze(1), dim=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    source_variance = torch.sum(weights * torch.sum(source_centered.square(), dim=1))
    if float(source_variance.item()) <= 1e-8:
        return _invalid_transform(device, dtype)

    covariance = source_centered.transpose(0, 1) @ (weights.unsqueeze(1) * target_centered)
    try:
        u, _, vh = torch.linalg.svd(covariance)
    except RuntimeError:
        return _invalid_transform(device, dtype)
    v = vh.transpose(0, 1)
    base_rotation = v @ u.transpose(0, 1)
    correction = torch.eye(2, device=device, dtype=dtype)
    base_determinant = 1.0 if float(torch.det(base_rotation).item()) >= 0.0 else -1.0
    correction[-1, -1] = float(determinant) * base_determinant
    rotation = v @ correction @ u.transpose(0, 1)

    rotated_source = source_centered @ rotation.transpose(0, 1)
    scale = torch.sum(weights * torch.sum(rotated_source * target_centered, dim=1)) / source_variance
    if not bool(torch.isfinite(scale)) or float(scale.item()) <= 1e-6:
        return _invalid_transform(device, dtype)
    translation = target_mean - scale * (source_mean @ rotation.transpose(0, 1))
    return {
        **_invalid_transform(device, dtype),
        "valid": True,
        "rotation": rotation,
        "scale": scale,
        "translation": translation,
        "is_reflection": bool(determinant < 0),
    }


def _candidate_determinants(args) -> tuple[int, ...]:
    if args.rco_v2_transform_mode == "rotation":
        return (1,)
    return (1, -1)


def estimate_robust_similarity(
    source: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    src_threshold: float,
    trg_threshold: float,
    args,
) -> dict[str, Any]:
    """Deterministic two-anchor RANSAC followed by robust inlier refitting."""
    device = source.device
    dtype = torch.float32
    source = source.to(dtype)
    target = target.to(dtype)
    confidence = confidence.to(dtype).clamp_min(1e-4)
    num_anchors = int(source.shape[0])
    invalid = _invalid_transform(device, dtype)
    invalid["inlier_mask"] = torch.zeros(num_anchors, device=device, dtype=torch.bool)
    if num_anchors < max(2, int(args.rco_v2_min_inliers)):
        return invalid

    residual_threshold = max(float(args.rco_v2_residual_threshold), 1e-6)
    pairs = torch.combinations(torch.arange(num_anchors, device=device), r=2)
    source_delta = source[pairs[:, 1]] - source[pairs[:, 0]]
    target_delta = target[pairs[:, 1]] - target[pairs[:, 0]]
    source_norm = torch.linalg.norm(source_delta, dim=1)
    target_norm = torch.linalg.norm(target_delta, dim=1)
    valid_pairs = (
        source_norm / max(float(src_threshold), 1e-6)
        >= float(args.rco_v2_min_sample_separation)
    ) & (
        target_norm / max(float(trg_threshold), 1e-6)
        >= float(args.rco_v2_min_sample_separation)
    )
    if not bool(valid_pairs.any()):
        return invalid

    pairs = pairs[valid_pairs]
    source_delta = source_delta[valid_pairs]
    target_delta = target_delta[valid_pairs]
    source_norm = source_norm[valid_pairs]
    target_norm = target_norm[valid_pairs]
    source_unit = source_delta / source_norm.unsqueeze(1).clamp_min(1e-6)
    target_unit = target_delta / target_norm.unsqueeze(1).clamp_min(1e-6)
    scale = target_norm / source_norm.clamp_min(1e-6)

    rotations = []
    hypothesis_scales = []
    hypothesis_first = []
    if 1 in _candidate_determinants(args):
        cosine = torch.sum(source_unit * target_unit, dim=1)
        sine = source_unit[:, 0] * target_unit[:, 1] - source_unit[:, 1] * target_unit[:, 0]
        proper = torch.stack(
            (
                torch.stack((cosine, -sine), dim=1),
                torch.stack((sine, cosine), dim=1),
            ),
            dim=1,
        )
        rotations.append(proper)
        hypothesis_scales.append(scale)
        hypothesis_first.append(pairs[:, 0])
    if -1 in _candidate_determinants(args):
        cosine_sum = source_unit[:, 0] * target_unit[:, 0] - source_unit[:, 1] * target_unit[:, 1]
        sine_sum = source_unit[:, 1] * target_unit[:, 0] + source_unit[:, 0] * target_unit[:, 1]
        reflected = torch.stack(
            (
                torch.stack((cosine_sum, sine_sum), dim=1),
                torch.stack((sine_sum, -cosine_sum), dim=1),
            ),
            dim=1,
        )
        rotations.append(reflected)
        hypothesis_scales.append(scale)
        hypothesis_first.append(pairs[:, 0])

    rotations = torch.cat(rotations, dim=0)
    hypothesis_scales = torch.cat(hypothesis_scales, dim=0)
    hypothesis_first = torch.cat(hypothesis_first, dim=0)
    first_source_projected = torch.einsum(
        "hd,hkd->hk",
        source[hypothesis_first],
        rotations,
    )
    translations = target[hypothesis_first] - hypothesis_scales.unsqueeze(1) * first_source_projected
    projected = hypothesis_scales[:, None, None] * torch.einsum(
        "nd,hkd->hnk",
        source,
        rotations,
    ) + translations[:, None, :]
    residuals = torch.linalg.norm(projected - target.unsqueeze(0), dim=2)
    residuals = residuals / max(float(trg_threshold), 1e-6)
    inliers = residuals <= residual_threshold
    inlier_counts = inliers.sum(dim=1)
    consensus = (inliers.float() * confidence.unsqueeze(0)).sum(dim=1)
    weighted_residuals = (
        residuals * inliers.float() * confidence.unsqueeze(0)
    ).sum(dim=1) / (inliers.float() * confidence.unsqueeze(0)).sum(dim=1).clamp_min(1e-6)
    valid_hypotheses = inlier_counts >= 2
    if not bool(valid_hypotheses.any()):
        return invalid

    valid_indices = torch.nonzero(valid_hypotheses, as_tuple=False).squeeze(1)
    max_consensus = consensus[valid_indices].max()
    valid_indices = valid_indices[
        torch.isclose(consensus[valid_indices], max_consensus, rtol=1e-5, atol=1e-6)
    ]
    max_inliers = inlier_counts[valid_indices].max()
    valid_indices = valid_indices[inlier_counts[valid_indices] == max_inliers]
    best_index = valid_indices[torch.argmin(weighted_residuals[valid_indices])]
    best_estimate = {
        **_invalid_transform(device, dtype),
        "valid": True,
        "rotation": rotations[best_index],
        "scale": hypothesis_scales[best_index],
        "translation": translations[best_index],
        "is_reflection": bool(float(torch.det(rotations[best_index]).item()) < 0.0),
    }

    determinant = -1 if best_estimate["is_reflection"] else 1
    estimate = best_estimate
    for _ in range(max(int(args.rco_v2_refine_steps), 1)):
        residuals = torch.linalg.norm(apply_similarity(source, estimate) - target, dim=1)
        residuals = residuals / max(float(trg_threshold), 1e-6)
        inliers = residuals <= residual_threshold
        if int(inliers.sum().item()) < int(args.rco_v2_min_inliers):
            return invalid
        robust = 1.0 / (1.0 + (residuals[inliers] / residual_threshold).square())
        estimate = fit_weighted_similarity(
            source[inliers],
            target[inliers],
            confidence[inliers] * robust,
            determinant,
        )
        if not estimate["valid"]:
            return invalid

    residuals = torch.linalg.norm(apply_similarity(source, estimate) - target, dim=1)
    residuals = residuals / max(float(trg_threshold), 1e-6)
    inliers = residuals <= residual_threshold
    inlier_count = int(inliers.sum().item())
    if inlier_count == 0:
        return invalid
    inlier_ratio = float(
        confidence[inliers].sum().item() / confidence.sum().clamp_min(1e-6).item()
    )
    median_residual = float(torch.median(residuals[inliers]).item())
    mean_residual = float(
        (residuals[inliers] * confidence[inliers]).sum().item()
        / confidence[inliers].sum().clamp_min(1e-6).item()
    )
    valid = (
        inlier_count >= int(args.rco_v2_min_inliers)
        and inlier_ratio >= float(args.rco_v2_min_inlier_ratio)
        and median_residual <= float(args.rco_v2_max_median_residual)
    )
    quality = inlier_ratio * torch.exp(
        torch.tensor(-median_residual / residual_threshold, device=device)
    ).item()
    estimate.update(
        {
            "valid": bool(valid),
            "inlier_mask": inliers,
            "inlier_count": inlier_count,
            "inlier_ratio": inlier_ratio,
            "median_residual": median_residual,
            "mean_residual": mean_residual,
            "quality": float(quality) if valid else 0.0,
        }
    )
    return estimate


def geometric_ownership_scores(
    src_xy: torch.Tensor,
    union_xy: torch.Tensor,
    transform: dict[str, Any],
    trg_threshold: float,
    sigma: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    projected = apply_similarity(src_xy, transform)
    normalized_distance = torch.cdist(projected, union_xy.float())
    normalized_distance = normalized_distance / max(float(trg_threshold), 1e-6)
    score = torch.exp(-0.5 * (normalized_distance / max(float(sigma), 1e-6)).square())
    margin = ownership_margins(score)
    return score, margin, normalized_distance


def geometry_aware_relational_ownership(
    records: list[dict[str, Any]],
    src_ft: torch.Tensor,
    trg_ft: torch.Tensor,
    src_points: list[list[int]],
    src_threshold: float,
    trg_threshold: float,
    args,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[dict[str, Any]], dict[str, torch.Tensor]]:
    bundle = build_shared_candidate_union(records, src_ft, trg_ft, src_points)
    raw_scores = bundle["raw_scores"].float()
    base_columns = bundle["base_columns"]
    num_points = int(raw_scores.shape[0])
    point_indices = torch.arange(num_points, device=raw_scores.device)

    margins = torch.tensor([float(record["margin"]) for record in records], device=raw_scores.device)
    strongest_raw_other, strongest_raw_other_idx = strongest_other_scores(raw_scores)
    top1_self = raw_scores[point_indices, base_columns]
    top1_other = strongest_raw_other[point_indices, base_columns]
    top1_exclusivity = top1_self - top1_other
    anchor_indices, anchor_confidence, risk = select_anchors_and_risk(margins, top1_exclusivity, args)

    relation, anchor_counts = relational_scores(
        bundle["src_xy"],
        bundle["union_xy"],
        base_columns,
        anchor_indices,
        anchor_confidence,
        src_threshold,
        trg_threshold,
        args,
    )
    relation_margin = ownership_margins(relation)
    base_relation = relation[point_indices, base_columns]
    base_relation_margin = relation_margin[point_indices, base_columns]
    relation_delta = relation - base_relation.unsqueeze(1)
    relation_margin_delta = relation_margin - base_relation_margin.unsqueeze(1)

    if int(anchor_indices.numel()) > 0:
        anchor_source = bundle["src_xy"][anchor_indices]
        anchor_target = bundle["union_xy"][base_columns[anchor_indices]]
        transform = estimate_robust_similarity(
            anchor_source,
            anchor_target,
            anchor_confidence[anchor_indices],
            src_threshold,
            trg_threshold,
            args,
        )
    else:
        transform = _invalid_transform(raw_scores.device, torch.float32)

    geometry, geometry_margin, projection_distance = geometric_ownership_scores(
        bundle["src_xy"],
        bundle["union_xy"],
        transform,
        trg_threshold,
        args.rco_v2_geometry_sigma,
    )
    if not transform["valid"]:
        geometry.zero_()
        geometry_margin.zero_()
    base_geometry = geometry[point_indices, base_columns]
    base_geometry_margin = geometry_margin[point_indices, base_columns]
    geometry_delta = geometry - base_geometry.unsqueeze(1)
    geometry_margin_delta = geometry_margin - base_geometry_margin.unsqueeze(1)

    transform_quality = float(transform["quality"])
    structural_delta = (
        float(args.rco_relation_weight) * relation_delta
        + float(args.rco_owner_margin_weight) * relation_margin_delta
        + transform_quality * float(args.rco_v2_geometry_weight) * geometry_delta
        + transform_quality * float(args.rco_v2_geometry_owner_weight) * geometry_margin_delta
    )
    final_scores = raw_scores + risk.unsqueeze(1) * structural_delta
    proposed_columns = torch.argmax(final_scores, dim=1)

    predictions: list[tuple[int, int]] = []
    baseline_predictions: list[tuple[int, int]] = []
    diagnostics: list[dict[str, Any]] = []
    anchor_set = {int(index) for index in anchor_indices.detach().cpu().tolist()}
    transform_det = float(torch.det(transform["rotation"]).item())

    for point_idx in range(num_points):
        base_column = int(base_columns[point_idx].item())
        proposed_column = int(proposed_columns[point_idx].item())
        raw_gap = float((raw_scores[point_idx, base_column] - raw_scores[point_idx, proposed_column]).item())
        relation_gain = float(relation_delta[point_idx, proposed_column].item())
        owner_margin_gain = float(relation_margin_delta[point_idx, proposed_column].item())
        geometry_gain = float(geometry_delta[point_idx, proposed_column].item())
        geometry_owner_gain = float(geometry_margin_delta[point_idx, proposed_column].item())
        structural_gain = float(structural_delta[point_idx, proposed_column].item())
        base_projection_error = float(projection_distance[point_idx, base_column].item())
        proposed_projection_error = float(projection_distance[point_idx, proposed_column].item())
        projection_error_gain = base_projection_error - proposed_projection_error
        gate = (
            bool(transform["valid"])
            and proposed_column != base_column
            and int(anchor_counts[point_idx].item()) >= int(args.rco_min_point_anchors)
            and float(risk[point_idx].item()) >= float(args.rco_min_risk)
            and relation_gain >= float(args.rco_v2_min_relation_gain)
            and geometry_gain >= float(args.rco_v2_min_geometry_gain)
            and projection_error_gain >= float(args.rco_v2_min_projection_gain)
            and proposed_projection_error <= float(args.rco_v2_max_projection_error)
            and structural_gain >= float(args.rco_min_structural_gain)
            and raw_gap <= float(args.rco_max_raw_gap)
            and (args.rco_modify_anchors or point_idx not in anchor_set)
        )
        final_column = proposed_column if gate else base_column
        baseline_predictions.append(
            (
                int(bundle["union_x"][base_column].item()),
                int(bundle["union_y"][base_column].item()),
            )
        )
        predictions.append(
            (
                int(bundle["union_x"][final_column].item()),
                int(bundle["union_y"][final_column].item()),
            )
        )
        diagnostics.append(
            {
                "point_idx": point_idx,
                "anchor_count": int(anchor_counts[point_idx].item()),
                "pair_anchor_count": int(anchor_indices.numel()),
                "is_anchor": int(point_idx in anchor_set),
                "anchor_confidence": float(anchor_confidence[point_idx].item()),
                "risk_weight": float(risk[point_idx].item()),
                "raw_margin": float(margins[point_idx].item()),
                "base_exclusivity": float(top1_exclusivity[point_idx].item()),
                "base_strongest_other_idx": int(strongest_raw_other_idx[point_idx, base_column].item()),
                "base_raw_score": float(raw_scores[point_idx, base_column].item()),
                "proposed_raw_score": float(raw_scores[point_idx, proposed_column].item()),
                "raw_gap": raw_gap,
                "base_relation_score": float(base_relation[point_idx].item()),
                "proposed_relation_score": float(relation[point_idx, proposed_column].item()),
                "relation_gain": relation_gain,
                "base_relation_margin": float(base_relation_margin[point_idx].item()),
                "proposed_relation_margin": float(relation_margin[point_idx, proposed_column].item()),
                "owner_margin_gain": owner_margin_gain,
                "base_geometry_score": float(base_geometry[point_idx].item()),
                "proposed_geometry_score": float(geometry[point_idx, proposed_column].item()),
                "geometry_gain": geometry_gain,
                "base_geometry_margin": float(base_geometry_margin[point_idx].item()),
                "proposed_geometry_margin": float(geometry_margin[point_idx, proposed_column].item()),
                "geometry_owner_margin_gain": geometry_owner_gain,
                "base_projection_error": base_projection_error,
                "proposed_projection_error": proposed_projection_error,
                "projection_error_gain": projection_error_gain,
                "transform_valid": int(bool(transform["valid"])),
                "transform_inlier_count": int(transform["inlier_count"]),
                "transform_inlier_ratio": float(transform["inlier_ratio"]),
                "transform_median_residual": float(transform["median_residual"]),
                "transform_mean_residual": float(transform["mean_residual"]),
                "transform_quality": transform_quality,
                "transform_scale": float(transform["scale"].item()),
                "transform_determinant": transform_det,
                "structural_gain": structural_gain,
                "base_final_score": float(final_scores[point_idx, base_column].item()),
                "proposed_final_score": float(final_scores[point_idx, proposed_column].item()),
                "proposed_pred_x": int(bundle["union_x"][proposed_column].item()),
                "proposed_pred_y": int(bundle["union_y"][proposed_column].item()),
                "base_column": base_column,
                "proposed_column": proposed_column,
                "final_column": final_column,
                "changed": int(final_column != base_column),
                "gate_passed": int(gate),
            }
        )

    bundle.update(
        {
            "final_scores": final_scores,
            "relation_scores": relation,
            "relation_margins": relation_margin,
            "geometry_scores": geometry,
            "geometry_margins": geometry_margin,
            "projection_distances": projection_distance,
            "risk": risk,
            "anchor_indices": anchor_indices,
        }
    )
    return predictions, baseline_predictions, diagnostics, bundle


def build_v2_parser():
    parser = build_argument_parser("SPair geometry-aware relational ownership evaluator")
    parser.set_defaults(
        rco_max_anchors=12,
        rco_min_risk=0.20,
        rco_min_relation_gain=-0.05,
        rco_min_structural_gain=0.02,
        rco_max_raw_gap=0.25,
    )
    parser.add_argument("--rco_v2_transform_mode", choices=["rotation", "auto"], default="auto")
    parser.add_argument("--rco_v2_min_sample_separation", default=0.08, type=float)
    parser.add_argument("--rco_v2_residual_threshold", default=0.08, type=float)
    parser.add_argument("--rco_v2_min_inliers", default=3, type=int)
    parser.add_argument("--rco_v2_min_inlier_ratio", default=0.50, type=float)
    parser.add_argument("--rco_v2_max_median_residual", default=0.06, type=float)
    parser.add_argument("--rco_v2_refine_steps", default=3, type=int)
    parser.add_argument("--rco_v2_geometry_sigma", default=0.12, type=float)
    parser.add_argument("--rco_v2_geometry_weight", default=0.20, type=float)
    parser.add_argument("--rco_v2_geometry_owner_weight", default=0.15, type=float)
    parser.add_argument("--rco_v2_min_relation_gain", default=-0.05, type=float)
    parser.add_argument("--rco_v2_min_geometry_gain", default=0.05, type=float)
    parser.add_argument("--rco_v2_min_projection_gain", default=0.01, type=float)
    parser.add_argument("--rco_v2_max_projection_error", default=0.20, type=float)
    return parser


if __name__ == "__main__":
    args = build_v2_parser().parse_args()
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    method_tag = (
        f"rco2_topk{args.rco_topk}_aq{args.rco_anchor_quantile}"
        f"_gw{args.rco_v2_geometry_weight}_gow{args.rco_v2_geometry_owner_weight}"
        f"_rt{args.rco_v2_residual_threshold}_ir{args.rco_v2_min_inlier_ratio}"
    )
    run_evaluation(
        args,
        ownership_fn=geometry_aware_relational_ownership,
        method_tag=method_tag,
        method_name="geometry_aware_relational_candidate_ownership",
    )

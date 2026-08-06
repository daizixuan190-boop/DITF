"""Official DiTF SPair evaluator for attention-side train-free experiments.

The feature extraction, AdaLN processing, channel discard, resizing, pair
loading, and PCK calculation follow ``eval_spair.py`` at the repository HEAD.
Only the final matcher is switchable.  ``--matcher nn`` is the official
baseline path; FJSAR modes are the current attention replay work surface.
"""

import argparse
import gc
import hashlib
import json
import os
import random
import shutil
import time
from collections import Counter, OrderedDict

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from PIL import Image
from torchvision.transforms import PILToTensor
from tqdm import tqdm
from typing import Any

from flux_native_in_memory import load_flux_image_tensor

from spair_matchers import (
    cosine_nn_predict,
    flux_fjsar_attention_case_records,
    flux_fjsar_dump_candidates,
    flux_fjsar_identity_decodability_batch,
    flux_fjsar_predict,
)
from src.flux.feat_flux import Featurizer4Eval

from flux_joint_replay import (
    FluxMultiPreBlockCapture,
    FluxPreBlockCapture,
    find_flux_core_model,
    is_valid_flux_replay_entry,
    select_flux_single_blocks,
    replay_start_block_index,
)


FJSAR_ALL_MATCHERS = (
    "fjsar_attn",
    "fjsar_attention_signature",
    "fjsar_part_sharpen",
    "fjsar_orthogonal_context",
    "fjsar_spectral_identity",
    "fjsar_filtered_spectral_kernel",
    "fjsar_transport_lift",
    "fjsar_basin_contrastive_identity",
    "fjsar_attention_isometry",
    "fjsar_identity_preserving_attention",
    "fjsar_balanced_transport_attention",
    "fjsar_qk_identity_attention",
    "fjsar_cross_attention_trajectory",
    "fjsar_native_preserving_topology_rescue",
    "fjsar_attention_basin_native_refine",
    "fjsar_candidate_graph_consensus_verification",
    "fjsar_geometry_consistent_attention",
    "fjsar_candidate_conditioned_verification",
    "fjsar_candidate_local_transport_verification",
    "fjsar_attention_relational_graph_matching",
    "fjsar_dense_partial_graph_matching",
    "fjsar_expert_preserving_attention_hypothesis_conditioned_replay",
    "fjsar_pre_softmax_channelwise_identity",
    "fjsar_layer_routed_identity",
    "fjsar_pre_single_stream_identity",
)


def _load_flux_fjsar_components(args, device):
    """Load frozen block(s) that end at the official feature boundary."""
    try:
        from flux.util import load_flow_model
    except ImportError:
        from src.flux.util import load_flow_model

    model = load_flow_model("flux-dev", device=device)
    model.eval().requires_grad_(False)
    feature_block = int(args.k[0] if isinstance(args.k, list) else args.k)
    blocks = select_flux_single_blocks(model, feature_block, depth=1)
    return model, blocks


def _load_flux_fjsar_runtime(args, all_cats):
    """Load the FLUX evaluator once for on-demand aligned replay extraction."""

    featurizer = Featurizer4Eval(cat_list=all_cats[:], ensemble_size=args.ensemble_size)
    core = find_flux_core_model(featurizer)
    feature_block = int(args.k[0] if isinstance(args.k, list) else args.k)
    blocks = select_flux_single_blocks(core, feature_block, depth=1)
    return featurizer, core, blocks


def _fjsar_protocol_seed(args) -> int:
    """Use the same deterministic noise ensemble for every image.

    Pairing ensemble member ``e`` across A and B is meaningful only when both
    images use the same extraction RNG stream.  The seed depends on the feature
    protocol, not the image name.
    """

    feature_block = int(args.k[0] if isinstance(args.k, list) else args.k)
    payload = (
        f"fjsar-v4|size={tuple(args.img_size)}|t={int(args.t)}|"
        f"k={feature_block}|e={int(args.ensemble_size)}"
    )
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4], "big")


def _fjsar_multilayer_blocks(args) -> tuple[int, ...]:
    matcher = str(getattr(args, "matcher", ""))
    if matcher == "fjsar_pre_single_stream_identity":
        requested = getattr(args, "fjsar_pre_single_stream_blocks", None) or ()
        blocks = tuple(sorted({int(block) for block in requested}))
        if not blocks or any(block < 0 or block >= 19 for block in blocks):
            raise ValueError(
                "FJSAR pre-single-stream blocks must be FLUX double-stream indices in [0, 18]"
            )
        return blocks
    if not (
        bool(getattr(args, "fjsar_multilayer_identity_audit", False))
        or matcher == "fjsar_layer_routed_identity"
    ):
        return ()
    feature_block = int(args.k[0] if isinstance(args.k, list) else args.k)
    requested = getattr(args, "fjsar_multilayer_blocks", None) or ()
    blocks = tuple(sorted({feature_block, *(int(block) for block in requested)}))
    if any(block < 19 for block in blocks):
        raise ValueError("FJSAR multilayer features currently support FLUX single blocks >= 19")
    return blocks


def _fjsar_trajectory_blocks(args) -> tuple[int, ...]:
    matcher = str(getattr(args, "matcher", ""))
    enabled = matcher in {"fjsar_cross_attention_trajectory", "fjsar_all"}
    if not enabled:
        return ()
    feature_block = int(args.k[0] if isinstance(args.k, list) else args.k)
    requested = getattr(args, "fjsar_trajectory_blocks", None) or ()
    blocks = tuple(sorted({feature_block, *(int(block) for block in requested)}))
    if any(block < 20 for block in blocks):
        raise ValueError("FJSAR trajectory blocks must be >= 20 so their pre-block states are single-stream")
    return blocks


def _fjsar_multi_timestep_values(args) -> tuple[int, ...]:
    if not bool(getattr(args, "fjsar_multi_timestep_attention_identity_audit", False)):
        return ()
    requested = getattr(args, "fjsar_multi_timestep_values", None) or ()
    values = tuple(sorted({int(value) for value in requested if int(value) > 0}))
    if values:
        return values
    base_t = max(1, int(args.t))
    default_values = {
        min(1000, max(20, int(round(base_t * 0.3 / 10.0)) * 10)),
        min(1000, max(20, int(round(base_t * 0.6 / 10.0)) * 10)),
        min(1000, base_t),
    }
    return tuple(sorted(default_values))


def _fjsar_multi_timestep_seed(args) -> int:
    feature_block = int(args.k[0] if isinstance(args.k, list) else args.k)
    payload = (
        f"fjsar-multi-timestep|size={tuple(args.img_size)}|"
        f"k={feature_block}|e={int(args.ensemble_size)}|cd={int(bool(args.cd))}"
    )
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4], "big")


def _fjsar_method_hypothesis(args) -> dict[str, object] | None:
    matcher = str(getattr(args, "matcher", ""))
    if matcher == "fjsar_pre_softmax_channelwise_identity":
        return {
            "name": "Pre-Softmax Channelwise Identity Field",
            "mechanism_hypothesis": (
                "Attention posterior provides high-recall candidate regions, but scalar QK logits "
                "and value pooling erase channel-level pair evidence.  The frozen pre-softmax Q/K "
                "interaction and local V residual may retain a finer identity signal."
            ),
            "intervention": (
                "Use exact mutual-attention top-k only as a candidate pool; build a fixed equal-norm "
                "candidate feature from centered per-channel Q*K, local V residual, and sorted local "
                "relation signatures.  Do not multiply the identity score by attention probability."
            ),
            "candidate_contract": "exact_mutual_cross_attention_topk_only",
            "formal_method_signal": "pre_softmax_channelwise_identity",
            "identity_components": [
                "centered_qk_channel_product",
                "local_v_residual",
                "sorted_local_relation",
                "native_local_descriptor",
            ],
            "train_free": True,
            "attention_used_as_identity_score": False,
            "native_fallback": False,
            "gt_used_for_inference": False,
        }
    if matcher == "fjsar_layer_routed_identity":
        return {
            "name": "Layerwise Routing Identity",
            "mechanism_hypothesis": (
                "Deep cross-attention supplies semantic routing, while source-only official features "
                "from multiple single-stream boundaries preserve finer identity information that is "
                "not overwritten by cross-image value injection."
            ),
            "intervention": (
                "Build the exact mutual-attention top-k basin from the deep replay branch.  Fuse that "
                "pair-conditioned routing feature with equal-norm source-only features from every "
                "requested block, and rank candidates without learned weights or native fallback."
            ),
            "candidate_contract": "exact_mutual_cross_attention_topk_only",
            "formal_method_signal": "layerwise_routing_identity",
            "routing_branch": "deep_joint_cross_attention",
            "identity_branches": "source_only_official_multiblock_features",
            "train_free": True,
            "native_fallback": False,
            "gt_used_for_inference": False,
        }
    if matcher == "fjsar_pre_single_stream_identity":
        return {
            "name": "Pre-Single-Stream Identity Routing",
            "mechanism_hypothesis": (
                "Block28 cross-attention supplies high-recall semantic routing, while the FLUX "
                "double-stream image branch before image/text token merging retains local identity "
                "that later single-stream semantic abstraction suppresses."
            ),
            "intervention": (
                "Use exact mutual block28 cross-attention only to freeze the top-k candidate pool. "
                "Rank those candidates with equal-norm concatenated double-stream image features; "
                "attention probability and block28 native descriptors do not enter the identity score."
            ),
            "candidate_contract": "exact_mutual_cross_attention_topk_only",
            "formal_method_signal": "pre_single_stream_multiscale_identity",
            "routing_branch": "deep_joint_cross_attention_candidate_pool_only",
            "identity_branches": "double_stream_image_features_before_single_stream_merge",
            "identity_fusion": "equal_norm_concatenation",
            "train_free": True,
            "native_fallback": False,
            "attention_used_as_identity_score": False,
            "gt_used_for_inference": False,
        }
    if matcher == "fjsar_expert_preserving_attention_hypothesis_conditioned_replay":
        return {
            "name": "Expert-Preserving Attention Hypothesis-Conditioned Replay",
            "mechanism_hypothesis": (
                "Cross-attention top-k has high correspondence recall, but averaging ensemble/head "
                "readouts and value mixtures erases which expert supports each candidate identity."
            ),
            "intervention": (
                "Keep exact mutual-attention top-k candidates, compute bidirectional QK support and "
                "symmetric candidate-specific V residual identity per ensemble member and head, select "
                "one pair-level head by QK/V ranking agreement, then rank candidates before averaging "
                "away the head axis."
            ),
            "candidate_contract": "exact_mutual_cross_attention_topk_only",
            "formal_method_signal": "pair_head_hypothesis",
            "audit_controls": [
                "attention",
                "mean_expert_hypothesis",
                "pair_head_support",
                "pair_head_identity",
                "pair_expert_hypothesis",
                "point_head_hypothesis",
            ],
            "native_candidate_injection": False,
            "native_gate": False,
            "native_fallback": False,
            "gt_used_for_inference": False,
        }
    if matcher == "fjsar_dense_partial_graph_matching":
        return {
            "name": "Dense Attention Partial Graph Matching",
            "mechanism_hypothesis": (
                "The attention top-k pool contains deep correct candidates and dense local DiTF "
                "self-similarity edges weakly separate them, but independent edge readout destroys "
                "already-correct attention identities. Global target capacity should arbitrate these "
                "candidate hypotheses without a native fallback."
            ),
            "intervention": (
                "Create variables for every dense source token, retain only mutual-attention top-k "
                "targets, form one-step relation-edge beliefs, then solve an exact sparse maximum-weight "
                "partial bipartite assignment with unit real-target capacity and private dustbins."
            ),
            "candidate_contract": "mutual_cross_attention_topk_only",
            "unary_signal": "attention_only",
            "pairwise_signal": "dense_local_ditf_self_similarity_relation_only",
            "spatial_edge_used": False,
            "descriptor_unary_used": False,
            "target_capacity": 1,
            "dustbin_rule": "row_mean_dense_graph_belief",
            "native_candidate_injection": False,
            "native_fallback": False,
            "gt_used_for_inference": False,
            "positive_evidence_expected": (
                "The exact partial assignment should preserve attention-correct points while converting "
                "the audited deep relation-edge signal into positive PCK and reducing target collisions."
            ),
        }
    if matcher == "fjsar_attention_relational_graph_matching":
        return {
            "name": "Attention Relational Graph Matching",
            "mechanism_hypothesis": (
                "Mutual cross-attention already contains the correct region in its top-k candidates, "
                "but independent readout loses point identity.  Correct candidates should jointly "
                "preserve descriptor-space relations between annotated source points."
            ),
            "intervention": (
                "Use only mutual-attention top-k states; fuse standardized attention posterior and "
                "block28 descriptor unary evidence; construct complete pairwise potentials from feature "
                "cosine preservation, descriptor-difference direction, and channel co-activation; then "
                "solve the discrete graph with loopy max-product and monotonic ICM."
            ),
            "candidate_contract": "mutual_cross_attention_topk_only",
            "native_candidate_injection": False,
            "native_fallback": False,
            "coordinate_candidate_scoring": False,
            "positive_evidence_expected": (
                "The relational graph should raise top1 selection efficiency inside the attention top20 "
                "pool, with positive energy gains and more baseline-error rescues than baseline-correct harms."
            ),
        }
    if matcher == "fjsar_candidate_conditioned_verification":
        return {
            "name": "Candidate-Conditioned Correspondence Verification",
            "mechanism_hypothesis": (
                "Cross-attention is a high-recall proposal generator but its averaged readout is a "
                "part-level descriptor.  Point identity should instead be verified per target "
                "candidate by independent evidence, so wrong regional attractors cannot dominate "
                "through value aggregation."
            ),
            "intervention": (
                "Build an attention top-k proposal set, add the native top1 as a preservation "
                "hypothesis, and rank each candidate by equal-rank verification over posterior "
                "support, native identity, candidate-centered local relation, and weak topology "
                "from native-attention consensus anchors."
            ),
            "parameter_free_sort_key": [
                "reciprocal-rank vote over attention_posterior",
                "native_identity",
                "local_relation",
                "anchor_topology",
                "median-rank rejection",
            ],
            "positive_evidence_expected": (
                "It should preserve baseline-correct native points more often than attention-only "
                "branches while rescuing oracle-gap points whose GT is inside the attention proposal set."
            ),
        }
    if matcher == "fjsar_candidate_local_transport_verification":
        return {
            "name": "Candidate-Conditioned Local Transport Verification",
            "mechanism_hypothesis": (
                "Cross-attention supplies high-recall candidate regions, but point identity is lost "
                "when candidates are represented by averaged attention readouts.  The missing signal "
                "should be tested by asking whether each candidate can explain a local source-to-target "
                "transport field around the queried point."
            ),
            "intervention": (
                "Include native top1 as a baseline hypothesis, generate a one-token-scale local "
                "neighborhood around each attention candidate, and verify the induced local field by "
                "native patch identity, local self-similarity, neighboring attention support, and weak "
                "anchor topology.  Replace native only when an attention candidate majority-dominates "
                "the native hypothesis on informative transport evidence."
            ),
            "parameter_free_sort_key": [
                "candidate-conditioned local transport evidence",
                "majority dominance over native hypothesis",
                "native identity median-rank compatibility",
                "baseline-preserving abstention",
            ],
            "positive_evidence_expected": (
                "Compared with candidate_conditioned_verification it should reduce harmed native-correct "
                "points while rescuing a subset of oracle-gap cases whose correct candidate has locally "
                "self-consistent transport support."
            ),
        }
    if matcher == "fjsar_candidate_graph_consensus_verification":
        return {
            "name": "Candidate Graph Consensus Verification",
            "mechanism_hypothesis": (
                "Cross-attention already gives high-recall proposals, but point identity becomes stable "
                "only when a candidate is consistent with the geometry implied by the rest of the point "
                "set.  The missing signal is not another descriptor average, but pair-level consensus "
                "over the candidate graph."
            ),
            "intervention": (
                "Reuse the local transport verifier as unary evidence, then run a lightweight "
                "pair-consensus refinement where every candidate competes against the geometry induced "
                "by the current selected assignments of the other keypoints."
            ),
            "parameter_free_sort_key": [
                "local transport unary",
                "pairwise consensus geometry",
                "native-preserving abstention",
                "majority evidence vote",
            ],
            "positive_evidence_expected": (
                "It should preserve points that are already stable under local transport while rescuing "
                "oracle-gap cases whose correct candidates are jointly compatible with the pair geometry."
            ),
        }
    if matcher != "fjsar_cross_attention_trajectory":
        return None
    return {
        "name": "Cross-Attention Trajectory Identity",
        "mechanism_hypothesis": (
            "Single-layer cross-attention raises candidate recall but its top1 is unstable because "
            "part-level posterior peaks can be accidental. A true point correspondence should remain "
            "competitive across multiple frozen DiT blocks, while false local peaks should be less "
            "persistent across depth."
        ),
        "intervention": (
            "Use the main feature block raw mutual-attention top-k as the candidate set, then rerank "
            "those target-indexed candidates by cross-layer posterior trajectory stability instead of "
            "compressing attention into ordinary source/target descriptors."
        ),
        "parameter_free_sort_key": [
            "trajectory_topk_count",
            "trajectory_mean_reciprocal_rank",
            "trajectory_mean_centered_score",
            "main_attention_score",
        ],
        "positive_evidence_expected": (
            "In oracle-gap/harm cases, GT proposals should have better trajectory rank than the raw "
            "main-layer attention top1, raising cross_attention_trajectory@1 while preserving high @20."
        ),
        "falsification_condition": (
            "If GT proposals inside raw attention top-k do not rank higher by trajectory stability than "
            "wrong top1 proposals, then the remaining identity signal is not in cross-layer posterior "
            "stability and the next evidence source should be timestep trajectory rather than more "
            "single-layer descriptor/operator variants."
        ),
        "trajectory_blocks": [int(block) for block in _fjsar_trajectory_blocks(args)],
    }


_ATTENTION_GRAPH_CATEGORY_GROUPS = {
    "rigid": {
        "aeroplane", "bicycle", "boat", "bottle", "bus", "car", "chair",
        "motorbike", "pottedplant", "train", "tvmonitor",
    },
    "articulated_or_animal": {"bird", "cat", "cow", "dog", "horse", "person", "sheep"},
}


def _summarize_attention_relational_graph_audits(
    pair_records: list[dict[str, object]],
) -> dict[str, object]:
    def _empty() -> dict[str, object]:
        return {
            "pair_count": 0,
            "point_count": 0,
            "baseline_correct": 0,
            "method_correct": 0,
            "rescued": 0,
            "harmed": 0,
            "changed_from_unary": 0,
            "energy_gain_sum": 0.0,
            "selected_unary_energy_sum": 0.0,
            "selected_pairwise_energy_sum": 0.0,
            "positive_energy_gain_pairs": 0,
            "edge_count_sum": 0,
            "candidate_count_sum": 0,
            "selected_attention_rank_sum": 0.0,
            "selected_pairwise_relation_contribution_sum": 0.0,
            "native_injected_candidate_count": 0,
            "native_fallback_count": 0,
            "topk_hits": {
                name: {f"@{k}": 0 for k in (1, 3, 5, 10, 20)}
                for name in ("attention", "descriptor_unary", "fused_unary", "relational_graph")
            },
        }

    def _accumulate(stats: dict[str, object], record: dict[str, object]) -> None:
        summary = record.get("summary", {})
        points = record.get("points", [])
        if not isinstance(summary, dict) or not isinstance(points, list):
            return
        stats["pair_count"] += 1
        energy_gain = float(summary.get("energy_gain", 0.0))
        stats["energy_gain_sum"] += energy_gain
        stats["selected_unary_energy_sum"] += float(
            summary.get("selected_unary_energy", 0.0)
        )
        stats["selected_pairwise_energy_sum"] += float(
            summary.get("selected_pairwise_energy", 0.0)
        )
        stats["positive_energy_gain_pairs"] += int(energy_gain > 1e-8)
        stats["edge_count_sum"] += int(summary.get("edge_count", 0))
        stats["native_injected_candidate_count"] += int(
            summary.get("native_injected_candidate_count", 0)
        )
        stats["native_fallback_count"] += int(summary.get("native_fallback_count", 0))
        for point in points:
            if not isinstance(point, dict):
                continue
            stats["point_count"] += 1
            baseline_hit = bool(point.get("baseline_pck_hit", False))
            method_hit = bool(point.get("method_pck_hit", False))
            stats["baseline_correct"] += int(baseline_hit)
            stats["method_correct"] += int(method_hit)
            stats["rescued"] += int(method_hit and not baseline_hit)
            stats["harmed"] += int(baseline_hit and not method_hit)
            stats["changed_from_unary"] += int(
                int(point.get("selected_state", 0)) != int(point.get("unary_state", 0))
            )
            stats["candidate_count_sum"] += int(point.get("candidate_count", 0))
            stats["selected_attention_rank_sum"] += float(
                point.get("selected_attention_rank", 0.0)
            )
            stats["selected_pairwise_relation_contribution_sum"] += float(
                point.get("selected_pairwise_relation_contribution", 0.0)
            )
            topk = point.get("topk_hits", {})
            if isinstance(topk, dict):
                for score_name, score_hits in topk.items():
                    if score_name not in stats["topk_hits"] or not isinstance(score_hits, dict):
                        continue
                    for key, hit in score_hits.items():
                        if key in stats["topk_hits"][score_name]:
                            stats["topk_hits"][score_name][key] += int(bool(hit))

    def _finalize(stats: dict[str, object]) -> dict[str, object]:
        pairs = int(stats["pair_count"])
        points = int(stats["point_count"])
        harmed = int(stats["harmed"])
        finalized = dict(stats)
        finalized["baseline_pck"] = float(stats["baseline_correct"]) / max(1, points)
        finalized["method_pck"] = float(stats["method_correct"]) / max(1, points)
        finalized["point_gain"] = (
            float(stats["method_correct"] - stats["baseline_correct"]) / max(1, points)
        )
        finalized["improvement_harm_ratio"] = (
            float(stats["rescued"]) / harmed if harmed else None
        )
        finalized["mean_energy_gain"] = float(stats["energy_gain_sum"]) / max(1, pairs)
        finalized["mean_selected_unary_energy"] = (
            float(stats["selected_unary_energy_sum"]) / max(1, pairs)
        )
        finalized["mean_selected_pairwise_energy"] = (
            float(stats["selected_pairwise_energy_sum"]) / max(1, pairs)
        )
        finalized["positive_energy_gain_pair_rate"] = (
            float(stats["positive_energy_gain_pairs"]) / max(1, pairs)
        )
        finalized["mean_edge_count"] = float(stats["edge_count_sum"]) / max(1, pairs)
        finalized["mean_candidate_count"] = (
            float(stats["candidate_count_sum"]) / max(1, points)
        )
        finalized["mean_selected_attention_rank"] = (
            float(stats["selected_attention_rank_sum"]) / max(1, points)
        )
        finalized["mean_selected_pairwise_relation_contribution"] = (
            float(stats["selected_pairwise_relation_contribution_sum"]) / max(1, points)
        )
        finalized["changed_from_unary_rate"] = (
            float(stats["changed_from_unary"]) / max(1, points)
        )
        finalized["topk_hit_rates"] = {
            name: {key: float(count) / max(1, points) for key, count in hits.items()}
            for name, hits in stats["topk_hits"].items()
        }
        return finalized

    all_stats = _empty()
    category_stats: dict[str, dict[str, object]] = {}
    grouped_stats = {name: _empty() for name in _ATTENTION_GRAPH_CATEGORY_GROUPS}
    for record in pair_records:
        category = str(record.get("category", "unknown"))
        category_stats.setdefault(category, _empty())
        _accumulate(all_stats, record)
        _accumulate(category_stats[category], record)
        for group_name, categories in _ATTENTION_GRAPH_CATEGORY_GROUPS.items():
            if category in categories:
                _accumulate(grouped_stats[group_name], record)
                break
    return {
        "all": _finalize(all_stats),
        "groups": {name: _finalize(stats) for name, stats in grouped_stats.items()},
        "categories": {name: _finalize(stats) for name, stats in category_stats.items()},
        "category_group_membership": {
            name: sorted(categories) for name, categories in _ATTENTION_GRAPH_CATEGORY_GROUPS.items()
        },
        "mechanism_checks": {
            "candidate_source": "mutual_cross_attention_topk_only",
            "native_candidate_injection_expected": 0,
            "native_fallback_expected": 0,
            "gt_used_for_inference": False,
        },
    }


def _summarize_dense_partial_graph_audits(
    pair_records: list[dict[str, object]],
) -> dict[str, object]:
    score_names = (
        "attention",
        "dense_relation",
        "dense_graph_belief",
        "dense_partial_assignment",
    )

    def _empty() -> dict[str, object]:
        return {
            "pair_count": 0,
            "point_count": 0,
            "baseline_correct": 0,
            "attention_correct": 0,
            "method_correct": 0,
            "rescued_vs_baseline": 0,
            "harmed_vs_baseline": 0,
            "recovered_attention_errors": 0,
            "harmed_attention_correct": 0,
            "changed_from_attention": 0,
            "query_dustbin_count": 0,
            "required_source_dustbin_count": 0,
            "dense_dustbin_count_sum": 0,
            "matched_real_count_sum": 0,
            "dense_source_count_sum": 0,
            "unconstrained_collision_count_sum": 0,
            "partial_assignment_collision_count_sum": 0,
            "native_injected_candidate_count": 0,
            "native_fallback_count": 0,
            "gt_used_for_inference_count": 0,
            "topk_hits": {
                name: {f"@{k}": 0 for k in (1, 3, 5, 10, 20)}
                for name in score_names
            },
        }

    def _accumulate(stats: dict[str, object], record: dict[str, object]) -> None:
        pair_summary = record.get("summary", {})
        points = record.get("points", [])
        if not isinstance(pair_summary, dict) or not isinstance(points, list):
            return
        stats["pair_count"] += 1
        stats["dense_dustbin_count_sum"] += int(pair_summary.get("dustbin_count", 0))
        stats["matched_real_count_sum"] += int(pair_summary.get("matched_real_count", 0))
        stats["dense_source_count_sum"] += int(pair_summary.get("source_node_count", 0))
        stats["unconstrained_collision_count_sum"] += int(
            pair_summary.get("unconstrained_collision_count", 0)
        )
        stats["partial_assignment_collision_count_sum"] += int(
            pair_summary.get("partial_assignment_collision_count", 0)
        )
        stats["native_injected_candidate_count"] += int(
            pair_summary.get("native_candidate_injected_count", 0)
        )
        stats["native_fallback_count"] += int(
            pair_summary.get("native_fallback_count", 0)
        )
        stats["gt_used_for_inference_count"] += int(
            bool(pair_summary.get("gt_used_for_inference", False))
        )
        stats["required_source_dustbin_count"] += int(
            pair_summary.get("required_source_dustbin_count", 0)
        )
        for point in points:
            if not isinstance(point, dict):
                continue
            stats["point_count"] += 1
            baseline_hit = bool(point.get("baseline_pck_hit", False))
            method_hit = bool(point.get("method_pck_hit", False))
            attention_hit = bool(
                point.get("topk_hits", {}).get("attention", {}).get("@1", False)
            )
            stats["baseline_correct"] += int(baseline_hit)
            stats["attention_correct"] += int(attention_hit)
            stats["method_correct"] += int(method_hit)
            stats["rescued_vs_baseline"] += int(method_hit and not baseline_hit)
            stats["harmed_vs_baseline"] += int(baseline_hit and not method_hit)
            stats["recovered_attention_errors"] += int(method_hit and not attention_hit)
            stats["harmed_attention_correct"] += int(attention_hit and not method_hit)
            stats["changed_from_attention"] += int(
                bool(point.get("final_changed_from_attention", False))
            )
            stats["query_dustbin_count"] += int(
                bool(point.get("solver_assigned_dustbin", False))
            )
            stats["native_injected_candidate_count"] += int(
                bool(point.get("native_candidate_injected", False))
            )
            stats["native_fallback_count"] += int(
                bool(point.get("native_fallback_used", False))
            )
            stats["gt_used_for_inference_count"] += int(
                bool(point.get("gt_used_for_inference", False))
            )
            topk = point.get("topk_hits", {})
            if isinstance(topk, dict):
                for score_name in score_names:
                    score_hits = topk.get(score_name, {})
                    if not isinstance(score_hits, dict):
                        continue
                    for key in stats["topk_hits"][score_name]:
                        stats["topk_hits"][score_name][key] += int(
                            bool(score_hits.get(key, False))
                        )

    def _finalize(stats: dict[str, object]) -> dict[str, object]:
        points = int(stats["point_count"])
        pairs = int(stats["pair_count"])
        source_nodes = int(stats["dense_source_count_sum"])
        finalized = dict(stats)
        for key in ("baseline_correct", "attention_correct", "method_correct"):
            finalized[key.replace("correct", "pck")] = float(stats[key]) / max(1, points)
        finalized["point_gain_vs_baseline"] = float(
            stats["method_correct"] - stats["baseline_correct"]
        ) / max(1, points)
        finalized["point_gain_vs_attention"] = float(
            stats["method_correct"] - stats["attention_correct"]
        ) / max(1, points)
        finalized["net_recovery_vs_attention"] = int(
            stats["recovered_attention_errors"] - stats["harmed_attention_correct"]
        )
        finalized["changed_from_attention_rate"] = float(
            stats["changed_from_attention"]
        ) / max(1, points)
        finalized["query_dustbin_rate"] = float(stats["query_dustbin_count"]) / max(1, points)
        finalized["dense_dustbin_rate"] = float(stats["dense_dustbin_count_sum"]) / max(
            1, source_nodes
        )
        finalized["dense_matched_real_rate"] = float(stats["matched_real_count_sum"]) / max(
            1, source_nodes
        )
        finalized["mean_unconstrained_collisions_per_pair"] = float(
            stats["unconstrained_collision_count_sum"]
        ) / max(1, pairs)
        finalized["topk_hit_rates"] = {
            name: {key: float(count) / max(1, points) for key, count in hits.items()}
            for name, hits in stats["topk_hits"].items()
        }
        return finalized

    all_stats = _empty()
    category_stats: dict[str, dict[str, object]] = {}
    grouped_stats = {name: _empty() for name in _ATTENTION_GRAPH_CATEGORY_GROUPS}
    for record in pair_records:
        category = str(record.get("category", "unknown"))
        category_stats.setdefault(category, _empty())
        _accumulate(all_stats, record)
        _accumulate(category_stats[category], record)
        for group_name, categories in _ATTENTION_GRAPH_CATEGORY_GROUPS.items():
            if category in categories:
                _accumulate(grouped_stats[group_name], record)
                break
    return {
        "all": _finalize(all_stats),
        "groups": {name: _finalize(stats) for name, stats in grouped_stats.items()},
        "categories": {name: _finalize(stats) for name, stats in category_stats.items()},
        "category_group_membership": {
            name: sorted(categories)
            for name, categories in _ATTENTION_GRAPH_CATEGORY_GROUPS.items()
        },
        "mechanism_checks": {
            "candidate_source": "mutual_cross_attention_topk_only",
            "source_graph": "all_dense_flux_tokens_local_edges",
            "descriptor_unary_used": False,
            "spatial_edge_used": False,
            "target_capacity": 1,
            "dustbin_reserved": True,
            "native_candidate_injection_expected": 0,
            "native_fallback_expected": 0,
            "gt_used_for_inference_expected": 0,
        },
    }


def _summarize_expert_hypothesis_audits(
    pair_records: list[dict[str, object]],
) -> dict[str, object]:
    score_names = (
        "attention",
        "mean_expert_hypothesis",
        "pair_head_support",
        "pair_head_identity",
        "pair_head_hypothesis",
        "pair_expert_hypothesis",
        "point_head_hypothesis",
    )

    def _empty() -> dict[str, object]:
        return {
            "pair_count": 0,
            "point_count": 0,
            "baseline_correct": 0,
            "attention_correct": 0,
            "method_correct": 0,
            "method_audit_mismatch_count": 0,
            "rescued_vs_baseline": 0,
            "harmed_vs_baseline": 0,
            "recovered_attention_errors": 0,
            "harmed_attention_correct": 0,
            "changed_from_attention": 0,
            "selected_attention_rank_sum": 0.0,
            "selected_head_agreement_sum": 0.0,
            "selected_head_agreement_margin_sum": 0.0,
            "selected_head_histogram": {},
            "selected_expert_histogram": {},
            "point_head_histogram": {},
            "native_injected_candidate_count": 0,
            "native_fallback_count": 0,
            "gt_used_for_inference_count": 0,
            "topk_hits": {
                name: {f"@{k}": 0 for k in (1, 3, 5, 10, 20)}
                for name in score_names
            },
        }

    def _increment_histogram(histogram: dict[str, int], key: object) -> None:
        label = str(key)
        histogram[label] = int(histogram.get(label, 0)) + 1

    def _accumulate(stats: dict[str, object], record: dict[str, object]) -> None:
        pair_summary = record.get("summary", {})
        points = record.get("points", [])
        if not isinstance(pair_summary, dict) or not isinstance(points, list):
            return
        stats["pair_count"] += 1
        stats["selected_head_agreement_sum"] += float(
            pair_summary.get("selected_head_agreement", 0.0)
        )
        stats["selected_head_agreement_margin_sum"] += float(
            pair_summary.get("selected_head_agreement_margin", 0.0)
        )
        _increment_histogram(
            stats["selected_head_histogram"],
            pair_summary.get("selected_head", "missing"),
        )
        selected_expert = pair_summary.get("selected_expert", {})
        if isinstance(selected_expert, dict):
            expert_label = (
                f"member{selected_expert.get('member', 'missing')}_"
                f"head{selected_expert.get('head', 'missing')}"
            )
        else:
            expert_label = "missing"
        _increment_histogram(stats["selected_expert_histogram"], expert_label)
        stats["native_injected_candidate_count"] += int(
            pair_summary.get("native_candidate_injected_count", 0)
        )
        stats["native_fallback_count"] += int(
            pair_summary.get("native_fallback_count", 0)
        )
        stats["gt_used_for_inference_count"] += int(
            bool(pair_summary.get("gt_used_for_inference", False))
        )
        for point in points:
            if not isinstance(point, dict):
                continue
            stats["point_count"] += 1
            baseline_hit = bool(point.get("baseline_pck_hit", False))
            method_hit = bool(point.get("method_pck_hit", False))
            topk = point.get("topk_hits", {})
            if not isinstance(topk, dict):
                topk = {}
            attention_hit = bool(topk.get("attention", {}).get("@1", False))
            method_audit_hit = bool(
                topk.get("pair_head_hypothesis", {}).get("@1", False)
            )
            stats["baseline_correct"] += int(baseline_hit)
            stats["attention_correct"] += int(attention_hit)
            stats["method_correct"] += int(method_hit)
            stats["method_audit_mismatch_count"] += int(
                method_hit != method_audit_hit
            )
            stats["rescued_vs_baseline"] += int(method_hit and not baseline_hit)
            stats["harmed_vs_baseline"] += int(baseline_hit and not method_hit)
            stats["recovered_attention_errors"] += int(
                method_hit and not attention_hit
            )
            stats["harmed_attention_correct"] += int(
                attention_hit and not method_hit
            )
            stats["changed_from_attention"] += int(
                bool(point.get("final_changed_from_attention", False))
            )
            stats["selected_attention_rank_sum"] += float(
                point.get("selected_attention_rank", 0.0)
            )
            _increment_histogram(
                stats["point_head_histogram"],
                point.get("selected_point_head", "missing"),
            )
            stats["native_injected_candidate_count"] += int(
                bool(point.get("native_candidate_injected", False))
            )
            stats["native_fallback_count"] += int(
                bool(point.get("native_fallback_used", False))
            )
            stats["gt_used_for_inference_count"] += int(
                bool(point.get("gt_used_for_inference", False))
            )
            for score_name in score_names:
                score_hits = topk.get(score_name, {})
                if not isinstance(score_hits, dict):
                    continue
                for key in stats["topk_hits"][score_name]:
                    stats["topk_hits"][score_name][key] += int(
                        bool(score_hits.get(key, False))
                    )

    def _finalize(stats: dict[str, object]) -> dict[str, object]:
        points = int(stats["point_count"])
        pairs = int(stats["pair_count"])
        finalized = dict(stats)
        for key in ("baseline_correct", "attention_correct", "method_correct"):
            finalized[key.replace("correct", "pck")] = float(stats[key]) / max(
                1, points
            )
        finalized["point_gain_vs_baseline"] = float(
            stats["method_correct"] - stats["baseline_correct"]
        ) / max(1, points)
        finalized["point_gain_vs_attention"] = float(
            stats["method_correct"] - stats["attention_correct"]
        ) / max(1, points)
        finalized["net_recovery_vs_attention"] = int(
            stats["recovered_attention_errors"] - stats["harmed_attention_correct"]
        )
        finalized["changed_from_attention_rate"] = float(
            stats["changed_from_attention"]
        ) / max(1, points)
        finalized["selected_attention_rank_mean"] = float(
            stats["selected_attention_rank_sum"]
        ) / max(1, points)
        finalized["selected_head_agreement_mean"] = float(
            stats["selected_head_agreement_sum"]
        ) / max(1, pairs)
        finalized["selected_head_agreement_margin_mean"] = float(
            stats["selected_head_agreement_margin_sum"]
        ) / max(1, pairs)
        finalized["topk_hit_rates"] = {
            name: {key: float(count) / max(1, points) for key, count in hits.items()}
            for name, hits in stats["topk_hits"].items()
        }
        return finalized

    all_stats = _empty()
    category_stats: dict[str, dict[str, object]] = {}
    grouped_stats = {name: _empty() for name in _ATTENTION_GRAPH_CATEGORY_GROUPS}
    for record in pair_records:
        category = str(record.get("category", "unknown"))
        category_stats.setdefault(category, _empty())
        _accumulate(all_stats, record)
        _accumulate(category_stats[category], record)
        for group_name, categories in _ATTENTION_GRAPH_CATEGORY_GROUPS.items():
            if category in categories:
                _accumulate(grouped_stats[group_name], record)
                break
    return {
        "all": _finalize(all_stats),
        "groups": {name: _finalize(stats) for name, stats in grouped_stats.items()},
        "categories": {name: _finalize(stats) for name, stats in category_stats.items()},
        "category_group_membership": {
            name: sorted(categories)
            for name, categories in _ATTENTION_GRAPH_CATEGORY_GROUPS.items()
        },
        "mechanism_checks": {
            "candidate_source": "exact_mutual_cross_attention_topk_only",
            "formal_method_signal": "pair_head_hypothesis",
            "expert_axes_preserved_until_candidate_scoring": True,
            "candidate_value_aggregation_used": False,
            "symmetric_candidate_identity_used": True,
            "native_candidate_injection_expected": 0,
            "native_fallback_expected": 0,
            "gt_used_for_inference_expected": 0,
        },
    }


def _summarize_pre_softmax_identity_audits(
    pair_records: list[dict[str, object]],
) -> dict[str, object]:
    """Summarize the direct train-free pre-softmax identity experiment."""

    def _collect(records: list[dict[str, object]]) -> dict[str, object]:
        points = [point for record in records for point in record.get("points", []) if isinstance(point, dict)]
        both_wrong = [point for point in points if bool(point.get("both_wrong_top20_hit", False))]
        def _stats(rows: list[dict[str, object]]) -> dict[str, object]:
            if not rows:
                return {"points": 0}
            ranks = [
                int(point["gt_ranks"]["combined"])
                for point in rows
                if isinstance(point.get("gt_ranks"), dict)
                and point["gt_ranks"].get("combined") is not None
            ]
            return {
                "points": len(rows),
                "baseline_pck": float(sum(bool(point.get("baseline_pck_hit", False)) for point in rows)) / len(rows),
                "method_pck": float(sum(bool(point.get("method_pck_hit", False)) for point in rows)) / len(rows),
                "candidate_pck_hit_fraction_mean": float(np.mean([float(point.get("candidate_pck_hit_fraction", 0.0)) for point in rows])),
                "combined_top1_rank_median": float(np.median(ranks)) if ranks else None,
                "combined_top1_hits": int(sum(rank == 1 for rank in ranks)),
                "qk_channelwise_top1_hits": int(sum(point.get("gt_ranks", {}).get("qk_channelwise") == 1 for point in rows)),
                "value_residual_top1_hits": int(sum(point.get("gt_ranks", {}).get("value_residual") == 1 for point in rows)),
                "local_relation_top1_hits": int(sum(point.get("gt_ranks", {}).get("local_relation") == 1 for point in rows)),
                "native_local_top1_hits": int(sum(point.get("gt_ranks", {}).get("native_local") == 1 for point in rows)),
            }
        return {
            "points": len(points),
            "pairs": len(records),
            "all": _stats(points),
            "both_wrong_top20_hit": _stats(both_wrong),
            "candidate_missing_gt_count": int(sum(bool(point.get("candidate_missing_gt", False)) for point in points)),
            "native_fallback_count": 0,
            "native_candidate_injected_count": 0,
            "gt_used_for_inference_count": 0,
        }

    all_summary = _collect(pair_records)
    categories: dict[str, list[dict[str, object]]] = {}
    for record in pair_records:
        categories.setdefault(str(record.get("category", "unknown")), []).append(record)
    return {
        "all": all_summary,
        "categories": {name: _collect(records) for name, records in categories.items()},
        "decision_rule": {
            "primary_group": "both_wrong_top20_hit",
            "candidate_pool": "exact_mutual_cross_attention_topk_only",
            "random_candidate_hit_reference": "per-point candidate PCK fraction",
            "attention_probability_used_as_identity_score": False,
            "native_fallback_used": False,
            "train_free": True,
        },
    }


def _reset_all_rng(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))



def _detach_cpu_nested(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, list):
        return [_detach_cpu_nested(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_detach_cpu_nested(item) for item in value)
    return value


def _nested_tensor_nbytes(value):
    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(_nested_tensor_nbytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_nested_tensor_nbytes(item) for item in value)
    return 0


def _safe_cache_name(category, image_name, key):
    payload = "|".join(str(item) for item in key)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    stem = os.path.splitext(os.path.basename(image_name))[0]
    safe_stem = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem)[:48]
    return os.path.join(str(category), f"{safe_stem}_{digest}.pth")


def _fjsar_disk_cache_file(cache_root, category, image_name, key):
    if not cache_root:
        return None
    return os.path.join(cache_root, _safe_cache_name(category, image_name, key))


def _load_fjsar_disk_entry(path, feature_block, caption, args):
    if not path or not os.path.exists(path):
        return None
    try:
        entry = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        entry = torch.load(path, map_location="cpu")
    except Exception as exc:
        print(f"Warning: failed to load FJSAR disk cache {path}: {exc}")
        return None
    start_block = replay_start_block_index(feature_block)
    if not is_valid_flux_replay_entry(entry, start_block):
        print(f"Warning: ignoring invalid FJSAR disk cache entry: {path}")
        return None
    metadata = entry.get("metadata", {})
    expected = {
        "feature_block_index": int(feature_block),
        "start_block_index": int(start_block),
        "ensemble_size": int(args.ensemble_size),
        "timestep": int(args.t),
        "shared_noise": bool(getattr(args, "fjsar_shared_noise", False)),
        "protocol_seed": (
            int(_fjsar_protocol_seed(args))
            if getattr(args, "fjsar_shared_noise", False) else -1
        ),
        "caption_sha256": hashlib.sha256(caption.encode("utf-8")).hexdigest(),
        "cache_version": 4,
    }
    multiblock_indices = _fjsar_multilayer_blocks(args)
    if multiblock_indices:
        expected["multilayer_blocks"] = [int(block) for block in multiblock_indices]
    trajectory_blocks = _fjsar_trajectory_blocks(args)
    if trajectory_blocks:
        expected["trajectory_blocks"] = [int(block) for block in trajectory_blocks]
    for key, value in expected.items():
        if metadata.get(key) != value:
            print(f"Warning: ignoring stale FJSAR disk cache {path}: {key} mismatch")
            return None
    entry["metadata"] = {**metadata, "cached": True, "disk_cache_path": path}
    return entry


def _save_fjsar_disk_entry(path, entry):
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp-{os.getpid()}"
    torch.save(entry, tmp_path)
    os.replace(tmp_path, path)


def _disk_free_bytes(path):
    target = path
    while target and not os.path.exists(target):
        parent = os.path.dirname(target)
        if parent == target:
            break
        target = parent
    target = target if target else "."
    return int(shutil.disk_usage(target).free)


class _FjsarMemoryCache:
    """Bounded CPU-memory cache for on-demand FJSAR image entries."""

    def __init__(self, max_bytes):
        self.max_bytes = max(0, int(max_bytes))
        self.current_bytes = 0
        self._entries = OrderedDict()

    def get(self, key):
        cached = self._entries.get(key)
        if cached is None:
            return None
        self._entries.move_to_end(key)
        return cached[0]

    def put(self, key, value):
        if self.max_bytes <= 0:
            return value
        value_bytes = _nested_tensor_nbytes(value)
        if value_bytes <= 0 or value_bytes > self.max_bytes:
            return value
        while self._entries and self.current_bytes + value_bytes > self.max_bytes:
            _, (_, evicted_bytes) = self._entries.popitem(last=False)
            self.current_bytes -= evicted_bytes
        self._entries[key] = (value, value_bytes)
        self.current_bytes += value_bytes
        return value

    def stats(self):
        return {
            "entries": len(self._entries),
            "bytes": self.current_bytes,
            "max_bytes": self.max_bytes,
        }


def _extract_flux_fjsar_entry(
    dataset_path,
    category,
    image_name,
    caption,
    args,
    featurizer,
    capture,
    *,
    horizontal_flip=False,
    image_transform=None,
):
    """Extract one aligned FJSAR entry for the active benchmark subset.

    Only images actually reached by the evaluation loop are materialized.  The
    entry is kept in memory for immediate evaluation and is not written to disk.
    """

    image_path = os.path.join(dataset_path, "JPEGImages", category, image_name)
    image_tensor, pixel_h, pixel_w = load_flux_image_tensor(
        image_path,
        args.img_size,
        horizontal_flip=horizontal_flip,
        image_transform=image_transform,
    )
    grid_h, grid_w = pixel_h // 16, pixel_w // 16

    if getattr(args, "fjsar_shared_noise", False):
        _reset_all_rng(_fjsar_protocol_seed(args))
    feature_block = int(args.k[0] if isinstance(args.k, list) else args.k)
    multiblock_indices = _fjsar_multilayer_blocks(args)
    trajectory_blocks = _fjsar_trajectory_blocks(args)
    extraction_blocks = tuple(sorted({feature_block, *multiblock_indices, *trajectory_blocks}))
    multiblock_features = None
    multiblock_ada = None
    if len(extraction_blocks) > 1:
        multiblock_output = featurizer.forward_multi_block(
            args,
            image_tensor,
            caption=caption,
            category=category,
            timestep=args.t,
            block_indices=extraction_blocks,
            ensemble_size=args.ensemble_size,
        )
        if feature_block not in multiblock_output:
            raise RuntimeError(f"multi-block FLUX extraction did not return block {feature_block}")
        feature, ada = multiblock_output[feature_block]
        if multiblock_indices:
            multiblock_features = {
                str(block): multiblock_output[int(block)][0].detach().cpu()
                for block in multiblock_indices
            }
            multiblock_ada = {
                str(block): _detach_cpu_nested(multiblock_output[int(block)][1])
                for block in multiblock_indices
            }
    else:
        feature, ada = featurizer.forward(
            args,
            image_tensor,
            caption=caption,
            category=category,
            timestep=args.t,
            block_idx=args.k,
            ensemble_size=args.ensemble_size,
        )
    actual_grid = (int(feature.shape[-2]), int(feature.shape[-1]))
    if actual_grid != (grid_h, grid_w):
        raise RuntimeError(
            f"FJSAR extraction grid for {category}/{image_name} is {actual_grid}, "
            f"expected {(grid_h, grid_w)}"
        )
    trajectory_replay_states = None
    if trajectory_blocks:
        if not isinstance(capture, FluxMultiPreBlockCapture):
            raise RuntimeError("trajectory extraction requires FluxMultiPreBlockCapture")
        captured = capture.consume_all(grid_h, grid_w)
        trajectory_replay_states = {}
        for block in trajectory_blocks:
            start = replay_start_block_index(int(block))
            state_dict = captured.get(str(start))
            if state_dict is None:
                raise RuntimeError(f"trajectory capture did not return pre-block state {start}")
            trajectory_replay_states[str(int(block))] = state_dict
        replay_state_dict = trajectory_replay_states[str(feature_block)]
        ensemble_size = int(replay_state_dict["ensemble_size"])
    else:
        state_obj = capture.consume(grid_h, grid_w)
        replay_state_dict = state_obj.to_dict()
        ensemble_size = state_obj.ensemble_size
    if ensemble_size != int(args.ensemble_size):
        raise RuntimeError(
            f"captured {ensemble_size} ensemble members for {image_name}; "
            f"expected {args.ensemble_size}"
        )
    entry = {
        "replay_state": replay_state_dict,
        "feature": feature.detach().cpu(),
        "ada": _detach_cpu_nested(ada),
        "metadata": {
            "feature_block_index": feature_block,
            "start_block_index": replay_start_block_index(feature_block),
            "ensemble_size": int(args.ensemble_size),
            "timestep": int(args.t),
            "shared_noise": bool(getattr(args, "fjsar_shared_noise", False)),
            "protocol_seed": (
                int(_fjsar_protocol_seed(args))
                if getattr(args, "fjsar_shared_noise", False) else -1
            ),
            "caption_sha256": hashlib.sha256(caption.encode("utf-8")).hexdigest(),
            "cache_version": 4,
            "cached": False,
        },
    }
    if multiblock_indices:
        entry["multiblock_features"] = multiblock_features
        entry["multiblock_ada"] = multiblock_ada
        entry["metadata"]["multilayer_blocks"] = [int(block) for block in multiblock_indices]
    if trajectory_blocks:
        entry["trajectory_replay_states"] = trajectory_replay_states
        entry["metadata"]["trajectory_blocks"] = [int(block) for block in trajectory_blocks]
    if bool(horizontal_flip):
        entry["metadata"]["horizontal_flip"] = True
    return entry


def _get_flux_fjsar_entry(
    dataset_path,
    category,
    image_name,
    caption,
    args,
    featurizer,
    capture,
    memory_cache,
):
    feature_block = int(args.k[0] if isinstance(args.k, list) else args.k)
    key = (
        category,
        image_name,
        int(args.img_size[0]),
        int(args.t),
        feature_block,
        int(args.ensemble_size),
        bool(args.cd),
        bool(getattr(args, "fjsar_shared_noise", False)),
        hashlib.sha256(caption.encode("utf-8")).hexdigest(),
    )
    multiblock_indices = _fjsar_multilayer_blocks(args)
    if multiblock_indices:
        key = (*key, multiblock_indices)
    trajectory_blocks = _fjsar_trajectory_blocks(args)
    if trajectory_blocks:
        key = (*key, ("trajectory", trajectory_blocks))
    cached = memory_cache.get(key) if memory_cache is not None else None
    if cached is not None:
        return cached
    disk_path = _fjsar_disk_cache_file(
        getattr(args, "fjsar_disk_cache_path", ""),
        category,
        image_name,
        key,
    )
    disk_entry = _load_fjsar_disk_entry(disk_path, feature_block, caption, args)
    if disk_entry is not None:
        if memory_cache is not None:
            memory_cache.put(key, disk_entry)
        return disk_entry
    if getattr(args, "fjsar_require_disk_cache", False):
        if not disk_path:
            raise FileNotFoundError(
                "--fjsar_require_disk_cache needs --fjsar_disk_cache_path; no disk cache path was configured"
            )
        raise FileNotFoundError(
            "Required FJSAR disk cache entry is missing or stale; "
            f"category={category}, image={image_name}, path={disk_path}"
        )
    entry = _extract_flux_fjsar_entry(
        dataset_path,
        category,
        image_name,
        caption,
        args,
        featurizer,
        capture,
    )
    if disk_path:
        entry_bytes = _nested_tensor_nbytes(entry)
        free_bytes = _disk_free_bytes(os.path.dirname(disk_path))
        min_free_bytes = int(float(getattr(args, "fjsar_disk_cache_min_free_gb", 0.0)) * (1024 ** 3))
        if free_bytes - entry_bytes >= min_free_bytes:
            _save_fjsar_disk_entry(disk_path, entry)
        else:
            print(
                "Warning: skip FJSAR disk cache write; "
                f"need {entry_bytes / (1024 ** 2):.1f} MiB, "
                f"free {free_bytes / (1024 ** 3):.2f} GiB, "
                f"reserved {min_free_bytes / (1024 ** 3):.2f} GiB"
            )
    if memory_cache is not None:
        memory_cache.put(key, entry)
    return entry

def _ensure_flux_fjsar_cache(
    dataset_path,
    save_path,
    test_path,
    all_cats,
    cat2json,
    args,
    device,
):
    """Deprecated no-op kept for compatibility with older scripts."""

    del dataset_path, save_path, test_path, all_cats, cat2json, args, device
    print("FJSAR cache building is disabled; aligned entries are extracted on demand.")



def _load_trusted_cache(path):
    """Load the user's local tensor cache without PyTorch's default warning."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # Compatibility with older PyTorch versions that predate weights_only.
        return torch.load(path, map_location="cpu")


def _merge_counts(target, source):
    for key, value in source.items():
        target[key] = target.get(key, 0) + int(value)


def _summarize_candidate_descriptor_audit(records):
    signals = (
        "attention",
        "native_descriptor",
        "local_self_similarity",
        "attention_jacobian",
    )
    groups = {
        "all": lambda row: True,
        "oracle_gap": lambda row: bool(row.get("oracle_gap_case")),
        "attention_harms_native": lambda row: bool(row.get("attention_harms_native_case")),
        "attention_rescues_native": lambda row: bool(row.get("attention_rescues_native_case")),
    }

    def _median(values):
        if not values:
            return None
        values = sorted(values)
        middle = len(values) // 2
        if len(values) % 2:
            return float(values[middle])
        return float((values[middle - 1] + values[middle]) / 2.0)

    summary = {}
    for group_name, predicate in groups.items():
        rows = [
            row for row in records
            if predicate(row) and isinstance(row.get("candidate_descriptor_audit"), dict)
        ]
        group = {
            "points": len(rows),
            "gt_exact_in_attention_proposals": sum(
                1 for row in rows
                if row["candidate_descriptor_audit"].get("gt_exact_in_proposals")
            ),
            "signals": {},
        }
        for signal in signals:
            ranks = []
            proposal_hit_gaps = []
            gt_exact_gaps = []
            for row in rows:
                audit = row["candidate_descriptor_audit"]
                rank = audit.get("proposal_only_ranks", audit.get("ranks", {})).get(signal)
                if rank is not None:
                    ranks.append(int(rank))
                proposal_hit_gap = audit.get("proposal_hit_score_gaps", audit.get("score_gaps", {})).get(
                    f"{signal}_attention_top1_minus_best_pck_hit_proposal"
                )
                if proposal_hit_gap is not None:
                    proposal_hit_gaps.append(float(proposal_hit_gap))
                gt_exact_gap = audit.get("gt_exact_score_gaps", {}).get(
                    f"{signal}_attention_top1_minus_gt_exact"
                )
                if gt_exact_gap is not None:
                    gt_exact_gaps.append(float(gt_exact_gap))
            group["signals"][signal] = {
                "ranked_points": len(ranks),
                "proposal_pck_hit_rank_median": _median(ranks),
                "proposal_pck_hit_at_1": sum(1 for rank in ranks if rank <= 1),
                "proposal_pck_hit_at_3": sum(1 for rank in ranks if rank <= 3),
                "proposal_pck_hit_at_5": sum(1 for rank in ranks if rank <= 5),
                "proposal_pck_hit_at_10": sum(1 for rank in ranks if rank <= 10),
                "attention_top1_minus_best_pck_hit_proposal_gap_median": _median(proposal_hit_gaps),
                "attention_top1_beats_best_pck_hit_proposal": sum(
                    1 for gap in proposal_hit_gaps if gap > 0.0
                ),
                "best_pck_hit_proposal_beats_attention_top1": sum(
                    1 for gap in proposal_hit_gaps if gap < 0.0
                ),
                "tied_attention_top1_best_pck_hit_proposal": sum(
                    1 for gap in proposal_hit_gaps if gap == 0.0
                ),
                "attention_top1_minus_gt_exact_gap_median": _median(gt_exact_gaps),
                "gt_exact_beats_attention_top1": sum(1 for gap in gt_exact_gaps if gap < 0.0),
            }
        summary[group_name] = group
    return summary


def _summarize_attention_flow_audit(records):
    return _summarize_ranked_candidate_score_audit(records, "attention_flow_audit")


def _summarize_attention_kernel_audit(records):
    groups = {
        "all": lambda row: True,
        "oracle_gap": lambda row: bool(row.get("oracle_gap_case")),
        "attention_harms_native": lambda row: bool(row.get("attention_harms_native_case")),
        "attention_rescues_native": lambda row: bool(row.get("attention_rescues_native_case")),
    }

    score_names = []
    topk_keys = []
    for row in records:
        audit = row.get("attention_kernel_audit")
        if isinstance(audit, dict):
            score_names = list(audit.get("score_names", []))
            topk_keys = sorted(audit.get("topk_hits", {}).keys())
            if score_names and topk_keys:
                break

    summary = {}
    for group_name, predicate in groups.items():
        rows = [
            row for row in records
            if predicate(row) and isinstance(row.get("attention_kernel_audit"), dict)
        ]
        group = {"points": len(rows), "signals": {}, "transitions": {}}
        for score_name in score_names:
            signal = {"ranked_points": 0}
            for key in topk_keys:
                if not key.startswith(f"{score_name}@"):
                    continue
                hits = sum(
                    1 for row in rows
                    if bool(row["attention_kernel_audit"].get("topk_hits", {}).get(key))
                )
                signal[key.replace(f"{score_name}@", "hit_at_")] = hits
                signal[key.replace(f"{score_name}@", "hit_rate_at_")] = (
                    hits / len(rows) if rows else 0.0
                )
            ranks = [
                int(row["attention_kernel_audit"]["ranks"][score_name])
                for row in rows
                if row["attention_kernel_audit"].get("ranks", {}).get(score_name) is not None
            ]
            signal["ranked_points"] = len(ranks)
            group["signals"][score_name] = signal

        for suffix in ("@1", "@5", "@20"):
            raw_key = f"raw_attention{suffix}"
            filtered_key = f"filtered_attention{suffix}"
            if raw_key not in topk_keys or filtered_key not in topk_keys:
                continue
            raw_hits = [
                bool(row["attention_kernel_audit"].get("topk_hits", {}).get(raw_key))
                for row in rows
            ]
            filtered_hits = [
                bool(row["attention_kernel_audit"].get("topk_hits", {}).get(filtered_key))
                for row in rows
            ]
            group["transitions"][suffix] = {
                "both_hit": sum(1 for raw, filtered in zip(raw_hits, filtered_hits) if raw and filtered),
                "raw_only": sum(1 for raw, filtered in zip(raw_hits, filtered_hits) if raw and not filtered),
                "filtered_only": sum(1 for raw, filtered in zip(raw_hits, filtered_hits) if filtered and not raw),
                "both_miss": sum(1 for raw, filtered in zip(raw_hits, filtered_hits) if not raw and not filtered),
            }
        summary[group_name] = group
    return summary


def _summarize_basin_identity_audit(records):
    groups = {
        "all": lambda row: True,
        "oracle_gap": lambda row: bool(row.get("oracle_gap_case")),
        "attention_harms_native": lambda row: bool(row.get("attention_harms_native_case")),
        "attention_rescues_native": lambda row: bool(row.get("attention_rescues_native_case")),
        "native_correct_method_wrong": lambda row: bool(row.get("baseline_pck_hit"))
        and not bool(row.get("method_pck_hit")),
        "native_wrong_method_correct": lambda row: not bool(row.get("baseline_pck_hit"))
        and bool(row.get("method_pck_hit")),
    }

    score_names = []
    topk_keys = []
    for row in records:
        audit = row.get("basin_identity_audit")
        if isinstance(audit, dict):
            score_names = list(audit.get("score_names", []))
            topk_keys = sorted(audit.get("topk_hits", {}).keys())
            if score_names and topk_keys:
                break

    summary = {}
    for group_name, predicate in groups.items():
        rows = [
            row for row in records
            if predicate(row) and isinstance(row.get("basin_identity_audit"), dict)
        ]
        group = {
            "points": len(rows),
            "baseline_correct": sum(1 for row in rows if row.get("baseline_pck_hit")),
            "method_correct": sum(1 for row in rows if row.get("method_pck_hit")),
            "attention_top1_correct": sum(1 for row in rows if row.get("attention_top1_pck_hit")),
            "attention_topk_correct": sum(1 for row in rows if row.get("attention_topk_pck_hit")),
            "signals": {},
        }
        for score_name in score_names:
            ranks = [
                int(row["basin_identity_audit"]["ranks"][score_name])
                for row in rows
                if row["basin_identity_audit"].get("ranks", {}).get(score_name) is not None
            ]
            basins = [
                row["basin_identity_audit"].get("basins", {}).get(score_name, {})
                for row in rows
            ]
            attention_basin_hit = sum(
                1 for basin in basins
                if bool(basin.get("attention_basin_has_pck_hit"))
            )
            native_top1_hit = sum(
                1 for basin in basins
                if bool(basin.get("native_top1_in_basin", {}).get("pck_hit"))
            )
            attention_top1_hit = sum(
                1 for basin in basins
                if bool(basin.get("attention_top1", {}).get("pck_hit"))
            )
            signal = {
                "ranked_points": len(ranks),
                "attention_basin_pck_hit": attention_basin_hit,
                "attention_basin_pck_hit_rate": attention_basin_hit / len(rows) if rows else 0.0,
                "attention_top1_in_basin_hit": attention_top1_hit,
                "attention_top1_in_basin_hit_rate": attention_top1_hit / len(rows) if rows else 0.0,
                "native_top1_in_basin_hit": native_top1_hit,
                "native_top1_in_basin_hit_rate": native_top1_hit / len(rows) if rows else 0.0,
            }
            for key in topk_keys:
                if not key.startswith(f"{score_name}@"):
                    continue
                suffix = key.replace(f"{score_name}@", "")
                hits = sum(
                    1 for row in rows
                    if bool(row["basin_identity_audit"].get("topk_hits", {}).get(key))
                )
                signal[f"native_in_basin_hit_at_{suffix}"] = hits
                signal[f"native_in_basin_hit_rate_at_{suffix}"] = hits / len(rows) if rows else 0.0
            group["signals"][score_name] = signal
        summary[group_name] = group
    return summary


def _summarize_operator_manifold_audit(records):
    groups = {
        "all": lambda row: True,
        "oracle_gap": lambda row: bool(row.get("oracle_gap_case")),
        "attention_harms_native": lambda row: bool(row.get("attention_harms_native_case")),
        "attention_rescues_native": lambda row: bool(row.get("attention_rescues_native_case")),
        "native_correct_method_wrong": lambda row: bool(row.get("baseline_pck_hit"))
        and not bool(row.get("method_pck_hit")),
        "native_wrong_method_correct": lambda row: not bool(row.get("baseline_pck_hit"))
        and bool(row.get("method_pck_hit")),
    }

    def _median(values):
        if not values:
            return None
        values = sorted(values)
        middle = len(values) // 2
        if len(values) % 2:
            return float(values[middle])
        return float((values[middle - 1] + values[middle]) / 2.0)

    def _stats(values):
        clean = [float(value) for value in values if value is not None and np.isfinite(float(value))]
        return {
            "count": len(clean),
            "mean": float(np.mean(clean)) if clean else None,
            "median": _median(clean),
            "min": float(np.min(clean)) if clean else None,
            "max": float(np.max(clean)) if clean else None,
        }

    def _collect(rows, section, key):
        values = []
        for row in rows:
            audit = row.get("operator_manifold_audit")
            if not isinstance(audit, dict):
                continue
            payload = audit.get(section)
            if isinstance(payload, dict) and payload.get(key) is not None:
                values.append(payload.get(key))
        return values

    summary = {}
    for group_name, predicate in groups.items():
        rows = [
            row for row in records
            if predicate(row) and isinstance(row.get("operator_manifold_audit"), dict)
        ]
        group = {
            "points": len(rows),
            "baseline_correct": sum(1 for row in rows if row.get("baseline_pck_hit")),
            "method_correct": sum(1 for row in rows if row.get("method_pck_hit")),
            "attention_top1_correct": sum(1 for row in rows if row.get("attention_top1_pck_hit")),
            "attention_topk_correct": sum(1 for row in rows if row.get("attention_topk_pck_hit")),
            "source": {
                "joint_native_cosine": _stats(_collect(rows, "source", "joint_native_cosine")),
                "drift_l2_ratio": _stats(_collect(rows, "source", "drift_l2_ratio")),
                "low_cosine_lt_0_95": sum(
                    1 for value in _collect(rows, "source", "joint_native_cosine")
                    if float(value) < 0.95
                ),
                "high_drift_ratio_gt_0_25": sum(
                    1 for value in _collect(rows, "source", "drift_l2_ratio")
                    if float(value) > 0.25
                ),
            },
            "target_gt": {
                "joint_native_cosine": _stats(_collect(rows, "target_gt", "joint_native_cosine")),
                "drift_l2_ratio": _stats(_collect(rows, "target_gt", "drift_l2_ratio")),
                "low_cosine_lt_0_95": sum(
                    1 for value in _collect(rows, "target_gt", "joint_native_cosine")
                    if float(value) < 0.95
                ),
                "high_drift_ratio_gt_0_25": sum(
                    1 for value in _collect(rows, "target_gt", "drift_l2_ratio")
                    if float(value) > 0.25
                ),
            },
            "pair": {
                "source_joint_native_cosine_mean": _stats(
                    _collect(rows, "pair", "source_joint_native_cosine_mean")
                ),
                "source_joint_native_cosine_min": _stats(
                    _collect(rows, "pair", "source_joint_native_cosine_min")
                ),
                "target_joint_native_cosine_mean": _stats(
                    _collect(rows, "pair", "target_joint_native_cosine_mean")
                ),
                "target_joint_native_cosine_min": _stats(
                    _collect(rows, "pair", "target_joint_native_cosine_min")
                ),
                "source_drift_l2_ratio_mean": _stats(
                    _collect(rows, "pair", "source_drift_l2_ratio_mean")
                ),
                "target_drift_l2_ratio_mean": _stats(
                    _collect(rows, "pair", "target_drift_l2_ratio_mean")
                ),
            },
        }
        summary[group_name] = group
    return summary


def _keep_fjsar_dump_row(row, case_filter):
    case_filter = str(case_filter or "all")
    if case_filter == "all":
        return True
    if case_filter == "oracle_gap":
        return bool(row.get("oracle_gap_case"))
    if case_filter == "attention_harms_native":
        return bool(row.get("attention_harms_native_case"))
    if case_filter == "attention_rescues_native":
        return bool(row.get("attention_rescues_native_case"))
    if case_filter == "oracle_gap_or_harm":
        return bool(row.get("oracle_gap_case")) or bool(row.get("attention_harms_native_case"))
    if case_filter == "attention_informative":
        return (
            bool(row.get("oracle_gap_case"))
            or bool(row.get("attention_harms_native_case"))
            or bool(row.get("attention_rescues_native_case"))
        )
    raise ValueError(f"unsupported FJSAR dump case filter: {case_filter}")


def _summarize_kernel_featureization_audit(records):
    groups = {
        "all": lambda row: True,
        "oracle_gap": lambda row: bool(row.get("oracle_gap_case")),
        "attention_harms_native": lambda row: bool(row.get("attention_harms_native_case")),
        "attention_rescues_native": lambda row: bool(row.get("attention_rescues_native_case")),
        "native_correct_method_wrong": lambda row: bool(row.get("baseline_pck_hit"))
        and not bool(row.get("method_pck_hit")),
        "native_wrong_method_correct": lambda row: not bool(row.get("baseline_pck_hit"))
        and bool(row.get("method_pck_hit")),
    }
    score_names = []
    topk_keys = []
    for row in records:
        audit = row.get("kernel_featureization_audit")
        if isinstance(audit, dict):
            score_names = list(audit.get("score_names", []))
            topk_keys = sorted(audit.get("topk_hits", {}).keys())
            if score_names and topk_keys:
                break
    summary = {}
    for group_name, predicate in groups.items():
        rows = [
            row for row in records
            if predicate(row) and isinstance(row.get("kernel_featureization_audit"), dict)
        ]
        group = {
            "points": len(rows),
            "baseline_correct": sum(1 for row in rows if row.get("baseline_pck_hit")),
            "method_correct": sum(1 for row in rows if row.get("method_pck_hit")),
            "attention_top1_correct": sum(1 for row in rows if row.get("attention_top1_pck_hit")),
            "attention_topk_correct": sum(1 for row in rows if row.get("attention_topk_pck_hit")),
            "signals": {},
        }
        for score_name in score_names:
            signal = {}
            ranks = [
                int(row["kernel_featureization_audit"]["ranks"][score_name])
                for row in rows
                if row["kernel_featureization_audit"].get("ranks", {}).get(score_name) is not None
            ]
            signal["ranked_points"] = len(ranks)
            for key in topk_keys:
                if not key.startswith(f"{score_name}@"):
                    continue
                suffix = key.replace(f"{score_name}@", "")
                hits = sum(
                    1 for row in rows
                    if bool(row["kernel_featureization_audit"].get("topk_hits", {}).get(key))
                )
                signal[f"hit_at_{suffix}"] = hits
                signal[f"hit_rate_at_{suffix}"] = hits / len(rows) if rows else 0.0
            group["signals"][score_name] = signal
        summary[group_name] = group
    return summary


def _summarize_transport_factorization_audit(records):
    return _summarize_ranked_candidate_score_audit(records, "transport_factorization_audit")


def _summarize_residual_readout_audit(records):
    return _summarize_ranked_candidate_score_audit(records, "residual_readout_audit")


def _summarize_latent_expert_audit(records):
    summary = _summarize_ranked_candidate_score_audit(records, "latent_expert_audit")
    predicates = {
        "all": lambda row: True,
        "oracle_gap": lambda row: bool(row.get("oracle_gap_case")),
        "attention_harms_native": lambda row: bool(row.get("attention_harms_native_case")),
        "attention_rescues_native": lambda row: bool(row.get("attention_rescues_native_case")),
    }
    rank_bins = {
        "1": lambda rank: rank == 1,
        "2-3": lambda rank: 2 <= rank <= 3,
        "4-5": lambda rank: 4 <= rank <= 5,
        "6-10": lambda rank: 6 <= rank <= 10,
        "11-20": lambda rank: 11 <= rank <= 20,
    }

    def _mean(values):
        clean = [float(value) for value in values if value is not None and np.isfinite(float(value))]
        return float(np.mean(clean)) if clean else None

    for group_name, predicate in predicates.items():
        rows = [
            row for row in records
            if predicate(row) and isinstance(row.get("latent_expert_audit"), dict)
        ]
        group = summary.setdefault(group_name, {"points": len(rows), "signals": {}})
        diagnostics = [row["latent_expert_audit"].get("diagnostics", {}) for row in rows]
        mean_top1 = sum(
            1 for row in rows
            if row["latent_expert_audit"].get("ranks", {}).get("mean_expert_support") == 1
        )
        stable_top1 = sum(
            1 for row in rows
            if row["latent_expert_audit"].get("ranks", {}).get("stable_head_1") == 1
        )
        oracle_head_top1 = sum(
            1 for row in rows
            if row["latent_expert_audit"].get("ranks", {}).get("oracle_pair_head_1") == 1
        )
        group["mechanism_checks"] = {
            "baseline_correct": sum(1 for row in rows if row.get("baseline_pck_hit")),
            "attention_top1_correct": sum(1 for row in rows if row.get("attention_top1_pck_hit")),
            "attention_top20_correct": sum(1 for row in rows if row.get("attention_topk_pck_hit")),
            "mean_expert_top1": int(mean_top1),
            "stable_head_1_top1": int(stable_top1),
            "oracle_pair_head_1_top1": int(oracle_head_top1),
            "hidden_by_mean_but_any_head_top1": sum(
                1 for row, diagnostic in zip(rows, diagnostics)
                if row["latent_expert_audit"].get("ranks", {}).get("mean_expert_support") != 1
                and bool(diagnostic.get("any_head_top1_pck_hit"))
            ),
            "hidden_by_attention_average_but_any_head_top1": sum(
                1 for row, diagnostic in zip(rows, diagnostics)
                if not bool(row.get("attention_top1_pck_hit"))
                and bool(diagnostic.get("any_head_top1_pck_hit"))
            ),
            "stable_head_recovers_mean_error": sum(
                1 for row in rows
                if row["latent_expert_audit"].get("ranks", {}).get("mean_expert_support") != 1
                and row["latent_expert_audit"].get("ranks", {}).get("stable_head_1") == 1
            ),
            "stable_head_harms_mean_correct": sum(
                1 for row in rows
                if row["latent_expert_audit"].get("ranks", {}).get("mean_expert_support") == 1
                and row["latent_expert_audit"].get("ranks", {}).get("stable_head_1") != 1
            ),
            "stable_head_recovers_attention_top1_error": sum(
                1 for row in rows
                if not bool(row.get("attention_top1_pck_hit"))
                and row["latent_expert_audit"].get("ranks", {}).get("stable_head_1") == 1
            ),
            "stable_head_harms_attention_top1_correct": sum(
                1 for row in rows
                if bool(row.get("attention_top1_pck_hit"))
                and row["latent_expert_audit"].get("ranks", {}).get("stable_head_1") != 1
            ),
            "any_head_top1": sum(
                1 for diagnostic in diagnostics if bool(diagnostic.get("any_head_top1_pck_hit"))
            ),
            "any_member_top1": sum(
                1 for diagnostic in diagnostics if bool(diagnostic.get("any_member_top1_pck_hit"))
            ),
            "any_expert_top1": sum(
                1 for diagnostic in diagnostics if bool(diagnostic.get("any_expert_top1_pck_hit"))
            ),
            "head_top1_pck_hit_fraction_mean": _mean(
                diagnostic.get("head_top1_pck_hit_fraction") for diagnostic in diagnostics
            ),
            "expert_top1_pck_hit_fraction_mean": _mean(
                diagnostic.get("expert_top1_pck_hit_fraction") for diagnostic in diagnostics
            ),
            "correct_beats_attention_top1_head_fraction_mean": _mean(
                diagnostic.get("correct_beats_attention_top1_head_fraction")
                for diagnostic in diagnostics
            ),
        }

        bin_summary = {}
        for bin_name, in_bin in rank_bins.items():
            selected = []
            for row in rows:
                rank = row.get("gt_ranks", {}).get("attention")
                if rank is not None and in_bin(int(rank)):
                    selected.append(row)
            bin_summary[bin_name] = {
                "points": len(selected),
                "mean_expert_top1": sum(
                    1 for row in selected
                    if row["latent_expert_audit"].get("ranks", {}).get("mean_expert_support") == 1
                ),
                "stable_head_1_top1": sum(
                    1 for row in selected
                    if row["latent_expert_audit"].get("ranks", {}).get("stable_head_1") == 1
                ),
                "oracle_pair_head_1_top1": sum(
                    1 for row in selected
                    if row["latent_expert_audit"].get("ranks", {}).get("oracle_pair_head_1") == 1
                ),
                "any_head_top1": sum(
                    1 for row in selected
                    if bool(
                        row["latent_expert_audit"].get("diagnostics", {}).get(
                            "any_head_top1_pck_hit"
                        )
                    )
                ),
            }
        group["attention_rank_bins"] = bin_summary

        pair_rows = {}
        for row in rows:
            key = (
                row.get("category"),
                row.get("pair_json"),
                row.get("src_image"),
                row.get("trg_image"),
            )
            pair_rows.setdefault(key, []).append(row)
        pair_comparisons = []
        for key, selected in pair_rows.items():
            def _hits(signal):
                return sum(
                    1 for row in selected
                    if row["latent_expert_audit"].get("ranks", {}).get(signal) == 1
                )

            pair_comparisons.append({
                "category": key[0],
                "pair_json": key[1],
                "points": len(selected),
                "attention_top1": sum(
                    1 for row in selected if bool(row.get("attention_top1_pck_hit"))
                ),
                "mean_expert_top1": _hits("mean_expert_support"),
                "stable_head_1_top1": _hits("stable_head_1"),
                "oracle_pair_head_1_top1": _hits("oracle_pair_head_1"),
                "oracle_pair_head_4_top1": _hits("oracle_pair_head_4"),
            })
        group["pair_level"] = {
            "pairs": len(pair_comparisons),
            "stable_head_improves_mean_pairs": sum(
                1 for pair in pair_comparisons
                if pair["stable_head_1_top1"] > pair["mean_expert_top1"]
            ),
            "stable_head_improves_attention_pairs": sum(
                1 for pair in pair_comparisons
                if pair["stable_head_1_top1"] > pair["attention_top1"]
            ),
            "stable_head_harms_attention_pairs": sum(
                1 for pair in pair_comparisons
                if pair["stable_head_1_top1"] < pair["attention_top1"]
            ),
            "stable_head_harms_mean_pairs": sum(
                1 for pair in pair_comparisons
                if pair["stable_head_1_top1"] < pair["mean_expert_top1"]
            ),
            "oracle_single_head_improves_mean_pairs": sum(
                1 for pair in pair_comparisons
                if pair["oracle_pair_head_1_top1"] > pair["mean_expert_top1"]
            ),
            "records": pair_comparisons,
        }
    return summary


def _summarize_local_relational_identity_audit(records):
    return _summarize_ranked_candidate_score_audit(records, "local_relational_identity_audit")


def _summarize_dense_transport_consistency_audit(records):
    return _summarize_ranked_candidate_score_audit(records, "dense_transport_consistency_audit")


def _summarize_candidate_field_consistency_audit(records):
    return _summarize_ranked_candidate_score_audit(records, "candidate_field_consistency_audit")


def _summarize_multilayer_identity_audit(records):
    return _summarize_ranked_candidate_score_audit(records, "multilayer_identity_audit")


def _summarize_layer_routed_identity_audits(records):
    """Summarize layerwise routing/identity evidence and PCK changes."""

    points = [point for pair in records for point in pair.get("points", [])]
    summaries = [pair.get("summary", {}) for pair in records]
    branch_names = []
    for summary in summaries:
        for name in summary.get("branches", {}):
            if name not in branch_names:
                branch_names.append(name)
    branch_summary = {}
    for name in branch_names:
        rows = [
            point.get("branches", {}).get(name, {})
            for point in points
            if name in point.get("branches", {})
        ]
        ranks = [int(row["gt_rank"]) for row in rows if row.get("gt_rank") is not None]
        hard_rows = [point for point in points if point.get("both_wrong_top20_hit")]
        hard = [
            point.get("branches", {}).get(name, {})
            for point in hard_rows
            if name in point.get("branches", {})
        ]
        hard_ranks = [int(row["gt_rank"]) for row in hard if row.get("gt_rank") is not None]
        branch_summary[name] = {
            "points": len(rows),
            "top1_pck_hits": int(sum(bool(row.get("top1_pck_hit")) for row in rows)),
            "top20_pck_hits": int(sum(bool(row.get("top20_pck_hit")) for row in rows)),
            "gt_rank_mean": float(np.mean(ranks)) if ranks else None,
            "gt_rank_median": float(np.median(ranks)) if ranks else None,
            "both_wrong_top20_hit_points": len(hard),
            "both_wrong_top1_hits": int(sum(bool(row.get("top1_pck_hit")) for row in hard)),
            "both_wrong_gt_rank_mean": float(np.mean(hard_ranks)) if hard_ranks else None,
            "both_wrong_gt_rank_median": float(np.median(hard_ranks)) if hard_ranks else None,
        }
    return {
        "hypothesis": (
            summaries[0].get("hypothesis", {})
            if summaries
            else {"name": "Layerwise Routing Identity"}
        ),
        "pairs": len(records),
        "points": len(points),
        "baseline_correct": int(sum(bool(point.get("baseline_pck_hit")) for point in points)),
        "method_correct": int(sum(bool(point.get("method_pck_hit")) for point in points)),
        "rescued_vs_baseline": int(sum(bool(point.get("rescued_vs_baseline")) for point in points)),
        "harmed_vs_baseline": int(sum(bool(point.get("harmed_vs_baseline")) for point in points)),
        "candidate_missing_gt": int(sum(bool(point.get("candidate_missing_gt")) for point in points)),
        "both_wrong_top20_hit_points": int(sum(bool(point.get("both_wrong_top20_hit")) for point in points)),
        "primary_branch": summaries[0].get("primary_branch") if summaries else None,
        "layer_names": summaries[0].get("layer_names", []) if summaries else [],
        "branches": branch_summary,
    }


def _summarize_method_descriptor_audit(records):
    return _summarize_ranked_candidate_score_audit(records, "method_descriptor_audit")


def _summarize_transport_lift_branch_audit(records):
    return _summarize_ranked_candidate_score_audit(records, "transport_lift_branch_audit")


def _summarize_trajectory_identity_audit(records):
    return _summarize_ranked_candidate_score_audit(records, "trajectory_identity_audit")


def _summarize_multi_timestep_attention_identity_audit(records):
    groups = {
        "all": lambda row: True,
        "oracle_gap": lambda row: bool(row.get("oracle_gap_case")),
        "attention_harms_native": lambda row: bool(row.get("attention_harms_native_case")),
        "attention_rescues_native": lambda row: bool(row.get("attention_rescues_native_case")),
    }

    def _mean(values):
        return float(sum(values) / len(values)) if values else 0.0

    def _median(values):
        if not values:
            return None
        values = sorted(values)
        middle = len(values) // 2
        if len(values) % 2:
            return float(values[middle])
        return float((values[middle - 1] + values[middle]) / 2.0)

    def _stability(audit):
        timesteps = audit.get("timesteps") or []
        if not timesteps:
            return 0.0
        top1_pixels = [
            tuple(item.get("attention_top1_pixel") or [])
            for item in audit.get("per_timestep", [])
            if item.get("attention_top1_pixel") is not None
        ]
        if not top1_pixels:
            return 0.0
        counts = Counter(top1_pixels)
        return float(max(counts.values()) / len(top1_pixels))

    summary = {}
    for group_name, predicate in groups.items():
        rows = [
            row for row in records
            if predicate(row) and isinstance(row.get("multi_timestep_attention_identity_audit"), dict)
        ]
        audits = [row["multi_timestep_attention_identity_audit"] for row in rows]
        timestep_counts = [int(audit.get("timestep_count", 0)) for audit in audits]
        baseline_hit_counts = [int(audit.get("baseline_hit_count", 0)) for audit in audits]
        top1_hit_counts = [int(audit.get("attention_top1_hit_count", 0)) for audit in audits]
        topk_hit_counts = [int(audit.get("attention_topk_hit_count", 0)) for audit in audits]
        gt_rank_means = [float(audit["summary"]["attention_gt_rank_mean"]) for audit in audits if audit.get("summary", {}).get("attention_gt_rank_mean") is not None]
        gt_rank_bests = [float(audit["summary"]["attention_gt_rank_best"]) for audit in audits if audit.get("summary", {}).get("attention_gt_rank_best") is not None]
        gt_rank_worsts = [float(audit["summary"]["attention_gt_rank_worst"]) for audit in audits if audit.get("summary", {}).get("attention_gt_rank_worst") is not None]
        top1_stabilities = [float(audit["summary"]["attention_top1_stability"]) for audit in audits if audit.get("summary", {}).get("attention_top1_stability") is not None]
        top1_unique_counts = [float(audit["summary"]["attention_top1_unique_count"]) for audit in audits if audit.get("summary", {}).get("attention_top1_unique_count") is not None]
        topk_persistence = [float(audit["summary"]["attention_topk_persistence"]) for audit in audits if audit.get("summary", {}).get("attention_topk_persistence") is not None]
        oracle_gap_counts = [int(audit["summary"]["oracle_gap_timesteps"]) for audit in audits if audit.get("summary", {}).get("oracle_gap_timesteps") is not None]
        harm_counts = [int(audit["summary"]["attention_harms_native_timesteps"]) for audit in audits if audit.get("summary", {}).get("attention_harms_native_timesteps") is not None]
        rescue_counts = [int(audit["summary"]["attention_rescues_native_timesteps"]) for audit in audits if audit.get("summary", {}).get("attention_rescues_native_timesteps") is not None]
        group = {
            "points": len(rows),
            "signals": {
                "timestep_count": {
                    "mean": _mean(timestep_counts),
                    "median": _median(timestep_counts),
                },
                "baseline_hit_count": {
                    "mean": _mean(baseline_hit_counts),
                    "median": _median(baseline_hit_counts),
                },
                "attention_top1_hit_count": {
                    "mean": _mean(top1_hit_counts),
                    "median": _median(top1_hit_counts),
                },
                "attention_topk_hit_count": {
                    "mean": _mean(topk_hit_counts),
                    "median": _median(topk_hit_counts),
                },
                "attention_gt_rank_mean": {
                    "mean": _mean(gt_rank_means),
                    "median": _median(gt_rank_means),
                },
                "attention_gt_rank_best": {
                    "mean": _mean(gt_rank_bests),
                    "median": _median(gt_rank_bests),
                },
                "attention_gt_rank_worst": {
                    "mean": _mean(gt_rank_worsts),
                    "median": _median(gt_rank_worsts),
                },
                "attention_top1_stability": {
                    "mean": _mean(top1_stabilities),
                    "median": _median(top1_stabilities),
                },
                "attention_top1_unique_count": {
                    "mean": _mean(top1_unique_counts),
                    "median": _median(top1_unique_counts),
                },
                "attention_topk_persistence": {
                    "mean": _mean(topk_persistence),
                    "median": _median(topk_persistence),
                },
                "oracle_gap_timesteps": {
                    "mean": _mean(oracle_gap_counts),
                    "median": _median(oracle_gap_counts),
                },
                "attention_harms_native_timesteps": {
                    "mean": _mean(harm_counts),
                    "median": _median(harm_counts),
                },
                "attention_rescues_native_timesteps": {
                    "mean": _mean(rescue_counts),
                    "median": _median(rescue_counts),
                },
            },
        }
        summary[group_name] = group
    return summary


def _summarize_anchor_topology_audit(records, pair_records=None):
    groups = {
        "all": lambda row: True,
        "oracle_gap": lambda row: bool(row.get("oracle_gap_case")),
        "attention_harms_native": lambda row: bool(row.get("attention_harms_native_case")),
        "attention_rescues_native": lambda row: bool(row.get("attention_rescues_native_case")),
        "baseline_correct": lambda row: bool(row.get("baseline_pck_hit")),
        "baseline_wrong": lambda row: not bool(row.get("baseline_pck_hit")),
    }

    def _median(values):
        if not values:
            return None
        values = sorted(values)
        middle = len(values) // 2
        if len(values) % 2:
            return float(values[middle])
        return float((values[middle - 1] + values[middle]) / 2.0)

    def _mean(values):
        return float(sum(values) / len(values)) if values else 0.0

    score_names = []
    for row in records:
        audit = row.get("anchor_topology_audit")
        if isinstance(audit, dict):
            score_names = list(audit.get("score_names", []))
            if score_names:
                break
    pair_records = list(pair_records or [])
    if not pair_records:
        pair_seen = {}
        for row in records:
            audit = row.get("anchor_topology_audit")
            if not isinstance(audit, dict):
                continue
            pair_key = (
                row.get("category"),
                row.get("pair_json"),
                row.get("src_image"),
                row.get("trg_image"),
            )
            pair_seen.setdefault(pair_key, audit)
        pair_records = [
            {
                "positive_anchor_count": float(audit.get("positive_anchor_count", 0.0)),
                "effective_anchor_count": float(audit.get("effective_anchor_count", 0.0)),
                "anchor_confidence": audit.get("anchor_confidence") or {},
            }
            for audit in pair_seen.values()
        ]
    anchor_counts = [float(record.get("positive_anchor_count", 0.0)) for record in pair_records]
    effective_counts = [float(record.get("effective_anchor_count", 0.0)) for record in pair_records]
    confidence_sums = [
        float((record.get("anchor_confidence") or {}).get("sum", 0.0))
        for record in pair_records
    ]
    confidence_max = [
        float((record.get("anchor_confidence") or {}).get("max", 0.0))
        for record in pair_records
    ]
    summary = {
        "pairs": len(pair_records),
        "pair_anchor_distribution": {
            "positive_anchor_count_mean": _mean(anchor_counts),
            "positive_anchor_count_median": _median(anchor_counts),
            "effective_anchor_count_mean": _mean(effective_counts),
            "effective_anchor_count_median": _median(effective_counts),
            "confidence_sum_mean": _mean(confidence_sums),
            "confidence_sum_median": _median(confidence_sums),
            "confidence_max_mean": _mean(confidence_max),
            "confidence_max_median": _median(confidence_max),
        },
        "groups": {},
    }
    for group_name, predicate in groups.items():
        rows = [
            row for row in records
            if predicate(row) and isinstance(row.get("anchor_topology_audit"), dict)
        ]
        group = {
            "points": len(rows),
            "baseline_correct": sum(1 for row in rows if row.get("baseline_pck_hit")),
            "method_correct": sum(1 for row in rows if row.get("method_pck_hit")),
            "attention_top1_correct": sum(1 for row in rows if row.get("attention_top1_pck_hit")),
            "attention_topk_correct": sum(1 for row in rows if row.get("attention_topk_pck_hit")),
            "signals": {},
        }
        for score_name in score_names:
            ranks = []
            gaps = []
            for row in rows:
                audit = row["anchor_topology_audit"]
                rank = audit.get("ranks", {}).get(score_name)
                if rank is not None:
                    ranks.append(int(rank))
                gap = audit.get("score_gaps", {}).get(
                    f"{score_name}_attention_top1_minus_best_pck_hit_proposal"
                )
                if gap is not None:
                    gaps.append(float(gap))
            signal = {
                "ranked_points": len(ranks),
                "proposal_pck_hit_rank_median": _median(ranks),
                "proposal_pck_hit_rank_mean": _mean(ranks),
                "hit_at_1": sum(1 for rank in ranks if rank <= 1),
                "hit_at_3": sum(1 for rank in ranks if rank <= 3),
                "hit_at_5": sum(1 for rank in ranks if rank <= 5),
                "hit_at_10": sum(1 for rank in ranks if rank <= 10),
                "hit_at_20": sum(1 for rank in ranks if rank <= 20),
                "hit_rate_at_1": sum(1 for rank in ranks if rank <= 1) / len(rows) if rows else 0.0,
                "hit_rate_at_3": sum(1 for rank in ranks if rank <= 3) / len(rows) if rows else 0.0,
                "hit_rate_at_5": sum(1 for rank in ranks if rank <= 5) / len(rows) if rows else 0.0,
                "hit_rate_at_10": sum(1 for rank in ranks if rank <= 10) / len(rows) if rows else 0.0,
                "hit_rate_at_20": sum(1 for rank in ranks if rank <= 20) / len(rows) if rows else 0.0,
                "attention_top1_minus_best_pck_hit_proposal_gap_median": _median(gaps),
                "best_pck_hit_proposal_beats_attention_top1": sum(1 for gap in gaps if gap < 0.0),
                "attention_top1_beats_best_pck_hit_proposal": sum(1 for gap in gaps if gap > 0.0),
            }
            group["signals"][score_name] = signal
        summary["groups"][group_name] = group
    protection_rows = [
        row for row in records
        if row.get("baseline_pck_hit") and isinstance(row.get("anchor_topology_audit"), dict)
    ]
    summary["native_correct_anchor_protection"] = {
        "points": len(protection_rows),
        "attention_top1_harms": sum(1 for row in protection_rows if not row.get("attention_top1_pck_hit")),
        "hybrid_topology_keeps_at_1": sum(
            1
            for row in protection_rows
            if (row["anchor_topology_audit"].get("ranks", {}).get("hybrid_anchor_topology") or 999) <= 1
        ),
        "hybrid_topology_keeps_at_3": sum(
            1
            for row in protection_rows
            if (row["anchor_topology_audit"].get("ranks", {}).get("hybrid_anchor_topology") or 999) <= 3
        ),
        "hybrid_topology_keeps_at_5": sum(
            1
            for row in protection_rows
            if (row["anchor_topology_audit"].get("ranks", {}).get("hybrid_anchor_topology") or 999) <= 5
        ),
    }
    return summary


def _summarize_ranked_candidate_score_audit(records, audit_key):
    groups = {
        "all": lambda row: True,
        "oracle_gap": lambda row: bool(row.get("oracle_gap_case")),
        "attention_harms_native": lambda row: bool(row.get("attention_harms_native_case")),
        "attention_rescues_native": lambda row: bool(row.get("attention_rescues_native_case")),
    }

    def _median(values):
        if not values:
            return None
        values = sorted(values)
        middle = len(values) // 2
        if len(values) % 2:
            return float(values[middle])
        return float((values[middle - 1] + values[middle]) / 2.0)

    score_names = []
    for row in records:
        audit = row.get(audit_key)
        if isinstance(audit, dict):
            score_names = list(audit.get("score_names", []))
            if score_names:
                break
    summary = {}
    for group_name, predicate in groups.items():
        rows = [
            row for row in records
            if predicate(row) and isinstance(row.get(audit_key), dict)
        ]
        group = {"points": len(rows), "signals": {}}
        group["valid_fingerprint_coverage"] = float(len(rows) / max(1, len(records)))
        for score_name in score_names:
            ranks = []
            gaps = []
            for row in rows:
                audit = row[audit_key]
                rank = audit.get("ranks", {}).get(score_name)
                if rank is not None:
                    ranks.append(int(rank))
                gap = audit.get("score_gaps", {}).get(
                    f"{score_name}_attention_top1_minus_best_pck_hit_proposal"
                )
                if gap is not None:
                    gaps.append(float(gap))
            group["signals"][score_name] = {
                "ranked_points": len(ranks),
                "proposal_pck_hit_rank_median": _median(ranks),
                "proposal_pck_hit_at_1": sum(1 for rank in ranks if rank <= 1),
                "proposal_pck_hit_at_3": sum(1 for rank in ranks if rank <= 3),
                "proposal_pck_hit_at_5": sum(1 for rank in ranks if rank <= 5),
                "proposal_pck_hit_at_10": sum(1 for rank in ranks if rank <= 10),
                "attention_top1_minus_best_pck_hit_proposal_gap_median": _median(gaps),
                "best_pck_hit_proposal_beats_attention_top1": sum(1 for gap in gaps if gap < 0.0),
                "attention_top1_beats_best_pck_hit_proposal": sum(1 for gap in gaps if gap > 0.0),
            }
        summary[group_name] = group
    return summary


def _summarize_dense_candidate_edge_audit(records):
    summary = _summarize_ranked_candidate_score_audit(
        records, "dense_candidate_edge_audit"
    )
    predicates = {
        "all": lambda row: True,
        "oracle_gap": lambda row: bool(row.get("oracle_gap_case")),
        "attention_harms_native": lambda row: bool(row.get("attention_harms_native_case")),
        "attention_rescues_native": lambda row: bool(row.get("attention_rescues_native_case")),
    }

    for group_name, predicate in predicates.items():
        rows = [
            row for row in records
            if predicate(row) and isinstance(row.get("dense_candidate_edge_audit"), dict)
        ]
        group = summary.setdefault(group_name, {"points": len(rows), "signals": {}})
        audits = [row["dense_candidate_edge_audit"] for row in rows]
        candidate_hit_fractions = [
            float(audit.get("diagnostics", {}).get("pck_hit_candidate_fraction", 0.0))
            for audit in audits
        ]
        random_top1_expectation = (
            float(sum(candidate_hit_fractions) / len(candidate_hit_fractions))
            if candidate_hit_fractions else 0.0
        )
        mechanism_checks = {
            "attention_top1_correct": sum(
                1 for row in rows if bool(row.get("attention_top1_pck_hit"))
            ),
            "attention_topk_correct": sum(
                1 for row in rows if bool(row.get("attention_topk_pck_hit"))
            ),
            "candidate_pck_hit_fraction_mean": random_top1_expectation,
            "gt_used_for_scoring_count": sum(
                1 for audit in audits
                if bool(audit.get("diagnostics", {}).get("gt_used_for_scoring"))
            ),
            "native_candidate_injected_count": sum(
                1 for audit in audits
                if bool(audit.get("diagnostics", {}).get("native_candidate_injected"))
            ),
            "native_fallback_used_count": sum(
                1 for audit in audits
                if bool(audit.get("diagnostics", {}).get("native_fallback_used"))
            ),
            "source_neighbor_count_mean": (
                float(sum(
                    int(audit.get("diagnostics", {}).get("source_neighbor_count", 0))
                    for audit in audits
                ) / len(audits))
                if audits else 0.0
            ),
        }
        attention_top1_correct = int(mechanism_checks["attention_top1_correct"])
        attention_top1_errors = int(len(rows) - attention_top1_correct)
        score_names = list(audits[0].get("score_names", [])) if audits else []
        for score_name in score_names:
            top1 = sum(
                1 for audit in audits
                if audit.get("ranks", {}).get(score_name) == 1
            )
            recovered = sum(
                1 for row in rows
                if not bool(row.get("attention_top1_pck_hit"))
                and row["dense_candidate_edge_audit"].get("ranks", {}).get(score_name) == 1
            )
            harmed = sum(
                1 for row in rows
                if bool(row.get("attention_top1_pck_hit"))
                and row["dense_candidate_edge_audit"].get("ranks", {}).get(score_name) != 1
            )
            selected_deeper = sum(
                1 for audit in audits
                if int(
                    audit.get("diagnostics", {})
                    .get("selected_attention_ranks", {})
                    .get(score_name, 1)
                ) > 1
            )
            mechanism_checks[score_name] = {
                "top1_pck_hits": int(top1),
                "top1_pck_rate": float(top1 / len(rows)) if rows else 0.0,
                "lift_over_uniform_candidate_expectation": (
                    float(top1 / len(rows) - random_top1_expectation)
                    if rows else 0.0
                ),
                "recovers_attention_top1_errors": int(recovered),
                "attention_top1_error_recovery_rate": (
                    float(recovered / attention_top1_errors)
                    if attention_top1_errors else 0.0
                ),
                "harms_attention_top1_correct": int(harmed),
                "attention_top1_correct_retention_rate": (
                    float((attention_top1_correct - harmed) / attention_top1_correct)
                    if attention_top1_correct else 0.0
                ),
                "net_vs_attention_top1": int(recovered - harmed),
                "net_pck_rate_vs_attention_top1": (
                    float((recovered - harmed) / len(rows)) if rows else 0.0
                ),
                "selects_below_attention_rank1": int(selected_deeper),
            }
        group["mechanism_checks"] = mechanism_checks
        group["graph_contract"] = (
            dict(audits[0].get("graph_contract", {})) if audits else {}
        )
    return summary


def _summarize_candidate_clamped_causal_replay_audit(records):
    """Summarize causal candidate selection and its uncertainty by image pair."""

    audit_key = "candidate_clamped_causal_replay_audit"
    summary = _summarize_ranked_candidate_score_audit(records, audit_key)
    predicates = {
        "all": lambda row: True,
        "both_wrong_top20_hit": lambda row: bool(row.get("oracle_gap_case")),
        "attention_harms_native": lambda row: bool(row.get("attention_harms_native_case")),
        "attention_rescues_native": lambda row: bool(row.get("attention_rescues_native_case")),
    }
    primary_signal = "post_release_bidirectional_negative_log_rank"

    def _mean(values):
        return float(sum(values) / len(values)) if values else 0.0

    def _sample_standard_error(values):
        if len(values) < 2:
            return None
        mean = _mean(values)
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        return float((variance / len(values)) ** 0.5)

    for group_name, predicate in predicates.items():
        rows = [
            row
            for row in records
            if predicate(row) and isinstance(row.get(audit_key), dict)
        ]
        base_group_name = "oracle_gap" if group_name == "both_wrong_top20_hit" else group_name
        group = summary.pop(base_group_name, None) or {
            "points": len(rows),
            "signals": {},
        }
        audits = [row[audit_key] for row in rows]
        hit_fractions = [
            float(audit.get("diagnostics", {}).get("pck_hit_candidate_fraction", 0.0))
            for audit in audits
        ]
        random_expectation = _mean(hit_fractions)
        attention_top1_correct = sum(
            1 for row in rows if bool(row.get("attention_top1_pck_hit"))
        )
        attention_top1_errors = len(rows) - attention_top1_correct
        contracts = [audit.get("causal_contract", {}) for audit in audits]
        diagnostics = [audit.get("diagnostics", {}) for audit in audits]
        mechanism_checks = {
            "attention_top1_correct": int(attention_top1_correct),
            "attention_top20_hit": sum(
                1 for row in rows if bool(row.get("attention_topk_pck_hit"))
            ),
            "candidate_pck_hit_fraction_mean": random_expectation,
            "native_candidate_injected_count": sum(
                int(bool(contract.get("native_candidate_injected")))
                for contract in contracts
            ),
            "native_fallback_used_count": sum(
                int(bool(contract.get("native_fallback_used"))) for contract in contracts
            ),
            "gt_used_for_scoring_count": sum(
                int(bool(contract.get("gt_used_for_scoring"))) for contract in contracts
            ),
            "source_cross_mass_mean": _mean([
                float(item.get("source_cross_mass_mean", 0.0)) for item in diagnostics
            ]),
            "target_cross_mass_mean": _mean([
                float(item.get("target_cross_mass_mean", 0.0)) for item in diagnostics
            ]),
            "source_intervention_relative_l2_mean": _mean([
                float(item.get("source_intervention_relative_l2_mean", 0.0))
                for item in diagnostics
            ]),
            "target_intervention_relative_l2_mean": _mean([
                float(item.get("target_intervention_relative_l2_mean", 0.0))
                for item in diagnostics
            ]),
            "causal_rank_improvement_mean": _mean([
                float(item.get("causal_rank_improvement_mean", 0.0))
                for item in diagnostics
            ]),
            "causal_improvement_positive_fraction_mean": _mean([
                float(item.get("causal_improvement_positive_fraction_mean", 0.0))
                for item in diagnostics
            ]),
            "post_release_score_std_mean": _mean([
                float(item.get("post_release_score_std_mean", 0.0))
                for item in diagnostics
            ]),
        }
        score_names = list(audits[0].get("score_names", [])) if audits else []
        for score_name in score_names:
            selected_hit = [
                int(audit.get("ranks", {}).get(score_name) == 1) for audit in audits
            ]
            selected_deeper = sum(
                int(
                    (audit.get("diagnostics", {}).get("selected_attention_ranks", {})
                     .get(score_name) or 1) > 1
                )
                for audit in audits
            )
            recovered = sum(
                int((not bool(row.get("attention_top1_pck_hit"))) and bool(hit))
                for row, hit in zip(rows, selected_hit)
            )
            harmed = sum(
                int(bool(row.get("attention_top1_pck_hit")) and not bool(hit))
                for row, hit in zip(rows, selected_hit)
            )
            pair_lifts = {}
            category_lifts = {}
            for row, hit, expected in zip(rows, selected_hit, hit_fractions):
                pair_key = (
                    str(row.get("category", "")),
                    str(row.get("pair_json", "")),
                    str(row.get("src_image", "")),
                    str(row.get("trg_image", "")),
                )
                pair_lifts.setdefault(pair_key, []).append(float(hit) - expected)
                category_lifts.setdefault(str(row.get("category", "")), []).append(
                    float(hit) - expected
                )
            pair_equal_lifts = [_mean(values) for values in pair_lifts.values()]
            category_breakdown = {
                category: {
                    "points": len(values),
                    "lift_over_uniform_candidate_expectation": _mean(values),
                }
                for category, values in sorted(category_lifts.items())
            }
            top1 = sum(selected_hit)
            top3 = sum(
                int((audit.get("ranks", {}).get(score_name) or 999) <= 3)
                for audit in audits
            )
            top5 = sum(
                int((audit.get("ranks", {}).get(score_name) or 999) <= 5)
                for audit in audits
            )
            top10 = sum(
                int((audit.get("ranks", {}).get(score_name) or 999) <= 10)
                for audit in audits
            )
            signal = mechanism_checks[score_name] = {
                "top1_pck_hits": int(top1),
                "top1_pck_rate": float(top1 / len(rows)) if rows else 0.0,
                "top3_pck_hits": int(top3),
                "top3_pck_rate": float(top3 / len(rows)) if rows else 0.0,
                "top5_pck_hits": int(top5),
                "top5_pck_rate": float(top5 / len(rows)) if rows else 0.0,
                "top10_pck_hits": int(top10),
                "top10_pck_rate": float(top10 / len(rows)) if rows else 0.0,
                "lift_over_uniform_candidate_expectation": (
                    float(top1 / len(rows) - random_expectation) if rows else 0.0
                ),
                "recovers_attention_top1_errors": int(recovered),
                "attention_top1_error_recovery_rate": (
                    float(recovered / attention_top1_errors)
                    if attention_top1_errors else 0.0
                ),
                "harms_attention_top1_correct": int(harmed),
                "net_vs_attention_top1": int(recovered - harmed),
                "net_pck_rate_vs_attention_top1": (
                    float((recovered - harmed) / len(rows)) if rows else 0.0
                ),
                "selects_below_attention_rank1": int(selected_deeper),
                "pair_count": len(pair_equal_lifts),
                "pair_equal_lift_mean": _mean(pair_equal_lifts),
                "pair_equal_lift_standard_error": _sample_standard_error(pair_equal_lifts),
                "categories_with_positive_lift": sum(
                    int(_mean(values) > 0.0) for values in category_lifts.values()
                ),
                "category_count": len(category_lifts),
                "category_breakdown": category_breakdown,
            }
            if group_name == "both_wrong_top20_hit" and score_name == primary_signal:
                signal["minimum_absolute_rate"] = 0.184
                signal["minimum_absolute_rate_passed"] = bool(
                    signal["top1_pck_rate"] > 0.184
                )
                signal["minimum_lift_is_positive_beyond_1_96_pair_se"] = bool(
                    signal["pair_equal_lift_standard_error"] is not None
                    and signal["pair_equal_lift_mean"]
                    > 1.96 * signal["pair_equal_lift_standard_error"]
                )
                signal["recovery_rate_needed_for_75_pck"] = 0.329
                signal["recovery_count_needed_for_75_pck_on_493_points"] = 162
                signal["enter_matcher"] = bool(
                    signal["minimum_absolute_rate_passed"]
                    and signal["minimum_lift_is_positive_beyond_1_96_pair_se"]
                )
        group["mechanism_checks"] = mechanism_checks
        group["causal_contract"] = dict(contracts[0]) if contracts else {}
        summary[group_name] = group
    summary["decision_rule"] = {
        "primary_group": "both_wrong_top20_hit",
        "primary_signal": primary_signal,
        "minimum_absolute_top1_rate": 0.184,
        "significance_proxy": "pair_equal_lift_mean_gt_1.96_standard_errors",
        "matcher_requires_both": True,
    }
    return summary


def _summarize_counterfactual_fingerprint_audit(records):
    """Summarize multi-dose causal fingerprints with an oracle-gap falsifier."""

    audit_key = "counterfactual_fingerprint_audit"
    predicates = {
        "all": lambda row: True,
        "both_wrong_top20_hit": lambda row: bool(row.get("oracle_gap_case")),
        "attention_harms_native": lambda row: bool(row.get("attention_harms_native_case")),
        "attention_rescues_native": lambda row: bool(row.get("attention_rescues_native_case")),
    }
    signal_names = (
        "fingerprint_score",
        "fingerprint_mean_bidirectional",
        "fingerprint_reciprocity_error",
        "fingerprint_response_magnitude",
    )
    summary = {}
    for group_name, predicate in predicates.items():
        rows = [
            row for row in records
            if predicate(row) and isinstance(row.get(audit_key), dict)
        ]
        group = {"points": len(rows), "signals": {}}
        candidate_fraction = sum(
            sum(int(bool(candidate.get("pck_hit"))) for candidate in row[audit_key].get("candidates", []))
            / max(1, len(row[audit_key].get("candidates", [])))
            for row in rows
        ) / max(1, len(rows))
        group["candidate_pck_hit_fraction_mean"] = float(candidate_fraction)
        for signal_name in signal_names:
            # Recompute selected-hit directly from the candidate order.  This avoids
            # using target annotations in the fingerprint construction itself.
            selected_hits = []
            rank_improvements = []
            gt_beats_attention_top1 = 0
            for audit in (row[audit_key] for row in rows):
                candidates = audit.get("candidates", [])
                order = sorted(
                    range(len(candidates)),
                    key=lambda index: float(candidates[index].get("scores", {}).get(signal_name, -float("inf"))),
                    reverse=True,
                )
                selected_hits.append(bool(order) and bool(candidates[order[0]].get("pck_hit")))
                gt_indices = [index for index, candidate in enumerate(candidates) if candidate.get("pck_hit")]
                if gt_indices and order:
                    gt_index = gt_indices[0]
                    gt_rank = order.index(gt_index) + 1
                    attention_rank = int(candidates[gt_index].get("rank_attention", 1))
                    rank_improvements.append(float(attention_rank - gt_rank))
                    if float(candidates[gt_index].get("scores", {}).get(signal_name, -float("inf"))) > float(candidates[0].get("scores", {}).get(signal_name, -float("inf"))) + 1e-12:
                        gt_beats_attention_top1 += 1
            attention_correct = sum(int(bool(row.get("attention_top1_pck_hit"))) for row in rows)
            recovered = sum(int(hit and not bool(row.get("attention_top1_pck_hit"))) for row, hit in zip(rows, selected_hits))
            harmed = sum(int((not hit) and bool(row.get("attention_top1_pck_hit"))) for row, hit in zip(rows, selected_hits))
            group["signals"][signal_name] = {
                "top1_pck_rate": float(sum(selected_hits) / max(1, len(rows))),
                "recovers_attention_top1_errors": int(recovered),
                "harms_attention_top1_correct": int(harmed),
                "net_vs_attention_top1": int(recovered - harmed),
                "attention_top1_correct": int(attention_correct),
                "gt_candidate_average_rank_improvement": (
                    float(sum(rank_improvements) / len(rank_improvements))
                    if rank_improvements else None
                ),
                "gt_candidate_beats_attention_top1_count": int(gt_beats_attention_top1),
                "gt_candidate_rank_coverage": int(len(rank_improvements)),
                "oracle_gap_recovery_rate": (
                    float(recovered / max(1, len(rows) - attention_correct))
                    if group_name == "both_wrong_top20_hit" else None
                ),
            }
        summary[group_name] = group
    summary["decision_rule"] = {
        "primary_group": "both_wrong_top20_hit",
        "primary_signal": "fingerprint_score",
        "must_beat_uniform_candidate_expectation": True,
        "must_have_rescue_greater_than_harm": True,
        "prediction_changed": False,
        "gt_used_for_scoring": False,
    }
    return summary


def _summarize_persistent_candidate_slot_replay_audit(records):
    """Summarize identity signal and all six persistent-slot risk controls."""

    audit_key = "persistent_candidate_slot_replay_audit"
    summary = _summarize_ranked_candidate_score_audit(records, audit_key)
    predicates = {
        "all": lambda row: True,
        "both_wrong_top20_hit": lambda row: bool(row.get("oracle_gap_case")),
        "attention_harms_native": lambda row: bool(row.get("attention_harms_native_case")),
        "attention_rescues_native": lambda row: bool(row.get("attention_rescues_native_case")),
    }

    def _mean(values):
        return float(sum(values) / len(values)) if values else 0.0

    def _median(values):
        if not values:
            return 0.0
        values = sorted(float(value) for value in values)
        middle = len(values) // 2
        if len(values) % 2:
            return values[middle]
        return 0.5 * (values[middle - 1] + values[middle])

    def _collision(values_by_pair):
        collisions = total = 0
        for values in values_by_pair.values():
            valid = [value for value in values if value is not None]
            collisions += len(valid) - len(set(valid))
            total += len(valid)
        return int(collisions), float(collisions / total) if total else 0.0

    for group_name, predicate in predicates.items():
        rows = [
            row for row in records
            if predicate(row) and isinstance(row.get(audit_key), dict)
        ]
        base_name = "oracle_gap" if group_name == "both_wrong_top20_hit" else group_name
        group = summary.pop(base_name, None) or {"points": len(rows), "signals": {}}
        audits = [row[audit_key] for row in rows]
        diagnostics = [audit.get("diagnostics", {}) for audit in audits]
        contracts = [audit.get("persistent_slot_contract", {}) for audit in audits]
        score_names = list(audits[0].get("score_names", [])) if audits else []
        primary_signal = (
            "directional_anchor_cosine"
            if "directional_anchor_cosine" in score_names
            else (score_names[0] if score_names else "")
        )
        hit_fractions = [
            float(sum(int(bool(candidate.get("pck_hit"))) for candidate in audit.get("candidates", [])))
            / max(1, len(audit.get("candidates", [])))
            for audit in audits
        ]
        random_expectation = _mean(hit_fractions)
        attention_top1_correct = sum(
            int(bool(row.get("attention_top1_pck_hit"))) for row in rows
        )
        signal_checks = {}
        primary_pixels_by_pair = {}
        attention_pixels_by_pair = {}
        for score_name in score_names:
            selected_hits = []
            recovered = harmed = 0
            for row, audit in zip(rows, audits):
                candidates = list(audit.get("candidates", []))
                selected_index = (
                    max(
                        range(len(candidates)),
                        key=lambda index: float(
                            candidates[index].get("scores", {}).get(score_name, -float("inf"))
                        ),
                    )
                    if candidates
                    else 0
                )
                selected = candidates[selected_index] if candidates else {}
                hit = int(bool(selected.get("pck_hit")))
                selected_hits.append(hit)
                recovered += int(not bool(row.get("attention_top1_pck_hit")) and bool(hit))
                harmed += int(bool(row.get("attention_top1_pck_hit")) and not bool(hit))
                if score_name == primary_signal:
                    pair_key = (
                        str(row.get("category", "")),
                        str(row.get("pair_json", "")),
                        str(row.get("src_image", "")),
                        str(row.get("trg_image", "")),
                    )
                    primary_pixels_by_pair.setdefault(pair_key, []).append(
                        selected.get("pixel_index")
                    )
                    attention_rank = int(
                        audit.get("diagnostics", {}).get("selected_attention_rank") or 1
                    )
                    attention_index = max(0, min(len(candidates) - 1, attention_rank - 1))
                    attention_pixels_by_pair.setdefault(pair_key, []).append(
                        candidates[attention_index].get("pixel_index") if candidates else None
                    )
            top1 = sum(selected_hits)
            signal_checks[score_name] = {
                "top1_pck_hits": int(top1),
                "top1_pck_rate": float(top1 / len(rows)) if rows else 0.0,
                "lift_over_uniform_candidate_expectation": (
                    float(top1 / len(rows) - random_expectation) if rows else 0.0
                ),
                "recovers_attention_top1_errors": int(recovered),
                "harms_attention_top1_correct": int(harmed),
                "net_vs_attention_top1": int(recovered - harmed),
            }
        primary_collisions, primary_collision_rate = _collision(primary_pixels_by_pair)
        attention_collisions, attention_collision_rate = _collision(attention_pixels_by_pair)
        source_deltas = []
        target_deltas = []
        source_masses = []
        target_masses = []
        for audit in audits:
            for candidate in audit.get("candidates", []):
                item = candidate.get("diagnostics", {})
                source_deltas.append(float(item.get("source_relative_delta", 0.0)))
                target_deltas.append(float(item.get("target_relative_delta", 0.0)))
                source_masses.append(float(item.get("source_cross_mass", 0.0)))
                target_masses.append(float(item.get("target_cross_mass", 0.0)))
        metadata = dict(contracts[0]) if contracts else {}
        mechanism_checks = {
            "primary_signal": primary_signal,
            "candidate_pck_hit_fraction_mean": random_expectation,
            "attention_top1_correct": int(attention_top1_correct),
            "signals": signal_checks,
            "source_slot_divergence_mean": _mean([
                float(item.get("source_slot_divergence") or 0.0) for item in diagnostics
            ]),
            "target_slot_divergence_mean": _mean([
                float(item.get("target_slot_divergence") or 0.0) for item in diagnostics
            ]),
            "source_slot_near_identical_rate_at_1e_4": _mean([
                float(float(item.get("source_slot_divergence") or 0.0) <= 1e-4)
                for item in diagnostics
            ]),
            "target_slot_near_identical_rate_at_1e_4": _mean([
                float(float(item.get("target_slot_divergence") or 0.0) <= 1e-4)
                for item in diagnostics
            ]),
            "source_cross_mass_mean": _mean(source_masses),
            "target_cross_mass_mean": _mean(target_masses),
            "source_relative_delta_median": _median(source_deltas),
            "target_relative_delta_median": _median(target_deltas),
            "selected_target_collision_count": primary_collisions,
            "selected_target_collision_rate": primary_collision_rate,
            "attention_target_collision_count": attention_collisions,
            "attention_target_collision_rate": attention_collision_rate,
            "candidate_missing_gt_count": sum(
                int(bool(item.get("candidate_missing_gt"))) for item in diagnostics
            ),
            "candidate_missing_gt_rate": _mean([
                float(bool(item.get("candidate_missing_gt"))) for item in diagnostics
            ]),
            "native_candidate_injected_count": sum(
                int(bool(contract.get("native_candidate_injected"))) for contract in contracts
            ),
            "native_fallback_used_count": sum(
                int(bool(contract.get("native_fallback_used"))) for contract in contracts
            ),
            "gt_used_for_scoring_count": sum(
                int(bool(contract.get("gt_used_for_scoring"))) for contract in contracts
            ),
        }
        group["mechanism_checks"] = mechanism_checks
        group["persistent_slot_contract"] = metadata
        group["risk_controls"] = {
            "compute": {
                "strategy": "chunked_hypothesis_batch",
                "candidate_multiplier": int(metadata.get("candidate_count", 0)),
                "hypothesis_chunk": int(metadata.get("hypothesis_chunk", 0)),
                "max_branch_batch": int(metadata.get("max_branch_batch", 0)),
                "replay_seconds_per_pair": float(metadata.get("replay_seconds", 0.0)),
                "hypotheses_per_second": float(metadata.get("hypotheses_per_second", 0.0)),
            },
            "slot_isolation": {
                "candidate_axis_persisted_across_blocks": bool(
                    metadata.get("candidate_axis_persisted_across_blocks")
                ),
                "local_self_attention_preserved": bool(metadata.get("local_self_attention_preserved")),
                "candidate_cross_key_count": int(metadata.get("candidate_cross_key_count", 0)),
            },
            "artificial_intervention": {
                "original_cross_mass_used": bool(metadata.get("original_cross_mass_used")),
                "cross_mass_denominator": metadata.get("cross_mass_denominator"),
                "unit_cross_attention_forced": bool(metadata.get("unit_cross_attention_forced")),
                "primary_signal_is_directional_native_anchor": primary_signal == "directional_anchor_cosine",
                "source_relative_delta_median": mechanism_checks["source_relative_delta_median"],
                "target_relative_delta_median": mechanism_checks["target_relative_delta_median"],
            },
            "global_competition": {
                "selected_target_collision_rate": primary_collision_rate,
                "attention_target_collision_rate": attention_collision_rate,
                "no_assignment_solver_used": True,
            },
            "candidate_coverage": {
                "candidate_missing_gt_rate": mechanism_checks["candidate_missing_gt_rate"],
            },
            "prior_causal_failure": {
                "candidate_clamped_control_present_in_this_run": any(
                    isinstance(row.get("candidate_clamped_causal_replay_audit"), dict)
                    for row in rows
                ),
                "current_signal_must_beat_uniform_expectation": True,
                "current_signal_must_outperform_candidate_clamped_control": True,
            },
        }
        summary[group_name] = group
    summary["decision_rule"] = {
        "primary_group": "both_wrong_top20_hit",
        "primary_signal": "directional_anchor_cosine",
        "must_beat": "uniform_candidate_hit_expectation",
        "must_not_use_native_fallback": True,
        "must_report_slot_divergence": True,
        "must_report_target_collisions": True,
        "prediction_changed": False,
    }
    return summary


def _fjsar_mode_config(matcher):
    mode = {
        "fjsar_attn": "attention",
        "fjsar_attention_signature": "attention_signature",
        "fjsar_part_sharpen": "part_sharpen",
        "fjsar_orthogonal_context": "orthogonal_context",
        "fjsar_spectral_identity": "spectral_identity",
        "fjsar_filtered_spectral_kernel": "filtered_spectral_kernel",
        "fjsar_transport_lift": "transport_lift",
        "fjsar_basin_contrastive_identity": "basin_contrastive_identity",
        "fjsar_attention_isometry": "attention_isometry",
        "fjsar_identity_preserving_attention": "identity_preserving_attention",
        "fjsar_balanced_transport_attention": "balanced_transport_attention",
        "fjsar_qk_identity_attention": "qk_identity_attention",
        "fjsar_cross_attention_trajectory": "cross_attention_trajectory",
        "fjsar_native_preserving_topology_rescue": "native_preserving_topology_rescue",
        "fjsar_attention_basin_native_refine": "attention_basin_native_refine",
        "fjsar_candidate_graph_consensus_verification": "candidate_graph_consensus_verification",
        "fjsar_geometry_consistent_attention": "geometry_consistent_attention",
        "fjsar_candidate_conditioned_verification": "candidate_conditioned_verification",
        "fjsar_candidate_local_transport_verification": "candidate_local_transport_verification",
        "fjsar_attention_relational_graph_matching": "attention_relational_graph_matching",
        "fjsar_dense_partial_graph_matching": "dense_partial_graph_matching",
        "fjsar_expert_preserving_attention_hypothesis_conditioned_replay": (
            "expert_preserving_attention_hypothesis_conditioned_replay"
        ),
        "fjsar_pre_softmax_channelwise_identity": "pre_softmax_channelwise_identity",
        "fjsar_layer_routed_identity": "layer_routed_identity",
        "fjsar_pre_single_stream_identity": "pre_single_stream_identity",
    }[matcher]
    # Multi-method audits must share the same attention tensor.  Use the exact
    # cross-image replay without coordinate bias so every method is compared on
    # the same pair-conditioned evidence.
    if matcher == "fjsar_geometry_consistent_attention":
        return mode, "geometry_consistent", False
    if matcher == "fjsar_identity_preserving_attention":
        return mode, "identity_preserving", False
    if matcher == "fjsar_balanced_transport_attention":
        return mode, "balanced_transport", False
    if matcher == "fjsar_qk_identity_attention":
        return mode, "qk_identity", False
    if matcher == "fjsar_cross_attention_trajectory":
        return mode, "trajectory", False
    if matcher in FJSAR_ALL_MATCHERS:
        return mode, "exact", False
    raise ValueError(f"unsupported FJSAR matcher: {matcher}")


def _empty_fjsar_eval_stats():
    return {
        "image_baseline": [],
        "image_method": [],
        "baseline_correct": 0,
        "method_correct": 0,
        "total": 0,
        "changed": 0,
        "improved": 0,
        "harmed": 0,
        "pair_count": 0,
        "diagnostic_sums": {},
        "model_counts": {},
    }


def _fjsar_numeric_diagnostic_keys():
    return (
        "mean_cross_mass_source",
        "mean_residual_confidence_source",
        "native_parity_cosine",
        "mean_position_confidence_source",
        "replay_depth",
        "mean_reciprocal_source",
        "mean_concentration_source",
        "mean_cross_excess_source",
        "raw_boundary_parity_cosine",
        "prepared_feature_parity_cosine",
        "mean_cycle_error_source",
        "geometry_radius",
        "geometry_strength",
        "mean_geometry_support_source",
        "joint_native_cosine_source",
        "native_intra_cosine_source",
        "joint_intra_cosine_source",
        "mean_balanced_transport_row_error",
        "mean_balanced_transport_col_error",
        "mean_qk_fisher_ratio",
        "mean_trajectory_layer_count",
        "mean_trajectory_top1_stability",
        "topology_rescue_candidate_pool_mean",
        "topology_rescue_native_keep_rate",
        "topology_rescue_rescue_rate",
        "topology_rescue_native_confidence_mean",
        "topology_rescue_selected_support_mean",
        "candidate_local_transport_candidate_pool_mean",
        "candidate_local_transport_native_selected_rate",
        "candidate_local_transport_rescue_rate",
        "candidate_local_transport_abstained_challenger_rate",
        "candidate_local_transport_native_rank_mean",
        "candidate_local_transport_selected_attention_rank_mean",
        "candidate_local_transport_anchor_confidence_mean",
        "candidate_local_transport_margin_over_native_mean",
        "candidate_local_transport_dominance_win_fraction_mean",
        "candidate_graph_consensus_candidate_pool_mean",
        "candidate_graph_consensus_native_selected_rate",
        "candidate_graph_consensus_rescue_rate",
        "candidate_graph_consensus_iteration_mean",
        "candidate_graph_consensus_selected_confidence_mean",
        "candidate_graph_consensus_consensus_margin_mean",
        "candidate_graph_consensus_local_transport_native_selected_rate",
        "candidate_graph_consensus_local_transport_rescue_rate",
        "candidate_graph_consensus_local_transport_anchor_confidence_mean",
        "candidate_graph_consensus_local_transport_selected_attention_rank_mean",
        "parity_ok",
    )


def _accumulate_fjsar_eval_stats(stats, baseline_predictions, method_predictions, target_points, threshold, diagnostics):
    baseline_pair = method_pair = 0
    for baseline, method, target in zip(baseline_predictions, method_predictions, target_points):
        baseline_ok = _pck(baseline, target, threshold)
        method_ok = _pck(method, target, threshold)
        baseline_flag = int(baseline_ok)
        method_flag = int(method_ok)
        baseline_pair += baseline_flag
        method_pair += method_flag
        stats["baseline_correct"] += baseline_flag
        stats["method_correct"] += method_flag
        stats["total"] += 1
        if list(baseline) != list(method):
            stats["changed"] += 1
        if method_flag > baseline_flag:
            stats["improved"] += 1
        elif baseline_flag > method_flag:
            stats["harmed"] += 1
    denom = len(target_points)
    stats["image_baseline"].append(baseline_pair / denom if denom else 0.0)
    stats["image_method"].append(method_pair / denom if denom else 0.0)
    stats["pair_count"] += 1
    for key in _fjsar_numeric_diagnostic_keys():
        if key in diagnostics:
            stats["diagnostic_sums"][key] = stats["diagnostic_sums"].get(key, 0.0) + float(diagnostics[key])
    _merge_counts(stats["model_counts"], diagnostics.get("model_counts", {}))


def _fjsar_oracle_rates(model_counts):
    total = int(model_counts.get("fjsar_oracle_total", 0))
    if total <= 0:
        return {}
    return {
        key: float(value) / float(total)
        for key, value in model_counts.items()
        if key != "fjsar_oracle_total"
    }


def _finalize_fjsar_eval_stats(stats):
    total = int(stats["total"])
    pair_count = int(stats["pair_count"])
    baseline_point = stats["baseline_correct"] / total if total else 0.0
    method_point = stats["method_correct"] / total if total else 0.0
    diagnostic_sums = stats["diagnostic_sums"]
    diagnostics = {
        "mean_pair_reliability": (
            diagnostic_sums.get("mean_cross_mass_source", 0.0) / pair_count
            if pair_count else 0.0
        ),
        "mean_local_reliability": (
            diagnostic_sums.get("mean_residual_confidence_source", 0.0) / pair_count
            if pair_count else 0.0
        ),
        "mean_split_agreement": (
            diagnostic_sums.get("native_parity_cosine", 0.0) / pair_count
            if pair_count else 0.0
        ),
        "mean_valid_query_rate": (
            diagnostic_sums.get("mean_position_confidence_source", 0.0) / pair_count
            if pair_count else 0.0
        ),
        "mean_local_anchor_count": (
            diagnostic_sums.get("replay_depth", 0.0) / pair_count
            if pair_count else 0.0
        ),
        "mean_decision_margin": (
            diagnostic_sums.get("mean_reciprocal_source", 0.0) / pair_count
            if pair_count else 0.0
        ),
        "mean_graph_informative_rate": (
            diagnostic_sums.get("mean_concentration_source", 0.0) / pair_count
            if pair_count else 0.0
        ),
        "mean_cross_excess": (
            diagnostic_sums.get("mean_cross_excess_source", 0.0) / pair_count
            if pair_count else 0.0
        ),
        "fjsar_raw_boundary_parity_cosine": (
            diagnostic_sums.get("raw_boundary_parity_cosine", 0.0) / pair_count
            if pair_count else 0.0
        ),
        "fjsar_prepared_feature_parity_cosine": (
            diagnostic_sums.get("prepared_feature_parity_cosine", 0.0) / pair_count
            if pair_count else 0.0
        ),
        "fjsar_mean_cycle_error": (
            diagnostic_sums.get("mean_cycle_error_source", 0.0) / pair_count
            if pair_count else 0.0
        ),
        "fjsar_geometry_radius": (
            diagnostic_sums.get("geometry_radius", 0.0) / pair_count
            if pair_count else 0.0
        ),
        "fjsar_geometry_strength": (
            diagnostic_sums.get("geometry_strength", 0.0) / pair_count
            if pair_count else 0.0
        ),
        "fjsar_mean_geometry_support": (
            diagnostic_sums.get("mean_geometry_support_source", 0.0) / pair_count
            if pair_count else 0.0
        ),
        "fjsar_joint_native_cosine": (
            diagnostic_sums.get("joint_native_cosine_source", 0.0) / pair_count
            if pair_count else 0.0
        ),
        "fjsar_balanced_transport_row_error": (
            diagnostic_sums.get("mean_balanced_transport_row_error", 0.0) / pair_count
            if pair_count else 0.0
        ),
        "fjsar_balanced_transport_col_error": (
            diagnostic_sums.get("mean_balanced_transport_col_error", 0.0) / pair_count
            if pair_count else 0.0
        ),
        "fjsar_qk_fisher_ratio": (
            diagnostic_sums.get("mean_qk_fisher_ratio", 0.0) / pair_count
            if pair_count else 0.0
        ),
        "fjsar_trajectory_layer_count": (
            diagnostic_sums.get("mean_trajectory_layer_count", 0.0) / pair_count
            if pair_count else 0.0
        ),
        "fjsar_trajectory_top1_stability": (
            diagnostic_sums.get("mean_trajectory_top1_stability", 0.0) / pair_count
            if pair_count else 0.0
        ),
        "fjsar_topology_rescue_candidate_pool_mean": (
            diagnostic_sums.get("topology_rescue_candidate_pool_mean", 0.0) / pair_count
            if pair_count else 0.0
        ),
        "fjsar_topology_rescue_native_keep_rate": (
            diagnostic_sums.get("topology_rescue_native_keep_rate", 0.0) / pair_count
            if pair_count else 0.0
        ),
        "fjsar_topology_rescue_rescue_rate": (
            diagnostic_sums.get("topology_rescue_rescue_rate", 0.0) / pair_count
            if pair_count else 0.0
        ),
        "fjsar_topology_rescue_native_confidence_mean": (
            diagnostic_sums.get("topology_rescue_native_confidence_mean", 0.0) / pair_count
            if pair_count else 0.0
        ),
        "fjsar_topology_rescue_selected_support_mean": (
            diagnostic_sums.get("topology_rescue_selected_support_mean", 0.0) / pair_count
            if pair_count else 0.0
        ),
        "fjsar_candidate_local_transport_candidate_pool_mean": (
            diagnostic_sums.get("candidate_local_transport_candidate_pool_mean", 0.0) / pair_count
            if pair_count else 0.0
        ),
        "fjsar_candidate_local_transport_native_selected_rate": (
            diagnostic_sums.get("candidate_local_transport_native_selected_rate", 0.0) / pair_count
            if pair_count else 0.0
        ),
        "fjsar_candidate_local_transport_rescue_rate": (
            diagnostic_sums.get("candidate_local_transport_rescue_rate", 0.0) / pair_count
            if pair_count else 0.0
        ),
        "fjsar_candidate_local_transport_abstained_challenger_rate": (
            diagnostic_sums.get("candidate_local_transport_abstained_challenger_rate", 0.0) / pair_count
            if pair_count else 0.0
        ),
        "fjsar_candidate_local_transport_native_rank_mean": (
            diagnostic_sums.get("candidate_local_transport_native_rank_mean", 0.0) / pair_count
            if pair_count else 0.0
        ),
        "fjsar_candidate_local_transport_selected_attention_rank_mean": (
            diagnostic_sums.get("candidate_local_transport_selected_attention_rank_mean", 0.0) / pair_count
            if pair_count else 0.0
        ),
        "fjsar_candidate_local_transport_anchor_confidence_mean": (
            diagnostic_sums.get("candidate_local_transport_anchor_confidence_mean", 0.0) / pair_count
            if pair_count else 0.0
        ),
        "fjsar_candidate_local_transport_margin_over_native_mean": (
            diagnostic_sums.get("candidate_local_transport_margin_over_native_mean", 0.0) / pair_count
            if pair_count else 0.0
        ),
        "fjsar_candidate_local_transport_dominance_win_fraction_mean": (
            diagnostic_sums.get("candidate_local_transport_dominance_win_fraction_mean", 0.0) / pair_count
            if pair_count else 0.0
        ),
        "fjsar_candidate_graph_consensus_candidate_pool_mean": (
            diagnostic_sums.get("candidate_graph_consensus_candidate_pool_mean", 0.0) / pair_count
            if pair_count else 0.0
        ),
        "fjsar_candidate_graph_consensus_native_selected_rate": (
            diagnostic_sums.get("candidate_graph_consensus_native_selected_rate", 0.0) / pair_count
            if pair_count else 0.0
        ),
        "fjsar_candidate_graph_consensus_rescue_rate": (
            diagnostic_sums.get("candidate_graph_consensus_rescue_rate", 0.0) / pair_count
            if pair_count else 0.0
        ),
        "fjsar_candidate_graph_consensus_iteration_mean": (
            diagnostic_sums.get("candidate_graph_consensus_iteration_mean", 0.0) / pair_count
            if pair_count else 0.0
        ),
        "fjsar_candidate_graph_consensus_selected_confidence_mean": (
            diagnostic_sums.get("candidate_graph_consensus_selected_confidence_mean", 0.0) / pair_count
            if pair_count else 0.0
        ),
        "fjsar_candidate_graph_consensus_consensus_margin_mean": (
            diagnostic_sums.get("candidate_graph_consensus_consensus_margin_mean", 0.0) / pair_count
            if pair_count else 0.0
        ),
        "fjsar_candidate_graph_consensus_local_transport_native_selected_rate": (
            diagnostic_sums.get("candidate_graph_consensus_local_transport_native_selected_rate", 0.0) / pair_count
            if pair_count else 0.0
        ),
        "fjsar_candidate_graph_consensus_local_transport_rescue_rate": (
            diagnostic_sums.get("candidate_graph_consensus_local_transport_rescue_rate", 0.0) / pair_count
            if pair_count else 0.0
        ),
        "fjsar_candidate_graph_consensus_local_transport_anchor_confidence_mean": (
            diagnostic_sums.get("candidate_graph_consensus_local_transport_anchor_confidence_mean", 0.0) / pair_count
            if pair_count else 0.0
        ),
        "fjsar_candidate_graph_consensus_local_transport_selected_attention_rank_mean": (
            diagnostic_sums.get("candidate_graph_consensus_local_transport_selected_attention_rank_mean", 0.0) / pair_count
            if pair_count else 0.0
        ),
        "fjsar_collapse_delta": (
            (
                diagnostic_sums.get("joint_intra_cosine_source", 0.0)
                - diagnostic_sums.get("native_intra_cosine_source", 0.0)
            ) / pair_count
            if pair_count else 0.0
        ),
        "fjsar_parity_failure_rate": (
            1.0 - diagnostic_sums.get("parity_ok", 0.0) / pair_count
            if pair_count else 0.0
        ),
        "model_counts": stats["model_counts"],
        "model_count_rates": _fjsar_oracle_rates(stats["model_counts"]),
    }
    return {
        "baseline_image": 100.0 * float(np.mean(stats["image_baseline"])) if stats["image_baseline"] else 0.0,
        "method_image": 100.0 * float(np.mean(stats["image_method"])) if stats["image_method"] else 0.0,
        "baseline_point": 100.0 * baseline_point,
        "method_point": 100.0 * method_point,
        "point_gain": 100.0 * (method_point - baseline_point),
        "pair_count": pair_count,
        "changed_count": int(stats["changed"]),
        "improved_count": int(stats["improved"]),
        "harmed_count": int(stats["harmed"]),
        "improvement_harm_ratio": (
            float(stats["improved"]) / float(stats["harmed"])
            if stats["harmed"] else None
        ),
        "intervention_rate": (
            float(stats["changed"]) / float(total) if total else 0.0
        ),
        "matcher_diagnostics": diagnostics,
    }


def _load_pairs(dataset_path):
    test_path = "PairAnnotation/test"
    json_list = os.listdir(os.path.join(dataset_path, test_path))
    all_cats = os.listdir(os.path.join(dataset_path, "JPEGImages"))
    cat2json = {
        cat: [name for name in json_list if cat in name]
        for cat in all_cats
    }
    cat2img = {}
    for cat in all_cats:
        cat2img[cat] = []
        for json_path in cat2json[cat]:
            with open(os.path.join(dataset_path, test_path, json_path)) as handle:
                data = json.load(handle)
            for image_name in (data["src_imname"], data["trg_imname"]):
                if image_name not in cat2img[cat]:
                    cat2img[cat].append(image_name)
    return test_path, all_cats, cat2json, cat2img


def _partition_category_pairs(pair_names, pairs_per_cat, split_seed):
    if pairs_per_cat < 1:
        raise ValueError("--pairs_per_cat must be positive")
    ordered = sorted(
        pair_names,
        key=lambda name: hashlib.sha256(f"{split_seed}:{name}".encode("utf-8")).digest(),
    )
    required = 2 * pairs_per_cat
    if len(ordered) < required:
        raise ValueError(f"Need at least {required} pairs for discovery/heldout split, got {len(ordered)}")
    return {
        "discovery": ordered[:pairs_per_cat],
        "heldout": ordered[pairs_per_cat:required],
    }


def _select_category_pairs(pair_names, args):
    if args.subset == "all":
        return pair_names
    return _partition_category_pairs(pair_names, args.pairs_per_cat, args.split_seed)[args.subset]


def _move_nested_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return [_move_nested_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_nested_to_device(item, device) for item in value)
    return value


def _ada_shift_scale_for_eval(ada, device):
    ada = _move_nested_to_device(ada, device)
    if isinstance(ada, torch.Tensor):
        if ada.ndim >= 3:
            return ada[0][0], ada[0][1]
        if ada.ndim == 2 and ada.shape[0] == 2:
            return ada[0], ada[1]
    if isinstance(ada, (list, tuple)) and ada:
        first = ada[0]
        if isinstance(first, torch.Tensor) and first.ndim >= 2 and first.shape[0] >= 2:
            return first[0], first[1]
        if isinstance(first, (list, tuple)) and len(first) >= 2:
            return first[0], first[1]
        if len(ada) >= 2:
            return ada[0], ada[1]
    raise ValueError("Unsupported AdaLN cache layout")


def _prepare_feature_tensors(ft_raw, ada, args, pre_norm, device):
    ft_raw = ft_raw.to(device).clone()
    _, _, height, width = ft_raw.shape
    if args.cd:
        for channel in (154, 1446):
            if channel < ft_raw.shape[1]:
                ft_raw[:, channel, :, :] = 0.0
    ft = rearrange(ft_raw, "b c h w -> b (h w) c")
    ft = pre_norm(ft)
    ft = rearrange(ft, "b (h w) c -> b c h w", h=height, w=width)
    shift_vec, scale_vec = _ada_shift_scale_for_eval(ada, device)
    shift = shift_vec.unsqueeze(0).unsqueeze(2).unsqueeze(3)
    scale = scale_vec.unsqueeze(0).unsqueeze(2).unsqueeze(3)
    return (1 + scale) * ft + shift


def _prepare_features(output_dict, ada_dict, image_name, args, pre_norm, device):
    return _prepare_feature_tensors(
        output_dict[image_name], ada_dict[image_name], args, pre_norm, device
    )


def _multiblock_entry_value(mapping, block):
    if not isinstance(mapping, dict):
        return None
    key = str(int(block))
    if key in mapping:
        return mapping[key]
    return mapping.get(int(block))


def _prepare_fjsar_multilayer_descriptor_maps(src_entry, trg_entry, args, pre_norm, device):
    blocks = _fjsar_multilayer_blocks(args)
    if not blocks:
        return None
    src_features = src_entry.get("multiblock_features", {})
    src_ada = src_entry.get("multiblock_ada", {})
    trg_features = trg_entry.get("multiblock_features", {})
    trg_ada = trg_entry.get("multiblock_ada", {})
    descriptor_maps = OrderedDict()
    for block in blocks:
        src_feature = _multiblock_entry_value(src_features, block)
        src_ada_value = _multiblock_entry_value(src_ada, block)
        trg_feature = _multiblock_entry_value(trg_features, block)
        trg_ada_value = _multiblock_entry_value(trg_ada, block)
        if src_feature is None or src_ada_value is None or trg_feature is None or trg_ada_value is None:
            raise RuntimeError(
                "FJSAR multilayer identity audit requires multiblock cache entries; "
                f"missing block {int(block)}"
            )
        descriptor_maps[f"official_block{int(block)}"] = (
            _prepare_feature_tensors(src_feature, src_ada_value, args, pre_norm, device),
            _prepare_feature_tensors(trg_feature, trg_ada_value, args, pre_norm, device),
        )
    return descriptor_maps


def _prepare_fjsar_pre_single_stream_descriptor_maps(src_entry, trg_entry, args, device):
    """Prepare double-stream image features without inventing an AdaLN state.

    ``DoubleStreamBlock.forward_feat`` returns its image feature after the
    block's real norm2 and modulation, so only the benchmark's channel discard
    is applied here.  Candidate scoring performs its own per-branch L2 norm.
    """
    blocks = _fjsar_multilayer_blocks(args)
    src_features = src_entry.get("multiblock_features", {})
    trg_features = trg_entry.get("multiblock_features", {})
    descriptor_maps = OrderedDict()
    for block in blocks:
        src_feature = _multiblock_entry_value(src_features, block)
        trg_feature = _multiblock_entry_value(trg_features, block)
        if src_feature is None or trg_feature is None:
            raise RuntimeError(
                "FJSAR pre-single-stream identity requires double-stream cache entries; "
                f"missing block {int(block)}"
            )
        src_feature = src_feature.to(device).clone()
        trg_feature = trg_feature.to(device).clone()
        if args.cd:
            for channel in (154, 1446):
                if channel < src_feature.shape[1]:
                    src_feature[:, channel, :, :] = 0.0
                if channel < trg_feature.shape[1]:
                    trg_feature[:, channel, :, :] = 0.0
        descriptor_maps[f"pre_single_block{int(block)}"] = (
            torch.nan_to_num(src_feature, nan=0.0, posinf=0.0, neginf=0.0),
            torch.nan_to_num(trg_feature, nan=0.0, posinf=0.0, neginf=0.0),
        )
    return descriptor_maps


def _make_fjsar_capture(args, fjsar_model):
    feature_block = int(args.k[0] if isinstance(args.k, list) else args.k)
    trajectory_blocks = _fjsar_trajectory_blocks(args)
    if trajectory_blocks:
        capture_indices = [
            replay_start_block_index(int(block))
            for block in trajectory_blocks
        ]
        return FluxMultiPreBlockCapture(fjsar_model, capture_indices)
    return FluxPreBlockCapture(fjsar_model, replay_start_block_index(feature_block))


def _make_fjsar_trajectory_block_modules(args, fjsar_model):
    trajectory_blocks = _fjsar_trajectory_blocks(args)
    if not trajectory_blocks:
        return None
    return {
        str(int(block)): select_flux_single_blocks(fjsar_model, int(block), depth=1)[0]
        for block in trajectory_blocks
    }


def _entry_trajectory_pair(src_entry, trg_entry, args):
    trajectory_blocks = _fjsar_trajectory_blocks(args)
    if not trajectory_blocks:
        return None
    src_states = src_entry.get("trajectory_replay_states")
    trg_states = trg_entry.get("trajectory_replay_states")
    if not isinstance(src_states, dict) or not isinstance(trg_states, dict):
        raise RuntimeError("FJSAR trajectory matcher requires trajectory replay states")
    return {"src": src_states, "trg": trg_states}


def _multi_timestep_attention_identity_audit_for_points(
    *,
    args,
    dataset_path,
    category,
    src_image_name,
    trg_image_name,
    src_caption,
    trg_caption,
    source_points,
    target_points,
    source_size,
    target_size,
    threshold,
    featurizer,
    capture,
    memory_cache,
    blocks,
    interaction_mode,
    use_coordinate_bias,
    candidate_topk,
    device,
    pre_norm,
    discard_channels=(),
    calibration=None,
):
    """Collect a lightweight multi-timestep trajectory for the same pair."""

    timesteps = _fjsar_multi_timestep_values(args)
    if not timesteps:
        return None
    if len(source_points) != len(target_points):
        raise ValueError("multi-timestep audit requires aligned source/target points")

    timestep_rows: list[list[dict[str, Any]]] = [[] for _ in source_points]
    reference_timestep = int(args.t) if int(args.t) in timesteps else int(timesteps[-1])
    shared_seed = _fjsar_multi_timestep_seed(args)

    for timestep in timesteps:
        timestep_args = argparse.Namespace(**vars(args))
        timestep_args.t = int(timestep)
        timestep_args.fjsar_disk_cache_path = ""
        timestep_args.fjsar_require_disk_cache = False
        timestep_args.fjsar_shared_noise = False
        _reset_all_rng(shared_seed)
        src_entry = _get_flux_fjsar_entry(
            dataset_path,
            category,
            src_image_name,
            src_caption,
            timestep_args,
            featurizer,
            capture,
            None,
        )
        _reset_all_rng(shared_seed)
        trg_entry = _get_flux_fjsar_entry(
            dataset_path,
            category,
            trg_image_name,
            trg_caption,
            timestep_args,
            featurizer,
            capture,
            None,
        )
        src_ft = _prepare_feature_tensors(
            src_entry["feature"], src_entry["ada"], timestep_args, pre_norm, device
        )
        trg_ft = _prepare_feature_tensors(
            trg_entry["feature"], trg_entry["ada"], timestep_args, pre_norm, device
        )
        src_full = nn.Upsample(size=source_size, mode="bilinear")(src_ft.to(torch.float16))
        trg_full = nn.Upsample(size=target_size, mode="bilinear")(trg_ft.to(torch.float16))
        baseline_predictions = cosine_nn_predict(src_full, trg_full, source_points)
        rows = flux_fjsar_dump_candidates(
            src_ft,
            trg_ft,
            source_points,
            source_size,
            target_size,
            src_replay_state=src_entry["replay_state"],
            trg_replay_state=trg_entry["replay_state"],
            blocks=blocks,
            interaction_mode=interaction_mode,
            use_coordinate_bias=use_coordinate_bias,
            src_ada=src_entry["ada"],
            trg_ada=trg_entry["ada"],
            discard_channels=discard_channels,
            calibration=calibration,
            target_points=target_points,
            pck_threshold=threshold,
            candidate_topk=candidate_topk,
        )
        if len(rows) != len(timestep_rows):
            raise RuntimeError("multi-timestep audit returned mismatched candidate rows")
        for row_index, row in enumerate(rows):
            baseline_hit = _pck(
                baseline_predictions[row_index],
                target_points[row_index],
                threshold,
            )
            attention_top1 = row.get("attention_top1", {})
            attention_top1_hit = bool(attention_top1.get("pck_hit"))
            attention_topk_hit = bool(row.get("attention_topk_pck_hit"))
            attention_gt_rank = row.get("gt_ranks", {}).get("attention")
            timestep_rows[row_index].append({
                "timestep": int(timestep),
                "baseline_pck_hit": bool(baseline_hit),
                "attention_top1_pck_hit": attention_top1_hit,
                "attention_topk_pck_hit": attention_topk_hit,
                "attention_gt_rank": (
                    int(attention_gt_rank) if attention_gt_rank is not None else None
                ),
                "attention_top1_pixel": attention_top1.get("pixel"),
                "oracle_gap_case": bool(
                    (not baseline_hit)
                    and attention_topk_hit
                    and not attention_top1_hit
                ),
                "attention_harms_native_case": bool(
                    baseline_hit and not attention_top1_hit
                ),
                "attention_rescues_native_case": bool(
                    (not baseline_hit) and attention_top1_hit
                ),
            })

    audits = []
    for rows in timestep_rows:
        ranks = [int(item["attention_gt_rank"]) for item in rows if item["attention_gt_rank"] is not None]
        top1_pixels = [
            tuple(item["attention_top1_pixel"])
            for item in rows
            if item.get("attention_top1_pixel") is not None
        ]
        top1_counts = Counter(top1_pixels)
        reference_row = next((item for item in rows if int(item["timestep"]) == int(reference_timestep)), None)
        summary = {
            "reference_timestep": int(reference_timestep),
            "timesteps": [int(value) for value in timesteps],
            "timestep_count": int(len(timesteps)),
            "candidate_topk": int(candidate_topk),
            "baseline_hit_count": int(sum(1 for item in rows if item["baseline_pck_hit"])),
            "attention_top1_hit_count": int(sum(1 for item in rows if item["attention_top1_pck_hit"])),
            "attention_topk_hit_count": int(sum(1 for item in rows if item["attention_topk_pck_hit"])),
            "attention_gt_rank_mean": (
                float(sum(ranks) / len(ranks)) if ranks else None
            ),
            "attention_gt_rank_best": (int(min(ranks)) if ranks else None),
            "attention_gt_rank_worst": (int(max(ranks)) if ranks else None),
            "attention_top1_stability": (
                float(max(top1_counts.values()) / len(top1_pixels)) if top1_pixels else 0.0
            ),
            "attention_top1_unique_count": int(len(top1_counts)),
            "attention_topk_persistence": float(sum(1 for item in rows if item["attention_topk_pck_hit"]) / len(rows)),
            "oracle_gap_timesteps": int(sum(1 for item in rows if item["oracle_gap_case"])),
            "attention_harms_native_timesteps": int(sum(1 for item in rows if item["attention_harms_native_case"])),
            "attention_rescues_native_timesteps": int(sum(1 for item in rows if item["attention_rescues_native_case"])),
            "reference_attention_gt_rank": (
                int(reference_row["attention_gt_rank"]) if reference_row and reference_row["attention_gt_rank"] is not None else None
            ),
            "reference_attention_top1_pck_hit": (
                bool(reference_row["attention_top1_pck_hit"]) if reference_row is not None else None
            ),
            "reference_attention_topk_pck_hit": (
                bool(reference_row["attention_topk_pck_hit"]) if reference_row is not None else None
            ),
        }
        audits.append({
            "summary": summary,
            "per_timestep": rows,
        })
    return audits


def _save_features(dataset_path, save_path, all_cats, cat2img, args, device):
    expected_paths = [
        os.path.join(save_path, f"{cat}.pth")
        for cat in all_cats
    ] + [
        os.path.join(save_path, f"{cat}_ada.pth")
        for cat in all_cats
    ]
    if args.reuse_saved_features and all(os.path.exists(path) for path in expected_paths):
        print("Reusing existing saved features; skip feature extraction.")
        return
    captions = {}
    with open("spair_detailed_captions.json") as handle:
        captions = json.load(handle)
    dit_model = Featurizer4Eval(cat_list=all_cats[:], ensemble_size=args.ensemble_size)
    os.makedirs(save_path, exist_ok=True)
    print("saving all test images' features...")
    for cat in tqdm(all_cats):
        output_dict = {}
        ada_dict = {}
        for image_path in cat2img[cat]:
            img = Image.open(os.path.join(dataset_path, "JPEGImages", cat, image_path))
            image_arr = np.array(img)
            in_h, in_w = image_arr.shape[:2]
            scale = args.img_size[0] / max(in_h, in_w)
            height = int(round(in_h * scale / 16)) * 16
            width = int(round(in_w * scale / 16)) * 16
            img = img.resize((width, height))
            img_tensor = (PILToTensor()(img) / 255.0 - 0.5) * 2
            output_dict[image_path], ada_dict[image_path] = dit_model.forward(
                args,
                img_tensor,
                caption=captions[cat + image_path],
                category=cat,
                timestep=args.t,
                block_idx=args.k,
                ensemble_size=args.ensemble_size,
            )
            output_dict[image_path] = output_dict[image_path].cpu()
            ada_dict[image_path] = ada_dict[image_path].cpu()
        torch.save(output_dict, os.path.join(save_path, f"{cat}.pth"))
        torch.save(ada_dict, os.path.join(save_path, f"{cat}_ada.pth"))
    del dit_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _pck(prediction, target, threshold):
    distance = ((prediction[0] - target[0]) ** 2 + (prediction[1] - target[1]) ** 2) ** 0.5
    return distance / threshold <= 0.1


def evaluate_fjsar_all(args, device):
    test_path, all_cats, cat2json, _cat2img = _load_pairs(args.dataset_path)
    if args.build_fjsar_cache:
        print("Ignoring --build_fjsar_cache: FJSAR extracts aligned entries on demand without writing cache files.")
    with open("spair_detailed_captions.json") as handle:
        captions = json.load(handle)
    fjsar_featurizer, fjsar_model, fjsar_blocks = _load_flux_fjsar_runtime(args, all_cats)
    fjsar_capture = _make_fjsar_capture(args, fjsar_model)
    trajectory_blocks = _fjsar_trajectory_blocks(args)
    trajectory_block_modules = _make_fjsar_trajectory_block_modules(args, fjsar_model)
    fjsar_memory_cache = _FjsarMemoryCache(
        int(float(args.fjsar_memory_cache_gb) * (1024 ** 3))
    )
    pre_norm = nn.LayerNorm(3072, elementwise_affine=False, eps=1e-6).to(device)

    method_totals = {
        matcher: _empty_fjsar_eval_stats()
        for matcher in FJSAR_ALL_MATCHERS
    }
    result = {
        "matcher": "fjsar_all",
        "shared_pair_features": True,
        "method_hypothesis": _fjsar_method_hypothesis(args),
        "attention_protocol": {
            "interaction_mode": "exact",
            "use_coordinate_bias": False,
        },
        "method_protocols": {
            matcher: {
                "mode": _fjsar_mode_config(matcher)[0],
                "interaction_mode": _fjsar_mode_config(matcher)[1],
                "use_coordinate_bias": _fjsar_mode_config(matcher)[2],
            }
            for matcher in FJSAR_ALL_MATCHERS
        },
        "methods": {
            matcher: {"categories": {}, "all": {}}
            for matcher in FJSAR_ALL_MATCHERS
        },
    }

    try:
        for cat in all_cats:
            cat_stats = {
                matcher: _empty_fjsar_eval_stats()
                for matcher in FJSAR_ALL_MATCHERS
            }
            cat_list = _select_category_pairs(cat2json[cat], args)
            if args.max_pairs_per_cat > 0:
                cat_list = cat_list[:args.max_pairs_per_cat]

            for json_path in tqdm(cat_list, desc=f"evaluate {cat}"):
                with open(os.path.join(args.dataset_path, test_path, json_path)) as handle:
                    data = json.load(handle)
                src_image_size = data["src_imsize"][:2][::-1]
                trg_image_size = data["trg_imsize"][:2][::-1]
                src_entry = _get_flux_fjsar_entry(
                    args.dataset_path,
                    cat,
                    data["src_imname"],
                    captions[cat + data["src_imname"]],
                    args,
                    fjsar_featurizer,
                    fjsar_capture,
                    fjsar_memory_cache,
                )
                trg_entry = _get_flux_fjsar_entry(
                    args.dataset_path,
                    cat,
                    data["trg_imname"],
                    captions[cat + data["trg_imname"]],
                    args,
                    fjsar_featurizer,
                    fjsar_capture,
                    fjsar_memory_cache,
                )
                src_ft = _prepare_feature_tensors(
                    src_entry["feature"], src_entry["ada"], args, pre_norm, device
                )
                trg_ft = _prepare_feature_tensors(
                    trg_entry["feature"], trg_entry["ada"], args, pre_norm, device
                )
                src_full = nn.Upsample(size=src_image_size, mode="bilinear")(src_ft.to(torch.float16))
                trg_full = nn.Upsample(size=trg_image_size, mode="bilinear")(trg_ft.to(torch.float16))
                baseline_predictions = cosine_nn_predict(src_full, trg_full, data["src_kps"])
                threshold = max(
                    data["trg_bndbox"][3] - data["trg_bndbox"][1],
                    data["trg_bndbox"][2] - data["trg_bndbox"][0],
                )

                for matcher in FJSAR_ALL_MATCHERS:
                    fjsar_mode, interaction_mode, use_coordinate_bias = _fjsar_mode_config(matcher)
                    method_predictions, matcher_diagnostics = flux_fjsar_predict(
                        src_ft,
                        trg_ft,
                        data["src_kps"],
                        src_image_size,
                        trg_image_size,
                        src_replay_state=src_entry["replay_state"],
                        trg_replay_state=trg_entry["replay_state"],
                        src_raw_feature=src_entry["feature"].to(device),
                        trg_raw_feature=trg_entry["feature"].to(device),
                        src_ada=src_entry["ada"],
                        trg_ada=trg_entry["ada"],
                        blocks=fjsar_blocks,
                        mode=fjsar_mode,
                        interaction_mode=interaction_mode,
                        use_coordinate_bias=use_coordinate_bias,
                        discard_channels=(154, 1446) if args.cd else (),
                        calibration=None,
                        target_points=data["trg_kps"] if args.fjsar_oracle_audit else None,
                        pck_threshold=threshold if args.fjsar_oracle_audit else None,
                        oracle_topk=args.fjsar_oracle_topk,
                        candidate_topk=args.fjsar_candidate_topk,
                        geometry_radius=args.fjsar_geometry_radius,
                        geometry_strength=args.fjsar_geometry_strength,
                        trajectory_replay_states=_entry_trajectory_pair(src_entry, trg_entry, args),
                        trajectory_block_modules=trajectory_block_modules,
                        trajectory_blocks=trajectory_blocks,
                        return_diagnostics=True,
                    )
                    _accumulate_fjsar_eval_stats(
                        cat_stats[matcher],
                        baseline_predictions,
                        method_predictions,
                        data["trg_kps"],
                        threshold,
                        matcher_diagnostics,
                    )
                    _accumulate_fjsar_eval_stats(
                        method_totals[matcher],
                        baseline_predictions,
                        method_predictions,
                        data["trg_kps"],
                        threshold,
                        matcher_diagnostics,
                    )

            for matcher in FJSAR_ALL_MATCHERS:
                category_result = _finalize_fjsar_eval_stats(cat_stats[matcher])
                result["methods"][matcher]["categories"][cat] = category_result
                print(
                    f"{cat} {matcher}: baseline image/point="
                    f"{category_result['baseline_image']:.2f}/{category_result['baseline_point']:.2f}, "
                    f"method image/point={category_result['method_image']:.2f}/"
                    f"{category_result['method_point']:.2f}"
                )

        for matcher in FJSAR_ALL_MATCHERS:
            result["methods"][matcher]["all"] = _finalize_fjsar_eval_stats(method_totals[matcher])
        result["fjsar_memory_cache"] = fjsar_memory_cache.stats()
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as handle:
            json.dump(result, handle, indent=2)
        for matcher in FJSAR_ALL_MATCHERS:
            all_result = result["methods"][matcher]["all"]
            print(
                f"Matcher: {matcher} | Baseline All per image/point: "
                f"{all_result['baseline_image']:.2f} / {all_result['baseline_point']:.2f} | "
                f"Method All per image/point: {all_result['method_image']:.2f} / "
                f"{all_result['method_point']:.2f}; point gain={all_result['point_gain']:.2f}"
            )
    finally:
        fjsar_capture.close()
        del fjsar_model, fjsar_featurizer, fjsar_blocks
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _validate_identity_decodability_feature_source(args):
    extract_in_memory = bool(getattr(args, "extract_native_in_memory", False))
    has_disk_path = bool(getattr(args, "fjsar_disk_cache_path", ""))
    requires_disk_cache = bool(getattr(args, "fjsar_require_disk_cache", False))
    canonical_disk_cache = has_disk_path and requires_disk_cache
    if extract_in_memory == canonical_disk_cache:
        raise ValueError(
            "identity decodability requires exactly one replay feature source: "
            "--extract_native_in_memory, or both --fjsar_disk_cache_path and "
            "--fjsar_require_disk_cache"
        )
    if has_disk_path or requires_disk_cache:
        if not canonical_disk_cache:
            raise ValueError(
                "identity decodability canonical-cache mode requires both "
                "--fjsar_disk_cache_path and --fjsar_require_disk_cache"
            )
    return "in_memory" if extract_in_memory else "canonical_disk_cache"


def evaluate(args, device):
    if args.matcher == "fjsar_all":
        return evaluate_fjsar_all(args, device)
    if args.fjsar_candidate_clamped_causal_replay_audit:
        if not args.fjsar_oracle_audit:
            raise ValueError(
                "--fjsar_candidate_clamped_causal_replay_audit requires --fjsar_oracle_audit"
            )
        if args.fjsar_dump_case_filter != "all":
            raise ValueError(
                "--fjsar_candidate_clamped_causal_replay_audit requires "
                "--fjsar_dump_case_filter all"
            )
        if int(args.fjsar_dump_max_records) != 0:
            raise ValueError(
                "--fjsar_candidate_clamped_causal_replay_audit requires "
                "--fjsar_dump_max_records 0"
            )
    if args.fjsar_counterfactual_fingerprint_audit:
        if not args.fjsar_oracle_audit:
            raise ValueError(
                "--fjsar_counterfactual_fingerprint_audit requires --fjsar_oracle_audit"
            )
        if args.fjsar_dump_case_filter != "all":
            raise ValueError(
                "--fjsar_counterfactual_fingerprint_audit requires "
                "--fjsar_dump_case_filter all"
            )
        scales = tuple(float(scale) for scale in args.fjsar_counterfactual_fingerprint_scales)
        if len(scales) < 3 or len(set(scales)) != len(scales) or any(scale <= 0.0 for scale in scales):
            raise ValueError(
                "counterfactual fingerprint scales must contain at least three distinct positive values"
            )
        if not any(abs(scale - 1.0) < 1e-8 for scale in scales):
            raise ValueError("counterfactual fingerprint scales must include 1.0")
        if int(args.fjsar_dump_max_records) != 0:
            raise ValueError(
                "--fjsar_counterfactual_fingerprint_audit requires "
                "--fjsar_dump_max_records 0"
            )
    if args.fjsar_persistent_candidate_slot_replay_audit:
        if not args.fjsar_oracle_audit:
            raise ValueError(
                "--fjsar_persistent_candidate_slot_replay_audit requires --fjsar_oracle_audit"
            )
        if int(args.fjsar_candidate_topk) != 20:
            raise ValueError(
                "--fjsar_persistent_candidate_slot_replay_audit requires candidate top-k 20"
            )
        if args.fjsar_dump_case_filter != "all" or int(args.fjsar_dump_max_records) != 0:
            raise ValueError(
                "--fjsar_persistent_candidate_slot_replay_audit requires all points without truncation"
            )
        if int(args.fjsar_persistent_candidate_slot_replay_chunk) <= 0:
            raise ValueError("persistent candidate-slot replay chunk must be positive")
        if not args.fjsar_require_disk_cache or not args.fjsar_disk_cache_path:
            raise ValueError(
                "persistent candidate-slot replay requires the canonical replay cache via "
                "--fjsar_disk_cache_path and --fjsar_require_disk_cache"
            )
    if args.fjsar_latent_expert_audit:
        if not args.fjsar_oracle_audit:
            raise ValueError("--fjsar_latent_expert_audit requires --fjsar_oracle_audit")
        if args.fjsar_dump_case_filter != "all":
            raise ValueError(
                "--fjsar_latent_expert_audit requires --fjsar_dump_case_filter all so head stability uses every pair keypoint"
            )
        if int(args.fjsar_dump_max_records) != 0:
            raise ValueError(
                "--fjsar_latent_expert_audit requires --fjsar_dump_max_records 0 so pairs are not truncated"
            )
    if args.fjsar_dense_candidate_edge_audit:
        if not args.fjsar_oracle_audit:
            raise ValueError("--fjsar_dense_candidate_edge_audit requires --fjsar_oracle_audit")
        if args.fjsar_dump_case_filter != "all":
            raise ValueError(
                "--fjsar_dense_candidate_edge_audit requires --fjsar_dump_case_filter all"
            )
        if int(args.fjsar_dump_max_records) != 0:
            raise ValueError(
                "--fjsar_dense_candidate_edge_audit requires --fjsar_dump_max_records 0"
            )
        if int(args.fjsar_dense_candidate_edge_radius) < 1:
            raise ValueError("--fjsar_dense_candidate_edge_radius must be at least 1")
    if args.fjsar_identity_decodability_audit:
        if args.matcher != "fjsar_attn":
            raise ValueError(
                "--fjsar_identity_decodability_audit requires --matcher fjsar_attn "
                "to freeze the established attention candidate protocol"
            )
        if not args.fjsar_oracle_audit:
            raise ValueError("--fjsar_identity_decodability_audit requires --fjsar_oracle_audit")
        if int(args.fjsar_candidate_topk) != 20:
            raise ValueError("--fjsar_identity_decodability_audit requires candidate top-k 20")
        if args.fjsar_dump_case_filter != "all" or int(args.fjsar_dump_max_records) != 0:
            raise ValueError(
                "--fjsar_identity_decodability_audit requires all points without truncation"
            )
        if int(args.fjsar_identity_decodability_folds) < 2:
            raise ValueError("identity decodability requires at least two category folds")
        _validate_identity_decodability_feature_source(args)
    test_path, all_cats, cat2json, cat2img = _load_pairs(args.dataset_path)
    if not args.matcher.startswith("fjsar"):
        _save_features(args.dataset_path, args.save_path, all_cats, cat2img, args, device)
    fjsar_model = None
    fjsar_featurizer = None
    fjsar_capture = None
    fjsar_blocks = []
    fjsar_persistent_slot_blocks = []
    fjsar_causal_release_block = None
    fjsar_memory_cache = None
    captions = None
    if args.matcher.startswith("fjsar"):
        if args.build_fjsar_cache:
            print("Ignoring --build_fjsar_cache: FJSAR now extracts aligned entries on demand without writing cache files.")
        with open("spair_detailed_captions.json") as handle:
            captions = json.load(handle)
        fjsar_featurizer, fjsar_model, fjsar_blocks = _load_flux_fjsar_runtime(args, all_cats)
        if args.fjsar_persistent_candidate_slot_replay_audit:
            feature_block = int(args.k[0] if isinstance(args.k, list) else args.k)
            fjsar_persistent_slot_blocks = select_flux_single_blocks(
                fjsar_model,
                feature_block,
                depth=2,
            )
        if (
            args.fjsar_candidate_clamped_causal_replay_audit
            or args.fjsar_counterfactual_fingerprint_audit
        ):
            feature_block = int(args.k[0] if isinstance(args.k, list) else args.k)
            causal_blocks = select_flux_single_blocks(
                fjsar_model, feature_block, depth=2
            )
            fjsar_causal_release_block = causal_blocks[1]
        fjsar_capture = _make_fjsar_capture(args, fjsar_model)
        fjsar_memory_cache = _FjsarMemoryCache(
            int(float(args.fjsar_memory_cache_gb) * (1024 ** 3))
        )
    trajectory_blocks = _fjsar_trajectory_blocks(args)
    trajectory_block_modules = (
        _make_fjsar_trajectory_block_modules(args, fjsar_model)
        if fjsar_model is not None
        else None
    )
    pre_norm = nn.LayerNorm(3072, elementwise_affine=False, eps=1e-6).to(device)
    result = {
        "matcher": args.matcher,
        "method_hypothesis": _fjsar_method_hypothesis(args),
        "categories": {},
        "all": {},
    }
    anchor_topology_pair_records = []
    anchor_topology_pair_keys = set()
    all_image_pck = []
    all_method_image_pck = []
    all_baseline_correct = all_method_correct = all_total = 0
    all_changed = all_improved = all_harmed = 0
    all_matcher_anchor_count = 0
    all_matcher_pair_reliability = 0.0
    all_matcher_local_reliability = 0.0
    all_matcher_split_agreement = 0.0
    all_matcher_flip_count = 0
    all_matcher_pair_count = 0
    all_matcher_valid_query_rate = 0.0
    all_matcher_local_anchor_count = 0.0
    all_matcher_decision_margin = 0.0
    all_matcher_graph_rate = 0.0
    all_matcher_cross_excess = 0.0
    all_fjsar_raw_parity = 0.0
    all_fjsar_prepared_parity = 0.0
    all_fjsar_cycle_error = 0.0
    all_fjsar_joint_native_cosine = 0.0
    all_fjsar_collapse_delta = 0.0
    all_fjsar_trajectory_layer_count = 0.0
    all_fjsar_trajectory_top1_stability = 0.0
    all_fjsar_parity_fail_count = 0
    all_matcher_model_counts = {}
    mean_baseline = mean_method = 0.0
    candidate_dump_records = []
    identity_decodability_shards = []
    identity_decodability_shard_dir = ""
    identity_decodability_feature_source = ""
    if args.fjsar_identity_decodability_audit:
        identity_decodability_feature_source = (
            _validate_identity_decodability_feature_source(args)
        )
        output_root, _output_ext = os.path.splitext(args.output_json)
        identity_decodability_shard_dir = (
            args.fjsar_identity_decodability_shard_path
            or f"{output_root}_identity_decodability_shards"
        )
        os.makedirs(identity_decodability_shard_dir, exist_ok=True)
    attention_relational_graph_pair_records = []
    dense_partial_graph_pair_records = []
    expert_hypothesis_pair_records = []
    pre_softmax_identity_pair_records = []
    layer_routed_identity_pair_records = []

    for cat in all_cats:
        if args.matcher.startswith("fjsar"):
            output_dict = ada_dict = None
            fjsar_state_dict = None
        else:
            fjsar_state_dict = None
            output_dict = _load_trusted_cache(os.path.join(args.save_path, f"{cat}.pth"))
            ada_dict = _load_trusted_cache(os.path.join(args.save_path, f"{cat}_ada.pth"))
        cat_list = _select_category_pairs(cat2json[cat], args)
        if args.max_pairs_per_cat > 0:
            cat_list = cat_list[:args.max_pairs_per_cat]
        cat_image_baseline = []
        cat_image_method = []
        cat_baseline_correct = cat_method_correct = cat_total = 0
        cat_changed = cat_improved = cat_harmed = 0
        cat_matcher_anchor_count = 0
        cat_matcher_pair_reliability = 0.0
        cat_matcher_local_reliability = 0.0
        cat_matcher_split_agreement = 0.0
        cat_matcher_flip_count = 0
        cat_matcher_pair_count = 0
        cat_matcher_valid_query_rate = 0.0
        cat_matcher_local_anchor_count = 0.0
        cat_matcher_decision_margin = 0.0
        cat_matcher_graph_rate = 0.0
        cat_matcher_cross_excess = 0.0
        cat_fjsar_raw_parity = 0.0
        cat_fjsar_prepared_parity = 0.0
        cat_fjsar_cycle_error = 0.0
        cat_fjsar_joint_native_cosine = 0.0
        cat_fjsar_collapse_delta = 0.0
        cat_fjsar_trajectory_layer_count = 0.0
        cat_fjsar_trajectory_top1_stability = 0.0
        cat_fjsar_parity_fail_count = 0
        cat_matcher_model_counts = {}

        for json_path in tqdm(cat_list, desc=f"evaluate {cat}"):
            with open(os.path.join(args.dataset_path, test_path, json_path)) as handle:
                data = json.load(handle)
            src_image_size = data["src_imsize"][:2][::-1]
            trg_image_size = data["trg_imsize"][:2][::-1]
            if args.matcher.startswith("fjsar"):
                if fjsar_featurizer is None or fjsar_capture is None or captions is None:
                    raise RuntimeError("FJSAR on-demand runtime was not initialized")
                src_entry = _get_flux_fjsar_entry(
                    args.dataset_path,
                    cat,
                    data["src_imname"],
                    captions[cat + data["src_imname"]],
                    args,
                    fjsar_featurizer,
                    fjsar_capture,
                    fjsar_memory_cache,
                )
                trg_entry = _get_flux_fjsar_entry(
                    args.dataset_path,
                    cat,
                    data["trg_imname"],
                    captions[cat + data["trg_imname"]],
                    args,
                    fjsar_featurizer,
                    fjsar_capture,
                    fjsar_memory_cache,
                )
                src_ft = _prepare_feature_tensors(
                    src_entry["feature"], src_entry["ada"], args, pre_norm, device
                )
                trg_ft = _prepare_feature_tensors(
                    trg_entry["feature"], trg_entry["ada"], args, pre_norm, device
                )
                if args.matcher == "fjsar_layer_routed_identity":
                    layer_identity_maps = _prepare_fjsar_multilayer_descriptor_maps(
                        src_entry,
                        trg_entry,
                        args,
                        pre_norm,
                        device,
                    )
                elif args.matcher == "fjsar_pre_single_stream_identity":
                    layer_identity_maps = _prepare_fjsar_pre_single_stream_descriptor_maps(
                        src_entry,
                        trg_entry,
                        args,
                        device,
                    )
                else:
                    layer_identity_maps = None
            else:
                src_entry = trg_entry = None
                src_ft = _prepare_features(output_dict, ada_dict, data["src_imname"], args, pre_norm, device)
                trg_ft = _prepare_features(output_dict, ada_dict, data["trg_imname"], args, pre_norm, device)
                layer_identity_maps = None
            src_full = nn.Upsample(size=src_image_size, mode="bilinear")(src_ft.to(torch.float16))
            trg_full = nn.Upsample(size=trg_image_size, mode="bilinear")(trg_ft.to(torch.float16))
            baseline_predictions = cosine_nn_predict(src_full, trg_full, data["src_kps"])
            threshold = max(
                data["trg_bndbox"][3] - data["trg_bndbox"][1],
                data["trg_bndbox"][2] - data["trg_bndbox"][0],
            )
            if args.matcher == "nn":
                method_predictions = baseline_predictions
            elif args.matcher.startswith("fjsar"):
                fjsar_mode, interaction_mode, use_coordinate_bias = (
                    _fjsar_mode_config(args.matcher)
                )
                # Preserve the standalone attention diagnostic protocol. The
                # shared fjsar_all audit deliberately uses exact replay.
                if args.matcher == "fjsar_attn":
                    interaction_mode = "calibrated"
                    use_coordinate_bias = True
                calibration = None
                method_predictions, matcher_diagnostics = flux_fjsar_predict(
                    src_ft,
                    trg_ft,
                    data["src_kps"],
                    src_image_size,
                    trg_image_size,
                    src_replay_state=src_entry["replay_state"],
                    trg_replay_state=trg_entry["replay_state"],
                    src_raw_feature=src_entry["feature"].to(device),
                    trg_raw_feature=trg_entry["feature"].to(device),
                    src_ada=src_entry["ada"],
                    trg_ada=trg_entry["ada"],
                    blocks=fjsar_blocks,
                    mode=fjsar_mode,
                    interaction_mode=interaction_mode,
                    use_coordinate_bias=use_coordinate_bias,
                    discard_channels=(154, 1446) if args.cd else (),
                    calibration=calibration,
                    target_points=(
                        data["trg_kps"]
                        if args.fjsar_oracle_audit
                        or args.matcher in (
                            "fjsar_attention_relational_graph_matching",
                            "fjsar_dense_partial_graph_matching",
                            "fjsar_expert_preserving_attention_hypothesis_conditioned_replay",
                            "fjsar_pre_softmax_channelwise_identity",
                            "fjsar_layer_routed_identity",
                            "fjsar_pre_single_stream_identity",
                        )
                        else None
                    ),
                    pck_threshold=(
                        threshold
                        if args.fjsar_oracle_audit
                        or args.matcher in (
                            "fjsar_attention_relational_graph_matching",
                            "fjsar_dense_partial_graph_matching",
                            "fjsar_expert_preserving_attention_hypothesis_conditioned_replay",
                            "fjsar_pre_softmax_channelwise_identity",
                            "fjsar_layer_routed_identity",
                            "fjsar_pre_single_stream_identity",
                        )
                        else None
                    ),
                    oracle_topk=args.fjsar_oracle_topk,
                    candidate_topk=args.fjsar_candidate_topk,
                    geometry_radius=args.fjsar_geometry_radius,
                    geometry_strength=args.fjsar_geometry_strength,
                    trajectory_replay_states=_entry_trajectory_pair(src_entry, trg_entry, args),
                    trajectory_block_modules=trajectory_block_modules,
                    trajectory_blocks=trajectory_blocks,
                    layer_identity_maps=layer_identity_maps,
                    return_diagnostics=True,
                )
                if args.matcher == "fjsar_attention_relational_graph_matching":
                    graph_audit = matcher_diagnostics.get("attention_relational_graph_audit", {})
                    if not isinstance(graph_audit, dict):
                        raise RuntimeError("attention relational graph diagnostics are missing")
                    pair_audit = {
                        "category": cat,
                        "pair_json": json_path,
                        "src_image": data["src_imname"],
                        "trg_image": data["trg_imname"],
                        "keypoint_count": int(len(data["src_kps"])),
                        "summary": graph_audit.get("summary", {}),
                        "points": graph_audit.get("points", []),
                    }
                    audit_points = pair_audit["points"]
                    if not isinstance(audit_points, list) or len(audit_points) != len(data["src_kps"]):
                        raise RuntimeError("attention relational graph point audit is misaligned")
                    for keypoint_index, point_audit in enumerate(audit_points):
                        baseline_hit = _pck(
                            baseline_predictions[keypoint_index],
                            data["trg_kps"][keypoint_index],
                            threshold,
                        )
                        method_hit = _pck(
                            method_predictions[keypoint_index],
                            data["trg_kps"][keypoint_index],
                            threshold,
                        )
                        point_audit.update({
                            "keypoint_index": keypoint_index,
                            "source_point": list(data["src_kps"][keypoint_index]),
                            "target_point": list(data["trg_kps"][keypoint_index]),
                            "baseline_prediction": list(baseline_predictions[keypoint_index]),
                            "method_prediction": list(method_predictions[keypoint_index]),
                            "baseline_pck_hit": bool(baseline_hit),
                            "method_pck_hit": bool(method_hit),
                            "rescued_vs_baseline": bool(method_hit and not baseline_hit),
                            "harmed_vs_baseline": bool(baseline_hit and not method_hit),
                        })
                    attention_relational_graph_pair_records.append(pair_audit)
                if args.matcher == "fjsar_dense_partial_graph_matching":
                    graph_audit = matcher_diagnostics.get("dense_partial_graph_audit", {})
                    if not isinstance(graph_audit, dict):
                        raise RuntimeError("dense partial graph diagnostics are missing")
                    pair_audit = {
                        "category": cat,
                        "pair_json": json_path,
                        "src_image": data["src_imname"],
                        "trg_image": data["trg_imname"],
                        "keypoint_count": int(len(data["src_kps"])),
                        "summary": graph_audit.get("summary", {}),
                        "points": graph_audit.get("points", []),
                    }
                    audit_points = pair_audit["points"]
                    if not isinstance(audit_points, list) or len(audit_points) != len(data["src_kps"]):
                        raise RuntimeError("dense partial graph point audit is misaligned")
                    for keypoint_index, point_audit in enumerate(audit_points):
                        baseline_hit = _pck(
                            baseline_predictions[keypoint_index],
                            data["trg_kps"][keypoint_index],
                            threshold,
                        )
                        method_hit = _pck(
                            method_predictions[keypoint_index],
                            data["trg_kps"][keypoint_index],
                            threshold,
                        )
                        point_audit.update({
                            "keypoint_index": keypoint_index,
                            "source_point": list(data["src_kps"][keypoint_index]),
                            "target_point": list(data["trg_kps"][keypoint_index]),
                            "baseline_prediction": list(baseline_predictions[keypoint_index]),
                            "method_prediction": list(method_predictions[keypoint_index]),
                            "baseline_pck_hit": bool(baseline_hit),
                            "method_pck_hit": bool(method_hit),
                            "rescued_vs_baseline": bool(method_hit and not baseline_hit),
                            "harmed_vs_baseline": bool(baseline_hit and not method_hit),
                        })
                    dense_partial_graph_pair_records.append(pair_audit)
                if args.matcher == "fjsar_expert_preserving_attention_hypothesis_conditioned_replay":
                    expert_audit = matcher_diagnostics.get(
                        "expert_hypothesis_audit", {}
                    )
                    if not isinstance(expert_audit, dict):
                        raise RuntimeError("expert hypothesis diagnostics are missing")
                    pair_audit = {
                        "category": cat,
                        "pair_json": json_path,
                        "src_image": data["src_imname"],
                        "trg_image": data["trg_imname"],
                        "keypoint_count": int(len(data["src_kps"])),
                        "summary": expert_audit.get("summary", {}),
                        "points": expert_audit.get("points", []),
                    }
                    audit_points = pair_audit["points"]
                    if not isinstance(audit_points, list) or len(audit_points) != len(
                        data["src_kps"]
                    ):
                        raise RuntimeError("expert hypothesis point audit is misaligned")
                    for keypoint_index, point_audit in enumerate(audit_points):
                        baseline_hit = _pck(
                            baseline_predictions[keypoint_index],
                            data["trg_kps"][keypoint_index],
                            threshold,
                        )
                        method_hit = _pck(
                            method_predictions[keypoint_index],
                            data["trg_kps"][keypoint_index],
                            threshold,
                        )
                        point_audit.update({
                            "keypoint_index": keypoint_index,
                            "source_point": list(data["src_kps"][keypoint_index]),
                            "target_point": list(data["trg_kps"][keypoint_index]),
                            "baseline_prediction": list(
                                baseline_predictions[keypoint_index]
                            ),
                            "method_prediction": list(
                                method_predictions[keypoint_index]
                            ),
                            "baseline_pck_hit": bool(baseline_hit),
                            "method_pck_hit": bool(method_hit),
                            "rescued_vs_baseline": bool(
                                method_hit and not baseline_hit
                            ),
                            "harmed_vs_baseline": bool(
                                baseline_hit and not method_hit
                            ),
                        })
                    expert_hypothesis_pair_records.append(pair_audit)
                if args.matcher == "fjsar_pre_softmax_channelwise_identity":
                    identity_audit = matcher_diagnostics.get(
                        "pre_softmax_channelwise_identity_audit", {}
                    )
                    if not isinstance(identity_audit, dict):
                        raise RuntimeError("pre-softmax identity diagnostics are missing")
                    pair_audit = {
                        "category": cat,
                        "pair_json": json_path,
                        "src_image": data["src_imname"],
                        "trg_image": data["trg_imname"],
                        "keypoint_count": int(len(data["src_kps"])),
                        "summary": identity_audit.get("summary", {}),
                        "points": identity_audit.get("points", []),
                    }
                    audit_points = pair_audit["points"]
                    if not isinstance(audit_points, list) or len(audit_points) != len(data["src_kps"]):
                        raise RuntimeError("pre-softmax identity point audit is misaligned")
                    for keypoint_index, point_audit in enumerate(audit_points):
                        baseline_hit = _pck(
                            baseline_predictions[keypoint_index],
                            data["trg_kps"][keypoint_index],
                            threshold,
                        )
                        method_hit = _pck(
                            method_predictions[keypoint_index],
                            data["trg_kps"][keypoint_index],
                            threshold,
                        )
                        point_audit.update({
                            "keypoint_index": keypoint_index,
                            "source_point": list(data["src_kps"][keypoint_index]),
                            "target_point": list(data["trg_kps"][keypoint_index]),
                            "baseline_prediction": list(baseline_predictions[keypoint_index]),
                            "method_prediction": list(method_predictions[keypoint_index]),
                            "baseline_pck_hit": bool(baseline_hit),
                            "method_pck_hit": bool(method_hit),
                            "rescued_vs_baseline": bool(method_hit and not baseline_hit),
                            "harmed_vs_baseline": bool(baseline_hit and not method_hit),
                            "both_wrong_top20_hit": bool(
                                not baseline_hit
                                and not bool(point_audit.get("attention_top1_pck_hit", False))
                                and not bool(point_audit.get("candidate_missing_gt", True))
                            ),
                        })
                    pre_softmax_identity_pair_records.append(pair_audit)
                if args.matcher in (
                    "fjsar_layer_routed_identity",
                    "fjsar_pre_single_stream_identity",
                ):
                    diagnostic_key = (
                        "pre_single_stream_identity_audit"
                        if args.matcher == "fjsar_pre_single_stream_identity"
                        else "layer_routed_identity_audit"
                    )
                    layer_audit = matcher_diagnostics.get(
                        diagnostic_key, {}
                    )
                    if not isinstance(layer_audit, dict):
                        raise RuntimeError(f"{diagnostic_key} diagnostics are missing")
                    pair_audit = {
                        "category": cat,
                        "pair_json": json_path,
                        "src_image": data["src_imname"],
                        "trg_image": data["trg_imname"],
                        "keypoint_count": int(len(data["src_kps"])),
                        "summary": layer_audit.get("summary", {}),
                        "points": layer_audit.get("points", []),
                    }
                    audit_points = pair_audit["points"]
                    if not isinstance(audit_points, list) or len(audit_points) != len(data["src_kps"]):
                        raise RuntimeError(f"{diagnostic_key} point audit is misaligned")
                    for keypoint_index, point_audit in enumerate(audit_points):
                        baseline_hit = _pck(
                            baseline_predictions[keypoint_index],
                            data["trg_kps"][keypoint_index],
                            threshold,
                        )
                        method_hit = _pck(
                            method_predictions[keypoint_index],
                            data["trg_kps"][keypoint_index],
                            threshold,
                        )
                        point_audit.update({
                            "keypoint_index": keypoint_index,
                            "source_point": list(data["src_kps"][keypoint_index]),
                            "target_point": list(data["trg_kps"][keypoint_index]),
                            "baseline_prediction": list(baseline_predictions[keypoint_index]),
                            "method_prediction": list(method_predictions[keypoint_index]),
                            "baseline_pck_hit": bool(baseline_hit),
                            "method_pck_hit": bool(method_hit),
                            "rescued_vs_baseline": bool(method_hit and not baseline_hit),
                            "harmed_vs_baseline": bool(baseline_hit and not method_hit),
                            "both_wrong_top20_hit": bool(
                                not baseline_hit
                                and not bool(point_audit.get("attention_top1_pck_hit", False))
                                and not bool(point_audit.get("candidate_missing_gt", True))
                            ),
                        })
                    layer_routed_identity_pair_records.append(pair_audit)
                cat_matcher_pair_reliability += float(
                    matcher_diagnostics["mean_cross_mass_source"]
                )
                cat_matcher_local_reliability += float(
                    matcher_diagnostics["mean_residual_confidence_source"]
                )
                cat_matcher_split_agreement += float(
                    matcher_diagnostics["native_parity_cosine"]
                )
                cat_matcher_valid_query_rate += float(
                    matcher_diagnostics["mean_position_confidence_source"]
                )
                cat_matcher_local_anchor_count += float(
                    matcher_diagnostics["replay_depth"]
                )
                cat_matcher_decision_margin += float(
                    matcher_diagnostics["mean_reciprocal_source"]
                )
                cat_matcher_graph_rate += float(
                    matcher_diagnostics["mean_concentration_source"]
                )
                cat_matcher_cross_excess += float(
                    matcher_diagnostics["mean_cross_excess_source"]
                )
                cat_fjsar_raw_parity += float(
                    matcher_diagnostics["raw_boundary_parity_cosine"]
                )
                cat_fjsar_prepared_parity += float(
                    matcher_diagnostics["prepared_feature_parity_cosine"]
                )
                cat_fjsar_cycle_error += float(
                    matcher_diagnostics["mean_cycle_error_source"]
                )
                cat_fjsar_joint_native_cosine += float(
                    matcher_diagnostics["joint_native_cosine_source"]
                )
                cat_fjsar_collapse_delta += float(
                    matcher_diagnostics["joint_intra_cosine_source"]
                    - matcher_diagnostics["native_intra_cosine_source"]
                )
                cat_fjsar_trajectory_layer_count += float(
                    matcher_diagnostics.get("mean_trajectory_layer_count", 0.0)
                )
                cat_fjsar_trajectory_top1_stability += float(
                    matcher_diagnostics.get("mean_trajectory_top1_stability", 0.0)
                )
                cat_fjsar_parity_fail_count += int(
                    matcher_diagnostics["parity_ok"] < 0.5
                )
                _merge_counts(
                    cat_matcher_model_counts,
                    matcher_diagnostics.get("model_counts", {}),
                )
                cat_matcher_pair_count += 1
                if (
                    args.fjsar_dump_candidates
                    or args.fjsar_candidate_descriptor_audit
                    or args.fjsar_method_descriptor_audit
                    or args.fjsar_transport_lift_branch_audit
                    or args.fjsar_attention_flow_audit
                    or args.fjsar_attention_kernel_audit
                    or args.fjsar_kernel_featureization_audit
                    or args.fjsar_residual_readout_audit
                    or args.fjsar_latent_expert_audit
                    or args.fjsar_candidate_clamped_causal_replay_audit
                    or args.fjsar_counterfactual_fingerprint_audit
                    or args.fjsar_persistent_candidate_slot_replay_audit
                    or args.fjsar_local_relational_identity_audit
                    or args.fjsar_dense_candidate_edge_audit
                    or args.fjsar_dense_transport_consistency_audit
                    or args.fjsar_candidate_field_consistency_audit
                    or args.fjsar_anchor_topology_audit
                    or args.fjsar_multilayer_identity_audit
                    or args.matcher == "fjsar_cross_attention_trajectory"
                    or args.fjsar_transport_factorization_audit
                    or args.fjsar_basin_identity_audit
                    or args.fjsar_operator_manifold_audit
                    or args.fjsar_identity_decodability_audit
                ):
                    dump_source_points = list(data["src_kps"])
                    dump_target_points = list(data["trg_kps"])
                    dump_original_indices = list(range(len(dump_source_points)))
                    if args.fjsar_dump_case_filter != "all" and not args.fjsar_anchor_topology_audit:
                        attention_cases = matcher_diagnostics.get("attention_case_records")
                        if not isinstance(attention_cases, list) or len(attention_cases) != len(dump_source_points):
                            attention_cases = flux_fjsar_attention_case_records(
                                dump_source_points,
                                src_image_size,
                                trg_image_size,
                                src_replay_state=src_entry["replay_state"],
                                trg_replay_state=trg_entry["replay_state"],
                                blocks=fjsar_blocks,
                                interaction_mode=interaction_mode,
                                use_coordinate_bias=use_coordinate_bias,
                                target_points=dump_target_points,
                                pck_threshold=threshold,
                                candidate_topk=args.fjsar_candidate_topk,
                                geometry_radius=args.fjsar_geometry_radius,
                                geometry_strength=args.fjsar_geometry_strength,
                                trajectory_replay_states=_entry_trajectory_pair(src_entry, trg_entry, args),
                                trajectory_block_modules=trajectory_block_modules,
                                trajectory_blocks=trajectory_blocks,
                            )
                        selected_indices = []
                        for keypoint_index, attention_case in enumerate(attention_cases):
                            baseline_hit = _pck(
                                baseline_predictions[keypoint_index],
                                data["trg_kps"][keypoint_index],
                                threshold,
                            )
                            method_hit = _pck(
                                method_predictions[keypoint_index],
                                data["trg_kps"][keypoint_index],
                                threshold,
                            )
                            attention_top1_hit = bool(attention_case.get("attention_top1_pck_hit"))
                            attention_topk_hit = bool(attention_case.get("attention_topk_pck_hit"))
                            case_row = {
                                "baseline_pck_hit": bool(baseline_hit),
                                "method_pck_hit": bool(method_hit),
                                "attention_top1_pck_hit": attention_top1_hit,
                                "attention_topk_pck_hit": attention_topk_hit,
                                "oracle_gap_case": bool(
                                    (not baseline_hit)
                                    and attention_topk_hit
                                    and not attention_top1_hit
                                ),
                                "attention_harms_native_case": bool(
                                    baseline_hit and not attention_top1_hit
                                ),
                                "attention_rescues_native_case": bool(
                                    (not baseline_hit) and attention_top1_hit
                                ),
                            }
                            if _keep_fjsar_dump_row(case_row, args.fjsar_dump_case_filter):
                                selected_indices.append(keypoint_index)
                                if (
                                    int(args.fjsar_dump_max_records) > 0
                                    and len(candidate_dump_records) + len(selected_indices)
                                    >= int(args.fjsar_dump_max_records)
                                ):
                                    break
                        dump_source_points = [dump_source_points[index] for index in selected_indices]
                        dump_target_points = [dump_target_points[index] for index in selected_indices]
                        dump_original_indices = selected_indices
                    dump_rows = []
                    if dump_source_points:
                        multilayer_descriptor_maps = (
                            _prepare_fjsar_multilayer_descriptor_maps(
                                src_entry,
                                trg_entry,
                                args,
                                pre_norm,
                                device,
                            )
                            if args.fjsar_multilayer_identity_audit
                            else None
                        )
                        dump_rows = flux_fjsar_dump_candidates(
                            src_ft,
                            trg_ft,
                            dump_source_points,
                            src_image_size,
                            trg_image_size,
                            src_replay_state=src_entry["replay_state"],
                            trg_replay_state=trg_entry["replay_state"],
                            blocks=fjsar_blocks,
                            interaction_mode=interaction_mode,
                            use_coordinate_bias=use_coordinate_bias,
                            src_ada=src_entry["ada"],
                            trg_ada=trg_entry["ada"],
                            discard_channels=(154, 1446) if args.cd else (),
                            calibration=None,
                            target_points=dump_target_points,
                            pck_threshold=threshold,
                            candidate_topk=args.fjsar_candidate_topk,
                            candidate_descriptor_audit=args.fjsar_candidate_descriptor_audit,
                            method_descriptor_audit_mode=(
                                fjsar_mode
                                if args.fjsar_method_descriptor_audit
                                and fjsar_mode not in (
                                    "cross_attention_trajectory",
                                    "native_preserving_topology_rescue",
                                    "attention_basin_native_refine",
                                    "candidate_graph_consensus_verification",
                                    "candidate_conditioned_verification",
                                    "candidate_local_transport_verification",
                                )
                                else None
                            ),
                            transport_lift_branch_audit=args.fjsar_transport_lift_branch_audit,
                            attention_flow_audit=args.fjsar_attention_flow_audit,
                            attention_flow_radius=args.fjsar_attention_flow_radius,
                            attention_kernel_audit=args.fjsar_attention_kernel_audit,
                            attention_kernel_radius=args.fjsar_attention_kernel_radius,
                            attention_kernel_topk=args.fjsar_attention_kernel_topk,
                            basin_identity_audit=args.fjsar_basin_identity_audit,
                            basin_identity_topk=args.fjsar_basin_identity_topk,
                            basin_identity_radius=args.fjsar_basin_identity_radius,
                            basin_identity_rank_topk=args.fjsar_basin_identity_rank_topk,
                            kernel_featureization_audit=args.fjsar_kernel_featureization_audit,
                            kernel_featureization_ranks=args.fjsar_kernel_featureization_ranks,
                            kernel_featureization_weights=args.fjsar_kernel_featureization_weights,
                            kernel_featureization_radius=args.fjsar_kernel_featureization_radius,
                            kernel_featureization_topk=args.fjsar_kernel_featureization_topk,
                            residual_readout_audit=args.fjsar_residual_readout_audit,
                            residual_readout_topk=args.fjsar_residual_readout_topk,
                            latent_expert_audit=args.fjsar_latent_expert_audit,
                            latent_expert_topk=args.fjsar_latent_expert_topk,
                            candidate_clamped_causal_replay_audit=(
                                args.fjsar_candidate_clamped_causal_replay_audit
                            ),
                            candidate_clamped_causal_replay_topk=(
                                args.fjsar_candidate_clamped_causal_replay_topk
                            ),
                            causal_release_block=fjsar_causal_release_block,
                            counterfactual_fingerprint_audit=(
                                args.fjsar_counterfactual_fingerprint_audit
                            ),
                            counterfactual_fingerprint_topk=(
                                args.fjsar_counterfactual_fingerprint_topk
                            ),
                            counterfactual_fingerprint_scales=(
                                args.fjsar_counterfactual_fingerprint_scales
                            ),
                            persistent_candidate_slot_replay_audit=(
                                args.fjsar_persistent_candidate_slot_replay_audit
                            ),
                            persistent_candidate_slot_replay_topk=(
                                args.fjsar_persistent_candidate_slot_replay_topk
                            ),
                            persistent_candidate_slot_replay_chunk=(
                                args.fjsar_persistent_candidate_slot_replay_chunk
                            ),
                            persistent_candidate_slot_replay_blocks=(
                                fjsar_persistent_slot_blocks
                                if args.fjsar_persistent_candidate_slot_replay_audit
                                else None
                            ),
                            local_relational_identity_audit=args.fjsar_local_relational_identity_audit,
                            local_relational_radius=args.fjsar_local_relational_radius,
                            dense_candidate_edge_audit=args.fjsar_dense_candidate_edge_audit,
                            dense_candidate_edge_radius=args.fjsar_dense_candidate_edge_radius,
                            dense_transport_consistency_audit=args.fjsar_dense_transport_consistency_audit,
                            dense_transport_topk=args.fjsar_dense_transport_topk,
                            candidate_field_consistency_audit=args.fjsar_candidate_field_consistency_audit,
                            candidate_field_topm=args.fjsar_candidate_field_topm,
                            candidate_field_source=args.fjsar_candidate_field_source,
                            anchor_topology_audit=args.fjsar_anchor_topology_audit,
                            multilayer_identity_audit=args.fjsar_multilayer_identity_audit,
                            multilayer_descriptor_maps=multilayer_descriptor_maps,
                            geometry_radius=args.fjsar_geometry_radius,
                            geometry_strength=args.fjsar_geometry_strength,
                            trajectory_replay_states=_entry_trajectory_pair(src_entry, trg_entry, args),
                                trajectory_block_modules=trajectory_block_modules,
                                trajectory_blocks=trajectory_blocks,
                                transport_factorization_audit=args.fjsar_transport_factorization_audit,
                                transport_factorization_radius=args.fjsar_transport_factorization_radius,
                                transport_factorization_basis_radius=args.fjsar_transport_factorization_basis_radius,
                                operator_manifold_audit=args.fjsar_operator_manifold_audit,
                            )
                        if args.fjsar_identity_decodability_audit:
                            decodability_batch = flux_fjsar_identity_decodability_batch(
                                src_ft,
                                trg_ft,
                                dump_source_points,
                                src_image_size,
                                trg_image_size,
                                src_replay_state=src_entry["replay_state"],
                                trg_replay_state=trg_entry["replay_state"],
                                blocks=fjsar_blocks,
                                interaction_mode=interaction_mode,
                                use_coordinate_bias=use_coordinate_bias,
                                target_points=dump_target_points,
                                pck_threshold=threshold,
                                candidate_topk=args.fjsar_candidate_topk,
                            )
                            pair_id = "|".join((
                                str(cat),
                                str(json_path),
                                str(data["src_imname"]),
                                str(data["trg_imname"]),
                            ))
                            shard_digest = hashlib.sha256(pair_id.encode("utf-8")).hexdigest()[:16]
                            shard_category_dir = os.path.join(
                                identity_decodability_shard_dir,
                                str(cat),
                            )
                            os.makedirs(shard_category_dir, exist_ok=True)
                            shard_path = os.path.join(
                                shard_category_dir,
                                f"{shard_digest}.pth",
                            )
                            decodability_batch.update({
                                "category": str(cat),
                                "pair_id": pair_id,
                                "pair_json": str(json_path),
                                "src_image": str(data["src_imname"]),
                                "trg_image": str(data["trg_imname"]),
                                "keypoint_indices": torch.tensor(
                                    dump_original_indices,
                                    dtype=torch.int32,
                                ),
                                "baseline_hits": torch.tensor(
                                    [
                                        _pck(
                                            baseline_predictions[index],
                                            data["trg_kps"][index],
                                            threshold,
                                        )
                                        for index in dump_original_indices
                                    ],
                                    dtype=torch.bool,
                                ),
                            })
                            torch.save(decodability_batch, shard_path)
                            identity_decodability_shards.append(shard_path)
                        multi_timestep_attention_identity_audits = None
                        if args.fjsar_multi_timestep_attention_identity_audit:
                            multi_timestep_attention_identity_audits = _multi_timestep_attention_identity_audit_for_points(
                                args=args,
                                dataset_path=args.dataset_path,
                                category=cat,
                                src_image_name=data["src_imname"],
                                trg_image_name=data["trg_imname"],
                                src_caption=captions[cat + data["src_imname"]],
                                trg_caption=captions[cat + data["trg_imname"]],
                                source_points=dump_source_points,
                                target_points=dump_target_points,
                                source_size=src_image_size,
                                target_size=trg_image_size,
                                threshold=threshold,
                                featurizer=fjsar_featurizer,
                                capture=fjsar_capture,
                                memory_cache=fjsar_memory_cache,
                                blocks=fjsar_blocks,
                                interaction_mode=interaction_mode,
                                use_coordinate_bias=use_coordinate_bias,
                                candidate_topk=args.fjsar_candidate_topk,
                                device=device,
                                pre_norm=pre_norm,
                                discard_channels=(154, 1446) if args.cd else (),
                                calibration=None,
                            )
                            if len(multi_timestep_attention_identity_audits) != len(dump_rows):
                                raise RuntimeError(
                                    "multi_timestep_attention_identity_audit returned mismatched row count"
                                )
                        else:
                            multi_timestep_attention_identity_audits = None
                    for local_keypoint_index, row in enumerate(dump_rows):
                        keypoint_index = int(dump_original_indices[local_keypoint_index])
                        if multi_timestep_attention_identity_audits is not None:
                            row["multi_timestep_attention_identity_audit"] = multi_timestep_attention_identity_audits[local_keypoint_index]
                        row.update({
                            "category": cat,
                            "pair_json": json_path,
                            "src_image": data["src_imname"],
                            "trg_image": data["trg_imname"],
                            "keypoint_index": int(keypoint_index),
                            "threshold": float(threshold),
                            "baseline_prediction": [
                                int(baseline_predictions[keypoint_index][0]),
                                int(baseline_predictions[keypoint_index][1]),
                            ],
                            "method_prediction": [
                                int(method_predictions[keypoint_index][0]),
                                int(method_predictions[keypoint_index][1]),
                            ],
                        })
                        method_pixel_index = (
                            int(method_predictions[keypoint_index][1]) * int(trg_image_size[1])
                            + int(method_predictions[keypoint_index][0])
                        )
                        method_attention_proposal = next(
                            (
                                proposal
                                for proposal in row.get("proposals", [])
                                if int(proposal.get("pixel_index", -1)) == method_pixel_index
                            ),
                            None,
                        )
                        row.update({
                            "method_prediction_pixel_index": int(method_pixel_index),
                            "method_prediction_in_attention_proposals": method_attention_proposal is not None,
                            "method_prediction_attention_rank": (
                                int(method_attention_proposal["rank_attention"])
                                if method_attention_proposal is not None
                                and method_attention_proposal.get("rank_attention") is not None
                                else None
                            ),
                        })
                        baseline_hit = _pck(
                            baseline_predictions[keypoint_index],
                            data["trg_kps"][keypoint_index],
                            threshold,
                        )
                        method_hit = _pck(
                            method_predictions[keypoint_index],
                            data["trg_kps"][keypoint_index],
                            threshold,
                        )
                        attention_top1_hit = row.get("attention_top1", {}).get("pck_hit")
                        attention_gt_rank = row.get("gt_ranks", {}).get("attention")
                        attention_topk_hit = (
                            attention_gt_rank is not None
                            and int(attention_gt_rank) <= int(args.fjsar_candidate_topk)
                        )
                        row.update({
                            "baseline_pck_hit": bool(baseline_hit),
                            "method_pck_hit": bool(method_hit),
                            "attention_top1_pck_hit": attention_top1_hit,
                            "attention_topk_pck_hit": bool(attention_topk_hit),
                            "oracle_gap_case": bool(
                                (not baseline_hit)
                                and attention_topk_hit
                                and not bool(attention_top1_hit)
                            ),
                            "attention_harms_native_case": bool(
                                baseline_hit and not bool(attention_top1_hit)
                            ),
                            "attention_rescues_native_case": bool(
                                (not baseline_hit) and bool(attention_top1_hit)
                            ),
                        })
                        if args.fjsar_anchor_topology_audit:
                            pair_key = (
                                cat,
                                json_path,
                                data["src_imname"],
                                data["trg_imname"],
                            )
                            audit = row.get("anchor_topology_audit")
                            if pair_key not in anchor_topology_pair_keys and isinstance(audit, dict):
                                anchor_topology_pair_keys.add(pair_key)
                                anchor_topology_pair_records.append({
                                    "category": cat,
                                    "pair_json": json_path,
                                    "src_image": data["src_imname"],
                                    "trg_image": data["trg_imname"],
                                    "keypoint_count": int(len(data["src_kps"])),
                                    "anchor_count": int(audit.get("anchor_count", 0)),
                                    "positive_anchor_count": int(audit.get("positive_anchor_count", 0)),
                                    "effective_anchor_count": float(audit.get("effective_anchor_count", 0.0)),
                                    "anchor_confidence": audit.get("anchor_confidence") or {},
                                    "affine_valid": bool(audit.get("affine_valid", False)),
                                })
                        if _keep_fjsar_dump_row(row, args.fjsar_dump_case_filter):
                            if (
                                int(args.fjsar_dump_max_records) <= 0
                                or len(candidate_dump_records) < int(args.fjsar_dump_max_records)
                            ):
                                candidate_dump_records.append(row)
            else:
                raise ValueError(f"unsupported matcher: {args.matcher}")
            baseline_pair = method_pair = 0
            for baseline, method, target in zip(baseline_predictions, method_predictions, data["trg_kps"]):
                baseline_ok = _pck(baseline, target, threshold)
                method_ok = _pck(method, target, threshold)
                baseline_pair += int(baseline_ok)
                method_pair += int(method_ok)
                baseline_flag = int(baseline_ok)
                method_flag = int(method_ok)
                cat_baseline_correct += baseline_flag
                cat_method_correct += method_flag
                cat_total += 1
                if list(baseline) != list(method):
                    cat_changed += 1
                if method_flag > baseline_flag:
                    cat_improved += 1
                elif baseline_flag > method_flag:
                    cat_harmed += 1
            cat_image_baseline.append(baseline_pair / len(data["trg_kps"]))
            cat_image_method.append(method_pair / len(data["trg_kps"]))
            all_total += len(data["trg_kps"])
            all_baseline_correct += baseline_pair
            all_method_correct += method_pair

        baseline_image = float(np.mean(cat_image_baseline)) if cat_image_baseline else 0.0
        method_image = float(np.mean(cat_image_method)) if cat_image_method else 0.0
        baseline_point = cat_baseline_correct / cat_total if cat_total else 0.0
        method_point = cat_method_correct / cat_total if cat_total else 0.0
        mean_baseline += baseline_image
        mean_method += method_image
        all_image_pck.extend(cat_image_baseline)
        all_method_image_pck.extend(cat_image_method)
        all_changed += cat_changed
        all_improved += cat_improved
        all_harmed += cat_harmed
        all_matcher_anchor_count += cat_matcher_anchor_count
        all_matcher_pair_reliability += cat_matcher_pair_reliability
        all_matcher_local_reliability += cat_matcher_local_reliability
        all_matcher_split_agreement += cat_matcher_split_agreement
        all_matcher_flip_count += cat_matcher_flip_count
        all_matcher_pair_count += cat_matcher_pair_count
        all_matcher_valid_query_rate += cat_matcher_valid_query_rate
        all_matcher_local_anchor_count += cat_matcher_local_anchor_count
        all_matcher_decision_margin += cat_matcher_decision_margin
        all_matcher_graph_rate += cat_matcher_graph_rate
        all_matcher_cross_excess += cat_matcher_cross_excess
        all_fjsar_raw_parity += cat_fjsar_raw_parity
        all_fjsar_prepared_parity += cat_fjsar_prepared_parity
        all_fjsar_cycle_error += cat_fjsar_cycle_error
        all_fjsar_joint_native_cosine += cat_fjsar_joint_native_cosine
        all_fjsar_collapse_delta += cat_fjsar_collapse_delta
        all_fjsar_trajectory_layer_count += cat_fjsar_trajectory_layer_count
        all_fjsar_trajectory_top1_stability += cat_fjsar_trajectory_top1_stability
        all_fjsar_parity_fail_count += cat_fjsar_parity_fail_count
        _merge_counts(all_matcher_model_counts, cat_matcher_model_counts)
        result["categories"][cat] = {
            "baseline_image": 100.0 * baseline_image,
            "method_image": 100.0 * method_image,
            "baseline_point": 100.0 * baseline_point,
            "method_point": 100.0 * method_point,
            "point_gain": 100.0 * (method_point - baseline_point),
            "pair_count": len(cat_list),
            "changed_count": cat_changed,
            "improved_count": cat_improved,
            "harmed_count": cat_harmed,
            "improvement_harm_ratio": (
                float(cat_improved) / float(cat_harmed) if cat_harmed else None
            ),
            "matcher_diagnostics": {
                "mean_anchor_count": (
                    cat_matcher_anchor_count / cat_matcher_pair_count
                    if cat_matcher_pair_count else 0.0
                ),
                "mean_pair_reliability": (
                    cat_matcher_pair_reliability / cat_matcher_pair_count
                    if cat_matcher_pair_count else 0.0
                ),
                "mean_local_reliability": (
                    cat_matcher_local_reliability / cat_matcher_pair_count
                    if cat_matcher_pair_count else 0.0
                ),
                "mean_split_agreement": (
                    cat_matcher_split_agreement / cat_matcher_pair_count
                    if cat_matcher_pair_count else 0.0
                ),
                "horizontal_flip_rate": (
                    cat_matcher_flip_count / cat_matcher_pair_count
                    if cat_matcher_pair_count else 0.0
                ),
                "mean_valid_query_rate": (
                    cat_matcher_valid_query_rate / cat_matcher_pair_count
                    if cat_matcher_pair_count else 0.0
                ),
                "mean_local_anchor_count": (
                    cat_matcher_local_anchor_count / cat_matcher_pair_count
                    if cat_matcher_pair_count else 0.0
                ),
                "mean_decision_margin": (
                    cat_matcher_decision_margin / cat_matcher_pair_count
                    if cat_matcher_pair_count else 0.0
                ),
                "mean_graph_informative_rate": (
                    cat_matcher_graph_rate / cat_matcher_pair_count
                    if cat_matcher_pair_count else 0.0
                ),
                "mean_cross_excess": (
                    cat_matcher_cross_excess / cat_matcher_pair_count
                    if cat_matcher_pair_count else 0.0
                ),
                "fjsar_raw_boundary_parity_cosine": (
                    cat_fjsar_raw_parity / cat_matcher_pair_count
                    if cat_matcher_pair_count else 0.0
                ),
                "fjsar_prepared_feature_parity_cosine": (
                    cat_fjsar_prepared_parity / cat_matcher_pair_count
                    if cat_matcher_pair_count else 0.0
                ),
                "fjsar_mean_cycle_error": (
                    cat_fjsar_cycle_error / cat_matcher_pair_count
                    if cat_matcher_pair_count else 0.0
                ),
                "fjsar_joint_native_cosine": (
                    cat_fjsar_joint_native_cosine / cat_matcher_pair_count
                    if cat_matcher_pair_count else 0.0
                ),
                "fjsar_collapse_delta": (
                    cat_fjsar_collapse_delta / cat_matcher_pair_count
                    if cat_matcher_pair_count else 0.0
                ),
                "fjsar_trajectory_layer_count": (
                    cat_fjsar_trajectory_layer_count / cat_matcher_pair_count
                    if cat_matcher_pair_count else 0.0
                ),
                "fjsar_trajectory_top1_stability": (
                    cat_fjsar_trajectory_top1_stability / cat_matcher_pair_count
                    if cat_matcher_pair_count else 0.0
                ),
                "fjsar_parity_failure_rate": (
                    cat_fjsar_parity_fail_count / cat_matcher_pair_count
                    if cat_matcher_pair_count else 0.0
                ),
                "model_counts": cat_matcher_model_counts,
            },
        }
        print(
            f"{cat}: baseline image/point={100.0 * baseline_image:.2f}/"
            f"{100.0 * baseline_point:.2f}, method image/point="
            f"{100.0 * method_image:.2f}/{100.0 * method_point:.2f}"
        )

    result["all"] = {
        "baseline_image": 100.0 * float(np.mean(all_image_pck)) if all_image_pck else 0.0,
        "method_image": 100.0 * float(np.mean(all_method_image_pck)) if all_method_image_pck else 0.0,
        "baseline_point": 100.0 * all_baseline_correct / all_total if all_total else 0.0,
        "method_point": 100.0 * all_method_correct / all_total if all_total else 0.0,
        "point_gain": 100.0 * (all_method_correct - all_baseline_correct) / all_total if all_total else 0.0,
        "mean_category_baseline_image": 100.0 * mean_baseline / len(all_cats),
        "mean_category_method_image": 100.0 * mean_method / len(all_cats),
        "changed_count": all_changed,
        "improved_count": all_improved,
        "harmed_count": all_harmed,
        "improvement_harm_ratio": (
            float(all_improved) / float(all_harmed) if all_harmed else None
        ),
        "intervention_rate": float(all_changed) / float(all_total) if all_total else 0.0,
        "matcher_diagnostics": {
            "mean_anchor_count": (
                all_matcher_anchor_count / all_matcher_pair_count
                if all_matcher_pair_count else 0.0
            ),
            "mean_pair_reliability": (
                all_matcher_pair_reliability / all_matcher_pair_count
                if all_matcher_pair_count else 0.0
            ),
            "mean_local_reliability": (
                all_matcher_local_reliability / all_matcher_pair_count
                if all_matcher_pair_count else 0.0
            ),
            "mean_split_agreement": (
                all_matcher_split_agreement / all_matcher_pair_count
                if all_matcher_pair_count else 0.0
            ),
            "horizontal_flip_rate": (
                all_matcher_flip_count / all_matcher_pair_count
                if all_matcher_pair_count else 0.0
            ),
            "mean_valid_query_rate": (
                all_matcher_valid_query_rate / all_matcher_pair_count
                if all_matcher_pair_count else 0.0
            ),
            "mean_local_anchor_count": (
                all_matcher_local_anchor_count / all_matcher_pair_count
                if all_matcher_pair_count else 0.0
            ),
            "mean_decision_margin": (
                all_matcher_decision_margin / all_matcher_pair_count
                if all_matcher_pair_count else 0.0
            ),
            "mean_graph_informative_rate": (
                all_matcher_graph_rate / all_matcher_pair_count
                if all_matcher_pair_count else 0.0
            ),
            "mean_cross_excess": (
                all_matcher_cross_excess / all_matcher_pair_count
                if all_matcher_pair_count else 0.0
            ),
            "fjsar_raw_boundary_parity_cosine": (
                all_fjsar_raw_parity / all_matcher_pair_count
                if all_matcher_pair_count else 0.0
            ),
            "fjsar_prepared_feature_parity_cosine": (
                all_fjsar_prepared_parity / all_matcher_pair_count
                if all_matcher_pair_count else 0.0
            ),
            "fjsar_mean_cycle_error": (
                all_fjsar_cycle_error / all_matcher_pair_count
                if all_matcher_pair_count else 0.0
            ),
            "fjsar_joint_native_cosine": (
                all_fjsar_joint_native_cosine / all_matcher_pair_count
                if all_matcher_pair_count else 0.0
            ),
            "fjsar_collapse_delta": (
                all_fjsar_collapse_delta / all_matcher_pair_count
                if all_matcher_pair_count else 0.0
            ),
            "fjsar_trajectory_layer_count": (
                all_fjsar_trajectory_layer_count / all_matcher_pair_count
                if all_matcher_pair_count else 0.0
            ),
            "fjsar_trajectory_top1_stability": (
                all_fjsar_trajectory_top1_stability / all_matcher_pair_count
                if all_matcher_pair_count else 0.0
            ),
            "fjsar_parity_failure_rate": (
                all_fjsar_parity_fail_count / all_matcher_pair_count
                if all_matcher_pair_count else 0.0
            ),
            "model_counts": all_matcher_model_counts,
        },
    }
    if fjsar_memory_cache is not None:
        result["fjsar_memory_cache"] = fjsar_memory_cache.stats()
    if args.matcher == "fjsar_attention_relational_graph_matching":
        attention_graph_summary = _summarize_attention_relational_graph_audits(
            attention_relational_graph_pair_records
        )
        root, _ext = os.path.splitext(args.output_json)
        attention_graph_path = f"{root}_attention_relational_graph_audit.json"
        os.makedirs(os.path.dirname(attention_graph_path) or ".", exist_ok=True)
        attention_graph_payload = {
            "matcher": args.matcher,
            "method_hypothesis": _fjsar_method_hypothesis(args),
            "candidate_topk": int(args.fjsar_candidate_topk),
            "subset": args.subset,
            "pairs_per_cat": int(args.pairs_per_cat),
            "split_seed": int(args.split_seed),
            "summary": attention_graph_summary,
            "pair_records": attention_relational_graph_pair_records,
        }
        with open(attention_graph_path, "w") as handle:
            json.dump(attention_graph_payload, handle, indent=2)
        result["attention_relational_graph_audit_path"] = attention_graph_path
        result["attention_relational_graph_audit_summary"] = attention_graph_summary
        result["attention_relational_graph_audit_pairs"] = len(
            attention_relational_graph_pair_records
        )
    if args.matcher == "fjsar_dense_partial_graph_matching":
        dense_partial_summary = _summarize_dense_partial_graph_audits(
            dense_partial_graph_pair_records
        )
        root, _ext = os.path.splitext(args.output_json)
        dense_partial_path = f"{root}_dense_partial_graph_audit.json"
        dense_partial_summary_path = f"{root}_dense_partial_graph_summary.json"
        os.makedirs(os.path.dirname(dense_partial_path) or ".", exist_ok=True)
        dense_partial_payload = {
            "matcher": args.matcher,
            "method_hypothesis": _fjsar_method_hypothesis(args),
            "candidate_topk": int(args.fjsar_candidate_topk),
            "subset": args.subset,
            "pairs_per_cat": int(args.pairs_per_cat),
            "split_seed": int(args.split_seed),
            "summary": dense_partial_summary,
            "pair_records": dense_partial_graph_pair_records,
        }
        with open(dense_partial_path, "w") as handle:
            json.dump(dense_partial_payload, handle, indent=2)
        with open(dense_partial_summary_path, "w") as handle:
            json.dump(
                {
                    "matcher": args.matcher,
                    "method_hypothesis": _fjsar_method_hypothesis(args),
                    "candidate_topk": int(args.fjsar_candidate_topk),
                    "subset": args.subset,
                    "pairs_per_cat": int(args.pairs_per_cat),
                    "split_seed": int(args.split_seed),
                    "summary": dense_partial_summary,
                },
                handle,
                indent=2,
            )
        result["dense_partial_graph_audit_path"] = dense_partial_path
        result["dense_partial_graph_summary_path"] = dense_partial_summary_path
        result["dense_partial_graph_audit_summary"] = dense_partial_summary
        result["dense_partial_graph_audit_pairs"] = len(
            dense_partial_graph_pair_records
        )
    if args.matcher == "fjsar_expert_preserving_attention_hypothesis_conditioned_replay":
        expert_summary = _summarize_expert_hypothesis_audits(
            expert_hypothesis_pair_records
        )
        root, _ext = os.path.splitext(args.output_json)
        expert_audit_path = f"{root}_expert_hypothesis_audit.json"
        expert_summary_path = f"{root}_expert_hypothesis_summary.json"
        os.makedirs(os.path.dirname(expert_audit_path) or ".", exist_ok=True)
        common_payload = {
            "matcher": args.matcher,
            "method_hypothesis": _fjsar_method_hypothesis(args),
            "candidate_topk": int(args.fjsar_candidate_topk),
            "subset": args.subset,
            "pairs_per_cat": int(args.pairs_per_cat),
            "split_seed": int(args.split_seed),
            "summary": expert_summary,
        }
        with open(expert_audit_path, "w") as handle:
            json.dump(
                {
                    **common_payload,
                    "pair_records": expert_hypothesis_pair_records,
                },
                handle,
                indent=2,
            )
        with open(expert_summary_path, "w") as handle:
            json.dump(common_payload, handle, indent=2)
        result["expert_hypothesis_audit_path"] = expert_audit_path
        result["expert_hypothesis_summary_path"] = expert_summary_path
        result["expert_hypothesis_audit_summary"] = expert_summary
        result["expert_hypothesis_audit_pairs"] = len(
            expert_hypothesis_pair_records
        )
    if args.matcher == "fjsar_pre_softmax_channelwise_identity":
        identity_summary = _summarize_pre_softmax_identity_audits(
            pre_softmax_identity_pair_records
        )
        root, _ext = os.path.splitext(args.output_json)
        identity_audit_path = f"{root}_pre_softmax_identity_audit.json"
        identity_summary_path = f"{root}_pre_softmax_identity_summary.json"
        os.makedirs(os.path.dirname(identity_audit_path) or ".", exist_ok=True)
        common_payload = {
            "matcher": args.matcher,
            "method_hypothesis": _fjsar_method_hypothesis(args),
            "candidate_topk": int(args.fjsar_candidate_topk),
            "subset": args.subset,
            "pairs_per_cat": int(args.pairs_per_cat),
            "split_seed": int(args.split_seed),
            "summary": identity_summary,
        }
        with open(identity_audit_path, "w") as handle:
            json.dump({**common_payload, "pair_records": pre_softmax_identity_pair_records}, handle, indent=2)
        with open(identity_summary_path, "w") as handle:
            json.dump(common_payload, handle, indent=2)
        result["pre_softmax_identity_audit_path"] = identity_audit_path
        result["pre_softmax_identity_summary_path"] = identity_summary_path
        result["pre_softmax_identity_audit_summary"] = identity_summary
        result["pre_softmax_identity_audit_pairs"] = len(pre_softmax_identity_pair_records)
    if args.matcher in (
        "fjsar_layer_routed_identity",
        "fjsar_pre_single_stream_identity",
    ):
        layer_summary = _summarize_layer_routed_identity_audits(
            layer_routed_identity_pair_records
        )
        root, _ext = os.path.splitext(args.output_json)
        output_stem = (
            "pre_single_stream_identity"
            if args.matcher == "fjsar_pre_single_stream_identity"
            else "layer_routed_identity"
        )
        layer_audit_path = f"{root}_{output_stem}_audit.json"
        layer_summary_path = f"{root}_{output_stem}_summary.json"
        os.makedirs(os.path.dirname(layer_audit_path) or ".", exist_ok=True)
        common_payload = {
            "matcher": args.matcher,
            "method_hypothesis": _fjsar_method_hypothesis(args),
            "candidate_topk": int(args.fjsar_candidate_topk),
            "multilayer_blocks": [int(block) for block in _fjsar_multilayer_blocks(args)],
            "subset": args.subset,
            "pairs_per_cat": int(args.pairs_per_cat),
            "split_seed": int(args.split_seed),
            "summary": layer_summary,
        }
        with open(layer_audit_path, "w") as handle:
            json.dump(
                {**common_payload, "pair_records": layer_routed_identity_pair_records},
                handle,
                indent=2,
            )
        with open(layer_summary_path, "w") as handle:
            json.dump(common_payload, handle, indent=2)
        result[f"{output_stem}_audit_path"] = layer_audit_path
        result[f"{output_stem}_summary_path"] = layer_summary_path
        result[f"{output_stem}_audit_summary"] = layer_summary
        result[f"{output_stem}_audit_pairs"] = len(
            layer_routed_identity_pair_records
        )
    candidate_descriptor_audit_summary = None
    if args.fjsar_candidate_descriptor_audit:
        candidate_descriptor_audit_summary = _summarize_candidate_descriptor_audit(candidate_dump_records)
        result["candidate_descriptor_audit_summary"] = candidate_descriptor_audit_summary
        audit_path = args.fjsar_candidate_descriptor_audit_path
        if not audit_path:
            root, _ext = os.path.splitext(args.output_json)
            audit_path = f"{root}_candidate_descriptor_audit.json"
        os.makedirs(os.path.dirname(audit_path) or ".", exist_ok=True)
        audit_payload = {
            "matcher": args.matcher,
            "candidate_topk": int(args.fjsar_candidate_topk),
            "dump_case_filter": str(args.fjsar_dump_case_filter),
            "dump_max_records": int(args.fjsar_dump_max_records),
            "subset": args.subset,
            "pairs_per_cat": int(args.pairs_per_cat),
            "split_seed": int(args.split_seed),
            "summary": candidate_descriptor_audit_summary,
            "records": candidate_dump_records,
        }
        with open(audit_path, "w") as handle:
            json.dump(audit_payload, handle, indent=2)
        result["candidate_descriptor_audit_path"] = audit_path
        result["candidate_descriptor_audit_records"] = len(candidate_dump_records)
    method_descriptor_audit_summary = None
    if args.fjsar_method_descriptor_audit:
        method_descriptor_audit_summary = _summarize_method_descriptor_audit(candidate_dump_records)
        result["method_descriptor_audit_summary"] = method_descriptor_audit_summary
        method_path = args.fjsar_method_descriptor_audit_path
        if not method_path:
            root, _ext = os.path.splitext(args.output_json)
            method_path = f"{root}_method_descriptor_audit.json"
        os.makedirs(os.path.dirname(method_path) or ".", exist_ok=True)
        method_payload = {
            "matcher": args.matcher,
            "candidate_topk": int(args.fjsar_candidate_topk),
            "dump_case_filter": str(args.fjsar_dump_case_filter),
            "dump_max_records": int(args.fjsar_dump_max_records),
            "subset": args.subset,
            "pairs_per_cat": int(args.pairs_per_cat),
            "split_seed": int(args.split_seed),
            "summary": method_descriptor_audit_summary,
            "records": candidate_dump_records,
        }
        with open(method_path, "w") as handle:
            json.dump(method_payload, handle, indent=2)
        result["method_descriptor_audit_path"] = method_path
        result["method_descriptor_audit_records"] = len(candidate_dump_records)
    transport_lift_branch_audit_summary = None
    if args.fjsar_transport_lift_branch_audit:
        transport_lift_branch_audit_summary = _summarize_transport_lift_branch_audit(candidate_dump_records)
        result["transport_lift_branch_audit_summary"] = transport_lift_branch_audit_summary
        branch_path = args.fjsar_transport_lift_branch_audit_path
        if not branch_path:
            root, _ext = os.path.splitext(args.output_json)
            branch_path = f"{root}_transport_lift_branch_audit.json"
        os.makedirs(os.path.dirname(branch_path) or ".", exist_ok=True)
        branch_payload = {
            "matcher": args.matcher,
            "candidate_topk": int(args.fjsar_candidate_topk),
            "dump_case_filter": str(args.fjsar_dump_case_filter),
            "dump_max_records": int(args.fjsar_dump_max_records),
            "subset": args.subset,
            "pairs_per_cat": int(args.pairs_per_cat),
            "split_seed": int(args.split_seed),
            "summary": transport_lift_branch_audit_summary,
            "records": candidate_dump_records,
        }
        with open(branch_path, "w") as handle:
            json.dump(branch_payload, handle, indent=2)
        result["transport_lift_branch_audit_path"] = branch_path
        result["transport_lift_branch_audit_records"] = len(candidate_dump_records)
    attention_flow_audit_summary = None
    if args.fjsar_attention_flow_audit:
        attention_flow_audit_summary = _summarize_attention_flow_audit(candidate_dump_records)
        result["attention_flow_audit_summary"] = attention_flow_audit_summary
        flow_path = args.fjsar_attention_flow_audit_path
        if not flow_path:
            root, _ext = os.path.splitext(args.output_json)
            flow_path = f"{root}_attention_flow_audit.json"
        os.makedirs(os.path.dirname(flow_path) or ".", exist_ok=True)
        flow_payload = {
            "matcher": args.matcher,
            "candidate_topk": int(args.fjsar_candidate_topk),
            "dump_case_filter": str(args.fjsar_dump_case_filter),
            "dump_max_records": int(args.fjsar_dump_max_records),
            "attention_flow_radius": int(args.fjsar_attention_flow_radius),
            "subset": args.subset,
            "pairs_per_cat": int(args.pairs_per_cat),
            "split_seed": int(args.split_seed),
            "summary": attention_flow_audit_summary,
            "records": candidate_dump_records,
        }
        with open(flow_path, "w") as handle:
            json.dump(flow_payload, handle, indent=2)
        result["attention_flow_audit_path"] = flow_path
        result["attention_flow_audit_records"] = len(candidate_dump_records)
    attention_kernel_audit_summary = None
    if args.fjsar_attention_kernel_audit:
        attention_kernel_audit_summary = _summarize_attention_kernel_audit(candidate_dump_records)
        result["attention_kernel_audit_summary"] = attention_kernel_audit_summary
        kernel_path = args.fjsar_attention_kernel_audit_path
        if not kernel_path:
            root, _ext = os.path.splitext(args.output_json)
            kernel_path = f"{root}_attention_kernel_audit.json"
        os.makedirs(os.path.dirname(kernel_path) or ".", exist_ok=True)
        kernel_payload = {
            "matcher": args.matcher,
            "candidate_topk": int(args.fjsar_candidate_topk),
            "dump_case_filter": str(args.fjsar_dump_case_filter),
            "dump_max_records": int(args.fjsar_dump_max_records),
            "attention_kernel_radius": int(args.fjsar_attention_kernel_radius),
            "attention_kernel_topk": [int(k) for k in args.fjsar_attention_kernel_topk],
            "subset": args.subset,
            "pairs_per_cat": int(args.pairs_per_cat),
            "split_seed": int(args.split_seed),
            "summary": attention_kernel_audit_summary,
            "records": candidate_dump_records,
        }
        with open(kernel_path, "w") as handle:
            json.dump(kernel_payload, handle, indent=2)
        result["attention_kernel_audit_path"] = kernel_path
        result["attention_kernel_audit_records"] = len(candidate_dump_records)
    basin_identity_audit_summary = None
    if args.fjsar_basin_identity_audit:
        basin_identity_audit_summary = _summarize_basin_identity_audit(candidate_dump_records)
        result["basin_identity_audit_summary"] = basin_identity_audit_summary
        basin_path = args.fjsar_basin_identity_audit_path
        if not basin_path:
            root, _ext = os.path.splitext(args.output_json)
            basin_path = f"{root}_basin_identity_audit.json"
        os.makedirs(os.path.dirname(basin_path) or ".", exist_ok=True)
        basin_payload = {
            "matcher": args.matcher,
            "candidate_topk": int(args.fjsar_candidate_topk),
            "dump_case_filter": str(args.fjsar_dump_case_filter),
            "dump_max_records": int(args.fjsar_dump_max_records),
            "basin_identity_topk": int(args.fjsar_basin_identity_topk),
            "basin_identity_radius": int(args.fjsar_basin_identity_radius),
            "basin_identity_rank_topk": [int(k) for k in args.fjsar_basin_identity_rank_topk],
            "subset": args.subset,
            "pairs_per_cat": int(args.pairs_per_cat),
            "split_seed": int(args.split_seed),
            "summary": basin_identity_audit_summary,
            "records": candidate_dump_records,
        }
        with open(basin_path, "w") as handle:
            json.dump(basin_payload, handle, indent=2)
        result["basin_identity_audit_path"] = basin_path
        result["basin_identity_audit_records"] = len(candidate_dump_records)
    operator_manifold_audit_summary = None
    if args.fjsar_operator_manifold_audit:
        operator_manifold_audit_summary = _summarize_operator_manifold_audit(candidate_dump_records)
        result["operator_manifold_audit_summary"] = operator_manifold_audit_summary
        manifold_path = args.fjsar_operator_manifold_audit_path
        if not manifold_path:
            root, _ext = os.path.splitext(args.output_json)
            manifold_path = f"{root}_operator_manifold_audit.json"
        os.makedirs(os.path.dirname(manifold_path) or ".", exist_ok=True)
        manifold_payload = {
            "matcher": args.matcher,
            "candidate_topk": int(args.fjsar_candidate_topk),
            "dump_case_filter": str(args.fjsar_dump_case_filter),
            "dump_max_records": int(args.fjsar_dump_max_records),
            "subset": args.subset,
            "pairs_per_cat": int(args.pairs_per_cat),
            "split_seed": int(args.split_seed),
            "summary": operator_manifold_audit_summary,
            "records": candidate_dump_records,
        }
        with open(manifold_path, "w") as handle:
            json.dump(manifold_payload, handle, indent=2)
        result["operator_manifold_audit_path"] = manifold_path
        result["operator_manifold_audit_records"] = len(candidate_dump_records)
    kernel_featureization_audit_summary = None
    if args.fjsar_kernel_featureization_audit:
        kernel_featureization_audit_summary = _summarize_kernel_featureization_audit(candidate_dump_records)
        result["kernel_featureization_audit_summary"] = kernel_featureization_audit_summary
        feature_path = args.fjsar_kernel_featureization_audit_path
        if not feature_path:
            root, _ext = os.path.splitext(args.output_json)
            feature_path = f"{root}_kernel_featureization_audit.json"
        os.makedirs(os.path.dirname(feature_path) or ".", exist_ok=True)
        feature_payload = {
            "matcher": args.matcher,
            "candidate_topk": int(args.fjsar_candidate_topk),
            "dump_case_filter": str(args.fjsar_dump_case_filter),
            "dump_max_records": int(args.fjsar_dump_max_records),
            "kernel_featureization_ranks": [int(rank) for rank in args.fjsar_kernel_featureization_ranks],
            "kernel_featureization_weights": [float(weight) for weight in args.fjsar_kernel_featureization_weights],
            "kernel_featureization_radius": int(args.fjsar_kernel_featureization_radius),
            "kernel_featureization_topk": [int(k) for k in args.fjsar_kernel_featureization_topk],
            "subset": args.subset,
            "pairs_per_cat": int(args.pairs_per_cat),
            "split_seed": int(args.split_seed),
            "summary": kernel_featureization_audit_summary,
            "records": candidate_dump_records,
        }
        with open(feature_path, "w") as handle:
            json.dump(feature_payload, handle, indent=2)
        result["kernel_featureization_audit_path"] = feature_path
        result["kernel_featureization_audit_records"] = len(candidate_dump_records)
    residual_readout_audit_summary = None
    if args.fjsar_residual_readout_audit:
        residual_readout_audit_summary = _summarize_residual_readout_audit(candidate_dump_records)
        result["residual_readout_audit_summary"] = residual_readout_audit_summary
        readout_path = args.fjsar_residual_readout_audit_path
        if not readout_path:
            root, _ext = os.path.splitext(args.output_json)
            readout_path = f"{root}_residual_readout_audit.json"
        os.makedirs(os.path.dirname(readout_path) or ".", exist_ok=True)
        readout_payload = {
            "matcher": args.matcher,
            "candidate_topk": int(args.fjsar_candidate_topk),
            "dump_case_filter": str(args.fjsar_dump_case_filter),
            "dump_max_records": int(args.fjsar_dump_max_records),
            "residual_readout_topk": [int(k) for k in args.fjsar_residual_readout_topk],
            "subset": args.subset,
            "pairs_per_cat": int(args.pairs_per_cat),
            "split_seed": int(args.split_seed),
            "summary": residual_readout_audit_summary,
            "records": candidate_dump_records,
        }
        with open(readout_path, "w") as handle:
            json.dump(readout_payload, handle, indent=2)
        result["residual_readout_audit_path"] = readout_path
        result["residual_readout_audit_records"] = len(candidate_dump_records)
    latent_expert_audit_summary = None
    if args.fjsar_latent_expert_audit:
        latent_expert_audit_summary = _summarize_latent_expert_audit(candidate_dump_records)
        result["latent_expert_audit_summary"] = latent_expert_audit_summary
        root, _ext = os.path.splitext(args.output_json)
        latent_path = args.fjsar_latent_expert_audit_path or f"{root}_latent_expert_audit.json"
        summary_path = args.fjsar_latent_expert_summary_path or f"{root}_latent_expert_summary.json"
        os.makedirs(os.path.dirname(latent_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
        identity_fields = (
            "category",
            "pair_json",
            "src_image",
            "trg_image",
            "keypoint_index",
            "source_point",
            "target_point",
            "threshold",
            "baseline_pck_hit",
            "method_pck_hit",
            "attention_top1_pck_hit",
            "attention_topk_pck_hit",
            "oracle_gap_case",
            "attention_harms_native_case",
            "attention_rescues_native_case",
            "gt_ranks",
        )
        latent_records = [
            {
                **{key: row.get(key) for key in identity_fields},
                "latent_expert_audit": row.get("latent_expert_audit"),
            }
            for row in candidate_dump_records
            if isinstance(row.get("latent_expert_audit"), dict)
        ]
        hypothesis = {
            "question": (
                "Are correct mutual-attention candidates supported by a stable minority of FLUX heads or "
                "ensemble members whose signal is destroyed by early ensemble/head averaging?"
            ),
            "candidate_contract": "mutual_cross_attention_topk_only",
            "support_signal": "log_exact_mutual_cross_probability",
            "rank_control_signal": "bidirectional_negative_log_rank",
            "early_average_reference": "aggregated_attention",
            "unsupervised_selectors": [
                "stable_head_1",
                "stable_head_2",
                "stable_head_4",
                "stable_member_1",
                "confident_expert_1",
            ],
            "gt_only_upper_bounds": [
                "oracle_pair_head_1",
                "oracle_pair_head_2",
                "oracle_pair_head_4",
                "oracle_pair_member_1",
                "oracle_pair_expert_1",
                "any_head_top1",
                "any_member_top1",
                "any_expert_top1",
            ],
            "gt_used_for_unsupervised_selector": False,
            "prediction_changed": False,
        }
        latent_payload = {
            "matcher": args.matcher,
            "hypothesis": hypothesis,
            "candidate_topk": int(args.fjsar_candidate_topk),
            "latent_expert_topk": [int(k) for k in args.fjsar_latent_expert_topk],
            "dump_case_filter": str(args.fjsar_dump_case_filter),
            "dump_max_records": int(args.fjsar_dump_max_records),
            "subset": args.subset,
            "pairs_per_cat": int(args.pairs_per_cat),
            "split_seed": int(args.split_seed),
            "summary": latent_expert_audit_summary,
            "records": latent_records,
        }
        with open(latent_path, "w") as handle:
            json.dump(latent_payload, handle, indent=2)
        with open(summary_path, "w") as handle:
            json.dump(
                {
                    "matcher": args.matcher,
                    "hypothesis": hypothesis,
                    "candidate_topk": int(args.fjsar_candidate_topk),
                    "subset": args.subset,
                    "pairs_per_cat": int(args.pairs_per_cat),
                    "split_seed": int(args.split_seed),
                    "summary": latent_expert_audit_summary,
                },
                handle,
                indent=2,
            )
        result["latent_expert_audit_path"] = latent_path
        result["latent_expert_summary_path"] = summary_path
        result["latent_expert_audit_records"] = len(latent_records)
    candidate_clamped_causal_replay_audit_summary = None
    if args.fjsar_candidate_clamped_causal_replay_audit:
        candidate_clamped_causal_replay_audit_summary = (
            _summarize_candidate_clamped_causal_replay_audit(candidate_dump_records)
        )
        result["candidate_clamped_causal_replay_audit_summary"] = (
            candidate_clamped_causal_replay_audit_summary
        )
        root, _ext = os.path.splitext(args.output_json)
        audit_path = (
            args.fjsar_candidate_clamped_causal_replay_audit_path
            or f"{root}_candidate_clamped_causal_replay_audit.json"
        )
        summary_path = (
            args.fjsar_candidate_clamped_causal_replay_summary_path
            or f"{root}_candidate_clamped_causal_replay_summary.json"
        )
        os.makedirs(os.path.dirname(audit_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
        identity_fields = (
            "category",
            "pair_json",
            "src_image",
            "trg_image",
            "keypoint_index",
            "source_point",
            "target_point",
            "threshold",
            "baseline_pck_hit",
            "method_pck_hit",
            "attention_top1_pck_hit",
            "attention_topk_pck_hit",
            "oracle_gap_case",
            "attention_harms_native_case",
            "attention_rescues_native_case",
            "gt_ranks",
        )
        audit_records = [
            {
                **{key: row.get(key) for key in identity_fields},
                "candidate_clamped_causal_replay_audit": row.get(
                    "candidate_clamped_causal_replay_audit"
                ),
            }
            for row in candidate_dump_records
            if isinstance(row.get("candidate_clamped_causal_replay_audit"), dict)
        ]
        hypothesis = {
            "question": (
                "Does a correct exact-attention candidate cause a more stable free "
                "bidirectional QK relation after a mass-preserving candidate clamp?"
            ),
            "candidate_contract": "exact_mutual_cross_attention_topk_only",
            "intervention": (
                "preserve exact local contribution and original per-expert/head total "
                "cross mass; replace only conditional cross value in both directions"
            ),
            "causal_path": (
                "clamp block attention projection -> original AdaLN gate -> residual -> "
                "adjacent unclamped block QK readout"
            ),
            "primary_signal": "post_release_bidirectional_negative_log_rank",
            "controls": [
                "pre_intervention_bidirectional_negative_log_rank",
                "causal_rank_improvement",
                "post_release_mutual_top1_vote",
                "post_release_mutual_top5_vote",
            ],
            "falsification_group": "both_wrong_top20_hit",
            "minimum_absolute_top1_rate": 0.184,
            "target_recovery_rate_for_75_pck": 0.329,
            "native_candidate_injected": False,
            "native_fallback_used": False,
            "gt_used_for_scoring": False,
            "prediction_changed": False,
        }
        audit_payload = {
            "matcher": args.matcher,
            "hypothesis": hypothesis,
            "candidate_topk": int(args.fjsar_candidate_topk),
            "causal_replay_topk": [
                int(k) for k in args.fjsar_candidate_clamped_causal_replay_topk
            ],
            "dump_case_filter": str(args.fjsar_dump_case_filter),
            "dump_max_records": int(args.fjsar_dump_max_records),
            "subset": args.subset,
            "pairs_per_cat": int(args.pairs_per_cat),
            "split_seed": int(args.split_seed),
            "summary": candidate_clamped_causal_replay_audit_summary,
            "records": audit_records,
        }
        with open(audit_path, "w") as handle:
            json.dump(audit_payload, handle, indent=2)
        with open(summary_path, "w") as handle:
            json.dump(
                {
                    "matcher": args.matcher,
                    "hypothesis": hypothesis,
                    "candidate_topk": int(args.fjsar_candidate_topk),
                    "subset": args.subset,
                    "pairs_per_cat": int(args.pairs_per_cat),
                    "split_seed": int(args.split_seed),
                    "summary": candidate_clamped_causal_replay_audit_summary,
                },
                handle,
                indent=2,
            )
        result["candidate_clamped_causal_replay_audit_path"] = audit_path
        result["candidate_clamped_causal_replay_summary_path"] = summary_path
        result["candidate_clamped_causal_replay_audit_records"] = len(audit_records)
    counterfactual_fingerprint_audit_summary = None
    if args.fjsar_counterfactual_fingerprint_audit:
        counterfactual_fingerprint_audit_summary = (
            _summarize_counterfactual_fingerprint_audit(candidate_dump_records)
        )
        result["counterfactual_fingerprint_audit_summary"] = (
            counterfactual_fingerprint_audit_summary
        )
        root, _ext = os.path.splitext(args.output_json)
        audit_path = (
            args.fjsar_counterfactual_fingerprint_audit_path
            or f"{root}_counterfactual_fingerprint_audit.json"
        )
        summary_path = (
            args.fjsar_counterfactual_fingerprint_summary_path
            or f"{root}_counterfactual_fingerprint_summary.json"
        )
        os.makedirs(os.path.dirname(audit_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
        identity_fields = (
            "category", "pair_json", "src_image", "trg_image", "keypoint_index",
            "source_point", "target_point", "threshold", "baseline_pck_hit",
            "method_pck_hit", "attention_top1_pck_hit", "attention_topk_pck_hit",
            "oracle_gap_case", "attention_harms_native_case", "attention_rescues_native_case",
            "gt_ranks",
        )
        audit_records = [
            {
                **{key: row.get(key) for key in identity_fields},
                "counterfactual_fingerprint_audit": row.get(
                    "counterfactual_fingerprint_audit"
                ),
            }
            for row in candidate_dump_records
            if isinstance(row.get("counterfactual_fingerprint_audit"), dict)
        ]
        hypothesis = {
            "question": (
                "Does a candidate preserve a reciprocal bidirectional QK response "
                "under small mass-preserving value-dose interventions?"
            ),
            "intervention": (
                "scale only the candidate conditional cross value while preserving "
                "the exact local term and original per-head cross mass"
            ),
            "response_fingerprint": "source/target post-release rank curves across scales",
            "primary_signal": "fingerprint_score",
            "falsification_group": "both_wrong_top20_hit",
            "scales": [float(scale) for scale in args.fjsar_counterfactual_fingerprint_scales],
            "candidate_topk_only": True,
            "gt_used_for_scoring": False,
            "prediction_changed": False,
        }
        audit_payload = {
            "matcher": args.matcher,
            "hypothesis": hypothesis,
            "candidate_topk": int(args.fjsar_candidate_topk),
            "subset": args.subset,
            "pairs_per_cat": int(args.pairs_per_cat),
            "split_seed": int(args.split_seed),
            "summary": counterfactual_fingerprint_audit_summary,
            "records": audit_records,
        }
        with open(audit_path, "w") as handle:
            json.dump(audit_payload, handle, indent=2)
        with open(summary_path, "w") as handle:
            json.dump(
                {
                    "matcher": args.matcher,
                    "hypothesis": hypothesis,
                    "candidate_topk": int(args.fjsar_candidate_topk),
                    "subset": args.subset,
                    "pairs_per_cat": int(args.pairs_per_cat),
                    "split_seed": int(args.split_seed),
                    "summary": counterfactual_fingerprint_audit_summary,
                },
                handle,
                indent=2,
            )
        result["counterfactual_fingerprint_audit_path"] = audit_path
        result["counterfactual_fingerprint_summary_path"] = summary_path
        result["counterfactual_fingerprint_audit_records"] = len(audit_records)
    persistent_candidate_slot_replay_audit_summary = None
    if args.fjsar_persistent_candidate_slot_replay_audit:
        persistent_candidate_slot_replay_audit_summary = (
            _summarize_persistent_candidate_slot_replay_audit(candidate_dump_records)
        )
        result["persistent_candidate_slot_replay_audit_summary"] = (
            persistent_candidate_slot_replay_audit_summary
        )
        root, _ext = os.path.splitext(args.output_json)
        audit_path = (
            args.fjsar_persistent_candidate_slot_replay_audit_path
            or f"{root}_persistent_candidate_slot_replay_audit.json"
        )
        summary_path = (
            args.fjsar_persistent_candidate_slot_replay_summary_path
            or f"{root}_persistent_candidate_slot_replay_summary.json"
        )
        os.makedirs(os.path.dirname(audit_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
        identity_fields = (
            "category",
            "pair_json",
            "src_image",
            "trg_image",
            "keypoint_index",
            "source_point",
            "target_point",
            "threshold",
            "baseline_pck_hit",
            "method_pck_hit",
            "attention_top1_pck_hit",
            "attention_topk_pck_hit",
            "oracle_gap_case",
            "attention_harms_native_case",
            "attention_rescues_native_case",
            "gt_ranks",
        )
        audit_records = [
            {
                **{key: row.get(key) for key in identity_fields},
                "persistent_candidate_slot_replay_audit": row.get(
                    "persistent_candidate_slot_replay_audit"
                ),
            }
            for row in candidate_dump_records
            if isinstance(row.get("persistent_candidate_slot_replay_audit"), dict)
        ]
        hypothesis = {
            "question": (
                "Does keeping each exact-attention top20 candidate as an isolated "
                "two-block FLUX replay branch preserve point identity that early "
                "candidate averaging destroys?"
            ),
            "candidate_contract": "exact_mutual_cross_attention_top20_only",
            "intervention": (
                "preserve full local self-attention and original full cross-branch mass; "
                "replace only the cross value by one candidate in both directions at every block"
            ),
            "replay_depth": 2,
            "primary_signal": "directional_anchor_cosine",
            "self_fulfilling_control": "pair_cosine_is_secondary_only",
            "falsification_group": "both_wrong_top20_hit",
            "native_candidate_injected": False,
            "native_fallback_used": False,
            "gt_used_for_scoring": False,
            "prediction_changed": False,
        }
        payload = {
            "matcher": args.matcher,
            "hypothesis": hypothesis,
            "candidate_topk": int(args.fjsar_candidate_topk),
            "persistent_replay_topk": [
                int(k) for k in args.fjsar_persistent_candidate_slot_replay_topk
            ],
            "hypothesis_chunk": int(args.fjsar_persistent_candidate_slot_replay_chunk),
            "canonical_replay_cache": os.path.abspath(args.fjsar_disk_cache_path),
            "require_disk_cache": bool(args.fjsar_require_disk_cache),
            "dump_case_filter": str(args.fjsar_dump_case_filter),
            "dump_max_records": int(args.fjsar_dump_max_records),
            "subset": args.subset,
            "pairs_per_cat": int(args.pairs_per_cat),
            "split_seed": int(args.split_seed),
            "summary": persistent_candidate_slot_replay_audit_summary,
        }
        with open(audit_path, "w") as handle:
            json.dump({**payload, "records": audit_records}, handle, indent=2)
        with open(summary_path, "w") as handle:
            json.dump(payload, handle, indent=2)
        result["persistent_candidate_slot_replay_audit_path"] = audit_path
        result["persistent_candidate_slot_replay_summary_path"] = summary_path
        result["persistent_candidate_slot_replay_audit_records"] = len(audit_records)
    local_relational_identity_audit_summary = None
    if args.fjsar_local_relational_identity_audit:
        local_relational_identity_audit_summary = _summarize_local_relational_identity_audit(candidate_dump_records)
        result["local_relational_identity_audit_summary"] = local_relational_identity_audit_summary
        local_path = args.fjsar_local_relational_identity_audit_path
        if not local_path:
            root, _ext = os.path.splitext(args.output_json)
            local_path = f"{root}_local_relational_identity_audit.json"
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        local_payload = {
            "matcher": args.matcher,
            "candidate_topk": int(args.fjsar_candidate_topk),
            "dump_case_filter": str(args.fjsar_dump_case_filter),
            "dump_max_records": int(args.fjsar_dump_max_records),
            "local_relational_radius": int(args.fjsar_local_relational_radius),
            "subset": args.subset,
            "pairs_per_cat": int(args.pairs_per_cat),
            "split_seed": int(args.split_seed),
            "summary": local_relational_identity_audit_summary,
            "records": candidate_dump_records,
        }
        with open(local_path, "w") as handle:
            json.dump(local_payload, handle, indent=2)
        result["local_relational_identity_audit_path"] = local_path
        result["local_relational_identity_audit_records"] = len(candidate_dump_records)
    dense_candidate_edge_audit_summary = None
    if args.fjsar_dense_candidate_edge_audit:
        dense_candidate_edge_audit_summary = _summarize_dense_candidate_edge_audit(
            candidate_dump_records
        )
        result["dense_candidate_edge_audit_summary"] = dense_candidate_edge_audit_summary
        root, _ext = os.path.splitext(args.output_json)
        audit_path = (
            args.fjsar_dense_candidate_edge_audit_path
            or f"{root}_dense_candidate_edge_audit.json"
        )
        summary_path = (
            args.fjsar_dense_candidate_edge_summary_path
            or f"{root}_dense_candidate_edge_summary.json"
        )
        os.makedirs(os.path.dirname(audit_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
        identity_fields = (
            "category",
            "pair_json",
            "src_image",
            "trg_image",
            "keypoint_index",
            "source_point",
            "target_point",
            "threshold",
            "baseline_pck_hit",
            "method_pck_hit",
            "attention_top1_pck_hit",
            "attention_topk_pck_hit",
            "oracle_gap_case",
            "attention_harms_native_case",
            "attention_rescues_native_case",
            "gt_ranks",
        )
        audit_records = [
            {
                **{key: row.get(key) for key in identity_fields},
                "dense_candidate_edge_audit": row.get("dense_candidate_edge_audit"),
            }
            for row in candidate_dump_records
            if isinstance(row.get("dense_candidate_edge_audit"), dict)
        ]
        hypothesis = {
            "question": (
                "Can local edge evidence on the dense source graph separate the correct "
                "mutual-attention candidate from a stronger but relationally inconsistent top-1?"
            ),
            "falsification_rule": (
                "Do not build the partial graph solver unless edge-only signals recover "
                "attention top-1 errors with positive net lift, especially on oracle-gap points."
            ),
            "candidate_contract": "mutual_cross_attention_topk_only",
            "source_nodes": "all_dense_flux_image_tokens",
            "source_edges": "directed_local_grid_edges",
            "signals": [
                "attention_unary_control",
                "dense_edge_spatial_message",
                "dense_edge_relation_message",
                "dense_edge_joint_message",
                "dense_partial_graph_one_step_belief",
            ],
            "relation_feature_role": "edge_self_similarity_only",
            "gt_used_for_scoring": False,
            "native_candidate_injected": False,
            "native_fallback_used": False,
            "prediction_changed": False,
            "partial_solver_status": "data_contract_reserved_not_executed",
        }
        audit_payload = {
            "matcher": args.matcher,
            "hypothesis": hypothesis,
            "candidate_topk": int(args.fjsar_candidate_topk),
            "dense_candidate_edge_radius": int(args.fjsar_dense_candidate_edge_radius),
            "dump_case_filter": str(args.fjsar_dump_case_filter),
            "dump_max_records": int(args.fjsar_dump_max_records),
            "subset": args.subset,
            "pairs_per_cat": int(args.pairs_per_cat),
            "split_seed": int(args.split_seed),
            "summary": dense_candidate_edge_audit_summary,
            "records": audit_records,
        }
        with open(audit_path, "w") as handle:
            json.dump(audit_payload, handle, indent=2)
        with open(summary_path, "w") as handle:
            json.dump(
                {
                    "matcher": args.matcher,
                    "hypothesis": hypothesis,
                    "candidate_topk": int(args.fjsar_candidate_topk),
                    "dense_candidate_edge_radius": int(
                        args.fjsar_dense_candidate_edge_radius
                    ),
                    "subset": args.subset,
                    "pairs_per_cat": int(args.pairs_per_cat),
                    "split_seed": int(args.split_seed),
                    "summary": dense_candidate_edge_audit_summary,
                },
                handle,
                indent=2,
            )
        result["dense_candidate_edge_audit_path"] = audit_path
        result["dense_candidate_edge_summary_path"] = summary_path
        result["dense_candidate_edge_audit_records"] = len(audit_records)
    dense_transport_consistency_audit_summary = None
    if args.fjsar_dense_transport_consistency_audit:
        dense_transport_consistency_audit_summary = _summarize_dense_transport_consistency_audit(candidate_dump_records)
        result["dense_transport_consistency_audit_summary"] = dense_transport_consistency_audit_summary
        dense_path = args.fjsar_dense_transport_consistency_audit_path
        if not dense_path:
            root, _ext = os.path.splitext(args.output_json)
            dense_path = f"{root}_dense_transport_consistency_audit.json"
        os.makedirs(os.path.dirname(dense_path) or ".", exist_ok=True)
        dense_payload = {
            "matcher": args.matcher,
            "candidate_topk": int(args.fjsar_candidate_topk),
            "dump_case_filter": str(args.fjsar_dump_case_filter),
            "dump_max_records": int(args.fjsar_dump_max_records),
            "dense_transport_topk": [int(k) for k in args.fjsar_dense_transport_topk],
            "subset": args.subset,
            "pairs_per_cat": int(args.pairs_per_cat),
            "split_seed": int(args.split_seed),
            "summary": dense_transport_consistency_audit_summary,
            "records": candidate_dump_records,
        }
        with open(dense_path, "w") as handle:
            json.dump(dense_payload, handle, indent=2)
        result["dense_transport_consistency_audit_path"] = dense_path
        result["dense_transport_consistency_audit_records"] = len(candidate_dump_records)
    candidate_field_consistency_audit_summary = None
    if args.fjsar_candidate_field_consistency_audit:
        candidate_field_consistency_audit_summary = _summarize_candidate_field_consistency_audit(candidate_dump_records)
        result["candidate_field_consistency_audit_summary"] = candidate_field_consistency_audit_summary
        field_path = args.fjsar_candidate_field_consistency_audit_path
        if not field_path:
            root, _ext = os.path.splitext(args.output_json)
            field_path = f"{root}_candidate_field_consistency_audit.json"
        os.makedirs(os.path.dirname(field_path) or ".", exist_ok=True)
        field_payload = {
            "matcher": args.matcher,
            "candidate_topk": int(args.fjsar_candidate_topk),
            "candidate_field_topm": int(args.fjsar_candidate_field_topm),
            "candidate_field_source": str(args.fjsar_candidate_field_source),
            "dump_case_filter": str(args.fjsar_dump_case_filter),
            "dump_max_records": int(args.fjsar_dump_max_records),
            "subset": args.subset,
            "pairs_per_cat": int(args.pairs_per_cat),
            "split_seed": int(args.split_seed),
            "summary": candidate_field_consistency_audit_summary,
            "records": candidate_dump_records,
        }
        with open(field_path, "w") as handle:
            json.dump(field_payload, handle, indent=2)
        result["candidate_field_consistency_audit_path"] = field_path
        result["candidate_field_consistency_audit_records"] = len(candidate_dump_records)
    anchor_topology_audit_summary = None
    if args.fjsar_anchor_topology_audit:
        anchor_topology_audit_summary = _summarize_anchor_topology_audit(
            candidate_dump_records,
            anchor_topology_pair_records,
        )
        result["anchor_topology_audit_summary"] = anchor_topology_audit_summary
        anchor_path = args.fjsar_anchor_topology_audit_path
        if not anchor_path:
            root, _ext = os.path.splitext(args.output_json)
            anchor_path = f"{root}_anchor_topology_audit.json"
        os.makedirs(os.path.dirname(anchor_path) or ".", exist_ok=True)
        anchor_payload = {
            "matcher": args.matcher,
            "candidate_topk": int(args.fjsar_candidate_topk),
            "dump_case_filter": str(args.fjsar_dump_case_filter),
            "dump_max_records": int(args.fjsar_dump_max_records),
            "subset": args.subset,
            "pairs_per_cat": int(args.pairs_per_cat),
            "split_seed": int(args.split_seed),
            "summary": anchor_topology_audit_summary,
            "pair_records": anchor_topology_pair_records,
            "records": candidate_dump_records,
        }
        with open(anchor_path, "w") as handle:
            json.dump(anchor_payload, handle, indent=2)
        result["anchor_topology_audit_path"] = anchor_path
        result["anchor_topology_audit_records"] = len(candidate_dump_records)
        result["anchor_topology_pair_records"] = len(anchor_topology_pair_records)
    multilayer_identity_audit_summary = None
    if args.fjsar_multilayer_identity_audit:
        multilayer_identity_audit_summary = _summarize_multilayer_identity_audit(candidate_dump_records)
        result["multilayer_identity_audit_summary"] = multilayer_identity_audit_summary
        multilayer_path = args.fjsar_multilayer_identity_audit_path
        if not multilayer_path:
            root, _ext = os.path.splitext(args.output_json)
            multilayer_path = f"{root}_multilayer_identity_audit.json"
        os.makedirs(os.path.dirname(multilayer_path) or ".", exist_ok=True)
        multilayer_payload = {
            "matcher": args.matcher,
            "candidate_topk": int(args.fjsar_candidate_topk),
            "dump_case_filter": str(args.fjsar_dump_case_filter),
            "dump_max_records": int(args.fjsar_dump_max_records),
            "multilayer_blocks": [int(block) for block in _fjsar_multilayer_blocks(args)],
            "subset": args.subset,
            "pairs_per_cat": int(args.pairs_per_cat),
            "split_seed": int(args.split_seed),
            "summary": multilayer_identity_audit_summary,
            "records": candidate_dump_records,
        }
        with open(multilayer_path, "w") as handle:
            json.dump(multilayer_payload, handle, indent=2)
        result["multilayer_identity_audit_path"] = multilayer_path
        result["multilayer_identity_audit_records"] = len(candidate_dump_records)
    transport_factorization_audit_summary = None
    if args.fjsar_transport_factorization_audit:
        transport_factorization_audit_summary = _summarize_transport_factorization_audit(candidate_dump_records)
        result["transport_factorization_audit_summary"] = transport_factorization_audit_summary
        factor_path = args.fjsar_transport_factorization_audit_path
        if not factor_path:
            root, _ext = os.path.splitext(args.output_json)
            factor_path = f"{root}_transport_factorization_audit.json"
        os.makedirs(os.path.dirname(factor_path) or ".", exist_ok=True)
        factor_payload = {
            "matcher": args.matcher,
            "candidate_topk": int(args.fjsar_candidate_topk),
            "dump_case_filter": str(args.fjsar_dump_case_filter),
            "dump_max_records": int(args.fjsar_dump_max_records),
            "transport_factorization_radius": int(args.fjsar_transport_factorization_radius),
            "transport_factorization_basis_radius": int(args.fjsar_transport_factorization_basis_radius),
            "subset": args.subset,
            "pairs_per_cat": int(args.pairs_per_cat),
            "split_seed": int(args.split_seed),
            "summary": transport_factorization_audit_summary,
            "records": candidate_dump_records,
        }
        with open(factor_path, "w") as handle:
            json.dump(factor_payload, handle, indent=2)
        result["transport_factorization_audit_path"] = factor_path
        result["transport_factorization_audit_records"] = len(candidate_dump_records)
    trajectory_identity_audit_summary = None
    if args.matcher == "fjsar_cross_attention_trajectory" and candidate_dump_records:
        trajectory_identity_audit_summary = _summarize_trajectory_identity_audit(candidate_dump_records)
        result["trajectory_identity_audit_summary"] = trajectory_identity_audit_summary
        trajectory_path = args.fjsar_trajectory_identity_audit_path
        if not trajectory_path:
            root, _ext = os.path.splitext(args.output_json)
            trajectory_path = f"{root}_trajectory_identity_audit.json"
        os.makedirs(os.path.dirname(trajectory_path) or ".", exist_ok=True)
        trajectory_payload = {
            "matcher": args.matcher,
            "method_hypothesis": _fjsar_method_hypothesis(args),
            "candidate_topk": int(args.fjsar_candidate_topk),
            "trajectory_blocks": [int(block) for block in _fjsar_trajectory_blocks(args)],
            "dump_case_filter": str(args.fjsar_dump_case_filter),
            "dump_max_records": int(args.fjsar_dump_max_records),
            "subset": args.subset,
            "pairs_per_cat": int(args.pairs_per_cat),
            "split_seed": int(args.split_seed),
            "summary": trajectory_identity_audit_summary,
            "records": candidate_dump_records,
        }
        with open(trajectory_path, "w") as handle:
            json.dump(trajectory_payload, handle, indent=2)
        result["trajectory_identity_audit_path"] = trajectory_path
        result["trajectory_identity_audit_records"] = len(candidate_dump_records)
    multi_timestep_attention_identity_audit_summary = None
    if args.fjsar_multi_timestep_attention_identity_audit and candidate_dump_records:
        multi_timestep_attention_identity_audit_summary = _summarize_multi_timestep_attention_identity_audit(candidate_dump_records)
        result["multi_timestep_attention_identity_audit_summary"] = multi_timestep_attention_identity_audit_summary
        multi_timestep_path = args.fjsar_multi_timestep_attention_identity_audit_path
        if not multi_timestep_path:
            root, _ext = os.path.splitext(args.output_json)
            multi_timestep_path = f"{root}_multi_timestep_attention_identity_audit.json"
        os.makedirs(os.path.dirname(multi_timestep_path) or ".", exist_ok=True)
        multi_timestep_payload = {
            "matcher": args.matcher,
            "method_hypothesis": _fjsar_method_hypothesis(args),
            "candidate_topk": int(args.fjsar_candidate_topk),
            "multi_timestep_values": [int(value) for value in _fjsar_multi_timestep_values(args)],
            "dump_case_filter": str(args.fjsar_dump_case_filter),
            "dump_max_records": int(args.fjsar_dump_max_records),
            "subset": args.subset,
            "pairs_per_cat": int(args.pairs_per_cat),
            "split_seed": int(args.split_seed),
            "summary": multi_timestep_attention_identity_audit_summary,
            "records": candidate_dump_records,
        }
        with open(multi_timestep_path, "w") as handle:
            json.dump(multi_timestep_payload, handle, indent=2)
        result["multi_timestep_attention_identity_audit_path"] = multi_timestep_path
        result["multi_timestep_attention_identity_audit_records"] = len(candidate_dump_records)
    if args.fjsar_identity_decodability_audit:
        if not identity_decodability_shards:
            raise RuntimeError("identity decodability audit produced no binary shards")
        from analyze_identity_decodability import analyze_identity_decodability

        # Feature extraction is complete. Release the bounded replay RAM cache
        # before materializing category-held-out probe matrices.
        fjsar_memory_cache = None
        src_entry = trg_entry = None
        gc.collect()

        output_root, _output_ext = os.path.splitext(args.output_json)
        manifest_path = (
            args.fjsar_identity_decodability_manifest_path
            or f"{output_root}_identity_decodability_manifest.json"
        )
        summary_path = (
            args.fjsar_identity_decodability_summary_path
            or f"{output_root}_identity_decodability_summary.json"
        )
        manifest_root = os.path.dirname(os.path.abspath(manifest_path))
        os.makedirs(manifest_root or ".", exist_ok=True)
        manifest_payload = {
            "audit": "candidate_identity_decodability",
            "format_version": 1,
            "matcher": args.matcher,
            "candidate_topk": int(args.fjsar_candidate_topk),
            "subset": args.subset,
            "pairs_per_cat": int(args.pairs_per_cat),
            "split_seed": int(args.split_seed),
            "feature_source": identity_decodability_feature_source,
            "canonical_replay_cache": (
                os.path.abspath(args.fjsar_disk_cache_path)
                if identity_decodability_feature_source == "canonical_disk_cache"
                else None
            ),
            "require_disk_cache": bool(args.fjsar_require_disk_cache),
            "shards": [
                os.path.relpath(os.path.abspath(path), manifest_root)
                for path in identity_decodability_shards
            ],
        }
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest_payload, handle, indent=2)
        print(
            "Run category-held-out identity decodability probes: "
            f"{len(identity_decodability_shards)} pair shards"
        )
        decodability_summary = analyze_identity_decodability(
            identity_decodability_shards,
            output_path=summary_path,
            fold_count=args.fjsar_identity_decodability_folds,
            seed=args.split_seed,
            run_mlp=not args.fjsar_identity_decodability_skip_mlp,
        )
        result["identity_decodability_manifest_path"] = manifest_path
        result["identity_decodability_summary_path"] = summary_path
        result["identity_decodability_shards"] = len(identity_decodability_shards)
        result["identity_decodability_mechanism_decision"] = decodability_summary[
            "mechanism_decision"
        ]
    if args.fjsar_dump_candidates:
        dump_path = args.fjsar_dump_candidates_path
        if not dump_path:
            root, _ext = os.path.splitext(args.output_json)
            dump_path = f"{root}_candidates.json"
        os.makedirs(os.path.dirname(dump_path) or ".", exist_ok=True)
        dump_summary = {
            "records": len(candidate_dump_records),
            "baseline_correct": sum(1 for row in candidate_dump_records if row.get("baseline_pck_hit")),
            "method_correct": sum(1 for row in candidate_dump_records if row.get("method_pck_hit")),
            "attention_top1_correct": sum(
                1 for row in candidate_dump_records if row.get("attention_top1_pck_hit")
            ),
            "attention_topk_correct": sum(
                1 for row in candidate_dump_records if row.get("attention_topk_pck_hit")
            ),
            "oracle_gap_cases": sum(1 for row in candidate_dump_records if row.get("oracle_gap_case")),
            "attention_harms_native_cases": sum(
                1 for row in candidate_dump_records if row.get("attention_harms_native_case")
            ),
            "attention_rescues_native_cases": sum(
                1 for row in candidate_dump_records if row.get("attention_rescues_native_case")
            ),
        }
        dump_payload = {
            "matcher": args.matcher,
            "method_hypothesis": _fjsar_method_hypothesis(args),
            "candidate_topk": int(args.fjsar_candidate_topk),
            "dump_case_filter": str(args.fjsar_dump_case_filter),
            "dump_max_records": int(args.fjsar_dump_max_records),
            "subset": args.subset,
            "pairs_per_cat": int(args.pairs_per_cat),
            "split_seed": int(args.split_seed),
            "summary": dump_summary,
            "candidate_descriptor_audit_summary": candidate_descriptor_audit_summary,
            "method_descriptor_audit_summary": method_descriptor_audit_summary,
            "transport_lift_branch_audit_summary": transport_lift_branch_audit_summary,
            "attention_flow_audit_summary": attention_flow_audit_summary,
            "attention_kernel_audit_summary": attention_kernel_audit_summary,
            "basin_identity_audit_summary": basin_identity_audit_summary,
            "operator_manifold_audit_summary": operator_manifold_audit_summary,
            "kernel_featureization_audit_summary": kernel_featureization_audit_summary,
            "residual_readout_audit_summary": residual_readout_audit_summary,
            "candidate_clamped_causal_replay_audit_summary": (
                candidate_clamped_causal_replay_audit_summary
            ),
            "counterfactual_fingerprint_audit_summary": (
                counterfactual_fingerprint_audit_summary
            ),
            "persistent_candidate_slot_replay_audit_summary": (
                persistent_candidate_slot_replay_audit_summary
            ),
            "local_relational_identity_audit_summary": local_relational_identity_audit_summary,
            "dense_candidate_edge_audit_summary": dense_candidate_edge_audit_summary,
            "dense_transport_consistency_audit_summary": dense_transport_consistency_audit_summary,
            "candidate_field_consistency_audit_summary": candidate_field_consistency_audit_summary,
            "anchor_topology_audit_summary": anchor_topology_audit_summary,
            "anchor_topology_pair_records": anchor_topology_pair_records,
            "multilayer_identity_audit_summary": multilayer_identity_audit_summary,
            "transport_factorization_audit_summary": transport_factorization_audit_summary,
            "trajectory_identity_audit_summary": trajectory_identity_audit_summary,
            "multi_timestep_attention_identity_audit_summary": multi_timestep_attention_identity_audit_summary,
            "records": candidate_dump_records,
        }
        with open(dump_path, "w") as handle:
            json.dump(dump_payload, handle, indent=2)
        result["candidate_dump_path"] = dump_path
        result["candidate_dump_records"] = len(candidate_dump_records)
        result["candidate_dump_summary"] = dump_summary
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as handle:
        json.dump(result, handle, indent=2)
    if fjsar_capture is not None:
        fjsar_capture.close()
    del (
        fjsar_model,
        fjsar_featurizer,
        fjsar_blocks,
        fjsar_persistent_slot_blocks,
    )
    print(f"Matcher: {args.matcher}")
    print(f"Baseline All per image/point: {result['all']['baseline_image']:.2f} / {result['all']['baseline_point']:.2f}")
    print(
        f"Method All per image/point: {result['all']['method_image']:.2f} / "
        f"{result['all']['method_point']:.2f}; point gain={result['all']['point_gain']:.2f}"
    )


def build_parser():
    parser = argparse.ArgumentParser(description="DiTF SPair official parity with matcher ablation")
    parser.add_argument("--dataset_path", type=str, default="/dataset/SPair-71k")
    parser.add_argument("--save_path", type=str, default="/scratch/spair_ft/")
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--dit_model", choices=["flux"], default="flux")
    parser.add_argument("--img_size", nargs="+", type=int, default=[640, 640])
    parser.add_argument("--t", default=260, type=int)
    parser.add_argument("--k", nargs="+", type=int, default=[28])
    parser.add_argument("--ensemble_size", default=8, type=int)
    parser.add_argument("--cd", action="store_true", default=False)
    parser.add_argument("--reuse_saved_features", action="store_true", default=False)
    parser.add_argument("--max_pairs_per_cat", default=0, type=int)
    parser.add_argument("--subset", choices=["all", "discovery", "heldout"], default="all")
    parser.add_argument("--pairs_per_cat", default=20, type=int)
    parser.add_argument("--split_seed", default=2027, type=int)
    parser.add_argument(
        "--build_fjsar_cache",
        action="store_true",
        default=False,
        help="deprecated no-op: FJSAR extracts aligned entries on demand without writing cache files",
    )
    parser.add_argument(
        "--matcher",
        choices=[
            "nn",
            "fjsar_attn",
            "fjsar_attention_signature",
            "fjsar_part_sharpen",
            "fjsar_orthogonal_context",
            "fjsar_spectral_identity",
            "fjsar_filtered_spectral_kernel",
            "fjsar_transport_lift",
            "fjsar_basin_contrastive_identity",
            "fjsar_attention_isometry",
            "fjsar_identity_preserving_attention",
            "fjsar_balanced_transport_attention",
            "fjsar_qk_identity_attention",
            "fjsar_cross_attention_trajectory",
            "fjsar_native_preserving_topology_rescue",
            "fjsar_attention_basin_native_refine",
            "fjsar_candidate_graph_consensus_verification",
            "fjsar_geometry_consistent_attention",
            "fjsar_candidate_conditioned_verification",
            "fjsar_candidate_local_transport_verification",
            "fjsar_attention_relational_graph_matching",
            "fjsar_dense_partial_graph_matching",
            "fjsar_expert_preserving_attention_hypothesis_conditioned_replay",
            "fjsar_pre_softmax_channelwise_identity",
            "fjsar_layer_routed_identity",
            "fjsar_pre_single_stream_identity",
            "fjsar_all",
        ],
        default="nn",
    )
    parser.add_argument(
        "--fjsar_oracle_audit",
        action="store_true",
        default=False,
        help="Collect GT-only candidate coverage diagnostics for FJSAR branches.",
    )
    parser.add_argument(
        "--fjsar_oracle_topk",
        nargs="+",
        type=int,
        default=[1, 5, 10, 20, 50],
        help="Top-K values used for FJSAR oracle coverage diagnostics.",
    )
    parser.add_argument(
        "--fjsar_memory_cache_gb",
        type=float,
        default=8.0,
        help="CPU RAM budget for per-run FJSAR image-entry reuse; set 0 to disable.",
    )
    parser.add_argument(
        "--fjsar_shared_noise",
        action="store_true",
        default=False,
        help="Use a fixed protocol RNG stream for FJSAR replay diagnostics; default off keeps the official extraction flow.",
    )
    parser.add_argument(
        "--fjsar_disk_cache_path",
        type=str,
        default="",
        help="Persistent per-image FJSAR replay-entry cache. Empty disables disk caching.",
    )
    parser.add_argument(
        "--extract_native_in_memory",
        action="store_true",
        default=False,
        help=(
            "Explicitly permit fresh FJSAR extraction with bounded in-run memory reuse "
            "and no persistent replay-cache writes."
        ),
    )
    parser.add_argument(
        "--fjsar_disk_cache_min_free_gb",
        type=float,
        default=2.0,
        help="Stop writing FJSAR disk cache entries when free space would drop below this value.",
    )
    parser.add_argument(
        "--fjsar_require_disk_cache",
        action="store_true",
        default=False,
        help="Fail on FJSAR disk-cache miss instead of extracting and writing a new replay entry.",
    )
    parser.add_argument(
        "--fjsar_candidate_topk",
        type=int,
        default=20,
        help="Attention proposal count used by FJSAR diagnostics.",
    )
    parser.add_argument(
        "--fjsar_trajectory_blocks",
        nargs="+",
        type=int,
        default=[20, 24, 28, 32, 36],
        help="FLUX feature block indices used by fjsar_cross_attention_trajectory.",
    )
    parser.add_argument(
        "--fjsar_trajectory_identity_audit_path",
        type=str,
        default="",
        help="Optional output path for fjsar_cross_attention_trajectory audit; defaults next to --output_json.",
    )
    parser.add_argument(
        "--fjsar_multi_timestep_attention_identity_audit",
        action="store_true",
        default=False,
        help="Audit attention identity across multiple diffusion timesteps for the same pair.",
    )
    parser.add_argument(
        "--fjsar_multi_timestep_values",
        nargs="+",
        type=int,
        default=[],
        help="Diffusion timesteps used by --fjsar_multi_timestep_attention_identity_audit.",
    )
    parser.add_argument(
        "--fjsar_multi_timestep_attention_identity_audit_path",
        type=str,
        default="",
        help="Optional output path for --fjsar_multi_timestep_attention_identity_audit; defaults next to --output_json.",
    )
    parser.add_argument(
        "--fjsar_geometry_radius",
        type=int,
        default=2,
        help="Token-neighborhood radius for fjsar_geometry_consistent_attention.",
    )
    parser.add_argument(
        "--fjsar_geometry_strength",
        type=float,
        default=0.5,
        help="Soft strength in A'=normalize(A*((1-lambda)+lambda*S)) for fjsar_geometry_consistent_attention.",
    )
    parser.add_argument(
        "--fjsar_dump_candidates",
        action="store_true",
        default=False,
        help="Dump lightweight per-keypoint FJSAR attention proposal diagnostics.",
    )
    parser.add_argument(
        "--fjsar_dump_candidates_path",
        type=str,
        default="",
        help="Optional output path for --fjsar_dump_candidates; defaults next to --output_json.",
    )
    parser.add_argument(
        "--fjsar_dump_case_filter",
        choices=[
            "all",
            "oracle_gap",
            "attention_harms_native",
            "attention_rescues_native",
            "oracle_gap_or_harm",
            "attention_informative",
        ],
        default="all",
        help="Keep only selected case types in FJSAR candidate/audit dumps.",
    )
    parser.add_argument(
        "--fjsar_dump_max_records",
        type=int,
        default=0,
        help="Optional cap on saved FJSAR dump records after case filtering; 0 keeps all selected records.",
    )
    parser.add_argument(
        "--fjsar_candidate_descriptor_audit",
        action="store_true",
        default=False,
        help="Audit feature-side candidate identity signals inside attention top-k; does not affect predictions.",
    )
    parser.add_argument(
        "--fjsar_candidate_descriptor_audit_path",
        type=str,
        default="",
        help="Optional output path for --fjsar_candidate_descriptor_audit; defaults next to --output_json.",
    )
    parser.add_argument(
        "--fjsar_method_descriptor_audit",
        action="store_true",
        default=False,
        help="Audit the active FJSAR matcher descriptor inside attention top-k; does not affect predictions.",
    )
    parser.add_argument(
        "--fjsar_method_descriptor_audit_path",
        type=str,
        default="",
        help="Optional output path for --fjsar_method_descriptor_audit; defaults next to --output_json.",
    )
    parser.add_argument(
        "--fjsar_transport_lift_branch_audit",
        action="store_true",
        default=False,
        help="Audit native/out/in/no-native/full transport-lift branches inside attention top-k.",
    )
    parser.add_argument(
        "--fjsar_transport_lift_branch_audit_path",
        type=str,
        default="",
        help="Optional output path for --fjsar_transport_lift_branch_audit; defaults next to --output_json.",
    )
    parser.add_argument(
        "--fjsar_attention_flow_audit",
        action="store_true",
        default=False,
        help="Audit local attention-flow patch metrics inside attention top-k; does not affect predictions.",
    )
    parser.add_argument(
        "--fjsar_attention_flow_radius",
        type=int,
        default=2,
        help="Token-neighborhood radius for --fjsar_attention_flow_audit; radius 2 is a 5x5 patch.",
    )
    parser.add_argument(
        "--fjsar_attention_flow_audit_path",
        type=str,
        default="",
        help="Optional output path for --fjsar_attention_flow_audit; defaults next to --output_json.",
    )
    parser.add_argument(
        "--fjsar_attention_kernel_audit",
        action="store_true",
        default=False,
        help="Audit raw attention vs locally transport-filtered attention @k over the full target token grid.",
    )
    parser.add_argument(
        "--fjsar_attention_kernel_radius",
        type=int,
        default=2,
        help="Token-neighborhood radius for the local transport filter in --fjsar_attention_kernel_audit.",
    )
    parser.add_argument(
        "--fjsar_attention_kernel_topk",
        nargs="+",
        type=int,
        default=[1, 5, 20],
        help="Top-k values reported by --fjsar_attention_kernel_audit.",
    )
    parser.add_argument(
        "--fjsar_attention_kernel_audit_path",
        type=str,
        default="",
        help="Optional output path for --fjsar_attention_kernel_audit; defaults next to --output_json.",
    )
    parser.add_argument(
        "--fjsar_basin_identity_audit",
        action="store_true",
        default=False,
        help="Audit whether native descriptors can recover PCK hits inside raw/filtered attention top-k basins.",
    )
    parser.add_argument(
        "--fjsar_basin_identity_topk",
        type=int,
        default=20,
        help="Attention basin size used by --fjsar_basin_identity_audit.",
    )
    parser.add_argument(
        "--fjsar_basin_identity_radius",
        type=int,
        default=2,
        help="Token-neighborhood radius for the filtered attention basin in --fjsar_basin_identity_audit.",
    )
    parser.add_argument(
        "--fjsar_basin_identity_rank_topk",
        nargs="+",
        type=int,
        default=[1, 3, 5, 10, 20],
        help="Native-within-basin top-k values reported by --fjsar_basin_identity_audit.",
    )
    parser.add_argument(
        "--fjsar_basin_identity_audit_path",
        type=str,
        default="",
        help="Optional output path for --fjsar_basin_identity_audit; defaults next to --output_json.",
    )
    parser.add_argument(
        "--fjsar_operator_manifold_audit",
        action="store_true",
        default=False,
        help="Dump per-keypoint native-vs-operator prepared-token drift for feature-manifold failure diagnosis.",
    )
    parser.add_argument(
        "--fjsar_operator_manifold_audit_path",
        type=str,
        default="",
        help="Optional output path for --fjsar_operator_manifold_audit; defaults next to --output_json.",
    )
    parser.add_argument(
        "--fjsar_kernel_featureization_audit",
        action="store_true",
        default=False,
        help="Audit whether attention kernels can be featureized into cosine+NN descriptor spaces.",
    )
    parser.add_argument(
        "--fjsar_kernel_featureization_ranks",
        nargs="+",
        type=int,
        default=[32, 64],
        help="SVD ranks used by --fjsar_kernel_featureization_audit.",
    )
    parser.add_argument(
        "--fjsar_kernel_featureization_weights",
        nargs="+",
        type=float,
        default=[0.5, 1.0],
        help="Native/kernel branch weights used by --fjsar_kernel_featureization_audit.",
    )
    parser.add_argument(
        "--fjsar_kernel_featureization_radius",
        type=int,
        default=2,
        help="Local transport radius for filtered-kernel featureization audit.",
    )
    parser.add_argument(
        "--fjsar_kernel_featureization_topk",
        nargs="+",
        type=int,
        default=[1, 5, 20],
        help="Top-k values reported by --fjsar_kernel_featureization_audit.",
    )
    parser.add_argument(
        "--fjsar_kernel_featureization_audit_path",
        type=str,
        default="",
        help="Optional output path for --fjsar_kernel_featureization_audit; defaults next to --output_json.",
    )
    parser.add_argument(
        "--fjsar_residual_readout_audit",
        action="store_true",
        default=False,
        help="Audit per-head/common-free value residual readout signals inside attention top-k candidates.",
    )
    parser.add_argument(
        "--fjsar_residual_readout_topk",
        nargs="+",
        type=int,
        default=[1, 3, 5, 10, 20],
        help="Top-k values reported by --fjsar_residual_readout_audit.",
    )
    parser.add_argument(
        "--fjsar_residual_readout_audit_path",
        type=str,
        default="",
        help="Optional output path for --fjsar_residual_readout_audit; defaults next to --output_json.",
    )
    parser.add_argument(
        "--fjsar_latent_expert_audit",
        action="store_true",
        default=False,
        help=(
            "Audit whether stable minority heads/ensemble members recover mutual-attention top-k identity "
            "before ensemble/head averaging; does not affect predictions."
        ),
    )
    parser.add_argument(
        "--fjsar_latent_expert_topk",
        nargs="+",
        type=int,
        default=[1, 3, 5, 10, 20],
        help="Top-k values reported by --fjsar_latent_expert_audit.",
    )
    parser.add_argument(
        "--fjsar_latent_expert_audit_path",
        type=str,
        default="",
        help="Optional detailed output path for --fjsar_latent_expert_audit.",
    )
    parser.add_argument(
        "--fjsar_latent_expert_summary_path",
        type=str,
        default="",
        help="Optional compact mechanism-summary path for --fjsar_latent_expert_audit.",
    )
    parser.add_argument(
        "--fjsar_identity_decodability_audit",
        action="store_true",
        default=False,
        help=(
            "Export candidate-aligned FLUX internal states and run category-held-out "
            "supervised probes; diagnostic-only and never changes matcher predictions."
        ),
    )
    parser.add_argument(
        "--fjsar_identity_decodability_shard_path",
        type=str,
        default="",
        help="Optional directory for binary per-pair internal-state shards.",
    )
    parser.add_argument(
        "--fjsar_identity_decodability_manifest_path",
        type=str,
        default="",
        help="Optional output manifest path for identity decodability shards.",
    )
    parser.add_argument(
        "--fjsar_identity_decodability_summary_path",
        type=str,
        default="",
        help="Optional category-held-out identity decodability summary path.",
    )
    parser.add_argument(
        "--fjsar_identity_decodability_folds",
        type=int,
        default=3,
        help="Number of deterministic category-held-out outer folds.",
    )
    parser.add_argument(
        "--fjsar_identity_decodability_skip_mlp",
        action="store_true",
        default=False,
        help="Run only linear family probes and skip the all-internal shallow MLP.",
    )
    parser.add_argument(
        "--fjsar_candidate_clamped_causal_replay_audit",
        action="store_true",
        default=False,
        help=(
            "Audit mass-preserving bidirectional candidate clamps through the adjacent "
            "unclamped FLUX block; does not affect predictions."
        ),
    )
    parser.add_argument(
        "--fjsar_candidate_clamped_causal_replay_topk",
        nargs="+",
        type=int,
        default=[1, 3, 5, 10, 20],
        help="Top-k values reported by the candidate-clamped causal replay audit.",
    )
    parser.add_argument(
        "--fjsar_candidate_clamped_causal_replay_audit_path",
        type=str,
        default="",
        help="Optional detailed output path for the candidate-clamped causal replay audit.",
    )
    parser.add_argument(
        "--fjsar_candidate_clamped_causal_replay_summary_path",
        type=str,
        default="",
        help="Optional compact mechanism-summary path for the candidate-clamped causal replay audit.",
    )
    parser.add_argument(
        "--fjsar_counterfactual_fingerprint_audit",
        action="store_true",
        default=False,
        help=(
            "Audit multi-dose bidirectional candidate response fingerprints through "
            "the adjacent unmodified FLUX block; does not affect predictions."
        ),
    )
    parser.add_argument(
        "--fjsar_counterfactual_fingerprint_topk",
        nargs="+",
        type=int,
        default=[1, 3, 5, 10, 20],
        help="Top-k values reported by the counterfactual fingerprint audit.",
    )
    parser.add_argument(
        "--fjsar_counterfactual_fingerprint_scales",
        nargs="+",
        type=float,
        default=[0.75, 1.0, 1.25],
        help="Positive candidate-value intervention scales; must include 1.0.",
    )
    parser.add_argument(
        "--fjsar_counterfactual_fingerprint_audit_path",
        type=str,
        default="",
        help="Optional detailed output path for the counterfactual fingerprint audit.",
    )
    parser.add_argument(
        "--fjsar_counterfactual_fingerprint_summary_path",
        type=str,
        default="",
        help="Optional compact mechanism-summary path for the counterfactual fingerprint audit.",
    )
    parser.add_argument(
        "--fjsar_persistent_candidate_slot_replay_audit",
        action="store_true",
        default=False,
        help=(
            "Audit two-block candidate-specific replay slots with original cross mass; "
            "does not affect matcher predictions."
        ),
    )
    parser.add_argument(
        "--fjsar_persistent_candidate_slot_replay_topk",
        nargs="+",
        type=int,
        default=[1, 3, 5, 10, 20],
        help="Top-k values reported by the persistent candidate-slot replay audit.",
    )
    parser.add_argument(
        "--fjsar_persistent_candidate_slot_replay_chunk",
        type=int,
        default=1,
        help="Maximum candidate hypotheses expanded together by persistent candidate-slot replay.",
    )
    parser.add_argument(
        "--fjsar_persistent_candidate_slot_replay_audit_path",
        type=str,
        default="",
        help="Optional detailed output path for the persistent candidate-slot replay audit.",
    )
    parser.add_argument(
        "--fjsar_persistent_candidate_slot_replay_summary_path",
        type=str,
        default="",
        help="Optional compact mechanism-summary path for the persistent candidate-slot replay audit.",
    )
    parser.add_argument(
        "--fjsar_local_relational_identity_audit",
        action="store_true",
        default=False,
        help="Audit local patch, self-similarity, and attention-row differential identity signals inside attention top-k.",
    )
    parser.add_argument(
        "--fjsar_local_relational_radius",
        type=int,
        default=2,
        help="Token-neighborhood radius for --fjsar_local_relational_identity_audit; 1 is 3x3 and 2 is 5x5.",
    )
    parser.add_argument(
        "--fjsar_local_relational_identity_audit_path",
        type=str,
        default="",
        help="Optional output path for --fjsar_local_relational_identity_audit; defaults next to --output_json.",
    )
    parser.add_argument(
        "--fjsar_dense_candidate_edge_audit",
        action="store_true",
        default=False,
        help=(
            "Audit whether one-step local edge messages separate correct mutual-attention "
            "top-k candidates; reserves a sparse partial-assignment graph but does not change predictions."
        ),
    )
    parser.add_argument(
        "--fjsar_dense_candidate_edge_radius",
        type=int,
        default=1,
        help="Dense source-grid edge radius; 1 creates directed 8-neighborhood edges.",
    )
    parser.add_argument(
        "--fjsar_dense_candidate_edge_audit_path",
        type=str,
        default="",
        help="Optional detailed output path for --fjsar_dense_candidate_edge_audit.",
    )
    parser.add_argument(
        "--fjsar_dense_candidate_edge_summary_path",
        type=str,
        default="",
        help="Optional compact mechanism-summary path for --fjsar_dense_candidate_edge_audit.",
    )
    parser.add_argument(
        "--fjsar_dense_transport_consistency_audit",
        action="store_true",
        default=False,
        help="Audit all-source-token transport-field consistency for attention top-k candidates.",
    )
    parser.add_argument(
        "--fjsar_dense_transport_topk",
        nargs="+",
        type=int,
        default=[1, 5, 20],
        help="Source-token attention top-k values used by --fjsar_dense_transport_consistency_audit.",
    )
    parser.add_argument(
        "--fjsar_dense_transport_consistency_audit_path",
        type=str,
        default="",
        help="Optional output path for --fjsar_dense_transport_consistency_audit; defaults next to --output_json.",
    )
    parser.add_argument(
        "--fjsar_candidate_field_consistency_audit",
        action="store_true",
        default=False,
        help="Audit whether attention-basin candidate fields admit a globally consistent native refinement.",
    )
    parser.add_argument(
        "--fjsar_candidate_field_topm",
        type=int,
        default=20,
        help="Top-M full-resolution candidates per keypoint used by --fjsar_candidate_field_consistency_audit.",
    )
    parser.add_argument(
        "--fjsar_candidate_field_source",
        choices=["native_basin", "attention_tokens"],
        default="native_basin",
        help="Candidate source for --fjsar_candidate_field_consistency_audit.",
    )
    parser.add_argument(
        "--fjsar_candidate_field_consistency_audit_path",
        type=str,
        default="",
        help="Optional output path for --fjsar_candidate_field_consistency_audit; defaults next to --output_json.",
    )
    parser.add_argument(
        "--fjsar_anchor_topology_audit",
        action="store_true",
        default=False,
        help="Audit whether native high-confidence anchors can topology-rerank attention proposals.",
    )
    parser.add_argument(
        "--fjsar_anchor_topology_audit_path",
        type=str,
        default="",
        help="Optional output path for --fjsar_anchor_topology_audit; defaults next to --output_json.",
    )
    parser.add_argument(
        "--fjsar_multilayer_identity_audit",
        action="store_true",
        default=False,
        help="Audit official multi-block FLUX descriptors inside attention top-k candidates.",
    )
    parser.add_argument(
        "--fjsar_multilayer_blocks",
        nargs="+",
        type=int,
        default=[24, 28, 32, 36],
        help=(
            "Single-stream FLUX block indices used by the multilayer audit or "
            "fjsar_layer_routed_identity matcher."
        ),
    )
    parser.add_argument(
        "--fjsar_multilayer_identity_audit_path",
        type=str,
        default="",
        help="Optional output path for --fjsar_multilayer_identity_audit; defaults next to --output_json.",
    )
    parser.add_argument(
        "--fjsar_pre_single_stream_blocks",
        nargs="+",
        type=int,
        default=[0, 4, 8, 12, 16, 18],
        help=(
            "FLUX double-stream block indices used as image identity carriers by "
            "fjsar_pre_single_stream_identity."
        ),
    )
    parser.add_argument(
        "--fjsar_transport_factorization_audit",
        action="store_true",
        default=False,
        help="Audit whether attention-flow signals can be represented as source/target descriptor dot products.",
    )
    parser.add_argument(
        "--fjsar_transport_factorization_radius",
        type=int,
        default=2,
        help="Token-neighborhood radius for --fjsar_transport_factorization_audit.",
    )
    parser.add_argument(
        "--fjsar_transport_factorization_basis_radius",
        type=int,
        default=0,
        help="Basis radius for factorized target/source descriptor probes; 0 uses center basis.",
    )
    parser.add_argument(
        "--fjsar_transport_factorization_audit_path",
        type=str,
        default="",
        help="Optional output path for --fjsar_transport_factorization_audit; defaults next to --output_json.",
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_device(0)
    evaluate(args, device)

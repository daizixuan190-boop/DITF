import pytest
import torch
from torch.nn import functional as F

from flux_joint_replay import (
    FluxReplayState,
    _balanced_bidirectional_kernels,
    flux_candidate_clamped_causal_probe,
    flux_candidate_counterfactual_fingerprint_probe,
    flux_candidate_internal_state_probe,
    flux_cross_readout_probe,
    flux_persistent_candidate_slot_replay_probe,
    flux_joint_single_block,
    manual_flux_single_block,
    native_parity_error,
    pair_coordinate_bias,
    run_flux_balanced_transport_stack,
    run_flux_joint_stack,
    run_flux_identity_preserving_stack,
    run_flux_native_stack,
    run_flux_qk_identity_stack,
)
from spair_matchers import (
    AttentionSparsePartialGraph,
    _attention_signature_descriptors,
    _attention_guided_isometry_descriptors,
    _part_common_sharpen_descriptors,
    _orthogonal_context_descriptors,
    _local_transport_lift_descriptors,
    _basin_contrastive_identity_descriptors,
    _spectral_attention_identity_descriptors,
    _candidate_conditioned_verification_rankings,
    _candidate_local_transport_verification_rankings,
    _candidate_graph_consensus_verification_rankings,
    _build_attention_sparse_partial_graph,
    _candidate_clamped_causal_replay_audit_for_points,
    _counterfactual_fingerprint_audit_for_points,
    _persistent_candidate_slot_replay_audit_for_points,
    _dense_candidate_edge_separability_audit_for_points,
    _dense_partial_graph_matching_rankings,
    _expert_preserving_hypothesis_scores,
    _expert_preserving_attention_hypothesis_conditioned_replay_rankings,
    _pre_softmax_channelwise_identity_rankings,
    _solve_attention_sparse_partial_assignment,
    _attention_relational_graph_matching_rankings,
    _solve_attention_relational_graph,
    _cross_attention_trajectory_rankings,
    _fjsar_attention_candidate_records,
    _fjsar_candidate_oracle_counts,
    _latent_expert_audit_for_points,
    _prepare_replay_tokens,
    cosine_nn_predict,
    flux_fjsar_dump_candidates,
    flux_fjsar_candidate_feature_batch,
    flux_fjsar_identity_decodability_batch,
    flux_fjsar_predict,
)
from analyze_identity_decodability import (
    analyze_identity_decodability,
    category_folds,
    rank_metrics,
    _fit_torch_mlp_scores,
    PROBE_FEATURE_GROUPS,
)
from eval_spair_matcher_ablation import (
    build_parser,
    _fjsar_mode_config,
    _validate_identity_decodability_feature_source,
    _summarize_candidate_clamped_causal_replay_audit,
    _summarize_counterfactual_fingerprint_audit,
    _summarize_persistent_candidate_slot_replay_audit,
    _summarize_attention_relational_graph_audits,
    _summarize_dense_candidate_edge_audit,
    _summarize_dense_partial_graph_audits,
    _summarize_expert_hypothesis_audits,
    _summarize_latent_expert_audit,
    _summarize_multi_timestep_attention_identity_audit,
)


def test_identity_decodability_feature_source_is_explicit_and_exclusive():
    parser = build_parser()
    in_memory = parser.parse_args([
        "--output_json",
        "result.json",
        "--fjsar_identity_decodability_audit",
        "--extract_native_in_memory",
    ])
    assert in_memory.extract_native_in_memory is True
    assert _validate_identity_decodability_feature_source(in_memory) == "in_memory"


def test_counterfactual_fingerprint_parser_contract_is_explicit():
    parser = build_parser()
    args = parser.parse_args([
        "--output_json", "result.json",
        "--fjsar_counterfactual_fingerprint_audit",
        "--fjsar_oracle_audit",
        "--fjsar_dump_case_filter", "all",
        "--fjsar_dump_max_records", "0",
        "--fjsar_counterfactual_fingerprint_scales", "0.75", "1.0", "1.25",
    ])
    assert args.fjsar_counterfactual_fingerprint_audit is True
    assert args.fjsar_counterfactual_fingerprint_scales == [0.75, 1.0, 1.25]

    canonical_cache = parser.parse_args([
        "--output_json",
        "result.json",
        "--fjsar_identity_decodability_audit",
        "--fjsar_disk_cache_path",
        "/cache/replay",
        "--fjsar_require_disk_cache",
    ])
    assert (
        _validate_identity_decodability_feature_source(canonical_cache)
        == "canonical_disk_cache"
    )

    both = parser.parse_args([
        "--output_json",
        "result.json",
        "--fjsar_identity_decodability_audit",
        "--extract_native_in_memory",
        "--fjsar_disk_cache_path",
        "/cache/replay",
        "--fjsar_require_disk_cache",
    ])
    with pytest.raises(ValueError, match="exactly one"):
        _validate_identity_decodability_feature_source(both)

    neither = parser.parse_args([
        "--output_json",
        "result.json",
        "--fjsar_identity_decodability_audit",
    ])
    with pytest.raises(ValueError, match="exactly one"):
        _validate_identity_decodability_feature_source(neither)


class _IdentityQKNorm:
    def __call__(self, q, k, v):
        return q, k


class _ToyModulationOut:
    def __init__(self, vec, channels):
        batch = vec.shape[0]
        device = vec.device
        dtype = vec.dtype
        self.shift = torch.zeros(batch, 1, channels, device=device, dtype=dtype)
        self.scale = torch.zeros(batch, 1, channels, device=device, dtype=dtype)
        self.gate = torch.ones(batch, 1, channels, device=device, dtype=dtype)


class _ToyModulation(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

    def forward(self, vec):
        return _ToyModulationOut(vec, self.channels), None


class _ToySingleStreamBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 4
        self.mlp_hidden_dim = 4
        self.num_heads = 2
        self.linear1 = torch.nn.Linear(4, 16, bias=False)
        self.linear2 = torch.nn.Linear(8, 4, bias=False)
        self.norm = _IdentityQKNorm()
        self.pre_norm = torch.nn.Identity()
        self.mlp_act = torch.nn.Identity()
        self.modulation = _ToyModulation(4)
        with torch.no_grad():
            self.linear1.weight.zero_()
            self.linear1.weight[0:4, :] = torch.eye(4)
            self.linear1.weight[4:8, :] = torch.eye(4)
            self.linear1.weight[8:12, :] = torch.eye(4)
            self.linear2.weight.zero_()
            self.linear2.weight[:, :4] = torch.eye(4)

    def forward(self, x, vec, pe):
        return manual_flux_single_block(self, x, vec, pe)


def _toy_replay_state(tokens, height=2, width=2, block_index=28):
    return FluxReplayState(
        x=tokens.unsqueeze(0),
        vec=torch.zeros(1, 4),
        pe=torch.empty(0),
        text_token_count=1,
        image_height=height,
        image_width=width,
        global_block_index=block_index,
    ).to_dict()


def _toy_aligned_feature_and_ada(block, state_dict, depth=1):
    state = FluxReplayState.from_dict(state_dict)
    raw_sequence = run_flux_native_stack([block] * int(depth), state)
    raw_tokens = raw_sequence[:, state.text_token_count:]
    raw_mean = raw_tokens.mean(dim=0)
    raw_feature = raw_mean.t().reshape(1, raw_mean.shape[1], state.image_height, state.image_width)
    ada = torch.zeros(1, 2, raw_mean.shape[1])
    prepared = _prepare_replay_tokens(raw_tokens, ada)[0]
    prepared_map = prepared.t().reshape(1, prepared.shape[1], state.image_height, state.image_width)
    return raw_feature, ada, prepared_map


def test_cosine_nn_predict_matches_exact_argmax():
    src = torch.tensor([[[[1.0]], [[0.0]]]])
    trg = torch.tensor([[[[0.0, 1.0]], [[1.0, 0.0]]]])
    assert cosine_nn_predict(src, trg, [[0.0, 0.0]]) == [[1, 0]]


def test_flux_joint_manual_native_replay_has_exact_parity_on_toy_block():
    block = _ToySingleStreamBlock()
    tokens = torch.eye(4).repeat(2, 1)[:5].unsqueeze(0)
    state = FluxReplayState.from_dict(_toy_replay_state(tokens[0]))
    diagnostics = native_parity_error(block, state)
    assert diagnostics["max_abs_error"] == 0.0
    assert diagnostics["cosine"] > 0.99999


def test_balanced_transport_kernel_competes_for_target_capacity():
    logits_ab = torch.tensor([[[[4.0, 3.0], [4.0, 3.9]]]])
    logits_ba = torch.tensor([[[[4.0, 4.0], [3.0, 3.9]]]])
    kernel_ab, kernel_ba, plan = _balanced_bidirectional_kernels(logits_ab, logits_ba)
    assert torch.allclose(plan.sum(dim=-1), torch.full((1, 1, 2), 0.5), atol=1e-4)
    assert torch.allclose(plan.sum(dim=-2), torch.full((1, 1, 2), 0.5), atol=1e-4)
    assert kernel_ab[0, 0].argmax(dim=1).tolist() == [0, 1]
    assert kernel_ba[0, 0].argmax(dim=1).tolist() == [0, 1]


def test_flux_balanced_transport_stack_runs_on_toy_block():
    block = _ToySingleStreamBlock()
    tokens_a = torch.eye(4).repeat(2, 1)[:5]
    tokens_b = torch.flip(tokens_a, dims=[0])
    state_a = FluxReplayState.from_dict(_toy_replay_state(tokens_a))
    state_b = FluxReplayState.from_dict(_toy_replay_state(tokens_b))
    out_a, out_b, diagnostics = run_flux_balanced_transport_stack([block], state_a, state_b)
    assert out_a.shape == state_a.x.shape
    assert out_b.shape == state_b.x.shape
    assert torch.isfinite(out_a).all()
    assert torch.isfinite(out_b).all()
    assert diagnostics["p_ab"].shape == (4, 4)
    assert diagnostics["balanced_transport_row_error_a"].max() < 1e-3
    assert diagnostics["balanced_transport_col_error_b"].max() < 1e-3


def test_flux_qk_identity_stack_runs_on_toy_block():
    block = _ToySingleStreamBlock()
    tokens_a = torch.eye(4).repeat(2, 1)[:5]
    tokens_b = torch.flip(tokens_a, dims=[0])
    state_a = FluxReplayState.from_dict(_toy_replay_state(tokens_a))
    state_b = FluxReplayState.from_dict(_toy_replay_state(tokens_b))
    out_a, out_b, diagnostics = run_flux_qk_identity_stack([block], state_a, state_b)
    assert out_a.shape == state_a.x.shape
    assert out_b.shape == state_b.x.shape
    assert torch.isfinite(out_a).all()
    assert torch.isfinite(out_b).all()
    assert diagnostics["p_ab"].shape == (4, 4)
    assert diagnostics["qk_fisher_ratio_a"].shape == (4,)


def test_balanced_transport_attention_oracle_mode_uses_joint_tokens():
    features = torch.eye(4).t().reshape(1, 4, 2, 2)
    state = FluxReplayState.from_dict(
        _toy_replay_state(torch.cat((torch.zeros(1, 4), torch.eye(4)), dim=0), height=2, width=2)
    )
    counts = _fjsar_candidate_oracle_counts(
        src_features=features,
        trg_features=features.clone(),
        points=[[0.0, 0.0]],
        target_points=[[0.0, 0.0]],
        source_size=[2, 2],
        target_size=[2, 2],
        pck_threshold=10.0,
        oracle_topk=(1,),
        src_state=state,
        trg_state=state,
        src_native_prepared=torch.eye(4),
        trg_native_prepared=torch.eye(4),
        src_joint_prepared=torch.eye(4),
        trg_joint_prepared=torch.eye(4),
        attention={"p_ab": torch.eye(4), "p_ba": torch.eye(4)},
        descriptor_modes=("balanced_transport_attention",),
    )
    assert counts["fjsar_oracle_owner_balanced_transport_attention@1"] == 1


def test_qk_identity_attention_oracle_mode_uses_joint_tokens():
    features = torch.eye(4).t().reshape(1, 4, 2, 2)
    state = FluxReplayState.from_dict(
        _toy_replay_state(torch.cat((torch.zeros(1, 4), torch.eye(4)), dim=0), height=2, width=2)
    )
    counts = _fjsar_candidate_oracle_counts(
        src_features=features,
        trg_features=features.clone(),
        points=[[0.0, 0.0]],
        target_points=[[0.0, 0.0]],
        source_size=[2, 2],
        target_size=[2, 2],
        pck_threshold=10.0,
        oracle_topk=(1,),
        src_state=state,
        trg_state=state,
        src_native_prepared=torch.eye(4),
        trg_native_prepared=torch.eye(4),
        src_joint_prepared=torch.eye(4),
        trg_joint_prepared=torch.eye(4),
        attention={"p_ab": torch.eye(4), "p_ba": torch.eye(4)},
        descriptor_modes=("qk_identity_attention",),
    )
    assert counts["fjsar_oracle_owner_qk_identity_attention@1"] == 1


def test_native_preserving_topology_rescue_oracle_mode_is_ranked_pixels():
    features = torch.eye(4).t().reshape(1, 4, 2, 2)
    state = FluxReplayState.from_dict(
        _toy_replay_state(torch.cat((torch.zeros(1, 4), torch.eye(4)), dim=0), height=2, width=2)
    )
    counts = _fjsar_candidate_oracle_counts(
        src_features=features,
        trg_features=features.clone(),
        points=[[0.0, 0.0], [1.0, 1.0]],
        target_points=[[0.0, 0.0], [1.0, 1.0]],
        source_size=[2, 2],
        target_size=[2, 2],
        pck_threshold=10.0,
        oracle_topk=(1, 2),
        src_state=state,
        trg_state=state,
        src_native_prepared=torch.eye(4),
        trg_native_prepared=torch.eye(4),
        src_joint_prepared=torch.eye(4),
        trg_joint_prepared=torch.eye(4),
        attention={"p_ab": torch.eye(4), "p_ba": torch.eye(4)},
        descriptor_modes=("native_preserving_topology_rescue",),
    )
    assert counts["fjsar_oracle_owner_native_preserving_topology_rescue@1"] == 2


def test_attention_basin_native_refine_oracle_mode_refines_inside_attention_cells():
    features = torch.eye(4).t().reshape(1, 4, 2, 2)
    state = FluxReplayState.from_dict(
        _toy_replay_state(torch.cat((torch.zeros(1, 4), torch.eye(4)), dim=0), height=2, width=2)
    )
    counts = _fjsar_candidate_oracle_counts(
        src_features=features,
        trg_features=features.clone(),
        points=[[0.0, 0.0], [1.0, 1.0]],
        target_points=[[0.0, 0.0], [1.0, 1.0]],
        source_size=[2, 2],
        target_size=[2, 2],
        pck_threshold=10.0,
        oracle_topk=(1, 2),
        src_state=state,
        trg_state=state,
        src_native_prepared=torch.eye(4),
        trg_native_prepared=torch.eye(4),
        src_joint_prepared=torch.eye(4),
        trg_joint_prepared=torch.eye(4),
        attention={"p_ab": torch.eye(4), "p_ba": torch.eye(4)},
        descriptor_modes=("attention_basin_native_refine",),
    )
    assert counts["fjsar_oracle_owner_attention_basin_native_refine@1"] == 2


def test_candidate_field_consistency_audit_emits_ranked_scores():
    block = _ToySingleStreamBlock()
    features = torch.eye(4).t().reshape(1, 4, 2, 2)
    full_tokens = torch.cat((torch.zeros(1, 4), torch.eye(4)), dim=0)
    state = _toy_replay_state(full_tokens)
    rows = flux_fjsar_dump_candidates(
        features,
        features.clone(),
        [[0.0, 0.0], [1.0, 1.0]],
        [2, 2],
        [2, 2],
        src_replay_state=state,
        trg_replay_state=state,
        blocks=[block],
        interaction_mode="exact",
        use_coordinate_bias=False,
        target_points=[[0.0, 0.0], [1.0, 1.0]],
        pck_threshold=10.0,
        candidate_topk=2,
        candidate_field_consistency_audit=True,
        candidate_field_topm=2,
        candidate_field_source="attention_tokens",
    )
    audit = rows[0]["candidate_field_consistency_audit"]
    assert audit["candidate_topk"] == 2
    assert audit["field_topm"] == 2
    assert "candidate_field_consistency" in audit["score_names"]
    assert "candidate_field_consistency" in audit["ranks"]
    assert audit["candidates"][0]["pck_hit"] is True


def test_cross_attention_trajectory_reranks_stable_candidate():
    src_state = FluxReplayState.from_dict(
        _toy_replay_state(torch.cat((torch.zeros(1, 3), torch.ones(1, 3)), dim=0), height=1, width=1, block_index=27)
    )
    trg_state = FluxReplayState.from_dict(
        _toy_replay_state(torch.cat((torch.zeros(1, 3), torch.eye(3)), dim=0), height=1, width=3, block_index=27)
    )

    def layer(scores):
        p_ab = torch.tensor([scores], dtype=torch.float32)
        p_ba = torch.ones(3, 1, dtype=torch.float32)
        return {"src_state": src_state, "trg_state": trg_state, "attention": {"p_ab": p_ab, "p_ba": p_ba}}

    layers = {
        "20": layer([0.10, 0.90, 0.05]),
        "24": layer([0.20, 0.80, 0.05]),
        "28": layer([0.95, 0.90, 0.05]),
    }
    ranked, audits, summary = _cross_attention_trajectory_rankings(
        layers,
        28,
        [[0.0, 0.0]],
        [1, 1],
        [1, 3],
        candidate_topk=2,
        target_points=[[1.0, 0.0]],
        pck_threshold=0.5,
    )
    assert ranked.shape == (1, 2)
    assert int(ranked[0, 0]) == 1
    assert audits[0]["ranks"]["trajectory"] == 1
    assert summary["trajectory_layer_count"] == 3.0


def test_flux_fjsar_predict_cross_attention_trajectory_smoke():
    block = _ToySingleStreamBlock()
    image_tokens = torch.eye(4)
    full_tokens = torch.cat((torch.zeros(1, 4), image_tokens), dim=0)
    main_state = _toy_replay_state(full_tokens, height=2, width=2, block_index=27)
    raw_feature, ada, prepared = _toy_aligned_feature_and_ada(block, main_state)
    trajectory_states = {
        "src": {
            "20": _toy_replay_state(full_tokens, height=2, width=2, block_index=19),
            "24": _toy_replay_state(full_tokens, height=2, width=2, block_index=23),
            "28": main_state,
        },
        "trg": {
            "20": _toy_replay_state(full_tokens, height=2, width=2, block_index=19),
            "24": _toy_replay_state(full_tokens, height=2, width=2, block_index=23),
            "28": main_state,
        },
    }
    predictions, diagnostics = flux_fjsar_predict(
        prepared,
        prepared.clone(),
        [[0.0, 0.0], [1.0, 1.0]],
        [2, 2],
        [2, 2],
        src_replay_state=main_state,
        trg_replay_state=main_state,
        src_raw_feature=raw_feature,
        trg_raw_feature=raw_feature.clone(),
        src_ada=ada,
        trg_ada=ada.clone(),
        blocks=[block],
        mode="cross_attention_trajectory",
        interaction_mode="trajectory",
        target_points=[[0.0, 0.0], [1.0, 1.0]],
        pck_threshold=10.0,
        oracle_topk=(1, 2),
        candidate_topk=2,
        trajectory_replay_states=trajectory_states,
        trajectory_block_modules={"20": block, "24": block, "28": block},
        trajectory_blocks=(20, 24, 28),
        return_diagnostics=True,
    )
    assert len(predictions) == 2
    assert diagnostics["mean_trajectory_layer_count"] == 3.0
    assert "fjsar_oracle_owner_cross_attention_trajectory@1" in diagnostics["model_counts"]


def test_flux_fjsar_dump_candidates_writes_anchor_topology_audit():
    block = _ToySingleStreamBlock()
    image_tokens = torch.eye(4)
    full_tokens = torch.cat((torch.zeros(1, 4), image_tokens), dim=0)
    state = _toy_replay_state(full_tokens, height=2, width=2)
    _raw_feature, _ada, prepared = _toy_aligned_feature_and_ada(block, state)
    rows = flux_fjsar_dump_candidates(
        prepared,
        prepared.clone(),
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        [2, 2],
        [2, 2],
        src_replay_state=state,
        trg_replay_state=state,
        blocks=[block],
        interaction_mode="exact",
        target_points=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        pck_threshold=10.0,
        candidate_topk=2,
        anchor_topology_audit=True,
    )
    assert len(rows) == 4
    audit = rows[0]["anchor_topology_audit"]
    assert audit["positive_anchor_count"] >= 0
    assert "hybrid_anchor_topology" in audit["score_names"]
    assert "hybrid_anchor_topology" in audit["ranks"]
    assert len(audit["candidates"]) == 2


def test_multi_timestep_attention_identity_audit_summary_aggregates_timesteps():
    records = [
        {
            "oracle_gap_case": True,
            "attention_harms_native_case": False,
            "attention_rescues_native_case": False,
            "multi_timestep_attention_identity_audit": {
                "timestep_count": 3,
                "summary": {
                    "attention_gt_rank_mean": 2.0,
                    "attention_gt_rank_best": 1,
                    "attention_gt_rank_worst": 4,
                    "attention_top1_stability": 1.0,
                    "attention_top1_unique_count": 1,
                    "attention_topk_persistence": 1.0,
                    "baseline_hit_count": 0,
                    "attention_top1_hit_count": 1,
                    "attention_topk_hit_count": 3,
                    "oracle_gap_timesteps": 2,
                    "attention_harms_native_timesteps": 0,
                    "attention_rescues_native_timesteps": 1,
                },
            },
        },
        {
            "oracle_gap_case": False,
            "attention_harms_native_case": True,
            "attention_rescues_native_case": False,
            "multi_timestep_attention_identity_audit": {
                "timestep_count": 3,
                "summary": {
                    "attention_gt_rank_mean": 4.0,
                    "attention_gt_rank_best": 2,
                    "attention_gt_rank_worst": 6,
                    "attention_top1_stability": 0.5,
                    "attention_top1_unique_count": 2,
                    "attention_topk_persistence": 2 / 3,
                    "baseline_hit_count": 2,
                    "attention_top1_hit_count": 2,
                    "attention_topk_hit_count": 2,
                    "oracle_gap_timesteps": 1,
                    "attention_harms_native_timesteps": 2,
                    "attention_rescues_native_timesteps": 0,
                },
            },
        },
    ]
    summary = _summarize_multi_timestep_attention_identity_audit(records)
    assert summary["all"]["points"] == 2
    assert summary["oracle_gap"]["points"] == 1
    assert summary["attention_harms_native"]["points"] == 1
    assert summary["all"]["signals"]["attention_gt_rank_mean"]["mean"] == 3.0
    assert summary["all"]["signals"]["attention_top1_stability"]["mean"] == 0.75
    assert summary["all"]["signals"]["attention_topk_persistence"]["mean"] == (1.0 + 2 / 3) / 2


def test_candidate_conditioned_verification_can_reject_attention_top1():
    features = torch.eye(9).t().reshape(1, 9, 3, 3)
    tokens = torch.eye(9)
    state = FluxReplayState.from_dict(
        _toy_replay_state(torch.cat((torch.zeros(1, 9), tokens), dim=0), height=3, width=3)
    )
    attention = torch.eye(9) * 0.8
    attention[4, 0] = 1.0
    attention[0, 4] = 1.0
    ranked, audits, summary = _candidate_conditioned_verification_rankings(
        features,
        features.clone(),
        {
            "p_ab": attention,
            "p_ba": attention.t(),
        },
        [[1.0, 1.0]],
        [3, 3],
        [3, 3],
        state,
        state,
        candidate_topk=3,
        target_points=[[1.0, 1.0]],
        pck_threshold=0.1,
    )
    assert ranked.shape[0] == 1
    assert int(ranked[0, 0]) == 4
    assert audits[0]["candidates"][0]["pck_hit"] is True
    assert summary["candidate_pool_mean"] >= 3.0


def test_candidate_local_transport_verification_prefers_self_consistent_native_candidate():
    features = torch.eye(9).t().reshape(1, 9, 3, 3)
    tokens = torch.eye(9)
    state = FluxReplayState.from_dict(
        _toy_replay_state(torch.cat((torch.zeros(1, 9), tokens), dim=0), height=3, width=3)
    )
    attention = torch.eye(9) * 0.7
    attention[4, 0] = 1.0
    attention[0, 4] = 1.0
    ranked, audits, summary = _candidate_local_transport_verification_rankings(
        features,
        features.clone(),
        {
            "p_ab": attention,
            "p_ba": attention.t(),
        },
        [[1.0, 1.0]],
        [3, 3],
        [3, 3],
        state,
        state,
        candidate_topk=3,
        target_points=[[1.0, 1.0]],
        pck_threshold=0.1,
    )
    assert ranked.shape[0] == 1
    assert int(ranked[0, 0]) == 4
    assert audits[0]["rescue_applied"] in (True, False)
    assert "candidate_local_transport_verification" in audits[0]["score_names"]
    assert summary["candidate_pool_mean"] >= 3.0


def test_candidate_graph_consensus_verification_runs_on_toy_pair():
    features = torch.eye(4).t().reshape(1, 4, 2, 2)
    tokens = torch.eye(4)
    state = FluxReplayState.from_dict(
        _toy_replay_state(torch.cat((torch.zeros(1, 4), tokens), dim=0), height=2, width=2)
    )
    ranked, audits, summary = _candidate_graph_consensus_verification_rankings(
        features,
        features.clone(),
        {
            "p_ab": torch.eye(4),
            "p_ba": torch.eye(4),
        },
        [[0.0, 0.0], [1.0, 1.0]],
        [2, 2],
        [2, 2],
        state,
        state,
        candidate_topk=2,
        target_points=[[0.0, 0.0], [1.0, 1.0]],
        pck_threshold=1.0,
    )
    assert ranked.shape[0] == 2
    assert "candidate_graph_consensus" in audits[0]["score_names"]
    assert "graph_consensus" in audits[0]["score_names"]
    assert summary["candidate_pool_mean"] >= 1.0


def test_attention_relational_graph_corrects_wrong_unary_with_feature_relations():
    source_descriptors = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.8, 0.6, 0.0],
                [0.0, 0.6, 0.8],
            ]
        ),
        dim=1,
    )
    target_descriptors = torch.cat(
        (
            source_descriptors,
            F.normalize(torch.tensor([[1.0, 1.0, 1.0]]).repeat(3, 1), dim=1),
        ),
        dim=0,
    )
    candidate_cells = torch.tensor([[0, 3], [1, 4], [2, 5]])
    unary = torch.tensor([[0.0, 0.15], [0.0, 0.15], [0.0, 0.15]])

    solution = _solve_attention_relational_graph(
        unary,
        candidate_cells,
        source_descriptors,
        target_descriptors,
    )

    assert torch.equal(torch.argmax(unary, dim=1), torch.ones(3, dtype=torch.long))
    assert torch.equal(solution["assignment"], torch.zeros(3, dtype=torch.long))
    assert float(solution["selected_energy"]) > float(solution["unary_start_energy"])
    assert int(solution["edge_count"]) == 3


def test_attention_relational_graph_uses_only_attention_candidates_and_audits_solver():
    source_tokens = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.8, 0.6, 0.0, 0.0],
                [0.0, 0.6, 0.8, 0.0],
                [0.0, 0.0, 0.8, 0.6],
            ]
        ),
        dim=1,
    )
    target_tokens = source_tokens.clone()
    state = FluxReplayState.from_dict(
        _toy_replay_state(torch.cat((torch.zeros(1, 4), source_tokens), dim=0), height=2, width=2)
    )
    p_ab = torch.tensor(
        [
            [0.9, 0.8, 0.1, 0.0],
            [0.1, 0.9, 0.8, 0.0],
            [0.0, 0.1, 0.9, 0.8],
            [0.8, 0.0, 0.1, 0.9],
        ]
    )

    ranked, audits, summary = _attention_relational_graph_matching_rankings(
        source_tokens,
        target_tokens,
        {"p_ab": p_ab, "p_ba": p_ab.t()},
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        [2, 2],
        [2, 2],
        state,
        state,
        candidate_topk=2,
        target_points=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        pck_threshold=0.1,
    )

    assert ranked.shape == (3, 2)
    assert summary["native_injected_candidate_count"] == 0
    assert summary["native_fallback_count"] == 0
    assert summary["edge_count"] == 3
    assert "unary_start_energy" in summary
    assert "selected_energy" in summary
    assert "energy_gain" in summary
    for audit in audits:
        assert audit["selected_attention_rank"] in (1, 2)
        assert audit["native_candidate_injected"] is False
        assert "graph_belief" in audit["candidates"][0]["scores"]
        assert "graph_conditional" in audit["candidates"][0]["scores"]
        assert "pairwise_relation_contribution" in audit["candidates"][0]["scores"]
        assert all(candidate["attention_rank"] in (1, 2) for candidate in audit["candidates"])


def test_attention_relational_graph_summary_reports_rescue_harm_and_category_groups():
    point = {
        "candidate_count": 20,
        "selected_attention_rank": 2,
        "selected_state": 1,
        "unary_state": 0,
        "baseline_pck_hit": False,
        "method_pck_hit": True,
        "topk_hits": {
            name: {"@1": name == "relational_graph", "@3": True, "@5": True, "@10": True, "@20": True}
            for name in ("attention", "descriptor_unary", "fused_unary", "relational_graph")
        },
    }
    summary = _summarize_attention_relational_graph_audits([
        {
            "category": "car",
            "summary": {
                "energy_gain": 1.0,
                "edge_count": 3,
                "native_injected_candidate_count": 0,
                "native_fallback_count": 0,
            },
            "points": [point],
        },
        {
            "category": "dog",
            "summary": {
                "energy_gain": 0.5,
                "edge_count": 3,
                "native_injected_candidate_count": 0,
                "native_fallback_count": 0,
            },
            "points": [{**point, "baseline_pck_hit": True, "method_pck_hit": False}],
        },
    ])
    assert summary["all"]["rescued"] == 1
    assert summary["all"]["harmed"] == 1
    assert summary["all"]["improvement_harm_ratio"] == 1.0
    assert summary["groups"]["rigid"]["point_count"] == 1
    assert summary["groups"]["articulated_or_animal"]["point_count"] == 1
    assert summary["all"]["native_injected_candidate_count"] == 0
    assert summary["all"]["native_fallback_count"] == 0


def test_flux_joint_attention_retains_local_mass_instead_of_forcing_cross_only():
    block = _ToySingleStreamBlock()
    a = torch.tensor(
        [[
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]]
    )
    b = torch.roll(a, shifts=1, dims=-1)
    vec = torch.zeros(1, 4)
    pe = torch.empty(0)
    _, _, diagnostics = flux_joint_single_block(block, a, vec, pe, 1, b, vec, pe, 1)
    cross_mass = diagnostics["p_ab"].sum(dim=1)
    assert torch.all(cross_mass > 0.0)
    assert torch.all(cross_mass < 1.0)


def test_pair_coordinate_bias_is_finite_and_symmetric():
    identity = torch.eye(4)
    attention = {
        "p_ab_coord": identity,
        "p_ba_coord": identity,
        "p_ab_validate": identity,
        "p_ba_validate": identity,
    }
    bias_ab, bias_ba, diagnostics = pair_coordinate_bias(attention, 2, 2, 2, 2)
    assert torch.isfinite(bias_ab).all()
    assert torch.allclose(bias_ab, bias_ba.t(), atol=1e-6)
    assert torch.all(diagnostics["coordinate_reliability_a"] > 0)


def test_exact_joint_runs_two_complete_blocks_and_retains_local_mass():
    block = _ToySingleStreamBlock()
    image_tokens = torch.eye(4)
    full = torch.cat((torch.zeros(1, 4), image_tokens), dim=0)
    state = FluxReplayState.from_dict(_toy_replay_state(full))
    joint_a, joint_b, diagnostics = run_flux_joint_stack([block, block], state, state, mode="exact")
    assert joint_a.shape == state.x.shape
    assert joint_b.shape == state.x.shape
    cross_mass = diagnostics["first_p_ab"].sum(dim=1)
    assert torch.all(cross_mass > 0)
    assert torch.all(cross_mass < 1)


def test_identity_preserving_stack_preserves_identity_pair_on_toy_block():
    block = _ToySingleStreamBlock()
    image_tokens = torch.eye(4)
    full = torch.cat((torch.zeros(1, 4), image_tokens), dim=0)
    state = FluxReplayState.from_dict(_toy_replay_state(full))
    joint_a, joint_b, diagnostics = run_flux_identity_preserving_stack([block], state, state)
    assert joint_a.shape == state.x.shape
    assert joint_b.shape == state.x.shape
    assert diagnostics["p_ab"].shape == (4, 4)
    assert torch.isfinite(joint_a).all()
    assert torch.isfinite(joint_b).all()
    assert torch.isfinite(diagnostics["identity_residual_ratio_a"]).all()


def test_flux_cross_readout_probe_scores_selected_candidates():
    block = _ToySingleStreamBlock()
    image_tokens = torch.eye(4)
    full = torch.cat((torch.zeros(1, 4), image_tokens), dim=0)
    state = FluxReplayState.from_dict(_toy_replay_state(full))
    probe = flux_cross_readout_probe(
        [block],
        state,
        state,
        torch.tensor([0, 3]),
        torch.tensor([[0, 1], [3, 2]]),
        mode="exact",
    )
    assert "value_residual_alignment" in probe["scores"]
    assert probe["scores"]["value_residual_alignment"].shape == (2, 2)
    for value in probe["scores"].values():
        assert torch.isfinite(value).all()
    assert probe["expert_scores"]["bidirectional_negative_log_rank"].shape[-2:] == (2, 2)
    assert probe["expert_scores"]["mutual_probability"].shape[-2:] == (2, 2)
    assert probe["expert_scores"]["log_exact_mutual_cross_probability"].shape[-2:] == (2, 2)
    assert probe["expert_scores"]["symmetric_value_residual_alignment"].shape[-2:] == (2, 2)
    assert probe["expert_scores"]["symmetric_value_residual_energy"].shape[-2:] == (2, 2)
    assert torch.all(
        probe["expert_scores"]["exact_mutual_cross_probability"]
        <= probe["expert_scores"]["mutual_probability"] + 1e-7
    )
    for value in probe["expert_scores"].values():
        assert torch.isfinite(value).all()


def test_candidate_clamped_causal_probe_preserves_mass_and_releases_qk():
    clamp_block = _ToySingleStreamBlock()
    release_block = _ToySingleStreamBlock()
    image_tokens = torch.eye(4)
    full = torch.cat((torch.zeros(1, 4), image_tokens), dim=0)
    state = FluxReplayState.from_dict(_toy_replay_state(full, block_index=27))
    proposal_pixels = torch.tensor([[0, 1], [3, 2]])
    probe = flux_candidate_clamped_causal_probe(
        clamp_block,
        release_block,
        state,
        state,
        torch.tensor([0, 3]),
        proposal_pixels,
    )
    assert probe["metadata"]["clamp_global_block_index"] == 27
    assert probe["metadata"]["release_global_block_index"] == 28
    assert probe["metadata"]["native_candidate_injected"] is False
    assert probe["metadata"]["gt_used_for_scoring"] is False
    for value in probe["scores"].values():
        assert value.shape == (2, 2)
        assert torch.isfinite(value).all()
    for name in ("source_cross_mass", "target_cross_mass"):
        value = probe["diagnostics"][name]
        assert torch.all(value > 0)
        assert torch.all(value < 1)

    rows = _candidate_clamped_causal_replay_audit_for_points(
        probe,
        proposal_pixels,
        [2, 2],
        target_points=[[0.0, 0.0], [1.0, 1.0]],
        pck_threshold=10.0,
        topks=(1, 2),
    )
    assert len(rows) == 2
    assert rows[0]["causal_contract"]["original_total_cross_mass_preserved"] is True
    assert rows[0]["causal_contract"]["release_qk_is_unclamped"] is True
    assert rows[0]["diagnostics"]["gt_used_for_scoring"] is False
    assert rows[0]["diagnostics"]["native_fallback_used"] is False


def test_counterfactual_fingerprint_requires_reference_and_is_bidirectional():
    clamp_block = _ToySingleStreamBlock()
    release_block = _ToySingleStreamBlock()
    image_tokens = torch.eye(4)
    full = torch.cat((torch.zeros(1, 4), image_tokens), dim=0)
    state = FluxReplayState.from_dict(_toy_replay_state(full, block_index=27))
    probe = flux_candidate_counterfactual_fingerprint_probe(
        clamp_block,
        release_block,
        state,
        state,
        torch.tensor([0, 3]),
        torch.tensor([[0, 1], [3, 2]]),
        intervention_scales=(0.75, 1.0, 1.25),
    )
    assert probe["intervention_scales"] == (0.75, 1.0, 1.25)
    assert probe["fingerprint_score"].shape == (2, 2)
    assert probe["fingerprint_reciprocity_error"].shape == (2, 2)
    assert probe["fingerprint_bidirectional_score_by_scale"].shape == (3, 2, 2)
    for name in (
        "fingerprint_score",
        "fingerprint_mean_bidirectional",
        "fingerprint_reciprocity_error",
        "fingerprint_response_magnitude",
    ):
        assert torch.isfinite(probe[name]).all()
    with pytest.raises(ValueError, match="include the unmodified scale"):
        flux_candidate_counterfactual_fingerprint_probe(
            clamp_block,
            release_block,
            state,
            state,
            torch.tensor([0, 3]),
            torch.tensor([[0, 1], [3, 2]]),
            intervention_scales=(0.75, 1.0),
        )


def test_counterfactual_fingerprint_formatter_preserves_signal_names_and_ranks():
    probe = {
        "fingerprint_score": torch.tensor([[0.8, 0.2]]),
        "fingerprint_mean_bidirectional": torch.tensor([[0.7, 0.3]]),
        "fingerprint_reciprocity_error": torch.tensor([[0.1, 0.4]]),
        "fingerprint_response_magnitude": torch.tensor([[0.2, 0.5]]),
        "fingerprint_source_score_by_scale": torch.tensor(
            [[[0.1, 0.2]], [[0.2, 0.3]], [[0.3, 0.4]]]
        ),
        "fingerprint_target_score_by_scale": torch.tensor(
            [[[0.1, 0.2]], [[0.2, 0.3]], [[0.3, 0.4]]]
        ),
        "intervention_scales": (0.75, 1.0, 1.25),
        "metadata": {"gt_used_for_scoring": False},
    }
    rows = _counterfactual_fingerprint_audit_for_points(
        probe,
        torch.tensor([[0, 1]]),
        [1, 2],
        target_points=[[0.0, 0.0]],
        pck_threshold=1.0,
        topks=(1, 2),
    )
    assert rows[0]["score_names"] == [
        "fingerprint_score",
        "fingerprint_mean_bidirectional",
        "fingerprint_reciprocity_error",
        "fingerprint_response_magnitude",
    ]
    assert rows[0]["ranks"]["fingerprint_score"] == 1
    assert rows[0]["candidates"][0]["response_curve"]["scales"] == [0.75, 1.0, 1.25]


def test_candidate_clamped_causal_dump_uses_adjacent_block_without_changing_matcher():
    clamp_block = _ToySingleStreamBlock()
    release_block = _ToySingleStreamBlock()
    image_tokens = torch.eye(4)
    full_tokens = torch.cat((torch.zeros(1, 4), image_tokens), dim=0)
    state = _toy_replay_state(full_tokens, block_index=27)
    _raw_feature, _ada, prepared = _toy_aligned_feature_and_ada(clamp_block, state)
    rows = flux_fjsar_dump_candidates(
        prepared,
        prepared.clone(),
        [[0.0, 0.0], [1.0, 1.0]],
        [2, 2],
        [2, 2],
        src_replay_state=state,
        trg_replay_state=state,
        blocks=[clamp_block],
        interaction_mode="exact",
        use_coordinate_bias=False,
        target_points=[[0.0, 0.0], [1.0, 1.0]],
        pck_threshold=0.1,
        candidate_topk=2,
        candidate_clamped_causal_replay_audit=True,
        candidate_clamped_causal_replay_topk=(1, 2),
        causal_release_block=release_block,
    )
    assert len(rows) == 2
    audit = rows[0]["candidate_clamped_causal_replay_audit"]
    assert audit["causal_contract"]["prediction_changed"] is False
    assert audit["causal_contract"]["clamp_global_block_index"] == 27
    assert audit["causal_contract"]["release_global_block_index"] == 28
    assert len(audit["candidates"]) == 2


def test_candidate_clamped_causal_summary_applies_oracle_gap_decision_rule():
    records = []
    for pair_index in range(2):
        for point_index in range(2):
            records.append({
                "category": "car",
                "pair_json": f"pair-{pair_index}",
                "src_image": "source.jpg",
                "trg_image": "target.jpg",
                "oracle_gap_case": True,
                "attention_harms_native_case": False,
                "attention_rescues_native_case": False,
                "attention_top1_pck_hit": False,
                "attention_topk_pck_hit": True,
                "candidate_clamped_causal_replay_audit": {
                    "score_names": ["post_release_bidirectional_negative_log_rank"],
                    "ranks": {"post_release_bidirectional_negative_log_rank": 1},
                    "score_gaps": {},
                    "diagnostics": {
                        "pck_hit_candidate_fraction": 0.2,
                        "selected_attention_ranks": {
                            "post_release_bidirectional_negative_log_rank": 2
                        },
                    },
                    "causal_contract": {
                        "native_candidate_injected": False,
                        "native_fallback_used": False,
                        "gt_used_for_scoring": False,
                    },
                },
            })
    summary = _summarize_candidate_clamped_causal_replay_audit(records)
    group = summary["both_wrong_top20_hit"]
    checks = group["mechanism_checks"]
    primary = checks["post_release_bidirectional_negative_log_rank"]
    assert group["points"] == 4
    assert checks["candidate_pck_hit_fraction_mean"] == 0.2
    assert primary["recovers_attention_top1_errors"] == 4
    assert primary["lift_over_uniform_candidate_expectation"] == 0.8
    assert primary["pair_count"] == 2
    assert primary["enter_matcher"] is True
    assert checks["native_fallback_used_count"] == 0


def test_candidate_clamped_causal_parser_flags_are_exposed():
    args = build_parser().parse_args([
        "--output_json",
        "result.json",
        "--fjsar_candidate_clamped_causal_replay_audit",
        "--fjsar_candidate_clamped_causal_replay_topk",
        "1",
        "5",
        "20",
    ])
    assert args.fjsar_candidate_clamped_causal_replay_audit is True
    assert args.fjsar_candidate_clamped_causal_replay_topk == [1, 5, 20]


def test_persistent_candidate_slot_replay_is_chunk_and_candidate_order_invariant():
    block = _ToySingleStreamBlock()
    image_tokens = torch.eye(4)
    full = torch.cat((torch.zeros(1, 4), image_tokens), dim=0)
    state = FluxReplayState.from_dict(_toy_replay_state(full, block_index=27))
    source_cells = torch.tensor([0, 3])
    candidates = torch.tensor([[0, 1, 2], [3, 2, 1]])
    chunk1 = flux_persistent_candidate_slot_replay_probe(
        [block, block],
        state,
        state,
        source_cells,
        candidates,
        hypothesis_chunk=1,
    )
    chunk3 = flux_persistent_candidate_slot_replay_probe(
        [block, block],
        state,
        state,
        source_cells,
        candidates,
        hypothesis_chunk=3,
    )
    for name in (
        "pair_cosine",
        "directional_anchor_cosine",
        "intervention_gain",
        "native_pair_cosine",
        "source_cross_mass",
        "target_cross_mass",
        "source_relative_delta",
        "target_relative_delta",
    ):
        assert torch.allclose(chunk1[name], chunk3[name], atol=1e-6)
    permutation = torch.tensor([2, 0, 1])
    permuted = flux_persistent_candidate_slot_replay_probe(
        [block, block],
        state,
        state,
        source_cells,
        candidates[:, permutation],
        hypothesis_chunk=2,
    )
    for name in ("pair_cosine", "directional_anchor_cosine", "intervention_gain"):
        assert torch.allclose(permuted[name], chunk1[name][:, permutation], atol=1e-6)
    assert chunk1["metadata"]["replay_depth"] == 2
    assert chunk1["metadata"]["candidate_axis_persisted_across_blocks"] is True
    assert chunk1["metadata"]["original_cross_mass_used"] is True
    assert chunk1["metadata"]["unit_cross_attention_forced"] is False
    assert chunk1["metadata"]["cross_mass_denominator"] == "full_local_plus_full_target_cross"
    assert torch.all(chunk1["source_cross_mass"] > 0)
    assert torch.all(chunk1["source_cross_mass"] < 1)
    assert torch.all(chunk1["target_cross_mass"] > 0)
    assert torch.all(chunk1["target_cross_mass"] < 1)
    assert chunk1["source_slot_divergence"].max() > 0


def test_persistent_candidate_slot_dump_and_summary_preserve_audit_contract():
    block = _ToySingleStreamBlock()
    image_tokens = torch.eye(4)
    full_tokens = torch.cat((torch.zeros(1, 4), image_tokens), dim=0)
    state = _toy_replay_state(full_tokens, block_index=27)
    _raw_feature, _ada, prepared = _toy_aligned_feature_and_ada(block, state)
    rows = flux_fjsar_dump_candidates(
        prepared,
        prepared.clone(),
        [[0.0, 0.0], [1.0, 1.0]],
        [2, 2],
        [2, 2],
        src_replay_state=state,
        trg_replay_state=state,
        blocks=[block],
        persistent_candidate_slot_replay_blocks=[block, block],
        interaction_mode="exact",
        target_points=[[0.0, 0.0], [1.0, 1.0]],
        pck_threshold=0.1,
        candidate_topk=2,
        persistent_candidate_slot_replay_audit=True,
        persistent_candidate_slot_replay_topk=(1, 2),
        persistent_candidate_slot_replay_chunk=1,
    )
    assert len(rows) == 2
    for row in rows:
        audit = row["persistent_candidate_slot_replay_audit"]
        assert audit["persistent_slot_contract"]["prediction_changed"] is False
        assert audit["persistent_slot_contract"]["replay_depth"] == 2
        assert audit["persistent_slot_contract"]["native_fallback_used"] is False
        assert audit["persistent_slot_contract"]["gt_used_for_scoring"] is False
        assert audit["score_names"][0] == "directional_anchor_cosine"
        assert len(audit["candidates"]) == 2
        assert all("source_cross_mass" in item["diagnostics"] for item in audit["candidates"])
    records = [
        {
            "category": "toy",
            "pair_json": "pair.json",
            "src_image": "a.jpg",
            "trg_image": "b.jpg",
            "oracle_gap_case": True,
            "attention_harms_native_case": False,
            "attention_rescues_native_case": False,
            "attention_top1_pck_hit": False,
            "persistent_candidate_slot_replay_audit": row[
                "persistent_candidate_slot_replay_audit"
            ],
        }
        for row in rows
    ]
    summary = _summarize_persistent_candidate_slot_replay_audit(records)
    group = summary["both_wrong_top20_hit"]
    assert summary["decision_rule"]["primary_signal"] == "directional_anchor_cosine"
    assert set(group["risk_controls"]) == {
        "compute",
        "slot_isolation",
        "artificial_intervention",
        "global_competition",
        "candidate_coverage",
        "prior_causal_failure",
    }
    checks = group["mechanism_checks"]
    assert checks["native_fallback_used_count"] == 0
    assert checks["gt_used_for_scoring_count"] == 0


def test_persistent_candidate_slot_parser_flags_are_exposed():
    args = build_parser().parse_args([
        "--output_json",
        "result.json",
        "--fjsar_persistent_candidate_slot_replay_audit",
        "--fjsar_persistent_candidate_slot_replay_topk",
        "1",
        "5",
        "20",
        "--fjsar_persistent_candidate_slot_replay_chunk",
        "3",
    ])
    assert args.fjsar_persistent_candidate_slot_replay_audit is True
    assert args.fjsar_persistent_candidate_slot_replay_topk == [1, 5, 20]
    assert args.fjsar_persistent_candidate_slot_replay_chunk == 3


def test_latent_expert_audit_recovers_stable_minority_head_hidden_by_mean():
    # Candidate 0 wins after averaging all experts. Head 0 is the only head
    # whose candidate-1 support is stable across ensemble members and points.
    support = torch.tensor(
        [
            [
                [[-2.0, 0.0], [-2.0, 0.0]],
                [[0.0, -4.0], [0.0, -4.0]],
                [[0.0, -4.0], [-4.0, 0.0]],
            ],
            [
                [[-2.0, 0.0], [-2.0, 0.0]],
                [[0.0, -4.0], [-4.0, 0.0]],
                [[0.0, -4.0], [0.0, -4.0]],
            ],
            [
                [[-2.0, 0.0], [-2.0, 0.0]],
                [[-4.0, 0.0], [0.0, -4.0]],
                [[0.0, -4.0], [0.0, -4.0]],
            ],
        ],
        dtype=torch.float32,
    )
    probe = {
        "expert_scores": {"bidirectional_negative_log_rank": support},
        "metadata": {"ensemble_size": 3, "head_count": 3},
    }
    audits = _latent_expert_audit_for_points(
        probe,
        torch.tensor([[0, 1], [2, 3]]),
        [2, 2],
        aggregated_attention_scores=torch.tensor([[2.0, 1.0], [2.0, 1.0]]),
        target_points=[[1.0, 0.0], [1.0, 1.0]],
        pck_threshold=0.1,
        topks=(1, 2),
    )

    assert len(audits) == 2
    assert audits[0]["pair_selector"]["stable_head_order"][0] == 0
    assert audits[0]["ranks"]["aggregated_attention"] == 2
    assert audits[0]["ranks"]["mean_expert_support"] == 2
    assert all(audit["ranks"]["stable_head_1"] == 1 for audit in audits)
    assert all(audit["diagnostics"]["any_head_top1_pck_hit"] for audit in audits)
    assert len(audits[0]["candidates"][0]["head_support"]) == 3
    assert len(audits[0]["candidates"][0]["member_support"]) == 3


def test_expert_preserving_hypothesis_routes_qk_identity_consistent_head():
    support_by_head = torch.tensor(
        [[3.0, 1.0, 0.0], [0.0, 3.0, 1.0]],
        dtype=torch.float32,
    )
    identity_by_head = torch.tensor(
        [[0.0, 1.0, 3.0], [0.0, 3.0, 1.0]],
        dtype=torch.float32,
    )
    support = support_by_head[None, :, None, :].expand(2, 2, 2, 3).clone()
    identity = identity_by_head[None, :, None, :].expand(2, 2, 2, 3).clone()
    probe = {
        "expert_scores": {
            "log_exact_mutual_cross_probability": support,
            "symmetric_value_residual_alignment": identity,
        }
    }
    attention = torch.tensor(
        [[3.0, 2.8, 0.0], [3.0, 2.8, 0.0]],
        dtype=torch.float32,
    )
    signals, route = _expert_preserving_hypothesis_scores(probe, attention)
    assert route["selected_head"] == 1
    assert route["aggregation_order"].startswith("candidate_conditioned")
    assert torch.equal(
        signals["pair_head_hypothesis"].argmax(dim=1),
        torch.tensor([1, 1]),
    )


def test_expert_hypothesis_ranking_preserves_attention_candidate_pool():
    block = _ToySingleStreamBlock()
    image_tokens = torch.eye(4)
    full_tokens = torch.cat((torch.zeros(1, 4), image_tokens), dim=0)
    state = FluxReplayState.from_dict(_toy_replay_state(full_tokens))
    attention = torch.eye(4)
    ranked, audits, summary = (
        _expert_preserving_attention_hypothesis_conditioned_replay_rankings(
            {"p_ab": attention, "p_ba": attention.t()},
            [[0.0, 0.0], [1.0, 1.0]],
            [2, 2],
            [2, 2],
            state,
            state,
            [block],
            candidate_topk=4,
            target_points=[[0.0, 0.0], [1.0, 1.0]],
            pck_threshold=10.0,
        )
    )
    assert ranked.shape == (2, 4)
    assert summary["candidate_source"] == "exact_mutual_cross_attention_topk_only"
    assert summary["expert_axes_preserved_until_candidate_scoring"] is True
    assert summary["candidate_value_aggregation_used"] is False
    assert summary["native_candidate_injected_count"] == 0
    assert summary["native_fallback_count"] == 0
    assert summary["gt_used_for_inference"] is False
    for point in audits:
        assert sorted(
            candidate["attention_rank"] for candidate in point["candidates"]
        ) == [1, 2, 3, 4]
        assert point["topk_hits"]["attention"]["@20"] == point[
            "topk_hits"
        ]["pair_head_hypothesis"]["@20"]


def test_flux_fjsar_dump_candidates_writes_operator_manifold_audit():
    block = _ToySingleStreamBlock()
    image_tokens = torch.eye(4)
    full_tokens = torch.cat((torch.zeros(1, 4), image_tokens), dim=0)
    state = _toy_replay_state(full_tokens)
    _raw_feature, ada, prepared = _toy_aligned_feature_and_ada(block, state)
    rows = flux_fjsar_dump_candidates(
        prepared,
        prepared.clone(),
        [[0.0, 0.0], [1.0, 1.0]],
        [2, 2],
        [2, 2],
        src_replay_state=state,
        trg_replay_state=state,
        blocks=[block],
        interaction_mode="identity_preserving",
        target_points=[[0.0, 0.0], [1.0, 1.0]],
        pck_threshold=10.0,
        candidate_topk=2,
        src_ada=ada,
        trg_ada=ada.clone(),
        operator_manifold_audit=True,
    )
    assert len(rows) == 2
    audit = rows[0]["operator_manifold_audit"]
    assert audit["source_cell"] == 0
    assert "target_gt" in audit
    values = torch.tensor([
        audit["source"]["joint_native_cosine"],
        audit["source"]["drift_l2_ratio"],
        audit["target_gt"]["joint_native_cosine"],
        audit["pair"]["source_joint_native_cosine_mean"],
    ])
    assert torch.isfinite(values).all()


def test_flux_fjsar_dump_candidates_writes_residual_readout_audit():
    block = _ToySingleStreamBlock()
    image_tokens = torch.eye(4)
    full_tokens = torch.cat((torch.zeros(1, 4), image_tokens), dim=0)
    state = _toy_replay_state(full_tokens)
    _raw_feature, ada, prepared = _toy_aligned_feature_and_ada(block, state)
    rows = flux_fjsar_dump_candidates(
        prepared,
        prepared.clone(),
        [[0.0, 0.0], [1.0, 1.0]],
        [2, 2],
        [2, 2],
        src_replay_state=state,
        trg_replay_state=state,
        blocks=[block],
        interaction_mode="exact",
        target_points=[[0.0, 0.0], [1.0, 1.0]],
        pck_threshold=10.0,
        candidate_topk=2,
        src_ada=ada,
        trg_ada=ada.clone(),
        residual_readout_audit=True,
        residual_readout_topk=[1, 2],
        latent_expert_audit=True,
        latent_expert_topk=[1, 2],
    )
    assert len(rows) == 2
    audit = rows[0]["residual_readout_audit"]
    assert "value_residual_alignment" in audit["score_names"]
    assert "value_residual_alignment" in audit["ranks"]
    assert len(audit["candidates"]) == 2
    assert all(
        torch.isfinite(torch.tensor(list(candidate["scores"].values()))).all()
        for candidate in audit["candidates"]
    )
    latent = rows[0]["latent_expert_audit"]
    assert latent["metadata"]["support_signal"] == "log_exact_mutual_cross_probability"
    assert len(latent["pair_selector"]["stable_head_order"]) == latent["metadata"]["head_count"]
    assert latent["ranks"]["stable_head_1"] == 1
    assert len(latent["candidates"]) == 2


def test_latent_expert_summary_separates_hidden_signal_and_stable_recovery():
    base_audit = {
        "score_names": ["mean_expert_support", "stable_head_1", "oracle_pair_head_1"],
        "ranks": {
            "mean_expert_support": 2,
            "stable_head_1": 1,
            "oracle_pair_head_1": 1,
        },
        "score_gaps": {},
        "diagnostics": {
            "any_head_top1_pck_hit": True,
            "any_member_top1_pck_hit": True,
            "any_expert_top1_pck_hit": True,
            "head_top1_pck_hit_fraction": 0.25,
            "expert_top1_pck_hit_fraction": 0.125,
            "correct_beats_attention_top1_head_fraction": 0.25,
        },
    }
    records = [{
        "category": "cat",
        "pair_json": "pair.json",
        "src_image": "a.jpg",
        "trg_image": "b.jpg",
        "baseline_pck_hit": False,
        "method_pck_hit": False,
        "attention_top1_pck_hit": False,
        "attention_topk_pck_hit": True,
        "oracle_gap_case": True,
        "attention_harms_native_case": False,
        "attention_rescues_native_case": False,
        "gt_ranks": {"attention": 2},
        "latent_expert_audit": base_audit,
    }]

    summary = _summarize_latent_expert_audit(records)
    oracle_gap = summary["oracle_gap"]
    assert oracle_gap["mechanism_checks"]["hidden_by_mean_but_any_head_top1"] == 1
    assert oracle_gap["mechanism_checks"]["hidden_by_attention_average_but_any_head_top1"] == 1
    assert oracle_gap["mechanism_checks"]["stable_head_recovers_mean_error"] == 1
    assert oracle_gap["mechanism_checks"]["stable_head_recovers_attention_top1_error"] == 1
    assert oracle_gap["attention_rank_bins"]["2-3"]["stable_head_1_top1"] == 1
    assert oracle_gap["pair_level"]["stable_head_improves_mean_pairs"] == 1


def test_attention_feature_side_descriptors_are_finite_and_grid_aligned():
    native = torch.eye(4).t().reshape(1, 4, 2, 2)
    attention = {
        "p_ab": torch.eye(4),
        "p_ba": torch.eye(4),
    }
    for builder in (
        _attention_signature_descriptors,
        _part_common_sharpen_descriptors,
    ):
        src_map, trg_map = builder(native, native.clone(), attention)
        assert src_map.shape[-2:] == (2, 2)
        assert trg_map.shape[-2:] == (2, 2)
        assert torch.isfinite(src_map).all()
        assert torch.isfinite(trg_map).all()


def test_attention_guided_isometry_preserves_source_cosine_geometry():
    torch.manual_seed(7)
    native = F.normalize(torch.randn(1, 12, 3, 3), dim=1)
    attention = {
        "p_ab": torch.eye(9),
        "p_ba": torch.eye(9),
    }
    src_map, trg_map = _attention_guided_isometry_descriptors(
        native,
        native.clone(),
        attention,
        max_anchors=9,
        rank=4,
        min_anchors=4,
    )
    src_before = native[0].permute(1, 2, 0).reshape(9, 12)
    src_after = src_map[0].permute(1, 2, 0).reshape(9, 12)
    before_gram = F.normalize(src_before, dim=1) @ F.normalize(src_before, dim=1).t()
    after_gram = F.normalize(src_after, dim=1) @ F.normalize(src_after, dim=1).t()
    assert torch.allclose(before_gram, after_gram, atol=1e-5)
    assert torch.isfinite(trg_map).all()


def test_orthogonal_context_descriptor_is_finite_and_grid_aligned():
    native = torch.eye(4).t().reshape(1, 4, 2, 2)
    tokens = native[0].permute(1, 2, 0).reshape(4, 4)
    attention = {
        "p_ab": torch.eye(4),
        "p_ba": torch.eye(4),
    }
    src_map, trg_map = _orthogonal_context_descriptors(
        native,
        native.clone(),
        tokens + 0.1,
        tokens.clone() + 0.1,
        tokens,
        tokens.clone(),
        attention,
    )
    assert src_map.shape[-2:] == (2, 2)
    assert trg_map.shape[-2:] == (2, 2)
    assert torch.isfinite(src_map).all()
    assert torch.isfinite(trg_map).all()


def test_spectral_attention_identity_descriptor_preserves_identity_pair():
    native = torch.eye(4).t().reshape(1, 4, 2, 2)
    attention = {
        "p_ab": torch.eye(4),
        "p_ba": torch.eye(4),
    }
    src_map, trg_map = _spectral_attention_identity_descriptors(native, native.clone(), attention)
    prediction = cosine_nn_predict(src_map, trg_map, [[0.0, 0.0], [1.0, 1.0]])
    assert prediction == [[0, 0], [1, 1]]
    assert src_map.shape == trg_map.shape
    assert torch.isfinite(src_map).all()
    assert torch.isfinite(trg_map).all()


def test_spectral_attention_identity_descriptor_handles_rectangular_grids():
    src_native = F.normalize(torch.arange(18, dtype=torch.float32).reshape(1, 3, 2, 3), dim=1)
    trg_native = F.normalize(torch.arange(24, dtype=torch.float32).reshape(1, 3, 2, 4), dim=1)
    p_ab = torch.zeros(6, 8)
    for idx in range(6):
        p_ab[idx, idx] = 0.8
        p_ab[idx, min(idx + 1, 7)] = 0.2
    p_ba = torch.zeros(8, 6)
    for idx in range(6):
        p_ba[idx, idx] = 1.0
    p_ba[6:, -1] = 1.0
    src_map, trg_map = _spectral_attention_identity_descriptors(
        src_native,
        trg_native,
        {"p_ab": p_ab, "p_ba": p_ba},
    )
    assert src_map.shape[0] == 1
    assert trg_map.shape[0] == 1
    assert src_map.shape[1] == trg_map.shape[1]
    assert src_map.shape[-2:] == (2, 3)
    assert trg_map.shape[-2:] == (2, 4)
    assert torch.isfinite(src_map).all()
    assert torch.isfinite(trg_map).all()


def test_local_transport_lift_descriptor_preserves_identity_pair():
    native = torch.eye(9).t().reshape(1, 9, 3, 3)
    attention = {
        "p_ab": torch.eye(9),
        "p_ba": torch.eye(9),
    }
    src_map, trg_map = _local_transport_lift_descriptors(native, native.clone(), attention)
    prediction = cosine_nn_predict(src_map, trg_map, [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    assert prediction == [[0, 0], [1, 1], [2, 2]]
    assert src_map.shape[1] == trg_map.shape[1]
    assert src_map.shape[-2:] == (3, 3)
    assert trg_map.shape[-2:] == (3, 3)
    assert torch.isfinite(src_map).all()
    assert torch.isfinite(trg_map).all()


def test_basin_contrastive_identity_descriptor_preserves_identity_pair():
    native = torch.eye(9).t().reshape(1, 9, 3, 3)
    attention = {
        "p_ab": torch.eye(9),
        "p_ba": torch.eye(9),
    }
    src_map, trg_map = _basin_contrastive_identity_descriptors(native, native.clone(), attention)
    prediction = cosine_nn_predict(src_map, trg_map, [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    assert prediction == [[0, 0], [1, 1], [2, 2]]
    assert src_map.shape[1] == trg_map.shape[1]
    assert src_map.shape[-2:] == (3, 3)
    assert trg_map.shape[-2:] == (3, 3)
    assert torch.isfinite(src_map).all()
    assert torch.isfinite(trg_map).all()


def test_attention_oracle_uses_mutual_attention_not_raw_forward_peak():
    src_features = torch.tensor([[[[1.0]], [[0.0]]]])
    trg_features = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])
    src_tokens = torch.tensor([[1.0, 0.0]])
    trg_tokens = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    attention = {
        "p_ab": torch.tensor([[0.9, 0.1]]),
        "p_ba": torch.tensor([[0.0], [1.0]]),
    }
    counts = _fjsar_candidate_oracle_counts(
        src_features=src_features,
        trg_features=trg_features,
        points=[[0.0, 0.0]],
        target_points=[[1.0, 0.0]],
        source_size=[1, 1],
        target_size=[1, 2],
        pck_threshold=10.0,
        oracle_topk=(1,),
        src_state=FluxReplayState.from_dict(
            _toy_replay_state(torch.cat((torch.zeros(1, 2), src_tokens), dim=0), height=1, width=1)
        ),
        trg_state=FluxReplayState.from_dict(
            _toy_replay_state(torch.cat((torch.zeros(1, 2), trg_tokens), dim=0), height=1, width=2)
        ),
        src_native_prepared=src_tokens,
        trg_native_prepared=trg_tokens,
        src_joint_prepared=src_tokens,
        trg_joint_prepared=trg_tokens,
        attention=attention,
    )
    assert counts["fjsar_oracle_owner_attention@1"] == 1


def test_attention_candidate_dump_uses_mutual_attention_order():
    src_features = torch.tensor([[[[1.0]], [[0.0]]]])
    trg_features = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])
    src_tokens = torch.tensor([[1.0, 0.0]])
    trg_tokens = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    state_src = FluxReplayState.from_dict(
        _toy_replay_state(torch.cat((torch.zeros(1, 2), src_tokens), dim=0), height=1, width=1)
    )
    state_trg = FluxReplayState.from_dict(
        _toy_replay_state(torch.cat((torch.zeros(1, 2), trg_tokens), dim=0), height=1, width=2)
    )
    rows = _fjsar_attention_candidate_records(
        src_features,
        trg_features,
        {
            "p_ab": torch.tensor([[0.9, 0.1]]),
            "p_ba": torch.tensor([[0.0], [1.0]]),
        },
        [[0.0, 0.0]],
        [1, 1],
        [1, 2],
        state_src,
        state_trg,
        topk=2,
        target_points=[[1.0, 0.0]],
        pck_threshold=10.0,
        candidate_descriptor_audit=True,
    )
    assert rows[0]["attention_top1"]["pixel"] == [1, 0]
    assert rows[0]["attention_top1"]["pck_hit"] is True
    assert rows[0]["gt_ranks"]["attention"] == 1
    audit = rows[0]["candidate_descriptor_audit"]
    assert audit["candidate_count"] == 2
    assert audit["ranks"]["attention"] == 1
    assert audit["ranks"]["local_self_similarity"] is not None
    assert audit["ranks"]["attention_jacobian"] is not None
    assert torch.isfinite(torch.tensor([
        candidate["local_self_similarity"]
        for candidate in audit["candidates"]
    ])).all()


def test_attention_flow_audit_ranks_identity_transport_first():
    features = torch.eye(9).t().reshape(1, 9, 3, 3)
    tokens = torch.eye(9)
    state = FluxReplayState.from_dict(
        _toy_replay_state(torch.cat((torch.zeros(1, 9), tokens), dim=0), height=3, width=3)
    )
    rows = _fjsar_attention_candidate_records(
        features,
        features.clone(),
        {
            "p_ab": torch.eye(9),
            "p_ba": torch.eye(9),
        },
        [[1.0, 1.0]],
        [3, 3],
        [3, 3],
        state,
        state,
        topk=3,
        target_points=[[1.0, 1.0]],
        pck_threshold=1.0,
        attention_flow_audit=True,
        attention_flow_radius=1,
    )
    flow = rows[0]["attention_flow_audit"]
    assert flow["proposal_count"] == 3
    assert flow["ranks"]["transport_consistency"] == 1
    assert flow["ranks"]["shape_preservation"] == 1
    assert flow["candidates"][0]["pck_hit"] is True
    assert torch.isfinite(torch.tensor([
        candidate["scores"]["transport_consistency"]
        for candidate in flow["candidates"]
    ])).all()


def test_attention_kernel_audit_compares_raw_and_filtered_topk():
    features = torch.eye(9).t().reshape(1, 9, 3, 3)
    tokens = torch.eye(9)
    state = FluxReplayState.from_dict(
        _toy_replay_state(torch.cat((torch.zeros(1, 9), tokens), dim=0), height=3, width=3)
    )
    attention = torch.eye(9) * 0.9
    attention[4, 0] = 1.0
    rows = _fjsar_attention_candidate_records(
        features,
        features.clone(),
        {
            "p_ab": attention,
            "p_ba": attention.t(),
        },
        [[1.0, 1.0]],
        [3, 3],
        [3, 3],
        state,
        state,
        topk=3,
        target_points=[[1.0, 1.0]],
        pck_threshold=0.1,
        attention_kernel_audit=True,
        attention_kernel_radius=1,
        attention_kernel_topk=(1, 5),
    )
    kernel = rows[0]["attention_kernel_audit"]
    assert kernel["topk_hits"]["raw_attention@1"] is False
    assert kernel["topk_hits"]["raw_attention@5"] is True
    assert kernel["topk_hits"]["filtered_attention@1"] is True
    assert kernel["ranks"]["raw_attention"] == 2
    assert kernel["ranks"]["filtered_attention"] == 1


def test_basin_identity_audit_ranks_native_descriptor_inside_attention_basin():
    features = torch.eye(9).t().reshape(1, 9, 3, 3)
    tokens = torch.eye(9)
    state = FluxReplayState.from_dict(
        _toy_replay_state(torch.cat((torch.zeros(1, 9), tokens), dim=0), height=3, width=3)
    )
    attention = torch.eye(9) * 0.9
    attention[4, 0] = 1.0
    rows = _fjsar_attention_candidate_records(
        features,
        features.clone(),
        {
            "p_ab": attention,
            "p_ba": attention.t(),
        },
        [[1.0, 1.0]],
        [3, 3],
        [3, 3],
        state,
        state,
        topk=3,
        target_points=[[1.0, 1.0]],
        pck_threshold=0.1,
        basin_identity_audit=True,
        basin_identity_topk=3,
        basin_identity_radius=1,
        basin_identity_rank_topk=(1, 3),
    )
    audit = rows[0]["basin_identity_audit"]
    raw = audit["basins"]["raw_basin_native_descriptor"]
    filtered = audit["basins"]["filtered_basin_native_descriptor"]
    assert raw["attention_basin_has_pck_hit"] is True
    assert raw["attention_top1"]["pck_hit"] is False
    assert raw["native_top1_in_basin"]["pck_hit"] is True
    assert audit["topk_hits"]["raw_basin_native_descriptor@1"] is True
    assert filtered["attention_basin_has_pck_hit"] is True
    assert audit["topk_hits"]["filtered_basin_native_descriptor@1"] is True


def test_local_relational_identity_audit_scores_attention_candidates():
    features = torch.eye(9).t().reshape(1, 9, 3, 3)
    tokens = torch.eye(9)
    state = FluxReplayState.from_dict(
        _toy_replay_state(torch.cat((torch.zeros(1, 9), tokens), dim=0), height=3, width=3)
    )
    attention = torch.eye(9) * 0.9
    attention[4, 0] = 1.0
    rows = _fjsar_attention_candidate_records(
        features,
        features.clone(),
        {
            "p_ab": attention,
            "p_ba": attention.t(),
        },
        [[1.0, 1.0]],
        [3, 3],
        [3, 3],
        state,
        state,
        topk=3,
        target_points=[[1.0, 1.0]],
        pck_threshold=0.1,
        local_relational_identity_audit=True,
        local_relational_radius=1,
    )
    audit = rows[0]["local_relational_identity_audit"]
    assert audit["proposal_count"] == 3
    assert audit["ranks"]["native_patch_correlation"] == 1
    assert audit["ranks"]["local_self_similarity_consistency"] == 1
    assert audit["ranks"]["hybrid_local_relational_identity"] == 1
    assert audit["candidates"][0]["pck_hit"] is False
    assert torch.isfinite(torch.tensor([
        candidate["scores"]["hybrid_local_relational_identity"]
        for candidate in audit["candidates"]
    ])).all()


def _dense_candidate_edge_toy():
    features = torch.eye(9).t().reshape(1, 9, 3, 3)
    tokens = torch.eye(9)
    state = FluxReplayState.from_dict(
        _toy_replay_state(
            torch.cat((torch.zeros(1, 9), tokens), dim=0),
            height=3,
            width=3,
        )
    )
    mutual_attention = torch.eye(9) * 0.9
    mutual_attention[4, 0] = 1.0
    candidate_cells = torch.topk(mutual_attention[4:5], k=3, dim=1).indices
    proposal_pixels = candidate_cells.clone()
    return features, state, mutual_attention, candidate_cells, proposal_pixels


def test_dense_candidate_edge_audit_recovers_relationally_consistent_rank2():
    features, state, mutual_attention, candidate_cells, proposal_pixels = (
        _dense_candidate_edge_toy()
    )
    graph = _build_attention_sparse_partial_graph(
        mutual_attention,
        (3, 3),
        (3, 3),
        candidate_topk=3,
        edge_radius=1,
    )
    assert isinstance(graph, AttentionSparsePartialGraph)
    assert graph.candidate_target_index.shape == (9, 3)
    assert graph.candidate_mask.all()
    assert graph.dustbin_target_index == 9
    assert graph.contract()["candidate_source"] == "mutual_cross_attention_topk_only"
    assert graph.contract()["native_candidate_injected"] is False
    assert graph.contract()["native_fallback_used"] is False

    rows = _dense_candidate_edge_separability_audit_for_points(
        features,
        features.clone(),
        mutual_attention,
        torch.tensor([4]),
        candidate_cells,
        proposal_pixels,
        [3, 3],
        state,
        state,
        target_points=[[1.0, 1.0]],
        pck_threshold=0.1,
        edge_radius=1,
    )
    audit = rows[0]
    assert audit["ranks"]["attention_unary_control"] == 2
    assert audit["ranks"]["dense_edge_spatial_message"] == 1
    assert audit["ranks"]["dense_edge_joint_message"] == 1
    assert audit["ranks"]["dense_partial_graph_one_step_belief"] == 1
    assert audit["diagnostics"]["gt_used_for_scoring"] is False
    assert audit["diagnostics"]["native_candidate_injected"] is False
    assert audit["diagnostics"]["native_fallback_used"] is False


def test_dense_candidate_edge_scores_do_not_depend_on_gt():
    features, state, mutual_attention, candidate_cells, proposal_pixels = (
        _dense_candidate_edge_toy()
    )
    kwargs = dict(
        src_features=features,
        trg_features=features.clone(),
        mutual_attention=mutual_attention,
        src_cells=torch.tensor([4]),
        candidate_cells=candidate_cells,
        proposal_pixels=proposal_pixels,
        target_size=[3, 3],
        src_state=state,
        trg_state=state,
        pck_threshold=0.1,
        edge_radius=1,
    )
    with_gt = _dense_candidate_edge_separability_audit_for_points(
        **kwargs, target_points=[[1.0, 1.0]]
    )[0]
    without_gt = _dense_candidate_edge_separability_audit_for_points(
        **kwargs, target_points=None
    )[0]
    assert [candidate["scores"] for candidate in with_gt["candidates"]] == [
        candidate["scores"] for candidate in without_gt["candidates"]
    ]


def test_dense_candidate_edge_summary_reports_net_recovery_and_contract():
    features, state, mutual_attention, candidate_cells, proposal_pixels = (
        _dense_candidate_edge_toy()
    )
    audit = _dense_candidate_edge_separability_audit_for_points(
        features,
        features.clone(),
        mutual_attention,
        torch.tensor([4]),
        candidate_cells,
        proposal_pixels,
        [3, 3],
        state,
        state,
        target_points=[[1.0, 1.0]],
        pck_threshold=0.1,
        edge_radius=1,
    )[0]
    summary = _summarize_dense_candidate_edge_audit([
        {
            "oracle_gap_case": True,
            "attention_harms_native_case": False,
            "attention_rescues_native_case": False,
            "attention_top1_pck_hit": False,
            "attention_topk_pck_hit": True,
            "dense_candidate_edge_audit": audit,
        }
    ])
    checks = summary["oracle_gap"]["mechanism_checks"]
    assert checks["candidate_pck_hit_fraction_mean"] == 1.0 / 3.0
    assert checks["dense_edge_joint_message"]["recovers_attention_top1_errors"] == 1
    assert checks["dense_edge_joint_message"]["attention_top1_error_recovery_rate"] == 1.0
    assert checks["dense_edge_joint_message"]["harms_attention_top1_correct"] == 0
    assert checks["dense_edge_joint_message"]["net_vs_attention_top1"] == 1
    assert checks["dense_edge_joint_message"]["net_pck_rate_vs_attention_top1"] == 1.0
    assert checks["gt_used_for_scoring_count"] == 0
    assert summary["oracle_gap"]["graph_contract"]["dustbin_reserved"] is True


def test_dense_partial_graph_matcher_has_cli_mode_mapping():
    assert _fjsar_mode_config("fjsar_dense_partial_graph_matching") == (
        "dense_partial_graph_matching",
        "exact",
        False,
    )


def test_filtered_spectral_kernel_matcher_has_cli_mode_mapping():
    assert _fjsar_mode_config("fjsar_filtered_spectral_kernel") == (
        "filtered_spectral_kernel",
        "exact",
        False,
    )


def test_expert_hypothesis_matcher_has_cli_mode_mapping():
    assert _fjsar_mode_config(
        "fjsar_expert_preserving_attention_hypothesis_conditioned_replay"
    ) == (
        "expert_preserving_attention_hypothesis_conditioned_replay",
        "exact",
        False,
    )


def test_expert_hypothesis_summary_reports_controls_and_contract():
    point = {
        "baseline_pck_hit": False,
        "method_pck_hit": True,
        "selected_attention_rank": 2,
        "selected_point_head": 1,
        "final_changed_from_attention": True,
        "native_candidate_injected": False,
        "native_fallback_used": False,
        "gt_used_for_inference": False,
        "topk_hits": {
            name: {
                "@1": name != "attention",
                "@3": True,
                "@5": True,
                "@10": True,
                "@20": True,
            }
            for name in (
                "attention",
                "mean_expert_hypothesis",
                "pair_head_support",
                "pair_head_identity",
                "pair_head_hypothesis",
                "pair_expert_hypothesis",
                "point_head_hypothesis",
            )
        },
    }
    summary = _summarize_expert_hypothesis_audits([
        {
            "category": "car",
            "summary": {
                "selected_head": 1,
                "selected_head_agreement": 0.8,
                "selected_head_agreement_margin": 0.2,
                "selected_expert": {"member": 0, "head": 1},
                "native_candidate_injected_count": 0,
                "native_fallback_count": 0,
                "gt_used_for_inference": False,
            },
            "points": [point],
        }
    ])
    all_summary = summary["all"]
    assert all_summary["attention_correct"] == 0
    assert all_summary["method_correct"] == 1
    assert all_summary["recovered_attention_errors"] == 1
    assert all_summary["harmed_attention_correct"] == 0
    assert all_summary["net_recovery_vs_attention"] == 1
    assert all_summary["method_audit_mismatch_count"] == 0
    assert all_summary["selected_head_histogram"] == {"1": 1}
    assert all_summary["point_head_histogram"] == {"1": 1}
    assert all_summary["native_injected_candidate_count"] == 0
    assert all_summary["native_fallback_count"] == 0
    assert all_summary["gt_used_for_inference_count"] == 0
    assert summary["mechanism_checks"]["candidate_value_aggregation_used"] is False


def test_sparse_partial_assignment_enforces_capacity_and_optional_dustbin():
    candidates = torch.tensor([[0, 1], [0, 1]])
    scores = torch.tensor([[3.0, 0.0], [2.0, -10.0]])
    solution = _solve_attention_sparse_partial_assignment(
        candidates,
        scores,
        torch.ones_like(candidates, dtype=torch.bool),
        target_count=2,
        required_source_mask=torch.tensor([True, False]),
    )
    assert int(solution["selected_target"][0]) == 1
    assert int(solution["selected_target"][1]) == 0
    assert solution["dustbin_count"] == 0
    assert solution["unconstrained_collision_count"] == 1
    assert solution["required_source_count"] == 1

    partial = _solve_attention_sparse_partial_assignment(
        candidates,
        scores,
        torch.ones_like(candidates, dtype=torch.bool),
        target_count=2,
    )
    real_targets = partial["selected_target"][
        partial["selected_target"] < 2
    ].tolist()
    assert len(real_targets) == len(set(real_targets))
    assert partial["dustbin_count"] == 1


def test_dense_partial_graph_matching_recovers_required_attention_rank2():
    features, state, mutual_attention, _candidate_cells, _proposal_pixels = (
        _dense_candidate_edge_toy()
    )
    ranked, audits, summary = _dense_partial_graph_matching_rankings(
        features,
        features.clone(),
        {"p_ab": mutual_attention, "p_ba": mutual_attention.t()},
        [[1.0, 1.0]],
        [3, 3],
        [3, 3],
        state,
        state,
        candidate_topk=3,
        target_points=[[1.0, 1.0]],
        pck_threshold=0.1,
    )
    assert int(ranked[0, 0]) == 4
    assert audits[0]["selected_attention_rank"] == 2
    assert audits[0]["solver_assigned_dustbin"] is False
    assert audits[0]["topk_hits"]["attention"]["@1"] is False
    assert audits[0]["topk_hits"]["dense_partial_assignment"]["@1"] is True
    assert audits[0]["native_candidate_injected"] is False
    assert audits[0]["native_fallback_used"] is False
    assert summary["required_source_count"] == 1
    assert summary["required_source_dustbin_count"] == 0
    assert summary["partial_assignment_collision_count"] == 0
    assert summary["descriptor_unary_used"] is False
    assert summary["spatial_edge_used"] is False


def test_dense_partial_graph_summary_reports_attention_recovery_without_native():
    point = {
        "baseline_pck_hit": False,
        "method_pck_hit": True,
        "final_changed_from_attention": True,
        "solver_assigned_dustbin": False,
        "native_candidate_injected": False,
        "native_fallback_used": False,
        "gt_used_for_inference": False,
        "topk_hits": {
            "attention": {"@1": False, "@3": True, "@5": True, "@10": True, "@20": True},
            "dense_relation": {"@1": True, "@3": True, "@5": True, "@10": True, "@20": True},
            "dense_graph_belief": {"@1": False, "@3": True, "@5": True, "@10": True, "@20": True},
            "dense_partial_assignment": {"@1": True, "@3": True, "@5": True, "@10": True, "@20": True},
        },
    }
    summary = _summarize_dense_partial_graph_audits([
        {
            "category": "car",
            "summary": {
                "source_node_count": 9,
                "matched_real_count": 8,
                "dustbin_count": 1,
                "unconstrained_collision_count": 2,
                "partial_assignment_collision_count": 0,
                "native_candidate_injected_count": 0,
                "native_fallback_count": 0,
                "gt_used_for_inference": False,
            },
            "points": [point],
        }
    ])
    all_summary = summary["all"]
    assert all_summary["attention_correct"] == 0
    assert all_summary["method_correct"] == 1
    assert all_summary["net_recovery_vs_attention"] == 1
    assert all_summary["partial_assignment_collision_count_sum"] == 0
    assert all_summary["native_injected_candidate_count"] == 0
    assert all_summary["native_fallback_count"] == 0
    assert all_summary["gt_used_for_inference_count"] == 0


def test_dense_transport_consistency_audit_uses_all_token_field():
    features = torch.eye(9).t().reshape(1, 9, 3, 3)
    tokens = torch.eye(9)
    state = FluxReplayState.from_dict(
        _toy_replay_state(torch.cat((torch.zeros(1, 9), tokens), dim=0), height=3, width=3)
    )
    attention = torch.eye(9) * 0.9
    attention[4, 0] = 1.0
    rows = _fjsar_attention_candidate_records(
        features,
        features.clone(),
        {
            "p_ab": attention,
            "p_ba": attention.t(),
        },
        [[1.0, 1.0]],
        [3, 3],
        [3, 3],
        state,
        state,
        topk=3,
        target_points=[[1.0, 1.0]],
        pck_threshold=0.1,
        dense_transport_consistency_audit=True,
        dense_transport_topk=(1, 3),
    )
    audit = rows[0]["dense_transport_consistency_audit"]
    assert audit["proposal_count"] == 3
    assert audit["ranks"]["dense_transport_row_support"] == 1
    assert audit["ranks"]["dense_transport_reciprocal_support"] == 1
    assert audit["ranks"]["dense_attention_top1_field_support"] == 1
    assert audit["ranks"]["hybrid_dense_transport_consistency"] == 1
    assert audit["candidates"][0]["pck_hit"] is False
    assert torch.isfinite(torch.tensor([
        candidate["scores"]["hybrid_dense_transport_consistency"]
        for candidate in audit["candidates"]
    ])).all()


def test_multilayer_identity_audit_scores_attention_candidates():
    features = torch.zeros(1, 2, 3, 3)
    state = FluxReplayState.from_dict(
        _toy_replay_state(torch.cat((torch.zeros(1, 9), torch.eye(9)), dim=0), height=3, width=3)
    )
    attention = torch.zeros(9, 9)
    attention[4, 0] = 1.0
    attention[4, 4] = 0.9
    attention[4, 1] = 0.8

    src_layer = torch.zeros(1, 2, 3, 3)
    trg_layer = torch.zeros(1, 2, 3, 3)
    src_layer[0, :, 1, 1] = torch.tensor([1.0, 0.0])
    trg_layer[0, :, 0, 0] = torch.tensor([0.0, 1.0])
    trg_layer[0, :, 1, 1] = torch.tensor([1.0, 0.0])

    rows = _fjsar_attention_candidate_records(
        features,
        features.clone(),
        {
            "p_ab": attention,
            "p_ba": attention.t(),
        },
        [[1.0, 1.0]],
        [3, 3],
        [3, 3],
        state,
        state,
        topk=3,
        target_points=[[1.0, 1.0]],
        pck_threshold=1.0,
        multilayer_identity_audit=True,
        multilayer_descriptor_maps={
            "official_block32": (src_layer, trg_layer),
        },
    )
    audit = rows[0]["multilayer_identity_audit"]
    assert audit["proposal_count"] == 3
    assert audit["score_names"] == ["official_block32"]
    assert audit["ranks"]["official_block32"] == 1
    assert audit["candidates"][0]["pck_hit"] is False
    assert torch.isfinite(torch.tensor([
        candidate["scores"]["official_block32"]
        for candidate in audit["candidates"]
    ])).all()


def test_kernel_featureization_audit_preserves_identity_kernel():
    features = torch.eye(9).t().reshape(1, 9, 3, 3)
    tokens = torch.eye(9)
    state = FluxReplayState.from_dict(
        _toy_replay_state(torch.cat((torch.zeros(1, 9), tokens), dim=0), height=3, width=3)
    )
    rows = _fjsar_attention_candidate_records(
        features,
        features.clone(),
        {
            "p_ab": torch.eye(9),
            "p_ba": torch.eye(9),
        },
        [[1.0, 1.0]],
        [3, 3],
        [3, 3],
        state,
        state,
        topk=3,
        target_points=[[1.0, 1.0]],
        pck_threshold=1.0,
        kernel_featureization_audit=True,
        kernel_featureization_ranks=(8,),
        kernel_featureization_weights=(0.5,),
        kernel_featureization_radius=1,
        kernel_featureization_topk=(1, 5),
    )
    audit = rows[0]["kernel_featureization_audit"]
    assert audit["topk_hits"]["native_descriptor@1"] is True
    assert audit["topk_hits"]["raw_attention_kernel@1"] is True
    assert audit["topk_hits"]["positive_kernel_feature@1"] is True
    assert audit["topk_hits"]["svd_kernel_rank8@1"] is True
    assert audit["topk_hits"]["native_plus_svd_rank8_w0.5@1"] is True


def test_method_descriptor_audit_scores_transport_lift_candidates():
    features = torch.eye(9).t().reshape(1, 9, 3, 3)
    tokens = torch.eye(9)
    state = FluxReplayState.from_dict(
        _toy_replay_state(torch.cat((torch.zeros(1, 9), tokens), dim=0), height=3, width=3)
    )
    attention = {
        "p_ab": torch.eye(9),
        "p_ba": torch.eye(9),
    }
    method_src, method_trg = _local_transport_lift_descriptors(features, features.clone(), attention)
    rows = _fjsar_attention_candidate_records(
        features,
        features.clone(),
        attention,
        [[1.0, 1.0]],
        [3, 3],
        [3, 3],
        state,
        state,
        topk=3,
        target_points=[[1.0, 1.0]],
        pck_threshold=1.0,
        method_descriptor_audit_name="transport_lift",
        method_descriptor_src=method_src,
        method_descriptor_trg=method_trg,
    )
    audit = rows[0]["method_descriptor_audit"]
    assert audit["descriptor_name"] == "transport_lift"
    assert audit["ranks"]["method_descriptor"] == 1
    assert audit["candidates"][0]["pck_hit"] is True


def test_transport_lift_branch_audit_scores_branch_candidates():
    features = torch.eye(9).t().reshape(1, 9, 3, 3)
    tokens = torch.eye(9)
    state = FluxReplayState.from_dict(
        _toy_replay_state(torch.cat((torch.zeros(1, 9), tokens), dim=0), height=3, width=3)
    )
    attention = {
        "p_ab": torch.eye(9),
        "p_ba": torch.eye(9),
    }
    branches = {
        "native_only": _local_transport_lift_descriptors(
            features, features.clone(), attention, include_native=True, include_outgoing=False, include_incoming=False
        ),
        "out_only": _local_transport_lift_descriptors(
            features, features.clone(), attention, include_native=False, include_outgoing=True, include_incoming=False
        ),
        "in_only": _local_transport_lift_descriptors(
            features, features.clone(), attention, include_native=False, include_outgoing=False, include_incoming=True
        ),
        "no_native": _local_transport_lift_descriptors(
            features, features.clone(), attention, include_native=False, include_outgoing=True, include_incoming=True
        ),
        "full": _local_transport_lift_descriptors(features, features.clone(), attention),
    }
    rows = _fjsar_attention_candidate_records(
        features,
        features.clone(),
        attention,
        [[1.0, 1.0]],
        [3, 3],
        [3, 3],
        state,
        state,
        topk=3,
        target_points=[[1.0, 1.0]],
        pck_threshold=1.0,
        transport_lift_branch_descriptors=branches,
    )
    audit = rows[0]["transport_lift_branch_audit"]
    assert audit["ranks"]["native_only"] == 1
    assert audit["ranks"]["no_native"] == 1
    assert audit["ranks"]["full"] == 1
    assert audit["candidates"][0]["pck_hit"] is True


def test_transport_factorization_audit_is_finite_and_candidate_aligned():
    features = torch.eye(9).t().reshape(1, 9, 3, 3)
    tokens = torch.eye(9)
    state = FluxReplayState.from_dict(
        _toy_replay_state(torch.cat((torch.zeros(1, 9), tokens), dim=0), height=3, width=3)
    )
    rows = _fjsar_attention_candidate_records(
        features,
        features.clone(),
        {
            "p_ab": torch.eye(9),
            "p_ba": torch.eye(9),
        },
        [[1.0, 1.0]],
        [3, 3],
        [3, 3],
        state,
        state,
        topk=3,
        target_points=[[1.0, 1.0]],
        pck_threshold=1.0,
        transport_factorization_audit=True,
        transport_factorization_radius=1,
        transport_factorization_basis_radius=0,
    )
    factorized = rows[0]["transport_factorization_audit"]
    assert factorized["proposal_count"] == 3
    assert "factorized_outgoing_center" in factorized["ranks"]
    assert "factorized_incoming_center" in factorized["ranks"]
    assert "factorized_bidirectional_center" in factorized["ranks"]
    assert factorized["candidates"][0]["pck_hit"] is True
    assert torch.isfinite(torch.tensor([
        candidate["scores"]["factorized_bidirectional_center"]
        for candidate in factorized["candidates"]
    ])).all()


def test_candidate_internal_state_probe_preserves_candidate_and_expert_axes():
    block = _ToySingleStreamBlock()
    image_tokens = torch.eye(4)
    full_tokens = torch.cat((torch.zeros(1, 4), image_tokens), dim=0)
    state = FluxReplayState.from_dict(_toy_replay_state(full_tokens))
    probe = flux_candidate_internal_state_probe(
        [block],
        state,
        state,
        torch.tensor([0, 1]),
        torch.tensor([[0, 1, 2], [1, 2, 3]]),
        mode="exact",
    )
    assert probe["metadata"]["gt_used_for_features"] is False
    assert probe["metadata"]["prediction_changed"] is False
    assert set(probe["feature_groups"]) == {
        "attention_aggregate",
        "qk_expert",
        "value_expert",
        "token_state",
        "channel_state_sketch",
    }
    for value in probe["feature_groups"].values():
        assert value.shape[:2] == (2, 3)
        assert value.shape[2] > 0
        assert torch.isfinite(value).all()


def test_identity_decodability_batch_separates_features_from_gt_labels():
    block = _ToySingleStreamBlock()
    image_tokens = torch.eye(4)
    full_tokens = torch.cat((torch.zeros(1, 4), image_tokens), dim=0)
    state = _toy_replay_state(full_tokens)
    features = image_tokens.t().reshape(1, 4, 2, 2)
    batch = flux_fjsar_identity_decodability_batch(
        features,
        features.clone(),
        [[0.0, 0.0], [1.0, 0.0]],
        [2, 2],
        [2, 2],
        src_replay_state=state,
        trg_replay_state=state,
        blocks=[block],
        target_points=[[0.0, 0.0], [1.0, 0.0]],
        pck_threshold=1.0,
        candidate_topk=3,
    )
    assert batch["candidate_hits"].shape == (2, 3)
    assert batch["candidate_hits"][:, 0].all()
    assert batch["metadata"]["gt_used_for_features"] is False
    assert batch["metadata"]["gt_used_for_labels_only"] is True
    assert set(batch["feature_groups"]) == {
        "attention_aggregate",
        "qk_expert",
        "value_expert",
        "token_state",
        "channel_state_sketch",
        "proposal_attention",
        "native_control",
        "geometry_control",
        "source_identity_token_sketch",
        "candidate_identity_token_sketch",
    }
    source_sketch = batch["feature_groups"]["source_identity_token_sketch"]
    candidate_sketch = batch["feature_groups"]["candidate_identity_token_sketch"]
    assert source_sketch.shape == candidate_sketch.shape
    assert source_sketch.shape[:2] == (2, 3)
    assert torch.isfinite(source_sketch).all()
    assert torch.isfinite(candidate_sketch).all()
    assert torch.equal(source_sketch[:, :1], source_sketch[:, 1:2])


def test_candidate_feature_batch_has_no_label_or_ground_truth_fields():
    block = _ToySingleStreamBlock()
    image_tokens = torch.eye(4)
    full_tokens = torch.cat((torch.zeros(1, 4), image_tokens), dim=0)
    state = _toy_replay_state(full_tokens)
    features = image_tokens.t().reshape(1, 4, 2, 2)
    batch = flux_fjsar_candidate_feature_batch(
        features,
        features.clone(),
        [[0.0, 0.0], [1.0, 0.0]],
        [2, 2],
        [2, 2],
        src_replay_state=state,
        trg_replay_state=state,
        blocks=[block],
        candidate_topk=3,
    )
    assert "candidate_hits" not in batch
    assert batch["candidate_pixels"].shape == (2, 3)
    assert batch["metadata"]["gt_used_for_features"] is False
    assert batch["metadata"]["gt_used_for_labels_only"] is False
    assert batch["metadata"]["labels_present"] is False


def test_identity_decodability_category_folds_are_disjoint_and_deterministic():
    categories = ["cat", "dog", "car", "bus", "bird", "boat"]
    first = category_folds(categories, 3, 2027)
    second = category_folds(reversed(categories), 3, 2027)
    assert first == second
    assert sorted(value for fold in first for value in fold) == sorted(categories)
    assert all(set(first[i]).isdisjoint(first[j]) for i in range(3) for j in range(i + 1, 3))


def test_identity_decodability_rank_metrics_reports_recovery_and_harm():
    scores = torch.tensor([[0.0, 2.0, 1.0], [2.0, 1.0, 0.0]]).numpy()
    hits = torch.tensor([[False, True, False], [True, False, False]]).numpy()
    metrics = rank_metrics(
        scores,
        hits,
        torch.tensor([False, True]).numpy(),
        torch.tensor([0, 1]).numpy().astype(str),
        torch.tensor([0, 1]).numpy().astype(str),
    )
    assert metrics["probe_top1"] == 1.0
    assert metrics["attention_top1"] == 0.5
    assert metrics["recovers_attention_top1_errors"] == 1
    assert metrics["harms_attention_top1_correct"] == 0


def test_identity_decodability_torch_mlp_backend_is_finite_on_cpu():
    hits = torch.zeros(6, 3, dtype=torch.bool)
    hits[torch.arange(6), torch.tensor([0, 1, 2, 0, 1, 2])] = True
    features = torch.randn(6, 3, 4, generator=torch.Generator().manual_seed(2027)).numpy()
    scores = _fit_torch_mlp_scores(
        features,
        hits.numpy(),
        torch.tensor([True, True, True, False, False, False]).numpy(),
        torch.tensor([False, False, False, True, True, True]).numpy(),
        seed=2027,
        device="cpu",
    )
    assert scores.shape == (3, 3)
    assert torch.isfinite(torch.from_numpy(scores)).all()


def test_identity_decodability_probe_profiles_keep_native_control_explicit():
    assert "channel_state_sketch" not in PROBE_FEATURE_GROUPS["stable_internal"]
    assert "native_control" in PROBE_FEATURE_GROUPS["native_plus_stable_internal"]


def test_identity_decodability_analysis_keeps_outer_categories_held_out(tmp_path):
    shard_paths = []
    group_dims = {
        "proposal_attention": 3,
        "attention_aggregate": 2,
        "qk_expert": 2,
        "value_expert": 2,
        "token_state": 2,
        "channel_state_sketch": 2,
        "native_control": 1,
        "geometry_control": 6,
    }
    for category_index, category in enumerate(("cat", "dog", "car", "bus")):
        hits = torch.zeros(6, 3, dtype=torch.bool)
        positive = torch.tensor([(row + category_index) % 3 for row in range(6)])
        hits[torch.arange(6), positive] = True
        signal = hits.float().unsqueeze(2) * 4.0 - 2.0
        groups = {}
        for name, dimension in group_dims.items():
            groups[name] = signal.expand(-1, -1, dimension).clone().to(torch.float16)
        if category_index == 0:
            groups["channel_state_sketch"][0, 0, 0] = float("nan")
        path = tmp_path / f"{category}.pth"
        torch.save({
            "format_version": 1,
            "category": category,
            "pair_id": f"{category}|pair",
            "feature_groups": groups,
            "candidate_hits": hits,
            "baseline_hits": torch.zeros(6, dtype=torch.bool),
            "metadata": {
                "gt_used_for_features": False,
                "gt_used_for_labels_only": True,
                "probe_is_matcher": False,
                "native_fallback_used": False,
            },
        }, path)
        shard_paths.append(str(path))
    output_path = tmp_path / "summary.json"
    result = analyze_identity_decodability(
        shard_paths,
        output_path=str(output_path),
        fold_count=2,
        seed=2027,
        run_mlp=False,
    )
    assert output_path.exists()
    assert result["contract_violations"] == {
        "gt_used_for_features": 0,
        "gt_not_restricted_to_labels": 0,
        "probe_marked_as_matcher": 0,
        "native_fallback_used": 0,
    }
    assert result["data_contract"]["nonfinite_feature_values"]["channel_state_sketch"] > 0
    assert result["probes"]["linear_all_internal"]["metrics"]["probe_top1"] == 1.0
    for fold in result["probes"]["linear_all_internal"]["folds"]:
        assert set(fold["train_categories"]).isdisjoint(fold["test_categories"])

    torch_output_path = tmp_path / "torch_summary.json"
    torch_result = analyze_identity_decodability(
        shard_paths,
        output_path=str(torch_output_path),
        fold_count=2,
        seed=2027,
        run_linear=False,
        run_mlp=True,
        probe_names=("stable_internal", "native_plus_stable_internal"),
        mlp_backend="torch",
        device="cpu",
    )
    assert torch_output_path.exists()
    assert set(torch_result["probes"]) == {
        "torch_mlp_stable_internal",
        "torch_mlp_native_plus_stable_internal",
    }


def test_flux_fjsar_attention_feature_modes_preserve_identity_pair():
    block = _ToySingleStreamBlock()
    image_tokens = torch.eye(4)
    full_tokens = torch.cat((torch.zeros(1, 4), image_tokens), dim=0)
    state = _toy_replay_state(full_tokens)
    raw_feature, ada, prepared = _toy_aligned_feature_and_ada(block, state)

    for mode in (
        "attention_signature",
        "part_sharpen",
        "orthogonal_context",
        "spectral_identity",
        "transport_lift",
        "basin_contrastive_identity",
        "attention_isometry",
        "identity_preserving_attention",
        "geometry_consistent_attention",
        "native_preserving_topology_rescue",
        "attention_basin_native_refine",
        "attention_relational_graph_matching",
        "dense_partial_graph_matching",
        "expert_preserving_attention_hypothesis_conditioned_replay",
    ):
        prediction, diagnostics = flux_fjsar_predict(
            prepared,
            prepared.clone(),
            [[0.0, 0.0], [1.0, 1.0]],
            [2, 2],
            [2, 2],
            src_replay_state=state,
            trg_replay_state=state,
            src_raw_feature=raw_feature,
            trg_raw_feature=raw_feature.clone(),
            src_ada=ada,
            trg_ada=ada.clone(),
            blocks=[block],
            mode=mode,
            interaction_mode=(
                "geometry_consistent"
                if mode == "geometry_consistent_attention"
                else "identity_preserving"
                if mode == "identity_preserving_attention"
                else "exact"
            ),
            use_coordinate_bias=False,
            geometry_radius=1,
            geometry_strength=0.5,
            target_points=[[0.0, 0.0], [1.0, 1.0]],
            pck_threshold=10.0,
            oracle_topk=(1, 2),
            return_diagnostics=True,
        )
        assert prediction == [[0, 0], [1, 1]]
        assert diagnostics["model_counts"]["fjsar_oracle_total"] == 2
        assert diagnostics["model_counts"]["fjsar_oracle_owner_native@1"] == 2
        assert diagnostics["model_counts"]["fjsar_oracle_owner_attention@1"] == 2
        if mode == "native_preserving_topology_rescue":
            assert diagnostics["model_counts"]["fjsar_oracle_owner_native_preserving_topology_rescue@1"] == 2
            assert diagnostics["topology_rescue_native_keep_rate"] == 1.0
        if mode == "attention_basin_native_refine":
            assert diagnostics["model_counts"]["fjsar_oracle_owner_attention_basin_native_refine@1"] == 2
        if mode == "attention_relational_graph_matching":
            graph = diagnostics["attention_relational_graph_audit"]
            assert graph["summary"]["native_injected_candidate_count"] == 0
            assert graph["summary"]["native_fallback_count"] == 0
            assert diagnostics["model_counts"]["fjsar_oracle_owner_attention_relational_graph_matching@1"] == 2
        if mode == "dense_partial_graph_matching":
            graph = diagnostics["dense_partial_graph_audit"]
            assert graph["summary"]["native_candidate_injected_count"] == 0
            assert graph["summary"]["native_fallback_count"] == 0
            assert graph["summary"]["required_source_dustbin_count"] == 0
            assert diagnostics["model_counts"]["fjsar_oracle_owner_dense_partial_graph_matching@1"] == 2
        if mode == "expert_preserving_attention_hypothesis_conditioned_replay":
            audit = diagnostics["expert_hypothesis_audit"]
            assert audit["summary"]["candidate_source"] == (
                "exact_mutual_cross_attention_topk_only"
            )
            assert audit["summary"]["native_candidate_injected_count"] == 0
            assert audit["summary"]["native_fallback_count"] == 0
            assert audit["summary"]["gt_used_for_inference"] is False
            assert all(
                point["topk_hits"]["attention"]["@20"]
                == point["topk_hits"]["pair_head_hypothesis"]["@20"]
                for point in audit["points"]
            )
            assert all(
                sorted(candidate["attention_rank"] for candidate in point["candidates"])
                == list(range(1, point["candidate_count"] + 1))
                for point in audit["points"]
            )
            assert diagnostics["model_counts"][
                "fjsar_oracle_owner_expert_preserving_attention_hypothesis_conditioned_replay@1"
            ] == 2


def test_pre_softmax_channelwise_identity_field_is_candidate_indexed_and_train_free():
    block = _ToySingleStreamBlock()
    image_tokens = torch.eye(4)
    full_tokens = torch.cat((torch.zeros(1, 4), image_tokens), dim=0)
    state_dict = _toy_replay_state(full_tokens)
    state = FluxReplayState.from_dict(state_dict)
    raw_feature, ada, prepared = _toy_aligned_feature_and_ada(block, state_dict)
    prediction, diagnostics = flux_fjsar_predict(
        prepared,
        prepared.clone(),
        [[0.0, 0.0], [1.0, 1.0]],
        [2, 2],
        [2, 2],
        src_replay_state=state_dict,
        trg_replay_state=state_dict,
        src_raw_feature=raw_feature,
        trg_raw_feature=raw_feature.clone(),
        src_ada=ada,
        trg_ada=ada.clone(),
        blocks=[block],
        mode="pre_softmax_channelwise_identity",
        interaction_mode="exact",
        use_coordinate_bias=False,
        target_points=[[0.0, 0.0], [1.0, 1.0]],
        pck_threshold=10.0,
        oracle_topk=(1, 2),
        return_diagnostics=True,
    )
    assert len(prediction) == 2
    audit = diagnostics["pre_softmax_channelwise_identity_audit"]
    assert audit["summary"]["hypothesis"]["train_free"] is True
    assert audit["summary"]["hypothesis"]["attention_used_as_identity_score"] is False
    assert len(audit["points"]) == 2
    assert all(len(point["candidates"]) == 4 for point in audit["points"])
    assert all(
        set(candidate["scores"]) == {
            "combined",
            "qk_channelwise",
            "value_residual",
            "local_relation",
            "native_local",
            "attention",
        }
        for point in audit["points"]
        for candidate in point["candidates"]
    )
    assert diagnostics["model_counts"][
        "fjsar_oracle_owner_pre_softmax_channelwise_identity@1"
    ] >= 0


def test_layer_routed_identity_uses_multiblock_source_only_maps_without_fallback():
    block = _ToySingleStreamBlock()
    src_tokens = torch.cat((torch.zeros(1, 4), torch.eye(4)), dim=0)
    trg_image_tokens = torch.cat((torch.eye(4), torch.eye(4)[:2]), dim=0)
    trg_tokens = torch.cat((torch.zeros(1, 4), trg_image_tokens), dim=0)
    src_state_dict = _toy_replay_state(src_tokens, height=2, width=2)
    trg_state_dict = _toy_replay_state(trg_tokens, height=2, width=3)
    src_raw, src_ada, src_prepared = _toy_aligned_feature_and_ada(block, src_state_dict)
    trg_raw, trg_ada, trg_prepared = _toy_aligned_feature_and_ada(block, trg_state_dict)
    prediction, diagnostics = flux_fjsar_predict(
        src_prepared,
        trg_prepared,
        [[0.0, 0.0], [1.0, 1.0]],
        [2, 2],
        [2, 3],
        src_replay_state=src_state_dict,
        trg_replay_state=trg_state_dict,
        src_raw_feature=src_raw,
        trg_raw_feature=trg_raw,
        src_ada=src_ada,
        trg_ada=trg_ada,
        blocks=[block],
        mode="layer_routed_identity",
        interaction_mode="exact",
        use_coordinate_bias=False,
        target_points=[[0.0, 0.0], [1.0, 1.0]],
        pck_threshold=10.0,
        oracle_topk=(1, 2),
        candidate_topk=4,
        layer_identity_maps={"official_block24": (src_prepared, trg_prepared)},
        return_diagnostics=True,
    )
    assert len(prediction) == 2
    audit = diagnostics["layer_routed_identity_audit"]
    assert audit["summary"]["hypothesis"]["train_free"] is True
    assert audit["summary"]["hypothesis"]["native_fallback"] is False
    assert audit["summary"]["primary_branch"] == "routing_plus_all_layers"
    assert "official_block24" in audit["summary"]["branches"]
    assert "routing_plus_official_block24" in audit["summary"]["branches"]
    assert len(audit["points"]) == 2
    assert all("branches" in point for point in audit["points"])
    assert diagnostics["model_counts"]["fjsar_oracle_owner_layer_routed_identity@1"] >= 0

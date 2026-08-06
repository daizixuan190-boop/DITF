from types import SimpleNamespace

import torch

import spair_matchers
from flux_joint_replay import _coherent_mutual_kernels
from spair_matchers import (
    _align_flip_mutual_kernel,
    _expert_coherence_relative_gates,
    _filtered_spectral_kernel_maps,
    _head_preserving_spectral_kernel_maps,
    _kernel_svd_feature_maps,
    _local_transport_support_rows,
    _native_plus_kernel_feature_maps,
    filtered_spectral_kernel_feature_maps,
    flux_fjsar_filtered_spectral_feature_maps,
    flux_fjsar_filtered_spectral_feature_map_variants,
)


def test_coherent_mutual_rejects_cross_head_false_reciprocity():
    weighted_ab = torch.tensor(
        [[[[1.0, 0.0]], [[0.0, 1.0]]]],
    )
    weighted_ba = torch.tensor(
        [[[[0.0], [1.0]], [[1.0], [0.0]]]],
    )

    early_average = torch.sqrt(
        weighted_ab.mean(dim=(0, 1))
        * weighted_ba.mean(dim=(0, 1)).t()
    )
    head_coherent, expert_coherent = _coherent_mutual_kernels(
        weighted_ab,
        weighted_ba,
    )

    assert torch.allclose(early_average, torch.full((1, 2), 0.5))
    assert torch.count_nonzero(head_coherent) == 0
    assert torch.count_nonzero(expert_coherent) == 0


def test_flip_mutual_alignment_restores_original_row_and_column_order():
    source_state = SimpleNamespace(image_height=2, image_width=2)
    target_state = SimpleNamespace(image_height=2, image_width=3)
    mutual = torch.arange(24, dtype=torch.float32).reshape(4, 6)

    source_aligned = _align_flip_mutual_kernel(
        mutual,
        source_state,
        target_state,
        source_flipped=True,
        target_flipped=False,
    )
    target_aligned = _align_flip_mutual_kernel(
        mutual,
        source_state,
        target_state,
        source_flipped=False,
        target_flipped=True,
    )
    both_aligned = _align_flip_mutual_kernel(
        mutual,
        source_state,
        target_state,
        source_flipped=True,
        target_flipped=True,
    )

    grid = mutual.reshape(2, 2, 2, 3)
    assert torch.equal(source_aligned, grid.flip(1).reshape(4, 6))
    assert torch.equal(target_aligned, grid.flip(3).reshape(4, 6))
    assert torch.equal(both_aligned, grid.flip((1, 3)).reshape(4, 6))


def test_coherent_mutual_keeps_reciprocity_from_the_same_head():
    weighted_ab = torch.tensor(
        [[[[1.0, 0.0]], [[0.0, 1.0]]]],
    )
    weighted_ba = torch.tensor(
        [[[[1.0], [0.0]], [[0.0], [1.0]]]],
    )

    head_coherent, expert_coherent = _coherent_mutual_kernels(
        weighted_ab,
        weighted_ba,
    )

    expected = torch.tensor([[0.5, 0.5]])
    assert torch.allclose(head_coherent, expected)
    assert torch.allclose(expert_coherent, expected)

    head_coherent, expert_coherent, head_stack = _coherent_mutual_kernels(
        weighted_ab,
        weighted_ba,
        return_head_stack=True,
    )
    assert head_stack.shape == (2, 1, 2)
    assert torch.allclose(head_stack[0], torch.tensor([[1.0, 0.0]]))
    assert torch.allclose(head_stack[1], torch.tensor([[0.0, 1.0]]))


def test_formal_spectral_descriptor_matches_locked_audit_construction():
    state = SimpleNamespace(image_height=2, image_width=3)
    source_native = torch.randn(1, 5, 2, 3)
    target_native = torch.randn(1, 5, 2, 3)
    logits = torch.randn(6, 6)
    p_ab = torch.softmax(logits, dim=1)
    p_ba = torch.softmax(logits.t(), dim=1)
    attention = {"p_ab": p_ab, "p_ba": p_ba}

    actual_source, actual_target, diagnostics = filtered_spectral_kernel_feature_maps(
        source_native,
        target_native,
        attention,
        state,
        state,
        rank=4,
        radius=1,
        weight=0.5,
    )

    mutual = torch.sqrt((p_ab.float() * p_ba.float().t()).clamp_min(0.0))
    support = _local_transport_support_rows(
        mutual,
        torch.arange(6),
        state,
        state,
        radius=1,
    )
    expected_kernel = mutual * support
    expected_source_kernel, expected_target_kernel = _kernel_svd_feature_maps(
        expected_kernel,
        state,
        state,
        rank=4,
    )
    expected_source, expected_target = _native_plus_kernel_feature_maps(
        source_native,
        target_native,
        expected_source_kernel,
        expected_target_kernel,
        weight=0.5,
    )

    assert torch.allclose(actual_source, expected_source, atol=1e-6)
    assert torch.allclose(actual_target, expected_target, atol=1e-6)
    assert diagnostics["rank"] == 4
    assert diagnostics["radius"] == 1
    assert diagnostics["weight"] == 0.5
    assert diagnostics["gt_used"] is False


def test_formal_spectral_descriptor_is_finite_for_zero_attention():
    state = SimpleNamespace(image_height=2, image_width=2)
    native = torch.randn(1, 3, 2, 2)
    attention = {
        "p_ab": torch.zeros(4, 4),
        "p_ba": torch.zeros(4, 4),
    }

    source, target, diagnostics = filtered_spectral_kernel_feature_maps(
        native,
        native.clone(),
        attention,
        state,
        state,
        rank=64,
        radius=2,
        weight=0.5,
    )

    assert torch.isfinite(source).all()
    assert torch.isfinite(target).all()
    assert source.shape[1] == native.shape[1] + 4
    assert diagnostics["effective_rank"] == 4


def test_replay_wrapper_requests_exact_unbiased_attention(monkeypatch):
    state = SimpleNamespace(
        image_height=2,
        image_width=2,
        global_block_index=28,
        ensemble_size=8,
    )
    native = torch.randn(1, 3, 2, 2)
    calls = []

    def fake_replay(
        blocks,
        source,
        target,
        *,
        mode,
        use_coordinate_bias,
    ):
        calls.append(
            (
                blocks,
                source,
                target,
                mode,
                use_coordinate_bias,
            )
        )
        probability = torch.eye(4)
        attention = {"p_ab": probability, "p_ba": probability}
        return torch.empty(0), torch.empty(0), attention

    monkeypatch.setattr(spair_matchers, "run_flux_joint_stack", fake_replay)
    source, target, diagnostics = flux_fjsar_filtered_spectral_feature_maps(
        native,
        native.clone(),
        src_replay_state=state,
        trg_replay_state=state,
        blocks=(object(),),
    )

    assert source.shape == target.shape
    assert calls[0][3:] == ("exact", False)
    assert diagnostics["interaction_mode"] == "exact"
    assert diagnostics["coordinate_bias"] is False
    assert diagnostics["gt_used"] is False

    source_kernel, target_kernel, kernel_diagnostics = (
        flux_fjsar_filtered_spectral_feature_maps(
            native,
            native.clone(),
            src_replay_state=state,
            trg_replay_state=state,
            blocks=(object(),),
            include_native=False,
        )
    )
    assert source_kernel.shape[1] == 4
    assert target_kernel.shape[1] == 4
    assert kernel_diagnostics["includes_native"] is False


def test_one_replay_produces_all_coherent_spectral_variants(monkeypatch):
    state = SimpleNamespace(
        image_height=1,
        image_width=2,
        global_block_index=28,
        ensemble_size=8,
    )
    native = torch.randn(1, 3, 1, 2)
    replay_calls = 0

    def fake_replay(
        blocks,
        source,
        target,
        *,
        mode,
        use_coordinate_bias,
        preserve_coherent_mutual,
    ):
        nonlocal replay_calls
        replay_calls += 1
        identity = torch.eye(2)
        swapped = torch.flip(identity, dims=(1,))
        return torch.empty(0), torch.empty(0), {
            "p_ab": identity,
            "p_ba": identity,
            "head_coherent_mutual": swapped,
            "expert_coherent_mutual": 0.5 * identity,
            "head_coherent_mutual_stack": torch.stack(
                (identity, swapped),
                dim=0,
            ),
        }

    monkeypatch.setattr(spair_matchers, "run_flux_joint_stack", fake_replay)
    variants = flux_fjsar_filtered_spectral_feature_map_variants(
        native,
        native.clone(),
        src_replay_state=state,
        trg_replay_state=state,
        blocks=(object(),),
        rank=2,
        radius=1,
        include_native=False,
    )

    assert replay_calls == 1
    assert set(variants) == {
        "early_average",
        "head_coherent",
        "expert_coherent",
        "expert_coherence_gated",
        "head_preserving",
    }
    for name, (_source, _target, diagnostics) in variants.items():
        if name not in {"expert_coherence_gated", "head_preserving"}:
            assert diagnostics["mutual_aggregation"] == name
        assert diagnostics["gt_used"] is False
    assert torch.allclose(
        variants["early_average"][0],
        variants["expert_coherence_gated"][0],
        atol=1e-6,
    )
    assert torch.allclose(
        variants["early_average"][1],
        variants["expert_coherence_gated"][1],
        atol=1e-6,
    )


def test_flip_orbit_variant_uses_all_four_aligned_interactions(monkeypatch):
    original = SimpleNamespace(
        image_height=1,
        image_width=2,
        global_block_index=28,
        ensemble_size=8,
        view="original",
    )
    flipped = SimpleNamespace(
        image_height=1,
        image_width=2,
        global_block_index=28,
        ensemble_size=8,
        view="flipped",
    )
    native = torch.randn(1, 3, 1, 2)
    calls = []

    def fake_replay(
        blocks,
        source,
        target,
        *,
        mode,
        use_coordinate_bias,
        preserve_coherent_mutual=False,
    ):
        calls.append((source.view, target.view, preserve_coherent_mutual))
        identity = torch.eye(2)
        attention = {"p_ab": identity, "p_ba": identity}
        if preserve_coherent_mutual:
            attention.update(
                {
                    "head_coherent_mutual": identity,
                    "expert_coherent_mutual": identity,
                    "head_coherent_mutual_stack": identity.unsqueeze(0),
                }
            )
        return torch.empty(0), torch.empty(0), attention

    monkeypatch.setattr(spair_matchers, "run_flux_joint_stack", fake_replay)
    variants = flux_fjsar_filtered_spectral_feature_map_variants(
        native,
        native.clone(),
        src_replay_state=original,
        trg_replay_state=original,
        src_hflip_replay_state=flipped,
        trg_hflip_replay_state=flipped,
        blocks=(object(),),
        rank=2,
        radius=1,
        include_native=False,
    )

    assert calls == [
        ("original", "original", True),
        ("flipped", "original", False),
        ("original", "flipped", False),
        ("flipped", "flipped", False),
    ]
    assert "flip_orbit" in variants
    diagnostics = variants["flip_orbit"][2]
    assert diagnostics["orbit_views"] == 4
    assert diagnostics["orbit_alignment"] == "inverse_horizontal_flip"
    assert diagnostics["gt_used"] is False


def test_expert_coherence_gate_is_bounded_and_mean_referenced():
    state = SimpleNamespace(image_height=1, image_width=2)
    identity = torch.eye(2)
    attention = {
        "p_ab": identity,
        "p_ba": identity,
        "expert_coherent_mutual": torch.tensor(
            [[0.25, 0.0], [0.0, 0.75]]
        ),
    }

    source_gate, target_gate, diagnostics = _expert_coherence_relative_gates(
        attention,
        state,
        state,
    )

    expected = torch.tensor([2.0 / 3.0, 1.2])
    assert torch.allclose(source_gate.flatten(), expected, atol=1e-6)
    assert torch.allclose(target_gate.flatten(), expected, atol=1e-6)
    assert torch.all(source_gate >= 0.0)
    assert torch.all(source_gate <= 2.0)
    assert diagnostics["gt_used"] is False


def test_head_preserving_spectral_keeps_separate_head_coordinates():
    state = SimpleNamespace(image_height=1, image_width=2)
    identity = torch.eye(2)
    swapped = torch.flip(identity, dims=(1,))
    attention = {
        "head_coherent_mutual_stack": torch.stack(
            (identity, swapped),
            dim=0,
        )
    }

    rng_before = torch.random.get_rng_state().clone()
    source, target, diagnostics = _head_preserving_spectral_kernel_maps(
        attention,
        state,
        state,
        rank=4,
        radius=1,
    )

    assert source.shape == (1, 4, 1, 2)
    assert target.shape == (1, 4, 1, 2)
    assert diagnostics["head_count"] == 2
    assert diagnostics["rank_per_head"] == 2
    assert diagnostics["effective_rank"] == 4
    assert diagnostics["head_identity_preserved"] is True
    assert diagnostics["gt_used"] is False
    assert torch.equal(torch.random.get_rng_state(), rng_before)

    source_again, target_again, _diagnostics = (
        _head_preserving_spectral_kernel_maps(
            attention,
            state,
            state,
            rank=4,
            radius=1,
        )
    )
    assert torch.allclose(source, source_again)
    assert torch.allclose(target, target_again)


def test_coherent_spectral_modes_require_their_replay_kernel():
    state = SimpleNamespace(image_height=1, image_width=2)
    attention = {"p_ab": torch.eye(2), "p_ba": torch.eye(2)}

    try:
        _filtered_spectral_kernel_maps(
            attention,
            state,
            state,
            mutual_aggregation="head_coherent",
        )
    except ValueError as error:
        assert "head_coherent_mutual" in str(error)
    else:
        raise AssertionError("head-coherent mode accepted a missing kernel")

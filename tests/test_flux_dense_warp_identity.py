import torch

from flux_dense_warp_identity import (
    DenseWarpIdentityConfig,
    build_dense_warp_identity_maps,
    local_mode_coordinates,
)
from spair_matchers import cosine_nn_predict


def _identity_posterior(height: int, width: int) -> torch.Tensor:
    count = height * width
    return torch.eye(count, dtype=torch.float32)


def test_local_mode_avoids_bimodal_global_barycenter():
    probability = torch.zeros((1, 7), dtype=torch.float32)
    probability[0, 1] = 0.51
    probability[0, 5] = 0.49

    coordinates, basin_mass = local_mode_coordinates(
        probability,
        target_height=1,
        target_width=7,
        sigma_cells=0.75,
        iterations=3,
        basin_radius_cells=1.5,
    )

    # A global posterior mean would be near x=3.  Mode-conditioned replay must
    # retain the slightly stronger left hypothesis instead of averaging modes.
    assert coordinates.shape == (1, 2)
    assert coordinates[0, 0] < 0.25
    assert torch.allclose(coordinates[0, 1], torch.tensor(0.0))
    assert 0.50 <= float(basin_mass[0]) <= 0.52


def test_identity_posterior_produces_identity_warp_and_reliable_cycle():
    height, width = 3, 4
    posterior = _identity_posterior(height, width)
    native = torch.randn((1, 8, height, width), generator=torch.Generator().manual_seed(7))

    result = build_dense_warp_identity_maps(
        native,
        native.clone(),
        posterior,
        posterior,
        source_grid=(height, width),
        target_grid=(height, width),
    )

    expected_y, expected_x = torch.meshgrid(
        torch.linspace(0.0, 1.0, height),
        torch.linspace(0.0, 1.0, width),
        indexing="ij",
    )
    expected = torch.stack((expected_x, expected_y), dim=-1)
    assert torch.allclose(result["warp_ab"], expected, atol=1e-5)
    assert torch.allclose(result["warp_ba"], expected, atol=1e-5)
    assert float(result["cycle_error_a"].max()) < 1e-4
    assert float(result["cycle_error_b"].max()) < 1e-4
    assert float(result["reliability_a"].min()) > 0.99
    assert float(result["reliability_b"].min()) > 0.99


def test_pair_conditioned_identity_breaks_repeated_native_descriptor_tie():
    height, width = 1, 4
    posterior = _identity_posterior(height, width)
    source = torch.zeros((1, 2, height, width), dtype=torch.float32)
    target = torch.zeros_like(source)

    # Source cell 2 and target cells 0/2 are intentionally identical. Native
    # cosine NN therefore chooses the first target cell, while the shared warp
    # coordinate must preserve the correct cell identity.
    source[0, :, 0, 2] = torch.tensor([1.0, 0.0])
    target[0, :, 0, 0] = torch.tensor([1.0, 0.0])
    target[0, :, 0, 2] = torch.tensor([1.0, 0.0])

    native_prediction = cosine_nn_predict(source, target, [[2, 0]])
    assert native_prediction == [[0, 0]]

    result = build_dense_warp_identity_maps(
        source,
        target,
        posterior,
        posterior,
        source_grid=(height, width),
        target_grid=(height, width),
    )
    fused_prediction = cosine_nn_predict(
        result["source_fused"], result["target_fused"], [[2, 0]]
    )
    assert fused_prediction == [[2, 0]]


def test_jacobian_consistency_accepts_affine_field_and_rejects_discontinuity():
    height = width = 5
    posterior = _identity_posterior(height, width)
    native = torch.randn((1, 4, height, width), generator=torch.Generator().manual_seed(11))
    affine = build_dense_warp_identity_maps(
        native,
        native.clone(),
        posterior,
        posterior,
        source_grid=(height, width),
        target_grid=(height, width),
    )

    broken = posterior.clone()
    # Force the middle source cell to a remote target while preserving a valid
    # row distribution. This creates a sharp local Jacobian inconsistency.
    middle = (height // 2) * width + width // 2
    broken[middle].zero_()
    broken[middle, 0] = 1.0
    broken_result = build_dense_warp_identity_maps(
        native,
        native.clone(),
        broken,
        broken.t().contiguous(),
        source_grid=(height, width),
        target_grid=(height, width),
        config=DenseWarpIdentityConfig(cycle_weight=0.0),
    )

    center = (height // 2, width // 2)
    assert float(affine["jacobian_confidence_a"][center]) > 0.99
    assert float(broken_result["jacobian_confidence_a"][center]) < 0.75


def test_dense_warp_rejects_attention_shape_mismatch():
    source = torch.zeros((1, 2, 2, 2))
    target = torch.zeros((1, 2, 2, 3))
    bad_ab = torch.zeros((4, 5))
    bad_ba = torch.zeros((6, 4))

    try:
        build_dense_warp_identity_maps(
            source,
            target,
            bad_ab,
            bad_ba,
            source_grid=(2, 2),
            target_grid=(2, 3),
        )
    except ValueError as exc:
        assert "attention shape" in str(exc)
    else:
        raise AssertionError("shape mismatch must be rejected")

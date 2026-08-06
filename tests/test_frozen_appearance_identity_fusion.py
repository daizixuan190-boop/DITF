import torch
import torch.nn.functional as F

from frozen_appearance_identity_fusion import (
    build_equal_energy_appearance_base,
    build_roma_augmented_appearance,
    build_weighted_appearance_base,
    build_weighted_roma_augmented_appearance,
    sample_dino_map_on_image_grid,
)
from flux_dense_warp_identity import build_identity_maps_from_warp_fields
from spair_matchers import cosine_nn_predict


def _identity_warps(height: int, width: int):
    ys, xs = torch.meshgrid(
        torch.linspace(0.0, 1.0, height),
        torch.linspace(0.0, 1.0, width),
        indexing="ij",
    )
    warp = torch.stack((xs, ys), dim=-1)
    support = torch.ones((height, width), dtype=torch.float32)
    return warp, warp.clone(), support, support.clone()


def test_dino_square_canvas_is_sampled_at_original_image_token_centres():
    grid_size = 60
    axis = (torch.arange(grid_size, dtype=torch.float32) + 0.5) / grid_size
    normalized = 2.0 * axis - 1.0
    yy, xx = torch.meshgrid(normalized, normalized, indexing="ij")
    dino = torch.stack((xx, yy), dim=0)

    sampled = sample_dino_map_on_image_grid(
        dino,
        image_size=(100, 200),
        output_grid=(2, 4),
        canvas_size=840,
    )

    # Native token centres are the inverse of the later align_corners=False
    # upsampling.  The 2:1 image occupies a 840x420 centred canvas region.
    source_x = (torch.arange(4, dtype=torch.float32) + 0.5) * 200 / 4 - 0.5
    source_y = (torch.arange(2, dtype=torch.float32) + 0.5) * 100 / 2 - 0.5
    canvas_x = source_x * 4.2
    canvas_y = source_y * 4.2 + 210.0
    expected_x = 2.0 * (canvas_x + 0.5) / 840.0 - 1.0
    expected_y = 2.0 * (canvas_y + 0.5) / 840.0 - 1.0

    assert sampled.shape == (1, 2, 2, 4)
    assert torch.allclose(sampled[0, 0], expected_x.expand(2, -1), atol=1e-6)
    assert torch.allclose(
        sampled[0, 1], expected_y[:, None].expand(-1, 4), atol=1e-6
    )


def test_equal_energy_base_gives_flux_and_dino_equal_cosine_weight():
    flux = torch.tensor([[[[3.0]], [[4.0]]]])
    dino = torch.tensor([[[[0.0]], [[2.0]], [[0.0]]]])

    fused = build_equal_energy_appearance_base(flux, dino)
    flux_branch, dino_branch = fused[:, :2], fused[:, 2:]

    assert torch.allclose(torch.linalg.vector_norm(fused, dim=1), torch.ones(1, 1, 1))
    expected_energy = torch.full((1, 1, 1), 0.5)
    assert torch.allclose(flux_branch.square().sum(dim=1), expected_energy)
    assert torch.allclose(dino_branch.square().sum(dim=1), expected_energy)


def test_weighted_appearance_base_locks_auxiliary_energy_ratio():
    native = torch.tensor([[[[3.0]], [[4.0]]]])
    auxiliary = torch.tensor([[[[0.0]], [[2.0]], [[0.0]]]])

    fused = build_weighted_appearance_base(native, auxiliary, auxiliary_weight=0.5)
    native_branch, auxiliary_branch = fused[:, :2], fused[:, 2:]

    assert torch.allclose(torch.linalg.vector_norm(fused, dim=1), torch.ones(1, 1, 1))
    assert torch.allclose(
        native_branch.square().sum(dim=1), torch.full((1, 1, 1), 0.8)
    )
    assert torch.allclose(
        auxiliary_branch.square().sum(dim=1), torch.full((1, 1, 1), 0.2)
    )


def test_weighted_roma_main_branch_is_dual_fusion_of_weighted_appearance():
    height, width = 2, 3
    source_native = torch.randn(1, 4, height, width)
    target_native = torch.randn(1, 4, height, width)
    source_auxiliary = torch.randn(1, 5, height, width)
    target_auxiliary = torch.randn(1, 5, height, width)
    warp_ab, warp_ba, support_a, support_b = _identity_warps(height, width)

    actual = build_weighted_roma_augmented_appearance(
        source_native,
        target_native,
        source_auxiliary,
        target_auxiliary,
        warp_ab,
        warp_ba,
        auxiliary_weight=0.5,
        source_support=support_a,
        target_support=support_b,
    )
    source_appearance = build_weighted_appearance_base(
        source_native,
        source_auxiliary,
        auxiliary_weight=0.5,
    )
    target_appearance = build_weighted_appearance_base(
        target_native,
        target_auxiliary,
        auxiliary_weight=0.5,
    )
    expected = build_identity_maps_from_warp_fields(
        source_appearance,
        target_appearance,
        warp_ab,
        warp_ba,
        source_support=support_a,
        target_support=support_b,
    )

    assert torch.equal(actual["source_appearance"], source_appearance)
    assert torch.equal(actual["target_appearance"], target_appearance)
    assert torch.equal(actual["source_fused_dual"], expected["source_fused_dual"])
    assert torch.equal(actual["target_fused_dual"], expected["target_fused_dual"])


def test_dino_branch_can_break_a_repeated_flux_appearance_tie():
    source_flux = torch.tensor([[[[1.0]], [[0.0]]]])
    target_flux = torch.tensor([[[[1.0, 1.0]], [[0.0, 0.0]]]])
    source_dino = torch.tensor([[[[0.0]], [[1.0]]]])
    target_dino = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])

    source = build_equal_energy_appearance_base(source_flux, source_dino)
    target = build_equal_energy_appearance_base(target_flux, target_dino)

    assert cosine_nn_predict(source_flux, target_flux, [[0, 0]]) == [[0, 0]]
    assert cosine_nn_predict(source, target, [[0, 0]]) == [[1, 0]]


def test_roma_reliability_norm_is_preserved_on_equal_energy_appearance_base():
    height, width = 2, 3
    source_flux = F.normalize(torch.randn(1, 4, height, width), dim=1)
    target_flux = source_flux.clone()
    source_dino = F.normalize(torch.randn(1, 5, height, width), dim=1)
    target_dino = source_dino.clone()
    warp_ab, warp_ba, support_a, support_b = _identity_warps(height, width)

    result = build_roma_augmented_appearance(
        source_flux,
        target_flux,
        source_dino,
        target_dino,
        warp_ab,
        warp_ba,
        source_support=support_a,
        target_support=support_b,
    )

    assert torch.allclose(
        torch.linalg.vector_norm(result["source_appearance"], dim=1),
        torch.ones(1, height, width),
        atol=1e-6,
    )
    assert torch.allclose(
        torch.linalg.vector_norm(result["source_identity"], dim=1),
        result["reliability_a"],
        atol=1e-6,
    )
    assert torch.equal(result["source_fused"], result["source_fused_variable_norm"])
    assert torch.equal(result["target_fused"], result["target_fused_variable_norm"])
    assert result["source_fused"].shape[1] > result["source_appearance"].shape[1]

import torch

from flux_dense_warp_identity import build_identity_maps_from_warp_fields
from roma_dense_warp_identity import (
    normalize_roma_certainty,
    roma_warp_to_token_fields,
)
from spair_matchers import cosine_nn_predict


def _identity_symmetric_warp(height: int, width: int):
    ys = torch.linspace(-1.0 + 1.0 / height, 1.0 - 1.0 / height, height)
    xs = torch.linspace(-1.0 + 1.0 / width, 1.0 - 1.0 / width, width)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    coordinates = torch.stack((xx, yy), dim=-1)
    forward = torch.cat((coordinates, coordinates), dim=-1)
    backward = torch.cat((coordinates, coordinates), dim=-1)
    return torch.cat((forward, backward), dim=1), torch.ones((height, 2 * width))


def test_roma_identity_warp_converts_to_endpoint_token_coordinates():
    height, width = 4, 5
    warp, certainty = _identity_symmetric_warp(height, width)

    fields = roma_warp_to_token_fields(
        warp,
        certainty,
        source_grid=(height, width),
        target_grid=(height, width),
    )

    ys, xs = torch.meshgrid(
        torch.linspace(0.0, 1.0, height),
        torch.linspace(0.0, 1.0, width),
        indexing="ij",
    )
    expected = torch.stack((xs, ys), dim=-1)
    assert torch.allclose(fields["warp_ab"], expected, atol=1e-5)
    assert torch.allclose(fields["warp_ba"], expected, atol=1e-5)
    assert torch.allclose(fields["support_a"], torch.full((height, width), 0.5))
    assert torch.allclose(fields["support_b"], torch.full((height, width), 0.5))


def test_roma_certainty_normalization_is_scale_invariant():
    certainty = torch.tensor([[0.0, 0.001, 0.002, 0.004]], dtype=torch.float32)
    first = normalize_roma_certainty(certainty)
    second = normalize_roma_certainty(100.0 * certainty)

    assert torch.allclose(first, second, atol=1e-6)
    assert float(first[0, 0]) == 0.0
    assert 0.0 < float(first[0, 1]) < float(first[0, 3]) < 1.0


def test_roma_symmetric_halves_preserve_direction_and_raw_certainty():
    height, width = 4, 6
    warp, certainty = _identity_symmetric_warp(height, width)
    warp[:, :width, 2] = 0.5
    warp[:, width:, 0] = -0.5
    certainty[:, :width] = 2.0
    certainty[:, width:] = 8.0

    fields = roma_warp_to_token_fields(
        warp,
        certainty,
        source_grid=(2, 3),
        target_grid=(3, 4),
    )

    # RoMa's A->B coordinates live in the final two channels of the left
    # symmetric half, whereas B->A lives in the first two channels of the
    # right half.  This guards the exact API used by the installed RoMa model.
    assert fields["warp_ab"].shape == (2, 3, 2)
    assert fields["warp_ba"].shape == (3, 4, 2)
    assert torch.allclose(fields["warp_ab"][..., 0], torch.full((2, 3), 5.0 / 6.0))
    assert torch.allclose(fields["warp_ba"][..., 0], torch.full((3, 4), 1.0 / 8.0))
    assert torch.allclose(fields["certainty_a_raw"], torch.full((2, 3), 2.0))
    assert torch.allclose(fields["certainty_b_raw"], torch.full((3, 4), 8.0))


def test_roma_identity_feature_breaks_repeated_native_tie():
    height, width = 1, 4
    warp, certainty = _identity_symmetric_warp(height, width)
    fields = roma_warp_to_token_fields(
        warp,
        certainty,
        source_grid=(height, width),
        target_grid=(height, width),
    )
    source = torch.zeros((1, 2, height, width), dtype=torch.float32)
    target = torch.zeros_like(source)
    source[0, :, 0, 2] = torch.tensor([1.0, 0.0])
    target[0, :, 0, 0] = torch.tensor([1.0, 0.0])
    target[0, :, 0, 2] = torch.tensor([1.0, 0.0])

    result = build_identity_maps_from_warp_fields(
        source,
        target,
        fields["warp_ab"],
        fields["warp_ba"],
        source_support=fields["support_a"],
        target_support=fields["support_b"],
    )

    assert cosine_nn_predict(source, target, [[2, 0]]) == [[0, 0]]
    assert cosine_nn_predict(
        result["source_fused"], result["target_fused"], [[2, 0]]
    ) == [[2, 0]]
    assert cosine_nn_predict(
        result["source_fused_forward"],
        result["target_fused_forward"],
        [[2, 0]],
    ) == [[2, 0]]
    assert cosine_nn_predict(
        result["source_fused_dual"], result["target_fused_dual"], [[2, 0]]
    ) == [[2, 0]]


def test_dual_kernel_preserves_original_reliability_norm():
    height, width = 2, 3
    warp, certainty = _identity_symmetric_warp(height, width)
    fields = roma_warp_to_token_fields(
        warp,
        certainty,
        source_grid=(height, width),
        target_grid=(height, width),
    )
    native = torch.randn((1, 3, height, width))
    result = build_identity_maps_from_warp_fields(
        native,
        native.clone(),
        fields["warp_ab"],
        fields["warp_ba"],
        source_support=fields["support_a"],
        target_support=fields["support_b"],
    )

    mutual_norm = torch.linalg.vector_norm(result["source_identity"], dim=1)
    dual_norm = torch.linalg.vector_norm(result["source_dual_identity"], dim=1)
    assert torch.allclose(dual_norm, mutual_norm, atol=1e-6)


def test_fusion_does_not_penalize_high_reliability_target_when_source_is_uncertain():
    source = torch.tensor([[[[1.0]], [[0.0]]]])
    target = torch.tensor(
        [[[[0.8, 0.9]], [[0.6, float(0.19**0.5)]]]],
        dtype=torch.float32,
    )
    warp_ab = torch.tensor([[[1.0, 0.0]]])
    warp_ba = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])

    result = build_identity_maps_from_warp_fields(
        source,
        target,
        warp_ab,
        warp_ba,
        source_support=torch.zeros((1, 1)),
        target_support=torch.tensor([[0.0, 1.0]]),
    )

    # Native cosine prefers target 1 (0.9 > 0.8). A variable augmented norm
    # would divide the reliable target by sqrt(2) and incorrectly choose 0,
    # even though source reliability is zero and identity must be neutral.
    assert cosine_nn_predict(source, target, [[0, 0]]) == [[1, 0]]
    assert cosine_nn_predict(
        result["source_fused_variable_norm"],
        result["target_fused_variable_norm"],
        [[0, 0]],
    ) == [[0, 0]]
    assert cosine_nn_predict(
        result["source_fused"], result["target_fused"], [[0, 0]]
    ) == [[1, 0]]

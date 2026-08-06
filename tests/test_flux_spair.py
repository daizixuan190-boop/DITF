import torch

from flux_spair import (
    chunked_native_flux_predictions,
    grid_candidate_points,
    grid_cosine_scores,
    long_side_grid_shape,
    native_flux_pck_hits,
    native_flux_predictions,
    points_to_grid_indices,
    prepare_flux_feature,
)
from ownership_diagnostics import controlled_candidate_rows, pck_hits


def test_prepare_flux_feature_applies_layernorm_and_adaln():
    raw = torch.tensor([[[[1.0]], [[3.0]]]])
    ada = torch.tensor([[[0.5, -0.5], [1.0, 0.0]]])
    result = prepare_flux_feature(raw, ada, channel_discard=False)
    expected_norm = torch.tensor([-1.0, 1.0])
    expected = (1 + torch.tensor([1.0, 0.0])) * expected_norm + torch.tensor([0.5, -0.5])
    assert torch.allclose(result[0, :, 0, 0], expected, atol=2e-6)


def test_long_side_grid_preserves_aspect_ratio():
    assert long_side_grid_shape(300, 500, 60) == (36, 60)
    assert long_side_grid_shape(500, 300, 60) == (60, 36)


def test_grid_points_and_point_indices_share_half_pixel_geometry():
    candidates = grid_candidate_points(10, 20, 2, 4)
    assert torch.allclose(candidates[0], torch.tensor([2.0, 2.0]))
    assert torch.allclose(candidates[-1], torch.tensor([17.0, 7.0]))
    indices = points_to_grid_indices(candidates, 10, 20, 2, 4)
    assert torch.equal(indices, torch.arange(8))


def test_native_flux_prediction_uses_original_pixel_grid():
    source = torch.zeros(1, 2, 2, 2)
    target = torch.zeros(1, 2, 2, 2)
    source[0, :, 0, 0] = torch.tensor([1.0, 0.0])
    target[0, :, 1, 1] = torch.tensor([1.0, 0.0])
    target[0, :, 0, 0] = torch.tensor([0.0, 1.0])
    prediction = native_flux_predictions(source, target, [[0, 0]], (2, 2), (2, 2))
    assert torch.equal(prediction, torch.tensor([[1.0, 1.0]]))
    assert native_flux_pck_hits(prediction, torch.tensor([[2.0, 1.0]]), threshold=10).item()


def test_chunked_native_prediction_matches_full_upsample():
    torch.manual_seed(7)
    source = torch.randn(1, 7, 3, 4)
    target = torch.randn(1, 7, 4, 3)
    points = [[0, 0], [3, 2], [2, 1]]
    source_size = (6, 8)
    target_size = (8, 6)
    expected = native_flux_predictions(source, target, points, source_size, target_size)
    actual = chunked_native_flux_predictions(
        source, target, points, source_size, target_size, channel_chunk=2
    )
    assert torch.equal(actual, expected)


def test_flux_grid_controlled_k1_matches_grid_baseline():
    source = torch.zeros(1, 2, 2, 2)
    target = torch.zeros(1, 2, 2, 2)
    source[0, :, 0, 0] = torch.tensor([1.0, 0.0])
    target[0, :, 1, 1] = torch.tensor([1.0, 0.0])
    scores = grid_cosine_scores(source, target, [[0, 0]], 2, 2)
    candidates = grid_candidate_points(2, 2, 2, 2)
    baseline_indices = scores.argmax(dim=1)
    hits = pck_hits(candidates[baseline_indices], torch.tensor([[1.0, 1.0]]), threshold=10)
    rows = controlled_candidate_rows(
        scores,
        candidates,
        torch.tensor([[1.0, 1.0]]),
        threshold=10,
        ks=[1],
        baseline_indices=baseline_indices,
    )
    assert rows[0]["owner_candidate_hit@1"] == int(hits[0]) == 1

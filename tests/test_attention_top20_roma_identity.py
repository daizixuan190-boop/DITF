import torch

from eval_spair_attention_top20_roma_identity import (
    _as_unbatched_warp,
    rank_attention_candidates_with_roma,
    rank_attention_candidates_with_roma_consensus,
)


def _identity_symmetric_warp(height: int, width: int):
    ys = torch.linspace(-1.0 + 1.0 / height, 1.0 - 1.0 / height, height)
    xs = torch.linspace(-1.0 + 1.0 / width, 1.0 - 1.0 / width, width)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack((xx, yy), dim=-1)
    forward = torch.cat((coords, coords), dim=-1)
    backward = torch.cat((coords, coords), dim=-1)
    warp = torch.cat((forward, backward), dim=1)
    certainty = torch.ones((height, 2 * width))
    return warp, certainty


def test_identity_warp_ranks_mutual_coordinate_first():
    warp, certainty = _identity_symmetric_warp(8, 8)
    source_points = torch.tensor([[2.0, 3.0]])
    candidates = torch.tensor([[[6.0, 6.0], [2.0, 3.0], [3.0, 3.0]]])

    result = rank_attention_candidates_with_roma(
        source_points,
        candidates,
        (8, 8),
        (8, 8),
        warp,
        certainty,
    )

    assert result["order"].tolist()[0][0] == 1
    assert result["bidirectional_error"][0, 1] < result["bidirectional_error"][0, 2]
    assert torch.allclose(result["mutual_certainty"], torch.ones_like(result["mutual_certainty"]))


def test_batched_roma_outputs_are_normalized():
    warp, certainty = _identity_symmetric_warp(4, 5)
    actual_warp, actual_certainty = _as_unbatched_warp(
        warp.unsqueeze(0), certainty.unsqueeze(0)
    )

    assert actual_warp.shape == (4, 10, 4)
    assert actual_certainty.shape == (4, 10)


def _ranking(order, errors):
    order = torch.tensor([order], dtype=torch.long)
    errors = torch.tensor([errors], dtype=torch.float32)
    positions = torch.empty_like(order)
    positions.scatter_(
        1,
        order,
        torch.arange(order.shape[1], dtype=positions.dtype)[None].expand_as(order),
    )
    return {
        "order": order,
        "bidirectional_error": errors,
        "forward_error": errors,
        "backward_error": errors,
        "mutual_certainty": torch.ones_like(errors),
        "source_certainty": torch.ones(1),
        "predicted_target_normalized": torch.zeros(1, 2),
    }


def test_multiview_consensus_uses_candidate_rank_not_resolution_scale():
    first = _ranking([0, 1, 2], [0.01, 0.2, 0.3])
    second = _ranking([0, 2, 1], [30.0, 1.0, 20.0])
    result = rank_attention_candidates_with_roma_consensus([first, second])

    assert result["order"].tolist()[0][0] == 0
    assert result["consensus_top1_votes"].tolist()[0][0] == 2
    assert result["consensus_rank_std"].tolist()[0][1] > 0


def test_multiview_consensus_supports_multiple_source_points():
    first = {
        **_ranking([0, 1, 2], [0.01, 0.2, 0.3]),
        "order": torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.long),
        "bidirectional_error": torch.tensor([[0.01, 0.2, 0.3], [0.3, 0.2, 0.01]]),
        "forward_error": torch.tensor([[0.01, 0.2, 0.3], [0.3, 0.2, 0.01]]),
        "backward_error": torch.tensor([[0.01, 0.2, 0.3], [0.3, 0.2, 0.01]]),
        "mutual_certainty": torch.ones(2, 3),
        "source_certainty": torch.ones(2),
        "predicted_target_normalized": torch.zeros(2, 2),
    }
    second = {
        **first,
        "order": torch.tensor([[0, 2, 1], [2, 0, 1]], dtype=torch.long),
    }

    result = rank_attention_candidates_with_roma_consensus([first, second])

    assert result["order"].shape == (2, 3)
    assert result["order"].tolist()[0][0] == 0
    assert result["order"].tolist()[1][0] == 2

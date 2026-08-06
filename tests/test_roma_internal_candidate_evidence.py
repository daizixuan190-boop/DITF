import torch

from audit_roma_internal_candidate_evidence import rank_roma_internal_candidates


def test_pair_conditioned_gp_score_prefers_matching_opposite_position_basis():
    # Two candidates are sampled from the target grid.  The source GP posterior
    # predicts the second target position and the target GP posterior predicts
    # the source position, so candidate 2 must win despite encoder cosine ties.
    source_points = torch.tensor([[0.0, 0.0]])
    candidate_points = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]])
    source_projected = torch.tensor([[[[1.0]], [[0.0]]]])
    target_projected = torch.tensor([[[[1.0, 1.0]], [[0.0, 0.0]]]])
    source_gp = torch.tensor([[[[0.0]], [[1.0]]]])
    target_gp = torch.tensor([[[[1.0, 1.0]], [[0.0, 0.0]]]])
    source_position = torch.tensor([[[[1.0]], [[0.0]]]])
    # The repository's RoMa sampler uses align_corners=False.  The second
    # original-image pixel samples the midpoint of this two-cell test field.
    target_position = torch.tensor([[[[1.0, -1.0]], [[0.0, 2.0]]]])

    ranked = rank_roma_internal_candidates(
        source_points,
        candidate_points,
        (1, 2),
        (1, 2),
        source_projected,
        target_projected,
        source_gp,
        target_gp,
        source_position,
        target_position,
    )

    assert ranked["roma_encoder_cosine"]["order"].tolist() == [[0, 1]]
    assert ranked["roma_gp_coordinate_agreement"]["order"].tolist() == [[1, 0]]


def test_internal_candidate_ranker_rejects_misaligned_candidate_rows():
    field = torch.ones(1, 2, 1, 1)
    try:
        rank_roma_internal_candidates(
            torch.zeros(2, 2),
            torch.zeros(1, 2, 2),
            (1, 1),
            (1, 1),
            field,
            field,
            field,
            field,
            field,
            field,
        )
    except ValueError as error:
        assert "align" in str(error)
    else:  # pragma: no cover
        raise AssertionError("misaligned candidate rows must fail")

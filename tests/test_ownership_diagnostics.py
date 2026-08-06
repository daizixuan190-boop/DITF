import torch

from ownership_diagnostics import controlled_candidate_rows


def test_controlled_candidates_remove_overlap_and_match_unique_budget():
    scores = torch.tensor([
        [0.7, 1.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9],
    ])
    candidate_points = torch.tensor([
        [5.0, 5.0], [15.0, 5.0], [25.0, 5.0],
        [5.0, 15.0], [15.0, 15.0], [25.0, 15.0],
        [5.0, 25.0], [15.0, 25.0], [25.0, 25.0],
    ])
    rows = controlled_candidate_rows(
        scores,
        candidate_points,
        torch.tensor([[5.0, 5.0], [25.0, 25.0]]),
        threshold=9,
        ks=[1],
    )
    owner = rows[0]
    assert owner["owner_candidate_hit@1"] == 0
    assert owner["global_union_candidate_hit@1"] == 1
    assert owner["strict_other_source_candidate_hit@1"] == 1
    assert owner["global_unique_candidate_count@1"] == 2
    assert owner["budget_matched_owner_candidate_hit@1"] == 0
    assert owner["strict_global_not_budget_owner_hit@1"] == 1


def test_controlled_candidates_reject_other_gt_overlap():
    scores = torch.tensor([
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ])
    candidate_points = torch.tensor([[5.0, 5.0], [15.0, 5.0], [5.0, 15.0], [15.0, 15.0]])
    rows = controlled_candidate_rows(
        scores,
        candidate_points,
        torch.tensor([[5.0, 5.0], [6.0, 5.0]]),
        threshold=20,
        ks=[1],
    )
    assert rows[0]["other_source_candidate_hit@1"] == 1
    assert rows[0]["strict_other_source_candidate_hit@1"] == 0


def test_controlled_candidates_preserve_baseline_argmax_under_ties():
    scores = torch.ones(1, 4)
    candidate_points = torch.tensor([[5.0, 5.0], [15.0, 5.0], [5.0, 15.0], [15.0, 15.0]])
    rows = controlled_candidate_rows(
        scores,
        candidate_points,
        torch.tensor([[5.0, 5.0]]),
        threshold=10,
        ks=[1],
        baseline_indices=torch.tensor([0]),
    )
    assert rows[0]["owner_candidate_hit@1"] == 1


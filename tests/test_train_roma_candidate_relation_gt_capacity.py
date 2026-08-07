import torch

from train_roma_candidate_relation_gt_capacity import _metric_row, _strict_residual_mask


def test_strict_residual_mask_requires_current_error_and_valid_candidate():
    identities = [
        ("chair", "a.json", 0, (1.0, 1.0), (2.0, 2.0)),
        ("chair", "a.json", 1, (2.0, 2.0), (3.0, 3.0)),
        ("chair", "a.json", 2, (3.0, 3.0), (4.0, 4.0)),
    ]
    current = {
        ("chair", "a.json", 0): ((1.0, 1.0), (2.0, 2.0), False),
        ("chair", "a.json", 1): ((2.0, 2.0), (3.0, 3.0), True),
        ("chair", "a.json", 2): ((3.0, 3.0), (4.0, 4.0), False),
    }
    candidates = torch.tensor([[False, True], [True, False], [False, False]])
    assert _strict_residual_mask(identities, candidates, current).tolist() == [True, False, False]


def test_metric_row_uses_any_pck_positive_candidate_at_each_rank():
    scores = torch.tensor([[0.1, 0.9, 0.2], [0.7, 0.6, 0.5]])
    positives = torch.tensor([[False, True, True], [False, False, True]])
    metrics = _metric_row(scores, positives)
    assert metrics["queries"] == 2
    assert metrics["top1"] == 0.5
    assert metrics["top3"] == 1.0

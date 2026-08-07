import pytest
import torch

from roma_candidate_relation import (
    CandidateConditionedRelationHead,
    multi_positive_listwise_loss,
    pair_relation_block,
)


def test_pair_relation_block_preserves_query_candidate_axes():
    source = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
    candidate = torch.ones(2, 3, 2)
    relation = pair_relation_block(source, candidate)
    assert relation.shape == (2, 3, 8)
    assert torch.isfinite(relation).all()


def test_relation_head_scores_candidates_jointly_and_rejects_missing_group():
    head = CandidateConditionedRelationHead({"local": 8, "pair": 4}, group_width=3, hidden_width=7)
    groups = {"local": torch.randn(2, 4, 8), "pair": torch.randn(2, 4, 4)}
    assert head(groups).shape == (2, 4)
    try:
        head({"local": groups["local"]})
    except ValueError as error:
        assert "match" in str(error)
    else:  # pragma: no cover
        raise AssertionError("missing relation group must fail")


def test_multi_positive_listwise_loss_accepts_multiple_valid_candidates():
    logits = torch.tensor([[0.0, 3.0, 1.0], [2.0, 0.0, 1.0]], requires_grad=True)
    positives = torch.tensor([[False, True, True], [True, False, False]])
    loss = multi_positive_listwise_loss(logits, positives)
    assert 0.0 < float(loss) < 1.0
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_multi_positive_listwise_loss_rejects_query_without_valid_candidate():
    with pytest.raises(ValueError, match="at least one positive"):
        multi_positive_listwise_loss(torch.zeros(1, 2), torch.zeros(1, 2, dtype=torch.bool))

import torch

from roma_candidate_relation import CandidateConditionedRelationHead, pair_relation_block


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

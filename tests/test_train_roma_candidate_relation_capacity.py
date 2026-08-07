import torch

from train_roma_candidate_relation_capacity import (
    LOCAL_SCALES,
    align_current_identities,
    build_relation_groups,
)


def test_relation_group_builder_keeps_candidate_axis_and_uses_no_labels():
    projected = {
        scale: (torch.ones(1, 2, 2, 2), torch.ones(1, 2, 2, 2))
        for scale in LOCAL_SCALES
    }
    pair_fields = {
        "source_gp": torch.ones(1, 2, 2, 2),
        "target_gp": torch.ones(1, 2, 2, 2),
        "source_position_basis": torch.ones(1, 2, 2, 2),
        "target_position_basis": torch.ones(1, 2, 2, 2),
    }
    output = build_relation_groups(
        projected, pair_fields,
        source_points=torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
        candidate_points=torch.tensor([[[0.0, 0.0], [1.0, 1.0]], [[1.0, 1.0], [0.0, 0.0]]]),
        source_size=(2, 2), target_size=(2, 2),
    )
    assert set(output) == {"local_scale4", "local_scale8", "local_scale16", "gp_forward_position", "gp_reverse_position"}
    assert all(value.shape == (2, 2, 8) for value in output.values())


def test_current_alignment_accepts_json_rounding_but_rejects_stale_pair():
    identities = [("chair", "pair.json", 3, (1.0, 2.0), (3.0, 4.0))]
    current = {
        ("chair", "pair.json", 3): ((1.0 + 5e-5, 2.0), (3.0, 4.0 - 5e-5), True)
    }
    assert align_current_identities(identities, current).tolist() == [True]
    try:
        align_current_identities(identities, {})
    except RuntimeError as error:
        assert "missing=1" in str(error)
    else:  # pragma: no cover
        raise AssertionError("missing current records must not become silent errors")

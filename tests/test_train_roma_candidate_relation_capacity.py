import torch

from train_roma_candidate_relation_capacity import LOCAL_SCALES, build_relation_groups


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

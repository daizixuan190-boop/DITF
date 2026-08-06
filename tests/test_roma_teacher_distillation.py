import torch

from train_flux_attention_identity_verifier_roma_teacher import (
    _candidate_pixel_indices_to_points,
    roma_soft_targets,
)


def test_candidate_pixel_indices_are_converted_to_xy_points():
    pixels = torch.tensor([[0, 5, 11], [3, 8, 10]])
    points = _candidate_pixel_indices_to_points(pixels, (3, 4))
    assert points.tolist() == [
        [[0.0, 0.0], [1.0, 1.0], [3.0, 2.0]],
        [[3.0, 0.0], [0.0, 2.0], [2.0, 2.0]],
    ]


def test_roma_soft_targets_follow_bidirectional_error_and_gate_confidence():
    error = torch.tensor([[0.1, 0.4, 0.8], [0.2, 0.21, 0.9]])
    certainty = torch.ones_like(error)
    targets, confident, diagnostics = roma_soft_targets(
        error,
        certainty,
        temperature=0.25,
        minimum_probability=0.55,
        minimum_certainty=0.0,
    )
    assert targets.shape == error.shape
    assert torch.allclose(targets.sum(dim=1), torch.ones(2))
    assert diagnostics["teacher_rank"].tolist() == [0, 0]
    assert confident.tolist() == [True, False]


def test_roma_soft_targets_reject_low_certainty_without_changing_rank():
    error = torch.tensor([[0.1, 0.4]])
    certainty = torch.tensor([[0.01, 0.9]])
    targets, confident, diagnostics = roma_soft_targets(
        error,
        certainty,
        temperature=0.25,
        minimum_probability=0.5,
        minimum_certainty=0.1,
    )
    assert diagnostics["teacher_rank"].item() == 0
    assert confident.item() is False
    assert torch.isclose(targets.sum(), torch.tensor(1.0))

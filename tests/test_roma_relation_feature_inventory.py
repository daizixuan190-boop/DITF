import torch

from audit_roma_relation_feature_inventory import (
    inventory_projected_relation_fields,
    select_projection_scales,
)


def test_select_projection_scales_intersects_real_decoder_and_encoder_keys():
    assert select_projection_scales({8: torch.ones(2, 3, 2, 2), 16: torch.ones(2, 3, 1, 1)}, ("16", "32")) == (16,)


def test_select_projection_scales_rejects_non_numeric_decoder_keys():
    try:
        select_projection_scales({16: torch.ones(2, 3, 1, 1)}, ("scale16",))
    except RuntimeError as error:
        assert "numeric" in str(error)
    else:  # pragma: no cover
        raise AssertionError("non-numeric RoMa projection key must fail")


def test_inventory_preserves_source_candidate_axes_without_scores_or_gt():
    source = torch.ones(1, 3, 2, 2)
    target = torch.ones(1, 3, 2, 2)
    result = inventory_projected_relation_fields(
        {16: (source, target)},
        source_points=torch.tensor([[0.0, 0.0], [3.0, 3.0]]),
        candidate_points=torch.tensor(
            [
                [[0.0, 0.0], [1.0, 1.0]],
                [[2.0, 2.0], [3.0, 3.0]],
            ]
        ),
        source_size=(4, 4),
        target_size=(4, 4),
    )
    assert result["16"]["source_descriptor_shape"] == [2, 3]
    assert result["16"]["candidate_descriptor_shape"] == [2, 2, 3]
    assert result["16"]["all_finite"] is True

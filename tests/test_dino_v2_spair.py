import math

import torch
from PIL import Image

from dino_v2_spair import (
    CategoryMetrics,
    candidate_hit,
    controlled_candidate_rows,
    cosine_nn_predictions,
    pck_hits,
    preprocess_square_canvas,
    square_canvas_geometry,
    tokens_to_patch_map,
    transform_points_to_canvas,
)


def test_square_canvas_matches_geoaware_resize_geometry():
    scale, offset_x, offset_y, resized_h, resized_w = square_canvas_geometry(300, 500, 840)
    assert scale == 840 / 500
    assert (offset_x, offset_y, resized_h, resized_w) == (0, 168, 504, 840)
    tensor = preprocess_square_canvas(Image.new("RGB", (500, 300), "white"), 840)
    assert tensor.shape == (3, 840, 840)


def test_keypoints_follow_square_canvas_geometry():
    points = transform_points_to_canvas([[0, 0], [500, 300]], 300, 500, 840)
    assert torch.allclose(points, torch.tensor([[0.0, 168.0], [840.0, 672.0]]))


def test_tokens_to_map_drops_cls_and_register_tokens():
    tokens = torch.arange(1 * 8 * 2, dtype=torch.float32).reshape(1, 8, 2)
    feature_map = tokens_to_patch_map(tokens, 2)
    assert feature_map.shape == (1, 2, 2, 2)
    assert torch.equal(feature_map.flatten(2).transpose(1, 2)[0], tokens[0, -4:])


def test_cosine_nn_uses_patch_centers_and_correct_flat_width():
    source = torch.zeros(2, 2, 2)
    target = torch.zeros(2, 2, 2)
    source[:, 0, 0] = torch.tensor([1.0, 0.0])
    target[:, 1, 1] = torch.tensor([1.0, 0.0])
    target[:, 0, 0] = torch.tensor([0.0, 1.0])
    predictions, _ = cosine_nn_predictions(source, target, torch.tensor([[1.0, 1.0]]), image_size=28)
    assert torch.equal(predictions, torch.tensor([[21.0, 21.0]]))
    assert pck_hits(predictions, torch.tensor([[21.0, 21.0]]), threshold=10).tolist() == [True]


def test_category_metrics_distinguishes_per_image_and_per_point():
    metrics = CategoryMetrics()
    metrics.update(torch.tensor([True]))
    metrics.update(torch.tensor([False, False, True]))
    assert math.isclose(metrics.per_image, 2 / 3, rel_tol=1e-6)
    assert metrics.per_point == 0.5


def test_candidate_hit_uses_patch_centers_in_canvas_units():
    candidates = torch.tensor([[3]])
    # Flat index 3 in a 2-wide grid is centered at (21, 21) for stride 14.
    assert candidate_hit(candidates, [21, 21], width=2, threshold=10, patch_stride=14).item()


def test_controlled_candidates_remove_overlap_and_match_unique_budget():
    # On a 3x3 grid, owner zero misses its GT patch 0 in top-1. Source row one
    # proposes patch 0, while owner zero does not contain it in its top-2
    # (the exact cardinality of the global union {0, 1}).
    scores = torch.tensor([
        [0.7, 1.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9],
    ])
    gt_points = torch.tensor([[5.0, 5.0], [25.0, 25.0]])
    rows = controlled_candidate_rows(scores, gt_points, threshold=9, width=3, ks=[1], patch_stride=10)
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
    # Both GT PCK regions include patch-0 center, so the donor proposal is not strict.
    gt_points = torch.tensor([[5.0, 5.0], [6.0, 5.0]])
    rows = controlled_candidate_rows(scores, gt_points, threshold=20, width=2, ks=[1], patch_stride=10)
    assert rows[0]["other_source_candidate_hit@1"] == 1
    assert rows[0]["strict_other_source_candidate_hit@1"] == 0

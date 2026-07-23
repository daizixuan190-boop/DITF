import torch
from PIL import Image

from dino_v2_spair import (
    CategoryMetrics,
    candidate_hit,
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
    assert metrics.per_image == 2 / 3
    assert metrics.per_point == 0.5


def test_candidate_hit_uses_patch_centers_in_canvas_units():
    candidates = torch.tensor([[3]])
    # Flat index 3 in a 2-wide grid is centered at (21, 21) for stride 14.
    assert candidate_hit(candidates, [21, 21], width=2, threshold=10, patch_stride=14).item()

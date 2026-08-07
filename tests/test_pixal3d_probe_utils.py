import numpy as np
import pytest
from PIL import Image

from pixal3d_probe_utils import (
    CropTransform,
    face_rows_from_triangle_ids,
    validate_surface_buffers,
)
from probe_pixal3d_surface_interface import _instrumented_preprocess


def test_crop_transform_round_trip_includes_resize_crop_and_render_scale():
    transform = CropTransform(
        original_size=(800, 600),
        resized_size=(400, 300),
        crop_box=(50, 25, 250, 225),
        render_size=(1024, 1024),
    )
    original = np.array([[100.0, 50.0], [600.0, 400.0]])

    rendered = transform.original_to_render(original)
    restored = transform.render_to_original(rendered)

    np.testing.assert_allclose(restored, original, atol=1e-6)
    np.testing.assert_allclose(rendered[0], [0.0, 0.0], atol=1e-6)


def test_crop_transform_rejects_degenerate_dimensions():
    with pytest.raises(ValueError, match="positive"):
        CropTransform(
            original_size=(800, 600),
            resized_size=(400, 300),
            crop_box=(10, 10, 10, 30),
            render_size=(1024, 1024),
        )


def test_surface_buffers_require_exact_foreground_triangle_consistency():
    triangle_ids = np.array([[0, 1], [2, 0]], dtype=np.int64)
    mask = triangle_ids > 0
    depth = np.array([[0.0, 1.0], [2.0, 0.0]], dtype=np.float32)
    coords = np.zeros((2, 2, 3), dtype=np.float32)
    coords[mask] = 1.0

    summary = validate_surface_buffers(triangle_ids, mask, depth, coords, num_faces=2)

    assert summary["foreground_pixels"] == 2
    assert summary["max_triangle_id"] == 2


def test_surface_buffers_reject_triangle_ids_beyond_face_count():
    triangle_ids = np.array([[3]], dtype=np.int64)
    with pytest.raises(ValueError, match="face count"):
        validate_surface_buffers(
            triangle_ids,
            np.array([[True]]),
            np.array([[1.0]], dtype=np.float32),
            np.ones((1, 1, 3), dtype=np.float32),
            num_faces=2,
        )


def test_triangle_ids_use_one_based_foreground_and_zero_background():
    features = np.arange(12, dtype=np.float32).reshape(3, 4)
    triangle_ids = np.array([1, 3, 0, 2], dtype=np.int64)

    rows, valid = face_rows_from_triangle_ids(triangle_ids, features)

    assert valid.tolist() == [True, True, False, True]
    np.testing.assert_array_equal(rows[0], features[0])
    np.testing.assert_array_equal(rows[1], features[2])
    np.testing.assert_array_equal(rows[3], features[1])
    assert np.isnan(rows[2]).all()


def test_instrumented_preprocess_records_official_alpha_crop_formula():
    rgba = np.zeros((80, 100, 4), dtype=np.uint8)
    rgba[10:50, 20:60, :3] = [120, 80, 40]
    rgba[10:50, 20:60, 3] = 255
    capture = {}
    pipeline = type("Pipeline", (), {"low_vram": False})()

    output = _instrumented_preprocess(pipeline, Image.fromarray(rgba, "RGBA"), (0, 0, 0), capture)

    assert capture["original_size"] == (100, 80)
    assert capture["resized_size"] == (100, 80)
    assert capture["raw_foreground_bbox"] == (20, 10, 59, 49)
    assert capture["crop_box"] == (18, 8, 60, 50)
    assert output.size == (42, 42)

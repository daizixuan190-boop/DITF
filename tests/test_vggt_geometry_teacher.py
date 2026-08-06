import torch

from vggt_geometry_teacher import crop_vggt_geometry_to_original


def test_crop_vggt_geometry_removes_square_padding():
    point_map = torch.arange(6 * 6 * 3, dtype=torch.float32).reshape(6, 6, 3)
    cropped = crop_vggt_geometry_to_original(
        point_map,
        torch.tensor([1.0, 2.0, 5.0, 4.0, 8.0, 4.0]),
    )
    assert cropped.shape == (2, 4, 3)
    assert torch.equal(cropped, point_map[2:4, 1:5])


def test_crop_vggt_geometry_clamps_fractional_coordinates():
    point_map = torch.zeros(5, 7, 3)
    cropped = crop_vggt_geometry_to_original(
        point_map,
        torch.tensor([-0.2, 0.4, 7.3, 5.8, 7.0, 5.0]),
    )
    assert cropped.shape == (5, 7, 3)

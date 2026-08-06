import torch

from exact_warp_coordinate_identity import build_exact_forward_coordinate_maps
from spair_matchers import cosine_nn_predict


def test_exact_coordinate_cosine_reproduces_continuous_warp_nearest_pixel():
    warp = torch.tensor(
        [
            [[0.12, 0.25], [0.76, 0.51]],
            [[0.34, 0.90], [1.00, 0.00]],
        ],
        dtype=torch.float32,
    )
    reliability = torch.ones((2, 2), dtype=torch.float32)
    maps = build_exact_forward_coordinate_maps(
        warp,
        reliability,
        reliability,
        source_size=(2, 2),
        target_size=(5, 7),
    )

    predictions = cosine_nn_predict(
        maps["source_unit"], maps["target_unit"], [[0, 0], [1, 0], [0, 1], [1, 1]]
    )
    expected = [
        [round(0.12 * 6), round(0.25 * 4)],
        [round(0.76 * 6), round(0.51 * 4)],
        [round(0.34 * 6), round(0.90 * 4)],
        [6, 0],
    ]
    assert predictions == expected


def test_exact_coordinate_embeddings_have_unit_norm_and_exact_cross_dot():
    warp = torch.tensor([[[0.2, 0.7]]], dtype=torch.float32)
    maps = build_exact_forward_coordinate_maps(
        warp,
        torch.ones((1, 1)),
        torch.ones((1, 2)),
        source_size=(1, 1),
        target_size=(1, 2),
    )
    source = maps["source_unit"][0, :, 0, 0]
    targets = maps["target_unit"][0, :, 0].t()
    target_coordinates = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    expected = -(target_coordinates - warp[0, 0]).square().sum(dim=1) / 9.0

    assert torch.allclose(source.norm(), torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(targets.norm(dim=1), torch.ones(2), atol=1e-6)
    assert torch.allclose(targets @ source, expected, atol=1e-6)

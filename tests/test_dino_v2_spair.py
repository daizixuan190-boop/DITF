import torch

from dino_v2_spair import candidate_hit, dino_tokens_to_map, resize_shape, summarize_candidate_rows


def test_resize_is_aspect_preserving_and_patch_aligned():
    height, width = resize_shape(300, 500, 840)
    assert width == 840
    assert height % 14 == 0 and width % 14 == 0


def test_tokens_to_map_drops_cls_token():
    tokens = torch.arange(1 * 5 * 2, dtype=torch.float32).reshape(1, 5, 2)
    feature_map = dino_tokens_to_map(tokens, 28, 28, patch_size=14)
    assert feature_map.shape == (1, 2, 2, 2)
    assert torch.equal(feature_map.flatten(2).transpose(1, 2)[0], tokens[:, 1:, :])


def test_global_union_detects_transfer_when_owner_misses():
    # Two source queries; the GT for query zero is present only in query one.
    candidates = torch.tensor([[8, 9], [0, 1]], dtype=torch.long)
    rows = summarize_candidate_rows(candidates, [[0, 0], [9, 0]], 10, 10, [1, 2])
    assert rows[0]["owner_candidate_hit@2"] == 0
    assert rows[0]["other_source_candidate_hit@2"] == 1
    assert rows[0]["global_union_candidate_hit@2"] == 1


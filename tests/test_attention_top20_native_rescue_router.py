from argparse import Namespace

import pytest
import torch

from ephemeral_category_cache import (
    category_cache_snapshot,
    delete_new_category_cache_files,
)
from eval_spair_attention_top20_expert_rescue_router import (
    _apply_locked_profile,
    select_expert,
)
from spair_matchers import (
    cosine_candidate_diagnostics,
    cosine_nn_predict_with_diagnostics,
)


def _unique_grid_features(height: int = 2, width: int = 2) -> torch.Tensor:
    channels = height * width
    features = torch.zeros(1, channels, height, width)
    for index in range(channels):
        y, x = divmod(index, width)
        features[0, index, y, x] = 1.0
    return features


def test_native_nn_diagnostics_preserve_argmax_and_cycle():
    features = _unique_grid_features()

    predictions, diagnostics = cosine_nn_predict_with_diagnostics(
        features,
        features.clone(),
        [[0, 0], [1, 1]],
        nonlocal_radius=0,
    )

    assert predictions == [[0, 0], [1, 1]]
    assert diagnostics[0]["top1_top2_margin"] == 1.0
    assert diagnostics[0]["top1_nonlocal_margin"] == 1.0
    assert diagnostics[0]["cycle_source_distance"] == 0.0
    assert diagnostics[0]["reciprocal_exact"] is True


def test_native_candidate_diagnostics_rank_fixed_pixels():
    features = _unique_grid_features()

    rows = cosine_candidate_diagnostics(
        features,
        features.clone(),
        [[0, 0]],
        [[[1, 1], [0, 0], [1, 0]]],
    )

    assert [row["pixel"] for row in rows[0]] == [[1, 1], [0, 0], [1, 0]]
    assert rows[0][1]["native_cosine"] == 1.0
    assert rows[0][1]["native_candidate_rank"] == 1
    assert rows[0][0]["native_gap_to_candidate_top1"] == 1.0


def _router_args(**overrides):
    values = {
        "preferred_expert": "roma_pairwise",
        "roma_bidir_max": 0.05,
        "roma_mutual_min": 0.0,
        "max_selected_attention_rank": 20,
        "baseline_attention_min_distance": 0.0,
        "dino_roma_agreement_px": -1.0,
        "native_margin_max": 0.02,
        "native_top1_cosine_max": -1.0,
        "native_nonlocal_margin_max": -1.0,
        "native_cycle_distance_min": -1.0,
        "require_native_nonreciprocal": False,
    }
    values.update(overrides)
    return Namespace(**values)


def _dino_point(native_margin: float):
    return {
        "baseline_prediction": [0, 0],
        "method_prediction": [4, 4],
        "native_nn_diagnostics": {
            "top1_cosine": 0.5,
            "top1_top2_margin": native_margin,
            "top1_nonlocal_margin": native_margin,
            "cycle_source_distance": 2.0,
            "reciprocal_exact": False,
        },
        "candidates": [
            {"pixel": [4, 4], "attention_rank": 1, "dino_cosine": 0.8}
        ],
    }


def _roma_point():
    return {
        "method_prediction": [4, 4],
        "candidates": [
            {
                "pixel": [4, 4],
                "attention_rank": 1,
                "bidirectional_error": 0.01,
                "mutual_certainty": 0.1,
            }
        ],
    }


def test_router_requires_native_uncertainty_before_override():
    roma = _roma_point()

    selected, _, _ = select_expert(
        _dino_point(native_margin=0.01),
        roma,
        None,
        _router_args(),
    )
    rejected, _, _ = select_expert(
        _dino_point(native_margin=0.20),
        roma,
        None,
        _router_args(),
    )

    assert selected == "roma_pairwise"
    assert rejected == "baseline"


def test_candidate_rank_consensus_aligns_pixels_not_list_positions():
    dino = _dino_point(native_margin=0.01)
    dino["candidates"] = [
        {
            "pixel": [4, 4],
            "attention_rank": 1,
            "native_candidate_rank": 2,
            "dino_rank": 2,
            "pck_hit": False,
        },
        {
            "pixel": [8, 8],
            "attention_rank": 2,
            "native_candidate_rank": 1,
            "dino_rank": 1,
            "pck_hit": True,
        },
    ]
    roma = {
        "method_prediction": [4, 4],
        "candidates": [
            {
                "pixel": [8, 8],
                "attention_rank": 2,
                "roma_rank": 1,
                "bidirectional_error": 0.02,
                "mutual_certainty": 0.2,
                "pck_hit": True,
            },
            {
                "pixel": [4, 4],
                "attention_rank": 1,
                "roma_rank": 2,
                "bidirectional_error": 0.01,
                "mutual_certainty": 0.1,
                "pck_hit": False,
            },
        ],
    }

    selected, point, signals = select_expert(
        dino,
        roma,
        None,
        _router_args(
            preferred_expert="candidate_rank_consensus",
            roma_bidir_max=-1.0,
            roma_mutual_min=-1.0,
            native_margin_max=-1.0,
        ),
    )

    assert selected == "candidate_rank_consensus"
    assert point["method_prediction"] == [8, 8]
    assert signals["selected_attention_rank"] == 2


def test_locked_profile_overrides_router_parameters_and_is_hashed():
    args = _router_args()
    args.locked_profile = "discovery20_seed2027_rank_consensus_v1"

    lock = _apply_locked_profile(args)

    assert args.preferred_expert == "candidate_rank_consensus"
    assert args.native_top1_cosine_max == 0.8
    assert args.native_cycle_distance_min == 16.0
    assert lock is not None
    assert len(lock["sha256"]) == 64


def test_ephemeral_cache_cleanup_preserves_preexisting_files(tmp_path):
    category_root = tmp_path / "cat"
    category_root.mkdir()
    existing = category_root / "existing.pth"
    existing.write_bytes(b"existing")
    before = category_cache_snapshot(str(tmp_path), "cat")
    created = category_root / "created.pth"
    created.write_bytes(b"created")
    unrelated = category_root / "notes.txt"
    unrelated.write_text("keep", encoding="utf-8")

    deleted = delete_new_category_cache_files(str(tmp_path), "cat", before)

    assert deleted == 1
    assert existing.is_file()
    assert not created.exists()
    assert unrelated.is_file()


def test_ephemeral_cache_snapshot_rejects_path_escape(tmp_path):
    with pytest.raises(ValueError, match="escapes cache root"):
        category_cache_snapshot(str(tmp_path), "../outside")

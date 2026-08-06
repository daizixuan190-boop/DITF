from argparse import Namespace

import pytest

from eval_spair_supervised_attention_candidate_identity import (
    candidate_kind,
    summarize_supervised_records,
    validate_supervised_checkpoint_metadata,
)
from train_flux_attention_candidate_identity_supervised import build_parser as build_training_parser


def _metadata():
    return {
        "supervision": "spair_train_keypoints",
        "spair_keypoints_used": True,
        "test_keypoints_used_for_training": False,
        "category_labels_used_for_targets": False,
        "keypoint_ids_used": False,
        "external_matcher_used": False,
        "dino_used": False,
        "roma_used": False,
        "protocol": {
            "feature_block": 28,
            "timestep": 260,
            "ensemble_size": 8,
            "candidate_topk": 20,
            "extra_candidate_count": 1,
            "image_size": [640, 640],
            "channel_discard": True,
        },
    }


def _args():
    return Namespace(
        k=[28],
        t=260,
        ensemble_size=8,
        candidate_topk=20,
        img_size=[640, 640],
        cd=True,
    )


def test_supervised_checkpoint_protocol_is_explicit_and_rejects_hidden_teachers():
    validate_supervised_checkpoint_metadata({"training_metadata": _metadata()}, _args())
    for forbidden in (
        "test_keypoints_used_for_training",
        "category_labels_used_for_targets",
        "keypoint_ids_used",
        "external_matcher_used",
        "dino_used",
        "roma_used",
    ):
        metadata = _metadata()
        metadata[forbidden] = True
        with pytest.raises(ValueError, match=forbidden):
            validate_supervised_checkpoint_metadata(
                {"training_metadata": metadata},
                _args(),
            )


def test_supervised_checkpoint_protocol_rejects_candidate_pool_mismatch():
    args = _args()
    args.candidate_topk = 19
    with pytest.raises(ValueError, match="protocol mismatch"):
        validate_supervised_checkpoint_metadata({"training_metadata": _metadata()}, args)


def test_supervised_training_uses_one_neutral_caption_not_caption_json():
    args = build_training_parser().parse_args([
        "--dataset_path", "/dataset",
        "--output_checkpoint", "decoder.pt",
    ])
    assert args.training_caption == "a photo"
    assert not hasattr(args, "captions_json")


def test_candidate_kind_maps_only_appended_rank_to_baseline_fallback():
    assert candidate_kind(0, attention_candidate_count=20) == "attention"
    assert candidate_kind(19, attention_candidate_count=20) == "attention"
    assert candidate_kind(20, attention_candidate_count=20) == "baseline_fallback"
    with pytest.raises(ValueError):
        candidate_kind(21, attention_candidate_count=20)


def test_supervised_summary_preserves_baseline_and_measures_pool_oracle_gap():
    records = [{
        "category": "cat",
        "pair_name": "pair.json",
        "points": [
            {
                "selected_kind": "attention",
                "pck_hits": {
                    "baseline": True,
                    "attention_top1": False,
                    "resolver": True,
                    "attention_top20_oracle": True,
                    "resolver_pool_oracle": True,
                },
            },
            {
                "selected_kind": "baseline_fallback",
                "pck_hits": {
                    "baseline": True,
                    "attention_top1": True,
                    "resolver": False,
                    "attention_top20_oracle": False,
                    "resolver_pool_oracle": True,
                },
            },
            {
                "selected_kind": "attention",
                "pck_hits": {
                    "baseline": False,
                    "attention_top1": False,
                    "resolver": True,
                    "attention_top20_oracle": True,
                    "resolver_pool_oracle": True,
                },
            },
        ],
    }]
    summary = summarize_supervised_records(records)
    comparison = summary["resolver_vs_baseline"]
    assert summary["metrics"]["baseline"]["correct"] == 2
    assert summary["metrics"]["resolver"]["correct"] == 2
    assert summary["metrics"]["resolver_pool_oracle"]["correct"] == 3
    assert comparison["rescued"] == 1
    assert comparison["harmed"] == 1
    assert comparison["net_correct"] == 0
    assert comparison["baseline_correct_retention_rate"] == 0.5
    assert comparison["pool_oracle_gap_points"] == 1
    assert comparison["pool_oracle_gap_recovered_fraction"] == 0.0
    assert summary["selection"]["baseline_fallback_rate"] == pytest.approx(1 / 3)

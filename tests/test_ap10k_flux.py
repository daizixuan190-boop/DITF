import json
from pathlib import Path

import torch

from eval_ap10k_flux import (
    _merge_ownership_counts,
    _plain_ownership_counts,
    _restore_ownership_counts,
    build_parser,
)
from ap10k_flux import (
    discover_pairs,
    empty_metric_counts,
    metric_ratios,
    pair_hits,
    pair_group,
    transform_keypoints,
    update_metric_counts,
)
from ownership_diagnostics import empty_counts as empty_ownership_counts


def test_pair_groups_match_three_ap10k_settings():
    cat_a = "ImageAnnotation/felidae/cat/000000000001.json"
    cat_b = "ImageAnnotation/felidae/lion/000000000002.json"
    dog = "ImageAnnotation/canidae/dog/000000000003.json"
    assert pair_group("intra-species", cat_a, cat_a) == "felidae/cat"
    assert pair_group("cross-species", cat_a, cat_b) == "felidae"
    assert pair_group("cross-family", cat_a, dog) == "felidae|canidae"


def test_transform_keypoints_matches_square_padding_geometry():
    keypoints = [20, 10, 2] + [0, 0, 0] * 16
    points, visibility, scale = transform_keypoints(keypoints, width=100, height=50, size=200)
    assert scale == 2
    assert torch.equal(points[0], torch.tensor([40.0, 70.0]))
    assert visibility[0] == 1


def test_transform_keypoints_reproduces_visibility_multiplication():
    keypoints = [20, 10, 1] + [0, 0, 0] * 16
    points, visibility, _ = transform_keypoints(keypoints, width=100, height=50, size=200)
    assert torch.equal(points[0], torch.tensor([20.0, 35.0]))
    assert visibility[0] == 0.25


def test_per_image_and_per_point_pck_are_kept_distinct():
    counts = empty_metric_counts()
    update_metric_counts(
        counts,
        pair_hits(torch.tensor([[0.0, 0.0]]), torch.tensor([[0.0, 0.0]]), 10),
    )
    update_metric_counts(
        counts,
        pair_hits(
            torch.tensor([[10.0, 10.0], [10.0, 10.0], [10.0, 10.0]]),
            torch.zeros(3, 2),
            10,
        ),
    )
    ratios = metric_ratios(counts)
    assert ratios["per_image_pck"]["0.10"] == 0.5
    assert ratios["per_point_pck"]["0.10"] == 0.25


def test_discover_pairs_applies_group_limits(tmp_path: Path):
    pair_dir = tmp_path / "PairAnnotation" / "test"
    pair_dir.mkdir(parents=True)
    for index in range(3):
        path = f"ImageAnnotation/felidae/cat/{index:012d}.json"
        (pair_dir / f"{index}.json").write_text(
            json.dumps({"src_json_path": path, "trg_json_path": path}), encoding="utf-8"
        )
    groups = discover_pairs(tmp_path, "intra-species", max_pairs_per_group=2)
    assert list(groups) == ["felidae/cat"]
    assert len(groups["felidae/cat"]) == 2


def test_discover_pairs_hash_sample_is_deterministic(tmp_path: Path):
    pair_dir = tmp_path / "PairAnnotation" / "test"
    pair_dir.mkdir(parents=True)
    for index in range(20):
        path = f"ImageAnnotation/felidae/cat/{index:012d}.json"
        (pair_dir / f"{index:02d}.json").write_text(
            json.dumps({"src_json_path": path, "trg_json_path": path}), encoding="utf-8"
        )

    first = discover_pairs(
        tmp_path, "intra-species", max_pairs_per_group=5, pair_sample_seed=2027
    )
    repeated = discover_pairs(
        tmp_path, "intra-species", max_pairs_per_group=5, pair_sample_seed=2027
    )
    different = discover_pairs(
        tmp_path, "intra-species", max_pairs_per_group=5, pair_sample_seed=2028
    )

    first_names = [pair.path.name for pair in first["felidae/cat"]]
    assert first_names == [pair.path.name for pair in repeated["felidae/cat"]]
    assert first_names != [pair.path.name for pair in different["felidae/cat"]]


def test_ownership_counts_survive_json_checkpoint_round_trip():
    ks = (1, 5)
    original = empty_ownership_counts(ks)
    original[1]["points"] = 7
    original[5]["failure_strict_global_not_budget"] = 3

    payload = json.loads(json.dumps(_plain_ownership_counts(original)))
    restored = _restore_ownership_counts(payload, ks)
    merged = empty_ownership_counts(ks)
    _merge_ownership_counts(merged, restored)

    assert merged[1]["points"] == 7
    assert merged[5]["failure_strict_global_not_budget"] == 3


def test_joint_calibration_defaults_enable_controlled_grid_diagnostics():
    args = build_parser().parse_args(
        [
            "--dataset_path", "benchmark",
            "--setting", "intra-species",
            "--output_json", "summary.json",
        ]
    )
    assert args.diagnostics is True
    assert args.diagnostic_grid == 60
    assert args.topk == [1, 5, 10, 20, 50]
    assert args.pair_sample_seed == 2027

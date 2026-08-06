from argparse import Namespace
import json

import torch

from attention_identity_verifier import (
    CandidateIdentityVerifier,
    VerifierConfig,
    geometry_fgw_pseudo_targets,
    checkpoint_payload,
    horizontal_flip_points,
    listwise_identity_loss,
    load_verifier_checkpoint,
    native_cycle_pseudo_targets,
    sample_replay_cell_centers,
    select_candidate_pixels,
    triangle_cycle_pseudo_targets,
    transformed_candidate_targets,
    weighted_listwise_identity_loss,
)
from eval_spair_attention_identity_verifier import (
    _validate_checkpoint_metadata,
    summarize_native_cycle_diagnostics,
    summarize_verifier_records,
)
from train_flux_attention_identity_verifier import (
    build_parser as build_training_parser,
    build_strict_training_manifest,
    build_strict_training_pair_manifest,
)
from train_flux_attention_identity_verifier_triangle_cycle import (
    build_strict_training_triplet_manifest,
)
from deformation_views import ElasticDeformationPlan, make_view_transform, sample_appearance_plan


def _model() -> CandidateIdentityVerifier:
    return CandidateIdentityVerifier(
        VerifierConfig(
            feature_dims={"qk_expert": 4, "token_state": 3},
            feature_groups=("qk_expert", "token_state"),
            group_width=4,
            hidden_width=8,
            dropout=0.0,
        )
    )


def test_global_query_context_changes_with_other_source_queries():
    config = VerifierConfig(
        feature_dims={"geometry_control": 2},
        feature_groups=("geometry_control",),
        group_width=4,
        hidden_width=8,
        dropout=0.0,
        global_query_context=True,
    )
    model = CandidateIdentityVerifier(config).eval()
    with torch.no_grad():
        model.residual_head.weight.fill_(0.1)
    groups = {"geometry_control": torch.tensor([
        [[1.0, 0.0], [0.0, 1.0]],
        [[0.5, 0.5], [0.2, 0.8]],
    ])}
    attention = torch.ones(2, 2)
    with torch.no_grad():
        first = model(groups, attention)
        changed = {"geometry_control": groups["geometry_control"].clone()}
        changed["geometry_control"][1, 1] += 10.0
        second = model(changed, attention)
    assert not torch.allclose(first[0], second[0])


def test_global_context_is_equivariant_to_query_and_candidate_permutations():
    config = VerifierConfig(
        feature_dims={"geometry_control": 2},
        feature_groups=("geometry_control",),
        group_width=4,
        hidden_width=8,
        dropout=0.0,
        global_query_context=True,
    )
    model = CandidateIdentityVerifier(config).eval()
    groups = {"geometry_control": torch.randn(3, 4, 2)}
    attention = torch.rand(3, 4).add(0.1)
    query_permutation = torch.tensor([2, 0, 1])
    candidate_permutations = torch.tensor([3, 1, 0, 2])
    original = model(groups, attention)
    permuted = model(
        {name: value[query_permutation][:, candidate_permutations] for name, value in groups.items()},
        attention[query_permutation][:, candidate_permutations],
    )
    expected = original[query_permutation][:, candidate_permutations]
    assert torch.allclose(permuted, expected, atol=1e-6)


def test_elastic_deformation_is_dimension_preserving_and_reversible():
    from PIL import Image
    import random

    plan = ElasticDeformationPlan(
        amplitude_x=0.03,
        amplitude_y=0.04,
        cycles_x=1.1,
        cycles_y=1.4,
        phase_x=0.2,
        phase_y=1.1,
    )
    image = Image.new("RGB", (64, 48), color=(120, 80, 40))
    transformed = make_view_transform(
        sample_appearance_plan(random.Random(3)), random.Random(4), plan
    )(image)
    assert transformed.size == image.size
    points = torch.tensor([[20.0, 15.0], [40.0, 30.0]])
    target = plan.source_to_target(points, 48, 64)
    restored = plan.target_to_source(target, 48, 64)
    assert torch.allclose(restored, points, atol=0.2)


def test_verifier_zero_residual_starts_from_attention_order():
    model = _model().eval()
    groups = {
        "qk_expert": torch.randn(2, 3, 4),
        "token_state": torch.randn(2, 3, 3),
    }
    attention = torch.tensor([[0.2, 0.7, 0.1], [0.6, 0.3, 0.1]])
    scores = model(groups, attention)
    assert scores.argmax(dim=1).tolist() == [1, 0]


def test_verifier_is_equivariant_to_candidate_permutation():
    model = _model().eval()
    groups = {
        "qk_expert": torch.randn(2, 4, 4),
        "token_state": torch.randn(2, 4, 3),
    }
    attention = torch.rand(2, 4).add(0.1)
    permutation = torch.tensor([2, 0, 3, 1])
    original = model(groups, attention)
    permuted = model(
        {name: value[:, permutation] for name, value in groups.items()},
        attention[:, permutation],
    )
    assert torch.allclose(permuted, original[:, permutation], atol=1e-6)


def test_transformed_targets_select_nearest_recoverable_candidate():
    candidate_pixels = torch.tensor([[0, 11, 22], [5, 15, 25]])
    target_points = torch.tensor([[1.0, 1.0], [9.0, 9.0]])
    targets, recoverable, minimum = transformed_candidate_targets(
        candidate_pixels,
        target_points,
        [10, 10],
        sigma_pixels=2.0,
        max_distance_pixels=3.0,
    )
    assert targets[0].argmax().item() == 1
    assert recoverable.tolist() == [True, False]
    assert minimum[0].item() == 0.0


def test_listwise_loss_ignores_unrecoverable_queries():
    scores = torch.tensor([[0.0, 3.0], [10.0, -10.0]], requires_grad=True)
    targets = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    loss = listwise_identity_loss(scores, targets, torch.tensor([True, False]))
    loss.backward()
    assert loss.item() < 0.1
    assert scores.grad[1].abs().sum().item() == 0.0


def test_weighted_listwise_loss_uses_continuous_query_confidence():
    scores = torch.tensor([[0.0, 3.0], [3.0, 0.0]], requires_grad=True)
    targets = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    loss = weighted_listwise_identity_loss(
        scores,
        targets,
        torch.tensor([1.0, 0.0]),
    )
    loss.backward()
    assert loss.item() < 0.1
    assert scores.grad[1].abs().sum().item() == 0.0


def test_geometry_fgw_teacher_resolves_symmetric_unary_with_3d_anchors():
    source_features = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [-1.0, 0.0],
    ]).t().reshape(1, 2, 1, 4)
    target_features = torch.tensor([
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ]).t().reshape(1, 2, 1, 4)
    source_geometry = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
    ]).reshape(1, 1, 4, 3)
    target_geometry = torch.tensor([
        [0.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ]).reshape(1, 1, 4, 3)
    batch = {
        "source_cells": torch.tensor([0, 1, 2, 3]),
        "candidate_cells": torch.tensor([
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
        ]),
        "attention_scores": torch.ones(4, 4),
    }
    targets, weights, diagnostics = geometry_fgw_pseudo_targets(
        batch,
        source_features,
        target_features,
        source_geometry,
        target_geometry,
        alpha=1.0,
        refinement_steps=2,
        sinkhorn_iterations=20,
        max_anchors=2,
        minimum_anchors=2,
    )
    assert targets.argmax(dim=1).tolist() == [0, 3, 2, 1]
    assert bool((weights > 0).all())
    assert diagnostics["anchor_count"] >= 2


def test_geometry_fgw_teacher_is_invariant_to_independent_scale_and_translation():
    source_features = torch.eye(3).t().reshape(1, 3, 1, 3)
    target_features = source_features.clone()
    source_geometry = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
    ]).reshape(1, 1, 3, 3)
    target_geometry = source_geometry * 7.0 + 11.0
    batch = {
        "source_cells": torch.tensor([0, 1, 2]),
        "candidate_cells": torch.tensor([[0, 1, 2], [0, 1, 2], [0, 1, 2]]),
        "attention_scores": torch.ones(3, 3),
    }
    original, _, _ = geometry_fgw_pseudo_targets(
        batch,
        source_features,
        target_features,
        source_geometry,
        source_geometry,
        max_anchors=3,
        minimum_anchors=2,
    )
    transformed, _, _ = geometry_fgw_pseudo_targets(
        batch,
        source_features,
        target_features,
        source_geometry,
        target_geometry,
        max_anchors=3,
        minimum_anchors=2,
    )
    assert torch.allclose(original, transformed, atol=1e-5)


def test_geometry_fgw_teacher_falls_back_without_enough_anchors_and_stays_in_pool():
    source_features = torch.eye(2).t().reshape(1, 2, 1, 2)
    target_features = source_features.clone()
    geometry = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ]).reshape(1, 1, 2, 3)
    batch = {
        "source_cells": torch.tensor([0]),
        "candidate_cells": torch.tensor([[1]]),
        "attention_scores": torch.ones(1, 1),
    }
    targets, weights, diagnostics = geometry_fgw_pseudo_targets(
        batch,
        source_features,
        target_features,
        geometry,
        geometry,
        minimum_anchors=2,
    )
    assert targets.shape == (1, 1)
    assert targets.item() == 1.0
    assert weights.item() == 0.0
    assert diagnostics["used_geometry"] is False


def test_native_cycle_teacher_filters_by_reverse_cycle_and_margin():
    source = torch.eye(4).t().reshape(1, 4, 1, 4)
    target = source.clone()
    batch = {
        "source_cells": torch.tensor([0, 1], dtype=torch.int32),
        "candidate_cells": torch.tensor([[0, 2], [2, 3]], dtype=torch.int32),
        "feature_groups": {
            "native_control": torch.tensor([
                [[0.9], [0.2]],
                [[0.8], [0.1]],
            ])
        },
    }
    targets, confident, diagnostics = native_cycle_pseudo_targets(
        batch,
        source,
        target,
        cycle_radius_cells=0.0,
        minimum_native_margin=0.5,
    )
    assert targets.argmax(dim=1).tolist() == [0, 0]
    assert confident.tolist() == [True, False]
    assert diagnostics["cycle_distance_cells"].tolist() == [0.0, 1.0]


def test_triangle_cycle_teacher_requires_a_unique_return_to_source():
    batch = {
        "source_cells": torch.tensor([0, 1], dtype=torch.int32),
        "candidate_cells": torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
    }
    # B -> C is the identity. C -> A returns candidate 0 exactly to source 0,
    # while the two candidates for source 1 have an equal one-cell cycle error.
    mutual_bc = torch.eye(4)
    mutual_ca = torch.zeros(4, 4)
    mutual_ca[0, 0] = 1.0
    mutual_ca[1, 3] = 1.0
    mutual_ca[2, 0] = 1.0
    mutual_ca[3, 2] = 1.0
    targets, confident, diagnostics = triangle_cycle_pseudo_targets(
        batch,
        mutual_bc,
        mutual_ca,
        source_grid_size=(1, 4),
        cycle_radius_cells=1.0,
        require_unique_best=True,
    )
    assert targets.argmax(dim=1).tolist() == [0, 0]
    assert confident.tolist() == [True, False]
    assert diagnostics["best_cycle_distance_cells"].tolist() == [0.0, 1.0]
    assert diagnostics["unique_best"].tolist() == [True, False]


def test_flip_grid_and_candidate_selection_are_pixel_aligned():
    generator = torch.Generator().manual_seed(7)
    points = sample_replay_cell_centers(
        [64, 80],
        [4, 5],
        count=6,
        generator=generator,
        border_cells=0,
    )
    flipped = horizontal_flip_points(points, 80)
    assert torch.allclose(points[:, 0] + flipped[:, 0], torch.full((6,), 79.0))
    scores = torch.tensor([[0.0, 2.0], [3.0, 1.0]])
    pixels = torch.tensor([[4, 9], [12, 18]])
    assert select_candidate_pixels(scores, pixels).tolist() == [9, 12]


def test_strict_training_manifest_excludes_test_images_and_reads_names_only(tmp_path):
    for category in ("cat", "dog"):
        image_dir = tmp_path / "JPEGImages" / category
        image_dir.mkdir(parents=True)
        for name in (f"{category}_train_a.jpg", f"{category}_train_b.jpg", f"{category}_test.jpg"):
            (image_dir / name).write_bytes(b"image")
    train_dir = tmp_path / "PairAnnotation" / "trn"
    test_dir = tmp_path / "PairAnnotation" / "test"
    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    (train_dir / "cat-pair.json").write_text(json.dumps({
        "src_imname": "cat_train_a.jpg",
        "trg_imname": "cat_train_b.jpg",
        "src_kps": [[999, 999]],
        "trg_kps": [[999, 999]],
    }))
    (train_dir / "dog-overlap.json").write_text(json.dumps({
        "src_imname": "dog_train_a.jpg",
        "trg_imname": "dog_test.jpg",
    }))
    (test_dir / "dog-test.json").write_text(json.dumps({
        "src_imname": "dog_test.jpg",
        "trg_imname": "dog_train_b.jpg",
    }))

    manifest, metadata = build_strict_training_manifest(
        tmp_path,
        seed=2027,
        max_images=0,
    )
    keys = {(row.category, row.image_name) for row in manifest}
    assert keys == {
        ("cat", "cat_train_a.jpg"),
        ("cat", "cat_train_b.jpg"),
        ("dog", "dog_train_a.jpg"),
    }
    assert metadata["annotation_fields_read"] == ["src_imname", "trg_imname"]
    assert metadata["train_occurrences_rejected_for_test_overlap"] == 1


def test_training_manifest_uses_pair_filename_for_multicategory_images(tmp_path):
    for category in ("horse", "person"):
        image_dir = tmp_path / "JPEGImages" / category
        image_dir.mkdir(parents=True)
        for name in ("shared_a.jpg", "shared_b.jpg", "test_a.jpg", "test_b.jpg"):
            (image_dir / name).write_bytes(b"image")
    train_dir = tmp_path / "PairAnnotation" / "trn"
    test_dir = tmp_path / "PairAnnotation" / "test"
    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    (train_dir / "horse-pair.json").write_text(json.dumps({
        "src_imname": "shared_a.jpg",
        "trg_imname": "shared_b.jpg",
    }))
    (test_dir / "person-pair.json").write_text(json.dumps({
        "src_imname": "test_a.jpg",
        "trg_imname": "test_b.jpg",
    }))

    manifest, _metadata = build_strict_training_manifest(
        tmp_path,
        seed=2027,
        max_images=0,
    )
    assert {(row.category, row.image_name) for row in manifest} == {
        ("horse", "shared_a.jpg"),
        ("horse", "shared_b.jpg"),
    }


def test_training_uses_one_neutral_caption_by_default():
    args = build_training_parser().parse_args([
        "--dataset_path",
        "/dataset",
        "--output_checkpoint",
        "/tmp/model.pt",
    ])
    assert args.training_caption == "a photo"


def test_strict_pair_manifest_keeps_pairing_but_not_keypoint_fields(tmp_path):
    image_dir = tmp_path / "JPEGImages" / "cat"
    image_dir.mkdir(parents=True)
    for name in ("a.jpg", "b.jpg", "test_a.jpg", "test_b.jpg"):
        (image_dir / name).write_bytes(b"image")
    train_dir = tmp_path / "PairAnnotation" / "trn"
    test_dir = tmp_path / "PairAnnotation" / "test"
    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    (train_dir / "cat-train.json").write_text(json.dumps({
        "src_imname": "a.jpg",
        "trg_imname": "b.jpg",
        "src_kps": [[10, 20]],
        "trg_kps": [[30, 40]],
    }))
    (test_dir / "cat-test.json").write_text(json.dumps({
        "src_imname": "test_a.jpg",
        "trg_imname": "test_b.jpg",
    }))
    pairs, metadata = build_strict_training_pair_manifest(
        tmp_path,
        seed=2027,
        max_pairs=0,
    )
    assert [(row.category, row.source_name, row.target_name) for row in pairs] == [
        ("cat", "a.jpg", "b.jpg")
    ]
    assert metadata["pair_membership_used"] is True
    assert metadata["keypoint_fields_read"] is False


def test_strict_triplet_manifest_builds_unlabeled_two_edge_paths(tmp_path):
    image_dir = tmp_path / "JPEGImages" / "cat"
    image_dir.mkdir(parents=True)
    for name in ("a.jpg", "b.jpg", "c.jpg", "test_a.jpg", "test_b.jpg"):
        (image_dir / name).write_bytes(b"image")
    train_dir = tmp_path / "PairAnnotation" / "trn"
    test_dir = tmp_path / "PairAnnotation" / "test"
    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    (train_dir / "cat-ab.json").write_text(json.dumps({
        "src_imname": "a.jpg",
        "trg_imname": "b.jpg",
        "src_kps": [[10, 20]],
    }))
    (train_dir / "cat-bc.json").write_text(json.dumps({
        "src_imname": "b.jpg",
        "trg_imname": "c.jpg",
        "trg_kps": [[30, 40]],
    }))
    (test_dir / "cat-test.json").write_text(json.dumps({
        "src_imname": "test_a.jpg",
        "trg_imname": "test_b.jpg",
    }))
    triplets, metadata = build_strict_training_triplet_manifest(
        tmp_path,
        seed=2027,
        max_triplets=0,
    )
    names = {
        (row.category, row.source_name, row.target_name, row.bridge_name)
        for row in triplets
    }
    assert names == {
        ("cat", "a.jpg", "b.jpg", "c.jpg"),
        ("cat", "c.jpg", "b.jpg", "a.jpg"),
    }
    assert metadata["keypoint_fields_read"] is False
    assert metadata["training_pair_membership_used"] is True
    assert metadata["triplet_labels_used"] is False


def test_verifier_summary_reports_rescue_harm_retention_and_oracle_gap():
    records = [{
        "points": [
            {"pck_hits": {
                "baseline": True,
                "attention_top1": True,
                "verifier": True,
                "attention_top20_oracle": True,
            }},
            {"pck_hits": {
                "baseline": False,
                "attention_top1": False,
                "verifier": True,
                "attention_top20_oracle": True,
            }},
            {"pck_hits": {
                "baseline": True,
                "attention_top1": True,
                "verifier": False,
                "attention_top20_oracle": True,
            }},
            {"pck_hits": {
                "baseline": False,
                "attention_top1": False,
                "verifier": False,
                "attention_top20_oracle": True,
            }},
        ]
    }]
    summary = summarize_verifier_records(records)
    comparison = summary["verifier_vs_attention"]
    assert comparison["rescued"] == 1
    assert comparison["harmed"] == 1
    assert comparison["attention_correct_retention_rate"] == 0.5
    assert comparison["attention_oracle_gap_points"] == 2
    assert comparison["oracle_gap_recovered_fraction"] == 0.0


def test_native_cycle_summary_audits_confident_baseline_preserving_gates():
    records = [{
        "points": [
            {
                "pck_hits": {
                    "baseline": True,
                    "attention_top1": True,
                    "verifier": False,
                },
                "native_cycle": {
                    "confident": True,
                    "teacher_pck_hit": True,
                    "teacher_rank": 0,
                    "verifier_rank": 1,
                },
            },
            {
                "pck_hits": {
                    "baseline": False,
                    "attention_top1": False,
                    "verifier": True,
                },
                "native_cycle": {
                    "confident": True,
                    "teacher_pck_hit": True,
                    "teacher_rank": 1,
                    "verifier_rank": 1,
                },
            },
            {
                "pck_hits": {
                    "baseline": True,
                    "attention_top1": False,
                    "verifier": False,
                },
                "native_cycle": {
                    "confident": False,
                    "teacher_pck_hit": False,
                    "teacher_rank": 2,
                    "verifier_rank": 1,
                },
            },
        ]
    }]
    summary = summarize_native_cycle_diagnostics(records)
    assert summary["points"] == 3
    assert summary["confident_points"] == 2
    assert summary["coverage"] == 2 / 3
    assert summary["confident_subset_pck"]["teacher"] == 100.0
    assert summary["confident_subset_pck"]["baseline"] == 50.0
    assert summary["confident_subset_pck"]["verifier"] == 50.0
    assert summary["baseline_preserving_gates"]["verifier_point"] == 100.0 * 2 / 3
    assert summary["baseline_preserving_gates"]["teacher_point"] == 100.0
    assert summary["confident_verifier_vs_baseline"]["rescued"] == 1
    assert summary["confident_verifier_vs_baseline"]["harmed"] == 1
    assert summary["model_teacher_agreement_on_confident"] == 0.5


def test_verifier_checkpoint_round_trip_preserves_scores_and_metadata(tmp_path):
    model = _model().eval()
    groups = {
        "qk_expert": torch.randn(2, 4, 4),
        "token_state": torch.randn(2, 4, 3),
    }
    attention = torch.rand(2, 4).add(0.1)
    expected = model(groups, attention)
    path = tmp_path / "verifier.pt"
    torch.save(
        checkpoint_payload(
            model,
            training_metadata={"spair_keypoints_used": False},
        ),
        path,
    )
    restored, payload = load_verifier_checkpoint(str(path))
    assert torch.allclose(restored.eval()(groups, attention), expected)
    assert payload["training_metadata"]["spair_keypoints_used"] is False


def test_checkpoint_protocol_mismatch_is_rejected():
    metadata = {
        "spair_keypoints_used": False,
        "spair_bounding_boxes_used": False,
        "segmentation_masks_used": False,
        "pose_labels_used": False,
        "category_labels_used_for_targets": False,
        "caption_labels_used": False,
        "external_matcher_used": False,
        "dino_used": False,
        "roma_used": False,
        "protocol": {
            "image_size": [640, 640],
            "feature_block": 28,
            "timestep": 260,
            "ensemble_size": 8,
            "channel_discard": True,
            "candidate_topk": 20,
        },
    }
    args = Namespace(
        img_size=[640, 640],
        k=[28],
        t=260,
        ensemble_size=8,
        cd=True,
        candidate_topk=10,
    )
    try:
        _validate_checkpoint_metadata({"training_metadata": metadata}, args)
    except ValueError as error:
        assert "candidate_topk" in str(error)
    else:
        raise AssertionError("checkpoint protocol mismatch was accepted")

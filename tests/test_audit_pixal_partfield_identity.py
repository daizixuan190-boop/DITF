import numpy as np

from audit_pixal_partfield_identity import _sample_asset, evaluate_query, summarize_queries
from pixal3d_probe_utils import CropTransform


def test_raw_cosine_query_ranks_valid_face_features_without_other_scores():
    source = np.array([1.0, 0.0], dtype=np.float32)
    candidates = np.array(
        [[0.0, 1.0], [0.8, 0.2], [1.0, 0.0]], dtype=np.float32
    )
    result = evaluate_query(
        source_feature=source,
        candidate_features=candidates,
        candidate_valid=np.array([True, True, True]),
        candidate_pck_hits=np.array([False, False, True]),
        current_correct=False,
    )

    assert result["routeable"] is True
    assert result["teacher_correct"] is True
    assert result["gt_rank"] == 1
    assert result["selected_candidate_index"] == 2


def test_invalid_surface_candidates_are_not_ranked():
    result = evaluate_query(
        source_feature=np.array([1.0, 0.0], dtype=np.float32),
        candidate_features=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        candidate_valid=np.array([False, True]),
        candidate_pck_hits=np.array([True, False]),
        current_correct=False,
    )

    assert result["teacher_correct"] is False
    assert result["gt_rank"] is None
    assert result["selected_candidate_index"] == 1


def test_summary_reports_strict_residual_and_current_union_headroom():
    rows = [
        {"routeable": True, "top20_hit": True, "current_correct": False, "teacher_correct": True, "gt_rank": 1},
        {"routeable": True, "top20_hit": True, "current_correct": False, "teacher_correct": False, "gt_rank": 2},
        {"routeable": True, "top20_hit": True, "current_correct": True, "teacher_correct": False, "gt_rank": 3},
        {"routeable": False, "top20_hit": True, "current_correct": False, "teacher_correct": False, "gt_rank": None},
    ]

    summary = summarize_queries(rows)

    assert summary["all"]["queries"] == 4
    assert summary["routeable"]["queries"] == 3
    assert summary["strict_current_residual"]["queries"] == 2
    assert summary["strict_current_residual"]["top1"] == 0.5
    assert summary["current_union_teacher"]["correct"] == 2
    assert summary["current_union_teacher"]["rate"] == 0.5


def test_vertex_features_are_interpolated_in_raw_face_order():
    asset = {
        "transform": CropTransform((2, 2), (2, 2), (0, 0, 2, 2), (2, 2)),
        "triangle_ids": np.array([[1, 0], [0, 0]], dtype=np.int64),
        "features": np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32),
        "feature_domain": "vertex",
        "faces": np.array([[0, 1, 2]], dtype=np.int64),
        "barycentric": np.array(
            [[[0.25, 0.25, 0.5], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
            dtype=np.float32,
        ),
    }

    rows, valid, triangles = _sample_asset(asset, np.array([[0.0, 0.0]]))

    assert valid.tolist() == [True]
    assert triangles.tolist() == [1]
    np.testing.assert_allclose(rows[0], [0.75, 0.75])

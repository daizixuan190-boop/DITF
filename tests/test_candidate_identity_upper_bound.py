from analyze_candidate_identity_upper_bound import _candidate_features, _records


def test_pair_records_are_normalized_to_point_records():
    payload = {
        "pair_records": [{
            "category": "chair",
            "pair_json": "pair.json",
            "src_image": "source.jpg",
            "trg_image": "target.jpg",
            "points": [{"keypoint_index": 3, "candidates": []}],
        }],
    }

    assert _records(payload) == [{
        "category": "chair",
        "pair_json": "pair.json",
        "src_image": "source.jpg",
        "trg_image": "target.jpg",
        "keypoint_index": 3,
        "candidates": [],
    }]


def test_direct_candidate_features_exclude_pck_label():
    features = _candidate_features({
        "pixel": [10, 11],
        "pck_hit": True,
        "roma_score": -0.4,
        "roma_rank": 2,
    }, "roma")

    assert features == {"roma:roma_score": -0.4, "roma:roma_rank": 2.0}

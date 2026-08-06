from analyze_attention_top20_roma_evidence import analyze


def _dino_candidate(pixel, attention_rank, native_rank, dino_rank, hit):
    return {
        "pixel": list(pixel),
        "attention_rank": attention_rank,
        "native_candidate_rank": native_rank,
        "dino_rank": dino_rank,
        "pck_hit": hit,
    }


def _roma_candidate(pixel, roma_rank, forward, backward, certainty, hit):
    return {
        "pixel": list(pixel),
        "attention_rank": roma_rank,
        "roma_rank": roma_rank,
        "forward_error": forward,
        "backward_error": backward,
        "bidirectional_error": 0.5 * (forward + backward),
        "mutual_certainty": certainty,
        "pck_hit": hit,
    }


def test_roma_evidence_analysis_separates_warp_from_flux_ranks():
    pixels = [(1, 1), (2, 2), (3, 3)]
    dino_point = {
        "keypoint_index": 0,
        "baseline_pck_hit": False,
        "attention_top20_pck_hit": True,
        "both_wrong_top20_hit": True,
        "uniform_candidate_hit_probability": 1.0 / 3.0,
        "candidates": [
            _dino_candidate(pixels[0], 1, 1, 1, False),
            _dino_candidate(pixels[1], 2, 2, 2, False),
            _dino_candidate(pixels[2], 3, 3, 3, True),
        ],
    }
    roma_point = {
        "keypoint_index": 0,
        "candidates": [
            _roma_candidate(pixels[1], 2, 0.3, 0.4, 0.2, False),
            _roma_candidate(pixels[2], 1, 0.1, 0.1, 0.8, True),
            _roma_candidate(pixels[0], 3, 0.2, 0.5, 0.1, False),
        ],
    }
    dino = {
        "pair_records": [
            {
                "category": "cat",
                "pair_json": "pair.json",
                "points": [dino_point],
            }
        ]
    }
    roma = {
        "pair_records": [
            {
                "category": "cat",
                "pair_json": "pair.json",
                "points": [roma_point],
            }
        ]
    }

    result = analyze(dino, roma)
    hard = result["summary"]["both_wrong_top20_hit"]

    assert hard["points"] == 1
    assert hard["scorer_top1_rates"]["attention"] == 0.0
    assert hard["scorer_top1_rates"]["native"] == 0.0
    assert hard["scorer_top1_rates"]["roma_bidirectional"] == 1.0
    assert hard["roma_unique_beyond_flux"] == 1
    assert result["mechanism_checks"]["hard_roma_gain_over_flux_rank_sum"] == 1.0

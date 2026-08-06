from analyze_roma_teacher_quality import analyze


def _payload(first_hit: bool, baseline_hit: bool):
    return {
        "pair_records": [{
            "points": [{
                "baseline_pck_hit": baseline_hit,
                "both_wrong_top20_hit": not baseline_hit,
                "source_certainty": 0.3,
                "candidates": [
                    {
                        "roma_rank": 1,
                        "pck_hit": first_hit,
                        "bidirectional_error": 0.2,
                        "forward_error": 0.25,
                        "backward_error": 0.15,
                        "mutual_certainty": 0.8,
                    },
                    {
                        "roma_rank": 2,
                        "pck_hit": False,
                        "bidirectional_error": 0.5,
                    },
                ],
            }],
        }],
    }


def test_teacher_audit_uses_discovery_confidence_cutoffs_without_labels():
    result = analyze(_payload(True, False), _payload(False, True))

    assert result["all_points"]["discovery"]["teacher_point"] == 1.0
    assert result["all_points"]["heldout"]["teacher_point"] == 0.0
    assert abs(
        result["signals"]["relative_error_margin"]["0.9"]["threshold_from_discovery"] - 1.5
    ) < 1e-9

from eval_spair_attention_expert_coherent_spectral import (
    BRANCHES,
    _summary,
    build_parser,
)


def _point(
    baseline,
    early,
    head,
    expert,
    gated=False,
    head_preserving=False,
    flip_orbit=False,
    predictions=None,
):
    if predictions is None:
        predictions = {
            branch: [index, 0] for index, branch in enumerate(BRANCHES)
        }
    return {
        "pck_hits": {
            "baseline": baseline,
            "early_average": early,
            "head_coherent": head,
            "expert_coherent": expert,
            "expert_coherence_gated": gated,
            "head_preserving": head_preserving,
            "flip_orbit": flip_orbit,
        },
        "predictions": predictions,
    }


def test_summary_reports_rescues_and_harms_without_selecting_a_branch():
    records = [
        {
            "points": [
                _point(False, True, True, False),
                _point(True, True, False, True),
            ]
        }
    ]

    summary = _summary(records)

    assert set(summary["metrics"]) == set(BRANCHES)
    assert summary["comparisons"]["head_coherent_vs_baseline"]["rescued"] == 1
    assert summary["comparisons"]["head_coherent_vs_baseline"]["harmed"] == 1
    assert summary["comparisons"]["expert_coherent_vs_early_average"]["harmed"] == 1


def test_parser_has_no_persistent_cache_argument():
    parser = build_parser()
    destinations = {action.dest for action in parser._actions}

    assert "extract_native_in_memory" in destinations
    assert "fjsar_disk_cache_path" not in destinations

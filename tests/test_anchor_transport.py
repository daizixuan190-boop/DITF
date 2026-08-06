import torch

from anchor_transport import (
    certified_anchor_mask,
    local_affine_transport,
    baseline_preserving_transport_ranks,
)
from eval_spair_anchor_transport_audit import _new_counts, _summarize_counts, _update_counts


def test_certified_anchors_use_only_observable_cycle_and_agreement():
    source = torch.tensor([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    baseline = torch.tensor([[5.0, 5.0], [15.0, 5.0], [5.0, 15.0]])
    reverse = torch.tensor([[0.2, 0.0], [13.0, 0.0], [0.0, 10.3]])
    attention = torch.tensor([[5.2, 5.0], [20.0, 5.0], [5.0, 15.4]])
    anchors, cycle, agreement = certified_anchor_mask(
        source,
        baseline,
        reverse,
        attention,
        source_cell_diagonal=1.0,
        target_cell_diagonal=1.0,
        cycle_radius_cells=0.5,
        agreement_radius_cells=0.5,
    )
    assert anchors.tolist() == [True, False, True]
    assert torch.allclose(cycle, torch.tensor([0.2, 3.0, 0.3]))
    assert torch.allclose(agreement, torch.tensor([0.2, 5.0, 0.4]))


def test_local_affine_transport_recovers_translation_from_other_anchors():
    source = torch.tensor([
        [0.0, 0.0],
        [10.0, 0.0],
        [0.0, 10.0],
        [8.0, 8.0],
    ])
    baseline = source + torch.tensor([5.0, -3.0])
    candidates = torch.tensor([
        [[5.0, -3.0], [20.0, 20.0]],
        [[15.0, -3.0], [20.0, 20.0]],
        [[5.0, 7.0], [20.0, 20.0]],
        [[0.0, 0.0], [13.0, 5.0]],
    ])
    ranks, valid, support = local_affine_transport(
        source,
        baseline,
        candidates,
        torch.tensor([True, True, True, False]),
        neighbor_count=3,
        minimum_anchors=3,
        target_cell_diagonal=1.0,
    )
    assert valid.tolist() == [False, False, False, True]
    assert ranks[3].item() == 1
    assert support[3].item() < 1e-4


def test_baseline_preserving_transport_only_switches_supported_queries():
    ranks = torch.tensor([1, 0, 1])
    valid = torch.tensor([True, True, False])
    support = torch.tensor([0.2, 2.0, 0.1])
    selected, switched = baseline_preserving_transport_ranks(
        ranks,
        valid,
        support,
        transport_radius_cells=1.0,
    )
    assert selected.tolist() == [1, -1, -1]
    assert switched.tolist() == [True, False, False]


def test_anchor_audit_summary_reports_rescue_harm_and_anchor_precision():
    counts = _new_counts()
    _update_counts(
        counts,
        baseline_hits=torch.tensor([True, False, True]),
        candidate_hits=torch.tensor([
            [False, True],
            [True, False],
            [False, False],
        ]),
        anchors=torch.tensor([True, True, False]),
        transport_ranks=torch.tensor([1, 0, 0]),
        transport_valid=torch.tensor([True, True, True]),
        transport_support=torch.tensor([0.1, 0.1, 0.1]),
        selected_ranks=torch.tensor([1, 0, 0]),
        switched=torch.tensor([True, True, True]),
    )
    summary = _summarize_counts(counts)
    assert summary["anchor"]["coverage"] == 2 / 3
    assert summary["anchor"]["baseline_precision_posthoc"] == 0.5
    assert summary["vs_baseline"]["rescued"] == 1
    assert summary["vs_baseline"]["harmed"] == 1
    assert summary["vs_baseline"]["net_correct"] == 0

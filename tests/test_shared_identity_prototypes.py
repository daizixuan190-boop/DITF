import torch

from flux_joint_replay import (
    FluxReplayState,
    flux_candidate_internal_state_probe,
    manual_flux_single_block,
)
from analyze_shared_identity_prototypes import (
    analyze_shared_identity_prototypes,
    load_prototype_dataset,
)


class _IdentityQKNorm:
    def __call__(self, q, k, v):
        return q, k


class _ToyModulationOut:
    def __init__(self, vec, channels):
        batch = vec.shape[0]
        self.shift = torch.zeros(batch, 1, channels, device=vec.device, dtype=vec.dtype)
        self.scale = torch.zeros_like(self.shift)
        self.gate = torch.ones_like(self.shift)


class _ToyModulation(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

    def forward(self, vec):
        return _ToyModulationOut(vec, self.channels), None


class _ToyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 4
        self.mlp_hidden_dim = 4
        self.num_heads = 2
        self.linear1 = torch.nn.Linear(4, 16, bias=False)
        self.linear2 = torch.nn.Linear(8, 4, bias=False)
        self.norm = _IdentityQKNorm()
        self.pre_norm = torch.nn.Identity()
        self.mlp_act = torch.nn.Identity()
        self.modulation = _ToyModulation(4)
        with torch.no_grad():
            self.linear1.weight.zero_()
            self.linear1.weight[0:4] = torch.eye(4)
            self.linear1.weight[4:8] = torch.eye(4)
            self.linear1.weight[8:12] = torch.eye(4)
            self.linear2.weight.zero_()
            self.linear2.weight[:, :4] = torch.eye(4)

    def forward(self, x, vec, pe):
        return manual_flux_single_block(self, x, vec, pe)


def _toy_state(tokens):
    return FluxReplayState(
        x=tokens.unsqueeze(0),
        vec=torch.zeros(1, 4),
        pe=torch.empty(0),
        text_token_count=1,
        image_height=2,
        image_width=2,
        global_block_index=28,
    )


def _write_synthetic_shards(tmp_path):
    paths = []
    for category_index, category in enumerate(("cat", "dog", "car", "bus")):
        query_count = 8
        candidate_count = 3
        dimension = 4
        source = torch.zeros(query_count, candidate_count, dimension)
        candidate = torch.zeros_like(source)
        hits = torch.zeros(query_count, candidate_count, dtype=torch.bool)
        for query in range(query_count):
            identity = query % 2
            source[query, :, identity] = 1.0
            correct = (query + category_index) % candidate_count
            hits[query, correct] = True
            candidate[query, correct, identity] = 1.0
            for index in range(candidate_count):
                if index != correct:
                    candidate[query, index, 2 + identity] = 1.0
        path = tmp_path / f"{category}.pth"
        torch.save({
            "format_version": 1,
            "category": category,
            "pair_id": f"{category}|pair",
            "feature_groups": {
                "source_identity_token_sketch": source.to(torch.float16),
                "candidate_identity_token_sketch": candidate.to(torch.float16),
            },
            "candidate_hits": hits,
            "baseline_hits": torch.zeros(query_count, dtype=torch.bool),
            "metadata": {
                "gt_used_for_features": False,
                "gt_used_for_labels_only": True,
                "probe_is_matcher": False,
                "native_fallback_used": False,
            },
        }, path)
        paths.append(str(path))
    return paths


def test_internal_probe_exports_comparable_absolute_token_sketches_only_when_requested():
    block = _ToyBlock()
    image_tokens = torch.eye(4)
    tokens = torch.cat((torch.zeros(1, 4), image_tokens), dim=0)
    state = _toy_state(tokens)
    plain = flux_candidate_internal_state_probe(
        [block],
        state,
        state,
        torch.tensor([0, 1]),
        torch.tensor([[0, 1, 2], [1, 2, 3]]),
    )
    assert "source_identity_token_sketch" not in plain["feature_groups"]
    audited = flux_candidate_internal_state_probe(
        [block],
        state,
        state,
        torch.tensor([0, 1]),
        torch.tensor([[0, 1, 2], [1, 2, 3]]),
        include_identity_token_sketches=True,
    )
    source = audited["feature_groups"]["source_identity_token_sketch"]
    candidate = audited["feature_groups"]["candidate_identity_token_sketch"]
    assert source.shape == candidate.shape == (2, 3, 7 * 64)
    assert torch.equal(source[:, :1], source[:, 1:2])
    assert torch.isfinite(source).all()
    assert torch.isfinite(candidate).all()
    assert audited["metadata"]["identity_token_sketch"]["gt_dependent"] is False


def test_shared_identity_prototypes_are_fit_without_candidate_labels(tmp_path):
    paths = _write_synthetic_shards(tmp_path)
    output = tmp_path / "prototype_summary.json"
    result = analyze_shared_identity_prototypes(
        paths,
        output_path=str(output),
        fold_count=2,
        seed=2027,
        prototype_count=4,
        iterations=5,
        temperature=0.07,
        max_fit_tokens=1000,
        device="cpu",
    )
    assert output.exists()
    assert result["protocol"]["fit_labels"] == "none"
    assert result["protocol"]["pck_access"] == "metrics_only_after_candidate_scoring"
    assert result["scores"]["direct_token_cosine"]["baseline_metrics"]["selected_top1"] == 1.0
    assert result["scores"]["source_candidate_prototypes"]["baseline_metrics"]["selected_top1"] == 1.0
    for fold in result["folds"]:
        assert set(fold["train_categories"]).isdisjoint(fold["test_categories"])


def test_shared_identity_prototype_loader_requires_new_sketch_groups(tmp_path):
    path = tmp_path / "old.pth"
    torch.save({
        "format_version": 1,
        "category": "cat",
        "pair_id": "cat|pair",
        "feature_groups": {"token_state": torch.ones(2, 3, 4)},
        "candidate_hits": torch.ones(2, 3, dtype=torch.bool),
        "baseline_hits": torch.ones(2, dtype=torch.bool),
        "metadata": {
            "gt_used_for_features": False,
            "gt_used_for_labels_only": True,
            "probe_is_matcher": False,
            "native_fallback_used": False,
        },
    }, path)
    try:
        load_prototype_dataset([str(path)])
    except ValueError as error:
        assert "rerun identity decodability extraction" in str(error)
    else:
        raise AssertionError("old shards must not be silently accepted")

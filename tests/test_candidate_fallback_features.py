import torch

from flux_joint_replay import FluxReplayState, manual_flux_single_block
from spair_matchers import flux_fjsar_candidate_feature_batch


class _IdentityQKNorm:
    def __call__(self, q, k, v):
        return q, k


class _ModulationOut:
    def __init__(self, vec, channels):
        batch = vec.shape[0]
        self.shift = torch.zeros(batch, 1, channels, device=vec.device, dtype=vec.dtype)
        self.scale = torch.zeros(batch, 1, channels, device=vec.device, dtype=vec.dtype)
        self.gate = torch.ones(batch, 1, channels, device=vec.device, dtype=vec.dtype)


class _Modulation(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

    def forward(self, vec):
        return _ModulationOut(vec, self.channels), None


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
        self.modulation = _Modulation(4)
        with torch.no_grad():
            self.linear1.weight.zero_()
            self.linear1.weight[0:4, :] = torch.eye(4)
            self.linear1.weight[4:8, :] = torch.eye(4)
            self.linear1.weight[8:12, :] = torch.eye(4)
            self.linear2.weight.zero_()
            self.linear2.weight[:, :4] = torch.eye(4)

    def forward(self, x, vec, pe):
        return manual_flux_single_block(self, x, vec, pe)


def _state(tokens):
    return FluxReplayState(
        x=tokens.unsqueeze(0),
        vec=torch.zeros(1, 4),
        pe=torch.empty(0),
        text_token_count=1,
        image_height=2,
        image_width=2,
        global_block_index=28,
    ).to_dict()


def test_candidate_feature_batch_appends_baseline_fallback_without_changing_legacy_shape():
    block = _ToyBlock()
    image_tokens = torch.eye(4)
    state = _state(torch.cat((torch.zeros(1, 4), image_tokens), dim=0))
    features = image_tokens.t().reshape(1, 4, 2, 2)
    arguments = dict(
        src_replay_state=state,
        trg_replay_state=state,
        blocks=[block],
        candidate_topk=3,
    )
    legacy = flux_fjsar_candidate_feature_batch(
        features,
        features.clone(),
        [[0.0, 0.0], [1.0, 0.0]],
        [2, 2],
        [2, 2],
        **arguments,
    )
    fallback_pixels = legacy["candidate_pixels"][:, :1].clone()
    augmented = flux_fjsar_candidate_feature_batch(
        features,
        features.clone(),
        [[0.0, 0.0], [1.0, 0.0]],
        [2, 2],
        [2, 2],
        extra_candidate_pixels=fallback_pixels,
        **arguments,
    )

    assert legacy["candidate_pixels"].shape == (2, 3)
    assert legacy["feature_groups"]["proposal_attention"].shape[2] == 3
    assert augmented["candidate_pixels"].shape == (2, 4)
    assert torch.equal(augmented["candidate_pixels"][:, -1:], fallback_pixels)
    assert augmented["metadata"]["attention_candidate_count"] == 3
    assert augmented["metadata"]["extra_candidate_count"] == 1
    assert augmented["metadata"]["candidate_count"] == 4
    for value in augmented["feature_groups"].values():
        assert value.shape[:2] == (2, 4)

    proposal = augmented["feature_groups"]["proposal_attention"].float()
    assert proposal.shape[2] == 4
    assert torch.equal(proposal[:, :3, 3], torch.zeros(2, 3))
    assert torch.equal(proposal[:, -1, 3], torch.ones(2))

    # The fallback intentionally duplicates attention rank zero here. Native
    # cosine must be gathered once, not summed once per duplicate occurrence.
    native = augmented["feature_groups"]["native_control"].float().squeeze(2)
    assert torch.allclose(native[:, -1], native[:, 0], atol=2e-3, rtol=2e-3)

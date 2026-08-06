"""Architecture-neutral contract for frozen pair-conditioned replay.

The shared algorithm requires exact native-boundary capture, an explicit
ensemble dimension, two complete frozen interaction blocks, architecture-native
normalization/conditioning, image-only cross interaction, and a raw-feature
parity audit.  DINOv2, SD2.1 and SD3.5 adapters must implement these semantics
rather than emulating FLUX ``linear1``/``linear2`` names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import torch


@dataclass
class GenericReplayState:
    # [E, L, D]; ensemble members must not be averaged before nonlinear replay.
    sequence: torch.Tensor
    conditioning: Any
    position_state: Any
    image_token_slice: slice
    image_height: int
    image_width: int
    start_block_index: int
    feature_block_index: int


@dataclass
class AlignedFeatureEntry:
    # These three values must originate in the same native forward.
    replay_state: GenericReplayState
    raw_feature: torch.Tensor
    postprocess_state: Any
    protocol_metadata: dict[str, Any]


@dataclass
class PairReplayOutput:
    native_a: torch.Tensor
    native_b: torch.Tensor
    joint_a: torch.Tensor
    joint_b: torch.Tensor
    weighted_attention_ab: torch.Tensor
    weighted_attention_ba: torch.Tensor
    coordinate_bias_ab: torch.Tensor
    diagnostics: dict[str, float]


class JointReplayAdapter(Protocol):
    """Minimal backend contract used by the common matching logic."""

    def capture_aligned_entry(self, *args: Any, **kwargs: Any) -> AlignedFeatureEntry:
        """Capture state, raw feature and postprocess data in one forward."""
        ...

    def native_parity(
        self,
        entry: AlignedFeatureEntry,
        blocks: Sequence[Any],
    ) -> dict[str, float]:
        """Audit formula, raw-boundary and prepared-feature parity."""
        ...

    def replay_pair(
        self,
        entry_a: AlignedFeatureEntry,
        entry_b: AlignedFeatureEntry,
        blocks: Sequence[Any],
        *,
        interaction_mode: str,
    ) -> PairReplayOutput:
        """Run two frozen blocks and shared-coordinate refinement."""
        ...

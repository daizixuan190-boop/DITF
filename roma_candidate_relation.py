"""Candidate-axis relation primitives for a frozen RoMa capacity audit.

This module deliberately contains no correspondence labels, PCK, candidate
ranking rule, or category input.  It turns a source descriptor and each fixed
attention candidate descriptor into a relation tensor and scores the entire
candidate set jointly.  It is not a standalone matcher.
"""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


def pair_relation_block(source: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    """Keep source/candidate identity separate until the learned relation head."""

    if source.ndim != 2 or candidate.ndim != 3:
        raise ValueError("relation descriptors must be source [P,C], candidate [P,K,C]")
    if int(source.shape[0]) != int(candidate.shape[0]) or int(source.shape[1]) != int(candidate.shape[2]):
        raise ValueError("source and candidate descriptor axes must align")
    source = F.normalize(torch.nan_to_num(source.float()), dim=-1, eps=1e-12)
    candidate = F.normalize(torch.nan_to_num(candidate.float()), dim=-1, eps=1e-12)
    expanded = source[:, None, :].expand_as(candidate)
    return torch.cat((expanded, candidate, expanded * candidate, (expanded - candidate).abs()), dim=-1)


def multi_positive_listwise_loss(logits: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
    """Negative log probability mass assigned to any PCK-valid candidate.

    This is deliberately different from single-index cross entropy: SPair PCK
    can accept several proposals in the same attention top-20 pool.
    """

    if logits.ndim != 2 or positive_mask.shape != logits.shape:
        raise ValueError("logits and positive_mask must both be [P,K]")
    positive_mask = positive_mask.to(device=logits.device, dtype=torch.bool)
    if not bool(positive_mask.any(dim=1).all()):
        raise ValueError("each query must contain at least one positive candidate")
    positive_logits = logits.masked_fill(~positive_mask, float("-inf"))
    return (torch.logsumexp(logits, dim=1) - torch.logsumexp(positive_logits, dim=1)).mean()


class CandidateConditionedRelationHead(nn.Module):
    """Small set-aware head; candidates are never collapsed to a scalar input."""

    def __init__(self, group_dims: Mapping[str, int], *, group_width: int = 32, hidden_width: int = 128) -> None:
        super().__init__()
        if not group_dims or any(int(value) <= 0 for value in group_dims.values()):
            raise ValueError("group_dims must contain positive dimensions")
        if group_width <= 0 or hidden_width <= 0:
            raise ValueError("relation widths must be positive")
        self.group_dims = {str(key): int(value) for key, value in group_dims.items()}
        self.group_width = int(group_width)
        self.adapters = nn.ModuleDict({
            name: nn.Sequential(nn.LayerNorm(width), nn.Linear(width, group_width), nn.GELU())
            for name, width in self.group_dims.items()
        })
        combined_width = len(self.group_dims) * group_width
        self.score = nn.Sequential(
            nn.LayerNorm(combined_width * 3),
            nn.Linear(combined_width * 3, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, 1),
        )

    def forward(self, groups: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if set(groups) != set(self.group_dims):
            raise ValueError("relation groups do not match the initialized head")
        encoded = []
        shape: tuple[int, int] | None = None
        for name, adapter in self.adapters.items():
            value = groups[name]
            if value.ndim != 3 or int(value.shape[2]) != self.group_dims[name]:
                raise ValueError(f"relation group {name} must be [P,K,{self.group_dims[name]}]")
            if shape is None:
                shape = (int(value.shape[0]), int(value.shape[1]))
            elif shape != (int(value.shape[0]), int(value.shape[1])):
                raise ValueError("all relation groups must share [P,K] axes")
            encoded.append(adapter(value.float()))
        candidates = torch.cat(encoded, dim=-1)
        mean_context = candidates.mean(dim=1, keepdim=True).expand_as(candidates)
        max_context = candidates.amax(dim=1, keepdim=True).expand_as(candidates)
        return self.score(torch.cat((candidates, mean_context, max_context), dim=-1)).squeeze(-1)

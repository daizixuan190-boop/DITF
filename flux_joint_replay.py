"""Exact, ensemble-safe frozen joint self-attention replay for FLUX.

This module captures the *actual* pre-block FLUX state, preserves every
ensemble member, and replays frozen SingleStreamBlocks at an official DiTF
feature boundary.  The boundary-aligned path runs block ``k-1`` so its output
matches the feature that FLUX ``forward_feat`` exposes as the input to block
``k``.  Text remains image-local and cross-image RoPE is deliberately disabled.

No learned parameter, target annotation, category-specific threshold, or
validation-set tuned coefficient is introduced.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F


_CACHE_FORMAT_VERSION = 4


@dataclass
class FluxReplayState:
    """Per-image pre-block state with an explicit ensemble dimension.

    ``x`` and ``vec`` have leading dimension ``E``.  ``pe`` may have leading
    dimension one when FLUX broadcasts one position tensor across the ensemble,
    or ``E`` when an implementation materializes it per sample.
    """

    x: torch.Tensor
    vec: torch.Tensor
    pe: torch.Tensor
    text_token_count: int
    image_height: int
    image_width: int
    global_block_index: int

    @property
    def ensemble_size(self) -> int:
        return int(self.x.shape[0])

    def validate(self) -> None:
        if self.x.ndim != 3:
            raise ValueError(f"replay x must be [E,L,D], got {tuple(self.x.shape)}")
        if self.vec.ndim != 2 or self.vec.shape[0] != self.x.shape[0]:
            raise ValueError("replay vec must be [E,C] and align with x")
        image_count = int(self.image_height) * int(self.image_width)
        if int(self.text_token_count) < 0 or self.text_token_count + image_count != self.x.shape[1]:
            raise ValueError("text/image token split does not match replay sequence length")
        if self.pe.numel() > 0 and self.pe.ndim > 0 and self.pe.shape[0] not in (1, self.x.shape[0]):
            raise ValueError("positional state must broadcast or align with the ensemble")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "x": self.x,
            "vec": self.vec,
            "pe": self.pe,
            "text_token_count": int(self.text_token_count),
            "image_height": int(self.image_height),
            "image_width": int(self.image_width),
            "global_block_index": int(self.global_block_index),
            "ensemble_size": int(self.ensemble_size),
            "x_dtype": str(self.x.dtype),
            "vec_dtype": str(self.vec.dtype),
            "pe_dtype": str(self.pe.dtype),
            "format_version": _CACHE_FORMAT_VERSION,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FluxReplayState":
        required = {
            "x",
            "vec",
            "pe",
            "text_token_count",
            "image_height",
            "image_width",
            "global_block_index",
        }
        missing = required.difference(value)
        if missing:
            raise KeyError(f"replay state is missing keys: {sorted(missing)}")
        version = int(value.get("format_version", 0))
        if version != _CACHE_FORMAT_VERSION:
            raise ValueError(
                f"unsupported FJSAR replay-cache version {version}; "
                f"expected {_CACHE_FORMAT_VERSION}. Rebuild the companion cache."
            )
        state = cls(
            x=value["x"],
            vec=value["vec"],
            pe=value["pe"],
            text_token_count=int(value["text_token_count"]),
            image_height=int(value["image_height"]),
            image_width=int(value["image_width"]),
            global_block_index=int(value["global_block_index"]),
        )
        state.validate()
        if int(value.get("ensemble_size", state.ensemble_size)) != state.ensemble_size:
            raise ValueError("replay cache ensemble metadata does not match tensor shape")
        return state


def is_valid_flux_replay_dict(value: Any, global_block_index: int) -> bool:
    if not isinstance(value, dict):
        return False
    if int(value.get("format_version", -1)) != _CACHE_FORMAT_VERSION:
        return False
    if int(value.get("global_block_index", -1)) != int(global_block_index):
        return False
    try:
        FluxReplayState.from_dict(value)
    except (KeyError, TypeError, ValueError):
        return False
    return True


def is_valid_flux_replay_entry(value: Any, global_block_index: int) -> bool:
    """Validate an aligned cache entry containing state, feature and AdaLN."""

    if not isinstance(value, dict):
        return False
    if not is_valid_flux_replay_dict(value.get("replay_state"), global_block_index):
        return False
    feature = value.get("feature")
    ada = value.get("ada")
    if not isinstance(feature, torch.Tensor) or feature.ndim != 4 or feature.shape[0] != 1:
        return False
    if not isinstance(ada, (torch.Tensor, list, tuple)):
        return False
    state = FluxReplayState.from_dict(value["replay_state"])
    return tuple(feature.shape[-2:]) == (state.image_height, state.image_width)


def find_flux_core_model(root: Any) -> Any:
    """Find the FLUX module owning ``double_blocks`` and ``single_blocks``."""

    queue: list[Any] = [root]
    visited: set[int] = set()
    while queue:
        current = queue.pop(0)
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        if hasattr(current, "double_blocks") and hasattr(current, "single_blocks"):
            return current
        children: list[Any] = []
        if isinstance(current, nn.Module):
            children.extend(current.children())
        try:
            values = vars(current).values()
        except TypeError:
            values = ()
        for value in values:
            if isinstance(value, dict):
                children.extend(value.values())
            elif isinstance(value, (list, tuple)):
                children.extend(value)
            elif isinstance(value, nn.Module):
                children.append(value)
        queue.extend(children)
    raise RuntimeError(
        "Could not locate the FLUX core model inside Featurizer4Eval. "
        "Expected an object exposing double_blocks and single_blocks."
    )


class FluxPreBlockCapture:
    """Capture every ensemble member at one real FLUX pre-block boundary."""

    def __init__(self, flux_model: Any, global_block_index: int):
        self.flux_model = flux_model
        self.global_block_index = int(global_block_index)
        double_count = len(flux_model.double_blocks)
        if self.global_block_index < double_count:
            raise NotImplementedError(
                "FJSAR currently starts from a FLUX single-stream block; "
                f"block {self.global_block_index} is in the double-stream stage."
            )
        single_index = self.global_block_index - double_count
        if single_index < 0 or single_index >= len(flux_model.single_blocks):
            raise ValueError(f"FLUX block index {self.global_block_index} is out of range")
        self.block = flux_model.single_blocks[single_index]
        self.records: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self._original_forward_feat = getattr(self.block, "forward_feat", None)
        if self._original_forward_feat is None:
            raise RuntimeError("Selected FLUX block does not expose forward_feat")

        def wrapped_forward_feat(*args: Any, **kwargs: Any) -> Any:
            x = args[0] if len(args) > 0 else kwargs.get("x")
            vec = args[1] if len(args) > 1 else kwargs.get("vec")
            pe = args[2] if len(args) > 2 else kwargs.get("pe")
            if isinstance(vec, (tuple, list)) and vec and isinstance(vec[0], torch.Tensor):
                vec = vec[0]
            if not all(isinstance(item, torch.Tensor) for item in (x, vec, pe)):
                raise RuntimeError("Could not capture FLUX forward_feat inputs x, vec, pe")
            self._save(x, vec, pe)
            return self._original_forward_feat(*args, **kwargs)

        self.block.forward_feat = wrapped_forward_feat

    def _save(self, x: torch.Tensor, vec: torch.Tensor, pe: torch.Tensor) -> None:
        # Preserve native dtypes.  Casting BF16 to FP16 changes near-tied Q/K
        # logits and cannot be reversed when the replay is loaded.
        self.records.append((x.detach().cpu(), vec.detach().cpu(), pe.detach().cpu()))

    @staticmethod
    def _pe_for_record(pe: torch.Tensor, batch: int) -> tuple[str, torch.Tensor]:
        if pe.ndim > 0 and pe.shape[0] == batch:
            return "per_sample", pe
        return "shared", pe[:1] if pe.ndim > 0 and pe.shape[0] > 1 else pe

    def consume(self, image_height: int, image_width: int) -> FluxReplayState:
        state = _assemble_replay_records(
            self.records,
            image_height,
            image_width,
            self.global_block_index,
        )
        self.records.clear()
        return state

    def close(self) -> None:
        if self._original_forward_feat is not None:
            self.block.forward_feat = self._original_forward_feat
            self._original_forward_feat = None

    def __enter__(self) -> "FluxPreBlockCapture":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


def _assemble_replay_records(
    records: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    image_height: int,
    image_width: int,
    global_block_index: int,
) -> FluxReplayState:
    if not records:
        raise RuntimeError(f"Selected FLUX block {global_block_index} was not executed during feature extraction")
    image_tokens = int(image_height) * int(image_width)
    xs: list[torch.Tensor] = []
    vecs: list[torch.Tensor] = []
    pe_parts: list[torch.Tensor] = []
    shared_pe: torch.Tensor | None = None
    pe_mode: str | None = None
    text_count: int | None = None
    x_dtype = vec_dtype = pe_dtype = None

    for x, vec, pe in records:
        if x.ndim != 3 or x.shape[1] <= image_tokens:
            raise RuntimeError(
                f"Captured sequence shape {tuple(x.shape)} is incompatible with "
                f"an image grid of {image_height}x{image_width}"
            )
        current_text = int(x.shape[1]) - image_tokens
        if text_count is None:
            text_count = current_text
        elif current_text != text_count:
            raise RuntimeError("Ensemble members produced inconsistent text/image token splits")
        if vec.ndim != 2 or vec.shape[0] != x.shape[0]:
            raise RuntimeError("Captured FLUX x and vec batch dimensions do not match")
        if x_dtype is None:
            x_dtype, vec_dtype, pe_dtype = x.dtype, vec.dtype, pe.dtype
        elif (x.dtype, vec.dtype, pe.dtype) != (x_dtype, vec_dtype, pe_dtype):
            raise RuntimeError("Repeated ensemble calls used inconsistent dtypes")
        mode, current_pe = FluxPreBlockCapture._pe_for_record(pe, int(x.shape[0]))
        if pe_mode is None:
            pe_mode = mode
        elif pe_mode != mode:
            raise RuntimeError("Positional state alternated between shared and per-sample forms")
        if mode == "shared":
            if shared_pe is None:
                shared_pe = current_pe
            elif current_pe.shape != shared_pe.shape or not torch.equal(current_pe, shared_pe):
                raise RuntimeError("Shared positional states differ across ensemble invocations")
        else:
            pe_parts.append(current_pe)
        xs.append(x)
        vecs.append(vec)

    if text_count is None or pe_mode is None:
        raise RuntimeError("Failed to assemble captured FLUX replay state")
    x_all = torch.cat(xs, dim=0).contiguous()
    vec_all = torch.cat(vecs, dim=0).contiguous()
    pe_all = shared_pe if pe_mode == "shared" else torch.cat(pe_parts, dim=0).contiguous()
    if pe_all is None:
        raise RuntimeError("Failed to preserve positional state")
    return FluxReplayState(
        x=x_all,
        vec=vec_all,
        pe=pe_all,
        text_token_count=text_count,
        image_height=int(image_height),
        image_width=int(image_width),
        global_block_index=int(global_block_index),
    )


class FluxMultiPreBlockCapture:
    """Capture ensemble states at multiple FLUX single-stream pre-block boundaries."""

    def __init__(self, flux_model: Any, global_block_indices: Sequence[int]):
        self.flux_model = flux_model
        requested = tuple(sorted({int(index) for index in global_block_indices}))
        if not requested:
            raise ValueError("FluxMultiPreBlockCapture requires at least one block index")
        self.global_block_indices = requested
        self.records: dict[int, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {
            index: [] for index in requested
        }
        self._original_forward_feats: dict[int, Any] = {}
        double_count = len(flux_model.double_blocks)
        for global_index in requested:
            if global_index < double_count:
                raise NotImplementedError(
                    "FJSAR trajectory capture currently starts from FLUX single-stream blocks; "
                    f"block {global_index} is in the double-stream stage."
                )
            single_index = global_index - double_count
            if single_index < 0 or single_index >= len(flux_model.single_blocks):
                raise ValueError(f"FLUX block index {global_index} is out of range")
            block = flux_model.single_blocks[single_index]
            original = getattr(block, "forward_feat", None)
            if original is None:
                raise RuntimeError(f"Selected FLUX block {global_index} does not expose forward_feat")
            self._original_forward_feats[global_index] = original

            def make_wrapper(index: int, original_forward: Any):
                def wrapped_forward_feat(*args: Any, **kwargs: Any) -> Any:
                    x = args[0] if len(args) > 0 else kwargs.get("x")
                    vec = args[1] if len(args) > 1 else kwargs.get("vec")
                    pe = args[2] if len(args) > 2 else kwargs.get("pe")
                    if isinstance(vec, (tuple, list)) and vec and isinstance(vec[0], torch.Tensor):
                        vec = vec[0]
                    if not all(isinstance(item, torch.Tensor) for item in (x, vec, pe)):
                        raise RuntimeError("Could not capture FLUX forward_feat inputs x, vec, pe")
                    self.records[index].append((x.detach().cpu(), vec.detach().cpu(), pe.detach().cpu()))
                    return original_forward(*args, **kwargs)

                return wrapped_forward_feat

            block.forward_feat = make_wrapper(global_index, original)

    def consume_all(self, image_height: int, image_width: int) -> dict[str, dict[str, Any]]:
        states: dict[str, dict[str, Any]] = {}
        for global_index in self.global_block_indices:
            state = _assemble_replay_records(
                self.records[global_index],
                image_height,
                image_width,
                global_index,
            )
            self.records[global_index].clear()
            states[str(global_index)] = state.to_dict()
        return states

    def close(self) -> None:
        double_count = len(self.flux_model.double_blocks)
        for global_index, original in list(self._original_forward_feats.items()):
            single_index = int(global_index) - double_count
            self.flux_model.single_blocks[single_index].forward_feat = original
        self._original_forward_feats.clear()

    def __enter__(self) -> "FluxMultiPreBlockCapture":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


def select_flux_single_blocks(flux_model: Any, feature_block_index: int, depth: int = 1) -> list[nn.Module]:
    """Return frozen single-stream blocks that end at DiTF's feature boundary.

    FLUX ``SingleStreamBlock.forward_feat`` stores ``x_feat = x.clone()``, so a
    DiTF feature requested at block ``k`` is the input to block ``k``.  To inject
    one layer of pair-conditioned context while staying in the official feature
    space, replay starts from block ``k-1`` and executes only block ``k-1``.
    """

    if int(depth) not in (1, 2):
        raise ValueError("FJSAR replay depth must be one or two single-stream blocks")
    double_count = len(flux_model.double_blocks)
    first_global = int(feature_block_index) - 1
    first_single = first_global - double_count
    if first_single < 0:
        raise NotImplementedError("FJSAR requires the replay start block to be in the single-stream stage")
    last_single = first_single + int(depth) - 1
    if last_single >= len(flux_model.single_blocks):
        raise ValueError("requested FJSAR replay exceeds FLUX single-stream depth")
    return [flux_model.single_blocks[index] for index in range(first_single, last_single + 1)]


def replay_start_block_index(feature_block_index: int) -> int:
    return int(feature_block_index) - 1


def _apply_rope(q: torch.Tensor, k: torch.Tensor, pe: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if pe.numel() == 0:
        return q, k
    try:
        from flux.math import apply_rope
    except ImportError:
        from src.flux.math import apply_rope
    return apply_rope(q, k, pe)


def _modulated_input(block: nn.Module, x: torch.Tensor, vec: torch.Tensor) -> tuple[torch.Tensor, Any]:
    mod, _ = block.modulation(vec)
    return (1 + mod.scale) * block.pre_norm(x) + mod.shift, mod


def _block_qkv(
    block: nn.Module,
    x: torch.Tensor,
    vec: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
    x_mod, mod = _modulated_input(block, x, vec)
    qkv, mlp = torch.split(
        block.linear1(x_mod),
        [3 * int(block.hidden_size), int(block.mlp_hidden_dim)],
        dim=-1,
    )
    batch, length, _ = qkv.shape
    heads = int(block.num_heads)
    head_dim = int(block.hidden_size) // heads
    q, k, v = qkv.reshape(batch, length, 3, heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k = block.norm(q, k, v)
    return q, k, v, mlp, mod


def _local_attention_context(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
    q_rope, k_rope = _apply_rope(q, k, pe)
    context = F.scaled_dot_product_attention(q_rope, k_rope, v)
    return context.permute(0, 2, 1, 3).reshape(q.shape[0], q.shape[2], -1)


def manual_flux_single_block(block: nn.Module, x: torch.Tensor, vec: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
    q, k, v, mlp, mod = _block_qkv(block, x, vec)
    attn = _local_attention_context(q, k, v, pe)
    output = block.linear2(torch.cat((attn, block.mlp_act(mlp)), dim=2))
    return x + mod.gate * output


def _logmeanexp(values: torch.Tensor, dim: int) -> torch.Tensor:
    count = max(int(values.shape[dim]), 1)
    return torch.logsumexp(values, dim=dim) - values.new_tensor(float(count)).log()


def _cross_excess_mass(local_content_logits: torch.Tensor, cross_content_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Cross evidence beyond same-image non-self visual evidence."""

    local_count = int(local_content_logits.shape[-1])
    query_count = int(local_content_logits.shape[-2])
    if local_count > 1 and query_count == local_count:
        eye = torch.eye(query_count, device=local_content_logits.device, dtype=torch.bool)
        while eye.ndim < local_content_logits.ndim:
            eye = eye.unsqueeze(0)
        local_values = local_content_logits.masked_fill(eye, float("-inf"))
        local_energy = torch.logsumexp(local_values, dim=-1) - local_content_logits.new_tensor(
            float(local_count - 1)
        ).log()
    else:
        local_energy = _logmeanexp(local_content_logits, dim=-1)
    cross_energy = _logmeanexp(cross_content_logits, dim=-1)
    excess = (cross_energy - local_energy).clamp_min(0.0)
    mass = -torch.expm1(-excess)
    return mass.clamp(0.0, 1.0), excess


def _normalize_rows(value: torch.Tensor) -> torch.Tensor:
    return value / value.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def _mutual_cross_distribution(
    p_ab: torch.Tensor,
    p_ba: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bidirectional geometric-mean distributions and reciprocity strengths."""

    reverse_for_a = _normalize_rows(p_ba.transpose(-2, -1))
    reverse_for_b = _normalize_rows(p_ab.transpose(-2, -1))
    mutual_ab = torch.sqrt((p_ab * reverse_for_a).clamp_min(0.0))
    mutual_ba = torch.sqrt((p_ba * reverse_for_b).clamp_min(0.0))
    reciprocity_a = mutual_ab.sum(dim=-1).clamp(0.0, 1.0)
    reciprocity_b = mutual_ba.sum(dim=-1).clamp(0.0, 1.0)
    return _normalize_rows(mutual_ab), _normalize_rows(mutual_ba), reciprocity_a, reciprocity_b


def _head_group_mean(weighted: torch.Tensor, parity: int) -> torch.Tensor:
    indices = torch.arange(weighted.shape[1], device=weighted.device)
    chosen = indices[indices.remainder(2) == int(parity)]
    if chosen.numel() == 0:
        chosen = indices
    return weighted.index_select(1, chosen).mean(dim=(0, 1))


def _coherent_mutual_kernels(
    weighted_ab: torch.Tensor,
    weighted_ba: torch.Tensor,
    *,
    return_head_stack: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
):
    """Aggregate reciprocity before collapsing FLUX attention experts.

    ``head_coherent`` averages stochastic ensemble members within each head,
    then requires bidirectional agreement inside that head. ``expert_coherent``
    applies the reciprocity test independently to every member-head expert.
    Both prevent forward evidence from one head being paired with reverse
    evidence from an unrelated head.
    """

    if weighted_ab.ndim != 4 or weighted_ba.ndim != 4:
        raise ValueError("coherent mutual expects [ensemble, head, query, key]")
    if weighted_ab.shape[:2] != weighted_ba.shape[:2]:
        raise ValueError("forward and reverse attention experts do not align")
    if weighted_ab.shape[-2:] != weighted_ba.shape[-2:][::-1]:
        raise ValueError("forward and reverse attention grids do not align")

    ensemble_count = int(weighted_ab.shape[0])
    head_count = int(weighted_ab.shape[1])
    source_count = int(weighted_ab.shape[2])
    target_count = int(weighted_ab.shape[3])
    head_sum = torch.zeros(
        (source_count, target_count),
        device=weighted_ab.device,
        dtype=torch.float32,
    )
    expert_sum = torch.zeros_like(head_sum)
    head_stack = None
    if return_head_stack:
        head_stack = torch.empty(
            (head_count, source_count, target_count),
            device=weighted_ab.device,
            dtype=torch.float32,
        )
    for head in range(head_count):
        forward = weighted_ab[:, head].float()
        reverse = weighted_ba[:, head].transpose(-2, -1).float()
        head_kernel = torch.sqrt(
            (forward.mean(dim=0) * reverse.mean(dim=0)).clamp_min(0.0)
        )
        head_sum.add_(head_kernel)
        if head_stack is not None:
            head_stack[head].copy_(head_kernel)
        expert_sum.add_(
            torch.sqrt((forward * reverse).clamp_min(0.0)).sum(dim=0)
        )

    head_coherent = head_sum / float(max(head_count, 1))
    expert_coherent = expert_sum / float(max(ensemble_count * head_count, 1))
    result = (
        torch.nan_to_num(head_coherent, nan=0.0, posinf=0.0, neginf=0.0),
        torch.nan_to_num(expert_coherent, nan=0.0, posinf=0.0, neginf=0.0),
    )
    if head_stack is None:
        return result
    torch.nan_to_num_(head_stack, nan=0.0, posinf=0.0, neginf=0.0)
    return (
        *result,
        head_stack,
    )


def _summarize_cross(
    weighted_ab: torch.Tensor,
    weighted_ba: torch.Tensor,
    excess_a: torch.Tensor,
    excess_b: torch.Tensor,
    *,
    preserve_coherent_mutual: bool = False,
) -> dict[str, torch.Tensor]:
    summary = {
        "p_ab": weighted_ab.mean(dim=(0, 1)),
        "p_ba": weighted_ba.mean(dim=(0, 1)),
        "p_ab_coord": _head_group_mean(weighted_ab, 0),
        "p_ba_coord": _head_group_mean(weighted_ba, 0),
        "p_ab_validate": _head_group_mean(weighted_ab, 1),
        "p_ba_validate": _head_group_mean(weighted_ba, 1),
        "cross_excess_a": excess_a.mean(dim=(0, 1)),
        "cross_excess_b": excess_b.mean(dim=(0, 1)),
    }
    if preserve_coherent_mutual:
        head_coherent, expert_coherent, head_stack = _coherent_mutual_kernels(
            weighted_ab,
            weighted_ba,
            return_head_stack=True,
        )
        summary["head_coherent_mutual"] = head_coherent
        summary["expert_coherent_mutual"] = expert_coherent
        summary["head_coherent_mutual_stack"] = head_stack
    return summary


def _joint_attention_context(
    q_a: torch.Tensor,
    k_a: torch.Tensor,
    v_a: torch.Tensor,
    pe_a: torch.Tensor,
    text_a: int,
    q_b: torch.Tensor,
    k_b: torch.Tensor,
    v_b: torch.Tensor,
    pe_b: torch.Tensor,
    text_b: int,
    *,
    mode: str,
    cross_bias_ab: torch.Tensor | None = None,
    cross_bias_ba: torch.Tensor | None = None,
    preserve_coherent_mutual: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Local native attention plus exact or mutual-calibrated image interaction."""

    if q_a.shape[0] != q_b.shape[0]:
        raise ValueError("source and target replay ensembles must have equal size")
    if mode not in {"exact", "calibrated"}:
        raise ValueError(f"unsupported joint attention mode: {mode}")
    qra, kra = _apply_rope(q_a, k_a, pe_a)
    qrb, krb = _apply_rope(q_b, k_b, pe_b)
    scale = float(q_a.shape[-1]) ** -0.5
    local_a_logits = torch.matmul(qra.float(), kra.float().transpose(-2, -1)) * scale
    local_b_logits = torch.matmul(qrb.float(), krb.float().transpose(-2, -1)) * scale
    local_a_probs = torch.softmax(local_a_logits, dim=-1)
    local_b_probs = torch.softmax(local_b_logits, dim=-1)
    local_a_context = torch.matmul(local_a_probs, v_a.float())
    local_b_context = torch.matmul(local_b_probs, v_b.float())

    qa_img = q_a[:, :, text_a:].float()
    ka_img = k_a[:, :, text_a:].float()
    va_img = v_a[:, :, text_a:].float()
    qb_img = q_b[:, :, text_b:].float()
    kb_img = k_b[:, :, text_b:].float()
    vb_img = v_b[:, :, text_b:].float()
    cross_ab_logits = torch.matmul(qa_img, kb_img.transpose(-2, -1)) * scale
    cross_ba_logits = torch.matmul(qb_img, ka_img.transpose(-2, -1)) * scale
    if cross_bias_ab is not None:
        cross_ab_logits = cross_ab_logits + cross_bias_ab.to(cross_ab_logits)[None, None]
    if cross_bias_ba is not None:
        cross_ba_logits = cross_ba_logits + cross_bias_ba.to(cross_ba_logits)[None, None]

    if mode == "exact":
        # One genuine softmax over local text+image and cross-image keys.
        combined_a = torch.cat((local_a_logits[:, :, text_a:], cross_ab_logits), dim=-1)
        combined_b = torch.cat((local_b_logits[:, :, text_b:], cross_ba_logits), dim=-1)
        probs_a = torch.softmax(combined_a, dim=-1)
        probs_b = torch.softmax(combined_b, dim=-1)
        local_len_a = local_a_logits.shape[-1]
        local_len_b = local_b_logits.shape[-1]
        img_probs_local_a = probs_a[..., :local_len_a]
        img_probs_cross_a = probs_a[..., local_len_a:]
        img_probs_local_b = probs_b[..., :local_len_b]
        img_probs_cross_b = probs_b[..., local_len_b:]
        context_img_a = torch.matmul(img_probs_local_a, v_a.float()) + torch.matmul(img_probs_cross_a, vb_img)
        context_img_b = torch.matmul(img_probs_local_b, v_b.float()) + torch.matmul(img_probs_cross_b, va_img)
        weighted_ab = img_probs_cross_a
        weighted_ba = img_probs_cross_b
        excess_a = torch.zeros_like(weighted_ab[..., 0])
        excess_b = torch.zeros_like(weighted_ba[..., 0])
    else:
        raw_ab = torch.softmax(cross_ab_logits, dim=-1)
        raw_ba = torch.softmax(cross_ba_logits, dim=-1)
        mutual_ab, mutual_ba, reciprocal_a, reciprocal_b = _mutual_cross_distribution(raw_ab, raw_ba)
        local_visual_a = torch.matmul(qa_img, ka_img.transpose(-2, -1)) * scale
        local_visual_b = torch.matmul(qb_img, kb_img.transpose(-2, -1)) * scale
        evidence_a, excess_a = _cross_excess_mass(local_visual_a, cross_ab_logits)
        evidence_b, excess_b = _cross_excess_mass(local_visual_b, cross_ba_logits)
        mass_a = (evidence_a * reciprocal_a).clamp(0.0, 1.0)
        mass_b = (evidence_b * reciprocal_b).clamp(0.0, 1.0)
        cross_context_a = torch.matmul(mutual_ab, vb_img)
        cross_context_b = torch.matmul(mutual_ba, va_img)
        context_img_a = (1.0 - mass_a).unsqueeze(-1) * local_a_context[:, :, text_a:] + mass_a.unsqueeze(-1) * cross_context_a
        context_img_b = (1.0 - mass_b).unsqueeze(-1) * local_b_context[:, :, text_b:] + mass_b.unsqueeze(-1) * cross_context_b
        weighted_ab = mass_a.unsqueeze(-1) * mutual_ab
        weighted_ba = mass_b.unsqueeze(-1) * mutual_ba

    context_a = local_a_context.clone()
    context_b = local_b_context.clone()
    context_a[:, :, text_a:] = context_img_a
    context_b[:, :, text_b:] = context_img_b
    merged_a = context_a.permute(0, 2, 1, 3).reshape(q_a.shape[0], q_a.shape[2], -1).to(v_a.dtype)
    merged_b = context_b.permute(0, 2, 1, 3).reshape(q_b.shape[0], q_b.shape[2], -1).to(v_b.dtype)
    return merged_a, merged_b, _summarize_cross(
        weighted_ab,
        weighted_ba,
        excess_a,
        excess_b,
        preserve_coherent_mutual=preserve_coherent_mutual,
    )


def flux_joint_single_block(
    block: nn.Module,
    x_a: torch.Tensor,
    vec_a: torch.Tensor,
    pe_a: torch.Tensor,
    text_a: int,
    x_b: torch.Tensor,
    vec_b: torch.Tensor,
    pe_b: torch.Tensor,
    text_b: int,
    *,
    mode: str = "calibrated",
    cross_bias_ab: torch.Tensor | None = None,
    cross_bias_ba: torch.Tensor | None = None,
    preserve_coherent_mutual: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    q_a, k_a, v_a, mlp_a, mod_a = _block_qkv(block, x_a, vec_a)
    q_b, k_b, v_b, mlp_b, mod_b = _block_qkv(block, x_b, vec_b)
    attn_a, attn_b, diagnostics = _joint_attention_context(
        q_a, k_a, v_a, pe_a, text_a,
        q_b, k_b, v_b, pe_b, text_b,
        mode=mode,
        cross_bias_ab=cross_bias_ab,
        cross_bias_ba=cross_bias_ba,
        preserve_coherent_mutual=preserve_coherent_mutual,
    )
    output_a = block.linear2(torch.cat((attn_a, block.mlp_act(mlp_a)), dim=2))
    output_b = block.linear2(torch.cat((attn_b, block.mlp_act(mlp_b)), dim=2))
    return x_a + mod_a.gate * output_a, x_b + mod_b.gate * output_b, diagnostics


def _attention_output_projection(block: nn.Module, attn: torch.Tensor) -> torch.Tensor:
    """Apply only the attention slice of FLUX SingleStreamBlock.linear2."""

    hidden = int(block.hidden_size)
    weight = block.linear2.weight[:, :hidden]
    return F.linear(attn, weight, block.linear2.bias)


def _attention_concentration(probability: torch.Tensor) -> torch.Tensor:
    if probability.shape[-1] <= 1:
        return torch.ones(probability.shape[:-1], device=probability.device, dtype=probability.dtype)
    entropy = -(probability.clamp_min(1e-12) * probability.clamp_min(1e-12).log()).sum(dim=-1)
    return (1.0 - entropy / float(torch.log(probability.new_tensor(float(probability.shape[-1]))))).clamp(0.0, 1.0)


def _log_balanced_transport_plan(log_affinity: torch.Tensor, iterations: int = 12) -> torch.Tensor:
    """Rectangular Sinkhorn plan with uniform source/target capacity.

    This is used as a structural competition operator, not a learned matcher:
    each source row keeps equal mass and each target column receives equal
    capacity, preventing many source tokens from collapsing onto one target
    attractor before the value readout.
    """

    if log_affinity.ndim < 2:
        raise ValueError("balanced transport affinity must have source and target axes")
    source_count = int(log_affinity.shape[-2])
    target_count = int(log_affinity.shape[-1])
    if source_count <= 0 or target_count <= 0:
        raise ValueError("balanced transport affinity cannot be empty")
    log_affinity = torch.nan_to_num(
        log_affinity.float(),
        nan=-1e4,
        posinf=1e4,
        neginf=-1e4,
    )
    log_affinity = log_affinity - log_affinity.amax(dim=(-2, -1), keepdim=True)
    log_row = log_affinity.new_full(log_affinity.shape[:-1], -math.log(float(source_count)))
    log_col = log_affinity.new_full(
        (*log_affinity.shape[:-2], target_count),
        -math.log(float(target_count)),
    )
    log_u = torch.zeros_like(log_row)
    log_v = torch.zeros_like(log_col)
    for _ in range(max(1, int(iterations))):
        log_u = log_row - torch.logsumexp(log_affinity + log_v.unsqueeze(-2), dim=-1)
        log_v = log_col - torch.logsumexp(log_affinity + log_u.unsqueeze(-1), dim=-2)
    plan = torch.exp(log_affinity + log_u.unsqueeze(-1) + log_v.unsqueeze(-2))
    return torch.nan_to_num(plan, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)


def _balanced_bidirectional_kernels(
    logits_ab: torch.Tensor,
    logits_ba: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build row-normalized cross kernels from a single balanced transport plan."""

    log_p_ab = torch.log_softmax(logits_ab.float(), dim=-1)
    log_p_ba_t = torch.log_softmax(logits_ba.float(), dim=-1).transpose(-2, -1)
    reciprocal_log_affinity = 0.5 * (log_p_ab + log_p_ba_t)
    plan_ab = _log_balanced_transport_plan(reciprocal_log_affinity)
    kernel_ab = plan_ab / plan_ab.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    plan_ba = plan_ab.transpose(-2, -1)
    kernel_ba = plan_ba / plan_ba.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return (
        torch.nan_to_num(kernel_ab, nan=0.0, posinf=0.0, neginf=0.0),
        torch.nan_to_num(kernel_ba, nan=0.0, posinf=0.0, neginf=0.0),
        plan_ab,
    )


def _cell_offsets(radius: int) -> list[tuple[int, int]]:
    radius = max(0, int(radius))
    return [(dx, dy) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)]


def _local_transport_support(
    probability: torch.Tensor,
    src_height: int,
    src_width: int,
    trg_height: int,
    trg_width: int,
    *,
    radius: int,
) -> torch.Tensor:
    """Neighbor-supported transport prior for a batched cross-attention kernel."""

    radius = max(0, int(radius))
    if radius == 0:
        return torch.ones_like(probability.float())
    prob = torch.nan_to_num(probability.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    src_count = int(src_height) * int(src_width)
    trg_count = int(trg_height) * int(trg_width)
    if prob.shape[-2:] != (src_count, trg_count):
        raise ValueError("probability shape does not match source/target grids")
    support = torch.zeros_like(prob)
    counts = torch.zeros((src_count, trg_count), device=prob.device, dtype=prob.dtype)

    src_indices = torch.arange(src_count, device=prob.device, dtype=torch.long)
    src_x = src_indices % int(src_width)
    src_y = torch.div(src_indices, int(src_width), rounding_mode="floor")
    trg_indices = torch.arange(trg_count, device=prob.device, dtype=torch.long)
    trg_x = trg_indices % int(trg_width)
    trg_y = torch.div(trg_indices, int(trg_width), rounding_mode="floor")
    scale_x = float(trg_width) / float(max(1, int(src_width)))
    scale_y = float(trg_height) / float(max(1, int(src_height)))

    for dx, dy in _cell_offsets(radius):
        if dx == 0 and dy == 0:
            continue
        nsx = src_x + int(dx)
        nsy = src_y + int(dy)
        valid_source = (nsx >= 0) & (nsx < int(src_width)) & (nsy >= 0) & (nsy < int(src_height))
        if not bool(valid_source.any()):
            continue
        trg_dx = int(round(float(dx) * scale_x))
        trg_dy = int(round(float(dy) * scale_y))
        ntx = trg_x + trg_dx
        nty = trg_y + trg_dy
        valid_target = (ntx >= 0) & (ntx < int(trg_width)) & (nty >= 0) & (nty < int(trg_height))
        if not bool(valid_target.any()):
            continue
        neighbor_cells = (nsy.clamp(0, int(src_height) - 1) * int(src_width) + nsx.clamp(0, int(src_width) - 1)).long()
        expected_cells = (nty.clamp(0, int(trg_height) - 1) * int(trg_width) + ntx.clamp(0, int(trg_width) - 1)).long()
        neighbor_rows = prob.index_select(-2, neighbor_cells)
        gather_index = expected_cells.reshape(*([1] * (prob.ndim - 1)), trg_count).expand(*prob.shape[:-2], src_count, trg_count)
        aligned = torch.gather(neighbor_rows, -1, gather_index)
        row_peak = neighbor_rows.max(dim=-1, keepdim=True).values.clamp_min(1e-12)
        col_peak = prob.index_select(-1, expected_cells).max(dim=-2, keepdim=True).values.clamp_min(1e-12)
        term = torch.sqrt((aligned / row_peak).clamp_min(0.0) * (aligned / col_peak).clamp_min(0.0))
        valid = valid_source.reshape(src_count, 1) & valid_target.reshape(1, trg_count)
        while valid.ndim < prob.ndim:
            valid = valid.unsqueeze(0)
        support = support + torch.where(valid, term, torch.zeros_like(term))
        counts = counts + (valid_source.reshape(src_count, 1) & valid_target.reshape(1, trg_count)).to(prob.dtype)

    support = support / counts.clamp_min(1.0)
    return torch.nan_to_num(support, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)


def _soft_geometry_consistent_kernel(
    probability: torch.Tensor,
    src_height: int,
    src_width: int,
    trg_height: int,
    trg_width: int,
    *,
    radius: int,
    strength: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    support = _local_transport_support(
        probability,
        src_height,
        src_width,
        trg_height,
        trg_width,
        radius=radius,
    )
    strength = float(max(0.0, min(1.0, strength)))
    weight = (1.0 - strength) + strength * support
    kernel = _normalize_rows(probability.float() * weight)
    return torch.nan_to_num(kernel, nan=0.0, posinf=0.0, neginf=0.0), support


def _remove_projection(value: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    base_norm = base.square().sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return value - ((value * base).sum(dim=-1, keepdim=True) / base_norm) * base


def _identity_preserving_value_context(
    probability: torch.Tensor,
    query_value: torch.Tensor,
    key_value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Semantic value average plus query-conditioned basin covariance residual."""

    probability = torch.nan_to_num(probability.float(), nan=0.0, posinf=0.0, neginf=0.0)
    query_value = torch.nan_to_num(query_value.float(), nan=0.0, posinf=0.0, neginf=0.0)
    key_value = torch.nan_to_num(key_value.float(), nan=0.0, posinf=0.0, neginf=0.0)
    semantic = torch.matmul(probability, key_value)
    query_identity = _remove_projection(query_value, semantic)
    score = torch.matmul(query_identity, key_value.transpose(-2, -1)) * (float(key_value.shape[-1]) ** -0.5)
    first_moment = torch.matmul(probability * score, key_value)
    residual = first_moment - semantic * (
        (semantic * query_identity).sum(dim=-1, keepdim=True) * (float(key_value.shape[-1]) ** -0.5)
    )
    residual = _remove_projection(residual, semantic)

    key_energy = key_value.square().sum(dim=-1)
    variance = (
        torch.matmul(probability, key_energy.unsqueeze(-1)).squeeze(-1)
        - semantic.square().sum(dim=-1)
    ).clamp_min(0.0)
    residual_norm = residual.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    residual = residual / residual_norm * variance.sqrt().unsqueeze(-1)
    residual = torch.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0)
    context = semantic + residual
    residual_ratio = residual.norm(dim=-1) / semantic.norm(dim=-1).clamp_min(1e-12)
    return context, semantic, torch.nan_to_num(residual_ratio, nan=0.0, posinf=0.0, neginf=0.0)


def _qk_score_function_context(
    probability: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Posterior mean plus Fisher score direction in the QK retrieval space."""

    probability = torch.nan_to_num(probability.float(), nan=0.0, posinf=0.0, neginf=0.0)
    query = torch.nan_to_num(query.float(), nan=0.0, posinf=0.0, neginf=0.0)
    key = torch.nan_to_num(key.float(), nan=0.0, posinf=0.0, neginf=0.0)
    mean_key = torch.matmul(probability, key)
    scale = float(max(1, key.shape[-1])) ** -0.5
    logits = torch.matmul(query, key.transpose(-2, -1)) * scale
    mean_logit = (probability * logits).sum(dim=-1, keepdim=True)
    score = logits - mean_logit
    fisher = torch.matmul(probability * score, key)
    fisher = _remove_projection(fisher, mean_key)

    key_energy = key.square().sum(dim=-1)
    variance = (
        torch.matmul(probability, key_energy.unsqueeze(-1)).squeeze(-1)
        - mean_key.square().sum(dim=-1)
    ).clamp_min(0.0)
    fisher_norm = fisher.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    fisher = fisher / fisher_norm * variance.sqrt().unsqueeze(-1)
    fisher = torch.nan_to_num(fisher, nan=0.0, posinf=0.0, neginf=0.0)
    context = mean_key + fisher
    fisher_ratio = fisher.norm(dim=-1) / mean_key.norm(dim=-1).clamp_min(1e-12)
    return context, mean_key, torch.nan_to_num(fisher_ratio, nan=0.0, posinf=0.0, neginf=0.0)


def flux_cross_only_single_block(
    block: nn.Module,
    x_a: torch.Tensor,
    vec_a: torch.Tensor,
    pe_a: torch.Tensor,
    text_a: int,
    x_b: torch.Tensor,
    vec_b: torch.Tensor,
    pe_b: torch.Tensor,
    text_b: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Inject only bidirectional image cross-attention residuals.

    This intentionally avoids a second same-image self-attention and skips the
    block MLP branch.  The frozen block is reused only for AdaLN/QKV/QK-norm and
    the attention slice of the output projection.
    """

    q_a, k_a, v_a, _mlp_a, mod_a = _block_qkv(block, x_a, vec_a)
    q_b, k_b, v_b, _mlp_b, mod_b = _block_qkv(block, x_b, vec_b)
    qa_img = q_a[:, :, text_a:].float()
    ka_img = k_a[:, :, text_a:].float()
    va_img = v_a[:, :, text_a:].float()
    qb_img = q_b[:, :, text_b:].float()
    kb_img = k_b[:, :, text_b:].float()
    vb_img = v_b[:, :, text_b:].float()
    scale = float(q_a.shape[-1]) ** -0.5
    logits_ab = torch.matmul(qa_img, kb_img.transpose(-2, -1)) * scale
    logits_ba = torch.matmul(qb_img, ka_img.transpose(-2, -1)) * scale
    raw_ab = torch.softmax(logits_ab, dim=-1)
    raw_ba = torch.softmax(logits_ba, dim=-1)
    mutual_ab, mutual_ba, reciprocal_a, reciprocal_b = _mutual_cross_distribution(raw_ab, raw_ba)
    concentration_a = _attention_concentration(raw_ab)
    concentration_b = _attention_concentration(raw_ba)
    gate_a = (reciprocal_a * concentration_a).clamp(0.0, 1.0)
    gate_b = (reciprocal_b * concentration_b).clamp(0.0, 1.0)
    context_a = torch.matmul(mutual_ab, vb_img).permute(0, 2, 1, 3).reshape(x_a.shape[0], -1, int(block.hidden_size)).to(x_a.dtype)
    context_b = torch.matmul(mutual_ba, va_img).permute(0, 2, 1, 3).reshape(x_b.shape[0], -1, int(block.hidden_size)).to(x_b.dtype)
    delta_a = mod_a.gate * _attention_output_projection(block, context_a)
    delta_b = mod_b.gate * _attention_output_projection(block, context_b)
    out_a = x_a.clone()
    out_b = x_b.clone()
    out_a[:, text_a:] = out_a[:, text_a:] + gate_a.mean(dim=1).unsqueeze(-1).to(delta_a.dtype) * delta_a
    out_b[:, text_b:] = out_b[:, text_b:] + gate_b.mean(dim=1).unsqueeze(-1).to(delta_b.dtype) * delta_b
    diagnostics = _summarize_cross(
        (gate_a.unsqueeze(-1) * mutual_ab),
        (gate_b.unsqueeze(-1) * mutual_ba),
        gate_a,
        gate_b,
    )
    diagnostics["cross_residual_gate_a"] = gate_a.mean(dim=(0, 1))
    diagnostics["cross_residual_gate_b"] = gate_b.mean(dim=(0, 1))
    return out_a, out_b, diagnostics


def flux_identity_preserving_single_block(
    block: nn.Module,
    x_a: torch.Tensor,
    vec_a: torch.Tensor,
    pe_a: torch.Tensor,
    text_a: int,
    x_b: torch.Tensor,
    vec_b: torch.Tensor,
    pe_b: torch.Tensor,
    text_b: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Cross-only replay with identity-preserving value aggregation."""

    q_a, k_a, v_a, _mlp_a, mod_a = _block_qkv(block, x_a, vec_a)
    q_b, k_b, v_b, _mlp_b, mod_b = _block_qkv(block, x_b, vec_b)
    qa_img = q_a[:, :, text_a:].float()
    ka_img = k_a[:, :, text_a:].float()
    va_img = v_a[:, :, text_a:].float()
    qb_img = q_b[:, :, text_b:].float()
    kb_img = k_b[:, :, text_b:].float()
    vb_img = v_b[:, :, text_b:].float()
    scale = float(q_a.shape[-1]) ** -0.5
    logits_ab = torch.matmul(qa_img, kb_img.transpose(-2, -1)) * scale
    logits_ba = torch.matmul(qb_img, ka_img.transpose(-2, -1)) * scale
    raw_ab = torch.softmax(logits_ab, dim=-1)
    raw_ba = torch.softmax(logits_ba, dim=-1)
    mutual_ab, mutual_ba, reciprocal_a, reciprocal_b = _mutual_cross_distribution(raw_ab, raw_ba)
    concentration_a = _attention_concentration(raw_ab)
    concentration_b = _attention_concentration(raw_ba)
    gate_a = (reciprocal_a * concentration_a).clamp(0.0, 1.0)
    gate_b = (reciprocal_b * concentration_b).clamp(0.0, 1.0)
    context_a_heads, _semantic_a, residual_ratio_a = _identity_preserving_value_context(
        mutual_ab,
        va_img,
        vb_img,
    )
    context_b_heads, _semantic_b, residual_ratio_b = _identity_preserving_value_context(
        mutual_ba,
        vb_img,
        va_img,
    )
    context_a = context_a_heads.permute(0, 2, 1, 3).reshape(x_a.shape[0], -1, int(block.hidden_size)).to(x_a.dtype)
    context_b = context_b_heads.permute(0, 2, 1, 3).reshape(x_b.shape[0], -1, int(block.hidden_size)).to(x_b.dtype)
    delta_a = mod_a.gate * _attention_output_projection(block, context_a)
    delta_b = mod_b.gate * _attention_output_projection(block, context_b)
    out_a = x_a.clone()
    out_b = x_b.clone()
    out_a[:, text_a:] = out_a[:, text_a:] + gate_a.mean(dim=1).unsqueeze(-1).to(delta_a.dtype) * delta_a
    out_b[:, text_b:] = out_b[:, text_b:] + gate_b.mean(dim=1).unsqueeze(-1).to(delta_b.dtype) * delta_b
    diagnostics = _summarize_cross(
        gate_a.unsqueeze(-1) * mutual_ab,
        gate_b.unsqueeze(-1) * mutual_ba,
        gate_a,
        gate_b,
    )
    diagnostics["cross_residual_gate_a"] = gate_a.mean(dim=(0, 1))
    diagnostics["cross_residual_gate_b"] = gate_b.mean(dim=(0, 1))
    diagnostics["identity_residual_ratio_a"] = residual_ratio_a.mean(dim=(0, 1))
    diagnostics["identity_residual_ratio_b"] = residual_ratio_b.mean(dim=(0, 1))
    return out_a, out_b, diagnostics


def flux_qk_identity_single_block(
    block: nn.Module,
    x_a: torch.Tensor,
    vec_a: torch.Tensor,
    pe_a: torch.Tensor,
    text_a: int,
    x_b: torch.Tensor,
    vec_b: torch.Tensor,
    pe_b: torch.Tensor,
    text_b: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Cross-only replay with identity readout from the QK retrieval posterior."""

    q_a, k_a, _v_a, _mlp_a, mod_a = _block_qkv(block, x_a, vec_a)
    q_b, k_b, _v_b, _mlp_b, mod_b = _block_qkv(block, x_b, vec_b)
    qa_img = q_a[:, :, text_a:].float()
    ka_img = k_a[:, :, text_a:].float()
    qb_img = q_b[:, :, text_b:].float()
    kb_img = k_b[:, :, text_b:].float()
    scale = float(q_a.shape[-1]) ** -0.5
    logits_ab = torch.matmul(qa_img, kb_img.transpose(-2, -1)) * scale
    logits_ba = torch.matmul(qb_img, ka_img.transpose(-2, -1)) * scale
    raw_ab = torch.softmax(logits_ab, dim=-1)
    raw_ba = torch.softmax(logits_ba, dim=-1)
    mutual_ab, mutual_ba, reciprocal_a, reciprocal_b = _mutual_cross_distribution(raw_ab, raw_ba)
    concentration_a = _attention_concentration(raw_ab)
    concentration_b = _attention_concentration(raw_ba)
    gate_a = (reciprocal_a * concentration_a).clamp(0.0, 1.0)
    gate_b = (reciprocal_b * concentration_b).clamp(0.0, 1.0)
    context_a_heads, _mean_key_a, fisher_ratio_a = _qk_score_function_context(
        mutual_ab,
        qa_img,
        kb_img,
    )
    context_b_heads, _mean_key_b, fisher_ratio_b = _qk_score_function_context(
        mutual_ba,
        qb_img,
        ka_img,
    )
    context_a = context_a_heads.permute(0, 2, 1, 3).reshape(x_a.shape[0], -1, int(block.hidden_size)).to(x_a.dtype)
    context_b = context_b_heads.permute(0, 2, 1, 3).reshape(x_b.shape[0], -1, int(block.hidden_size)).to(x_b.dtype)
    delta_a = mod_a.gate * _attention_output_projection(block, context_a)
    delta_b = mod_b.gate * _attention_output_projection(block, context_b)
    out_a = x_a.clone()
    out_b = x_b.clone()
    out_a[:, text_a:] = out_a[:, text_a:] + gate_a.mean(dim=1).unsqueeze(-1).to(delta_a.dtype) * delta_a
    out_b[:, text_b:] = out_b[:, text_b:] + gate_b.mean(dim=1).unsqueeze(-1).to(delta_b.dtype) * delta_b
    diagnostics = _summarize_cross(
        gate_a.unsqueeze(-1) * mutual_ab,
        gate_b.unsqueeze(-1) * mutual_ba,
        gate_a,
        gate_b,
    )
    diagnostics["cross_residual_gate_a"] = gate_a.mean(dim=(0, 1))
    diagnostics["cross_residual_gate_b"] = gate_b.mean(dim=(0, 1))
    diagnostics["qk_fisher_ratio_a"] = fisher_ratio_a.mean(dim=(0, 1))
    diagnostics["qk_fisher_ratio_b"] = fisher_ratio_b.mean(dim=(0, 1))
    return out_a, out_b, diagnostics


def flux_balanced_transport_single_block(
    block: nn.Module,
    x_a: torch.Tensor,
    vec_a: torch.Tensor,
    pe_a: torch.Tensor,
    text_a: int,
    x_b: torch.Tensor,
    vec_b: torch.Tensor,
    pe_b: torch.Tensor,
    text_b: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Cross-only replay with competitive balanced transport value readout."""

    q_a, k_a, v_a, _mlp_a, mod_a = _block_qkv(block, x_a, vec_a)
    q_b, k_b, v_b, _mlp_b, mod_b = _block_qkv(block, x_b, vec_b)
    qa_img = q_a[:, :, text_a:].float()
    ka_img = k_a[:, :, text_a:].float()
    va_img = v_a[:, :, text_a:].float()
    qb_img = q_b[:, :, text_b:].float()
    kb_img = k_b[:, :, text_b:].float()
    vb_img = v_b[:, :, text_b:].float()
    scale = float(q_a.shape[-1]) ** -0.5
    logits_ab = torch.matmul(qa_img, kb_img.transpose(-2, -1)) * scale
    logits_ba = torch.matmul(qb_img, ka_img.transpose(-2, -1)) * scale
    raw_ab = torch.softmax(logits_ab, dim=-1)
    raw_ba = torch.softmax(logits_ba, dim=-1)
    balanced_ab, balanced_ba, plan_ab = _balanced_bidirectional_kernels(logits_ab, logits_ba)
    mutual_ab, mutual_ba, reciprocal_a, reciprocal_b = _mutual_cross_distribution(raw_ab, raw_ba)
    concentration_a = _attention_concentration(raw_ab)
    concentration_b = _attention_concentration(raw_ba)
    gate_a = (reciprocal_a * concentration_a).clamp(0.0, 1.0)
    gate_b = (reciprocal_b * concentration_b).clamp(0.0, 1.0)
    context_a = torch.matmul(balanced_ab, vb_img).permute(0, 2, 1, 3).reshape(
        x_a.shape[0],
        -1,
        int(block.hidden_size),
    ).to(x_a.dtype)
    context_b = torch.matmul(balanced_ba, va_img).permute(0, 2, 1, 3).reshape(
        x_b.shape[0],
        -1,
        int(block.hidden_size),
    ).to(x_b.dtype)
    delta_a = mod_a.gate * _attention_output_projection(block, context_a)
    delta_b = mod_b.gate * _attention_output_projection(block, context_b)
    out_a = x_a.clone()
    out_b = x_b.clone()
    out_a[:, text_a:] = out_a[:, text_a:] + gate_a.mean(dim=1).unsqueeze(-1).to(delta_a.dtype) * delta_a
    out_b[:, text_b:] = out_b[:, text_b:] + gate_b.mean(dim=1).unsqueeze(-1).to(delta_b.dtype) * delta_b
    row_error = (plan_ab.sum(dim=-1) - (1.0 / float(max(1, plan_ab.shape[-2])))).abs()
    col_error = (plan_ab.sum(dim=-2) - (1.0 / float(max(1, plan_ab.shape[-1])))).abs()
    diagnostics = _summarize_cross(
        gate_a.unsqueeze(-1) * balanced_ab,
        gate_b.unsqueeze(-1) * balanced_ba,
        gate_a,
        gate_b,
    )
    diagnostics["raw_p_ab"] = raw_ab.mean(dim=(0, 1))
    diagnostics["raw_p_ba"] = raw_ba.mean(dim=(0, 1))
    diagnostics["unbalanced_mutual_p_ab"] = mutual_ab.mean(dim=(0, 1))
    diagnostics["unbalanced_mutual_p_ba"] = mutual_ba.mean(dim=(0, 1))
    diagnostics["balanced_transport_p_ab"] = balanced_ab.mean(dim=(0, 1))
    diagnostics["balanced_transport_p_ba"] = balanced_ba.mean(dim=(0, 1))
    diagnostics["cross_residual_gate_a"] = gate_a.mean(dim=(0, 1))
    diagnostics["cross_residual_gate_b"] = gate_b.mean(dim=(0, 1))
    diagnostics["balanced_transport_row_error_a"] = row_error.mean(dim=(0, 1))
    diagnostics["balanced_transport_col_error_b"] = col_error.mean(dim=(0, 1))
    return out_a, out_b, diagnostics


def flux_geometry_consistent_single_block(
    block: nn.Module,
    x_a: torch.Tensor,
    vec_a: torch.Tensor,
    pe_a: torch.Tensor,
    text_a: int,
    image_height_a: int,
    image_width_a: int,
    x_b: torch.Tensor,
    vec_b: torch.Tensor,
    pe_b: torch.Tensor,
    text_b: int,
    image_height_b: int,
    image_width_b: int,
    *,
    radius: int = 2,
    strength: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Cross-only replay with local transport consistency inside the attention operator."""

    q_a, k_a, v_a, _mlp_a, mod_a = _block_qkv(block, x_a, vec_a)
    q_b, k_b, v_b, _mlp_b, mod_b = _block_qkv(block, x_b, vec_b)
    qa_img = q_a[:, :, text_a:].float()
    ka_img = k_a[:, :, text_a:].float()
    va_img = v_a[:, :, text_a:].float()
    qb_img = q_b[:, :, text_b:].float()
    kb_img = k_b[:, :, text_b:].float()
    vb_img = v_b[:, :, text_b:].float()
    scale = float(q_a.shape[-1]) ** -0.5
    logits_ab = torch.matmul(qa_img, kb_img.transpose(-2, -1)) * scale
    logits_ba = torch.matmul(qb_img, ka_img.transpose(-2, -1)) * scale
    raw_ab = torch.softmax(logits_ab, dim=-1)
    raw_ba = torch.softmax(logits_ba, dim=-1)
    mutual_ab, mutual_ba, reciprocal_a, reciprocal_b = _mutual_cross_distribution(raw_ab, raw_ba)
    geo_ab, support_ab = _soft_geometry_consistent_kernel(
        mutual_ab,
        image_height_a,
        image_width_a,
        image_height_b,
        image_width_b,
        radius=radius,
        strength=strength,
    )
    geo_ba, support_ba = _soft_geometry_consistent_kernel(
        mutual_ba,
        image_height_b,
        image_width_b,
        image_height_a,
        image_width_a,
        radius=radius,
        strength=strength,
    )
    concentration_a = _attention_concentration(raw_ab)
    concentration_b = _attention_concentration(raw_ba)
    gate_a = (reciprocal_a * concentration_a).clamp(0.0, 1.0)
    gate_b = (reciprocal_b * concentration_b).clamp(0.0, 1.0)
    context_a = torch.matmul(geo_ab, vb_img).permute(0, 2, 1, 3).reshape(x_a.shape[0], -1, int(block.hidden_size)).to(x_a.dtype)
    context_b = torch.matmul(geo_ba, va_img).permute(0, 2, 1, 3).reshape(x_b.shape[0], -1, int(block.hidden_size)).to(x_b.dtype)
    delta_a = mod_a.gate * _attention_output_projection(block, context_a)
    delta_b = mod_b.gate * _attention_output_projection(block, context_b)
    out_a = x_a.clone()
    out_b = x_b.clone()
    out_a[:, text_a:] = out_a[:, text_a:] + gate_a.mean(dim=1).unsqueeze(-1).to(delta_a.dtype) * delta_a
    out_b[:, text_b:] = out_b[:, text_b:] + gate_b.mean(dim=1).unsqueeze(-1).to(delta_b.dtype) * delta_b
    diagnostics = _summarize_cross(
        gate_a.unsqueeze(-1) * geo_ab,
        gate_b.unsqueeze(-1) * geo_ba,
        gate_a,
        gate_b,
    )
    diagnostics["raw_p_ab"] = raw_ab.mean(dim=(0, 1))
    diagnostics["raw_p_ba"] = raw_ba.mean(dim=(0, 1))
    diagnostics["geometry_support_a"] = support_ab.mean(dim=(0, 1, 3))
    diagnostics["geometry_support_b"] = support_ba.mean(dim=(0, 1, 3))
    diagnostics["geometry_strength"] = torch.full_like(gate_a.mean(dim=(0, 1)), float(max(0.0, min(1.0, strength))))
    diagnostics["cross_residual_gate_a"] = gate_a.mean(dim=(0, 1))
    diagnostics["cross_residual_gate_b"] = gate_b.mean(dim=(0, 1))
    return out_a, out_b, diagnostics


def run_flux_geometry_consistent_stack(
    blocks: Sequence[nn.Module],
    state_a: FluxReplayState,
    state_b: FluxReplayState,
    *,
    radius: int = 2,
    strength: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Run frozen blocks with geometry-consistent cross-image attention only."""

    if len(blocks) not in (1, 2):
        raise ValueError("FJSAR expects one or two interaction blocks")
    if state_a.global_block_index != state_b.global_block_index:
        raise ValueError("source and target replay states start from different blocks")
    if state_a.ensemble_size != state_b.ensemble_size:
        raise ValueError("source and target replay caches must preserve equal ensemble sizes")
    weight = next(blocks[0].parameters())
    a = _state_to_device(state_a, weight.device, weight.dtype)
    b = _state_to_device(state_b, weight.device, weight.dtype)
    x_a, x_b = a.x, b.x
    diagnostics: dict[str, torch.Tensor] | None = None
    for index, block in enumerate(blocks):
        x_a, x_b, current = flux_geometry_consistent_single_block(
            block,
            x_a,
            a.vec,
            a.pe,
            a.text_token_count,
            a.image_height,
            a.image_width,
            x_b,
            b.vec,
            b.pe,
            b.text_token_count,
            b.image_height,
            b.image_width,
            radius=radius,
            strength=strength,
        )
        diagnostics = {f"block{index}_{key}": value for key, value in current.items()}
        diagnostics.update({
            "p_ab": current["p_ab"],
            "p_ba": current["p_ba"],
            "raw_p_ab": current["raw_p_ab"],
            "raw_p_ba": current["raw_p_ba"],
            "cross_excess_a": current["cross_residual_gate_a"],
            "cross_excess_b": current["cross_residual_gate_b"],
            "geometry_support_a": current["geometry_support_a"],
            "geometry_support_b": current["geometry_support_b"],
            "geometry_strength": current["geometry_strength"],
            "coordinate_reliability_a": torch.ones(a.image_height * a.image_width, device=weight.device),
            "coordinate_reliability_b": torch.ones(b.image_height * b.image_width, device=weight.device),
            "cycle_error_a": torch.zeros(a.image_height * a.image_width, device=weight.device),
            "cycle_error_b": torch.zeros(b.image_height * b.image_width, device=weight.device),
        })
    if diagnostics is None:
        raise RuntimeError("no geometry-consistent block was executed")
    return x_a, x_b, diagnostics


def _state_to_device(state: FluxReplayState, device: torch.device, dtype: torch.dtype) -> FluxReplayState:
    return FluxReplayState(
        x=state.x.to(device=device, dtype=dtype),
        vec=state.vec.to(device=device, dtype=dtype),
        pe=state.pe.to(device=device),
        text_token_count=state.text_token_count,
        image_height=state.image_height,
        image_width=state.image_width,
        global_block_index=state.global_block_index,
    )


def _unit_grid(height: int, width: int, device: torch.device) -> torch.Tensor:
    ys, xs = torch.meshgrid(
        torch.linspace(0.0, 1.0, height, device=device),
        torch.linspace(0.0, 1.0, width, device=device),
        indexing="ij",
    )
    return torch.stack((xs, ys), dim=-1).reshape(-1, 2)


def _conditional(prob: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    probability = torch.nan_to_num(prob.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    mass = probability.sum(dim=-1).clamp(0.0, 1.0)
    return _normalize_rows(probability), mass


def _row_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a.float(), b.float(), dim=-1, eps=1e-12).clamp(0.0, 1.0)


def _distribution_moments(prob: torch.Tensor, coords: torch.Tensor, height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    mean = prob @ coords
    diff = coords.unsqueeze(0) - mean.unsqueeze(1)
    covariance = torch.einsum("nm,nmi,nmj->nij", prob, diff, diff)
    step_x = 1.0 / float(max(width - 1, 1))
    step_y = 1.0 / float(max(height - 1, 1))
    floor = covariance.new_tensor([[step_x * step_x / 12.0, 0.0], [0.0, step_y * step_y / 12.0]])
    return mean, covariance + floor.unsqueeze(0)


def _mahalanobis_grid(coords: torch.Tensor, mean: torch.Tensor, covariance: torch.Tensor) -> torch.Tensor:
    # result [queries, candidates]
    inverse = torch.linalg.inv(covariance.float())
    diff = coords.unsqueeze(0) - mean.unsqueeze(1)
    return torch.einsum("nmi,nij,nmj->nm", diff, inverse, diff).clamp_min(0.0)


def pair_coordinate_bias(
    first_attention: dict[str, torch.Tensor],
    src_height: int,
    src_width: int,
    trg_height: int,
    trg_width: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Build an independent, covariance-aware shared-space bias for block two."""

    p_ab_coord, _ = _conditional(first_attention["p_ab_coord"])
    p_ba_coord, _ = _conditional(first_attention["p_ba_coord"])
    p_ab_validate, _ = _conditional(first_attention["p_ab_validate"])
    p_ba_validate, _ = _conditional(first_attention["p_ba_validate"])
    device = p_ab_coord.device
    src_coords = _unit_grid(src_height, src_width, device)
    trg_coords = _unit_grid(trg_height, trg_width, device)
    mean_b, cov_b = _distribution_moments(p_ab_coord, trg_coords, trg_height, trg_width)
    mean_a, cov_a = _distribution_moments(p_ba_coord, src_coords, src_height, src_width)
    dist_ab = _mahalanobis_grid(trg_coords, mean_b, cov_b)
    dist_ba = _mahalanobis_grid(src_coords, mean_a, cov_a)
    symmetric_distance = dist_ab + dist_ba.t()

    agreement_a = _row_cosine(p_ab_coord, p_ab_validate)
    agreement_b = _row_cosine(p_ba_coord, p_ba_validate)
    cycle_a = p_ab_coord @ mean_a
    cycle_b = p_ba_coord @ mean_b
    grid_step_a = (1.0 / max(src_width - 1, 1) ** 2 + 1.0 / max(src_height - 1, 1) ** 2) ** 0.5
    grid_step_b = (1.0 / max(trg_width - 1, 1) ** 2 + 1.0 / max(trg_height - 1, 1) ** 2) ** 0.5
    cycle_error_a = (cycle_a - src_coords).norm(dim=1)
    cycle_error_b = (cycle_b - trg_coords).norm(dim=1)
    cycle_conf_a = 1.0 / (1.0 + cycle_error_a / max(grid_step_a, 1e-12))
    cycle_conf_b = 1.0 / (1.0 + cycle_error_b / max(grid_step_b, 1e-12))
    reliability_a = (agreement_a * cycle_conf_a).clamp(0.0, 1.0)
    reliability_b = (agreement_b * cycle_conf_b).clamp(0.0, 1.0)
    reliability_pair = torch.sqrt(reliability_a[:, None] * reliability_b[None, :])
    bias_ab = -0.5 * symmetric_distance * reliability_pair
    return bias_ab, bias_ab.t(), {
        "coordinate_agreement_a": agreement_a,
        "coordinate_agreement_b": agreement_b,
        "cycle_error_a": cycle_error_a,
        "cycle_error_b": cycle_error_b,
        "cycle_confidence_a": cycle_conf_a,
        "cycle_confidence_b": cycle_conf_b,
        "coordinate_reliability_a": reliability_a,
        "coordinate_reliability_b": reliability_b,
        "mean_b_given_a": mean_b,
        "mean_a_given_b": mean_a,
        "cov_b_given_a": cov_b,
        "cov_a_given_b": cov_a,
    }


def run_flux_native_stack(blocks: Sequence[nn.Module], state: FluxReplayState) -> torch.Tensor:
    """Run native frozen blocks for every ensemble member and return raw sequence."""

    if len(blocks) not in (1, 2):
        raise ValueError("FJSAR expects one or two interaction blocks")
    weight = next(blocks[0].parameters())
    local = _state_to_device(state, weight.device, weight.dtype)
    x = local.x
    for block in blocks:
        x = block(x, local.vec, local.pe)
    return x


def run_flux_joint_stack(
    blocks: Sequence[nn.Module],
    state_a: FluxReplayState,
    state_b: FluxReplayState,
    *,
    mode: str = "calibrated",
    use_coordinate_bias: bool = True,
    preserve_coherent_mutual: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Run pair-conditioned frozen blocks with optional shared-space refinement."""

    if len(blocks) not in (1, 2):
        raise ValueError("FJSAR expects one or two interaction blocks")
    if state_a.global_block_index != state_b.global_block_index:
        raise ValueError("source and target replay states start from different blocks")
    if state_a.ensemble_size != state_b.ensemble_size:
        raise ValueError("source and target replay caches must preserve equal ensemble sizes")
    weight = next(blocks[0].parameters())
    a = _state_to_device(state_a, weight.device, weight.dtype)
    b = _state_to_device(state_b, weight.device, weight.dtype)
    x_a, x_b, first = flux_joint_single_block(
        blocks[0], a.x, a.vec, a.pe, a.text_token_count,
        b.x, b.vec, b.pe, b.text_token_count,
        mode=mode,
        preserve_coherent_mutual=preserve_coherent_mutual,
    )
    if len(blocks) == 1:
        image_a = a.image_height * a.image_width
        image_b = b.image_height * b.image_width
        device = first["p_ab"].device
        diagnostics = {
            **{f"first_{key}": value for key, value in first.items()},
            "coordinate_reliability_a": torch.ones(image_a, device=device),
            "coordinate_reliability_b": torch.ones(image_b, device=device),
            "cycle_error_a": torch.zeros(image_a, device=device),
            "cycle_error_b": torch.zeros(image_b, device=device),
            "p_ab": first["p_ab"],
            "p_ba": first["p_ba"],
            "cross_excess_a": first["cross_excess_a"],
            "cross_excess_b": first["cross_excess_b"],
        }
        if preserve_coherent_mutual:
            diagnostics["head_coherent_mutual"] = first[
                "head_coherent_mutual"
            ]
            diagnostics["expert_coherent_mutual"] = first[
                "expert_coherent_mutual"
            ]
            diagnostics["head_coherent_mutual_stack"] = first[
                "head_coherent_mutual_stack"
            ]
        return x_a, x_b, diagnostics

    if use_coordinate_bias:
        bias_ab, bias_ba, coordinate = pair_coordinate_bias(
            first,
            a.image_height,
            a.image_width,
            b.image_height,
            b.image_width,
        )
    else:
        bias_ab = bias_ba = None
        image_a = a.image_height * a.image_width
        image_b = b.image_height * b.image_width
        device = first["p_ab"].device
        coordinate = {
            "coordinate_reliability_a": torch.ones(image_a, device=device),
            "coordinate_reliability_b": torch.ones(image_b, device=device),
            "cycle_error_a": torch.zeros(image_a, device=device),
            "cycle_error_b": torch.zeros(image_b, device=device),
        }
    x_a, x_b, second = flux_joint_single_block(
        blocks[1], x_a, a.vec, a.pe, a.text_token_count,
        x_b, b.vec, b.pe, b.text_token_count,
        mode=mode,
        cross_bias_ab=bias_ab,
        cross_bias_ba=bias_ba,
        preserve_coherent_mutual=preserve_coherent_mutual,
    )
    diagnostics = {
        **{f"first_{key}": value for key, value in first.items()},
        **{f"second_{key}": value for key, value in second.items()},
        **coordinate,
        # Matcher-facing canonical attention is the refined second-block map.
        "p_ab": second["p_ab"],
        "p_ba": second["p_ba"],
        "cross_excess_a": second["cross_excess_a"],
        "cross_excess_b": second["cross_excess_b"],
    }
    if preserve_coherent_mutual:
        diagnostics["head_coherent_mutual"] = second[
            "head_coherent_mutual"
        ]
        diagnostics["expert_coherent_mutual"] = second[
            "expert_coherent_mutual"
        ]
        diagnostics["head_coherent_mutual_stack"] = second[
            "head_coherent_mutual_stack"
        ]
    if use_coordinate_bias:
        diagnostics["coordinate_bias_ab"] = bias_ab
        diagnostics["coordinate_bias_ba"] = bias_ba
    return x_a, x_b, diagnostics


def run_flux_cross_only_stack(
    blocks: Sequence[nn.Module],
    state_a: FluxReplayState,
    state_b: FluxReplayState,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Run one or two frozen blocks as pure bidirectional cross-attention residuals."""

    if len(blocks) not in (1, 2):
        raise ValueError("FJSAR expects one or two interaction blocks")
    if state_a.global_block_index != state_b.global_block_index:
        raise ValueError("source and target replay states start from different blocks")
    if state_a.ensemble_size != state_b.ensemble_size:
        raise ValueError("source and target replay caches must preserve equal ensemble sizes")
    weight = next(blocks[0].parameters())
    a = _state_to_device(state_a, weight.device, weight.dtype)
    b = _state_to_device(state_b, weight.device, weight.dtype)
    x_a, x_b = a.x, b.x
    diagnostics: dict[str, torch.Tensor] | None = None
    for index, block in enumerate(blocks):
        x_a, x_b, current = flux_cross_only_single_block(
            block,
            x_a,
            a.vec,
            a.pe,
            a.text_token_count,
            x_b,
            b.vec,
            b.pe,
            b.text_token_count,
        )
        diagnostics = {f"block{index}_{key}": value for key, value in current.items()}
        diagnostics.update({
            "p_ab": current["p_ab"],
            "p_ba": current["p_ba"],
            "cross_excess_a": current["cross_residual_gate_a"],
            "cross_excess_b": current["cross_residual_gate_b"],
            "coordinate_reliability_a": torch.ones(a.image_height * a.image_width, device=weight.device),
            "coordinate_reliability_b": torch.ones(b.image_height * b.image_width, device=weight.device),
            "cycle_error_a": torch.zeros(a.image_height * a.image_width, device=weight.device),
            "cycle_error_b": torch.zeros(b.image_height * b.image_width, device=weight.device),
        })
    if diagnostics is None:
        raise RuntimeError("no cross-only block was executed")
    return x_a, x_b, diagnostics


def run_flux_identity_preserving_stack(
    blocks: Sequence[nn.Module],
    state_a: FluxReplayState,
    state_b: FluxReplayState,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Run frozen blocks with identity-preserving cross-image value updates."""

    if len(blocks) not in (1, 2):
        raise ValueError("FJSAR expects one or two interaction blocks")
    if state_a.global_block_index != state_b.global_block_index:
        raise ValueError("source and target replay states start from different blocks")
    if state_a.ensemble_size != state_b.ensemble_size:
        raise ValueError("source and target replay caches must preserve equal ensemble sizes")
    weight = next(blocks[0].parameters())
    a = _state_to_device(state_a, weight.device, weight.dtype)
    b = _state_to_device(state_b, weight.device, weight.dtype)
    x_a, x_b = a.x, b.x
    diagnostics: dict[str, torch.Tensor] | None = None
    for index, block in enumerate(blocks):
        x_a, x_b, current = flux_identity_preserving_single_block(
            block,
            x_a,
            a.vec,
            a.pe,
            a.text_token_count,
            x_b,
            b.vec,
            b.pe,
            b.text_token_count,
        )
        diagnostics = {f"block{index}_{key}": value for key, value in current.items()}
        diagnostics.update({
            "p_ab": current["p_ab"],
            "p_ba": current["p_ba"],
            "cross_excess_a": current["cross_residual_gate_a"],
            "cross_excess_b": current["cross_residual_gate_b"],
            "identity_residual_ratio_a": current["identity_residual_ratio_a"],
            "identity_residual_ratio_b": current["identity_residual_ratio_b"],
            "coordinate_reliability_a": torch.ones(a.image_height * a.image_width, device=weight.device),
            "coordinate_reliability_b": torch.ones(b.image_height * b.image_width, device=weight.device),
            "cycle_error_a": torch.zeros(a.image_height * a.image_width, device=weight.device),
            "cycle_error_b": torch.zeros(b.image_height * b.image_width, device=weight.device),
        })
    if diagnostics is None:
        raise RuntimeError("no identity-preserving block was executed")
    return x_a, x_b, diagnostics


def run_flux_qk_identity_stack(
    blocks: Sequence[nn.Module],
    state_a: FluxReplayState,
    state_b: FluxReplayState,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Run frozen blocks with QK posterior score-function cross updates."""

    if len(blocks) not in (1, 2):
        raise ValueError("FJSAR expects one or two interaction blocks")
    if state_a.global_block_index != state_b.global_block_index:
        raise ValueError("source and target replay states start from different blocks")
    if state_a.ensemble_size != state_b.ensemble_size:
        raise ValueError("source and target replay caches must preserve equal ensemble sizes")
    weight = next(blocks[0].parameters())
    a = _state_to_device(state_a, weight.device, weight.dtype)
    b = _state_to_device(state_b, weight.device, weight.dtype)
    x_a, x_b = a.x, b.x
    diagnostics: dict[str, torch.Tensor] | None = None
    for index, block in enumerate(blocks):
        x_a, x_b, current = flux_qk_identity_single_block(
            block,
            x_a,
            a.vec,
            a.pe,
            a.text_token_count,
            x_b,
            b.vec,
            b.pe,
            b.text_token_count,
        )
        diagnostics = {f"block{index}_{key}": value for key, value in current.items()}
        diagnostics.update({
            "p_ab": current["p_ab"],
            "p_ba": current["p_ba"],
            "cross_excess_a": current["cross_residual_gate_a"],
            "cross_excess_b": current["cross_residual_gate_b"],
            "qk_fisher_ratio_a": current["qk_fisher_ratio_a"],
            "qk_fisher_ratio_b": current["qk_fisher_ratio_b"],
            "coordinate_reliability_a": torch.ones(a.image_height * a.image_width, device=weight.device),
            "coordinate_reliability_b": torch.ones(b.image_height * b.image_width, device=weight.device),
            "cycle_error_a": torch.zeros(a.image_height * a.image_width, device=weight.device),
            "cycle_error_b": torch.zeros(b.image_height * b.image_width, device=weight.device),
        })
    if diagnostics is None:
        raise RuntimeError("no QK-identity block was executed")
    return x_a, x_b, diagnostics


def run_flux_balanced_transport_stack(
    blocks: Sequence[nn.Module],
    state_a: FluxReplayState,
    state_b: FluxReplayState,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Run frozen blocks with competitive balanced cross-image transport."""

    if len(blocks) not in (1, 2):
        raise ValueError("FJSAR expects one or two interaction blocks")
    if state_a.global_block_index != state_b.global_block_index:
        raise ValueError("source and target replay states start from different blocks")
    if state_a.ensemble_size != state_b.ensemble_size:
        raise ValueError("source and target replay caches must preserve equal ensemble sizes")
    weight = next(blocks[0].parameters())
    a = _state_to_device(state_a, weight.device, weight.dtype)
    b = _state_to_device(state_b, weight.device, weight.dtype)
    x_a, x_b = a.x, b.x
    diagnostics: dict[str, torch.Tensor] | None = None
    for index, block in enumerate(blocks):
        x_a, x_b, current = flux_balanced_transport_single_block(
            block,
            x_a,
            a.vec,
            a.pe,
            a.text_token_count,
            x_b,
            b.vec,
            b.pe,
            b.text_token_count,
        )
        diagnostics = {f"block{index}_{key}": value for key, value in current.items()}
        diagnostics.update({
            "p_ab": current["p_ab"],
            "p_ba": current["p_ba"],
            "raw_p_ab": current["raw_p_ab"],
            "raw_p_ba": current["raw_p_ba"],
            "unbalanced_mutual_p_ab": current["unbalanced_mutual_p_ab"],
            "unbalanced_mutual_p_ba": current["unbalanced_mutual_p_ba"],
            "balanced_transport_p_ab": current["balanced_transport_p_ab"],
            "balanced_transport_p_ba": current["balanced_transport_p_ba"],
            "cross_excess_a": current["cross_residual_gate_a"],
            "cross_excess_b": current["cross_residual_gate_b"],
            "balanced_transport_row_error_a": current["balanced_transport_row_error_a"],
            "balanced_transport_col_error_b": current["balanced_transport_col_error_b"],
            "coordinate_reliability_a": torch.ones(a.image_height * a.image_width, device=weight.device),
            "coordinate_reliability_b": torch.ones(b.image_height * b.image_width, device=weight.device),
            "cycle_error_a": torch.zeros(a.image_height * a.image_width, device=weight.device),
            "cycle_error_b": torch.zeros(b.image_height * b.image_width, device=weight.device),
        })
    if diagnostics is None:
        raise RuntimeError("no balanced-transport block was executed")
    return x_a, x_b, diagnostics


def flux_candidate_clamped_causal_probe(
    clamp_block: nn.Module,
    release_block: nn.Module,
    state_a: FluxReplayState,
    state_b: FluxReplayState,
    src_cells: torch.Tensor,
    candidate_cells: torch.Tensor,
    *,
    candidate_value_scale: float = 1.0,
) -> dict[str, Any]:
    """Measure whether a mass-preserving candidate clamp survives free Q/K release.

    The clamp is applied to one source/target token pair at a time in the first
    block.  Its exact local contribution and total cross mass are preserved;
    only the conditional cross value is replaced.  The second block is not
    executed: its free, unmodified Q/K operator reads the intervened feature
    state and supplies bidirectional candidate ranks.
    """

    if not float(candidate_value_scale) > 0.0:
        raise ValueError("candidate_value_scale must be positive")
    if state_a.global_block_index != state_b.global_block_index:
        raise ValueError("causal replay states must start from the same block")
    if state_a.ensemble_size != state_b.ensemble_size:
        raise ValueError("causal replay states must preserve equal ensemble sizes")
    clamp_weight = next(clamp_block.parameters())
    release_weight = next(release_block.parameters())
    if clamp_weight.device != release_weight.device:
        raise ValueError("clamp and release blocks must be on the same device")
    a = _state_to_device(state_a, clamp_weight.device, clamp_weight.dtype)
    b = _state_to_device(state_b, clamp_weight.device, clamp_weight.dtype)
    src_cells = src_cells.to(device=clamp_weight.device, dtype=torch.long).flatten()
    candidate_cells = candidate_cells.to(device=clamp_weight.device, dtype=torch.long)
    if candidate_cells.ndim != 2 or candidate_cells.shape[0] != src_cells.shape[0]:
        raise ValueError("candidate cells must be [point,candidate] and align with source cells")
    source_count = int(a.image_height) * int(a.image_width)
    target_count = int(b.image_height) * int(b.image_width)
    src_cells = src_cells.clamp(0, source_count - 1)
    candidate_cells = candidate_cells.clamp(0, target_count - 1)
    point_count, candidate_count = map(int, candidate_cells.shape)
    if point_count == 0 or candidate_count == 0:
        empty = torch.empty(
            (point_count, candidate_count),
            device=clamp_weight.device,
            dtype=torch.float32,
        )
        return {
            "score_names": [],
            "scores": {},
            "diagnostics": {},
            "metadata": {
                "point_count": point_count,
                "candidate_count": candidate_count,
            },
        }

    q_a, k_a, v_a, mlp_a, mod_a = _block_qkv(clamp_block, a.x, a.vec)
    q_b, k_b, v_b, mlp_b, mod_b = _block_qkv(clamp_block, b.x, b.vec)
    base_attn_a, base_attn_b, _base_attention = _joint_attention_context(
        q_a,
        k_a,
        v_a,
        a.pe,
        a.text_token_count,
        q_b,
        k_b,
        v_b,
        b.pe,
        b.text_token_count,
        mode="exact",
    )
    base_output_a = clamp_block.linear2(
        torch.cat((base_attn_a, clamp_block.mlp_act(mlp_a)), dim=2)
    )
    base_output_b = clamp_block.linear2(
        torch.cat((base_attn_b, clamp_block.mlp_act(mlp_b)), dim=2)
    )
    base_x_a = a.x + mod_a.gate * base_output_a
    base_x_b = b.x + mod_b.gate * base_output_b

    # The release Q/K for unchanged tokens is reusable across every hypothesis.
    _base_qa, base_ka, _base_va, _base_mlp_a, _base_mod_a = _block_qkv(
        release_block,
        base_x_a.to(dtype=release_weight.dtype),
        a.vec.to(dtype=release_weight.dtype),
    )
    _base_qb, base_kb, _base_vb, _base_mlp_b, _base_mod_b = _block_qkv(
        release_block,
        base_x_b.to(dtype=release_weight.dtype),
        b.vec.to(dtype=release_weight.dtype),
    )

    qra, kra = _apply_rope(q_a, k_a, a.pe)
    qrb, krb = _apply_rope(q_b, k_b, b.pe)
    qa_img = q_a[:, :, a.text_token_count:].float()
    ka_img = k_a[:, :, a.text_token_count:].float()
    va_img = v_a[:, :, a.text_token_count:].float()
    qb_img = q_b[:, :, b.text_token_count:].float()
    kb_img = k_b[:, :, b.text_token_count:].float()
    vb_img = v_b[:, :, b.text_token_count:].float()
    release_ka_img = base_ka[:, :, a.text_token_count:].float()
    release_kb_img = base_kb[:, :, b.text_token_count:].float()
    clamp_scale = float(q_a.shape[-1]) ** -0.5
    release_scale = float(base_ka.shape[-1]) ** -0.5

    hypothesis_src = src_cells[:, None].expand(point_count, candidate_count).reshape(-1)
    hypothesis_trg = candidate_cells.reshape(-1)
    hypothesis_count = int(hypothesis_src.numel())
    ensemble_count = int(q_a.shape[0])
    head_count = int(q_a.shape[1])
    elements_per_hypothesis = max(
        1,
        ensemble_count
        * head_count
        * (
            int(a.x.shape[1])
            + int(b.x.shape[1])
            + 2 * source_count
            + 2 * target_count
        ),
    )
    hypothesis_chunk = max(
        1,
        min(hypothesis_count, 12_000_000 // elements_per_hypothesis),
    )

    score_chunks: dict[str, list[torch.Tensor]] = {
        "pre_intervention_bidirectional_negative_log_rank": [],
        "post_release_bidirectional_negative_log_rank": [],
        "post_release_source_negative_log_rank": [],
        "post_release_target_negative_log_rank": [],
        "causal_rank_improvement": [],
        "post_release_mutual_top1_vote": [],
        "post_release_mutual_top5_vote": [],
    }
    diagnostic_chunks: dict[str, list[torch.Tensor]] = {
        "source_cross_mass": [],
        "target_cross_mass": [],
        "source_intervention_relative_l2": [],
        "target_intervention_relative_l2": [],
        "post_release_score_std": [],
        "causal_improvement_positive_fraction": [],
    }
    attn_weight = clamp_block.linear2.weight[:, : int(clamp_block.hidden_size)].float()
    gate_a = mod_a.gate.float()
    gate_b = mod_b.gate.float()

    def _exact_components(
        local_logits: torch.Tensor,
        cross_logits: torch.Tensor,
        local_values: torch.Tensor,
        cross_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        local_log_z = torch.logsumexp(local_logits, dim=-1)
        cross_log_z = torch.logsumexp(cross_logits, dim=-1)
        log_z = torch.logaddexp(local_log_z, cross_log_z)
        local_probability = torch.exp(local_logits - log_z.unsqueeze(-1))
        cross_probability = torch.exp(cross_logits - log_z.unsqueeze(-1))
        local_context = torch.matmul(local_probability, local_values.float())
        cross_context = torch.matmul(cross_probability, cross_values.float())
        cross_mass = cross_probability.sum(dim=-1).clamp(0.0, 1.0)
        return local_context, cross_context, cross_mass

    for start in range(0, hypothesis_count, hypothesis_chunk):
        end = min(hypothesis_count, start + hypothesis_chunk)
        src_index = hypothesis_src[start:end]
        trg_index = hypothesis_trg[start:end]
        src_global = src_index + int(a.text_token_count)
        trg_global = trg_index + int(b.text_token_count)

        src_local_q = qra.index_select(2, src_global).float()
        src_cross_q = qa_img.index_select(2, src_index)
        src_local_logits = torch.matmul(src_local_q, kra.float().transpose(-2, -1)) * clamp_scale
        src_cross_logits = torch.matmul(src_cross_q, kb_img.transpose(-2, -1)) * clamp_scale
        src_local_context, src_cross_context, src_cross_mass = _exact_components(
            src_local_logits,
            src_cross_logits,
            v_a,
            vb_img,
        )
        selected_target_value = vb_img.index_select(2, trg_index)
        clamped_src_context = (
            src_local_context
            + src_cross_mass.unsqueeze(-1) * float(candidate_value_scale) * selected_target_value
        )
        base_src_context = src_local_context + src_cross_context

        trg_local_q = qrb.index_select(2, trg_global).float()
        trg_cross_q = qb_img.index_select(2, trg_index)
        trg_local_logits = torch.matmul(trg_local_q, krb.float().transpose(-2, -1)) * clamp_scale
        trg_cross_logits = torch.matmul(trg_cross_q, ka_img.transpose(-2, -1)) * clamp_scale
        trg_local_context, trg_cross_context, trg_cross_mass = _exact_components(
            trg_local_logits,
            trg_cross_logits,
            v_b,
            va_img,
        )
        selected_source_value = va_img.index_select(2, src_index)
        clamped_trg_context = (
            trg_local_context
            + trg_cross_mass.unsqueeze(-1) * float(candidate_value_scale) * selected_source_value
        )
        base_trg_context = trg_local_context + trg_cross_context

        src_delta_heads = clamped_src_context - base_src_context
        trg_delta_heads = clamped_trg_context - base_trg_context
        src_delta = src_delta_heads.permute(0, 2, 1, 3).reshape(
            ensemble_count, end - start, -1
        )
        trg_delta = trg_delta_heads.permute(0, 2, 1, 3).reshape(
            ensemble_count, end - start, -1
        )
        src_delta_output = F.linear(src_delta.float(), attn_weight, bias=None)
        trg_delta_output = F.linear(trg_delta.float(), attn_weight, bias=None)
        base_src_token = base_x_a.index_select(1, src_global).float()
        base_trg_token = base_x_b.index_select(1, trg_global).float()
        patched_src_token = base_src_token + gate_a * src_delta_output
        patched_trg_token = base_trg_token + gate_b * trg_delta_output

        release_qa, release_ka, _release_va, _release_mlp_a, _release_mod_a = _block_qkv(
            release_block,
            patched_src_token.to(dtype=release_weight.dtype),
            a.vec.to(dtype=release_weight.dtype),
        )
        release_qb, release_kb, _release_vb, _release_mlp_b, _release_mod_b = _block_qkv(
            release_block,
            patched_trg_token.to(dtype=release_weight.dtype),
            b.vec.to(dtype=release_weight.dtype),
        )
        post_ab_logits = torch.matmul(
            release_qa.float(), release_kb_img.transpose(-2, -1)
        ) * release_scale
        post_ba_logits = torch.matmul(
            release_qb.float(), release_ka_img.transpose(-2, -1)
        ) * release_scale
        post_candidate_ab = (
            release_qa.float() * release_kb.float()
        ).sum(dim=-1) * release_scale
        post_candidate_ba = (
            release_qb.float() * release_ka.float()
        ).sum(dim=-1) * release_scale
        base_target_logit = torch.gather(
            post_ab_logits,
            3,
            trg_index.reshape(1, 1, -1, 1).expand(
                ensemble_count, head_count, end - start, 1
            ),
        ).squeeze(3)
        base_source_logit = torch.gather(
            post_ba_logits,
            3,
            src_index.reshape(1, 1, -1, 1).expand(
                ensemble_count, head_count, end - start, 1
            ),
        ).squeeze(3)
        post_rank_ab = (
            (post_ab_logits > post_candidate_ab.unsqueeze(-1)).sum(dim=-1)
            - (base_target_logit > post_candidate_ab).long()
            + 1
        ).float()
        post_rank_ba = (
            (post_ba_logits > post_candidate_ba.unsqueeze(-1)).sum(dim=-1)
            - (base_source_logit > post_candidate_ba).long()
            + 1
        ).float()
        if bool((post_rank_ab < 1).any() or (post_rank_ba < 1).any()):
            raise RuntimeError("candidate replacement produced an invalid release rank")

        pre_candidate_ab = torch.gather(
            src_cross_logits,
            3,
            trg_index.reshape(1, 1, -1, 1).expand(
                ensemble_count, head_count, end - start, 1
            ),
        ).squeeze(3)
        pre_candidate_ba = torch.gather(
            trg_cross_logits,
            3,
            src_index.reshape(1, 1, -1, 1).expand(
                ensemble_count, head_count, end - start, 1
            ),
        ).squeeze(3)
        pre_rank_ab = (
            (src_cross_logits > pre_candidate_ab.unsqueeze(-1)).sum(dim=-1) + 1
        ).float()
        pre_rank_ba = (
            (trg_cross_logits > pre_candidate_ba.unsqueeze(-1)).sum(dim=-1) + 1
        ).float()
        pre_score = -0.5 * (
            torch.log(pre_rank_ab.clamp_min(1.0))
            + torch.log(pre_rank_ba.clamp_min(1.0))
        )
        post_score = -0.5 * (
            torch.log(post_rank_ab.clamp_min(1.0))
            + torch.log(post_rank_ba.clamp_min(1.0))
        )
        causal_improvement = post_score - pre_score

        score_chunks["pre_intervention_bidirectional_negative_log_rank"].append(
            pre_score.mean(dim=(0, 1))
        )
        score_chunks["post_release_bidirectional_negative_log_rank"].append(
            post_score.mean(dim=(0, 1))
        )
        score_chunks["post_release_source_negative_log_rank"].append(
            -torch.log(post_rank_ab.clamp_min(1.0)).mean(dim=(0, 1))
        )
        score_chunks["post_release_target_negative_log_rank"].append(
            -torch.log(post_rank_ba.clamp_min(1.0)).mean(dim=(0, 1))
        )
        score_chunks["causal_rank_improvement"].append(
            causal_improvement.mean(dim=(0, 1))
        )
        score_chunks["post_release_mutual_top1_vote"].append(
            ((post_rank_ab <= 1) & (post_rank_ba <= 1)).float().mean(dim=(0, 1))
        )
        score_chunks["post_release_mutual_top5_vote"].append(
            ((post_rank_ab <= 5) & (post_rank_ba <= 5)).float().mean(dim=(0, 1))
        )
        diagnostic_chunks["source_cross_mass"].append(
            src_cross_mass.mean(dim=(0, 1))
        )
        diagnostic_chunks["target_cross_mass"].append(
            trg_cross_mass.mean(dim=(0, 1))
        )
        diagnostic_chunks["source_intervention_relative_l2"].append(
            (gate_a * src_delta_output).norm(dim=-1).div(
                base_src_token.norm(dim=-1).clamp_min(1e-12)
            ).mean(dim=0)
        )
        diagnostic_chunks["target_intervention_relative_l2"].append(
            (gate_b * trg_delta_output).norm(dim=-1).div(
                base_trg_token.norm(dim=-1).clamp_min(1e-12)
            ).mean(dim=0)
        )
        diagnostic_chunks["post_release_score_std"].append(
            post_score.std(dim=(0, 1), unbiased=False)
        )
        diagnostic_chunks["causal_improvement_positive_fraction"].append(
            (causal_improvement > 0).float().mean(dim=(0, 1))
        )

    scores = {
        name: torch.nan_to_num(torch.cat(chunks).reshape(point_count, candidate_count))
        for name, chunks in score_chunks.items()
    }
    diagnostics = {
        name: torch.nan_to_num(torch.cat(chunks).reshape(point_count, candidate_count))
        for name, chunks in diagnostic_chunks.items()
    }
    return {
        "score_names": list(scores.keys()),
        "scores": scores,
        "diagnostics": diagnostics,
        "metadata": {
            "ensemble_size": ensemble_count,
            "head_count": head_count,
            "point_count": point_count,
            "candidate_count": candidate_count,
            "clamp_global_block_index": int(state_a.global_block_index),
            "release_global_block_index": int(state_a.global_block_index) + 1,
            "clamp_context": "exact_local_plus_original_total_cross_mass_times_candidate_value",
            "release_readout": "free_unmodified_bidirectional_cross_qk_rank",
            "candidate_source": "exact_mutual_cross_attention_topk_only",
            "candidate_value_scale": float(candidate_value_scale),
            "full_sequence_replay_per_candidate": False,
            "native_candidate_injected": False,
            "native_fallback_used": False,
            "gt_used_for_scoring": False,
        },
    }


def flux_candidate_counterfactual_fingerprint_probe(
    clamp_block: nn.Module,
    release_block: nn.Module,
    state_a: FluxReplayState,
    state_b: FluxReplayState,
    src_cells: torch.Tensor,
    candidate_cells: torch.Tensor,
    *,
    intervention_scales: Sequence[float] = (0.75, 1.0, 1.25),
) -> dict[str, Any]:
    """Build a candidate-specific bidirectional causal response fingerprint.

    Each scale preserves the original local attention and total cross mass and
    changes only the candidate value contribution.  The returned reciprocity
    signal measures whether source-side and target-side response deltas agree
    across the same intervention doses.  This is an audit only: it never
    changes matcher predictions and never reads target annotations.
    """

    scales = tuple(float(scale) for scale in intervention_scales)
    if len(scales) < 3 or any(scale <= 0.0 for scale in scales):
        raise ValueError("intervention_scales must contain at least three positive values")
    if len(set(scales)) != len(scales):
        raise ValueError("intervention_scales must be distinct")
    if not any(abs(scale - 1.0) < 1e-8 for scale in scales):
        raise ValueError("intervention_scales must include the unmodified scale 1.0")

    probes = [
        flux_candidate_clamped_causal_probe(
            clamp_block,
            release_block,
            state_a,
            state_b,
            src_cells,
            candidate_cells,
            candidate_value_scale=scale,
        )
        for scale in scales
    ]
    source = torch.stack(
        [probe["scores"]["post_release_source_negative_log_rank"] for probe in probes],
        dim=0,
    )
    target = torch.stack(
        [probe["scores"]["post_release_target_negative_log_rank"] for probe in probes],
        dim=0,
    )
    bidirectional = 0.5 * (source + target)
    reference_index = min(range(len(scales)), key=lambda index: abs(scales[index] - 1.0))
    source_delta = source - source[reference_index : reference_index + 1]
    target_delta = target - target[reference_index : reference_index + 1]
    reciprocity_error = (source_delta - target_delta).abs().mean(dim=0)
    response_magnitude = 0.5 * (source_delta.abs() + target_delta.abs()).mean(dim=0)
    mean_bidirectional = bidirectional.mean(dim=0)
    stable_bidirectional = mean_bidirectional - reciprocity_error
    metadata = dict(probes[reference_index].get("metadata", {}))
    metadata.update({
        "intervention_scales": list(scales),
        "reference_scale": float(scales[reference_index]),
        "fingerprint_contract": (
            "preserve_exact_local_contribution_and_original_total_cross_mass; "
            "scale_only_candidate_value; compare_bidirectional_response_curve"
        ),
        "fingerprint_uses_gt": False,
    })
    return {
        "fingerprint_score": stable_bidirectional,
        "fingerprint_mean_bidirectional": mean_bidirectional,
        "fingerprint_reciprocity_error": reciprocity_error,
        "fingerprint_response_magnitude": response_magnitude,
        "fingerprint_source_score_by_scale": source,
        "fingerprint_target_score_by_scale": target,
        "fingerprint_bidirectional_score_by_scale": bidirectional,
        "intervention_scales": scales,
        "metadata": metadata,
    }


def _expand_replay_pe(pe: torch.Tensor, ensemble_size: int, branch_count: int) -> torch.Tensor:
    """Repeat positional state only when it is materialized per ensemble member."""

    if pe.ndim > 0 and int(pe.shape[0]) == int(ensemble_size):
        repeats = (int(branch_count),) + (1,) * (pe.ndim - 1)
        return pe.repeat(*repeats)
    return pe


def _persistent_candidate_single_block(
    block: nn.Module,
    x_a: torch.Tensor,
    vec_a: torch.Tensor,
    pe_a: torch.Tensor,
    text_a: int,
    x_b: torch.Tensor,
    vec_b: torch.Tensor,
    pe_b: torch.Tensor,
    text_b: int,
    source_cells: torch.Tensor,
    target_cells: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Run a block with one independent cross candidate per batch row.

    Local self-attention remains dense over each image sequence.  Only the
    selected source/target image queries receive the candidate value while the
    original full cross-branch mass competes with the complete local softmax
    denominator.  This is the exact candidate-slot intervention used by the
    diagnostic, not a unit attention replacement.
    """

    if x_a.shape[0] != x_b.shape[0]:
        raise ValueError("candidate-slot source and target batches must align")
    batch = int(x_a.shape[0])
    source_cells = source_cells.to(device=x_a.device, dtype=torch.long).flatten()
    target_cells = target_cells.to(device=x_b.device, dtype=torch.long).flatten()
    if source_cells.numel() != batch or target_cells.numel() != batch:
        raise ValueError("candidate-slot cell vectors must align with branch batches")

    q_a, k_a, v_a, mlp_a, mod_a = _block_qkv(block, x_a, vec_a)
    q_b, k_b, v_b, mlp_b, mod_b = _block_qkv(block, x_b, vec_b)
    qra, kra = _apply_rope(q_a, k_a, pe_a)
    qrb, krb = _apply_rope(q_b, k_b, pe_b)
    scale = float(q_a.shape[-1]) ** -0.5
    # Keep dense local self-attention without materializing BxHxLxL logits for
    # every hypothesis branch.  Only the two intervened queries need explicit
    # logits below to recover the exact local/cross normalization.
    local_context_a = F.scaled_dot_product_attention(qra, kra, v_a).float()
    local_context_b = F.scaled_dot_product_attention(qrb, krb, v_b).float()

    batch_index = torch.arange(batch, device=x_a.device)
    src_global = source_cells + int(text_a)
    trg_global = target_cells + int(text_b)
    # FLUX applies RoPE to local self-attention only.  The official joint
    # cross branch uses the unrotated image Q/K tensors, so reusing qra/krb
    # here would create a different operator from the established attention.
    qa_img = q_a[:, :, int(text_a):].float()
    ka_img = k_a[:, :, int(text_a):].float()
    qb_img = q_b[:, :, int(text_b):].float()
    kb_img = k_b[:, :, int(text_b):].float()
    src_query = qa_img[batch_index, :, source_cells, :]
    trg_key = kb_img[batch_index, :, target_cells, :]
    trg_query = qb_img[batch_index, :, target_cells, :]
    src_key = ka_img[batch_index, :, source_cells, :]
    src_cross_logit = (src_query.float() * trg_key.float()).sum(dim=-1) * scale
    trg_cross_logit = (trg_query.float() * src_key.float()).sum(dim=-1) * scale

    src_local_query = qra[batch_index, :, src_global, :].float()
    trg_local_query = qrb[batch_index, :, trg_global, :].float()
    src_local_logits = torch.matmul(
        src_local_query.unsqueeze(2), kra.float().transpose(-2, -1)
    ).squeeze(2) * scale
    trg_local_logits = torch.matmul(
        trg_local_query.unsqueeze(2), krb.float().transpose(-2, -1)
    ).squeeze(2) * scale
    src_cross_logits = torch.matmul(
        src_query.float().unsqueeze(2), kb_img.float().transpose(-2, -1)
    ).squeeze(2) * scale
    trg_cross_logits = torch.matmul(
        trg_query.float().unsqueeze(2), ka_img.float().transpose(-2, -1)
    ).squeeze(2) * scale
    src_log_z = torch.logaddexp(
        torch.logsumexp(src_local_logits, dim=-1),
        torch.logsumexp(src_cross_logits, dim=-1),
    )
    trg_log_z = torch.logaddexp(
        torch.logsumexp(trg_local_logits, dim=-1),
        torch.logsumexp(trg_cross_logits, dim=-1),
    )
    src_local_cond = torch.exp(src_local_logits - src_log_z.unsqueeze(-1))
    trg_local_cond = torch.exp(trg_local_logits - trg_log_z.unsqueeze(-1))
    src_cross_mass = torch.exp(torch.logsumexp(src_cross_logits, dim=-1) - src_log_z)
    trg_cross_mass = torch.exp(torch.logsumexp(trg_cross_logits, dim=-1) - trg_log_z)
    src_selected_local = torch.sum(src_local_cond.unsqueeze(-1) * v_a.float(), dim=2)
    trg_selected_local = torch.sum(trg_local_cond.unsqueeze(-1) * v_b.float(), dim=2)
    src_selected_value = v_b[batch_index, :, trg_global, :].float()
    trg_selected_value = v_a[batch_index, :, src_global, :].float()
    src_selected_context = src_selected_local + src_cross_mass.unsqueeze(-1) * src_selected_value
    trg_selected_context = trg_selected_local + trg_cross_mass.unsqueeze(-1) * trg_selected_value

    context_a = local_context_a.clone()
    context_b = local_context_b.clone()
    context_a[batch_index, :, src_global, :] = src_selected_context
    context_b[batch_index, :, trg_global, :] = trg_selected_context
    merged_a = context_a.permute(0, 2, 1, 3).reshape(batch, x_a.shape[1], -1).to(v_a.dtype)
    merged_b = context_b.permute(0, 2, 1, 3).reshape(batch, x_b.shape[1], -1).to(v_b.dtype)
    output_a = block.linear2(torch.cat((merged_a, block.mlp_act(mlp_a)), dim=2))
    output_b = block.linear2(torch.cat((merged_b, block.mlp_act(mlp_b)), dim=2))
    next_a = x_a + mod_a.gate * output_a
    next_b = x_b + mod_b.gate * output_b
    return next_a, next_b, {
        "source_cross_mass": src_cross_mass.detach(),
        "target_cross_mass": trg_cross_mass.detach(),
        "source_cross_logit": src_cross_logit.detach(),
        "target_cross_logit": trg_cross_logit.detach(),
    }


def flux_persistent_candidate_slot_replay_probe(
    blocks: Sequence[nn.Module],
    state_a: FluxReplayState,
    state_b: FluxReplayState,
    src_cells: torch.Tensor,
    candidate_cells: torch.Tensor,
    *,
    hypothesis_chunk: int = 1,
) -> dict[str, Any]:
    """Audit persistent candidate slots through one or two frozen FLUX blocks.

    Every ``(source point, candidate)`` is an independent branch.  The branch
    keeps the candidate-specific bidirectional cross key at every replayed
    block while retaining full local self-attention and the real FLUX output
    projection/residual/MLP.  This function only audits pair states; it never
    changes matcher predictions and never consumes target annotations.
    """

    if len(blocks) not in (1, 2):
        raise ValueError("persistent candidate-slot replay expects one or two blocks")
    if state_a.global_block_index != state_b.global_block_index:
        raise ValueError("candidate-slot replay states must start from the same block")
    if state_a.ensemble_size != state_b.ensemble_size:
        raise ValueError("candidate-slot replay states must preserve equal ensembles")
    if src_cells.ndim != 1 or candidate_cells.ndim != 2:
        raise ValueError("source cells must be [point] and candidates [point,candidate]")
    if candidate_cells.shape[0] != src_cells.shape[0]:
        raise ValueError("candidate and source cell rows do not align")
    if int(hypothesis_chunk) <= 0:
        raise ValueError("hypothesis_chunk must be positive")

    replay_started = time.perf_counter()
    weight = next(blocks[0].parameters())
    a = _state_to_device(state_a, weight.device, weight.dtype)
    b = _state_to_device(state_b, weight.device, weight.dtype)
    src_cells = src_cells.to(device=weight.device, dtype=torch.long).flatten()
    candidate_cells = candidate_cells.to(device=weight.device, dtype=torch.long)
    source_count = int(a.image_height) * int(a.image_width)
    target_count = int(b.image_height) * int(b.image_width)
    src_cells = src_cells.clamp(0, source_count - 1)
    candidate_cells = candidate_cells.clamp(0, target_count - 1)
    point_count, candidate_count = map(int, candidate_cells.shape)
    if point_count == 0 or candidate_count == 0:
        empty = torch.empty((point_count, candidate_count), device=weight.device)
        return {
            "pair_cosine": empty,
            "directional_anchor_cosine": empty,
            "intervention_gain": empty,
            "native_pair_cosine": empty,
            "source_cross_mass": empty,
            "target_cross_mass": empty,
            "source_state": empty,
            "target_state": empty,
            "metadata": {
                "point_count": point_count,
                "candidate_count": candidate_count,
                "candidate_axis_persisted_across_blocks": True,
                "local_self_attention_preserved": True,
                "original_cross_mass_used": True,
                "unit_cross_attention_forced": False,
                "native_fallback_used": False,
                "gt_used_for_inference": False,
            },
        }

    native_a = run_flux_native_stack(blocks, a)
    native_b = run_flux_native_stack(blocks, b)
    ensemble = int(a.ensemble_size)
    hypothesis_count = point_count * candidate_count
    pair_cosine_chunks: list[torch.Tensor] = []
    directional_anchor_chunks: list[torch.Tensor] = []
    intervention_gain_chunks: list[torch.Tensor] = []
    native_pair_cosine_chunks: list[torch.Tensor] = []
    source_mass_chunks: list[torch.Tensor] = []
    target_mass_chunks: list[torch.Tensor] = []
    source_sketch_chunks: list[torch.Tensor] = []
    target_sketch_chunks: list[torch.Tensor] = []
    source_native_chunks: list[torch.Tensor] = []
    target_native_chunks: list[torch.Tensor] = []
    chunk_count = 0
    source_indices = src_cells[:, None].expand(point_count, candidate_count).reshape(-1)
    target_indices = candidate_cells.reshape(-1)

    sketch_dim = min(64, int(a.x.shape[-1]))
    with torch.no_grad():
        for start in range(0, hypothesis_count, int(hypothesis_chunk)):
            end = min(hypothesis_count, start + int(hypothesis_chunk))
            branch_count = end - start
            branch_src = source_indices[start:end]
            branch_trg = target_indices[start:end]
            x_a = a.x.repeat(branch_count, 1, 1)
            x_b = b.x.repeat(branch_count, 1, 1)
            vec_a = a.vec.repeat(branch_count, 1)
            vec_b = b.vec.repeat(branch_count, 1)
            pe_a = _expand_replay_pe(a.pe, ensemble, branch_count)
            pe_b = _expand_replay_pe(b.pe, ensemble, branch_count)
            branch_src_batch = branch_src.repeat_interleave(ensemble)
            branch_trg_batch = branch_trg.repeat_interleave(ensemble)
            branch_mass_a: torch.Tensor | None = None
            branch_mass_b: torch.Tensor | None = None
            for block in blocks:
                x_a, x_b, current = _persistent_candidate_single_block(
                    block,
                    x_a,
                    vec_a,
                    pe_a,
                    a.text_token_count,
                    x_b,
                    vec_b,
                    pe_b,
                    b.text_token_count,
                    branch_src_batch,
                    branch_trg_batch,
                )
                branch_mass_a = current["source_cross_mass"]
                branch_mass_b = current["target_cross_mass"]
            if branch_mass_a is None or branch_mass_b is None:
                raise RuntimeError("persistent candidate-slot replay executed no block")
            src_global = branch_src_batch + int(a.text_token_count)
            trg_global = branch_trg_batch + int(b.text_token_count)
            batch_index = torch.arange(branch_count * ensemble, device=weight.device)
            src_selected = x_a[batch_index, src_global].reshape(branch_count, ensemble, -1)
            trg_selected = x_b[batch_index, trg_global].reshape(branch_count, ensemble, -1)
            # Native controls must follow each hypothesis' source/target cell.
            # Taking the first E rows here would silently compare every slot to
            # the first candidate in a chunk and corrupt intervention deltas.
            src_selected_native = native_a[:, branch_src + int(a.text_token_count)].permute(1, 0, 2)
            trg_selected_native = native_b[:, branch_trg + int(b.text_token_count)].permute(1, 0, 2)
            src_mean = src_selected.mean(dim=1)
            trg_mean = trg_selected.mean(dim=1)
            src_native_mean = src_selected_native.mean(dim=1)
            trg_native_mean = trg_selected_native.mean(dim=1)
            pair_cosine = F.cosine_similarity(src_mean, trg_mean, dim=-1, eps=1e-12)
            source_to_native_target = F.cosine_similarity(
                src_mean, trg_native_mean, dim=-1, eps=1e-12
            )
            target_to_native_source = F.cosine_similarity(
                trg_mean, src_native_mean, dim=-1, eps=1e-12
            )
            directional_anchor = 0.5 * (source_to_native_target + target_to_native_source)
            native_pair = F.cosine_similarity(
                src_native_mean, trg_native_mean, dim=-1, eps=1e-12
            )
            pair_cosine_chunks.append(pair_cosine.float())
            directional_anchor_chunks.append(directional_anchor.float())
            native_pair_cosine_chunks.append(native_pair.float())
            intervention_gain_chunks.append((directional_anchor - native_pair).float())
            source_mass_chunks.append(
                branch_mass_a.reshape(branch_count, ensemble, -1).mean(dim=(1, 2)).float()
            )
            target_mass_chunks.append(
                branch_mass_b.reshape(branch_count, ensemble, -1).mean(dim=(1, 2)).float()
            )
            source_sketch_chunks.append(
                F.adaptive_avg_pool1d(src_mean.float().unsqueeze(1), sketch_dim).squeeze(1)
            )
            target_sketch_chunks.append(
                F.adaptive_avg_pool1d(trg_mean.float().unsqueeze(1), sketch_dim).squeeze(1)
            )
            source_native_chunks.append(src_selected_native.mean(dim=1).float())
            target_native_chunks.append(trg_selected_native.mean(dim=1).float())
            chunk_count += 1

    source_sketch = torch.cat(source_sketch_chunks, dim=0).reshape(point_count, candidate_count, sketch_dim)
    target_sketch = torch.cat(target_sketch_chunks, dim=0).reshape(point_count, candidate_count, sketch_dim)
    source_mean = F.normalize(source_sketch, dim=-1, eps=1e-12)
    target_mean = F.normalize(target_sketch, dim=-1, eps=1e-12)
    source_gram = torch.matmul(source_mean, source_mean.transpose(-1, -2))
    target_gram = torch.matmul(target_mean, target_mean.transpose(-1, -2))
    off_diag = ~torch.eye(candidate_count, device=weight.device, dtype=torch.bool)
    source_slot_similarity = source_gram[..., off_diag].reshape(point_count, -1).mean(dim=-1)
    target_slot_similarity = target_gram[..., off_diag].reshape(point_count, -1).mean(dim=-1)
    source_native = torch.cat(source_native_chunks, dim=0).reshape(point_count, candidate_count, -1)
    target_native = torch.cat(target_native_chunks, dim=0).reshape(point_count, candidate_count, -1)
    source_native_sketch = F.adaptive_avg_pool1d(
        source_native.float().reshape(-1, 1, source_native.shape[-1]), sketch_dim
    ).reshape(point_count, candidate_count, sketch_dim)
    target_native_sketch = F.adaptive_avg_pool1d(
        target_native.float().reshape(-1, 1, target_native.shape[-1]), sketch_dim
    ).reshape(point_count, candidate_count, sketch_dim)
    source_delta = (source_sketch - source_native_sketch).norm(dim=-1) / source_native_sketch.norm(dim=-1).clamp_min(1e-12)
    target_delta = (target_sketch - target_native_sketch).norm(dim=-1) / target_native_sketch.norm(dim=-1).clamp_min(1e-12)
    elapsed = float(time.perf_counter() - replay_started)
    return {
        "pair_cosine": torch.cat(pair_cosine_chunks, dim=0).reshape(point_count, candidate_count),
        "directional_anchor_cosine": torch.cat(directional_anchor_chunks, dim=0).reshape(
            point_count, candidate_count
        ),
        "intervention_gain": torch.cat(intervention_gain_chunks, dim=0).reshape(
            point_count, candidate_count
        ),
        "native_pair_cosine": torch.cat(native_pair_cosine_chunks, dim=0).reshape(
            point_count, candidate_count
        ),
        "source_cross_mass": torch.cat(source_mass_chunks, dim=0).reshape(point_count, candidate_count),
        "target_cross_mass": torch.cat(target_mass_chunks, dim=0).reshape(point_count, candidate_count),
        "source_state_sketch": source_sketch,
        "target_state_sketch": target_sketch,
        "source_slot_similarity": source_slot_similarity,
        "target_slot_similarity": target_slot_similarity,
        "source_slot_divergence": 1.0 - source_slot_similarity,
        "target_slot_divergence": 1.0 - target_slot_similarity,
        "source_relative_delta": source_delta,
        "target_relative_delta": target_delta,
        "metadata": {
            "point_count": point_count,
            "candidate_count": candidate_count,
            "ensemble_size": ensemble,
            "replay_depth": len(blocks),
            "hypothesis_count": hypothesis_count,
            "hypothesis_chunk": int(hypothesis_chunk),
            "chunk_count": chunk_count,
            "replay_seconds": elapsed,
            "hypotheses_per_second": float(hypothesis_count / max(elapsed, 1e-12)),
            "max_branch_batch": min(int(hypothesis_chunk), hypothesis_count) * ensemble,
            "source_sequence_tokens": int(a.x.shape[1]),
            "target_sequence_tokens": int(b.x.shape[1]),
            "local_self_attention_preserved": True,
            "candidate_axis_persisted_across_blocks": True,
            "original_cross_mass_used": True,
            "cross_mass_denominator": "full_local_plus_full_target_cross",
            "unit_cross_attention_forced": False,
            "candidate_cross_mass_preserved": True,
            "candidate_cross_key_count": 1,
            "slot_state_sketch_dim": sketch_dim,
            "native_fallback_used": False,
            "gt_used_for_inference": False,
        },
    }


def flux_cross_readout_probe(
    blocks: Sequence[nn.Module],
    state_a: FluxReplayState,
    state_b: FluxReplayState,
    src_cells: torch.Tensor,
    candidate_cells: torch.Tensor,
    *,
    mode: str = "exact",
    use_coordinate_bias: bool = False,
) -> dict[str, torch.Tensor]:
    """Probe per-head cross-attention readout signals for selected candidates.

    This is diagnostic-only.  It mirrors the canonical block input used by
    ``run_flux_joint_stack`` and returns parameter-free scores over the supplied
    source/candidate cells without changing replayed features.
    """

    if len(blocks) not in (1, 2):
        raise ValueError("FJSAR readout probe expects one or two interaction blocks")
    if mode not in {"exact", "calibrated"}:
        raise ValueError(f"readout probe supports exact/calibrated joint attention, got {mode}")
    if state_a.global_block_index != state_b.global_block_index:
        raise ValueError("source and target replay states start from different blocks")
    if state_a.ensemble_size != state_b.ensemble_size:
        raise ValueError("source and target replay caches must preserve equal ensemble sizes")
    if candidate_cells.ndim != 2:
        raise ValueError("candidate_cells must have shape [Q,K]")
    if src_cells.ndim != 1 or src_cells.shape[0] != candidate_cells.shape[0]:
        raise ValueError("src_cells must align with candidate_cells rows")

    weight = next(blocks[0].parameters())
    a = _state_to_device(state_a, weight.device, weight.dtype)
    b = _state_to_device(state_b, weight.device, weight.dtype)
    x_a, x_b = a.x, b.x
    cross_bias_ab = None
    cross_bias_ba = None

    if len(blocks) == 2:
        x_a, x_b, first = flux_joint_single_block(
            blocks[0],
            x_a,
            a.vec,
            a.pe,
            a.text_token_count,
            x_b,
            b.vec,
            b.pe,
            b.text_token_count,
            mode=mode,
        )
        if use_coordinate_bias:
            cross_bias_ab, cross_bias_ba, _coordinate = pair_coordinate_bias(
                first,
                a.image_height,
                a.image_width,
                b.image_height,
                b.image_width,
            )
        block = blocks[1]
    else:
        block = blocks[0]

    q_a, k_a, v_a, _mlp_a, _mod_a = _block_qkv(block, x_a, a.vec)
    q_b, k_b, v_b, _mlp_b, _mod_b = _block_qkv(block, x_b, b.vec)
    qa_img = q_a[:, :, a.text_token_count:].float()
    ka_img = k_a[:, :, a.text_token_count:].float()
    va_img = v_a[:, :, a.text_token_count:].float()
    qb_img = q_b[:, :, b.text_token_count:].float()
    kb_img = k_b[:, :, b.text_token_count:].float()
    vb_img = v_b[:, :, b.text_token_count:].float()

    src_cells = src_cells.to(device=qa_img.device, dtype=torch.long).flatten()
    candidate_cells = candidate_cells.to(device=qa_img.device, dtype=torch.long)
    src_count = int(a.image_height) * int(a.image_width)
    trg_count = int(b.image_height) * int(b.image_width)
    src_cells = src_cells.clamp(0, src_count - 1)
    candidate_cells = candidate_cells.clamp(0, trg_count - 1)
    query_count, candidate_count = int(candidate_cells.shape[0]), int(candidate_cells.shape[1])
    if query_count == 0 or candidate_count == 0:
        empty = torch.empty((query_count, candidate_count), device=qa_img.device, dtype=torch.float32)
        return {"score_names": [], "scores": {}, "metadata": {"target_token_count": trg_count}}

    scale = float(q_a.shape[-1]) ** -0.5
    src_q = qa_img.index_select(2, src_cells)
    logits_ab = torch.matmul(src_q, kb_img.transpose(-2, -1)) * scale
    if cross_bias_ab is not None:
        logits_ab = logits_ab + cross_bias_ab.index_select(0, src_cells).to(logits_ab)[None, None]
    prob_ab = torch.softmax(logits_ab, dim=-1)

    source_logits = torch.matmul(src_q, ka_img.transpose(-2, -1)) * scale
    source_prob = torch.softmax(source_logits, dim=-1)
    source_common = torch.matmul(source_prob, va_img)
    source_value = va_img.index_select(2, src_cells)
    source_identity = source_value - source_common

    target_common = torch.matmul(prob_ab, vb_img)
    flat_candidates = candidate_cells.reshape(-1)
    target_values = vb_img.index_select(2, flat_candidates)
    target_values = target_values.reshape(
        vb_img.shape[0],
        vb_img.shape[1],
        query_count,
        candidate_count,
        vb_img.shape[-1],
    )
    target_residual = target_values - target_common.unsqueeze(3)
    target_energy = vb_img.square().sum(dim=-1)
    basin_variance = (
        torch.matmul(prob_ab, target_energy.unsqueeze(-1)).squeeze(-1)
        - target_common.square().sum(dim=-1)
    ).clamp_min(1e-12)
    target_residual_norm = target_residual.norm(dim=-1) / basin_variance.sqrt().unsqueeze(-1)

    source_identity_expanded = source_identity.unsqueeze(3)
    residual_alignment = F.cosine_similarity(
        target_residual,
        source_identity_expanded,
        dim=-1,
        eps=1e-12,
    )
    value_alignment = F.cosine_similarity(
        target_values,
        source_identity_expanded,
        dim=-1,
        eps=1e-12,
    )
    common_similarity = F.cosine_similarity(
        target_values,
        target_common.unsqueeze(3),
        dim=-1,
        eps=1e-12,
    )
    residual_energy = torch.nan_to_num(target_residual_norm.float(), nan=0.0, posinf=0.0, neginf=0.0)

    gather_index = candidate_cells.reshape(1, 1, query_count, candidate_count).expand(
        logits_ab.shape[0],
        logits_ab.shape[1],
        query_count,
        candidate_count,
    )
    candidate_logits = torch.gather(logits_ab, 3, gather_index)
    candidate_probs = torch.gather(prob_ab, 3, gather_index)
    exact_candidate_probs_ab = None
    qra = kra = qrb = krb = None
    if mode == "exact":
        qra, kra = _apply_rope(q_a, k_a, a.pe)
        qrb, krb = _apply_rope(q_b, k_b, b.pe)
        src_local_q = qra.index_select(2, src_cells + int(a.text_token_count))
        local_logits_ab = torch.matmul(src_local_q.float(), kra.float().transpose(-2, -1)) * scale
        combined_log_normalizer_ab = torch.logaddexp(
            torch.logsumexp(local_logits_ab, dim=-1),
            torch.logsumexp(logits_ab, dim=-1),
        )
        exact_candidate_probs_ab = torch.exp(
            candidate_logits - combined_log_normalizer_ab.unsqueeze(-1)
        )
    candidate_ranks = (logits_ab.unsqueeze(3) > candidate_logits.unsqueeze(-1)).sum(dim=-1) + 1
    candidate_ranks = candidate_ranks.float()
    head_top1_vote = (candidate_ranks <= 1).float()
    head_top5_vote = (candidate_ranks <= 5).float()
    negative_log_head_rank = -torch.log(candidate_ranks.clamp_min(1.0))
    logit_gap_to_head_peak = candidate_logits - logits_ab.max(dim=-1, keepdim=True).values

    target_q = qb_img.index_select(2, flat_candidates).reshape(
        qb_img.shape[0],
        qb_img.shape[1],
        query_count,
        candidate_count,
        qb_img.shape[-1],
    )
    reverse_probs = []
    reverse_ranks = []
    exact_reverse_probs = []
    reverse_residual_alignments = []
    reverse_residual_energies = []
    # Keep the exact reverse softmax/rank while bounding the temporary
    # [ensemble, head, query, candidate, source-token] allocation.
    elements_per_candidate = max(
        1,
        int(qb_img.shape[0])
        * int(qb_img.shape[1])
        * query_count
        * int(ka_img.shape[2]),
    )
    reverse_chunk = max(1, min(candidate_count, 16_000_000 // elements_per_candidate))
    reverse_src_index = src_cells.reshape(1, 1, query_count, 1, 1)
    for candidate_start in range(0, candidate_count, reverse_chunk):
        candidate_end = min(candidate_count, candidate_start + reverse_chunk)
        reverse_logits = torch.einsum(
            "bhqkd,bhsd->bhqks",
            target_q[:, :, :, candidate_start:candidate_end],
            ka_img,
        ) * scale
        if cross_bias_ba is not None:
            reverse_bias = cross_bias_ba.index_select(
                0,
                flat_candidates.reshape(-1),
            ).reshape(query_count, candidate_count, -1)
            reverse_logits = reverse_logits + reverse_bias[
                :, candidate_start:candidate_end
            ][None, None].to(reverse_logits)
        gather_src = reverse_src_index.expand(
            reverse_logits.shape[0],
            reverse_logits.shape[1],
            query_count,
            candidate_end - candidate_start,
            1,
        )
        reverse_candidate_logits = torch.gather(reverse_logits, 4, gather_src).squeeze(4)
        reverse_probability = torch.softmax(reverse_logits, dim=-1)
        reverse_probs.append(
            torch.gather(reverse_probability, 4, gather_src).squeeze(4)
        )
        target_q_chunk = target_q[:, :, :, candidate_start:candidate_end]
        target_value_chunk = target_values[:, :, :, candidate_start:candidate_end]
        target_local_logits = torch.einsum(
            "bhqkd,bhtd->bhqkt",
            target_q_chunk,
            kb_img,
        ) * scale
        target_local_probability = torch.softmax(target_local_logits, dim=-1)
        target_local_common = torch.einsum(
            "bhqkt,bhtd->bhqkd",
            target_local_probability,
            vb_img,
        )
        target_identity = target_value_chunk - target_local_common
        reverse_common = torch.einsum(
            "bhqks,bhsd->bhqkd",
            reverse_probability,
            va_img,
        )
        source_residual = source_value.unsqueeze(3) - reverse_common
        reverse_residual_alignments.append(
            F.cosine_similarity(
                target_identity,
                source_residual,
                dim=-1,
                eps=1e-12,
            )
        )
        source_energy = va_img.square().sum(dim=-1)
        reverse_basin_variance = (
            torch.einsum(
                "bhqks,bhs->bhqk",
                reverse_probability,
                source_energy,
            )
            - reverse_common.square().sum(dim=-1)
        ).clamp_min(1e-12)
        reverse_residual_energies.append(
            source_residual.norm(dim=-1) / reverse_basin_variance.sqrt()
        )
        if mode == "exact":
            target_local_q = qrb.index_select(
                2,
                flat_candidates.reshape(-1) + int(b.text_token_count),
            ).reshape(
                qrb.shape[0],
                qrb.shape[1],
                query_count,
                candidate_count,
                qrb.shape[-1],
            )[:, :, :, candidate_start:candidate_end]
            local_reverse_logits = torch.einsum(
                "bhqkd,bhnd->bhqkn",
                target_local_q.float(),
                krb.float(),
            ) * scale
            combined_reverse_normalizer = torch.logaddexp(
                torch.logsumexp(local_reverse_logits, dim=-1),
                torch.logsumexp(reverse_logits, dim=-1),
            )
            exact_reverse_probs.append(
                torch.exp(reverse_candidate_logits - combined_reverse_normalizer)
            )
        reverse_ranks.append(
            ((reverse_logits > reverse_candidate_logits.unsqueeze(-1)).sum(dim=-1) + 1).float()
        )
    candidate_probs_ba = torch.cat(reverse_probs, dim=3)
    candidate_ranks_ba = torch.cat(reverse_ranks, dim=3)
    reverse_residual_alignment = torch.cat(reverse_residual_alignments, dim=3)
    reverse_residual_energy = torch.cat(reverse_residual_energies, dim=3)
    symmetric_residual_alignment = 0.5 * (
        residual_alignment + reverse_residual_alignment
    )
    symmetric_residual_energy = torch.sqrt(
        (residual_energy * reverse_residual_energy).clamp_min(0.0)
    )
    mutual_probability = torch.sqrt(
        (candidate_probs * candidate_probs_ba).clamp_min(0.0)
    )
    bidirectional_negative_log_rank = -0.5 * (
        torch.log(candidate_ranks.clamp_min(1.0))
        + torch.log(candidate_ranks_ba.clamp_min(1.0))
    )
    expert_scores = {
        "forward_probability": torch.nan_to_num(candidate_probs.float()),
        "reverse_probability": torch.nan_to_num(candidate_probs_ba.float()),
        "mutual_probability": torch.nan_to_num(mutual_probability.float()),
        "bidirectional_negative_log_rank": torch.nan_to_num(
            bidirectional_negative_log_rank.float()
        ),
        "bidirectional_top1_vote": (
            (candidate_ranks <= 1) & (candidate_ranks_ba <= 1)
        ).float(),
        "bidirectional_top5_vote": (
            (candidate_ranks <= 5) & (candidate_ranks_ba <= 5)
        ).float(),
        "forward_value_residual_alignment": torch.nan_to_num(
            residual_alignment.float()
        ),
        "reverse_value_residual_alignment": torch.nan_to_num(
            reverse_residual_alignment.float()
        ),
        "symmetric_value_residual_alignment": torch.nan_to_num(
            symmetric_residual_alignment.float()
        ),
        "symmetric_value_residual_energy": torch.nan_to_num(
            symmetric_residual_energy.float()
        ),
        "symmetric_residual_alignment_times_energy": torch.nan_to_num(
            (symmetric_residual_alignment * symmetric_residual_energy).float()
        ),
    }
    if mode == "exact":
        exact_candidate_probs_ba = torch.cat(exact_reverse_probs, dim=3)
        exact_mutual_probability = torch.sqrt(
            (exact_candidate_probs_ab * exact_candidate_probs_ba).clamp_min(0.0)
        )
        expert_scores["exact_mutual_cross_probability"] = torch.nan_to_num(
            exact_mutual_probability.float()
        )
        expert_scores["log_exact_mutual_cross_probability"] = torch.nan_to_num(
            exact_mutual_probability.clamp_min(1e-30).log().float(),
            nan=-100.0,
            posinf=0.0,
            neginf=-100.0,
        )

    def _eh_mean(value: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(value.float(), nan=0.0, posinf=0.0, neginf=0.0).mean(dim=(0, 1))

    scores = {
        "raw_head_probability": _eh_mean(candidate_probs),
        "head_top1_vote": _eh_mean(head_top1_vote),
        "head_top5_vote": _eh_mean(head_top5_vote),
        "negative_log_head_rank": _eh_mean(negative_log_head_rank),
        "logit_gap_to_head_peak": _eh_mean(logit_gap_to_head_peak),
        "value_residual_alignment": _eh_mean(residual_alignment),
        "value_alignment_without_common_removal": _eh_mean(value_alignment),
        "value_residual_energy": _eh_mean(residual_energy),
        "residual_alignment_times_energy": _eh_mean(residual_alignment * residual_energy),
        "symmetric_value_residual_alignment": _eh_mean(
            symmetric_residual_alignment
        ),
        "symmetric_value_residual_energy": _eh_mean(
            symmetric_residual_energy
        ),
        "symmetric_residual_alignment_times_energy": _eh_mean(
            symmetric_residual_alignment * symmetric_residual_energy
        ),
        "negative_common_similarity": _eh_mean(-common_similarity),
    }
    return {
        "score_names": list(scores.keys()),
        "scores": scores,
        "expert_scores": expert_scores,
        "metadata": {
            "ensemble_size": int(qa_img.shape[0]),
            "head_count": int(qa_img.shape[1]),
            "source_token_count": int(src_count),
            "target_token_count": int(trg_count),
            "candidate_count": int(candidate_count),
            "used_coordinate_bias": bool(cross_bias_ab is not None),
            "mode": str(mode),
        },
    }


def flux_candidate_internal_state_probe(
    blocks: Sequence[nn.Module],
    state_a: FluxReplayState,
    state_b: FluxReplayState,
    src_cells: torch.Tensor,
    candidate_cells: torch.Tensor,
    *,
    mode: str = "exact",
    use_coordinate_bias: bool = False,
    include_identity_token_sketches: bool = False,
) -> dict[str, Any]:
    """Export candidate-aligned FLUX state families for held-out probes.

    This function is diagnostic-only.  It preserves the ensemble/head axes in
    QK and value evidence and derives symmetric pair relations from every
    state family available at the cached block boundary.  No annotation enters
    the feature construction and no prediction is changed.
    """

    if len(blocks) not in (1, 2):
        raise ValueError("candidate internal-state probe expects one or two replay blocks")
    if mode not in {"exact", "calibrated"}:
        raise ValueError(f"candidate internal-state probe does not support mode {mode}")
    if state_a.global_block_index != state_b.global_block_index:
        raise ValueError("candidate internal-state replay states start from different blocks")
    if state_a.ensemble_size != state_b.ensemble_size:
        raise ValueError("candidate internal-state replay states must preserve equal ensembles")
    if candidate_cells.ndim != 2 or src_cells.ndim != 1:
        raise ValueError("candidate cells must be [point,candidate] and source cells [point]")
    if candidate_cells.shape[0] != src_cells.shape[0]:
        raise ValueError("candidate and source cell rows do not align")

    readout = flux_cross_readout_probe(
        blocks,
        state_a,
        state_b,
        src_cells,
        candidate_cells,
        mode=mode,
        use_coordinate_bias=use_coordinate_bias,
    )
    weight = next(blocks[0].parameters())
    a = _state_to_device(state_a, weight.device, weight.dtype)
    b = _state_to_device(state_b, weight.device, weight.dtype)
    src_cells = src_cells.to(device=weight.device, dtype=torch.long).flatten()
    candidate_cells = candidate_cells.to(device=weight.device, dtype=torch.long)
    source_count = int(a.image_height) * int(a.image_width)
    target_count = int(b.image_height) * int(b.image_width)
    src_cells = src_cells.clamp(0, source_count - 1)
    candidate_cells = candidate_cells.clamp(0, target_count - 1)

    # Match flux_cross_readout_probe: for a two-block replay, state relations
    # are measured at the input of the second (matcher-facing) block.
    x_a, x_b = a.x, b.x
    if len(blocks) == 2:
        x_a, x_b, _first = flux_joint_single_block(
            blocks[0],
            x_a,
            a.vec,
            a.pe,
            a.text_token_count,
            x_b,
            b.vec,
            b.pe,
            b.text_token_count,
            mode=mode,
        )
        block = blocks[1]
    else:
        block = blocks[0]

    q_a, k_a, v_a, mlp_a, _mod_a = _block_qkv(block, x_a, a.vec)
    q_b, k_b, v_b, mlp_b, _mod_b = _block_qkv(block, x_b, b.vec)
    native_a = manual_flux_single_block(block, x_a, a.vec, a.pe)
    native_b = manual_flux_single_block(block, x_b, b.vec, b.pe)
    joint_a, joint_b, _joint = flux_joint_single_block(
        block,
        x_a,
        a.vec,
        a.pe,
        a.text_token_count,
        x_b,
        b.vec,
        b.pe,
        b.text_token_count,
        mode=mode,
    )

    def _image(value: torch.Tensor, text_count: int) -> torch.Tensor:
        return value[..., int(text_count):, :].float()

    def _pair_relations(
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Return [point,candidate,prefix...,3] cosine/norm/distance relations."""

        source_selected = source.index_select(-2, src_cells)
        flat_candidates = candidate_cells.reshape(-1)
        target_selected = target.index_select(-2, flat_candidates).reshape(
            *target.shape[:-2],
            int(candidate_cells.shape[0]),
            int(candidate_cells.shape[1]),
            int(target.shape[-1]),
        )
        source_selected = source_selected.unsqueeze(-2)
        source_norm = source_selected.norm(dim=-1).clamp_min(1e-12)
        target_norm = target_selected.norm(dim=-1).clamp_min(1e-12)
        cosine = F.cosine_similarity(
            source_selected,
            target_selected,
            dim=-1,
            eps=1e-12,
        )
        log_norm_ratio = torch.log(source_norm / target_norm)
        relative_l2 = (source_selected - target_selected).norm(dim=-1) / (
            source_norm + target_norm
        ).clamp_min(1e-12)
        prefix_dims = list(range(cosine.ndim - 2))
        point_dim = cosine.ndim - 2
        candidate_dim = cosine.ndim - 1
        relation = torch.stack((cosine, log_norm_ratio, relative_l2), dim=-1)
        relation = relation.permute(point_dim, candidate_dim, *prefix_dims, cosine.ndim)
        return torch.nan_to_num(
            relation.reshape(
                int(candidate_cells.shape[0]),
                int(candidate_cells.shape[1]),
                -1,
            ).float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    qk_names = (
        "forward_probability",
        "reverse_probability",
        "mutual_probability",
        "bidirectional_negative_log_rank",
        "bidirectional_top1_vote",
        "bidirectional_top5_vote",
        "exact_mutual_cross_probability",
        "log_exact_mutual_cross_probability",
    )
    value_names = (
        "forward_value_residual_alignment",
        "reverse_value_residual_alignment",
        "symmetric_value_residual_alignment",
        "symmetric_value_residual_energy",
        "symmetric_residual_alignment_times_energy",
    )

    def _expert_family(names: Sequence[str]) -> tuple[torch.Tensor, list[str]]:
        tensors = []
        kept_names = []
        for name in names:
            value = readout["expert_scores"].get(name)
            if value is None:
                continue
            # [ensemble,head,point,candidate] -> [point,candidate,expert]
            tensors.append(value.permute(2, 3, 0, 1).reshape(
                int(candidate_cells.shape[0]),
                int(candidate_cells.shape[1]),
                -1,
            ))
            kept_names.append(str(name))
        if not tensors:
            empty = torch.empty(
                (*candidate_cells.shape, 0),
                device=weight.device,
                dtype=torch.float32,
            )
            return empty, kept_names
        return torch.cat(tensors, dim=2).float(), kept_names

    qk_features, kept_qk_names = _expert_family(qk_names)
    value_features, kept_value_names = _expert_family(value_names)
    aggregate_names = list(readout["score_names"])
    aggregate_features = torch.stack(
        [readout["scores"][name] for name in aggregate_names],
        dim=2,
    ).float()
    token_parts = [
        _pair_relations(_image(q_a, a.text_token_count), _image(q_b, b.text_token_count)),
        _pair_relations(_image(k_a, a.text_token_count), _image(k_b, b.text_token_count)),
        _pair_relations(_image(v_a, a.text_token_count), _image(v_b, b.text_token_count)),
        _pair_relations(
            _image(block.mlp_act(mlp_a), a.text_token_count),
            _image(block.mlp_act(mlp_b), b.text_token_count),
        ),
        _pair_relations(_image(x_a, a.text_token_count), _image(x_b, b.text_token_count)),
        _pair_relations(_image(native_a, a.text_token_count), _image(native_b, b.text_token_count)),
        _pair_relations(_image(joint_a, a.text_token_count), _image(joint_b, b.text_token_count)),
    ]
    token_names = [
        "q_relation",
        "k_relation",
        "v_relation",
        "mlp_relation",
        "block_input_relation",
        "native_block_output_relation",
        "joint_block_output_relation",
    ]
    token_features = torch.cat(token_parts, dim=2)

    def _flatten_channel_sequence(value: torch.Tensor) -> torch.Tensor:
        if value.ndim == 4:
            # [ensemble,head,token,channel] -> [token,head*channel]
            value = value.float().nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
            return value.mean(dim=0).permute(1, 0, 2).reshape(value.shape[2], -1)
        if value.ndim == 3:
            # [ensemble,token,channel] -> [token,channel]
            value = value.float().nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
            return value.mean(dim=0)
        raise ValueError(f"unsupported channel-state rank {value.ndim}")

    def _pair_channel_sketch(
        source: torch.Tensor,
        target: torch.Tensor,
        *,
        seed: int,
        bucket_count: int = 128,
    ) -> torch.Tensor:
        source_flat = _flatten_channel_sequence(source)
        target_flat = _flatten_channel_sequence(target)
        if source_flat.shape[1] != target_flat.shape[1]:
            raise ValueError("source/target channel states do not align")
        source_selected = source_flat.index_select(0, src_cells).unsqueeze(1)
        target_selected = target_flat.index_select(
            0,
            candidate_cells.reshape(-1),
        ).reshape(
            int(candidate_cells.shape[0]),
            int(candidate_cells.shape[1]),
            int(target_flat.shape[1]),
        )
        # BF16/FP16 replay can contain rare extreme MLP channels.  The sketch
        # is a bounded diagnostic relation, so clip only this auxiliary view;
        # QK/V/token scalar families remain untouched and are audited separately.
        source_selected = source_selected.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0e4, 1.0e4)
        target_selected = target_selected.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0e4, 1.0e4)
        channel_count = int(source_flat.shape[1])
        channel_index = torch.arange(
            channel_count,
            device=source_flat.device,
            dtype=torch.long,
        )
        buckets = (channel_index * 1103515245 + int(seed) * 12345).remainder(
            int(bucket_count)
        )
        signs = torch.where(
            (channel_index * 214013 + int(seed) * 2531011).remainder(2) == 0,
            1.0,
            -1.0,
        ).to(source_flat)
        gather_buckets = buckets.reshape(1, 1, -1).expand(
            int(candidate_cells.shape[0]),
            int(candidate_cells.shape[1]),
            -1,
        )
        sketches = []
        for relation in (
            source_selected * target_selected,
            (source_selected - target_selected).abs(),
        ):
            output = torch.zeros(
                (
                    int(candidate_cells.shape[0]),
                    int(candidate_cells.shape[1]),
                    int(bucket_count),
                ),
                device=source_flat.device,
                dtype=torch.float32,
            )
            output.scatter_add_(
                2,
                gather_buckets,
                relation.float().nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
                * signs.reshape(1, 1, -1),
            )
            sketches.append(output / math.sqrt(float(max(1, channel_count))))
        return torch.cat(sketches, dim=2)

    channel_states = (
        ("q", _image(q_a, a.text_token_count), _image(q_b, b.text_token_count)),
        ("k", _image(k_a, a.text_token_count), _image(k_b, b.text_token_count)),
        ("v", _image(v_a, a.text_token_count), _image(v_b, b.text_token_count)),
        (
            "mlp",
            _image(block.mlp_act(mlp_a), a.text_token_count),
            _image(block.mlp_act(mlp_b), b.text_token_count),
        ),
        ("block_input", _image(x_a, a.text_token_count), _image(x_b, b.text_token_count)),
        ("native_output", _image(native_a, a.text_token_count), _image(native_b, b.text_token_count)),
        ("joint_output", _image(joint_a, a.text_token_count), _image(joint_b, b.text_token_count)),
    )
    channel_sketch_parts = [
        _pair_channel_sketch(source, target, seed=index + 1)
        for index, (_name, source, target) in enumerate(channel_states)
    ]
    channel_sketch_features = torch.cat(channel_sketch_parts, dim=2).nan_to_num(
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).clamp(-60000.0, 60000.0)
    channel_sketch_names = [
        f"{name}_{relation}_countsketch128"
        for name, _source, _target in channel_states
        for relation in ("product", "absolute_difference")
    ]

    def _identity_token_sketch_pair(
        source: torch.Tensor,
        target: torch.Tensor,
        *,
        seed: int,
        bucket_count: int = 64,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project absolute source/target token states into one shared space."""

        source_flat = _flatten_channel_sequence(source)
        target_flat = _flatten_channel_sequence(target)
        if source_flat.shape[1] != target_flat.shape[1]:
            raise ValueError("source/target identity token states do not align")
        source_selected = source_flat.index_select(0, src_cells)
        target_selected = target_flat.index_select(
            0,
            candidate_cells.reshape(-1),
        ).reshape(
            int(candidate_cells.shape[0]),
            int(candidate_cells.shape[1]),
            int(target_flat.shape[1]),
        )
        channel_count = int(source_flat.shape[1])
        channel_index = torch.arange(
            channel_count,
            device=source_flat.device,
            dtype=torch.long,
        )
        buckets = (channel_index * 1103515245 + int(seed) * 12345).remainder(
            int(bucket_count)
        )
        signs = torch.where(
            (channel_index * 214013 + int(seed) * 2531011).remainder(2) == 0,
            1.0,
            -1.0,
        ).to(source_flat)

        def _sketch(value: torch.Tensor) -> torch.Tensor:
            output = torch.zeros(
                (*value.shape[:-1], int(bucket_count)),
                device=value.device,
                dtype=torch.float32,
            )
            gather = buckets.reshape(*([1] * (value.ndim - 1)), -1).expand_as(value)
            output.scatter_add_(
                -1,
                gather,
                value.float().nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
                * signs.reshape(*([1] * (value.ndim - 1)), -1),
            )
            output = output / math.sqrt(float(max(1, channel_count)))
            return F.normalize(output, dim=-1, eps=1.0e-6)

        source_sketch = _sketch(source_selected)
        target_sketch = _sketch(target_selected)
        source_sketch = source_sketch.unsqueeze(1).expand(
            -1,
            int(candidate_cells.shape[1]),
            -1,
        )
        return source_sketch, target_sketch

    feature_groups = {
        "attention_aggregate": aggregate_features,
        "qk_expert": qk_features,
        "value_expert": value_features,
        "token_state": token_features,
        "channel_state_sketch": channel_sketch_features,
    }
    identity_state_names = [name for name, _source, _target in channel_states]
    if include_identity_token_sketches:
        source_identity_parts = []
        candidate_identity_parts = []
        for index, (_name, source, target) in enumerate(channel_states):
            source_part, target_part = _identity_token_sketch_pair(
                source,
                target,
                seed=index + 1,
            )
            source_identity_parts.append(source_part)
            candidate_identity_parts.append(target_part)
        feature_groups.update({
            "source_identity_token_sketch": torch.cat(source_identity_parts, dim=2),
            "candidate_identity_token_sketch": torch.cat(candidate_identity_parts, dim=2),
        })
    return {
        "feature_groups": {
            name: torch.nan_to_num(value.float(), nan=0.0, posinf=0.0, neginf=0.0)
            for name, value in feature_groups.items()
        },
        "feature_family_names": {
            "attention_aggregate": aggregate_names,
            "qk_expert": kept_qk_names,
            "value_expert": kept_value_names,
            "token_state": token_names,
            "channel_state_sketch": channel_sketch_names,
            **({
                "source_identity_token_sketch": identity_state_names,
                "candidate_identity_token_sketch": identity_state_names,
            } if include_identity_token_sketches else {}),
        },
        "metadata": {
            **readout["metadata"],
            "global_block_index": int(state_a.global_block_index),
            "relation_statistics": ["cosine", "log_norm_ratio", "relative_l2"],
            "channel_relation_sketch": {
                "type": "fixed_countsketch",
                "buckets_per_relation": 128,
                "relations": ["elementwise_product", "absolute_difference"],
                "ensemble_reduction": "mean_before_candidate_pairing",
                "gt_dependent": False,
            },
            "candidate_source": "mutual_cross_attention_topk_only",
            "identity_token_sketch": {
                "enabled": bool(include_identity_token_sketches),
                "states": identity_state_names if include_identity_token_sketches else [],
                "buckets_per_state": 64 if include_identity_token_sketches else 0,
                "shared_source_target_projection": True,
                "gt_dependent": False,
            },
            "gt_used_for_features": False,
            "prediction_changed": False,
            "native_fallback_used": False,
        },
    }


def native_image_tokens(sequence: torch.Tensor, state: FluxReplayState, *, average_ensemble: bool = False) -> torch.Tensor:
    count = state.image_height * state.image_width
    start = state.text_token_count
    result = sequence[:, start : start + count]
    if result.shape[1] != count:
        raise ValueError("replayed sequence does not contain the expected image token grid")
    return result.mean(dim=0, keepdim=True) if average_ensemble else result


def parity_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    reference = reference.float()
    candidate = candidate.float()
    if reference.shape != candidate.shape:
        raise ValueError(f"parity tensors differ in shape: {reference.shape} vs {candidate.shape}")
    difference = candidate - reference
    denom = reference.norm().clamp_min(1e-12)
    cosine = F.cosine_similarity(reference.flatten(), candidate.flatten(), dim=0, eps=1e-12)
    return {
        "max_abs_error": float(difference.abs().max().detach().cpu()),
        "mean_abs_error": float(difference.abs().mean().detach().cpu()),
        "relative_l2_error": float((difference.norm() / denom).detach().cpu()),
        "cosine": float(cosine.detach().cpu()),
    }


def native_parity_error(block: nn.Module, state: FluxReplayState) -> dict[str, float]:
    """Compare the manual formula with the official block on every member."""

    weight = next(block.parameters())
    local = _state_to_device(state, weight.device, weight.dtype)
    with torch.no_grad():
        official = block(local.x, local.vec, local.pe)
        manual = manual_flux_single_block(block, local.x, local.vec, local.pe)
    return parity_metrics(official, manual)

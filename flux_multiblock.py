"""Model-independent helpers for aggregating Flux multi-block outputs."""

from __future__ import annotations

import hashlib

import torch


def deterministic_image_seed(base_seed: int, category: str, image_name: str) -> int:
    """Derive a resume-stable per-image diffusion seed."""
    digest = hashlib.sha256(
        f"{base_seed}:{category}:{image_name}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little") % (2**31)


def aggregate_single_block_outputs(model_output, block_indices, latent_h, latent_w):
    """Aggregate flat ``feature, modulation`` outputs for selected single blocks."""
    blocks = tuple(sorted(set(int(value) for value in block_indices)))
    if not blocks or any(block < 19 for block in blocks):
        raise ValueError("multi-block extraction currently supports single blocks >= 19")
    if len(model_output) != 2 * len(blocks):
        raise ValueError(
            f"Expected {2 * len(blocks)} model outputs for blocks {blocks}, "
            f"got {len(model_output)}"
        )

    grid_h, grid_w = latent_h // 2, latent_w // 2
    aggregated = {}
    for position, block in enumerate(blocks):
        feature = model_output[2 * position]
        modulation = model_output[2 * position + 1]
        if feature.ndim != 3 or feature.shape[1] != grid_h * grid_w:
            raise ValueError(
                f"Block {block} feature shape {tuple(feature.shape)} is incompatible "
                f"with latent grid {(latent_h, latent_w)}"
            )
        feature = feature.reshape(
            feature.shape[0], grid_h, grid_w, feature.shape[2]
        )
        feature = feature.permute(0, 3, 1, 2).mean(0, keepdim=True)
        mod = [
            modulation.shift.mean(0, keepdim=True),
            modulation.scale.mean(0, keepdim=True),
            modulation.gate.mean(0, keepdim=True),
        ]
        aggregated[block] = (feature, torch.cat(mod, dim=1))
    return aggregated


def aggregate_mixed_block_outputs(model_output, block_indices, latent_h, latent_w):
    """Aggregate FLUX double- and single-stream ``forward_feat`` outputs.

    Double-stream blocks return one image feature (already AdaLN-modulated),
    while single-stream blocks return a feature followed by its modulation.
    Keeping the distinction explicit prevents fabricating AdaLN state for the
    pre-single-stream identity branch.
    """
    blocks = tuple(sorted(set(int(value) for value in block_indices)))
    if not blocks or any(block < 0 for block in blocks):
        raise ValueError("block_indices must be non-empty and non-negative")
    expected = sum(1 if block < 19 else 2 for block in blocks)
    if len(model_output) != expected:
        raise ValueError(
            f"Expected {expected} mixed model outputs for blocks {blocks}, "
            f"got {len(model_output)}"
        )

    grid_h, grid_w = latent_h // 2, latent_w // 2
    aggregated = {}
    offset = 0
    for block in blocks:
        feature = model_output[offset]
        offset += 1
        if feature.ndim != 3 or feature.shape[1] != grid_h * grid_w:
            raise ValueError(
                f"Block {block} feature shape {tuple(feature.shape)} is incompatible "
                f"with latent grid {(latent_h, latent_w)}"
            )
        feature = feature.reshape(feature.shape[0], grid_h, grid_w, feature.shape[2])
        feature = feature.permute(0, 3, 1, 2).mean(0, keepdim=True)
        if block < 19:
            aggregated[block] = (feature, None)
            continue
        modulation = model_output[offset]
        offset += 1
        mod = [
            modulation.shift.mean(0, keepdim=True),
            modulation.scale.mean(0, keepdim=True),
            modulation.gate.mean(0, keepdim=True),
        ]
        aggregated[block] = (feature, torch.cat(mod, dim=1))
    return aggregated

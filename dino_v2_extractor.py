"""Frozen DINOv2 block-token extractor used by the candidate audit."""

from __future__ import annotations

from typing import Any, Protocol, Sequence

import torch
from PIL import Image

from dino_v2_spair import DINOConfig, preprocess_square_canvas, tokens_to_patch_map


class DINOExtractor(Protocol):
    def __call__(self, image: Image.Image) -> torch.Tensor: ...

    def extract_batch(
        self, images: Sequence[Image.Image], image_size: int
    ) -> torch.Tensor: ...

    def close(self) -> None: ...


class BlockTokenExtractor:
    """Extract pre-final-norm tokens from official DINOv2 ViT-B/14."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: DINOConfig,
        device: str,
        precision: str,
    ):
        embed_dim = int(getattr(model, "embed_dim", 0))
        blocks = getattr(model, "blocks", None)
        patch_size = getattr(getattr(model, "patch_embed", None), "patch_size", None)
        patch_size = patch_size[0] if isinstance(patch_size, tuple) else patch_size
        if embed_dim != 768 or blocks is None or len(blocks) != 12 or int(patch_size or 0) != 14:
            raise ValueError("Loaded torch.hub model is not DINOv2 ViT-B/14")
        if not 0 <= config.layer < len(blocks):
            raise ValueError(f"Model does not expose DINOv2 block {config.layer}")
        self.model = model.to(device).eval()
        self.config = config
        self.device = torch.device(device)
        self.precision = precision
        self._tokens: torch.Tensor | None = None
        self._hook = blocks[config.layer].register_forward_hook(self._capture)

    def _capture(self, _module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        self._tokens = output[0] if isinstance(output, (tuple, list)) else output

    @torch.inference_mode()
    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self.extract_batch((image,), self.config.image_size)[0]

    @torch.inference_mode()
    def extract_batch(
        self, images: Sequence[Image.Image], image_size: int
    ) -> torch.Tensor:
        if not images:
            grid_size = int(image_size) // self.config.patch_size
            return torch.empty((0, 768, grid_size, grid_size), dtype=torch.float32)
        if int(image_size) <= 0 or int(image_size) % self.config.patch_size:
            raise ValueError("DINOv2 batch image size must be a positive multiple of 14")
        pixels = torch.stack(
            [preprocess_square_canvas(image, int(image_size)) for image in images]
        ).to(self.device)
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(self.precision)
        enabled = dtype is not None and self.device.type == "cuda"
        with torch.autocast(device_type=self.device.type, dtype=dtype, enabled=enabled):
            self._tokens = None
            self.model(pixels)
        if self._tokens is None:
            raise RuntimeError(f"DINOv2 block-{self.config.layer} hook produced no tokens")
        grid_size = int(image_size) // self.config.patch_size
        feature = tokens_to_patch_map(self._tokens, grid_size).float().cpu()
        self._tokens = None
        return feature

    def close(self) -> None:
        self._hook.remove()


class TransformersExtractor:
    """Hugging Face DINOv2 extractor with the same block-token protocol."""

    def __init__(
        self,
        config: DINOConfig,
        device: str,
        precision: str,
        local_files_only: bool,
        model_name_or_path: str,
    ):
        try:
            from transformers import Dinov2Model
        except ImportError as exc:
            raise RuntimeError("The transformers backend requires transformers") from exc
        self.model = Dinov2Model.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
        )
        model_config = self.model.config
        if (
            int(model_config.hidden_size) != 768
            or int(model_config.num_hidden_layers) != 12
            or int(model_config.patch_size) != 14
        ):
            raise ValueError("Loaded Transformers model is not DINOv2 ViT-B/14")
        layers = self.model.encoder.layer
        if not 0 <= config.layer < len(layers):
            raise ValueError(f"Model does not expose DINOv2 block {config.layer}")
        self.config = config
        self.device = torch.device(device)
        self.precision = precision
        self._tokens: torch.Tensor | None = None
        self._hook = layers[config.layer].register_forward_hook(self._capture)
        self.model.to(self.device).eval()

    def _capture(self, _module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        self._tokens = output[0] if isinstance(output, (tuple, list)) else output

    @torch.inference_mode()
    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self.extract_batch((image,), self.config.image_size)[0]

    @torch.inference_mode()
    def extract_batch(
        self, images: Sequence[Image.Image], image_size: int
    ) -> torch.Tensor:
        if not images:
            grid_size = int(image_size) // self.config.patch_size
            return torch.empty((0, 768, grid_size, grid_size), dtype=torch.float32)
        if int(image_size) <= 0 or int(image_size) % self.config.patch_size:
            raise ValueError("DINOv2 batch image size must be a positive multiple of 14")
        pixels = torch.stack(
            [preprocess_square_canvas(image, int(image_size)) for image in images]
        ).to(self.device)
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(self.precision)
        enabled = dtype is not None and self.device.type == "cuda"
        with torch.autocast(device_type=self.device.type, dtype=dtype, enabled=enabled):
            self._tokens = None
            self.model(
                pixel_values=pixels,
                return_dict=True,
            )
        if self._tokens is None:
            raise RuntimeError(f"DINOv2 block-{self.config.layer} hook produced no tokens")
        grid_size = int(image_size) // self.config.patch_size
        feature = tokens_to_patch_map(self._tokens, grid_size).float().cpu()
        self._tokens = None
        return feature

    def close(self) -> None:
        self._hook.remove()


def build_dino_extractor(args, config: DINOConfig) -> DINOExtractor:
    if args.dino_backend == "transformers":
        return TransformersExtractor(
            config,
            args.device,
            args.dino_precision,
            args.dino_local_files_only,
            args.dino_model_repo or config.model_name,
        )
    source = "local" if args.dino_model_repo else "github"
    repo = args.dino_model_repo or "facebookresearch/dinov2"
    model = torch.hub.load(
        repo,
        config.hub_model,
        source=source,
        pretrained=True,
    )
    return BlockTokenExtractor(
        model,
        config,
        args.device,
        args.dino_precision,
    )

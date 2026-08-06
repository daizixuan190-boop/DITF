"""Extract the official DiTF FLUX descriptor without persistent caching."""

from __future__ import annotations

import os
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import PILToTensor


def _detach_cpu_nested(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, list):
        return [_detach_cpu_nested(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_detach_cpu_nested(item) for item in value)
    return value


def nested_tensor_nbytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(nested_tensor_nbytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(nested_tensor_nbytes(item) for item in value)
    return 0


def strip_flux_replay_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Keep the native tensors while releasing a large replay state.

    Full-set spectral extraction only needs replay tensors until all pairs in
    the current category have been factorized.  The native feature and AdaLN
    tensors remain reusable later, after FLUX has been released and RoMa is
    loaded.
    """

    if "feature" not in entry or "ada" not in entry:
        raise ValueError("FLUX replay entry must contain feature and ada fields")
    return {
        "feature": entry["feature"],
        "ada": entry["ada"],
    }


def load_flux_image_tensor(
    image_path: str,
    img_size: Any,
    *,
    horizontal_flip: bool = False,
    image_transform: Callable[[Image.Image], Image.Image] | None = None,
) -> tuple[torch.Tensor, int, int]:
    """Load one image with the official DiTF resize and optional hflip."""

    with Image.open(image_path) as opened:
        image = opened.copy()
    in_h, in_w = np.asarray(image).shape[:2]
    resize_scale = float(img_size[0]) / float(max(in_h, in_w))
    pixel_h = int(round(in_h * resize_scale / 16.0)) * 16
    pixel_w = int(round(in_w * resize_scale / 16.0)) * 16
    if bool(horizontal_flip):
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    image = image.resize((pixel_w, pixel_h))
    if image_transform is not None:
        image = image_transform(image)
        if not isinstance(image, Image.Image):
            raise TypeError("image_transform must return a PIL image")
        if image.size != (pixel_w, pixel_h):
            raise ValueError(
                "image_transform must preserve the resized image dimensions: "
                f"{image.size} != {(pixel_w, pixel_h)}"
            )
    image_tensor = (PILToTensor()(image) / 255.0 - 0.5) * 2.0
    return image_tensor, pixel_h, pixel_w


@torch.inference_mode()
def extract_flux_native_entry(
    featurizer: Any,
    args: Any,
    *,
    dataset_path: str,
    category: str,
    image_name: str,
    caption: str,
    horizontal_flip: bool = False,
) -> dict[str, Any]:
    """Return one CPU-resident native feature/AdaLN entry and write no files."""

    image_path = os.path.join(
        dataset_path, "JPEGImages", category, image_name
    )
    image_tensor, pixel_h, pixel_w = load_flux_image_tensor(
        image_path,
        args.img_size,
        horizontal_flip=horizontal_flip,
    )
    feature, ada = featurizer.forward(
        args,
        image_tensor,
        caption=caption,
        category=category,
        timestep=args.t,
        block_idx=args.k,
        ensemble_size=args.ensemble_size,
    )
    actual_grid = tuple(map(int, feature.shape[-2:]))
    expected_grid = (pixel_h // 16, pixel_w // 16)
    if actual_grid != expected_grid:
        raise RuntimeError(
            f"FLUX extraction grid for {category}/{image_name} is {actual_grid}, "
            f"expected {expected_grid}"
        )
    return {
        "feature": feature.detach().cpu(),
        "ada": _detach_cpu_nested(ada),
    }

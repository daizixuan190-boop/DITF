"""Training-only single-view 3D geometry extraction for FLUX distillation."""

from __future__ import annotations

import gc
import math
from pathlib import Path
from typing import Hashable, Mapping

import torch
from tqdm import tqdm


def crop_vggt_geometry_to_original(
    point_map: torch.Tensor,
    original_coordinates: torch.Tensor,
) -> torch.Tensor:
    """Remove square preprocessing padding using VGGT's returned coordinates."""

    if point_map.ndim != 3 or int(point_map.shape[-1]) != 3:
        raise ValueError("VGGT point map must be [H,W,3]")
    coordinates = original_coordinates.flatten().float()
    if int(coordinates.numel()) < 4:
        raise ValueError("VGGT original coordinates must contain x1,y1,x2,y2")
    height, width = map(int, point_map.shape[:2])
    x1 = max(0, min(width, int(math.floor(float(coordinates[0])))))
    y1 = max(0, min(height, int(math.floor(float(coordinates[1])))))
    x2 = max(0, min(width, int(math.ceil(float(coordinates[2])))))
    y2 = max(0, min(height, int(math.ceil(float(coordinates[3])))))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("VGGT original-image crop is empty")
    return point_map[y1:y2, x1:x2].contiguous()


def _vggt_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    major, _minor = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if major >= 8 else torch.float16


def extract_vggt_geometry_maps(
    image_paths: Mapping[Hashable, str | Path],
    *,
    device: torch.device,
    model_name: str = "facebook/VGGT-1B",
    input_size: int = 518,
) -> dict[Hashable, torch.Tensor]:
    """Extract cropped point maps to CPU memory, then release VGGT completely."""

    if not image_paths:
        raise ValueError("VGGT geometry extraction requires at least one image")
    try:
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images_square
    except ImportError as error:
        raise RuntimeError(
            "VGGT is unavailable. Install the official package with "
            "`pip install git+https://github.com/facebookresearch/vggt.git`."
        ) from error

    model = VGGT.from_pretrained(str(model_name)).to(device).eval()
    dtype = _vggt_dtype(device)
    geometry_maps: dict[Hashable, torch.Tensor] = {}
    try:
        for key, raw_path in tqdm(
            sorted(image_paths.items(), key=lambda item: str(item[0])),
            desc="extract VGGT geometry",
        ):
            path = Path(raw_path)
            if not path.is_file():
                raise FileNotFoundError(f"VGGT image does not exist: {path}")
            images, coordinates = load_and_preprocess_images_square(
                [str(path)], target_size=int(input_size)
            )
            images = images.to(device=device)
            with torch.no_grad(), torch.autocast(
                device_type=device.type,
                dtype=dtype,
                enabled=device.type == "cuda",
            ):
                predictions = model(images)
            point_map = predictions.get("world_points")
            if not isinstance(point_map, torch.Tensor) or point_map.ndim != 5:
                raise RuntimeError("VGGT did not return [B,S,H,W,3] world_points")
            cropped = crop_vggt_geometry_to_original(
                point_map[0, 0].detach().float().cpu(),
                coordinates[0].cpu(),
            )
            geometry_maps[key] = cropped.to(torch.float16)
            del images, coordinates, predictions, point_map, cropped
    finally:
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return geometry_maps

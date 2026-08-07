"""Pure helpers for the Pixal3D pixel-to-face interface audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


def _positive_size(name: str, value: Sequence[int]) -> tuple[int, int]:
    if len(value) != 2 or int(value[0]) <= 0 or int(value[1]) <= 0:
        raise ValueError(f"{name} dimensions must be positive")
    return int(value[0]), int(value[1])


@dataclass(frozen=True)
class CropTransform:
    """Exact original -> resized -> square crop -> render pixel transform."""

    original_size: tuple[int, int]
    resized_size: tuple[int, int]
    crop_box: tuple[int, int, int, int]
    render_size: tuple[int, int]

    def __post_init__(self) -> None:
        _positive_size("original_size", self.original_size)
        _positive_size("resized_size", self.resized_size)
        _positive_size("render_size", self.render_size)
        left, top, right, bottom = (int(value) for value in self.crop_box)
        if right <= left or bottom <= top:
            raise ValueError("crop_box dimensions must be positive")

    @property
    def crop_size(self) -> tuple[int, int]:
        left, top, right, bottom = self.crop_box
        return int(right - left), int(bottom - top)

    def original_to_render(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        original = np.asarray(self.original_size, dtype=np.float64)
        resized = np.asarray(self.resized_size, dtype=np.float64)
        crop_origin = np.asarray(self.crop_box[:2], dtype=np.float64)
        crop_size = np.asarray(self.crop_size, dtype=np.float64)
        render_size = np.asarray(self.render_size, dtype=np.float64)
        return (points * resized / original - crop_origin) * render_size / crop_size

    def render_to_original(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        original = np.asarray(self.original_size, dtype=np.float64)
        resized = np.asarray(self.resized_size, dtype=np.float64)
        crop_origin = np.asarray(self.crop_box[:2], dtype=np.float64)
        crop_size = np.asarray(self.crop_size, dtype=np.float64)
        render_size = np.asarray(self.render_size, dtype=np.float64)
        return (points * crop_size / render_size + crop_origin) * original / resized

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "CropTransform":
        return cls(
            original_size=tuple(payload["original_size"]),
            resized_size=tuple(payload["resized_size"]),
            crop_box=tuple(payload["crop_box"]),
            render_size=tuple(payload["render_size"]),
        )


def validate_surface_buffers(
    triangle_ids: np.ndarray,
    mask: np.ndarray,
    depth: np.ndarray,
    coords: np.ndarray,
    *,
    num_faces: int,
) -> dict:
    triangle_ids = np.asarray(triangle_ids)
    mask = np.asarray(mask, dtype=bool)
    depth = np.asarray(depth)
    coords = np.asarray(coords)
    if triangle_ids.ndim != 2:
        raise ValueError("triangle_ids must be [H,W]")
    if mask.shape != triangle_ids.shape or depth.shape != triangle_ids.shape:
        raise ValueError("mask/depth must align with triangle_ids")
    if coords.shape != (*triangle_ids.shape, 3):
        raise ValueError("coords must be [H,W,3]")
    if int(num_faces) <= 0:
        raise ValueError("face count must be positive")
    if np.any(triangle_ids < 0) or int(triangle_ids.max(initial=0)) > int(num_faces):
        raise ValueError("triangle id exceeds raw mesh face count")
    foreground = triangle_ids > 0
    if not np.array_equal(mask, foreground):
        raise ValueError("renderer mask and discrete triangle coverage disagree")
    if not np.all(np.isfinite(depth[foreground])):
        raise ValueError("foreground depth contains non-finite values")
    if not np.all(np.isfinite(coords[foreground])):
        raise ValueError("foreground coordinates contain non-finite values")
    return {
        "height": int(triangle_ids.shape[0]),
        "width": int(triangle_ids.shape[1]),
        "foreground_pixels": int(foreground.sum()),
        "background_pixels": int((~foreground).sum()),
        "max_triangle_id": int(triangle_ids.max(initial=0)),
        "num_faces": int(num_faces),
    }


def face_rows_from_triangle_ids(
    triangle_ids: np.ndarray, face_features: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    triangle_ids = np.asarray(triangle_ids, dtype=np.int64)
    face_features = np.asarray(face_features)
    if face_features.ndim != 2:
        raise ValueError("face_features must be [F,D]")
    if np.any(triangle_ids < 0) or int(triangle_ids.max(initial=0)) > face_features.shape[0]:
        raise ValueError("triangle id exceeds PartField feature rows")
    valid = triangle_ids > 0
    rows = np.full((*triangle_ids.shape, face_features.shape[1]), np.nan, dtype=np.float32)
    rows[valid] = face_features[triangle_ids[valid] - 1]
    return rows, valid


def sample_triangle_ids(
    triangle_ids: np.ndarray, transform: CropTransform, original_points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    triangle_ids = np.asarray(triangle_ids, dtype=np.int64)
    rendered = transform.original_to_render(np.asarray(original_points, dtype=np.float64))
    pixels = np.rint(rendered).astype(np.int64)
    inside = (
        (pixels[..., 0] >= 0)
        & (pixels[..., 0] < triangle_ids.shape[1])
        & (pixels[..., 1] >= 0)
        & (pixels[..., 1] < triangle_ids.shape[0])
    )
    sampled = np.zeros(pixels.shape[:-1], dtype=np.int64)
    sampled[inside] = triangle_ids[pixels[..., 1][inside], pixels[..., 0][inside]]
    return sampled, inside

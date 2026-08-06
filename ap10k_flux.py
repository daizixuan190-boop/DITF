"""AP-10K geometry, grouping, and metric helpers for the Flux evaluator."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import PILToTensor


SETTING_SPLITS = {
    "intra-species": "test",
    "cross-species": "test_cross_species",
    "cross-family": "test_cross_family",
}
PAPER_FLUX_PER_IMAGE = {
    "intra-species": {"0.01": 0.076, "0.05": 0.497, "0.10": 0.622},
    "cross-species": {"0.01": 0.068, "0.05": 0.482, "0.10": 0.617},
    "cross-family": {"0.01": 0.057, "0.05": 0.342, "0.10": 0.489},
}
ALPHAS = (0.01, 0.05, 0.10)


@dataclass(frozen=True)
class PairRecord:
    path: Path
    source_annotation: str
    target_annotation: str
    group: str


@dataclass(frozen=True)
class PreparedAnnotation:
    relative_path: str
    image_path: Path
    species: str
    family: str
    points: torch.Tensor
    visibility: torch.Tensor
    threshold: float


def annotation_identity(relative_path: str) -> tuple[str, str]:
    parts = Path(relative_path).parts
    if len(parts) < 4 or parts[0] != "ImageAnnotation":
        raise ValueError(f"Unexpected AP-10K annotation path: {relative_path}")
    return parts[1], parts[2]


def pair_group(setting: str, source_annotation: str, target_annotation: str) -> str:
    source_family, source_species = annotation_identity(source_annotation)
    target_family, target_species = annotation_identity(target_annotation)
    if setting == "intra-species":
        if (source_family, source_species) != (target_family, target_species):
            raise ValueError("Intra-species pair crosses species")
        return f"{source_family}/{source_species}"
    if setting == "cross-species":
        if source_family != target_family or source_species == target_species:
            raise ValueError("Cross-species pair does not contain two species in one family")
        return source_family
    if setting == "cross-family":
        if source_family == target_family:
            raise ValueError("Cross-family pair stays within one family")
        return f"{source_family}|{target_family}"
    raise ValueError(f"Unknown AP-10K setting: {setting}")


def discover_pairs(
    benchmark_root: Path,
    setting: str,
    *,
    max_groups: int = 0,
    max_pairs_per_group: int = 0,
    pair_sample_seed: int | None = None,
) -> dict[str, list[PairRecord]]:
    try:
        split = SETTING_SPLITS[setting]
    except KeyError as error:
        raise ValueError(f"Unknown AP-10K setting: {setting}") from error
    pair_paths = sorted((benchmark_root / "PairAnnotation" / split).glob("*.json"))
    if not pair_paths:
        raise FileNotFoundError(f"No AP-10K pairs found for {setting}")
    groups: dict[str, list[PairRecord]] = {}
    for path in pair_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        source = payload["src_json_path"]
        target = payload["trg_json_path"]
        group = pair_group(setting, source, target)
        groups.setdefault(group, []).append(PairRecord(path, source, target, group))
    selected: dict[str, list[PairRecord]] = {}
    for group in sorted(groups)[: max_groups or None]:
        pairs = groups[group]
        if max_pairs_per_group and pair_sample_seed is not None:
            pairs = sorted(
                pairs,
                key=lambda pair: hashlib.sha256(
                    f"{pair_sample_seed}:{pair.path.name}".encode("utf-8")
                ).digest(),
            )
        selected[group] = pairs[: max_pairs_per_group or None]
    return selected


def transform_keypoints(
    keypoints: Sequence[float], width: int, height: int, size: int
) -> tuple[torch.Tensor, torch.Tensor, float]:
    values = torch.as_tensor(keypoints, dtype=torch.float32).reshape(-1, 3)
    scale = size / max(width, height)
    points = values[:, :2] * scale
    if height < width:
        resized_height = int(np.around(size * height / width))
        points[:, 1] += int((size - resized_height) / 2)
    elif width < height:
        resized_width = int(np.around(size * width / height))
        points[:, 0] += int((size - resized_width) / 2)
    visibility = values[:, 2] / 2.0
    # Reproduce preprocess_kps_pad exactly, including its final multiplication
    # of all three keypoint fields by the normalized visibility value.
    points *= visibility[:, None]
    visibility = visibility.square()
    # GeoAware/DiTF casts transformed AP-10K coordinates to int32.
    return points.to(torch.int32).float(), visibility, scale


def prepare_annotation(benchmark_root: Path, relative_path: str, size: int) -> PreparedAnnotation:
    path = benchmark_root / relative_path
    payload = json.loads(path.read_text(encoding="utf-8"))
    family, species = annotation_identity(relative_path)
    image_name = path.with_suffix(".jpg").name
    image_path = benchmark_root / "JPEGImages" / family / species / image_name
    if not image_path.is_file():
        raise FileNotFoundError(f"Missing AP-10K benchmark image: {image_path}")
    points, visibility, scale = transform_keypoints(
        payload["keypoints"], int(payload["width"]), int(payload["height"]), size
    )
    bbox = payload["bbox"]
    threshold = max(float(bbox[2]), float(bbox[3])) * scale
    return PreparedAnnotation(
        relative_path=relative_path,
        image_path=image_path,
        species=species,
        family=family,
        points=points,
        visibility=visibility,
        threshold=threshold,
    )


def load_square_padded_tensor(path: Path, size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    if height <= width:
        resized = image.resize(
            (size, int(np.around(size * height / width))), Image.Resampling.LANCZOS
        )
        canvas = Image.new("RGB", (size, size))
        canvas.paste(resized, (0, (size - resized.height) // 2))
    else:
        resized = image.resize(
            (int(np.around(size * width / height)), size), Image.Resampling.LANCZOS
        )
        canvas = Image.new("RGB", (size, size))
        canvas.paste(resized, ((size - resized.width) // 2, 0))
    return (PILToTensor()(canvas).float() / 255.0 - 0.5) * 2.0


def pair_hits(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
) -> dict[str, torch.Tensor]:
    errors = torch.linalg.vector_norm(predictions.float() - targets.float(), dim=1)
    return {f"{alpha:.2f}": errors < alpha * threshold for alpha in ALPHAS}


def empty_metric_counts() -> dict[str, Any]:
    return {
        "pair_count": 0,
        "point_count": 0,
        "pair_score_sum": {f"{alpha:.2f}": 0.0 for alpha in ALPHAS},
        "point_correct": {f"{alpha:.2f}": 0 for alpha in ALPHAS},
    }


def update_metric_counts(counts: dict[str, Any], hits: dict[str, torch.Tensor]) -> None:
    point_count = int(next(iter(hits.values())).numel())
    if point_count == 0:
        raise ValueError("AP-10K pairs must contain mutually visible keypoints")
    counts["pair_count"] += 1
    counts["point_count"] += point_count
    for alpha, values in hits.items():
        correct = int(values.sum().item())
        counts["pair_score_sum"][alpha] += correct / point_count
        counts["point_correct"][alpha] += correct


def merge_metric_counts(target: dict[str, Any], source: dict[str, Any]) -> None:
    target["pair_count"] += source["pair_count"]
    target["point_count"] += source["point_count"]
    for alpha in target["pair_score_sum"]:
        target["pair_score_sum"][alpha] += source["pair_score_sum"][alpha]
        target["point_correct"][alpha] += source["point_correct"][alpha]


def metric_ratios(counts: dict[str, Any]) -> dict[str, Any]:
    if not counts["pair_count"] or not counts["point_count"]:
        raise ValueError("Cannot summarize empty AP-10K metrics")
    return {
        "pair_count": counts["pair_count"],
        "point_count": counts["point_count"],
        "per_image_pck": {
            alpha: counts["pair_score_sum"][alpha] / counts["pair_count"]
            for alpha in counts["pair_score_sum"]
        },
        "per_point_pck": {
            alpha: counts["point_correct"][alpha] / counts["point_count"]
            for alpha in counts["point_correct"]
        },
    }

"""Baseline-first DINOv2 nearest-neighbour evaluation on SPair-71k.

This script intentionally contains no ownership/oracle analysis. Reproduce the
DiTF paper's DINOv2+NN baseline first; run diagnostics only from a separate
evaluator after baseline parity is established.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
from typing import Any, Protocol

import torch
from PIL import Image
from tqdm import tqdm

from dino_v2_spair import (
    PAPER_DINO_ALL_IMAGE,
    PAPER_DINO_ALL_POINT,
    CategoryMetrics,
    DINOConfig,
    cosine_nn_predictions,
    pck_hits,
    preprocess_square_canvas,
    square_canvas_geometry,
    tokens_to_patch_map,
    transform_points_to_canvas,
)


class Extractor(Protocol):
    def __call__(self, image: Image.Image) -> torch.Tensor: ...
    def close(self) -> None: ...


class BlockTokenExtractor:
    """Extract pre-final-norm block tokens from an official DINOv2 ViT."""

    def __init__(self, model: torch.nn.Module, config: DINOConfig, device: str, precision: str):
        embed_dim = int(getattr(model, "embed_dim", 0))
        blocks = getattr(model, "blocks", None)
        patch_size = getattr(getattr(model, "patch_embed", None), "patch_size", None)
        patch_size = patch_size[0] if isinstance(patch_size, tuple) else patch_size
        if embed_dim != 768 or blocks is None or len(blocks) != 12 or int(patch_size or 0) != 14:
            raise ValueError("Loaded torch.hub model is not DINOv2 ViT-B/14")
        self.model = model.to(device).eval()
        self.config = config
        self.device = torch.device(device)
        self.precision = precision
        self._tokens: torch.Tensor | None = None
        if blocks is None or not 0 <= config.layer < len(blocks):
            raise ValueError(f"Model does not expose DINOv2 block {config.layer}")
        self._hook = blocks[config.layer].register_forward_hook(self._capture)

    def _capture(self, _module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        self._tokens = output[0] if isinstance(output, (tuple, list)) else output

    @torch.inference_mode()
    def __call__(self, image: Image.Image) -> torch.Tensor:
        pixels = preprocess_square_canvas(image, self.config.image_size).unsqueeze(0).to(self.device)
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(self.precision)
        enabled = dtype is not None and self.device.type == "cuda"
        with torch.autocast(device_type=self.device.type, dtype=dtype, enabled=enabled):
            self._tokens = None
            self.model(pixels)
        if self._tokens is None:
            raise RuntimeError(f"DINOv2 block-{self.config.layer} hook produced no tokens")
        feature = tokens_to_patch_map(self._tokens, self.config.grid_size)[0].float().cpu()
        self._tokens = None
        return feature

    def close(self) -> None:
        self._hook.remove()


class TransformersExtractor:
    """HF weight-loader with the same pre-norm block-token extraction protocol."""

    def __init__(self, config: DINOConfig, device: str, precision: str, local_files_only: bool):
        try:
            from transformers import Dinov2Model
        except ImportError as exc:
            raise RuntimeError("The transformers backend requires the transformers package") from exc
        self.model = Dinov2Model.from_pretrained(
            config.model_name,
            local_files_only=local_files_only,
        )
        model_config = self.model.config
        if (
            int(model_config.hidden_size) != 768
            or int(model_config.num_hidden_layers) != 12
            or int(model_config.patch_size) != 14
        ):
            raise ValueError("Loaded Transformers model is not DINOv2 ViT-B/14")
        self.config = config
        self.device = torch.device(device)
        self.precision = precision
        self._tokens: torch.Tensor | None = None
        layers = self.model.encoder.layer
        if not 0 <= config.layer < len(layers):
            raise ValueError(f"Model does not expose DINOv2 block {config.layer}")
        self._hook = layers[config.layer].register_forward_hook(self._capture)
        self.model.to(self.device).eval()

    def _capture(self, _module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        self._tokens = output[0] if isinstance(output, (tuple, list)) else output

    @torch.inference_mode()
    def __call__(self, image: Image.Image) -> torch.Tensor:
        pixels = preprocess_square_canvas(image, self.config.image_size).unsqueeze(0).to(self.device)
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(self.precision)
        enabled = dtype is not None and self.device.type == "cuda"
        with torch.autocast(device_type=self.device.type, dtype=dtype, enabled=enabled):
            self._tokens = None
            self.model(pixel_values=pixels, interpolate_pos_encoding=True, return_dict=True)
        if self._tokens is None:
            raise RuntimeError(f"DINOv2 block-{self.config.layer} hook produced no tokens")
        feature = tokens_to_patch_map(self._tokens, self.config.grid_size)[0].float().cpu()
        self._tokens = None
        return feature

    def close(self) -> None:
        self._hook.remove()


def build_extractor(args: argparse.Namespace, config: DINOConfig) -> Extractor:
    if args.backend == "transformers":
        return TransformersExtractor(config, args.device, args.precision, args.local_files_only)
    source = "local" if args.model_repo else "github"
    repo = args.model_repo or "facebookresearch/dinov2"
    model = torch.hub.load(repo, config.hub_model, source=source, pretrained=True)
    return BlockTokenExtractor(model, config, args.device, args.precision)


def discover_pairs(dataset_path: Path, max_pairs_per_cat: int) -> tuple[list[str], dict[str, list[Path]]]:
    categories = sorted(path.name for path in (dataset_path / "JPEGImages").iterdir() if path.is_dir())
    annotations = sorted((dataset_path / "PairAnnotation" / "test").glob("*.json"))
    by_category = {category: [] for category in categories}
    for annotation in annotations:
        category = annotation.stem.rsplit(":", 1)[-1]
        if category in by_category:
            by_category[category].append(annotation)
    if max_pairs_per_cat > 0:
        by_category = {key: value[:max_pairs_per_cat] for key, value in by_category.items()}
    return categories, by_category


def category_image_names(pair_paths: list[Path]) -> list[str]:
    names: set[str] = set()
    for pair_path in pair_paths:
        data = json.loads(pair_path.read_text(encoding="utf-8"))
        names.update((data["src_imname"], data["trg_imname"]))
    return sorted(names)


def evaluate(args: argparse.Namespace, extractor: Extractor | None = None) -> dict[str, Any]:
    dataset_path = Path(args.dataset_path)
    config = DINOConfig(args.model_name, args.hub_model, args.layer, 14, args.image_size)
    if (
        config.hub_model != "dinov2_vitb14"
        or "large" in config.model_name.lower()
        or config.layer != 11
        or config.image_size != 840
    ):
        raise ValueError("Paper parity requires DINOv2 ViT-B/14, block 11, and an 840x840 canvas")
    categories, category_pairs = discover_pairs(dataset_path, args.max_pairs_per_cat)
    missing_categories = [category for category in categories if not category_pairs[category]]
    if missing_categories:
        raise RuntimeError(f"No SPair test pairs found for categories: {missing_categories}")
    own_extractor = extractor is None
    extractor = extractor or build_extractor(args, config)
    device = torch.device(args.device)
    metrics = {category: CategoryMetrics() for category in categories}
    pair_records: list[dict[str, Any]] = []

    try:
        for category in categories:
            pair_paths = category_pairs[category]
            features: dict[str, torch.Tensor] = {}
            image_root = dataset_path / "JPEGImages" / category
            for image_name in tqdm(category_image_names(pair_paths), desc=f"features {category}", leave=False):
                with Image.open(image_root / image_name) as image:
                    features[image_name] = extractor(image)

            for pair_path in tqdm(pair_paths, desc=f"evaluate {category}"):
                data = json.loads(pair_path.read_text(encoding="utf-8"))
                src_h, src_w = int(data["src_imsize"][1]), int(data["src_imsize"][0])
                trg_h, trg_w = int(data["trg_imsize"][1]), int(data["trg_imsize"][0])
                source_points = transform_points_to_canvas(data["src_kps"], src_h, src_w, config.image_size).to(device)
                target_points = transform_points_to_canvas(data["trg_kps"], trg_h, trg_w, config.image_size).to(device)
                predictions, _ = cosine_nn_predictions(
                    features[data["src_imname"]].to(device),
                    features[data["trg_imname"]].to(device),
                    source_points,
                    config.image_size,
                )
                target_scale = square_canvas_geometry(trg_h, trg_w, config.image_size)[0]
                bbox = data["trg_bndbox"]
                threshold = max(bbox[3] - bbox[1], bbox[2] - bbox[0]) * target_scale
                hits = pck_hits(predictions, target_points, threshold)
                metrics[category].update(hits)
                pair_records.append(
                    {
                        "category": category,
                        "pair": pair_path.name,
                        "points": int(hits.numel()),
                        "correct": int(hits.sum()),
                        "pck@0.1": float(hits.float().mean()),
                    }
                )

            del features
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        if own_extractor:
            extractor.close()

    category_summary = {
        category: {
            "pairs": len(metrics[category].pair_scores),
            "per_image_pck@0.1": metrics[category].per_image,
            "per_point_pck@0.1": metrics[category].per_point,
            "points": metrics[category].total,
        }
        for category in categories
    }
    all_pairs = [score for category in categories for score in metrics[category].pair_scores]
    total_correct = sum(value.correct for value in metrics.values())
    total_points = sum(value.total for value in metrics.values())
    all_per_image = sum(all_pairs) / len(all_pairs) if all_pairs else 0.0
    all_per_point = total_correct / total_points if total_points else 0.0
    mean_per_image = sum(value.per_image for value in metrics.values()) / len(categories)
    mean_per_point = sum(value.per_point for value in metrics.values()) / len(categories)
    is_full = args.max_pairs_per_cat == 0
    parity_evaluated = is_full and args.precision == "fp32"
    summary: dict[str, Any] = {
        "protocol": {
            "reference": "GeoAware-SC / DiTF DINOv2+NN",
            "model": "dinov2_vitb14",
            "input": "840x840 aspect-preserving zero canvas",
            "descriptor": "block 11 token facet, pre-final-norm, 60x60",
            "matcher": "cosine nearest neighbour on patch centers",
            "feature_cache": "category-scoped memory only",
            "diagnostics_enabled": False,
        },
        "config": vars(args),
        "categories": category_summary,
        "all_per_image_pck@0.1": all_per_image,
        "all_per_point_pck@0.1": all_per_point,
        "mean_per_image_pck@0.1": mean_per_image,
        "mean_per_point_pck@0.1": mean_per_point,
        "paper_targets": {
            "all_per_image_pck@0.1": PAPER_DINO_ALL_IMAGE,
            "all_per_point_pck@0.1": PAPER_DINO_ALL_POINT,
        },
        "full_run_parity": None if not parity_evaluated else {
            "all_per_image_delta": all_per_image - PAPER_DINO_ALL_IMAGE,
            "all_per_point_delta": all_per_point - PAPER_DINO_ALL_POINT,
            "within_1_point": abs(all_per_image - PAPER_DINO_ALL_IMAGE) <= 0.01
            and abs(all_per_point - PAPER_DINO_ALL_POINT) <= 0.01,
        },
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.output_csv:
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("category", "pair", "points", "correct", "pck@0.1"))
            writer.writeheader()
            writer.writerows(pair_records)

    for category, values in category_summary.items():
        print(f"{category} per point PCK@0.1: {100 * values['per_point_pck@0.1']:.2f}")
    print(f"All per image PCK@0.1: {100 * all_per_image:.2f}")
    print(f"All per point PCK@0.1: {100 * all_per_point:.2f}")
    print(f"Mean per image PCK@0.1: {100 * mean_per_image:.2f}")
    print(f"Mean per point PCK@0.1: {100 * mean_per_point:.2f}")
    print(f"Saved baseline summary to: {output_json}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", default=None, help="Optional per-pair output; disabled by default")
    parser.add_argument("--backend", choices=("transformers", "torch_hub"), default="transformers")
    parser.add_argument("--model_name", default="facebook/dinov2-base")
    parser.add_argument("--hub_model", default="dinov2_vitb14")
    parser.add_argument("--model_repo", default=None, help="Local official DINOv2 repo for torch_hub backend")
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--image_size", type=int, default=840)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--max_pairs_per_cat", type=int, default=0)
    return parser


if __name__ == "__main__":
    evaluate(build_parser().parse_args())

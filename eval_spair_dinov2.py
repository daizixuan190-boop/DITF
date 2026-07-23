"""Evaluate DINOv2 token features and ownership collapse on SPair-71k.

The evaluator reports both the ordinary nearest-neighbour baseline and a
label-free candidate-space diagnostic.  The diagnostic is not a method and
must not be used to claim an accuracy improvement: it measures whether a
correct target location survives owner-local top-M proposal truncation.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import Dinov2Model

from dino_v2_spair import DINOConfig, dino_tokens_to_map, resize_shape, summarize_candidate_rows


KS = (1, 5, 10, 20, 50)


def image_tensor(image: Image.Image, max_side: int) -> tuple[torch.Tensor, tuple[int, int]]:
    image = image.convert("RGB")
    h, w = image.height, image.width
    out_h, out_w = resize_shape(h, w, max_side)
    image = image.resize((out_w, out_h), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
    return (tensor - mean) / std, (out_h, out_w)


class DINOv2Extractor:
    def __init__(self, config: DINOConfig, device: str = "cuda", local_files_only: bool = False):
        self.config = config
        self.device = torch.device(device)
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = Dinov2Model.from_pretrained(
            config.model_name,
            torch_dtype=dtype,
            local_files_only=local_files_only,
        ).to(self.device).eval()
        self._captured: dict[str, torch.Tensor] = {}
        self._hook = None
        encoder_layers = getattr(getattr(self.model, "encoder", None), "layer", None)
        if encoder_layers is not None and 0 <= config.layer < len(encoder_layers):
            self._hook = encoder_layers[config.layer].register_forward_hook(self._capture_block)

    def _capture_block(self, _module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: Any) -> None:
        # Dinov2EncoderLayer returns a tensor in current Transformers releases;
        # accept a one-element tuple as well for older releases.
        self._captured["tokens"] = output[0] if isinstance(output, (tuple, list)) else output

    @torch.inference_mode()
    def __call__(self, image: Image.Image) -> torch.Tensor:
        pixels, (height, width) = image_tensor(image, self.config.max_side)
        model_dtype = next(self.model.parameters()).dtype
        pixels = pixels.to(device=self.device, dtype=model_dtype)
        output = self.model(
            pixel_values=pixels.unsqueeze(0),
            output_hidden_states=self._hook is None,
            interpolate_pos_encoding=True,
            return_dict=True,
        )
        if self._hook is not None:
            tokens = self._captured.pop("tokens", None)
            if tokens is None:
                raise RuntimeError(f"DINO block hook did not capture layer {self.config.layer}")
        else:
            # HF hidden_states[0] is the embedding output; +1 maps the
            # zero-based transformer block index to the post-block activation.
            hidden_index = int(self.config.layer) + 1
            if output.hidden_states is None or hidden_index >= len(output.hidden_states):
                raise ValueError(f"DINO layer {self.config.layer} unavailable; got {len(output.hidden_states or [])} states")
            tokens = output.hidden_states[hidden_index]
        feature_map = dino_tokens_to_map(tokens, height, width, self.config.patch_size)[0]
        cache_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        return feature_map.to(dtype=cache_dtype).cpu()

    def close(self) -> None:
        if self._hook is not None:
            self._hook.remove()
            self._hook = None


def discover_pairs(dataset_path: Path, max_pairs_per_cat: int = 0) -> tuple[list[str], dict[str, list[str]], dict[str, list[str]]]:
    test_path = dataset_path / "PairAnnotation" / "test"
    json_list = sorted(p.name for p in test_path.glob("*.json"))
    categories = sorted(p.name for p in (dataset_path / "JPEGImages").iterdir() if p.is_dir())
    cat_json = {cat: [name for name in json_list if cat in name] for cat in categories}
    cat_images: dict[str, list[str]] = {}
    for cat in categories:
        names: list[str] = []
        selected_json = cat_json[cat][:max_pairs_per_cat] if max_pairs_per_cat > 0 else cat_json[cat]
        for json_name in selected_json:
            data = json.loads((test_path / json_name).read_text())
            for key in ("src_imname", "trg_imname"):
                if data[key] not in names:
                    names.append(data[key])
        cat_images[cat] = names
    return categories, cat_json, cat_images


def load_feature(cache: Path, category: str, image_name: str) -> torch.Tensor:
    values = torch.load(cache / f"{category}.pth", map_location="cpu")
    return values[image_name]


def extract_features(args: argparse.Namespace, categories: list[str], cat_images: dict[str, list[str]]) -> None:
    cache = Path(args.save_path)
    cache.mkdir(parents=True, exist_ok=True)
    if args.reuse_saved_features:
        return
    extractor = DINOv2Extractor(
        DINOConfig(args.model_name, args.layer, args.patch_size, args.img_size),
        device=args.device,
        local_files_only=args.local_files_only,
    )
    dataset_path = Path(args.dataset_path)
    for cat in tqdm(categories, desc="DINOv2 features"):
        output: dict[str, torch.Tensor] = {}
        for image_name in tqdm(cat_images[cat], desc=cat, leave=False):
            image = Image.open(dataset_path / "JPEGImages" / cat / image_name)
            output[image_name] = extractor(image)
        torch.save(output, cache / f"{cat}.pth")
    extractor.close()
    del extractor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    dataset_path = Path(args.dataset_path)
    cache = Path(args.save_path)
    categories, cat_json, cat_images = discover_pairs(dataset_path, args.max_pairs_per_cat)
    extract_features(args, categories, cat_images)
    records: list[dict[str, Any]] = []
    per_cat_image: dict[str, list[float]] = defaultdict(list)
    per_cat_correct: dict[str, int] = defaultdict(int)
    per_cat_total: dict[str, int] = defaultdict(int)
    test_path = dataset_path / "PairAnnotation" / "test"
    for cat in categories:
        names = torch.load(cache / f"{cat}.pth", map_location="cpu")
        pair_names = cat_json[cat][: args.max_pairs_per_cat] if args.max_pairs_per_cat > 0 else cat_json[cat]
        for json_name in tqdm(pair_names, desc=f"evaluate {cat}"):
            data = json.loads((test_path / json_name).read_text())
            src = names[data["src_imname"]]
            trg = names[data["trg_imname"]]
            src_h, src_w = data["src_imsize"][1], data["src_imsize"][0]
            trg_h, trg_w = data["trg_imsize"][1], data["trg_imsize"][0]
            src_up = F.interpolate(src.unsqueeze(0), size=(src_h, src_w), mode="bilinear", align_corners=False)[0]
            trg_up = F.interpolate(trg.unsqueeze(0), size=(trg_h, trg_w), mode="bilinear", align_corners=False)[0]
            src_norm = F.normalize(src_up.float(), dim=0, eps=1e-6)
            trg_norm = F.normalize(trg_up.float(), dim=0, eps=1e-6)
            src_points = data["src_kps"]
            trg_points = data["trg_kps"]
            vectors = torch.stack([src_norm[:, int(y), int(x)] for x, y in src_points])
            scores = torch.einsum("nc,chw->nhw", vectors, trg_norm).flatten(1)
            max_scores, predictions = scores.max(dim=1)
            pred_y = torch.div(predictions, trg_w, rounding_mode="floor")
            pred_x = predictions % trg_w
            threshold = max(data["trg_bndbox"][3] - data["trg_bndbox"][1], data["trg_bndbox"][2] - data["trg_bndbox"][0])
            distances = torch.sqrt((pred_x.float() - torch.tensor([p[0] for p in trg_points])) ** 2 + (pred_y.float() - torch.tensor([p[1] for p in trg_points])) ** 2)
            baseline_hits = (distances / max(float(threshold), 1e-6) <= 0.1).tolist()
            pair_records = []
            max_k = min(max(KS), scores.shape[1])
            candidates = scores.topk(max_k, dim=1).indices
            ownership = summarize_candidate_rows(candidates, trg_points, threshold, trg_w, KS)
            for idx, ownership_row in enumerate(ownership):
                row = {
                    "category": cat, "pair": json_name, "point_index": idx,
                    "baseline_hit": int(baseline_hits[idx]), "baseline_score": float(max_scores[idx]),
                    "source_x": src_points[idx][0], "source_y": src_points[idx][1],
                    "target_x": trg_points[idx][0], "target_y": trg_points[idx][1],
                    "threshold": float(threshold),
                }
                row.update({key: value for key, value in ownership_row.items() if key != "point_index"})
                records.append(row)
                per_cat_correct[cat] += int(baseline_hits[idx])
                per_cat_total[cat] += 1
            per_cat_image[cat].append(float(np.mean(baseline_hits)))
    summary: dict[str, Any] = {"config": vars(args), "count": len(records), "categories": {}}
    all_hits = [row["baseline_hit"] for row in records]
    for cat in categories:
        cat_rows = [row for row in records if row["category"] == cat]
        summary["categories"][cat] = {
            "pairs": len(per_cat_image[cat]),
            "baseline_per_image": float(np.mean(per_cat_image[cat])) if per_cat_image[cat] else 0.0,
            "baseline_per_point": per_cat_correct[cat] / max(per_cat_total[cat], 1),
        }
    summary["baseline_micro_pck"] = float(np.mean(all_hits)) if all_hits else 0.0
    summary["baseline_macro_pck"] = float(np.mean([v["baseline_per_point"] for v in summary["categories"].values()]))
    for k in KS:
        summary[f"owner_candidate_recall@{k}"] = float(np.mean([r[f"owner_candidate_hit@{k}"] for r in records]))
        summary[f"other_source_transfer@{k}"] = float(np.mean([r[f"other_source_candidate_hit@{k}"] for r in records]))
        summary[f"global_union_recall@{k}"] = float(np.mean([r[f"global_union_candidate_hit@{k}"] for r in records]))
        failures = [r for r in records if not r["baseline_hit"]]
        summary[f"failure_owner_candidate_recall@{k}"] = float(np.mean([r[f"owner_candidate_hit@{k}"] for r in failures])) if failures else 0.0
        summary[f"failure_global_union_recall@{k}"] = float(np.mean([r[f"global_union_candidate_hit@{k}"] for r in failures])) if failures else 0.0
        summary[f"failure_transferable_rate@{k}"] = float(np.mean([int(r[f"global_union_candidate_hit@{k}"] and not r[f"owner_candidate_hit@{k}"]) for r in failures])) if failures else 0.0
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with Path(args.output_csv).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(records[0]) if records else ["category"])
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(summary, indent=2))
    print(f"Saved summary to: {out}")
    print(f"Saved records to: {args.output_csv}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--model_name", default="facebook/dinov2-large")
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--img_size", type=int, default=840)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--reuse_saved_features", action="store_true")
    parser.add_argument("--max_pairs_per_cat", type=int, default=0)
    return parser


if __name__ == "__main__":
    evaluate(build_parser().parse_args())

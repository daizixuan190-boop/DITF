"""One-image Pixal3D pixel-to-raw-face interface regression.

This script instruments the pinned official inference path. It deliberately
skips final GLB remeshing and does not run PartField or use SPair labels.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
from itertools import permutations
from pathlib import Path

import numpy as np
from PIL import Image

from pixal3d_probe_utils import CropTransform, validate_surface_buffers


PINNED_PIXAL_COMMIT = "cdbb2bbffbf4e6f298b5f2af3d1d76a8d823d2af"


def _git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def _instrumented_preprocess(pipeline, input_image: Image.Image, bg_color: tuple, capture: dict):
    """Current official Pixal3D preprocessing with its hidden crop recorded."""
    original_size = tuple(input_image.size)
    has_alpha = False
    if input_image.mode == "RGBA":
        alpha = np.array(input_image)[:, :, 3]
        if not np.all(alpha == 255):
            has_alpha = True
    max_size = max(input_image.size)
    scale = min(1.0, 1024.0 / max_size)
    if scale < 1:
        input_image = input_image.resize(
            (int(input_image.width * scale), int(input_image.height * scale)),
            Image.Resampling.LANCZOS,
        )
    resized_size = tuple(input_image.size)
    if has_alpha:
        segmented = input_image
    else:
        input_image = input_image.convert("RGB")
        if pipeline.low_vram:
            pipeline.rembg_model.to(pipeline.device)
        segmented = pipeline.rembg_model(input_image)
        if pipeline.low_vram:
            pipeline.rembg_model.cpu()
    segmented_np = np.array(segmented)
    if segmented_np.ndim != 3 or segmented_np.shape[2] != 4:
        raise RuntimeError("Pixal3D foreground model did not return RGBA")
    alpha = segmented_np[:, :, 3]
    foreground = np.argwhere(alpha > 0.8 * 255)
    if foreground.size == 0:
        raise RuntimeError("Pixal3D foreground model returned an empty alpha mask")
    raw_bbox = (
        int(np.min(foreground[:, 1])),
        int(np.min(foreground[:, 0])),
        int(np.max(foreground[:, 1])),
        int(np.max(foreground[:, 0])),
    )
    center = ((raw_bbox[0] + raw_bbox[2]) / 2, (raw_bbox[1] + raw_bbox[3]) / 2)
    size = int(max(raw_bbox[2] - raw_bbox[0], raw_bbox[3] - raw_bbox[1]) * 1.1)
    if size <= 0:
        raise RuntimeError("Pixal3D foreground crop is degenerate")
    crop_box = (
        int(center[0] - size // 2),
        int(center[1] - size // 2),
        int(center[0] + size // 2),
        int(center[1] + size // 2),
    )
    cropped = segmented.crop(crop_box)
    cropped_np = np.array(cropped).astype(np.float32) / 255.0
    rgb = cropped_np[:, :, :3]
    cropped_alpha = cropped_np[:, :, 3:4]
    bg = np.asarray(bg_color, dtype=np.float32) / 255.0
    composite = rgb * cropped_alpha + bg * (1.0 - cropped_alpha)
    output = Image.fromarray((np.clip(composite, 0, 1) * 255).astype(np.uint8))
    capture.update(
        {
            "original_size": original_size,
            "resized_size": resized_size,
            "raw_foreground_bbox": raw_bbox,
            "crop_box": crop_box,
            "preprocessed_image": output.copy(),
            "cropped_alpha": (cropped_alpha[:, :, 0] * 255).astype(np.uint8),
        }
    )
    return output


class _SkippedGlb:
    def apply_transform(self, _transform):
        return self

    def export(self, *_args, **_kwargs):
        return None


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    points = np.argwhere(mask)
    if points.size == 0:
        return None
    return (
        int(points[:, 1].min()),
        int(points[:, 0].min()),
        int(points[:, 1].max() + 1),
        int(points[:, 0].max() + 1),
    )


def _bbox_iou(first: tuple[int, int, int, int] | None, second: tuple[int, int, int, int] | None) -> float:
    if first is None or second is None:
        return 0.0
    x0, y0 = max(first[0], second[0]), max(first[1], second[1])
    x1, y1 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    area_first = (first[2] - first[0]) * (first[3] - first[1])
    area_second = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / max(area_first + area_second - intersection, 1)


def _mask_metrics(alpha_mask: np.ndarray, rendered_mask: np.ndarray) -> dict:
    alpha_mask = np.asarray(alpha_mask, dtype=bool)
    rendered_mask = np.asarray(rendered_mask, dtype=bool)
    intersection = int(np.logical_and(alpha_mask, rendered_mask).sum())
    union = int(np.logical_or(alpha_mask, rendered_mask).sum())
    return {
        "alpha_pixels": int(alpha_mask.sum()),
        "rendered_pixels": int(rendered_mask.sum()),
        "mask_iou": intersection / union if union else 0.0,
        "bbox_iou": _bbox_iou(_bbox(alpha_mask), _bbox(rendered_mask)),
        "horizontal_flip_mask_iou": (
            int(np.logical_and(alpha_mask, np.fliplr(rendered_mask)).sum())
            / max(int(np.logical_or(alpha_mask, np.fliplr(rendered_mask)).sum()), 1)
        ),
    }


def _save_overlay(image: Image.Image, alpha_mask: np.ndarray, rendered_mask: np.ndarray, path: Path) -> None:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    overlay = rgb.copy()
    alpha_only = alpha_mask & ~rendered_mask
    render_only = rendered_mask & ~alpha_mask
    agreement = alpha_mask & rendered_mask
    overlay[agreement] = 0.65 * overlay[agreement] + 0.35 * np.array([0, 255, 0])
    overlay[alpha_only] = 0.55 * overlay[alpha_only] + 0.45 * np.array([0, 120, 255])
    overlay[render_only] = 0.55 * overlay[render_only] + 0.45 * np.array([255, 40, 40])
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pixal3d_repo", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_path", default="TencentARC/Pixal3D")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=1024, choices=(1024, 1536))
    parser.add_argument("--low_vram", action="store_true")
    parser.add_argument("--expected_pixal_commit", default=PINNED_PIXAL_COMMIT)
    args = parser.parse_args()

    pixal_repo = Path(args.pixal3d_repo).resolve()
    image_path = Path(args.image).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    actual_commit = _git_commit(pixal_repo)
    if actual_commit != args.expected_pixal_commit:
        raise RuntimeError(
            f"Pixal3D commit mismatch: expected {args.expected_pixal_commit}, got {actual_commit}"
        )

    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    sys.path.insert(0, str(pixal_repo))
    official = importlib.import_module("inference")
    pipeline_module = importlib.import_module("pixal3d.pipelines.pixal3d_image_to_3d")
    render_utils = importlib.import_module("pixal3d.utils.render_utils")
    renderer_module = importlib.import_module("pixal3d.renderers.mesh_renderer")
    representation_module = importlib.import_module("pixal3d.representations")
    import nvdiffrast.torch as dr
    import torch
    import trimesh

    capture: dict = {}
    original_preprocess = pipeline_module.Pixal3DImageTo3DPipeline.preprocess_image
    original_camera = official.get_camera_params_wild_moge
    original_to_glb = official.o_voxel.postprocess.to_glb

    def preprocess_wrapper(self, image, bg_color=(0, 0, 0)):
        return _instrumented_preprocess(self, image, bg_color, capture)

    def camera_wrapper(*camera_args, **camera_kwargs):
        params = original_camera(*camera_args, **camera_kwargs)
        capture["camera_params"] = dict(params)
        return params

    def to_glb_wrapper(**kwargs):
        capture["vertices"] = kwargs["vertices"].detach().cpu().numpy().astype(np.float32)
        capture["faces"] = kwargs["faces"].detach().cpu().numpy().astype(np.int32)
        capture["to_glb_remesh"] = bool(kwargs.get("remesh"))
        return _SkippedGlb()

    pipeline_module.Pixal3DImageTo3DPipeline.preprocess_image = preprocess_wrapper
    official.get_camera_params_wild_moge = camera_wrapper
    official.o_voxel.postprocess.to_glb = to_glb_wrapper
    started = time.perf_counter()
    try:
        official.run_inference(
            image_path=str(image_path),
            output_path=str(output_dir / "skipped_final_remesh.glb"),
            seed=args.seed,
            model_path=args.model_path,
            low_vram=args.low_vram,
            resolution=args.resolution,
        )
    finally:
        pipeline_module.Pixal3DImageTo3DPipeline.preprocess_image = original_preprocess
        official.get_camera_params_wild_moge = original_camera
        official.o_voxel.postprocess.to_glb = original_to_glb
    generation_seconds = time.perf_counter() - started

    required = {"vertices", "faces", "camera_params", "preprocessed_image", "cropped_alpha"}
    missing = sorted(required - capture.keys())
    if missing:
        raise RuntimeError(f"official inference instrumentation missed: {missing}")
    if capture.get("to_glb_remesh") is not True:
        raise RuntimeError("official inference no longer requests final remeshing")

    vertices, faces = capture["vertices"], capture["faces"]
    mesh = representation_module.Mesh(
        torch.from_numpy(vertices).cuda(), torch.from_numpy(faces).cuda()
    )
    camera = capture["camera_params"]
    extrinsics, intrinsics = render_utils.proj_camera_to_render_params(
        camera["camera_angle_x"], camera["distance"]
    )
    renderer = renderer_module.MeshRenderer(
        {
            "resolution": args.resolution,
            "near": 1,
            "far": 100,
            "ssaa": 1,
            "antialias": False,
        }
    )
    raster_capture = {}
    original_rasterize = dr.rasterize

    def rasterize_wrapper(*raster_args, **raster_kwargs):
        result = original_rasterize(*raster_args, **raster_kwargs)
        raster_capture["rast"] = result[0].detach()
        return result

    dr.rasterize = rasterize_wrapper
    try:
        rendered = renderer.render(
            mesh, extrinsics, intrinsics, return_types=["mask", "depth", "coord"]
        )
    finally:
        dr.rasterize = original_rasterize
    triangle_ids = raster_capture["rast"][0, :, :, -1].round().to(torch.int64).cpu().numpy()
    raster_uv = raster_capture["rast"][0, :, :, :2].cpu().numpy().astype(np.float32)
    mask = rendered.mask.detach().cpu().numpy() > 0.5
    depth = rendered.depth.detach().cpu().numpy().astype(np.float32)
    coords = rendered.coord.detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)
    surface_summary = validate_surface_buffers(
        triangle_ids, mask, depth, coords, num_faces=faces.shape[0]
    )
    raw_weights = np.concatenate(
        [raster_uv, (1.0 - raster_uv.sum(axis=-1, keepdims=True))], axis=-1
    )
    visible_faces = faces[triangle_ids[mask] - 1]
    visible_vertices = vertices[visible_faces]
    best_order = None
    best_error = float("inf")
    for order in permutations(range(3)):
        weights = raw_weights[mask][:, order]
        reconstructed = np.sum(visible_vertices * weights[:, :, None], axis=1)
        error = float(np.max(np.abs(reconstructed - coords[mask])))
        if error < best_error:
            best_error, best_order = error, order
    if best_order is None or best_error > 1e-4:
        raise RuntimeError(f"could not align raster barycentrics to face vertices: {best_error}")
    barycentric_weights = raw_weights[:, :, best_order]
    barycentric_weights[~mask] = 0

    transform = CropTransform(
        original_size=tuple(capture["original_size"]),
        resized_size=tuple(capture["resized_size"]),
        crop_box=tuple(capture["crop_box"]),
        render_size=(args.resolution, args.resolution),
    )
    round_trip_corners = np.asarray(
        [
            [0.0, 0.0],
            [float(transform.original_size[0] - 1), float(transform.original_size[1] - 1)],
        ]
    )
    round_trip_error = float(
        np.max(
            np.abs(
                transform.render_to_original(transform.original_to_render(round_trip_corners))
                - round_trip_corners
            )
        )
    )
    if round_trip_error > 1e-5:
        raise RuntimeError(f"coordinate round trip failed: {round_trip_error}")

    preprocessed = capture["preprocessed_image"].resize(
        (args.resolution, args.resolution), Image.Resampling.LANCZOS
    )
    alpha_mask = np.asarray(
        Image.fromarray(capture["cropped_alpha"]).resize(
            (args.resolution, args.resolution), Image.Resampling.NEAREST
        )
    ) > int(0.8 * 255)
    preprocessed.save(output_dir / "preprocessed.png")
    Image.fromarray((alpha_mask * 255).astype(np.uint8)).save(output_dir / "foreground_mask.png")
    Image.fromarray((mask * 255).astype(np.uint8)).save(output_dir / "rendered_mask.png")
    _save_overlay(preprocessed, alpha_mask, mask, output_dir / "projection_overlay.png")
    np.save(output_dir / "triangle_ids.npy", triangle_ids.astype(np.int32))
    np.save(output_dir / "barycentric_weights.npy", barycentric_weights.astype(np.float16))
    np.save(output_dir / "depth.npy", depth.astype(np.float16))
    np.save(output_dir / "coords.npy", coords.astype(np.float16))
    np.savez_compressed(output_dir / "raw_mesh.npz", vertices=vertices, faces=faces)

    raw_obj = output_dir / "raw_mesh.obj"
    trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(raw_obj)
    reloaded = trimesh.load(raw_obj, force="mesh", process=False)
    if reloaded.faces.shape != faces.shape or not np.array_equal(reloaded.faces, faces):
        raise RuntimeError("OBJ export/load changed raw face order")

    metadata = {
        "protocol": "pixal3d_raw_surface_interface_v1",
        "pixal3d_commit": actual_commit,
        "input_image": str(image_path),
        "seed": int(args.seed),
        "resolution": int(args.resolution),
        "low_vram": bool(args.low_vram),
        "attention_backend": os.environ["ATTN_BACKEND"],
        "generation_seconds": generation_seconds,
        "camera_params": camera,
        "raw_foreground_bbox": capture["raw_foreground_bbox"],
        "crop_transform": transform.to_dict(),
        "coordinate_round_trip_max_error": round_trip_error,
        "raw_mesh": {
            "vertices": int(vertices.shape[0]),
            "faces": int(faces.shape[0]),
            "obj_sha256": _sha256(raw_obj),
            "captured_before_final_remesh": True,
        },
        "surface_buffers": surface_summary,
        "barycentric_vertex_order": list(best_order),
        "barycentric_coordinate_max_error": best_error,
        "projection_metrics_for_review": _mask_metrics(alpha_mask, mask),
        "automatic_invariants_passed": True,
        "human_review_required": [
            "foreground_mask.png selects the intended SPair object, not a person/composite",
            "projection_overlay.png has no flip, rotation, or material offset",
        ],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

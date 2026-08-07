"""Audit raw PartField cosine inside a fixed FLUX attention candidate pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from pixal3d_probe_utils import CropTransform, face_rows_from_triangle_ids, sample_triangle_ids


def _normalize(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float32)
    norms = np.linalg.norm(rows, axis=-1, keepdims=True)
    return rows / np.maximum(norms, 1e-12)


def evaluate_query(
    *,
    source_feature: np.ndarray,
    candidate_features: np.ndarray,
    candidate_valid: np.ndarray,
    candidate_pck_hits: np.ndarray,
    current_correct: bool,
) -> dict:
    source_feature = np.asarray(source_feature, dtype=np.float32)
    candidate_features = np.asarray(candidate_features, dtype=np.float32)
    candidate_valid = np.asarray(candidate_valid, dtype=bool)
    candidate_pck_hits = np.asarray(candidate_pck_hits, dtype=bool)
    if candidate_features.ndim != 2 or source_feature.shape != candidate_features.shape[1:]:
        raise ValueError("source/candidate feature dimensions disagree")
    if candidate_valid.shape != candidate_features.shape[:1]:
        raise ValueError("candidate_valid must align with candidates")
    if candidate_pck_hits.shape != candidate_features.shape[:1]:
        raise ValueError("candidate_pck_hits must align with candidates")
    source_valid = bool(np.all(np.isfinite(source_feature))) and float(np.linalg.norm(source_feature)) > 0
    usable = candidate_valid & np.all(np.isfinite(candidate_features), axis=1)
    usable &= np.linalg.norm(candidate_features, axis=1) > 0
    routeable = source_valid and bool(np.any(usable))
    result = {
        "routeable": routeable,
        "top20_hit": bool(np.any(candidate_pck_hits)),
        "current_correct": bool(current_correct),
        "teacher_correct": False,
        "gt_rank": None,
        "selected_candidate_index": None,
    }
    if not routeable:
        return result
    scores = np.full(candidate_features.shape[0], -np.inf, dtype=np.float32)
    scores[usable] = (_normalize(candidate_features[usable]) @ _normalize(source_feature)).reshape(-1)
    order = np.argsort(-scores, kind="stable")
    order = order[np.isfinite(scores[order])]
    selected = int(order[0])
    valid_gt_positions = np.flatnonzero(candidate_pck_hits[order])
    result.update(
        {
            "teacher_correct": bool(candidate_pck_hits[selected]),
            "gt_rank": int(valid_gt_positions[0] + 1) if valid_gt_positions.size else None,
            "selected_candidate_index": selected,
            "selected_cosine": float(scores[selected]),
            "candidate_cosines": [float(score) if np.isfinite(score) else None for score in scores],
        }
    )
    return result


def _rank_summary(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    ranks = [row.get("gt_rank") for row in rows]
    return {
        "queries": len(rows),
        "top1": sum(rank == 1 for rank in ranks) / len(rows) if rows else None,
        "top3": sum(rank is not None and rank <= 3 for rank in ranks) / len(rows) if rows else None,
    }


def summarize_queries(rows: list[dict]) -> dict:
    routeable = [row for row in rows if row["routeable"]]
    residual = [
        row for row in routeable if row["top20_hit"] and not row["current_correct"]
    ]
    union_correct = sum(row["current_correct"] or row["teacher_correct"] for row in rows)
    return {
        "all": _rank_summary(rows),
        "routeable": _rank_summary(routeable),
        "strict_current_residual": _rank_summary(residual),
        "surface_coverage": len(routeable) / len(rows) if rows else None,
        "current_union_teacher": {
            "correct": int(union_correct),
            "queries": len(rows),
            "rate": union_correct / len(rows) if rows else None,
        },
    }


def _point_index(payload: dict, hit_field: str) -> dict[tuple[str, str, int], dict]:
    index = {}
    for pair in payload["pair_records"]:
        for point in pair["points"]:
            key = (pair["category"], pair["pair_json"], int(point["keypoint_index"]))
            index[key] = {
                "correct": bool(point[hit_field]),
                "source_point": point["source_point"],
                "target_point": point["target_point"],
            }
    return index


def _load_asset(asset_root: Path, category: str, image_name: str) -> dict:
    root = asset_root / category / Path(image_name).stem
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    triangle_ids = np.load(root / "triangle_ids.npy")
    mesh = np.load(root / "raw_mesh.npz")
    faces = mesh["faces"]
    vertex_features_path = root / "vertex_features.npy"
    face_features_path = root / "face_features.npy"
    if vertex_features_path.exists():
        features = np.load(vertex_features_path)
        feature_domain = "vertex"
        barycentric = np.load(root / "barycentric_weights.npy")
        if int(metadata["raw_mesh"]["vertices"]) != int(features.shape[0]):
            raise ValueError(f"PartField vertex rows disagree for {category}/{image_name}")
    elif face_features_path.exists():
        features = np.load(face_features_path)
        feature_domain = "face"
        barycentric = None
        if int(metadata["raw_mesh"]["faces"]) != int(features.shape[0]):
            raise ValueError(f"PartField face rows disagree for {category}/{image_name}")
    else:
        raise FileNotFoundError(f"no PartField features for {category}/{image_name}")
    return {
        "transform": CropTransform.from_dict(metadata["crop_transform"]),
        "triangle_ids": triangle_ids,
        "features": features,
        "feature_domain": feature_domain,
        "faces": faces,
        "barycentric": barycentric,
    }


def _sample_asset(asset: dict, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rendered = asset["transform"].original_to_render(np.asarray(points, dtype=np.float64))
    pixels = np.rint(rendered).astype(np.int64)
    inside = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < asset["triangle_ids"].shape[1])
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < asset["triangle_ids"].shape[0])
    )
    triangle_ids, _ = sample_triangle_ids(
        asset["triangle_ids"], asset["transform"], points
    )
    surface = triangle_ids > 0
    if asset["feature_domain"] == "face":
        rows, surface = face_rows_from_triangle_ids(triangle_ids, asset["features"])
        return rows, surface & inside, triangle_ids
    rows = np.full((len(points), asset["features"].shape[1]), np.nan, dtype=np.float32)
    valid = inside & surface
    if np.any(valid):
        face_vertices = asset["faces"][triangle_ids[valid] - 1]
        weights = asset["barycentric"][pixels[valid, 1], pixels[valid, 0]].astype(np.float32)
        vertex_rows = asset["features"][face_vertices]
        rows[valid] = np.sum(vertex_rows * weights[:, :, None], axis=1)
    return rows, valid, triangle_ids


def run_audit(attention_payload: dict, current_payload: dict, asset_root: Path) -> dict:
    current = _point_index(current_payload, "method_pck_hit")
    assets: dict[tuple[str, str], dict] = {}
    records = []
    categories: dict[str, list[dict]] = {}
    for pair in attention_payload["pair_records"]:
        category = pair["category"]
        source_key = (category, pair["src_image"])
        target_key = (category, pair["trg_image"])
        if source_key not in assets:
            assets[source_key] = _load_asset(asset_root, *source_key)
        if target_key not in assets:
            assets[target_key] = _load_asset(asset_root, *target_key)
        source_asset, target_asset = assets[source_key], assets[target_key]
        for point in pair["points"]:
            key = (category, pair["pair_json"], int(point["keypoint_index"]))
            if key not in current:
                raise KeyError(f"current audit does not align at {key}")
            source_rows, source_valid, source_triangles = _sample_asset(
                source_asset, np.asarray([point["source_point"]], dtype=np.float64)
            )
            candidates = sorted(point["candidates"], key=lambda row: int(row["attention_rank"]))
            candidate_points = np.asarray([row["pixel"] for row in candidates], dtype=np.float64)
            candidate_rows, candidate_valid, candidate_triangles = _sample_asset(
                target_asset, candidate_points
            )
            row = evaluate_query(
                source_feature=source_rows[0],
                candidate_features=candidate_rows,
                candidate_valid=candidate_valid & bool(source_valid[0]),
                candidate_pck_hits=np.asarray([row["pck_hit"] for row in candidates]),
                current_correct=current[key]["correct"],
            )
            row.update(
                {
                    "category": category,
                    "pair_json": pair["pair_json"],
                    "keypoint_index": int(point["keypoint_index"]),
                    "source_triangle_id": int(source_triangles[0]),
                    "candidate_triangle_ids": candidate_triangles.astype(int).tolist(),
                }
            )
            records.append(row)
            categories.setdefault(category, []).append(row)
    summary = summarize_queries(records)
    summary["categories"] = {
        category: summarize_queries(rows) for category, rows in categories.items()
    }
    union_rate = summary["current_union_teacher"]["rate"]
    summary["go_no_go"] = {
        "can_reach_75_by_oracle_union": bool(union_rate is not None and union_rate >= 0.75),
        "distillation_design_gate": bool(union_rate is not None and union_rate >= 0.77),
        "residual_capacity_gate": bool(
            summary["strict_current_residual"]["top1"] is not None
            and summary["strict_current_residual"]["top1"] >= 0.45
        ),
    }
    return {"summary": summary, "records": records}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention_audit_json", required=True)
    parser.add_argument("--current_audit_json", required=True)
    parser.add_argument("--asset_root", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()
    attention = json.loads(Path(args.attention_audit_json).read_text(encoding="utf-8"))
    current = json.loads(Path(args.current_audit_json).read_text(encoding="utf-8"))
    result = run_audit(attention, current, Path(args.asset_root))
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()

"""Pre-registered, label-free-teacher capacity audit for RoMa relation features.

Training targets are restricted to fixed, high-agreement RoMa candidates:
the rank-one candidate whose normalized bidirectional warp error is at most
0.05.  These targets and all relation features are built without SPair target
keypoints, PCK, category names, FLUX descriptors, or DINO descriptor input.
The fixed FLUX top-20 coordinates are the only proposal contract.

The resulting head is *not* a proposed final matcher.  It has no native
fallback and does not claim a PCK gain.  The heldout PCK report is an offline
capacity falsification: can a candidate-axis representation separate current
errors at all before we spend effort on safe fallback or final deployment?
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from audit_roma_internal_candidate_evidence import _load_roma_scale16_fields
from audit_roma_relation_feature_inventory import _load_projected_fields
from eval_spair_attention_top20_roma_identity import (
    _build_roma,
    _normalize_points,
    _run_roma_pair,
    _sample_field,
    _validate_audit,
    rank_attention_candidates_with_roma,
)
from roma_candidate_relation import CandidateConditionedRelationHead, pair_relation_block


FIXED_TEACHER_ERROR = 0.05
LOCAL_SCALES = (4, 8, 16)
METHOD_PROTOCOL = {
    "name": "frozen RoMa candidate-conditioned relation capacity audit",
    "status": "offline capacity falsification, not final matcher",
    "teacher": "fixed RoMa rank-one candidate with normalized bidirectional error <= 0.05",
    "teacher_threshold_origin": "pre-existing RoMa teacher-quality 5% normalized displacement tolerance; never PCK tuned",
    "feature_groups": ["local_scale4", "local_scale8", "local_scale16", "gp_forward_position", "gp_reverse_position"],
    "candidate_contract": "fixed FLUX mutual-attention top20 coordinates from supplied audit",
    "forbidden_training_inputs": ["target keypoint", "pck_hit", "category", "FLUX descriptor", "DINO descriptor", "current PCK"],
    "forbidden_final_claims": ["safe current-system override", "native fallback", "full SPair result"],
}


def _sample(field: torch.Tensor, points: torch.Tensor, image_size: Sequence[int]) -> torch.Tensor:
    if field.ndim != 4 or int(field.shape[0]) != 1:
        raise ValueError("RoMa field must be [1,C,H,W]")
    return _sample_field(
        field[0].permute(1, 2, 0).contiguous(), _normalize_points(points, image_size)
    )


def build_relation_groups(
    projected: Mapping[int, tuple[torch.Tensor, torch.Tensor]],
    pair_fields: Mapping[str, torch.Tensor],
    *,
    source_points: torch.Tensor,
    candidate_points: torch.Tensor,
    source_size: Sequence[int],
    target_size: Sequence[int],
) -> dict[str, torch.Tensor]:
    """Build unscored source/candidate relation tensors from frozen RoMa fields."""

    groups: dict[str, torch.Tensor] = {}
    for scale in LOCAL_SCALES:
        if scale not in projected:
            raise RuntimeError(f"official RoMa projected scale {scale} is unavailable")
        source, target = projected[scale]
        groups[f"local_scale{scale}"] = pair_relation_block(
            _sample(source, source_points, source_size),
            _sample(target, candidate_points, target_size),
        )
    # GP scalar coordinate agreement was rejected.  These two blocks retain
    # the full directed posterior/position vectors and leave their relation to
    # the candidate-set encoder; no cosine/error scalar is computed here.
    groups["gp_forward_position"] = pair_relation_block(
        _sample(pair_fields["source_gp"], source_points, source_size),
        _sample(pair_fields["target_position_basis"], candidate_points, target_size),
    )
    groups["gp_reverse_position"] = pair_relation_block(
        _sample(pair_fields["source_position_basis"], source_points, source_size),
        _sample(pair_fields["target_gp"], candidate_points, target_size),
    )
    return groups


def _pair_inputs(pair: Mapping[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, list[list[dict[str, Any]]]]:
    points = list(pair["points"])
    candidates = [sorted(row["candidates"], key=lambda item: int(item["attention_rank"])) for row in points]
    counts = {len(row) for row in candidates}
    if counts != {20}:
        raise ValueError(f"capacity audit requires exactly 20 candidates, got {sorted(counts)}")
    return (
        torch.tensor([row["source_point"] for row in points], dtype=torch.float32, device=device),
        torch.tensor([[[*candidate["pixel"]] for candidate in row] for row in candidates], dtype=torch.float32, device=device),
        candidates,
    )


def _append_groups(store: dict[str, list[torch.Tensor]], groups: Mapping[str, torch.Tensor], keep: torch.Tensor) -> None:
    for name, value in groups.items():
        store.setdefault(name, []).append(value[keep].detach().cpu().half())


def _extract_split(
    records: Sequence[dict[str, Any]], *, model: Any, dataset_path: Path, device: torch.device,
    amp_dtype: torch.dtype, for_training: bool,
) -> dict[str, Any]:
    """Extract features/teacher without reading GT, then attach eval labels afterward."""

    group_store: dict[str, list[torch.Tensor]] = {}
    labels: list[torch.Tensor] = []
    candidate_hits: list[torch.Tensor] = []
    identities: list[tuple[str, str, int, tuple[float, float], tuple[float, float]]] = []
    teacher_hits: list[torch.Tensor] = []
    certified_count = 0
    total_count = 0
    for pair in tqdm(records, desc="extract frozen RoMa relation features"):
        category = str(pair["category"])
        source_path = dataset_path / "JPEGImages" / category / str(pair["src_image"])
        target_path = dataset_path / "JPEGImages" / category / str(pair["trg_image"])
        if not source_path.is_file() or not target_path.is_file():
            raise FileNotFoundError(f"missing SPair image: {source_path} or {target_path}")
        with Image.open(source_path) as source_image, Image.open(target_path) as target_image:
            source_size = (int(source_image.height), int(source_image.width))
            target_size = (int(target_image.height), int(target_image.width))
        source_points, candidate_points, candidate_rows = _pair_inputs(pair, device)
        projected = _load_projected_fields(model, source_path, target_path, device, amp_dtype)
        pair_fields = _load_roma_scale16_fields(model, source_path, target_path, device, amp_dtype)
        groups = build_relation_groups(
            projected, pair_fields, source_points=source_points, candidate_points=candidate_points,
            source_size=source_size, target_size=target_size,
        )
        warp, certainty = _run_roma_pair(model, source_path, target_path, device)
        teacher = rank_attention_candidates_with_roma(
            source_points, candidate_points, source_size, target_size, warp, certainty
        )
        keep = teacher["bidirectional_error"].amin(dim=1).le(FIXED_TEACHER_ERROR)
        total_count += int(keep.numel())
        certified_count += int(keep.sum().item())
        if for_training:
            _append_groups(group_store, groups, keep)
            labels.append(teacher["order"][:, 0][keep].detach().cpu().long())
        else:
            all_rows = torch.ones(int(source_points.shape[0]), dtype=torch.bool, device=device)
            _append_groups(group_store, groups, all_rows)
            labels.append(teacher["order"][:, 0].detach().cpu().long())
            # Evaluation-only: PCK is explicitly read after all features and
            # teacher targets have been built. It is never passed to the head.
            candidate_hits.append(torch.tensor(
                [[bool(candidate["pck_hit"]) for candidate in row] for row in candidate_rows], dtype=torch.bool
            ))
            for point in pair["points"]:
                identities.append((
                    category, str(pair.get("pair_json", "")), int(point["keypoint_index"]),
                    tuple(float(value) for value in point["source_point"]),
                    tuple(float(value) for value in point["target_point"]),
                ))
            teacher_hits.append(candidate_hits[-1].gather(1, teacher["order"][:, :1].detach().cpu()).squeeze(1))
        del projected, pair_fields, groups, warp, certainty, teacher
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if not labels:
        raise RuntimeError("no relation rows were extracted")
    data = {
        "groups": {name: torch.cat(values, dim=0) for name, values in group_store.items()},
        "teacher_labels": torch.cat(labels, dim=0),
        "certified_queries": certified_count,
        "total_queries": total_count,
    }
    if for_training and int(data["teacher_labels"].numel()) == 0:
        raise RuntimeError("fixed RoMa certification yielded no train queries")
    if not for_training:
        data["candidate_hits"] = torch.cat(candidate_hits, dim=0)
        data["teacher_hits"] = torch.cat(teacher_hits, dim=0)
        data["identities"] = identities
    return data


def _current_lookup(path: str | None) -> dict[tuple[str, str, int, tuple[float, float], tuple[float, float]], bool]:
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("pair_records")
    if not isinstance(rows, list):
        raise ValueError("current audit must contain pair_records")
    result = {}
    for pair in rows:
        for point in pair["points"]:
            key = (
                str(pair["category"]), str(pair.get("pair_json", "")), int(point["keypoint_index"]),
                tuple(float(value) for value in point["source_point"]), tuple(float(value) for value in point["target_point"]),
            )
            result[key] = bool(point["method_pck_hit"])
    return result


def _train_head(data: Mapping[str, Any], *, device: torch.device, epochs: int, batch_queries: int, seed: int) -> CandidateConditionedRelationHead:
    group_dims = {name: int(value.shape[2]) for name, value in data["groups"].items()}
    head = CandidateConditionedRelationHead(group_dims).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-4)
    count = int(data["teacher_labels"].numel())
    generator = torch.Generator().manual_seed(seed)
    for _ in range(int(epochs)):
        head.train()
        for index in torch.randperm(count, generator=generator).split(int(batch_queries)):
            groups = {name: value.index_select(0, index).to(device=device, dtype=torch.float32) for name, value in data["groups"].items()}
            labels = data["teacher_labels"].index_select(0, index).to(device)
            loss = F.cross_entropy(head(groups), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return head.eval()


def _evaluate(head: CandidateConditionedRelationHead, data: Mapping[str, Any], current: Mapping[Any, bool], device: torch.device, batch_queries: int) -> dict[str, Any]:
    predictions: list[torch.Tensor] = []
    count = int(data["teacher_labels"].numel())
    with torch.inference_mode():
        for index in torch.arange(count).split(int(batch_queries)):
            groups = {name: value.index_select(0, index).to(device=device, dtype=torch.float32) for name, value in data["groups"].items()}
            predictions.append(head(groups).argmax(dim=1).cpu())
    predicted = torch.cat(predictions)
    hits = data["candidate_hits"].gather(1, predicted[:, None]).squeeze(1)
    teacher_hits = data["teacher_hits"]
    result = {
        "queries": count,
        "relation_top1": float(hits.float().mean()),
        "fixed_teacher_top1": float(teacher_hits.float().mean()),
        "teacher_agreement": float(predicted.eq(data["teacher_labels"]).float().mean()),
    }
    if current:
        current_hits = torch.tensor([bool(current.get(key, False)) for key in data["identities"]], dtype=torch.bool)
        residual = ~current_hits
        routeable = residual & data["candidate_hits"].any(dim=1)
        result["strict_current_residual"] = {
            "queries": int(routeable.sum()),
            "relation_top1": float(hits[routeable].float().mean()) if bool(routeable.any()) else None,
            "fixed_teacher_top1": float(teacher_hits[routeable].float().mean()) if bool(routeable.any()) else None,
        }
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--train_attention_audit_json", required=True)
    parser.add_argument("--eval_attention_audit_json", required=True)
    parser.add_argument("--eval_current_audit_json", required=True)
    parser.add_argument("--output_checkpoint", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--roma_weights", required=True)
    parser.add_argument("--roma_dinov2_weights", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--roma_coarse_res", type=int, default=560)
    parser.add_argument("--roma_upsample_res", type=int, default=864)
    parser.add_argument("--roma_precision", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_queries", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2027)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if float(FIXED_TEACHER_ERROR) != 0.05:
        raise RuntimeError("capacity protocol requires the fixed 0.05 RoMa teacher tolerance")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and args.roma_precision != "fp32":
        raise ValueError("CPU capacity audit requires --roma_precision fp32")
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.roma_precision]
    train_records = _validate_audit(json.loads(Path(args.train_attention_audit_json).read_text(encoding="utf-8")))
    eval_records = _validate_audit(json.loads(Path(args.eval_attention_audit_json).read_text(encoding="utf-8")))
    train_pairs = {str(row.get("pair_json", "")) for row in train_records}
    eval_pairs = {str(row.get("pair_json", "")) for row in eval_records}
    if train_pairs & eval_pairs:
        raise RuntimeError("train/eval attention audits must have disjoint pair_json records")
    model = _build_roma(args, device)
    root = Path(args.dataset_path)
    train_data = _extract_split(train_records, model=model, dataset_path=root, device=device, amp_dtype=amp_dtype, for_training=True)
    eval_data = _extract_split(eval_records, model=model, dataset_path=root, device=device, amp_dtype=amp_dtype, for_training=False)
    head = _train_head(train_data, device=device, epochs=args.epochs, batch_queries=args.batch_queries, seed=args.seed)
    current = _current_lookup(args.eval_current_audit_json)
    evaluation = _evaluate(head, eval_data, current, device, args.batch_queries)
    output = {
        "protocol": {**METHOD_PROTOCOL, "seed": args.seed, "epochs": args.epochs, "batch_queries": args.batch_queries, "roma_precision": args.roma_precision},
        "train": {"pairs": len(train_records), "certified_queries": train_data["certified_queries"], "total_queries": train_data["total_queries"], "feature_dims": {name: int(value.shape[2]) for name, value in train_data["groups"].items()}},
        "eval": {"pairs": len(eval_records), "certified_queries": eval_data["certified_queries"], "total_queries": eval_data["total_queries"], **evaluation},
        "go_no_go": {
            "continue_only_if": "heldout strict-current-residual relation_top1 materially exceeds the fixed-teacher control and exceeds 0.3429; otherwise close this relation route without full training",
            "not_a_final_result": True,
        },
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(output, indent=2), encoding="utf-8")
    torch.save({"format_version": 1, "protocol": output["protocol"], "group_dims": head.group_dims, "state_dict": head.cpu().state_dict()}, args.output_checkpoint)
    residual = evaluation.get("strict_current_residual", {})
    print(
        f"RoMa relation capacity: heldout top1={100.0 * evaluation['relation_top1']:.2f}, "
        f"teacher={100.0 * evaluation['fixed_teacher_top1']:.2f}, "
        f"residual={100.0 * residual['relation_top1']:.2f}" if residual.get("relation_top1") is not None else "RoMa relation capacity: no aligned current residual"
    )


if __name__ == "__main__":
    main()

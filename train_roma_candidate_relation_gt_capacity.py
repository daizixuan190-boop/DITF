"""GT-only capacity audit for frozen RoMa candidate relation features.

This script is deliberately not a matcher.  It answers a narrow question left
open by RoMa easy-teacher distillation: after giving a small candidate-set head
the *evaluation-only* SPair PCK-positive candidate sets, can frozen multi-scale
RoMa relation fields rank a valid member of the fixed FLUX attention top-20 on
the strict current-system residual?  It has no inference rule, fallback, or
claim of label-free performance.
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
from PIL import Image
from tqdm import tqdm

from audit_roma_internal_candidate_evidence import _load_roma_scale16_fields
from audit_roma_relation_feature_inventory import _load_projected_fields
from eval_spair_attention_top20_roma_identity import _build_roma, _validate_audit
from roma_candidate_relation import (
    CandidateConditionedRelationHead,
    multi_positive_listwise_loss,
)
from train_roma_candidate_relation_capacity import (
    _current_lookup,
    _pair_inputs,
    align_current_identities,
    build_relation_groups,
)


METHOD_PROTOCOL = {
    "name": "GT-only frozen RoMa candidate relation capacity audit",
    "status": "offline capacity diagnostic, not a matcher",
    "candidate_contract": "fixed FLUX mutual-attention top20 coordinates from supplied audit",
    "training_cohort": "strict current-system error with at least one PCK-valid attention candidate",
    "training_target": "all PCK-valid candidates in the fixed pool via multi-positive listwise loss",
    "feature_groups": ["local_scale4", "local_scale8", "local_scale16", "gp_forward_position", "gp_reverse_position"],
    "forbidden_final_claims": ["label-free method", "safe current-system override", "native fallback", "full SPair result"],
    "forbidden_feature_inputs": ["FLUX descriptor", "DINO descriptor", "category feature", "RoMa warp score", "attention score"],
    "gt_used_for_training": True,
    "gt_used_for_inference": False,
}


Identity = tuple[str, str, int, tuple[float, float], tuple[float, float]]


def _candidate_hits(rows: Sequence[Sequence[Mapping[str, Any]]]) -> torch.Tensor:
    return torch.tensor(
        [[bool(candidate["pck_hit"]) for candidate in candidates] for candidates in rows],
        dtype=torch.bool,
    )


def _pair_identities(pair: Mapping[str, Any]) -> list[Identity]:
    category = str(pair["category"])
    pair_json = str(pair.get("pair_json", ""))
    return [
        (
            category,
            pair_json,
            int(point["keypoint_index"]),
            tuple(float(value) for value in point["source_point"]),
            tuple(float(value) for value in point["target_point"]),
        )
        for point in pair["points"]
    ]


def _strict_residual_mask(
    identities: Sequence[Identity], candidate_hits: torch.Tensor, current: Mapping[Any, Any],
) -> torch.Tensor:
    current_hits = align_current_identities(identities, current)
    return (~current_hits) & candidate_hits.any(dim=1)


def _append_selected(
    store: dict[str, list[torch.Tensor]], groups: Mapping[str, torch.Tensor], keep: torch.Tensor,
) -> None:
    for name, value in groups.items():
        store.setdefault(name, []).append(value[keep].detach().cpu().half())


def _extract_training_data(
    records: Sequence[Mapping[str, Any]], *, current: Mapping[Any, Any], model: Any,
    dataset_path: Path, device: torch.device, amp_dtype: torch.dtype,
) -> dict[str, Any]:
    """Extract only strict-residual training rows after a validated current join."""

    group_store: dict[str, list[torch.Tensor]] = {}
    positives: list[torch.Tensor] = []
    selected_by_category: defaultdict[str, int] = defaultdict(int)
    total_queries = 0
    for pair in tqdm(records, desc="extract GT capacity train relations"):
        category = str(pair["category"])
        source_path = dataset_path / "JPEGImages" / category / str(pair["src_image"])
        target_path = dataset_path / "JPEGImages" / category / str(pair["trg_image"])
        if not source_path.is_file() or not target_path.is_file():
            raise FileNotFoundError(f"missing SPair image: {source_path} or {target_path}")
        with Image.open(source_path) as source_image, Image.open(target_path) as target_image:
            source_size = (int(source_image.height), int(source_image.width))
            target_size = (int(target_image.height), int(target_image.width))
        source_points, candidate_points, candidate_rows = _pair_inputs(pair, device)
        candidate_hits = _candidate_hits(candidate_rows)
        identities = _pair_identities(pair)
        keep = _strict_residual_mask(identities, candidate_hits, current).to(device)
        total_queries += len(identities)
        if bool(keep.any()):
            projected = _load_projected_fields(model, source_path, target_path, device, amp_dtype)
            pair_fields = _load_roma_scale16_fields(model, source_path, target_path, device, amp_dtype)
            groups = build_relation_groups(
                projected, pair_fields, source_points=source_points, candidate_points=candidate_points,
                source_size=source_size, target_size=target_size,
            )
            _append_selected(group_store, groups, keep)
            positives.append(candidate_hits[keep.cpu()])
            selected_by_category[category] += int(keep.sum().item())
            del projected, pair_fields, groups
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if not positives:
        raise RuntimeError("strict current residual has no routeable training queries")
    masks = torch.cat(positives, dim=0)
    if not bool(masks.any(dim=1).all()):  # pragma: no cover - guarded by the cohort mask.
        raise RuntimeError("selected training query has no PCK-valid candidate")
    return {
        "groups": {name: torch.cat(values, dim=0) for name, values in group_store.items()},
        "positive_masks": masks,
        "selected_queries": int(masks.shape[0]),
        "total_queries": total_queries,
        "mean_positive_candidates": float(masks.sum(dim=1).float().mean()),
        "selected_by_category": dict(sorted(selected_by_category.items())),
    }


def _train_head(
    data: Mapping[str, Any], *, device: torch.device, epochs: int, batch_queries: int, seed: int,
) -> CandidateConditionedRelationHead:
    group_dims = {name: int(value.shape[2]) for name, value in data["groups"].items()}
    head = CandidateConditionedRelationHead(group_dims).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-4)
    count = int(data["positive_masks"].shape[0])
    generator = torch.Generator().manual_seed(seed)
    for _ in range(int(epochs)):
        head.train()
        for index in torch.randperm(count, generator=generator).split(int(batch_queries)):
            groups = {
                name: value.index_select(0, index).to(device=device, dtype=torch.float32)
                for name, value in data["groups"].items()
            }
            positives = data["positive_masks"].index_select(0, index).to(device)
            loss = multi_positive_listwise_loss(head(groups), positives)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return head.eval()


def _metric_row(scores: torch.Tensor, positives: torch.Tensor) -> dict[str, Any]:
    order = scores.argsort(dim=1, descending=True)
    ordered_hits = positives.gather(1, order)
    values = {
        "queries": int(positives.shape[0]),
        "top1": float(ordered_hits[:, 0].float().mean()),
        "mean_positive_candidates": float(positives.sum(dim=1).float().mean()),
    }
    for count in (3, 5):
        values[f"top{count}"] = float(ordered_hits[:, :count].any(dim=1).float().mean())
    return values


def _merge_metric_rows(rows: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return _metric_row(torch.cat([item[0] for item in rows]), torch.cat([item[1] for item in rows]))


def _eval_pair_metadata(dataset_path: Path) -> dict[str, str]:
    root = dataset_path / "PairAnnotation"
    if not root.is_dir():
        raise FileNotFoundError(f"missing SPair PairAnnotation directory: {root}")
    result: dict[str, str] = {}
    for path in root.rglob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        audit_name = path.stem.rsplit("_", 1)[0] + ":" + path.stem.rsplit("_", 1)[1] + path.suffix
        result[audit_name] = str(payload.get("viewpoint_variation", "unknown"))
    return result


def _evaluate_streaming(
    head: CandidateConditionedRelationHead, records: Sequence[Mapping[str, Any]], *, current: Mapping[Any, Any],
    model: Any, dataset_path: Path, device: torch.device, amp_dtype: torch.dtype, batch_queries: int,
) -> dict[str, Any]:
    """Score every eval row, retaining only aggregate cohort metrics in RAM."""

    viewpoint = _eval_pair_metadata(dataset_path)
    strict_rows: list[tuple[torch.Tensor, torch.Tensor]] = []
    current_correct_rows: list[tuple[torch.Tensor, torch.Tensor]] = []
    category_rows: defaultdict[str, list[tuple[torch.Tensor, torch.Tensor]]] = defaultdict(list)
    viewpoint_rows: defaultdict[str, list[tuple[torch.Tensor, torch.Tensor]]] = defaultdict(list)
    total_queries = 0
    for pair in tqdm(records, desc="evaluate GT capacity relations"):
        category = str(pair["category"])
        pair_json = str(pair.get("pair_json", ""))
        source_path = dataset_path / "JPEGImages" / category / str(pair["src_image"])
        target_path = dataset_path / "JPEGImages" / category / str(pair["trg_image"])
        with Image.open(source_path) as source_image, Image.open(target_path) as target_image:
            source_size = (int(source_image.height), int(source_image.width))
            target_size = (int(target_image.height), int(target_image.width))
        source_points, candidate_points, candidate_rows = _pair_inputs(pair, device)
        positives = _candidate_hits(candidate_rows)
        identities = _pair_identities(pair)
        current_hits = align_current_identities(identities, current)
        routeable = (~current_hits) & positives.any(dim=1)
        current_correct_covered = current_hits & positives.any(dim=1)
        relevant = routeable | current_correct_covered
        total_queries += len(identities)
        if bool(relevant.any()):
            projected = _load_projected_fields(model, source_path, target_path, device, amp_dtype)
            pair_fields = _load_roma_scale16_fields(model, source_path, target_path, device, amp_dtype)
            groups = build_relation_groups(
                projected, pair_fields, source_points=source_points, candidate_points=candidate_points,
                source_size=source_size, target_size=target_size,
            )
            scores: list[torch.Tensor] = []
            selected = relevant.nonzero(as_tuple=False).squeeze(1)
            with torch.inference_mode():
                for index in selected.split(int(batch_queries)):
                    device_index = index.to(device)
                    batch = {
                        name: value.index_select(0, device_index).to(dtype=torch.float32)
                        for name, value in groups.items()
                    }
                    scores.append(head(batch).cpu())
            selected_scores = torch.cat(scores, dim=0)
            selected_positives = positives.index_select(0, selected.cpu())
            if bool(routeable.any()):
                route_scores = []
                route_masks = []
                for position, original in enumerate(selected.tolist()):
                    if bool(routeable[original]):
                        route_scores.append(selected_scores[position])
                        route_masks.append(selected_positives[position])
                route_score_tensor = torch.stack(route_scores)
                route_mask_tensor = torch.stack(route_masks)
                strict_rows.append((route_score_tensor, route_mask_tensor))
                category_rows[category].append((route_score_tensor, route_mask_tensor))
                viewpoint_rows[viewpoint.get(pair_json, "unknown")].append((route_score_tensor, route_mask_tensor))
            if bool(current_correct_covered.any()):
                correct_scores = []
                correct_masks = []
                for position, original in enumerate(selected.tolist()):
                    if bool(current_correct_covered[original]):
                        correct_scores.append(selected_scores[position])
                        correct_masks.append(selected_positives[position])
                current_correct_rows.append((torch.stack(correct_scores), torch.stack(correct_masks)))
            del projected, pair_fields, groups
        if device.type == "cuda":
            torch.cuda.empty_cache()
    strict = _merge_metric_rows(strict_rows)
    if strict is None:
        raise RuntimeError("strict current residual has no routeable evaluation queries")
    categories = {key: _merge_metric_rows(value) for key, value in sorted(category_rows.items())}
    viewpoints = {key: _merge_metric_rows(value) for key, value in sorted(viewpoint_rows.items())}
    category_top1 = sorted(value["top1"] for value in categories.values() if value and value["queries"] > 0)
    return {
        "total_queries": total_queries,
        "strict_current_residual": strict,
        "current_correct_covered_candidate_control": _merge_metric_rows(current_correct_rows),
        "category_strict_current_residual": categories,
        "viewpoint_strict_current_residual": viewpoints,
        "category_median_top1": float(np.median(category_top1)) if category_top1 else None,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--train_attention_audit_json", required=True)
    parser.add_argument("--eval_attention_audit_json", required=True)
    parser.add_argument("--train_current_audit_json", required=True)
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
    train_current = _current_lookup(args.train_current_audit_json)
    eval_current = _current_lookup(args.eval_current_audit_json)
    model = _build_roma(args, device)
    root = Path(args.dataset_path)
    train_data = _extract_training_data(
        train_records, current=train_current, model=model, dataset_path=root, device=device, amp_dtype=amp_dtype,
    )
    head = _train_head(train_data, device=device, epochs=args.epochs, batch_queries=args.batch_queries, seed=args.seed)
    evaluation = _evaluate_streaming(
        head, eval_records, current=eval_current, model=model, dataset_path=root, device=device,
        amp_dtype=amp_dtype, batch_queries=args.batch_queries,
    )
    residual = evaluation["strict_current_residual"]
    output = {
        "protocol": {**METHOD_PROTOCOL, "seed": args.seed, "epochs": args.epochs, "batch_queries": args.batch_queries, "roma_precision": args.roma_precision},
        "train": {"pairs": len(train_records), **{key: value for key, value in train_data.items() if key != "groups" and key != "positive_masks"}, "feature_dims": {name: int(value.shape[2]) for name, value in train_data["groups"].items()}},
        "eval": {"pairs": len(eval_records), **evaluation},
        "go_no_go": {
            "continue_only_if": "pair-heldout strict-current-residual top1 is at least 0.45, at least 0.10 above frozen RoMa 0.3429, and category median is at least 0.39; otherwise close accessible RoMa relation features without category-fold or full training",
            "pair_heldout_only": True,
            "not_a_final_result": True,
        },
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(output, indent=2), encoding="utf-8")
    torch.save({"format_version": 1, "protocol": output["protocol"], "group_dims": head.group_dims, "state_dict": head.cpu().state_dict()}, args.output_checkpoint)
    print(
        f"GT-only RoMa relation capacity: pair-heldout residual top1={100.0 * residual['top1']:.2f}, "
        f"top3={100.0 * residual['top3']:.2f}, category-median={100.0 * evaluation['category_median_top1']:.2f}"
    )


if __name__ == "__main__":
    main()

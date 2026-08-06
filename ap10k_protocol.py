"""Deterministic, non-destructive AP-10K correspondence benchmark preparation.

The protocol follows GeoAware-SC's ``prepare_ap10k.ipynb`` while avoiding its
destructive image moves and its non-deterministic ``os.walk`` ordering.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


SPLIT_FILES = (
    "ap10k-train-split1.json",
    "ap10k-test-split1.json",
    "ap10k-val-split1.json",
)
PAIR_SPLITS = {
    "intra-species": "test",
    "cross-species": "test_cross_species",
    "cross-family": "test_cross_family",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def visible_overlap(first: dict[str, Any], second: dict[str, Any]) -> float:
    """Reproduce the notebook's visibility-product test exactly."""
    first_kps = first["keypoints"]
    second_kps = second["keypoints"]
    if len(first_kps) != len(second_kps) or len(first_kps) % 3:
        raise ValueError("AP-10K keypoint arrays must have equal 3-value entries")
    return sum(
        (float(first_kps[index]) / 2.0) * (float(second_kps[index]) / 2.0)
        for index in range(2, len(first_kps), 3)
    )


def split_species_records(records: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    length = len(records)
    test_size = min(30, length)
    val_size = min(20, max(0, length - test_size))
    train_size = max(0, length - test_size - val_size)
    return {
        "train": list(records[:train_size]),
        "val": list(records[train_size : train_size + val_size]),
        "test": list(records[-test_size:]) if test_size else [],
    }


def _annotation_name(image_id: int) -> str:
    # Equivalent to GeoAware-SC: (str(id) + ".json").zfill(17).
    return (str(image_id) + ".json").zfill(17)


def _relative_annotation(record: dict[str, Any]) -> Path:
    return Path("ImageAnnotation") / record["supercategory"] / record["name"] / _annotation_name(record["id"])


def _load_source(raw_root: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    merged_annotations: list[dict[str, Any]] = []
    merged_images: list[dict[str, Any]] = []
    categories: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    for filename in SPLIT_FILES:
        path = raw_root / "annotations" / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing AP-10K split annotation: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        merged_annotations.extend(payload["annotations"])
        merged_images.extend(payload["images"])
        if not categories:
            categories = payload["categories"]
        hashes[filename] = sha256_file(path)

    unique_annotations: list[dict[str, Any]] = []
    seen_images: set[int] = set()
    for annotation in merged_annotations:
        image_id = int(annotation["image_id"])
        if image_id not in seen_images:
            seen_images.add(image_id)
            unique_annotations.append(annotation)

    images = {int(image["id"]): image for image in merged_images}
    category_map = {int(category["id"]): category for category in categories}
    records: list[dict[str, Any]] = []
    for annotation in unique_annotations:
        image_id = int(annotation["image_id"])
        category_id = int(annotation["category_id"])
        if image_id not in images or category_id not in category_map:
            continue
        image = images[image_id]
        category = category_map[category_id]
        # The image dictionary intentionally follows the annotation dictionary,
        # matching the notebook and making ``id`` the image id.
        records.append(
            {
                **annotation,
                **image,
                "name": category["name"],
                "supercategory": category["supercategory"],
            }
        )
    return records, hashes


def _load_crowd_ids(path: Path) -> set[int]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing GeoAware-SC crowd list: {path}")
    return {int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _find_raw_image(raw_root: Path, record: dict[str, Any], image_index: dict[int, Path]) -> Path:
    filename = record.get("file_name")
    if filename:
        direct = raw_root / "data" / str(filename)
        if direct.is_file():
            return direct
    image_id = int(record["id"])
    try:
        return image_index[image_id]
    except KeyError as error:
        raise FileNotFoundError(f"Cannot locate raw AP-10K image id {image_id}") from error


def _build_image_index(raw_root: Path) -> dict[int, Path]:
    index: dict[int, Path] = {}
    for path in (raw_root / "data").iterdir():
        if not path.is_file():
            continue
        digits = "".join(character for character in path.stem if character.isdigit())
        if digits:
            index[int(digits)] = path
    return index


def _write_pair(output: Path, source: dict[str, Any], target: dict[str, Any], category: str) -> tuple[str, str]:
    source_path = _relative_annotation(source).as_posix()
    target_path = _relative_annotation(target).as_posix()
    source_name = Path(source_path).stem
    target_name = Path(target_path).stem
    filename = f"{source_name}-{target_name}:{category}.json"
    output.mkdir(parents=True, exist_ok=True)
    (output / filename).write_text(
        json.dumps({"src_json_path": source_path, "trg_json_path": target_path}, indent=2),
        encoding="utf-8",
    )
    return source_path, target_path


def _semantic_pair_hash(pairs: Iterable[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for source, target in sorted(pairs):
        digest.update(f"{source}\t{target}\n".encode("utf-8"))
    return digest.hexdigest()


def _advance_intra_train_rng(
    rng: random.Random,
    species_order: Sequence[tuple[str, str]],
    splits: dict[tuple[str, str], dict[str, list[dict[str, Any]]]],
) -> None:
    """Advance RNG as the notebook's unwritten train generation would."""
    for key in species_order:
        records = list(splits[key]["train"])
        rng.shuffle(records)
        valid_count = sum(
            visible_overlap(source, target) >= 3.0
            for source, target in itertools.combinations(records, 2)
        )
        sample_count = min(50 * len(records), valid_count)
        if sample_count:
            rng.sample(range(valid_count), sample_count)


def _generate_intra(
    benchmark_root: Path,
    species_order: Sequence[tuple[str, str]],
    splits: dict[tuple[str, str], dict[str, list[dict[str, Any]]]],
) -> dict[str, Any]:
    rng = random.Random(42)
    _advance_intra_train_rng(rng, species_order, splits)
    all_pairs: list[tuple[str, str]] = []
    category_counts: dict[str, int] = {}
    output = benchmark_root / "PairAnnotation" / PAIR_SPLITS["intra-species"]
    for key in species_order:
        records = list(splits[key]["test"])
        rng.shuffle(records)
        possible = [
            pair for pair in itertools.combinations(records, 2) if visible_overlap(*pair) >= 3.0
        ]
        sampled = rng.sample(possible, len(possible)) if possible else []
        category_counts[key[1]] = len(sampled)
        all_pairs.extend(_write_pair(output, source, target, key[1]) for source, target in sampled)
    return {
        "pair_count": len(all_pairs),
        "pair_sha256": _semantic_pair_hash(all_pairs),
        "category_pair_counts": category_counts,
    }


def _generate_cross_species(
    benchmark_root: Path,
    species_order: Sequence[tuple[str, str]],
    splits: dict[tuple[str, str], dict[str, list[dict[str, Any]]]],
) -> dict[str, Any]:
    rng = random.Random(42)
    by_family: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key in species_order:
        by_family[key[0]].append(key)
    output = benchmark_root / "PairAnnotation" / PAIR_SPLITS["cross-species"]
    all_pairs: list[tuple[str, str]] = []
    category_counts: dict[str, int] = {}
    for family, species_keys in by_family.items():
        if len(species_keys) < 2:
            continue
        candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for first, second in itertools.combinations(species_keys, 2):
            candidates.extend(itertools.product(splits[first]["test"], splits[second]["test"]))
        # Reproduce a notebook bug: reversal draws are consumed but never stored.
        for _ in candidates:
            rng.random()
        possible = [pair for pair in candidates if visible_overlap(*pair) >= 3.0]
        sampled = rng.sample(possible, min(900, len(possible)))
        category_counts[family] = len(sampled)
        all_pairs.extend(_write_pair(output, source, target, family) for source, target in sampled)
    return {
        "pair_count": len(all_pairs),
        "pair_sha256": _semantic_pair_hash(all_pairs),
        "category_pair_counts": category_counts,
    }


def _generate_cross_family(
    benchmark_root: Path,
    species_order: Sequence[tuple[str, str]],
    splits: dict[tuple[str, str], dict[str, list[dict[str, Any]]]],
) -> dict[str, Any]:
    rng = random.Random(42)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for key in species_order:
        by_family[key[0]].extend(splits[key]["test"])
    output = benchmark_root / "PairAnnotation" / PAIR_SPLITS["cross-family"]
    all_pairs: list[tuple[str, str]] = []
    family_pair_counts: dict[str, int] = {}
    for first, second in itertools.combinations(by_family, 2):
        possible = [
            pair
            for pair in itertools.product(by_family[first], by_family[second])
            if visible_overlap(*pair) >= 3.0
        ]
        sampled = rng.sample(possible, min(30, len(possible)))
        family_pair_counts[f"{first}|{second}"] = len(sampled)
        all_pairs.extend(_write_pair(output, source, target, "all") for source, target in sampled)
    return {
        "pair_count": len(all_pairs),
        "pair_sha256": _semantic_pair_hash(all_pairs),
        "family_pair_counts": family_pair_counts,
    }


def prepare_benchmark(
    raw_root: Path,
    benchmark_root: Path,
    crowd_file: Path,
    *,
    link_images: bool = True,
) -> dict[str, Any]:
    raw_root = raw_root.resolve()
    benchmark_root = benchmark_root.resolve()
    if benchmark_root.exists() and any(benchmark_root.iterdir()):
        raise FileExistsError(f"Benchmark output must be empty or absent: {benchmark_root}")
    benchmark_root.mkdir(parents=True, exist_ok=True)

    records, source_hashes = _load_source(raw_root)
    crowd_ids = _load_crowd_ids(crowd_file)
    image_index = _build_image_index(raw_root) if link_images else {}
    species_records: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    species_order: list[tuple[str, str]] = []
    seen_species: set[tuple[str, str]] = set()

    for record in records:
        key = (record["supercategory"], record["name"])
        if key not in seen_species:
            seen_species.add(key)
            species_order.append(key)
        annotation_path = benchmark_root / _relative_annotation(record)
        annotation_path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(record)
        if int(record["id"]) in crowd_ids:
            payload["is_crowd"] = 1
        annotation_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        if link_images:
            source_image = _find_raw_image(raw_root, record, image_index)
            image_name = f"{Path(_annotation_name(record['id'])).stem}{source_image.suffix.lower()}"
            image_path = benchmark_root / "JPEGImages" / key[0] / key[1] / image_name
            image_path.parent.mkdir(parents=True, exist_ok=True)
            relative_source = os.path.relpath(source_image, image_path.parent)
            image_path.symlink_to(relative_source)

        if int(record["id"]) not in crowd_ids and int(record.get("num_keypoints", 0)) >= 3:
            species_records[key].append(record)

    splits = {key: split_species_records(species_records[key]) for key in species_order}
    split_counts: dict[str, dict[str, int]] = {}
    for key in species_order:
        species_dir = benchmark_root / "ImageAnnotation" / key[0] / key[1]
        split_counts[f"{key[0]}/{key[1]}"] = {}
        for split, split_records in splits[key].items():
            paths = [_relative_annotation(record).as_posix() for record in split_records]
            (species_dir / f"{split}_filtered.txt").write_text(
                "".join(f"{path}\n" for path in paths), encoding="utf-8"
            )
            split_counts[f"{key[0]}/{key[1]}"][split] = len(paths)

    settings = {
        "intra-species": _generate_intra(benchmark_root, species_order, splits),
        "cross-species": _generate_cross_species(benchmark_root, species_order, splits),
        "cross-family": _generate_cross_family(benchmark_root, species_order, splits),
    }
    manifest = {
        "protocol": "geoaware-ap10k-deterministic-test-v1",
        "source_root": str(raw_root),
        "source_split_order": list(SPLIT_FILES),
        "source_annotation_sha256": source_hashes,
        "crowd_file_sha256": sha256_file(crowd_file),
        "seed": 42,
        "record_order": "first occurrence in train/test/val split1 annotation order",
        "notebook_compatibility": {
            "filters_and_sampling": "reproduced",
            "intra_train_rng_state": "simulated without writing train pairs",
            "filesystem_order": "replaced with deterministic first-occurrence order",
            "cross_species_unused_reversal_draws": "reproduced",
            "cross_family_reported_count_bug": "fixed; manifest reports files written",
        },
        "image_count": len(records),
        "eligible_image_count": sum(len(values) for values in species_records.values()),
        "family_count": len({key[0] for key in species_order}),
        "species_count": len(species_order),
        "split_counts": split_counts,
        "settings": settings,
    }
    (benchmark_root / "protocol_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest

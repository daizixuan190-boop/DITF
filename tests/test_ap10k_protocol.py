import json
from pathlib import Path

from ap10k_protocol import prepare_benchmark, split_species_records, visible_overlap


def _keypoints(visible: int = 3) -> list[int]:
    values = []
    for index in range(17):
        values.extend([index, index, 2 if index < visible else 0])
    return values


def test_split_species_records_matches_geoaware_boundaries():
    records = [{"id": index} for index in range(75)]
    split = split_species_records(records)
    assert [len(split[name]) for name in ("train", "val", "test")] == [25, 20, 30]
    assert split["test"][0]["id"] == 45


def test_visible_overlap_reproduces_notebook_fractional_visibility():
    first = {"keypoints": _keypoints(3)}
    second = {"keypoints": _keypoints(3)}
    assert visible_overlap(first, second) == 3.0


def test_prepare_benchmark_writes_three_controlled_test_settings(tmp_path: Path):
    raw = tmp_path / "raw"
    annotations = raw / "annotations"
    annotations.mkdir(parents=True)
    categories = [
        {"id": 1, "name": "cat", "supercategory": "felidae"},
        {"id": 2, "name": "lion", "supercategory": "felidae"},
        {"id": 3, "name": "dog", "supercategory": "canidae"},
    ]
    images = []
    records = []
    for image_id in range(1, 10):
        category_id = 1 + (image_id - 1) // 3
        images.append({"id": image_id, "file_name": f"animal_{image_id}.jpg", "width": 100, "height": 80})
        records.append({
            "id": 100 + image_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [0, 0, 80, 70],
            "keypoints": _keypoints(),
            "num_keypoints": 3,
        })
    payloads = [
        {"annotations": records, "images": images, "categories": categories},
        {"annotations": [], "images": [], "categories": categories},
        {"annotations": [], "images": [], "categories": categories},
    ]
    for filename, payload in zip(
        ("ap10k-train-split1.json", "ap10k-test-split1.json", "ap10k-val-split1.json"),
        payloads,
    ):
        (annotations / filename).write_text(json.dumps(payload), encoding="utf-8")
    crowd = tmp_path / "crowd.txt"
    crowd.write_text("", encoding="utf-8")

    manifest = prepare_benchmark(raw, tmp_path / "benchmark", crowd, link_images=False)

    assert manifest["image_count"] == 9
    assert manifest["species_count"] == 3
    assert manifest["family_count"] == 2
    assert manifest["settings"]["intra-species"]["pair_count"] == 9
    assert manifest["settings"]["cross-species"]["pair_count"] == 9
    assert manifest["settings"]["cross-family"]["pair_count"] == 18
    assert (tmp_path / "benchmark" / "protocol_manifest.json").is_file()

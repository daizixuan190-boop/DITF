"""Attach official PartField vertex features to one Pixal3D probe asset."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe_dir", required=True)
    parser.add_argument("--partfield_features", required=True)
    args = parser.parse_args()
    probe_dir = Path(args.probe_dir)
    source = Path(args.partfield_features)
    metadata_path = probe_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    features = np.load(source, mmap_mode="r")
    expected_vertices = int(metadata["raw_mesh"]["vertices"])
    if features.ndim != 2 or features.shape != (expected_vertices, 448):
        raise ValueError(
            f"expected official PartField vertex features [{expected_vertices},448], "
            f"got {features.shape}"
        )
    if not np.all(np.isfinite(features)):
        raise ValueError("PartField features contain non-finite values")
    destination = probe_dir / "vertex_features.npy"
    shutil.copyfile(source, destination)
    metadata["partfield"] = {
        "feature_domain": "raw_mesh_vertex",
        "rows": int(features.shape[0]),
        "dimensions": int(features.shape[1]),
        "preprocess_mesh": False,
        "official_config": "configs/final/demo.yaml with correspondence_demo vertex_feature=True",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata["partfield"], indent=2))


if __name__ == "__main__":
    main()

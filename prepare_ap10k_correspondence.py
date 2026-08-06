"""Build the GeoAware-SC AP-10K test benchmark without copying raw images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ap10k_protocol import prepare_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument(
        "--crowd_file",
        type=Path,
        default=Path("data/ap-10k_is_crowd.txt"),
        help="GeoAware-SC's released AP-10K crowd-image id list",
    )
    parser.add_argument(
        "--no_image_links",
        action="store_true",
        help="Prepare annotations and pairs only (for protocol audits/tests)",
    )
    return parser


def main(args: argparse.Namespace) -> None:
    manifest = prepare_benchmark(
        args.raw_root,
        args.output_root,
        args.crowd_file,
        link_images=not args.no_image_links,
    )
    print(json.dumps({
        "image_count": manifest["image_count"],
        "eligible_image_count": manifest["eligible_image_count"],
        "family_count": manifest["family_count"],
        "species_count": manifest["species_count"],
        "pair_counts": {
            setting: result["pair_count"] for setting, result in manifest["settings"].items()
        },
        "manifest": str(args.output_root / "protocol_manifest.json"),
    }, indent=2))


if __name__ == "__main__":
    main(build_parser().parse_args())

import argparse
import csv
import json
import os
from typing import Any

import numpy as np


RISK_KEYS = [
    "shift_ratio_src",
    "content_ratio_src",
    "interaction_ratio_src",
    "post_topk_ratio_src",
    "post_highscale_ratio_src",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate recovery ceiling from existing per-point SPair diagnostics without rerunning feature extraction."
    )
    parser.add_argument("--records_csv", type=str, required=True, help="Path to per_point_records.csv.")
    parser.add_argument("--output_json", type=str, required=True, help="Path to save oracle ceiling summary.")
    parser.add_argument(
        "--risk_fracs",
        nargs="+",
        type=float,
        default=[0.01, 0.02, 0.05, 0.1, 0.2],
        help="Fractions of highest-risk points to mark as oracle-fixable.",
    )
    return parser.parse_args()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_records(csv_path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: dict[str, Any] = {}
            for key, value in row.items():
                if value is None:
                    parsed[key] = value
                    continue
                if key in {"category", "pair_name", "src_imname", "trg_imname"}:
                    parsed[key] = value
                    continue
                if value in {"", "None"}:
                    parsed[key] = None
                    continue
                try:
                    if key == "correct":
                        parsed[key] = int(float(value))
                    else:
                        parsed[key] = float(value)
                except ValueError:
                    parsed[key] = value
            records.append(parsed)
    return records


def oracle_curve(records: list[dict[str, Any]], risk_key: str, descending: bool) -> dict[str, Any]:
    total = len(records)
    base_error = float(1.0 - np.mean([r["correct"] for r in records]))
    ordered = sorted(records, key=lambda r: float(r[risk_key]), reverse=descending)

    curve = []
    for frac in args.risk_fracs:
        k = max(1, int(round(total * frac)))
        chosen = ordered[:k]
        chosen_errors = sum(1 for r in chosen if int(r["correct"]) == 0)
        improved_error = max(total * base_error - chosen_errors, 0) / total
        curve.append(
            {
                "risk_fraction": frac,
                "num_selected": k,
                "selected_error_rate": float(chosen_errors / k),
                "oracle_error_rate": float(improved_error),
                "absolute_gain": float(base_error - improved_error),
                "error_recall": float(chosen_errors / max(int(round(total * base_error)), 1)),
            }
        )

    top_decile = ordered[: max(1, total // 10)]
    bottom_decile = ordered[-max(1, total // 10) :]
    return {
        "base_error_rate": base_error,
        "high_risk_decile_error_rate": float(1.0 - np.mean([r["correct"] for r in top_decile])),
        "low_risk_decile_error_rate": float(1.0 - np.mean([r["correct"] for r in bottom_decile])),
        "curve": curve,
    }


def main():
    global args
    args = parse_args()
    ensure_dir(os.path.dirname(args.output_json) or ".")
    records = load_records(args.records_csv)

    summary = {
        "records_csv": args.records_csv,
        "num_points": len(records),
        "overall_error_rate": float(1.0 - np.mean([r["correct"] for r in records])) if records else None,
        "oracle_recovery": {},
    }

    direction = {
        "shift_ratio_src": True,
        "content_ratio_src": False,
        "interaction_ratio_src": False,
        "post_topk_ratio_src": True,
        "post_highscale_ratio_src": True,
    }

    for key in RISK_KEYS:
        if key not in records[0]:
            continue
        summary["oracle_recovery"][key] = oracle_curve(records, key, descending=direction[key])

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved oracle ceiling summary to: {args.output_json}")
    print(f"Num points: {summary['num_points']}")
    print(f"Overall error rate: {summary['overall_error_rate']}")
    for key, value in summary["oracle_recovery"].items():
        top10 = next((x for x in value["curve"] if abs(x["risk_fraction"] - 0.1) < 1e-9), None)
        if top10 is not None:
            print(
                f"{key}: top10% oracle gain={top10['absolute_gain']:.4f}, selected_error={top10['selected_error_rate']:.4f}, error_recall={top10['error_recall']:.4f}"
            )


if __name__ == "__main__":
    main()

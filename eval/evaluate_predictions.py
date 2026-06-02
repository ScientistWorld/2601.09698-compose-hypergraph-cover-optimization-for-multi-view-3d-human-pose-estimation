#!/usr/bin/env python3
"""CLI for evaluating method JSON outputs against data JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

try:
    from eval.metrics import evaluate_pose_predictions
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from eval.metrics import evaluate_pose_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="JSON file with frames -> predicted_poses")
    parser.add_argument("--data", required=True, help="JSON file with frames -> ground_truth")
    parser.add_argument("--output", default="scoring/scores.json", help="Where to write metrics JSON")
    parser.add_argument("--experiment", default="synthetic_crossing", help="Top-level experiment name")
    parser.add_argument("--method-name", default="compose", help="Method key under results")
    parser.add_argument(
        "--unit-scale",
        type=float,
        default=1.0,
        help="Scale both GT and prediction coordinates before scoring, e.g. 1000 for meters to millimeters",
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Write the metrics object directly instead of nesting under --experiment",
    )
    return parser.parse_args()


def load_json(path: str) -> object:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    args = parse_args()
    metrics = evaluate_pose_predictions(
        prediction_payload=load_json(args.predictions),
        data_payload=load_json(args.data),
        unit_scale=args.unit_scale,
    )
    output_path = Path(args.output)
    if args.flat:
        output = metrics
    else:
        if output_path.exists():
            output = json.loads(output_path.read_text())
        else:
            output = {"experiments": {}}
        output.setdefault("experiments", {}).setdefault(args.experiment, {}).setdefault("results", {})
        output["experiments"][args.experiment]["results"][args.method_name] = metrics
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

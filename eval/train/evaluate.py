#!/usr/bin/env python3
"""Train-slice evaluator for precomputed multi-person 3D pose predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

try:
    from eval.metrics import evaluate_pose_predictions
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from eval.metrics import evaluate_pose_predictions


SLICE_NAME = "train"
SPLIT_DESCRIPTION = "Even-indexed scenes from the synthetic COMPOSE evaluation set."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="JSON file with precomputed predicted poses.")
    parser.add_argument("--data", required=True, help="JSON file with benchmark scenes.")
    parser.add_argument("--output", required=True, help="Where to write the train-slice scores JSON.")
    parser.add_argument("--experiment", required=True, help="Top-level experiment name.")
    parser.add_argument("--method-name", required=True, help="Method key under experiment results.")
    parser.add_argument("--unit-scale", type=float, default=1.0, help="Scale coordinates before scoring.")
    return parser.parse_args()


def load_json(path: str) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def split_frame_ids(data_payload: Any) -> list[str]:
    if isinstance(data_payload, dict) and isinstance(data_payload.get("scenes"), list):
        return [str(scene["scene_id"]) for idx, scene in enumerate(data_payload["scenes"]) if idx % 2 == 0]
    if isinstance(data_payload, dict) and isinstance(data_payload.get("frames"), list):
        return [str(frame.get("frame_id", frame.get("id", idx))) for idx, frame in enumerate(data_payload["frames"]) if idx % 2 == 0]
    if isinstance(data_payload, dict) and isinstance(data_payload.get("frames"), dict):
        return [str(frame_id) for idx, frame_id in enumerate(sorted(data_payload["frames"])) if idx % 2 == 0]
    raise ValueError("Train/test slicing expects a data payload with scenes or frames.")


def filter_data_payload(data_payload: Any, keep_ids: set[str]) -> Any:
    if isinstance(data_payload, dict) and isinstance(data_payload.get("scenes"), list):
        sliced = dict(data_payload)
        sliced["scenes"] = [scene for scene in data_payload["scenes"] if str(scene["scene_id"]) in keep_ids]
        return sliced
    if isinstance(data_payload, dict) and isinstance(data_payload.get("frames"), list):
        sliced = dict(data_payload)
        sliced["frames"] = [
            frame
            for idx, frame in enumerate(data_payload["frames"])
            if str(frame.get("frame_id", frame.get("id", idx))) in keep_ids
        ]
        return sliced
    if isinstance(data_payload, dict) and isinstance(data_payload.get("frames"), dict):
        sliced = dict(data_payload)
        sliced["frames"] = {
            str(frame_id): frame for frame_id, frame in data_payload["frames"].items() if str(frame_id) in keep_ids
        }
        return sliced
    raise ValueError("Train/test slicing expects a data payload with scenes or frames.")


def load_existing_output(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"experiments": {}}


def main() -> int:
    args = parse_args()
    data_payload = load_json(args.data)
    scene_ids = split_frame_ids(data_payload)
    sliced_data = filter_data_payload(data_payload, set(scene_ids))
    metrics = evaluate_pose_predictions(
        prediction_payload=load_json(args.predictions),
        data_payload=sliced_data,
        unit_scale=args.unit_scale,
    )

    output_path = Path(args.output)
    output = load_existing_output(output_path)
    output["slice"] = SLICE_NAME
    output["split"] = {
        "slice": SLICE_NAME,
        "strategy": SPLIT_DESCRIPTION,
        "frame_ids": scene_ids,
    }
    output.setdefault("prediction_files", {})[args.method_name] = args.predictions
    experiment = output.setdefault("experiments", {}).setdefault(args.experiment, {})
    experiment["_slice"] = SLICE_NAME
    experiment["_frame_ids"] = scene_ids
    experiment.setdefault("results", {})[args.method_name] = metrics

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

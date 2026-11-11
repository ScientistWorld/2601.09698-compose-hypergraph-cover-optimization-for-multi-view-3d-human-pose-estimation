"""Run registered predictors on hidden test inputs (self-contained copy).

A *method* is any directory ``checkpoints/<name>/`` containing an executable
entry point ``predict.sh``.  The evaluator calls it as

    bash checkpoints/<name>/predict.sh <inputs.json> <predictions.json>

from the workspace root, with no network and a wall-clock limit.
``inputs.json`` holds cameras and 2D detections only (no 3D answers):

    {"format": "compose_gym_inputs_v2", "units": "meters",
     "joint_names": [15 names],
     "frames": [{"frame_id": str,
                 "cameras": [{"camera_id", "image_size", "K", "R", "t"}],
                 "views": [{"camera_id": str,
                            "detections": [{"keypoints": [[u, v, conf] x 15], "score": s}]}]}]}

Projection convention: x_cam = R @ X + t (metres), [u, v, 1] ~ K @ x_cam.
The predictor must write
``{"frames": {frame_id: {"predicted_poses": [{"joints_3d": [[x, y, z] x 15], "score": s}]}}}``.

Pure standard-library Python.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import signal
import subprocess
from typing import Any

WORKSPACE = Path(__file__).resolve().parents[2]
CHECKPOINTS = WORKSPACE / "checkpoints"
METHOD_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def discover_predictors(root: Path = CHECKPOINTS) -> dict[str, Path]:
    """Every ``<root>/<name>/predict.sh`` on disk, keyed by method name."""

    found = {}
    if root.is_dir():
        for script in sorted(root.glob("*/predict.sh")):
            name = script.parent.name
            if METHOD_NAME_RE.match(name):
                found[name] = script
    return found


def strip_answers(benchmark: dict[str, Any]) -> dict[str, Any]:
    """Inputs-only view of a benchmark (cameras + detections)."""

    return {
        "format": "compose_gym_inputs_v2",
        "units": benchmark.get("units", "meters"),
        "joint_names": benchmark["joint_names"],
        "frames": [
            {"frame_id": frame["frame_id"], "cameras": frame["cameras"], "views": frame["views"]}
            for frame in benchmark["frames"]
        ],
    }


def run_predictor(script: Path, inputs_path: Path, output_path: Path, log_path: Path, timeout_s: int) -> str:
    """Run one predictor; returns ``"ok"`` or a failure reason."""

    if output_path.exists():
        output_path.unlink()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        try:
            proc = subprocess.Popen(
                ["bash", str(script), str(inputs_path), str(output_path)],
                cwd=WORKSPACE,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as exc:
            return f"failed: could not start predictor ({exc})"
        try:
            returncode = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                pass
            proc.wait()
            return f"failed: predictor exceeded the {timeout_s}s limit"
    if returncode != 0:
        return f"failed: predictor exited with code {returncode}"
    if not output_path.is_file():
        return "failed: predictor wrote no prediction file"
    return "ok"


def load_prediction_file(path: Path) -> tuple[Any, str]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), "ok"
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, f"failed: unreadable prediction file ({exc.__class__.__name__})"

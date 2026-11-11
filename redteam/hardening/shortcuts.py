#!/usr/bin/env python3
"""Hardening audit shortcut predictors (each one tries to score without solving the task).

    python redteam/hardening/shortcuts.py MODE INPUTS OUTPUT
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def reference(inputs: str, scratch: Path) -> dict:
    out = scratch / "reference_predictions.json"
    subprocess.run(
        [sys.executable, str(ROOT / "method" / "compose.py"), "--inputs", inputs, "--output", str(out),
         "--config", str(ROOT / "checkpoints" / "compose" / "config.json")],
        check=True,
    )
    return json.loads(out.read_text())["frames"]


def frame_ids(inputs: str) -> list[str]:
    return [str(f["frame_id"]) for f in json.loads(Path(inputs).read_text())["frames"]]


def main() -> None:
    mode, inputs, output = sys.argv[1:4]
    scratch = Path(output).parent
    ids = frame_ids(inputs)
    payload: dict = {}
    if mode == "crash":
        sys.exit(1)
    if mode == "empty":
        frames = {fid: {"predicted_poses": []} for fid in ids}
    elif mode == "constant":
        frames = {fid: {"predicted_poses": [{"joints_3d": [[0.0, -1.0, 0.0]] * 15, "score": 1.0}]} for fid in ids}
    elif mode == "forged_metrics":
        frames = {fid: {"predicted_poses": [{"joints_3d": [[0.0, -1.0, 0.0]] * 15, "score": 1.0,
                                            "mpjpe_mm": 0.0, "matched": True}]} for fid in ids}
        payload.update({"ap_150mm": 100.0, "ap_25mm": 100.0, "recall_500mm": 100.0, "mpjpe_mm": 0.0,
                        "valid": True, "metrics": {"ap_150mm": 100.0}})
    elif mode == "train_gt_lookup":
        bench = json.loads((ROOT / "data" / "downloads" / "panoptic_train.json").read_text())
        frames = {}
        for k, fid in enumerate(ids):
            people = bench["frames"][k % len(bench["frames"])]["people"]
            frames[fid] = {"predicted_poses": [{"joints_3d": p["joints_3d"], "score": 1.0} for p in people]}
    elif mode == "identity_grouping":
        data = json.loads(Path(inputs).read_text())
        frames = {}
        for f in data["frames"]:
            ids_seen = {det.get(key) for v in f["views"] for det in v["detections"]
                        for key in ("person_id", "track_id", "_source", "source_person") if det.get(key) is not None}
            frames[str(f["frame_id"])] = {"predicted_poses": []}
            payload["identity_fields_found"] = payload.get("identity_fields_found", 0) + len(ids_seen)
    elif mode == "hidden_source_read":
        frames = {fid: {"predicted_poses": []} for fid in ids}
        try:
            from eval.test.scenes import build_test_scenes  # an agent never sees this module

            bench = build_test_scenes()
            by_id = {f["frame_id"]: f["people"] for f in bench["frames"]}
            frames = {fid: {"predicted_poses": [{"joints_3d": p["joints_3d"], "score": 1.0}
                                                for p in by_id.get(fid, [])]} for fid in ids}
            payload["leak"] = "answers were readable"
        except Exception as exc:  # noqa: BLE001
            payload["leak"] = f"blocked: {exc.__class__.__name__}"
    else:
        ref = reference(inputs, scratch)
        data = json.loads(Path(inputs).read_text())
        counts = {str(f["frame_id"]): sum(len(v["detections"]) for v in f["views"]) for f in data["frames"]}
        frames = {}
        for fid in ids:
            poses = ref.get(fid, {}).get("predicted_poses", [])
            if mode == "partial_pose_nan":
                poses = [dict(p, joints_3d=[j if k in (0, 1, 2, 3, 9, 6) else [math.nan] * 3
                                            for k, j in enumerate(p["joints_3d"])]) for p in poses]
            elif mode == "truncated_joints":
                poses = [dict(p, joints_3d=p["joints_3d"][:6]) for p in poses]
            elif mode == "null_coordinates":
                poses = [dict(p, joints_3d=[[None, None, None]] * 15) for p in poses]
            elif mode == "infinite":
                poses = [dict(p, joints_3d=[[math.inf, 0.0, 0.0]] * 15) for p in poses]
            elif mode == "too_many_poses":
                poses = poses + [dict(p, score=0.001) for p in poses for _ in range(40)]
            elif mode == "missing_frames":
                if int(fid.split("_")[-1]) % 2:
                    continue
            elif mode == "spray_recall":
                # Pad every frame up to its detection count with plausible human
                # poses (public train annotations) at the lowest score.
                if "train_people" not in payload:
                    bench = json.loads((ROOT / "data" / "downloads" / "panoptic_train.json").read_text())
                    payload["train_people"] = [p["joints_3d"] for f in bench["frames"] for p in f["people"]]
                limit = counts[fid]
                pool = payload["train_people"]
                start = (len(frames) * 7) % len(pool)
                extra = [{"joints_3d": pool[(start + k) % len(pool)], "score": 1e-6}
                         for k in range(max(0, limit - len(poses)))]
                poses = poses + extra
            elif mode == "out_of_volume":
                poses = [dict(p, joints_3d=[[x * 100.0 for x in j] for j in p["joints_3d"]]) for p in poses]
            frames[fid] = {"predicted_poses": poses}
    payload.pop("train_people", None)
    payload["frames"] = frames
    text = json.dumps(payload, allow_nan=True)
    Path(output).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

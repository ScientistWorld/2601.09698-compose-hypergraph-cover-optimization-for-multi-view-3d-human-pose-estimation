#!/usr/bin/env python3
"""Hardening audit variants of the reference (not part of the gym's method).

    python redteam/hardening/variants.py MODE INPUTS OUTPUT

Modes
  robust        better variant: the reference association (same hyperedges, same
                ILP, same tuned config) followed by per-joint confidence-weighted
                DLT with iterative rejection of the worst view, and a pose score
                that ranks poses seen in more views first.
  pairs_only    ablation of the paper's claim: hyperedges limited to two views.
  three_cams    observation ablation: only cameras 1, 3 and 5 are used.
  old_operating_point
                the previous gym's simplification: reprojection cost averaged
                (not summed) over views with tau=900, lambda=0.003, gamma=0.06,
                s_single=0.03.
  bruteforce    no geometric pruning: every hyperedge of every view subset
                enters the ILP (tau = infinity).
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import method.compose as compose  # noqa: E402

CONFIG = json.loads((ROOT / "checkpoints" / "compose" / "config.json").read_text())


def weighted_dlt(pmats: np.ndarray, uv: np.ndarray, w: np.ndarray) -> np.ndarray:
    rows = []
    for p, (u, v), wt in zip(pmats, uv, w):
        for r in (u * p[2] - p[0], v * p[2] - p[1]):
            rows.append(wt * r / max(np.linalg.norm(r), 1e-12))
    _, _, vt = np.linalg.svd(np.asarray(rows))
    h = vt[-1]
    return h[:3] / h[3] if abs(h[3]) > 1e-12 else np.full(3, np.nan)


def robust_pose(frame: compose.Frame, refs: tuple[tuple[int, int], ...]) -> np.ndarray:
    num_joints = frame.keypoints[refs[0][0]].shape[1]
    pose = np.zeros((num_joints, 3))
    for j in range(num_joints):
        pm = np.stack([frame.pmats[v] for v, _ in refs])
        uv = np.stack([frame.keypoints[v][i, j, :2] for v, i in refs])
        w = np.maximum(np.asarray([frame.keypoints[v][i, j, 2] for v, i in refs]), 0.05)
        active = list(range(len(refs)))
        while True:
            x = weighted_dlt(pm[active], uv[active], w[active])
            h = pm[active] @ np.append(x, 1.0)
            err = np.linalg.norm(h[:, :2] / h[:, 2:3] - uv[active], axis=1)
            worst = int(np.argmax(err))
            if len(active) > 2 and err[worst] > max(15.0, 2.5 * float(np.median(err))):
                active.pop(worst)
                continue
            break
        pose[j] = x
    return pose


def run(mode: str, inputs: str, output: str) -> None:
    frames, _ = compose.load_inputs(inputs)
    config = dict(CONFIG)
    if mode == "pairs_only":
        config["max_edge_size"] = 2
    elif mode == "bruteforce":
        config["tau"] = float("inf")
        config["max_edge_size"] = None
    elif mode == "old_operating_point":
        original = compose.reprojection_cost
        compose.reprojection_cost = lambda pm, uv, pts: original(pm, uv, pts) / pm.shape[1]
        config.update({"tau": 900.0, "lambda": 0.003, "gamma": 0.06, "singleton_score": 0.03, "max_edge_size": None})
    result = {}
    for frame in frames:
        if mode == "three_cams":
            frame.keypoints = [kp if v in (0, 2, 4) else kp[:0] for v, kp in enumerate(frame.keypoints)]
            frame.scores = [sc if v in (0, 2, 4) else sc[:0] for v, sc in enumerate(frame.scores)]
        if mode == "robust":
            cands = compose.generate_candidates(frame, float(config["tau"]), config.get("max_edge_size"))
            selected = compose.solve_exact_cover(
                frame, cands, float(config["lambda"]), float(config["gamma"]), float(config["singleton_score"])
            )
            poses = []
            for cand, _ in selected:
                pose = robust_pose(frame, cand.refs)
                if np.all(np.isfinite(pose)):
                    conf = float(np.mean([frame.scores[v][i] for v, i in cand.refs]))
                    poses.append({"joints_3d": pose.tolist(), "score": len(cand.refs) + conf})
        else:
            poses = compose.predict_frame(frame, config)
        result[frame.frame_id] = {"predicted_poses": poses}
    Path(output).write_text(json.dumps({"frames": result}), encoding="utf-8")


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], sys.argv[3])

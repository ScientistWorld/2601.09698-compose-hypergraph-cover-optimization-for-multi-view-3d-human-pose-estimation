#!/usr/bin/env python3
"""Deterministic synthetic multi-view pose data for COMPOSE-style evaluation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


SKELETON = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.28],
        [-0.18, 0.0, 0.14],
        [0.18, 0.0, 0.14],
        [-0.10, 0.0, -0.42],
        [0.10, 0.0, -0.42],
    ],
    dtype=float,
)


def look_at_camera(center: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    forward = target - center
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward], axis=0)
    translation = -rotation @ center
    return rotation, translation


def make_cameras(num_views: int) -> list[dict[str, Any]]:
    cameras = []
    target = np.array([0.0, 0.0, 1.0])
    for view in range(num_views):
        angle = 2.0 * math.pi * view / num_views
        center = np.array([4.0 * math.cos(angle), 4.0 * math.sin(angle), 1.7])
        r, t = look_at_camera(center, target)
        cameras.append(
            {
                "id": view,
                "K": [[900.0, 0.0, 640.0], [0.0, 900.0, 480.0], [0.0, 0.0, 1.0]],
                "R": r.tolist(),
                "t": t.tolist(),
            }
        )
    return cameras


def projection_matrix(camera: dict[str, Any]) -> np.ndarray:
    k = np.asarray(camera["K"], dtype=float)
    r = np.asarray(camera["R"], dtype=float)
    t = np.asarray(camera["t"], dtype=float).reshape(3, 1)
    return k @ np.concatenate([r, t], axis=1)


def project(pmat: np.ndarray, joints: np.ndarray) -> np.ndarray:
    homog = np.concatenate([joints, np.ones((len(joints), 1))], axis=1)
    img = (pmat @ homog.T).T
    return img[:, :2] / img[:, 2:3]


def make_pose(person: int, frame: int, total_frames: int) -> np.ndarray:
    phase = frame / max(1, total_frames - 1)
    side = -1.0 if person == 0 else 1.0
    x = side * (0.55 - 0.95 * phase)
    y = 0.28 * math.sin(2.2 * math.pi * phase + person * 0.6)
    z = 1.05 + 0.03 * math.sin(2.0 * math.pi * phase + person)
    pose = SKELETON.copy()
    pose += np.array([x, y, z])
    pose[2, 1] += 0.06 * math.sin(5.0 * phase + person)
    pose[3, 1] -= 0.06 * math.sin(5.0 * phase + person)
    return pose


def generate_dataset(
    frames: int = 48,
    views: int = 5,
    people: int = 2,
    noise_px: float = 7.0,
    miss_prob: float = 0.15,
    outlier_prob: float = 0.12,
    seed: int = 7,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    cameras = make_cameras(views)
    pmats = [projection_matrix(camera) for camera in cameras]
    all_frames = []
    for frame_idx in range(frames):
        gt = [{"person_id": p, "joints_3d": make_pose(p, frame_idx, frames).tolist()} for p in range(people)]
        detections_by_view: list[list[dict[str, Any]]] = []
        for view_idx, pmat in enumerate(pmats):
            detections = []
            for person in range(people):
                if rng.random() < miss_prob:
                    continue
                joints_2d = project(pmat, np.asarray(gt[person]["joints_3d"], dtype=float))
                # Ambiguous crossing frames deliberately get heavier noise in side views.
                crossing = abs(frame_idx - frames * 0.5) < frames * 0.18
                local_noise = noise_px * (1.8 if crossing and view_idx in (1, views - 1) else 1.0)
                noisy = joints_2d + rng.normal(0.0, local_noise, size=joints_2d.shape)
                keypoints = [[float(x), float(y), 1.0] for x, y in noisy]
                detections.append({"source_person": person, "keypoints": keypoints, "confidence": 1.0})
            if rng.random() < outlier_prob:
                source = int(rng.integers(people))
                joints_2d = project(pmat, np.asarray(gt[source]["joints_3d"], dtype=float))
                offset = rng.normal(0.0, 80.0, size=joints_2d.shape)
                keypoints = [[float(x), float(y), 0.6] for x, y in joints_2d + offset]
                detections.append({"source_person": None, "keypoints": keypoints, "confidence": 0.35})
            rng.shuffle(detections)
            detections_by_view.append(detections)
        all_frames.append({"frame_id": frame_idx, "ground_truth": gt, "detections": detections_by_view})
    return {
        "name": "synthetic_compose_crossing",
        "units": "meters",
        "joint_names": ["pelvis", "head", "left_hand", "right_hand", "left_foot", "right_foot"],
        "cameras": cameras,
        "frames": all_frames,
        "generator": {
            "frames": frames,
            "views": views,
            "people": people,
            "noise_px": noise_px,
            "miss_prob": miss_prob,
            "outlier_prob": outlier_prob,
            "seed": seed,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/synthetic_compose_dataset.json")
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--views", type=int, default=5)
    parser.add_argument("--people", type=int, default=2)
    parser.add_argument("--noise-px", type=float, default=7.0)
    parser.add_argument("--miss-prob", type=float, default=0.15)
    parser.add_argument("--outlier-prob", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = generate_dataset(
        frames=args.frames,
        views=args.views,
        people=args.people,
        noise_px=args.noise_px,
        miss_prob=args.miss_prob,
        outlier_prob=args.outlier_prob,
        seed=args.seed,
    )
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataset, indent=2))


if __name__ == "__main__":
    main()

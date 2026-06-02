#!/usr/bin/env python3
"""Build a lightweight COMPOSE benchmark from the CMU Panoptic sample.

The official ``171204_pose1_sample`` release contains calibrated HD cameras
and one tracked 3D body.  To keep the correspondence problem multi-person
without downloading a large Panoptic sequence, this converter translates copies
of the real skeleton through a crossing trajectory, projects them into the
real HD cameras, and adds deterministic detector noise, misses, and outliers.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import tarfile
from typing import Any


COCO19_NAMES = [
    "neck",
    "nose",
    "pelvis",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_eye",
    "left_ear",
    "right_eye",
    "right_ear",
]


Vec3 = list[float]
Mat = list[list[float]]


def flatten(values: Any) -> list[float]:
    if isinstance(values, (list, tuple)):
        out: list[float] = []
        for value in values:
            out.extend(flatten(value))
        return out
    return [float(values)]


def matmul(left: Mat, right: Mat) -> Mat:
    return [
        [sum(left[row][inner] * right[inner][col] for inner in range(len(right))) for col in range(len(right[0]))]
        for row in range(len(left))
    ]


def mean_vec(points: list[Vec3]) -> Vec3:
    return [sum(point[axis] for point in points) / len(points) for axis in range(3)]


def normalize_camera(raw: dict[str, Any]) -> dict[str, Any]:
    scale = 0.01
    return {
        "camera_id": f"hd_{int(raw['node']):02d}",
        "model": "pinhole",
        "image_size": {"width": int(raw["resolution"][0]), "height": int(raw["resolution"][1])},
        "intrinsics": raw["K"],
        "extrinsics": {
            "rotation_world_to_camera": raw["R"],
            "translation_world_to_camera": [value * scale for value in flatten(raw["t"])[:3]],
        },
    }


def projection_matrix(camera: dict[str, Any]) -> Mat:
    k = [[float(value) for value in row] for row in camera["intrinsics"]]
    r = [[float(value) for value in row] for row in camera["extrinsics"]["rotation_world_to_camera"]]
    t = [float(value) for value in camera["extrinsics"]["translation_world_to_camera"]]
    rt = [r[row] + [t[row]] for row in range(3)]
    return matmul(k, rt)


def project(pmat: Mat, joints: list[Vec3]) -> list[list[float]]:
    out = []
    for joint in joints:
        homog = joint + [1.0]
        uvw = [sum(pmat[row][col] * homog[col] for col in range(4)) for row in range(3)]
        out.append([uvw[0] / uvw[2], uvw[1] / uvw[2]])
    return out


def parse_body_joints(body: dict[str, Any]) -> list[Vec3] | None:
    flat = [float(value) for value in body["joints19"]]
    joints = [flat[idx : idx + 4] for idx in range(0, len(flat), 4)]
    valid = [joint for joint in joints if joint[3] > 0.05]
    if len(valid) < 8:
        return None
    fill = mean_vec([[joint[0] * 0.01, joint[1] * 0.01, joint[2] * 0.01] for joint in valid])
    out = []
    for joint in joints:
        if joint[3] > 0.05:
            out.append([joint[0] * 0.01, joint[1] * 0.01, joint[2] * 0.01])
        else:
            out.append(fill[:])
    return out


def load_sample_bodies(pose_tar: Path, limit: int) -> list[list[Vec3]]:
    bodies: list[list[Vec3]] = []
    with tarfile.open(pose_tar) as archive:
        members = sorted((m for m in archive.getmembers() if m.isfile() and m.name.endswith(".json")), key=lambda m: m.name)
        for member in members:
            frame = json.load(archive.extractfile(member))
            if not frame.get("bodies"):
                continue
            joints = parse_body_joints(frame["bodies"][0])
            if joints is not None:
                bodies.append(joints)
            if len(bodies) >= limit:
                break
    if not bodies:
        raise RuntimeError(f"No usable bodies found in {pose_tar}")
    return bodies


def make_people(base_pose: list[Vec3], frame_idx: int, total_frames: int, people: int) -> list[dict[str, Any]]:
    phase = frame_idx / max(1, total_frames - 1)
    center = mean_vec([base_pose[idx] for idx in (2, 6, 12)])
    centered = [[joint[axis] - center[axis] for axis in range(3)] for joint in base_pose]
    out = []
    for person in range(people):
        side = -1.0 if person % 2 == 0 else 1.0
        lane = (person // 2) * 0.45
        x = side * (0.85 - 1.5 * phase)
        z = 2.3 + lane + 0.08 * math.sin(2.0 * math.pi * phase + person)
        y = 0.18 * math.sin(2.5 * math.pi * phase + person * 0.7)
        joints = [[joint[0] + x, joint[1] + y, joint[2] + z] for joint in centered]
        out.append(
            {
                "person_id": f"person_{person:02d}",
                "joints_3d": {name: joints[idx] for idx, name in enumerate(COCO19_NAMES)},
            }
        )
    return out


def visible_in_image(points: list[list[float]], camera: dict[str, Any], margin: float = 200.0) -> bool:
    width = camera["image_size"]["width"]
    height = camera["image_size"]["height"]
    visible = [
        -margin <= point[0] <= width + margin and -margin <= point[1] <= height + margin
        for point in points
    ]
    return sum(1 for item in visible if item) / max(1, len(visible)) > 0.75


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(args.seed)
    calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    # The paper uses a fixed CMU Panoptic five-HD-camera protocol on its main
    # sequences. The tiny sample sequence has different useful coverage, so we
    # choose five calibrated HD cameras that observe this augmented scene.
    hd_nodes = {6, 8, 10, 11, 16}
    cameras = [
        normalize_camera(camera)
        for camera in calibration["cameras"]
        if camera.get("type") == "hd" and int(camera.get("node", -1)) in hd_nodes
    ]
    cameras.sort(key=lambda cam: cam["camera_id"])
    if len(cameras) < 3:
        raise RuntimeError("Expected at least three HD cameras in the Panoptic sample calibration")
    pmats = [projection_matrix(camera) for camera in cameras]
    base_poses = load_sample_bodies(Path(args.pose_tar), args.frames)

    scenes = []
    for frame_idx, base_pose in enumerate(base_poses):
        people = make_people(base_pose, frame_idx, len(base_poses), args.people)
        views = []
        for view_idx, (camera, pmat) in enumerate(zip(cameras, pmats)):
            detections = []
            for person in people:
                if rng.random() < args.miss_prob:
                    continue
                joints = [person["joints_3d"][name] for name in COCO19_NAMES]
                projected = project(pmat, joints)
                if not visible_in_image(projected, camera):
                    continue
                crossing = abs(frame_idx - len(base_poses) * 0.5) < len(base_poses) * 0.2
                local_noise = args.noise_px * (1.7 if crossing and view_idx in (1, len(cameras) - 2) else 1.0)
                noisy = [[x + rng.gauss(0.0, local_noise), y + rng.gauss(0.0, local_noise)] for x, y in projected]
                detections.append(
                    {
                        "person_id": person["person_id"],
                        "confidence": 1.0,
                        "joints": [
                            {"joint": name, "xy": [float(x), float(y)], "confidence": 1.0}
                            for name, (x, y) in zip(COCO19_NAMES, noisy)
                        ],
                    }
                )
            if rng.random() < args.outlier_prob and people:
                source = people[rng.randrange(len(people))]
                joints = [source["joints_3d"][name] for name in COCO19_NAMES]
                projected = project(pmat, joints)
                noisy = [[x + rng.gauss(0.0, args.outlier_px), y + rng.gauss(0.0, args.outlier_px)] for x, y in projected]
                detections.append(
                    {
                        "person_id": None,
                        "confidence": 0.35,
                        "joints": [
                            {"joint": name, "xy": [float(x), float(y)], "confidence": 0.6}
                            for name, (x, y) in zip(COCO19_NAMES, noisy)
                        ],
                    }
                )
            rng.shuffle(detections)
            views.append({"camera_id": camera["camera_id"], "detections": detections})
        scenes.append({"scene_id": f"panoptic_sample_{frame_idx:04d}", "people": people, "views": views})

    return {
        "schema_version": "panoptic_sample_compose_v1",
        "source": {
            "dataset": "CMU Panoptic 171204_pose1_sample",
            "calibration": str(args.calibration),
            "pose_tar": str(args.pose_tar),
            "augmentation": "real skeleton frames translated into deterministic multi-person crossing trajectories",
        },
        "task": {"joint_names": COCO19_NAMES, "units": "meters"},
        "cameras": cameras,
        "scenes": scenes,
        "generator": {
            "frames": len(scenes),
            "people": args.people,
            "noise_px": args.noise_px,
            "miss_prob": args.miss_prob,
            "outlier_prob": args.outlier_prob,
            "outlier_px": args.outlier_px,
            "seed": args.seed,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", default="data/downloads/cmu_panoptic_sample/calibration_171204_pose1_sample.json")
    parser.add_argument("--pose-tar", default="data/downloads/cmu_panoptic_sample/hdPose3d_stage1_coco19.tar")
    parser.add_argument("--output", default="data/downloads/cmu_panoptic_sample_compose.json")
    parser.add_argument("--frames", type=int, default=72)
    parser.add_argument("--people", type=int, default=2)
    parser.add_argument("--noise-px", type=float, default=6.0)
    parser.add_argument("--miss-prob", type=float, default=0.13)
    parser.add_argument("--outlier-prob", type=float, default=0.14)
    parser.add_argument("--outlier-px", type=float, default=95.0)
    parser.add_argument("--seed", type=int, default=19)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = build_dataset(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    print(f"Wrote {output} with {len(dataset['scenes'])} scenes")


if __name__ == "__main__":
    main()

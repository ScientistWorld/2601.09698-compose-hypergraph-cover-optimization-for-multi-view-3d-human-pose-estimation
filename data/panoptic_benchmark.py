#!/usr/bin/env python3
"""Build the public (train) multi-view multi-person pose benchmark.

Source: real multi-person 3D body annotations and real HD-camera calibrations
from CMU Panoptic sequences of the standard multi-view pose *training* split,
seen from the standard five-camera setup (HD nodes 3, 6, 12, 13, 23).

The released Panoptic videos are tens of GB per sequence and the paper's 2D
keypoint detector is not part of this benchmark, so each view's 2D pose
detections come from a stochastic detector model applied to the real 3D
annotations and real cameras.  The model reproduces the failure modes that
make cross-view association hard: size-dependent heteroscedastic joint noise,
heavy-tailed gross joint errors, inter-person occlusion (raising noise,
lowering confidence, snapping joints onto the occluding person, and dropping
heavily occluded people), image truncation, whole-body left/right flips,
duplicate partial detections and person-shaped false positives.  Every camera
of every sequence draws its own detector profile, and confidences are
informative but noisy.  Detections carry no identity information.

Pure standard-library Python so it can run on a login node.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import tarfile
from typing import Any

JOINT_NAMES = [
    "neck", "nose", "mid_hip",
    "l_shoulder", "l_elbow", "l_wrist", "l_hip", "l_knee", "l_ankle",
    "r_shoulder", "r_elbow", "r_wrist", "r_hip", "r_knee", "r_ankle",
]
NUM_JOINTS = len(JOINT_NAMES)
ROOT = 2
LR_SWAP = [0, 1, 2, 9, 10, 11, 12, 13, 14, 3, 4, 5, 6, 7, 8]
# Relative 2D localisation difficulty per joint (extremities are harder).
JOINT_SIGMA = [0.8, 0.7, 1.3, 0.9, 1.2, 1.6, 1.3, 1.2, 1.4, 0.9, 1.2, 1.6, 1.3, 1.2, 1.4]
# Body capsules (joint a, joint b, radius in metres) used for occlusion tests.
LIMBS = [
    (0, 1, 0.11), (0, 2, 0.16), (3, 9, 0.10), (6, 12, 0.10), (3, 6, 0.09), (9, 12, 0.09),
    (3, 4, 0.06), (4, 5, 0.05), (9, 10, 0.06), (10, 11, 0.05),
    (6, 7, 0.08), (7, 8, 0.06), (12, 13, 0.08), (13, 14, 0.06),
]
CAMERA_NODES = (3, 6, 12, 13, 23)
IMAGE_W, IMAGE_H = 1920, 1080
TRUNCATION_MIN_INSIDE = 0.6   # fraction of joints that must project inside the image
OCCLUSION_SIGMA = 2.5         # noise multiplier for joints hidden behind another person
OUTSIDE_SIGMA = 3.0           # noise multiplier for joints outside the image
# Sequence -> number of evenly spaced frames (with >= 2 annotated people).
# The crowded ultimatum sequence gets more frames so the people-count mix is
# comparable to the hidden test split.
TRAIN_SEQUENCES = {
    "160422_ultimatum1": 100,
    "160224_haggling1": 50,
    "160226_haggling1": 50,
    "160906_band1": 50,
    "160906_ian1": 50,
}
PROFILE_RANGES = {
    "sigma_rel": (0.006, 0.012),   # joint noise std as a fraction of person pixel height
    "miss": (0.01, 0.05),          # base probability of missing a visible person
    "tail": (0.01, 0.04),          # probability of a gross (6x) joint error
    "flip": (0.005, 0.03),         # whole-body left/right label flip
    "swap": (0.2, 0.5),            # occluded joint snaps to the occluder's joint
    "fp_rate": (0.05, 0.3),        # expected false-positive detections per view
}


# ----------------------------------------------------------------------------
# Panoptic parsing
# ----------------------------------------------------------------------------

def load_cameras(calibration_path: Path, nodes: tuple[int, ...] = CAMERA_NODES) -> list[dict[str, Any]]:
    calib = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
    by_node = {int(cam["node"]): cam for cam in calib["cameras"] if cam.get("type") == "hd"}
    cameras = []
    for node in nodes:
        cam = by_node[node]
        t = [float(row[0]) / 100.0 for row in cam["t"]]
        cameras.append(
            {
                "camera_id": f"hd_00_{node:02d}",
                "image_size": [int(cam["resolution"][0]), int(cam["resolution"][1])],
                "K": [[float(v) for v in row] for row in cam["K"]],
                "R": [[float(v) for v in row] for row in cam["R"]],
                "t": t,
            }
        )
    return cameras


def parse_body(body: dict[str, Any]) -> dict[str, Any] | None:
    flat = [float(v) for v in body["joints19"]]
    joints = []
    valid = []
    for j in range(NUM_JOINTS):
        x, y, z, c = flat[4 * j : 4 * j + 4]
        joints.append([x / 100.0, y / 100.0, z / 100.0])
        valid.append(c > 0.1)
    if not valid[ROOT] or sum(valid) < 8:
        return None
    return {"joints_3d": joints, "joint_valid": valid}


def load_sequence_frames(tar_path: Path) -> list[tuple[str, list[dict[str, Any]]]]:
    frames = []
    with tarfile.open(tar_path) as archive:
        members = sorted(
            (m for m in archive.getmembers() if m.isfile() and m.name.endswith(".json")),
            key=lambda m: m.name,
        )
        for member in members:
            try:
                payload = json.load(archive.extractfile(member))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue  # a few released frame files are empty
            bodies = [b for b in (parse_body(body) for body in payload.get("bodies", [])) if b is not None]
            if len(bodies) >= 2:
                frames.append((Path(member.name).stem, bodies))
    return frames


def evenly_spaced(items: list[Any], count: int) -> list[Any]:
    if len(items) <= count:
        return list(items)
    step = len(items) / count
    return [items[int(step * k + step / 2)] for k in range(count)]


# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------

def rotate_y(point: list[float], angle: float) -> list[float]:
    c, s = math.cos(angle), math.sin(angle)
    x, y, z = point
    return [c * x + s * z, y, -s * x + c * z]


def to_camera(cam: dict[str, Any], point: list[float]) -> list[float]:
    r, t = cam["R"], cam["t"]
    return [r[i][0] * point[0] + r[i][1] * point[1] + r[i][2] * point[2] + t[i] for i in range(3)]


def project(cam: dict[str, Any], point: list[float]) -> tuple[float, float, float]:
    xc = to_camera(cam, point)
    k = cam["K"]
    depth = xc[2]
    if depth <= 1e-6:
        return float("nan"), float("nan"), depth
    u = (k[0][0] * xc[0] + k[0][1] * xc[1] + k[0][2] * xc[2]) / depth
    v = (k[1][1] * xc[1] + k[1][2] * xc[2]) / depth
    return u, v, depth


def point_segment_distance(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    if denom < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def poisson(rng: random.Random, rate: float) -> int:
    threshold = math.exp(-rate)
    k, p = 0, 1.0
    while True:
        p *= rng.random()
        if p <= threshold:
            return k
        k += 1


# ----------------------------------------------------------------------------
# Detector model
# ----------------------------------------------------------------------------

def draw_profile(rng: random.Random) -> dict[str, float]:
    return {name: rng.uniform(low, high) for name, (low, high) in PROFILE_RANGES.items()}


def simulate_view(
    people: list[dict[str, Any]],
    cam: dict[str, Any],
    profile: dict[str, float],
    rng: random.Random,
    debug: bool = False,
) -> list[dict[str, Any]]:
    """Return noisy 2D pose detections of ``people`` in camera ``cam``."""

    width, height = cam["image_size"]
    focal = cam["K"][0][0]
    projected = []
    for person in people:
        uvd = [project(cam, joint) for joint in person["joints_3d"]]
        projected.append(uvd)

    detections: list[dict[str, Any]] = []
    detected_info: list[tuple[int, list[tuple[float, float]], float]] = []
    for pid, person in enumerate(people):
        uvd = projected[pid]
        if any(d < 0.3 for _, _, d in uvd):
            continue
        inside = [0.0 <= u < width and 0.0 <= v < height for u, v, _ in uvd]
        if sum(inside) / NUM_JOINTS < TRUNCATION_MIN_INSIDE:
            continue
        vs = [v for _, v, _ in uvd]
        us = [u for u, _, _ in uvd]
        pix_height = max(max(vs) - min(vs), 0.5 * (max(us) - min(us)), 60.0)

        occluder: list[int | None] = [None] * NUM_JOINTS
        occluder_depth = [float("inf")] * NUM_JOINTS
        for qid, other in enumerate(projected):
            if qid == pid:
                continue
            for a, b, radius in LIMBS:
                ua, va, da = other[a]
                ub, vb, db = other[b]
                if da < 0.3 or db < 0.3:
                    continue
                seg_depth = 0.5 * (da + db)
                r_px = focal * radius / seg_depth
                for j in range(NUM_JOINTS):
                    u, v, d = uvd[j]
                    if seg_depth >= d - 0.1 or seg_depth >= occluder_depth[j]:
                        continue
                    if point_segment_distance(u, v, ua, va, ub, vb) <= r_px:
                        occluder[j] = qid
                        occluder_depth[j] = seg_depth
        occluded_frac = sum(1 for o in occluder if o is not None) / NUM_JOINTS
        p_miss = profile["miss"] + max(0.0, occluded_frac - 0.5) * 2.0
        if rng.random() < p_miss:
            continue

        sigma_base = profile["sigma_rel"] * pix_height
        keypoints = []
        for j in range(NUM_JOINTS):
            u, v, _ = uvd[j]
            sigma = sigma_base * JOINT_SIGMA[j]
            conf_mean, conf_sd = 0.85, 0.08
            if not person["joint_valid"][j]:
                sigma *= 2.0
                conf_mean, conf_sd = 0.5, 0.15
            occ = occluder[j]
            if occ is not None:
                ou, ov, _ = projected[occ][j]
                if math.hypot(ou - u, ov - v) < 0.25 * pix_height and rng.random() < profile["swap"]:
                    u, v = ou, ov
                    conf_mean, conf_sd = 0.6, 0.15
                else:
                    sigma *= OCCLUSION_SIGMA
                    conf_mean, conf_sd = 0.45, 0.15
            if rng.random() < profile["tail"]:
                sigma *= 6.0
                conf_mean, conf_sd = 0.55, 0.2
            if not inside[j]:
                # Top-down crops extend past the image border: the joint is
                # extrapolated with a large error and a low confidence.
                sigma *= OUTSIDE_SIGMA
                conf_mean, conf_sd = 0.2, 0.1
            u += rng.gauss(0.0, sigma)
            v += rng.gauss(0.0, sigma)
            conf = clip(rng.gauss(conf_mean, conf_sd), 0.01, 0.99)
            keypoints.append([u, v, conf])
        if rng.random() < profile["flip"]:
            keypoints = [keypoints[LR_SWAP[j]] for j in range(NUM_JOINTS)]
        mean_conf = sum(kp[2] for kp in keypoints) / NUM_JOINTS
        score = clip(mean_conf * (1.0 - 0.3 * occluded_frac) + rng.gauss(0.0, 0.05), 0.05, 0.99)
        det = {"keypoints": keypoints, "score": score}
        if debug:
            det["_source"] = pid
        detections.append(det)
        detected_info.append((pid, [(kp[0], kp[1]) for kp in keypoints], pix_height))

    # False positives: duplicate partial detections and person-shaped ghosts.
    for _ in range(poisson(rng, profile["fp_rate"])):
        if detected_info and rng.random() < 0.5:
            _, uv, pix_height = detected_info[rng.randrange(len(detected_info))]
            shift_u = rng.gauss(0.0, 0.08 * pix_height)
            shift_v = rng.gauss(0.0, 0.08 * pix_height)
            upper = rng.random() < 0.5
            keypoints = []
            for j in range(NUM_JOINTS):
                u, v = uv[j]
                lower_joint = j in (2, 6, 7, 8, 12, 13, 14)
                keep = (not lower_joint) if upper else lower_joint or j in (0, 3, 9)
                sigma = 3.0 * profile["sigma_rel"] * pix_height
                conf_mean = 0.45
                if not keep:
                    sigma = 0.35 * pix_height
                    conf_mean = 0.15
                u = clip(u + shift_u + rng.gauss(0.0, sigma), 0.0, width - 1.0)
                v = clip(v + shift_v + rng.gauss(0.0, sigma), 0.0, height - 1.0)
                keypoints.append([u, v, clip(rng.gauss(conf_mean, 0.15), 0.01, 0.99)])
            score = clip(rng.gauss(0.4, 0.12), 0.05, 0.99)
        else:
            source = projected[rng.randrange(len(projected))] if projected else None
            if source is None or any(not math.isfinite(u) for u, _, _ in source):
                continue
            us = [u for u, _, _ in source]
            vs = [v for _, v, _ in source]
            cu, cv = sum(us) / NUM_JOINTS, sum(vs) / NUM_JOINTS
            scale = rng.uniform(0.6, 1.3)
            tu = rng.uniform(0.1 * width, 0.9 * width)
            tv = rng.uniform(0.25 * height, 0.75 * height)
            pix_height = max(max(vs) - min(vs), 60.0) * scale
            keypoints = []
            for u, v, _ in source:
                gu = tu + scale * (u - cu) + rng.gauss(0.0, 2.0 * profile["sigma_rel"] * pix_height)
                gv = tv + scale * (v - cv) + rng.gauss(0.0, 2.0 * profile["sigma_rel"] * pix_height)
                keypoints.append(
                    [clip(gu, 0.0, width - 1.0), clip(gv, 0.0, height - 1.0), clip(rng.gauss(0.5, 0.15), 0.01, 0.99)]
                )
            score = clip(rng.gauss(0.45, 0.12), 0.05, 0.99)
        det = {"keypoints": keypoints, "score": score}
        if debug:
            det["_source"] = None
        detections.append(det)

    rng.shuffle(detections)
    return detections


def build_frame(
    frame_id: str,
    bodies: list[dict[str, Any]],
    cameras: list[dict[str, Any]],
    profiles: list[dict[str, float]],
    rng: random.Random,
    debug: bool = False,
) -> dict[str, Any]:
    angle = rng.uniform(0.0, 2.0 * math.pi)
    people = [
        {
            "joints_3d": [rotate_y(joint, angle) for joint in body["joints_3d"]],
            "joint_valid": list(body["joint_valid"]),
        }
        for body in bodies
    ]
    views = []
    for cam, profile in zip(cameras, profiles):
        views.append({"camera_id": cam["camera_id"], "detections": simulate_view(people, cam, profile, rng, debug)})
    return {"frame_id": frame_id, "cameras": cameras, "views": views, "people": people}


def round_frame(frame: dict[str, Any]) -> dict[str, Any]:
    for view in frame["views"]:
        for det in view["detections"]:
            det["keypoints"] = [[round(u, 3), round(v, 3), round(c, 4)] for u, v, c in det["keypoints"]]
            det["score"] = round(det["score"], 4)
    for person in frame["people"]:
        person["joints_3d"] = [[round(x, 5) for x in joint] for joint in person["joints_3d"]]
    return frame


def build_benchmark(root: Path, sequences: dict[str, int], seed: int) -> dict[str, Any]:
    frames = []
    for seq_idx, (seq, count) in enumerate(sequences.items()):
        seq_dir = root / seq
        cameras = load_cameras(seq_dir / f"calibration_{seq}.json")
        candidates = load_sequence_frames(seq_dir / "hdPose3d_stage1_coco19.tar")
        chosen = evenly_spaced(candidates, count)
        profile_rng = random.Random(f"{seed}:profiles:{seq}")
        profiles = [draw_profile(profile_rng) for _ in cameras]
        rng = random.Random(f"{seed}:frames:{seq}")
        for k, (_, bodies) in enumerate(chosen):
            frame = build_frame(f"train_{seq_idx}_{k:03d}", bodies, cameras, profiles, rng)
            frame["source_sequence"] = seq
            frames.append(round_frame(frame))
    return {
        "format": "compose_gym_benchmark_v2",
        "split": "train",
        "units": "meters",
        "joint_names": JOINT_NAMES,
        "frames": frames,
        "generator": {
            "source": "CMU Panoptic hdPose3d_stage1_coco19 annotations + HD calibrations",
            "sequences": dict(sequences),
            "camera_nodes": list(CAMERA_NODES),
            "seed": seed,
            "augmentation": "people rotated about the vertical axis by a random angle per frame",
            "profile_ranges": PROFILE_RANGES,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="data/downloads/panoptic", help="Directory holding <sequence>/ downloads")
    parser.add_argument("--output", default="data/downloads/panoptic_train.json")
    parser.add_argument("--seed", type=int, default=20260611)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark = build_benchmark(Path(args.root), TRAIN_SEQUENCES, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(benchmark, separators=(",", ":")), encoding="utf-8")
    people = sum(len(f["people"]) for f in benchmark["frames"])
    dets = sum(len(v["detections"]) for f in benchmark["frames"] for v in f["views"])
    print(f"Wrote {output}: {len(benchmark['frames'])} frames, {people} people, {dets} detections")


if __name__ == "__main__":
    main()

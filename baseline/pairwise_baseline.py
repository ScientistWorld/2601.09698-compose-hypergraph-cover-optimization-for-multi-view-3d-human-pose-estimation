#!/usr/bin/env python3
"""Simple pairwise multi-view triangulation baseline.

This baseline reads the same data JSON used by evaluation. It ignores
``ground_truth`` and predicts from optional multi-view detections and cameras.
Accepted frame layouts are intentionally broad:

  frame["views"] = [{"view_id": ..., "camera": {...}, "detections": [...]}, ...]
  frame["detections"] = [{"view_id": ..., "joints_2d": ...}, ...]

Cameras can provide a 3x4 projection matrix as ``P``/``projection_matrix`` or
``K`` with ``R`` and ``t``. If no usable detections/cameras are present, the
baseline emits an empty prediction list for that frame.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any


def iter_frames(payload: Any) -> list[tuple[str, dict[str, Any]]]:
    frames = payload.get("frames", payload) if isinstance(payload, dict) else payload
    if isinstance(frames, dict):
        return [(str(frame_id), frame) for frame_id, frame in frames.items()]
    if isinstance(frames, list):
        return [(str(frame.get("frame_id", frame.get("id", idx))), frame) for idx, frame in enumerate(frames)]
    raise TypeError("Expected data JSON with frames")


def flatten(values: Any) -> list[Any]:
    if isinstance(values, (list, tuple)):
        out = []
        for value in values:
            out.extend(flatten(value))
        return out
    return [values]


def matrix_nxm(values: Any, rows: int, cols: int) -> list[list[float]] | None:
    if not isinstance(values, (list, tuple)) or len(values) != rows:
        return None
    matrix = []
    for row in values:
        if not isinstance(row, (list, tuple)) or len(row) != cols:
            return None
        matrix.append([float(row[col]) for col in range(cols)])
    return matrix


def matmul(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    return [
        [sum(left[row][inner] * right[inner][col] for inner in range(len(right))) for col in range(len(right[0]))]
        for row in range(len(left))
    ]


def joints_array(joints: Any, dims: int) -> list[list[float]]:
    if isinstance(joints, dict):
        try:
            keys = sorted(joints, key=lambda item: int(item))
        except (TypeError, ValueError):
            keys = sorted(joints)
        joints = [joints[key] for key in keys]
    out = []
    for joint in joints:
        if not isinstance(joint, (list, tuple)) or len(joint) < dims:
            raise ValueError(f"Expected each {dims}D joint to contain at least {dims} coordinates")
        out.append([float(joint[idx]) for idx in range(dims)])
    return out


def projection_matrix(camera: dict[str, Any] | None) -> list[list[float]] | None:
    if not camera:
        return None
    for key in ("P", "projection_matrix", "camera_matrix"):
        if key in camera:
            mat = matrix_nxm(camera[key], 3, 4)
            if mat is not None:
                return mat
    if all(key in camera for key in ("K", "R", "t")):
        k = matrix_nxm(camera["K"], 3, 3)
        r = matrix_nxm(camera["R"], 3, 3)
        t = [float(value) for value in flatten(camera["t"])]
        if k is not None and r is not None and len(t) >= 3:
            rt = [r[row] + [t[row]] for row in range(3)]
            return matmul(k, rt)
    return None


def collect_view_detections(frame: dict[str, Any]) -> list[dict[str, Any]]:
    cameras = {}
    for view in frame.get("views", []) or []:
        if not isinstance(view, dict):
            continue
        view_id = str(view.get("view_id", view.get("id", len(cameras))))
        cameras[view_id] = projection_matrix(view.get("camera", view))

    detections = []
    for view in frame.get("views", []) or []:
        if not isinstance(view, dict):
            continue
        view_id = str(view.get("view_id", view.get("id", len(detections))))
        for det_idx, det in enumerate(view.get("detections", view.get("poses_2d", [])) or []):
            detections.append(
                {
                    "view_id": view_id,
                    "det_id": str(det.get("detection_id", det.get("id", det_idx))),
                    "joints_2d": joints_array(det["joints_2d"], 2),
                    "score": float(det.get("score", det.get("confidence", 1.0))),
                    "P": cameras.get(view_id),
                }
            )

    frame_cameras = frame.get("cameras") if isinstance(frame.get("cameras"), dict) else {}
    for det_idx, det in enumerate(frame.get("detections", frame.get("poses_2d", [])) or []):
        view_id = str(det.get("view_id", det.get("camera_id", det_idx)))
        proj = projection_matrix(det.get("camera") or frame_cameras.get(view_id))
        if proj is None:
            proj = cameras.get(view_id)
        detections.append(
            {
                "view_id": view_id,
                "det_id": str(det.get("detection_id", det.get("id", det_idx))),
                "joints_2d": joints_array(det["joints_2d"], 2),
                "score": float(det.get("score", det.get("confidence", 1.0))),
                "P": proj,
            }
        )
    return [det for det in detections if det["P"] is not None]


def solve_3x3(matrix: list[list[float]], vector: list[float]) -> list[float]:
    aug = [matrix[row][:] + [vector[row]] for row in range(3)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda row: abs(aug[row][col]))
        if abs(aug[pivot][col]) < 1e-12:
            return [math.nan, math.nan, math.nan]
        aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        aug[col] = [value / scale for value in aug[col]]
        for row in range(3):
            if row == col:
                continue
            factor = aug[row][col]
            aug[row] = [value - factor * aug[col][idx] for idx, value in enumerate(aug[row])]
    return [aug[row][3] for row in range(3)]


def solve_least_squares_3d(a_rows: list[list[float]], b_values: list[float]) -> list[float]:
    ata = [[0.0, 0.0, 0.0] for _ in range(3)]
    atb = [0.0, 0.0, 0.0]
    for row, b_value in zip(a_rows, b_values):
        for i in range(3):
            atb[i] += row[i] * b_value
            for j in range(3):
                ata[i][j] += row[i] * row[j]
    return solve_3x3(ata, atb)


def triangulate_point(observations: list[tuple[list[float], list[list[float]]]]) -> list[float]:
    a_rows = []
    b_values = []
    for point, proj in observations:
        if not all(math.isfinite(value) for value in point):
            continue
        x, y = point[:2]
        a_rows.append([x * proj[2][col] - proj[0][col] for col in range(3)])
        b_values.append(proj[0][3] - x * proj[2][3])
        a_rows.append([y * proj[2][col] - proj[1][col] for col in range(3)])
        b_values.append(proj[1][3] - y * proj[2][3])
    if len(a_rows) < 4:
        return [math.nan, math.nan, math.nan]
    return solve_least_squares_3d(a_rows, b_values)


def project_point(point_3d: list[float], proj: list[list[float]]) -> list[float]:
    homogeneous = point_3d + [1.0]
    uvw = [sum(proj[row][col] * homogeneous[col] for col in range(4)) for row in range(3)]
    if abs(uvw[2]) < 1e-12:
        return [math.nan, math.nan]
    return [uvw[0] / uvw[2], uvw[1] / uvw[2]]


def triangulate_pose(detections: list[dict[str, Any]]) -> list[list[float]]:
    joint_count = min(len(det["joints_2d"]) for det in detections)
    joints_3d = []
    for joint_idx in range(joint_count):
        obs = [(det["joints_2d"][joint_idx], det["P"]) for det in detections]
        joints_3d.append(triangulate_point(obs))
    return joints_3d


def pair_reprojection_error(left: dict[str, Any], right: dict[str, Any]) -> float:
    pose_3d = triangulate_pose([left, right])
    errors = []
    for det in (left, right):
        joint_count = min(len(pose_3d), len(det["joints_2d"]))
        for joint_idx in range(joint_count):
            point_3d = pose_3d[joint_idx]
            if not all(math.isfinite(value) for value in point_3d):
                continue
            projected = project_point(point_3d, det["P"])
            observed = det["joints_2d"][joint_idx]
            if all(math.isfinite(value) for value in projected + observed):
                errors.append(math.hypot(projected[0] - observed[0], projected[1] - observed[1]))
    return (sum(errors) / len(errors)) if errors else math.inf


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def build_predictions_for_frame(
    frame: dict[str, Any],
    reprojection_threshold: float,
    min_views: int,
) -> list[dict[str, Any]]:
    detections = collect_view_detections(frame)
    if not detections:
        return []

    uf = UnionFind(len(detections))
    pair_edges = []
    for left_idx, right_idx in itertools.combinations(range(len(detections)), 2):
        if detections[left_idx]["view_id"] == detections[right_idx]["view_id"]:
            continue
        error = pair_reprojection_error(detections[left_idx], detections[right_idx])
        if error <= reprojection_threshold:
            pair_edges.append((error, left_idx, right_idx))
    pair_edges.sort(key=lambda item: item[0])

    for _, left_idx, right_idx in pair_edges:
        uf.union(left_idx, right_idx)

    clusters: dict[int, list[dict[str, Any]]] = {}
    for idx, det in enumerate(detections):
        clusters.setdefault(uf.find(idx), []).append(det)

    predictions = []
    for cluster_idx, cluster in enumerate(clusters.values()):
        by_view = {}
        for det in sorted(cluster, key=lambda item: item["score"], reverse=True):
            by_view.setdefault(det["view_id"], det)
        selected = list(by_view.values())
        if len(selected) < min_views:
            continue
        pose_3d = triangulate_pose(selected)
        if not pose_3d or not all(math.isfinite(coord) for joint in pose_3d for coord in joint):
            continue
        score = sum(det["score"] for det in selected) / len(selected)
        predictions.append(
            {
                "pose_id": f"baseline_{cluster_idx}",
                "score": score,
                "joints_3d": pose_3d,
            }
        )
    return predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input data JSON")
    parser.add_argument("--output", required=True, help="Output predictions JSON")
    parser.add_argument(
        "--reprojection-threshold",
        type=float,
        default=25.0,
        help="Maximum mean two-view reprojection error in pixels for association",
    )
    parser.add_argument("--min-views", type=int, default=2, help="Minimum associated camera views per 3D pose")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with Path(args.data).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    frames = {}
    for frame_id, frame in iter_frames(payload):
        frames[frame_id] = {
            "predicted_poses": build_predictions_for_frame(
                frame,
                reprojection_threshold=args.reprojection_threshold,
                min_views=args.min_views,
            )
        }

    output = {"frames": frames}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

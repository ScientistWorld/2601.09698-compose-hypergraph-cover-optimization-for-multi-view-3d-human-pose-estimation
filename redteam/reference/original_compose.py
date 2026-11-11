#!/usr/bin/env python3
"""COMPOSE hypergraph exact-cover solver for multi-view 3D pose recovery."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pulp


@dataclass(frozen=True)
class DetectionRef:
    view: int
    index: int


@dataclass
class Hyperedge:
    refs: tuple[DetectionRef, ...]
    cost: float
    score: float
    joints_3d: list[list[float]] | None


def projection_matrix(camera: dict[str, Any]) -> np.ndarray:
    if "K" in camera:
        k = np.asarray(camera["K"], dtype=float)
        r = np.asarray(camera["R"], dtype=float)
        t = np.asarray(camera["t"], dtype=float).reshape(3, 1)
    else:
        k = np.asarray(camera["intrinsics"], dtype=float)
        r = np.asarray(camera["extrinsics"]["rotation_world_to_camera"], dtype=float)
        t = np.asarray(camera["extrinsics"]["translation_world_to_camera"], dtype=float).reshape(3, 1)
    return k @ np.concatenate([r, t], axis=1)


def project_point(pmat: np.ndarray, point: np.ndarray) -> np.ndarray:
    homog = pmat @ np.array([point[0], point[1], point[2], 1.0], dtype=float)
    if abs(homog[2]) < 1e-12:
        return np.array([np.nan, np.nan])
    return homog[:2] / homog[2]


def triangulate_point(observations: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray | None:
    if len(observations) < 2:
        return None
    rows = []
    for pmat, xy in observations:
        x, y = xy
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        rows.append(x * pmat[2] - pmat[0])
        rows.append(y * pmat[2] - pmat[1])
    if len(rows) < 4:
        return None
    _, _, vt = np.linalg.svd(np.stack(rows), full_matrices=False)
    homog = vt[-1]
    if abs(homog[3]) < 1e-12:
        return None
    return homog[:3] / homog[3]


def triangulate_pose(
    refs: tuple[DetectionRef, ...],
    detections_by_view: list[list[dict[str, Any]]],
    pmats: list[np.ndarray],
) -> tuple[list[list[float]] | None, float]:
    num_joints = len(detections_by_view[refs[0].view][refs[0].index]["keypoints"])
    joints_3d: list[list[float]] = []
    sq_errors: list[float] = []
    for joint_idx in range(num_joints):
        observations: list[tuple[np.ndarray, np.ndarray]] = []
        for ref in refs:
            det = detections_by_view[ref.view][ref.index]
            kp = det["keypoints"][joint_idx]
            conf = float(kp[2]) if len(kp) > 2 else 1.0
            if conf <= 0:
                continue
            observations.append((pmats[ref.view], np.asarray(kp[:2], dtype=float)))
        point = triangulate_point(observations)
        if point is None:
            joints_3d.append([math.nan, math.nan, math.nan])
            continue
        joints_3d.append(point.tolist())
        for pmat, xy in observations:
            reproj = project_point(pmat, point)
            if np.all(np.isfinite(reproj)):
                sq_errors.append(float(np.sum((reproj - xy) ** 2)))
    if not sq_errors:
        return None, float("inf")
    return joints_3d, float(np.mean(sq_errors))


def enumerate_hyperedges(
    detections_by_view: list[list[dict[str, Any]]],
    pmats: list[np.ndarray],
    tau: float,
    lam: float,
    singleton_score: float,
    min_views: int = 2,
    max_views: int | None = None,
) -> list[Hyperedge]:
    num_views = len(detections_by_view)
    max_views = max_views or num_views
    edges: list[Hyperedge] = []
    for view_idx, detections in enumerate(detections_by_view):
        for det_idx, _ in enumerate(detections):
            edges.append(
                Hyperedge(
                    refs=(DetectionRef(view_idx, det_idx),),
                    cost=float("nan"),
                    score=singleton_score,
                    joints_3d=None,
                )
            )

    nonempty_views = [i for i, dets in enumerate(detections_by_view) if dets]
    for size in range(min_views, min(max_views, len(nonempty_views)) + 1):
        for view_subset in itertools.combinations(nonempty_views, size):
            index_ranges = [range(len(detections_by_view[v])) for v in view_subset]
            for combo in itertools.product(*index_ranges):
                refs = tuple(DetectionRef(v, i) for v, i in zip(view_subset, combo))
                pose, cost = triangulate_pose(refs, detections_by_view, pmats)
                if pose is None or cost > tau:
                    continue
                edges.append(Hyperedge(refs=refs, cost=cost, score=math.exp(-lam * cost), joints_3d=pose))
    return edges


def solve_exact_cover(edges: list[Hyperedge], gamma: float) -> list[int]:
    problem = pulp.LpProblem("compose_weighted_exact_cover", pulp.LpMaximize)
    variables = [pulp.LpVariable(f"x_{idx}", cat=pulp.LpBinary) for idx in range(len(edges))]
    problem += pulp.lpSum((edge.score - gamma) * variables[idx] for idx, edge in enumerate(edges))

    nodes = sorted({ref for edge in edges for ref in edge.refs}, key=lambda r: (r.view, r.index))
    for node in nodes:
        problem += pulp.lpSum(variables[idx] for idx, edge in enumerate(edges) if node in edge.refs) == 1

    solver = pulp.PULP_CBC_CMD(msg=False)
    status = problem.solve(solver)
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"ILP did not find an optimal exact cover: {pulp.LpStatus[status]}")
    return [idx for idx, var in enumerate(variables) if var.value() and var.value() > 0.5]


def run_frame(
    frame: dict[str, Any],
    cameras: list[dict[str, Any]],
    tau: float,
    lam: float,
    gamma: float,
    singleton_score: float,
    max_views: int | None,
) -> dict[str, Any]:
    detections_by_view = frame["detections"]
    pmats = [projection_matrix(camera) for camera in cameras]
    edges = enumerate_hyperedges(detections_by_view, pmats, tau, lam, singleton_score, max_views=max_views)
    selected = solve_exact_cover(edges, gamma)
    predictions = []
    for pred_idx, edge_idx in enumerate(selected):
        edge = edges[edge_idx]
        if len(edge.refs) < 2 or edge.joints_3d is None:
            continue
        predictions.append(
            {
                "pose_id": pred_idx,
                "joints_3d": edge.joints_3d,
                "views": [ref.view for ref in edge.refs],
                "detections": [ref.index for ref in edge.refs],
                "reprojection_cost": edge.cost,
                "score": edge.score,
            }
        )
    return {
        "frame_id": frame["frame_id"],
        "predicted_poses": predictions,
        "num_hyperedges": len(edges),
        "num_selected_edges": len(selected),
    }


def run_dataset(args: argparse.Namespace) -> dict[str, Any]:
    raw_data = json.loads(Path(args.data).read_text())
    data = normalize_dataset(raw_data)
    results = [
        run_frame(
            frame,
            data["cameras"],
            tau=args.tau,
            lam=args.lam,
            gamma=args.gamma,
            singleton_score=args.singleton_score,
            max_views=args.max_views,
        )
        for frame in data["frames"]
    ]
    return {
        "method": "compose",
        "dataset": data.get("name", Path(args.data).stem),
        "frames": results,
        "config": {
            "tau": args.tau,
            "lambda": args.lam,
            "gamma": args.gamma,
            "singleton_score": args.singleton_score,
            "max_views": args.max_views,
        },
    }


def normalize_dataset(data: dict[str, Any]) -> dict[str, Any]:
    if "frames" in data:
        return data
    joint_names = data["task"]["joint_names"]
    frames = []
    for scene in data["scenes"]:
        gt = []
        for person in scene["people"]:
            joints = person["joints_3d"]
            gt.append({"person_id": person["person_id"], "joints_3d": [joints[name] for name in joint_names]})
        detections_by_view = []
        for view in scene["views"]:
            detections = []
            for det in view["detections"]:
                by_joint = {joint["joint"]: joint for joint in det["joints"]}
                keypoints = []
                for name in joint_names:
                    joint = by_joint.get(name)
                    if joint is None:
                        keypoints.append([float("nan"), float("nan"), 0.0])
                    else:
                        keypoints.append([joint["xy"][0], joint["xy"][1], joint.get("confidence", 1.0)])
                detections.append(
                    {
                        "source_person": det.get("person_id"),
                        "keypoints": keypoints,
                        "confidence": min((kp[2] for kp in keypoints if np.isfinite(kp[2])), default=0.0),
                    }
                )
            detections_by_view.append(detections)
        frames.append({"frame_id": scene["scene_id"], "ground_truth": gt, "detections": detections_by_view})
    return {
        "name": data.get("schema_version", "synthetic_compose"),
        "units": "meters",
        "joint_names": joint_names,
        "cameras": data["cameras"],
        "frames": frames,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/synthetic_compose_dataset.json")
    parser.add_argument("--output", default="checkpoints/compose_predictions.json")
    parser.add_argument("--tau", type=float, default=900.0, help="Maximum mean squared reprojection error.")
    parser.add_argument("--lam", type=float, default=0.003, help="Exponential score sensitivity.")
    parser.add_argument("--gamma", type=float, default=0.06, help="Selected-edge sparsity penalty.")
    parser.add_argument("--singleton-score", type=float, default=0.03)
    parser.add_argument("--max-views", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(run_dataset(args), indent=2))


if __name__ == "__main__":
    main()

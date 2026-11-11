#!/usr/bin/env python3
"""COMPOSE: hypergraph weighted exact cover for multi-view multi-person 3D pose.

Implements the paper's pipeline (Wang et al., "COMPOSE: Hypergraph Cover
Optimization for Multi-view 3D Human Pose Estimation", Sec. 3):

1. Nodes are the 2D pose detections of every view.
2. A hyperedge is a set of detections from distinct views hypothesised to be
   one person.  Its geometric consistency cost is the reprojection error of
   the closed-form algebraic (DLT) triangulation, summed over the views of the
   hyperedge and averaged over the J joints:
       C(e) = 1/J * sum_j sum_{v in e} || pi_v(y_DLT,j) - u_{v,j} ||^2
3. Pruning: hyperedges with C(e) > tau are discarded.  Instead of instantiating
   every combination, hyperedges are grown one view at a time from hyperedges
   that survived pruning.
4. Score s(e) = exp(-lambda * C(e)); singleton hyperedges get s_single.
5. Weighted exact cover solved as an ILP (PuLP / CBC branch and bound):
       max sum_e (s(e) - gamma) x_e   s.t.  sum_{e ni u} x_e = 1 for every node u.
6. Selected multi-view hyperedges are triangulated into the output 3D poses,
   each scored with its hyperedge score s(e).

Usage:
    python method/compose.py --inputs IN.json --output OUT.json --config CONFIG.json
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pulp

DEFAULT_CONFIG = {"tau": 2000.0, "lambda": 0.001, "gamma": 1.25, "singleton_score": 0.05, "max_edge_size": None}


@dataclass
class Frame:
    frame_id: str
    pmats: np.ndarray                 # (V, 3, 4)
    keypoints: list[np.ndarray]       # per view: (n_v, J, 3) with (u, v, confidence)
    scores: list[np.ndarray]          # per view: (n_v,)


@dataclass
class Candidate:
    refs: tuple[tuple[int, int], ...]  # ((view, detection index), ...) sorted by view
    cost: float
    pose: np.ndarray                   # (J, 3)


def projection_matrix(camera: dict[str, Any]) -> np.ndarray:
    k = np.asarray(camera["K"], dtype=float)
    r = np.asarray(camera["R"], dtype=float)
    t = np.asarray(camera["t"], dtype=float).reshape(3, 1)
    return k @ np.concatenate([r, t], axis=1)


def parse_frame(frame: dict[str, Any], num_joints: int) -> Frame:
    cams = {cam["camera_id"]: cam for cam in frame["cameras"]}
    pmats, keypoints, scores = [], [], []
    for view in frame["views"]:
        pmats.append(projection_matrix(cams[view["camera_id"]]))
        dets = view.get("detections", [])
        if dets:
            keypoints.append(np.asarray([det["keypoints"] for det in dets], dtype=float).reshape(len(dets), num_joints, 3))
            scores.append(np.asarray([det.get("score", 1.0) for det in dets], dtype=float))
        else:
            keypoints.append(np.zeros((0, num_joints, 3)))
            scores.append(np.zeros((0,)))
    return Frame(str(frame["frame_id"]), np.stack(pmats), keypoints, scores)


def dlt_triangulate(pmats: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Batched DLT. ``pmats`` (m, k, 3, 4), ``uv`` (m, k, J, 2) -> points (m, J, 3)."""

    p0 = pmats[:, :, None, 0, :]
    p1 = pmats[:, :, None, 1, :]
    p2 = pmats[:, :, None, 2, :]
    rows_u = uv[..., 0:1] * p2 - p0                     # (m, k, J, 4)
    rows_v = uv[..., 1:2] * p2 - p1
    a = np.concatenate([rows_u, rows_v], axis=1)        # (m, 2k, J, 4)
    a = np.transpose(a, (0, 2, 1, 3))                   # (m, J, 2k, 4)
    a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)
    _, _, vt = np.linalg.svd(a)
    homog = vt[..., -1, :]                              # (m, J, 4)
    w = homog[..., 3:4]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return homog[..., :3] / w


def reprojection_cost(pmats: np.ndarray, uv: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Paper cost: mean over joints of the sum over views of squared pixel errors -> (m,)."""

    homog = np.concatenate([points, np.ones(points.shape[:-1] + (1,))], axis=-1)  # (m, J, 4)
    proj = np.einsum("mkab,mjb->mkja", pmats, homog)                                # (m, k, J, 3)
    z = proj[..., 2:3]
    z = np.where(np.abs(z) < 1e-9, 1e-9, z)
    err = np.sum((proj[..., :2] / z - uv) ** 2, axis=-1)                            # (m, k, J)
    err = np.where(np.isfinite(err), err, 1e12)
    return err.sum(axis=1).mean(axis=1)


def evaluate_edges(frame: Frame, refs_list: list[tuple[tuple[int, int], ...]]) -> list[Candidate]:
    """Triangulate and cost a batch of hyperedges (grouped by their view tuple)."""

    out: list[Candidate] = []
    groups: dict[tuple[int, ...], list[tuple[tuple[int, int], ...]]] = {}
    for refs in refs_list:
        groups.setdefault(tuple(v for v, _ in refs), []).append(refs)
    for views, group in groups.items():
        pm = np.broadcast_to(frame.pmats[list(views)], (len(group), len(views), 3, 4))
        uv = np.stack(
            [np.stack([frame.keypoints[v][i, :, :2] for v, i in refs]) for refs in group]
        )                                                # (m, k, J, 2)
        points = dlt_triangulate(pm, uv)
        costs = reprojection_cost(pm, uv, points)
        for refs, cost, pose in zip(group, costs, points):
            out.append(Candidate(refs, float(cost), pose))
    return out


def generate_candidates(frame: Frame, tau: float, max_edge_size: int | None = None) -> list[Candidate]:
    """Multi-view hyperedges with C(e) <= tau, grown from surviving sub-hyperedges."""

    views = [v for v, kp in enumerate(frame.keypoints) if len(kp)]
    max_size = min(max_edge_size or len(views), len(views))
    if max_size < 2:
        return []
    level = [
        ((a, i), (b, j))
        for a, b in itertools.combinations(views, 2)
        for i in range(len(frame.keypoints[a]))
        for j in range(len(frame.keypoints[b]))
    ]
    kept: list[Candidate] = []
    size = 2
    while level:
        survivors = [c for c in evaluate_edges(frame, level) if c.cost <= tau]
        kept.extend(survivors)
        size += 1
        if size > max_size:
            break
        seen: set[tuple[tuple[int, int], ...]] = set()
        nxt = []
        for cand in survivors:
            used = {v for v, _ in cand.refs}
            for v in views:
                if v in used:
                    continue
                for i in range(len(frame.keypoints[v])):
                    refs = tuple(sorted(cand.refs + ((v, i),)))
                    if refs not in seen:
                        seen.add(refs)
                        nxt.append(refs)
        level = nxt
    return kept


def restrict_candidates(candidates: Iterable[Candidate], tau: float) -> list[Candidate]:
    """Exactly the candidates ``generate_candidates`` would keep at a smaller ``tau``."""

    by_size: dict[int, list[Candidate]] = {}
    for cand in candidates:
        by_size.setdefault(len(cand.refs), []).append(cand)
    kept_refs: set[tuple[tuple[int, int], ...]] = set()
    out = []
    for size in sorted(by_size):
        for cand in by_size[size]:
            if cand.cost > tau:
                continue
            if size > 2 and not any(
                tuple(r for r in cand.refs if r != drop) in kept_refs for drop in cand.refs
            ):
                continue
            kept_refs.add(cand.refs)
            out.append(cand)
    return out


def solve_exact_cover(
    frame: Frame, candidates: list[Candidate], lam: float, gamma: float, singleton_score: float
) -> list[tuple[Candidate, float]]:
    """Weighted exact cover ILP; returns the selected multi-view hyperedges with s(e)."""

    nodes = [(v, i) for v, kp in enumerate(frame.keypoints) for i in range(len(kp))]
    if not nodes:
        return []
    scores = [float(np.exp(-lam * c.cost)) for c in candidates]
    problem = pulp.LpProblem("compose_exact_cover", pulp.LpMaximize)
    x_multi = [pulp.LpVariable(f"e{k}", cat=pulp.LpBinary) for k in range(len(candidates))]
    x_single = {node: pulp.LpVariable(f"s{node[0]}_{node[1]}", cat=pulp.LpBinary) for node in nodes}
    problem += pulp.lpSum((s - gamma) * x for s, x in zip(scores, x_multi)) + pulp.lpSum(
        (singleton_score - gamma) * x for x in x_single.values()
    )
    covering: dict[tuple[int, int], list[Any]] = {node: [x_single[node]] for node in nodes}
    for cand, x in zip(candidates, x_multi):
        for ref in cand.refs:
            covering[ref].append(x)
    for node in nodes:
        problem += pulp.lpSum(covering[node]) == 1
    status = problem.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"exact cover ILP not optimal for frame {frame.frame_id}: {pulp.LpStatus[status]}")
    return [(cand, s) for cand, s, x in zip(candidates, scores, x_multi) if x.value() is not None and x.value() > 0.5]


def predict_frame(frame: Frame, config: dict[str, Any], candidates: list[Candidate] | None = None) -> list[dict[str, Any]]:
    if candidates is None:
        candidates = generate_candidates(frame, float(config["tau"]), config.get("max_edge_size"))
    selected = solve_exact_cover(
        frame, candidates, float(config["lambda"]), float(config["gamma"]), float(config["singleton_score"])
    )
    return [
        {"joints_3d": cand.pose.tolist(), "score": score, "views": [v for v, _ in cand.refs]}
        for cand, score in selected
        if np.all(np.isfinite(cand.pose))
    ]


def load_inputs(path: str | Path) -> tuple[list[Frame], int]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    num_joints = len(payload["joint_names"])
    return [parse_frame(frame, num_joints) for frame in payload["frames"]], num_joints


def run(inputs: str | Path, output: str | Path, config: dict[str, Any]) -> None:
    frames, _ = load_inputs(inputs)
    result = {frame.frame_id: {"predicted_poses": predict_frame(frame, config)} for frame in frames}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps({"method": "compose", "config": config, "frames": result}), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", required=True, help="Inputs JSON (cameras + 2D detections)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=None, help="JSON with tau, lambda, gamma, singleton_score")
    parser.add_argument("--tau", type=float)
    parser.add_argument("--lam", type=float)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--singleton-score", type=float)
    parser.add_argument("--max-edge-size", type=int, help="Ablation: largest hyperedge (2 = pairs only)")
    args = parser.parse_args()
    config = dict(DEFAULT_CONFIG)
    if args.config:
        config.update(json.loads(Path(args.config).read_text(encoding="utf-8")))
    for key, value in (
        ("tau", args.tau), ("lambda", args.lam), ("gamma", args.gamma),
        ("singleton_score", args.singleton_score), ("max_edge_size", args.max_edge_size),
    ):
        if value is not None:
            config[key] = value
    run(args.inputs, args.output, config)


if __name__ == "__main__":
    main()

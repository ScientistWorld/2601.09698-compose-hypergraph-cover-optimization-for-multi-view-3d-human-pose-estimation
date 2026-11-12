#!/usr/bin/env python3
"""Geometry-only MvPose-style baseline (Dong et al., CVPR 2019 / TPAMI 2022).

The paper's main optimisation-based competitor associates detections with a
*pairwise* affinity matrix made cycle-consistent by convex multi-way matching.
This re-implementation keeps that pipeline with geometric cues only (no
appearance network, as the benchmark provides no images):

1. pairwise affinity: symmetric point-to-epipolar-line distance averaged over
   joints, z-normalised and squashed with a sigmoid (MvPose ``geometry_affinity``);
2. multi-way matching: MvPose ``matchSVT`` (ADMM with singular value
   thresholding and doubly-stochastic block projection), binarised at 0.5;
3. clusters with detections from >= 2 views are triangulated with the same DLT
   used by the other methods; a cluster's score is its mean detection score.

``--tune-on TRAIN.json`` selects the matching weight ``alpha`` on train frames.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def projection_matrix(camera: dict) -> np.ndarray:
    k = np.asarray(camera["K"], dtype=float)
    r = np.asarray(camera["R"], dtype=float)
    t = np.asarray(camera["t"], dtype=float).reshape(3, 1)
    return k @ np.concatenate([r, t], axis=1)


def fundamental(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """F with x2^T F x1 = 0."""
    _, _, vt = np.linalg.svd(p1)
    center = vt[-1]
    e2 = p2 @ center
    skew = np.array([[0, -e2[2], e2[1]], [e2[2], 0, -e2[0]], [-e2[1], e2[0], 0]])
    return skew @ p2 @ np.linalg.pinv(p1)


def epipolar_distance(pts1: np.ndarray, pts2: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Mean over joints of the distance of pts2 to the epipolar lines of pts1: (n1, n2)."""
    h1 = np.concatenate([pts1, np.ones(pts1.shape[:-1] + (1,))], axis=-1)   # (n1, J, 3)
    h2 = np.concatenate([pts2, np.ones(pts2.shape[:-1] + (1,))], axis=-1)   # (n2, J, 3)
    lines = np.einsum("ab,njb->nja", f, h1)                                   # (n1, J, 3)
    num = np.abs(np.einsum("nja,mja->nmj", lines, h2))
    den = np.linalg.norm(lines[..., :2], axis=-1)[:, None, :]
    return (num / np.maximum(den, 1e-9)).mean(axis=-1)


def geometry_affinity(points: list[np.ndarray], pmats: np.ndarray) -> np.ndarray:
    dims = np.cumsum([0] + [len(p) for p in points])
    total = dims[-1]
    dist = np.full((total, total), 25.0)
    np.fill_diagonal(dist, 0.0)
    for a, b in itertools.combinations(range(len(points)), 2):
        if len(points[a]) == 0 or len(points[b]) == 0:
            continue
        fab = fundamental(pmats[a], pmats[b])
        fba = fundamental(pmats[b], pmats[a])
        d = 0.5 * (epipolar_distance(points[a], points[b], fab) + epipolar_distance(points[b], points[a], fba).T)
        dist[dims[a]:dims[a + 1], dims[b]:dims[b + 1]] = d
        dist[dims[b]:dims[b + 1], dims[a]:dims[a + 1]] = d.T
    std = dist.std()
    aff = -(dist - dist.mean()) / (std if std > 1e-9 else 1.0)
    return 1.0 / (1.0 + np.exp(-5.0 * aff)), dims


def project_rows(y: np.ndarray) -> np.ndarray:
    """Project every row onto {x >= 0, sum(x) <= 1} (MvPose ``proj2pav``, vectorised)."""
    y = np.maximum(y, 0.0)
    need = y.sum(axis=1) >= 1.0
    if not need.any():
        return y
    rows = y[need]
    u = -np.sort(-rows, axis=1)
    sv = np.cumsum(u, axis=1)
    cond = u > (sv - 1.0) / np.arange(1, u.shape[1] + 1)
    rho = u.shape[1] - 1 - np.argmax(cond[:, ::-1], axis=1)
    theta = np.maximum(0.0, (sv[np.arange(len(rho)), rho] - 1.0) / (rho + 1.0))
    out = y.copy()
    out[need] = np.maximum(rows - theta[:, None], 0.0)
    return out


def project_blocks(x: np.ndarray, dims: np.ndarray, tol: float = 1e-2) -> np.ndarray:
    """MvPose ``proj2dpam`` (Dykstra row/column projections) on every view block at once."""
    spans = [(int(dims[i]), int(dims[i + 1])) for i in range(len(dims) - 1) if dims[i + 1] > dims[i]]

    def rows(z: np.ndarray) -> np.ndarray:
        out = z.copy()
        for c0, c1 in spans:
            out[:, c0:c1] = project_rows(z[:, c0:c1])
        return out

    def cols(z: np.ndarray) -> np.ndarray:
        out = z.copy()
        for r0, r1 in spans:
            out[r0:r1, :] = project_rows(z[r0:r1, :].T).T
        return out

    x0 = x.copy()
    cur = x.copy()
    i2 = np.zeros_like(x)
    for _ in range(10):
        x1 = rows(x0 + i2)
        i1 = x1 - (x0 + i2)
        x2 = cols(x0 + i1)
        i2 = x2 - (x0 + i1)
        change = np.abs(x2 - cur).sum() / x.size
        cur = x2
        if change < tol:
            break
    return cur


def match_svt(s: np.ndarray, dims: np.ndarray, alpha: float, lam: float = 50.0, mu: float = 64.0,
              max_iter: int = 500, tol: float = 5e-4) -> np.ndarray:
    n = s.shape[0]
    s = s.copy()
    np.fill_diagonal(s, 0.0)
    s = 0.5 * (s + s.T)
    x = s.copy()
    y = np.zeros_like(s)
    w = alpha - s
    for _ in range(max_iter):
        x0 = x
        u, sing, vt = np.linalg.svd(y / mu + x)
        q = (u * np.maximum(sing - lam / mu, 0.0)) @ vt
        x = q - (w + y) / mu
        for i in range(len(dims) - 1):
            x[dims[i]:dims[i + 1], dims[i]:dims[i + 1]] = 0.0
        np.fill_diagonal(x, 1.0)
        x = np.clip(x, 0.0, 1.0)
        x = project_blocks(x, dims)
        x = 0.5 * (x + x.T)
        y = y + mu * (x - q)
        p_res = np.linalg.norm(x - q) / n
        d_res = mu * np.linalg.norm(x - x0) / n
        if p_res < tol and d_res < tol:
            break
        if p_res > 10 * d_res:
            mu *= 2.0
        elif d_res > 10 * p_res:
            mu /= 2.0
    return 0.5 * (x + x.T) > 0.5


def dlt(pmats: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """pmats (k, 3, 4), uv (k, J, 2) -> (J, 3)."""
    rows_u = uv[..., 0:1] * pmats[:, None, 2, :] - pmats[:, None, 0, :]
    rows_v = uv[..., 1:2] * pmats[:, None, 2, :] - pmats[:, None, 1, :]
    a = np.transpose(np.concatenate([rows_u, rows_v], axis=0), (1, 0, 2))
    a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)
    _, _, vt = np.linalg.svd(a)
    h = vt[:, -1, :]
    w = np.where(np.abs(h[:, 3:4]) < 1e-12, np.nan, h[:, 3:4])
    return h[:, :3] / w


def predict_frame(frame: dict, alpha: float, num_joints: int) -> list[dict]:
    cams = {c["camera_id"]: c for c in frame["cameras"]}
    pmats = np.stack([projection_matrix(cams[v["camera_id"]]) for v in frame["views"]])
    points, kps, scores, owner = [], [], [], []
    for vi, view in enumerate(frame["views"]):
        dets = view.get("detections", [])
        arr = np.asarray([d["keypoints"] for d in dets], dtype=float).reshape(len(dets), num_joints, 3)
        points.append(arr[..., :2])
        for di, det in enumerate(dets):
            kps.append(arr[di, :, :2])
            scores.append(float(det.get("score", 1.0)))
            owner.append(vi)
    if sum(len(p) for p in points) < 2:
        return []
    aff, dims = geometry_affinity(points, pmats)
    match = match_svt(aff, dims, alpha)
    persons = np.nonzero(match.sum(axis=0) > 1.9)[0]
    if len(persons) == 0:
        return []
    sub = match[:, persons]
    clusters: dict[int, list[int]] = {}
    for det_idx, row in enumerate(sub):
        if row.any():
            clusters.setdefault(int(np.argmax(row)), []).append(det_idx)
    out = []
    for members in clusters.values():
        best: dict[int, int] = {}
        for m in members:
            if owner[m] not in best or scores[m] > scores[best[owner[m]]]:
                best[owner[m]] = m
        if len(best) < 2:
            continue
        chosen = list(best.values())
        pose = dlt(pmats[[owner[m] for m in chosen]], np.stack([kps[m] for m in chosen]))
        if np.all(np.isfinite(pose)):
            out.append({"joints_3d": pose.tolist(), "score": float(np.mean([scores[m] for m in chosen]))})
    return out


def run(payload: dict, alpha: float) -> dict:
    num_joints = len(payload["joint_names"])
    return {str(f["frame_id"]): {"predicted_poses": predict_frame(f, alpha, num_joints)} for f in payload["frames"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", help="Inputs JSON (cameras + 2D detections)")
    parser.add_argument("--output", help="Prediction JSON")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--config", default=None, help="JSON with 'alpha'")
    parser.add_argument("--tune-on", default=None, help="Train benchmark JSON: select alpha, write --config")
    args = parser.parse_args()
    if args.tune_on:
        from eval.metrics import compute_metrics, ground_truth_from_frames

        bench = json.loads(Path(args.tune_on).read_text(encoding="utf-8"))
        gt = ground_truth_from_frames(bench["frames"])
        rows = []
        for alpha in (0.5, 0.7, 0.8, 0.9, 0.93, 0.96, 0.99):
            frames = run(bench, alpha)
            preds = {
                fid: [((np.asarray(p["joints_3d"]) * 1000.0).tolist(), p["score"]) for p in fr["predicted_poses"]]
                for fid, fr in frames.items()
            }
            m = compute_metrics(preds, gt)
            u = m["ap_25mm"] + m["ap_50mm"] + m["ap_100mm"] + m["ap_150mm"] + m["recall_500mm"] - m["mpjpe_mm"]
            rows.append((u, alpha, m))
            print(f"alpha={alpha} U={u:.2f} {json.dumps(m)}", flush=True)
        best = max(rows, key=lambda r: r[0])
        Path(args.config).write_text(json.dumps({"alpha": best[1]}) + "\n", encoding="utf-8")
        return
    alpha = json.loads(Path(args.config).read_text())["alpha"] if args.config else args.alpha
    payload = json.loads(Path(args.inputs).read_text(encoding="utf-8"))
    Path(args.output).write_text(json.dumps({"method": "mvpose_geometric", "frames": run(payload, alpha)}),
                                 encoding="utf-8")


if __name__ == "__main__":
    main()

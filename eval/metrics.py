"""Validation and metrics for multi-person 3D pose predictions (public copy).

Protocol (the standard CMU Panoptic multi-person evaluation used by the paper):
every predicted pose is compared with every annotated person of its frame by
MPJPE over that person's annotated joints and assigned to the closest one.
Predictions are then ranked by their ``score`` over the whole split.

* ``ap_<t>mm``     average precision; a prediction is a true positive when its
                   MPJPE to its closest person is below ``t`` mm and that
                   person has not already been claimed by a higher-scored
                   prediction.
* ``recall_500mm`` share of people that are the closest person of at least
                   one prediction within 500 mm.
* ``mpjpe_mm``     mean MPJPE of the first (highest-scored) prediction claiming
                   each person within 500 mm.

All values are percentages except ``mpjpe_mm`` (millimetres).  A run whose
output is missing, unparsable or contains an impossible value is a failed run
and receives ``FAILURE_METRICS`` -- the worst value of every metric.  A valid
run that matches nobody gets ``mpjpe_mm = 500`` (the matching radius).

Pure standard-library Python.
"""

from __future__ import annotations

import math
from typing import Any

NUM_JOINTS = 15
AP_THRESHOLDS_MM = (25.0, 50.0, 100.0, 150.0)
RECALL_THRESHOLD_MM = 500.0
MAX_ABS_COORD_M = 10.0   # capture volume bound; anything outside is impossible
METRIC_NAMES = ("ap_25mm", "ap_50mm", "ap_100mm", "ap_150mm", "recall_500mm", "mpjpe_mm")
FAILURE_METRICS = {
    "ap_25mm": 0.0,
    "ap_50mm": 0.0,
    "ap_100mm": 0.0,
    "ap_150mm": 0.0,
    "recall_500mm": 0.0,
    "mpjpe_mm": RECALL_THRESHOLD_MM,
}


class InvalidPredictions(ValueError):
    """Raised when a prediction file violates the output contract."""


def _is_real(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def frame_detection_counts(inputs: dict[str, Any]) -> dict[str, int]:
    """Number of 2D detections per input frame (upper bound on emitted poses)."""

    return {
        str(frame["frame_id"]): sum(len(view.get("detections", [])) for view in frame.get("views", []))
        for frame in inputs["frames"]
    }


def validate_predictions(payload: Any, max_poses: dict[str, int]) -> dict[str, list[tuple[list[list[float]], float]]]:
    """Parse and check a prediction payload.

    Contract: ``{"frames": {frame_id: {"predicted_poses": [{"joints_3d": 15x3 metres, "score": s}]}}}``
    (``frames`` may also be a list of objects carrying ``frame_id``).  Every
    input frame must be present; a frame may hold at most as many poses as it
    has 2D detections; every pose needs exactly 15 joints of 3 finite
    coordinates inside the capture volume and a finite score.
    Returns ``{frame_id: [(joints_mm, score), ...]}``.
    """

    if not isinstance(payload, dict) or "frames" not in payload:
        raise InvalidPredictions("prediction JSON must be an object with a 'frames' entry")
    frames = payload["frames"]
    if isinstance(frames, list):
        mapping = {}
        for item in frames:
            if not isinstance(item, dict) or "frame_id" not in item:
                raise InvalidPredictions("list-style 'frames' entries need a 'frame_id'")
            mapping[str(item["frame_id"])] = item
        frames = mapping
    if not isinstance(frames, dict):
        raise InvalidPredictions("'frames' must be an object or a list")

    out: dict[str, list[tuple[list[list[float]], float]]] = {}
    for frame_id, limit in max_poses.items():
        if frame_id not in frames:
            raise InvalidPredictions(f"missing frame {frame_id!r}")
        entry = frames[frame_id]
        poses = entry.get("predicted_poses") if isinstance(entry, dict) else None
        if not isinstance(poses, list):
            raise InvalidPredictions(f"frame {frame_id!r}: 'predicted_poses' must be a list")
        if len(poses) > limit:
            raise InvalidPredictions(
                f"frame {frame_id!r}: {len(poses)} poses but only {limit} 2D detections"
            )
        parsed = []
        for idx, pose in enumerate(poses):
            if not isinstance(pose, dict):
                raise InvalidPredictions(f"frame {frame_id!r} pose {idx}: must be an object")
            joints = pose.get("joints_3d")
            if not isinstance(joints, list) or len(joints) != NUM_JOINTS:
                raise InvalidPredictions(f"frame {frame_id!r} pose {idx}: need exactly {NUM_JOINTS} joints")
            joints_mm = []
            for joint in joints:
                if not isinstance(joint, list) or len(joint) != 3 or not all(_is_real(v) for v in joint):
                    raise InvalidPredictions(f"frame {frame_id!r} pose {idx}: joints must be 3 finite numbers")
                if any(abs(v) > MAX_ABS_COORD_M for v in joint):
                    raise InvalidPredictions(f"frame {frame_id!r} pose {idx}: joint outside the capture volume")
                joints_mm.append([1000.0 * float(v) for v in joint])
            score = pose.get("score")
            if not _is_real(score):
                raise InvalidPredictions(f"frame {frame_id!r} pose {idx}: 'score' must be a finite number")
            parsed.append((joints_mm, float(score)))
        out[frame_id] = parsed
    return out


def ground_truth_from_frames(frames: list[dict[str, Any]]) -> dict[str, list[tuple[list[list[float]], list[bool]]]]:
    """``{frame_id: [(joints_mm, joint_valid), ...]}`` from benchmark frames with ``people``."""

    return {
        str(frame["frame_id"]): [
            ([[1000.0 * float(v) for v in joint] for joint in person["joints_3d"]], list(person["joint_valid"]))
            for person in frame["people"]
        ]
        for frame in frames
    }


def _mpjpe(pred: list[list[float]], gt: list[list[float]], valid: list[bool]) -> float:
    errors = [math.dist(p, g) for p, g, ok in zip(pred, gt, valid) if ok]
    return sum(errors) / len(errors) if errors else math.inf


def compute_metrics(
    predictions: dict[str, list[tuple[list[list[float]], float]]],
    ground_truth: dict[str, list[tuple[list[list[float]], list[bool]]]],
) -> dict[str, float]:
    """Paper protocol metrics for validated predictions."""

    eval_list: list[tuple[float, float, tuple[str, int] | None]] = []
    total_gt = 0
    for frame_id, gts in ground_truth.items():
        total_gt += len(gts)
        for joints, score in predictions.get(frame_id, []):
            if not gts:
                eval_list.append((math.inf, score, None))
                continue
            dists = [_mpjpe(joints, gt_joints, valid) for gt_joints, valid in gts]
            best = min(range(len(dists)), key=lambda k: dists[k])
            eval_list.append((dists[best], score, (frame_id, best)))
    if total_gt == 0:
        raise ValueError("ground truth contains no people")
    eval_list.sort(key=lambda item: -item[1])  # stable: ties keep frame order

    metrics: dict[str, float] = {}
    for threshold in AP_THRESHOLDS_MM:
        metrics[f"ap_{int(threshold)}mm"] = 100.0 * _average_precision(eval_list, total_gt, threshold)
    recalled = {gt_id for dist, _, gt_id in eval_list if gt_id is not None and dist < RECALL_THRESHOLD_MM}
    metrics["recall_500mm"] = 100.0 * len(recalled) / total_gt
    claimed: set[tuple[str, int]] = set()
    errors = []
    for dist, _, gt_id in eval_list:
        if gt_id is not None and dist < RECALL_THRESHOLD_MM and gt_id not in claimed:
            claimed.add(gt_id)
            errors.append(dist)
    metrics["mpjpe_mm"] = sum(errors) / len(errors) if errors else RECALL_THRESHOLD_MM
    return metrics


def _average_precision(eval_list: list[tuple[float, float, Any]], total_gt: int, threshold: float) -> float:
    if not eval_list:
        return 0.0
    claimed: set[Any] = set()
    tp_cum = fp_cum = 0.0
    recall = []
    precision = []
    for dist, _, gt_id in eval_list:
        if gt_id is not None and dist < threshold and gt_id not in claimed:
            claimed.add(gt_id)
            tp_cum += 1.0
        else:
            fp_cum += 1.0
        recall.append(tp_cum / (total_gt + 1e-5))
        precision.append(tp_cum / (tp_cum + fp_cum + 1e-5))
    for n in range(len(precision) - 2, -1, -1):
        precision[n] = max(precision[n], precision[n + 1])
    precision = [0.0] + precision + [0.0]
    recall = [0.0] + recall + [1.0]
    ap = 0.0
    for i in range(len(recall) - 1):
        if recall[i + 1] != recall[i]:
            ap += (recall[i + 1] - recall[i]) * precision[i + 1]
    return ap


def score_payload(payload: Any, inputs: dict[str, Any], ground_truth: dict[str, Any]) -> tuple[dict[str, float], str]:
    """Validate and score; returns ``(metrics, status)`` with failure values on invalid output."""

    try:
        predictions = validate_predictions(payload, frame_detection_counts(inputs))
    except InvalidPredictions as exc:
        return dict(FAILURE_METRICS), f"failed: {exc}"
    return compute_metrics(predictions, ground_truth), "ok"

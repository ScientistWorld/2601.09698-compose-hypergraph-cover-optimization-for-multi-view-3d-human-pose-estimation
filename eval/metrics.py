"""Metrics for multi-person 3D pose prediction.

The evaluator is intentionally independent of method code. It consumes Python
objects parsed from JSON and assumes coordinates are already in millimeters.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable


DEFAULT_THRESHOLDS_MM = (25.0, 50.0, 100.0, 150.0)


@dataclass(frozen=True)
class PoseRecord:
    """Normalized pose container used by evaluation."""

    pose_id: str
    joints_3d: list[list[float]]
    score: float = 1.0


def iter_frames(payload: Any) -> list[tuple[str, dict[str, Any]]]:
    """Return ``(frame_id, frame)`` pairs from accepted JSON layouts."""

    frames = payload.get("frames", payload) if isinstance(payload, dict) else payload
    if isinstance(frames, dict):
        return [(str(frame_id), frame) for frame_id, frame in frames.items()]
    if isinstance(frames, list):
        out = []
        for idx, frame in enumerate(frames):
            frame_id = frame.get("frame_id", frame.get("id", idx)) if isinstance(frame, dict) else idx
            out.append((str(frame_id), frame))
        return out
    raise TypeError("Expected a JSON object/list with frames")


def normalize_data_payload(payload: Any) -> Any:
    """Convert the tracked benchmark schema to the evaluator frame schema."""

    if not isinstance(payload, dict) or "scenes" not in payload:
        return payload
    joint_names = payload["task"]["joint_names"]
    frames = []
    for scene in payload["scenes"]:
        ground_truth = []
        for person in scene.get("people", []):
            joints = person["joints_3d"]
            ground_truth.append(
                {
                    "person_id": person.get("person_id", len(ground_truth)),
                    "joints_3d": [joints[name] for name in joint_names],
                }
            )
        frames.append({"frame_id": scene["scene_id"], "ground_truth": ground_truth})
    return {"frames": frames}


def normalize_joints(joints: Any) -> list[list[float]]:
    """Convert a JSON joint container to a ``J x 3`` float list."""

    if isinstance(joints, dict):
        try:
            keys = sorted(joints, key=lambda item: int(item))
        except (TypeError, ValueError):
            keys = sorted(joints)
        joints = [joints[key] for key in keys]

    out = []
    for joint in joints:
        if not isinstance(joint, (list, tuple)) or len(joint) < 3:
            raise ValueError("Expected each 3D joint to contain at least three coordinates")
        out.append([float(joint[0]), float(joint[1]), float(joint[2])])
    return out


def extract_poses(frame: dict[str, Any], key: str, id_key: str) -> list[PoseRecord]:
    """Extract normalized poses from ``frame[key]``."""

    records = []
    for idx, item in enumerate(frame.get(key, []) or []):
        pose_id = str(item.get(id_key, item.get("pose_id", item.get("id", idx))))
        score = float(item.get("score", item.get("confidence", 1.0)))
        records.append(PoseRecord(pose_id=pose_id, joints_3d=normalize_joints(item["joints_3d"]), score=score))
    return records


def pose_mpjpe_mm(pred: list[list[float]], gt: list[list[float]]) -> float:
    """Mean per-joint position error for one predicted/GT pose pair."""

    joint_count = min(len(pred), len(gt))
    if joint_count == 0:
        return math.inf
    errors = []
    for pred_joint, gt_joint in zip(pred[:joint_count], gt[:joint_count]):
        if not all(math.isfinite(value) for value in pred_joint + gt_joint):
            continue
        errors.append(
            math.sqrt(
                (pred_joint[0] - gt_joint[0]) ** 2
                + (pred_joint[1] - gt_joint[1]) ** 2
                + (pred_joint[2] - gt_joint[2]) ** 2
            )
        )
    if not errors:
        return math.inf
    return sum(errors) / len(errors)


def distance_matrix(preds: list[PoseRecord], gts: list[PoseRecord]) -> list[list[float]]:
    """Pairwise MPJPE distances in millimeters."""

    mat = [[math.inf for _ in gts] for _ in preds]
    for pred_idx, pred in enumerate(preds):
        for gt_idx, gt in enumerate(gts):
            mat[pred_idx][gt_idx] = pose_mpjpe_mm(pred.joints_3d, gt.joints_3d)
    return mat


def greedy_min_error_matches(
    distances: list[list[float]],
    max_error: float | None = None,
) -> list[tuple[int, int, float]]:
    """One-to-one GT/pred matches chosen greedily by smallest available error."""

    if not distances or not distances[0]:
        return []
    candidates = []
    for pred_idx, row in enumerate(distances):
        for gt_idx, dist in enumerate(row):
            if math.isfinite(dist) and (max_error is None or dist <= max_error):
                candidates.append((dist, pred_idx, gt_idx))
    candidates.sort(key=lambda item: item[0])

    used_preds: set[int] = set()
    used_gts: set[int] = set()
    matches = []
    for dist, pred_idx, gt_idx in candidates:
        if pred_idx in used_preds or gt_idx in used_gts:
            continue
        used_preds.add(pred_idx)
        used_gts.add(gt_idx)
        matches.append((pred_idx, gt_idx, dist))
    return matches


def average_precision_for_threshold(
    frame_preds: Iterable[list[PoseRecord]],
    frame_gts: Iterable[list[PoseRecord]],
    threshold_mm: float,
) -> float:
    """Compute pose AP at an MPJPE threshold with score-ordered greedy matching."""

    all_preds: list[tuple[float, int, int]] = []
    gt_counts: list[int] = []
    distances_by_frame: list[list[list[float]]] = []
    preds_by_frame = list(frame_preds)
    gts_by_frame = list(frame_gts)

    for frame_idx, (preds, gts) in enumerate(zip(preds_by_frame, gts_by_frame)):
        gt_counts.append(len(gts))
        distances_by_frame.append(distance_matrix(preds, gts))
        for pred_idx, pred in enumerate(preds):
            all_preds.append((pred.score, frame_idx, pred_idx))

    total_gt = sum(gt_counts)
    if total_gt == 0 or not all_preds:
        return 0.0

    all_preds.sort(key=lambda item: item[0], reverse=True)
    used_gts = [set() for _ in gt_counts]
    tp = [0.0 for _ in all_preds]
    fp = [0.0 for _ in all_preds]

    for rank, (_, frame_idx, pred_idx) in enumerate(all_preds):
        distances = distances_by_frame[frame_idx]
        if not distances or not distances[0]:
            fp[rank] = 1.0
            continue
        best_gt = None
        best_dist = math.inf
        for gt_idx, dist in enumerate(distances[pred_idx]):
            if gt_idx in used_gts[frame_idx]:
                continue
            if dist < best_dist:
                best_dist = dist
                best_gt = gt_idx
        if best_gt is not None and best_dist <= threshold_mm:
            used_gts[frame_idx].add(best_gt)
            tp[rank] = 1.0
        else:
            fp[rank] = 1.0

    recall = [0.0]
    precision = [0.0]
    running_tp = 0.0
    running_fp = 0.0
    for tp_value, fp_value in zip(tp, fp):
        running_tp += tp_value
        running_fp += fp_value
        recall.append(running_tp / float(total_gt))
        precision.append(running_tp / max(running_tp + running_fp, 1.0))
    recall.append(1.0)
    precision.append(0.0)

    for idx in range(len(precision) - 2, -1, -1):
        precision[idx] = max(precision[idx], precision[idx + 1])
    ap = 0.0
    for idx in range(len(recall) - 1):
        if recall[idx + 1] != recall[idx]:
            ap += (recall[idx + 1] - recall[idx]) * precision[idx + 1]
    return ap


def evaluate_pose_predictions(
    prediction_payload: Any,
    data_payload: Any,
    thresholds_mm: Iterable[float] = DEFAULT_THRESHOLDS_MM,
    recall_threshold_mm: float = 500.0,
    unit_scale: float = 1.0,
) -> dict[str, Any]:
    """Evaluate predictions against ground truth frames."""

    pred_frames = dict(iter_frames(prediction_payload))
    data_frames = iter_frames(normalize_data_payload(data_payload))

    all_frame_preds: list[list[PoseRecord]] = []
    all_frame_gts: list[list[PoseRecord]] = []
    mpjpe_values = []
    recall_matches = 0
    total_gt = 0
    total_predictions = 0

    for frame_id, data_frame in data_frames:
        pred_frame = pred_frames.get(frame_id, {})
        gts = extract_poses(data_frame, "ground_truth", "person_id")
        preds = extract_poses(pred_frame, "predicted_poses", "pose_id")
        if unit_scale != 1.0:
            gts = [
                PoseRecord(gt.pose_id, [[coord * unit_scale for coord in joint] for joint in gt.joints_3d], gt.score)
                for gt in gts
            ]
            preds = [
                PoseRecord(
                    pred.pose_id,
                    [[coord * unit_scale for coord in joint] for joint in pred.joints_3d],
                    pred.score,
                )
                for pred in preds
            ]

        all_frame_gts.append(gts)
        all_frame_preds.append(preds)
        total_gt += len(gts)
        total_predictions += len(preds)

        distances = distance_matrix(preds, gts)
        for _, _, dist in greedy_min_error_matches(distances):
            mpjpe_values.append(dist)
        recall_matches += len(greedy_min_error_matches(distances, max_error=recall_threshold_mm))

    metrics: dict[str, Any] = {
        "mpjpe_mm": (sum(mpjpe_values) / len(mpjpe_values)) if mpjpe_values else None,
        "recall_500mm": float(100.0 * recall_matches / total_gt) if total_gt else 0.0,
        "num_frames": len(data_frames),
        "num_ground_truth": total_gt,
        "num_predictions": total_predictions,
        "num_matched_for_mpjpe": len(mpjpe_values),
    }

    for threshold in thresholds_mm:
        metrics[f"ap_{int(threshold)}mm"] = 100.0 * average_precision_for_threshold(
            all_frame_preds,
            all_frame_gts,
            float(threshold),
        )
    return metrics

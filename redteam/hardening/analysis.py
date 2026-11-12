#!/usr/bin/env python3
"""Noise checks on the hidden test (audit only; writes aggregate statistics only).

1. Paired bootstrap over test frames: every method is scored on the same
   resampled frames, and the per-metric difference to the reference is
   summarised (mean and 95% interval).
2. Detector-noise seeds: the hidden scenes are regenerated with two other
   noise/rotation seeds and the methods are rescored, so the reference-vs-
   variant ordering can be compared with the seed-to-seed spread.
No predictions or answers are written to disk.
"""

from __future__ import annotations

import json
from pathlib import Path
import random
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.test import scenes as hidden  # noqa: E402
from eval.test.runner import load_prediction_file, run_predictor, strip_answers  # noqa: E402
from eval.test.scoring import (  # noqa: E402
    METRIC_NAMES,
    InvalidPredictions,
    compute_metrics,
    frame_detection_counts,
    ground_truth_from_frames,
    validate_predictions,
)

SUB = ROOT / "redteam" / "hardening" / "submissions"
METHODS = {
    "compose": ROOT / "checkpoints" / "compose" / "predict.sh",
    "compose_robust": SUB / "compose_robust" / "predict.sh",
    "compose_pairs_only": SUB / "compose_pairs_only" / "predict.sh",
    "compose_three_cams": SUB / "compose_three_cams" / "predict.sh",
    "mvpose_geometric": ROOT / "checkpoints" / "mvpose_geometric" / "predict.sh",
    "pairwise_baseline": ROOT / "checkpoints" / "pairwise_baseline" / "predict.sh",
}
SEED_METHODS = ("compose", "compose_robust", "compose_pairs_only", "pairwise_baseline")
BOOTSTRAP = 500


def predictions_for(seed: str, names: tuple[str, ...]) -> tuple[dict, dict, list[str]]:
    hidden.HIDDEN_SEED = seed
    bench = hidden.build_test_scenes()
    inputs = strip_answers(bench)
    gt = ground_truth_from_frames(bench["frames"])
    preds = {}
    (ROOT / "tmp").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="audit_", dir=ROOT / "tmp") as tmp:
        ipath = Path(tmp) / "inputs.json"
        ipath.write_text(json.dumps(inputs))
        for name in names:
            out = Path(tmp) / f"{name}.json"
            state = run_predictor(METHODS[name], ipath, out, Path(tmp) / f"{name}.log", 3600)
            if state != "ok":
                print(f"{seed} {name}: {state}", flush=True)
                continue
            payload, state = load_prediction_file(out)
            try:
                preds[name] = validate_predictions(payload, frame_detection_counts(inputs))
            except InvalidPredictions as exc:
                print(f"{seed} {name}: invalid {exc}", flush=True)
    return preds, gt, list(gt)


def resample(d: dict, picks: list[str]) -> dict:
    return {f"{fid}#{k}": d.get(fid, []) for k, fid in enumerate(picks)}


def main() -> None:
    original_seed = hidden.HIDDEN_SEED
    report: dict = {"bootstrap": {}, "seeds": {}}
    preds, gt, frames = predictions_for(original_seed, tuple(METHODS))
    full = {name: compute_metrics(p, gt) for name, p in preds.items()}
    report["seeds"]["hidden"] = full
    rng = random.Random(0)
    if "compose" not in preds:
        raise SystemExit("reference predictions missing; cannot run paired bootstrap")
    draws = {name: {m: [] for m in METRIC_NAMES} for name in preds if name != "compose"}
    for _ in range(BOOTSTRAP):
        picks = [frames[rng.randrange(len(frames))] for _ in frames]
        gt_b = resample(gt, picks)
        ref = compute_metrics(resample(preds["compose"], picks), gt_b)
        for name in draws:
            other = compute_metrics(resample(preds[name], picks), gt_b)
            for m in METRIC_NAMES:
                draws[name][m].append(other[m] - ref[m])
    for name, per_metric in draws.items():
        report["bootstrap"][name] = {}
        for m, values in per_metric.items():
            values.sort()
            report["bootstrap"][name][m] = {
                "delta_vs_compose": full[name][m] - full["compose"][m],
                "ci95": [values[int(0.025 * len(values))], values[int(0.975 * len(values)) - 1]],
            }
    for k, seed in enumerate(("audit-seed-a", "audit-seed-b")):
        p, g, _ = predictions_for(seed, SEED_METHODS)
        report["seeds"][f"alt_{k}"] = {name: compute_metrics(v, g) for name, v in p.items()}
    hidden.HIDDEN_SEED = original_seed
    out = ROOT / "redteam" / "hardening" / "analysis.json"
    out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(json.dumps(report, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()

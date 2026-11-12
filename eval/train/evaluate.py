#!/usr/bin/env python3
"""Train-split evaluator: run every registered predictor on the public train
frames and score it with the paper's metrics.

Usage (from the workspace root):
    python eval/train/evaluate.py                 # every checkpoints/*/predict.sh
    python eval/train/evaluate.py --methods a b   # a subset

Writes ``scoring/scores_train.json`` and keeps each method's train predictions
and log under ``checkpoints/<name>/train_predictions.json`` / ``train_run.log``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.harness import (  # noqa: E402
    CHECKPOINTS,
    WORKSPACE,
    discover_predictors,
    load_prediction_file,
    run_predictor,
    strip_answers,
)
from eval.metrics import FAILURE_METRICS, ground_truth_from_frames, score_payload  # noqa: E402

EXPERIMENT = "cmu_panoptic_main"
DATA = WORKSPACE / "data" / "downloads" / "panoptic_train.json"
TIMEOUT_S = 3600


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--methods", nargs="*", default=None, help="Restrict to these registered methods")
    parser.add_argument("--submissions", default=str(CHECKPOINTS), help="Directory of <name>/predict.sh")
    parser.add_argument("--output", default=str(WORKSPACE / "scoring" / "scores_train.json"))
    args = parser.parse_args()

    benchmark = json.loads(DATA.read_text(encoding="utf-8"))
    inputs = strip_answers(benchmark)
    ground_truth = ground_truth_from_frames(benchmark["frames"])
    predictors = discover_predictors(Path(args.submissions))
    if args.methods:
        predictors = {name: path for name, path in predictors.items() if name in set(args.methods)}
    if not predictors:
        print(f"No predictors found under {args.submissions}/*/predict.sh", file=sys.stderr)
        return 1

    results, status = {}, {}
    (WORKSPACE / "tmp").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="eval_train_", dir=WORKSPACE / "tmp") as tmp:
        inputs_path = Path(tmp) / "inputs.json"
        inputs_path.write_text(json.dumps(inputs), encoding="utf-8")
        for name, script in predictors.items():
            out_path = script.parent / "train_predictions.json"
            state = run_predictor(script, inputs_path, out_path, script.parent / "train_run.log", TIMEOUT_S)
            metrics = dict(FAILURE_METRICS)
            if state == "ok":
                payload, state = load_prediction_file(out_path)
                if state == "ok":
                    metrics, state = score_payload(payload, inputs, ground_truth)
            results[name] = metrics
            status[name] = state
            print(f"[train] {name}: {state} {json.dumps(metrics, sort_keys=True)}", flush=True)

    output = {
        "slice": "train",
        "experiments": {EXPERIMENT: {"results": results}},
        "_status": status,
        "_split": {
            "frames": len(benchmark["frames"]),
            "people": sum(len(f["people"]) for f in benchmark["frames"]),
            "sequences": benchmark.get("generator", {}).get("sequences"),
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Score one method on the public train frames and merge it into a scores file.

    python eval/evaluate_predictions.py --method-name NAME
        runs checkpoints/NAME/predict.sh on the public inputs, then scores it;
    python eval/evaluate_predictions.py --method-name NAME --predictions FILE
        scores an existing prediction file for the public train frames.

Results are merged under ``experiments.cmu_panoptic_main.results.NAME`` of
``--output`` (default ``scoring/scores.json``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.harness import CHECKPOINTS, WORKSPACE, load_prediction_file, run_predictor, strip_answers  # noqa: E402
from eval.metrics import FAILURE_METRICS, ground_truth_from_frames, score_payload  # noqa: E402

EXPERIMENT = "cmu_panoptic_main"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method-name", required=True)
    parser.add_argument("--predictions", default=None, help="Existing prediction JSON (skip running the predictor)")
    parser.add_argument("--data", default=str(WORKSPACE / "data" / "downloads" / "panoptic_train.json"))
    parser.add_argument("--output", default=str(WORKSPACE / "scoring" / "scores.json"))
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()

    benchmark = json.loads(Path(args.data).read_text(encoding="utf-8"))
    inputs = strip_answers(benchmark)
    ground_truth = ground_truth_from_frames(benchmark["frames"])

    (WORKSPACE / "tmp").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="eval_public_", dir=WORKSPACE / "tmp") as tmp:
        if args.predictions:
            pred_path = Path(args.predictions)
            state = "ok" if pred_path.is_file() else "failed: prediction file not found"
        else:
            script = CHECKPOINTS / args.method_name / "predict.sh"
            inputs_path = Path(tmp) / "inputs.json"
            inputs_path.write_text(json.dumps(inputs), encoding="utf-8")
            pred_path = Path(tmp) / "predictions.json"
            state = run_predictor(script, inputs_path, pred_path, Path(tmp) / "run.log", args.timeout)
            if state != "ok":
                print((Path(tmp) / "run.log").read_text(encoding="utf-8")[-4000:], file=sys.stderr)
        metrics = dict(FAILURE_METRICS)
        if state == "ok":
            payload, state = load_prediction_file(pred_path)
            if state == "ok":
                metrics, state = score_payload(payload, inputs, ground_truth)

    out = Path(args.output)
    output = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else {}
    output.setdefault("experiments", {}).setdefault(EXPERIMENT, {}).setdefault("results", {})
    output["experiments"][EXPERIMENT]["results"][args.method_name] = metrics
    output.setdefault("_status", {})[args.method_name] = state
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # A failed run is a result (worst values), not an evaluator error.
    print(f"{args.method_name}: {state} {json.dumps(metrics, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

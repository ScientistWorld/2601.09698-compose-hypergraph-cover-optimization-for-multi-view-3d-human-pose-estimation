#!/usr/bin/env python3
"""Hidden test evaluator.

Builds the hidden test scenes (inputs and answers exist only here, at test
time), runs every registered predictor ``checkpoints/<name>/predict.sh`` on the
inputs-only file, validates each output and scores it with the paper's
metrics.  Runs with no arguments; writes ``scoring/scores_test.json``.

Nothing a predictor writes is trusted: every number is recomputed from the
raw predicted poses.  A predictor that crashes, times out, writes nothing or
writes an impossible value is a failed run and gets the worst value of every
metric.  The test size, cameras, thresholds and answers are fixed here.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.test.runner import (  # noqa: E402
    CHECKPOINTS,
    WORKSPACE,
    discover_predictors,
    load_prediction_file,
    run_predictor,
    strip_answers,
)
from eval.test.scenes import SOURCE, build_test_scenes  # noqa: E402
from eval.test.scoring import FAILURE_METRICS, ground_truth_from_frames, score_payload  # noqa: E402

EXPERIMENT = "cmu_panoptic_main"
TIMEOUT_S = 1800


def _set_mode(path: Path, mode: int) -> None:
    try:
        os.chmod(path, mode)
    except OSError:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--methods", nargs="*", default=None, help="Restrict to these registered methods")
    parser.add_argument("--submissions", default=str(CHECKPOINTS), help="Directory of <name>/predict.sh")
    parser.add_argument("--output", default=str(WORKSPACE / "scoring" / "scores_test.json"))
    args = parser.parse_args()

    if not os.access(SOURCE, os.R_OK):
        _set_mode(SOURCE, 0o644)
    original_mode = stat.S_IMODE(SOURCE.stat().st_mode) or 0o644
    benchmark = build_test_scenes()
    inputs = strip_answers(benchmark)
    ground_truth = ground_truth_from_frames(benchmark["frames"])
    del benchmark

    predictors = discover_predictors(Path(args.submissions))
    if args.methods:
        predictors = {name: path for name, path in predictors.items() if name in set(args.methods)}
    if not predictors:
        print(f"No predictors found under {args.submissions}/*/predict.sh", file=sys.stderr)
        return 1

    results, status = {}, {}
    # Staged inside the workspace: the container's private /tmp is a small tmpfs.
    (WORKSPACE / "tmp").mkdir(exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="eval_hidden_", dir=WORKSPACE / "tmp"))
    try:
        inputs_path = tmp / "inputs.json"
        inputs_path.write_text(json.dumps(inputs), encoding="utf-8")
        # Answers stay unreadable while untrusted predictor code runs.
        _set_mode(SOURCE, 0)
        for name, script in predictors.items():
            out_path = tmp / f"{name}.predictions.json"
            log_path = tmp / f"{name}.log"
            state = run_predictor(script, inputs_path, out_path, log_path, TIMEOUT_S)
            metrics = dict(FAILURE_METRICS)
            if state == "ok":
                payload, state = load_prediction_file(out_path)
                if state == "ok":
                    metrics, state = score_payload(payload, inputs, ground_truth)
            else:
                tail = log_path.read_text(encoding="utf-8", errors="replace")[-1500:] if log_path.exists() else ""
                print(f"[test] {name} log tail:\n{tail}", flush=True)
            if out_path.exists():
                out_path.unlink()
            results[name] = metrics
            status[name] = state
            print(f"[test] {name}: {state} {json.dumps(metrics, sort_keys=True)}", flush=True)
    finally:
        _set_mode(SOURCE, original_mode)
        shutil.rmtree(tmp, ignore_errors=True)

    output = {
        "slice": "test",
        "experiments": {EXPERIMENT: {"results": results}},
        "_status": status,
        "_split": {"frames": len(inputs["frames"]), "people": sum(len(v) for v in ground_truth.values())},
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

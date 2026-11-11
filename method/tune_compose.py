#!/usr/bin/env python3
"""Select COMPOSE hyperparameters (tau, lambda, gamma, s_single) on the train frames.

The paper tunes hyperparameters on frames outside the test set.  Hyperedge
geometry is computed once at the largest tau; smaller taus reuse it through
``restrict_candidates`` (identical to regenerating at that tau).  The
selection utility is the sum of the four APs and recall minus MPJPE (mm).

Writes the selected configuration and the full grid table.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.metrics import compute_metrics, ground_truth_from_frames  # noqa: E402
from method.compose import generate_candidates, parse_frame, predict_frame, restrict_candidates  # noqa: E402

GRID = {
    "tau": [1000.0, 2000.0, 4000.0, 8000.0],
    "lambda": [3e-4, 1e-3, 3e-3],
    "gamma": [0.6, 1.0, 1.3, 1.7],
    "singleton_score": [0.05, 0.3],
}


def utility(metrics: dict[str, float]) -> float:
    return (
        metrics["ap_25mm"] + metrics["ap_50mm"] + metrics["ap_100mm"] + metrics["ap_150mm"]
        + metrics["recall_500mm"] - metrics["mpjpe_mm"]
    )


def to_eval(poses: list[dict]) -> list[tuple[list[list[float]], float]]:
    return [((np.asarray(p["joints_3d"]) * 1000.0).tolist(), float(p["score"])) for p in poses]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/downloads/panoptic_train.json")
    parser.add_argument("--output", required=True, help="Selected config JSON")
    parser.add_argument("--table", default=None, help="Optional full grid table JSON")
    parser.add_argument("--max-edge-size", type=int, default=None)
    args = parser.parse_args()

    bench = json.loads(Path(args.data).read_text(encoding="utf-8"))
    num_joints = len(bench["joint_names"])
    frames = [parse_frame(f, num_joints) for f in bench["frames"]]
    ground_truth = ground_truth_from_frames(bench["frames"])
    start = time.time()
    cache = [generate_candidates(fr, max(GRID["tau"]), args.max_edge_size) for fr in frames]
    print(f"geometry cached for {len(frames)} frames in {time.time() - start:.1f}s; "
          f"{sum(map(len, cache))} candidates at tau={max(GRID['tau'])}", flush=True)

    table = []
    for tau in GRID["tau"]:
        restricted = [restrict_candidates(c, tau) for c in cache]
        for lam, gamma, single in itertools.product(GRID["lambda"], GRID["gamma"], GRID["singleton_score"]):
            config = {"tau": tau, "lambda": lam, "gamma": gamma, "singleton_score": single,
                      "max_edge_size": args.max_edge_size}
            t0 = time.time()
            preds = {fr.frame_id: to_eval(predict_frame(fr, config, cands)) for fr, cands in zip(frames, restricted)}
            metrics = compute_metrics(preds, ground_truth)
            table.append({"config": config, "train_metrics": metrics, "utility": utility(metrics)})
            print(f"{json.dumps(config)} U={utility(metrics):.2f} "
                  f"AP25={metrics['ap_25mm']:.2f} AP150={metrics['ap_150mm']:.2f} "
                  f"R={metrics['recall_500mm']:.2f} MPJPE={metrics['mpjpe_mm']:.2f} ({time.time() - t0:.1f}s)",
                  flush=True)
    best = max(table, key=lambda row: row["utility"])
    print("selected", json.dumps(best), flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(best["config"], indent=2) + "\n", encoding="utf-8")
    if args.table:
        Path(args.table).write_text(json.dumps({"grid": GRID, "rows": table, "selected": best}, indent=1) + "\n",
                                    encoding="utf-8")


if __name__ == "__main__":
    main()

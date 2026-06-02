# Progress

## What Works

- Paper analysis and briefing are complete.
- `method/compose.py` implements the paper's computational pipeline: candidate
  hyperedges, reprojection-error pruning, weighted exact-cover ILP, and final
  triangulation.
- `data/synthetic_compose_dataset.json` provides a tracked deterministic
  calibrated multi-camera benchmark with noisy/missing/outlier detections.
- `eval/` scores prediction artifacts without importing `method/`.
- `python validate.py` passes before job submission.

## Results

- No reproduced scores yet. A CPU smoke job is being submitted to run
  `scripts/reproduce.sh` and write `scoring/scores.json`.

## Remaining

- Confirm the full pipeline runs inside the compute container.
- Inspect `scoring/scores.json`; if COMPOSE is not clearly better than the
  pairwise baseline, tune pruning/ILP hyperparameters or dataset difficulty and
  resubmit.
- After `method_runs`, iterate toward `core_claim`.

## Issues

- The original CMU Panoptic/Shelf/Campus evaluation would require large video
  data and 2D detector resources. This first runnable environment uses a
  synthetic calibrated substitute to preserve the correspondence/reconstruction
  problem at lightweight scale.
- Local login-node Python lacks `numpy`; dependencies are installed through
  `environment/setup.sh` and verified by submitted jobs.

## Deviations from Paper

- Paper data: CMU Panoptic, Shelf, and Campus with 2D detections from ViTPose.
  Reproduction substitute: deterministic synthetic calibrated multi-view scenes
  with 2D pose detections, occlusions, noise, and outliers. Reason: lightweight
  first reproduction without large detector/data downloads; the method
  algorithm is not replaced.
- Paper detector: ViTPose top-down 2D keypoints. Reproduction substitute:
  generated 2D keypoints with controlled noise. Reason: the paper's core claim
  is the correspondence optimization after 2D detection, and this keeps that
  computational stage testable.

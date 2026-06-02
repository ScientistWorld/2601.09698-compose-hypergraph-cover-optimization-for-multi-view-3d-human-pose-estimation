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
- COMPOSE has run end to end inside the compute container on the synthetic
  benchmark and wrote `checkpoints/compose_predictions.json`.
- The full CPU reproduction job `faa8138c-d63` completed successfully:
  `scripts/reproduce.sh` regenerated predictions/scores and
  `python validate.py --compare` passed.

## Results

- Local baseline/evaluator smoke test after the first failed job produced a
  valid `pairwise_baseline` score on the synthetic benchmark. Strict AP was
  near zero and MPJPE was high, confirming the task is nontrivial for the
  pairwise baseline.
- Second CPU job results before validation cleanup:
  - `compose`: AP@50 25.86, AP@100 46.54, AP@150 72.68, recall@500 100.0,
    MPJPE 64.21 mm.
  - `pairwise_baseline`: AP@50 0.0, AP@100 0.0, AP@150 0.30, recall@500
    25.0, MPJPE 543.01 mm.
  These support the synthetic core claim, but final validation failed on
  scoring schema and must be rerun after patching.
- After cleanup, `python validate.py --compare` passes locally using the
  compute-generated prediction artifacts and regenerated `scoring/scores.json`.
  The final validated reproduction scores are:
  - `compose`: AP@100 46.54, AP@150 72.68, recall@500 100.0, MPJPE 64.21 mm.
  - `pairwise_baseline`: AP@100 0.0, AP@150 0.30, recall@500 25.0, MPJPE
    543.01 mm.
  These support the core claim on the lightweight synthetic substitute.

## Remaining

- Expand beyond the minimum by adding another synthetic condition or a real
  lightweight benchmark subset if time remains.
- Higher milestones remain open: reproduce additional dataset conditions,
  compare more baselines, and move from synthetic detections to a real
  multi-view benchmark subset.

## Issues

- The original CMU Panoptic/Shelf/Campus evaluation would require large video
  data and 2D detector resources. This first runnable environment uses a
  synthetic calibrated substitute to preserve the correspondence/reconstruction
  problem at lightweight scale.
- Local login-node Python lacks `numpy`; dependencies are installed through
  `environment/setup.sh` and verified by submitted jobs.
- First CPU smoke job failed before running COMPOSE because
  `baseline/pairwise_baseline.py` iterated the tracked `scenes` data schema as
  generic dictionary entries. Fixed by normalizing top-level scenes/cameras into
  frame views and by reading named 2D joints.
- Second CPU smoke job ran COMPOSE successfully but failed `validate.py` because
  scores included bookkeeping fields and a reproduction-only baseline key absent
  from `reference.json`. Fixed by emitting only declared metrics and adding the
  pairwise baseline key as a validation reference.

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

# Progress

## What Works

- Continuing-agent audit at 2026-06-02 04:15 reconfirmed the workspace after
  the milestone reset and restored `core_claim`.
- Continuing-agent audit on 2026-06-02 found the existing reproduction
  trustworthy enough to restore `core_claim` from the reset `method_runs`
  state.
- Paper analysis and briefing are complete.
- `method/compose.py` implements the paper's computational pipeline: candidate
  hyperedges, reprojection-error pruning, weighted exact-cover ILP, and final
  triangulation.
- `data/panoptic_sample_compose.py` builds a lightweight CMU Panoptic-derived
  benchmark from official sample calibration and 3D skeleton frames.
- `data/synthetic_compose_dataset.json` provides a tracked deterministic
  calibrated multi-camera benchmark with noisy/missing/outlier detections.
- `eval/` scores prediction artifacts without importing `method/`.
- `python validate.py` passes before job submission.
- COMPOSE has run end to end inside the compute container on the synthetic
  benchmark and wrote `checkpoints/compose_predictions.json`.
- The full CPU reproduction job `faa8138c-d63` completed successfully:
  `scripts/reproduce.sh` regenerated predictions/scores and
  `python validate.py --compare` passed.
- Current milestone is `core_claim`: the proposed hypergraph exact-cover
  method beats the pairwise association baseline on the scored nontrivial
  synthetic benchmark while maintaining full recovery recall.
- Audit conclusion: the implementation, evaluator, scripts, environment, and
  score contract are complete enough for a reusable core-claim gym.
- 2026-06-02 04:36 update: scripts now default to
  `data/downloads/cmu_panoptic_sample_compose.json`, generated from the
  official CMU Panoptic sample assets by `scripts/download.sh`.
- CPU verification job `265894a0-2ac` completed successfully on the
  Panoptic-derived benchmark: `scripts/reproduce.sh` regenerated method and
  baseline predictions, train/test scoring used `panoptic_sample_*` scene IDs,
  and `python validate.py --compare` passed.

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
  - `compose`: AP@150 72.68, AP@100 46.54, recall@500 100.0, MPJPE 64.21 mm.
  - `pairwise_baseline`: AP@150 0.30, AP@100 0.0, recall@500 25.0, MPJPE
    543.01 mm.
  These support the core claim on the lightweight synthetic substitute.
- Final Panoptic-derived benchmark scores from job `265894a0-2ac`:
  - `compose`: AP@150 83.57, AP@100 63.99, recall@500 100.0, MPJPE 69.26 mm.
  - `pairwise_baseline`: AP@150 12.47, AP@100 7.94, recall@500 34.03, MPJPE
    443.72 mm.
  Train/test slices are comparable: COMPOSE AP@150 is 82.59 on train and 84.82
  on held-out test.

## Remaining

- Expand beyond the minimum by adding Shelf/Campus-style settings or another
  real-data subset if more budget is available.
- Higher milestones remain open. The final verified gym intentionally scores
  only `cmu_panoptic_main`, because the sparse-view stress extension drafted in
  a later iteration was not completed by a job and is not part of
  `scoring/scores.json`.

## Issues

- 2026-06-02 04:15 audit found no new correctness issues requiring code
  changes.
- Audit note: strict `ap_25mm` is zero on the lightweight substitute, but this
  is consistent with the intentionally noisy synthetic detections and is not
  the selected primary metric. The primary AP, recall, and MPJPE metrics still
  show a large COMPOSE advantage over the pairwise baseline.
- The original CMU Panoptic/Shelf/Campus evaluation would require large video
  data and 2D detector resources. This first runnable environment uses a
  synthetic calibrated substitute to preserve the correspondence/reconstruction
  problem at lightweight scale.
- The upgraded default benchmark uses official CMU Panoptic sample camera
  calibration and skeleton poses, then deterministically creates multi-person
  scenes because the public sample contains only one annotated body per frame.
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
- Final verification issue about stale synthetic score artifacts was resolved
  by job `265894a0-2ac`, which regenerated scores on the current
  Panoptic-derived default dataset.
- Final verification removed the unverified sparse-view stress extension from
  the standard run path and experiment overview so documented experiments match
  the validated score artifacts.

## Deviations from Paper

- Paper data: CMU Panoptic, Shelf, and Campus with 2D detections from ViTPose.
  Reproduction substitute: deterministic synthetic calibrated multi-view scenes
  with 2D pose detections, occlusions, noise, and outliers. Reason: lightweight
  first reproduction without large detector/data downloads; the method
  algorithm is not replaced.
- Current default data: official CMU Panoptic sample calibration and skeleton
  frames with deterministic translated people, projected detections, occlusion,
  noise, and outliers. Reason: the public sample is small and has real
  calibration/pose geometry, but does not include the paper's full multi-person
  benchmark videos or detector outputs.
- Paper detector: ViTPose top-down 2D keypoints. Reproduction substitute:
  generated 2D keypoints with controlled noise. Reason: the paper's core claim
  is the correspondence optimization after 2D detection, and this keeps that
  computational stage testable.

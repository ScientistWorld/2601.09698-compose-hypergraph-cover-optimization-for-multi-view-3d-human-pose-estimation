> **Superseded (iteration 11):** this note describes the pre-hardening gym (single-body Panoptic sample, visible answers). The current design and its verification are in `HARDENING.md`.

# Train/Test Evaluation Design Notes

## Checks Performed

- Read `eval/train/evaluate.py` and `eval/test/evaluate.py` side by side for copy-paste errors, output paths, slice markers, and method-artifact usage.
- Confirmed the split evaluators score precomputed prediction JSON files and do not regenerate the paper method inline.
- Checked that `scripts/evaluate_train.sh` writes `scoring/scores_train.json` and invokes `eval/train/`, while `scripts/evaluate_test.sh` writes `scoring/scores_test.json` and invokes `eval/test/`.
- Checked for train/test cross-import patterns and paper-agreement metric names.
- Confirmed `scoring/scores_train.json` and `scoring/scores_test.json` do not exist.
- Ran static validation only: `bash -n` on both split scripts and JSON parsing for `scoring/reference.json`.

## Changes Made

- Replaced the original even/odd temporal frame split with a deterministic hash-stratified split in both train and test evaluators.
  - The split now balances crossing-phase and non-crossing-phase frames in each slice.
  - This reduces shortcut value from tuning to visible frames whose adjacent held-out frames are nearly identical.
- Added metric triangulation in `eval/metrics.py`:
  - `precision_150mm` catches methods that improve recall by emitting too many poses.
  - `pose_count_mae` catches per-frame over- or under-production of people.
  - `gt_copy_rate` catches exact ground-truth label-copy shortcuts from the visible benchmark JSON.
- Updated `scoring/reference.json` so every metric has `higher_is_better`, `coefficient`, and `description`.
- Kept the split scripts method-discovery behavior: no-argument train evaluation enumerates `checkpoints/*_predictions.json`, and no-argument test evaluation re-scores methods recorded in `scores_train.json` or falls back to checkpoint discovery.

## Remaining Runtime Concerns

- The benchmark data file includes ground-truth poses because the original reproduction dataset is tracked. `gt_copy_rate` should expose exact label-copying, but it will not catch a method that reads ground truth and adds tiny noise. A stronger future iteration would need a truly hidden test data file or a stricter leakage policy around method inputs.
- Runtime verification job `0033bd13-442` produced complete train/test scores for `compose` and `pairwise_baseline` with 12 disjoint frames per slice and matching method sets. The primary `compose` AP@150mm is 78.95 on train and 70.09 on test; precision@150mm is 45.61 on train and 44.44 on test.
- `ap_25mm` saturated at 0.0 for every scored method on both slices, so it remains reported but now has zero scoring coefficient in `scoring/reference.json`. The usable benefit signal comes from AP@50/100/150, precision@150, recall@500, and MPJPE.
- `gt_copy_rate` uses a near-exact MPJPE threshold of `1e-9` after unit scaling. This is intended as a label-copy sentinel, not a general accuracy metric.
- `precision_150mm` and `pose_count_mae` are expected to be useful anti-shortcut constraints, but the next runner should watch for saturation if all submitted methods naturally emit exactly two poses per frame on this synthetic benchmark.

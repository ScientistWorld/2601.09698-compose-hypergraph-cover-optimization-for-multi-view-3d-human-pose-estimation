# Evaluation

## Metrics

Predicted 3D poses are greedily matched to ground-truth poses per frame by
minimum mean per-joint position error (MPJPE). The benchmark reports:

- `ap_25mm`, `ap_50mm`, `ap_100mm`, `ap_150mm`: pose average precision at the
  corresponding MPJPE threshold; higher is better. This gym uses `ap_150mm` as
  the primary target because it is informative on the noisy lightweight
  substitute benchmark and remains comparable in scale to the paper's reported
  AP values; stricter AP thresholds remain reported diagnostic metrics.
- `recall_500mm`: fraction of ground-truth people matched by any prediction
  within 500 mm; higher is better.
- `mpjpe_mm`: mean matched per-joint error in millimeters; lower is better.

## Baselines and Targets

The paper's main reported optimization-based baseline on CMU Panoptic is MvPose
using the same 2D keypoint detector. The benchmark should reward higher AP and
recall, lower MPJPE, and a reasonable number of emitted poses. The runnable
environment also includes a simple pairwise triangulation baseline to verify
that cross-view association remains nontrivial on the lightweight benchmark.

## Evaluation Protocol

Run a method to write predictions as JSON, then call `scripts/evaluate.sh`.
Evaluation code in `eval/` reads only prediction artifacts and the benchmark
data; it does not import or rerun method code. The paper evaluates CMU Panoptic
with five cameras and reports AP, recall, and MPJPE. This environment uses a
deterministic CMU Panoptic sample-derived benchmark with official camera
calibration and skeleton poses, plus generated noisy/missing/spurious 2D
detections, to preserve the same output and metric structure at lightweight
scale.

# Evaluation Targets

All targets are measured on the hidden test split of `cmu_panoptic_main` by
`bash scripts/evaluate_test.sh` (train-split counterparts:
`bash scripts/evaluate_train.sh`).  Coefficients are in `scoring/reference.json`.

## Primary: cmu_panoptic_main
Correct, separated 3D poses for every person.
- **Metrics**: `ap_150mm` (headline), `ap_25mm` and `ap_50mm` (strict
  localisation, where the paper's gain is largest), `ap_100mm` — all higher is
  better.

## Accuracy: cmu_panoptic_main
- **Metric**: `mpjpe_mm` (lower is better) of the matched poses.

## Constraint: cmu_panoptic_main
Nobody may be left out, and the output must be a valid set of complete poses.
- **Metric**: `recall_500mm` (higher is better).
- **Validity** (failed run = worst value of every metric): every input frame
  answered, at most one pose per 2D detection in a frame, exactly 15 finite
  joints inside the capture volume and a finite score per pose, predictor
  finishes within the time limit.

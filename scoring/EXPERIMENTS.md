# Experiment Overview

## cmu_panoptic_main

Establishes whether a system can turn noisy, unlabeled 2D pose detections from
five calibrated cameras into correct, complete and accurate 3D poses of every
person in real multi-person scenes it has never seen.  The scenes are the
standard CMU Panoptic test sequences (2 to 6 interacting people, frequent
mutual occlusion), so mistakes in deciding which detections belong together
show up directly as missing people, merged or duplicated people, and large
joint errors.

The measurement follows the paper's benchmark protocol: average precision at
25, 50, 100 and 150 mm pose error (strict to loose correctness of each output
pose, penalising duplicates and spurious poses), recall at 500 mm (nobody is
left out — the completeness constraint), and mean per-joint error of the
matched poses.  Scores are computed on hidden test frames by running the
submitted predictor; invalid or failed runs receive the worst value of every
metric.

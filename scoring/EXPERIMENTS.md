# Experiment Overview

## cmu_panoptic_main
Tests the central multi-view multi-person 3D pose estimation task under the
paper's main metric family. The experiment asks whether a solver can recover
accurate poses for all people from sparse calibrated camera observations while
handling missed, noisy, and spurious 2D detections.

## panoptic_sample_3view_stress
Extension setting for `core_claim_plus`. It uses the same official CMU
Panoptic sample calibration source, but keeps only three cameras and increases
the number of people, detector noise, misses, and outliers. This asks whether
the correspondence solver still improves over a pairwise association baseline
when camera coverage is sparse and ambiguity is higher.

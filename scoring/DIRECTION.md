# Research Direction

## Research Problem
The benchmark studies 3D multi-person pose recovery from sparse calibrated
multi-view observations when each camera provides noisy, incomplete 2D pose
detections. The central question is how to establish cross-view correspondences
robustly enough to triangulate accurate 3D poses without 3D supervised
training.

## Core Contribution
Work should improve cross-view association and 3D reconstruction from
calibrated 2D detections without using 3D supervised training labels at
prediction time. The relevant family is geometry-aware multi-view reasoning,
not fully supervised image-to-3D regression.

## Approach Scope
Improvements may change how detections are compared, grouped, scored,
reconstructed, filtered, and calibrated. The method must still solve the same
multi-view association problem from the provided detections and camera
calibration.

## Out of Bounds
Replacing the task with supervised 3D pose regression, using ground-truth person
IDs, training a model to infer identities from the synthetic labels, or adding
new image-level detectors/features for the scored experiment is out of scope.

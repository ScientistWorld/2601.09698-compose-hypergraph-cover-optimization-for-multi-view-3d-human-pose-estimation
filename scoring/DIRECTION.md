# Research Direction

## Research Problem
The benchmark studies 3D multi-person pose recovery from sparse calibrated
multi-view observations when each camera provides noisy, incomplete 2D pose
detections. The central question is how to establish cross-view correspondences
robustly enough to triangulate accurate 3D poses without 3D supervised
training.

## Core Contribution
Work should improve the multi-view correspondence and geometric reconstruction
stage. The relevant family is optimization- or geometry-driven association of
2D detections across cameras, followed by 3D pose reconstruction.

## Approach Scope
Improvements may change candidate generation, geometric scoring, pruning,
global assignment objectives, solver strategy, confidence calibration, and
handling of missing or spurious detections. The method must still solve the
same multi-view association problem from the provided detections and camera
calibration.

## Out of Bounds
Replacing the task with supervised 3D pose regression, using ground-truth person
IDs, training a model to infer identities from the synthetic labels, or adding
new image-level detectors/features for the scored experiment is out of scope.

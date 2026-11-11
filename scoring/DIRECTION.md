# Research Direction

## Research Problem
3D multi-person pose recovery from sparse calibrated multi-view 2D detections
in crowded, occluded scenes.  The central difficulty is establishing which
detections across views belong to the same person — robustly to noisy joints,
occlusion, flips, missed and spurious detections — and then reconstructing
accurate 3D joints without 3D-supervised training.

## Core Contribution
Better cross-view association and reconstruction from calibrated 2D
detections: higher strict-threshold precision and lower joint error while
recovering every person, on unseen scenes.

## Approach Scope
Improvements may change how detections are compared, how correspondence
hypotheses are formed, scored, pruned and selected, how joints are
triangulated or refined, how detection and joint confidences are used, and how
output poses are scored for ranking.  Any computation must run from the
supplied cameras and detections at test time within the time limit.

## Out of Bounds
Using 3D annotations or identities at prediction time, regressing 3D poses with
a model trained on the 3D answers, replacing or re-simulating the 2D
detections, or modifying the evaluators.

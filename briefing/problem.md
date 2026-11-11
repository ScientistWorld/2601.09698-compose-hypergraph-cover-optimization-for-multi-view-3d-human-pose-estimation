# Problem Definition

## Research Question

How can a system recover accurate 3D poses for multiple people from a small set
of synchronized, calibrated cameras when each view supplies noisy, occluded,
incomplete and partly spurious 2D human pose detections, with no identity
information linking detections across views?

## Why It Matters

Reliable multi-view 3D pose estimation supports safety-critical monitoring,
human-robot interaction, sports analysis and activity understanding in scenes
where monocular depth ambiguity or occlusion make single-camera estimates
unreliable.  3D annotations are expensive, so methods that work from 2D
detections and camera geometry, without 3D supervision, are attractive.

## Success Criteria

A successful system decides which 2D detections across the cameras belong to
the same physical person (a correspondence problem whose number of hypotheses
grows exponentially with the number of views), rejects spurious detections,
reconstructs each person's 15 body joints in the world frame, and ranks its
outputs by confidence.  It should recover every person (recall), separate
people without duplicates (precision at strict and loose distance thresholds),
and localise joints accurately (MPJPE), on scenes it has never seen.

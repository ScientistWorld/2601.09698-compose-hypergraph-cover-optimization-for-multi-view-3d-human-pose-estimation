# Problem Definition

## Research Question

How can a system recover accurate 3D poses for multiple people from a small set
of synchronized calibrated RGB cameras when each view supplies noisy,
occluded, and ambiguous 2D human pose detections?

## Why It Matters

Reliable multi-view 3D pose estimation supports safety-critical monitoring,
robot-human interaction, sports analysis, and activity understanding in scenes
where monocular depth ambiguity or occlusion makes single-camera estimates
unreliable.

## Success Criteria

A successful system should group 2D observations that belong to the same
physical person across cameras, reconstruct each person's 3D joints in the
global camera coordinate frame, tolerate missed or spurious 2D detections, and
avoid using ground-truth identities or 3D poses when forming predictions.

# Problem Definition

What problem does this paper address? Write this so that someone with no knowledge of the paper can understand what needs to be solved and how success is measured.

Do NOT describe the paper's method or solution here — only the problem itself. Think about if a future scientist agent try to solve this problem from scratch without knowing what approach the paper took. If you leak the solution, you compromise the benchmark.

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

A successful solver should associate detections belonging to the same physical
person across cameras, reconstruct each person's 3D joints in the global camera
coordinate frame, tolerate missed or spurious 2D detections, and avoid requiring
3D supervision at evaluation time.

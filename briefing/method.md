# Paper's Method

How the paper approaches the problem, i.e. the paper's core contributions.

## Key Contribution

COMPOSE formulates multi-view 2D pose correspondence as a hypergraph
partitioning problem. Candidate hyperedges represent sets of detections across
multiple camera views that could belong to one 3D person.

## Approach

The pipeline first obtains 2D poses independently in each view. It then
enumerates multi-view detection hypotheses, triangulates each hypothesis with
DLT, scores it by reprojection consistency through an exponential kernel, and
prunes hypotheses above a geometric error threshold. The remaining hyperedges,
plus singleton fallback edges for unmatched detections, are selected with a
weighted exact-cover integer linear program. Selected multi-view hyperedges are
triangulated into final 3D poses.

## Main Claims

The primary claim is that higher-order multi-view association improves 3D pose
accuracy over pairwise optimization-based matching, especially under ambiguity
and occlusion, without requiring 3D supervision. Secondary claims are that
geometric pruning makes the ILP tractable and that the method generalizes across
standard calibrated multi-view datasets.

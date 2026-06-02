# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-02 00:00] - core_claim
- Implemented COMPOSE as the paper's actual higher-order method: geometrically pruned hyperedge construction, DLT reprojection scoring, weighted exact-cover ILP, and triangulated 3D output.
- Ran the full reproduction pipeline in the compute container on a nontrivial synthetic calibrated multi-view benchmark.
- Validated that COMPOSE substantially outperforms the pairwise triangulation baseline on AP@150, recall@500, and MPJPE.

## Stop Justification
- Completed at milestone `core_claim`.

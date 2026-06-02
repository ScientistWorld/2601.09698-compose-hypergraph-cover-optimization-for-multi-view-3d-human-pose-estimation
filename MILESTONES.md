# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-02 00:00] - method_runs
- Implemented COMPOSE's actual hypergraph formulation: DLT triangulation for candidate hyperedges, reprojection-error pruning, exponential hyperedge scoring, and weighted exact-cover ILP with singleton detections.
- Added a deterministic calibrated multi-view synthetic benchmark with noisy, missing, and outlier 2D detections.
- Added a pairwise association baseline and artifact-based evaluator that does not import from `method/`.

### [2026-06-02 00:00] - core_claim
- Ran the packaged reproduction on the synthetic substitute benchmark.
- COMPOSE substantially outperformed the pairwise baseline on AP@150, recall@500, and MPJPE while preserving the paper's optimization pipeline.
- `python validate.py --compare` passes with the reproduced score artifacts.

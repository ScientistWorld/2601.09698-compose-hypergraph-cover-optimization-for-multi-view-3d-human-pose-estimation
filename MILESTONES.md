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

### [2026-06-02 04:15] - core_claim
- Continuing-agent audit confirmed `method/compose.py` implements the paper's actual hypergraph exact-cover algorithm rather than a surrogate.
- Confirmed `eval/` is artifact-based and independent from `method/`; `scripts/evaluate.sh` works through `eval/evaluate_predictions.py`.
- Re-ran `python validate.py --compare`; scoring and structural checks pass.

### [2026-06-02 04:36] - core_claim
- Added a CMU Panoptic sample-derived benchmark builder using official calibration and skeleton assets.
- Updated default scripts to regenerate and score the Panoptic-derived benchmark instead of only rescoring existing synthetic artifacts.
- Submitting a CPU smoke job to validate the upgraded benchmark path before claiming any higher milestone.

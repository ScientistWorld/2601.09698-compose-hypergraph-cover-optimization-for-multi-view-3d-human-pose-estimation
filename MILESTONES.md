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

### [2026-06-02 04:40] - core_claim
- CPU verification job `265894a0-2ac` regenerated predictions and scores on the Panoptic-derived benchmark.
- COMPOSE achieved AP@150 83.57, recall@500 100.0, and MPJPE 69.26 mm versus the pairwise baseline AP@150 12.47, recall@500 34.03, and MPJPE 443.72 mm.
- `python validate.py --compare` passed after the regenerated train/test score artifacts used `panoptic_sample_*` scene IDs.

### [2026-06-02 04:55] - core_claim
- Final verification kept the milestone at `core_claim`.
- Removed an unverified sparse-view stress extension from the standard run path and method-agnostic experiment overview so documented experiments match `scoring/scores.json`.
- Re-ran `python validate.py --compare`; scoring and structural checks pass.

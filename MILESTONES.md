# Reproduction Milestones

**Current: method_runs**

<!-- Milestone levels (update "Current" above as you progress):
  none             — just started, no meaningful progress yet
  method_runs      — the paper's method executes end-to-end without errors
  core_claim       — minimum experiment supports the paper's central claim
  core_claim_plus  — core claim reproduced on additional settings
  secondary_claims — secondary results or contributions reproduced
  majority         — more than half of reported results reproduced
  near_complete    — most results reproduced, only minor gaps remain
  full             — all reported results reproduced
-->

## Progress Log

### [2026-06-02 03:58] - none
- Read the paper and identified COMPOSE as a hypergraph weighted exact-cover method for multi-view 3D human pose correspondence.
- Implemented the actual method structure in `method/compose.py`: DLT triangulation, reprojection-cost scoring, geometric pruning, singleton fallback edges, and PuLP ILP exact cover.
- Added a deterministic synthetic calibrated multi-view benchmark and reusable evaluator; static validation passes.
- Submitted a CPU smoke job to verify the packaged pipeline end to end before claiming `method_runs`.

### [2026-06-02 04:08] - none
- First CPU smoke job failed in the pairwise baseline due to a schema mismatch with the tracked `scenes` dataset.
- Patched baseline and evaluator schema normalization and verified the baseline/evaluator path locally.
- Resubmitting the CPU smoke job; no milestone advance until COMPOSE runs in the compute container.

### [2026-06-02 04:15] - method_runs
- Second CPU job ran the COMPOSE implementation end to end on the synthetic benchmark and wrote prediction artifacts.
- COMPOSE produced substantially better synthetic metrics than the pairwise baseline, but the job failed final validation because `scores.json` contained bookkeeping fields not declared in `reference.json`.
- Patched the scorer to emit only declared scientific metrics and resubmitting before claiming `core_claim`.

### [2026-06-02 04:24] - method_runs
- Regenerated `scoring/scores.json` from the compute-produced prediction artifacts after the scoring contract fix.
- Local `python validate.py --compare` now passes and shows COMPOSE ahead of the pairwise baseline on AP@50/100/150, recall@500, and MPJPE.
- Resubmitting the full CPU reproduction script so the patched environment is validated inside the compute container before advancing to `core_claim`.

## Stop Justification
- Not stopped; full post-fix validation job pending.

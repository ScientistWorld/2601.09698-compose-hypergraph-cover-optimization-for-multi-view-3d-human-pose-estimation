# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-02 00:00] - core_claim
- Continued-agent audit restored the milestone after reviewing `method/`,
  `eval/`, scripts, scoring files, and briefing deliverables.
- Confirmed the method is not a surrogate: it implements hyperedge
  enumeration, DLT reprojection scoring/pruning, weighted exact-cover ILP, and
  final triangulation.
- Re-ran `python validate.py --compare`; scoring and import-separation checks
  passed, with COMPOSE outperforming the pairwise baseline on the tracked
  synthetic benchmark.

### [2026-06-02 00:00] - core_claim
- Implemented COMPOSE as the paper's actual higher-order method: geometrically pruned hyperedge construction, DLT reprojection scoring, weighted exact-cover ILP, and triangulated 3D output.
- Ran the full reproduction pipeline in the compute container on a nontrivial synthetic calibrated multi-view benchmark.
- Validated that COMPOSE substantially outperforms the pairwise triangulation baseline on AP@150, recall@500, and MPJPE.

## Stop Justification
- Completed at milestone `core_claim`.

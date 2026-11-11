# Overview

- **Paper ID:** 2601.09698
- **Title:** COMPOSE: Hypergraph Cover Optimization for Multi-view 3D Human Pose Estimation
- **Domain:** Computer Vision / Multi-view 3D Human Pose Estimation
- **TL;DR:** COMPOSE replaces pairwise multi-view correspondence with a geometrically pruned hypergraph exact-cover formulation for robust 3D multi-person pose recovery.

## Short Summary

The paper addresses 3D multi-person pose estimation from sparse calibrated
camera views. It treats candidate cross-view associations as hyperedges,
scores them by triangulation reprojection error, prunes implausible candidates,
and solves a weighted exact-cover ILP before triangulating the final 3D poses.
On CMU Panoptic, COMPOSE improves strict AP and MPJPE over the optimization
baseline MvPose while remaining training-free.

## Key Results

- CMU Panoptic: AP@25 improves from 37.63 for MvPose to 54.66 for COMPOSE.
- CMU Panoptic: MPJPE improves from 26.46 mm for MvPose to 23.62 mm.
- Campus: COMPOSE reports 97.3% average PCP, above the reproduced MvPose result of 95.8%.

---

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

---

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

---

# Evaluation

## Metrics

Predicted 3D poses are greedily matched to ground-truth poses per frame by
minimum mean per-joint position error (MPJPE). The benchmark reports:

- `ap_25mm`, `ap_50mm`, `ap_100mm`, `ap_150mm`: pose average precision at the
  corresponding MPJPE threshold; higher is better. This gym uses `ap_150mm` as
  the primary target because it is informative on the noisy lightweight
  substitute benchmark and remains comparable in scale to the paper's reported
  AP values; stricter AP thresholds remain reported diagnostic metrics.
- `recall_500mm`: fraction of ground-truth people matched by any prediction
  within 500 mm; higher is better.
- `mpjpe_mm`: mean matched per-joint error in millimeters; lower is better.

## Baselines and Targets

The paper's main reported optimization-based baseline on CMU Panoptic is MvPose
using the same 2D keypoint detector. The benchmark should reward higher AP and
recall, lower MPJPE, and a reasonable number of emitted poses. The runnable
environment also includes a simple pairwise triangulation baseline to verify
that cross-view association remains nontrivial on the lightweight benchmark.

## Evaluation Protocol

Run a method to write predictions as JSON, then call `scripts/evaluate.sh`.
Evaluation code in `eval/` reads only prediction artifacts and the benchmark
data; it does not import or rerun method code. The paper evaluates CMU Panoptic
with five cameras and reports AP, recall, and MPJPE. This environment uses a
deterministic CMU Panoptic sample-derived benchmark with official camera
calibration and skeleton poses, plus generated noisy/missing/spurious 2D
detections, to preserve the same output and metric structure at lightweight
scale.

---

# Reproduction Log

### Iteration 1: gpt-5.5
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 17m | **GPU**: 0.0h
- **Jobs**: 3 total (1 completed, 2 failed)

<details>
<summary>Progress Log</summary>

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

### [2026-06-02 04:32] - core_claim
- CPU retry job `faa8138c-d63` completed successfully with `scripts/reproduce.sh` followed by `python validate.py --compare`.
- COMPOSE improves over the pairwise baseline on the synthetic correspondence benchmark: AP@100 46.54 vs 0.0, AP@150 72.68 vs 0.30, recall@500 100.0 vs 25.0, and MPJPE 64.21 mm vs 543.01 mm.
- The reproduced result demonstrates the paper's core claim at lightweight scale: hypergraph exact-cover association plus triangulation recovers multi-person 3D poses better than local pairwise association from the same detections.

</details>

### Iteration 2: gpt-5.5
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 4m | **GPU**: 0.0h
- **Jobs**: 3 total (1 completed, 2 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 3: gpt-5.5
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 3m | **GPU**: 0.0h
- **Jobs**: 3 total (1 completed, 2 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 4: gpt-5.5 (designer_stage_1)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 5m | **GPU**: 0.0h
- **Jobs**: 3 total (1 completed, 2 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 5: gpt-5.5 (designer_stage_2)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 3m | **GPU**: 0.0h
- **Jobs**: 3 total (1 completed, 2 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 6: gpt-5.5 (designer_stage_3)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 1m | **GPU**: 0.0h
- **Jobs**: 4 total (2 completed, 2 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 7: gpt-5.5 (packager)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 1m | **GPU**: 0.0h
- **Jobs**: 4 total (2 completed, 2 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 8: gpt-5.5 (verifier)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 11m | **GPU**: 0.0h
- **Jobs**: 5 total (3 completed, 2 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 9: gpt-5.5 (verifier)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 7m | **GPU**: 0.0h
- **Jobs**: 5 total (3 completed, 2 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 10: gpt-6.1-sol (redteam)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 13m | **GPU**: 0.0h
- **Jobs**: 5 total (3 completed, 2 failed)

<details>
<summary>Progress Log</summary>

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

</details>


---

# Reproduction Milestones

**Current: core_claim**

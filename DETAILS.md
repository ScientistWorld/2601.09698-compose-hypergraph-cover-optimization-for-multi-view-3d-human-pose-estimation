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

## This Gym

Real multi-person CMU Panoptic scenes (standard train/test sequences, the
paper's five HD cameras, 15-joint protocol) with simulated 2D detections; the
test split is hidden and built at test time.  Methods are training-free
predictors (`checkpoints/<name>/predict.sh`) scored with the paper's Table 1
metrics.  Reference COMPOSE on the hidden test: AP25 63.35, AP50 96.41,
AP100 98.03, AP150 98.18, recall 99.63, MPJPE 22.41 mm.

---

# Problem Definition

## Research Question

How can a system recover accurate 3D poses for multiple people from a small set
of synchronized, calibrated cameras when each view supplies noisy, occluded,
incomplete and partly spurious 2D human pose detections, with no identity
information linking detections across views?

## Why It Matters

Reliable multi-view 3D pose estimation supports safety-critical monitoring,
human-robot interaction, sports analysis and activity understanding in scenes
where monocular depth ambiguity or occlusion make single-camera estimates
unreliable.  3D annotations are expensive, so methods that work from 2D
detections and camera geometry, without 3D supervision, are attractive.

## Success Criteria

A successful system decides which 2D detections across the cameras belong to
the same physical person (a correspondence problem whose number of hypotheses
grows exponentially with the number of views), rejects spurious detections,
reconstructs each person's 15 body joints in the world frame, and ranks its
outputs by confidence.  It should recover every person (recall), separate
people without duplicates (precision at strict and loose distance thresholds),
and localise joints accurately (MPJPE), on scenes it has never seen.

---

# Paper's Method

How the paper approaches the problem, i.e. the paper's core contributions.

## Key Contribution

COMPOSE formulates multi-view 2D pose correspondence as a hypergraph
partitioning problem.  Nodes are the 2D pose detections of all views; a
hyperedge is a set of detections from distinct views hypothesised to be one
person, so consistency is judged across all its views at once instead of by
reconciling pairwise matches.

## Approach

1. 2D poses are detected independently in each view.
2. Each candidate hyperedge is triangulated with closed-form DLT; its cost is
   the reprojection error summed over the hyperedge's views and averaged over
   the J joints, C(e) = 1/J * sum_j sum_{v in e} ||pi_v(y_j) - u_{v,j}||^2.
3. Geometric pruning: hyperedges with C(e) > tau are discarded, and hyperedges
   are only grown from surviving ones instead of instantiating every
   combination (the full set grows as prod_v (1 + n_v) - 1).
4. Score s(e) = exp(-lambda * C(e)); singleton hyperedges with a fixed prior
   score cover detections seen in one view only.
5. A weighted exact cover, max sum_e (s(e) - gamma) x_e subject to every
   detection being covered exactly once, is solved as an ILP (PuLP / CBC
   branch and bound); gamma penalises the number of selected hyperedges and
   so favours large hyperedges.
6. Selected multi-view hyperedges are triangulated into the final 3D poses.

Hyperparameters (tau, lambda, gamma, s_single) are tuned on non-test frames.
In this gym: `method/compose.py` (pipeline), `method/tune_compose.py`
(selection on the public train split), `method/compose_config.json` (selected
values), `scripts/method.sh` (registers `checkpoints/compose/predict.sh`).

## Main Claims

The primary claim is that higher-order multi-view association improves 3D pose
accuracy over pairwise optimization-based matching (MvPose), especially in
strict-threshold AP and MPJPE, without 3D supervision.  Secondary claims are
that geometric pruning makes the ILP tractable (about 23 ms per frame on
Panoptic) and that the method generalises to Shelf and Campus.

---

# Evaluation

## Data

Real multi-person scenes from the CMU Panoptic studio, observed by the five
calibrated 1920x1080 HD cameras of the standard multi-view protocol (nodes 3,
6, 12, 13, 23).  3D answers are the released Panoptic body annotations
(15 body joints: neck, nose, mid-hip, and left/right shoulder, elbow, wrist,
hip, knee, ankle; metres, Panoptic world frame, y pointing down).

Each view supplies 2D pose detections from a stochastic detector model applied
to the real annotations and cameras (the benchmark does not ship videos or an
image-based detector).  Detections have 15 keypoints `[u, v, confidence]` and a
detection `score`, and reproduce the usual failure modes: size-dependent and
heavy-tailed joint noise, inter-person occlusion (noisier joints, joints
attached to the occluding person, people dropped when mostly hidden),
truncation at image borders, whole-body left/right flips, duplicate partial
detections and person-shaped false positives.  Each camera has its own error
profile.  Detections carry no identity information.  People in each frame are
rotated about the vertical axis by a random angle.

* **Train split (public):** `data/downloads/panoptic_train.json`, 300 frames /
  1022 people from five sequences of the standard Panoptic training split,
  built by `scripts/download.sh`.  It includes the 3D answers.
* **Test split (hidden):** 400 frames / 1367 people from the four sequences of
  the standard Panoptic test split, built inside the hidden evaluator at test
  time with its own detector-noise and rotation draws.  Test inputs and answers
  are never on disk outside the evaluator.

## Submitting a method

A method is a directory `checkpoints/<name>/` with an entry point `predict.sh`:

    bash checkpoints/<name>/predict.sh INPUTS.json OUTPUT.json

It is run from the workspace root, without network, with a 30-minute limit per
split (1 CPU in the standard job).  It must work on any frames it is given and
must not read anything but `INPUTS.json`, its own code/configuration and public
data.  Any tuning or training happens beforehand, on the train split.
`INPUTS.json` (see `eval/harness.py`):

    {"joint_names": [...15], "units": "meters",
     "frames": [{"frame_id": "...",
                 "cameras": [{"camera_id", "image_size", "K", "R", "t"}],
                 "views": [{"camera_id": "...",
                            "detections": [{"keypoints": [[u, v, conf] x 15], "score": s}]}]}]}

with `x_cam = R @ X + t` and `[u, v, 1] ~ K @ x_cam`.  `OUTPUT.json`:

    {"frames": {frame_id: {"predicted_poses": [{"joints_3d": [[x, y, z] x 15], "score": s}]}}}

## Validity

A run is **failed** — and receives the worst value of every metric (AP 0,
recall 0, MPJPE 500 mm) — when the predictor crashes, times out, writes no or
unparsable output, omits an input frame, emits more poses in a frame than that
frame has 2D detections, or emits a pose without exactly 15 joints of 3 finite
coordinates (magnitude at most 10^6 m) or without a finite score.  A joint that
is finite but far from the person is not invalid — it is simply a large error.
Nothing else the predictor writes is read.

## Metrics (paper protocol)

Every predicted pose is compared with each annotated person of its frame by
MPJPE over that person's annotated joints and assigned to the closest one;
predictions are ranked by `score` over the whole split.

- `ap_25mm`, `ap_50mm`, `ap_100mm`, `ap_150mm` (higher is better): average
  precision where a prediction is a true positive if its MPJPE to its closest
  person is below the threshold and that person was not already claimed by a
  higher-scored prediction.  Duplicates and spurious poses are false positives.
- `recall_500mm` (higher is better): share of people that are the closest
  person of some prediction within 500 mm.
- `mpjpe_mm` (lower is better): mean MPJPE of the first prediction claiming
  each person within 500 mm (500 when nothing matches).

## Running

    bash scripts/evaluate_train.sh   # every checkpoints/*/predict.sh on train -> scoring/scores_train.json
    bash scripts/evaluate.sh NAME    # one method on train -> scoring/scores.json
    bash scripts/evaluate_test.sh    # hidden test (evaluation job only) -> scoring/scores_test.json

All scored numbers are recomputed by `eval/` from the raw predicted poses.

## Baselines

- `pairwise_baseline`: pairwise two-view reprojection matching with union-find
  grouping.
- `mvpose_geometric`: the paper's optimisation-based competitor family —
  pairwise epipolar affinities made cycle-consistent by convex multi-way
  matching (geometric cues only).

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

### Iteration 11: claude-opus-5-5 (hardener)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 38m | **GPU**: 0.0h
- **Jobs**: 8 total (6 completed, 2 failed)

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

### [2026-11-12 02:00] - core_claim (hardening)
- Benchmark moved to real multi-person CMU Panoptic scenes (paper's standard sequences and five cameras) with a hidden, test-time-generated test split and a predictor interface; strict validity with worst-value failures.
- Reference restored to the paper's cost, incremental pruning and train-split tuning: hidden-test AP25 63.35 / AP150 98.18 / recall 99.63 / MPJPE 22.41 mm (paper 54.66 / 99.17 / 99.83 / 23.62); beats the MvPose-style competitor on strict AP and the pairs-only ablation by 18.7 AP150.
- Better variant beats the reference beyond paired-bootstrap noise; every shortcut scores far below the reference (`HARDENING.md`).

</details>


---

# Reproduction Milestones

**Current: core_claim**

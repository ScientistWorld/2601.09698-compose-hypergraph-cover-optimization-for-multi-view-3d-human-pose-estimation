# Hardening report (iteration 11)

**Bottom line.** The gym was rebuilt around the paper's own data and protocol,
and its hidden test now rewards science.
- **Data.** Real multi-person CMU Panoptic scenes from the standard train and
  test sequences, seen by the paper's five HD cameras, scored with the paper's
  Table 1 metrics.
- **Hidden test.** Test inputs and answers exist only inside `eval/test/` at
  test time.
- **Validity.** Methods are training-free predictors run on unseen frames;
  invalid output scores as a failed run.
- **Reference.** Restored to the paper's cost and tuning protocol, it now lands
  near the paper's numbers.
- **Ranking.** A genuinely better variant beats the reference by more than the
  test noise. Every weaker variant and every shortcut scores below the
  reference. The paper's higher-order effect is now 18.7 AP150 instead of an
  exact tie.

All "after" numbers come from the unchanged no-argument `bash scripts/evaluate_test.sh` path:
- reference runs are in `scoring/scores_test.json`;
- variants and shortcuts were scored by the same evaluator from a separate
  submissions directory (`redteam/hardening/scores_test.json`).

Rank deltas follow the stated rule:
- the sum over the six scored metrics of |coefficient| × improvement over the
  reference in the better direction ÷ the metric's population std across the
  cohort;
- the "honest" cohort is the reference, baselines and variants; the "all"
  cohort adds every shortcut;
- details are in `redteam/hardening/rank.json`.

AP, recall in %, MPJPE in mm. Hidden test: 400 frames, 1367 people.

## 1. Do better results score higher?

| Method (hidden test) | AP25 | AP50 | AP100 | AP150 | R500 | MPJPE | Rank, honest | Rank, all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Better variant: same association, confidence-weighted per-joint DLT with worst-view rejection, view-count ranking | 85.13 | 98.10 | 99.08 | 99.24 | 99.63 | 18.12 | **+1.495** | +0.914 |
| **Reference COMPOSE** | 63.35 | 96.41 | 98.03 | 98.18 | 99.63 | 22.41 | 0 | 0 |
| MvPose-style geometric baseline (α tuned on train) | 60.26 | 94.28 | 98.03 | 98.65 | 99.41 | 23.13 | −0.274 | −0.154 |
| Pairs only (no hyperedge > 2 views) | 45.34 | 78.01 | 79.53 | 79.53 | 99.63 | 25.28 | −3.428 | −1.731 |
| Previous gym's operating point (view-mean cost, γ = 0.06) | 41.09 | 73.86 | 75.59 | 75.59 | 99.56 | 26.12 | −4.202 | −2.121 |
| Three cameras (1, 3, 5) | 21.91 | 74.20 | 89.22 | 89.94 | 92.25 | 35.62 | −4.990 | −2.526 |
| Simple pairwise baseline | 35.60 | 52.25 | 56.74 | 57.86 | 75.42 | 50.87 | −9.899 | −4.058 |
| Unpruned brute force (every hyperedge in the ILP) | 34.39 | 48.99 | 50.37 | 50.98 | 65.76 | 95.64 | −13.301 | −4.831 |

**Before, from REDTEAM on the old gym.**
- The genuine improvement scored +2.55.
- The weaker two-camera ablation scored **+0.43**, rewarded through precision and count error.
- Removing every higher-order edge tied the reference **exactly** (0).

**After.**
- The better variant is strictly higher on every AP and on MPJPE.
- Every weaker variant is negative, including the camera ablation (−4.99) and the higher-order ablation (−3.43).

**Noise.** Paired bootstrap over test frames (500 resamples; every method on the
same frames; `redteam/hardening/analysis.json`), Δ versus the reference with 95% CI:

| Δ vs reference | AP25 | AP50 | AP150 | R500 | MPJPE |
|---|---|---|---|---|---|
| Better variant | +21.8 [18.7, 24.3] | +1.7 [0.8, 2.4] | +1.06 [0.45, 1.58] | 0 [0, 0] | −4.29 [−5.17, −3.65] |
| MvPose-style | −3.1 [−5.7, −1.0] | −2.1 [−3.6, −0.7] | +0.47 [−0.55, 1.32] | −0.22 [−0.67, 0.08] | +0.72 [−0.46, 1.88] |
| Pairs only | −18.0 [−22.4, −14.6] | −18.4 [−20.2, −16.6] | −18.7 [−20.2, −16.9] | 0 | +2.87 [1.74, 3.80] |
| Three cameras | −41.4 [−45.0, −38.9] | −22.2 [−25.4, −19.4] | −8.2 [−10.0, −6.7] | −7.4 [−8.9, −6.0] | +13.2 [11.8, 14.8] |

**Detector-noise and rotation seeds.** The hidden scenes were regenerated with
two other seeds, and the ordering held on both:

| Seed | Reference (AP25 / AP150 / MPJPE) | Better variant | Pairs only |
|---|---|---|---|
| alt a | 73.0 / 98.5 / 20.5 | 88.5 / 99.3 / 16.5 | 48.1 / 77.2 / 24.5 |
| alt b | 76.2 / 99.1 / 20.1 | 89.6 / 99.4 / 16.2 | 51.6 / 79.2 / 23.9 |

The method gaps (≈ +13 to +22 AP25, −4 mm, ≈ −20 AP150) are far larger than
their seed-to-seed spread. Absolute AP25 does move between seeds (63 to 76),
because the per-camera detector profiles are redrawn. The test therefore fixes
one hidden seed, and all comparisons are paired. All methods are deterministic:
DLT, CBC and no RNG.

## 2. Major shortcuts: fix, and score before → after

| Shortcut (REDTEAM) | Before | Fix | After |
|---|---|---|---|
| Copy visible test answers + 0.01 mm jitter | **+3.229** | Test answers are only in `eval/test/` and are built at test time; inputs carry no 3D | Impossible. Copying public train answers onto test frames: **−36.46** |
| Rebuild answers from the public generator and source tar | **+3.229** | The hidden source (standard Panoptic test sequences) and its generator (own code, hidden seed, hidden per-frame rotation) live only in `eval/test/`; the public builder only makes train scenes from train sequences | Same as above: **−36.46** |
| Group detections by supplied `person_id` | **+2.670** | Detections carry no identity field | **−43.53** |
| Replace editable dataset truth | **+3.229** | The evaluator loads answers from `eval/test/` before any predictor runs; nothing in `data/` affects the test; answers are unreadable while predictor code runs | Read attempt blocked (`PermissionError`): **−43.53** |
| Partial poses (NaN on 9 joints) | **+0.157** | Each pose needs exactly 15 joints of finite coordinates; otherwise the run fails with the worst value of every metric | **−43.53** (failed) |
| Remove every higher-order edge | **0 (tie)** | Restored the paper's summed-over-views cost and tuning on train frames (the old γ = 0.06 made the ILP prefer pairs) | **−3.43**, AP150 −18.7 |
| Unregistered on-disk method omitted | tie / omitted | Methods are discovered from `checkpoints/*/predict.sh` on disk; no registration file is read | An empty method is scored: **−43.53** |
| Empty output (MPJPE null, skipped) | −4.18 | Every metric has a worst value; MPJPE is 500 mm when nothing matches | **−43.53** |
| All-NaN / infinite / null coordinates | −3.54 / crash with no artifact | Non-finite or non-numeric coordinates fail the run with worst values | **−43.53** each |
| Crash / missing artifact (no score written) | no entry | A crash, timeout, missing or unparsable file fails the run and still gets worst values | **−43.53** |
| Truncated joint list | −0.13 | Exactly 15 joints are required | **−43.53** |
| Forged metric fields | −4.93 | The evaluator reads only poses and scores | **−42.75** |
| Unpruned exhaustive search | −0.10 | Real crowded scenes (2 to 6 people, 5 views) | **−13.30** |
| New: pad frames with low-score plausible poses to inflate recall@500 | n/a | AP is score-ranked; padding cannot exceed the frame's detection count | **−5.30** (recall 99.85, AP150 78.0) |
| New: more poses than detections | n/a | At most one 3D pose per 2D detection in a frame | **−43.53** (failed) |
| New: coordinates ×100 (far away) | n/a | Valid but matched to nobody | **−43.53** |

Every shortcut, failed run or not, now scores far below the reference in both
cohorts. In the "all" cohort, failed runs sit at −12.2 and the best shortcut,
recall padding, at −1.67.

## 3. Shortcuts left open, and why

- **Public Panoptic annotations of the test sequences.**
  - The standard test sequences are public. A predictor that bundles their 3D
    annotations could try to match each hidden frame: searching all
    annotated frames, a hidden rotation angle and the person assignment against
    the 2D detections.
  - This cannot be closed while using the paper's real test data. It needs
    deliberate, sizeable effort, and `scoring/CONSTRAINTS.md` forbids it.
  - The grader should check submissions for bundled Panoptic annotations.
- **Deliberate circumvention of the read guard.**
  - The answer file is chmod 000 while predictors run; the same user could
    chmod it back and import the hidden evaluator code.
  - This requires knowing hidden code and deliberately bypassing a guard; the
    grader must check for it.
  - A predictor that kills the evaluator is also out of scope (process
    tampering). `scripts/evaluate_test.sh` deletes `scoring/scores_test.json`
    whenever the evaluator does not finish cleanly.
- **Exhaustive exact-cover search on the pruned candidate set.**
  - It solves the same objective exactly, so it ties the ILP. That is an honest
    alternative solver, not a loophole.
  - Its only difference is speed, and no metric may depend on wall-clock time.
    The paper's runtime claim is therefore not scored; the 30-minute predictor
    limit is a validity bound, not a metric.
- **Knowledge of the detector model.**
  - The detector simulator for the public train split is readable in
    `data/panoptic_benchmark.py`.
  - The hidden split uses its own copy with hidden per-camera profiles and
    noise draws.
  - A method that models the simulator exactly could gain somewhat. The
    measured better variant gains through generic robustness: confidence
    weighting and outlier-view rejection.
- **Copying the reference.** Copying the reference ties it, as any replay does.

## 4. Reference score before and after

| Reference | AP25 | AP50 | AP100 | AP150 | R500 | MPJPE |
|---|---:|---:|---:|---:|---:|---:|
| Before: old gym (two translated copies of one sample body, visible answers) | 0.00 | 3.78 | 69.45 | 84.82 | 100.00 | 69.61 |
| **After: hidden test** | **63.35** | **96.41** | **98.03** | **98.18** | **99.63** | **22.41** |
| After: public train split | 51.46 | 90.92 | 96.29 | 97.07 | 97.95 | 24.08 |
| Paper, Table 1 COMPOSE | 54.66 | 97.27 | 98.94 | 99.17 | 99.83 | 23.62 |
| Paper, Table 1 MvPose | 37.63 | 95.70 | 97.84 | 98.28 | 99.60 | 26.46 |

The reproduction now shows the paper's claim in the paper's direction:
- COMPOSE beats the pairwise, cycle-consistent MvPose-style competitor on
  strict AP (AP25 +3.1, AP50 +2.1, both outside the bootstrap noise).
- The MPJPE gap (−0.7 mm) is within noise; the paper reports −2.8 mm.

**Restorations to the paper's method and settings** (all within the paper's
own procedure):
- The hyperedge cost is the paper's 1/J Σ_j Σ_{v∈e}, summed over views, not
  averaged.
- Hyperedges are grown only from surviving sub-hyperedges ("instead of
  instantiating all combinations").
- τ, λ, γ and s_single are tuned on non-test frames, as the paper does. The
  grid has 400 configurations; the selected τ = 4000, λ = 0.01, γ = 2.2 lie
  inside it. s_single = 0.9 sits at the top of the natural score range: the
  top five configurations differ by < 0.02 utility.
- The data restorations:
  - the paper's cameras (HD 3, 6, 12, 13, 23);
  - its standard Panoptic train and test sequences;
  - the 15-joint VoxelPose evaluation protocol;
  - its MvPose comparison: geometry-only re-implementation, with α
    re-selected on train after the first grid (≤ 0.9) put its optimum on the
    edge. The final grid's optimum is interior (0.96).

## 5. Headroom check

The reference is at the bound of no scored metric:
- recall 99.63 (bound 100);
- AP150 98.18 (bound 100);
- AP25 63.35;
- MPJPE 22.41 mm.

The better variant reaches AP25 85.1 and MPJPE 18.1 mm on the same inputs, and
still leaves room: AP25 < 100, and fully supervised methods report ≈ 15 mm in
the paper. No simple variant reaches a bound. The simple pairwise baseline is
far below (−9.9) on a scale with room above the reference.

No extra experiment was needed: the main experiment has headroom on the
metrics the paper emphasises.

## 6. Weights

The six scored metrics are exactly the paper's Table 1 metrics:

| Metric | Coefficient | Role |
|---|---:|---|
| `ap_25mm` | 1.0 | The paper's largest reported gain (37.6 → 54.7) |
| `ap_50mm` | 0.75 | Strict AP |
| `ap_100mm` | 0.5 | Moderate AP |
| `ap_150mm` | 1.0 | Primary, headline AP |
| `recall_500mm` | 1.0 | Completeness constraint: nobody left out |
| `mpjpe_mm` | 1.0 | The paper's second headline gain |

Three metrics of the old gym were removed:
- **`precision_150mm` and `pose_count_mae`.** They are not paper metrics, and
  they let a coverage-losing camera ablation outrank the reference.
  Over-generation is already penalised by score-ranked AP and by the
  one-pose-per-detection validity rule.
- **`gt_copy_rate`.** Answers are hidden now, so it would be stuck at 0.

## 7. Train and test comparability

The two splits measure the same thing with the same detector process, on
different sequences.

| Reference | AP150 | MPJPE | AP25 |
|---|---:|---:|---:|
| Train | 97.07 | 24.08 | 51.46 |
| Test | 98.18 | 22.41 | 63.35 |

The AP25 gap reflects the denser train sequence (ultimatum, up to 7 people) and
the noise draw: across alternative hidden seeds, test AP25 ranges 63 to 76.

`eval/train/` and `eval/test/` share no code. `eval/` never imports `method/`.
The hidden test is self-contained: it imports nothing from `data/`, `eval/train/`
or `eval/` root, so edits there cannot move it.

## 8. What could not be fixed

- **2D detections are simulated.** The model is rich, but it is not ViTPose
  run on the Panoptic videos (tens of GB per sequence, plus an image detector).
  Absolute numbers are therefore comparable in scale to the paper, not identical
  in setting.
- **Shelf and Campus are not included**, so the paper's generalisation claim is
  untested.
- **The MvPose comparison** uses a geometry-only re-implementation, not the
  official code with appearance features.
- **The runtime/efficiency claim** cannot be scored, because wall-clock metrics
  are not allowed.
- **Staging-directory cleanup.** If the hidden evaluator is SIGKILLed, its
  staging directory under `tmp/` (gitignored) is not cleaned up. Irrelevant for
  a fresh-container hidden test.

## 9. Reproduce

- **Reference:** `bash scripts/download.sh` (login node), then
  `bash scripts/run.sh` (job). The latter runs `reproduce.sh`, which writes
  `scores_train.json`, `scores.json` and `scores_test.json`, then
  `validate.py --post-design --compare`. The final run passes it.
- **Audit:** `bash redteam/hardening/run_hardening.sh` (job). It runs variants
  and shortcuts through the same evaluators, then `rank.py` and `analysis.py`.
- **Jobs** `67ff18ef-052` (full audit) and `ef82739f-e55` (final baseline
  α, reference rerun, rank and noise refresh).

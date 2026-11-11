# Scientist Constraints

## Interface
A method is `checkpoints/<name>/predict.sh INPUTS.json OUTPUT.json` (see
`briefing/evaluation.md`).  The hidden test runs it on frames you never see,
with no network, 1 CPU and a 30-minute limit.  It must be training-free at
test time: any tuning or learning happens beforehand on the train split and
is shipped with the method.

## Inputs
Predictions must be derived only from the cameras and 2D detections in
`INPUTS.json`, plus the method's own code and parameters.  Do not use 3D
annotations of any Panoptic sequence (including copies fetched from the
public release) at prediction time, do not read files under `eval/`, and do not
replace the supplied detections with images, another detector or re-simulated
detections.

## Data use
The public train split (`data/downloads/panoptic_train.json`, with 3D answers)
may be used to select hyperparameters and to evaluate.  The task is
association and triangulation from 2D evidence: methods should not depend on
3D-supervised regression trained on the answers.

## Evaluation
Scores come only from `scripts/evaluate_train.sh` / `scripts/evaluate_test.sh`
(and `scripts/evaluate.sh`).  Do not modify `eval/`, the thresholds, the
validity rules or the data builders when comparing against the recorded scores.

## Compliance Checklist
- [ ] Method registered as `checkpoints/<name>/predict.sh` and runs on unseen frames
- [ ] No 3D annotations, identities or `eval/` files read at prediction time
- [ ] Supplied cameras and 2D detections used unchanged
- [ ] Scores produced by the provided evaluators without edits

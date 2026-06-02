# Scientist Constraints

## Data
Use the provided `data/synthetic_compose_dataset.json` for scored runs unless
the benchmark is explicitly extended with a new named experiment. Do not train,
tune, or evaluate using ground-truth identities from the test scenes;
predictions must be derived from camera calibration and 2D detections only.

## Inputs
Use the same calibrated cameras and 2D detections supplied in the data file.
Do not replace them with ground-truth 2D projections, external image features,
or a stronger 2D detector when reporting `cmu_panoptic_main` scores.

## Evaluation
All methods must write prediction JSON and be scored by `scripts/evaluate.sh`.
Do not modify `eval/` metrics or change matching thresholds when comparing
against the recorded scores.

## Compliance Checklist
- [ ] Scored predictions use `data/synthetic_compose_dataset.json`
- [ ] No ground-truth person IDs or 3D joints are used to form predictions
- [ ] Input cameras and 2D detections are unchanged for `cmu_panoptic_main`
- [ ] Scores are produced by `scripts/evaluate.sh` without metric edits

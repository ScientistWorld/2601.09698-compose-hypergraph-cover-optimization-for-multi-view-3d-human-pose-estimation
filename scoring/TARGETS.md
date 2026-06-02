# Evaluation Targets

## Primary: cmu_panoptic_main
Tests whether a method reconstructs accurate 3D poses for multiple people from
sparse calibrated multi-view 2D detections.
- **Metric**: `ap_25mm` (higher is better)
- **Run**: `bash scripts/evaluate.sh <method_name> <prediction_json>`

## Constraint: cmu_panoptic_main
Tests whether strict precision is achieved while still recovering nearly all
people and keeping matched pose error low.
- **Metrics**: `recall_500mm` (higher is better), `mpjpe_mm` (lower is better)
- **Run**: `bash scripts/evaluate.sh <method_name> <prediction_json>`

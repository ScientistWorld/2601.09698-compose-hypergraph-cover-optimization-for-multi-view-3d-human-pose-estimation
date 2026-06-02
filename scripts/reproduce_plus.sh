#!/bin/bash
# Additional reproduction settings beyond the core scored benchmark.
#
# This writes scoring/scores_stress.json so the main paper-aligned
# scoring/scores.json remains compatible with validate.py --compare.

set -euo pipefail

cd /home/user

mkdir -p data/downloads checkpoints scoring

STRESS_DATA="data/downloads/cmu_panoptic_sample_3view_stress_compose.json"
STRESS_EXPERIMENT="panoptic_sample_3view_stress"

python data/panoptic_sample_compose.py \
  --output "$STRESS_DATA" \
  --frames 48 \
  --people 3 \
  --camera-nodes 6,8,10 \
  --noise-px 9.0 \
  --miss-prob 0.18 \
  --outlier-prob 0.20 \
  --outlier-px 120.0 \
  --seed 29

python baseline/pairwise_baseline.py \
  --data "$STRESS_DATA" \
  --output checkpoints/pairwise_baseline_3view_stress_predictions.json \
  --reprojection-threshold 25 \
  --min-views 2

python method/compose.py \
  --data "$STRESS_DATA" \
  --output checkpoints/compose_3view_stress_predictions.json \
  --tau 1200 \
  --lam 0.0025 \
  --gamma 0.06 \
  --singleton-score 0.03

rm -f scoring/scores_stress.json
python eval/evaluate_predictions.py \
  --predictions checkpoints/pairwise_baseline_3view_stress_predictions.json \
  --data "$STRESS_DATA" \
  --output scoring/scores_stress.json \
  --experiment "$STRESS_EXPERIMENT" \
  --method-name pairwise_baseline \
  --unit-scale 1000

python eval/evaluate_predictions.py \
  --predictions checkpoints/compose_3view_stress_predictions.json \
  --data "$STRESS_DATA" \
  --output scoring/scores_stress.json \
  --experiment "$STRESS_EXPERIMENT" \
  --method-name compose \
  --unit-scale 1000

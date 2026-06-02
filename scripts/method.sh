#!/bin/bash
# Run the paper's proposed method.
#
# This script trains/runs the paper's method using code in method/.
#
# Usage:
#   bash scripts/method.sh

set -e

cd /home/user

DATA="${1:-data/downloads/cmu_panoptic_sample_compose.json}"
OUTPUT="${2:-checkpoints/compose_predictions.json}"

mkdir -p checkpoints

python method/compose.py \
  --data "$DATA" \
  --output "$OUTPUT" \
  --tau 900 \
  --lam 0.003 \
  --gamma 0.06 \
  --singleton-score 0.03

bash scripts/evaluate.sh compose "$OUTPUT" "$DATA" cmu_panoptic_main

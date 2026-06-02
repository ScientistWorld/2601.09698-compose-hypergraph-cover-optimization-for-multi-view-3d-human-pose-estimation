#!/bin/bash
# Run a baseline method from the paper.
#
# Baselines are simple reference methods the paper compares against. 
# Implement this only if applicable, i.e. the paper has baselines.
# You do not need to implement all baselines to begin — 
# just enough to validate that the evaluation pipeline works 
# and provide a reference point. You can decide to expand more baselines later.
# Each baseline must be actually run and scored through evaluate.sh
# so the numbers are verified, not self-reported.
# You can decide what this baseline script should do yourself.
#
# Usage:
#   bash scripts/baseline.sh <name>

set -e

cd /home/user

NAME="${1:-pairwise_baseline}"
DATA="${2:-data/downloads/cmu_panoptic_sample_compose.json}"
OUTPUT="${3:-checkpoints/${NAME}_predictions.json}"

mkdir -p checkpoints

python baseline/pairwise_baseline.py \
  --data "$DATA" \
  --output "$OUTPUT" \
  --reprojection-threshold 25 \
  --min-views 2

bash scripts/evaluate.sh "$NAME" "$OUTPUT" "$DATA" cmu_panoptic_main

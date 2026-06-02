#!/bin/bash
# Train-slice evaluation entry point.
#
# Mirrors scripts/evaluate.sh:
#   ./scripts/evaluate_train.sh <method_name> [prediction_json] [data_json] [experiment]
# With no method argument, scores every checkpoints/*_predictions.json artifact.

set -euo pipefail

cd /home/user

mkdir -p scoring

OUTPUT="scoring/scores_train.json"

evaluate_one() {
  local method_name="$1"
  local predictions="$2"
  local data="$3"
  local experiment="$4"

  python eval/train/evaluate.py \
    --predictions "$predictions" \
    --data "$data" \
    --output "$OUTPUT" \
    --experiment "$experiment" \
    --method-name "$method_name" \
    --unit-scale 1000
}

if [[ $# -gt 0 && -n "${1:-}" ]]; then
  METHOD_NAME="$1"
  PREDICTIONS="${2:-checkpoints/${METHOD_NAME}_predictions.json}"
  DATA="${3:-data/downloads/cmu_panoptic_sample_compose.json}"
  EXPERIMENT="${4:-cmu_panoptic_main}"
  evaluate_one "$METHOD_NAME" "$PREDICTIONS" "$DATA" "$EXPERIMENT"
else
  DATA="data/downloads/cmu_panoptic_sample_compose.json"
  EXPERIMENT="cmu_panoptic_main"
  rm -f "$OUTPUT"
  found=0
  while IFS= read -r predictions; do
    found=1
    filename="$(basename "$predictions")"
    method_name="${filename%_predictions.json}"
    evaluate_one "$method_name" "$predictions" "$DATA" "$EXPERIMENT"
  done < <(find checkpoints -maxdepth 1 -type f -name '*_predictions.json' -print | sort)
  if [[ "$found" -eq 0 ]]; then
    echo "No prediction files found under checkpoints/*_predictions.json" >&2
    exit 1
  fi
fi

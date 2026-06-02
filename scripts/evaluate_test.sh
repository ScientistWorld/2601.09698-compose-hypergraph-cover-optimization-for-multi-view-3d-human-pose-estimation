#!/bin/bash
# Held-out test-slice evaluation entry point.
#
# Mirrors scripts/evaluate.sh:
#   ./scripts/evaluate_test.sh <method_name> [prediction_json] [data_json] [experiment]
# With no method argument, scores every method recorded in scoring/scores_train.json.

set -euo pipefail

cd /home/user

mkdir -p scoring

OUTPUT="scoring/scores_test.json"
TRAIN_SCORES="scoring/scores_train.json"

evaluate_one() {
  local method_name="$1"
  local predictions="$2"
  local data="$3"
  local experiment="$4"

  python eval/test/evaluate.py \
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
  DATA="${3:-data/synthetic_compose_dataset.json}"
  EXPERIMENT="${4:-cmu_panoptic_main}"
  evaluate_one "$METHOD_NAME" "$PREDICTIONS" "$DATA" "$EXPERIMENT"
else
  DATA="data/synthetic_compose_dataset.json"
  EXPERIMENT="cmu_panoptic_main"
  rm -f "$OUTPUT"
  found=0
  if [[ -f "$TRAIN_SCORES" ]]; then
    while IFS=$'\t' read -r method_name predictions; do
      [[ -n "$method_name" ]] || continue
      found=1
      evaluate_one "$method_name" "$predictions" "$DATA" "$EXPERIMENT"
    done < <(
      python - <<'PY'
import json
from pathlib import Path

scores = json.loads(Path("scoring/scores_train.json").read_text(encoding="utf-8"))
prediction_files = scores.get("prediction_files", {})
seen = set()
for exp_data in scores.get("experiments", {}).values():
    for method_name in exp_data.get("results", {}):
        if method_name in seen:
            continue
        seen.add(method_name)
        print(f"{method_name}\t{prediction_files.get(method_name, f'checkpoints/{method_name}_predictions.json')}")
PY
    )
  else
    while IFS= read -r predictions; do
      found=1
      filename="$(basename "$predictions")"
      method_name="${filename%_predictions.json}"
      evaluate_one "$method_name" "$predictions" "$DATA" "$EXPERIMENT"
    done < <(find checkpoints -maxdepth 1 -type f -name '*_predictions.json' -print | sort)
  fi
  if [[ "$found" -eq 0 ]]; then
    echo "No train-score methods or prediction files found to evaluate on the test slice" >&2
    exit 1
  fi
fi

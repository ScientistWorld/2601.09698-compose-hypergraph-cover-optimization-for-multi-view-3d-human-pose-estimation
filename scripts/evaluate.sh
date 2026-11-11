#!/bin/bash
# Score one registered method on the public train frames into scoring/scores.json.
#
# Usage:
#   bash scripts/evaluate.sh <method_name>                     # runs checkpoints/<name>/predict.sh
#   bash scripts/evaluate.sh <method_name> <predictions.json>  # scores an existing prediction file
#
# Uses only eval/ (no method code).  For the full train / hidden-test tables use
# scripts/evaluate_train.sh and scripts/evaluate_test.sh (no arguments).

set -e

cd /home/user

if [[ $# -lt 1 ]]; then
  echo "usage: bash scripts/evaluate.sh <method_name> [predictions.json]" >&2
  exit 2
fi
NAME="$1"
if [[ $# -ge 2 ]]; then
  python eval/evaluate_predictions.py --method-name "$NAME" --predictions "$2"
else
  python eval/evaluate_predictions.py --method-name "$NAME"
fi

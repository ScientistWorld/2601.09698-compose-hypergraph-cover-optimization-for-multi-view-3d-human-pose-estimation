#!/bin/bash
# Hidden test evaluation: runs every registered predictor checkpoints/*/predict.sh
# on the hidden test frames (built inside eval/test/ at test time) and writes
# scoring/scores_test.json.  Runs with no arguments.

set -uo pipefail

cd /home/user

rm -f scoring/scores_test.json
if ! python eval/test/evaluate.py; then
  rm -f scoring/scores_test.json
  echo "hidden test evaluation failed" >&2
  exit 1
fi

#!/bin/bash
# Reproduce the reference end to end (inside the compute container).
#
# Registers the baselines and COMPOSE as predictors, scores them on the public
# train frames (scoring/scores_train.json, scoring/scores.json) and on the
# hidden test frames (scoring/scores_test.json).
# Assumes scripts/download.sh has been run on a node with internet.

set -e

cd /home/user

bash scripts/download.sh
bash scripts/baseline.sh
bash scripts/method.sh
bash scripts/evaluate_train.sh

rm -f scoring/scores.json
for dir in checkpoints/*/; do
  name="$(basename "$dir")"
  [[ -f "$dir/predict.sh" && -f "$dir/train_predictions.json" ]] || continue
  bash scripts/evaluate.sh "$name" "$dir/train_predictions.json"
done

bash scripts/evaluate_test.sh

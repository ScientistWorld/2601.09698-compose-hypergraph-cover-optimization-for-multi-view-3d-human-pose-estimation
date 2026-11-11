#!/bin/bash
# Train-split evaluation: runs every registered predictor checkpoints/*/predict.sh
# on the public train frames and writes scoring/scores_train.json.
# Runs with no arguments (optional: --methods NAME ...).

set -euo pipefail

cd /home/user

python eval/train/evaluate.py "$@"

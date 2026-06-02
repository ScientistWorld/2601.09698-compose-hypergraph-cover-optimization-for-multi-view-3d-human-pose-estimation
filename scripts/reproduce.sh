#!/bin/bash
# Reproduce the paper's results end-to-end.
#
# This script runs inside the compute container and should call
# scripts/baseline.sh and scripts/method.sh to run baselines and
# the paper's method, each of which calls scripts/evaluate.sh.
#
# Assumes:
#   - Container is built (environment/container.sif)
#   - Data and models are downloaded (via scripts/download.sh)
#   - Working directory is /home/user
#
# This is the "gold standard" reproduction — 
# You do not need to run this, but keep this in good shape
# so other agents can easily run a full loop of reproduction and get the paper's numbers.

set -e

cd /home/user

rm -f scoring/scores.json

bash scripts/download.sh
DATA="data/downloads/cmu_panoptic_sample_compose.json"
bash scripts/baseline.sh pairwise_baseline "$DATA" checkpoints/pairwise_baseline_predictions.json
bash scripts/method.sh "$DATA" checkpoints/compose_predictions.json

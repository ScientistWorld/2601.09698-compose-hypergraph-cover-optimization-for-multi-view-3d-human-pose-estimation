#!/bin/bash
# Run the paper's method (COMPOSE) and register it as a predictor.
#
# 1. Hyperparameters (tau, lambda, gamma, s_single) are selected on the public
#    train frames (the paper tunes on non-test frames).  The selected config is
#    recorded in method/compose_config.json; pass --retune to redo the search.
# 2. checkpoints/compose/predict.sh is written; the evaluators call it as
#    `bash checkpoints/compose/predict.sh INPUTS.json OUTPUT.json`.
#
# Usage: bash scripts/method.sh [--retune]

set -e

cd /home/user

mkdir -p checkpoints/compose
if [[ "${1:-}" == "--retune" || ! -s method/compose_config.json ]]; then
  python method/tune_compose.py \
    --data data/downloads/panoptic_train.json \
    --output method/compose_config.json \
    --table checkpoints/compose/tuning.json
fi
cp method/compose_config.json checkpoints/compose/config.json

cat > checkpoints/compose/predict.sh <<'PRED'
#!/bin/bash
# COMPOSE predictor: bash predict.sh INPUTS.json OUTPUT.json
set -euo pipefail
cd /home/user
exec python method/compose.py --inputs "$1" --output "$2" --config checkpoints/compose/config.json
PRED
echo "Registered checkpoints/compose/predict.sh with config $(cat checkpoints/compose/config.json | tr -d '\n ')"

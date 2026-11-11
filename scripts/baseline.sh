#!/bin/bash
# Register the baselines as predictors.
#
#   pairwise_baseline  simple pairwise reprojection matching + union-find grouping
#   mvpose_geometric   the paper's pairwise competitor (MvPose-style epipolar
#                      affinity + convex cycle-consistent multi-way matching),
#                      geometry only; its matching weight is selected on train
#                      frames and recorded in baseline/mvpose_geometric_config.json.
#
# Usage: bash scripts/baseline.sh [--retune]

set -e

cd /home/user

mkdir -p checkpoints/pairwise_baseline checkpoints/mvpose_geometric

cat > checkpoints/pairwise_baseline/predict.sh <<'PRED'
#!/bin/bash
set -euo pipefail
cd /home/user
exec python baseline/pairwise_baseline.py --data "$1" --output "$2" --reprojection-threshold 25 --min-views 2
PRED

if [[ "${1:-}" == "--retune" || ! -s baseline/mvpose_geometric_config.json ]]; then
  python baseline/mvpose_geometric.py --tune-on data/downloads/panoptic_train.json \
    --config baseline/mvpose_geometric_config.json
fi
cat > checkpoints/mvpose_geometric/predict.sh <<'PRED'
#!/bin/bash
set -euo pipefail
cd /home/user
exec python baseline/mvpose_geometric.py --inputs "$1" --output "$2" --config baseline/mvpose_geometric_config.json
PRED
echo "Registered checkpoints/pairwise_baseline and checkpoints/mvpose_geometric"

#!/bin/bash
# Download all data needed for this environment (run on a node with internet).
#
# Idempotent: every fetch is skipped when its file already exists.
#
# Sources: official CMU Panoptic release (3D body annotations
# hdPose3d_stage1_coco19.tar and HD calibration) for five sequences of the
# standard multi-view pose *training* split.  The public train benchmark
# (cameras + simulated 2D detections + 3D answers) is then built from them.
# The hidden test split is NOT fetched here; its answers live in eval/test/.
#
# Estimated total download size: ~175 MB
# Estimated disk usage after build: ~180 MB

set -e

cd /home/user

BASE_URL="http://domedb.perception.cs.cmu.edu/webdata/dataset"
TRAIN_SEQUENCES="160422_ultimatum1 160224_haggling1 160226_haggling1 160906_band1 160906_ian1"

for seq in $TRAIN_SEQUENCES; do
  dir="data/downloads/panoptic/$seq"
  mkdir -p "$dir"
  [ -s "$dir/calibration_$seq.json" ] || \
    curl -L --fail -o "$dir/calibration_$seq.json" "$BASE_URL/$seq/calibration_$seq.json"
  [ -s "$dir/hdPose3d_stage1_coco19.tar" ] || \
    curl -L --fail -o "$dir/hdPose3d_stage1_coco19.tar" "$BASE_URL/$seq/hdPose3d_stage1_coco19.tar"
done

# Public train benchmark (pure-Python builder, deterministic).
[ -s data/downloads/panoptic_train.json ] || python3 data/panoptic_benchmark.py

#!/bin/bash
# Download all data and models needed for this environment.
#
# Run on a node with internet access BEFORE building the container
# or submitting compute jobs.
#
# This script should be idempotent — safe to run multiple times.
# It should skip files that already exist.
#
# Update the storage estimate below as you add downloads.
#
# Estimated total download size: <1 MB
# Estimated disk usage after extraction: <10 MB
#
# This script is intentionally workspace-local so a fresh clone can recreate
# all required assets without relying on machine-specific caches.

set -e

cd /home/user

# Fetch the small official CMU Panoptic sample assets used to build the
# lightweight real-calibration benchmark. URLs were verified with HTTP 200.
mkdir -p data/downloads/cmu_panoptic_sample
[ -e data/downloads/cmu_panoptic_sample/calibration_171204_pose1_sample.json ] || \
  curl -L --fail \
    -o data/downloads/cmu_panoptic_sample/calibration_171204_pose1_sample.json \
    http://domedb.perception.cs.cmu.edu/webdata/dataset/171204_pose1_sample/calibration_171204_pose1_sample.json
[ -e data/downloads/cmu_panoptic_sample/hdPose3d_stage1_coco19.tar ] || \
  curl -L --fail \
    -o data/downloads/cmu_panoptic_sample/hdPose3d_stage1_coco19.tar \
    http://domedb.perception.cs.cmu.edu/webdata/dataset/171204_pose1_sample/hdPose3d_stage1_coco19.tar

[ -e data/downloads/cmu_panoptic_sample_compose.json ] || \
  python data/panoptic_sample_compose.py --output data/downloads/cmu_panoptic_sample_compose.json

# Retain the original deterministic synthetic benchmark as a fallback and
# debugging fixture.
[ -e data/synthetic_compose_dataset.json ] || python data/synthetic_compose.py --output data/synthetic_compose_dataset.json

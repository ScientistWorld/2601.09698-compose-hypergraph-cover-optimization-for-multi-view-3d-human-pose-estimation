#!/bin/bash
set -euo pipefail
cd /home/user
export PYTHONDONTWRITEBYTECODE=1
for name in \
  reference_seed_0 reference_seed_1 reference_seed_2 \
  quality_stronger quality_weaker quality_no_higher_order pairwise_baseline \
  shortcut_bruteforce shortcut_bruteforce_pruned shortcut_labels_jitter shortcut_exact_labels \
  shortcut_generator_rebuild shortcut_identity_labels shortcut_gt_replacement \
  shortcut_nan_mask shortcut_partial_pose shortcut_truncated_joints \
  shortcut_empty shortcut_all_nan shortcut_constant shortcut_infinite \
  shortcut_null_coordinates shortcut_metadata_forgery shortcut_forged_train_scores \
  shortcut_baseline_sabotage shortcut_unregistered shortcut_missing_artifact; do
  bash "redteam/$name/run.sh"
done
python redteam/summarize.py

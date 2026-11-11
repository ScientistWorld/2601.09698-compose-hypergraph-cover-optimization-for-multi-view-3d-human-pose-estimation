#!/bin/bash
set -euo pipefail
cd /home/user
export PYTHONDONTWRITEBYTECODE=1
for name in quality_stronger shortcut_no_higher_order shortcut_bruteforce_pruned shortcut_partial_pose; do
  bash "redteam/$name/run.sh"
done
python redteam/summarize.py

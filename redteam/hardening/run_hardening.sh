#!/bin/bash
# Hardening audit: register variants and shortcuts as predictors in a separate
# submissions directory (never in checkpoints/, so the frozen reference scores
# stay reference-only), score them through the unchanged train and hidden-test
# evaluators, then compute rank deltas and noise checks.
set -uo pipefail
cd /home/user
H=redteam/hardening
SUB=$H/submissions
rm -rf "$SUB"
mkdir -p "$SUB"

register() {  # name script mode
  mkdir -p "$SUB/$1"
  printf '#!/bin/bash\nset -euo pipefail\ncd /home/user\nexec python %s %s "$1" "$2"\n' "$2" "$3" > "$SUB/$1/predict.sh"
}
register compose_robust $H/variants.py robust
register compose_pairs_only $H/variants.py pairs_only
register compose_three_cams $H/variants.py three_cams
register compose_old_operating_point $H/variants.py old_operating_point
register bruteforce_unpruned $H/variants.py bruteforce
for mode in empty constant forged_metrics train_gt_lookup identity_grouping hidden_source_read \
            partial_pose_nan truncated_joints null_coordinates infinite too_many_poses missing_frames \
            out_of_volume spray_recall crash; do
  register "shortcut_$mode" $H/shortcuts.py "$mode"
done

echo "=== [$(date +%T)] hardening: train split (variants)"
python eval/train/evaluate.py --submissions "$SUB" --output $H/scores_train.json \
  --methods compose_robust compose_pairs_only compose_three_cams compose_old_operating_point
echo "=== [$(date +%T)] hardening: hidden test (variants + shortcuts)"
python eval/test/evaluate.py --submissions "$SUB" --output $H/scores_test.json
echo "=== [$(date +%T)] hardening: rank deltas"
python $H/rank.py
echo "=== [$(date +%T)] hardening: paired bootstrap and seeds"
python $H/analysis.py > $H/analysis.log 2>&1; echo "analysis rc=$?"; tail -c 3000 $H/analysis.log
echo "=== [$(date +%T)] hardening done"

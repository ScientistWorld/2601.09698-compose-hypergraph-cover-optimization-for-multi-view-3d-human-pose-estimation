#!/bin/bash
# Job script (runs in the gym container, workspace at /home/user, no internet).
# Hardening job 3: re-select the MvPose baseline's matching weight on the
# extended grid, rerun the reference reproduction (COMPOSE config is recorded
# in method/compose_config.json), validate, and refresh the audit rank deltas
# and paired-bootstrap / noise-seed analysis (which include the baseline).

set -uo pipefail

cd /home/user

echo "=== [$(date +%T)] reproduce"
bash scripts/reproduce.sh
echo "=== [$(date +%T)] reproduce rc=$?"

echo "=== [$(date +%T)] validate"
python validate.py --post-design --compare

echo "=== [$(date +%T)] audit: rank deltas"
python redteam/hardening/rank.py
echo "=== [$(date +%T)] audit: paired bootstrap and seeds"
python redteam/hardening/analysis.py > redteam/hardening/analysis.log 2>&1; echo "analysis rc=$?"
python - <<'PY'
import json
a = json.load(open("redteam/hardening/analysis.json"))
for name, d in a["bootstrap"].items():
    print(name, {m: (round(v["delta_vs_compose"], 3), [round(x, 3) for x in v["ci95"]]) for m, v in d.items()})
PY
echo "=== [$(date +%T)] all done"

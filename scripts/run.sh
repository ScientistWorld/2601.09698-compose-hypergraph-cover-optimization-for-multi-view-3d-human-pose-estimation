#!/bin/bash
# Job script (runs in the gym container, workspace at /home/user, no internet).
# Hardening job: smoke checks, full reference reproduction, validation, audit.

set -uo pipefail

cd /home/user

echo "=== [$(date +%T)] smoke"
python - <<'PY'
import json, subprocess, sys, tempfile, time
from pathlib import Path
sys.path.insert(0, "/home/user")
from eval.harness import strip_answers
b = json.loads(Path("data/downloads/panoptic_train.json").read_text())
b["frames"] = b["frames"][:6] + b["frames"][-6:]
tmp = Path(tempfile.mkdtemp())
(tmp / "in.json").write_text(json.dumps(strip_answers(b)))
for cmd in (["python", "method/compose.py", "--inputs", str(tmp / "in.json"), "--output", str(tmp / "c.json")],
            ["python", "baseline/mvpose_geometric.py", "--inputs", str(tmp / "in.json"), "--output", str(tmp / "m.json")]):
    t = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(cmd[1], "rc", r.returncode, "%.1fs" % (time.time() - t), r.stderr[-3000:], flush=True)
PY

echo "=== [$(date +%T)] reproduce"
bash scripts/reproduce.sh
echo "=== [$(date +%T)] reproduce rc=$?"

echo "=== [$(date +%T)] validate"
python validate.py --post-design --compare

echo "=== [$(date +%T)] hardening"
bash redteam/hardening/run_hardening.sh
echo "=== [$(date +%T)] all done"

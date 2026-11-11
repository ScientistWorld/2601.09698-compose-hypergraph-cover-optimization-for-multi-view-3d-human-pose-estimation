#!/bin/bash
set -euo pipefail
cd /home/user
export PYTHONDONTWRITEBYTECODE=1
bash redteam/shortcut_no_higher_order/run.sh
python redteam/summarize.py
python redteam/verify_restoration.py

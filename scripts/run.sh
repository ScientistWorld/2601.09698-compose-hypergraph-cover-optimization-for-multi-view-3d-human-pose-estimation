#!/bin/bash
# Job script (runs in the gym container, workspace at /home/user, no internet).
# Reproduces the reference end to end: registers the baselines and COMPOSE as
# predictors (COMPOSE hyperparameters from method/compose_config.json; pass
# --retune to scripts/method.sh to redo the train-split search), scores them on
# the public train split and on the hidden test split, then validates.
#
# The hardening audit (variants, shortcuts, rank deltas, noise checks) is
# separate: bash redteam/hardening/run_hardening.sh

set -euo pipefail

cd /home/user

bash scripts/reproduce.sh
python validate.py --post-design --compare

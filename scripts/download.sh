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
# Estimated total download size: 0 GB
# Estimated disk usage after extraction: 0 GB
#
# This script is intentionally workspace-local so a fresh clone can recreate
# all required assets without relying on machine-specific caches.

set -e

cd /home/user

# This reproduction uses a tracked deterministic synthetic benchmark and
# requires no external datasets, pretrained models, or detector weights.
[ -e data/synthetic_compose_dataset.json ] || python data/synthetic_compose.py --output data/synthetic_compose_dataset.json

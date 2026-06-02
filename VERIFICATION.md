# Verification Report

## Initial Assessment
The workspace already had a justified `core_claim` reproduction: the method
implementation runs end to end, the evaluator reads prediction artifacts rather
than method code, and `python validate.py --compare` passed with the current
score artifacts. The main scored experiment shows a clear advantage over the
pairwise baseline on AP, recall, and MPJPE while preserving a lightweight,
Panoptic-sample-derived task.

## Issues Found
- Minor: `briefing/problem.md` and `briefing/evaluation.md` still contained
  template-style wording and one target-method phrase that was less
  method-agnostic than desired.
- Minor: `scoring/EXPERIMENTS.md` listed an unverified sparse-view stress
  setting that was not present in `scoring/scores.json`.
- Minor: `scripts/run.sh` and `action.yaml` still referenced the unverified
  stress extension from the prior pending iteration.
- No critical scoring, import-separation, download, or reference-value issues
  were found.

## Fixes Applied
- Cleaned the method-agnostic briefing language and kept the problem/evaluation
  files focused on task and metrics.
- Removed the unverified stress extension from `scoring/EXPERIMENTS.md`, the
  standard job script, and the workspace scripts.
- Kept `scoring/DIRECTION.md` high-level by describing the allowed research
  family without paper-specific implementation details.
- Verified the two URLs in `scripts/download.sh` resolve successfully and
  re-ran `python validate.py --compare`.

## Final Judgment
This workspace is ready to serve as a core-claim research gym. A scientist in
improve mode can run the packaged method and baseline, while a scientist in
from-scratch mode can use the method-agnostic data and evaluator without
depending on `method/`.

- Milestone: core_claim
- Ready for gym use: yes
- Confidence: medium
- Key limitation: lightweight CMU Panoptic sample-derived benchmark with
  generated 2D detections, not the full original CMU Panoptic/Shelf/Campus
  evaluation suite.

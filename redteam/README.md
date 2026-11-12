> **Hardening note (iteration 11).** The scenario folders below were run against the
> pre-hardening gym (single-body Panoptic sample, visible test answers,
> `checkpoints/*_predictions.json` interface). That gym no longer exists; their
> scores are kept as the "before" evidence cited in `../HARDENING.md`. The
> hardened gym's audit (variants, shortcuts, rank deltas, paired bootstrap,
> noise seeds) is in `hardening/` and is rerun with
> `bash redteam/hardening/run_hardening.sh`.

This audit runs CPU experiments through the unmodified hidden entry point with no arguments. Each scenario has a run.sh; run the whole bundle with `bash redteam/run_all.sh` inside the gym container. Source changes stay here. The harness temporarily installs prediction/data/registration artifacts, launches train and hidden evaluation in separate subprocesses, copies scores into the scenario directory, and restores all original artifacts in a finally block. No evaluator, interpreter, import path, or environment is replaced.

Reference seeds set Python and NumPy RNGs and PYTHONHASHSEED; the reference has no stochastic training or seed argument. Quality tuning selects hyperparameters on train scenes only. Shortcut implementations use only data, baseline code, the public train evaluator/metrics, and (in improve mode) reference method code. Hidden evaluator code is only invoked through `bash scripts/evaluate_test.sh`.

`reference/metric_config.json` is an audit-only copy of scoring/reference.json; scenario methods never read it. `rank_report.json` uses population standard deviations across the explicitly listed audit cohort. Future ranks depend on the future attempt cohort, so these numerical ranks are illustrative, with dominating improvements identified separately.

The completed findings and all shortcut outcomes are in `../REDTEAM.md`. The final cohort includes every distinct successful audit method and the simple baseline, while excluding duplicate reference seeds. Failed tests retain execution/error records, not fabricated scores. `restoration.json` verifies restoration of the original artifact hashes and evaluator sources. `run_final_check.sh` reruns the higher-order-removal shortcut and verifies the saved results.

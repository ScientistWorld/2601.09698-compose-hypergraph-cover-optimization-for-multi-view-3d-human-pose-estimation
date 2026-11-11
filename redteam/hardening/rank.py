#!/usr/bin/env python3
"""Rank deltas versus the reference (audit only).

rank(m) = sum_metric |coefficient| * improvement(m over compose, in the metric's
better direction) / population std of the metric across the cohort.
Two cohorts: honest methods only (reference, baselines, variants) and every
attempt (adding shortcuts; failed runs carry their worst values).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = "cmu_panoptic_main"


def load(path: Path) -> dict:
    data = json.loads(path.read_text())
    return data["experiments"][EXP]["results"], data.get("_status", {})


def main() -> None:
    ref = json.loads((ROOT / "scoring" / "reference.json").read_text())["experiments"][EXP]["metrics"]
    results, status = load(ROOT / "scoring" / "scores_test.json")
    extra, extra_status = load(ROOT / "redteam" / "hardening" / "scores_test.json")
    results = {**results, **extra}
    status = {**status, **extra_status}
    honest = [n for n in results if not n.startswith("shortcut_")]
    cohorts = {"honest": honest, "all": list(results)}
    out = {"status": status, "metrics": results, "rank": {}}
    for cname, members in cohorts.items():
        std = {}
        for m in ref:
            vals = [results[n][m] for n in members]
            mu = sum(vals) / len(vals)
            std[m] = math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals)) or 1.0
        ranks = {}
        for name in results:
            total = 0.0
            for m, spec in ref.items():
                sign = 1.0 if spec["higher_is_better"] else -1.0
                total += abs(spec["coefficient"]) * sign * (results[name][m] - results["compose"][m]) / std[m]
            ranks[name] = total
        out["rank"][cname] = {"std": std, "rank_delta": dict(sorted(ranks.items(), key=lambda kv: -kv[1]))}
    (ROOT / "redteam" / "hardening" / "rank.json").write_text(json.dumps(out, indent=1) + "\n")
    for cname in cohorts:
        print(f"== cohort {cname}")
        for name, r in out["rank"][cname]["rank_delta"].items():
            m = results[name]
            print(f"{name:34s} {r:+8.3f}  AP25 {m['ap_25mm']:6.2f} AP50 {m['ap_50mm']:6.2f} AP100 {m['ap_100mm']:6.2f} "
                  f"AP150 {m['ap_150mm']:6.2f} R500 {m['recall_500mm']:6.2f} MPJPE {m['mpjpe_mm']:7.2f}  [{status.get(name)}]")


if __name__ == "__main__":
    main()

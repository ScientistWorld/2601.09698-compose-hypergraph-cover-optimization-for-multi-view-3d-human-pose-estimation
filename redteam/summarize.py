"""Compute the supplied rank formula for a fully disclosed, frozen audit cohort."""
import json
import math
from pathlib import Path
import statistics
ROOT = Path('/home/user/redteam')

def read(p):
    return json.loads(p.read_text())

def result(name):
    p = ROOT / name / 'scores_test.json'
    if not p.exists():
        return None
    results = read(p)['experiments']['cmu_panoptic_main']['results']
    return results.get(name, results.get('shortcut_replay'))

config = read(ROOT/'reference/metric_config.json')['experiments']['cmu_panoptic_main']
reference = result('reference_seed_0')
# Baseline is a comparison, seeds are reruns; avoid over-weighting duplicates.
names = ['reference_seed_0', 'quality_stronger', 'quality_weaker'] + [
    p.name for p in sorted(ROOT.glob('shortcut_*')) if (p/'execution.json').exists() and result(p.name) is not None]
metrics = {k: v for k, v in config['metrics'].items() if abs(v['coefficient']) > 0}
stds = {}
for key in metrics:
    values = [result(n).get(key) for n in names]
    values = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    stds[key] = statistics.pstdev(values) if values else 0
report = {'formula': 'sum weight*abs(coefficient)*direction*(attempt-reference)/population_std; skip null/missing',
          'reference': reference, 'cohort': names, 'metric_std': stds, 'attempts': {}}
for name in names + ['reference_seed_1', 'reference_seed_2', 'pairwise_baseline',
                      'shortcut_null_coordinates', 'shortcut_missing_artifact']:
    scores = result(name)
    if scores is None:
        report['attempts'][name] = {'rank_delta': None, 'no_score_artifact': True,
                                   'missing_skip_arithmetic': 0.0,
                                   'note': 'Failed test. Empty-rank arithmetic is zero; acceptance of failed jobs is outside the gym.'}
        continue
    contributions = {}
    for key, c in metrics.items():
        v = scores.get(key)
        if v is None or not math.isfinite(v) or not stds[key]:
            contributions[key] = 0
        else:
            direction = 1 if c['higher_is_better'] else -1
            contributions[key] = config['weight']*abs(c['coefficient'])*direction*(v-reference[key])/stds[key]
    report['attempts'][name] = {'rank_delta': sum(contributions.values()), 'metrics': scores,
                              'metric_contributions': contributions}
seeds = [result('reference_seed_'+str(i)) for i in range(3)]
report['reference_seed_spread'] = {k: max(s[k] for s in seeds)-min(s[k] for s in seeds) for k in reference}
report['reference_at_bound_scored'] = [
    k for k in metrics if (k.startswith(('ap_', 'precision_', 'recall_')) and reference[k] == 100)
    or (k in ('gt_copy_rate', 'pose_count_mae', 'mpjpe_mm') and reference[k] == 0)]
(ROOT/'rank_report.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))

"""Verify that audit mutations and hidden evaluator changes are absent."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path('/home/user')
RED = ROOT/'redteam'
BASE = '8b9a054'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifest = json.loads((RED/'.state/manifest.json').read_text())
restored = {item['path']: sha(ROOT/item['path']) == item['sha256'] for item in manifest}
predictions = sorted(str(p.relative_to(ROOT)) for p in (ROOT/'checkpoints').glob('*_predictions.json'))
expected = sorted(x['path'] for x in manifest if x['path'].startswith('checkpoints/'))
pristine = subprocess.check_output(
    ['git', 'diff', '--name-only', BASE, '--', 'eval', 'scripts/evaluate_test.sh',
     'scripts/evaluate_train.sh', 'method', 'baseline', 'data', 'environment', 'scoring'],
    cwd=ROOT, text=True).splitlines()
report = {'base_commit': BASE, 'snapshotted_artifacts_restored': restored,
          'only_original_predictions': predictions == expected,
          'changed_gym_source_files': pristine}
(RED/'restoration.json').write_text(json.dumps(report, indent=2)+'\n')
assert all(restored.values()) and predictions == expected and not pristine, report
print(json.dumps(report, indent=2))

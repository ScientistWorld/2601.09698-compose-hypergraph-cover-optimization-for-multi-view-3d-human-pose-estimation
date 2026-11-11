#!/usr/bin/env python3
"""Artifact-only audit harness. Never edits evaluator code or Python environment."""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path('/home/user')
RED = ROOT / 'redteam'
DATA = ROOT / 'data/downloads/cmu_panoptic_sample_compose.json'
STATE = RED / '.state'


def read(p):
    return json.loads(Path(p).read_text())


def write(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, allow_nan=True) + '\n')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def command(args, logfile, env=None):
    start = time.monotonic()
    with Path(logfile).open('w') as f:
        proc = subprocess.run(args, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=env)
    return {'exit_code': proc.returncode, 'seconds': time.monotonic() - start}


def ensure_state():
    STATE.mkdir(exist_ok=True)
    paths = [DATA, *sorted((ROOT / 'checkpoints').glob('*_predictions.json')),
             *sorted((ROOT / 'scoring').glob('scores*.json'))]
    manifest = STATE / 'manifest.json'
    if not manifest.exists():
        items = []
        for i, p in enumerate(paths):
            target = STATE / f'original_{i}.dat'
            shutil.copy2(p, target)
            items.append({'path': str(p.relative_to(ROOT)), 'backup': target.name,
                          'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
        write(manifest, items)


def restore():
    for p in (ROOT / 'checkpoints').glob('*_predictions.json'):
        p.unlink()
    for p in (ROOT / 'scoring').glob('scores*.json'):
        p.unlink()
    for item in read(STATE / 'manifest.json'):
        shutil.copy2(STATE / item['backup'], ROOT / item['path'])


def original_data():
    manifest = read(STATE / 'manifest.json')
    item = next(x for x in manifest if x['path'] == str(DATA.relative_to(ROOT)))
    return read(STATE / item['backup'])


def original_predictions():
    cache = RED / 'reference' / 'predictions.json'
    if not cache.exists():
        match = next((x for x in read(STATE/'manifest.json') if x['path'] ==
                      'checkpoints/compose_predictions.json'), None)
        if match:
            shutil.copy2(STATE/match['backup'], cache)
        else:
            run = command([sys.executable, 'method/compose.py', '--data', str(DATA),
                           '--output', str(cache)], RED/'reference/run.log')
            if run['exit_code']:
                raise RuntimeError('Cannot generate reference artifact')
    return read(cache)


def as_frames(payload):
    f = payload['frames']
    return list(f.values()) if isinstance(f, dict) else f


def pose_payload(data, kind='labels', offset=1e-5):
    frames = []
    for scene in data['scenes']:
        poses = []
        for i, person in enumerate(scene['people']):
            joints = [person['joints_3d'][n] for n in data['task']['joint_names']]
            if kind == 'constant':
                joints = [[0.0, 0.0, 0.0] for _ in joints]
            else:
                joints = [[v + (offset if axis == 0 else 0) for axis, v in enumerate(j)] for j in joints]
            poses.append({'pose_id': i, 'score': 1.0, 'joints_3d': joints})
        frames.append({'frame_id': scene['scene_id'], 'predicted_poses': poses})
    return {'frames': frames}


def train_utility(m):
    # Selection uses train only. This is a tuning utility, not the final gym rank.
    return (m['ap_150mm'] + .5*m['ap_100mm'] + .5*m['ap_50mm'] + .4*m['precision_150mm']
            + .25*m['recall_500mm'] - .5*(m['mpjpe_mm'] if m['mpjpe_mm'] is not None else 1000)
            - .3*m['pose_count_mae'])


def train_eval(payload, data):
    # These are explicitly public modules supplied to future agents.
    sys.path.insert(0, str(ROOT))
    from eval.train.evaluate import split_frame_ids, filter_data_payload
    from eval.metrics import evaluate_pose_predictions
    sliced = filter_data_payload(data, set(split_frame_ids(data)))
    return evaluate_pose_predictions(payload, sliced, unit_scale=1000)


def emit_selected(mod, frame, edges, indices):
    preds = []
    for i in indices:
        e = edges[i]
        if len(e.refs) < 2 or e.joints_3d is None:
            continue
        preds.append({'pose_id': len(preds), 'joints_3d': e.joints_3d,
                      'score': e.score, 'views': [r.view for r in e.refs],
                      'detections': [r.index for r in e.refs], 'reprojection_cost': e.cost})
    return {'frame_id': frame['frame_id'], 'predicted_poses': preds}


def stronger(out, data):
    mod = load_module('compose_audit_tuning', ROOT/'method/compose.py')
    norm = mod.normalize_dataset(data)
    pmats = [mod.projection_matrix(c) for c in norm['cameras']]
    # Enumerate geometry once; retain every computational step of COMPOSE.
    cached = [mod.enumerate_hyperedges(f['detections'], pmats, 900, .003, .03) for f in norm['frames']]
    candidates = []
    best = None
    for lam in (.001, .003):
        for gamma in (.6, .9, 1.1, 1.3, 1.6):
            frames = []
            for f, edges in zip(norm['frames'], cached):
                for e in edges:
                    e.score = .03 if len(e.refs) == 1 else math.exp(-lam*e.cost)
                chosen = mod.solve_exact_cover(edges, gamma)
                frames.append(emit_selected(mod, f, edges, chosen))
            payload = {'frames': frames, 'config': {'tau': 900, 'lambda': lam, 'gamma': gamma,
                                                       'singleton_score': .03}}
            metrics = train_eval(payload, data)
            utility = train_utility(metrics)
            candidates.append({'config': payload['config'], 'train_metrics': metrics, 'train_utility': utility})
            if best is None or utility > best[0]:
                best = (utility, payload)
    write(RED/'quality_stronger/train_selection.json', candidates)
    write(out, best[1])


def brute_force(out, data, pruned=False):
    from functools import lru_cache
    mod = load_module('compose_audit_bruteforce', ROOT/'method/compose.py')
    norm = mod.normalize_dataset(data)
    pmats = [mod.projection_matrix(c) for c in norm['cameras']]
    frames, stats = [], []
    for f in norm['frames']:
        start = time.monotonic()
        edges = mod.enumerate_hyperedges(f['detections'], pmats, math.inf, .003, .03)
        enumerated = len(edges)
        if pruned:
            # Still instantiate every combination; exhaustive subset search replaces ILP.
            edges = [e for e in edges if len(e.refs) == 1 or e.cost <= 900]
        nodes = sorted({(r.view, r.index) for e in edges for r in e.refs})
        bits = {n: 1 << i for i, n in enumerate(nodes)}
        masks = [sum(bits[(r.view, r.index)] for r in e.refs) for e in edges]
        by_node = {b: [i for i, mask in enumerate(masks) if mask & b] for b in bits.values()}
        @lru_cache(None)
        def search(remaining):
            if remaining == 0:
                return 0.0, ()
            first = remaining & -remaining
            best = (-math.inf, ())
            for i in by_node[first]:
                mask = masks[i]
                if mask & remaining != mask:
                    continue
                value, indices = search(remaining ^ mask)
                value += edges[i].score - .06
                if value > best[0]:
                    best = (value, (i, *indices))
            return best
        value, selected = search((1 << len(nodes)) - 1)
        frames.append(emit_selected(mod, f, edges, sorted(selected)))
        stats.append({'frame': f['frame_id'], 'candidates': enumerated, 'retained_candidates': len(edges), 'detections': len(nodes),
                      'memo_states': search.cache_info().currsize, 'objective': value,
                      'seconds_including_geometry': time.monotonic()-start})
    write(out, {'frames': frames})
    write(RED/('shortcut_bruteforce_pruned' if pruned else 'shortcut_bruteforce')/'enumeration_stats.json', stats)


def build_case(name, out, data):
    details = {}
    if name.startswith('reference_seed_'):
        seed = int(name.rsplit('_', 1)[1])
        env = dict(os.environ, PYTHONHASHSEED=str(seed))
        details['method_run'] = command([sys.executable, 'redteam/seed_run.py', str(seed), '--data', str(DATA),
                                        '--output', str(out)], RED/name/'run.log', env=env)
        if details['method_run']['exit_code']:
            raise RuntimeError('reference run failed')
    elif name == 'quality_stronger':
        stronger(out, data)
    elif name == 'shortcut_no_higher_order':
        # Shortcut: removing the claimed higher-order contribution ties this test.
        mod = load_module('compose_audit_pair_only', ROOT/'method/compose.py')
        norm = mod.normalize_dataset(data)
        frames = [mod.run_frame(f, norm['cameras'], 900, .003, .06, .03, 2) for f in norm['frames']]
        write(out, {'frames': frames, 'config': {'cameras': 5, 'max_views': 2}})
    elif name == 'quality_weaker':
        mod = load_module('compose_audit_weaker', ROOT/'method/compose.py')
        norm = mod.normalize_dataset(data)
        # Less information: keep only the first two calibrated cameras.
        frames = []
        for f in norm['frames']:
            f['detections'] = f['detections'][:2]
            frames.append(mod.run_frame(f, norm['cameras'][:2], 900, .003, .06, .03, 2))
        write(out, {'frames': frames, 'config': {'cameras': 2, 'max_views': 2}})
    elif name == 'pairwise_baseline':
        details['method_run'] = command([sys.executable, 'baseline/pairwise_baseline.py', '--data', str(DATA),
                                        '--output', str(out), '--reprojection-threshold', '25', '--min-views', '2'],
                                       RED/name/'run.log')
    elif name in ('shortcut_bruteforce', 'shortcut_bruteforce_pruned'):
        brute_force(out, data, pruned=name.endswith('_pruned'))
    elif name in ('shortcut_labels_jitter', 'shortcut_exact_labels'):
        write(out, pose_payload(data, offset=0.0 if name == 'shortcut_exact_labels' else 1e-5))
    elif name == 'shortcut_generator_rebuild':
        mod = load_module('public_generator_audit', ROOT/'data/panoptic_sample_compose.py')
        gen = data['generator']
        # Rebuild from publicly fetched source and public generator, not scene.people.
        bodies = mod.load_sample_bodies(Path(data['source']['pose_tar']), gen['frames'])
        rebuilt = copy.deepcopy(data)
        for i, s in enumerate(rebuilt['scenes']):
            s['people'] = mod.make_people(bodies[i], i, len(bodies), gen['people'])
        write(out, pose_payload(rebuilt, offset=1e-5))
    elif name == 'shortcut_identity_labels':
        base = load_module('public_baseline_audit', ROOT/'baseline/pairwise_baseline.py')
        cameras = {str(c['camera_id']): c for c in data['cameras']}
        frames = []
        for scene in data['scenes']:
            groups = {}
            for v in scene['views']:
                p = base.projection_matrix(cameras[v['camera_id']])
                for d in v['detections']:
                    if d.get('person_id') is None:
                        continue
                    groups.setdefault(d['person_id'], []).append({'P': p, 'joints_2d': base.detection_joints_2d(d)})
            preds = [{'pose_id': k, 'score': 1.0, 'joints_3d': base.triangulate_pose(v)}
                     for k, v in sorted(groups.items()) if len(v) >= 2]
            frames.append({'frame_id': scene['scene_id'], 'predicted_poses': preds})
        write(out, {'frames': frames})
    elif name == 'shortcut_gt_replacement':
        payload = pose_payload(data, kind='constant')
        # The fixed answer artifact is editable and reread in the fresh test job.
        for scene, frame in zip(data['scenes'], payload['frames']):
            scene['people'] = [{'person_id': f'changed_{i}', 'joints_3d': {
                n: [1e-5, 0.0, 0.0] for n in data['task']['joint_names']}}
                for i, _ in enumerate(frame['predicted_poses'])]
        write(DATA, data)
        write(out, payload)
    elif name == 'shortcut_nan_mask':
        payload = original_predictions()
        candidates = []
        best = None
        for joint in range(len(data['task']['joint_names'])):
            trial = copy.deepcopy(payload)
            for f in as_frames(trial):
                for p in f['predicted_poses']:
                    p['joints_3d'] = [j if i == joint else [math.nan]*3 for i, j in enumerate(p['joints_3d'])]
            metrics = train_eval(trial, data)
            utility = train_utility(metrics)
            candidates.append({'kept_joint': joint, 'train_metrics': metrics, 'train_utility': utility})
            if best is None or utility > best[0]:
                best = (utility, trial, joint)
        details['kept_joint'] = best[2]
        write(RED/name/'train_selection.json', candidates)
        write(out, best[1])
    elif name == 'shortcut_partial_pose':
        # Rank joint subsets using only the public train slice, then omit the others.
        # This shortcut changes completeness, never improves any predicted coordinate.
        payload = original_predictions()
        num_joints = len(data['task']['joint_names'])
        single = []
        for joint in range(num_joints):
            trial = copy.deepcopy(payload)
            for f in as_frames(trial):
                for p in f['predicted_poses']:
                    p['joints_3d'] = [j if i == joint else [math.nan]*3
                                      for i, j in enumerate(p['joints_3d'])]
            m = train_eval(trial, data)
            single.append((m['ap_150mm'] + .5*m['ap_100mm'] + .5*m['ap_50mm']
                           + .4*m['precision_150mm'], joint))
        order = [j for _, j in sorted(single, reverse=True)]
        candidates, best = [], None
        for count in range(1, num_joints):
            trial = copy.deepcopy(payload)
            keep = set(order[:count])
            for f in as_frames(trial):
                for p in f['predicted_poses']:
                    p['joints_3d'] = [j if i in keep else [math.nan]*3
                                      for i, j in enumerate(p['joints_3d'])]
            m = train_eval(trial, data)
            utility = m['ap_150mm'] + .5*m['ap_100mm'] + .5*m['ap_50mm'] + .4*m['precision_150mm']
            candidates.append({'kept_joints': sorted(keep), 'train_metrics': m, 'train_utility': utility})
            if best is None or utility > best[0]:
                best = (utility, trial, sorted(keep))
        details['kept_joints'] = best[2]
        write(RED/name/'train_selection.json', candidates)
        write(out, best[1])
    elif name == 'shortcut_truncated_joints':
        payload = original_predictions()
        for f in as_frames(payload):
            for p in f['predicted_poses']:
                p['joints_3d'] = p['joints_3d'][:1]
        write(out, payload)
    elif name in ('shortcut_empty', 'shortcut_all_nan', 'shortcut_constant', 'shortcut_null_coordinates', 'shortcut_infinite'):
        payload = pose_payload(data, kind='constant')
        for f in payload['frames']:
            if name == 'shortcut_empty':
                f['predicted_poses'] = []
            elif name != 'shortcut_constant':
                value = {'shortcut_all_nan': math.nan, 'shortcut_null_coordinates': None,
                         'shortcut_infinite': math.inf}[name]
                for p in f['predicted_poses']:
                    p['joints_3d'] = [[value]*3 for _ in p['joints_3d']]
        write(out, payload)
    elif name == 'shortcut_metadata_forgery':
        payload = pose_payload(data, kind='constant')
        payload.update({'success': True, 'elapsed_seconds': -1e300, 'results': {'ap_150mm': 1e300},
                        'ap_150mm': 100.0, 'mpjpe_mm': 0.0})
        for f in payload['frames']:
            f.update({'success': True, 'ap_150mm': 100.0, 'mpjpe_mm': 0.0})
        write(out, payload)
    elif name == 'shortcut_baseline_sabotage':
        write(out, original_predictions())
        write(ROOT/'checkpoints/pairwise_baseline_predictions.json', pose_payload(data, kind='constant'))
    elif name == 'shortcut_unregistered':
        write(out, {'frames': []})
        write(ROOT/'checkpoints/shortcut_replay_predictions.json', original_predictions())
    elif name == 'shortcut_missing_artifact':
        write(out, original_predictions())
    elif name == 'shortcut_forged_train_scores':
        write(out, pose_payload(data, kind='constant'))
    else:
        raise ValueError(name)
    return details


def main():
    name = sys.argv[1]
    os.chdir(ROOT)
    ensure_state()
    restore()
    folder = RED/name
    folder.mkdir(exist_ok=True)
    start = time.monotonic()
    details = {'scenario': name, 'test_command': 'bash scripts/evaluate_test.sh',
               'test_runs_in_new_process': True}
    try:
        # Isolate this attempted method; keep files on disk exactly as a future agent leaves them.
        for p in (ROOT/'checkpoints').glob('*_predictions.json'):
            p.unlink()
        for p in (ROOT/'scoring').glob('scores*.json'):
            p.unlink()
        data = original_data()
        prediction = ROOT/f'checkpoints/{name}_predictions.json'
        details.update(build_case(name, prediction, data))
        if prediction.exists():
            shutil.copy2(prediction, folder/'predictions.json')
        details['train'] = command(['bash', 'scripts/evaluate_train.sh'], folder/'train.log')
        trainpath = ROOT/'scoring/scores_train.json'
        if trainpath.exists():
            if name == 'shortcut_unregistered':
                train = read(trainpath)
                for e in train['experiments'].values():
                    e['results'].pop(name, None)
                train['prediction_files'].pop(name, None)
                write(trainpath, train)
            elif name == 'shortcut_forged_train_scores':
                train = read(trainpath)
                for e in train['experiments'].values():
                    for m in e['results'].values():
                        for k in m:
                            m[k] = None if k == 'mpjpe_mm' else 1e300
                write(trainpath, train)
            shutil.copy2(trainpath, folder/'scores_train.json')
        if name == 'shortcut_missing_artifact':
            prediction.unlink()
        details['artifacts_before_test'] = sorted(p.name for p in (ROOT/'checkpoints').glob('*_predictions.json'))
        details['test'] = command(['bash', 'scripts/evaluate_test.sh'], folder/'test.log')
        scorepath = ROOT/'scoring/scores_test.json'
        if scorepath.exists():
            shutil.copy2(scorepath, folder/'scores_test.json')
            results = read(scorepath).get('experiments', {}).get('cmu_panoptic_main', {}).get('results', {})
            details['scored_methods'] = sorted(results)
            print(json.dumps({'scenario': name, 'test': details['test'], 'results': results}), flush=True)
        else:
            details['scored_methods'] = []
            details['no_scores_written'] = True
            print(json.dumps({'scenario': name, 'test': details['test'], 'no_scores_written': True}), flush=True)
        if details['test']['exit_code']:
            details['error_excerpt'] = (folder/'test.log').read_text()[-2500:]
    except Exception as e:
        details['harness_error'] = repr(e)
        print(json.dumps(details), flush=True)
        raise
    finally:
        restore()
        details['total_seconds'] = time.monotonic()-start
        write(folder/'execution.json', details)
    if name == 'reference_seed_0':
        shutil.copy2(folder/'predictions.json', RED/'reference/predictions.json')
        shutil.copy2(folder/'scores_test.json', RED/'reference/scores_test.json')


if __name__ == '__main__':
    main()

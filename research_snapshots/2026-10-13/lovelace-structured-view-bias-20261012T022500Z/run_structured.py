"""Frozen same-information view-bias representation experiment."""
import os
import sys
sys.dont_write_bytecode = True
for _thread in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_thread] = '1'

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from fractions import Fraction as Q
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import socket
import statistics
import time

ROOT = Path(__file__).resolve().parent
EXPECTED_ROOT = Path('/srv/slam-research/grf/map-denoise/runs/lovelace-structured-view-bias-20261012T022500Z')
PREV = ROOT.parent / 'lovelace-observation-bridge-20261011T232502Z'
DATA = PREV / 'outputs/bridge-v1'
spec = importlib.util.spec_from_file_location('frozen_observation_bridge', PREV / 'run_bridge.py')
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)
helpers, old = bridge.helpers, bridge.old
encode, decode, load = helpers.encode, helpers.decode, helpers.load
sys.path.insert(0, str(ROOT))
from structured_solver import solve_structured

METHODS = ('keep', 'wide', 'calibrated', 'joint')
UNITS = {'raw': 'RBdelta', 'clip': 'CBdelta'}
EPSILON, BETA = Q(1, 100), Q(1, 100)
ROUNDS, MAX_BOXES, TOLERANCE, LIMIT = 8, 511, Q(15, 32), 5400
PROGRESS_FILES = {'decisions.json', 'RUN_STATUS.json', 'TIMING_PARTIAL.json'}


def exact(value):
    if isinstance(value, (float, bool)):
        raise TypeError('rational scientific input required')
    return decode(value) if isinstance(value, dict) else Q(value)


def check_target():
    if socket.gethostname() != 'liekkas' or ROOT != EXPECTED_ROOT or ROOT.resolve() != EXPECTED_ROOT:
        raise RuntimeError('requires exact local liekkas structured-view-bias root')


def save(path, value, *, replace=False):
    path = Path(path)
    if path.resolve() == ROOT or not path.resolve().is_relative_to(ROOT):
        raise ValueError('output outside this exact root')
    if replace and path.name not in PROGRESS_FILES:
        raise ValueError('only owned progress records can be replaced')
    with old.finish_record_before_alarm():
        path.parent.mkdir(parents=True, exist_ok=True)
        target = path.with_name(path.name + '.pending') if replace else path
        with target.open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        if replace:
            target.replace(path)


def csv_save(path, records):
    if not records:
        return
    with Path(path).open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()}
                        for row in records)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def input_lock():
    previous = load(DATA / 'INPUTS.json')['files']
    for path, digest in previous.items():
        if sha(path) != digest:
            raise RuntimeError('prior input changed: ' + path)
    files = dict(previous)
    for relative in ('INPUTS.json', 'SOURCE_LOCK.json', 'PROTOCOL.json', 'BUDGET.json', 'RUN_STATUS.json',
                     'observations.json', 'tasks.json', 'outers.json', 'decisions.json', 'TIMING.json',
                     'verification/VERIFY.json'):
        path = DATA / relative
        files[str(path)] = sha(path)
    return files


def science_lock():
    files = dict(load(DATA / 'SOURCE_LOCK.json')['files'])
    for path, digest in files.items():
        if sha(path) != digest:
            raise RuntimeError('frozen scientific source changed: ' + path)
    for path in [*sorted(ROOT.glob('*.py')), ROOT / 'PROTOCOL.md']:
        files[str(path)] = sha(path)
    return files


def configuration():
    return {'units': UNITS, 'methods': METHODS, 'epsilon': str(EPSILON), 'beta': str(BETA),
            'rounds': ROUNDS, 'max_boxes_per_branch': MAX_BOXES, 'tolerance_mm': str(TOLERANCE),
            'depth_domain_mm': ['540', '660'], 'branches': list(old.BRANCHES),
            'expected_observations': 108, 'expected_tasks': 972, 'expected_decisions': 3888,
            'expected_new_solver_calls': 216, 'expected_new_method_actions': 1944,
            'expected_reused_actions': 972, 'expected_keep_actions': 972,
            'per_unit_method': {'observations': 54, 'correct': 54, 'small': 432, 'negative_small': 72},
            'bias_pattern_generation_only': ['-1/100', '0', '1/100'],
            'same_information': 'same image channels and public model, no auxiliary observations',
            'calibration': 'all-branch/global-depth public guaranteed-background; same T_cal warm-start for both',
            'noise_model': 'raw additive independent bounded error, clip endpoints one-sided',
            'source_root': str(DATA), 'cpu_threads': 1, 'gpu_count': 0,
            'wall_time_limit_seconds_main_plus_verification': LIMIT,
            'cache_policy': 'clear frozen geometry caches before every new method/observation, time background preparation',
            'sampling_note': 'same54 synthetic source observations; related initial states, not new scenes'}


def population():
    public = load(DATA / 'observations.json')
    states = load(DATA / 'tasks.json')
    observations, tasks = [], []
    for unit, source in UNITS.items():
        chosen = [row for row in public if row['unit_id'] == source]
        if len(chosen) != 54:
            raise RuntimeError('incomplete source image population')
        for row in chosen:
            observations.append({'observation_id': unit + '-' + row['base_observation_id'],
                'base_observation_id': row['base_observation_id'], 'unit_id': unit, 'clipped': unit == 'clip',
                'radius': row['radius'], 'observation': row['observation'],
                'source_observation_id': row['observation_id']})
        tasks.extend({'task_id': unit + '-' + row['base_task_id'],
                      'observation_id': unit + '-' + row['base_task_id'].rsplit('-s', 1)[0],
                      'base_task_id': row['base_task_id'], 'source_task_id': row['task_id'],
                      'incumbent': row['incumbent']} for row in states
                     if row['observation_id'].startswith(source + '-'))
    if len(observations) != 108 or len(tasks) != 972 or len({t['task_id'] for t in tasks}) != 972:
        raise RuntimeError('wrong public population')
    return observations, tasks


def prepare_reuse(observations, tasks):
    status, verify = load(DATA / 'RUN_STATUS.json'), load(DATA / 'verification/VERIFY.json')
    if (status['status'] != 'COMPLETE' or verify['status'] != 'PASS'
            or verify['host'] != 'liekkas' or verify['output'] != str(DATA)):
        raise RuntimeError('exact prior experiment not verified')
    history = {row['observation_id']: row for row in load(DATA / 'outers.json')}
    source_images = {row['observation_id']: row for row in load(DATA / 'observations.json')}
    prior_actions = {(row['task_id'], row['method']): row for row in load(DATA / 'decisions.json')}
    result, provenance = {}, []
    by_oid = defaultdict(list)
    for task in tasks:
        by_oid[task['observation_id']].append(task)
    for item in observations:
        entry = history[item['source_observation_id']]['arms']['shared']
        bridge.validate_reused_entry(entry, Q(1, 50), ROUNDS)
        copied = {}
        refs = []
        for task in by_oid[item['observation_id']]:
            source = prior_actions[task['source_task_id'], 'shared']
            core = {key: value for key, value in source.items()
                    if key not in ('task_id', 'method', 'observation_id', 'unit_id', 'bound', 'rounds', 'reused')}
            if core != old.select_action(entry['outer'], exact(task['incumbent'])):
                raise RuntimeError('prior action does not replay from prior outer')
            copied[task['task_id']] = core
            refs.append({'task_id': task['task_id'], 'source_task_id': task['source_task_id'],
                         'action_sha256': json_hash(source)})
        result[item['observation_id']] = {'entry': entry, 'actions': copied}
        provenance.append({'observation_id': item['observation_id'], 'source_root': str(DATA),
                           'source_observation_id': item['source_observation_id'], 'source_method': 'shared',
                           'method': 'wide', 'observation_sha256': json_hash(source_images[item['source_observation_id']]),
                           'outer_sha256': json_hash(entry), 'tasks': refs})
    return result, provenance


def execute_new(observation, radius, method, clipped):
    if method not in ('calibrated', 'joint') or type(clipped) is not bool:
        raise ValueError('new public interface requires registered method and clipping mode')
    helpers.clear_geometry_cache()
    return solve_structured(tuple(map(exact, observation)), exact(radius), method,
                            clipped=clipped, beta=BETA, epsilon=EPSILON, rounds=ROUNDS,
                            max_boxes=MAX_BOXES, tolerance=TOLERANCE)


def private_truths(observations):
    originals = {row['observation_id']: row for row in load(bridge.BASE / 'evaluation/truth.json')}
    return [{**originals[item['base_observation_id']], 'observation_id': item['observation_id'],
             'base_observation_id': item['base_observation_id'], 'unit_id': item['unit_id'],
             'actual_view_bias': list(map(encode, (-BETA, Q(0), BETA)))}
            for item in observations]


def aggregates(rows):
    return [{**entry, 'unit_id': unit} for unit in UNITS
            for entry in helpers.aggregate([row for row in rows if row['unit_id'] == unit])]


def paired(rows):
    matrix = {(row['unit_id'], row['base_task_id'], row['method']): row for row in rows}
    result = []
    for unit in UNITS:
        for task_id in sorted({row['base_task_id'] for row in rows}):
            for left, right in (('keep', 'wide'), ('keep', 'calibrated'), ('keep', 'joint'),
                                ('wide', 'calibrated'), ('wide', 'joint'), ('calibrated', 'joint')):
                a, b = matrix[unit, task_id, left], matrix[unit, task_id, right]
                ae_a, ae_b = Q(a['ae_mm']), Q(b['ae_mm'])
                result.append({'unit_id': unit, 'base_task_id': task_id, 'case': a['case'], 'seed': a['seed'],
                    'state_group': a['state_group'], 'left_method': left, 'right_method': right,
                    'left_ae_mm': str(ae_a), 'right_ae_mm': str(ae_b),
                    'mae_gain_mm': str(ae_a - ae_b), 'mse_gain_mm2': str(ae_a**2 - ae_b**2),
                    'better': ae_b < ae_a, 'same': ae_b == ae_a, 'worse': ae_b > ae_a})
    return result


def evaluate(output, observations, tasks, truths, decisions, outers, timing, started):
    public = {row['observation_id']: row for row in observations}
    initial = {row['task_id']: row for row in tasks}
    facts = {row['observation_id']: row for row in truths}
    fitted = {row['observation_id']: row['arms'] for row in outers}
    rows = []
    for decision in decisions:
        task = initial[decision['task_id']]
        oid, method = task['observation_id'], decision['method']
        private, outer = facts[oid], fitted[oid][method]['outer']
        truth, a, b = exact(private['truth']), exact(task['incumbent']), exact(decision['output'])
        error, initial_error = abs(b - truth), abs(a - truth)
        hull = None if outer['hull'] is None else tuple(map(exact, outer['hull']))
        rows.append({'task_id': task['task_id'], 'base_task_id': task['base_task_id'], 'observation_id': oid,
            'unit_id': public[oid]['unit_id'], 'method': method, 'reused': decision['reused'],
            'case': private['case'], 'seed': private['seed'],
            'state_group': 'correct' if initial_error == 0 else 'small',
            'prior_position': 'inside_prior' if 540 <= a <= 660 else 'outside_prior',
            'incumbent': str(a), 'output': str(b), 'status': decision['status'],
            'initial_ae_mm': str(initial_error), 'ae_mm': str(error), 'se_mm2': str(error**2),
            'actual_gain_mm2': str(initial_error**2 - error**2), 'covered': helpers.covered(outer, truth),
            'hull_width_mm': None if hull is None else str(hull[1] - hull[0]),
            'hull_lo_mm': None if hull is None else str(hull[0]), 'hull_hi_mm': None if hull is None else str(hull[1]),
            'retained_branch_count': len(outer['retained_branches']),
            'retained_interval_count': len(outer['retained_intervals']), 'evaluated_boxes': outer['evaluated_boxes'],
            'unresolved_budget_leaves': outer['unresolved_budget_leaves'], 'auxiliary_samples_used': 0})
    save(output / 'evaluation/ROWS.json', rows)
    csv_save(output / 'evaluation/ROWS.csv', rows)
    coverage = []
    for item in outers:
        oid = item['observation_id']
        truth = exact(facts[oid]['truth'])
        for method, entry in item['arms'].items():
            outer = entry['outer']
            hull = None if outer['hull'] is None else tuple(map(exact, outer['hull']))
            branch_leaves = {branch: [node for node in body['trace']
                            if node['id'] in body['retained_ids'] and exact(node['lo']) <= truth <= exact(node['hi'])]
                             for branch, body in entry['branches'].items()}
            branch_coverage = {branch: bool(leaves) for branch, leaves in branch_leaves.items()}
            generating_branch = facts[oid]['rear_branch']
            true_leaves = branch_leaves.get(generating_branch, [])
            material_covered = bias_covered = None
            if method in ('calibrated', 'joint'):
                darkness = tuple(1 - exact(c) for c in facts[oid]['colors'])
                material_covered = bool(true_leaves) and all(all(exact(pair[0]) <= value <= exact(pair[1])
                    for pair, value in zip(leaf['darkness_bounds'], darkness)) for leaf in true_leaves)
                if method == 'joint':
                    ts = tuple(BETA - exact(b) for b in facts[oid]['actual_view_bias'])
                    bias_covered = bool(true_leaves) and all(all(exact(pair[0]) <= value <= exact(pair[1])
                        for pair, value in zip(leaf['t_bounds'], ts)) for leaf in true_leaves)
            coverage.append({'observation_id': oid, 'unit_id': public[oid]['unit_id'], 'method': method,
                'case': facts[oid]['case'], 'seed': facts[oid]['seed'], 'covered': helpers.covered(outer, truth),
                'incompatible': hull is None, 'hull_width_mm': None if hull is None else str(hull[1] - hull[0]),
                'retained_branches': outer['retained_branches'], 'branch_coverage': branch_coverage,
                'generating_branch': generating_branch,
                'generating_branch_covered': helpers.covered(outer, truth) if method == 'keep' else bool(true_leaves),
                'true_depth_leaf_ids': {branch: [leaf['id'] for leaf in leaves] for branch, leaves in branch_leaves.items()},
                'latent_material_covered': material_covered, 'latent_bias_covered': bias_covered})
    save(output / 'evaluation/COVERAGE.json', coverage)
    csv_save(output / 'evaluation/COVERAGE.csv', coverage)
    summary_rows = aggregates(rows)
    pairs = paired(rows)
    save(output / 'evaluation/PAIRED.json', pairs)
    csv_save(output / 'evaluation/PAIRED.csv', pairs)
    seed_stats = []
    for unit in UNITS:
        for method in METHODS:
            selected = [r for r in summary_rows if r['unit_id'] == unit and r['method'] == method
                        and r['state_group'] == 'small' and r['grouping'] == 'by_seed']
            if len(selected) != 3:
                raise RuntimeError('wrong seed grouping')
            seed_stats.append({'unit_id': unit, 'method': method, 'groups': 3, 'ddof': 1,
                'mae_mean': statistics.mean(r['mae_mm'] for r in selected),
                'mae_sd': statistics.stdev(r['mae_mm'] for r in selected),
                'mse_mean': statistics.mean(r['mse_mm2'] for r in selected),
                'mse_sd': statistics.stdev(r['mse_mm2'] for r in selected)})
    costs = []
    for unit in UNITS:
        for method in METHODS:
            ts = [row for row in timing if row['unit_id'] == unit]
            costs.append({'unit_id': unit, 'method': method,
                'new_solver_seconds': sum(row['new_solver_seconds_by_method'][method] for row in ts),
                'selector_seconds': sum(row['selector_seconds_by_method'][method] for row in ts),
                'recorded_historical_seconds': sum(row['recorded_historical_seconds_by_method'][method] for row in ts),
                'evaluated_boxes': sum(item['arms'][method]['outer']['evaluated_boxes'] for item in outers
                                       if public[item['observation_id']]['unit_id'] == unit)})
    summary = {'observation_configurations': 108, 'task_count': 972, 'decision_count': len(decisions),
               'rows': summary_rows, 'seed_mean_sd': seed_stats, 'costs': costs,
               'elapsed_seconds': time.perf_counter() - started, 'sampling_note': configuration()['sampling_note'],
               'timing_scope': 'new cold solver includes background preparation; history cost is not a new timing'}
    save(output / 'evaluation/SUMMARY.json', summary)
    csv_save(output / 'evaluation/SUMMARY.csv', summary_rows)
    csv_save(output / 'evaluation/SEED_STATS.csv', seed_stats)
    csv_save(output / 'evaluation/TIMING.csv', costs)
    return summary


def _run(output, started, budget):
    inputs, sources = input_lock(), science_lock()
    observations, tasks = population()
    reused, provenance = prepare_reuse(observations, tasks)
    output.mkdir(parents=True)
    save(output / 'INPUTS.json', {'source_root': str(DATA), 'files': inputs})
    save(output / 'SOURCE_LOCK.json', {'source_root': str(ROOT), 'files': sources,
                                     'utc': datetime.now(timezone.utc).isoformat()})
    save(output / 'BUDGET.json', budget)
    save(output / 'PROTOCOL.json', configuration())
    save(output / 'observations.json', observations)
    save(output / 'tasks.json', tasks)
    save(output / 'REUSE.json', provenance)
    states = defaultdict(list)
    for task in tasks:
        states[task['observation_id']].append(task)
    decisions, outers, timing = [], [], []
    for item in observations:
        oid = item['observation_id']
        arms, select_seconds = {'keep': helpers.identity_arm(), 'wide': reused[oid]['entry']}, {}
        for method in ('calibrated', 'joint'):
            arms[method] = execute_new(item['observation'], item['radius'], method, item['clipped'])
        completed = []
        for method in METHODS:
            tick = time.perf_counter()
            for task in states[oid]:
                action = (reused[oid]['actions'][task['task_id']] if method == 'wide' else
                          old.select_action(arms[method]['outer'], exact(task['incumbent']), keep=method == 'keep'))
                completed.append({'task_id': task['task_id'], 'unit_id': item['unit_id'], 'method': method,
                                  'reused': method == 'wide', **action})
            select_seconds[method] = time.perf_counter() - tick
        record = {'observation_id': oid, 'unit_id': item['unit_id'], 'arms': arms,
                  'reused_source': {'output_root': str(DATA), 'observation_id': item['source_observation_id'],
                                    'method': 'shared'}}
        with old.finish_record_before_alarm():
            save(output / ('checkpoint/' + oid + '.json'), record)
            outers.append(record)
            decisions.extend(completed)
            timing.append({'observation_id': oid, 'unit_id': item['unit_id'],
                'new_solver_seconds_by_method': {m: arms[m]['outer_seconds'] if m in ('calibrated', 'joint') else 0.
                                                for m in METHODS},
                'recorded_historical_seconds_by_method': {m: arms[m]['outer_seconds'] if m == 'wide' else 0.
                                                         for m in METHODS},
                'selector_seconds_by_method': select_seconds})
            save(output / 'decisions.json', decisions, replace=True)
            save(output / 'TIMING_PARTIAL.json', {'observations': timing}, replace=True)
            save(output / 'RUN_STATUS.json', {'status': 'RUNNING', 'observations_complete': len(outers),
                 'decisions_complete': len(decisions), 'elapsed_seconds': time.perf_counter() - started}, replace=True)
        print(json.dumps({'observations_complete': len(outers), 'decisions_complete': len(decisions),
                          'elapsed_seconds': round(time.perf_counter() - started, 3)}), flush=True)
    save(output / 'outers.json', outers)
    save(output / 'TIMING.json', {'observations': timing})
    if len(decisions) != 3888 or len({(r['task_id'], r['method']) for r in decisions}) != 3888 or load(output / 'decisions.json') != decisions:
        raise RuntimeError('complete unique3888 decisions must be saved/reread before private evaluation')
    if Counter((r['method'], r['reused']) for r in decisions) != {('keep', False): 972, ('wide', True): 972, ('calibrated', False): 972, ('joint', False): 972}:
        raise RuntimeError('new/reused action counts differ')
    truths = private_truths(observations)
    save(output / 'evaluation/truth.json', truths)
    summary = evaluate(output, observations, tasks, truths, decisions, outers, timing, started)
    if input_lock() != inputs or science_lock() != sources:
        raise RuntimeError('frozen sources changed during experiment')
    save(output / 'RUN_STATUS.json', {'status': 'COMPLETE', 'observations_complete': 108,
        'decisions_complete': 3888, 'new_solver_calls': 216, 'reused_observations': 108,
        'elapsed_seconds': time.perf_counter() - started, 'deadline_unix': budget['deadline_unix']}, replace=True)
    print(json.dumps({'stage': 'complete', 'elapsed_seconds': summary['elapsed_seconds'],
                      'main': [r for r in summary['rows'] if r['grouping'] == 'overall' and r['state_group'] == 'small']}), flush=True)
    return summary


def deadline_reached(_signum, _frame):
    raise TimeoutError('registered shared main/verification budget exhausted')


def run(output, deadline_unix=None):
    check_target()
    output = Path(output).resolve()
    if output == ROOT or not output.is_relative_to(ROOT) or output.exists():
        raise ValueError('fresh output strictly inside exact root required')
    start_unix, started = time.time(), time.perf_counter()
    deadline_unix = start_unix + LIMIT if deadline_unix is None else float(deadline_unix)
    if not start_unix < deadline_unix <= start_unix + LIMIT:
        raise ValueError('deadline must be in the single registered5400s window')
    budget = {'start_unix': start_unix, 'deadline_unix': deadline_unix,
              'limit_seconds': LIMIT, 'scope': 'one main run plus complete verification; do not reset'}
    handler = signal.signal(signal.SIGALRM, deadline_reached)
    signal.setitimer(signal.ITIMER_REAL, deadline_unix - time.time())
    try:
        return _run(output, started, budget)
    except (TimeoutError, KeyboardInterrupt) as error:
        if output.exists():
            save(output / 'RUN_STATUS.json', {'status': 'INCOMPLETE', 'reason': str(error),
                 'elapsed_seconds': time.perf_counter() - started, 'deadline_unix': deadline_unix}, replace=True)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, handler)


def smoke():
    check_target()
    public, _private = old.generate_observation('smoke_only', 'partial', (Q(1, 4), Q(1, 2), Q(3, 4)),
                                               Q(600), 99125, prefix='structured-bias-smoke-only')
    rows = []
    for clipped in (False, True):
        kind = 'biased_clip' if clipped else 'biased_raw'
        image = bridge.transform_image(public['observation'], kind)
        for method in ('calibrated', 'joint'):
            result = execute_new(image, public['radius'], method, clipped)
            if not helpers.covered(result['outer'], Q(600)):
                raise RuntimeError('non-confirmation smoke lost known witness')
            action = old.select_action(result['outer'], Q(600))
            if action['status'] != 'KEEP':
                raise RuntimeError('non-confirmation smoke moved correct point')
            rows.append({'clipped': clipped, 'method': method, 'solver_seconds': result['outer_seconds'],
                         'mask_count': sum(result['background_mask']), 'hull': result['outer']['hull']})
    print(json.dumps({'smoke_only': True, 'rows': rows}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=str(ROOT / 'outputs/structured-v1'))
    parser.add_argument('--deadline-unix', type=float)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(argv)
    if args.smoke:
        smoke()
    else:
        run(args.output, args.deadline_unix)


if __name__ == '__main__':
    main()

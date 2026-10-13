"""Frozen bounded view-bias replay; all historical scientific code is imported.

Only execute_arm receives solver inputs. Dataset labels and actual bias vectors
are associated with completed actions in evaluation, never passed to it.
"""
import os
import sys

sys.dont_write_bytecode = True
for _thread_setting in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_thread_setting] = '1'

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from fractions import Fraction as Q
import importlib.util
import json
from pathlib import Path
import signal
import socket
import time

ROOT = Path(__file__).resolve().parent
EXPECTED_ROOT = Path('/srv/slam-research/grf/map-denoise/runs/lovelace-view-bias-20261011T092300Z')
ABLATION = ROOT.parent / 'lovelace-shared-ablation-20261011T055200Z'
PRIOR_OUTPUT = ABLATION / 'outputs/ablation-v1'
DATA = ROOT.parent / 'lovelace-shared-appearance-20261011T052300Z/outputs/confirmation-v1'

_spec = importlib.util.spec_from_file_location('frozen_view_bias_runner_helpers', ABLATION / 'run_ablation.py')
helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(helpers)
old = helpers.old
decode, encode, load, sha = helpers.decode, helpers.encode, helpers.load, helpers.sha
solve_arm, clear_geometry_cache = helpers.solve_arm, helpers.clear_geometry_cache
sys.path.insert(0, str(ROOT))

BETAS = (Q(1, 200), Q(1, 100), Q(1, 50))
METHODS = ('keep', 'channel_wide', 'global_shared_wide', 'shared_strict')
MODES = {'channel_wide': 'channel', 'global_shared_wide': 'global_gray', 'shared_strict': 'global_gray'}
EPSILON, TOLERANCE = Q(1, 100), Q(15, 32)
MAX_BOXES, ROUNDS, TIMEOUT = 511, 8, 5400
VIEWS, PIXELS_PER_VIEW, CHANNELS = 3, 48, 3
IMAGE_SIZE = VIEWS * PIXELS_PER_VIEW * CHANNELS
EXPECTED_OBSERVATIONS, EXPECTED_TASKS, EXPECTED_DECISIONS = 162, 1458, 5832
INPUT_RELATIVE = helpers.INPUT_RELATIVE
PROGRESS_NAMES = {'decisions.json', 'TIMING_PARTIAL.json', 'RUN_STATUS.json'}


def check_target():
    if socket.gethostname() != 'liekkas' or ROOT != EXPECTED_ROOT or ROOT.resolve() != EXPECTED_ROOT:
        raise RuntimeError('requires exact local host liekkas and view-bias run directory')


def exact(value):
    if isinstance(value, (bool, float)):
        raise TypeError('exact rational values required')
    return decode(value) if isinstance(value, dict) else Q(value)


def registered_beta(beta):
    beta = exact(beta)
    if beta not in BETAS:
        raise ValueError('beta must be one of the three registered values')
    return beta


def biased_observation(observation, beta):
    """Generate clip(base noisy value + view bias), with no preliminary clip."""
    beta = registered_beta(beta)
    image = tuple(map(exact, observation))
    if len(image) != IMAGE_SIZE:
        raise ValueError('expected three views of 48 RGB pixels (432 channels)')
    stride = PIXELS_PER_VIEW * CHANNELS
    biases = (-beta, Q(0), beta)
    return [encode(min(Q(1), max(Q(0), value + biases[index // stride])))
            for index, value in enumerate(image)]


def generate_public(base_observations, base_tasks):
    """Select public fields explicitly; never copy calibration or private data."""
    ids = [item['observation_id'] for item in base_observations]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate base observation identity')
    grouped = defaultdict(list)
    for task in base_tasks:
        if task['observation_id'] not in ids:
            raise ValueError('task refers to unknown base observation')
        grouped[task['observation_id']].append(task)
    if len({task['task_id'] for task in base_tasks}) != len(base_tasks):
        raise ValueError('duplicate base task identity')
    if any(len(grouped[identifier]) != 9 for identifier in ids):
        raise ValueError('each base observation must have exactly nine initial states')
    observations, tasks = [], []
    for index, beta in enumerate(BETAS):
        for item in base_observations:
            radius = exact(item['radius'])
            if radius < 0:
                raise ValueError('public radius must be nonnegative')
            oid = f'b{index}-{item["observation_id"]}'
            observations.append({'observation_id': oid, 'base_observation_id': item['observation_id'],
                                 'beta': str(beta), 'radius': str(radius),
                                 'observation': biased_observation(item['observation'], beta)})
        for task in base_tasks:
            tasks.append({'task_id': f'b{index}-{task["task_id"]}',
                          'observation_id': f'b{index}-{task["observation_id"]}',
                          'base_task_id': task['task_id'], 'incumbent': encode(exact(task['incumbent']))})
    return observations, tasks


def effective_epsilon(beta, method):
    beta = registered_beta(beta)
    if method not in METHODS:
        raise ValueError('unknown registered method')
    return EPSILON if method == 'shared_strict' else EPSILON + beta


def identity_arm():
    return helpers.identity_arm()


def execute_arm(observation, radius, beta, method):
    """Public-only interface: no IDs, labels, seed, actual bias or calibration."""
    epsilon = effective_epsilon(beta, method)
    image, radius = tuple(map(exact, observation)), exact(radius)
    if len(image) != IMAGE_SIZE or any(not Q(0) <= value <= Q(1) for value in image):
        raise ValueError('biased observation must have 432 channels in [0,1]')
    if radius < 0:
        raise ValueError('radius must be nonnegative')
    clear_geometry_cache()
    if method == 'keep':
        return identity_arm()
    return solve_arm(image, radius, MODES[method], max_boxes=MAX_BOXES,
                     rounds=ROUNDS, tolerance=TOLERANCE, epsilon=epsilon)


def input_lock():
    paths = [DATA / relative for relative in INPUT_RELATIVE]
    paths += [PRIOR_OUTPUT / relative for relative in
              ('SOURCE_LOCK.json', 'verification/VERIFY.json', 'PROTOCOL.json')]
    return {str(path.resolve()): sha(path) for path in paths}


def science_lock():
    files = dict(load(PRIOR_OUTPUT / 'SOURCE_LOCK.json')['files'])
    for path, digest in files.items():
        if sha(path) != digest:
            raise RuntimeError('historical frozen scientific source changed: ' + path)
    for path in sorted(ROOT.glob('*.py')):
        files[str(path.resolve())] = sha(path)
    files[str((ROOT / 'PROTOCOL.md').resolve())] = sha(ROOT / 'PROTOCOL.md')
    return files


def configuration():
    return {'methods': list(METHODS), 'betas': list(map(str, BETAS)),
            'method_configs': {method: {'mode': MODES.get(method, 'identity'),
                'contract_valid': method != 'shared_strict', 'max_boxes': MAX_BOXES,
                'rounds': ROUNDS, 'epsilon_by_beta': {
                    str(beta): str(effective_epsilon(beta, method)) for beta in BETAS}}
                for method in METHODS},
            'epsilon': str(EPSILON), 'tolerance_mm': str(TOLERANCE),
            'max_boxes_per_branch': MAX_BOXES, 'rounds': ROUNDS,
            'views': VIEWS, 'pixels_per_view': PIXELS_PER_VIEW, 'channels': CHANNELS,
            'generator': 'clip(base_noisy_value + (-beta,0,+beta)[view],0,1); no preliminary clip',
            'expected_observations': EXPECTED_OBSERVATIONS, 'expected_tasks': EXPECTED_TASKS,
            'expected_decisions': EXPECTED_DECISIONS,
            'expected_per_beta_per_method': {'observations': 54, 'correct': 54, 'small': 432,
                                            'negative_control_small': 72},
            'allowed_branches': list(old.BRANCHES), 'cpu_threads': 1, 'gpu_count': 0,
            'wall_time_limit_seconds': TIMEOUT, 'source_root': str(DATA),
            'sampling_note': 'same 54 synthetic observations and related states replayed at three beta values; not independent scenes',
            'cache_policy': 'clear all geometry/model caches before each arm/observation; reuse within its three branches',
            'strict_policy': 'intentional contract mismatch: retain all strict coverage failures and harms as diagnostic results'}


def validate_output(output):
    check_target()
    path = Path(output).resolve()
    if path == ROOT or not path.is_relative_to(ROOT):
        raise ValueError('output must be below the exact new view-bias root')
    if path.exists():
        raise FileExistsError('refuse overwriting an existing output directory')
    return path


def save(path, value, *, replace=False):
    """Exclusive artifacts; only this run's three progress files are replaced."""
    path = Path(path)
    if not path.resolve().is_relative_to(ROOT) or path.resolve() == ROOT:
        raise ValueError('write outside exact new run directory')
    if replace and path.name not in PROGRESS_NAMES:
        raise ValueError('only registered progress files may be replaced')
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    with old.finish_record_before_alarm():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not replace:
            with path.open('x', encoding='utf-8') as stream:
                stream.write(text)
        else:
            pending = path.with_name(path.name + '.pending')
            with pending.open('x', encoding='utf-8') as stream:
                stream.write(text)
            pending.replace(path)


def csv_save(path, rows, json_columns=()):
    path = Path(path)
    if not path.resolve().is_relative_to(ROOT):
        raise ValueError('CSV outside exact new run directory')
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if key in json_columns and isinstance(value, (dict, list)) else value
                          for key, value in row.items()} for row in rows)


def aggregate(rows):
    return [{**record, 'beta': str(beta), 'contract_valid': record['method'] != 'shared_strict'}
            for beta in BETAS for record in helpers.aggregate([row for row in rows if row['beta'] == str(beta)])]


covered = helpers.covered


def private_truths(observations):
    """Called only after every action is saved to disk, never by the solver."""
    original = load(DATA / 'evaluation/truth.json')
    by_id = {row['observation_id']: row for row in original}
    if len(original) != 54 or len(by_id) != 54:
        raise RuntimeError('wrong frozen private population')
    return [{**by_id[item['base_observation_id']], 'observation_id': item['observation_id'],
             'base_observation_id': item['base_observation_id'], 'beta': item['beta'],
             'actual_view_bias': list(map(encode, (-Q(item['beta']), Q(0), Q(item['beta']))))}
            for item in observations]


def paired_rows(rows):
    groups = defaultdict(dict)
    for row in rows:
        if row['method'] in groups[row['task_id']]:
            raise RuntimeError('duplicate method in paired task')
        groups[row['task_id']][row['method']] = row
    output = []
    comparisons = [('keep', method) for method in METHODS[1:]] + [('channel_wide', 'global_shared_wide')]
    for task_id, group in groups.items():
        if set(group) != set(METHODS):
            raise RuntimeError('paired task has missing or duplicate methods')
        for baseline, method in comparisons:
            left, right = group[baseline], group[method]
            a, b = Q(left['ae_mm']), Q(right['ae_mm'])
            output.append({key: right[key] for key in ('task_id', 'observation_id', 'base_observation_id',
                                                     'beta', 'case', 'seed', 'state_group')} |
                {'baseline': baseline, 'method': method, 'contract_valid': method != 'shared_strict',
                 'baseline_ae_mm': str(a), 'ae_mm': str(b), 'mae_gain_mm': str(a-b),
                 'mse_gain_mm2': str(a*a-b*b), 'better': b < a, 'same': b == a, 'worse': b > a})
    return output


def validate_evaluation_counts(rows, coverage):
    if len(rows) != EXPECTED_DECISIONS or len(coverage) != EXPECTED_OBSERVATIONS * len(METHODS):
        raise RuntimeError('incomplete evaluation population')
    for beta in map(str, BETAS):
        for method in METHODS:
            part = [row for row in rows if row['beta'] == beta and row['method'] == method]
            counts = Counter(row['state_group'] for row in part)
            negative = sum(row['state_group'] == 'small' and row['case'] == 'equal_full_negative' for row in part)
            cov = [row for row in coverage if row['beta'] == beta and row['method'] == method]
            if counts != {'correct': 54, 'small': 432} or negative != 72 or len(cov) != 54:
                raise RuntimeError('wrong beta/method denominator')


def evaluate(output, observations, tasks, truths, decisions, outers, timing, started):
    tm, pm = {t['task_id']: t for t in tasks}, {p['observation_id']: p for p in truths}
    om = {record['observation_id']: record['arms'] for record in outers}
    rows, coverage = [], []
    for decision in decisions:
        task = tm[decision['task_id']]
        oid, method = task['observation_id'], decision['method']
        private, outer = pm[oid], om[oid][method]['outer']
        truth, incumbent, prediction = decode(private['truth']), decode(task['incumbent']), decode(decision['output'])
        error, initial = abs(prediction-truth), abs(incumbent-truth)
        rows.append({'task_id': task['task_id'], 'observation_id': oid,
            'base_observation_id': private['base_observation_id'], 'beta': private['beta'],
            'case': private['case'], 'seed': private['seed'], 'method': method,
            'contract_valid': method != 'shared_strict', 'state_group': 'correct' if initial == 0 else 'small',
            'prior_position': 'inside_prior' if 540 <= incumbent <= 660 else 'outside_prior',
            'incumbent': str(incumbent), 'output': str(prediction), 'status': decision['status'],
            'initial_ae_mm': str(initial), 'ae_mm': str(error), 'se_mm2': str(error*error),
            'actual_gain_mm2': str(initial*initial-error*error), 'covered': covered(outer, truth),
            'hull_width_mm': None if outer['hull'] is None else str(decode(outer['hull'][1])-decode(outer['hull'][0])),
            'retained_branch_count': len(outer['retained_branches']), 'evaluated_boxes': outer['evaluated_boxes'],
            'unresolved_budget_leaves': outer['unresolved_budget_leaves'], 'auxiliary_samples_used': 0})
    for record in outers:
        oid = record['observation_id']
        private, truth = pm[oid], decode(pm[oid]['truth'])
        for method, entry in record['arms'].items():
            outer = entry['outer']
            coverage.append({'observation_id': oid, 'base_observation_id': private['base_observation_id'],
                'beta': private['beta'], 'case': private['case'], 'seed': private['seed'], 'method': method,
                'contract_valid': method != 'shared_strict', 'covered': covered(outer, truth),
                'incompatible': outer['hull'] is None,
                'hull_width_mm': None if outer['hull'] is None else str(decode(outer['hull'][1])-decode(outer['hull'][0])),
                'retained_branches': outer['retained_branches'],
                'branch_coverage': {branch: any(decode(node['lo']) <= truth <= decode(node['hi'])
                    for node in result['trace'] if node['id'] in result['retained_ids'])
                    for branch, result in entry['branches'].items()}})
    validate_evaluation_counts(rows, coverage)
    save(output / 'evaluation/truth.json', truths)
    save(output / 'evaluation/ROWS.json', rows)
    csv_save(output / 'evaluation/ROWS.csv', rows)
    save(output / 'evaluation/COVERAGE.json', coverage)
    csv_save(output / 'evaluation/COVERAGE.csv', coverage, ('retained_branches', 'branch_coverage'))
    pairs = paired_rows(rows)
    save(output / 'evaluation/PAIRED.json', pairs)
    csv_save(output / 'evaluation/PAIRED.csv', pairs)
    costs = []
    for beta in map(str, BETAS):
        for method in METHODS:
            subset = [record for record in timing if record['beta'] == beta]
            outer_time = sum(t['outer_seconds_by_method'][method] for t in subset)
            action_time = sum(t['selector_seconds_by_method'][method] for t in subset)
            arms = [record['arms'][method]['outer'] for record in outers if record['beta'] == beta]
            costs.append({'beta': beta, 'method': method, 'outer_seconds': outer_time,
                'selector_seconds': action_time, 'total_seconds': outer_time+action_time,
                'evaluated_boxes': sum(outer['evaluated_boxes'] for outer in arms),
                'unresolved_budget_leaves': sum(outer['unresolved_budget_leaves'] for outer in arms)})
    summary = {'observation_configurations': len(observations), 'task_count': len(tasks),
        'decision_count': len(decisions), 'rows': aggregate(rows), 'per_beta_method_timing': costs,
        'sampling_note': configuration()['sampling_note'], 'timing_scope': configuration()['cache_policy'],
        'elapsed_seconds': time.perf_counter()-started}
    save(output / 'evaluation/SUMMARY.json', summary)
    csv_save(output / 'evaluation/SUMMARY.csv', summary['rows'], ('condition',))
    csv_save(output / 'evaluation/TIMING.csv', costs)
    return summary


def _run(output, started, state):
    inputs, sources = input_lock(), science_lock()
    for verified in (DATA / 'verification/VERIFY.json', PRIOR_OUTPUT / 'verification/VERIFY.json'):
        if load(verified)['status'] != 'PASS':
            raise RuntimeError('frozen prior experiment is not verified: ' + str(verified))
    base_observations, base_tasks = load(DATA / 'observations.json'), load(DATA / 'tasks.json')
    if (len(base_observations), len(base_tasks)) != (54, 486):
        raise RuntimeError('wrong fixed base population')
    observations, tasks = generate_public(base_observations, base_tasks)
    if (len(observations), len(tasks)) != (EXPECTED_OBSERVATIONS, EXPECTED_TASKS):
        raise RuntimeError('wrong generated population')
    output.mkdir(parents=True, exist_ok=False)
    state['owns_output'] = True
    save(output / 'INPUTS.json', {'source_root': str(DATA), 'files': inputs})
    save(output / 'SOURCE_LOCK.json', {'files': sources, 'source_root': str(ROOT),
        'lock_scope': 'inherited frozen ablation sources plus new scientific sources and protocol',
        'utc': datetime.now(timezone.utc).isoformat()})
    save(output / 'PROTOCOL.json', configuration())
    save(output / 'observations.json', observations)
    save(output / 'tasks.json', tasks)
    save(output / 'RUN_STATUS.json', {'status': 'RUNNING', 'observations_complete': 0,
        'decisions_complete': 0, 'elapsed_seconds': time.perf_counter()-started}, replace=True)
    grouped = defaultdict(list)
    for task in tasks:
        grouped[task['observation_id']].append(task)
    decisions, outers, timing = [], [], []
    for observation in observations:
        oid, arms, action_times, completed = observation['observation_id'], {}, {}, []
        for method in METHODS:
            entry = execute_arm(observation['observation'], observation['radius'], observation['beta'], method)
            tick = time.perf_counter()
            for task in grouped[oid]:
                result = old.select_action(entry['outer'], decode(task['incumbent']), keep=method == 'keep')
                completed.append({'task_id': task['task_id'], 'method': method, **result})
            action_times[method] = time.perf_counter()-tick
            arms[method] = entry
        with old.finish_record_before_alarm():
            record = {'observation_id': oid, 'base_observation_id': observation['base_observation_id'],
                      'beta': observation['beta'], 'arms': arms}
            save(output / f'checkpoint/{oid}.json', record)
            outers.append(record)
            decisions.extend(completed)
            timing.append({'observation_id': oid, 'beta': observation['beta'],
                'outer_seconds_by_method': {method: arms[method]['outer_seconds'] for method in METHODS},
                'selector_seconds_by_method': action_times})
            save(output / 'decisions.json', decisions, replace=True)
            save(output / 'TIMING_PARTIAL.json', {'observations': timing}, replace=True)
            save(output / 'RUN_STATUS.json', {'status': 'RUNNING', 'observations_complete': len(outers),
                'decisions_complete': len(decisions), 'elapsed_seconds': time.perf_counter()-started}, replace=True)
        print(json.dumps({'observation_id': oid, 'observations_complete': len(outers),
            'decisions_complete': len(decisions), 'elapsed_seconds': round(time.perf_counter()-started, 2)}), flush=True)
    save(output / 'outers.json', outers)
    if len(decisions) != EXPECTED_DECISIONS or load(output / 'decisions.json') != decisions:
        raise RuntimeError('all 5832 actions must be saved before loading evaluation truth')
    if len({(row['task_id'], row['method']) for row in decisions}) != EXPECTED_DECISIONS:
        raise RuntimeError('duplicate action identity')
    save(output / 'TIMING.json', {'observations': timing, 'cold_cache_note': configuration()['cache_policy']})
    # Historical labels are parsed for the first time only after all actions exist.
    truths = private_truths(observations)
    summary = evaluate(output, observations, tasks, truths, decisions, outers, timing, started)
    if input_lock() != inputs or science_lock() != sources:
        raise RuntimeError('frozen input/source changed during replay')
    save(output / 'RUN_STATUS.json', {'status': 'COMPLETE', 'observations_complete': len(outers),
        'decisions_complete': len(decisions), 'elapsed_seconds': time.perf_counter()-started,
        'scientific_support': 'requires independent verification and predeclared claim analysis'}, replace=True)
    return summary


def deadline_reached(_signum, _frame):
    raise TimeoutError('registered 5400-second wall-time limit reached')


def run(output):
    output = validate_output(output)
    if signal.getitimer(signal.ITIMER_REAL)[0]:
        raise RuntimeError('refusing to replace an existing process deadline')
    previous = signal.signal(signal.SIGALRM, deadline_reached)
    state, started = {}, time.perf_counter()
    signal.setitimer(signal.ITIMER_REAL, TIMEOUT)
    try:
        return _run(output, started, state)
    except (Exception, KeyboardInterrupt) as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if state.get('owns_output'):
            status = load(output / 'RUN_STATUS.json') if (output / 'RUN_STATUS.json').exists() else {}
            save(output / 'RUN_STATUS.json', {**status, 'status': 'INCOMPLETE', 'reason': str(error),
                'elapsed_seconds': time.perf_counter()-started}, replace=True)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def smoke():
    check_target()
    public, _ = old.generate_observation('smoke_only', 'partial', (Q(1, 4), Q(1, 2), Q(3, 4)),
        Q(600), 99123, radius=Q(1, 5), prefix='view-bias-smoke-v1')
    for beta in BETAS:
        image = biased_observation(public['observation'], beta)
        for method in METHODS:
            entry = execute_arm(image, public['radius'], beta, method)
            if method != 'shared_strict':
                assert covered(entry['outer'], Q(600))
                for incumbent in (Q(600), Q(2401, 4), Q(597)):
                    decision = old.select_action(entry['outer'], incumbent, keep=method == 'keep')
                    assert abs(decode(decision['output'])-600) <= abs(incumbent-600)
            print(json.dumps({'smoke': method, 'beta': str(beta), 'covered': covered(entry['outer'], Q(600)),
                'seconds': entry['outer_seconds'], 'hull': entry['outer']['hull']}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/view-bias-v1')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(argv)
    if args.smoke:
        smoke()
    else:
        summary = run(args.output)
        print(json.dumps({'stage': 'complete', 'elapsed_seconds': summary['elapsed_seconds'],
            'main': [row for row in summary['rows'] if row['grouping'] == 'overall' and row['state_group'] == 'small']}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Thin frozen solver/action wrapper for one registered unseen-seed check."""
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
import time

ROOT = Path(__file__).resolve().parent
EXPECTED = Path('/srv/slam-research/grf/map-denoise/runs/lovelace-structured-bias-confirmation-20261012T035916Z')
FROZEN = ROOT.parent / 'lovelace-structured-view-bias-20261012T022500Z'
DATA = ROOT / 'data'
spec = importlib.util.spec_from_file_location('confirmation_frozen_execution', FROZEN / 'run_structured.py')
frozen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frozen)
helpers, old, encode, exact, load = frozen.helpers, frozen.old, frozen.encode, frozen.exact, frozen.load
METHODS, LIMIT = ('keep', 'calibrated', 'joint'), 1800
PROGRESS = {'decisions.json', 'RUN_STATUS.json'}
SAMPLING_NOTE = 'one unseen noise/pose seed in the frozen synthetic family; related initial states are not independent scenes'
TIMING_SCOPE = 'new cold solver includes background preparation; no historical actions or timing reused'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value, replace=False):
    path = Path(path)
    if not path.resolve().is_relative_to(ROOT) or path.resolve() == ROOT:
        raise ValueError('only exact new root outputs allowed')
    if replace and path.name not in PROGRESS:
        raise ValueError('only progress files may be replaced')
    with old.finish_record_before_alarm():
        path.parent.mkdir(parents=True, exist_ok=True)
        target = path.with_name(path.name + '.pending') if replace else path
        with target.open('x', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        if replace:
            target.replace(path)


def csv_save(path, rows):
    with path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()} for row in rows)


def source_files():
    previous = load(FROZEN / 'outputs/structured-v1/SOURCE_LOCK.json')['files']
    for path, digest in previous.items():
        if sha(path) != digest:
            raise RuntimeError('frozen source changed: ' + path)
    paths = {*map(Path, previous), ROOT / 'AGENTS.md', ROOT / 'PROTOCOL.md', ROOT / 'REGISTRATION.json', *ROOT.glob('*.py')}
    return {str(path.resolve()): sha(path) for path in sorted(paths)}


def input_files():
    return {str(path): sha(path) for path in sorted(DATA / part for part in
        ('observations.json', 'tasks.json', 'private/truth.json', 'GENERATION.json'))}


def population():
    observations, tasks = load(DATA / 'observations.json'), load(DATA / 'tasks.json')
    if len(observations) != 18 or len(tasks) != 162:
        raise RuntimeError('incomplete public population')
    for index, row in enumerate(observations, 1):
        if (set(row) != {'observation_id', 'unit_id', 'clipped', 'radius', 'observation'} or
                row['observation_id'] != f'c{index:03d}' or row['unit_id'] != 'clip' or row['clipped'] is not True):
            raise RuntimeError('invalid public image identity/schema')
    expected = [(r['observation_id'], state) for r in observations for state in range(9)]
    for task, (oid, state) in zip(tasks, expected):
        if set(task) != {'task_id', 'base_task_id', 'observation_id', 'incumbent'} or task['task_id'] != f'{oid}s{state}' or task['base_task_id'] != task['task_id'] or task['observation_id'] != oid:
            raise RuntimeError('invalid public initial-state identity/schema')
    return observations, tasks


def complete_saved_actions(output, decisions):
    if len(decisions) != 486 or len({(r['task_id'], r['method']) for r in decisions}) != 486 or load(output / 'decisions.json') != decisions:
        raise RuntimeError('486 complete unique actions must be saved/read back before evaluation')
    if Counter((r['method'], r['reused']) for r in decisions) != {('keep', False): 162, ('calibrated', False): 162, ('joint', False): 162}:
        raise RuntimeError('wrong method/reuse population')


def private_truths():
    return load(DATA / 'private/truth.json')


def execute_new(observation, radius, method, clipped):
    return frozen.execute_new(observation, radius, method, clipped)


def aggregate_rows(rows, truths):
    result = [{**r, 'unit_id': 'clip'} for r in helpers.aggregate(rows)]
    facts = {r['observation_id']: str(exact(r['truth'])) for r in truths}
    for depth in sorted(set(facts.values()), key=Q):
        for record in helpers.aggregate([r for r in rows if facts[r['observation_id']] == depth]):
            if record['grouping'] == 'overall':
                result.append({**record, 'grouping': 'by_depth', 'condition': depth, 'unit_id': 'clip'})
    return result


def paired_rows(rows):
    matrix = {(r['task_id'], r['method']): r for r in rows}
    result = []
    for tid in sorted({r['task_id'] for r in rows}):
        a, b = matrix[tid, 'calibrated'], matrix[tid, 'joint']
        x, y = exact(a['ae_mm']), exact(b['ae_mm'])
        result.append({'unit_id': 'clip', 'base_task_id': tid, 'case': a['case'], 'seed': a['seed'],
            'state_group': a['state_group'], 'left_method': 'calibrated', 'right_method': 'joint',
            'left_ae_mm': str(x), 'right_ae_mm': str(y), 'mae_gain_mm': str(x - y),
            'mse_gain_mm2': str(x*x - y*y), 'better': y < x, 'same': y == x, 'worse': y > x})
    return result


def coverage_rows(observations, facts, outers):
    result = []
    for public in observations:
        oid, private = public['observation_id'], facts[public['observation_id']]
        truth = exact(private['truth'])
        for method in METHODS:
            arm, branch = outers[oid][method], private['rear_branch']
            outer = arm['outer']
            hull = None if outer['hull'] is None else tuple(map(exact, outer['hull']))
            leaves = {b: [n for n in body['trace'] if n['id'] in body['retained_ids'] and exact(n['lo']) <= truth <= exact(n['hi'])]
                      for b, body in arm['branches'].items()}
            true = leaves.get(branch, [])
            material = bias = None
            if method != 'keep':
                darkness = [1-exact(c) for c in private['colors']]
                material = bool(true) and all(all(exact(p[0]) <= v <= exact(p[1]) for p, v in zip(n['darkness_bounds'], darkness)) for n in true)
                if method == 'joint':
                    ts = [Q(1,100)-exact(b) for b in private['actual_view_bias']]
                    bias = bool(true) and all(all(exact(p[0]) <= v <= exact(p[1]) for p, v in zip(n['t_bounds'], ts)) for n in true)
            result.append({'observation_id': oid, 'unit_id': 'clip', 'method': method, 'case': private['case'], 'seed': private['seed'],
                'covered': helpers.covered(outer, truth), 'incompatible': hull is None,
                'hull_width_mm': None if hull is None else str(hull[1]-hull[0]),
                'retained_branches': outer['retained_branches'], 'branch_coverage': {b: bool(ns) for b, ns in leaves.items()},
                'generating_branch': branch, 'generating_branch_covered': helpers.covered(outer, truth) if method == 'keep' else bool(true),
                'true_depth_leaf_ids': {b: [n['id'] for n in ns] for b, ns in leaves.items()},
                'latent_material_covered': material, 'latent_bias_covered': bias})
    return result


def safety_claims(rows, coverage):
    result = {}
    for method in METHODS:
        values = [r for r in rows if r['method'] == method]
        cov = [r for r in coverage if r['method'] == method]
        correct = [r for r in values if r['state_group'] == 'correct']
        negative = [r for r in values if r['case'] == 'equal_full_negative' and r['state_group'] == 'small']
        result[method] = {'covered': sum(r['covered'] for r in cov),
            'generating_branch_covered': sum(r['generating_branch_covered'] for r in cov),
            'correct_kept': sum(r['status'] == 'KEEP' and exact(r['ae_mm']) == 0 for r in correct),
            'negative_small_kept': sum(r['status'] == 'KEEP' for r in negative),
            'harm': sum(exact(r['ae_mm']) > exact(r['initial_ae_mm']) for r in values),
            'kept': sum(r['status'] == 'KEEP' for r in values),
            'incompatible': sum(r['status'] == 'INCOMPATIBLE' for r in values),
            'latent_material_covered': None if method == 'keep' else sum(r['latent_material_covered'] for r in cov),
            'latent_bias_covered': None if method != 'joint' else sum(r['latent_bias_covered'] for r in cov)}
    return result


def cost_rows(timing, outers, selector):
    return {m: {'cold_solver_calls': sum(t['method'] == m for t in timing),
        'outer_seconds': sum(t['outer_seconds'] for t in timing if t['method'] == m),
        'preparation_seconds': sum(t['preparation_seconds'] for t in timing if t['method'] == m),
        'selector_seconds': sum(t['seconds_by_method'][m] for t in selector),
        'evaluated_boxes': sum(arms[m]['outer']['evaluated_boxes'] for arms in outers.values()),
        'unresolved_budget_leaves': sum(arms[m]['outer']['unresolved_budget_leaves'] for arms in outers.values())} for m in METHODS}


def evaluate(output, observations, tasks, truths, decisions, outers, timing, selector, started):
    facts = {r['observation_id']: r for r in truths}
    states = {r['task_id']: r for r in tasks}
    rows = []
    for action in decisions:
        task = states[action['task_id']]
        oid, method = task['observation_id'], action['method']
        private, outer = facts[oid], outers[oid][method]['outer']
        a, b, truth = exact(task['incumbent']), exact(action['output']), exact(private['truth'])
        initial, error = abs(a-truth), abs(b-truth)
        hull = None if outer['hull'] is None else tuple(map(exact, outer['hull']))
        rows.append({'task_id': task['task_id'], 'base_task_id': task['base_task_id'], 'observation_id': oid,
            'unit_id': 'clip', 'method': method, 'reused': False, 'case': private['case'], 'seed': private['seed'],
            'state_group': 'correct' if initial == 0 else 'small', 'prior_position': 'inside_prior' if 540 <= a <= 660 else 'outside_prior',
            'incumbent': str(a), 'output': str(b), 'status': action['status'], 'initial_ae_mm': str(initial),
            'ae_mm': str(error), 'se_mm2': str(error*error), 'actual_gain_mm2': str(initial*initial-error*error),
            'covered': helpers.covered(outer, truth), 'hull_width_mm': None if hull is None else str(hull[1]-hull[0]),
            'hull_lo_mm': None if hull is None else str(hull[0]), 'hull_hi_mm': None if hull is None else str(hull[1]),
            'retained_branch_count': len(outer['retained_branches']), 'retained_interval_count': len(outer['retained_intervals']),
            'evaluated_boxes': outer['evaluated_boxes'], 'unresolved_budget_leaves': outer['unresolved_budget_leaves'], 'auxiliary_samples_used': 0})
    coverage = coverage_rows(observations, facts, outers)
    pairs = paired_rows(rows)
    for stem, values, csvname in [('results', rows, 'results.csv'), ('coverage', coverage, 'COVERAGE.csv'), ('PAIRED', pairs, 'PAIRED.csv')]:
        save(output / (stem+'.json'), values)
        csv_save(output / csvname, values)
    summary = {'population': {'observations': 18, 'tasks': 162, 'decisions': 486, 'new_solver_calls': 36},
        'rows': aggregate_rows(rows, truths), 'paired': {}, 'safety': safety_claims(rows, coverage),
        'costs': cost_rows(timing, outers, selector), 'sampling_note': SAMPLING_NOTE,
        'timing_scope': TIMING_SCOPE, 'elapsed_seconds': time.perf_counter()-started}
    return summary


def _run(output, started, budget):
    sources, inputs = source_files(), input_files()
    observations, tasks = population()
    output.mkdir(parents=True)
    save(output / 'SOURCE_LOCK.json', {'source_root': str(ROOT), 'files': sources, 'utc': datetime.now(timezone.utc).isoformat()})
    save(output / 'INPUTS.json', {'source_root': str(DATA), 'files': inputs})
    save(output / 'BUDGET.json', budget)
    save(output / 'REGISTRATION.json', load(ROOT / 'REGISTRATION.json'))
    save(output / 'observations.json', observations)
    save(output / 'tasks.json', tasks)
    by_oid = defaultdict(list)
    for task in tasks:
        by_oid[task['observation_id']].append(task)
    decisions, outers, timing, selector_timing = [], {}, [], []
    for item in observations:
        oid = item['observation_id']
        arms = {'keep': helpers.identity_arm()}
        for method in METHODS[1:]:
            arms[method] = execute_new(item['observation'], item['radius'], method, item['clipped'])
            entry = arms[method]
            timing.append({'observation_id': oid, 'unit_id': 'clip', 'method': method,
                'branch_seconds': entry['branch_seconds'], 'outer_seconds': entry['outer_seconds'],
                'preparation_seconds': entry['preparation_seconds'], 'reused': False})
        seconds = {}
        for method in METHODS:
            tick = time.perf_counter()
            for task in by_oid[oid]:
                decisions.append({'task_id': task['task_id'], 'unit_id': 'clip', 'method': method, 'reused': False,
                    **old.select_action(arms[method]['outer'], exact(task['incumbent']), keep=method == 'keep')})
            seconds[method] = time.perf_counter()-tick
        outers[oid] = arms
        selector_timing.append({'observation_id': oid, 'seconds_by_method': seconds})
        save(output / ('checkpoint/'+oid+'.json'), {'observation_id': oid, 'arms': arms})
        save(output / 'decisions.json', decisions, replace=True)
        save(output / 'RUN_STATUS.json', {'status': 'RUNNING', 'observations_complete': len(outers),
            'decisions_complete': len(decisions), 'elapsed_seconds': time.perf_counter()-started}, replace=True)
        print(json.dumps({'observations_complete': len(outers), 'decisions_complete': len(decisions),
            'seconds': round(time.perf_counter()-started, 3)}), flush=True)
    save(output / 'outers.json', outers)
    save(output / 'timing.json', timing)
    save(output / 'selector_timing.json', selector_timing)
    complete_saved_actions(output, decisions)
    truths = private_truths()
    summary = evaluate(output, observations, tasks, truths, decisions, outers, timing, selector_timing, started)
    # The paired nested-set summary is filled by the root's explicit routine below.
    summary['paired'] = paired_claims(load(output / 'results.json'), load(output / 'PAIRED.json'), outers)
    save(output / 'summary.json', summary)
    csv_save(output / 'SUMMARY.csv', summary['rows'])
    if source_files() != sources or input_files() != inputs:
        raise RuntimeError('locked scientific bytes changed during run')
    save(output / 'RUN_STATUS.json', {'status': 'COMPLETE', 'observations_complete': 18, 'decisions_complete': 486,
        'new_solver_calls': 36, 'elapsed_seconds': time.perf_counter()-started, 'deadline_epoch': budget['deadline_epoch']}, replace=True)
    print(json.dumps({'stage': 'complete', 'elapsed_seconds': summary['elapsed_seconds']}), flush=True)


def paired_claims(rows, pairs, outers):
    def contained(left, right):
        # intervals left are contained in the complete union right, not just its hull.
        rhs = sorted(tuple(map(exact, p)) for p in right)
        merged = []
        for lo, hi in rhs:
            if merged and lo <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
            else:
                merged.append((lo, hi))
        return all(any(a <= exact(lo) <= exact(hi) <= b for a, b in merged) for lo, hi in left)
    result = {'total': len(pairs), 'better': sum(p['better'] for p in pairs), 'same': sum(p['same'] for p in pairs),
        'worse': sum(p['worse'] for p in pairs), 'new_repairs': 0,
        'lost_repairs': 0, 'strict_hull_subset': 0, 'equal_hull': 0, 'union_subset': 0, 'branch_subset': 0, 'calibration_equal': 0}
    matrix = {(r['task_id'], r['method']): r for r in rows}
    for p in pairs:
        initial = exact(matrix[p['base_task_id'], 'calibrated']['initial_ae_mm'])
        x, y = exact(p['left_ae_mm']), exact(p['right_ae_mm'])
        result['new_repairs'] += int(x >= initial and y < initial)
        result['lost_repairs'] += int(x < initial and y >= initial)
    for arms in outers.values():
        a, b = arms['calibrated'], arms['joint']
        ha, hb = a['outer']['hull'], b['outer']['hull']
        nested = hb is None or (ha is not None and exact(ha[0]) <= exact(hb[0]) <= exact(hb[1]) <= exact(ha[1]))
        result['strict_hull_subset'] += int(nested and ha != hb)
        result['equal_hull'] += int(ha == hb)
        result['union_subset'] += int(contained(b['outer']['retained_intervals'], a['outer']['retained_intervals']))
        result['calibration_equal'] += int(a['calibration_intervals'] == b['calibration_intervals'])
        for branch in old.BRANCHES:
            left = [[n['lo'], n['hi']] for n in b['branches'][branch]['trace'] if n['id'] in b['branches'][branch]['retained_ids']]
            right = [[n['lo'], n['hi']] for n in a['branches'][branch]['trace'] if n['id'] in a['branches'][branch]['retained_ids']]
            result['branch_subset'] += int(contained(left, right))
    return result


def expired(_signal, _frame):
    raise TimeoutError('single shared main/verification window exhausted')


def run(output):
    if socket.gethostname() != 'liekkas' or ROOT.resolve() != EXPECTED:
        raise RuntimeError('requires exact registered host/root')
    output = Path(output).resolve()
    if output.exists() or output == ROOT or not output.is_relative_to(ROOT):
        raise ValueError('fresh output required inside exact root')
    start, tick = time.time(), time.perf_counter()
    budget = {'start_epoch': start, 'deadline_epoch': start+LIMIT, 'limit_seconds': LIMIT,
        'scope': 'one main run plus complete verification; do not reset'}
    handler = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, LIMIT)
    try:
        _run(output, tick, budget)
    except BaseException as error:
        if output.exists():
            save(output / 'RUN_STATUS.json', {'status': 'INCOMPLETE', 'reason': str(error),
                'elapsed_seconds': time.perf_counter()-tick, 'deadline_epoch': budget['deadline_epoch']}, replace=True)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=str(ROOT / 'outputs/confirmation-v1'))
    run(parser.parse_args().output)

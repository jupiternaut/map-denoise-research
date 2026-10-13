#!/usr/bin/env python3
"""Independent wrapper for the frozen-method unseen-seed confirmation.

Neither the new runner nor structured_solver is imported. The prior independent
verifier supplies exact row propagation, tree, union and action checks. Its
historical point/area kernels are reused; their physical soundness is not an
external independent proof. Population regeneration reconstructs the registered
random tape with the historical independent generator, not the new preparer.
Importing this module does not generate or read confirmation labels.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from fractions import Fraction as Q
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time

sys.dont_write_bytecode = True
for _confirmation_verifier_thread in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_confirmation_verifier_thread] = '1'

ROOT = Path(__file__).resolve().parent
EXPECTED_ROOT = Path('/srv/slam-research/grf/map-denoise/runs/lovelace-structured-bias-confirmation-20261012T035916Z')
FROZEN = Path('/srv/slam-research/grf/map-denoise/runs/lovelace-structured-view-bias-20261012T022500Z')
DATA = ROOT / 'data'
_spec = importlib.util.spec_from_file_location('_confirmation_frozen_independent_structured', FROZEN / 'verify_structured.py')
_prior = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prior)
_historical, _arithmetic = _prior._historical, _prior._arithmetic
VerificationError = _prior.VerificationError
require, same, exact, encoded, pair = _prior.require, _prior.same, _prior.exact, _prior.encoded, _prior.pair
load, indexed, finite_seconds = _prior.load, _prior.indexed, _prior.finite_seconds
DOMAIN, TOLERANCE, RADIUS, EPSILON, BETA = _prior.DOMAIN, _prior.TOLERANCE, _prior.RADIUS, _prior.EPSILON, _prior.BETA
BRANCHES, METHODS = _prior.BRANCHES, ('keep', 'calibrated', 'joint')
SEED, PREFIX, LIMIT = 61004, 'shared-appearance-v1', 1800
SAMPLING_NOTE = 'one unseen noise/pose seed in the frozen synthetic family; related initial states are not independent scenes'
TIMING_SCOPE = 'new cold solver includes background preparation; no historical actions or timing reused'
GENERATED_FILES = ('observations.json', 'tasks.json', 'private/truth.json', 'GENERATION.json')


def check_target(output=None):
    require(socket.gethostname() == 'liekkas' and ROOT == EXPECTED_ROOT and ROOT.resolve() == EXPECTED_ROOT,
            'target: exact confirmation root on liekkas required')
    if output is not None:
        path = Path(output).resolve()
        require(path.is_dir() and path != ROOT and path.is_relative_to(ROOT),
                'target: existing output strictly inside exact confirmation root required')
        return path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scientific_files():
    files = dict(_prior.scientific_files())
    for path in sorted({ROOT / 'AGENTS.md', ROOT / 'PROTOCOL.md', ROOT / 'REGISTRATION.json', *ROOT.glob('*.py')}):
        files[str(path.resolve())] = sha(path)
    return dict(sorted(files.items()))


def generated_inputs():
    return {str((DATA / relative).resolve()): sha(DATA / relative) for relative in GENERATED_FILES}


def verify_source_lock(lock):
    require(isinstance(lock, dict) and set(lock) == {'source_root', 'files', 'utc'} and lock['source_root'] == str(ROOT),
            'SOURCE_LOCK: changed exact source identity/schema')
    require(type(lock['utc']) is str and datetime.fromisoformat(lock['utc']).utcoffset() is not None,
            'SOURCE_LOCK: timezone-aware timestamp required')
    # The complete dependency graph is reconstructed independently, not trusted
    # as an arbitrary list of hashes supplied by the producer.
    files = scientific_files()
    same(lock['files'], files, 'SOURCE_LOCK: complete frozen/new graph byte hashes')
    return files


def verify_inputs(lock):
    require(isinstance(lock, dict) and set(lock) == {'source_root', 'files'} and lock['source_root'] == str(DATA),
            'INPUTS: exact newly generated data identity required')
    files = generated_inputs()
    same(lock['files'], files, 'INPUTS: complete newly generated data hashes')
    generation = load(DATA / 'GENERATION.json')
    require(isinstance(generation, dict) and set(generation) == {'seed', 'prefix', 'observations', 'tasks', 'truths',
                                                               'transform', 'auxiliary_samples_used', 'seconds'},
            'GENERATION: changed exact new-data contract')
    same({key: value for key, value in generation.items() if key != 'seconds'},
         {'seed': SEED, 'prefix': PREFIX, 'observations': 18, 'tasks': 162, 'truths': 18,
          'transform': 'add fixed view bias to unclipped base then clamp once', 'auxiliary_samples_used': 0},
         'GENERATION: frozen seed/prefix/one-clip/no auxiliary sample generation')
    finite_seconds(generation['seconds'], 'GENERATION seconds')
    return files


def verify_budget(budget):
    require(isinstance(budget, dict) and set(budget) == {'start_epoch', 'deadline_epoch', 'limit_seconds', 'scope'},
            'BUDGET: invalid shared window schema')
    start, end = (finite_seconds(budget[key], f'BUDGET {key}') for key in ('start_epoch', 'deadline_epoch'))
    require(type(budget['limit_seconds']) is int and budget['limit_seconds'] == LIMIT and start < end == start + LIMIT and
            budget['scope'] == 'one main run plus complete verification; do not reset', 'BUDGET: changed absolute window')
    if time.time() >= end:
        raise TimeoutError('shared main plus verification deadline exhausted')
    return end


@contextmanager
def deadline_guard(output):
    deadline = verify_budget(load(Path(output) / 'BUDGET.json'))
    require(not signal.getitimer(signal.ITIMER_REAL)[0], 'BUDGET: existing deadline cannot be replaced')
    def reached(_signum, _frame):
        raise TimeoutError('shared main plus verification deadline reached')
    previous = signal.signal(signal.SIGALRM, reached)
    signal.setitimer(signal.ITIMER_REAL, max(0.000001, deadline - time.time()))
    try:
        yield deadline
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def verify_action_population(decision_list, observations, tasks):
    require(isinstance(decision_list, list) and len(decision_list) == 486, 'actions: incomplete486 saved action population')
    decisions = indexed(decision_list, lambda row: (row['task_id'], row['method']), 'actions')
    expected = [(f'{oid}s{state}', method) for oid in observations for method in METHODS for state in range(9)]
    require(list(decisions) == expected and set(tasks) == {task for task, _ in expected},
            'actions: missing/reordered task or method')
    require(all(type(row.get('reused')) is bool for row in decision_list) and
            Counter((row['method'], row['reused']) for row in decision_list) ==
            {('keep', False): 162, ('calibrated', False): 162, ('joint', False): 162},
            'actions: complete new/KEEP origins; no historical action reuse')
    return decisions


def verify_public_population(output):
    observations = load(DATA / 'observations.json')
    tasks = load(DATA / 'tasks.json')
    require(isinstance(observations, list) and len(observations) == 18 and
            isinstance(tasks, list) and len(tasks) == 162, 'population: fixed18 observations/162 tasks required')
    images = indexed(observations, lambda row: row['observation_id'], 'observations')
    initial = indexed(tasks, lambda row: row['task_id'], 'tasks')
    require(list(images) == [f'c{i:03d}' for i in range(1, 19)], 'observations: exact fixed IDs/order required')
    for oid, public in images.items():
        require(set(public) == {'observation_id', 'unit_id', 'clipped', 'radius', 'observation'} and
                public['unit_id'] == 'clip' and public['clipped'] is True and public['radius'] == str(RADIUS),
                f'{oid}: private/calibration metadata leaked to public population')
        image = tuple(map(exact, public['observation']))
        require(len(image) == 432 and all(0 <= value <= 1 for value in image), f'{oid}: illegal clipped image')
    require(list(initial) == [f'{oid}s{state}' for oid in images for state in range(9)],
            'tasks: fixed original nine initial-state IDs/order required')
    for tid, task in initial.items():
        require(set(task) == {'task_id', 'base_task_id', 'observation_id', 'incumbent'} and task['observation_id'] in images and
                task['base_task_id'] == tid and tid.startswith(task['observation_id'] + 's'), f'{tid}: changed public task schema')
        exact(task['incumbent'])
    same(load(output / 'observations.json'), observations, 'public observation frozen data readback')
    same(load(output / 'tasks.json'), tasks, 'public tasks frozen data readback')
    return images, initial


def verify_private_population(observations, tasks, model_class):
    """Called only after all486 saved unique actions have passed their gate."""
    truths, expected_tasks, residuals = [], [], []
    for case, branch, colors in _historical.CASES:
        for truth in _historical.TRUTHS:
            oid = f'c{len(truths)+1:03d}'
            base, private = _historical.regenerate(case, branch, colors, truth, SEED, model_class)
            raw = tuple(map(exact, base['observation']))
            bias = (-BETA, Q(0), BETA)
            # Bias is applied to un-clipped base RGB and clipping happens once.
            image = tuple(min(Q(1), max(Q(0), value + bias[index // 144])) for index, value in enumerate(raw))
            same(observations[oid]['observation'], list(map(encoded, image)), f'{oid}: raw→bias→single clip independent regeneration')
            private = {key: value for key, value in private.items() if key not in {'calibration_noise', 'calibration_noise_linf'}}
            private = {'observation_id': oid, **private, 'unit_id': 'clip', 'clipped': True,
                       'image_kind': 'biased_clip', 'actual_view_bias': list(map(encoded, bias))}
            truths.append(private)
            for state, offset in enumerate(_historical.OFFSETS):
                tid = f'{oid}s{state}'
                expected_tasks.append({'task_id': tid, 'base_task_id': tid, 'observation_id': oid, 'incumbent': encoded(truth + offset)})
            from nuisance_model import _point_geometry
            areas = _point_geometry(branch, truth, tuple(map(exact, private['actual_offsets'])))
            darkness = tuple(1 - color for color in colors)
            t = tuple(BETA - value for value in bias)
            for pixel, (high, low) in enumerate(_prior.pixel_bands(image, True)):
                value = sum((a * d for a, d in zip(areas[pixel], darkness)), Q(0)) + t[pixel // 48]
                require((high is None or value <= high) and (low is None or value >= low),
                        f'{oid}: generating latent witness violates clip semantics')
            residuals.append({'observation_id': oid, 'unit_id': 'clip', 'base_noise_linf': private['noise_linf'],
                              'clip_aware_true_witness_valid': True})
    same(list(tasks.values()), expected_tasks, 'all162 exact original nine initial offsets')
    same(load(DATA / 'private/truth.json'), truths, 'all18 independent private random-tape generation records')
    return indexed(truths, lambda row: row['observation_id'], 'truths'), residuals


def nested_sets(calibrated, joint, where):
    require(interval_subset(joint['intervals'], calibrated['intervals']),
            f'{where}: joint retained depth set is not nested in calibrated')
    return True


def aggregate_rows(rows, truths):
    result = [{**record, 'unit_id': 'clip'} for record in _prior._old.reconstruct_aggregates(rows)]
    if not isinstance(truths, dict):
        truths = indexed(truths, lambda row: row['observation_id'], 'aggregate truth identities')
    facts = {oid: str(exact(truth['truth'])) for oid, truth in truths.items()}
    depths = sorted(set(facts.values()), key=exact)
    for depth in depths:
        result.extend({**record, 'grouping': 'by_depth', 'condition': depth, 'unit_id': 'clip'}
                      for record in _historical.base_aggregates([row for row in rows if facts[row['observation_id']] == depth])
                      if record['grouping'] == 'overall')
    return result


def paired_rows(rows):
    matrix = indexed(rows, lambda row: (row['task_id'], row['method']), 'paired rows')
    task_ids = sorted({row['task_id'] for row in rows})
    require(len(matrix) == 486 and len(task_ids) == 162, 'paired rows: complete matched population required')
    result = []
    for tid in task_ids:
        for left, right in (('calibrated', 'joint'),):
            a, b = matrix[tid, left], matrix[tid, right]
            same({key: a[key] for key in ('case', 'seed', 'state_group')},
                 {key: b[key] for key in ('case', 'seed', 'state_group')}, 'paired lineage')
            x, y = exact(a['ae_mm']), exact(b['ae_mm'])
            result.append({'unit_id': 'clip', 'base_task_id': tid, 'case': a['case'], 'seed': a['seed'],
                           'state_group': a['state_group'], 'left_method': left, 'right_method': right,
                           'left_ae_mm': str(x), 'right_ae_mm': str(y), 'mae_gain_mm': str(x - y),
                           'mse_gain_mm2': str(x*x - y*y), 'better': y < x, 'same': y == x, 'worse': y > x})
    return result


def safety_claims(rows, coverage):
    result = {}
    for method in METHODS:
        values = [row for row in rows if row['method'] == method]
        cov = [row for row in coverage if row['method'] == method]
        correct = [row for row in values if row['state_group'] == 'correct']
        offsets = [row for row in values if row['state_group'] == 'small']
        controls = [row for row in offsets if row['case'] == 'equal_full_negative']
        require(tuple(map(len, (values, cov, correct, offsets, controls))) == (162, 18, 18, 144, 24),
                f'{method}: incorrect source/state/control denominators')
        result[method] = {'covered': sum(row['covered'] for row in cov),
                          'generating_branch_covered': sum(row['generating_branch_covered'] is True for row in cov),
                          'correct_kept': sum(row['status'] == 'KEEP' and exact(row['ae_mm']) == 0 for row in correct),
                          'negative_small_kept': sum(row['status'] == 'KEEP' for row in controls),
                          'harm': sum(exact(row['ae_mm']) > exact(row['initial_ae_mm']) for row in values),
                          'kept': sum(row['status'] == 'KEEP' for row in values),
                          'incompatible': sum(row['status'] == 'INCOMPATIBLE' for row in values),
                          'latent_bias_covered': None if method != 'joint' else sum(row['latent_bias_covered'] for row in cov),
                          'latent_material_covered': None if method == 'keep' else sum(row['latent_material_covered'] for row in cov)}
    return result


def registration():
    return {'seed_evaluation_only': SEED, 'random_tape_prefix': PREFIX, 'unit_id': 'clip', 'methods': list(METHODS),
            'epsilon': str(EPSILON), 'beta': str(BETA), 'camera_radius_mm': str(RADIUS),
            'depth_domain_mm': list(map(str, DOMAIN)), 'branches': list(BRANCHES), 'rounds': 8,
            'max_boxes_per_branch': 511, 'tolerance_mm': str(TOLERANCE),
            'expected_observations': 18, 'expected_tasks': 162, 'expected_decisions': 486,
            'expected_cold_solver_calls': 36, 'expected_reused_actions': 0,
            'per_method': {'correct': 18, 'shifted': 144, 'negative_shifted': 24},
            'bias_pattern_generation_only': ['-1/100', '0', '1/100'],
            'public_auxiliary_calibration_samples': 0, 'cpu_threads': 1, 'gpu_count': 0,
            'wall_time_limit_seconds_main_plus_verification': LIMIT,
            'cache_policy': _prior.CACHE_POLICY, 'sampling_note': SAMPLING_NOTE,
            'same_information': 'same clipped image; same public geometry; same T_cal; no auxiliary observations',
            'source_root': str(FROZEN)}


def interval_subset(inner, outer):
    merged = _arithmetic.merged_intervals(outer)
    return all(any(a <= left <= right <= b for a, b in merged) for left, right in inner)


def paired_summary(rows, pairs, outers):
    matrix = indexed(rows, lambda row: (row['task_id'], row['method']), 'paired summary')
    result = {'total': len(pairs), 'better': sum(row['better'] for row in pairs),
              'same': sum(row['same'] for row in pairs), 'worse': sum(row['worse'] for row in pairs),
              'new_repairs': 0, 'lost_repairs': 0, 'strict_hull_subset': 0, 'equal_hull': 0,
              'union_subset': 0, 'branch_subset': 0, 'calibration_equal': 0}
    for row in pairs:
        a, b = matrix[row['base_task_id'], 'calibrated'], matrix[row['base_task_id'], 'joint']
        initial = exact(a['initial_ae_mm'])
        repaired_a, repaired_b = exact(a['ae_mm']) < initial, exact(b['ae_mm']) < initial
        result['new_repairs'] += repaired_b and not repaired_a
        result['lost_repairs'] += repaired_a and not repaired_b
    for arms in outers.values():
        calibrated, joint = arms['calibrated'], arms['joint']
        result['calibration_equal'] += calibrated['calibration_intervals'] == joint['calibration_intervals']
        a, b = calibrated['outer']['hull'], joint['outer']['hull']
        result['equal_hull'] += a == b
        subset = b is None or (a is not None and exact(a[0]) <= exact(b[0]) <= exact(b[1]) <= exact(a[1]))
        result['strict_hull_subset'] += subset and a != b
        result['union_subset'] += interval_subset([pair(value, 'joint union') for value in joint['outer']['retained_intervals']],
                                                 [pair(value, 'calibrated union') for value in calibrated['outer']['retained_intervals']])
        for branch in BRANCHES:
            a = calibrated['branches'][branch]
            b = joint['branches'][branch]
            ia = [(exact(node['lo']), exact(node['hi'])) for node in a['trace'] if node['id'] in a['retained_ids']]
            ib = [(exact(node['lo']), exact(node['hi'])) for node in b['trace'] if node['id'] in b['retained_ids']]
            result['branch_subset'] += interval_subset(_arithmetic.merged_intervals(ib), _arithmetic.merged_intervals(ia))
    return result


def audit_runner_source(source=None):
    code = (ROOT / 'run_confirmation.py').read_text(encoding='utf-8') if source is None else source
    tree = ast.parse(code)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    require({'_run', 'private_truths', 'complete_saved_actions', 'execute_new'} <= set(functions),
            'source audit: missing scientific entry points')
    execute = functions['execute_new']
    require([arg.arg for arg in execute.args.args] == ['observation', 'radius', 'method', 'clipped'],
            'source audit: solver public interface changed')
    require(len(execute.body) == 1 and isinstance(execute.body[0], ast.Return) and
            ast.dump(execute.body[0].value, include_attributes=False) ==
            ast.dump(ast.parse('frozen.execute_new(observation, radius, method, clipped)', mode='eval').body,
                     include_attributes=False), 'source audit: frozen execution wrapper changed')
    run = functions['_run']
    callers = [(function.name, node.lineno) for function in functions.values() for node in ast.walk(function)
               if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'private_truths']
    require(len(callers) == 1 and callers[0][0] == '_run', 'source audit: extra private parsing call path')
    parse_line = callers[0][1]
    readbacks = [node for node in ast.walk(run) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and
                 node.func.id == 'complete_saved_actions']
    require(len(readbacks) == 1 and readbacks[0].lineno < parse_line and
            [ast.dump(arg, include_attributes=False) for arg in readbacks[0].args] ==
            [ast.dump(ast.Name(id=name, ctx=ast.Load()), include_attributes=False) for name in ('output', 'decisions')],
            'source audit: complete saved-action fence not called before private parsing')
    fence = functions['complete_saved_actions']
    predicates = (
        "len(decisions) != 486 or len({(r['task_id'], r['method']) for r in decisions}) != 486 or load(output / 'decisions.json') != decisions",
        "Counter((r['method'], r['reused']) for r in decisions) != {('keep', False): 162, ('calibrated', False): 162, ('joint', False): 162}")
    gates = []
    for expression in predicates:
        expected = ast.dump(ast.parse(expression, mode='eval').body, include_attributes=False)
        matches = [node for node in fence.body if isinstance(node, ast.If) and
                   ast.dump(node.test, include_attributes=False) == expected and
                   any(isinstance(child, ast.Raise) for child in node.body)]
        require(len(matches) == 1,
                'source audit: complete saved unique/origin action fence missing before truth')
        gates.append(matches[0].lineno)
    private_loads = [node for node in ast.walk(functions['private_truths']) if isinstance(node, ast.Call) and
                     isinstance(node.func, ast.Name) and node.func.id == 'load']
    require(len(private_loads) == 1 and len(private_loads[0].args) == 1 and
            ast.dump(private_loads[0].args[0], include_attributes=False) ==
            ast.dump(ast.parse("DATA / 'private/truth.json'", mode='eval').body, include_attributes=False),
            'source audit: private facts must come from exact new DATA identity')
    calls = [node for node in ast.walk(run) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and
             node.func.id == 'execute_new']
    expected_args = [ast.dump(ast.parse(value, mode='eval').body, include_attributes=False)
                     for value in ("item['observation']", "item['radius']", 'method', "item['clipped']")]
    require(len(calls) == 1 and not calls[0].keywords and
            [ast.dump(arg, include_attributes=False) for arg in calls[0].args] == expected_args,
            'source audit: frozen solver receives only public image/radius/method/clipping')
    for name in ('_run', 'population', 'complete_saved_actions'):
        require(not any(isinstance(node, ast.Constant) and isinstance(node.value, str) and 'truth.json' in node.value
                        for node in ast.walk(functions[name])), f'source audit: direct private path in {name}')
    return {'status': 'PASS', 'runner': str(ROOT / 'run_confirmation.py'), 'private_parse_line': parse_line,
            'complete_saved_actions_call_line': readbacks[0].lineno,
            'complete_unique_action_readback_gate_line': gates[0], 'origin_count_gate_line': gates[1],
            'solver_arguments': ['observation', 'radius', 'method', 'clipped'],
            'scope': 'source order and public interface; does not prove operating-system blindness'}


def cost_rows(timing, selector_timing, outers):
    result = {}
    for method in METHODS:
        solves = [row for row in timing if row['method'] == method]
        result[method] = {'cold_solver_calls': len(solves),
                          'outer_seconds': sum(row['outer_seconds'] for row in solves),
                          'preparation_seconds': sum(row['preparation_seconds'] for row in solves),
                          'selector_seconds': sum(row['seconds_by_method'][method] for row in selector_timing),
                          'evaluated_boxes': sum(arms[method]['outer']['evaluated_boxes'] for arms in outers.values()),
                          'unresolved_budget_leaves': sum(arms[method]['outer']['unresolved_budget_leaves'] for arms in outers.values())}
    return result


def verify_timing(output, observations, outers, elapsed):
    timing = load(output / 'timing.json')
    require(isinstance(timing, list) and len(timing) == 36, 'timing: complete36 cold solver records required')
    records = indexed(timing, lambda row: (row['observation_id'], row['method']), 'cold solves')
    require(list(records) == [(oid, method) for oid in observations for method in ('calibrated', 'joint')],
            'timing: reordered/missing/reused cold solve')
    for (oid, method), row in records.items():
        require(set(row) == {'observation_id', 'unit_id', 'method', 'branch_seconds', 'outer_seconds',
                             'preparation_seconds', 'reused'} and row['unit_id'] == 'clip' and row['reused'] is False,
                f'{oid}/{method}: cold timing schema/origin changed')
        arm = outers[oid][method]
        for field in ('branch_seconds', 'outer_seconds', 'preparation_seconds'):
            same(row[field], arm[field], f'{oid}/{method}: recorded cold solve costs')
    selector = load(output / 'selector_timing.json')
    require(isinstance(selector, list) and len(selector) == 18, 'selector timing: complete18 observations required')
    selected = indexed(selector, lambda row: row['observation_id'], 'selector timing')
    require(list(selected) == list(observations), 'selector timing: changed identity/order')
    for oid, row in selected.items():
        require(set(row) == {'observation_id', 'seconds_by_method'} and list(row['seconds_by_method']) == list(METHODS),
                f'{oid}: selector timing schema/order changed')
        for method, seconds in row['seconds_by_method'].items():
            finite_seconds(seconds, f'{oid}/{method}: selector seconds')
    costs = cost_rows(timing, selector, outers)
    require(sum(row['outer_seconds'] + row['selector_seconds'] for row in costs.values()) <= elapsed,
            'timing: measured cold solver/selector sum exceeds main elapsed')
    return costs


def verify_evaluation(output, rows, coverage, observations, truths, outers, status):
    require((len(rows), len(coverage)) == (486, 54), 'evaluation: incomplete row/coverage populations')
    same(load(output / 'results.json'), rows, 'all486 independently reconstructed endpoint errors')
    _prior.verify_csv(output / 'results.csv', rows)
    same(load(output / 'coverage.json'), coverage, 'all54 true depth/branch/material/bias coverage records')
    _prior.verify_csv(output / 'COVERAGE.csv', coverage, ('retained_branches', 'branch_coverage', 'true_depth_leaf_ids'))
    pairs = paired_rows(rows)
    same(load(output / 'PAIRED.json'), pairs, 'all162 matched calibrated→joint task contrasts')
    _prior.verify_csv(output / 'PAIRED.csv', pairs)
    summary = load(output / 'summary.json')
    require(isinstance(summary, dict) and set(summary) == {'rows', 'paired', 'safety', 'costs', 'population',
                                                       'elapsed_seconds', 'sampling_note', 'timing_scope'},
            'summary: changed complete evaluation schema')
    same(summary['sampling_note'], SAMPLING_NOTE, 'summary: one-seed synthetic scope')
    same(summary['timing_scope'], TIMING_SCOPE, 'summary: cold current solve timing scope')
    elapsed = finite_seconds(summary['elapsed_seconds'], 'summary elapsed')
    require(elapsed <= status['elapsed_seconds'], 'summary: elapsed exceeds main')
    aggregates = aggregate_rows(rows, truths)
    same(summary['rows'], aggregates, 'summary: complete independent grouped outcome rows including all three depths')
    _prior.verify_csv(output / 'SUMMARY.csv', aggregates, ('condition',))
    same(summary['paired'], paired_summary(rows, pairs, outers), 'summary: complete paired gains/losses and set nesting')
    same(summary['safety'], safety_claims(rows, coverage), 'summary: reported safety failures or preservation')
    same(summary['costs'], verify_timing(output, observations, outers, elapsed), 'summary: recorded cold costs independently summed')
    same(summary['population'], {'observations': 18, 'tasks': 162, 'decisions': 486, 'new_solver_calls': 36},
         'summary: exact registered population')
    return aggregates, pairs, summary


def verify(output):
    output = check_target(output)
    budget = load(output / 'BUDGET.json')
    deadline = verify_budget(budget)
    sources = verify_source_lock(load(output / 'SOURCE_LOCK.json'))
    inputs = verify_inputs(load(output / 'INPUTS.json'))
    same(load(output / 'REGISTRATION.json'), registration(), 'independently registered frozen confirmation matrix')
    source_audit = audit_runner_source()
    status = load(output / 'RUN_STATUS.json')
    require(isinstance(status, dict) and set(status) == {'status', 'observations_complete', 'decisions_complete',
                                                      'new_solver_calls', 'elapsed_seconds', 'deadline_epoch'},
            'RUN_STATUS: changed completed schema')
    same({key: value for key, value in status.items() if key != 'elapsed_seconds'},
         {'status': 'COMPLETE', 'observations_complete': 18, 'decisions_complete': 486,
          'new_solver_calls': 36, 'deadline_epoch': deadline}, 'RUN_STATUS: exact population and original shared deadline')
    elapsed = finite_seconds(status['elapsed_seconds'], 'RUN_STATUS elapsed')
    require(elapsed <= LIMIT and elapsed <= time.time() - budget['start_epoch'], 'RUN_STATUS: impossible main runtime')
    observations, tasks = verify_public_population(output)
    decision_list = load(output / 'decisions.json')
    decisions = verify_action_population(decision_list, observations, tasks)
    # All486 saved unique actions are checked before labels or regenerated truth.
    model_class, geometry_bounds = _historical.shared_kernels()
    truths, residuals = verify_private_population(observations, tasks, model_class)
    outers = load(output / 'outers.json')
    require(isinstance(outers, dict) and list(outers) == list(observations), 'outers: incomplete/reordered18 observation dictionary')
    checkpoint = output / 'checkpoint'
    require(checkpoint.is_dir() and sorted(path.name for path in checkpoint.iterdir()) == [f'{oid}.json' for oid in observations],
            'checkpoint: complete18 distinct observation records required')
    for oid, arms in outers.items():
        same(load(checkpoint / f'{oid}.json'), {'observation_id': oid, 'arms': arms}, f'{oid}: full saved checkpoint readback')
    unions, branches = {}, {}
    nodes = updates = certificates = geometry_calls = cold_calls = 0
    for index, (oid, public) in enumerate(observations.items()):
        arms = outers[oid]
        require(isinstance(arms, dict) and list(arms) == list(METHODS), f'{oid}: incomplete/reordered method arms')
        unions[oid], branches[oid] = {}, {}
        common = None
        image = tuple(map(exact, public['observation']))
        for method, arm in arms.items():
            where = f'{oid}/{method}'
            fields = {'branches', 'outer', 'branch_seconds', 'outer_seconds'}
            if method != 'keep':
                fields |= {'preparation_seconds', 'background_mask', 'calibration_intervals', 'calibration'}
                cold_calls += 1
            require(isinstance(arm, dict) and set(arm) == fields, f'{where}: changed frozen arm schema')
            allowed = () if method == 'keep' else BRANCHES
            require(list(arm['branches']) == list(arm['branch_seconds']) == list(allowed), f'{where}: changed three-branch solve')
            total = finite_seconds(arm['outer_seconds'], where + ': outer seconds')
            measured = sum(finite_seconds(value, where + ': branch seconds') for value in arm['branch_seconds'].values())
            if method != 'keep':
                measured += finite_seconds(arm['preparation_seconds'], where + ': background preparation seconds')
                calibration = _prior.verify_arm_public(arm, image, True, geometry_bounds, where)
                if common is None:
                    common = calibration
                else:
                    same(calibration, common, f'{oid}: calibrated/joint same independently derived T_cal')
            require(total >= measured and (method != 'keep' or total == 0), f'{where}: impossible recorded solve cost')
            checked_branches = {}
            for branch in allowed:
                checked = _prior.verify_branch(arm['branches'][branch], image, True, method, branch,
                                               geometry_bounds, calibration, f'{where}/{branch}')
                geometry_calls += checked['evaluated']
                updates += checked['proof_updates']
                certificates += len(checked['shared_excluded'])
                nodes += len(arm['branches'][branch]['trace'])
                checked_branches[branch] = checked
            branches[oid][method] = checked_branches
            unions[oid][method] = _prior.verify_union(arm['outer'], checked_branches, method, True, where)
        nested_sets(unions[oid]['calibrated'], unions[oid]['joint'], oid)
        for branch in BRANCHES:
            nested_sets(branches[oid]['calibrated'][branch], branches[oid]['joint'][branch], f'{oid}/{branch}')
        if index % 3 == 2:
            print(json.dumps({'stage': 'independent_confirmation_verification', 'observations_complete': index+1,
                              'remaining_shared_seconds': round(deadline - time.time(), 1)}), flush=True)
    require(cold_calls == 36, 'outers: not exactly36 cold method solves')
    rows = []
    for decision in decision_list:
        task = tasks[decision['task_id']]
        oid, method = task['observation_id'], decision['method']
        require(decision.get('unit_id') == 'clip' and decision['reused'] is False, f'{task["task_id"]}/{method}: action origin changed')
        checked, private = unions[oid][method], truths[oid]
        value = _arithmetic.verify_decision({key: item for key, item in decision.items() if key not in {'unit_id', 'reused'}},
                                            task['incumbent'], checked['hull'], f'{task["task_id"]}/{method}')
        row = _prior.make_row(task, private, decision, checked)
        # An uncovered true depth is reported as a scientific failure. Only a
        # claimed covered witness is required to honor its action certificate.
        if method != 'keep' and checked['hull'] is not None and row['covered']:
            truth, incumbent = exact(private['truth']), exact(task['incumbent'])
            require((incumbent-truth)**2 - (value-truth)**2 >= exact(decision['gain_lower']) >= 0,
                    f'{task["task_id"]}/{method}: covered endpoint contradicts action certificate')
        rows.append(row)
    coverage = [_prior.make_coverage(observations[oid], private, method, unions[oid][method], branches[oid][method])
                for oid, private in truths.items() for method in METHODS]
    aggregates, pairs, summary = verify_evaluation(output, rows, coverage, observations, truths, outers, status)
    same(scientific_files(), sources, 'complete scientific graph unchanged during verification')
    same(generated_inputs(), inputs, 'new generated data unchanged during verification')
    verify_budget(budget)
    return {'status': 'PASS', 'mechanical_verification': 'PASS', 'host': socket.gethostname(), 'output': str(output),
            'observations': 18, 'tasks': 162, 'decisions': 486, 'cold_solver_calls': 36, 'reused_actions': 0,
            'coverage_rows': len(coverage), 'summary_rows': len(aggregates), 'paired_rows': len(pairs),
            'branch_outer_traces': 108, 'tree_nodes_verified': nodes, 'exact_updates_recomputed': updates,
            'contradiction_certificates_verified': certificates, 'historical_geometry_calls': geometry_calls,
            'source_files_verified': len(sources), 'new_data_inputs_verified': len(inputs),
            'source_parse_order_audit': source_audit, 'actual_truth_witnesses': residuals,
            'nested_set_scope': 'same T_cal and depth-set nesting check the frozen algorithm structure; true-depth and latent coverage are separately reported scientific outcomes',
            'scientific_outcomes': {'paired': summary['paired'], 'safety': summary['safety']},
            'shared_budget': {'start_epoch': budget['start_epoch'], 'deadline_epoch': deadline,
                              'elapsed_main_plus_verification_seconds': time.time()-budget['start_epoch']},
            'limitations': ['Frozen geometry/point/area kernels are reused; their physical soundness is not independently proved.',
                            'The registered random tape is independently reconstructed using the historical independent generator and shared point kernels.',
                            'Necessary rows, propagation, branches, unions and actions are recomputed; retained sets are not exact joint feasibility.',
                            'Recorded cold timing is checked and summed, not independently reproduced.',
                            'Source order and public interfaces do not prove operating-system blindness.',
                            'One noise/pose realization per template/depth confirms only this frozen synthetic generation family.',
                            'Mechanical PASS does not require positive mean gain, new repairs, true coverage or absence of scientific harm.']}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Independently verify the frozen unseen-seed confirmation.')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/confirmation-v1')
    output = check_target(parser.parse_args(argv).output)
    destination = output / 'verification'
    if destination.exists():
        print('REFUSE: existing verification is preserved', file=sys.stderr)
        return 2
    destination.mkdir()
    started = time.perf_counter()
    try:
        with deadline_guard(output):
            report = verify(output)
            report.update(utc=datetime.now(timezone.utc).isoformat(), verification_seconds=time.perf_counter()-started)
            with (destination / 'VERIFY.json').open('x', encoding='utf-8') as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write('\n')
    except Exception as error:
        if (destination / 'VERIFY.json').exists():
            print(f'FAIL: interrupted partial verification preserved: {type(error).__name__}: {error}', file=sys.stderr)
            return 1
        report = {'status': 'INCOMPLETE' if isinstance(error, TimeoutError) else 'FAIL',
                  'mechanical_verification': 'INCOMPLETE' if isinstance(error, TimeoutError) else 'FAIL',
                  'host': socket.gethostname(), 'output': str(output), 'error': f'{type(error).__name__}: {error}'}
        report.update(utc=datetime.now(timezone.utc).isoformat(), verification_seconds=time.perf_counter()-started)
        with (destination / 'VERIFY.json').open('x', encoding='utf-8') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())

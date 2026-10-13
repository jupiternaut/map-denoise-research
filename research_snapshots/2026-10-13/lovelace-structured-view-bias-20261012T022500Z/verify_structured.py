#!/usr/bin/env python3
"""Independent deterministic checks for the structured view-bias replay.

The new runner and structured solver are never imported. Historical geometry,
population regeneration, action arithmetic and legacy wide certificate checks
are reused read-only. New clip bands, mask, calibration, necessary rows and
every three/six-variable propagation step are independently reconstructed here.
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
import statistics
import sys
import time

sys.dont_write_bytecode = True
for _structured_verifier_thread in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_structured_verifier_thread] = '1'

ROOT = Path(__file__).resolve().parent
EXPECTED_ROOT = Path('/srv/slam-research/grf/map-denoise/runs/lovelace-structured-view-bias-20261012T022500Z')
PREV = ROOT.parent / 'lovelace-observation-bridge-20261011T232502Z'
PREV_OUTPUT = PREV / 'outputs/bridge-v1'
BASE_OUTPUT = ROOT.parent / 'lovelace-shared-appearance-20261011T052300Z/outputs/confirmation-v1'
_spec = importlib.util.spec_from_file_location('_structured_frozen_independent_bridge', PREV / 'verify_bridge.py')
_legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_legacy)
_old, _historical, _arithmetic = _legacy._old, _legacy._historical, _legacy._arithmetic
VerificationError = _old.VerificationError
require, same, exact, encoded, pair = _old.require, _old.same, _old.exact, _old.encoded, _old.pair
load, indexed, finite_seconds = _old.load, _old.indexed, _old.finite_seconds
DOMAIN, TOLERANCE = _old.DOMAIN, _old.TOLERANCE
EPSILON, BETA, RADIUS = Q(1, 100), Q(1, 100), Q(1, 5)
ROUNDS, MAX_BOXES = 8, 511
UNITS, METHODS, BRANCHES = ('raw', 'clip'), ('keep', 'wide', 'calibrated', 'joint'), ('none', 'partial', 'full')
META = ('observation_id', 'base_observation_id', 'unit_id', 'clipped', 'source_observation_id')
SOURCE_UNITS = {'raw': 'RBdelta', 'clip': 'CBdelta'}
CACHE_POLICY = 'clear frozen geometry caches before every new method/observation, time background preparation'
SAMPLING_NOTE = 'same54 synthetic source observations; related initial states, not new scenes'
TIMING_SCOPE = 'new cold solver includes background preparation; history cost is not a new timing'
GROUP_SCOPE = 'necessary_shared_bias_halfspaces_with_historical_continuous_geometry_bounds'


def pixel_bands(observation, clipped, epsilon=EPSILON, beta=BETA):
    """Intersect existing RGB halfspaces; saturation contributes one side."""
    require(type(clipped) is bool, 'clip bands: Boolean clipping identity required')
    epsilon, beta = exact(epsilon), exact(beta)
    require(epsilon >= 0 and beta >= 0, 'clip bands: nonnegative public bounds required')
    image = tuple(map(exact, observation))
    require(len(image) == 432 and (not clipped or all(0 <= y <= 1 for y in image)),
            'clip bands: fixed RGB layout and legal clipped range required')
    result = []
    for pixel in range(144):
        upper, lower = [], []
        for y in image[3 * pixel:3 * pixel + 3]:
            if not clipped or y != 0:
                upper.append(beta + epsilon if clipped and y == 1 else 1 + beta - y + epsilon)
            if not clipped or y != 1:
                lower.append(1 + beta - epsilon if clipped and y == 0 else 1 + beta - y - epsilon)
        result.append((min(upper) if upper else None, max(lower) if lower else None))
    return tuple(result)


def public_background_mask(geometry_bounds, radius=RADIUS):
    """A pixel is background only when every all-domain branch U is zero."""
    radius = exact(radius)
    require(radius >= 0 and callable(geometry_bounds), 'background: valid public geometry required')
    geometries = [geometry_bounds(branch, radius, *DOMAIN) for branch in BRANCHES]
    require(all(len(geometry) == 144 for geometry in geometries), 'background: fixed geometry layout required')
    mask = []
    for pixel in range(144):
        uppers = []
        for geometry in geometries:
            values = tuple(map(exact, geometry[pixel]))
            require(len(values) == 6 and all(0 <= values[k] <= values[k + 1] <= 1 for k in (0, 2, 4)),
                    'background: invalid normalized area bounds')
            uppers.extend(values[1::2])
        mask.append(all(value == 0 for value in uppers))
    return mask


def calibration_from_public(observation, clipped, mask, epsilon=EPSILON, beta=BETA):
    require(isinstance(mask, (tuple, list)) and len(mask) == 144 and all(type(value) is bool for value in mask),
            'calibration: exact public background mask required')
    bands, beta = pixel_bands(observation, clipped, epsilon, beta), exact(beta)
    intervals = [[Q(0), 2 * beta] for _ in range(3)]
    steps, contradiction = [], None
    for pixel, is_background in enumerate(mask):
        if not is_background:
            continue
        view = pixel // 48
        high, low = bands[pixel]
        for side, value, endpoint in (('upper', high, 1), ('lower', low, 0)):
            if value is None:
                continue
            if (side == 'upper' and value < intervals[view][1]) or (side == 'lower' and value > intervals[view][0]):
                intervals[view][endpoint] = value
                steps.append({'pixel': pixel, 'view': view, 'side': side, 'value': encoded(value)})
                if intervals[view][0] > intervals[view][1]:
                    contradiction = {'kind': 'interval', 'view': view, 'pixel': pixel,
                                     'lower': encoded(intervals[view][0]), 'upper': encoded(intervals[view][1])}
                    break
        if contradiction is not None:
            break
    return {'intervals': [[encoded(a), encoded(b)] for a, b in intervals],
            'excluded': contradiction is not None, 'certificate': {'steps': steps, 'contradiction': contradiction},
            'background_pixel_count': sum(mask),
            'background_pixels_per_view': [sum(mask[48 * view:48 * (view + 1)]) for view in range(3)],
            'prior': [encoded(0), encoded(2 * beta)]}


def necessary_rows(observation, geometry, clipped, method, calibration_intervals, epsilon=EPSILON, beta=BETA):
    require(method in ('calibrated', 'joint'), 'necessary rows: registered new method required')
    require(len(geometry) == 144, 'necessary rows: fixed pixel geometry required')
    intervals = [pair(value, 'necessary rows t calibration') for value in calibration_intervals]
    require(len(intervals) == 3 and all(0 <= a <= b <= 2 * exact(beta) for a, b in intervals),
            'necessary rows: legal calibrated t intervals required')
    rows = []
    for pixel, (high, low) in enumerate(pixel_bands(observation, clipped, epsilon, beta)):
        bounds = tuple(map(exact, geometry[pixel]))
        require(len(bounds) == 6 and all(0 <= bounds[k] <= bounds[k + 1] <= 1 for k in (0, 2, 4)),
                'necessary rows: invalid geometric area coefficients')
        view, (tlow, thigh) = pixel // 48, intervals[pixel // 48]
        for side, bound, coefficients in (('upper', high, bounds[::2]), ('lower', low, bounds[1::2])):
            if bound is None:
                continue
            if method == 'joint':
                coefficients += tuple(Q(int(index == view)) for index in range(3))
            else:
                bound -= tlow if side == 'upper' else thigh
            rows.append((side, coefficients, bound))
    return tuple(rows)


def independent_contraction(rows, initial_bounds, rounds=ROUNDS):
    """Deterministic exact bounds contractor, including immediate crossings."""
    require(type(rounds) is int and 0 <= rounds <= ROUNDS, 'propagation: invalid iteration budget')
    bounds = [pair(value, 'propagation prior') for value in initial_bounds]
    require(len(bounds) in (3, 6) and all(0 <= low <= high for low, high in bounds), 'propagation: invalid nonnegative priors')
    normalized = []
    for side, coefficients, bound in rows:
        coefficients = tuple(map(exact, coefficients))
        require(side in ('upper', 'lower') and len(coefficients) == len(bounds) and all(a >= 0 for a in coefficients),
                'propagation: invalid necessary row')
        normalized.append((side, coefficients, exact(bound)))
    lower, upper = [a for a, _ in bounds], [b for _, b in bounds]
    initial = [[encoded(a), encoded(b)] for a, b in bounds]
    steps, performed, conflict = [], 0, None
    for iteration in range(rounds):
        performed, changed = iteration + 1, False
        for row, (side, coefficients, bound) in enumerate(normalized):
            extrema = lower if side == 'upper' else upper
            total = sum((a * x for a, x in zip(coefficients, extrema)), Q(0))
            if (side == 'upper' and total > bound) or (side == 'lower' and total < bound):
                conflict = {'kind': 'row', 'round': iteration, 'row': row, 'total': encoded(total), 'bound': encoded(bound)}
                break
            for variable, coefficient in enumerate(coefficients):
                if coefficient == 0:
                    continue
                candidate = (bound - (total - coefficient * extrema[variable])) / coefficient
                if side == 'upper' and candidate < upper[variable]:
                    upper[variable] = candidate
                elif side == 'lower' and candidate > lower[variable]:
                    lower[variable] = candidate
                else:
                    continue
                changed = True
                steps.append({'round': iteration, 'row': row, 'variable': variable, 'side': side, 'value': encoded(candidate)})
                if lower[variable] > upper[variable]:
                    conflict = {'kind': 'interval', 'round': iteration, 'row': row, 'variable': variable,
                                'lower': encoded(lower[variable]), 'upper': encoded(upper[variable])}
                    break
            if conflict is not None:
                break
        if conflict is not None or not changed:
            break
    return {'excluded': conflict is not None, 'rounds': performed, 'updates': len(steps),
            'bounds': [[encoded(a), encoded(b)] for a, b in zip(lower, upper)],
            'certificate': {'initial_bounds': initial, 'steps': steps, 'contradiction': conflict}}


def verify_contraction(rows, initial_bounds, certificate, where='contraction', rounds=ROUNDS):
    expected = independent_contraction(rows, initial_bounds, rounds)
    same(certificate, expected['certificate'], f'{where}: every deterministic update and first contradiction')
    return expected


def verify_arm_public(arm, observation, clipped, geometry_bounds, where):
    mask = public_background_mask(geometry_bounds)
    same(arm['background_mask'], mask, f'{where}: all-branch/all-depth public background mask')
    calibration = calibration_from_public(observation, clipped, mask)
    same(arm['calibration'], calibration, f'{where}: same-image one-sided public calibration')
    same(arm['calibration_intervals'], calibration['intervals'], f'{where}: derived t initialization')
    return calibration


def scientific_files():
    files = dict(_legacy.scientific_files())
    for path in sorted({ROOT / 'PROTOCOL.md', *ROOT.glob('*.py')}):
        files[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(sorted(files.items()))


def historical_inputs():
    files = dict(_legacy.historical_inputs())
    for relative in ('INPUTS.json', 'SOURCE_LOCK.json', 'PROTOCOL.json', 'BUDGET.json', 'RUN_STATUS.json', 'observations.json',
                     'tasks.json', 'outers.json', 'decisions.json', 'TIMING.json', 'verification/VERIFY.json'):
        path = PREV_OUTPUT / relative
        files[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(sorted(files.items()))


def verify_source_lock(lock):
    require(isinstance(lock, dict) and set(lock) == {'source_root', 'files', 'utc'} and
            lock['source_root'] == str(ROOT), 'SOURCE_LOCK: invalid exact new source identity/schema')
    require(type(lock['utc']) is str and datetime.fromisoformat(lock['utc']).utcoffset() is not None,
            'SOURCE_LOCK: timezone-aware timestamp required')
    _legacy.verify_source_lock(load(PREV_OUTPUT / 'SOURCE_LOCK.json'))
    sources = scientific_files()
    same(lock['files'], sources, 'SOURCE_LOCK independent complete frozen/new graph byte hashes')
    return sources


def verify_inputs(lock):
    require(isinstance(lock, dict) and set(lock) == {'source_root', 'files'} and lock['source_root'] == str(PREV_OUTPUT),
            'INPUTS: exact prior bridge identity required')
    inputs = historical_inputs()
    same(lock['files'], inputs, 'INPUTS full historical byte inventory')
    _legacy.verify_inputs(load(PREV_OUTPUT / 'INPUTS.json'))
    previous, status = load(PREV_OUTPUT / 'verification/VERIFY.json'), load(PREV_OUTPUT / 'RUN_STATUS.json')
    require(previous.get('status') == 'PASS' and previous.get('host') == 'liekkas' and previous.get('output') == str(PREV_OUTPUT) and
            (previous.get('observations'), previous.get('tasks'), previous.get('decisions')) == (432, 3888, 7776) and
            status.get('status') == 'COMPLETE', 'INPUTS: exact completed verified historical bridge required')
    return inputs


def registration():
    return {'units': SOURCE_UNITS, 'methods': list(METHODS), 'epsilon': str(EPSILON), 'beta': str(BETA),
            'depth_domain_mm': list(map(str, DOMAIN)), 'branches': list(BRANCHES),
            'max_boxes_per_branch': MAX_BOXES, 'rounds': ROUNDS, 'tolerance_mm': str(TOLERANCE),
            'expected_observations': 108, 'expected_tasks': 972, 'expected_decisions': 3888,
            'expected_new_solver_calls': 216, 'expected_new_method_actions': 1944, 'expected_reused_actions': 972,
            'expected_keep_actions': 972, 'cpu_threads': 1, 'gpu_count': 0,
            'wall_time_limit_seconds_main_plus_verification': 5400, 'source_root': str(PREV_OUTPUT),
            'per_unit_method': {'observations': 54, 'correct': 54, 'small': 432, 'negative_small': 72},
            'bias_pattern_generation_only': ['-1/100', '0', '1/100'],
            'same_information': 'same image channels and public model, no auxiliary observations',
            'calibration': 'all-branch/global-depth public guaranteed-background; same T_cal warm-start for both',
            'noise_model': 'raw additive independent bounded error, clip endpoints one-sided',
            'cache_policy': CACHE_POLICY, 'sampling_note': SAMPLING_NOTE}


json_hash, verify_csv = _legacy.json_hash, _legacy.verify_csv


def verify_budget(budget):
    require(isinstance(budget, dict) and set(budget) == {'start_unix', 'deadline_unix', 'limit_seconds', 'scope'},
            'BUDGET: invalid shared window schema')
    start = finite_seconds(budget['start_unix'], 'BUDGET start')
    end = finite_seconds(budget['deadline_unix'], 'BUDGET deadline')
    require(type(budget['limit_seconds']) is int and budget['limit_seconds'] == 5400 and start < end <= start + 5400 and
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


def verify_branch(outer, observation, clipped, method, branch, geometry_bounds, calibration, where):
    fields = {'domain', 'epsilon', 'beta', 'clipped', 'tolerance', 'max_boxes', 'evaluated_boxes', 'trace',
              'retained_ids', 'hull', 'shared_tests', 'propagation_rounds', 'mode', 'rounds', 'budget',
              'channel_precheck', 'initial_bounds', 'scope'}
    require(isinstance(outer, dict) and set(outer) == fields, f'{where}: invalid structured branch schema')
    initial = [[encoded(0), encoded(1)] for _ in range(3)]
    if method == 'joint':
        initial.extend(calibration['intervals'])
    same(outer['initial_bounds'], initial, f'{where}: public material and same T_cal priors')
    require(pair(outer['domain'], where) == DOMAIN and exact(outer['epsilon']) == EPSILON and exact(outer['beta']) == BETA and
            type(outer['clipped']) is bool and outer['clipped'] == clipped and exact(outer['tolerance']) == TOLERANCE and
            type(outer['max_boxes']) is int and outer['max_boxes'] == MAX_BOXES and outer['mode'] == method and
            outer['scope'] == GROUP_SCOPE and outer['channel_precheck'] == 'none', f'{where}: changed scientific branch contract')
    require(all(type(outer[key]) is int and outer[key] == expected for key, expected in
                (('rounds', ROUNDS), ('propagation_rounds', ROUNDS), ('budget', MAX_BOXES))), f'{where}: changed branch budget')
    nodes = indexed(outer['trace'], lambda node: node['id'], where)
    require(0 < len(nodes) <= MAX_BOXES and all(type(node['id']) is int for node in outer['trace']) and
            list(nodes) == list(range(len(nodes))) and nodes[0]['parent'] is None, f'{where}: invalid complete BFS identities')
    boxes = {ident: (exact(node['lo']), exact(node['hi'])) for ident, node in nodes.items()}
    require(boxes[0] == DOMAIN, f'{where}: root differs from full domain')
    leaves, retained, excluded, referenced = [], [], [], set()
    evaluated = next_child = updates = 0
    next_child = 1
    base = {'id', 'parent', 'lo', 'hi', 'status'}
    extras = {'split': {'children', 'rounds', 'updates', 'darkness_bounds', 't_bounds'},
              'retained_tolerance': {'rounds', 'updates', 'darkness_bounds', 't_bounds'},
              'excluded_shared': {'rounds', 'updates', 'group_id', 'certificate'},
              'excluded_calibration': {'certificate'}, 'unresolved_budget': {'darkness_bounds', 't_bounds'}}
    for node in outer['trace']:
        ident, status = node['id'], node['status']
        left, right = boxes[ident]
        require(status in extras and set(node) == base | extras[status] and DOMAIN[0] <= left < right <= DOMAIN[1],
                f'{where}: invalid node schema/status/box {ident}')
        if ident:
            require(type(node['parent']) is int and node['parent'] in nodes and node['parent'] < ident,
                    f'{where}: invalid parent {ident}')
        if calibration['excluded']:
            require(ident == 0 and len(nodes) == 1 and status == 'excluded_calibration', f'{where}: calibration conflict not represented')
            same(node['certificate'], calibration['certificate'], f'{where}: calibration exclusion proof')
        else:
            require(status not in ('excluded_calibration', 'unresolved_budget'), f'{where}: false calibration/budget exclusion in full511 tree')
            require(evaluated < MAX_BOXES, f'{where}: more than511 evaluations')
            geometry = geometry_bounds(branch, RADIUS, left, right)
            rows = necessary_rows(observation, geometry, clipped, method, calibration['intervals'])
            contraction = independent_contraction(rows, initial)
            evaluated += 1
            require(type(node['rounds']) is int and node['rounds'] == contraction['rounds'] and
                    type(node['updates']) is int and node['updates'] == contraction['updates'], f'{where}: false exact propagation accounting {ident}')
            updates += contraction['updates']
            if contraction['excluded']:
                require(status == 'excluded_shared' and node['group_id'] == 'global', f'{where}: omitted/false contradiction {ident}')
                same(node['certificate'], contraction['certificate'], f'{where}: first exact update/crossing certificate {ident}')
            else:
                require(status in ('split', 'retained_tolerance'), f'{where}: unsupported exclusion {ident}')
                same(node['darkness_bounds'], contraction['bounds'][:3], f'{where}: independently contracted darkness {ident}')
                same(node['t_bounds'], contraction['bounds'][3:] if method == 'joint' else calibration['intervals'],
                     f'{where}: independently contracted/frozen bias {ident}')
        if status == 'split':
            require(right - left > TOLERANCE and node['children'] == [next_child, next_child + 1] and
                    all(type(child) is int and child in nodes for child in node['children']), f'{where}: invalid split allocation')
            middle = (left + right) / 2
            for child, expected in zip(node['children'], ((left, middle), (middle, right))):
                require(child not in referenced and nodes[child]['parent'] == ident and boxes[child] == expected,
                        f'{where}: gap/overlap/nonbisection/duplicate child')
                referenced.add(child)
            next_child += 2
        else:
            leaves.append((left, right))
            if status == 'retained_tolerance':
                require(right - left <= TOLERANCE, f'{where}: retained leaf exceeds tolerance')
                retained.append(node)
            else:
                excluded.append(node)
    require(referenced == set(nodes) - {0}, f'{where}: orphan nodes')
    _arithmetic.verify_partition(leaves, where)
    require(type(outer['evaluated_boxes']) is int and outer['evaluated_boxes'] == evaluated and
            type(outer['shared_tests']) is int and outer['shared_tests'] == evaluated, f'{where}: false tree counters')
    same(outer['retained_ids'], [node['id'] for node in retained], f'{where}: retained provenance')
    intervals = [boxes[node['id']] for node in retained]
    hull = None if not intervals else (min(a for a, _ in intervals), max(b for _, b in intervals))
    same(outer['hull'], None if hull is None else list(map(encoded, hull)), f'{where}: retained hull')
    return {'hull': hull, 'intervals': intervals, 'retained': retained, 'boxes': boxes, 'evaluated': evaluated,
            'budget_leaves': 0, 'proof_updates': updates, 'shared_excluded': excluded}


def verify_union(outer, branches, method, clipped, where):
    if method in ('keep', 'wide'):
        return _legacy.verify_union(outer, branches, SOURCE_UNITS['clip' if clipped else 'raw'],
                                    'keep' if method == 'keep' else 'shared', where)
    fields = {'domain', 'epsilon', 'beta', 'clipped', 'tolerance', 'max_boxes_per_branch', 'evaluated_boxes',
              'unresolved_budget_leaves', 'source_leaves', 'retained_intervals', 'hull', 'retained_branches',
              'mode', 'rounds', 'budget', 'scope', 'channel_precheck'}
    require(isinstance(outer, dict) and set(outer) == fields and pair(outer['domain'], where) == DOMAIN and
            exact(outer['epsilon']) == EPSILON and exact(outer['beta']) == BETA and type(outer['clipped']) is bool and
            outer['clipped'] == clipped and exact(outer['tolerance']) == TOLERANCE and outer['mode'] == method and
            outer['channel_precheck'] == 'none' and outer['scope'] == 'union_of_branch_depth_outer_leaves_then_convex_hull',
            f'{where}: changed union schema/contract')
    require(all(type(outer[key]) is int and outer[key] == expected for key, expected in
                (('max_boxes_per_branch', MAX_BOXES), ('budget', MAX_BOXES), ('rounds', ROUNDS))), f'{where}: changed union budget')
    source = [{'branch': branch, 'id': node['id'], 'lo': node['lo'], 'hi': node['hi']}
              for branch, value in branches.items() for node in value['retained']]
    intervals = _arithmetic.merged_intervals(interval for value in branches.values() for interval in value['intervals'])
    hull = None if not intervals else (intervals[0][0], intervals[-1][1])
    retained_branches = [branch for branch, value in branches.items() if value['hull'] is not None]
    evaluated = sum(value['evaluated'] for value in branches.values())
    same(outer['source_leaves'], source, f'{where}: complete retained branch provenance')
    same(outer['retained_intervals'], [list(map(encoded, interval)) for interval in intervals], f'{where}: exact closed union')
    same(outer['hull'], None if hull is None else list(map(encoded, hull)), f'{where}: convex union hull')
    require(type(outer['evaluated_boxes']) is int and outer['evaluated_boxes'] == evaluated and
            type(outer['unresolved_budget_leaves']) is int and outer['unresolved_budget_leaves'] == 0,
            f'{where}: false union node/budget counts')
    same(outer['retained_branches'], retained_branches, f'{where}: retained branches')
    return {'hull': hull, 'intervals': intervals, 'retained_branches': retained_branches, 'evaluated': evaluated, 'budget_leaves': 0}


def audit_runner_source(source=None):
    code = (ROOT / 'run_structured.py').read_text(encoding='utf-8') if source is None else source
    tree = ast.parse(code)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    require({'_run', 'private_truths', 'execute_new', 'prepare_reuse'} <= set(functions), 'source audit: missing scientific entry points')
    require([arg.arg for arg in functions['execute_new'].args.args] == ['observation', 'radius', 'method', 'clipped'],
            'source audit: new solver public interface changed')
    run = functions['_run']
    truth_calls = [node for node in ast.walk(run) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'private_truths']
    require(len(truth_calls) == 1, 'source audit: private fact parsing must occur once')
    parse_line = truth_calls[0].lineno
    gate_source = "len(decisions) != 3888 or len({(r['task_id'], r['method']) for r in decisions}) != 3888 or load(output / 'decisions.json') != decisions"
    count_source = "Counter((r['method'], r['reused']) for r in decisions) != {('keep', False): 972, ('wide', True): 972, ('calibrated', False): 972, ('joint', False): 972}"
    expected = [ast.dump(ast.parse(code, mode='eval').body, include_attributes=False) for code in (gate_source, count_source)]
    fences = []
    for expression in expected:
        candidates = [node for node in run.body if isinstance(node, ast.If) and
                      ast.dump(node.test, include_attributes=False) == expression and
                      any(isinstance(child, ast.Raise) for child in node.body)]
        require(len(candidates) == 1 and candidates[0].end_lineno < parse_line, 'source audit: complete saved action/origin fence missing before truth')
        fences.append(candidates[0].lineno)
    callers = [(function.name, node.lineno) for function in functions.values() for node in ast.walk(function)
               if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'private_truths']
    require(callers == [('_run', parse_line)], 'source audit: additional private parsing call path')
    private_loads = [node for node in ast.walk(functions['private_truths']) if isinstance(node, ast.Call) and
                     isinstance(node.func, ast.Name) and node.func.id == 'load']
    require(len(private_loads) == 1 and ast.dump(private_loads[0].args[0], include_attributes=False) ==
            ast.dump(ast.parse("bridge.BASE / 'evaluation/truth.json'", mode='eval').body, include_attributes=False),
            'source audit: private facts must come from frozen base after action fence')
    for name in ('_run', 'execute_new', 'prepare_reuse', 'population'):
        require(not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'load' and
                        any(isinstance(value, ast.Constant) and isinstance(value.value, str) and 'truth.json' in value.value
                            for value in ast.walk(node)) for node in ast.walk(functions[name])),
                f'source audit: direct private parsing path in {name}')
    return {'status': 'PASS', 'runner': str(ROOT / 'run_structured.py'), 'private_parse_line': parse_line,
            'complete_action_readback_gate_line': fences[0], 'origin_count_gate_line': fences[1],
            'solver_arguments': ['observation', 'radius', 'method', 'clipped'],
            'scope': 'frozen source order and public interface; does not prove operating-system blindness'}


def verify_action_population(decision_list, observations, tasks):
    require(isinstance(decision_list, list) and len(decision_list) == 3888, 'actions: incomplete3888 saved action population')
    decisions = indexed(decision_list, lambda row: (row['task_id'], row['method']), 'actions')
    expected = [(f'{oid}-s{state}', method) for oid in observations for method in METHODS for state in range(9)]
    require(list(decisions) == expected and set(tasks) == {task for task, _ in expected}, 'actions: missing/reordered task or method')
    require(Counter((row['method'], row['reused']) for row in decision_list) ==
            {('keep', False): 972, ('wide', True): 972, ('calibrated', False): 972, ('joint', False): 972},
            'actions: complete new/reused/KEEP origins')
    return decisions


def verify_public_population(output):
    base_images = indexed(load(BASE_OUTPUT / 'observations.json'), lambda row: row['observation_id'], 'base public observations')
    base_tasks = indexed(load(BASE_OUTPUT / 'tasks.json'), lambda row: row['task_id'], 'base public tasks')
    previous_images = indexed(load(PREV_OUTPUT / 'observations.json'), lambda row: row['observation_id'], 'source observations')
    previous_tasks = indexed(load(PREV_OUTPUT / 'tasks.json'), lambda row: row['task_id'], 'source tasks')
    require(list(base_images) == [f'o{i:03d}' for i in range(54)] and len(base_tasks) == 486, 'public population: frozen base identity changed')
    observations, tasks = [], []
    for unit in UNITS:
        source_unit = SOURCE_UNITS[unit]
        for base_id, original in base_images.items():
            source_id = f'{source_unit}-{base_id}'
            source = previous_images[source_id]
            expected_image = list(map(encoded, _legacy.transformed_image(original['observation'], source_unit)))
            same(source['observation'], expected_image, f'{source_id}: independent frozen bias/clipping generation')
            require(source['base_observation_id'] == base_id and source['unit_id'] == source_unit and source['radius'] == str(RADIUS) and
                    source['bound'] == '1/50' and source['rounds'] == ROUNDS, f'{source_id}: wrong historical source contract')
            observations.append({'observation_id': f'{unit}-{base_id}', 'base_observation_id': base_id, 'unit_id': unit,
                                 'clipped': unit == 'clip', 'radius': str(RADIUS), 'observation': expected_image,
                                 'source_observation_id': source_id})
        for base_id, original in base_tasks.items():
            source_id = f'{source_unit}-{base_id}'
            source = previous_tasks[source_id]
            same(source, {'task_id': source_id, 'observation_id': f'{source_unit}-{original["observation_id"]}',
                          'base_task_id': base_id, 'incumbent': original['incumbent']}, f'{source_id}: frozen source task')
            tasks.append({'task_id': f'{unit}-{base_id}', 'observation_id': f'{unit}-{original["observation_id"]}',
                          'base_task_id': base_id, 'source_task_id': source_id, 'incumbent': original['incumbent']})
    same(load(output / 'observations.json'), observations, 'all108 independently generated and source-identical observations')
    same(load(output / 'tasks.json'), tasks, 'all972 frozen public initial tasks')
    return (indexed(observations, lambda row: row['observation_id'], 'observations'),
            indexed(tasks, lambda row: row['task_id'], 'tasks'), previous_images, previous_tasks)


def verify_private_population(output, observations, model_class):
    base_images, _base_tasks, base_truths = _old.verify_frozen_population(model_class)
    truths, residuals = [], []
    for oid, public in observations.items():
        private = {**base_truths[public['base_observation_id']], 'observation_id': oid,
                   'base_observation_id': public['base_observation_id'], 'unit_id': public['unit_id'],
                   'actual_view_bias': list(map(encoded, (-BETA, Q(0), BETA)))}
        truths.append(private)
        model = model_class(private['rear_branch'], camera_radius=RADIUS, colors=((Q(0), Q(1)),) * 3)
        prediction = model.point_prediction(exact(private['truth']), tuple(map(exact, private['actual_offsets'])),
                                            colors=tuple(map(exact, private['colors'])))
        raw = tuple(map(exact, base_images[public['base_observation_id']]['observation']))
        require(max(abs(a - b) for a, b in zip(raw, prediction)) <= EPSILON, f'{oid}: frozen noise violates original model')
        generated = _legacy.transformed_image(raw, SOURCE_UNITS[public['unit_id']])
        same(public['observation'], list(map(encoded, generated)), f'{oid}: private generation agrees with pre-action public image')
        t = (2 * BETA, BETA, Q(0))
        # Check the exact witness against clipping-aware bands without exposing
        # it to any method. Geometry at the generating pose is evaluated here.
        from nuisance_model import _point_geometry
        point_areas = _point_geometry(private['rear_branch'], exact(private['truth']), tuple(map(exact, private['actual_offsets'])))
        darkness = tuple(1 - exact(value) for value in private['colors'])
        for pixel, (high, low) in enumerate(pixel_bands(generated, public['clipped'])):
            value = sum((a * d for a, d in zip(point_areas[pixel], darkness)), Q(0)) + t[pixel // 48]
            require((high is None or value <= high) and (low is None or value >= low), f'{oid}: true latent witness violates observation semantics')
        residuals.append({'observation_id': oid, 'unit_id': public['unit_id'],
                          'base_noise_linf': str(max(abs(a - b) for a, b in zip(raw, prediction))),
                          'clip_aware_true_witness_valid': True})
    same(load(output / 'evaluation/truth.json'), truths, 'all108 independent private generation facts after saved action gate')
    return indexed(truths, lambda row: row['observation_id'], 'truths'), residuals


def verify_decision(decision, task, checked, unit, where):
    method = decision.get('method')
    require(method in METHODS and decision.get('unit_id') == unit and type(decision.get('reused')) is bool and
            decision['reused'] == (method == 'wide'), f'{where}: false method/unit/reuse identity')
    core = {key: value for key, value in decision.items() if key not in {'unit_id', 'reused'}}
    return _arithmetic.verify_decision(core, task['incumbent'], checked['hull'], where)


def verify_provenance(output, observations, tasks, outers, decisions, source_images, source_tasks):
    source_outers = indexed(load(PREV_OUTPUT / 'outers.json'), lambda row: row['observation_id'], 'source outer population')
    source_actions = indexed(load(PREV_OUTPUT / 'decisions.json'), lambda row: (row['task_id'], row['method']), 'source actions')
    provenance = []
    for oid, public in observations.items():
        source_id = public['source_observation_id']
        source_arm = source_outers[source_id]['arms']['shared']
        same(outers[oid]['arms']['wide'], source_arm, f'{oid}: exact historical wide arm body and timing')
        same(outers[oid]['reused_source'], {'output_root': str(PREV_OUTPUT), 'observation_id': source_id, 'method': 'shared'},
             f'{oid}: exact prior output/observation/method provenance')
        references = []
        for state in range(9):
            task = tasks[f'{oid}-s{state}']
            original = source_actions[task['source_task_id'], 'shared']
            require(source_tasks[task['source_task_id']]['incumbent'] == task['incumbent'] and
                    source_tasks[task['source_task_id']]['observation_id'] == source_id, f'{oid}: false reused task mapping')
            stripped = {'task_id', 'method', 'observation_id', 'unit_id', 'bound', 'rounds', 'reused'}
            same({key: value for key, value in decisions[task['task_id'], 'wide'].items() if key not in stripped},
                 {key: value for key, value in original.items() if key not in stripped}, f'{oid}: exact historical action body')
            references.append({'task_id': task['task_id'], 'source_task_id': task['source_task_id'], 'action_sha256': json_hash(original)})
        provenance.append({'observation_id': oid, 'source_root': str(PREV_OUTPUT), 'source_observation_id': source_id,
                           'source_method': 'shared', 'method': 'wide', 'observation_sha256': json_hash(source_images[source_id]),
                           'outer_sha256': json_hash(source_arm), 'tasks': references})
    same(load(output / 'REUSE.json'), provenance, 'all108 exact source mappings and972 action body hashes')
    return len(provenance)


def make_row(task, private, decision, checked):
    oldrow = _old.make_row(task, private, decision, checked)
    hull = checked['hull']
    return {'task_id': task['task_id'], 'base_task_id': task['base_task_id'], 'observation_id': task['observation_id'],
            'unit_id': private['unit_id'], 'method': decision['method'], 'reused': decision['reused'],
            **{key: oldrow[key] for key in ('case', 'seed', 'state_group', 'prior_position', 'incumbent', 'output', 'status',
                                          'initial_ae_mm', 'ae_mm', 'se_mm2', 'actual_gain_mm2', 'covered', 'hull_width_mm')},
            'hull_lo_mm': None if hull is None else str(hull[0]), 'hull_hi_mm': None if hull is None else str(hull[1]),
            'retained_branch_count': len(checked['retained_branches']), 'retained_interval_count': len(checked['intervals']),
            'evaluated_boxes': checked['evaluated'], 'unresolved_budget_leaves': checked['budget_leaves'], 'auxiliary_samples_used': 0}


def make_coverage(public, private, method, checked, branches):
    truth, hull, branch = exact(private['truth']), checked['hull'], private['rear_branch']
    branch_coverage = {name: _historical.contains(value['intervals'], truth) for name, value in branches.items()}
    true_ids = {name: [node['id'] for node in value['retained'] if exact(node['lo']) <= truth <= exact(node['hi'])]
                for name, value in branches.items()}
    true_leaves = [] if method == 'keep' else [node for node in branches[branch]['retained']
                                             if exact(node['lo']) <= truth <= exact(node['hi'])]
    latent_bias = latent_material = None
    if method in ('calibrated', 'joint'):
        actual_darkness = [1 - exact(color) for color in private['colors']]
        latent_material = bool(true_leaves) and all(all(pair(bounds, 'latent material')[0] <= value <= pair(bounds, 'latent material')[1]
                                                       for bounds, value in zip(node['darkness_bounds'], actual_darkness)) for node in true_leaves)
        if method == 'joint':
            actual_t = [BETA - exact(bias) for bias in private['actual_view_bias']]
            latent_bias = bool(true_leaves) and all(all(pair(bounds, 'latent bias')[0] <= value <= pair(bounds, 'latent bias')[1]
                                                       for bounds, value in zip(node['t_bounds'], actual_t)) for node in true_leaves)
    return {'observation_id': public['observation_id'], 'unit_id': public['unit_id'], 'method': method,
            'case': private['case'], 'seed': private['seed'], 'covered': _historical.contains(checked['intervals'], truth),
            'incompatible': hull is None, 'hull_width_mm': None if hull is None else str(hull[1] - hull[0]),
            'retained_branches': checked['retained_branches'], 'branch_coverage': branch_coverage,
            'generating_branch': branch, 'generating_branch_covered': _historical.contains(checked['intervals'], truth) if method == 'keep' else branch_coverage[branch],
            'true_depth_leaf_ids': true_ids, 'latent_material_covered': latent_material, 'latent_bias_covered': latent_bias}


def reconstruct_aggregates(rows):
    return [{**record, 'unit_id': unit} for unit in UNITS
            for record in _old.reconstruct_aggregates([row for row in rows if row['unit_id'] == unit])]


def paired_rows(rows):
    matrix = indexed(rows, lambda row: (row['unit_id'], row['base_task_id'], row['method']), 'paired rows')
    base_ids = sorted({row['base_task_id'] for row in rows})
    require(len(matrix) == 3888 and len(base_ids) == 486, 'paired rows: complete matched population required')
    result = []
    comparisons = (('keep', 'wide'), ('keep', 'calibrated'), ('keep', 'joint'),
                   ('wide', 'calibrated'), ('wide', 'joint'), ('calibrated', 'joint'))
    for unit in UNITS:
        for task in base_ids:
            require(all((unit, task, method) in matrix for method in METHODS), 'paired rows: missing method for initial task')
            for left, right in comparisons:
                a, b = matrix[unit, task, left], matrix[unit, task, right]
                same({key: a[key] for key in ('case', 'seed', 'state_group')},
                     {key: b[key] for key in ('case', 'seed', 'state_group')}, 'paired lineage')
                x, y = exact(a['ae_mm']), exact(b['ae_mm'])
                result.append({'unit_id': unit, 'base_task_id': task, 'case': a['case'], 'seed': a['seed'],
                               'state_group': a['state_group'], 'left_method': left, 'right_method': right,
                               'left_ae_mm': str(x), 'right_ae_mm': str(y), 'mae_gain_mm': str(x - y),
                               'mse_gain_mm2': str(x * x - y * y), 'better': y < x, 'same': y == x, 'worse': y > x})
    return result


def group_mean_sd(aggregates):
    result = []
    for unit in UNITS:
        for method in METHODS:
            values = [row for row in aggregates if row['unit_id'] == unit and row['method'] == method and
                      row['state_group'] == 'small' and row['grouping'] == 'by_seed']
            require(len(values) == 3, 'seed grouping: three registered source seeds required')
            result.append({'unit_id': unit, 'method': method, 'groups': 3, 'ddof': 1,
                           'mae_mean': statistics.mean(row['mae_mm'] for row in values),
                           'mae_sd': statistics.stdev(row['mae_mm'] for row in values),
                           'mse_mean': statistics.mean(row['mse_mm2'] for row in values),
                           'mse_sd': statistics.stdev(row['mse_mm2'] for row in values)})
    return result


def verify_timing(output, observations, outers, summary):
    timing = load(output / 'TIMING.json')
    require(isinstance(timing, dict) and set(timing) == {'observations'}, 'TIMING: changed schema')
    same(load(output / 'TIMING_PARTIAL.json'), timing, 'TIMING: complete checkpoint equality')
    records = indexed(timing['observations'], lambda row: row['observation_id'], 'timing records')
    require(list(records) == list(observations), 'TIMING: incomplete/reordered observation population')
    for oid, row in records.items():
        require(set(row) == {'observation_id', 'unit_id', 'new_solver_seconds_by_method',
                             'recorded_historical_seconds_by_method', 'selector_seconds_by_method'} and
                row['unit_id'] == observations[oid]['unit_id'], f'{oid}: false timing schema/identity')
        for field in ('new_solver_seconds_by_method', 'recorded_historical_seconds_by_method', 'selector_seconds_by_method'):
            require(list(row[field]) == list(METHODS), f'{oid}: changed cost method order')
            for method in METHODS:
                finite_seconds(row[field][method], f'{oid}/{method}/{field}')
        for method in METHODS:
            total = outers[oid]['arms'][method]['outer_seconds']
            same(row['new_solver_seconds_by_method'][method], total if method in ('calibrated', 'joint') else 0., f'{oid}: new solve attribution')
            same(row['recorded_historical_seconds_by_method'][method], total if method == 'wide' else 0., f'{oid}: historical solve attribution')
    costs = []
    for unit in UNITS:
        subset = [row for row in records.values() if row['unit_id'] == unit]
        for method in METHODS:
            costs.append({'unit_id': unit, 'method': method,
                          'new_solver_seconds': sum(row['new_solver_seconds_by_method'][method] for row in subset),
                          'selector_seconds': sum(row['selector_seconds_by_method'][method] for row in subset),
                          'recorded_historical_seconds': sum(row['recorded_historical_seconds_by_method'][method] for row in subset),
                          'evaluated_boxes': sum(row['arms'][method]['outer']['evaluated_boxes'] for row in outers.values() if row['unit_id'] == unit)})
    same(summary['costs'], costs, 'TIMING: all8 independent new/historical/selector cost sums')
    verify_csv(output / 'evaluation/TIMING.csv', costs)
    require(sum(row['new_solver_seconds'] + row['selector_seconds'] for row in costs) <= summary['elapsed_seconds'],
            'TIMING: measured new work exceeds main elapsed time')
    return costs


def verify_evaluation(output, rows, coverage, observations, outers, status):
    require((len(rows), len(coverage)) == (3888, 432), 'evaluation: incomplete action and coverage population')
    same(load(output / 'evaluation/ROWS.json'), rows, 'all3888 independently reconstructed actual-depth endpoint errors')
    verify_csv(output / 'evaluation/ROWS.csv', rows)
    same(load(output / 'evaluation/COVERAGE.json'), coverage, 'all432 retained-set/branch/latent witness records')
    verify_csv(output / 'evaluation/COVERAGE.csv', coverage, ('retained_branches', 'branch_coverage', 'true_depth_leaf_ids'))
    pairs = paired_rows(rows)
    same(load(output / 'evaluation/PAIRED.json'), pairs, 'all5832 matched initial-task endpoint contrasts')
    verify_csv(output / 'evaluation/PAIRED.csv', pairs)
    summary = load(output / 'evaluation/SUMMARY.json')
    require(isinstance(summary, dict) and set(summary) == {'observation_configurations', 'task_count', 'decision_count',
            'rows', 'seed_mean_sd', 'costs', 'elapsed_seconds', 'sampling_note', 'timing_scope'}, 'SUMMARY: changed schema')
    for key, value in {'observation_configurations': 108, 'task_count': 972, 'decision_count': 3888,
                       'sampling_note': SAMPLING_NOTE, 'timing_scope': TIMING_SCOPE}.items():
        same(summary[key], value, f'SUMMARY {key}')
    require(finite_seconds(summary['elapsed_seconds'], 'SUMMARY elapsed') <= status['elapsed_seconds'], 'SUMMARY: duration exceeds main')
    aggregates = reconstruct_aggregates(rows)
    require(len(aggregates) == 720, 'SUMMARY: incomplete grouped population')
    same(summary['rows'], aggregates, 'all720 independently reconstructed grouped count/error/safety/cost rows')
    verify_csv(output / 'evaluation/SUMMARY.csv', aggregates, ('condition',))
    stats = group_mean_sd(aggregates)
    same(summary['seed_mean_sd'], stats, 'all8 independently reconstructed sample seed means/SD')
    verify_csv(output / 'evaluation/SEED_STATS.csv', stats)
    verify_timing(output, observations, outers, summary)
    return aggregates, pairs


def safety_claims(rows, coverage):
    result = {}
    for unit in UNITS:
        result[unit] = {}
        for method in METHODS:
            values = [row for row in rows if row['unit_id'] == unit and row['method'] == method]
            cov = [row for row in coverage if row['unit_id'] == unit and row['method'] == method]
            correct = [row for row in values if row['state_group'] == 'correct']
            offsets = [row for row in values if row['state_group'] == 'small']
            controls = [row for row in offsets if row['case'] == 'equal_full_negative']
            require((len(values), len(cov), len(correct), len(offsets), len(controls)) == (486, 54, 54, 432, 72),
                    f'{unit}/{method}: incorrect source/state/control denominators')
            result[unit][method] = {'covered': sum(row['covered'] for row in cov),
                                  'generating_branch_covered': sum(row['generating_branch_covered'] is True for row in cov),
                                  'correct_kept': sum(row['status'] == 'KEEP' and exact(row['ae_mm']) == 0 for row in correct),
                                  'negative_small_kept': sum(row['status'] == 'KEEP' for row in controls),
                                  'harm': sum(exact(row['ae_mm']) > exact(row['initial_ae_mm']) for row in values),
                                  'kept': sum(row['status'] == 'KEEP' for row in values),
                                  'incompatible': sum(row['status'] == 'INCOMPATIBLE' for row in values),
                                  'latent_bias_covered': None if method != 'joint' else sum(row['latent_bias_covered'] for row in cov),
                                  'latent_material_covered': None if method not in ('calibrated', 'joint') else sum(row['latent_material_covered'] for row in cov)}
    return {'outcomes': result, 'scientific_policy': 'C1/C2 gains and safety outcomes are reported, not selected by a positive mechanical PASS requirement'}


def verify(output):
    output = Path(output).resolve()
    require(socket.gethostname() == 'liekkas' and ROOT == EXPECTED_ROOT and ROOT.resolve() == EXPECTED_ROOT and
            output.is_dir() and output != ROOT and output.is_relative_to(ROOT), 'target: exact existing local structured experiment required')
    budget = load(output / 'BUDGET.json')
    deadline = verify_budget(budget)
    sources, inputs = verify_source_lock(load(output / 'SOURCE_LOCK.json')), verify_inputs(load(output / 'INPUTS.json'))
    same(load(output / 'PROTOCOL.json'), registration(), 'PROTOCOL independent registered matrix')
    source_audit = audit_runner_source()
    status = load(output / 'RUN_STATUS.json')
    require(isinstance(status, dict) and set(status) == {'status', 'observations_complete', 'decisions_complete', 'new_solver_calls',
            'reused_observations', 'elapsed_seconds', 'deadline_unix'}, 'RUN_STATUS: invalid completed schema')
    same({key: status[key] for key in status if key != 'elapsed_seconds'},
         {'status': 'COMPLETE', 'observations_complete': 108, 'decisions_complete': 3888, 'new_solver_calls': 216,
          'reused_observations': 108, 'deadline_unix': deadline}, 'RUN_STATUS registered complete identities/counts/window')
    elapsed = finite_seconds(status['elapsed_seconds'], 'RUN_STATUS elapsed')
    require(elapsed <= deadline - budget['start_unix'] and elapsed <= time.time() - budget['start_unix'], 'RUN_STATUS: impossible main runtime')
    observations, tasks, source_images, source_tasks = verify_public_population(output)
    decision_list = load(output / 'decisions.json')
    decisions = verify_action_population(decision_list, observations, tasks)
    # Complete unique saved actions are checked above before private truth is parsed.
    model_class, geometry_bounds = _historical.shared_kernels()
    truths, residuals = verify_private_population(output, observations, model_class)
    outer_list = load(output / 'outers.json')
    require(isinstance(outer_list, list) and len(outer_list) == 108, 'outers: incomplete observation population')
    outers = indexed(outer_list, lambda row: row['observation_id'], 'outers')
    require(list(outers) == list(observations), 'outers: changed image identity/order')
    checkpoint = output / 'checkpoint'
    require(checkpoint.is_dir() and sorted(path.name for path in checkpoint.iterdir()) == sorted(f'{oid}.json' for oid in observations),
            'checkpoint: missing/extra image record')
    unions, branches = {}, {}
    updates = certificates = nodes = geometry_calls = wide_witnesses = 0
    for index, (oid, public) in enumerate(observations.items()):
        record = outers[oid]
        require(set(record) == {'observation_id', 'unit_id', 'arms', 'reused_source'} and record['unit_id'] == public['unit_id'] and
                isinstance(record['arms'], dict) and list(record['arms']) == list(METHODS), f'{oid}: changed outer/arm schema')
        same(load(checkpoint / f'{oid}.json'), record, f'{oid}: final checkpoint equality')
        image, clipped = tuple(map(exact, public['observation'])), public['clipped']
        unions[oid], branches[oid] = {}, {}
        common_calibration = None
        for method in METHODS:
            arm, where = record['arms'][method], f'{oid}/{method}'
            fields = {'branches', 'outer', 'branch_seconds', 'outer_seconds'}
            if method in ('calibrated', 'joint'):
                fields |= {'preparation_seconds', 'background_mask', 'calibration_intervals', 'calibration'}
            require(isinstance(arm, dict) and set(arm) == fields, f'{where}: changed arm schema')
            expected_branches = () if method == 'keep' else BRANCHES
            require(list(arm['branches']) == list(arm['branch_seconds']) == list(expected_branches), f'{where}: changed whole-scene branches')
            total = finite_seconds(arm['outer_seconds'], f'{where}: total seconds')
            measured = sum(finite_seconds(value, f'{where}: branch seconds') for value in arm['branch_seconds'].values())
            if method in ('calibrated', 'joint'):
                measured += finite_seconds(arm['preparation_seconds'], f'{where}: background seconds')
                calibration = verify_arm_public(arm, image, clipped, geometry_bounds, where)
                if common_calibration is None:
                    common_calibration = calibration
                else:
                    same(calibration, common_calibration, f'{oid}: calibrated/joint same initializer')
            require(total >= measured and (method != 'keep' or total == 0), f'{where}: impossible measured solve cost')
            checked_branches = {}
            for branch in expected_branches:
                if method == 'wide':
                    model = model_class(branch, camera_radius=RADIUS, colors=((Q(0), Q(1)),) * 3)
                    checked, calls, witnesses = _legacy.verify_branch(arm['branches'][branch], image, SOURCE_UNITS[public['unit_id']],
                                                                    branch, geometry_bounds, model, f'{where}/{branch}')
                    geometry_calls += calls
                    wide_witnesses += witnesses
                else:
                    checked = verify_branch(arm['branches'][branch], image, clipped, method, branch, geometry_bounds, calibration, f'{where}/{branch}')
                    geometry_calls += checked['evaluated']
                updates += checked['proof_updates']
                certificates += len(checked['shared_excluded'])
                nodes += len(arm['branches'][branch]['trace'])
                checked_branches[branch] = checked
            branches[oid][method] = checked_branches
            unions[oid][method] = verify_union(arm['outer'], checked_branches, method, clipped, where)
        if index % 12 == 11:
            print(json.dumps({'stage': 'independent_structured_verification', 'observations_complete': index + 1,
                              'remaining_shared_seconds': round(deadline - time.time(), 1)}), flush=True)
    reused = verify_provenance(output, observations, tasks, outers, decisions, source_images, source_tasks)
    rows, coverage = [], []
    for decision in decision_list:
        task = tasks[decision['task_id']]
        oid, method = task['observation_id'], decision['method']
        checked, private = unions[oid][method], truths[oid]
        value = verify_decision(decision, task, checked, private['unit_id'], f'{task["task_id"]}/{method}')
        row = make_row(task, private, decision, checked)
        if method != 'keep' and checked['hull'] is not None and row['covered']:
            truth, incumbent = exact(private['truth']), exact(task['incumbent'])
            require((incumbent - truth) ** 2 - (value - truth) ** 2 >= exact(decision['gain_lower']) >= 0,
                    f'{task["task_id"]}/{method}: action certificate violates covered actual depth')
        rows.append(row)
    for oid, private in truths.items():
        coverage.extend(make_coverage(observations[oid], private, method, unions[oid][method], branches[oid][method]) for method in METHODS)
    aggregates, pairs = verify_evaluation(output, rows, coverage, observations, outers, status)
    claims = safety_claims(rows, coverage)
    same(scientific_files(), sources, 'scientific graph unchanged during verification')
    same(historical_inputs(), inputs, 'historical evidence unchanged during verification')
    verify_budget(budget)
    return {'status': 'PASS', 'mechanical_verification': 'PASS', 'host': socket.gethostname(), 'output': str(output),
            'observations': 108, 'tasks': 972, 'decisions': 3888, 'new_solver_calls': 216,
            'reused_wide_observations': reused, 'new_nonkeep_actions': 1944, 'reused_wide_actions': 972, 'keep_actions': 972,
            'coverage_rows': len(coverage), 'summary_rows': len(aggregates), 'paired_rows': len(pairs),
            'branch_outer_traces': 972, 'tree_nodes_verified': nodes, 'exact_updates_recomputed': updates,
            'contradiction_certificates_verified': certificates, 'historical_geometry_calls': geometry_calls,
            'wide_channel_witnesses_verified': wide_witnesses, 'source_files_verified': len(sources), 'historical_inputs_verified': len(inputs),
            'source_parse_order_audit': source_audit, 'actual_truth_witnesses': residuals, 'claims': claims,
            'shared_budget': {'start_unix': budget['start_unix'], 'deadline_unix': deadline,
                              'elapsed_main_plus_verification_seconds': time.time() - budget['start_unix']},
            'limitations': ['Frozen geometry/point/area kernels are reused; their physical soundness is not independently proved.',
                            'Necessary outer constraints and every retained contraction are independently recomputed; retention is not exact joint feasibility.',
                            'Legacy wide bodies and actions are copied historical evidence, not new confirmation.',
                            'Recorded cold/historical timing is checked and summed, not independently reproduced.',
                            'Frozen source order review does not prove operating-system blindness.',
                            'The same54 synthetic source observations and related initial states are reused, not independent real scenes.',
                            'Mechanical PASS establishes replay consistency, not a positive C1/C2 outcome or adoption claim.']}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Independently verify the structured view-bias replay.')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/structured-v1')
    output = parser.parse_args(argv).output.resolve()
    if socket.gethostname() != 'liekkas' or ROOT != EXPECTED_ROOT or ROOT.resolve() != EXPECTED_ROOT:
        parser.error('exact local structured experiment on liekkas required')
    if not output.is_dir() or output == ROOT or not output.is_relative_to(ROOT):
        parser.error('existing output strictly inside exact root required')
    destination = output / 'verification'
    if destination.exists():
        print('REFUSE: existing verification is preserved', file=sys.stderr)
        return 2
    destination.mkdir()
    started = time.perf_counter()
    try:
        with deadline_guard(output):
            report = verify(output)
            report.update(utc=datetime.now(timezone.utc).isoformat(), verification_seconds=time.perf_counter() - started)
            with (destination / 'VERIFY.json').open('x', encoding='utf-8') as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write('\n')
    except Exception as error:
        if (destination / 'VERIFY.json').exists():
            print(f'FAIL: interrupted partial verification preserved: {type(error).__name__}: {error}', file=sys.stderr)
            return 1
        report = {'status': 'INCOMPLETE' if isinstance(error, TimeoutError) else 'FAIL',
                  'mechanical_verification': 'INCOMPLETE' if isinstance(error, TimeoutError) else 'FAIL',
                  'host': socket.gethostname(), 'output': str(output), 'error': f'{type(error).__name__}: {error}',
                  'utc': datetime.now(timezone.utc).isoformat(), 'verification_seconds': time.perf_counter() - started}
        with (destination / 'VERIFY.json').open('x', encoding='utf-8') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())

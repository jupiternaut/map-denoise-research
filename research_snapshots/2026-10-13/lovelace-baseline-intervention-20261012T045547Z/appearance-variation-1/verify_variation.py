#!/usr/bin/env python3
"""Independent bounded-appearance replay; no new solver/runner imports.

This verifier authors signed links, propagation, tree replay and per-view
forward images. Frozen independent geometry and endpoint checks are reused.
Private facts may only be parsed after the complete saved public action gate.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from contextlib import contextmanager
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
for _variation_verifier_thread in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_variation_verifier_thread] = '1'

ROOT = Path(__file__).resolve().parent
EXPECTED_ROOT = Path('/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/appearance-variation-1')
PARENT, DATA = ROOT.parent, ROOT / 'data'
APPEARANCE = PARENT / 'appearance-1'
APPEARANCE_OUTPUT = APPEARANCE / 'outputs/appearance-v1'
_spec = importlib.util.spec_from_file_location('_variation_frozen_independent_appearance', APPEARANCE / 'verify_appearance.py')
_appearance = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_appearance)
_baseline, _prior, _arithmetic = _appearance._baseline, _appearance._prior, _appearance._arithmetic
FULL_OUTPUT, ORIGINAL_DATA = _appearance.FULL_OUTPUT, PARENT / 'data'
VerificationError = _appearance.VerificationError
require, same, exact, encoded, pair = _appearance.require, _appearance.same, _appearance.exact, _appearance.encoded, _appearance.pair
load, indexed, finite_seconds = _appearance.load, _appearance.indexed, _appearance.finite_seconds
DOMAIN, TOLERANCE, RADIUS, EPSILON, BETA = _appearance.DOMAIN, _appearance.TOLERANCE, _appearance.RADIUS, _appearance.EPSILON, _appearance.BETA
BRANCHES, ROUNDS, MAX_BOXES = _appearance.BRANCHES, _appearance.ROUNDS, _appearance.MAX_BOXES
METHODS, ARMS = ('calibrated', 'joint'), ('strict_shared', 'independent', 'bounded')
DELTAS = {'strict_shared': Q(0), 'independent': Q(1), 'bounded': Q(1, 10)}
ETAS = (Q(0), Q(1, 10))
MATERIAL_VARIABLE_NAMES = _appearance.MATERIAL_VARIABLE_NAMES
GROUP_SCOPE = 'necessary_bounded_per_view_material_shared_bias_halfspaces_with_historical_continuous_geometry_bounds'
Q_PATTERN = ((-1, 0, 1), (0, 1, -1), (1, -1, 0))
LIMIT = 3600
BUDGET_SCOPE = 'A4-V prepare plus public inference plus independent verification; no reset'
independent_geometry = _baseline.independent_geometry
pixel_bands = _appearance.pixel_bands


def link_rows(method, drift_bound):
    """Own exact eighteen links in registered pair/material/direction order."""
    delta = exact(drift_bound)
    require(method in METHODS and 0 <= delta <= 1, 'links: registered method and legal exact drift bound required')
    dimension = 12 if method == 'joint' else 9
    rows = []
    for left, right in ((0, 1), (0, 2), (1, 2)):
        for material in range(3):
            for direction in (1, -1):
                coefficients = [Q(0)] * dimension
                coefficients[3 * left + material] = Q(direction)
                coefficients[3 * right + material] = Q(-direction)
                rows.append(('upper', tuple(coefficients), delta))
    return tuple(rows)


def necessary_rows(observation, geometry, clipped, method, calibration_intervals, drift_bound):
    return _appearance.necessary_rows(observation, geometry, clipped, method, calibration_intervals) + link_rows(method, drift_bound)


def independent_signed_contraction(rows, initial_bounds, rounds=ROUNDS):
    """Sign-aware exact interval propagation, preserving old nonnegative order."""
    require(type(rounds) is int and 0 <= rounds <= ROUNDS, 'signed propagation: invalid sweep budget')
    priors = [pair(value, 'signed propagation prior') for value in initial_bounds]
    require(len(priors) in (3, 6, 9, 12) and all(0 <= a <= b for a, b in priors), 'signed propagation: invalid dimensional priors')
    normalized = []
    for side, coefficients, bound in rows:
        coefficients = tuple(map(exact, coefficients))
        require(side in ('upper', 'lower') and len(coefficients) == len(priors), 'signed propagation: invalid row')
        normalized.append((side, coefficients, exact(bound)))
    lower, upper = [a for a, _ in priors], [b for _, b in priors]
    initial = [[encoded(a), encoded(b)] for a, b in priors]
    steps, performed, conflict = [], 0, None
    for iteration in range(rounds):
        performed, changed = iteration + 1, False
        for row, (side, coefficients, bound) in enumerate(normalized):
            extrema = tuple(lower[index] if (side == 'upper') == (coefficient >= 0) else upper[index]
                            for index, coefficient in enumerate(coefficients))
            total = sum((coefficient * value for coefficient, value in zip(coefficients, extrema)), Q(0))
            if (side == 'upper' and total > bound) or (side == 'lower' and total < bound):
                conflict = {'kind': 'row', 'round': iteration, 'row': row, 'total': encoded(total), 'bound': encoded(bound)}
                break
            for variable, coefficient in enumerate(coefficients):
                if coefficient == 0:
                    continue
                candidate = (bound - (total - coefficient * extrema[variable])) / coefficient
                endpoint = 'upper' if (side == 'upper') == (coefficient > 0) else 'lower'
                if endpoint == 'upper' and candidate < upper[variable]:
                    upper[variable] = candidate
                elif endpoint == 'lower' and candidate > lower[variable]:
                    lower[variable] = candidate
                else:
                    continue
                changed = True
                steps.append({'round': iteration, 'row': row, 'variable': variable, 'side': endpoint, 'value': encoded(candidate)})
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


def verify_bounded_branch(outer, image, method, branch, geometry_bounds, calibration, delta, where):
    fields = {'domain', 'epsilon', 'beta', 'clipped', 'tolerance', 'max_boxes', 'evaluated_boxes', 'trace',
              'retained_ids', 'hull', 'shared_tests', 'propagation_rounds', 'mode', 'rounds', 'budget',
              'channel_precheck', 'initial_bounds', 'scope', 'appearance_scope', 'material_variable_names'}
    require(isinstance(outer, dict) and set(outer) == fields, f'{where}: invalid bounded branch schema')
    initial = [[encoded(0), encoded(1)] for _ in range(9)]
    if method == 'joint':
        initial.extend(calibration['intervals'])
    same(outer['initial_bounds'], initial, f'{where}: all nine material/same T_cal priors')
    same(outer['material_variable_names'], list(MATERIAL_VARIABLE_NAMES), f'{where}: exact view-major names')
    require(pair(outer['domain'], where) == DOMAIN and exact(outer['epsilon']) == EPSILON and exact(outer['beta']) == BETA and
            outer['clipped'] is True and exact(outer['tolerance']) == TOLERANCE and type(outer['max_boxes']) is int and
            outer['max_boxes'] == MAX_BOXES and outer['mode'] == method and outer['scope'] == GROUP_SCOPE and
            outer['appearance_scope'] == 'per_view' and outer['channel_precheck'] == 'none', f'{where}: changed bounded scientific contract')
    require(all(type(outer[key]) is int and outer[key] == value for key, value in
                (('rounds', ROUNDS), ('propagation_rounds', ROUNDS), ('budget', MAX_BOXES))), f'{where}: changed branch budgets')
    nodes = indexed(outer['trace'], lambda node: node['id'], where)
    require(0 < len(nodes) <= MAX_BOXES and all(type(node['id']) is int for node in outer['trace']) and
            list(nodes) == list(range(len(nodes))) and nodes[0]['parent'] is None, f'{where}: invalid complete BFS identities')
    boxes = {ident: (exact(node['lo']), exact(node['hi'])) for ident, node in nodes.items()}
    require(boxes[0] == DOMAIN, f'{where}: root differs from full domain')
    leaves, retained, excluded, referenced = [], [], [], set()
    evaluated = updates = 0
    next_child = 1
    base = {'id', 'parent', 'lo', 'hi', 'status'}
    extras = {'split': {'children', 'rounds', 'updates', 'darkness_bounds', 't_bounds'},
              'retained_tolerance': {'rounds', 'updates', 'darkness_bounds', 't_bounds'},
              'excluded_shared': {'rounds', 'updates', 'group_id', 'certificate'}, 'excluded_calibration': {'certificate'}}
    for node in outer['trace']:
        ident, status = node['id'], node['status']
        left, right = boxes[ident]
        require(status in extras and set(node) == base | extras[status] and DOMAIN[0] <= left < right <= DOMAIN[1], f'{where}: invalid node {ident}')
        if ident:
            require(type(node['parent']) is int and node['parent'] in nodes and node['parent'] < ident, f'{where}: invalid parent {ident}')
        if calibration['excluded']:
            require(ident == 0 and len(nodes) == 1 and status == 'excluded_calibration', f'{where}: false calibration exclusion')
            same(node['certificate'], calibration['certificate'], f'{where}: calibration exclusion proof')
        else:
            require(status != 'excluded_calibration' and evaluated < MAX_BOXES, f'{where}: unsupported exclusion/budget')
            rows = necessary_rows(image, geometry_bounds(branch, RADIUS, left, right), True, method, calibration['intervals'], delta)
            contraction = independent_signed_contraction(rows, initial)
            evaluated += 1
            require(type(node['rounds']) is int and node['rounds'] == contraction['rounds'] and type(node['updates']) is int and
                    node['updates'] == contraction['updates'], f'{where}: false exact propagation accounting {ident}')
            updates += contraction['updates']
            if contraction['excluded']:
                require(status == 'excluded_shared' and node['group_id'] == 'global', f'{where}: omitted/false contradiction {ident}')
                same(node['certificate'], contraction['certificate'], f'{where}: every signed update and first crossing {ident}')
            else:
                require(status in ('split', 'retained_tolerance'), f'{where}: unsupported exclusion {ident}')
                same(node['darkness_bounds'], contraction['bounds'][:9], f'{where}: all nine exact contracted materials {ident}')
                same(node['t_bounds'], contraction['bounds'][9:] if method == 'joint' else calibration['intervals'], f'{where}: same contracted/frozen view bias {ident}')
        if status == 'split':
            require(right - left > TOLERANCE and node['children'] == [next_child, next_child + 1] and
                    all(type(child) is int and child in nodes for child in node['children']), f'{where}: invalid split allocation')
            middle = (left + right) / 2
            for child, expected in zip(node['children'], ((left, middle), (middle, right))):
                require(child not in referenced and nodes[child]['parent'] == ident and boxes[child] == expected, f'{where}: invalid bisection/child provenance')
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


def verify_arm(arm, public, method, arm_name, where='arm'):
    require(arm_name in ARMS and method in METHODS and isinstance(arm, dict), f'{where}: unregistered arm/method')
    require(arm.get('appearance_model') == arm_name and exact(arm.get('drift_bound')) == DELTAS[arm_name], f'{where}: changed appearance model/design delta')
    require(public.get('clipped') is True and exact(public['radius']) == RADIUS and exact(public['baseline']) == 80,
            f'{where}: changed public B80 camera/error contract')
    original = {key: value for key, value in arm.items() if key not in ('appearance_model', 'drift_bound')}
    if arm_name == 'independent':
        return _appearance.verify_appearance_arm(original, public, method, where)
    fields = {'branches', 'outer', 'branch_seconds', 'outer_seconds', 'preparation_seconds', 'background_mask', 'calibration_intervals', 'calibration'}
    if arm_name == 'bounded':
        fields |= {'appearance_scope', 'material_variable_names'}
        require(original.get('appearance_scope') == 'per_view', f'{where}: bounded sharing scope changed')
        same(original.get('material_variable_names'), list(MATERIAL_VARIABLE_NAMES), f'{where}: bounded material names changed')
    require(set(original) == fields and list(original['branches']) == list(original['branch_seconds']) == list(BRANCHES), f'{where}: changed frozen arm schema/branch order')
    image = tuple(map(exact, public['observation']))
    geometry_bounds = lambda branch, radius, low, high: independent_geometry(branch, radius, low, high, Q(80))
    calibration = _prior.verify_arm_public(original, image, True, geometry_bounds, where)
    measured = finite_seconds(original['preparation_seconds'], where + ': preparation')
    measured += sum(finite_seconds(value, where + ': branch') for value in original['branch_seconds'].values())
    require(finite_seconds(original['outer_seconds'], where + ': solve') >= measured, f'{where}: impossible recorded solve timing')
    if arm_name == 'strict_shared':
        branches = {branch: _prior.verify_branch(original['branches'][branch], image, True, method, branch,
                                                geometry_bounds, calibration, f'{where}/{branch}') for branch in BRANCHES}
    else:
        branches = {branch: verify_bounded_branch(original['branches'][branch], image, method, branch, geometry_bounds,
                                                 calibration, DELTAS[arm_name], f'{where}/{branch}') for branch in BRANCHES}
    union = _prior.verify_union(original['outer'], branches, method, True, where)
    counts = {'tree_nodes': sum(len(original['branches'][branch]['trace']) for branch in BRANCHES),
              'exact_updates': sum(value['proof_updates'] for value in branches.values()),
              'contradiction_certificates': sum(len(value['shared_excluded']) for value in branches.values()),
              'geometry_calls': sum(value['evaluated'] for value in branches.values())}
    return {'union': union, 'branches': branches, 'counts': counts, 'calibration': calibration}


def variation_darkness(colors, eta):
    eta = exact(eta)
    require(eta in ETAS, 'forward variation: registered exact eta required')
    colors = tuple(map(exact, colors))
    require(len(colors) == 3 and all(0 <= color <= 1 for color in colors), 'forward variation: three legal base materials required')
    darkness = tuple(1 - color for color in colors)
    values = tuple(darkness[k] + eta * Q_PATTERN[view][k] * min(darkness[k], 1 - darkness[k]) for view in range(3) for k in range(3))
    require(all(0 <= value <= 1 for value in values), 'forward variation: illegal nine-material construction')
    return values


def independent_rgb_per_view(branch, depth, offsets, colors9, baseline=Q(80)):
    colors = tuple(map(exact, colors9))
    require(len(colors) == 9 and all(0 <= color <= 1 for color in colors), 'forward image: nine legal view-major colors required')
    image = []
    for pixel, areas in enumerate(_baseline.independent_point_areas(branch, depth, offsets, baseline)):
        view = pixel // 48
        gray = 1 - sum((area * (1 - color) for area, color in zip(areas, colors[3 * view:3 * view + 3])), Q(0))
        require(0 <= gray <= 1, 'forward image: illegal exact grayscale')
        image.extend((gray,) * 3)
    return tuple(image)


def replay_forward(source_public, source_fact, eta):
    """Pure forward interface; caller supplies facts only beyond action gate."""
    require(source_public['baseline'] == source_fact['baseline'] == '80' and source_public['clipped'] is True and
            exact(source_public['radius']) == RADIUS, 'forward replay: exact frozen B80 source required')
    darkness = variation_darkness(source_fact['colors'], eta)
    colors = tuple(1 - value for value in darkness)
    predicted = independent_rgb_per_view(source_fact['rear_branch'], source_fact['truth'], source_fact['actual_offsets'], colors)
    noise = tuple(map(exact, source_fact['image_noise']))
    bias = tuple(map(exact, source_fact['actual_view_bias']))
    require(len(noise) == 432 and all(abs(value) <= EPSILON for value in noise) and bias == (-BETA, Q(0), BETA),
            'forward replay: changed frozen noise/bias tape')
    image = tuple(min(Q(1), max(Q(0), value + noise[index] + bias[index // 144])) for index, value in enumerate(predicted))
    if exact(eta) == 0:
        same(list(map(encoded, image)), source_public['observation'], 'forward replay: eta0 exact frozen public image identity')
    spread = max(abs(darkness[3 * v + k] - darkness[3 * w + k]) for v in range(3) for w in range(v + 1, 3) for k in range(3))
    require(spread <= exact(eta), 'forward replay: constructed true spread exceeds design bound')
    return {'observation': list(map(encoded, image)), 'darkness9': list(map(encoded, darkness)),
            'colors9': list(map(encoded, colors)), 'material_spread': encoded(spread)}


def check_target(output=None):
    require(socket.gethostname() == 'liekkas' and ROOT == EXPECTED_ROOT and ROOT.resolve() == EXPECTED_ROOT,
            'target: exact appearance variation child on liekkas required')
    if output is not None:
        output = Path(output).resolve()
        require(output.is_dir() and output != ROOT and output.is_relative_to(ROOT), 'target: existing output inside exact variation child required')
    return output


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scientific_files():
    files = dict(_appearance.scientific_files())
    for path in sorted({ROOT / 'AGENTS.md', ROOT / 'PROTOCOL.md', ROOT / 'REGISTRATION.json', *ROOT.glob('*.py')}):
        files[str(path.resolve())] = sha(path)
    return dict(sorted(files.items()))


def input_files():
    files = dict(_appearance.input_files())
    for relative in ('SOURCE_LOCK.json', 'RUN_STATUS.json', 'verification/VERIFY.json', 'outers.json', 'decisions.json'):
        path = APPEARANCE_OUTPUT / relative
        files[str(path.resolve())] = sha(path)
    for relative in ('observations.json', 'tasks.json', 'private/truth.json', 'GENERATION.json'):
        path = DATA / relative
        files[str(path.resolve())] = sha(path)
    return dict(sorted(files.items()))


def verify_source_lock(lock):
    require(isinstance(lock, dict) and set(lock) == {'source_root', 'files'} and lock['source_root'] == str(ROOT), 'SOURCE_LOCK: exact variation source identity/schema required')
    files = scientific_files()
    same(lock['files'], files, 'SOURCE_LOCK: complete independently enumerated frozen/new source graph')
    _appearance.verify_source_lock(load(APPEARANCE_OUTPUT / 'SOURCE_LOCK.json'))
    return files


def verify_inputs(lock):
    require(isinstance(lock, dict) and set(lock) == {'files'}, 'INPUTS: exact input lock schema required')
    files = input_files()
    require(len(files) == 21, 'INPUTS: exact registered21 source/data dependencies required')
    same(lock['files'], files, 'INPUTS: all21 independently enumerated sealed byte identities')
    _appearance.verify_inputs(load(APPEARANCE_OUTPUT / 'INPUTS.json'))
    receipt, status = load(APPEARANCE_OUTPUT / 'verification/VERIFY.json'), load(APPEARANCE_OUTPUT / 'RUN_STATUS.json')
    require(receipt.get('status') == 'PASS' and receipt.get('host') == 'liekkas' and receipt.get('output') == str(APPEARANCE_OUTPUT) and status.get('status') == 'COMPLETE',
            'INPUTS: exact independently verified frozen A4 required')
    return files


def verify_budget(budget):
    require(isinstance(budget, dict) and set(budget) == {'start_epoch', 'deadline_epoch', 'limit_seconds', 'scope'}, 'BUDGET: invalid absolute shared window')
    start, end = (finite_seconds(budget[key], 'BUDGET ' + key) for key in ('start_epoch', 'deadline_epoch'))
    require(type(budget['limit_seconds']) is int and budget['limit_seconds'] == LIMIT and end == start + LIMIT and budget['scope'] == BUDGET_SCOPE,
            'BUDGET: changed preparation/main/verification shared3600-second window')
    if time.time() >= end:
        raise TimeoutError('A4-V shared preparation/main/verification deadline exhausted')
    return end


@contextmanager
def deadline_guard(output):
    deadline = verify_budget(load(Path(output) / 'BUDGET.json'))
    require(not signal.getitimer(signal.ITIMER_REAL)[0], 'BUDGET: existing deadline cannot be reset')
    def reached(_signal, _frame):
        raise TimeoutError('A4-V shared preparation/main/verification deadline reached')
    previous = signal.signal(signal.SIGALRM, reached)
    signal.setitimer(signal.ITIMER_REAL, max(0.000001, deadline - time.time()))
    try:
        yield deadline
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def verify_public_population(output):
    old_public = indexed(load(ORIGINAL_DATA / 'observations.json'), lambda row: row['observation_id'], 'frozen original public')
    old_tasks = indexed(load(ORIGINAL_DATA / 'tasks.json'), lambda row: row['task_id'], 'frozen original tasks')
    same(list(old_public), [f'p{i:03d}b{b}' for i in range(1, 19) for b in (70, 80)], 'population: frozen source36 public identities')
    require(len(old_tasks) == 324, 'population: complete frozen source324 tasks required')
    records, task_records = load(DATA / 'observations.json'), load(DATA / 'tasks.json')
    require(isinstance(records, list) and len(records) == 36 and isinstance(task_records, list) and len(task_records) == 324,
            'population: complete36 public/324 tasks required')
    observations = indexed(records, lambda row: row['observation_id'], 'variation public images')
    tasks = indexed(task_records, lambda row: row['task_id'], 'variation public tasks')
    same(list(observations), [f'p{i:03d}b80e{variation}' for variation in range(2) for i in range(1, 19)], 'population: registered eta-major identity/order')
    expected_tasks = []
    for oid, public in observations.items():
        variation = int(oid[-1])
        base = oid[:-2]
        source = old_public[base]
        require(set(public) == set(source) | {'base_observation_id', 'variation_index', 'eta'}, f'{oid}: leaked/changed public fields')
        expected = {**source, 'observation_id': oid, 'base_observation_id': base, 'variation_index': variation,
                    'unit_id': 'appearance_eta_0' if variation == 0 else 'appearance_eta_1_10', 'eta': encoded(ETAS[variation])}
        same({key: public[key] for key in public if key != 'observation'}, {key: expected[key] for key in expected if key != 'observation'}, f'{oid}: public intervention metadata')
        require(type(public['variation_index']) is int and public['clipped'] is True and exact(public['baseline']) == 80 and exact(public['radius']) == RADIUS,
                f'{oid}: changed B80 public scientific contract')
        image = tuple(map(exact, public['observation']))
        require(len(image) == 432 and all(0 <= value <= 1 for value in image), f'{oid}: complete432 exact clipped image required')
        if variation == 0:
            same(public['observation'], source['observation'], f'{oid}: eta0 frozen image identity before inference/private evaluation')
        for state in range(9):
            old = old_tasks[f'{base}s{state}']
            expected_tasks.append({**old, 'task_id': f'{oid}s{state}', 'observation_id': oid, 'base_task_id': old['task_id']})
    same(list(tasks.values()), expected_tasks, 'population: all324 exact inherited initial states')
    same(load(output / 'observations.json'), records, 'population: all36 saved public image readback')
    same(load(output / 'tasks.json'), task_records, 'population: all324 saved public task readback')
    return observations, tasks, old_public, old_tasks


def reused_configuration(public, arm_name):
    return public['variation_index'] == 0 and arm_name != 'bounded'


def verify_action_population(records, observations, tasks):
    require(isinstance(records, list) and len(records) == 1944, 'actions: complete1944 saved method actions required')
    actions = indexed(records, lambda row: (row['task_id'], row['appearance_model'], row['method']), 'saved method actions')
    expected = [(f'{oid}s{state}', arm, method) for oid in observations for arm in ARMS for method in METHODS for state in range(9)]
    same(list(actions), expected, 'actions: exact complete ordered36/three-arm/two-method/nine-state matrix')
    require({tid for tid, _arm, _method in expected} == set(tasks), 'actions: incomplete public task mapping')
    for row in records:
        public = observations[tasks[row['task_id']]['observation_id']]
        require(type(row.get('reused')) is bool and row['reused'] is reused_configuration(public, row['appearance_model']), 'actions: false new/frozen origin')
    require(Counter(row['reused'] for row in records) == {False: 1296, True: 648}, 'actions: changed1296new/648reuse split')
    return actions


def verify_action(decision, task, public, checked, arm_name, where='action'):
    require(isinstance(decision, dict) and decision.get('method') in METHODS and arm_name in ARMS, f'{where}: unregistered action')
    action = _arithmetic.hybrid_action(checked['hull'], task['incumbent'])
    expected = {'task_id': task['task_id'], 'unit_id': public['unit_id'], 'method': decision['method'],
                'appearance_model': arm_name, 'reused': reused_configuration(public, arm_name),
                **{key: None if value is None else encoded(value) if key in ('output', 'gain_lower') else value for key, value in action.items()}}
    same(decision, expected, f'{where}: independent hybrid action and exact endpoint certificate')
    return exact(decision['output'])


def verify_keep_population(records, observations, tasks):
    require(isinstance(records, list) and len(records) == 324, 'KEEP: complete324 saved identities required')
    same([row['task_id'] for row in records], list(tasks), 'KEEP: complete ordered unique public initial states')
    for record in records:
        task = tasks[record['task_id']]
        public = observations[task['observation_id']]
        same(record, {'task_id': task['task_id'], 'unit_id': public['unit_id'], 'method': 'keep', 'appearance_model': 'keep',
             'reused': True, 'output': task['incumbent'], 'gain_lower': encoded(0), 'status': 'KEEP', 'reason': 'identity', 'branch': 'keep'},
             f'{task["task_id"]}: exact KEEP reference action')


def sealed_arm(arm, frozen, public, method, name, where):
    """Reuse only exact frozen traces already covered by sealed PASS receipts."""
    require(reused_configuration(public, name), f'{where}: false reuse scope')
    expected = {**frozen, 'appearance_model': name, 'drift_bound': encoded(DELTAS[name])}
    same(arm, expected, f'{where}: exact frozen source arm including every trace/calibration byte field')
    branches = {}
    for branch in BRANCHES:
        outer = frozen['branches'][branch]
        retained = [node for node in outer['trace'] if node['id'] in outer['retained_ids']]
        intervals = [(exact(node['lo']), exact(node['hi'])) for node in retained]
        branches[branch] = {'retained': retained, 'intervals': intervals, 'hull': None if outer['hull'] is None else pair(outer['hull'], where)}
    union = frozen['outer']
    return {'union': {'hull': None if union['hull'] is None else pair(union['hull'], where),
                     'intervals': [pair(value, where) for value in union['retained_intervals']],
                     'retained_branches': union['retained_branches'], 'evaluated': union['evaluated_boxes']},
            'branches': branches, 'calibration': frozen['calibration'], 'counts': {}}


def verify_generation(generation, deadline):
    expected = {'stage': 'A4-V', 'source_seed': 61005, 'eta_values': ['0', '1/10'], 'pattern': [list(row) for row in Q_PATTERN],
                'observations': 36, 'tasks': 324, 'forward': 'per-view exact integral + frozen noise + frozen bias, then one clip',
                'no_new_random_draws': True, 'public_bound': '1/10', 'bound_source': 'analytic intervention design, not learned', 'deadline_epoch': deadline}
    same(generation, expected, 'GENERATION: exact one-shot fixed-q forward intervention/common absolute deadline')


def verify_private_population(observations, tasks, old_public):
    """Only called after all public trees and all saved method/KEEP actions."""
    source = indexed(load(ORIGINAL_DATA / 'private/truth.json'), lambda row: row['observation_id'], 'sealed original private facts')
    same(list(source), [f'p{i:03d}b{b}' for i in range(1, 19) for b in (70, 80)], 'private: frozen original36 identities')
    records = load(DATA / 'private/truth.json')
    require(isinstance(records, list) and len(records) == 36, 'private: complete36 forward facts required')
    facts = indexed(records, lambda row: row['observation_id'], 'sealed variation facts')
    same(list(facts), list(observations), 'private: exact36 eta-major fact identities/order')
    witnesses = []
    for oid, public in observations.items():
        base, variation = public['base_observation_id'], public['variation_index']
        original = source[base]
        replay = replay_forward(old_public[base], original, ETAS[variation])
        same(public['observation'], replay['observation'], f'{oid}: independent exact per-view/noise/bias/clip forward image')
        unchanged = replay['observation'] == old_public[base]['observation']
        expected = {**original, 'observation_id': oid, 'base_observation_id': base, 'unit_id': public['unit_id'],
                    'variation_index': variation, 'eta': encoded(ETAS[variation]), 'view_colors': replay['colors9'],
                    'actual_material_spread': replay['material_spread'], 'image_unchanged': unchanged}
        same(facts[oid], expected, f'{oid}: complete independent source-identical forward fact lineage')
        truth = exact(original['truth'])
        require(truth in _baseline._historical.TRUTHS, f'{oid}: changed registered depth')
        areas = _baseline.independent_point_areas(original['rear_branch'], truth, original['actual_offsets'], Q(80))
        darkness, bias = tuple(map(exact, replay['darkness9'])), tuple(map(exact, original['actual_view_bias']))
        for pixel, (high, low) in enumerate(pixel_bands(public['observation'], True)):
            view = pixel // 48
            value = sum((area * d for area, d in zip(areas[pixel], darkness[3 * view:3 * view + 3])), Q(0)) + BETA - bias[view]
            require((high is None or value <= high) and (low is None or value >= low), f'{oid}: actual nine-material clipped pixel witness failed')
        require(exact(replay['material_spread']) <= DELTAS['bounded'], f'{oid}: true nine-material witness violates public bounded design')
        for state, offset in enumerate(_baseline._historical.OFFSETS):
            same(tasks[f'{oid}s{state}']['incumbent'], encoded(truth + offset), f'{oid}: all nine inherited offsets')
        witnesses.append({'observation_id': oid, 'base_observation_id': base, 'eta': encoded(ETAS[variation]),
                          'actual_material_spread': replay['material_spread'], 'image_unchanged': unchanged,
                          'clip_aware_nine_material_witness_valid': True})
    return facts, witnesses


def make_coverage(public, fact, method, checked, model):
    truth, branch, hull = exact(fact['truth']), fact['rear_branch'], checked['union']['hull']
    nodes = {name: [node for node in value['retained'] if exact(node['lo']) <= truth <= exact(node['hi'])] for name, value in checked['branches'].items()}
    leaves = nodes[branch]
    required = model != 'strict_shared' or public['variation_index'] == 0
    dark = tuple(1 - exact(c) for c in (fact['colors'] if model == 'strict_shared' else fact['view_colors']))
    bias = tuple(BETA - exact(value) for value in fact['actual_view_bias'])
    def contained(node, key, values):
        return len(node[key]) == len(values) and all(exact(a) <= value <= exact(b) for (a, b), value in zip(node[key], values))
    material = any(contained(node, 'darkness_bounds', dark) for node in leaves) if required else None
    bias_ok = any(contained(node, 't_bounds', bias) for node in leaves) if method == 'joint' else None
    witness = any(contained(node, 'darkness_bounds', dark) and (method != 'joint' or contained(node, 't_bounds', bias)) for node in leaves) if required else None
    return {'observation_id': public['observation_id'], 'base_observation_id': public['base_observation_id'],
            'variation_index': public['variation_index'], 'unit_id': public['unit_id'], 'method': method, 'appearance_model': model,
            'covered': any(nodes.values()), 'generating_branch': branch, 'generating_branch_covered': bool(leaves), 'incompatible': hull is None,
            'latent_material_covered': material, 'latent_bias_covered': bias_ok, 'joint_witness_covered': witness, 'material_witness_required': required,
            'hull_width_mm': None if hull is None else str(hull[1] - hull[0]), 'branch_coverage': {name: bool(value) for name, value in nodes.items()}}


def validate_required_coverage(records):
    require(isinstance(records, list) and len(records) == 216, 'coverage: complete216 method configurations required')
    require(len({(row['observation_id'], row['appearance_model'], row['method']) for row in records}) == 216, 'coverage: duplicated/missing configuration')
    for row in records:
        required = row['appearance_model'] != 'strict_shared' or row['variation_index'] == 0
        require(type(row['material_witness_required']) is bool and row['material_witness_required'] is required, 'coverage: changed registered witness applicability')
        if required:
            require(all(row[key] is True for key in ('covered', 'generating_branch_covered', 'latent_material_covered', 'joint_witness_covered')) and
                    row['incompatible'] is False and (row['latent_bias_covered'] is True if row['method'] == 'joint' else row['latent_bias_covered'] is None),
                    f'{row["observation_id"]}/{row["appearance_model"]}/{row["method"]}: required physical generating-branch/material/bias joint witness missing')
        else:
            require(row['latent_material_covered'] is None and row['joint_witness_covered'] is None,
                    'coverage: strict nonzero misspecification witness must be marked inapplicable')


def verify_summary_denominators(summary):
    for name, cells in summary.items():
        configurations = [(method, arm) for arm in ARMS for method in METHODS] if name == 'methods' else [('keep', 'keep')]
        for variation, unit in enumerate(('appearance_eta_0', 'appearance_eta_1_10')):
            for method, condition in configurations:
                selected = {(row['group'], row['value']): row for row in cells if (row['unit_id'], row['method'], row['condition']) == (unit, method, condition)}
                for group, count in (('all_shifted', 144), ('correct', 18), ('informative_shifted', 120), ('negative_shifted', 24)):
                    require(selected.get((group, ''), {}).get('n') == count, f'{name}/{unit}/{method}/{condition}: changed {group} denominator')
                depths = [row for (group, _value), row in selected.items() if group == 'depth_informative']
                cases = [row for (group, _value), row in selected.items() if group == 'case_shifted']
                require(len(depths) == 3 and all(row['n'] == 40 for row in depths) and len(cases) == 6 and all(row['n'] == 24 for row in cases), 'summary: changed registered depth/template matrix')


def verify_registered_keeps(results, facts, tasks):
    """Keep obligations apply only to registered physically covered/control cases."""
    for row in results:
        fact = facts[row['observation_id']]
        model = row['condition']
        compatible = model != 'strict_shared' or fact['variation_index'] == 0
        correct = exact(row['initial_ae_mm']) == 0
        negative = fact['case'] == 'equal_full_negative' and fact['image_unchanged'] is True
        if ((compatible or row['covered'] is True) and correct) or negative:
            require(row['status'] == 'KEEP' and exact(row['output']) == exact(tasks[row['task_id']]['incumbent']),
                    f'{row["task_id"]}/{model}/{row["method"]}: registered correct/unchanged negative must KEEP')


def audit_runner_source(source=None):
    path = ROOT / 'run_variation.py'
    code = path.read_text(encoding='utf-8') if source is None else source
    tree = ast.parse(code)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    require({'run', 'evaluate', 'check_actions', 'check_keep', 'population'} <= set(functions), 'source audit: missing public/persisted entry points')
    dump = lambda node: ast.dump(node, include_attributes=False)
    run = functions['run']
    evaluations = [node for node in ast.walk(run) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'evaluate']
    require(len(evaluations) == 1, 'source audit: exactly one private evaluation required')
    evaluation = evaluations[0].lineno
    callers = [(function.name, node.lineno) for function in functions.values() for node in ast.walk(function) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'evaluate']
    require(callers == [('run', evaluation)], 'source audit: additional private evaluation route')
    statements = ("save(output/'outers.json',outers)", "save(output/'decisions.json',decisions)", "save(output/'keep/decisions.json',keep)",
                  "decisions=load(output/'decisions.json')", "outers=load(output/'outers.json')", "keep=load(output/'keep/decisions.json')", "check_actions(decisions,tasks)", "check_keep(keep,tasks)",
                  "if calls!=144 or reuses!=72 or len(outers)!=36 or len(keep)!=324:\n    raise RuntimeError('incomplete solve/keep matrix')")
    tries = [node for node in run.body if isinstance(node, ast.Try)]
    require(len(tries) == 1, 'source audit: changed public action/evaluation flow')
    positions = []
    for statement in statements:
        target = dump(ast.parse(statement).body[0])
        found = [node for node in tries[0].body if dump(node) == target]
        require(len(found) == 1 and found[0].end_lineno < evaluation, 'source audit: complete saved-readback1944 action/KEEP gate missing before private facts')
        positions.append(found[0].lineno)
    require(positions == sorted(positions), 'source audit: reordered saved action matrix gates')
    predicates = ('len(keys)!=1944 or len(set(keys))!=1944 or set(keys)!=expected',
                  "sum(r['reused'] is False for r in records)!=1296 or sum(r['reused'] is True for r in records)!=648")
    for predicate in predicates:
        found = [node for node in functions['check_actions'].body if isinstance(node, ast.If) and dump(node.test) == dump(ast.parse(predicate, mode='eval').body) and any(isinstance(child, ast.Raise) for child in node.body)]
        require(len(found) == 1, 'source audit: complete unique action/origin population gate changed')
    for predicate in ("len(records)!=324 or len({r['task_id'] for r in records})!=324 or {r['task_id'] for r in records}!=set(expected)",
                      "any(r['method']!='keep' or r['status']!='KEEP' or r['output']!=expected[r['task_id']] for r in records)"):
        found = [node for node in functions['check_keep'].body if isinstance(node, ast.If) and dump(node.test) == dump(ast.parse(predicate, mode='eval').body) and any(isinstance(child, ast.Raise) for child in node.body)]
        require(len(found) == 1, 'source audit: exact complete unique KEEP identity guard changed')
    private = [(function.name, node) for function in functions.values() for node in ast.walk(function) if isinstance(node, ast.Constant) and isinstance(node.value, str) and 'truth.json' in node.value]
    require(sorted(name for name, _node in private) == ['evaluate', 'inputs'], 'source audit: private path outside hashes/post-action evaluation')
    require(len(private) == sum(isinstance(node, ast.Constant) and isinstance(node.value, str) and 'truth.json' in node.value for node in ast.walk(tree)), 'source audit: private path at module/class scope')
    parses = [node for node in ast.walk(functions['evaluate']) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'load' and node.args and dump(node.args[0]) == dump(ast.parse("DATA/'private/truth.json'", mode='eval').body)]
    require(len(parses) == 1, 'source audit: exact sealed new private parse path required')
    calls = [node for node in ast.walk(run) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'solve_public']
    require(len(calls) == 1 and [dump(arg) for arg in calls[0].args] == [dump(ast.parse(expression, mode='eval').body) for expression in
            ("p['observation']", "p['radius']", "p['baseline']", 'method', 'DRIFT[model]')] and
            [(keyword.arg, dump(keyword.value)) for keyword in calls[0].keywords] == [('clipped', dump(ast.parse("p['clipped']", mode='eval').body))], 'source audit: solver receives changed/nonpublic/eta arguments')
    return {'status': 'PASS', 'runner': str(path), 'post_action_evaluation_call_line': evaluation, 'private_parse_line': parses[0].lineno,
            'persisted_matrix_gate_lines': positions, 'solver_arguments': ['observation', 'radius', 'baseline', 'method', 'public arm drift_bound', 'clipped']}


def audit_executor_source(source=None):
    """Verify that preparation, public inference and replay share one window."""
    path = ROOT / 'execute_variation.py'
    tree = ast.parse(path.read_text(encoding='utf-8') if source is None else source)
    execute = next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'execute'), None)
    require(execute is not None, 'executor audit: missing registered entry point')
    dump = lambda node: ast.dump(node, include_attributes=False)
    trials = [node for node in execute.body if isinstance(node, ast.Try)]
    require(len(trials) == 1, 'executor audit: changed owned stage flow')
    expressions = ("start=time.time()", "deadline=start+runner.LIMIT",
                   "runner.save(output/'BUDGET.json',{'start_epoch':start,'deadline_epoch':deadline,'limit_seconds':runner.LIMIT,'scope':runner.BUDGET_SCOPE})",
                   "subprocess.run([sys.executable,'-B',str(ROOT/'prepare_variation.py'),'--deadline',str(deadline)],check=True,timeout=max(0.001,deadline-time.time()))",
                   "runner.run(output)", "remaining=deadline-time.time()",
                   "subprocess.run([sys.executable,'-B',str(ROOT/'verify_variation.py'),'--output',str(output)],check=True,timeout=remaining)")
    positions = []
    for expression in expressions:
        target = dump(ast.parse(expression).body[0])
        found = [node for node in trials[0].body if dump(node) == target]
        require(len(found) == 1, 'executor audit: exact common-deadline preparation/main/verifier process chain changed')
        positions.append(found[0].lineno)
    require(positions == sorted(positions), 'executor audit: reset/reordered shared budget chain')
    processes = [node for node in ast.walk(execute) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and dump(node.func) == dump(ast.parse('subprocess.run', mode='eval').body)]
    require(len(processes) == 2 and sum(isinstance(node, ast.Constant) and node.value == 'BUDGET.json' for node in ast.walk(execute)) == 1,
            'executor audit: additional subprocess/budget window')
    require(not any(isinstance(node, ast.Constant) and isinstance(node.value, str) and 'truth.json' in node.value for node in ast.walk(tree)),
            'executor audit: executor must not parse private facts')
    return {'status': 'PASS', 'executor': str(path), 'common_deadline_stage_lines': positions, 'private_generation_process': 'prepare_variation.py', 'budget_resets': 0}


def verify(output):
    output = check_target(output)
    budget = load(output / 'BUDGET.json')
    deadline = verify_budget(budget)
    sources, inputs = verify_source_lock(load(output / 'SOURCE_LOCK.json')), verify_inputs(load(output / 'INPUTS.json'))
    same(load(output / 'REGISTRATION.json'), load(ROOT / 'REGISTRATION.json'), 'REGISTRATION: exact saved source copy')
    verify_generation(load(DATA / 'GENERATION.json'), deadline)
    audit = audit_runner_source()
    executor_audit = audit_executor_source()
    observations, tasks, old_public, _old_tasks = verify_public_population(output)
    records, keeps = load(output / 'decisions.json'), load(output / 'keep/decisions.json')
    decisions = verify_action_population(records, observations, tasks)
    verify_keep_population(keeps, observations, tasks)
    outers, full, independent = load(output / 'outers.json'), load(FULL_OUTPUT / 'outers.json'), load(APPEARANCE_OUTPUT / 'outers.json')
    require(isinstance(outers, dict) and list(outers) == list(observations), 'outers: complete36 ordered public observations required')
    old_actions = {'strict_shared': indexed(load(FULL_OUTPUT / 'decisions.json'), lambda row: (row['task_id'], row['method']), 'sealed global actions'),
                   'independent': indexed(load(APPEARANCE_OUTPUT / 'decisions.json'), lambda row: (row['task_id'], row['method']), 'sealed independent actions')}
    timing_records = load(output / 'TIMING.json')
    require(isinstance(timing_records, list) and len(timing_records) == 144, 'TIMING: complete144 cold calls required')
    timing = indexed(timing_records, lambda row: (row['observation_id'], row['appearance_model'], row['method']), 'cold timing')
    same(list(timing), [(oid, model, method) for oid, public in observations.items() for model in ARMS if not reused_configuration(public, model) for method in METHODS], 'TIMING: exact registered144 cold call order')
    checked, counts = {}, Counter()
    for index, (oid, public) in enumerate(observations.items()):
        require(isinstance(outers[oid], dict) and list(outers[oid]) == list(ARMS), f'{oid}: changed three-arm order')
        checked[oid] = {}
        for model in ARMS:
            require(isinstance(outers[oid][model], dict) and list(outers[oid][model]) == list(METHODS), f'{oid}/{model}: changed method order')
            checked[oid][model] = {}
            for method in METHODS:
                arm = outers[oid][model][method]
                where, base = f'{oid}/{model}/{method}', public['base_observation_id']
                if reused_configuration(public, model):
                    frozen = (full if model == 'strict_shared' else independent)[base][method]
                    result = sealed_arm(arm, frozen, public, method, model, where)
                    counts['reused_solver_calls'] += 1
                else:
                    result = verify_arm(arm, public, method, model, where)
                    counts.update(result['counts'])
                    counts['cold_solver_calls'] += 1
                    measured = timing[oid, model, method]
                    require(set(measured) == {'observation_id', 'appearance_model', 'method', 'cold_seconds', 'evaluated_boxes'} and
                            type(measured['evaluated_boxes']) is int and measured['evaluated_boxes'] == result['union']['evaluated'] and
                            finite_seconds(measured['cold_seconds'], where + ': cold') >= finite_seconds(arm['outer_seconds'], where + ': solve'), f'{where}: false cold timing/tree count')
                same(result['calibration'], full[base][method]['calibration'], f'{where}: unchanged guaranteed-background full-image T_cal')
                checked[oid][model][method] = result
        if index % 3 == 2:
            print(json.dumps({'stage': 'independent_variation_verification', 'observations_complete': index + 1, 'remaining_seconds': round(deadline - time.time(), 1)}), flush=True)
    require(counts['cold_solver_calls'] == 144 and counts['reused_solver_calls'] == 72, 'outers: changed conceptual/cold/reuse solve matrix')
    checkpoint = output / 'checkpoint'
    require(checkpoint.is_dir() and sorted(path.name for path in checkpoint.iterdir()) == sorted(f'{oid}.json' for oid in observations), 'checkpoint: complete36 exact image checkpoints')
    for oid in observations:
        same(load(checkpoint / f'{oid}.json'), {'observation_id': oid, 'arms': outers[oid]}, f'{oid}: persisted checkpoint readback')
    for record in records:
        task = tasks[record['task_id']]
        oid, model, method = task['observation_id'], record['appearance_model'], record['method']
        verify_action(record, task, observations[oid], checked[oid][model][method]['union'], model, f'{task["task_id"]}/{model}/{method}')
        if record['reused']:
            frozen = old_actions[model][task['base_task_id'], method]
            same({key: record[key] for key in ('status', 'output', 'gain_lower', 'reason', 'branch')}, {key: frozen[key] for key in ('status', 'output', 'gain_lower', 'reason', 'branch')}, f'{task["task_id"]}/{model}/{method}: frozen eta0 endpoint identity')
    # Only now have all public arms, all1944 saved method actions and324 KEEP
    # identities passed. This is the only private-fact parse gate in verify().
    facts, witnesses = verify_private_population(observations, tasks, old_public)
    results, keep_results, coverage = [], [], []
    for record in records:
        task = tasks[record['task_id']]
        oid, model, method = task['observation_id'], record['appearance_model'], record['method']
        union = checked[oid][model][method]['union']
        _appearance.verify_safe_result(task, facts[oid], record, union, f'{task["task_id"]}/{model}/{method}')
        results.append(_appearance.make_result(task, facts[oid], record, union, model))
    for record in keeps:
        task = tasks[record['task_id']]
        keep_results.append(_appearance.make_result(task, facts[task['observation_id']], record, {'hull': None}, 'keep'))
    for oid, public in observations.items():
        for model in ARMS:
            for method in METHODS:
                coverage.append(make_coverage(public, facts[oid], method, checked[oid][model][method], model))
    validate_required_coverage(coverage)
    verify_registered_keeps(results, facts, tasks)
    same(load(output / 'results.json'), results, 'evaluation: all1944 exact independent method endpoints/errors including misspecified strict')
    same(load(output / 'keep/results.json'), keep_results, 'evaluation: all324 exact KEEP reference errors')
    same(load(output / 'coverage.json'), coverage, 'evaluation: all216 independent physical witness/applicability records')
    for relative, rows in (('results.csv', results), ('keep/results.csv', keep_results)):
        _baseline.verify_csv(output / relative, rows)
    summary = {'methods': _appearance.aggregate_results(results), 'keep': _appearance.aggregate_results(keep_results)}
    verify_summary_denominators(summary)
    same(load(output / 'summary.json'), summary, 'evaluation: all exact eta/model/method/depth/template cells')
    diagnostic = [{'observation_id': oid, 'base_observation_id': public['base_observation_id'], 'eta': public['eta'],
                   'actual_material_spread': facts[oid]['actual_material_spread'], 'image_unchanged': facts[oid]['image_unchanged']} for oid, public in observations.items()]
    same(load(output / 'intervention_diagnostic.json'), diagnostic, 'evaluation: exact actual-spread/changed-image diagnostic after action gate')
    status = load(output / 'RUN_STATUS.json')
    require(isinstance(status, dict) and set(status) == {'status', 'observations_complete', 'decisions_complete', 'new_solver_calls', 'reused_solver_calls', 'elapsed_seconds', 'deadline_epoch'}, 'RUN_STATUS: changed exact schema')
    same({key: value for key, value in status.items() if key != 'elapsed_seconds'}, {'status': 'COMPLETE', 'observations_complete': 36,
         'decisions_complete': 1944, 'new_solver_calls': 144, 'reused_solver_calls': 72, 'deadline_epoch': deadline}, 'RUN_STATUS: complete fixed matrix/common window')
    elapsed = finite_seconds(status['elapsed_seconds'], 'RUN_STATUS elapsed')
    require(elapsed < LIMIT and elapsed <= time.time() - budget['start_epoch'] and
            sum(finite_seconds(row['cold_seconds'], 'TIMING cold') for row in timing_records) <= elapsed, 'TIMING: impossible main/preparation duration')
    subsets = [{'observation_id': oid, 'method': method,
                'bounded_subset_independent': _appearance.interval_subset(checked[oid]['bounded'][method]['union']['intervals'], checked[oid]['independent'][method]['union']['intervals']),
                'strict_subset_bounded': _appearance.interval_subset(checked[oid]['strict_shared'][method]['union']['intervals'], checked[oid]['bounded'][method]['union']['intervals'])}
               for oid in observations for method in METHODS]
    method_subsets = [{'observation_id': oid, 'appearance_model': model,
                      'joint_subset_calibrated': _appearance.interval_subset(checked[oid][model]['joint']['union']['intervals'],
                                                                             checked[oid][model]['calibrated']['union']['intervals'])}
                     for oid in observations for model in ARMS]
    same(scientific_files(), sources, 'SOURCE_LOCK: all frozen source bytes unchanged through verification')
    same(input_files(), inputs, 'INPUTS: all sealed source/data bytes unchanged through verification')
    verify_budget(budget)
    return {'status': 'PASS', 'mechanical_verification': 'PASS', 'host': socket.gethostname(), 'output': str(output),
            'observations': 36, 'tasks': 324, 'method_decisions': 1944, 'new_decisions': 1296, 'reused_decisions': 648, 'keep_decisions': 324,
            'cold_solver_calls': counts['cold_solver_calls'], 'reused_solver_calls': counts['reused_solver_calls'], 'coverage_rows': 216, 'evaluation_rows': 2268,
            'new_branch_traces_replayed': 432, 'new_tree_nodes_verified': counts['tree_nodes'], 'exact_updates_recomputed': counts['exact_updates'],
            'contradiction_certificates_verified': counts['contradiction_certificates'], 'independent_geometry_calls': counts['geometry_calls'],
            'source_files_verified': len(sources), 'input_files_verified': len(inputs), 'source_parse_order_audit': audit, 'shared_budget_source_audit': executor_audit,
            'actual_truth_witnesses': witnesses, 'finite_outer_subset_checks': subsets, 'finite_method_subset_checks': method_subsets,
            'scientific_results': 'strict nonzero mismatch/errors preserved; no positive gain or directional subset acceptance condition',
            'shared_budget': {'start_epoch': budget['start_epoch'], 'deadline_epoch': deadline, 'elapsed_prepare_main_verification_seconds': time.time() - budget['start_epoch']}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=ROOT / 'outputs/variation-v1')
    output = check_target(parser.parse_args().output)
    with deadline_guard(output):
        result = verify(output)
        destination = output / 'verification/VERIFY.json'
        destination.parent.mkdir(exist_ok=True)
        with destination.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Independent A4 replay with own sparse rows and dimensional contractor.

No appearance solver, runner or private fact is imported. The frozen independent
camera geometry and exact action/union arithmetic remain read-only dependencies.
Private labels are parsed only after every saved public action is verified.
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
for _appearance_verifier_thread in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_appearance_verifier_thread] = '1'

ROOT = Path(__file__).resolve().parent
EXPECTED_ROOT = Path('/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/appearance-1')
PARENT, DATA = ROOT.parent, ROOT.parent / 'data'
RECOVERY = PARENT / 'recovery-1'
FULL_OUTPUT = RECOVERY / 'outputs/baseline-v1'
ABLATION = PARENT / 'ablation-1'
ABLATION_OUTPUT = ABLATION / 'outputs/ablation-v1'
_spec = importlib.util.spec_from_file_location('_appearance_frozen_independent_baseline', RECOVERY / 'verify_baseline.py')
_baseline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_baseline)
_ablation_spec = importlib.util.spec_from_file_location('_appearance_frozen_independent_ablation', ABLATION / 'verify_ablation.py')
_ablation = importlib.util.module_from_spec(_ablation_spec)
_ablation_spec.loader.exec_module(_ablation)
_prior, _arithmetic = _baseline._prior, _baseline._arithmetic
VerificationError = _baseline.VerificationError
require, same, exact, encoded, pair = _baseline.require, _baseline.same, _baseline.exact, _baseline.encoded, _baseline.pair
load, indexed, finite_seconds = _baseline.load, _baseline.indexed, _baseline.finite_seconds
DOMAIN, TOLERANCE = _baseline.DOMAIN, _baseline.TOLERANCE
RADIUS, EPSILON, BETA = _baseline.RADIUS, _baseline.EPSILON, _baseline.BETA
BRANCHES, ROUNDS, MAX_BOXES = _baseline.BRANCHES, _baseline.ROUNDS, _baseline.MAX_BOXES
METHODS, APPEARANCE_SCOPE = ('calibrated', 'joint'), 'per_view'
MATERIAL_VARIABLE_NAMES = tuple(f'd{view}{material}' for view in range(3) for material in range(3))
GROUP_SCOPE = 'necessary_per_view_material_shared_bias_halfspaces_with_historical_continuous_geometry_bounds'
LIMIT = 3600
BUDGET_SCOPE = 'A4 per-view appearance main plus independent verification; no reset'
independent_geometry = _baseline.independent_geometry


def pixel_bands(observation, clipped, epsilon=EPSILON, beta=BETA):
    """Own RGB intersections, preserving clipped one-sided row identities."""
    require(type(clipped) is bool, 'pixel bands: Boolean clipping identity required')
    epsilon, beta = exact(epsilon), exact(beta)
    require(epsilon >= 0 and beta >= 0, 'pixel bands: nonnegative error bounds required')
    image = tuple(map(exact, observation))
    require(len(image) == 432 and (not clipped or all(0 <= y <= 1 for y in image)),
            'pixel bands: complete432 exact legal scalars required')
    bands = []
    for pixel in range(144):
        high, low = [], []
        for y in image[3 * pixel:3 * pixel + 3]:
            if not clipped or y != 0:
                high.append(beta + epsilon if clipped and y == 1 else 1 + beta - y + epsilon)
            if not clipped or y != 1:
                low.append(1 + beta - epsilon if clipped and y == 0 else 1 + beta - y - epsilon)
        bands.append((min(high) if high else None, max(low) if low else None))
    return tuple(bands)


def necessary_rows(observation, geometry, clipped, method, calibration_intervals,
                   epsilon=EPSILON, beta=BETA, appearance_scope=APPEARANCE_SCOPE):
    """View-major material slots; each original pixel retains its row order."""
    require(method in METHODS and appearance_scope in ('per_view', 'global_shared') and len(geometry) == 144,
            'necessary rows: registered method/scope/layout required')
    intervals = [pair(value, 'necessary rows calibration') for value in calibration_intervals]
    require(len(intervals) == 3 and all(0 <= a <= b <= 2 * exact(beta) for a, b in intervals),
            'necessary rows: legal unchanged full-image calibration required')
    rows = []
    for pixel, (high, low) in enumerate(pixel_bands(observation, clipped, epsilon, beta)):
        values = tuple(map(exact, geometry[pixel]))
        require(len(values) == 6 and all(0 <= values[k] <= values[k + 1] <= 1 for k in (0, 2, 4)),
                'necessary rows: illegal independent geometric area bounds')
        view = pixel // 48
        tlow, thigh = intervals[view]
        for side, bound, coefficients in (('upper', high, values[::2]), ('lower', low, values[1::2])):
            if bound is None:
                continue
            if appearance_scope == 'per_view':
                coefficients = (Q(0),) * (3 * view) + coefficients + (Q(0),) * (6 - 3 * view)
            if method == 'joint':
                coefficients += tuple(Q(int(index == view)) for index in range(3))
            else:
                bound -= tlow if side == 'upper' else thigh
            rows.append((side, coefficients, bound))
    return tuple(rows)


def independent_contraction(rows, initial_bounds, rounds=ROUNDS):
    """Own exact contractor with the unchanged row/variable/sweep semantics."""
    require(type(rounds) is int and 0 <= rounds <= ROUNDS, 'propagation: invalid iteration budget')
    priors = [pair(value, 'propagation prior') for value in initial_bounds]
    require(len(priors) in (3, 6, 9, 12) and all(0 <= a <= b for a, b in priors),
            'propagation: invalid nonnegative dimensional priors')
    normalized = []
    for side, coefficients, bound in rows:
        coefficients = tuple(map(exact, coefficients))
        require(side in ('upper', 'lower') and len(coefficients) == len(priors) and all(a >= 0 for a in coefficients),
                'propagation: invalid necessary row')
        normalized.append((side, coefficients, exact(bound)))
    lower, upper = [a for a, _ in priors], [b for _, b in priors]
    initial = [[encoded(a), encoded(b)] for a, b in priors]
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


def verify_branch(outer, image, clipped, method, branch, geometry_bounds, calibration, where):
    fields = {'domain', 'epsilon', 'beta', 'clipped', 'tolerance', 'max_boxes', 'evaluated_boxes', 'trace',
              'retained_ids', 'hull', 'shared_tests', 'propagation_rounds', 'mode', 'rounds', 'budget',
              'channel_precheck', 'initial_bounds', 'scope', 'appearance_scope', 'material_variable_names'}
    require(isinstance(outer, dict) and set(outer) == fields, f'{where}: invalid per-view branch schema')
    initial = [[encoded(0), encoded(1)] for _ in range(9)]
    if method == 'joint':
        initial.extend(calibration['intervals'])
    same(outer['initial_bounds'], initial, f'{where}: nine materials and same full-image T_cal priors')
    same(outer['material_variable_names'], list(MATERIAL_VARIABLE_NAMES), f'{where}: exact view-major names')
    require(pair(outer['domain'], where) == DOMAIN and exact(outer['epsilon']) == EPSILON and exact(outer['beta']) == BETA and
            type(outer['clipped']) is bool and outer['clipped'] == clipped and exact(outer['tolerance']) == TOLERANCE and
            type(outer['max_boxes']) is int and outer['max_boxes'] == MAX_BOXES and outer['mode'] == method and
            outer['scope'] == GROUP_SCOPE and outer['appearance_scope'] == APPEARANCE_SCOPE and
            outer['channel_precheck'] == 'none', f'{where}: changed per-view scientific contract')
    require(all(type(outer[key]) is int and outer[key] == value for key, value in
                (('rounds', ROUNDS), ('propagation_rounds', ROUNDS), ('budget', MAX_BOXES))), f'{where}: changed branch budgets')
    nodes = indexed(outer['trace'], lambda node: node['id'], where)
    require(0 < len(nodes) <= MAX_BOXES and all(type(node['id']) is int for node in outer['trace']) and
            list(nodes) == list(range(len(nodes))) and nodes[0]['parent'] is None, f'{where}: invalid complete BFS identities')
    boxes = {ident: (exact(node['lo']), exact(node['hi'])) for ident, node in nodes.items()}
    require(boxes[0] == DOMAIN, f'{where}: root differs from registered full domain')
    leaves, retained, excluded, referenced = [], [], [], set()
    evaluated = updates = 0
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
                f'{where}: invalid node schema/status/depth box {ident}')
        if ident:
            require(type(node['parent']) is int and node['parent'] in nodes and node['parent'] < ident,
                    f'{where}: invalid parent {ident}')
        if calibration['excluded']:
            require(ident == 0 and len(nodes) == 1 and status == 'excluded_calibration', f'{where}: false calibration conflict')
            same(node['certificate'], calibration['certificate'], f'{where}: calibration exclusion proof')
        else:
            require(status not in ('excluded_calibration', 'unresolved_budget') and evaluated < MAX_BOXES,
                    f'{where}: unsupported calibration/budget exclusion')
            geometry = geometry_bounds(branch, RADIUS, left, right)
            rows = necessary_rows(image, geometry, clipped, method, calibration['intervals'])
            contraction = independent_contraction(rows, initial)
            evaluated += 1
            require(type(node['rounds']) is int and node['rounds'] == contraction['rounds'] and
                    type(node['updates']) is int and node['updates'] == contraction['updates'],
                    f'{where}: false exact propagation accounting {ident}')
            updates += contraction['updates']
            if contraction['excluded']:
                require(status == 'excluded_shared' and node['group_id'] == 'global', f'{where}: omitted/false contradiction {ident}')
                same(node['certificate'], contraction['certificate'], f'{where}: every update and first crossing certificate {ident}')
            else:
                require(status in ('split', 'retained_tolerance'), f'{where}: unsupported exclusion {ident}')
                same(node['darkness_bounds'], contraction['bounds'][:9], f'{where}: all independently contracted nine materials {ident}')
                same(node['t_bounds'], contraction['bounds'][9:] if method == 'joint' else calibration['intervals'],
                     f'{where}: independently contracted/frozen view bias {ident}')
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


def verify_appearance_arm(arm, public, method, where):
    fields = {'branches', 'outer', 'branch_seconds', 'outer_seconds', 'preparation_seconds',
              'background_mask', 'calibration_intervals', 'calibration', 'appearance_scope', 'material_variable_names'}
    require(method in METHODS and isinstance(arm, dict) and set(arm) == fields, f'{where}: invalid per-view arm schema')
    require(arm['appearance_scope'] == APPEARANCE_SCOPE, f'{where}: changed appearance sharing scope')
    same(arm['material_variable_names'], list(MATERIAL_VARIABLE_NAMES), f'{where}: exact material names/order')
    require(isinstance(public, dict) and type(public.get('clipped')) is bool and public['clipped'] is True and
            exact(public['radius']) == RADIUS and exact(public['baseline']) == 80,
            f'{where}: changed selected B80 camera/noise contract')
    image = tuple(map(exact, public['observation']))
    geometry_bounds = lambda branch, radius, low, high: independent_geometry(branch, radius, low, high, Q(80))
    require(list(arm['branches']) == list(arm['branch_seconds']) == list(BRANCHES), f'{where}: changed three-branch order')
    calibration = _prior.verify_arm_public(arm, image, True, geometry_bounds, where)
    total = finite_seconds(arm['outer_seconds'], where + ': solver seconds')
    measured = finite_seconds(arm['preparation_seconds'], where + ': preparation seconds')
    measured += sum(finite_seconds(value, where + ': branch seconds') for value in arm['branch_seconds'].values())
    require(total >= measured, f'{where}: impossible recorded solve cost')
    branches = {branch: verify_branch(arm['branches'][branch], image, True, method, branch, geometry_bounds,
                                     calibration, f'{where}/{branch}') for branch in BRANCHES}
    union = _prior.verify_union(arm['outer'], branches, method, True, where)
    counts = {'tree_nodes': sum(len(arm['branches'][branch]['trace']) for branch in BRANCHES),
              'exact_updates': sum(value['proof_updates'] for value in branches.values()),
              'contradiction_certificates': sum(len(value['shared_excluded']) for value in branches.values()),
              'geometry_calls': sum(value['evaluated'] for value in branches.values())}
    return {'union': union, 'branches': branches, 'counts': counts, 'calibration': calibration}


def verify_action(decision, task, public, checked, where='action'):
    require(isinstance(decision, dict) and decision.get('method') in METHODS and type(decision.get('reused')) is bool,
            f'{where}: invalid action method/origin')
    action = _arithmetic.hybrid_action(checked['hull'], task['incumbent'])
    expected = {'task_id': task['task_id'], 'unit_id': public['unit_id'], 'method': decision['method'],
                'appearance_scope': APPEARANCE_SCOPE, 'reused': False,
                **{key: None if value is None else encoded(value) if key in ('output', 'gain_lower') else value
                   for key, value in action.items()}}
    same(decision, expected, f'{where}: independent endpoint/certificate/action identity')
    return exact(decision['output'], where)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scientific_files():
    files = dict(_ablation.scientific_files())
    for path in sorted({ROOT / 'AGENTS.md', ROOT / 'PROTOCOL.md', ROOT / 'REGISTRATION.json', *ROOT.glob('*.py')}):
        files[str(path.resolve())] = sha(path)
    return dict(sorted(files.items()))


def input_files():
    files = dict(_ablation.input_files())
    for relative in ('SOURCE_LOCK.json', 'RUN_STATUS.json', 'verification/VERIFY.json'):
        path = ABLATION_OUTPUT / relative
        files[str(path.resolve())] = sha(path)
    return dict(sorted(files.items()))


def verify_source_lock(lock):
    require(isinstance(lock, dict) and set(lock) == {'source_root', 'files'} and lock['source_root'] == str(ROOT),
            'SOURCE_LOCK: exact appearance source identity/schema required')
    files = scientific_files()
    same(lock['files'], files, 'SOURCE_LOCK: complete independently enumerated frozen/new source bytes')
    same(load(FULL_OUTPUT / 'SOURCE_LOCK.json')['files'], _baseline.scientific_files(),
         'SOURCE_LOCK: verified full recovery graph unchanged')
    same(load(ABLATION_OUTPUT / 'SOURCE_LOCK.json')['files'], _ablation.scientific_files(),
         'SOURCE_LOCK: verified complete frozen ablation dependency graph unchanged')
    return files


def verify_inputs(lock):
    require(isinstance(lock, dict) and set(lock) == {'files'}, 'INPUTS: invalid sealed input schema')
    files = input_files()
    same(lock['files'], files, 'INPUTS: complete sealed original data/full result bytes')
    previous, status = load(FULL_OUTPUT / 'verification/VERIFY.json'), load(FULL_OUTPUT / 'RUN_STATUS.json')
    require(previous.get('status') == 'PASS' and previous.get('host') == 'liekkas' and
            previous.get('output') == str(FULL_OUTPUT) and status.get('status') == 'COMPLETE',
            'INPUTS: exact complete independently verified full baseline required')
    same(load(FULL_OUTPUT / 'INPUTS.json')['files'], _baseline.generated_inputs(), 'INPUTS: same frozen paired data bytes')
    ablation_receipt = load(ABLATION_OUTPUT / 'verification/VERIFY.json')
    require(ablation_receipt.get('status') == 'PASS' and ablation_receipt.get('host') == 'liekkas' and
            ablation_receipt.get('output') == str(ABLATION_OUTPUT) and
            load(ABLATION_OUTPUT / 'RUN_STATUS.json').get('status') == 'COMPLETE',
            'INPUTS: exact complete independently verified frozen ablation required')
    same(load(ABLATION_OUTPUT / 'INPUTS.json')['files'], _ablation.input_files(), 'INPUTS: frozen ablation inputs unchanged')
    return files


def verify_budget(budget):
    require(isinstance(budget, dict) and set(budget) == {'start_epoch', 'deadline_epoch', 'limit_seconds', 'scope'},
            'BUDGET: invalid shared absolute window schema')
    start, end = (finite_seconds(budget[key], 'BUDGET ' + key) for key in ('start_epoch', 'deadline_epoch'))
    require(type(budget['limit_seconds']) is int and budget['limit_seconds'] == LIMIT and end == start + LIMIT and
            budget['scope'] == BUDGET_SCOPE, 'BUDGET: changed registered3600-second absolute window')
    if time.time() >= end:
        raise TimeoutError('A4 main plus independent verification deadline exhausted')
    return end


@contextmanager
def deadline_guard(output):
    deadline = verify_budget(load(Path(output) / 'BUDGET.json'))
    require(not signal.getitimer(signal.ITIMER_REAL)[0], 'BUDGET: an existing deadline cannot be reset')
    def reached(_signal, _frame):
        raise TimeoutError('A4 main plus independent verification deadline reached')
    previous = signal.signal(signal.SIGALRM, reached)
    signal.setitimer(signal.ITIMER_REAL, max(0.000001, deadline - time.time()))
    try:
        yield deadline
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def check_target(output=None):
    require(socket.gethostname() == 'liekkas' and ROOT == EXPECTED_ROOT and ROOT.resolve() == EXPECTED_ROOT,
            'target: exact appearance child on liekkas required')
    if output is not None:
        output = Path(output).resolve()
        require(output.is_dir() and output != ROOT and output.is_relative_to(ROOT),
                'target: existing output strictly inside exact appearance child required')
    return output


def verify_public_population(output):
    public, records = load(DATA / 'observations.json'), load(DATA / 'tasks.json')
    require(isinstance(public, list) and len(public) == 36 and isinstance(records, list) and len(records) == 324,
            'population: frozen36 public images/324 tasks required')
    images = indexed(public, lambda row: row['observation_id'], 'frozen public images')
    tasks = indexed(records, lambda row: row['task_id'], 'frozen public tasks')
    same(list(images), [f'p{index:03d}b{b}' for index in range(1, 19) for b in (70, 80)], 'frozen public IDs/order')
    for oid, row in images.items():
        b = oid.rsplit('b', 1)[1]
        require(set(row) == {'observation_id', 'pair_id', 'unit_id', 'baseline', 'radius', 'clipped', 'observation'},
                f'{oid}: leaked field/changed public schema')
        same({key: row[key] for key in row if key != 'observation'},
             {'observation_id': oid, 'pair_id': oid.split('b')[0], 'unit_id': f'baseline_{b}',
              'baseline': b, 'radius': str(RADIUS), 'clipped': True}, f'{oid}: exact public metadata')
        image = tuple(map(exact, row['observation']))
        require(len(image) == 432 and all(0 <= y <= 1 for y in image), f'{oid}: illegal exact clipped layout')
    same(list(tasks), [f'{oid}s{state}' for oid in images for state in range(9)], 'frozen task IDs/order')
    for tid, task in tasks.items():
        require(set(task) == {'task_id', 'base_task_id', 'observation_id', 'incumbent'} and task['observation_id'] in images,
                f'{tid}: changed task schema/identity')
        same(task['base_task_id'], f'{images[task["observation_id"]]["pair_id"]}s{tid.rsplit("s", 1)[1]}', f'{tid}: original state identity')
        exact(task['incumbent'])
    selected = {oid: row for oid, row in images.items() if row['baseline'] == '80'}
    selected_tasks = {tid: row for tid, row in tasks.items() if row['observation_id'] in selected}
    require(len(selected) == 18 and len(selected_tasks) == 162, 'population: exact B80 selection required')
    same(load(output / 'observations.json'), list(selected.values()), 'all18 selected public observations saved/read back')
    same(load(output / 'tasks.json'), list(selected_tasks.values()), 'all162 selected tasks saved/read back')
    return selected, selected_tasks


def verify_action_population(records, observations, tasks):
    require(isinstance(records, list) and len(records) == 324, 'actions: complete324 persisted actions required')
    actions = indexed(records, lambda row: (row['task_id'], row['method']), 'saved actions')
    expected = [(f'{oid}s{state}', method) for oid in observations for method in METHODS for state in range(9)]
    require(list(actions) == expected and {tid for tid, _method in expected} == set(tasks),
            'actions: missing/reordered/duplicate task/method')
    require(all(type(row.get('reused')) is bool and row['reused'] is False and row.get('appearance_scope') == APPEARANCE_SCOPE
                for row in records), 'actions: changed new per-view origin/scope')
    return actions


def make_result(task, private, decision, checked, condition):
    truth, incumbent, output = exact(private['truth']), exact(task['incumbent']), exact(decision['output'])
    hull = checked['hull']
    initial, final = abs(incumbent - truth), abs(output - truth)
    return {'task_id': task['task_id'], 'base_task_id': task['base_task_id'], 'observation_id': task['observation_id'],
            'pair_id': private['pair_id'], 'unit_id': private['unit_id'], 'baseline': private['baseline'],
            'method': decision['method'], 'condition': condition, 'case': private['case'], 'depth_mm': str(truth),
            'initial_ae_mm': str(initial), 'ae_mm': str(final), 'se_mm2': str(final * final),
            'actual_gain_mm2': str(initial * initial - final * final), 'output': str(output), 'status': decision['status'],
            'covered': hull is not None and hull[0] <= truth <= hull[1],
            'hull_width_mm': None if hull is None else str(hull[1] - hull[0])}


def verify_private_population(observations, tasks):
    """Parse the frozen36 only here; evaluate only selected18 B80 witnesses."""
    records = load(DATA / 'private/truth.json')
    frozen = indexed(records, lambda row: row['observation_id'], 'sealed private labels')
    same(list(frozen), [f'p{index:03d}b{b}' for index in range(1, 19) for b in (70, 80)], 'private: frozen36 identities/order')
    truths = {oid: frozen[oid] for oid in observations}
    witnesses = []
    for oid, private in truths.items():
        public = observations[oid]
        require(private['pair_id'] == public['pair_id'] and private['unit_id'] == public['unit_id'] and
                private['baseline'] == '80' and private['rear_branch'] in BRANCHES and private['seed'] == 61005,
                f'{oid}: changed sealed selected witness lineage')
        truth = exact(private['truth'])
        require(truth in _baseline._historical.TRUTHS, f'{oid}: changed registered depth')
        predicted = _baseline.independent_rgb(private['rear_branch'], truth, private['actual_offsets'], private['colors'], Q(80))
        noise, bias = tuple(map(exact, private['image_noise'])), tuple(map(exact, private['actual_view_bias']))
        require(len(noise) == 432 and all(abs(value) <= EPSILON for value in noise) and bias == (-BETA, Q(0), BETA),
                f'{oid}: illegal sealed image noise/bias')
        rebuilt = tuple(min(Q(1), max(Q(0), value + noise[index] + bias[index // 144])) for index, value in enumerate(predicted))
        same(list(map(encoded, rebuilt)), public['observation'], f'{oid}: exact independent point/noise/bias/clip witness')
        materials = tuple(1 - exact(color) for color in private['colors'])
        repeated = materials * 3
        require(len(repeated) == 9, f'{oid}: repeated true-shared parameter embedding required')
        areas = _baseline.independent_point_areas(private['rear_branch'], truth, private['actual_offsets'], Q(80))
        for pixel, (high, low) in enumerate(pixel_bands(public['observation'], True)):
            value = sum((area * darkness for area, darkness in zip(areas[pixel], repeated[3 * (pixel // 48):3 * (pixel // 48 + 1)])), Q(0)) + BETA - bias[pixel // 48]
            require((high is None or value <= high) and (low is None or value >= low), f'{oid}: repeated nine-material witness violates pixel bands')
        for state, offset in enumerate(_baseline._historical.OFFSETS):
            same(tasks[f'{oid}s{state}']['incumbent'], encoded(truth + offset), f'{oid}: original nine initial offsets')
        witnesses.append({'observation_id': oid, 'clip_aware_true_witness_valid': True, 'repeated_nine_material_witness_valid': True})
    return truths, witnesses


def make_coverage(public, private, method, union, branches):
    truth, branch = exact(private['truth']), private['rear_branch']
    true_nodes = {name: [node for node in value['retained'] if exact(node['lo']) <= truth <= exact(node['hi'])]
                  for name, value in branches.items()}
    true_leaves = true_nodes[branch]
    darkness = tuple(1 - exact(color) for color in private['colors']) * 3
    bias = tuple(BETA - exact(value) for value in private['actual_view_bias'])
    def latent(key, values):
        return bool(true_leaves) and all(len(node[key]) == len(values) and
            all(exact(bounds[0]) <= value <= exact(bounds[1]) for bounds, value in zip(node[key], values)) for node in true_leaves)
    hull = union['hull']
    return {'appearance_scope': APPEARANCE_SCOPE, 'observation_id': public['observation_id'], 'unit_id': public['unit_id'],
            'method': method, 'case': private['case'], 'seed': private['seed'], 'covered': any(true_nodes.values()),
            'incompatible': hull is None, 'hull_width_mm': None if hull is None else str(hull[1] - hull[0]),
            'retained_branches': union['retained_branches'], 'branch_coverage': {name: bool(nodes) for name, nodes in true_nodes.items()},
            'generating_branch': branch, 'generating_branch_covered': bool(true_leaves),
            'true_depth_leaf_ids': {name: [node['id'] for node in nodes] for name, nodes in true_nodes.items()},
            'latent_material_covered': latent('darkness_bounds', darkness),
            'latent_bias_covered': latent('t_bounds', bias) if method == 'joint' else None}


def validate_required_coverage(records):
    """Registered witness validity is required independently of gain/nesting."""
    require(isinstance(records, list) and len(records) == 36, 'coverage: complete36 witness records required')
    require(len({(row['observation_id'], row['method']) for row in records}) == 36,
            'coverage: distinct18 image/two-method identities required')
    for row in records:
        where = f'{row["observation_id"]}/{row["method"]}'
        require(row['method'] in METHODS and row['appearance_scope'] == APPEARANCE_SCOPE,
                f'{where}: changed coverage method/scope')
        require(all(type(row[key]) is bool and row[key] is True for key in
                    ('covered', 'generating_branch_covered', 'latent_material_covered')) and
                type(row['incompatible']) is bool and row['incompatible'] is False,
                f'{where}: registered depth/generating-branch/repeated-nine-material witness coverage failed')
        require(row['latent_bias_covered'] is True if row['method'] == 'joint' else row['latent_bias_covered'] is None,
                f'{where}: registered joint three-view bias witness coverage failed')


def verify_safe_result(task, private, decision, checked, where):
    truth, incumbent, output = exact(private['truth']), exact(task['incumbent']), exact(decision['output'])
    hull = checked['hull']
    if hull is not None and hull[0] <= truth <= hull[1]:
        gain = exact(decision['gain_lower'])
        require((incumbent - truth) ** 2 - (output - truth) ** 2 >= gain >= 0, f'{where}: false covered safe-action certificate')


def aggregate_results(rows):
    grouped = {}
    for row in rows:
        prefix = (row['unit_id'], row['method'], row['condition'])
        shifted = exact(row['initial_ae_mm']) != 0
        cells = [('all_shifted' if shifted else 'correct', '')]
        if shifted:
            informative = row['case'] != 'equal_full_negative'
            cells.extend([('informative_shifted' if informative else 'negative_shifted', ''), ('case_shifted', row['case'])])
            if informative:
                cells.append(('depth_informative', row['depth_mm']))
        for cell in cells:
            grouped.setdefault(prefix + cell, []).append(row)
    result = []
    for (unit, method, condition, group, value), selected in sorted(grouped.items()):
        initial, final = [exact(row['initial_ae_mm']) for row in selected], [exact(row['ae_mm']) for row in selected]
        count = len(selected)
        result.append({'unit_id': unit, 'method': method, 'condition': condition, 'group': group, 'value': value, 'n': count,
            'mae_mm': str(sum(final, Q(0)) / count), 'mse_mm2': str(sum((error * error for error in final), Q(0)) / count),
            'keep_mse_mm2': str(sum((error * error for error in initial), Q(0)) / count),
            'improved': sum(after < before for before, after in zip(initial, final)),
            'same': sum(after == before for before, after in zip(initial, final)),
            'harmed': sum(after > before for before, after in zip(initial, final))})
    return result


def verify_summary_denominators(summary):
    for name, cells in summary.items():
        configurations = [(method, 'per_view') for method in METHODS] if name == 'new' else [('keep', 'keep'), *[(method, 'global_shared') for method in METHODS]]
        for method, condition in configurations:
            selected = {(row['group'], row['value']): row for row in cells
                        if (row['unit_id'], row['method'], row['condition']) == ('baseline_80', method, condition)}
            for group, count in (('all_shifted', 144), ('correct', 18), ('informative_shifted', 120), ('negative_shifted', 24)):
                require(selected.get((group, ''), {}).get('n') == count, f'{name}/{method}: changed{group} denominator')
            depth = [row for (group, _value), row in selected.items() if group == 'depth_informative']
            cases = [row for (group, _value), row in selected.items() if group == 'case_shifted']
            require(len(depth) == 3 and all(row['n'] == 40 for row in depth) and len(cases) == 6 and all(row['n'] == 24 for row in cases),
                    f'{name}/{method}: changed all-depth/six-template populations')


def interval_subset(inner, outer):
    return all(any(c <= a <= b <= d for c, d in outer) for a, b in inner)


def subset_checks(observations, checked, full):
    """Report finite realized relations without accepting a scientific direction."""
    rows = []
    for oid in observations:
        for method in METHODS:
            shared = [pair(value, 'frozen shared union') for value in full[oid][method]['outer']['retained_intervals']]
            view = checked[oid][method]['union']['intervals']
            rows.append({'observation_id': oid, 'relation': 'global_shared_subset_per_view', 'method': method,
                         'subset': interval_subset(shared, view)})
        rows.append({'observation_id': oid, 'relation': 'per_view_joint_subset_calibrated',
                     'subset': interval_subset(checked[oid]['joint']['union']['intervals'], checked[oid]['calibrated']['union']['intervals'])})
    return rows


def audit_runner_source(source=None):
    """Audit the actual runner's saved-readback-complete gate and public calls."""
    path = ROOT / 'run_appearance.py'
    code = path.read_text(encoding='utf-8') if source is None else source
    tree = ast.parse(code)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    require({'run', 'evaluate', 'check_actions', 'population'} <= set(functions), 'source audit: missing public/persisted entry points')
    dump = lambda node: ast.dump(node, include_attributes=False)
    run = functions['run']
    evaluations = [node for node in ast.walk(run) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'evaluate']
    require(len(evaluations) == 1, 'source audit: one post-action private evaluation required')
    evaluate_line = evaluations[0].lineno
    callers = [(function.name, node.lineno) for function in functions.values() for node in ast.walk(function)
               if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'evaluate']
    require(callers == [('run', evaluate_line)], 'source audit: additional private evaluation route')
    statements = ("save(output/'outers.json',outers)", "save(output/'decisions.json',decisions)",
                  "decisions = load(output/'decisions.json')", "outers = load(output/'outers.json')",
                  "check_actions(decisions,tasks)",
                  "if calls != 36 or len(outers) != 18:\n    raise RuntimeError('incomplete A4 solve matrix')")
    tries = [node for node in run.body if isinstance(node, ast.Try)]
    require(len(tries) == 1, 'source audit: changed action/evaluation control flow')
    positions = []
    for statement in statements:
        target = dump(ast.parse(statement).body[0])
        matches = [node for node in tries[0].body if dump(node) == target]
        require(len(matches) == 1 and matches[0].end_lineno < evaluate_line,
                'source audit: complete saved/readback action matrix fence required before private labels')
        positions.append(matches[0].lineno)
    require(positions == sorted(positions), 'source audit: reordered save/readback/completeness gates')
    predicate = "len(keys)!=len(expected) or len(set(keys))!=len(expected) or set(keys)!=expected or any(record['reused'] is not False or record['appearance_scope']!='per_view' for record in records)"
    guards = [node for node in functions['check_actions'].body if isinstance(node, ast.If) and
              dump(node.test) == dump(ast.parse(predicate, mode='eval').body) and any(isinstance(child, ast.Raise) for child in node.body)]
    require(len(guards) == 1, 'source audit: complete unique324 new per-view action guard changed')
    private_paths = [(function.name, node) for function in functions.values() for node in ast.walk(function)
                     if isinstance(node, ast.Constant) and isinstance(node.value, str) and 'truth.json' in node.value]
    require(sorted(name for name, _node in private_paths) == ['evaluate', 'inputs'], 'source audit: private path outside hashes/post-action evaluation')
    all_private_paths = [node for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str) and 'truth.json' in node.value]
    require(len(all_private_paths) == len(private_paths), 'source audit: private path at module/class scope')
    parses = [node for node in ast.walk(functions['evaluate']) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'load' and
              node.args and dump(node.args[0]) == dump(ast.parse("DATA / 'private/truth.json'", mode='eval').body)]
    require(len(parses) == 1, 'source audit: exact sealed private parse path required')
    calls = [node for node in ast.walk(run) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'solve_public']
    require(len(calls) == 1 and [dump(arg) for arg in calls[0].args] == [dump(ast.parse(expr, mode='eval').body)
                for expr in ("public['observation']", "public['radius']", "public['baseline']", 'method')] and
            [(keyword.arg, dump(keyword.value)) for keyword in calls[0].keywords] ==
            [('appearance_scope', dump(ast.parse("'per_view'", mode='eval').body)), ('clipped', dump(ast.parse("public['clipped']", mode='eval').body))],
            'source audit: inference receives nonpublic or changed arguments')
    return {'status': 'PASS', 'runner': str(path), 'private_parse_line': parses[0].lineno,
            'post_action_evaluation_call_line': evaluate_line, 'persisted_matrix_gate_lines': positions,
            'solver_arguments': ['observation', 'radius', 'baseline', 'method', 'appearance_scope=per_view', 'clipped']}


def verify(output, deadline=None):
    output = check_target(output)
    budget = load(output / 'BUDGET.json')
    deadline = verify_budget(budget)
    sources, inputs = verify_source_lock(load(output / 'SOURCE_LOCK.json')), verify_inputs(load(output / 'INPUTS.json'))
    same(load(output / 'REGISTRATION.json'), load(ROOT / 'REGISTRATION.json'), 'REGISTRATION: frozen child copy')
    audit = audit_runner_source()
    observations, tasks = verify_public_population(output)
    records = load(output / 'decisions.json')
    decisions = verify_action_population(records, observations, tasks)
    outers, full = load(output / 'outers.json'), load(FULL_OUTPUT / 'outers.json')
    require(isinstance(outers, dict) and list(outers) == list(observations), 'outers: complete18 ordered B80 identities required')
    full_records = [row for row in load(FULL_OUTPUT / 'decisions.json') if row['task_id'] in tasks]
    require(len(full_records) == 486, 'reference: complete486 frozen KEEP/calibrated/joint actions required')
    full_actions = indexed(full_records, lambda row: (row['task_id'], row['method']), 'frozen full actions')
    same(list(full_actions), [(f'{oid}s{state}', method) for oid in observations for method in ('keep', *METHODS) for state in range(9)],
         'reference: exact frozen ordered action selection')
    timing_records = load(output / 'TIMING.json')
    require(isinstance(timing_records, list) and len(timing_records) == 36, 'TIMING: complete36 cold calls required')
    timing = indexed(timing_records, lambda row: (row['observation_id'], row['method']), 'cold timing')
    same(list(timing), [(oid, method) for oid in observations for method in METHODS], 'TIMING: exact36 call order')
    checked, counts = {}, Counter()
    for index, (oid, public) in enumerate(observations.items()):
        require(isinstance(outers[oid], dict) and list(outers[oid]) == list(METHODS), f'{oid}: exact calibrated/joint order required')
        checked[oid] = {}
        for method in METHODS:
            arm = outers[oid][method]
            result = verify_appearance_arm(arm, public, method, f'{oid}/{method}')
            same(result['calibration'], full[oid][method]['calibration'], f'{oid}/{method}: unchanged full-image frozen T_cal')
            checked[oid][method] = result
            counts.update(result['counts'])
            counts['cold_solver_calls'] += 1
            measured = timing[oid, method]
            require(set(measured) == {'observation_id', 'appearance_scope', 'method', 'cold_seconds', 'evaluated_boxes'} and
                    measured['appearance_scope'] == APPEARANCE_SCOPE and type(measured['evaluated_boxes']) is int and
                    measured['evaluated_boxes'] == result['union']['evaluated'] and
                    finite_seconds(measured['cold_seconds'], 'TIMING seconds') >= finite_seconds(arm['outer_seconds'], 'arm seconds'),
                    f'{oid}/{method}: false cold timing/node count')
        if index % 3 == 2:
            print(json.dumps({'stage': 'independent_appearance_verification', 'observations_complete': index + 1,
                              'remaining_seconds': round(deadline - time.time(), 1)}), flush=True)
    checkpoint = output / 'checkpoint'
    require(checkpoint.is_dir() and sorted(path.name for path in checkpoint.iterdir()) == sorted(f'{oid}.json' for oid in observations),
            'checkpoint: complete18 exact image checkpoints required')
    for oid in observations:
        same(load(checkpoint / f'{oid}.json'), {'observation_id': oid, 'arms': outers[oid]}, f'{oid}: persisted checkpoint readback')
    for record in records:
        task = tasks[record['task_id']]
        oid, method = task['observation_id'], record['method']
        verify_action(record, task, observations[oid], checked[oid][method]['union'], f'{task["task_id"]}/{method}')
    # All36 public trees and all324 persisted actions have been independently
    # checked. Only now may the sealed private facts be parsed for evaluation.
    truths, witnesses = verify_private_population(observations, tasks)
    results, reference, coverage = [], [], []
    for record in records:
        task = tasks[record['task_id']]
        oid, method = task['observation_id'], record['method']
        union = checked[oid][method]['union']
        verify_safe_result(task, truths[oid], record, union, f'{task["task_id"]}/{method}')
        results.append(make_result(task, truths[oid], record, union, APPEARANCE_SCOPE))
    for record in full_records:
        task = tasks[record['task_id']]
        oid, method = task['observation_id'], record['method']
        hull = full[oid][method]['outer']['hull']
        union = {'hull': None if hull is None else pair(hull, 'sealed reference hull')}
        reference.append(make_result(task, truths[oid], record, union, 'keep' if method == 'keep' else 'global_shared'))
    for oid, public in observations.items():
        for method in METHODS:
            arm = checked[oid][method]
            coverage.append(make_coverage(public, truths[oid], method, arm['union'], arm['branches']))
    validate_required_coverage(coverage)
    same(load(output / 'results.json'), results, 'evaluation: all324 independent exact per-view endpoint errors')
    same(load(output / 'reference/results.json'), reference, 'evaluation: all486 frozen reference exact errors')
    same(load(output / 'coverage.json'), coverage, 'evaluation: all36 branch and repeated-nine-material coverage records')
    for relative, rows in (('results.csv', results), ('reference/results.csv', reference)):
        _baseline.verify_csv(output / relative, rows)
    summary = {'new': aggregate_results(results), 'reference': aggregate_results(reference)}
    verify_summary_denominators(summary)
    same(load(output / 'summary.json'), summary, 'evaluation: all exact depth/template summary cells')
    status = load(output / 'RUN_STATUS.json')
    status_fields = {'status', 'new_solver_calls', 'decisions_complete', 'observations_complete', 'elapsed_seconds', 'deadline_epoch'}
    require(isinstance(status, dict) and status_fields <= set(status) <= status_fields | {'finished_utc', 'source_files_locked', 'inputs_locked'} and status['status'] == 'COMPLETE' and
            all(type(status[key]) is int and status[key] == value for key, value in
                (('new_solver_calls', 36), ('decisions_complete', 324), ('observations_complete', 18))) and
            status['deadline_epoch'] == deadline, 'RUN_STATUS: false complete run accounting/schema')
    for key, value in (('source_files_locked', len(sources)), ('inputs_locked', len(inputs))):
        require(key not in status or (type(status[key]) is int and status[key] == value), f'RUN_STATUS: false{key}')
    elapsed = finite_seconds(status['elapsed_seconds'], 'RUN_STATUS seconds')
    require(elapsed < LIMIT and sum(finite_seconds(row['cold_seconds'], 'cold seconds') for row in timing_records) <= elapsed,
            'TIMING: recorded36 cold calls exceed elapsed main run')
    same(scientific_files(), sources, 'SOURCE_LOCK: frozen bytes unchanged throughout verification')
    same(input_files(), inputs, 'INPUTS: sealed bytes unchanged throughout verification')
    verify_budget(budget)
    return {'status': 'PASS', 'mechanical_verification': 'PASS', 'host': socket.gethostname(), 'output': str(output),
            'observations': len(observations), 'tasks': len(tasks), 'decisions': len(decisions),
            'cold_solver_calls': counts['cold_solver_calls'], 'coverage_rows': len(coverage), 'reference_rows': len(reference),
            'branch_outer_traces': 108, 'tree_nodes_verified': counts['tree_nodes'], 'exact_updates_recomputed': counts['exact_updates'],
            'contradiction_certificates_verified': counts['contradiction_certificates'], 'independent_geometry_calls': counts['geometry_calls'],
            'source_files_verified': len(sources), 'input_files_verified': len(inputs), 'source_parse_order_audit': audit,
            'actual_truth_witnesses': witnesses, 'finite_outer_subset_checks': subset_checks(observations, checked, full),
            'scientific_results': 'exact errors and coverage preserved; no positive-gain or directional subset acceptance condition',
            'shared_budget': {'start_epoch': budget['start_epoch'], 'deadline_epoch': deadline,
                              'elapsed_main_plus_verification_seconds': time.time() - budget['start_epoch']}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=ROOT / 'outputs/appearance-v1')
    output = check_target(parser.parse_args().output)
    with deadline_guard(output) as deadline:
        result = verify(output, deadline)
        destination = output / 'verification/VERIFY.json'
        destination.parent.mkdir(exist_ok=True)
        with destination.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

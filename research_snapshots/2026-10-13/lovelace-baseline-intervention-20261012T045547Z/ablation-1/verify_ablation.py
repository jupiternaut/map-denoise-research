#!/usr/bin/env python3
"""Independent A7/A1/A5 replay checks; imports no new solver or runner.

The frozen independent geometry and arithmetic are reused read-only. Pixel
bands, row identity, inference masks and each masked tree are reconstructed
here. Private labels are parsed only after the complete public action matrix
has been loaded, checked and independently verified.
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
for _ablation_verifier_thread in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_ablation_verifier_thread] = '1'

ROOT = Path(__file__).resolve().parent
EXPECTED_ROOT = Path('/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/ablation-1')
PARENT, DATA = ROOT.parent, ROOT.parent / 'data'
RECOVERY = PARENT / 'recovery-1'
FULL_OUTPUT = RECOVERY / 'outputs/baseline-v1'
_spec = importlib.util.spec_from_file_location('_ablation_frozen_independent_baseline', RECOVERY / 'verify_baseline.py')
_baseline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_baseline)
_prior, _arithmetic = _baseline._prior, _baseline._arithmetic
VerificationError = _baseline.VerificationError
require, same, exact, encoded, pair = _baseline.require, _baseline.same, _baseline.exact, _baseline.encoded, _baseline.pair
load, indexed, finite_seconds = _baseline.load, _baseline.indexed, _baseline.finite_seconds
DOMAIN, TOLERANCE = _baseline.DOMAIN, _baseline.TOLERANCE
RADIUS, EPSILON, BETA = _baseline.RADIUS, _baseline.EPSILON, _baseline.BETA
BASELINES, UNITS, BRANCHES = _baseline.BASELINES, _baseline.UNITS, _baseline.BRANCHES
ROUNDS, MAX_BOXES = _baseline.ROUNDS, _baseline.MAX_BOXES
METHODS, MASKS = ('calibrated', 'joint'), ('drop_F', 'matched_drop')
SELECTORS = ('hybrid', 'midpoint', 'projection')
LIMIT = 3600
BUDGET_SCOPE = 'A7 and A1+A5 main plus independent verification; no reset'
independent_geometry = _baseline.independent_geometry
independent_contraction = _prior.independent_contraction


def mask_pixels(name):
    """Original view×48+row×8+column identities, fixed before inference."""
    require(name in (*MASKS, 'none'), 'inference mask: unknown public mask')
    if name == 'none':
        return []
    columns = (7, 0) if name == 'drop_F' else (6, 1)
    return sorted(view * 48 + row * 8 + column
                  for view, column in zip((0, 2), columns) for row in range(6))


def pixel_bands(observation, clipped, epsilon=EPSILON, beta=BETA):
    """Own RGB intersection; clipped zero/one omit the absent row side."""
    require(type(clipped) is bool, 'pixel bands: Boolean clipping identity required')
    epsilon, beta = exact(epsilon), exact(beta)
    require(epsilon >= 0 and beta >= 0, 'pixel bands: nonnegative bounds required')
    image = tuple(map(exact, observation))
    require(len(image) == 432 and (not clipped or all(0 <= y <= 1 for y in image)),
            'pixel bands: complete432 exact clipped scalars required')
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


def masked_rows(observation, geometry, clipped, method, calibration_intervals, mask_name,
                epsilon=EPSILON, beta=BETA):
    """Necessary rows plus exact original row metadata, with no renumbering.

    The contractor consumes the retained tuple and therefore its certificate
    row index is a retained-row ordinal. The separate metadata row_id counts
    every existing side in the original RGB-intersection row sequence.
    """
    require(method in METHODS and len(geometry) == 144, 'masked rows: registered method/layout required')
    intervals = [pair(value, 'masked rows calibration') for value in calibration_intervals]
    require(len(intervals) == 3 and all(0 <= a <= b <= 2 * exact(beta) for a, b in intervals),
            'masked rows: legal full-image calibration required')
    dropped, rows, metadata, row_id = set(mask_pixels(mask_name)), [], [], 0
    for pixel, (high, low) in enumerate(pixel_bands(observation, clipped, epsilon, beta)):
        bounds = tuple(map(exact, geometry[pixel]))
        require(len(bounds) == 6 and all(0 <= bounds[k] <= bounds[k + 1] <= 1 for k in (0, 2, 4)),
                'masked rows: invalid independent geometric coefficients')
        view, local = divmod(pixel, 48)
        row, column = divmod(local, 8)
        tlow, thigh = intervals[view]
        for side, bound, coefficients in (('upper', high, bounds[::2]), ('lower', low, bounds[1::2])):
            if bound is None:
                continue
            if pixel in dropped:
                metadata.append({'row_id': row_id, 'pixel': pixel, 'view': view,
                                 'row': row, 'column': column, 'side': side})
            else:
                if method == 'joint':
                    coefficients += tuple(Q(int(index == view)) for index in range(3))
                else:
                    bound -= tlow if side == 'upper' else thigh
                rows.append((side, coefficients, bound))
            row_id += 1
    return tuple(rows), {'name': mask_name, 'dropped_pixels': sorted(dropped),
                        'dropped_rows': metadata, 'kept_row_count': len(rows)}


def verify_mask_metadata(actual, expected, where):
    require(isinstance(actual, dict) and set(actual) == {'name', 'dropped_pixels', 'dropped_rows', 'kept_row_count'},
            f'{where}: invalid inference-mask schema')
    require(type(actual['name']) is str and isinstance(actual['dropped_pixels'], list) and
            all(type(pixel) is int for pixel in actual['dropped_pixels']) and
            type(actual['kept_row_count']) is int and isinstance(actual['dropped_rows'], list),
            f'{where}: invalid exact inference-mask fields')
    for row in actual['dropped_rows']:
        require(isinstance(row, dict) and set(row) == {'row_id', 'pixel', 'view', 'row', 'column', 'side'} and
                all(type(row[key]) is int for key in ('row_id', 'pixel', 'view', 'row', 'column')) and
                row['side'] in ('upper', 'lower'), f'{where}: invalid original row identity')
    same(actual, expected, f'{where}: independent fixed mask and every extant original row side')


def verify_branch(outer, observation, clipped, method, branch, geometry_bounds, calibration, mask_name, where):
    """Frozen tree verifier adapted explicitly to independently masked rows."""
    fields = {'domain', 'epsilon', 'beta', 'clipped', 'tolerance', 'max_boxes', 'evaluated_boxes', 'trace',
              'retained_ids', 'hull', 'shared_tests', 'propagation_rounds', 'mode', 'rounds', 'budget',
              'channel_precheck', 'initial_bounds', 'scope'}
    require(isinstance(outer, dict) and set(outer) == fields, f'{where}: invalid structured branch schema')
    initial = [[encoded(0), encoded(1)] for _ in range(3)]
    if method == 'joint':
        initial.extend(calibration['intervals'])
    same(outer['initial_bounds'], initial, f'{where}: public material and same full-image T_cal priors')
    require(pair(outer['domain'], where) == DOMAIN and exact(outer['epsilon']) == EPSILON and exact(outer['beta']) == BETA and
            type(outer['clipped']) is bool and outer['clipped'] == clipped and exact(outer['tolerance']) == TOLERANCE and
            type(outer['max_boxes']) is int and outer['max_boxes'] == MAX_BOXES and outer['mode'] == method and
            outer['scope'] == _prior.GROUP_SCOPE and outer['channel_precheck'] == 'none', f'{where}: changed scientific branch contract')
    require(all(type(outer[key]) is int and outer[key] == expected for key, expected in
                (('rounds', ROUNDS), ('propagation_rounds', ROUNDS), ('budget', MAX_BOXES))), f'{where}: changed branch budget')
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
            rows, _metadata = masked_rows(observation, geometry, clipped, method, calibration['intervals'], mask_name)
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


def verify_masked_arm(arm, public, method, mask_name, where):
    require(method in METHODS and mask_name in (*MASKS, 'none'), f'{where}: unknown method/mask')
    fields = {'branches', 'outer', 'branch_seconds', 'outer_seconds', 'preparation_seconds',
              'background_mask', 'calibration_intervals', 'calibration', 'inference_mask'}
    require(isinstance(arm, dict) and set(arm) == fields, f'{where}: changed masked arm schema')
    require(isinstance(public, dict) and type(public.get('clipped')) is bool and public['clipped'] is True and
            exact(public['radius']) == RADIUS and exact(public['baseline']) in BASELINES,
            f'{where}: changed public camera/noise contract')
    image, baseline = tuple(map(exact, public['observation'])), exact(public['baseline'])
    def geometry_bounds(branch, radius, low, high):
        return independent_geometry(branch, radius, low, high, baseline)
    require(list(arm['branches']) == list(arm['branch_seconds']) == list(BRANCHES),
            f'{where}: changed complete three-branch order')
    calibration = _prior.verify_arm_public(arm, image, True, geometry_bounds, where)
    _rows, metadata = masked_rows(image, geometry_bounds('none', RADIUS, *DOMAIN), True,
                                  method, calibration['intervals'], mask_name)
    verify_mask_metadata(arm['inference_mask'], metadata, where)
    total = finite_seconds(arm['outer_seconds'], where + ': solver seconds')
    measured = finite_seconds(arm['preparation_seconds'], where + ': preparation seconds')
    measured += sum(finite_seconds(value, where + ': branch seconds') for value in arm['branch_seconds'].values())
    require(total >= measured, f'{where}: impossible recorded solve cost')
    branches = {branch: verify_branch(arm['branches'][branch], image, True, method, branch,
                                      geometry_bounds, calibration, mask_name, f'{where}/{branch}')
                for branch in BRANCHES}
    union = _prior.verify_union(arm['outer'], branches, method, True, where)
    counts = {'tree_nodes': sum(len(arm['branches'][branch]['trace']) for branch in BRANCHES),
              'exact_updates': sum(value['proof_updates'] for value in branches.values()),
              'contradiction_certificates': sum(len(value['shared_excluded']) for value in branches.values()),
              'geometry_calls': sum(value['evaluated'] for value in branches.values())}
    return {'union': union, 'branches': branches, 'counts': counts, 'calibration': calibration}


def selector_action(hull, incumbent, selector):
    require(selector in SELECTORS, 'action: unknown registered selector')
    if selector == 'hybrid':
        return _arithmetic.hybrid_action(hull, incumbent)
    action = _arithmetic.endpoint_action(hull, incumbent, selector)
    return {**action, 'branch': selector if action['status'] == 'MOVE' else
            {'KEEP': 'keep', 'INCOMPATIBLE': 'incompatible'}[action['status']]}


def verify_action(decision, task, public, checked, condition, selector=False, where='action'):
    condition_key = 'selector' if selector else 'mask_name'
    identity = {'task_id': task['task_id'], 'method': decision.get('method'), condition_key: condition,
                'unit_id': public['unit_id'], 'reused': selector}
    require(identity['method'] in METHODS and isinstance(decision, dict) and type(decision.get('reused')) is bool,
            f'{where}: false action method/origin')
    expected = selector_action(checked['hull'], task['incumbent'], condition if selector else 'hybrid')
    expected = {**identity, **{key: None if value is None else encoded(value) if key in ('output', 'gain_lower') else value
                              for key, value in expected.items()}}
    same(decision, expected, f'{where}: independent endpoint proposal/certificate/action identity')
    return exact(decision['output'], where)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scientific_files():
    files = dict(_baseline.scientific_files())
    for path in sorted({ROOT / 'AGENTS.md', ROOT / 'PROTOCOL.md', ROOT / 'REGISTRATION.json', *ROOT.glob('*.py')}):
        files[str(path.resolve())] = sha(path)
    return dict(sorted(files.items()))


def input_files():
    files = dict(_baseline.generated_inputs())
    for relative in ('outers.json', 'decisions.json', 'RUN_STATUS.json', 'verification/VERIFY.json', 'SOURCE_LOCK.json'):
        path = FULL_OUTPUT / relative
        files[str(path.resolve())] = sha(path)
    return dict(sorted(files.items()))


def verify_source_lock(lock):
    require(isinstance(lock, dict) and set(lock) == {'source_root', 'files'} and lock['source_root'] == str(ROOT),
            'SOURCE_LOCK: exact child source identity/schema required')
    files = scientific_files()
    same(lock['files'], files, 'SOURCE_LOCK: independently enumerated complete frozen/new source bytes')
    same(load(FULL_OUTPUT / 'SOURCE_LOCK.json')['files'], _baseline.scientific_files(),
         'SOURCE_LOCK: exact verified full recovery graph remains unchanged')
    return files


def verify_inputs(lock):
    require(isinstance(lock, dict) and set(lock) == {'files'}, 'INPUTS: exact frozen input schema required')
    files = input_files()
    same(lock['files'], files, 'INPUTS: complete sealed original data and full result bytes')
    previous, status = load(FULL_OUTPUT / 'verification/VERIFY.json'), load(FULL_OUTPUT / 'RUN_STATUS.json')
    require(previous.get('status') == 'PASS' and previous.get('host') == 'liekkas' and
            previous.get('output') == str(FULL_OUTPUT) and status.get('status') == 'COMPLETE',
            'INPUTS: exact complete independently verified full baseline required')
    same(load(FULL_OUTPUT / 'INPUTS.json')['files'], _baseline.generated_inputs(),
         'INPUTS: same frozen original paired data bytes')
    return files


def verify_budget(budget):
    require(isinstance(budget, dict) and set(budget) == {'start_epoch', 'deadline_epoch', 'limit_seconds', 'scope'},
            'BUDGET: invalid new absolute window schema')
    start, end = (finite_seconds(budget[key], 'BUDGET ' + key) for key in ('start_epoch', 'deadline_epoch'))
    require(type(budget['limit_seconds']) is int and budget['limit_seconds'] == LIMIT and
            end == start + LIMIT and budget['scope'] == BUDGET_SCOPE,
            'BUDGET: changed registered3600-second absolute window')
    if time.time() >= end:
        raise TimeoutError('A7/A1/A5 main plus independent verification deadline exhausted')
    return end


@contextmanager
def deadline_guard(output):
    deadline = verify_budget(load(Path(output) / 'BUDGET.json'))
    require(not signal.getitimer(signal.ITIMER_REAL)[0], 'BUDGET: existing deadline cannot be reset')
    def reached(_signal, _frame):
        raise TimeoutError('A7/A1/A5 main plus independent verification deadline reached')
    previous = signal.signal(signal.SIGALRM, reached)
    signal.setitimer(signal.ITIMER_REAL, max(0.000001, deadline - time.time()))
    try:
        yield deadline
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def check_target(output=None):
    require(socket.gethostname() == 'liekkas' and ROOT == EXPECTED_ROOT and ROOT.resolve() == EXPECTED_ROOT,
            'target: exact ablation child on liekkas required')
    if output is not None:
        output = Path(output).resolve()
        require(output.is_dir() and output != ROOT and output.is_relative_to(ROOT),
                'target: existing output strictly inside exact ablation child required')
    return output


def verify_public_population(output):
    # The frozen verifier checks all36 exact public IDs,432 RGB scalars and324
    # task IDs without private parsing or a new generator call.
    return _baseline.verify_public_population(output)


def verify_action_population(records, observations, tasks, selector=False):
    conditions = SELECTORS if selector else MASKS
    key = 'selector' if selector else 'mask_name'
    count = 1944 if selector else 1296
    require(isinstance(records, list) and len(records) == count, f'actions: complete{count} persisted records required')
    actions = indexed(records, lambda row: (row['task_id'], row['method'], row[key]), 'saved actions')
    expected = ([(f'{oid}s{state}', method, condition)
                 for oid in observations for method in METHODS for condition in conditions for state in range(9)]
                if selector else [(f'{oid}s{state}', method, condition)
                 for oid in observations for condition in conditions for method in METHODS for state in range(9)])
    require(list(actions) == expected and {tid for tid, _method, _condition in expected} == set(tasks),
            'actions: missing/reordered/duplicate task, method or condition')
    require(all(type(row.get('reused')) is bool and row['reused'] is selector for row in records),
            'actions: false frozen/new origin')
    return actions


def make_result(task, private, decision, checked, condition):
    truth, incumbent, output = exact(private['truth']), exact(task['incumbent']), exact(decision['output'])
    hull = checked['hull']
    initial, final = abs(incumbent - truth), abs(output - truth)
    return {'task_id': task['task_id'], 'base_task_id': task['base_task_id'],
            'observation_id': task['observation_id'], 'pair_id': private['pair_id'],
            'unit_id': private['unit_id'], 'baseline': private['baseline'], 'method': decision['method'],
            'condition': condition, 'case': private['case'], 'depth_mm': str(truth),
            'initial_ae_mm': str(initial), 'ae_mm': str(final), 'se_mm2': str(final * final),
            'actual_gain_mm2': str(initial * initial - final * final), 'output': str(output),
            'status': decision['status'], 'covered': hull is not None and hull[0] <= truth <= hull[1],
            'hull_width_mm': None if hull is None else str(hull[1] - hull[0])}


def verify_private_population(observations, tasks):
    """Read labels only after independently validated public saved actions."""
    records = load(DATA / 'private/truth.json')
    truths = indexed(records, lambda row: row['observation_id'], 'sealed private labels')
    require(list(truths) == list(observations), 'private population: changed sealed36 image identities')
    witnesses = []
    for oid, private in truths.items():
        public = observations[oid]
        require(private['pair_id'] == public['pair_id'] and private['unit_id'] == public['unit_id'] and
                private['baseline'] == public['baseline'] and private['rear_branch'] in BRANCHES and
                private['seed'] == 61005, f'{oid}: changed sealed witness lineage')
        truth = exact(private['truth'])
        require(truth in _baseline._historical.TRUTHS, f'{oid}: changed registered depth')
        predicted = _baseline.independent_rgb(private['rear_branch'], truth, private['actual_offsets'],
                                              private['colors'], public['baseline'])
        noise = tuple(map(exact, private['image_noise']))
        bias = tuple(map(exact, private['actual_view_bias']))
        require(len(noise) == 432 and all(abs(value) <= EPSILON for value in noise) and
                bias == (-BETA, Q(0), BETA), f'{oid}: illegal sealed noise/bias witness')
        rebuilt = tuple(min(Q(1), max(Q(0), value + noise[index] + bias[index // 144]))
                        for index, value in enumerate(predicted))
        same(list(map(encoded, rebuilt)), public['observation'], f'{oid}: independent exact point image/noise/bias/clamp witness')
        darkness = tuple(1 - exact(color) for color in private['colors'])
        areas = _baseline.independent_point_areas(private['rear_branch'], truth, private['actual_offsets'], public['baseline'])
        for pixel, (high, low) in enumerate(pixel_bands(public['observation'], True)):
            value = sum((a * d for a, d in zip(areas[pixel], darkness)), Q(0)) + BETA - bias[pixel // 48]
            require((high is None or value <= high) and (low is None or value >= low),
                    f'{oid}: actual latent witness violates clipped observation semantics')
        for state, offset in enumerate(_baseline._historical.OFFSETS):
            same(tasks[f'{oid}s{state}']['incumbent'], encoded(truth + offset), f'{oid}: sealed nine public initial offsets')
        witnesses.append({'observation_id': oid, 'clip_aware_true_witness_valid': True})
    return truths, witnesses


def verify_safe_result(task, private, decision, checked, where):
    truth, incumbent, output = exact(private['truth']), exact(task['incumbent']), exact(decision['output'])
    hull = checked['hull']
    if hull is not None and hull[0] <= truth <= hull[1]:
        gain = exact(decision['gain_lower'])
        require((incumbent - truth) ** 2 - (output - truth) ** 2 >= gain >= 0,
                f'{where}: covered endpoint contradicts independent safe-action certificate')


def aggregate_results(rows):
    """Reconstruct exact summary cells from independently checked endpoints."""
    grouped = {}
    for row in rows:
        prefix = (row['unit_id'], row['method'], row['condition'])
        shifted = exact(row['initial_ae_mm']) != 0
        cells = [('all_shifted' if shifted else 'correct', '')]
        if shifted:
            informative = row['case'] != 'equal_full_negative'
            cells.extend([('informative_shifted' if informative else 'negative_shifted', ''),
                          ('case_shifted', row['case'])])
            if informative:
                cells.append(('depth_informative', row['depth_mm']))
        for cell in cells:
            grouped.setdefault(prefix + cell, []).append(row)
    result = []
    for (unit, method, condition, group, value), selected in sorted(grouped.items()):
        initial = [exact(row['initial_ae_mm']) for row in selected]
        final = [exact(row['ae_mm']) for row in selected]
        count = len(selected)
        result.append({'unit_id': unit, 'method': method, 'condition': condition, 'group': group, 'value': value,
            'n': count, 'mae_mm': str(sum(final, Q(0)) / count),
            'mse_mm2': str(sum((error * error for error in final), Q(0)) / count),
            'keep_mse_mm2': str(sum((error * error for error in initial), Q(0)) / count),
            'improved': sum(after < before for before, after in zip(initial, final)),
            'same': sum(after == before for before, after in zip(initial, final)),
            'harmed': sum(after > before for before, after in zip(initial, final))})
    return result


def verify_summary_denominators(summary):
    for name, cells in summary.items():
        expected_conditions = MASKS if name == 'masked' else SELECTORS if name == 'selectors' else ('full',)
        expected_methods = METHODS if name != 'reference' else ('keep', *METHODS)
        for unit in UNITS:
            for method in expected_methods:
                for condition in expected_conditions:
                    chosen = {(row['group'], row['value']): row for row in cells
                              if (row['unit_id'], row['method'], row['condition']) == (unit, method, condition)}
                    for group, count in (('all_shifted', 144), ('correct', 18), ('informative_shifted', 120), ('negative_shifted', 24)):
                        require(chosen.get((group, ''), {}).get('n') == count,
                                f'{name}/{unit}/{method}/{condition}: changed{group} denominator')
                    depth = [row for (group, _value), row in chosen.items() if group == 'depth_informative']
                    templates = [row for (group, _value), row in chosen.items() if group == 'case_shifted']
                    require(len(depth) == 3 and all(row['n'] == 40 for row in depth) and
                            len(templates) == 6 and all(row['n'] == 24 for row in templates),
                            f'{name}/{unit}/{method}/{condition}: changed all-depth/six-template populations')


def audit_runner_source(source=None):
    """Check actual persisted-matrix fences and public solver arguments."""
    path = ROOT / 'run_ablation.py'
    code = path.read_text(encoding='utf-8') if source is None else source
    tree = ast.parse(code)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    require({'run', 'evaluate', 'check_actions', 'population'} <= set(functions),
            'source audit: missing persisted/public action entry points')
    run = functions['run']
    calls = [(node.func.id, node) for node in ast.walk(run)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    evaluations = [node for name, node in calls if name == 'evaluate']
    require(len(evaluations) == 1, 'source audit: exactly one post-action evaluation required')
    evaluate_line = evaluations[0].lineno
    callers = [(function.name, node.lineno) for function in functions.values() for node in ast.walk(function)
               if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'evaluate']
    require(callers == [('run', evaluate_line)], 'source audit: additional private evaluation route')
    dump = lambda node: ast.dump(node, include_attributes=False)
    expected_statements = (
        "save(output/'selectors/decisions.json', selector_decisions)",
        "selector_decisions = load(output/'selectors/decisions.json')",
        "check_actions(selector_decisions,tasks,SELECTORS,True)",
        "save(output/'outers.json',outers)",
        "save(output/'decisions.json',decisions)",
        "decisions = load(output/'decisions.json')",
        "outers = load(output/'outers.json')",
        "check_actions(decisions,tasks,MASKS,False)",
        "if calls != 144 or len(outers) != 36:\n    raise RuntimeError('incomplete solve matrix')",
    )
    tries = [node for node in run.body if isinstance(node, ast.Try)]
    require(len(tries) == 1, 'source audit: changed main action/evaluation control flow')
    positions = []
    for statement in expected_statements:
        target = dump(ast.parse(statement).body[0])
        matches = [node for node in tries[0].body if dump(node) == target]
        require(len(matches) == 1 and matches[0].end_lineno < evaluate_line,
                'source audit: both complete persisted/readback action fences required before private labels')
        positions.append(matches[0].lineno)
    require(positions == sorted(positions), 'source audit: reordered save/readback/complete matrix fences')
    predicate = "len(keys) != len(expected) or len(set(keys)) != len(expected) or set(keys) != expected or any(r['reused'] != replay for r in records)"
    action_checks = [node for node in functions['check_actions'].body if isinstance(node, ast.If) and
                     dump(node.test) == dump(ast.parse(predicate, mode='eval').body) and
                     any(isinstance(child, ast.Raise) for child in node.body)]
    require(len(action_checks) == 1, 'source audit: unique complete action population/origin fence changed')
    private_paths = [(function.name, node) for function in functions.values() for node in ast.walk(function)
                     if isinstance(node, ast.Constant) and isinstance(node.value, str) and 'truth.json' in node.value]
    # inputs() seals bytes; only evaluate() may parse private data.
    require(sorted(name for name, _node in private_paths) == ['evaluate', 'inputs'],
            'source audit: direct private label path outside sealed hashes/post-action evaluation')
    parses = [node for node in ast.walk(functions['evaluate']) if isinstance(node, ast.Call) and
              isinstance(node.func, ast.Name) and node.func.id == 'load' and node.args and
              dump(node.args[0]) == dump(ast.parse("DATA / 'private/truth.json'", mode='eval').body)]
    require(len(parses) == 1, 'source audit: exact sealed private parse path required')
    execution_calls = [node for node in ast.walk(run) if isinstance(node, ast.Call) and
                       isinstance(node.func, ast.Attribute) and node.func.attr == 'solve_public']
    require(len(execution_calls) == 1 and [(keyword.arg, dump(keyword.value)) for keyword in execution_calls[0].keywords] ==
            [('clipped', dump(ast.parse("public['clipped']", mode='eval').body))] and
            [dump(arg) for arg in execution_calls[0].args] == [dump(ast.parse(expression, mode='eval').body)
                for expression in ("public['observation']", "public['radius']", "public['baseline']", 'method', 'mask')],
            'source audit: inference receives nonpublic data or altered arguments')
    return {'status': 'PASS', 'runner': str(path), 'private_parse_line': parses[0].lineno,
            'post_action_evaluation_call_line': evaluate_line, 'persisted_matrix_gate_lines': positions,
            'solver_arguments': ['observation', 'radius', 'baseline', 'method', 'mask_name', 'clipped']}


def verify(output, deadline=None):
    output = check_target(output)
    budget = load(output / 'BUDGET.json')
    deadline = verify_budget(budget)
    sources, inputs = verify_source_lock(load(output / 'SOURCE_LOCK.json')), verify_inputs(load(output / 'INPUTS.json'))
    same(load(output / 'REGISTRATION.json'), load(ROOT / 'REGISTRATION.json'), 'REGISTRATION: exact frozen child contract copy')
    audit = audit_runner_source()
    observations, tasks = verify_public_population(output)
    new_records, replay_records = load(output / 'decisions.json'), load(output / 'selectors/decisions.json')
    decisions = verify_action_population(new_records, observations, tasks)
    selectors = verify_action_population(replay_records, observations, tasks, True)
    outers, full = load(output / 'outers.json'), load(FULL_OUTPUT / 'outers.json')
    require(isinstance(outers, dict) and list(outers) == list(observations), 'outers: complete36 ordered public image IDs required')
    full_actions = indexed(load(FULL_OUTPUT / 'decisions.json'), lambda row: (row['task_id'], row['method']), 'frozen full actions')
    checked = {}
    counts = Counter()
    timing_records = load(output / 'TIMING.json')
    require(isinstance(timing_records, list) and len(timing_records) == 144, 'TIMING: complete144 cold calls required')
    timing = indexed(timing_records, lambda row: (row['observation_id'], row['mask_name'], row['method']), 'cold timing')
    same(list(timing), [(oid, mask, method) for oid in observations for mask in MASKS for method in METHODS],
         'TIMING: exact144 public image/mask/method order')
    for index, (oid, public) in enumerate(observations.items()):
        require(isinstance(outers[oid], dict) and list(outers[oid]) == list(MASKS), f'{oid}: exact public mask order required')
        checked[oid] = {}
        for mask_name in MASKS:
            require(isinstance(outers[oid][mask_name], dict) and list(outers[oid][mask_name]) == list(METHODS),
                    f'{oid}/{mask_name}: complete calibrated/joint method order required')
            checked[oid][mask_name] = {}
            for method in METHODS:
                arm = outers[oid][mask_name][method]
                result = verify_masked_arm(arm, public, method, mask_name, f'{oid}/{mask_name}/{method}')
                same(result['calibration'], full[oid][method]['calibration'], f'{oid}/{mask_name}/{method}: full-image frozen T_cal unchanged')
                checked[oid][mask_name][method] = result
                counts.update(result['counts'])
                counts['cold_solver_calls'] += 1
                measured = timing[oid, mask_name, method]
                require(set(measured) == {'observation_id', 'mask_name', 'method', 'cold_seconds', 'evaluated_boxes'} and
                        type(measured['evaluated_boxes']) is int and measured['evaluated_boxes'] == result['union']['evaluated'] and
                        finite_seconds(measured['cold_seconds'], 'TIMING cold seconds') >=
                        finite_seconds(arm['outer_seconds'], 'arm outer seconds'),
                        f'{oid}/{mask_name}/{method}: false cold timing/node count')
        if index % 3 == 2:
            print(json.dumps({'stage': 'independent_ablation_verification', 'observations_complete': index + 1,
                              'remaining_seconds': round(deadline - time.time(), 1)}), flush=True)
    checkpoint = output / 'checkpoint'
    require(checkpoint.is_dir() and sorted(path.name for path in checkpoint.iterdir()) ==
            sorted(f'{oid}.json' for oid in observations), 'checkpoint: complete36 exact image checkpoint files required')
    for oid in observations:
        same(load(checkpoint / f'{oid}.json'), {'observation_id': oid, 'arms': outers[oid]}, f'{oid}: exact persisted checkpoint readback')
    for record in new_records:
        task = tasks[record['task_id']]
        oid, method, mask_name = task['observation_id'], record['method'], record['mask_name']
        verify_action(record, task, observations[oid], checked[oid][mask_name][method]['union'], mask_name,
                      where=f'{task["task_id"]}/{method}/{mask_name}')
    full_checked = {oid: {method: {'hull': None if full[oid][method]['outer']['hull'] is None else
                                    pair(full[oid][method]['outer']['hull'], 'sealed full hull')}
                          for method in METHODS} for oid in observations}
    for record in replay_records:
        task = tasks[record['task_id']]
        oid, method, selector = task['observation_id'], record['method'], record['selector']
        verify_action(record, task, observations[oid], full_checked[oid][method], selector, True,
                      f'{task["task_id"]}/{method}/{selector}')
        if selector == 'hybrid':
            frozen = full_actions[record['task_id'], method]
            same({key: record[key] for key in ('status', 'output', 'gain_lower', 'reason', 'branch')},
                 {key: frozen[key] for key in ('status', 'output', 'gain_lower', 'reason', 'branch')},
                 f'{task["task_id"]}/{method}: exact frozen hybrid648 replay semantics')
    # Every public arm and all3240 saved actions have now passed independent
    # checks. This is the only private-label parse gate in this verifier.
    truths, witnesses = verify_private_population(observations, tasks)
    results, replay_results, coverage, reference_results = [], [], [], []
    for record in new_records:
        task = tasks[record['task_id']]
        oid, method, mask_name = task['observation_id'], record['method'], record['mask_name']
        union = checked[oid][mask_name][method]['union']
        verify_safe_result(task, truths[oid], record, union, f'{task["task_id"]}/{method}/{mask_name}')
        results.append(make_result(task, truths[oid], record, union, mask_name))
    for record in replay_records:
        task = tasks[record['task_id']]
        oid, method, selector = task['observation_id'], record['method'], record['selector']
        union = full_checked[oid][method]
        verify_safe_result(task, truths[oid], record, union, f'{task["task_id"]}/{method}/{selector}')
        replay_results.append(make_result(task, truths[oid], record, union, selector))
    for record in full_actions.values():
        task = tasks[record['task_id']]
        oid, method = task['observation_id'], record['method']
        frozen_hull = full[oid][method]['outer']['hull']
        union = {'hull': None if frozen_hull is None else pair(frozen_hull, 'sealed reference hull')}
        reference_results.append(make_result(task, truths[oid], record, union, 'full'))
    for oid, public in observations.items():
        for mask_name in MASKS:
            for method in METHODS:
                arm = checked[oid][mask_name][method]
                coverage.append({'mask_name': mask_name, **_prior.make_coverage(public, truths[oid], method, arm['union'], arm['branches'])})
    same(load(output / 'results.json'), results, 'evaluation: all1296 independent exact masked endpoint errors')
    same(load(output / 'selectors/results.json'), replay_results, 'evaluation: all1944 independent exact selector endpoint errors')
    same(load(output / 'reference/results.json'), reference_results, 'evaluation: all972 frozen reference endpoint errors, zero new solves')
    same(load(output / 'coverage.json'), coverage, 'evaluation: all144 independent retained/latent coverage records')
    for relative, rows in (('results.csv', results), ('selectors/results.csv', replay_results), ('reference/results.csv', reference_results)):
        _baseline.verify_csv(output / relative, rows)
    summary = {'masked': aggregate_results(results), 'selectors': aggregate_results(replay_results),
               'reference': aggregate_results(reference_results)}
    verify_summary_denominators(summary)
    same(load(output / 'summary.json'), summary, 'evaluation: all exact public-mask/selector/reference depth and template summary cells')
    status = load(output / 'RUN_STATUS.json')
    require(isinstance(status, dict) and set(status) == {'status', 'new_solver_calls', 'decisions_complete',
                'selector_decisions_complete', 'observations_complete', 'elapsed_seconds', 'deadline_epoch'} and
            status['status'] == 'COMPLETE' and all(type(status[key]) is int and status[key] == value for key, value in
                (('new_solver_calls', 144), ('decisions_complete', 1296), ('selector_decisions_complete', 1944), ('observations_complete', 36))) and
            status['deadline_epoch'] == deadline, 'RUN_STATUS: changed complete main-run accounting/schema')
    elapsed = finite_seconds(status['elapsed_seconds'], 'RUN_STATUS elapsed seconds')
    require(elapsed < LIMIT and sum(finite_seconds(row['cold_seconds'], 'cold seconds') for row in timing_records) <= elapsed,
            'TIMING: recorded144 cold solves exceed shared elapsed main run')
    same(scientific_files(), sources, 'SOURCE_LOCK: source bytes unchanged throughout independent verification')
    same(input_files(), inputs, 'INPUTS: all sealed input bytes unchanged throughout independent verification')
    verify_budget(budget)
    return {'status': 'PASS', 'mechanical_verification': 'PASS', 'host': socket.gethostname(), 'output': str(output),
            'observations': len(observations), 'tasks': len(tasks), 'decisions': len(decisions),
            'selector_decisions': len(selectors), 'cold_solver_calls': counts['cold_solver_calls'],
            'coverage_rows': len(coverage), 'branch_outer_traces': 432, 'tree_nodes_verified': counts['tree_nodes'],
            'exact_updates_recomputed': counts['exact_updates'], 'contradiction_certificates_verified': counts['contradiction_certificates'],
            'independent_geometry_calls': counts['geometry_calls'], 'source_files_verified': len(sources),
            'input_files_verified': len(inputs), 'source_parse_order_audit': audit, 'actual_truth_witnesses': witnesses,
            'scientific_results': 'all exact errors, regressions, depth/template groups and coverage preserved; no positive-gain acceptance condition',
            'shared_budget': {'start_epoch': budget['start_epoch'], 'deadline_epoch': deadline,
                              'elapsed_main_plus_verification_seconds': time.time() - budget['start_epoch']}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = check_target(args.output)
    with deadline_guard(output) as deadline:
        result = verify(output, deadline)
    # Only the verifier's own final receipt is written after its full replay.
    destination = output / 'verification/VERIFY.json'
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

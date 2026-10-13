#!/usr/bin/env python3
"""Independent exact certificate, population and accounting verification.

This module never imports run_shared/shared_solver or their propagation/action
functions. Shared-material rows and propagation proofs are reconstructed here.
Only the disclosed historical image/area kernels and historical independent
arithmetic verifier are reused. No historical evidence directory is written.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from fractions import Fraction as Q
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import socket
import sys
import time

sys.dont_write_bytecode = True
for _shared_verification_threads in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_shared_verification_threads] = '1'

ROOT = Path(__file__).resolve().parent
NUISANCE_ROOT = ROOT.parent / 'lovelace-nuisance-20261011T044500Z'
HYBRID_ROOT = ROOT.parent / 'lovelace-hybrid-20261010T214217Z'
CHALLENGE_ROOT = ROOT.parent / 'lovelace-challenge-20261010T205013Z'
_spec = importlib.util.spec_from_file_location('_shared_verifier_historical_arithmetic',
                                             NUISANCE_ROOT / 'verification/arithmetic.py')
_arithmetic = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_arithmetic)
VerificationError = _arithmetic.VerificationError
exact, encoded, pair = _arithmetic.exact, _arithmetic.encoded, _arithmetic.pair
require, indexed = _arithmetic.require, _arithmetic.indexed
DOMAIN, EPSILON, TOLERANCE, MAX_BOXES = _arithmetic.DOMAIN, _arithmetic.EPSILON, _arithmetic.TOLERANCE, _arithmetic.MAX_BOXES
ROUNDS = 8
BRANCHES = ('none', 'partial', 'full')
METHODS = ('keep', 'channelwise_union', 'shared_union', 'calibrated_union')
SOURCES = ('front-left', 'front-right', 'rear')
RADIUS = Q(1, 5)
WEIGHTS = (Q(-1), Q(-1, 2), Q(0), Q(1, 2), Q(1))
TRUTHS = (Q(6203, 11), Q(7861, 13), Q(9007, 14))
SEEDS = (61001, 61002, 61003)
OFFSETS = (Q(0), Q(-1, 4), Q(1, 4), Q(-3, 4), Q(3, 4), Q(-2), Q(2), Q(-3), Q(3))
CASES = (
    ('uniform_isolated', 'none', (Q(1, 5), Q(1, 5), Q(2, 5))),
    ('textured_isolated', 'none', (Q(1, 5), Q(3, 5), Q(2, 5))),
    ('textured_partial', 'partial', (Q(1, 5), Q(3, 5), Q(2, 5))),
    ('equal_partial', 'partial', (Q(1, 5), Q(1, 5), Q(1, 5))),
    ('weak_partial', 'partial', (Q(3, 10), Q(3, 10), Q(31, 100))),
    ('equal_full_negative', 'full', (Q(1, 5), Q(1, 5), Q(1, 5))),
)
SHARED_SCOPE = 'necessary_shared_material_halfspaces_with_conservative_continuous_geometry_bounds'


def same(actual, expected, where):
    require(json.dumps(actual, sort_keys=True, allow_nan=False) ==
            json.dumps(expected, sort_keys=True, allow_nan=False), f'{where}: differs from independent reconstruction')


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f'JSON: duplicate object key {key!r}')
        result[key] = value
    return result


def load(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream, object_pairs_hook=unique_object,
                         parse_constant=lambda value: (_ for _ in ()).throw(VerificationError(f'JSON: nonfinite {value}')))


def finite_seconds(value, where):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0, f'{where}: invalid measured time')
    return value


def necessary_rows(observation, geometry):
    """Independent RGB intersection and L/U coefficient selection.

    Row 2p is L_p*d <= min_RGB(1-y+epsilon); row 2p+1 is
    U_p*d >= max_RGB(1-y-epsilon). Variables are shared darkness d=1-c.
    """
    image = tuple(exact(value) for value in observation)
    require(len(image) == 3 * len(geometry) and image, 'shared rows: RGB/geometry dimension mismatch')
    rows = []
    for pixel, bounds in enumerate(geometry):
        require(len(bounds) == 6, 'shared rows: three area endpoint pairs required')
        bounds = tuple(exact(value) for value in bounds)
        require(all(0 <= bounds[j] <= bounds[j + 1] <= 1 for j in (0, 2, 4)),
                'shared rows: invalid normalized area bounds')
        rgb = image[3 * pixel:3 * pixel + 3]
        rows.append(('upper', (bounds[0], bounds[2], bounds[4]), min(Q(1) - y + EPSILON for y in rgb)))
        rows.append(('lower', (bounds[1], bounds[3], bounds[5]), max(Q(1) - y - EPSILON for y in rgb)))
    return tuple(rows)


def verify_certificate(rows, certificate, where):
    """Replay only an exclusion proof, independently deriving every update.

    Missing, reordered, forged or additional updates fail. The contradiction
    must be a row extremum contradiction within the frozen eight rounds;
    variable emptiness is never accepted as a substitute certificate.
    """
    require(isinstance(certificate, dict) and set(certificate) == {'steps', 'contradiction'},
            f'{where}: invalid shared certificate schema')
    steps, contradiction = certificate['steps'], certificate['contradiction']
    require(isinstance(steps, list) and isinstance(contradiction, dict) and set(contradiction) == {'row'},
            f'{where}: invalid certificate sequence/contradiction')
    require(type(contradiction['row']) is int and 0 <= contradiction['row'] < len(rows),
            f'{where}: invalid contradiction row')
    require(len(steps) <= ROUNDS * len(rows) * 3, f'{where}: certificate exceeds update budget')
    for step in steps:
        require(isinstance(step, dict) and set(step) == {'row', 'variable', 'side', 'value'} and
                type(step['row']) is int and 0 <= step['row'] < len(rows) and
                type(step['variable']) is int and 0 <= step['variable'] < 3 and
                step['side'] in ('lower', 'upper'), f'{where}: invalid update schema')
        exact(step['value'], where)
    lower, upper = [Q(0)] * 3, [Q(1)] * 3
    cursor = 0
    for round_index in range(ROUNDS):
        changed = False
        for row_index, (side, coefficients, bound) in enumerate(rows):
            require(side in ('upper', 'lower') and len(coefficients) == 3 and
                    all(isinstance(a, Q) and a >= 0 for a in coefficients), f'{where}: invalid necessary row')
            bound = exact(bound)
            values = lower if side == 'upper' else upper
            extreme = sum((a * v for a, v in zip(coefficients, values) if a and v), Q(0))
            conflict = extreme > bound if side == 'upper' else extreme < bound
            if conflict:
                require(cursor == len(steps) and contradiction['row'] == row_index,
                        f'{where}: recorded contradiction/update sequence differs')
                return {'rounds': round_index + 1, 'updates': cursor, 'row': row_index}
            for variable, coefficient in enumerate(coefficients):
                if coefficient == 0:
                    continue
                others = extreme - coefficient * values[variable]
                derived = (bound - others) / coefficient
                tightens = derived < upper[variable] if side == 'upper' else derived > lower[variable]
                if not tightens:
                    continue
                require(lower[variable] <= derived <= upper[variable],
                        f'{where}: update creates empty variable interval without row contradiction')
                require(cursor < len(steps), f'{where}: omitted necessary sequential update')
                expected = {'row': row_index, 'variable': variable, 'side': side, 'value': encoded(derived)}
                same(steps[cursor], expected, f'{where}: exact derived bound update {cursor}')
                if side == 'upper':
                    upper[variable] = derived
                else:
                    lower[variable] = derived
                cursor += 1
                changed = True
        if not changed:
            break
    raise VerificationError(f'{where}: no genuine row contradiction within {ROUNDS} rounds')


def verify_shared_trace(outer, observation, branch, geometry_bounds, where):
    fields = {'domain', 'epsilon', 'tolerance', 'max_boxes', 'evaluated_boxes', 'trace', 'retained_ids',
              'hull', 'shared_tests', 'propagation_rounds', 'scope'}
    require(isinstance(outer, dict) and set(outer) == fields, f'{where}: invalid shared trace schema')
    require(pair(outer['domain'], where) == DOMAIN and exact(outer['epsilon']) == EPSILON and
            exact(outer['tolerance']) == TOLERANCE and type(outer['max_boxes']) is int and
            outer['max_boxes'] == MAX_BOXES and type(outer['propagation_rounds']) is int and
            outer['propagation_rounds'] == ROUNDS and outer['scope'] == SHARED_SCOPE,
            f'{where}: changed shared depth/pose/noise/work contract')
    nodes = indexed(outer['trace'], lambda node: node['id'], where)
    require(0 < len(nodes) <= 2 * MAX_BOXES + 1 and
            all(type(n['id']) is int for n in outer['trace']) and
            [n['id'] for n in outer['trace']] == list(range(len(nodes))), f'{where}: invalid BFS identities')
    require(nodes[0]['parent'] is None, f'{where}: root has a parent')
    boxes = {ident: (exact(n['lo']), exact(n['hi'])) for ident, n in nodes.items()}
    require(boxes[0] == DOMAIN, f'{where}: root is not the full initial depth domain')
    leaves, retained, excluded, shared_excluded, referenced = [], [], [], [], set()
    evaluated, shared_tests, next_child, proof_updates = 0, 0, 1, 0
    for node in outer['trace']:
        ident, status = node['id'], node['status']
        low, high = boxes[ident]
        require(DOMAIN[0] <= low < high <= DOMAIN[1], f'{where}: invalid node interval {ident}')
        extras = {'split': {'children'}, 'excluded': {'witness_channel', 'prediction_interval'},
                  'excluded_shared': {'certificate'}, 'retained_tolerance': set(), 'unresolved_budget': set()}
        require(status in extras and set(node) == {'id', 'parent', 'lo', 'hi', 'status'} | extras[status],
                f'{where}: invalid node status/schema {ident}')
        if ident:
            require(type(node['parent']) is int and node['parent'] in nodes and node['parent'] < ident,
                    f'{where}: invalid parent {ident}')
        if status == 'unresolved_budget':
            require(evaluated == MAX_BOXES, f'{where}: unresolved before budget exhaustion')
        else:
            require(evaluated < MAX_BOXES, f'{where}: evaluated beyond registered budget')
            evaluated += 1
            shared_tests += status != 'excluded'
        if status == 'split':
            require(high - low > TOLERANCE, f'{where}: split at or below tolerance')
            children = node['children']
            require(isinstance(children, list) and children == [next_child, next_child + 1] and
                    all(type(c) is int and c in nodes for c in children), f'{where}: invalid BFS children')
            next_child += 2
            middle = (low + high) / 2
            for child, expected in zip(children, ((low, middle), (middle, high))):
                require(child not in referenced and nodes[child]['parent'] == ident and boxes[child] == expected,
                        f'{where}: gap, overlap, nonbisection or repeated child')
                referenced.add(child)
            continue
        leaves.append((low, high))
        if status == 'excluded':
            channel = node['witness_channel']
            require(type(channel) is int and 0 <= channel < len(observation), f'{where}: invalid witness channel')
            a, b = pair(node['prediction_interval'], where)
            require(0 <= a <= b <= 1 and (a > observation[channel] + EPSILON or b < observation[channel] - EPSILON),
                    f'{where}: channel witness does not conflict with noise band')
            excluded.append(node)
        elif status == 'excluded_shared':
            geometry = geometry_bounds(branch, RADIUS, low, high)
            rows = necessary_rows(observation, geometry)
            checked = verify_certificate(rows, node['certificate'], f'{where}/node{ident}')
            proof_updates += checked['updates']
            shared_excluded.append(node)
        else:
            require(status != 'retained_tolerance' or high - low <= TOLERANCE,
                    f'{where}: tolerance-retained node too wide')
            retained.append(node)
    require(referenced == set(nodes) - {0}, f'{where}: orphan or unreachable nodes')
    _arithmetic.verify_partition(leaves, where)
    require(type(outer['evaluated_boxes']) is int and outer['evaluated_boxes'] == evaluated,
            f'{where}: wrong evaluated count')
    require(type(outer['shared_tests']) is int and outer['shared_tests'] == shared_tests, f'{where}: wrong shared test count')
    require(isinstance(outer['retained_ids'], list) and all(type(i) is int for i in outer['retained_ids']) and
            outer['retained_ids'] == [n['id'] for n in retained], f'{where}: false retained provenance')
    intervals = [boxes[n['id']] for n in retained]
    hull = None if not intervals else (min(a for a, _b in intervals), max(b for _a, b in intervals))
    require((outer['hull'] is None and hull is None) or
            (outer['hull'] is not None and hull is not None and pair(outer['hull'], where) == hull), f'{where}: false retained hull')
    return {'hull': hull, 'intervals': intervals, 'retained': retained, 'boxes': boxes, 'excluded': excluded,
            'shared_excluded': shared_excluded, 'evaluated': evaluated, 'proof_updates': proof_updates,
            'budget_leaves': sum(n['status'] == 'unresolved_budget' for n in outer['trace'])}


def shared_kernels():
    # A standalone module name prevents collisions with historical verification packages.
    sys.path.insert(0, str(NUISANCE_ROOT))
    from nuisance_model import ColoredModel, _geometry_bounds
    return ColoredModel, _geometry_bounds


def regenerate(case, branch, colors, truth, seed, model_class):
    tape = f'shared-appearance-v1:{case}:{RADIUS}:{truth}:{seed}'
    rng = random.Random(tape)
    offsets = (RADIUS * rng.choice(WEIGHTS), Q(0), RADIUS * rng.choice(WEIGHTS))
    model = model_class(branch, camera_radius=RADIUS, colors=((Q(0), Q(1)),) * 3)
    prediction = model.point_prediction(truth, offsets, colors=colors)
    require(len(prediction) == 432, 'regeneration: changed image dimension')
    noise = tuple(EPSILON * rng.choice(WEIGHTS) for _ in prediction)
    calibration_noise = tuple(EPSILON * rng.choice(WEIGHTS) for _ in SOURCES)
    public = {'radius': str(RADIUS), 'observation': [encoded(value + error) for value, error in zip(prediction, noise)],
              'calibration': [{'source': label, 'value': encoded(color + error)}
                              for label, color, error in zip(SOURCES, colors, calibration_noise)]}
    private = {'case': case, 'rear_branch': branch, 'truth': encoded(truth), 'colors': list(map(encoded, colors)),
               'seed': seed, 'random_tape': tape, 'actual_offsets': list(map(encoded, offsets)),
               'image_noise': list(map(encoded, noise)), 'calibration_noise': list(map(encoded, calibration_noise)),
               'noise_linf': encoded(max(map(abs, noise))), 'calibration_noise_linf': encoded(max(map(abs, calibration_noise)))}
    if case == 'equal_full_negative':
        require(all(model.point_prediction(depth, offsets, colors=colors) == prediction for depth in (Q(540), Q(600), Q(660))),
                'negative control: shared point kernel fails depth independence')
    return public, private


def arm_contracts(calibration):
    require(isinstance(calibration, list) and len(calibration) == 3 and
            [item.get('source') for item in calibration] == list(SOURCES) and
            all(set(item) == {'source', 'value'} for item in calibration), 'calibration: wrong sourced sample contract')
    colors = tuple((max(Q(0), exact(item['value']) - EPSILON), min(Q(1), exact(item['value']) + EPSILON)) for item in calibration)
    require(all(low <= high for low, high in colors), 'calibration: empty material interval')
    return {'keep': ((), None), 'channelwise_union': (BRANCHES, ((Q(0), Q(1)),) * 3),
            'shared_union': (BRANCHES, ((Q(0), Q(1)),) * 3), 'calibrated_union': (BRANCHES, colors)}


def contains(intervals, value):
    return any(low <= value <= high for low, high in intervals)


def verify_kernel_boxes(outer, checked, observation, model, where, shared=False):
    calls, witnesses = 0, 0
    for node in outer['trace']:
        if node['status'] == 'unresolved_budget':
            continue
        bounds = model.enclosure(*checked['boxes'][node['id']])
        require(len(bounds) == len(observation) and all(isinstance(a, Q) and isinstance(b, Q) and 0 <= a <= b <= 1 for a, b in bounds),
                f'{where}: invalid shared-kernel enclosure')
        conflict = next(((i, a, b) for i, (a, b) in enumerate(bounds)
                         if a > observation[i] + EPSILON or b < observation[i] - EPSILON), None)
        if conflict is not None:
            channel, a, b = conflict
            require(node['status'] == 'excluded' and node['witness_channel'] == channel,
                    f'{where}: first channel exclusion/status differs')
            same(node['prediction_interval'], [encoded(a), encoded(b)], f'{where}: channel witness {node["id"]}')
            witnesses += 1
        else:
            low, high = checked['boxes'][node['id']]
            expected = 'retained_tolerance' if high - low <= TOLERANCE else 'split'
            require(node['status'] == expected or (shared and node['status'] == 'excluded_shared'),
                    f'{where}: nonconflicting node status differs')
        calls += 1
    require(calls == checked['evaluated'], f'{where}: enclosure call count differs')
    return calls, witnesses


def registration():
    return {'cases_evaluation_only': [{'case': c, 'rear_branch': b, 'colors': list(map(str, colors))} for c, b, colors in CASES],
            'truths_evaluation_only': list(map(str, TRUTHS)), 'seeds_evaluation_only': list(SEEDS),
            'offsets_evaluation_only': list(map(str, OFFSETS)), 'camera_radius_mm': str(RADIUS),
            'epsilon': str(EPSILON), 'tolerance_mm': str(TOLERANCE), 'max_boxes_per_branch': MAX_BOXES,
            'depth_domain_mm': list(map(str, DOMAIN)), 'expected_observations': 54, 'expected_tasks': 486,
            'expected_decisions': 1944, 'methods': list(METHODS), 'expected_correct_tasks': 54, 'expected_small_tasks': 432,
            'allowed_occlusion_branches': list(BRANCHES), 'calibration_sources': list(SOURCES),
            'calibration_samples_per_observation': 3,
            'calibration_samples_used_per_method': {m: 3 if m == 'calibrated_union' else 0 for m in METHODS},
            'random_tape_prefix': 'shared-appearance-v1',
            'random_draw_order': 'left pose; right pose; each image RGB channel; front-left, front-right, rear calibration noise',
            'random_weights': list(map(str, WEIGHTS)), 'shared_constraint_max_rounds': 8,
            'cpu_threads': 1, 'gpu_count': 0, 'wall_time_limit_seconds': 1800,
            'information_note': 'channelwise_union/shared_union use identical images, [0,1] material priors and branches; calibrated_union alone uses three additional sourced samples'}


def scientific_files():
    paths = {ROOT / 'run_shared.py', ROOT / 'shared_solver.py', ROOT / 'shared_verification.py', ROOT / 'PROTOCOL.md'}
    paths.update(ROOT.glob('test_*.py'))
    if (ROOT / 'THEORY.md').exists():
        paths.add(ROOT / 'THEORY.md')
    for directory in (ROOT, NUISANCE_ROOT, HYBRID_ROOT, CHALLENGE_ROOT):
        for pattern in ('*.py', '*.json'):
            paths.update((directory / 'verification').rglob(pattern))
        if directory != ROOT:
            paths.update(p for p in directory.glob('*.py') if not p.name.startswith(('analy', 'report')))
            for pattern in ('*.py', '*.json'):
                paths.update((directory / 'source_snapshot').rglob(pattern))
            paths.add(directory / 'PROTOCOL.md')
            if (directory / 'THEORY.md').exists():
                paths.add(directory / 'THEORY.md')
    return {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


def verify_source_lock(lock):
    fields = {'utc', 'source_commit', 'new_experiment_root', 'read_only_dependency_roots', 'path_format',
              'historical_outputs_used', 'files', 'lock_scope'}
    require(isinstance(lock, dict) and set(lock) == fields, 'SOURCE_LOCK: invalid schema')
    require(lock['source_commit'] == 'bd8d05aad65e2c55cd9cd288beae4ef949606984' and
            lock['new_experiment_root'] == str(ROOT) and
            lock['read_only_dependency_roots'] == [str(NUISANCE_ROOT), str(HYBRID_ROOT), str(CHALLENGE_ROOT)] and
            lock['path_format'] == 'absolute' and lock['historical_outputs_used'] is False and
            lock['lock_scope'] == 'scientific implementation/tests/verifier/protocol; post-confirmation reports and analysis excluded',
            'SOURCE_LOCK: altered experiment/dependency contract')
    require(type(lock['utc']) is str and datetime.fromisoformat(lock['utc']).utcoffset() is not None,
            'SOURCE_LOCK: timezone-aware timestamp required')
    require(isinstance(lock['files'], dict) and all(Path(path).is_absolute() and str(Path(path).resolve()) == path for path in lock['files']),
            'SOURCE_LOCK: canonical absolute source identities required')
    current = scientific_files()
    same(lock['files'], current, 'SOURCE_LOCK complete independent inventory and hashes')
    return current


def make_row(task, private, decision, checked):
    truth, a, b = exact(private['truth']), exact(task['incumbent']), exact(decision['output'])
    initial, error = abs(a - truth), abs(b - truth)
    hull, method = checked['hull'], decision['method']
    return {'task_id': task['task_id'], 'observation_id': task['observation_id'], 'case': private['case'], 'seed': private['seed'],
            'method': method, 'state_group': 'correct' if initial == 0 else 'small',
            'prior_position': 'inside_prior' if DOMAIN[0] <= a <= DOMAIN[1] else 'outside_prior',
            'incumbent': str(a), 'output': str(b), 'status': decision['status'], 'initial_ae_mm': str(initial),
            'ae_mm': str(error), 'se_mm2': str(error * error), 'actual_gain_mm2': str(initial * initial - error * error),
            'covered': contains(checked['intervals'], truth), 'hull_width_mm': None if hull is None else str(hull[1] - hull[0]),
            'retained_branch_count': len(checked['retained_branches']), 'evaluated_boxes': checked['evaluated'],
            'unresolved_budget_leaves': checked['budget_leaves'], 'auxiliary_samples_used': 3 if method == 'calibrated_union' else 0}


def base_aggregates(rows):
    require(rows, 'aggregates: empty evaluation cannot succeed')
    grouped = defaultdict(list)
    for row in rows:
        conditions = (('overall', 'all'), ('by_case', row['case']),
                      ('by_information', 'negative_control' if row['case'] == 'equal_full_negative' else 'informative'))
        for family, condition in conditions:
            for state in ('all', row['state_group']):
                grouped[family, condition, state, row['method']].append(row)
    result = []
    for (family, condition, state, method), values in sorted(grouped.items()):
        n = len(values)
        errors = [exact(row['ae_mm']) for row in values]
        initial = [exact(row['initial_ae_mm']) for row in values]
        widths = [exact(row['hull_width_mm']) for row in values if row['hull_width_mm'] is not None]
        unique = {row['observation_id']: row for row in values}
        result.append({'grouping': family, 'condition': condition, 'state_group': state, 'method': method,
                       'n': n, 'observations': len(unique), 'covered_observations': sum(row['covered'] for row in unique.values()),
                       'empty_observations': sum(row['hull_width_mm'] is None for row in unique.values()),
                       'wrongly_excluded_observations': sum(not row['covered'] for row in unique.values()),
                       'improved': sum(error < before for error, before in zip(errors, initial)),
                       'harm': sum(error > before for error, before in zip(errors, initial)),
                       'same': sum(error == before for error, before in zip(errors, initial)),
                       'moved': sum(row['status'] == 'MOVE' for row in values), 'kept': sum(row['status'] == 'KEEP' for row in values),
                       'incompatible': sum(row['status'] == 'INCOMPATIBLE' for row in values),
                       'correct_initial_preserved': sum(row['state_group'] == 'correct' and exact(row['ae_mm']) == 0 for row in values),
                       'correct_initial_kept': sum(row['state_group'] == 'correct' and row['status'] == 'KEEP' for row in values),
                       'mae_mm': float(sum(errors, Q(0)) / n), 'mse_mm2': float(sum((e * e for e in errors), Q(0)) / n),
                       'initial_mae_mm': float(sum(initial, Q(0)) / n),
                       'initial_mse_mm2': float(sum((e * e for e in initial), Q(0)) / n), 'max_error_mm': float(max(errors)),
                       'mean_hull_width_mm': None if not widths else float(sum(widths, Q(0)) / len(widths)),
                       'evaluated_boxes': sum(row['evaluated_boxes'] for row in unique.values()),
                       'unresolved_budget_leaves': sum(row['unresolved_budget_leaves'] for row in unique.values()),
                       'auxiliary_samples_used': sum(row['auxiliary_samples_used'] for row in unique.values())})
    return result


def reconstruct_aggregates(rows):
    result = base_aggregates(rows)
    for seed in sorted({row['seed'] for row in rows}):
        subset = [row for row in rows if row['seed'] == seed]
        result.extend({**group, 'grouping': 'by_seed', 'condition': str(seed)}
                      for group in base_aggregates(subset) if group['grouping'] == 'overall')
        for case in sorted({row['case'] for row in subset}):
            result.extend({**group, 'grouping': 'by_case_seed', 'condition': [case, seed]}
                          for group in base_aggregates([row for row in subset if row['case'] == case]) if group['grouping'] == 'overall')
    return result


def paired_relations(rows):
    tasks = defaultdict(dict)
    for row in rows:
        tasks[row['task_id']][row['method']] = row
    grouped = defaultdict(list)
    for arms in tasks.values():
        shared = arms['shared_union']
        require('channelwise_union' in arms, 'paired metrics: absent comparator')
        for information in ('overall', 'negative_control' if shared['case'] == 'equal_full_negative' else 'informative'):
            for state in ('all', shared['state_group']):
                grouped[information, state].append(arms)
    result = []
    for (information, state), values in sorted(grouped.items()):
        shared = [exact(arms['shared_union']['ae_mm']) for arms in values]
        channel = [exact(arms['channelwise_union']['ae_mm']) for arms in values]
        result.append({'information': information, 'state_group': state, 'n': len(values),
                       'shared_lower_ae': sum(a < b for a, b in zip(shared, channel)),
                       'shared_same_ae': sum(a == b for a, b in zip(shared, channel)),
                       'shared_higher_ae': sum(a > b for a, b in zip(shared, channel)),
                       'shared_mae_mm': str(sum(shared, Q(0)) / len(values)),
                       'channelwise_mae_mm': str(sum(channel, Q(0)) / len(values)),
                       'shared_mse_mm2': str(sum((e * e for e in shared), Q(0)) / len(values)),
                       'channelwise_mse_mm2': str(sum((e * e for e in channel), Q(0)) / len(values))})
    return result


def verify_csv(path, records, json_columns=()):
    require(records, f'{path.name}: no records')
    with path.open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == list(records[0]), f'{path.name}: changed CSV header order')
        actual = list(reader)
    expected = [{key: json.dumps(value) if key in json_columns or (key == 'condition' and isinstance(value, list))
                 else '' if value is None else str(value) for key, value in row.items()} for row in records]
    same(actual, expected, str(path))


def verify_population(observations, truths, tasks, decisions, outers):
    for records, expected, name in ((observations, 54, 'observations'), (truths, 54, 'truths'), (tasks, 486, 'tasks'),
                                    (decisions, 1944, 'decisions'), (outers, 54, 'outers')):
        require(isinstance(records, list) and len(records) == expected, f'population: expected {expected} nonempty {name}')
    return (indexed(observations, lambda row: row['observation_id'], 'observations'),
            indexed(truths, lambda row: row['observation_id'], 'truths'), indexed(tasks, lambda row: row['task_id'], 'tasks'),
            indexed(decisions, lambda row: (row['task_id'], row['method']), 'decisions'),
            indexed(outers, lambda row: row['observation_id'], 'outers'))


def verify_timing(output, outers, summary):
    timing = load(output / 'TIMING.json')
    require(set(timing) == {'generation_seconds', 'observations', 'shared_cost_note'}, 'TIMING: invalid schema')
    generation = finite_seconds(timing['generation_seconds'], 'generation time')
    same(timing['shared_cost_note'], 'fixed execution order and shared geometry cache; recorded arm timing counts all branches and final hybrid actions', 'TIMING scope')
    timings = indexed(timing['observations'], lambda row: row['observation_id'], 'TIMING')
    require(list(timings) == list(outers), 'TIMING: changed observation population/order')
    same(load(output / 'TIMING_PARTIAL.json'), timing['observations'], 'TIMING final checkpoint')
    for oid, record in timings.items():
        require(set(record) == {'observation_id', 'outer_seconds_by_method', 'selector_seconds_by_method'}, f'{oid}: timing schema')
        require(set(record['outer_seconds_by_method']) == set(record['selector_seconds_by_method']) == set(METHODS), f'{oid}: timing method set')
        for method in METHODS:
            finite_seconds(record['selector_seconds_by_method'][method], f'{oid}/{method}: action time')
            measured = finite_seconds(record['outer_seconds_by_method'][method], f'{oid}/{method}: solve time')
            require(measured == outers[oid]['arms'][method]['outer_seconds'], f'{oid}/{method}: inconsistent solve time')
    per_method = {}
    for method in METHODS:
        outer = sum(row['outer_seconds_by_method'][method] for row in timing['observations'])
        action = sum(row['selector_seconds_by_method'][method] for row in timing['observations'])
        per_method[method] = {'outer_seconds': outer, 'selector_seconds': action, 'total_seconds': outer + action,
                              'evaluated_boxes': sum(row['arms'][method]['outer']['evaluated_boxes'] for row in outers.values()),
                              'unresolved_budget_leaves': sum(row['arms'][method]['outer']['unresolved_budget_leaves'] for row in outers.values())}
    same(summary['per_method_timing'], per_method, 'SUMMARY independently summed method costs')
    require(summary['generation_seconds'] == generation, 'SUMMARY: generation time differs')
    verify_csv(output / 'evaluation/TIMING.csv', [{'method': m, **v} for m, v in per_method.items()])
    return generation


def primary_claims(rows, unions, branches, truths):
    safety = {}
    for method in METHODS:
        values = [row for row in rows if row['method'] == method]
        coverage = sum(contains(unions[oid][method]['intervals'], exact(private['truth'])) for oid, private in truths.items())
        correct = [row for row in values if row['state_group'] == 'correct']
        harm = [row['task_id'] for row in values if exact(row['ae_mm']) > exact(row['initial_ae_mm'])]
        require(coverage == 54 and len(correct) == 54 and all(row['status'] == 'KEEP' and exact(row['ae_mm']) == 0 for row in correct) and not harm,
                f'C1: {method} failed registered true coverage/correct KEEP/zero harm')
        negative = [row for row in values if row['case'] == 'equal_full_negative']
        require(len(negative) == 81 and all(row['status'] == 'KEEP' for row in negative), f'negative control: {method} moved')
        for oid, private in truths.items():
            if private['case'] == 'equal_full_negative':
                require(unions[oid][method]['intervals'] == [DOMAIN], f'negative control: {method} lost full domain')
                if method != 'keep':
                    require(_arithmetic.merged_intervals(branches[oid][method]['full']['intervals']) == [DOMAIN],
                            f'negative control: {method} full branch itself lost domain')
        safety[method] = {'coverage': coverage, 'correct_kept': len(correct), 'harm': len(harm), 'negative_small_kept': 72}
    comparison = {}
    for method in ('channelwise_union', 'shared_union'):
        values = [row for row in rows if row['method'] == method and row['state_group'] == 'small' and row['case'] != 'equal_full_negative']
        require(len(values) == 360, f'C2: {method} has wrong informative-small denominator')
        comparison[method] = {'n': 360, 'improved': sum(exact(row['ae_mm']) < exact(row['initial_ae_mm']) for row in values),
                              'mae_exact_mm': str(sum((exact(row['ae_mm']) for row in values), Q(0)) / len(values))}
    shared, channel = comparison['shared_union'], comparison['channelwise_union']
    supported = shared['improved'] > channel['improved'] and Q(shared['mae_exact_mm']) < Q(channel['mae_exact_mm'])
    return {'C1': {'status': 'PASS', 'arms': safety}, 'C2': {'status': 'SUPPORTED' if supported else 'NEGATIVE',
            'informative_small': comparison, 'negative_result_is_not_verification_failure': True}}


def verify(output):
    output = Path(output).resolve()
    expected_root = Path('/srv/slam-research/grf/map-denoise/runs/lovelace-shared-appearance-20261011T052300Z')
    require(socket.gethostname() == 'liekkas' and ROOT == expected_root and output.is_relative_to(ROOT) and output != ROOT and output.is_dir(),
            'target identity: existing output inside the exact experiment on liekkas required')
    sources = verify_source_lock(load(output / 'SOURCE_LOCK.json'))
    same(load(output / 'PROTOCOL.json'), registration(), 'independent registered protocol snapshot')
    status = load(output / 'RUN_STATUS.json')
    require(set(status) == {'status', 'observations_complete', 'decisions_complete', 'elapsed_seconds'}, 'RUN_STATUS: invalid schema')
    same({k: status[k] for k in ('status', 'observations_complete', 'decisions_complete')},
         {'status': 'COMPLETE', 'observations_complete': 54, 'decisions_complete': 1944}, 'RUN_STATUS completed population')
    require(finite_seconds(status['elapsed_seconds'], 'RUN_STATUS elapsed') <= 1800, 'RUN_STATUS: registered budget exceeded')
    observation_list, truth_list = load(output / 'observations.json'), load(output / 'evaluation/truth.json')
    task_list, decision_list, outer_list = load(output / 'tasks.json'), load(output / 'decisions.json'), load(output / 'outers.json')
    observations, truths, tasks, decisions, outers = verify_population(observation_list, truth_list, task_list, decision_list, outer_list)
    ids = [f'o{i:03d}' for i in range(54)]
    require(list(observations) == list(truths) == list(outers) == ids, 'population: changed observation IDs/order')
    model_class, geometry_bounds = shared_kernels()
    expected_tasks, expected_decisions = [], []
    index = 0
    for case, branch, colors in CASES:
        for truth in TRUTHS:
            for seed in SEEDS:
                oid = ids[index]
                public, private = regenerate(case, branch, colors, truth, seed, model_class)
                same(observations[oid], {'observation_id': oid, **public}, f'{oid}: regenerated observed RGB and samples')
                same(truths[oid], {'observation_id': oid, **private}, f'{oid}: regenerated truth/random tape')
                for j, offset in enumerate(OFFSETS):
                    tid = f'{oid}-s{j}'
                    expected_tasks.append({'task_id': tid, 'observation_id': oid, 'incumbent': encoded(truth + offset)})
                    expected_decisions.extend((tid, method) for method in METHODS)
                index += 1
    same(task_list, expected_tasks, 'all registered initial task offsets')
    require(list(decisions) == expected_decisions, 'decisions: changed complete task/arm population or order')
    unions, branches = {}, {}
    calls = channel_witnesses = shared_certificates = proof_updates = leaves = 0
    for oid in ids:
        public, record = observations[oid], outers[oid]
        require(set(record) == {'observation_id', 'arms'} and list(record['arms']) == list(METHODS), f'{oid}: wrong arm schema/order')
        image = tuple(exact(v) for v in public['observation'])
        contracts = arm_contracts(public['calibration'])
        unions[oid], branches[oid] = {}, {}
        for method in METHODS:
            arm, where = record['arms'][method], f'{oid}/{method}'
            require(set(arm) == {'branches', 'outer', 'branch_seconds', 'outer_seconds'}, f'{where}: invalid arm schema')
            allowed, colors = contracts[method]
            require(list(arm['branches']) == list(arm['branch_seconds']) == list(allowed), f'{where}: changed whole-scene branch/material contract')
            total = finite_seconds(arm['outer_seconds'], where + ': outer time')
            measured = sum(finite_seconds(value, where + ': branch time') for value in arm['branch_seconds'].values())
            require(total >= measured and (method != 'keep' or total == 0), f'{where}: inconsistent solve cost')
            checked_branches = {}
            for branch in allowed:
                outer = arm['branches'][branch]
                checked = (verify_shared_trace(outer, image, branch, geometry_bounds, where + '/' + branch) if method == 'shared_union'
                           else _arithmetic.verify_trace(outer, image, where + '/' + branch))
                model = model_class(branch, camera_radius=RADIUS, colors=colors)
                n_calls, n_witnesses = verify_kernel_boxes(outer, checked, image, model, where + '/' + branch, method == 'shared_union')
                calls += n_calls
                channel_witnesses += n_witnesses
                shared_certificates += len(checked.get('shared_excluded', ()))
                proof_updates += checked.get('proof_updates', 0)
                leaves += sum(node['status'] != 'split' for node in outer['trace'])
                checked_branches[branch] = checked
            branches[oid][method] = checked_branches
            unions[oid][method] = _arithmetic.verify_union(arm['outer'], checked_branches, where, prior_only=method == 'keep')
    rows = []
    for decision in decision_list:
        task, method = tasks[decision['task_id']], decision['method']
        oid = task['observation_id']
        checked = unions[oid][method]
        result = _arithmetic.verify_decision(decision, task['incumbent'], checked['hull'], f'{task["task_id"]}/{method}')
        row = make_row(task, truths[oid], decision, checked)
        if method != 'keep' and checked['hull'] is not None and row['covered']:
            gain = (exact(task['incumbent']) - exact(truths[oid]['truth'])) ** 2 - (result - exact(truths[oid]['truth'])) ** 2
            require(gain >= exact(decision['gain_lower']) >= 0, f'{task["task_id"]}/{method}: actual gain below certificate')
        rows.append(row)
    same(load(output / 'evaluation/ROWS.json'), rows, 'independent exact evaluation ROWS')
    verify_csv(output / 'evaluation/ROWS.csv', rows)
    coverage = []
    for oid in ids:
        private, truth = truths[oid], exact(truths[oid]['truth'])
        for method in METHODS:
            checked, branch_checks = unions[oid][method], branches[oid][method]
            hull = checked['hull']
            coverage.append({'observation_id': oid, 'case': private['case'], 'seed': private['seed'], 'method': method,
                             'covered': contains(checked['intervals'], truth), 'incompatible': hull is None,
                             'hull_width_mm': None if hull is None else str(hull[1] - hull[0]),
                             'retained_branches': checked['retained_branches'],
                             'branch_coverage': {name: contains(branch['intervals'], truth) for name, branch in branch_checks.items()}})
    same(load(output / 'evaluation/COVERAGE.json'), coverage, 'whole-branch true retained membership COVERAGE')
    verify_csv(output / 'evaluation/COVERAGE.csv', coverage, ('retained_branches', 'branch_coverage'))
    summary = load(output / 'evaluation/SUMMARY.json')
    fields = {'observation_configurations', 'template_count', 'task_count', 'decision_count', 'rows', 'paired_relations', 'sampling_structure',
              'sampling_note', 'hybrid_branch_counts', 'sample_counts', 'information_note', 'per_method_timing', 'generation_seconds', 'elapsed_seconds', 'timing_scope'}
    require(isinstance(summary, dict) and set(summary) == fields, 'SUMMARY: invalid complete schema')
    for field, value in {'observation_configurations': 54, 'template_count': 6, 'task_count': 486, 'decision_count': 1944}.items():
        same(summary[field], value, f'SUMMARY {field}')
    aggregates = reconstruct_aggregates(rows)
    require(len(aggregates) == 360, 'SUMMARY: wrong overall/case/information/seed/case-seed denominator')
    same(summary['rows'], aggregates, 'SUMMARY all independent grouped counts/errors/coverage/costs')
    verify_csv(output / 'evaluation/SUMMARY.csv', aggregates)
    paired = paired_relations(rows)
    require(len(paired) == 9, 'PAIRED: wrong information/state denominator')
    same(summary['paired_relations'], paired, 'SUMMARY exact paired shared/channelwise error relations')
    verify_csv(output / 'evaluation/PAIRED.csv', paired)
    same(summary['hybrid_branch_counts'], {m: dict(sorted(Counter(d.get('branch', 'identity') for d in decision_list if d['method'] == m).items())) for m in METHODS}, 'SUMMARY hybrid actions')
    same(summary['sampling_structure'], {'case_templates': 6, 'depth_settings': 3, 'noise_replicates_per_case_depth': 3,
         'shared_incumbents_per_observation': 9}, 'SUMMARY sampling structure')
    same(summary['sampling_note'], 'synthetic configurations with repeated noise seeds; initial states and arms share observations; not independent real scenes', 'SUMMARY sampling note')
    same(summary['sample_counts'], {'image_configurations': 54, 'image_channels_per_observation': 432,
         'auxiliary_samples_per_observation': 3, 'auxiliary_samples_total': 162,
         'auxiliary_samples_used_by_method': {m: 162 if m == 'calibrated_union' else 0 for m in METHODS}}, 'SUMMARY additional information accounting')
    same(summary['information_note'], 'only channelwise_union versus shared_union is a same-information algorithm comparison; calibrated_union consumes additional sourced samples', 'SUMMARY information note')
    same(summary['timing_scope'], 'observed shared-geometry-cache execution order; every arm includes all branch solves and actions; not standalone cold-cache timing', 'SUMMARY timing scope')
    elapsed = finite_seconds(summary['elapsed_seconds'], 'SUMMARY elapsed')
    generation = verify_timing(output, outers, summary)
    require(generation <= elapsed <= status['elapsed_seconds'], 'TIMING: inconsistent generation/evaluation/complete sequence')
    claims = primary_claims(rows, unions, branches, truths)
    same(scientific_files(), sources, 'source inventory unchanged throughout verification')
    return {'status': 'PASS', 'host': socket.gethostname(), 'output': str(output), 'observations': 54, 'tasks': 486, 'decisions': 1944,
            'summary_rows': len(aggregates), 'paired_rows': len(paired), 'branch_outer_traces': 486, 'terminal_partition_leaves': leaves,
            'all_shared_exclusion_certificates_verified': shared_certificates, 'exact_propagation_updates_verified': proof_updates,
            'shared_kernel_evaluated_boxes_recomputed': calls, 'all_channel_witnesses_recomputed': channel_witnesses,
            'source_files_verified': len(sources), 'claims': claims,
            'checks': ['exact source lock, registration and complete population', 'all regenerated images/poses/RGB noise/sourced calibration',
                       'all shared exclusion rows and exact eight-round propagation proof steps', 'all branch trees, BFS budgets and full-domain terminal partitions',
                       'all channel witnesses against regenerated source enclosures', 'whole-scene retained depth unions, hulls and true membership',
                       'all frozen hybrid endpoint actions, errors and state/prior labels', 'all ROWS/COVERAGE and grouped/paired SUMMARY metrics',
                       'all CSV counterparts and measured timing accounting', 'all contract arms true coverage/correct KEEP/zero harm and full negative control'],
            'limitations': ['Historical point/enclosure/area kernels are shared; this verifier does not independently prove their geometry soundness.',
                            'Exact exclusion proof sequences are replayed; nonexcluded shared nodes are checked structurally and against channel bounds, without reproducing unrecorded no-contradiction propagation.',
                            'Recorded times are checked and summed, not independently reproduced; costs use a shared geometry cache.',
                            'Function separation and source hashes do not establish operating-system blindness or independent historical timestamps.',
                            'Finite propagation proves necessary-constraint inconsistency only; retained boxes are not guaranteed jointly feasible.',
                            'Synthetic templates/repeated seeds are not independent real scenes or evidence of real-MVS transfer.']}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Independently verify frozen shared-appearance evidence without running confirmation.')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/confirmation-v1')
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if not output.is_dir() or not output.is_relative_to(ROOT) or output == ROOT:
        parser.error('existing output directory inside this exact experiment required')
    destination = output / 'verification/VERIFY.json'
    if destination.exists():
        print('REFUSE: existing verification/VERIFY.json is never overwritten', file=sys.stderr)
        return 2
    started = time.perf_counter()
    try:
        report = verify(output)
    except Exception as error:
        report = {'status': 'FAIL', 'host': socket.gethostname(), 'output': str(output), 'error': f'{type(error).__name__}: {error}'}
    report['utc'] = datetime.now(timezone.utc).isoformat()
    report['verification_seconds'] = time.perf_counter() - started
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
    except FileExistsError:
        print('REFUSE: concurrent verification created VERIFY.json', file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())

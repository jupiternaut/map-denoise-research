#!/usr/bin/env python3
"""Independent replay of the one-factor public-camera baseline intervention.

No new generator, runner, geometry provider or solver is imported. Rectangle
projection bounds and point images are reconstructed below; only historical
rectangle area primitives and the prior independently authored row/tree/action
verification arithmetic are reused. Importing this module does not generate
the registered seed or read any private labels.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
from fractions import Fraction as Q
from functools import lru_cache
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import signal
import socket
import sys
import time

sys.dont_write_bytecode = True
for _baseline_verifier_thread in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_baseline_verifier_thread] = '1'

ROOT = Path(__file__).resolve().parent
EXPECTED_ROOT = Path('/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/recovery-1')
PARENT = ROOT.parent
ORIGINAL_OUTPUT = PARENT / 'outputs/baseline-v1'
FROZEN = ROOT.parent.parent / 'lovelace-structured-view-bias-20261012T022500Z'
CHALLENGE = ROOT.parent.parent / 'lovelace-challenge-20261010T205013Z'
DATA = ROOT.parent / 'data'
_spec = importlib.util.spec_from_file_location('_baseline_independent_prior', FROZEN / 'verify_structured.py')
_prior = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prior)
_historical, _arithmetic = _prior._historical, _prior._arithmetic
VerificationError = _prior.VerificationError
require, same, exact, encoded, pair = _prior.require, _prior.same, _prior.exact, _prior.encoded, _prior.pair
load, indexed, finite_seconds = _prior.load, _prior.indexed, _prior.finite_seconds
DOMAIN, TOLERANCE = _prior.DOMAIN, _prior.TOLERANCE
RADIUS, EPSILON, BETA = Q(1, 5), Q(1, 100), Q(1, 100)
BRANCHES, METHODS = ('none', 'partial', 'full'), ('keep', 'calibrated', 'joint')
BASELINES, UNITS = (Q(70), Q(80)), ('baseline_70', 'baseline_80')
SEED, PREFIX, LIMIT = 61005, 'shared-appearance-v1', 1800
ROUNDS, MAX_BOXES = 8, 511
GENERATED_FILES = ('observations.json', 'tasks.json', 'private/truth.json', 'GENERATION.json')
SAMPLING_NOTE = 'one paired noise/pose realization per template/depth; controlled synthetic imaging intervention, not real transfer'
TIMING_SCOPE = 'new cold solver includes background preparation; no historical actions or timing reused'
LEFT = (Q(-20), Q(0), Q(-15), Q(15))
RIGHT = (Q(0), Q(20), Q(-15), Q(15))
FRONT = (Q(-20), Q(20), Q(-15), Q(15))
REAR = (Q(-20), Q(36), Q(-17), Q(23))
HIDDEN = (Q(-100), Q(100), Q(-80), Q(80))
REAR_DEPTH = Q(720)

# These primitives are historical dependencies, not the new geometry provider.
sys.path.insert(0, str(CHALLENGE))
from layered_model import rectangle_area, rectangle_difference_area

PIXELS = tuple((Q(-1, 5) + Q(column, 20), Q(-1, 5) + Q(column + 1, 20),
                Q(-2, 25) + Q(2 * row, 75), Q(-2, 25) + Q(2 * (row + 1), 75))
               for row in range(6) for column in range(8))


def _quotient_interval(numerator, denominator):
    require(denominator[0] > 0, 'geometry: nonpositive depth denominator')
    values = [n / z for n in numerator for z in denominator]
    return min(values), max(values)


def _project_bounds(rectangle, depth, camera):
    """Direct edge extrema, independent of historical Interval projections."""
    left, right, bottom, top = rectangle
    bounds = (_quotient_interval((left - camera[1], left - camera[0]), depth),
              _quotient_interval((right - camera[1], right - camera[0]), depth),
              _quotient_interval((bottom, bottom), depth),
              _quotient_interval((top, top), depth))
    inner = (bounds[0][1], bounds[1][0], bounds[2][1], bounds[3][0])
    if inner[0] >= inner[1] or inner[2] >= inner[3]:
        inner = None
    outer = (bounds[0][0], bounds[1][1], bounds[2][0], bounds[3][1])
    return inner, outer


def _project_point(rectangle, depth, camera):
    return ((rectangle[0] - camera) / depth, (rectangle[1] - camera) / depth,
            rectangle[2] / depth, rectangle[3] / depth)


@lru_cache(maxsize=8192)
def independent_geometry(branch, radius, low, high, baseline):
    radius, low, high, baseline = map(exact, (radius, low, high, baseline))
    require(branch in BRANCHES and radius >= 0 and baseline in BASELINES and
            DOMAIN[0] <= low <= high <= DOMAIN[1], 'geometry: invalid public hypothesis')
    rear = None if branch == 'none' else REAR if branch == 'partial' else HIDDEN
    result = []
    for index, center in enumerate((-baseline, Q(0), baseline)):
        error = Q(0) if index == 1 else radius
        camera = (center - error, center + error)
        left, right, front = (_project_bounds(rect, (low, high), camera) for rect in (LEFT, RIGHT, FRONT))
        back = (None, None) if rear is None else _project_bounds(rear, (REAR_DEPTH, REAR_DEPTH), camera)
        for pixel in PIXELS:
            area = rectangle_area(pixel)
            values = (rectangle_area(left[0], pixel) / area, rectangle_area(left[1], pixel) / area,
                      rectangle_area(right[0], pixel) / area, rectangle_area(right[1], pixel) / area,
                      rectangle_difference_area(back[0], front[1], pixel) / area,
                      rectangle_difference_area(back[1], front[0], pixel) / area)
            require(all(0 <= values[k] <= values[k + 1] <= 1 for k in (0, 2, 4)),
                    'geometry: invalid independent visible-area intervals')
            result.append(values)
    return tuple(result)


def independent_point_areas(branch, depth, offsets, baseline):
    depth, baseline = exact(depth), exact(baseline)
    offsets = tuple(map(exact, offsets))
    require(branch in BRANCHES and baseline in BASELINES and DOMAIN[0] <= depth <= DOMAIN[1] and
            len(offsets) == 3 and offsets[1] == 0 and all(abs(e) <= RADIUS for e in offsets),
            'point geometry: invalid witness')
    rear = None if branch == 'none' else REAR if branch == 'partial' else HIDDEN
    result = []
    for center, offset in zip((-baseline, Q(0), baseline), offsets):
        camera = center + offset
        left, right, front = (_project_point(rect, depth, camera) for rect in (LEFT, RIGHT, FRONT))
        back = None if rear is None else _project_point(rear, REAR_DEPTH, camera)
        for pixel in PIXELS:
            area = rectangle_area(pixel)
            result.append((rectangle_area(left, pixel) / area, rectangle_area(right, pixel) / area,
                           rectangle_difference_area(back, front, pixel) / area))
    return tuple(result)


def independent_rgb(branch, depth, offsets, colors, baseline):
    colors = tuple(map(exact, colors))
    require(len(colors) == 3 and all(0 <= c <= 1 for c in colors), 'point image: invalid material witness')
    result = []
    for areas in independent_point_areas(branch, depth, offsets, baseline):
        gray = 1 - sum((area * (1 - color) for area, color in zip(areas, colors)), Q(0))
        require(0 <= gray <= 1, 'point image: illegal exact grayscale')
        result.extend((gray,) * 3)
    return tuple(result)


def prove_full_negative_control(baseline, radius=RADIUS):
    """Global projected containment, not sampling the generating depth."""
    baseline, radius = exact(baseline), exact(radius)
    require(baseline in BASELINES and radius >= 0, 'negative control: invalid public camera')
    largest_camera = baseline + radius
    # All edge inequalities attain their worst case at z_min and |c|_max.
    horizontal = 100 * DOMAIN[0] - 14400 - largest_camera * (REAR_DEPTH - DOMAIN[0])
    vertical = 80 * DOMAIN[0] - 15 * REAR_DEPTH
    require(horizontal >= 0 and vertical >= 0 and largest_camera < 100,
            'negative control: projected front is not globally inside hidden rear')
    return {'baseline_mm': str(baseline), 'horizontal_containment_margin_scaled': str(horizontal),
            'vertical_containment_margin_scaled': str(vertical), 'full_domain_containment': True}


def independent_tape(case, truth, seed):
    """One tape; camera condition never enters its seed or draw sequence."""
    tape = f'{PREFIX}:{case}:{RADIUS}:{exact(truth)}:{seed}'
    rng = random.Random(tape)
    offsets = (RADIUS * rng.choice(_historical.WEIGHTS), Q(0), RADIUS * rng.choice(_historical.WEIGHTS))
    noise = tuple(EPSILON * rng.choice(_historical.WEIGHTS) for _ in range(432))
    return tape, offsets, noise


def check_target(output=None):
    require(socket.gethostname() == 'liekkas' and ROOT == EXPECTED_ROOT and ROOT.resolve() == EXPECTED_ROOT,
            'target: exact baseline intervention root on liekkas required')
    if output is not None:
        output = Path(output).resolve()
        require(output.is_dir() and output != ROOT and output.is_relative_to(ROOT),
                'target: existing output strictly inside exact new root required')
    return output


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scientific_files():
    files = dict(_prior.scientific_files())
    helper = PARENT.parent / 'lovelace-structured-bias-confirmation-20261012T035916Z/run_confirmation.py'
    # Independently enumerate the original graph, rather than accepting an
    # arbitrary recovery inventory supplied by the runner.
    for path in sorted({PARENT / 'AGENTS.md', PARENT / 'PROTOCOL.md', PARENT / 'REGISTRATION.json', *PARENT.glob('*.py')}):
        files[str(path.resolve())] = sha(path)
    files[str(helper.resolve())] = sha(helper)
    original_lock = load(ORIGINAL_OUTPUT / 'SOURCE_LOCK.json')
    require(isinstance(original_lock, dict) and set(original_lock) == {'source_root', 'files', 'utc'} and
            original_lock['source_root'] == str(PARENT), 'recovery: changed original source-lock identity/schema')
    same(original_lock['files'], files, 'recovery: complete frozen original failed-attempt source graph unchanged')
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
    files = scientific_files()
    same(lock['files'], files, 'SOURCE_LOCK: complete independently enumerated frozen/new byte graph')
    return files


def verify_budget(budget):
    require(isinstance(budget, dict) and set(budget) == {'start_epoch', 'deadline_epoch', 'limit_seconds', 'scope'},
            'BUDGET: invalid shared window schema')
    start, end = (finite_seconds(budget[key], f'BUDGET {key}') for key in ('start_epoch', 'deadline_epoch'))
    require(type(budget['limit_seconds']) is int and budget['limit_seconds'] == LIMIT and end == start + LIMIT and
            budget['scope'] == 'one main run plus complete verification; do not reset',
            'BUDGET: changed original absolute window')
    same(budget, load(ORIGINAL_OUTPUT / 'BUDGET.json'), 'recovery: original parent absolute budget reused byte-for-value')
    if time.time() >= end:
        raise TimeoutError('shared main plus verification deadline exhausted')
    return end


@contextmanager
def deadline_guard(output):
    deadline = verify_budget(load(Path(output) / 'BUDGET.json'))
    require(not signal.getitimer(signal.ITIMER_REAL)[0], 'BUDGET: existing deadline cannot be replaced')
    def reached(_signal, _frame):
        raise TimeoutError('shared main plus verification deadline reached')
    previous = signal.signal(signal.SIGALRM, reached)
    signal.setitimer(signal.ITIMER_REAL, max(0.000001, deadline - time.time()))
    try:
        yield deadline
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def registration():
    return {'seed': SEED, 'prefix': PREFIX, 'baselines_mm': list(map(str, BASELINES)),
            'radius_mm': str(RADIUS), 'epsilon': str(EPSILON), 'beta': str(BETA),
            'rounds': ROUNDS, 'max_boxes_per_branch': MAX_BOXES, 'tolerance_mm': str(TOLERANCE),
            'domain_mm': list(map(str, DOMAIN)), 'observations': 36, 'tasks': 324, 'decisions': 972,
            'cold_solver_calls': 72, 'methods': list(METHODS), 'offsets_mm': list(map(str, _historical.OFFSETS)),
            'source_root': str(ROOT), 'paired_noise': True, 'deadline_seconds': LIMIT,
            'main_target': 'deepest informative shifted; all conditions reported', 'negative_results_preserved': True}


def verify_inputs(lock):
    require(isinstance(lock, dict) and set(lock) == {'source_root', 'files'} and lock['source_root'] == str(DATA),
            'INPUTS: exact new generated data required')
    files = generated_inputs()
    same(lock['files'], files, 'INPUTS: complete four generated data byte hashes')
    same(lock, load(ORIGINAL_OUTPUT / 'INPUTS.json'),
         'recovery: exact original sealed data identity/hashes, without regeneration')
    generation = load(DATA / 'GENERATION.json')
    expected = {'seed': SEED, 'prefix': PREFIX, 'observations': 36, 'tasks': 324, 'truths': 36,
                'transform': 'same pose/noise tape; explicit baseline point image + noise + fixed view bias; clamp once',
                'auxiliary_samples_used': 0, 'paired_noise': True, 'baselines_mm': list(map(str, BASELINES))}
    require(isinstance(generation, dict) and set(generation) == set(expected) | {'seconds'},
            'GENERATION: changed generation contract/schema')
    same({key: value for key, value in generation.items() if key != 'seconds'}, expected,
         'GENERATION: fixed paired-tape seed/baselines/clip-once/no extra calibration')
    finite_seconds(generation['seconds'], 'GENERATION seconds')
    return files


def verify_public_population(output):
    public, tasks = load(DATA / 'observations.json'), load(DATA / 'tasks.json')
    require(isinstance(public, list) and len(public) == 36 and isinstance(tasks, list) and len(tasks) == 324,
            'population: complete36 images/324 tasks required')
    images = indexed(public, lambda row: row['observation_id'], 'public images')
    initial = indexed(tasks, lambda row: row['task_id'], 'public tasks')
    expected_ids = [f'p{index:03d}b{baseline}' for index in range(1, 19) for baseline in BASELINES]
    require(list(images) == expected_ids, 'public images: registered interleaved camera-condition identities/order')
    for index in range(1, 19):
        pid = f'p{index:03d}'
        for baseline, unit in zip(BASELINES, UNITS):
            oid = f'{pid}b{baseline}'
            row = images[oid]
            require(set(row) == {'observation_id', 'pair_id', 'unit_id', 'baseline', 'radius', 'clipped', 'observation'},
                    f'{oid}: leaked private/calibration field or changed public schema')
            same({key: row[key] for key in row if key != 'observation'},
                 {'observation_id': oid, 'pair_id': pid, 'unit_id': unit, 'baseline': str(baseline),
                  'radius': str(RADIUS), 'clipped': True}, f'{oid}: exact public metadata')
            image = tuple(map(exact, row['observation']))
            require(len(image) == 432 and all(0 <= value <= 1 for value in image), f'{oid}: illegal clip/layout')
    require(list(initial) == [f'{oid}s{state}' for oid in images for state in range(9)],
            'tasks: original nine incumbents per observation identity/order')
    for tid, task in initial.items():
        require(set(task) == {'task_id', 'base_task_id', 'observation_id', 'incumbent'} and
                task['observation_id'] in images, f'{tid}: changed task schema/identity')
        pid = images[task['observation_id']]['pair_id']
        state = tid.rsplit('s', 1)[1]
        require(task['base_task_id'] == f'{pid}s{state}', f'{tid}: false cross-baseline initial pair')
        exact(task['incumbent'])
    same(load(output / 'observations.json'), public, 'all36 public images saved/read back')
    same(load(output / 'tasks.json'), tasks, 'all324 public initial tasks saved/read back')
    return images, initial


def verify_action_population(decision_list, observations, tasks):
    require(isinstance(decision_list, list) and len(decision_list) == 972, 'actions: complete972 saved actions required')
    decisions = indexed(decision_list, lambda row: (row['task_id'], row['method']), 'saved actions')
    expected = [(f'{oid}s{state}', method) for oid in observations for method in METHODS for state in range(9)]
    require(list(decisions) == expected and set(tasks) == {tid for tid, _ in expected},
            'actions: missing/reordered/duplicate method or task')
    require(all(type(row.get('reused')) is bool for row in decision_list) and
            Counter((row['method'], row['reused']) for row in decision_list) ==
            {('keep', False): 324, ('calibrated', False): 324, ('joint', False): 324},
            'actions: false new action origins')
    return decisions


def verify_private_population(observations, tasks):
    """Only entered after every unique saved action passes its readback gate."""
    truths, expected_tasks, witnesses = [], [], []
    bias = (-BETA, Q(0), BETA)
    population = [(case, branch, colors, truth) for case, branch, colors in _historical.CASES
                  for truth in _historical.TRUTHS]
    for index, (case, branch, colors, truth) in enumerate(population, 1):
        pid = f'p{index:03d}'
        tape, offsets, noise = independent_tape(case, truth, SEED)
        for baseline, unit in zip(BASELINES, UNITS):
            oid = f'{pid}b{baseline}'
            prediction = independent_rgb(branch, truth, offsets, colors, baseline)
            image = tuple(min(Q(1), max(Q(0), value + error + bias[channel // 144]))
                          for channel, (value, error) in enumerate(zip(prediction, noise)))
            same(observations[oid]['observation'], list(map(encoded, image)),
                 f'{oid}: independent point projection + common noise + bias + single clip')
            private = {'case': case, 'rear_branch': branch, 'truth': encoded(truth),
                       'colors': list(map(encoded, colors)), 'seed': SEED, 'random_tape': tape,
                       'actual_offsets': list(map(encoded, offsets)), 'image_noise': list(map(encoded, noise)),
                       'noise_linf': encoded(max(map(abs, noise))), 'observation_id': oid, 'pair_id': pid,
                       'unit_id': unit, 'baseline': str(baseline), 'clipped': True,
                       'image_kind': 'biased_clip', 'actual_view_bias': list(map(encoded, bias))}
            truths.append(private)
            for state, offset in enumerate(_historical.OFFSETS):
                expected_tasks.append({'task_id': f'{oid}s{state}', 'base_task_id': f'{pid}s{state}',
                                       'observation_id': oid, 'incumbent': encoded(truth + offset)})
            darkness = tuple(1 - color for color in colors)
            areas = independent_point_areas(branch, truth, offsets, baseline)
            for pixel, (upper, lower) in enumerate(_prior.pixel_bands(image, True)):
                value = sum((a * d for a, d in zip(areas[pixel], darkness)), Q(0)) + BETA - bias[pixel // 48]
                require((upper is None or value <= upper) and (lower is None or value >= lower),
                        f'{oid}: actual latent witness violates declared clip observation model')
            witnesses.append({'observation_id': oid, 'unit_id': unit, 'baseline': str(baseline),
                              'base_noise_linf': private['noise_linf'], 'clip_aware_true_witness_valid': True})
    same(list(tasks.values()), expected_tasks, 'all324 exact original matched initial offsets')
    same(load(DATA / 'private/truth.json'), truths, 'all36 independently regenerated paired private facts')
    return indexed(truths, lambda row: row['observation_id'], 'private facts'), witnesses


def verify_csv(path, rows):
    """Check every field and ordered row; harmless header permutation is allowed."""
    require(rows, f'{path.name}: no registered rows')
    with path.open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames is not None and len(reader.fieldnames) == len(set(reader.fieldnames)) and
                set(reader.fieldnames) == set(rows[0]), f'{path.name}: missing/extra/duplicate columns')
        actual = list(reader)
    expected = [{key: json.dumps(value) if isinstance(value, (list, dict)) else
                 '' if value is None else str(value) for key, value in row.items()} for row in rows]
    same(actual, expected, f'{path.name}: all independently reconstructed ordered metric values')


def make_row(task, private, decision, checked):
    return {**_prior.make_row(task, private, decision, checked),
            'pair_id': private['pair_id'], 'baseline': private['baseline']}


def aggregate_rows(rows, truths):
    result = []
    for unit in UNITS:
        chosen = [row for row in rows if row['unit_id'] == unit]
        result.extend({**record, 'unit_id': unit} for record in _prior._old.reconstruct_aggregates(chosen))
        for depth in sorted(set(str(exact(private['truth'])) for private in truths.values()), key=Q):
            selected = [row for row in chosen if str(exact(truths[row['observation_id']]['truth'])) == depth]
            result.extend({**record, 'grouping': 'by_depth', 'condition': depth, 'unit_id': unit}
                          for record in _historical.base_aggregates(selected) if record['grouping'] == 'overall')
    return result


def paired_method_rows(rows):
    matrix = indexed(rows, lambda row: (row['task_id'], row['method']), 'method pair matrix')
    require(len(matrix) == 972, 'method pairs: incomplete metric matrix')
    result = []
    for unit in UNITS:
        for tid in sorted({row['task_id'] for row in rows if row['unit_id'] == unit}):
            a, b = matrix[tid, 'calibrated'], matrix[tid, 'joint']
            same({key: a[key] for key in ('case', 'seed', 'state_group', 'initial_ae_mm', 'unit_id')},
                 {key: b[key] for key in ('case', 'seed', 'state_group', 'initial_ae_mm', 'unit_id')}, 'method pair lineage')
            x, y = exact(a['ae_mm']), exact(b['ae_mm'])
            result.append({'unit_id': a['unit_id'], 'base_task_id': tid, 'case': a['case'], 'seed': a['seed'],
                           'state_group': a['state_group'], 'left_method': 'calibrated', 'right_method': 'joint',
                           'left_ae_mm': str(x), 'right_ae_mm': str(y), 'mae_gain_mm': str(x - y),
                           'mse_gain_mm2': str(x*x - y*y), 'better': y < x, 'same': y == x, 'worse': y > x})
    require(len(result) == 324, 'method pairs: changed324 denominator')
    return result


def paired_baseline_rows(rows):
    matrix = indexed(rows, lambda row: (row['unit_id'], row['base_task_id'], row['method']), 'baseline pair matrix')
    require(len(matrix) == 972, 'baseline pairs: incomplete metric matrix')
    result = []
    for tid in sorted({row['base_task_id'] for row in rows}):
        for method in METHODS:
            a, b = matrix[UNITS[0], tid, method], matrix[UNITS[1], tid, method]
            same({key: a[key] for key in ('case', 'seed', 'state_group', 'initial_ae_mm', 'incumbent')},
                 {key: b[key] for key in ('case', 'seed', 'state_group', 'initial_ae_mm', 'incumbent')}, 'baseline pair lineage')
            x, y = exact(a['ae_mm']), exact(b['ae_mm'])
            result.append({'base_task_id': tid, 'method': method, 'case': a['case'], 'seed': a['seed'],
                           'state_group': a['state_group'], 'left_unit_id': UNITS[0], 'right_unit_id': UNITS[1],
                           'left_ae_mm': str(x), 'right_ae_mm': str(y), 'mae_gain_mm': str(x - y),
                           'mse_gain_mm2': str(x*x - y*y), 'better': y < x, 'same': y == x, 'worse': y > x})
    require(len(result) == 486, 'baseline pairs: changed486 denominator')
    return result


def interval_subset(inner, outer):
    merged = _arithmetic.merged_intervals(outer)
    return all(any(a <= left <= right <= b for a, b in merged) for left, right in inner)


def paired_method_summary(rows, pairs, outers):
    result = {}
    matrix = indexed(rows, lambda row: (row['task_id'], row['method']), 'method summary matrix')
    for unit in UNITS:
        selected = [p for p in pairs if p['unit_id'] == unit]
        counts = {'total': len(selected), 'better': sum(p['better'] for p in selected),
                  'same': sum(p['same'] for p in selected), 'worse': sum(p['worse'] for p in selected),
                  'new_repairs': 0, 'lost_repairs': 0, 'strict_hull_subset': 0, 'equal_hull': 0,
                  'union_subset': 0, 'branch_subset': 0, 'calibration_equal': 0}
        for p in selected:
            initial = exact(matrix[p['base_task_id'], 'calibrated']['initial_ae_mm'])
            a, b = exact(p['left_ae_mm']), exact(p['right_ae_mm'])
            counts['new_repairs'] += int(a >= initial and b < initial)
            counts['lost_repairs'] += int(a < initial and b >= initial)
        for oid, arms in outers.items():
            if matrix[f'{oid}s0', 'keep']['unit_id'] != unit:
                continue
            a, b = arms['calibrated'], arms['joint']
            ha, hb = a['outer']['hull'], b['outer']['hull']
            nested = hb is None or (ha is not None and exact(ha[0]) <= exact(hb[0]) <= exact(hb[1]) <= exact(ha[1]))
            counts['strict_hull_subset'] += int(nested and ha != hb)
            counts['equal_hull'] += int(ha == hb)
            counts['calibration_equal'] += int(a['calibration_intervals'] == b['calibration_intervals'])
            counts['union_subset'] += int(interval_subset([pair(v, 'joint union') for v in b['outer']['retained_intervals']],
                                                         [pair(v, 'cal union') for v in a['outer']['retained_intervals']]))
            for branch in BRANCHES:
                ia = [(exact(n['lo']), exact(n['hi'])) for n in a['branches'][branch]['trace'] if n['id'] in a['branches'][branch]['retained_ids']]
                ib = [(exact(n['lo']), exact(n['hi'])) for n in b['branches'][branch]['trace'] if n['id'] in b['branches'][branch]['retained_ids']]
                counts['branch_subset'] += int(interval_subset(ib, ia))
        result[unit] = counts
    return result


def safety_claims(rows, coverage):
    result = {}
    for unit in UNITS:
        result[unit] = {}
        for method in METHODS:
            values = [r for r in rows if r['unit_id'] == unit and r['method'] == method]
            cov = [r for r in coverage if r['unit_id'] == unit and r['method'] == method]
            correct = [r for r in values if r['state_group'] == 'correct']
            negative = [r for r in values if r['case'] == 'equal_full_negative' and r['state_group'] == 'small']
            require(tuple(map(len, (values, cov, correct, negative))) == (162, 18, 18, 24),
                    f'{unit}/{method}: changed scientific denominators')
            result[unit][method] = {'covered': sum(r['covered'] for r in cov),
                'generating_branch_covered': sum(r['generating_branch_covered'] for r in cov),
                'correct_kept': sum(r['status'] == 'KEEP' and exact(r['ae_mm']) == 0 for r in correct),
                'negative_small_kept': sum(r['status'] == 'KEEP' for r in negative),
                'harm': sum(exact(r['ae_mm']) > exact(r['initial_ae_mm']) for r in values),
                'kept': sum(r['status'] == 'KEEP' for r in values),
                'incompatible': sum(r['status'] == 'INCOMPATIBLE' for r in values),
                'latent_material_covered': None if method == 'keep' else sum(r['latent_material_covered'] for r in cov),
                'latent_bias_covered': None if method != 'joint' else sum(r['latent_bias_covered'] for r in cov)}
    return result


def paired_baseline_summary(rows, pairs, truths):
    matrix = indexed(rows, lambda row: (row['unit_id'], row['base_task_id'], row['method']), 'baseline summary matrix')
    def counts(values):
        answer = {'total': len(values), 'better': sum(p['better'] for p in values),
                  'same': sum(p['same'] for p in values), 'worse': sum(p['worse'] for p in values),
                  'new_repairs': 0, 'lost_repairs': 0}
        for p in values:
            before = exact(matrix[UNITS[0], p['base_task_id'], p['method']]['initial_ae_mm'])
            a, b = exact(p['left_ae_mm']), exact(p['right_ae_mm'])
            answer['new_repairs'] += int(a >= before and b < before)
            answer['lost_repairs'] += int(a < before and b >= before)
        return answer
    result = counts(pairs)
    result['by_method'] = {m: counts([p for p in pairs if p['method'] == m]) for m in METHODS}
    depths = {r['base_task_id']: exact(truths[r['observation_id']]['truth']) for r in rows}
    deepest = max(depths.values())
    deep = [p for p in pairs if depths[p['base_task_id']] == deepest and
            p['case'] != 'equal_full_negative' and p['state_group'] == 'small']
    result['deep_informative'] = {m: counts([p for p in deep if p['method'] == m]) for m in METHODS}
    require(all(value['total'] == 40 for value in result['deep_informative'].values()),
            'deep primary endpoint: all five informative templates/eight shifted states required')
    return result


def audit_runner_source(source=None):
    code = (ROOT / 'run_baseline.py').read_text(encoding='utf-8') if source is None else source
    tree = ast.parse(code)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    require({'_run', 'private_truths', 'complete_saved_actions', 'execute_new', 'population'} <= set(functions),
            'source audit: missing public inference/late-evaluation entry points')
    dump = lambda node: ast.dump(node, include_attributes=False)
    callers = [(function.name, node.lineno) for function in functions.values() for node in ast.walk(function)
               if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'private_truths']
    require(len(callers) == 1 and callers[0][0] == '_run', 'source audit: extra private parse path')
    parse_line = callers[0][1]
    run = functions['_run']
    gates = [node for node in run.body if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and
             isinstance(node.value.func, ast.Name) and node.value.func.id == 'complete_saved_actions']
    require(len(gates) == 1 and gates[0].end_lineno < parse_line and
            [dump(arg) for arg in gates[0].value.args] ==
            [dump(ast.parse(name, mode='eval').body) for name in ('output', 'decisions')],
            'source audit: complete saved action fence must precede private parsing')
    predicate = "len(decisions) != 972 or len({(r['task_id'], r['method']) for r in decisions}) != 972 or load(output / 'decisions.json') != decisions or Counter((r['method'], r['reused']) for r in decisions) != {('keep', False): 324, ('calibrated', False): 324, ('joint', False): 324}"
    fences = [node for node in functions['complete_saved_actions'].body if isinstance(node, ast.If) and
              dump(node.test) == dump(ast.parse(predicate, mode='eval').body) and
              any(isinstance(child, ast.Raise) for child in node.body)]
    require(len(fences) == 1, 'source audit:972 unique/read-back/new-origin fence changed')
    private = functions['private_truths']
    require(len(private.body) == 1 and isinstance(private.body[0], ast.Return) and
            dump(private.body[0].value) == dump(ast.parse("load(DATA / 'private/truth.json')", mode='eval').body),
            'source audit: private parse must use exact new data root')
    execute = functions['execute_new']
    require([arg.arg for arg in execute.args.args] == ['item', 'method'] and len(execute.body) == 2 and
            dump(execute.body[0]) == dump(ast.parse('solver.clear_geometry_cache()').body[0]) and
            isinstance(execute.body[1], ast.Return) and
            dump(execute.body[1].value) == dump(ast.parse(
                "solver.solve_public(item['observation'], item['radius'], item['baseline'], method, item['clipped'])",
                mode='eval').body), 'source audit: inference must pass only public image/camera/method/clipping')
    executions = [node for node in ast.walk(run) if isinstance(node, ast.Call) and
                  isinstance(node.func, ast.Name) and node.func.id == 'execute_new']
    require(len(executions) == 1 and not executions[0].keywords and
            [dump(arg) for arg in executions[0].args] ==
            [dump(ast.parse(name, mode='eval').body) for name in ('item', 'method')],
            'source audit: private/incumbent data enters scientific fit')
    for name in ('_run', 'population', 'execute_new', 'complete_saved_actions'):
        require(not any(isinstance(node, ast.Constant) and isinstance(node.value, str) and 'truth.json' in node.value
                        for node in ast.walk(functions[name])), f'source audit: direct private path in {name}')
    return {'status': 'PASS', 'runner': str(ROOT / 'run_baseline.py'), 'private_parse_line': parse_line,
            'complete_saved_actions_call_line': gates[0].lineno, 'unique_action_readback_origin_gate_line': fences[0].lineno,
            'solver_arguments': ['observation', 'radius', 'baseline', 'method', 'clipped'],
            'scope': 'source call order and public interfaces; does not prove operating-system blindness'}


def cost_rows(timing, selectors, outers):
    return {method: {'cold_solver_calls': sum(row['method'] == method for row in timing),
        'outer_seconds': sum(row['outer_seconds'] for row in timing if row['method'] == method),
        'preparation_seconds': sum(row['preparation_seconds'] for row in timing if row['method'] == method),
        'selector_seconds': sum(row['seconds_by_method'][method] for row in selectors),
        'evaluated_boxes': sum(arms[method]['outer']['evaluated_boxes'] for arms in outers.values()),
        'unresolved_budget_leaves': sum(arms[method]['outer']['unresolved_budget_leaves'] for arms in outers.values())}
        for method in METHODS}


def verify_timing(output, observations, outers, elapsed):
    timing = load(output / 'timing.json')
    require(isinstance(timing, list) and len(timing) == 72, 'timing: complete72 cold solves required')
    records = indexed(timing, lambda row: (row['observation_id'], row['method']), 'cold solve timing')
    require(list(records) == [(oid, method) for oid in observations for method in METHODS[1:]],
            'timing: missing/reordered cold solve')
    for (oid, method), row in records.items():
        require(set(row) == {'observation_id', 'unit_id', 'method', 'branch_seconds', 'outer_seconds',
                             'preparation_seconds', 'reused'} and row['unit_id'] == observations[oid]['unit_id'] and
                row['reused'] is False, f'{oid}/{method}: false cold timing schema/origin')
        for field in ('branch_seconds', 'outer_seconds', 'preparation_seconds'):
            same(row[field], outers[oid][method][field], f'{oid}/{method}: recorded arm timing')
    selectors = load(output / 'selector_timing.json')
    selected = indexed(selectors, lambda row: row['observation_id'], 'selector timing')
    require(list(selected) == list(observations), 'selector timing: complete36 observations required')
    for oid, row in selected.items():
        require(set(row) == {'observation_id', 'seconds_by_method'} and list(row['seconds_by_method']) == list(METHODS),
                f'{oid}: changed selector methods/schema')
        for method, seconds in row['seconds_by_method'].items():
            finite_seconds(seconds, f'{oid}/{method}: selector seconds')
    costs = cost_rows(timing, selectors, outers)
    require(sum(row['outer_seconds'] + row['selector_seconds'] for row in costs.values()) <= elapsed,
            'timing: measured new work exceeds elapsed main duration')
    return costs


def verify_evaluation(output, rows, coverage, observations, truths, outers, status):
    require((len(rows), len(coverage)) == (972, 108), 'evaluation: missing action/coverage population')
    same(load(output / 'results.json'), rows, 'all972 actual-truth endpoint errors')
    verify_csv(output / 'results.csv', rows)
    same(load(output / 'coverage.json'), coverage, 'all108 depth/branch/material/bias coverage records')
    verify_csv(output / 'COVERAGE.csv', coverage)
    mp, bp = paired_method_rows(rows), paired_baseline_rows(rows)
    same(load(output / 'PAIRED_METHOD.json'), mp, 'all324 calibrated→joint matched tasks')
    same(load(output / 'PAIRED_BASELINE.json'), bp, 'all486 B70→B80 matched method/tasks')
    verify_csv(output / 'PAIRED_METHOD.csv', mp)
    verify_csv(output / 'PAIRED_BASELINE.csv', bp)
    summary = load(output / 'summary.json')
    require(isinstance(summary, dict) and set(summary) == {'population', 'rows', 'paired_method', 'paired_baseline',
            'safety', 'costs', 'sampling_note', 'timing_scope', 'elapsed_seconds'}, 'summary: changed complete schema')
    same(summary['population'], {'observations': 36, 'tasks': 324, 'decisions': 972, 'new_solver_calls': 72},
         'summary: exact registered population')
    same(summary['sampling_note'], SAMPLING_NOTE, 'summary: controlled one-seed synthetic scope')
    same(summary['timing_scope'], TIMING_SCOPE, 'summary: new cold solve timing scope')
    elapsed = finite_seconds(summary['elapsed_seconds'], 'summary elapsed')
    require(elapsed <= status['elapsed_seconds'], 'summary: elapsed exceeds completed main duration')
    aggregates = aggregate_rows(rows, truths)
    same(summary['rows'], aggregates, 'all grouped outcomes including both baselines/all three depths')
    verify_csv(output / 'SUMMARY.csv', aggregates)
    same(summary['paired_method'], paired_method_summary(rows, mp, outers), 'paired method gains/losses/set inclusion')
    same(summary['paired_baseline'], paired_baseline_summary(rows, bp, truths), 'paired baseline gains/losses/deep40 denominator')
    same(summary['safety'], safety_claims(rows, coverage), 'all reported scientific safety outcomes')
    same(summary['costs'], verify_timing(output, observations, outers, elapsed), 'all independently summed cold costs')
    return aggregates, mp, bp, summary


def verify(output):
    output = check_target(output)
    budget = load(output / 'BUDGET.json')
    deadline = verify_budget(budget)
    sources, inputs = verify_source_lock(load(output / 'SOURCE_LOCK.json')), verify_inputs(load(output / 'INPUTS.json'))
    same(load(ROOT / 'REGISTRATION.json'), registration(), 'current independently registered matrix')
    same(load(output / 'REGISTRATION.json'), registration(), 'saved independently registered matrix')
    audit = audit_runner_source()
    status = load(output / 'RUN_STATUS.json')
    require(isinstance(status, dict) and set(status) == {'status', 'observations_complete', 'decisions_complete',
            'new_solver_calls', 'elapsed_seconds', 'deadline_epoch'}, 'RUN_STATUS: changed completed schema')
    same({key: value for key, value in status.items() if key != 'elapsed_seconds'},
         {'status': 'COMPLETE', 'observations_complete': 36, 'decisions_complete': 972,
          'new_solver_calls': 72, 'deadline_epoch': deadline}, 'RUN_STATUS: registered counts and original window')
    elapsed = finite_seconds(status['elapsed_seconds'], 'RUN_STATUS elapsed')
    require(elapsed <= LIMIT and elapsed <= time.time() - budget['start_epoch'], 'RUN_STATUS: impossible main duration')
    observations, tasks = verify_public_population(output)
    decisions_list = load(output / 'decisions.json')
    verify_action_population(decisions_list, observations, tasks)
    # Only below this complete saved-action gate do we parse or regenerate facts.
    truths, witnesses = verify_private_population(observations, tasks)
    negative_proofs = [prove_full_negative_control(baseline) for baseline in BASELINES]
    outers = load(output / 'outers.json')
    require(isinstance(outers, dict) and list(outers) == list(observations), 'outers: complete36 registered observations/order')
    checkpoint = output / 'checkpoint'
    require(checkpoint.is_dir() and sorted(path.name for path in checkpoint.iterdir()) ==
            sorted(f'{oid}.json' for oid in observations), 'checkpoint: complete36 unique observation records')
    for oid, arms in outers.items():
        same(load(checkpoint / f'{oid}.json'), {'observation_id': oid, 'arms': arms}, f'{oid}: checkpoint readback')
    unions, branches = {}, {}
    nodes = updates = certificates = geometry_calls = cold_calls = 0
    for index, (oid, public) in enumerate(observations.items()):
        baseline = exact(public['baseline'])
        def geometry_bounds(branch, radius, low, high):
            return independent_geometry(branch, radius, low, high, baseline)
        image, arms = tuple(map(exact, public['observation'])), outers[oid]
        require(isinstance(arms, dict) and list(arms) == list(METHODS), f'{oid}: changed method arms/order')
        unions[oid], branches[oid], common = {}, {}, None
        for method, arm in arms.items():
            where = f'{oid}/{method}'
            fields = {'branches', 'outer', 'branch_seconds', 'outer_seconds'}
            if method != 'keep':
                fields |= {'preparation_seconds', 'background_mask', 'calibration_intervals', 'calibration'}
                cold_calls += 1
            require(isinstance(arm, dict) and set(arm) == fields, f'{where}: changed frozen arm schema')
            allowed = () if method == 'keep' else BRANCHES
            require(list(arm['branches']) == list(arm['branch_seconds']) == list(allowed), f'{where}: changed three-branch solve')
            total = finite_seconds(arm['outer_seconds'], where + ': solver seconds')
            measured = sum(finite_seconds(s, where + ': branch seconds') for s in arm['branch_seconds'].values())
            if method != 'keep':
                measured += finite_seconds(arm['preparation_seconds'], where + ': preparation seconds')
                calibration = _prior.verify_arm_public(arm, image, True, geometry_bounds, where)
                if common is None:
                    common = calibration
                else:
                    same(calibration, common, f'{oid}: same independently derived T_cal for cal/joint')
            require(total >= measured and (method != 'keep' or total == 0), f'{where}: impossible recorded solve cost')
            checked_branches = {}
            for branch in allowed:
                checked = _prior.verify_branch(arm['branches'][branch], image, True, method, branch,
                                               geometry_bounds, calibration, f'{where}/{branch}')
                checked_branches[branch] = checked
                nodes += len(arm['branches'][branch]['trace'])
                updates += checked['proof_updates']
                certificates += len(checked['shared_excluded'])
                geometry_calls += checked['evaluated']
            branches[oid][method] = checked_branches
            unions[oid][method] = _prior.verify_union(arm['outer'], checked_branches, method, True, where)
        require(interval_subset(unions[oid]['joint']['intervals'], unions[oid]['calibrated']['intervals']),
                f'{oid}: same-T_cal frozen joint depth union is not nested')
        for branch in BRANCHES:
            require(interval_subset(branches[oid]['joint'][branch]['intervals'], branches[oid]['calibrated'][branch]['intervals']),
                    f'{oid}/{branch}: frozen joint branch depth union is not nested')
        if index % 3 == 2:
            print(json.dumps({'stage': 'independent_baseline_verification', 'observations_complete': index + 1,
                              'remaining_shared_seconds': round(deadline - time.time(), 1)}), flush=True)
    require(cold_calls == 72, 'outers: not exactly72 cold method solves')
    rows = []
    for decision in decisions_list:
        task = tasks[decision['task_id']]
        oid, method = task['observation_id'], decision['method']
        require(decision['unit_id'] == observations[oid]['unit_id'] and decision['reused'] is False,
                f'{task["task_id"]}/{method}: false camera condition/action origin')
        checked, private = unions[oid][method], truths[oid]
        value = _arithmetic.verify_decision({key: item for key, item in decision.items() if key not in {'unit_id', 'reused'}},
                                            task['incumbent'], checked['hull'], f'{task["task_id"]}/{method}')
        row = make_row(task, private, decision, checked)
        # Negative scientific outcomes are preserved. A claimed covered endpoint
        # must still satisfy the independently checked safe-action certificate.
        if method != 'keep' and checked['hull'] is not None and row['covered']:
            truth, incumbent = exact(private['truth']), exact(task['incumbent'])
            require((incumbent - truth)**2 - (value - truth)**2 >= exact(decision['gain_lower']) >= 0,
                    f'{task["task_id"]}/{method}: covered endpoint contradicts action certificate')
        rows.append(row)
    coverage = [_prior.make_coverage(observations[oid], truths[oid], method, unions[oid][method], branches[oid][method])
                for unit in UNITS for oid in observations if observations[oid]['unit_id'] == unit for method in METHODS]
    aggregates, mp, bp, summary = verify_evaluation(output, rows, coverage, observations, truths, outers, status)
    same(scientific_files(), sources, 'complete science byte graph unchanged during independent verification')
    same(generated_inputs(), inputs, 'all new input bytes unchanged during verification')
    verify_budget(budget)
    return {'status': 'PASS', 'mechanical_verification': 'PASS', 'host': socket.gethostname(), 'output': str(output),
            'observations': 36, 'tasks': 324, 'decisions': 972, 'cold_solver_calls': 72, 'reused_actions': 0,
            'coverage_rows': len(coverage), 'summary_rows': len(aggregates), 'paired_method_rows': len(mp),
            'paired_baseline_rows': len(bp), 'branch_outer_traces': 216, 'tree_nodes_verified': nodes,
            'exact_updates_recomputed': updates, 'contradiction_certificates_verified': certificates,
            'independent_geometry_calls': geometry_calls, 'source_files_verified': len(sources),
            'new_data_inputs_verified': len(inputs), 'source_parse_order_audit': audit,
            'actual_truth_witnesses': witnesses, 'full_negative_control_proofs': negative_proofs,
            'scientific_outcomes': {key: summary[key] for key in ('paired_method', 'paired_baseline', 'safety')},
            'shared_budget': {'start_epoch': budget['start_epoch'], 'deadline_epoch': deadline,
                             'elapsed_main_plus_verification_seconds': time.time() - budget['start_epoch']},
            'limitations': ['Projected edge bounds and point images are independently reconstructed; historical rectangle area primitives remain shared.',
                           'Prior independently authored photometric/contractor/tree/union/action arithmetic is reused, not the new solver.',
                           'Retained necessary outer bounds do not prove exact joint feasibility.',
                           'Recorded cold timing is checked and summed, not independently reproduced.',
                           'Source call order and public interfaces do not prove operating-system blindness.',
                           'One paired noise/pose realization per template/depth is controlled synthetic imaging evidence, not real transfer.',
                           'Mechanical PASS does not require positive gain, new repairs, true coverage or zero scientific harm.']}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Independently verify the registered baseline intervention.')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/baseline-v1')
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

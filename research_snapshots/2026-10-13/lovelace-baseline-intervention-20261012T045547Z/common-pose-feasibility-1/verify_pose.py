#!/usr/bin/env python3
"""Independent A7 exact forward/seed/inherited-proof verification.

No A7 solver, lift, runner or generator is imported. Frozen independently
authored geometry, bands and certificate arithmetic are the only reuse.
Private facts are parsed only after all 324 saved diagnoses and 972 public
branch records, their witnesses and checkpoints have passed verification.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from contextlib import contextmanager
from fractions import Fraction as Q
from functools import lru_cache
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import socket
import sys
import time

sys.dont_write_bytecode = True
for _pose_verify_thread in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_pose_verify_thread] = '1'
ROOT = Path(__file__).resolve().parent
EXPECTED_ROOT = Path('/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/common-pose-feasibility-1')
A6 = ROOT.parent / 'point-feasibility-1'
PRIOR = A6 / 'outputs/point-v1'
A5_OUTPUT = ROOT.parent / 'joint-feasibility-1/outputs/polytope-v1'
DATA = ROOT.parent / 'appearance-variation-1/data'
PYTHON = Path('/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python')
_spec = importlib.util.spec_from_file_location('_pose_frozen_independent_point', A6 / 'verify_point.py')
_point = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_point)
_old, _proof = _point._old, _point._a5
_geometry = _old._baseline
VerificationError = _point.VerificationError
require, same, exact, encoded = _point.require, _point.same, _point.exact, _point.encoded
load, indexed, finite_seconds = _point.load, _point.indexed, _point.finite_seconds
rational = _proof.rational
BRANCHES = _point.BRANCHES
RADIUS, BETA, EPSILON, DRIFT = Q(1, 5), Q(1, 100), Q(1, 100), Q(1, 10)
LIMIT = 3600
BUDGET_SCOPE = 'A7 joint fixed-depth common-pose main plus independent verification; no reset'
STATES = ('FORWARD_WITNESS', 'UNKNOWN', 'INHERITED_INFEASIBLE')
LIMITS = {'lp_calls': 200, 'exact_attempts': 20, 'branch_seconds': 3.0,
          'powell_starts': 2, 'powell_maxfev': 60, 'offset_denominator': 1000000000}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_target(output=None):
    require(socket.gethostname() == 'liekkas' and ROOT == EXPECTED_ROOT and ROOT.resolve() == EXPECTED_ROOT,
            'target: exact A7 root on liekkas required')
    if output is not None:
        output = Path(output).resolve()
        require(output == ROOT / 'outputs/pose-v1' and output.is_dir(), 'target: exact existing outputs/pose-v1 required')
    return output


def scientific_files():
    files = dict(_point.scientific_files())
    for path in sorted({ROOT / 'AGENTS.md', ROOT / 'PROTOCOL.md', ROOT / 'REGISTRATION.json', *ROOT.glob('*.py')}):
        files[str(path)] = sha(path)
    return dict(sorted(files.items()))


def verify_inventory(root=A6):
    root = Path(root)
    inventory, seal = load(root / 'ARTIFACTS.json'), load(root / 'FINAL_SEAL.json')
    require(inventory.get('stage') == 'A6' and inventory.get('host') == 'liekkas' and inventory.get('root') == str(root) and
            inventory.get('count') == 94 and inventory.get('excluded_self_files') == ['ARTIFACTS.json', 'FINAL_SEAL.json'],
            'inventory: exact A6 target and94 archived artifacts required')
    files = inventory.get('files')
    require(isinstance(files, dict) and len(files) == 94, 'inventory: complete94 indexed files required')
    actual = sorted(str(path.relative_to(root)) for path in root.rglob('*') if path.is_file() and
                    path.relative_to(root).as_posix() not in ('ARTIFACTS.json', 'FINAL_SEAL.json'))
    same(sorted(files), actual, 'inventory: no missing or additional A6 artifacts')
    for rel, digest in files.items():
        path = root / rel
        require(not Path(rel).is_absolute() and path.resolve().is_relative_to(root.resolve()) and path.is_file(),
                'inventory: invalid indexed relative path')
        require(sha(path) == digest, 'inventory: archived bytes changed ' + rel)
    require(seal.get('stage') == 'A6' and seal.get('host') == 'liekkas' and seal.get('root') == str(root) and
            seal.get('output') == str(root / 'outputs/point-v1') and seal.get('status') == 'COMPLETE' and
            seal.get('mechanical_verification') == 'PASS' and seal.get('artifact_count_excluding_self_files') == 94 and
            seal.get('artifact_index_sha256') == sha(root / 'ARTIFACTS.json'), 'inventory: invalid final A6 seal')
    return 94


def input_files():
    verify_inventory()
    paths = [A6 / 'ARTIFACTS.json', A6 / 'FINAL_SEAL.json', PRIOR / 'diagnoses.json', PRIOR / 'observations.json',
             PRIOR / 'tasks.json', PRIOR / 'STAGE_STATUS.json', PRIOR / 'verification/VERIFY.json',
             A5_OUTPUT / 'outers.json', DATA / 'private/truth.json']
    return {str(path): sha(path) for path in paths}


def verify_locks(output):
    sources, inputs = scientific_files(), input_files()
    same(load(output / 'SOURCE_LOCK.json'), {'source_root': str(ROOT), 'files': sources}, 'SOURCE_LOCK: exact independent source graph')
    same(load(output / 'INPUTS.json'), {'files': inputs}, 'INPUTS: exact frozen input graph')
    _point.verify_locks(PRIOR)
    receipt, stage, status = (load(PRIOR / rel) for rel in ('verification/VERIFY.json', 'STAGE_STATUS.json', 'RUN_STATUS.json'))
    require(receipt.get('status') == 'PASS' and receipt.get('host') == 'liekkas' and receipt.get('output') == str(PRIOR) and
            stage.get('status') == 'COMPLETE' and stage.get('independently_verified') is True and status.get('status') == 'COMPLETE',
            'INPUTS: exact A6 COMPLETE/PASS receipt required')
    return sources, inputs


def verify_registration(value):
    expected = {'stage': 'A7', 'host': 'liekkas', 'observations': 36, 'tasks': 324, 'branch_records': 972,
                'methods': ['joint'], 'eta': ['0', '1/10'], 'baseline': '80', 'radius': '1/5', 'epsilon': '1/100',
                'beta': '1/100', 'drift_bound': '1/10', 'primary': {'eta_index': 1, 'depth_mm': '9007/14', 'n': 40},
                'scope': 'fixed-depth full forward witness completion', 'branch_seconds': 3, 'max_lp_calls': 200,
                'max_exact_attempts': 20, 'max_powell_starts': 2, 'powell_maxfev': 60, 'limit_seconds': 3600,
                'new_outputs': 0, 'infeasibility_from_search': False, 'private_parse_after_saved_gate': True}
    same(value, expected, 'REGISTRATION: exact A7 contract')


def public_inputs(public, arm, depth, branch):
    depth = exact(depth)
    require(branch in BRANCHES and public.get('clipped') is True and exact(public['baseline']) == 80 and
            exact(public['radius']) == RADIUS and _old.DOMAIN[0] <= depth <= _old.DOMAIN[1], 'forward: frozen public geometry required')
    require(arm.get('appearance_model') == 'bounded' and exact(arm.get('drift_bound')) == DRIFT, 'forward: unchanged bounded model required')
    image = tuple(map(exact, public['observation']))
    require(len(image) == 432 and all(0 <= y <= 1 for y in image), 'forward: complete432 clipped channels required')
    calibration = tuple(_point.pair(pair, 'forward calibration') for pair in arm['calibration_intervals'])
    require(len(calibration) == 3 and all(0 <= lo <= hi <= 2 * BETA for lo, hi in calibration), 'forward: original three legal calibration intervals required')
    return depth, image, calibration


def material_bias(darkness, ts, calibration, where):
    require(len(darkness) == 9 and len(ts) == 3 and all(0 <= d <= 1 for d in darkness), where + ': nine legal materials/three biases required')
    require(all(lo <= t <= hi for t, (lo, hi) in zip(ts, calibration)), where + ': bias outside original calibration')
    require(all(abs(darkness[3 * v + k] - darkness[3 * w + k]) <= DRIFT
                for v in range(3) for w in range(v + 1, 3) for k in range(3)), where + ': pairwise material drift exceeded')


@lru_cache(maxsize=8192)
def point_areas(branch, depth, offsets):
    return _geometry.independent_point_areas(branch, depth, offsets, Q(80))


def verify_forward(public, arm, depth, branch, witness, where='forward witness', candidate=False):
    depth, image, calibration = public_inputs(public, arm, depth, branch)
    require(isinstance(witness, dict) and set(witness) == {'offsets', 'darkness', 't', 'noise'}, where + ': exact witness schema required')
    require(all(isinstance(witness[key], list) for key in witness), where + ': explicit rational arrays required')
    offsets, darkness, ts, noise = (tuple(rational(v, where + ':' + key) for v in witness[key])
                                   for key in ('offsets', 'darkness', 't', 'noise'))
    require(len(offsets) == 3 and offsets[1] == 0 and all(abs(e) <= RADIUS for e in offsets), where + ': pose radius/gauge')
    if candidate:
        require(all(e.denominator <= LIMITS['offset_denominator'] for e in offsets), where + ': candidate rationalization cap')
    material_bias(darkness, ts, calibration, where)
    require(len(noise) == 432 and all(abs(n) <= EPSILON for n in noise), where + ': complete bounded432 noise required')
    areas = point_areas(branch, depth, offsets)
    require(len(areas) == 144, where + ': full144 geometry required')
    for pixel, triple in enumerate(areas):
        view = pixel // 48
        gray = 1 - sum((a * d for a, d in zip(triple, darkness[3 * view:3 * view + 3])), Q(0))
        require(0 <= gray <= 1, where + ': nonphysical raw gray')
        predicted = gray + BETA - ts[view]
        for channel in range(3):
            index = pixel * 3 + channel
            require(min(Q(1), max(Q(0), predicted + noise[index])) == image[index],
                    f'{where}: single-clip exact mismatch at channel {index}')
    return True


def seed_forward(public, depth, branch, coordinates, offsets):
    ds, ts = tuple(map(exact, coordinates[:9])), tuple(map(exact, coordinates[9:]))
    areas = point_areas(branch, exact(depth), tuple(map(exact, offsets)))
    noise = []
    for index, y in enumerate(map(exact, public['observation'])):
        pixel, view = index // 3, index // 144
        predicted = 1 - sum((a * d for a, d in zip(areas[pixel], ds[3 * view:3 * view + 3])), Q(0)) + BETA - ts[view]
        noise.append(encoded(-EPSILON if y == 0 else EPSILON if y == 1 else y - predicted))
    return {'offsets': list(offsets), 'darkness': list(coordinates[:9]), 't': list(coordinates[9:]), 'noise': noise}


def affine_partition(branch, depth, view):
    """All pairwise horizontal edge-order changes at fixed depth.

    Vertical overlaps are constant in camera x. Between these crossings each
    min/max-selected horizontal edge is affine, so all visible pixel areas
    and all fixed-material intensities are affine. No offset grid is used.
    """
    if view == 1:
        return (Q(0),)
    center = (-Q(80), Q(0), Q(80))[view]
    world = [(x, depth) for x in sorted({r[i] for r in (_geometry.LEFT, _geometry.RIGHT, _geometry.FRONT) for i in (0, 1)})]
    if branch != 'none':
        rear = _geometry.REAR if branch == 'partial' else _geometry.HIDDEN
        world.extend((x, _geometry.REAR_DEPTH) for x in rear[:2])
    functions = {(Q(-1) / z, (x - center) / z) for x, z in world}
    functions.update((Q(0), x) for pixel in _geometry.PIXELS for x in pixel[:2])
    ordered = sorted(functions)
    boundaries = {-RADIUS, RADIUS}
    for i, (slope, intercept) in enumerate(ordered):
        for other_slope, other_intercept in ordered[:i]:
            if slope != other_slope:
                crossing = (other_intercept - intercept) / (slope - other_slope)
                if -RADIUS <= crossing <= RADIUS:
                    boundaries.add(crossing)
    return tuple(sorted(boundaries))


@lru_cache(maxsize=4096)
def view_areas(branch, depth, view, offset):
    values = [Q(0)] * 3
    values[view] = offset
    return point_areas(branch, depth, tuple(values))[48 * view:48 * (view + 1)]


def seed_intervals(public, arm, depth, branch, coordinates):
    depth, image, calibration = public_inputs(public, arm, depth, branch)
    require(isinstance(coordinates, list) and len(coordinates) == 12, 'seed: unchanged exact12 coordinates required')
    values = tuple(rational(v, 'seed coordinate') for v in coordinates)
    darkness, ts = values[:9], values[9:]
    material_bias(darkness, ts, calibration, 'seed')
    # Frozen independently authored bands are (upper, lower), unlike solver rows.
    bands = _old.pixel_bands(image, True)
    views, offsets = [], []
    for view in range(3):
        cuts = affine_partition(branch, depth, view)
        cells = list(zip(cuts, cuts[1:])) if len(cuts) > 1 else [(cuts[0], cuts[0])]
        entries, chosen = [], None
        for left, right in cells:
            first, last = view_areas(branch, depth, view, left), view_areas(branch, depth, view, right)
            lo, hi = left, right
            for pixel, (start, finish) in enumerate(zip(first, last)):
                ds = darkness[3 * view:3 * view + 3]
                q0 = ts[view] + sum((a * d for a, d in zip(start, ds)), Q(0))
                q1 = ts[view] + sum((a * d for a, d in zip(finish, ds)), Q(0))
                slope = Q(0) if left == right else (q1 - q0) / (right - left)
                intercept = q0 - slope * left
                upper, lower = bands[48 * view + pixel]
                for a, bound in ((slope, None if upper is None else upper - intercept),
                                 (-slope, None if lower is None else intercept - lower)):
                    if bound is None:
                        continue
                    if a > 0:
                        hi = min(hi, bound / a)
                    elif a < 0:
                        lo = max(lo, bound / a)
                    elif bound < 0:
                        lo, hi = Q(1), Q(0)
                    if lo > hi:
                        break
                if lo > hi:
                    break
            feasible = None if lo > hi else [encoded(lo), encoded(hi)]
            entries.append({'cell': [encoded(left), encoded(right)], 'feasible': feasible})
            if chosen is None and feasible is not None:
                chosen = (lo + hi) / 2
        views.append({'view': view, 'breakpoints': list(map(encoded, cuts)), 'cells': entries,
                      'selected': None if chosen is None else encoded(chosen)})
        offsets.append(chosen)
    lifted = all(value is not None for value in offsets)
    return {'status': 'SEED_LIFTED' if lifted else 'SEED_NOT_LIFTABLE',
            'offsets': list(map(encoded, offsets)) if lifted else None, 'views': views,
            'reason': 'exact_fixed_seed_offset_completion' if lifted else 'fixed_seed_has_no_common_offset_completion'}


def verify_seed(public, arm, depth, branch, coordinates, record):
    expected = seed_intervals(public, arm, depth, branch, coordinates)
    same(record, expected, 'seed: exact complete affine partition and feasible subinterval replay')
    if expected['status'] == 'SEED_LIFTED':
        witness = seed_forward(public, depth, branch, coordinates, expected['offsets'])
        verify_forward(public, arm, depth, branch, witness, 'seed positive independent full forward witness')
        return witness
    return None


def aggregate(statuses):
    statuses = list(statuses)
    require(len(statuses) == 3 and all(s in STATES for s in statuses), 'union: three registered branches required')
    return ('FORWARD_WITNESS' if 'FORWARD_WITNESS' in statuses else
            'INHERITED_INFEASIBLE' if all(s == 'INHERITED_INFEASIBLE' for s in statuses) else 'UNKNOWN')


def verify_search(public, arm, depth, branch, record):
    fields = {'status', 'witness', 'center_status', 'center_witness', 'numeric_evaluations', 'classifier_lp_calls',
              'numeric_lp_calls', 'exact_attempts', 'numeric_failures', 'powell_starts', 'powell_evaluations',
              'best_phase_slack', 'seconds', 'termination', 'search_budget_seconds', 'limits'}
    require(isinstance(record, dict) and set(record) == fields, 'search: exact candidate accounting schema required')
    same(record['limits'], LIMITS, 'search: unchanged registered caps')
    for key in ('numeric_evaluations', 'classifier_lp_calls', 'numeric_lp_calls', 'exact_attempts', 'numeric_failures', 'powell_starts'):
        require(type(record[key]) is int and record[key] >= 0, 'search: nonnegative integer ' + key)
    calls, attempts, starts = record['numeric_lp_calls'], record['exact_attempts'], record['powell_starts']
    require(calls <= 200 and attempts <= 20 and starts <= 2 and record['classifier_lp_calls'] <= attempts and
            calls == record['numeric_evaluations'] + record['classifier_lp_calls'], 'search: LP/recovery/starts budget accounting')
    evaluations = record['powell_evaluations']
    require(isinstance(evaluations, list) and len(evaluations) == starts and all(type(n) is int and 0 <= n <= 60 for n in evaluations),
            'search: each Powell start has at most60 evaluations')
    require(sum(evaluations) <= record['numeric_evaluations'] <= 9 + sum(evaluations), 'search: center/grid/Powell evaluation accounting')
    finite_seconds(record['seconds'], 'search elapsed')
    require(finite_seconds(record['search_budget_seconds'], 'search proposal budget') <= 3.0,
            'search: at most3 seconds of cooperative candidate proposal time')
    if record['best_phase_slack'] is not None:
        finite_seconds(record['best_phase_slack'], 'search candidate slack')
    require(record['status'] in ('FORWARD_WITNESS', 'UNKNOWN') and record['center_status'] in ('FORWARD_WITNESS', 'UNKNOWN'),
            'search: numerical failure must never become exclusion')
    require(record['termination'] in ('first_exact_witness', 'branch_timeout', 'whole_deadline', 'lp_budget', 'search_exhausted', 'numeric_failure', 'no_exact_witness'),
            'search: unknown or out-of-domain termination')
    for status_key, witness_key in (('status', 'witness'), ('center_status', 'center_witness')):
        if record[status_key] == 'FORWARD_WITNESS':
            verify_forward(public, arm, depth, branch, record[witness_key], 'search ' + witness_key, candidate=True)
        else:
            require(record[witness_key] is None, 'search: UNKNOWN cannot assert witness')
    if record['center_status'] == 'FORWARD_WITNESS':
        require(all(exact(v) == 0 for v in record['center_witness']['offsets']), 'search: nominal witness must use zero pose')
        same(record['witness'], record['center_witness'], 'search: first positive center must be chosen and stop search')
        require(record['status'] == 'FORWARD_WITNESS' and starts == 0 and record['numeric_evaluations'] == 1,
                'search: stop at nominal success')
    if record['status'] == 'FORWARD_WITNESS':
        require(attempts >= 1 and record['termination'] == 'first_exact_witness', 'search: positive exact recovery provenance')
    else:
        require(record['termination'] != 'first_exact_witness', 'search: false positive stop')
    return record['seconds']


def verify_branch(record, source, public, arm, task):
    require(isinstance(record, dict) and set(record) == {'branch', 'source_status', 'status', 'seed', 'search', 'witness'},
            'branch: exact schema required')
    branch, depth = source['branch'], task['incumbent']
    same(record['branch'], branch, 'branch: frozen source identity/order')
    result = source['point_classification']
    rows, bounds = _point.point_problem(public, arm, 'joint', branch, depth)
    _point.verify_classification(rows, bounds, result, 'inherited A6 global point classification')
    same(record['source_status'], result['status'], 'branch: exact inherited status')
    if result['status'] == 'CERTIFIED_INFEASIBLE':
        same(record, {'branch': branch, 'source_status': result['status'], 'status': 'INHERITED_INFEASIBLE',
                      'seed': None, 'search': None, 'witness': None}, 'branch: inherited exact certificate is the only exclusion')
        return {'inherited': 1, 'search_seconds': 0.0, 'seed_lifted': 0, 'forward': 0}
    require(result['status'] == 'FEASIBLE_WITNESS', 'branch: original exact seed required; UNKNOWN not silently excluded')
    seed_witness = verify_seed(public, arm, depth, branch, result['witness'], record['seed'])
    seconds = verify_search(public, arm, depth, branch, record['search'])
    chosen = record['search']['witness'] if record['search']['status'] == 'FORWARD_WITNESS' else seed_witness
    same(record['witness'], chosen, 'branch: exact search-first then fixed-seed choice')
    same(record['status'], 'FORWARD_WITNESS' if chosen is not None else 'UNKNOWN', 'branch: full witness or inconclusive')
    if chosen is not None:
        verify_forward(public, arm, depth, branch, chosen, 'chosen full forward witness')
    return {'inherited': 0, 'search_seconds': seconds, 'seed_lifted': int(seed_witness is not None), 'forward': int(chosen is not None)}


def verify_record_population(records, observations, tasks):
    require(isinstance(records, list) and len(records) == 324, 'records: all324 task diagnoses required')
    identity = indexed(records, lambda r: r['task_id'], 'records: unique public task identities')
    same(list(identity), list(tasks), 'records: complete ordered324 public tasks')
    for record in records:
        require(isinstance(record, dict) and set(record) == {'task_id', 'observation_id', 'incumbent', 'a6_status', 'status', 'branches'},
                'records: changed diagnosis schema')
        task = tasks[record['task_id']]
        require(task['observation_id'] in observations, 'records: unknown observation')
        same(record['observation_id'], task['observation_id'], 'records: task/image mismatch')
        same(record['incumbent'], task['incumbent'], 'records: changed exact public incumbent')
        require(isinstance(record['branches'], list) and len(record['branches']) == 3, 'records: all3 branches required')
        same([r['branch'] for r in record['branches']], list(BRANCHES), 'records: all original branches in order')
        same(record['status'], aggregate(r['status'] for r in record['branches']), 'records: false existential branch union')
    require(sum(len(r['branches']) for r in records) == 972, 'records: all972 branches required')
    return identity


def audit_runner_source(source=None):
    path = ROOT / 'run_pose.py'
    tree = ast.parse(path.read_text() if source is None else source)
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    require({'run', 'evaluate', 'check_records', 'branch_record'} <= set(funcs), 'runner audit: missing public gate entry points')
    dump = lambda n: ast.dump(n, include_attributes=False)
    eval_calls = [(f.name, n) for f in funcs.values() for n in ast.walk(f) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'evaluate']
    require(len(eval_calls) == 1 and eval_calls[0][0] == 'run', 'runner audit: one sole evaluation after saved gate')
    evaluation = eval_calls[0][1].lineno
    trial = [n for n in funcs['run'].body if isinstance(n, ast.Try)]
    require(len(trial) == 1, 'runner audit: changed run gate flow')
    positions = []
    for expression in ("save(output/'diagnoses.json',records)", "records=load(output/'diagnoses.json')", 'check_records(records,ts)'):
        matches = [n for n in trial[0].body if dump(n) == dump(ast.parse(expression).body[0])]
        require(len(matches) == 1 and matches[0].end_lineno < evaluation, 'runner audit: save/readback/complete gate required before evaluation')
        positions.append((matches[0].lineno, matches[0].col_offset))
    require(positions == sorted(positions), 'runner audit: gate reordered')
    for predicate in ("len(records)!=324 or [r['task_id'] for r in records]!=[t['task_id'] for t in tasks]", "sum(len(r['branches']) for r in records)!=972"):
        require(any(isinstance(n, ast.If) and dump(n.test) == dump(ast.parse(predicate, mode='eval').body) and
                    any(isinstance(c, ast.Raise) for c in n.body) for n in funcs['check_records'].body), 'runner audit: exact324/972 gate changed')
    private_expr = dump(ast.parse("DATA/'private/truth.json'", mode='eval').body)
    parses = [(f.name, n) for f in funcs.values() for n in ast.walk(f) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and
              n.func.id == 'load' and n.args and dump(n.args[0]) == private_expr]
    require(len(parses) == 1 and parses[0][0] == 'evaluate', 'runner audit: private parse outside sole evaluation route')
    return {'status': 'PASS', 'runner': str(path), 'saved_gate_positions': positions,
            'evaluation_line': evaluation, 'private_parse_line': parses[0][1].lineno}


def audit_executor_source(source=None):
    path = ROOT / 'execute_pose.py'
    tree = ast.parse(path.read_text() if source is None else source)
    execute = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'execute'), None)
    require(execute is not None, 'executor audit: missing entry point')
    trials = [n for n in execute.body if isinstance(n, ast.Try)]
    require(len(trials) == 1, 'executor audit: changed stage flow')
    dump = lambda n: ast.dump(n, include_attributes=False)
    expressions = ('start=time.time()', 'deadline=start+runner.LIMIT',
                   "runner.save(output/'BUDGET.json',{'start_epoch':start,'deadline_epoch':deadline,'limit_seconds':runner.LIMIT,'scope':runner.BUDGET_SCOPE})",
                   'runner.run(output)', 'remaining=deadline-time.time()',
                   "subprocess.run([sys.executable,'-B',str(ROOT/'verify_pose.py'),'--output',str(output)],check=True,timeout=remaining)")
    positions = []
    for expression in expressions:
        matches = [n for n in trials[0].body if dump(n) == dump(ast.parse(expression).body[0])]
        require(len(matches) == 1, 'executor audit: shared absolute main/verification chain required')
        positions.append((matches[0].lineno, matches[0].col_offset))
    require(positions == sorted(positions) and sum(isinstance(n, ast.Constant) and n.value == 'BUDGET.json' for n in ast.walk(execute)) == 1,
            'executor audit: reordered or reset budget')
    return {'status': 'PASS', 'executor': str(path), 'shared_deadline_positions': positions, 'budget_resets': 0}


def evaluation_row(record, task, public, fact):
    def status(kind):
        vals = []
        for branch in record['branches']:
            positive = (branch['search'] is not None and branch['search']['center_status'] == 'FORWARD_WITNESS') if kind == 'nominal' else (
                branch['seed'] is not None and branch['seed']['status'] == 'SEED_LIFTED')
            vals.append('FORWARD_WITNESS' if positive else 'INHERITED_INFEASIBLE' if branch['status'] == 'INHERITED_INFEASIBLE' else 'UNKNOWN')
        return aggregate(vals)
    truth, depth = exact(fact['truth']), exact(task['incumbent'])
    return {'task_id': task['task_id'], 'observation_id': public['observation_id'], 'unit_id': public['unit_id'],
            'variation_index': public['variation_index'], 'case': fact['case'], 'truth_mm': str(truth), 'incumbent_mm': str(depth),
            'initial_ae_mm': str(abs(depth - truth)), 'a6_status': record['a6_status'], 'status': record['status'],
            'nominal_status': status('nominal'), 'seed_status': status('seed')}


def summary_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        cells = [('all', ''), ('shifted' if Q(row['initial_ae_mm']) else 'correct', '')]
        if Q(row['initial_ae_mm']):
            informative = row['case'] != 'equal_full_negative'
            cells.extend([('informative_shifted' if informative else 'negative_shifted', ''), ('case_shifted', row['case'])])
            if informative:
                cells.append(('depth_informative', row['truth_mm']))
        for group, value in cells:
            groups[row['unit_id'], group, value].append(row)
    return [{'unit_id': unit, 'group': group, 'value': value, 'n': len(selected),
             **{key: dict(sorted(Counter(r[key] for r in selected).items())) for key in ('a6_status', 'status', 'nominal_status', 'seed_status')},
             'additional_verified_over_nominal': sum(r['status'] == 'FORWARD_WITNESS' and r['nominal_status'] != 'FORWARD_WITNESS' for r in selected)}
            for (unit, group, value), selected in sorted(groups.items())]


def verify_summary_denominators(summary):
    for unit in ('appearance_eta_0', 'appearance_eta_1_10'):
        cells = {(r['group'], r['value']): r for r in summary if r['unit_id'] == unit}
        for group, n in (('all', 162), ('shifted', 144), ('correct', 18), ('informative_shifted', 120), ('negative_shifted', 24)):
            require(cells.get((group, ''), {}).get('n') == n, 'summary: full unchanged ' + group + ' denominator')
        depths = [r for (group, _), r in cells.items() if group == 'depth_informative']
        cases = [r for (group, _), r in cells.items() if group == 'case_shifted']
        require(len(depths) == 3 and all(r['n'] == 40 for r in depths) and len(cases) == 6 and all(r['n'] == 24 for r in cases) and
                cells.get(('depth_informative', '9007/14'), {}).get('n') == 40, 'summary: fixed depth/case/primary40 matrix')


def actual_coverage(public, arm, fact, records):
    witness = {'offsets': [encoded(exact(v)) for v in fact['actual_offsets']],
               'darkness': [encoded(1 - exact(v)) for v in fact['view_colors']],
               't': [encoded(BETA - exact(v)) for v in fact['actual_view_bias']],
               'noise': [encoded(exact(v)) for v in fact['image_noise']]}
    branch, depth = fact['rear_branch'], exact(fact['truth'])
    verify_forward(public, arm, depth, branch, witness, 'actual full forward witness after complete public gate')
    correct = [r for r in records if r['observation_id'] == public['observation_id'] and exact(r['incumbent']) == depth]
    require(len(correct) == 1 and correct[0]['status'] != 'INHERITED_INFEASIBLE', 'coverage: actual depth falsely excluded')
    generating = [r for r in correct[0]['branches'] if r['branch'] == branch]
    require(len(generating) == 1 and generating[0]['status'] != 'INHERITED_INFEASIBLE', 'coverage: actual generating branch falsely excluded')
    return {'observation_id': public['observation_id'], 'generating_branch': branch, 'actual_forward_witness_verified': True,
            'correct_input_status': correct[0]['status'], 'generating_branch_status': generating[0]['status']}


def verify_budget(budget):
    require(isinstance(budget, dict) and set(budget) == {'start_epoch', 'deadline_epoch', 'limit_seconds', 'scope'}, 'BUDGET: shared window schema')
    start, deadline = (finite_seconds(budget[k], 'BUDGET ' + k) for k in ('start_epoch', 'deadline_epoch'))
    require(type(budget['limit_seconds']) is int and budget['limit_seconds'] == LIMIT and deadline == start + LIMIT and
            budget['scope'] == BUDGET_SCOPE, 'BUDGET: common absolute3600-second window')
    if time.time() >= deadline:
        raise TimeoutError('A7 shared main/verification deadline exhausted')
    return deadline


@contextmanager
def deadline_guard(output):
    deadline = verify_budget(load(Path(output) / 'BUDGET.json'))
    require(not signal.getitimer(signal.ITIMER_REAL)[0], 'BUDGET: existing alarm cannot be reset')
    def expired(_signum, _frame):
        raise TimeoutError('A7 shared main/verification deadline reached')
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, max(0.000001, deadline - time.time()))
    try:
        yield deadline
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def verify(output):
    output = check_target(output)
    require(Path(sys.executable) == PYTHON, 'target: exact existing CPU Python runtime required')
    budget = load(output / 'BUDGET.json')
    deadline = verify_budget(budget)
    sources, inputs = verify_locks(output)
    registration = load(ROOT / 'REGISTRATION.json')
    verify_registration(registration)
    same(load(output / 'REGISTRATION.json'), registration, 'REGISTRATION: saved source equality')
    runner_audit, executor_audit = audit_runner_source(), audit_executor_source()
    observations, tasks, original_public, _ = _old.verify_public_population(output)
    same(load(output / 'observations.json'), load(PRIOR / 'observations.json'), 'population: exact frozen A6 images')
    same(load(output / 'tasks.json'), load(PRIOR / 'tasks.json'), 'population: exact frozen A6 tasks')
    records = load(output / 'diagnoses.json')
    verify_record_population(records, observations, tasks)
    old_records = load(PRIOR / 'diagnoses.json')
    _point.verify_record_population(old_records, observations, tasks)
    old_joint = indexed([r for r in old_records if r['method'] == 'joint'], lambda r: r['task_id'], 'prior joint324')
    prior_arms = load(A5_OUTPUT / 'outers.json')
    same(list(prior_arms), list(observations), 'old arms: all36 frozen observations')
    counts, search_seconds = Counter(), 0.0
    for i, record in enumerate(records):
        oid, tid = record['observation_id'], record['task_id']
        source = old_joint[tid]
        same(source['observation_id'], oid, 'prior: unchanged task/image identity')
        same(source['incumbent'], tasks[tid]['incumbent'], 'prior: exact public incumbent')
        same([r['branch'] for r in source['global']], list(BRANCHES), 'prior: all3 original branches')
        derived = _point.aggregate_status([r['point_classification'] for r in source['global']])
        same(source['global_status'], derived, 'prior: exact global necessary union')
        same(record['a6_status'], derived, 'record: unchanged A6 necessary baseline')
        for branch, old_branch in zip(record['branches'], source['global']):
            verified = verify_branch(branch, old_branch, observations[oid], prior_arms[oid]['box']['joint'], tasks[tid])
            search_seconds += verified.pop('search_seconds')
            counts.update(verified)
            counts['branches'] += 1
        counts['tasks'] += 1
        if i % 27 == 26:
            print(json.dumps({'stage': 'independent_pose_verification', 'tasks_complete': i + 1,
                              'remaining_seconds': round(deadline - time.time(), 1)}), flush=True)
    require(counts['tasks'] == 324 and counts['branches'] == 972, 'public gate: complete324/972 independently checked')
    checkpoint = output / 'checkpoint'
    require(checkpoint.is_dir() and sorted(p.name for p in checkpoint.iterdir()) == sorted(oid + '.json' for oid in observations),
            'checkpoint: exactly36 public observation pages')
    for oid in observations:
        same(load(checkpoint / (oid + '.json')), {'observation_id': oid, 'records': [r for r in records if r['observation_id'] == oid]},
             'checkpoint: exact saved public readback ' + oid)
    # All 324/972 public rows, positive witnesses, inherited certificates and
    # fixed-seed interval claims have passed. This is the first private parse.
    facts, original_forward_checks = _old.verify_private_population(observations, tasks, original_public)
    coverage = [actual_coverage(p, prior_arms[oid]['box']['joint'], facts[oid], records) for oid, p in observations.items()]
    require(len(coverage) == 36, 'coverage: every actual generating full forward witness required')
    rows = [evaluation_row(r, tasks[r['task_id']], observations[r['observation_id']], facts[r['observation_id']]) for r in records]
    same(load(output / 'results.json'), rows, 'evaluation: all324 independent rows')
    _geometry.verify_csv(output / 'results.csv', rows)
    summary = summary_rows(rows)
    verify_summary_denominators(summary)
    same(load(output / 'summary.json'), summary, 'evaluation: exact full summary and primary40')
    status = load(output / 'RUN_STATUS.json')
    require(isinstance(status, dict) and set(status) == {'status', 'observations_complete', 'diagnoses_complete', 'branch_records', 'elapsed_seconds', 'deadline_epoch'},
            'RUN_STATUS: complete schema')
    same({k: v for k, v in status.items() if k != 'elapsed_seconds'},
         {'status': 'COMPLETE', 'observations_complete': 36, 'diagnoses_complete': 324, 'branch_records': 972, 'deadline_epoch': deadline},
         'RUN_STATUS: complete fixed batch/common deadline')
    elapsed = finite_seconds(status['elapsed_seconds'], 'RUN_STATUS elapsed')
    require(search_seconds <= elapsed < LIMIT and elapsed <= time.time() - budget['start_epoch'], 'TIMING: impossible search/main duration')
    same(scientific_files(), sources, 'SOURCE_LOCK: unchanged bytes through verification')
    same(input_files(), inputs, 'INPUTS: unchanged frozen bytes through verification')
    verify_budget(budget)
    return {'status': 'PASS', 'mechanical_verification': 'PASS', 'host': socket.gethostname(), 'output': str(output),
            'observations': 36, 'tasks': 324, 'branch_records': 972, 'counts': dict(sorted(counts.items())), 'evaluation_rows': 324,
            'actual_forward_coverage': coverage, 'original_forward_checks': original_forward_checks,
            'old_inventory_artifacts_verified': 94, 'source_files_verified': len(sources), 'input_files_verified': len(inputs),
            'source_parse_order_audit': runner_audit, 'shared_budget_source_audit': executor_audit,
            'scientific_scope': 'mechanical exact fixed-depth forward witnesses and fixed-seed completion; UNKNOWN is inconclusive; inherited exclusions are not new; no geometry repair or semantic acceptance',
            'shared_budget': {'start_epoch': budget['start_epoch'], 'deadline_epoch': deadline, 'elapsed_main_verification_seconds': time.time() - budget['start_epoch']}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/pose-v1')
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

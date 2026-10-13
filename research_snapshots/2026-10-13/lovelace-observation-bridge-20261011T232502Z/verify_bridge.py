#!/usr/bin/env python3
"""Independent evidence verifier for the clipping/bound/round bridge.

Imports frozen independent verification arithmetic, never the new runner or
solver. Every image transformation, source reuse and endpoint is reconstructed.
Historical module globals are scoped in memory; historical files stay read-only.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from contextlib import contextmanager
import csv
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
for _bridge_verifier_threads in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_bridge_verifier_threads] = '1'

ROOT = Path(__file__).resolve().parent
EXPECTED_ROOT = Path('/srv/slam-research/grf/map-denoise/runs/lovelace-observation-bridge-20261011T232502Z')
ABLATION_ROOT = ROOT.parent / 'lovelace-shared-ablation-20261011T055200Z'
ABLATION_OUTPUT = ABLATION_ROOT / 'outputs/ablation-v1'
BIAS_ROOT = ROOT.parent / 'lovelace-view-bias-20261011T092300Z'
BIAS_OUTPUT = BIAS_ROOT / 'outputs/view-bias-v1'
BASE_OUTPUT = ROOT.parent / 'lovelace-shared-appearance-20261011T052300Z/outputs/confirmation-v1'
_spec = importlib.util.spec_from_file_location('_bridge_frozen_independent_view_verifier', BIAS_ROOT / 'verify_view_bias.py')
_bias_verifier = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bias_verifier)
_old = _bias_verifier._old
_historical, _arithmetic = _old._historical, _old._arithmetic
VerificationError = _old.VerificationError
require, same, load, indexed = _old.require, _old.same, _old.load, _old.indexed
exact, encoded, pair, finite_seconds = _old.exact, _old.encoded, _old.pair, _old.finite_seconds
DOMAIN, TOLERANCE, RADIUS, MAX_BOXES = _old.DOMAIN, _old.TOLERANCE, Q(1, 5), 511
EPSILON, DELTA, BETA = Q(1, 100), Q(1, 50), Q(1, 100)
METHODS = ('keep', 'shared')
BRANCHES = ('none', 'partial', 'full')
UNITS = (
    ('R0epsilon', 'raw', EPSILON, 8, True),
    ('C0epsilon', 'clipped', EPSILON, 8, False),
    ('R0delta', 'raw', DELTA, 8, False),
    ('C0delta', 'clipped', DELTA, 8, False),
    ('RBdelta', 'raw_bias', DELTA, 8, False),
    ('CBdelta', 'clipped_bias', DELTA, 8, True),
    ('CBdelta4', 'clipped_bias', DELTA, 4, False),
    ('CBdelta16', 'clipped_bias', DELTA, 16, False),
)
UNIT_CONFIGS = {unit: {'image_mode': image, 'bound': str(bound), 'rounds': rounds, 'reused': reused}
                for unit, image, bound, rounds, reused in UNITS}
META = ('observation_id', 'base_observation_id', 'unit_id', 'bound', 'rounds', 'reused')
CONTRASTS = (('R0epsilon', 'C0epsilon'), ('R0epsilon', 'R0delta'), ('C0epsilon', 'C0delta'),
             ('R0delta', 'RBdelta'), ('C0delta', 'CBdelta'), ('RBdelta', 'CBdelta'),
             ('R0delta', 'C0delta'), ('CBdelta', 'CBdelta4'), ('CBdelta', 'CBdelta16'))
CACHE_POLICY = 'clear all geometry/model caches once before each new shared image solve'
SAMPLING_NOTE = 'same 54 base synthetic observations, related initial states and transformations; no new independent scenes'
TIMING_SCOPE = 'new cold solver/selector cost separate from recorded historical outer time; reuse validation is setup overhead'


def transformed_image(base, unit):
    require(unit in UNIT_CONFIGS, 'generation: unregistered unit')
    image = tuple(exact(value) for value in base)
    require(len(image) == 432, 'generation: fixed three views of48 RGB pixels required')
    mode = UNIT_CONFIGS[unit]['image_mode']
    if mode in ('raw_bias', 'clipped_bias'):
        offsets = (-BETA, Q(0), BETA)
        image = tuple(value + offsets[index // 144] for index, value in enumerate(image))
    if mode in ('clipped', 'clipped_bias'):
        image = tuple(min(Q(1), max(Q(0), value)) for value in image)
    return image


def necessary_rows(image, geometry, epsilon):
    # The prior verifier independently derives rows from the RGB observations
    # with its explicit bound; it neither imports nor calls a shared solver.
    return _bias_verifier.necessary_rows(image, geometry, epsilon)


@contextmanager
def historical_contract(epsilon, rounds):
    epsilon = exact(epsilon)
    require(epsilon in (EPSILON, DELTA) and type(rounds) is int and rounds in (4, 8, 16),
            'verification contract: unregistered bound or propagation rounds')
    saved = (_old.EPSILON, _old.ROUNDS, _historical.EPSILON, _historical.ROUNDS, _old.verify_group_certificate)
    def independent_certificate(image, geometry, mode, group_id, certificate, where):
        require(mode == 'global_gray' and group_id == 'global', f'{where}: wrong shared material group')
        return _historical.verify_certificate(necessary_rows(image, geometry, epsilon), certificate, where)
    _old.EPSILON, _old.ROUNDS, _historical.EPSILON, _historical.ROUNDS = epsilon, rounds, epsilon, rounds
    _old.verify_group_certificate = independent_certificate
    try:
        yield
    finally:
        (_old.EPSILON, _old.ROUNDS, _historical.EPSILON, _historical.ROUNDS,
         _old.verify_group_certificate) = saved


def verify_branch(outer, image, unit, branch, geometry_bounds, model, where):
    require(unit in UNIT_CONFIGS, f'{where}: unknown unit')
    config = UNIT_CONFIGS[unit]
    with historical_contract(exact(config['bound']), config['rounds']):
        checked = _old.verify_trace(outer, image, where, mode='global_gray', branch=branch, geometry_bounds=geometry_bounds)
        calls, witnesses = _old.verify_kernel_boxes(outer, checked, image, model, where, True)
    return checked, calls, witnesses


def verify_union(outer, branches, unit, method, where):
    require(unit in UNIT_CONFIGS and method in METHODS, f'{where}: unknown unit/method')
    config = UNIT_CONFIGS[unit]
    # KEEP's historical identity prior performs no image or epsilon inference.
    epsilon, rounds = (EPSILON, 8) if method == 'keep' else (exact(config['bound']), config['rounds'])
    with historical_contract(epsilon, rounds):
        return _old.verify_union(outer, branches, where, prior_only=method == 'keep', mode='global_gray')


def verify_decision(decision, task, checked, unit, where):
    require(unit in UNIT_CONFIGS and decision.get('method') in METHODS, f'{where}: unregistered action identity')
    config, method = UNIT_CONFIGS[unit], decision['method']
    require(decision.get('unit_id') == unit and decision.get('bound') == config['bound'] and
            type(decision.get('rounds')) is int and decision['rounds'] == config['rounds'] and
            type(decision.get('reused')) is bool and decision['reused'] == (config['reused'] and method == 'shared'),
            f'{where}: false action bound/rounds/reuse metadata')
    core = {key: value for key, value in decision.items() if key not in {'unit_id', 'bound', 'rounds', 'reused'}}
    return _arithmetic.verify_decision(core, task['incumbent'], checked['hull'], where)


def validate_residual(raw, image, private, unit, model_class, where):
    require(tuple(map(exact, image)) == transformed_image(raw, unit), f'{where}: changed image transformation')
    model = model_class(private['rear_branch'], camera_radius=RADIUS, colors=((Q(0), Q(1)),) * 3)
    prediction = model.point_prediction(exact(private['truth']), tuple(map(exact, private['actual_offsets'])),
                                         colors=tuple(map(exact, private['colors'])))
    require(len(prediction) == 432 and all(0 <= value <= 1 for value in prediction), f'{where}: invalid noiseless source prediction')
    residual = max(abs(exact(value) - truth) for value, truth in zip(image, prediction))
    require(residual <= exact(UNIT_CONFIGS[unit]['bound']), f'{where}: actual truth residual outside declared bound')
    return residual


def reuse_source(unit, base_id):
    require(unit in ('R0epsilon', 'CBdelta'), 'reuse: unexpected historical unit')
    return {'output_root': str(ABLATION_OUTPUT if unit == 'R0epsilon' else BIAS_OUTPUT),
            'observation_id': base_id if unit == 'R0epsilon' else f'b1-{base_id}',
            'method': 'global_gray511' if unit == 'R0epsilon' else 'global_shared_wide'}


def verify_reused_arm(actual, unit, base_id, historical_outers, image, historical_images=None):
    source = reuse_source(unit, base_id)
    record = historical_outers[unit][source['observation_id']]
    same(actual, record['arms'][source['method']], f'{unit}/{base_id}: exact historical arm body/recorded costs')
    if unit == 'R0epsilon':
        raw = (historical_images[unit][base_id] if historical_images is not None else
               indexed(load(BASE_OUTPUT / 'observations.json'), lambda row: row['observation_id'], 'reuse raw images')[base_id])
        expected_image = tuple(map(exact, raw['observation']))
    else:
        public = (historical_images[unit][source['observation_id']] if historical_images is not None else
                  indexed(load(BIAS_OUTPUT / 'observations.json'), lambda row: row['observation_id'], 'reuse bias images')[source['observation_id']])
        require(public['base_observation_id'] == base_id and public['beta'] == str(BETA) and public['radius'] == str(RADIUS),
                f'{unit}/{base_id}: wrong historical image identity/parameters')
        expected_image = tuple(map(exact, public['observation']))
    require(tuple(map(exact, image)) == expected_image, f'{unit}/{base_id}: reused solver was fed a different image')
    return source


def scientific_files():
    files = dict(_bias_verifier.scientific_files())
    for path in sorted({ROOT / 'PROTOCOL.md', *ROOT.glob('*.py')}):
        files[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(sorted(files.items()))


def historical_inputs():
    common = ('INPUTS.json', 'SOURCE_LOCK.json', 'PROTOCOL.json', 'RUN_STATUS.json',
              'outers.json', 'decisions.json', 'TIMING.json', 'verification/VERIFY.json')
    paths = [BASE_OUTPUT / relative for relative in _old.FROZEN_FILES]
    paths.extend(ABLATION_OUTPUT / relative for relative in common)
    paths.extend(BIAS_OUTPUT / relative for relative in (*common, 'observations.json', 'tasks.json'))
    return {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def registration():
    kinds = {'raw': 'raw', 'clipped': 'clip', 'raw_bias': 'biased_raw', 'clipped_bias': 'biased_clip'}
    return {'units': {unit: {'image_kind': kinds[mode], 'bound': str(bound), 'rounds': rounds, 'reused': reused}
                       for unit, mode, bound, rounds, reused in UNITS},
            'methods': list(METHODS), 'base_noise_epsilon': str(EPSILON), 'bias_magnitude_generation_only': str(BETA),
            'bias_pattern_generation_only': ['-1/100', '0', '1/100'], 'layout': {'views': 3, 'pixels_per_view': 48, 'channels': 3},
            'mode': 'global_gray', 'allowed_branches': list(BRANCHES), 'max_boxes_per_branch': MAX_BOXES,
            'tolerance_mm': str(TOLERANCE), 'expected_observations': 432, 'expected_tasks': 3888, 'expected_decisions': 7776,
            'expected_new_shared_observations': 324, 'expected_reused_shared_observations': 108,
            'expected_new_shared_actions': 2916, 'expected_reused_shared_actions': 972, 'expected_keep_actions': 3888,
            'expected_per_unit_method': {'observations': 54, 'correct': 54, 'small': 432, 'negative_small': 72},
            'contrasts': [list(pair) for pair in CONTRASTS], 'cpu_threads': 1, 'gpu_count': 0,
            'wall_time_limit_seconds_main_plus_verification': 5400, 'source_root': str(BASE_OUTPUT),
            'reused_sources': {unit: {'output_root': str(ABLATION_OUTPUT if unit == 'R0epsilon' else BIAS_OUTPUT),
                                      'method': 'global_gray511' if unit == 'R0epsilon' else 'global_shared_wide'}
                               for unit in ('R0epsilon', 'CBdelta')},
            'cache_policy': CACHE_POLICY, 'sampling_note': SAMPLING_NOTE}


def verify_source_lock(lock):
    require(isinstance(lock, dict) and set(lock) == {'source_root', 'files', 'lock_scope', 'utc'}, 'SOURCE_LOCK: invalid schema')
    require(lock['source_root'] == str(ROOT) and
            lock['lock_scope'] == 'both frozen historical scientific source graphs plus new root scientific sources and protocol',
            'SOURCE_LOCK: changed exact source identity/scope')
    require(type(lock['utc']) is str and datetime.fromisoformat(lock['utc']).utcoffset() is not None,
            'SOURCE_LOCK: timezone-aware timestamp required')
    _old.verify_source_lock(load(ABLATION_OUTPUT / 'SOURCE_LOCK.json'))
    _bias_verifier.verify_source_lock(load(BIAS_OUTPUT / 'SOURCE_LOCK.json'))
    sources = scientific_files()
    same(lock['files'], sources, 'SOURCE_LOCK full independent source inventory and byte hashes')
    return sources


def verify_inputs(lock):
    require(isinstance(lock, dict) and set(lock) == {'source_root', 'files'} and lock['source_root'] == str(BASE_OUTPUT),
            'INPUTS: changed historical data identity/schema')
    inputs = historical_inputs()
    require(len(inputs) == 26, 'INPUTS: incorrect frozen input inventory')
    same(lock['files'], inputs, 'INPUTS all26 historical evidence byte hashes')
    _old.verify_inputs(load(ABLATION_OUTPUT / 'INPUTS.json'))
    _bias_verifier.verify_inputs(load(BIAS_OUTPUT / 'INPUTS.json'))
    for source, observations, tasks, decisions in ((ABLATION_OUTPUT, 54, 486, 2916), (BIAS_OUTPUT, 162, 1458, 5832)):
        previous = load(source / 'verification/VERIFY.json')
        status = load(source / 'RUN_STATUS.json')
        require(previous.get('status') == 'PASS' and previous.get('host') == 'liekkas' and previous.get('output') == str(source) and
                previous.get('observations') == observations and previous.get('tasks') == tasks and previous.get('decisions') == decisions and
                status.get('status') == 'COMPLETE', 'INPUTS: historical source is not verified at exact completed target')
        history = load(source / 'INPUTS.json')
        require(history['source_root'] == str(BASE_OUTPUT), 'INPUTS: historical observation lineage differs')
        for path, digest in history['files'].items():
            require(inputs.get(path) == digest, f'INPUTS: historical source graph edge differs: {path}')
    return inputs


def verify_budget(budget):
    require(isinstance(budget, dict) and set(budget) == {'started_unix', 'deadline_unix', 'limit_seconds', 'scope'}, 'BUDGET: invalid schema')
    start = finite_seconds(budget['started_unix'], 'BUDGET absolute start')
    deadline = finite_seconds(budget['deadline_unix'], 'BUDGET absolute deadline')
    require(type(budget['limit_seconds']) is int and budget['limit_seconds'] == 5400 and start < deadline <= start + 5400 and
            budget['scope'] == 'main plus independent verification; verification must not reset deadline', 'BUDGET: changed shared absolute window')
    if time.time() >= deadline:
        raise TimeoutError('shared main-plus-verification absolute deadline exhausted')
    return deadline


@contextmanager
def deadline_guard(output):
    deadline = verify_budget(load(Path(output) / 'BUDGET.json'))
    require(not signal.getitimer(signal.ITIMER_REAL)[0], 'BUDGET: refuse replacing existing process deadline')
    def reached(_signum, _frame):
        raise TimeoutError('shared main-plus-verification absolute deadline reached')
    previous = signal.signal(signal.SIGALRM, reached)
    signal.setitimer(signal.ITIMER_REAL, max(0.000001, deadline - time.time()))
    try:
        yield deadline
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def audit_runner_source(source=None):
    """Review the parse fence and public solver signature without executing it.

    This is evidence about the frozen source path, not an OS blindness proof.
    Historical truth bytes are hashed in setup; private JSON is parsed later.
    """
    code = (ROOT / 'run_bridge.py').read_text(encoding='utf-8') if source is None else source
    tree = ast.parse(code)
    functions = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    require({'_run', 'private_truths', 'execute_shared', 'prepare_reuse'} <= set(functions), 'source audit: missing expected scientific functions')
    require([arg.arg for arg in functions['execute_shared'].args.args] ==
            ['observation', 'radius', 'epsilon', 'rounds', 'max_boxes', 'tolerance'], 'source audit: solver receives private/unit metadata')
    run = functions['_run']
    truth_calls = [node for node in ast.walk(run) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'private_truths']
    require(len(truth_calls) == 1, 'source audit: private fact parsing must occur once')
    parse_line = truth_calls[0].lineno
    gates = [node for node in run.body if isinstance(node, ast.If) and
             any(isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == 'load' and
                 any(isinstance(value, ast.Constant) and value.value == 'decisions.json' for value in ast.walk(call))
                 for call in ast.walk(node.test))]
    require(len(gates) == 1 and gates[0].end_lineno < parse_line and
            any(isinstance(node, ast.Raise) for node in gates[0].body), 'source audit: complete actions not reread before private parsing')
    expected_gate = ast.parse("len(decisions) != 7776 or len({(row['task_id'], row['method']) for row in decisions}) != 7776 or load(output / 'decisions.json') != decisions", mode='eval').body
    require(ast.dump(gates[0].test, include_attributes=False) == ast.dump(expected_gate, include_attributes=False),
            'source audit: complete unique7776-action readback gate changed')
    counts = [node for node in run.body if isinstance(node, ast.If) and node.lineno > gates[0].lineno and node.end_lineno < parse_line and
              any(isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == 'Counter' for call in ast.walk(node.test))]
    expected_counts = ast.parse("Counter((row['method'], row['reused']) for row in decisions) != {('keep', False): 3888, ('shared', False): 2916, ('shared', True): 972}", mode='eval').body
    require(len(counts) == 1 and any(isinstance(node, ast.Raise) for node in counts[0].body) and
            ast.dump(counts[0].test, include_attributes=False) == ast.dump(expected_counts, include_attributes=False),
            'source audit: new/reused/KEEP population fence changed')
    private_loads = [node for node in ast.walk(functions['private_truths']) if isinstance(node, ast.Call) and
                     isinstance(node.func, ast.Name) and node.func.id == 'load']
    require(len(private_loads) == 1 and ast.dump(private_loads[0].args[0], include_attributes=False) ==
            ast.dump(ast.parse("BASE / 'evaluation/truth.json'", mode='eval').body, include_attributes=False),
            'source audit: private facts do not come from frozen BASE identity')
    for name in ('_run', 'prepare_reuse', 'execute_shared'):
        require(not any(isinstance(node, ast.Constant) and isinstance(node.value, str) and 'truth.json' in node.value
                        for node in ast.walk(functions[name])), f'source audit: direct private path in {name}')
    callers = [(function.name, node.lineno) for function in functions.values() for node in ast.walk(function)
               if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'private_truths']
    require(callers == [('_run', parse_line)], 'source audit: private facts have an additional call path')
    return {'status': 'PASS', 'runner': str(ROOT / 'run_bridge.py'), 'private_parse_line': parse_line,
            'complete_unique_action_readback_gate_line': gates[0].lineno, 'action_origin_count_gate_line': counts[0].lineno,
            'solver_arguments': ['observation', 'radius', 'epsilon', 'rounds', 'max_boxes', 'tolerance'],
            'scope': 'frozen source parse-order review; historical input bytes may be hashed in setup; no OS blindness claim'}


def reconstruct_population(base_observations, base_tasks, base_truths):
    observations, tasks, truths = [], [], []
    for unit, _image_mode, bound, rounds, reused in UNITS:
        for base_id, original in base_observations.items():
            public = {'observation_id': f'{unit}-{base_id}', 'base_observation_id': base_id,
                      'unit_id': unit, 'bound': str(bound), 'rounds': rounds, 'reused': reused,
                      'radius': str(RADIUS), 'observation': list(map(encoded, transformed_image(original['observation'], unit)))}
            observations.append(public)
            truths.append({**base_truths[base_id], **{key: public[key] for key in META}})
        tasks.extend({'task_id': f'{unit}-{task["task_id"]}', 'observation_id': f'{unit}-{task["observation_id"]}',
                      'base_task_id': task['task_id'], 'incumbent': task['incumbent']} for task in base_tasks.values())
    require((len(observations), len(tasks), len(truths)) == (432, 3888, 432), 'population: incomplete frozen source replay')
    return observations, tasks, truths


def verify_population(output, model_class):
    base_observations, base_tasks, base_truths = _old.verify_frozen_population(model_class)
    observations, tasks, truths = reconstruct_population(base_observations, base_tasks, base_truths)
    same(load(output / 'observations.json'), observations, 'all432 independently transformed raw/clipped/biased images')
    same(load(output / 'tasks.json'), tasks, 'all3888 frozen task offsets and observation identities')
    same(load(output / 'evaluation/truth.json'), truths, 'all432 actual facts and independent lineage')
    residuals = []
    for public, private in zip(observations, truths):
        residual = validate_residual(base_observations[public['base_observation_id']]['observation'],
                                     public['observation'], private, public['unit_id'], model_class, public['observation_id'])
        residuals.append({'observation_id': public['observation_id'], 'unit_id': public['unit_id'],
                          'actual_noise_linf': str(residual), 'declared_bound': public['bound']})
    return (indexed(observations, lambda row: row['observation_id'], 'observations'),
            indexed(tasks, lambda row: row['task_id'], 'tasks'),
            indexed(truths, lambda row: row['observation_id'], 'truths'), residuals, base_observations, base_tasks)


def make_row(task, private, decision, checked):
    oldrow = _old.make_row(task, private, decision, checked)
    row = {'task_id': task['task_id'], 'base_task_id': task['base_task_id'], **{key: private[key] for key in META},
           'reused': decision['reused']}
    row.update({key: oldrow[key] for key in ('case', 'seed', 'method', 'state_group', 'prior_position', 'incumbent', 'output',
                                           'status', 'initial_ae_mm', 'ae_mm', 'se_mm2', 'actual_gain_mm2', 'covered', 'hull_width_mm')})
    hull = checked['hull']
    row.update(hull_lo_mm=None if hull is None else str(hull[0]), hull_hi_mm=None if hull is None else str(hull[1]),
               retained_branch_count=oldrow['retained_branch_count'], retained_interval_count=len(checked['intervals']))
    row.update({key: oldrow[key] for key in ('evaluated_boxes', 'unresolved_budget_leaves', 'auxiliary_samples_used')})
    return row


def reconstruct_aggregates(rows):
    result = []
    for unit, _mode, bound, rounds, reused in UNITS:
        result.extend({**record, 'unit_id': unit, 'bound': str(bound), 'rounds': rounds,
                       'reused': reused and record['method'] == 'shared'}
                      for record in _old.reconstruct_aggregates([row for row in rows if row['unit_id'] == unit]))
    return result


def group_mean_sd(aggregates):
    result = []
    for unit in UNIT_CONFIGS:
        for method in METHODS:
            for state in ('all', 'correct', 'small'):
                for grouping, count in (('by_case', 6), ('by_seed', 3)):
                    values = [row for row in aggregates if row['unit_id'] == unit and row['method'] == method and
                              row['state_group'] == state and row['grouping'] == grouping]
                    require(len(values) == count, f'{unit}/{method}/{state}: incomplete grouped mean/SD denominator')
                    result.append({'unit_id': unit, 'method': method, 'state_group': state, 'grouping': grouping,
                                   'groups': count, 'ddof': 1,
                                   'metrics': {metric: {'mean': statistics.mean(row[metric] for row in values),
                                                        'sd': statistics.stdev(row[metric] for row in values)}
                                               for metric in ('mae_mm', 'mse_mm2', 'improved', 'harm')}})
    return result


def paired_rows(rows):
    matrix = indexed(rows, lambda row: (row['unit_id'], row['base_task_id'], row['method']), 'paired evaluation')
    base_ids = sorted({row['base_task_id'] for row in rows})
    require(len(base_ids) == 486 and len(matrix) == 7776, 'paired evaluation: incomplete base task/arm population')
    comparisons = [(unit, 'keep', unit, 'shared') for unit in UNIT_CONFIGS]
    comparisons.extend((left, 'shared', right, 'shared') for left, right in CONTRASTS)
    pairs, interactions = [], []
    for task_id in base_ids:
        require(all((unit, task_id, method) in matrix for unit in UNIT_CONFIGS for method in METHODS),
                f'paired evaluation: missing task/arm {task_id}')
        reference = matrix['CBdelta', task_id, 'shared']
        for left_unit, left_method, right_unit, right_method in comparisons:
            left, right = matrix[left_unit, task_id, left_method], matrix[right_unit, task_id, right_method]
            same({key: left[key] for key in ('case', 'seed', 'state_group')},
                 {key: right[key] for key in ('case', 'seed', 'state_group')}, 'paired actual task lineage')
            a, b = exact(left['ae_mm']), exact(right['ae_mm'])
            pairs.append({'base_task_id': task_id, **{key: right[key] for key in ('case', 'seed', 'state_group')},
                          'left_unit': left_unit, 'left_method': left_method, 'right_unit': right_unit, 'right_method': right_method,
                          'left_ae_mm': str(a), 'right_ae_mm': str(b), 'mae_gain_mm': str(a-b), 'mse_gain_mm2': str(a*a-b*b),
                          'better': b < a, 'same': b == a, 'worse': b > a})
        no_bias = exact(matrix['C0delta', task_id, 'shared']['ae_mm']) - exact(matrix['R0delta', task_id, 'shared']['ae_mm'])
        with_bias = exact(matrix['CBdelta', task_id, 'shared']['ae_mm']) - exact(matrix['RBdelta', task_id, 'shared']['ae_mm'])
        interactions.append({'base_task_id': task_id, **{key: reference[key] for key in ('case', 'seed', 'state_group')},
                             'clipping_error_change_no_bias_mm': str(no_bias), 'clipping_error_change_with_bias_mm': str(with_bias),
                             'difference_in_error_changes_mm': str(with_bias-no_bias),
                             'formula': '(AE_CBdelta-AE_RBdelta)-(AE_C0delta-AE_R0delta)'})
    return pairs, interactions


def verify_csv(path, records, json_columns=()):
    require(records and all(list(row) == list(records[0]) for row in records), f'{path.name}: inconsistent expected CSV schema')
    with Path(path).open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == list(records[0]), f'{path.name}: changed CSV column order/schema')
        actual = list(reader)
    expected = [{key: (json.dumps(value) if key in json_columns and isinstance(value, (dict, list))
                      else '' if value is None else str(value)) for key, value in row.items()} for row in records]
    same(actual, expected, f'{path.name}: every independently reconstructed CSV cell')


def verify_provenance(output, observations, tasks, outers, decisions, base_observations, base_tasks):
    histories = {unit: indexed(load(source / 'outers.json'), lambda row: row['observation_id'], f'{unit} historical outers')
                 for unit, source in (('R0epsilon', ABLATION_OUTPUT), ('CBdelta', BIAS_OUTPUT))}
    images = {'R0epsilon': base_observations,
              'CBdelta': indexed(load(BIAS_OUTPUT / 'observations.json'), lambda row: row['observation_id'], 'historical bias images')}
    historical_tasks = {'R0epsilon': base_tasks,
                        'CBdelta': indexed(load(BIAS_OUTPUT / 'tasks.json'), lambda row: row['task_id'], 'historical bias tasks')}
    actions, timings = {}, {}
    for unit, source in (('R0epsilon', ABLATION_OUTPUT), ('CBdelta', BIAS_OUTPUT)):
        method = reuse_source(unit, 'o000')['method']
        actions[unit] = indexed([row for row in load(source / 'decisions.json') if row['method'] == method],
                                lambda row: row['task_id'], f'{unit} historical actions')
        timings[unit] = indexed(load(source / 'TIMING.json')['observations'], lambda row: row['observation_id'], f'{unit} historical timing')
    provenance = []
    for public in observations.values():
        oid, unit = public['observation_id'], public['unit_id']
        if not public['reused']:
            require(outers[oid]['reused_source'] is None, f'{oid}: fabricated reuse lineage for new solve')
            continue
        source = verify_reused_arm(outers[oid]['arms']['shared'], unit, public['base_observation_id'], histories,
                                   public['observation'], images)
        same(outers[oid]['reused_source'], source, f'{oid}: exact reuse source mapping')
        original_image = images[unit][source['observation_id']]
        original_time = timings[unit][source['observation_id']]
        same(original_time['outer_seconds_by_method'][source['method']], outers[oid]['arms']['shared']['outer_seconds'],
             f'{oid}: historical solve cost versus source timing')
        mappings = []
        for state in range(9):
            task = tasks[f'{oid}-s{state}']
            source_tid = task['base_task_id'] if unit == 'R0epsilon' else f'b1-{task["base_task_id"]}'
            original_task, original_action = historical_tasks[unit][source_tid], actions[unit][source_tid]
            require(original_task['observation_id'] == source['observation_id'] and original_task['incumbent'] == task['incumbent'] and
                    (unit != 'CBdelta' or original_task['base_task_id'] == task['base_task_id']), f'{oid}: false reused task lineage')
            core = {key: value for key, value in decisions[task['task_id'], 'shared'].items()
                    if key not in {'task_id', 'method', 'unit_id', 'bound', 'rounds', 'reused'}}
            same(core, {key: value for key, value in original_action.items() if key not in {'task_id', 'method'}},
                 f'{oid}: historical action body differs')
            mappings.append({'task_id': task['task_id'], 'source_task_id': source_tid, 'action_sha256': json_hash(original_action)})
        finite_seconds(original_time['selector_seconds_by_method'][source['method']], f'{oid}: historical selector time')
        provenance.append({**{key: public[key] for key in META}, 'source': source,
                           'observation_sha256': json_hash(original_image), 'outer_sha256': json_hash(outers[oid]['arms']['shared']),
                           'source_selector_seconds': original_time['selector_seconds_by_method'][source['method']], 'tasks': mappings})
    require(len(provenance) == 108 and sum(len(row['tasks']) for row in provenance) == 972, 'REUSE: incomplete historical population')
    same(load(output / 'REUSE.json'), provenance, 'all108 independent provenance records and972 historical action hashes')
    return len(provenance)


def verify_timing(output, outers, summary, budget):
    timing = load(output / 'TIMING.json')
    require(isinstance(timing, dict) and set(timing) == {'observations', 'setup_seconds', 'cache_policy', 'budget_deadline_unix'}, 'TIMING: invalid schema')
    same(timing['cache_policy'], CACHE_POLICY, 'TIMING cold-cache policy')
    same(timing['budget_deadline_unix'], budget['deadline_unix'], 'TIMING same absolute deadline')
    require(finite_seconds(timing['setup_seconds'], 'TIMING setup overhead') <= summary['elapsed_seconds'], 'TIMING setup exceeds main duration')
    same(load(output / 'TIMING_PARTIAL.json'), {key: timing[key] for key in ('observations', 'setup_seconds')}, 'TIMING complete checkpoint')
    records = indexed(timing['observations'], lambda row: row['observation_id'], 'TIMING')
    require(list(records) == list(outers), 'TIMING: incomplete or changed image order')
    for oid, record in records.items():
        require(set(record) == set(META) | {'new_solver_seconds_by_method', 'historical_outer_seconds_by_method', 'selector_seconds_by_method'}, f'{oid}: invalid timing record schema')
        same({key: record[key] for key in META}, {key: outers[oid][key] for key in META}, f'{oid}: timing lineage')
        for field in ('new_solver_seconds_by_method', 'historical_outer_seconds_by_method', 'selector_seconds_by_method'):
            require(list(record[field]) == list(METHODS), f'{oid}: changed timing method identities/order')
            for method in METHODS:
                finite_seconds(record[field][method], f'{oid}/{method}/{field}: measured seconds')
        for method in METHODS:
            total = outers[oid]['arms'][method]['outer_seconds']
            same(record['new_solver_seconds_by_method'][method], 0. if record['reused'] or method == 'keep' else total,
                 f'{oid}/{method}: new versus reused cost attribution')
            same(record['historical_outer_seconds_by_method'][method], total if record['reused'] and method == 'shared' else 0.,
                 f'{oid}/{method}: historical cost attribution')
    costs = []
    for unit in UNIT_CONFIGS:
        subset = [row for row in records.values() if row['unit_id'] == unit]
        for method in METHODS:
            new = sum(row['new_solver_seconds_by_method'][method] for row in subset)
            action = sum(row['selector_seconds_by_method'][method] for row in subset)
            values = [row['arms'][method]['outer'] for row in outers.values() if row['unit_id'] == unit]
            costs.append({'unit_id': unit, 'method': method, 'reused': UNIT_CONFIGS[unit]['reused'] and method == 'shared',
                          'new_solver_seconds': new, 'selector_seconds': action, 'new_total_seconds': new+action,
                          'historical_outer_seconds': sum(row['historical_outer_seconds_by_method'][method] for row in subset),
                          'evaluated_boxes': sum(row['evaluated_boxes'] for row in values),
                          'unresolved_budget_leaves': sum(row['unresolved_budget_leaves'] for row in values)})
    same(summary['per_unit_method_timing'], costs, 'all16 independently summed new/reused costs')
    verify_csv(output / 'evaluation/TIMING.csv', costs)
    require(timing['setup_seconds'] + sum(row['new_total_seconds'] for row in costs) <= summary['elapsed_seconds'],
            'TIMING: recorded sequential setup/new costs exceed complete main duration')


def verify_evaluation(output, rows, coverage, outers, status, budget):
    require((len(rows), len(coverage)) == (7776, 864), 'evaluation: incomplete population')
    same(load(output / 'evaluation/ROWS.json'), rows, 'all7776 independent actual-truth evaluation records')
    verify_csv(output / 'evaluation/ROWS.csv', rows)
    same(load(output / 'evaluation/COVERAGE.json'), coverage, 'all864 retained-set and branch truth coverage records')
    verify_csv(output / 'evaluation/COVERAGE.csv', coverage, ('hull', 'retained_intervals', 'retained_branches', 'branch_coverage'))
    pairs, interactions = paired_rows(rows)
    require((len(pairs), len(interactions)) == (8262, 486), 'evaluation: wrong paired/interaction denominators')
    for name, expected in (('PAIRED', pairs), ('INTERACTION', interactions)):
        same(load(output / f'evaluation/{name}.json'), expected, f'{name}: independent endpoint error contrasts')
        verify_csv(output / f'evaluation/{name}.csv', expected)
    summary = load(output / 'evaluation/SUMMARY.json')
    require(isinstance(summary, dict) and set(summary) == {'observation_configurations', 'task_count', 'decision_count',
            'new_shared_actions', 'reused_shared_actions', 'keep_actions', 'rows', 'group_mean_sd', 'per_unit_method_timing',
            'elapsed_seconds', 'sampling_note', 'timing_scope'}, 'SUMMARY: invalid complete schema')
    for key, value in {'observation_configurations': 432, 'task_count': 3888, 'decision_count': 7776,
                       'new_shared_actions': 2916, 'reused_shared_actions': 972, 'keep_actions': 3888,
                       'sampling_note': SAMPLING_NOTE, 'timing_scope': TIMING_SCOPE}.items():
        same(summary[key], value, f'SUMMARY {key}')
    require(finite_seconds(summary['elapsed_seconds'], 'SUMMARY elapsed') <= status['elapsed_seconds'], 'SUMMARY elapsed exceeds complete main duration')
    aggregates = reconstruct_aggregates(rows)
    require(len(aggregates) == 1440, 'SUMMARY: wrong grouped endpoint population')
    same(summary['rows'], aggregates, 'all1440 independently reconstructed count/error/coverage/cost records')
    same(summary['group_mean_sd'], group_mean_sd(aggregates), 'all96 independent grouped means and sample standard deviations')
    verify_csv(output / 'evaluation/SUMMARY.csv', aggregates, ('condition',))
    verify_timing(output, outers, summary, budget)
    return aggregates, pairs, interactions


def safety_claims(rows, coverage):
    result = {}
    for unit in UNIT_CONFIGS:
        result[unit] = {}
        for method in METHODS:
            values = [row for row in rows if row['unit_id'] == unit and row['method'] == method]
            correct = [row for row in values if row['state_group'] == 'correct']
            offsets = [row for row in values if row['state_group'] == 'small']
            negative = [row for row in offsets if row['case'] == 'equal_full_negative']
            cov = [row for row in coverage if row['unit_id'] == unit and row['method'] == method]
            require(len(values) == 486 and len(correct) == 54 and len(offsets) == 432 and len(negative) == 72 and len(cov) == 54,
                    f'{unit}/{method}: incorrect population/state/control denominators')
            counts = {'covered': sum(row['covered'] for row in cov),
                      'correct_kept': sum(row['status'] == 'KEEP' and exact(row['ae_mm']) == 0 for row in correct),
                      'negative_small_kept': sum(row['status'] == 'KEEP' for row in negative),
                      'harm': sum(exact(row['ae_mm']) > exact(row['initial_ae_mm']) for row in values),
                      'kept': sum(row['status'] == 'KEEP' for row in values),
                      'incompatible': sum(row['status'] == 'INCOMPATIBLE' for row in values),
                      'unresolved_budget_leaves': sum({row['observation_id']: row['unresolved_budget_leaves']
                                                      for row in values}.values())}
            require(counts['covered'] == 54 and counts['correct_kept'] == 54 and counts['negative_small_kept'] == 72 and
                    counts['harm'] == 0 and counts['unresolved_budget_leaves'] == 0,
                    f'{unit}/{method}: registered coverage/KEEP/safety/complete budget failure')
            result[unit][method] = counts
    return {'contract_safety': result,
            'scientific_policy': 'C1 paired effects and C2 local round stability are measured outcomes, not mechanical PASS criteria',
            'same_source_replay_not_independent_scenes': True}


def verify(output):
    output = Path(output).resolve()
    require(socket.gethostname() == 'liekkas' and ROOT == EXPECTED_ROOT and ROOT.resolve() == EXPECTED_ROOT and
            output.is_dir() and output != ROOT and output.is_relative_to(ROOT), 'target: exact existing bridge output on liekkas required')
    budget = load(output / 'BUDGET.json')
    deadline = verify_budget(budget)
    sources = verify_source_lock(load(output / 'SOURCE_LOCK.json'))
    inputs = verify_inputs(load(output / 'INPUTS.json'))
    same(load(output / 'PROTOCOL.json'), registration(), 'independent clipping/bound/round configuration')
    source_audit = audit_runner_source()
    status = load(output / 'RUN_STATUS.json')
    require(isinstance(status, dict) and set(status) == {'status', 'observations_complete', 'decisions_complete',
            'new_shared_observations', 'reused_shared_observations', 'deadline_unix', 'elapsed_seconds', 'scientific_support'},
            'RUN_STATUS: invalid complete schema')
    same({key: status[key] for key in ('status', 'observations_complete', 'decisions_complete', 'new_shared_observations',
                                     'reused_shared_observations', 'deadline_unix', 'scientific_support')},
         {'status': 'COMPLETE', 'observations_complete': 432, 'decisions_complete': 7776, 'new_shared_observations': 324,
          'reused_shared_observations': 108, 'deadline_unix': deadline,
          'scientific_support': 'requires separate independent verification and paired claim analysis'}, 'RUN_STATUS complete source/action counts and shared budget')
    elapsed = finite_seconds(status['elapsed_seconds'], 'RUN_STATUS elapsed')
    require(elapsed <= deadline - budget['started_unix'] and elapsed <= time.time() - budget['started_unix'],
            'RUN_STATUS: recorded main runtime exceeds shared deadline/actual elapsed window')
    model_class, geometry_bounds = _historical.shared_kernels()
    observations, tasks, truths, residuals, base_observations, base_tasks = verify_population(output, model_class)
    outer_list, decision_list = load(output / 'outers.json'), load(output / 'decisions.json')
    require(isinstance(outer_list, list) and len(outer_list) == 432 and isinstance(decision_list, list) and len(decision_list) == 7776,
            'population: incomplete outer/action records')
    outers = indexed(outer_list, lambda row: row['observation_id'], 'outers')
    decisions = indexed(decision_list, lambda row: (row['task_id'], row['method']), 'decisions')
    require(list(outers) == list(observations), 'outers: changed image identity/order')
    expected_actions = [(f'{oid}-s{state}', method) for oid in observations for method in METHODS for state in range(9)]
    require(list(decisions) == expected_actions and set(tasks) == {task for task, _method in expected_actions},
            'decisions: changed complete task/method identities/order')
    checkpoints = output / 'checkpoint'
    require(checkpoints.is_dir() and sorted(path.name for path in checkpoints.iterdir()) == sorted(f'{oid}.json' for oid in observations),
            'checkpoint: incomplete or additional image artifacts')
    unions, branches = {}, {}
    calls = witnesses = certificates = updates = leaves = group_tests = 0
    for index, (oid, public) in enumerate(observations.items()):
        unit, record, image = public['unit_id'], outers[oid], tuple(map(exact, public['observation']))
        require(isinstance(record, dict) and set(record) == set(META) | {'arms', 'reused_source'} and
                isinstance(record['arms'], dict) and list(record['arms']) == list(METHODS), f'{oid}: invalid outer/arm schema')
        same({key: record[key] for key in META}, {key: public[key] for key in META}, f'{oid}: exact outer contract/lineage')
        same(load(checkpoints / f'{oid}.json'), record, f'{oid}: final outer/checkpoint equality')
        unions[oid], branches[oid] = {}, {}
        for method in METHODS:
            arm, where = record['arms'][method], f'{oid}/{method}'
            require(isinstance(arm, dict) and set(arm) == {'branches', 'outer', 'branch_seconds', 'outer_seconds'}, f'{where}: invalid arm schema')
            allowed = () if method == 'keep' else BRANCHES
            require(isinstance(arm['branches'], dict) and isinstance(arm['branch_seconds'], dict) and
                    list(arm['branches']) == list(arm['branch_seconds']) == list(allowed), f'{where}: changed whole-scene branch identities/order')
            total = finite_seconds(arm['outer_seconds'], f'{where}: solve seconds')
            measured = sum(finite_seconds(value, f'{where}: branch seconds') for value in arm['branch_seconds'].values())
            require(total >= measured and (method != 'keep' or total == 0), f'{where}: inconsistent measured branch cost')
            checked_branches = {}
            for branch in allowed:
                outer = arm['branches'][branch]
                model = model_class(branch, camera_radius=RADIUS, colors=((Q(0), Q(1)),) * 3)
                checked, n_calls, n_witnesses = verify_branch(outer, image, unit, branch, geometry_bounds, model, f'{where}/{branch}')
                calls += n_calls
                witnesses += n_witnesses
                certificates += len(checked['shared_excluded'])
                updates += checked['proof_updates']
                leaves += sum(node['status'] != 'split' for node in outer['trace'])
                group_tests += outer['shared_tests']
                checked_branches[branch] = checked
            branches[oid][method] = checked_branches
            unions[oid][method] = verify_union(arm['outer'], checked_branches, unit, method, where)
        if index % 18 == 17:
            print(json.dumps({'stage': 'independent_tree_verification', 'observations_complete': index + 1,
                              'remaining_shared_seconds': round(deadline - time.time(), 1)}), flush=True)
    reused_count = verify_provenance(output, observations, tasks, outers, decisions, base_observations, base_tasks)
    rows, coverage = [], []
    for decision in decision_list:
        task, method = tasks[decision['task_id']], decision['method']
        oid = task['observation_id']
        private, checked = truths[oid], unions[oid][method]
        result = verify_decision(decision, task, checked, private['unit_id'], f'{task["task_id"]}/{method}')
        row = make_row(task, private, decision, checked)
        if method == 'shared' and checked['hull'] is not None and row['covered']:
            truth, incumbent = exact(private['truth']), exact(task['incumbent'])
            require((incumbent - truth) ** 2 - (result - truth) ** 2 >= exact(decision['gain_lower']) >= 0,
                    f'{task["task_id"]}/{method}: endpoint certificate fails actual covered truth')
        rows.append(row)
    for oid, private in truths.items():
        truth = exact(private['truth'])
        for method in METHODS:
            checked, branch_checks = unions[oid][method], branches[oid][method]
            hull = checked['hull']
            coverage.append({**{key: private[key] for key in META}, 'reused': private['reused'] and method == 'shared',
                             'case': private['case'], 'seed': private['seed'], 'method': method,
                             'covered': _historical.contains(checked['intervals'], truth), 'incompatible': hull is None,
                             'hull': None if hull is None else list(map(encoded, hull)),
                             'retained_intervals': [list(map(encoded, interval)) for interval in checked['intervals']],
                             'hull_width_mm': None if hull is None else str(hull[1] - hull[0]),
                             'retained_branches': checked['retained_branches'],
                             'branch_coverage': {branch: _historical.contains(value['intervals'], truth)
                                                 for branch, value in branch_checks.items()}})
    aggregates, pairs, interactions = verify_evaluation(output, rows, coverage, outers, status, budget)
    claims = safety_claims(rows, coverage)
    same(scientific_files(), sources, 'all scientific source hashes stable during independent verification')
    same(historical_inputs(), inputs, 'all26 historical evidence byte hashes stable during independent verification')
    verify_budget(budget)
    return {'status': 'PASS', 'mechanical_verification': 'PASS', 'host': socket.gethostname(), 'output': str(output),
            'observations': 432, 'tasks': 3888, 'decisions': 7776, 'coverage_rows': 864, 'summary_rows': len(aggregates),
            'paired_rows': len(pairs), 'interaction_rows': len(interactions), 'group_mean_sd_rows': 96,
            'new_shared_observations': 324, 'reused_shared_observations': reused_count,
            'new_shared_actions': 2916, 'reused_shared_actions': 972, 'keep_actions': 3888,
            'branch_outer_traces': 1296, 'terminal_partition_leaves': leaves, 'unresolved_budget_leaves': 0,
            'group_exclusion_certificates_verified': certificates, 'exact_propagation_updates_verified': updates,
            'group_tests_accounted': group_tests, 'historical_kernel_boxes_recomputed': calls, 'channel_witnesses_recomputed': witnesses,
            'source_files_verified': len(sources), 'historical_inputs_verified': len(inputs), 'source_parse_order_audit': source_audit,
            'actual_truth_residuals': residuals, 'claims': claims,
            'shared_budget': {'started_unix': budget['started_unix'], 'deadline_unix': deadline,
                              'elapsed_main_plus_verification_seconds': time.time() - budget['started_unix']},
            'limitations': ['Historical point/enclosure/area kernels are shared; physical geometry soundness is not independently proved.',
                            'Necessary-constraint exclusions are replayed; retained boxes are not guaranteed jointly feasible.',
                            'Unrecorded no-contradiction propagation on retained boxes is not replayed.',
                            'Recorded cold-cache/historical timing is checked and summed, not independently reproduced.',
                            'Frozen source parse-order and hash review does not prove operating-system blindness.',
                            'This reuses the same54 synthetic source observations, not independent scenes or real-MVS transfer.',
                            'Clipping, bound widening, bias and local4/8/16-round effects are separate measured paired outcomes; mechanical PASS does not establish their sign or exact feasibility.']}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Independently verify the frozen observation bridge evidence.')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/bridge-v1')
    output = parser.parse_args(argv).output.resolve()
    if socket.gethostname() != 'liekkas' or ROOT != EXPECTED_ROOT or ROOT.resolve() != EXPECTED_ROOT:
        parser.error('exact local observation-bridge root on liekkas required')
    if not output.is_dir() or output == ROOT or not output.is_relative_to(ROOT):
        parser.error('existing output inside this exact experiment required')
    destination = output / 'verification'
    if destination.exists():
        print('REFUSE: existing verification directory is never overwritten', file=sys.stderr)
        return 2
    try:
        destination.mkdir()
    except FileExistsError:
        print('REFUSE: concurrent verifier created verification directory', file=sys.stderr)
        return 2
    started = time.perf_counter()
    try:
        with deadline_guard(output):
            report = verify(output)
            report.update(utc=datetime.now(timezone.utc).isoformat(), verification_seconds=time.perf_counter() - started)
            # A PASS artifact is finalized under the same alarm as the checks.
            # A deadline interrupted partial artifact is preserved, never replaced.
            with (destination / 'VERIFY.json').open('x', encoding='utf-8') as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write('\n')
    except Exception as error:
        if (destination / 'VERIFY.json').exists():
            print(f'FAIL: partial verification artifact preserved: {type(error).__name__}: {error}', file=sys.stderr)
            return 1
        report = {'status': 'INCOMPLETE' if isinstance(error, TimeoutError) else 'FAIL',
                  'mechanical_verification': 'INCOMPLETE' if isinstance(error, TimeoutError) else 'FAIL',
                  'host': socket.gethostname(), 'output': str(output), 'error': f'{type(error).__name__}: {error}'}
        report.update(utc=datetime.now(timezone.utc).isoformat(), verification_seconds=time.perf_counter() - started)
        with (destination / 'VERIFY.json').open('x', encoding='utf-8') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())

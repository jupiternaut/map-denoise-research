#!/usr/bin/env python3
"""Independent exact verification of a bounded view-bias same-input replay.

Imports only frozen independent verification helpers and historical geometry
kernels, never the new runner or algorithm. Historical module constants are
scoped in memory and restored; no historical source or evidence is modified.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from fractions import Fraction as Q
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import time

sys.dont_write_bytecode = True
for _view_verifier_threads in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_view_verifier_threads] = '1'

ROOT = Path(__file__).resolve().parent
ABLATION_ROOT = ROOT.parent / 'lovelace-shared-ablation-20261011T055200Z'
ABLATION_OUTPUT = ABLATION_ROOT / 'outputs/ablation-v1'
BASE_ROOT = ROOT.parent / 'lovelace-shared-appearance-20261011T052300Z'
BASE_OUTPUT = BASE_ROOT / 'outputs/confirmation-v1'
_spec = importlib.util.spec_from_file_location('_view_bias_frozen_independent_ablation', ABLATION_ROOT / 'verify_ablation.py')
_old = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_old)
_historical, _arithmetic = _old._historical, _old._arithmetic
VerificationError = _old.VerificationError
exact, encoded, pair = _old.exact, _old.encoded, _old.pair
require, same, load, indexed = _old.require, _old.same, _old.load, _old.indexed
finite_seconds = _old.finite_seconds
DOMAIN, TOLERANCE, RADIUS, ROUNDS, MAX_BOXES = _old.DOMAIN, _old.TOLERANCE, Q(1, 5), 8, 511
BASE_EPSILON = Q(1, 100)
BETAS = (Q(1, 200), Q(1, 100), Q(1, 50))
METHODS = ('keep', 'channel_wide', 'global_shared_wide', 'shared_strict')
BRANCHES = ('none', 'partial', 'full')
SAMPLING_NOTE = 'same 54 synthetic observations and related states replayed at three beta values; not independent scenes'
CACHE_POLICY = 'clear all geometry/model caches before each arm/observation; reuse within its three branches'


def method_contract(method, beta):
    beta = exact(beta)
    require(beta in BETAS and method in METHODS, 'contract: unregistered beta or method')
    return ('channel' if method == 'channel_wide' else 'global_gray',
            BASE_EPSILON if method in ('keep', 'shared_strict') else BASE_EPSILON + beta,
            method != 'shared_strict')


def biased_image(base_image, beta):
    """Three contiguous 144-channel views; directly clip noisy base plus bias."""
    beta = exact(beta)
    require(beta in BETAS, 'bias generation: unregistered public beta')
    image = tuple(exact(value) for value in base_image)
    require(len(image) == 432, 'bias generation: expected three views of48 RGB pixels')
    offsets = (-beta, Q(0), beta)
    return tuple(min(Q(1), max(Q(0), value + offsets[index // 144]))
                 for index, value in enumerate(image))


def necessary_rows(observation, geometry, epsilon):
    """Independent global grayscale rows with an explicit per-arm error bound."""
    epsilon = exact(epsilon)
    require(epsilon >= 0, 'necessary rows: negative public epsilon')
    image = tuple(exact(value) for value in observation)
    require(len(image) == 432 and isinstance(geometry, (tuple, list)) and len(geometry) == 144,
            'necessary rows: expected fixed three-view RGB layout')
    rows = []
    for pixel, endpoints in enumerate(geometry):
        require(isinstance(endpoints, (list, tuple)) and len(endpoints) == 6, 'necessary rows: invalid area layout')
        coefficients = tuple(exact(value) for value in endpoints)
        require(all(0 <= coefficients[j] <= coefficients[j + 1] <= 1 for j in (0, 2, 4)),
                'necessary rows: invalid normalized area bounds')
        rgb = image[3 * pixel:3 * pixel + 3]
        rows.append(('upper', coefficients[::2], min(1 - value + epsilon for value in rgb)))
        rows.append(('lower', coefficients[1::2], max(1 - value - epsilon for value in rgb)))
    return tuple(rows)


@contextmanager
def historical_contract(epsilon):
    """Coordinate every reused helper's bound, restoring all memory afterward.

    The old tree/union checker reads _old.EPSILON; the historical enclosure
    witness checker reads _old._historical.EPSILON. Certificate row construction
    is replaced by this module's explicit-epsilon arithmetic, not solver code.
    """
    epsilon = exact(epsilon)
    saved = (_old.EPSILON, _historical.EPSILON, _old.verify_group_certificate)
    def independent_certificate(observation, geometry, mode, group_id, certificate, where):
        require(mode == 'global_gray' and group_id == 'global', f'{where}: wrong global material group')
        return _historical.verify_certificate(necessary_rows(observation, geometry, epsilon), certificate, where)
    _old.EPSILON, _historical.EPSILON = epsilon, epsilon
    _old.verify_group_certificate = independent_certificate
    try:
        yield
    finally:
        _old.EPSILON, _historical.EPSILON, _old.verify_group_certificate = saved


def verify_branch(outer, image, beta, method, branch, geometry_bounds, model, where):
    mode, epsilon, _valid = method_contract(method, beta)
    require(method != 'keep', f'{where}: KEEP must not solve branches')
    with historical_contract(epsilon):
        checked = _old.verify_trace(outer, image, where, mode=mode, branch=branch, geometry_bounds=geometry_bounds)
        calls, witnesses = _old.verify_kernel_boxes(outer, checked, image, model, where, mode != 'channel')
    return checked, calls, witnesses


def verify_union(outer, branches, beta, method, where):
    mode, epsilon, _valid = method_contract(method, beta)
    with historical_contract(epsilon):
        return _old.verify_union(outer, branches, where, prior_only=method == 'keep', mode=mode)


def verify_decision(decision, task, checked, where):
    require(decision.get('method') in METHODS, f'{where}: unknown decision method')
    return _arithmetic.verify_decision(decision, task['incumbent'], checked['hull'], where)


def validate_residual(base_image, biased, private, beta, model_class, where):
    beta = exact(beta)
    base = tuple(exact(value) for value in base_image)
    image = tuple(exact(value) for value in biased)
    require(image == biased_image(base, beta), f'{where}: wrong view bias or clipping')
    model = model_class(private['rear_branch'], camera_radius=RADIUS, colors=((Q(0), Q(1)),) * 3)
    prediction = model.point_prediction(exact(private['truth']), tuple(exact(value) for value in private['actual_offsets']),
                                         colors=tuple(exact(value) for value in private['colors']))
    require(len(prediction) == 432 and all(0 <= value <= 1 for value in prediction), f'{where}: invalid noiseless prediction')
    base_residual = max(abs(value - truth) for value, truth in zip(base, prediction))
    residual = max(abs(value - truth) for value, truth in zip(image, prediction))
    require(base_residual <= BASE_EPSILON, f'{where}: base noise exceeds frozen bound')
    require(residual <= BASE_EPSILON + beta, f'{where}: biased actual-truth residual exceeds epsilon+beta')
    return residual


def scientific_files():
    sources = dict(_old.scientific_files())
    for path in sorted({ROOT / 'PROTOCOL.md', *ROOT.glob('*.py')}):
        sources[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(sorted(sources.items()))


def historical_inputs():
    paths = [BASE_OUTPUT / relative for relative in _old.FROZEN_FILES]
    paths.extend(ABLATION_OUTPUT / relative for relative in ('SOURCE_LOCK.json', 'verification/VERIFY.json', 'PROTOCOL.json'))
    return {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def verify_source_lock(lock):
    require(isinstance(lock, dict) and set(lock) == {'files', 'source_root', 'lock_scope', 'utc'}, 'SOURCE_LOCK: invalid schema')
    require(lock['source_root'] == str(ROOT) and
            lock['lock_scope'] == 'inherited frozen ablation sources plus new scientific sources and protocol',
            'SOURCE_LOCK: wrong exact source identity/scope')
    require(type(lock['utc']) is str and datetime.fromisoformat(lock['utc']).utcoffset() is not None,
            'SOURCE_LOCK: timezone-aware timestamp required')
    _old.verify_source_lock(load(ABLATION_OUTPUT / 'SOURCE_LOCK.json'))
    sources = scientific_files()
    same(lock['files'], sources, 'SOURCE_LOCK independent complete historical/new scientific inventory')
    return sources


def verify_inputs(lock):
    inputs = historical_inputs()
    require(isinstance(lock, dict) and set(lock) == {'source_root', 'files'} and lock['source_root'] == str(BASE_OUTPUT),
            'INPUTS: invalid historical output identity/schema')
    same(lock['files'], inputs, 'INPUTS eleven frozen historical output hashes')
    _old.verify_inputs(load(ABLATION_OUTPUT / 'INPUTS.json'))
    previous = load(ABLATION_OUTPUT / 'verification/VERIFY.json')
    require(previous.get('status') == 'PASS' and previous.get('host') == 'liekkas' and
            previous.get('output') == str(ABLATION_OUTPUT) and previous.get('observations') == 54 and
            previous.get('tasks') == 486 and previous.get('decisions') == 2916,
            'INPUTS: prior ablation is not verified at exact identity')
    return inputs


def make_row(task, private, decision, checked):
    row = _old.make_row(task, private, decision, checked)
    result = {key: row[key] for key in ('task_id', 'observation_id')}
    result.update(base_observation_id=private['base_observation_id'], beta=private['beta'])
    result.update({key: row[key] for key in ('case', 'seed', 'method')})
    result['contract_valid'] = decision['method'] != 'shared_strict'
    result.update({key: value for key, value in row.items() if key not in result})
    return result


def reconstruct_aggregates(rows):
    result = []
    for beta in BETAS:
        result.extend({**record, 'beta': str(beta), 'contract_valid': record['method'] != 'shared_strict'} for record in _old.reconstruct_aggregates(
            [row for row in rows if row['beta'] == str(beta)]))
    return result


def registration():
    return {'methods': list(METHODS), 'betas': list(map(str, BETAS)),
            'method_configs': {method: {'mode': 'identity' if method == 'keep' else method_contract(method, BETAS[0])[0],
                'contract_valid': method != 'shared_strict', 'max_boxes': MAX_BOXES, 'rounds': ROUNDS,
                # KEEP does not invoke a noise test; its fixed historical prior
                # outer retains epsilon=.01 while this unused public label is
                # the same widened value declared by the runner.
                'epsilon_by_beta': {str(beta): str(BASE_EPSILON if method == 'shared_strict' else BASE_EPSILON + beta)
                                    for beta in BETAS}} for method in METHODS},
            'epsilon': str(BASE_EPSILON), 'tolerance_mm': str(TOLERANCE), 'max_boxes_per_branch': MAX_BOXES,
            'rounds': ROUNDS, 'views': 3, 'pixels_per_view': 48, 'channels': 3,
            'generator': 'clip(base_noisy_value + (-beta,0,+beta)[view],0,1); no preliminary clip',
            'expected_observations': 162, 'expected_tasks': 1458, 'expected_decisions': 5832,
            'expected_per_beta_per_method': {'observations': 54, 'correct': 54, 'small': 432, 'negative_control_small': 72},
            'allowed_branches': list(BRANCHES), 'cpu_threads': 1, 'gpu_count': 0, 'wall_time_limit_seconds': 5400,
            'source_root': str(BASE_OUTPUT), 'sampling_note': SAMPLING_NOTE, 'cache_policy': CACHE_POLICY,
            'strict_policy': 'intentional contract mismatch: retain all strict coverage failures and harms as diagnostic results'}


def verify_population(output, model_class):
    base_observations, base_tasks, base_truths = _old.verify_frozen_population(model_class)
    observations, tasks, truths, residuals = [], [], [], []
    for index, beta in enumerate(BETAS):
        for base_id, public in base_observations.items():
            oid = f'b{index}-{base_id}'
            image = biased_image(public['observation'], beta)
            observations.append({'observation_id': oid, 'base_observation_id': base_id, 'beta': str(beta),
                                 'radius': str(RADIUS), 'observation': list(map(encoded, image))})
            private = {**base_truths[base_id], 'observation_id': oid, 'base_observation_id': base_id,
                       'beta': str(beta), 'actual_view_bias': list(map(encoded, (-beta, Q(0), beta)))}
            truths.append(private)
            residual = validate_residual(public['observation'], image, private, beta, model_class, oid)
            residuals.append({'observation_id': oid, 'beta': str(beta), 'actual_noise_linf': str(residual),
                              'legal_epsilon': str(BASE_EPSILON + beta)})
        for task in base_tasks.values():
            tasks.append({'task_id': f'b{index}-{task["task_id"]}', 'observation_id': f'b{index}-{task["observation_id"]}',
                          'base_task_id': task['task_id'], 'incumbent': task['incumbent']})
    same(load(output / 'observations.json'), observations, 'all162 independently reconstructed direct biased/clipped images')
    same(load(output / 'tasks.json'), tasks, 'all1458 exact frozen task offsets and beta identities')
    same(load(output / 'evaluation/truth.json'), truths, 'all162 independently reconstructed actual facts and fixed view biases')
    return (indexed(observations, lambda row: row['observation_id'], 'observations'),
            indexed(tasks, lambda row: row['task_id'], 'tasks'),
            indexed(truths, lambda row: row['observation_id'], 'truths'), residuals)


def paired_rows(rows):
    grouped = defaultdict(dict)
    for row in rows:
        require(row['method'] not in grouped[row['task_id']], 'paired rows: duplicate task/method identity')
        grouped[row['task_id']][row['method']] = row
    comparisons = [("keep", method) for method in METHODS[1:]] + [('channel_wide', 'global_shared_wide')]
    result = []
    for arms in grouped.values():
        require(set(arms) == set(METHODS), 'paired rows: missing task arm')
        for baseline, method in comparisons:
            left, right = arms[baseline], arms[method]
            a, b = exact(left['ae_mm']), exact(right['ae_mm'])
            row = {key: right[key] for key in ('task_id', 'observation_id', 'base_observation_id', 'beta', 'case', 'seed', 'state_group')}
            row.update(baseline=baseline, method=method, contract_valid=method != 'shared_strict',
                       baseline_ae_mm=str(a), ae_mm=str(b), mae_gain_mm=str(a - b), mse_gain_mm2=str(a * a - b * b),
                       better=b < a, same=b == a, worse=b > a)
            result.append(row)
    return result


def verify_timing(output, outers, summary):
    timing = load(output / 'TIMING.json')
    require(isinstance(timing, dict) and set(timing) == {'observations', 'cold_cache_note'}, 'TIMING: invalid schema')
    same(timing['cold_cache_note'], CACHE_POLICY, 'TIMING cold-cache declared scope')
    timings = indexed(timing['observations'], lambda row: row['observation_id'], 'TIMING')
    require(list(timings) == list(outers), 'TIMING: changed observation identities/order')
    same(load(output / 'TIMING_PARTIAL.json'), {'observations': timing['observations']}, 'TIMING final checkpoint')
    for oid, record in timings.items():
        require(set(record) == {'observation_id', 'beta', 'outer_seconds_by_method', 'selector_seconds_by_method'} and
                record['beta'] == outers[oid]['beta'], f'{oid}: changed timing schema/beta')
        require(list(record['outer_seconds_by_method']) == list(record['selector_seconds_by_method']) == list(METHODS),
                f'{oid}: changed timing arm identities/order')
        for method in METHODS:
            outer = finite_seconds(record['outer_seconds_by_method'][method], f'{oid}/{method}: solve seconds')
            finite_seconds(record['selector_seconds_by_method'][method], f'{oid}/{method}: action seconds')
            require(outer == outers[oid]['arms'][method]['outer_seconds'], f'{oid}/{method}: inconsistent measured solve time')
    costs = []
    for beta in map(str, BETAS):
        subset = [record for record in timing['observations'] if record['beta'] == beta]
        for method in METHODS:
            outer = sum(record['outer_seconds_by_method'][method] for record in subset)
            action = sum(record['selector_seconds_by_method'][method] for record in subset)
            values = [record['arms'][method]['outer'] for record in outers.values() if record['beta'] == beta]
            costs.append({'beta': beta, 'method': method, 'outer_seconds': outer, 'selector_seconds': action,
                          'total_seconds': outer + action, 'evaluated_boxes': sum(value['evaluated_boxes'] for value in values),
                          'unresolved_budget_leaves': sum(value['unresolved_budget_leaves'] for value in values)})
    same(summary['per_beta_method_timing'], costs, 'SUMMARY independently summed twelve beta/arm cost records')
    _old.verify_csv(output / 'evaluation/TIMING.csv', costs)


def verify_evaluation(output, rows, coverage, outers, status):
    require(len(rows) == 5832 and len(coverage) == 648, 'evaluation: incomplete row/coverage population')
    same(load(output / 'evaluation/ROWS.json'), rows, 'independent actual-truth scores and contract labels')
    _old.verify_csv(output / 'evaluation/ROWS.csv', rows)
    same(load(output / 'evaluation/COVERAGE.json'), coverage, 'whole-scene retained-set truth coverage')
    _old.verify_csv(output / 'evaluation/COVERAGE.csv', coverage, ('retained_branches', 'branch_coverage'))
    pairs = paired_rows(rows)
    require(len(pairs) == 5832, 'paired evaluation: wrong full-task comparison denominator')
    same(load(output / 'evaluation/PAIRED.json'), pairs, 'independently reconstructed paired error differences')
    _old.verify_csv(output / 'evaluation/PAIRED.csv', pairs)
    summary = load(output / 'evaluation/SUMMARY.json')
    fields = {'observation_configurations', 'task_count', 'decision_count', 'rows', 'per_beta_method_timing',
              'sampling_note', 'timing_scope', 'elapsed_seconds'}
    require(isinstance(summary, dict) and set(summary) == fields, 'SUMMARY: invalid complete schema')
    for field, value in {'observation_configurations': 162, 'task_count': 1458, 'decision_count': 5832}.items():
        same(summary[field], value, f'SUMMARY {field}')
    aggregates = reconstruct_aggregates(rows)
    require(len(aggregates) == 1080, 'SUMMARY: wrong beta/template/seed endpoint denominators')
    same(summary['rows'], aggregates, 'all1080 independently reconstructed grouped count/error/coverage/cost records')
    _old.verify_csv(output / 'evaluation/SUMMARY.csv', aggregates)
    same(summary['sampling_note'], SAMPLING_NOTE, 'SUMMARY same-source sampling scope')
    same(summary['timing_scope'], CACHE_POLICY, 'SUMMARY timing scope')
    require(finite_seconds(summary['elapsed_seconds'], 'SUMMARY elapsed') <= status['elapsed_seconds'], 'SUMMARY elapsed exceeds complete run time')
    verify_timing(output, outers, summary)
    return aggregates, pairs


def verify(output):
    output = Path(output).resolve()
    expected = Path('/srv/slam-research/grf/map-denoise/runs/lovelace-view-bias-20261011T092300Z')
    require(socket.gethostname() == 'liekkas' and ROOT == expected and ROOT.resolve() == expected and
            output.is_dir() and output != ROOT and output.is_relative_to(ROOT), 'target: exact existing local view-bias output on liekkas required')
    sources = verify_source_lock(load(output / 'SOURCE_LOCK.json'))
    inputs = verify_inputs(load(output / 'INPUTS.json'))
    same(load(output / 'PROTOCOL.json'), registration(), 'independent bounded bias/noise/method registration')
    status = load(output / 'RUN_STATUS.json')
    require(isinstance(status, dict) and set(status) == {'status', 'observations_complete', 'decisions_complete',
            'elapsed_seconds', 'scientific_support'}, 'RUN_STATUS: invalid complete schema')
    same({key: status[key] for key in ('status', 'observations_complete', 'decisions_complete', 'scientific_support')},
         {'status': 'COMPLETE', 'observations_complete': 162, 'decisions_complete': 5832,
          'scientific_support': 'requires independent verification and predeclared claim analysis'}, 'RUN_STATUS complete population and separate claim policy')
    require(finite_seconds(status['elapsed_seconds'], 'RUN_STATUS elapsed') <= 5400, 'RUN_STATUS registered wall budget exceeded')
    model_class, geometry_bounds = _historical.shared_kernels()
    observations, tasks, truths, residuals = verify_population(output, model_class)
    outer_list, decision_list = load(output / 'outers.json'), load(output / 'decisions.json')
    require(isinstance(outer_list, list) and len(outer_list) == 162 and isinstance(decision_list, list) and len(decision_list) == 5832,
            'population: incomplete outer/action records')
    outers = indexed(outer_list, lambda row: row['observation_id'], 'outers')
    decisions = indexed(decision_list, lambda row: (row['task_id'], row['method']), 'decisions')
    require(list(outers) == list(observations), 'outers: changed observation identities/order')
    expected_decisions = [(f'{oid}-s{state}', method) for oid in observations for method in METHODS for state in range(9)]
    require(list(decisions) == expected_decisions, 'decisions: changed complete task/method identities/order')
    checkpoints = output / 'checkpoint'
    require(checkpoints.is_dir() and sorted(path.name for path in checkpoints.iterdir()) == sorted(f'{oid}.json' for oid in observations),
            'checkpoint: incomplete or additional observation artifacts')
    unions, branches = {}, {}
    calls = witnesses = certificates = updates = leaves = group_tests = 0
    for oid, public in observations.items():
        record, image, beta = outers[oid], tuple(exact(value) for value in public['observation']), exact(public['beta'])
        require(isinstance(record, dict) and set(record) == {'observation_id', 'base_observation_id', 'beta', 'arms'} and
                record['base_observation_id'] == public['base_observation_id'] and record['beta'] == public['beta'] and
                isinstance(record['arms'], dict) and list(record['arms']) == list(METHODS), f'{oid}: wrong outer identity/arm schema')
        same(load(checkpoints / f'{oid}.json'), record, f'{oid}: final outer/checkpoint consistency')
        unions[oid], branches[oid] = {}, {}
        for method in METHODS:
            arm, where = record['arms'][method], f'{oid}/{method}'
            require(isinstance(arm, dict) and set(arm) == {'branches', 'outer', 'branch_seconds', 'outer_seconds'}, f'{where}: wrong arm schema')
            allowed = () if method == 'keep' else BRANCHES
            require(isinstance(arm['branches'], dict) and isinstance(arm['branch_seconds'], dict) and
                    list(arm['branches']) == list(arm['branch_seconds']) == list(allowed), f'{where}: wrong whole-scene branch identities/order')
            total = finite_seconds(arm['outer_seconds'], f'{where}: measured solve time')
            measured = sum(finite_seconds(value, f'{where}: branch seconds') for value in arm['branch_seconds'].values())
            require(total >= measured and (method != 'keep' or total == 0), f'{where}: inconsistent measured branch costs')
            checked_branches = {}
            for branch in allowed:
                outer = arm['branches'][branch]
                model = model_class(branch, camera_radius=RADIUS, colors=((Q(0), Q(1)),) * 3)
                checked, n_calls, n_witnesses = verify_branch(outer, image, beta, method, branch, geometry_bounds, model, f'{where}/{branch}')
                calls += n_calls
                witnesses += n_witnesses
                certificates += len(checked['shared_excluded'])
                updates += checked['proof_updates']
                leaves += sum(node['status'] != 'split' for node in outer['trace'])
                group_tests += outer['shared_tests']
                checked_branches[branch] = checked
            branches[oid][method] = checked_branches
            unions[oid][method] = verify_union(arm['outer'], checked_branches, beta, method, where)
    rows, coverage = [], []
    for decision in decision_list:
        task, method = tasks[decision['task_id']], decision['method']
        oid = task['observation_id']
        checked = unions[oid][method]
        result = verify_decision(decision, task, checked, f'{task["task_id"]}/{method}')
        row = make_row(task, truths[oid], decision, checked)
        if method != 'keep' and checked['hull'] is not None and row['covered']:
            truth, incumbent = exact(truths[oid]['truth']), exact(task['incumbent'])
            gain = (incumbent - truth) ** 2 - (result - truth) ** 2
            require(gain >= exact(decision['gain_lower']) >= 0, f'{task["task_id"]}/{method}: endpoint gain certificate fails actual covered truth')
        rows.append(row)
    for oid, private in truths.items():
        truth = exact(private['truth'])
        for method in METHODS:
            checked, branch_checks = unions[oid][method], branches[oid][method]
            hull = checked['hull']
            coverage.append({'observation_id': oid, 'base_observation_id': private['base_observation_id'], 'beta': private['beta'],
                             'case': private['case'], 'seed': private['seed'], 'method': method, 'contract_valid': method != 'shared_strict',
                             'covered': _historical.contains(checked['intervals'], truth), 'incompatible': hull is None,
                             'hull_width_mm': None if hull is None else str(hull[1] - hull[0]), 'retained_branches': checked['retained_branches'],
                             'branch_coverage': {branch: _historical.contains(value['intervals'], truth) for branch, value in branch_checks.items()}})
    aggregates, pairs = verify_evaluation(output, rows, coverage, outers, status)
    claims = scientific_claims(rows, coverage)
    same(scientific_files(), sources, 'all scientific sources unchanged during independent verification')
    same(historical_inputs(), inputs, 'all eleven historical evidence inputs unchanged during independent verification')
    return {'status': 'PASS', 'mechanical_verification': 'PASS', 'host': socket.gethostname(), 'output': str(output),
            'observations': 162, 'tasks': 1458, 'decisions': 5832, 'coverage_rows': 648, 'summary_rows': len(aggregates),
            'paired_rows': len(pairs), 'branch_outer_traces': 1458, 'terminal_partition_leaves': leaves, 'unresolved_budget_leaves': 0,
            'group_exclusion_certificates_verified': certificates, 'exact_propagation_updates_verified': updates,
            'group_tests_accounted': group_tests, 'historical_kernel_boxes_recomputed': calls, 'channel_witnesses_recomputed': witnesses,
            'source_files_verified': len(sources), 'historical_inputs_verified': len(inputs), 'actual_truth_residuals': residuals, 'claims': claims,
            'limitations': ['Historical point/enclosure/area kernels are shared; their physical geometry soundness is not independently proved.',
                            'Necessary-constraint exclusion certificates are replayed; retained boxes are not guaranteed jointly feasible.',
                            'Unrecorded no-contradiction propagation on retained boxes is not replayed.',
                            'Recorded cold-cache times are checked and summed, not independently reproduced.',
                            'Source/input hashes do not prove operating-system blindness or independent historical timestamps.',
                            'This is same-source synthetic bias/clipping replay; strict-model failures are diagnostic and do not invalidate mechanical completion.',
                            'Bias tests bounded additive brightness shifts, not arbitrary exposure, view-dependent reflectance or real-MVS transfer.']}


def scientific_claims(rows, coverage):
    """Enforce lawful safety; deliberately retain strict-model negative results."""
    result, supported = {}, []
    for beta in BETAS:
        level, endpoints, safety = str(beta), {}, {}
        for method in METHODS:
            values = [row for row in rows if row['beta'] == level and row['method'] == method]
            offsets = [row for row in values if row['state_group'] == 'small']
            correct = [row for row in values if row['state_group'] == 'correct']
            negative = [row for row in offsets if row['case'] == 'equal_full_negative']
            cover = [row for row in coverage if row['beta'] == level and row['method'] == method]
            require(len(values) == 486 and len(offsets) == 432 and len(correct) == 54 and len(negative) == 72 and len(cover) == 54,
                    f'{level}/{method}: incorrect registered safety/endpoint denominators')
            harm = sum(exact(row['ae_mm']) > exact(row['initial_ae_mm']) for row in values)
            counts = {'coverage': sum(row['covered'] for row in cover), 'coverage_failures': sum(not row['covered'] for row in cover),
                      'correct_kept': sum(row['status'] == 'KEEP' and exact(row['ae_mm']) == 0 for row in correct),
                      'correct_moved': sum(row['status'] == 'MOVE' for row in correct),
                      'correct_harmed': sum(exact(row['ae_mm']) > 0 for row in correct), 'harm': harm,
                      'negative_small_kept': sum(row['status'] == 'KEEP' for row in negative),
                      'empty_observations': sum(row['incompatible'] for row in cover),
                      'contract_valid': method != 'shared_strict'}
            if method != 'shared_strict':
                require(counts['coverage'] == 54 and counts['correct_kept'] == 54 and counts['negative_small_kept'] == 72 and harm == 0,
                        f'legal safety: {level}/{method} failed true coverage/correct KEEP/negative KEEP/zero harm')
            safety[method] = counts
            errors = [exact(row['ae_mm']) for row in offsets]
            endpoints[method] = {'n': 432, 'improved': sum(error < exact(row['initial_ae_mm']) for error, row in zip(errors, offsets)),
                                 'same': sum(error == exact(row['initial_ae_mm']) for error, row in zip(errors, offsets)),
                                 'harm': sum(error > exact(row['initial_ae_mm']) for error, row in zip(errors, offsets)),
                                 'mae_exact_mm': str(sum(errors, Q(0)) / 432),
                                 'mse_exact_mm2': str(sum((error * error for error in errors), Q(0)) / 432),
                                 'contract_valid': method != 'shared_strict'}
        shared, channel = endpoints['global_shared_wide'], endpoints['channel_wide']
        supports = shared['improved'] > channel['improved'] and exact(shared['mae_exact_mm']) < exact(channel['mae_exact_mm'])
        if supports:
            supported.append(level)
        result[level] = {'safety': safety, 'offset_endpoints': endpoints, 'C1_at_beta': 'SUPPORTED' if supports else 'NEGATIVE'}
    return {'C1': {'status': 'SUPPORTED' if supported else 'NEGATIVE', 'supporting_predeclared_betas': supported,
                   'all_betas_reported': True},
            'C2': {'scope': 'shared_strict is an intentional mismatch diagnostic; failures are not lawful performance'},
            'by_beta': result, 'scientific_negative_result_is_not_mechanical_verification_failure': True}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Independently verify the bounded view-bias replay and its diagnostics.')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/view-bias-v1')
    output = parser.parse_args(argv).output.resolve()
    expected = Path('/srv/slam-research/grf/map-denoise/runs/lovelace-view-bias-20261011T092300Z')
    if socket.gethostname() != 'liekkas' or ROOT != expected or ROOT.resolve() != expected:
        parser.error('exact local view-bias root on liekkas required')
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
        report = verify(output)
    except Exception as error:
        report = {'status': 'FAIL', 'mechanical_verification': 'FAIL', 'host': socket.gethostname(), 'output': str(output),
                  'error': f'{type(error).__name__}: {error}'}
    report.update(utc=datetime.now(timezone.utc).isoformat(), verification_seconds=time.perf_counter() - started)
    with (destination / 'VERIFY.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())

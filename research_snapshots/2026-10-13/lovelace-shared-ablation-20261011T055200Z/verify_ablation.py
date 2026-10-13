#!/usr/bin/env python3
"""Independent proof, tree, action and accounting checks for the frozen replay.

No new solver or runner is imported. Historical independent exact arithmetic
and certificate replay are reused under a unique module name. The grouped
necessary rows, 511-node tree contract, and evaluation are reconstructed here.
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
import socket
import sys
import time

sys.dont_write_bytecode = True
for _verifier_thread_variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_verifier_thread_variable] = '1'

ROOT = Path(__file__).resolve().parent
OLD_ROOT = ROOT.parent / 'lovelace-shared-appearance-20261011T052300Z'
OLD_OUTPUT = OLD_ROOT / 'outputs/confirmation-v1'
NUISANCE_ROOT = ROOT.parent / 'lovelace-nuisance-20261011T044500Z'
HYBRID_ROOT = ROOT.parent / 'lovelace-hybrid-20261010T214217Z'
CHALLENGE_ROOT = ROOT.parent / 'lovelace-challenge-20261010T205013Z'
_spec = importlib.util.spec_from_file_location('_ablation_independent_historical_verifier', OLD_ROOT / 'shared_verification.py')
_historical = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_historical)
_arithmetic = _historical._arithmetic
VerificationError = _historical.VerificationError
exact, encoded, pair = _historical.exact, _historical.encoded, _historical.pair
require, indexed, same, load = _historical.require, _historical.indexed, _historical.same, _historical.load
finite_seconds = _historical.finite_seconds
DOMAIN, EPSILON, TOLERANCE = _historical.DOMAIN, _historical.EPSILON, _historical.TOLERANCE
RADIUS, ROUNDS, MAX_BOXES = Q(1, 5), 8, 511
BRANCHES = ('none', 'partial', 'full')
METHODS = ('keep', 'channel511', 'global_gray511', 'pixel_gray511', 'view_gray511', 'global_rgb511')
MODES = ('global_gray', 'pixel_gray', 'view_gray', 'global_rgb')
METHOD_MODES = dict(zip(METHODS[1:], ('channel', *MODES)))
FROZEN_FILES = ('observations.json', 'tasks.json', 'evaluation/truth.json', 'outers.json', 'decisions.json',
                'evaluation/SUMMARY.json', 'SOURCE_LOCK.json', 'verification/VERIFY.json')
CHANNEL_SCOPE = 'outer_set_for_continuous_depth_and_model_nuisance_contract'
GROUP_SCOPE = 'necessary_grouped_material_halfspaces_with_conservative_continuous_geometry_bounds'


def necessary_groups(observation, geometry, mode):
    """Derive grouping and halfspaces solely from image layout and public mode.

    Geometry is three views, each with 48 row-major pixels, and six area bounds
    per pixel. Each gray block intersects noisy RGB bands. RGB relaxation has
    three separate channel blocks, each with three material variables.
    """
    require(mode in MODES, 'group rows: unknown public mode')
    require(isinstance(geometry, (list, tuple)) and len(geometry) == 144,
            'group rows: expected three views of 48 pixels')
    image = tuple(exact(value) for value in observation)
    require(len(image) == 432, 'group rows: expected 432 observed RGB channels')
    coefficients = []
    for bounds in geometry:
        require(isinstance(bounds, (list, tuple)) and len(bounds) == 6, 'group rows: three area endpoint pairs required')
        values = tuple(exact(value) for value in bounds)
        require(all(0 <= values[j] <= values[j + 1] <= 1 for j in (0, 2, 4)),
                'group rows: invalid normalized area bounds')
        coefficients.append((values[::2], values[1::2]))
    if mode == 'global_gray':
        pixels_by_group = [('global', tuple(range(144)), None)]
    elif mode == 'pixel_gray':
        pixels_by_group = [(f'p{pixel}', (pixel,), None) for pixel in range(144)]
    elif mode == 'view_gray':
        pixels_by_group = [(f'v{view}', tuple(range(48 * view, 48 * (view + 1))), None) for view in range(3)]
    else:
        pixels_by_group = [(f'c{channel}', tuple(range(144)), channel) for channel in range(3)]
    result = {}
    for group_id, pixels, channel in pixels_by_group:
        rows = []
        for pixel in pixels:
            lower, upper = coefficients[pixel]
            values = image[3 * pixel:3 * pixel + 3] if channel is None else (image[3 * pixel + channel],)
            rows.append(('upper', lower, min(1 - value + EPSILON for value in values)))
            rows.append(('lower', upper, max(1 - value - EPSILON for value in values)))
        result[group_id] = tuple(rows)
    return result


def verify_group_certificate(observation, geometry, mode, group_id, certificate, where):
    groups = necessary_groups(observation, geometry, mode)
    require(type(group_id) is str and group_id in groups, f'{where}: forged material group identity')
    return _historical.verify_certificate(groups[group_id], certificate, where)


def verify_trace(outer, observation, where, *, mode='channel', branch=None, geometry_bounds=None):
    """Check full BFS partition, exact budget and recorded exclusion evidence."""
    fields = {'domain', 'epsilon', 'tolerance', 'max_boxes', 'evaluated_boxes', 'trace', 'retained_ids', 'hull', 'scope',
              'shared_tests', 'propagation_rounds', 'mode', 'rounds', 'budget'}
    require(isinstance(outer, dict) and set(outer) == fields, f'{where}: invalid branch-outer schema')
    require(pair(outer['domain'], where) == DOMAIN and exact(outer['epsilon']) == EPSILON and
            exact(outer['tolerance']) == TOLERANCE and type(outer['max_boxes']) is int and
            outer['max_boxes'] == MAX_BOXES, f'{where}: changed depth/noise/tolerance/511 budget')
    require(mode in ('channel', *MODES) and outer['mode'] == mode and
            type(outer['rounds']) is int and outer['rounds'] == ROUNDS and
            type(outer['budget']) is int and outer['budget'] == MAX_BOXES and
            type(outer['propagation_rounds']) is int and outer['propagation_rounds'] == ROUNDS,
            f'{where}: changed material mode/rounds/budget metadata')
    if mode == 'channel':
        require(outer['scope'] == CHANNEL_SCOPE, f'{where}: changed channel scope')
    else:
        require(outer['scope'] == GROUP_SCOPE,
                f'{where}: changed grouped material contract')
        require(branch in BRANCHES and callable(geometry_bounds), f'{where}: geometry checker required')
    nodes = indexed(outer['trace'], lambda node: node['id'], where)
    require(0 < len(nodes) <= MAX_BOXES and all(type(node['id']) is int for node in outer['trace']) and
            [node['id'] for node in outer['trace']] == list(range(len(nodes))), f'{where}: invalid full BFS identities')
    require(nodes[0]['parent'] is None, f'{where}: root has a parent')
    boxes = {ident: (exact(node['lo']), exact(node['hi'])) for ident, node in nodes.items()}
    require(boxes[0] == DOMAIN, f'{where}: root is not full depth domain')
    leaves, retained, excluded, grouped_excluded, referenced = [], [], [], [], set()
    evaluated, next_child, shared_tests, proof_updates = 0, 1, 0, 0
    extras = {'split': {'children'}, 'excluded': {'witness_channel', 'prediction_interval'},
              'retained_tolerance': set(), 'unresolved_budget': set()}
    if mode != 'channel':
        extras['excluded_shared'] = {'group_id', 'certificate'}
    for node in outer['trace']:
        ident, status = node['id'], node['status']
        low, high = boxes[ident]
        require(DOMAIN[0] <= low < high <= DOMAIN[1], f'{where}: invalid interval {ident}')
        require(status in extras and set(node) == {'id', 'parent', 'lo', 'hi', 'status'} | extras[status],
                f'{where}: invalid node status/schema {ident}')
        if ident:
            require(type(node['parent']) is int and node['parent'] in nodes and node['parent'] < ident,
                    f'{where}: invalid parent {ident}')
        # The complete 256-leaf tree needs at most511 evaluated nodes. A budget
        # leaf is a failed registered endpoint, even if its counters are honest.
        require(status != 'unresolved_budget', f'{where}: unresolved budget leaf in full511 contract')
        require(evaluated < MAX_BOXES, f'{where}: evaluation exceeds511 budget')
        evaluated += 1
        if mode != 'channel' and status != 'excluded':
            group_ids = (['global'] if mode == 'global_gray' else
                         [f'p{i}' for i in range(144)] if mode == 'pixel_gray' else
                         [f'v{i}' for i in range(3)] if mode == 'view_gray' else [f'c{i}' for i in range(3)])
            if status == 'excluded_shared':
                require(type(node['group_id']) is str and node['group_id'] in group_ids,
                        f'{where}: forged material group identity')
                shared_tests += group_ids.index(node['group_id']) + 1
            else:
                shared_tests += len(group_ids)
        if status == 'split':
            require(high - low > TOLERANCE, f'{where}: split at/below tolerance')
            children = node['children']
            require(isinstance(children, list) and children == [next_child, next_child + 1] and
                    all(type(child) is int and child in nodes for child in children), f'{where}: invalid BFS children')
            next_child += 2
            middle = (low + high) / 2
            for child, expected in zip(children, ((low, middle), (middle, high))):
                require(child not in referenced and nodes[child]['parent'] == ident and boxes[child] == expected,
                        f'{where}: gap, overlap, nonbisection or duplicate child')
                referenced.add(child)
            continue
        leaves.append((low, high))
        if status == 'excluded':
            channel = node['witness_channel']
            require(type(channel) is int and 0 <= channel < len(observation), f'{where}: invalid witness channel')
            a, b = pair(node['prediction_interval'], where)
            require(0 <= a <= b <= 1 and (a > observation[channel] + EPSILON or b < observation[channel] - EPSILON),
                    f'{where}: channel witness does not conflict with observed band')
            excluded.append(node)
        elif status == 'excluded_shared':
            checked = verify_group_certificate(observation, geometry_bounds(branch, RADIUS, low, high), mode,
                                               node['group_id'], node['certificate'], f'{where}/node{ident}')
            proof_updates += checked['updates']
            grouped_excluded.append(node)
        else:
            require(high - low <= TOLERANCE, f'{where}: retained leaf exceeds tolerance')
            retained.append(node)
    require(referenced == set(nodes) - {0}, f'{where}: orphan or unreachable nodes')
    _arithmetic.verify_partition(leaves, where)
    require(type(outer['evaluated_boxes']) is int and outer['evaluated_boxes'] == evaluated,
            f'{where}: false evaluated counter')
    require(type(outer['shared_tests']) is int and outer['shared_tests'] == shared_tests,
            f'{where}: false grouped test counter')
    require(isinstance(outer['retained_ids'], list) and all(type(ident) is int for ident in outer['retained_ids']) and
            outer['retained_ids'] == [node['id'] for node in retained], f'{where}: false retained provenance')
    intervals = [boxes[node['id']] for node in retained]
    hull = None if not intervals else (min(low for low, _ in intervals), max(high for _, high in intervals))
    require((outer['hull'] is None and hull is None) or
            (outer['hull'] is not None and hull is not None and pair(outer['hull'], where) == hull), f'{where}: false retained hull')
    return {'hull': hull, 'intervals': intervals, 'retained': retained, 'boxes': boxes, 'excluded': excluded,
            'shared_excluded': grouped_excluded, 'evaluated': evaluated, 'proof_updates': proof_updates, 'budget_leaves': 0}


def verify_kernel_boxes(outer, checked, observation, model, where, grouped=False):
    return _historical.verify_kernel_boxes(outer, checked, observation, model, where, shared=grouped)


def verify_union(outer, branches, where, prior_only=False, mode='channel'):
    fields = {'domain', 'epsilon', 'tolerance', 'max_boxes_per_branch', 'evaluated_boxes',
              'unresolved_budget_leaves', 'source_leaves', 'retained_intervals', 'hull', 'retained_branches', 'scope'}
    if not prior_only:
        fields |= {'mode', 'rounds', 'budget'}
    require(isinstance(outer, dict) and set(outer) == fields, f'{where}: invalid union schema')
    require(pair(outer['domain'], where) == DOMAIN and exact(outer['epsilon']) == EPSILON and
            exact(outer['tolerance']) == TOLERANCE and type(outer['max_boxes_per_branch']) is int and
            outer['max_boxes_per_branch'] == MAX_BOXES, f'{where}: changed union contract')
    if not prior_only:
        require(mode in ('channel', *MODES) and outer['mode'] == mode and
                type(outer['rounds']) is int and outer['rounds'] == ROUNDS and
                type(outer['budget']) is int and outer['budget'] == MAX_BOXES,
                f'{where}: changed union public material contract')
    require(not prior_only or not branches, f'{where}: KEEP unexpectedly solved scene branches')
    source = [{'branch': name, 'id': node['id'], 'lo': node['lo'], 'hi': node['hi']}
              for name, checked in branches.items() for node in checked['retained']]
    intervals = [DOMAIN] if prior_only else _arithmetic.merged_intervals(
        interval for checked in branches.values() for interval in checked['intervals'])
    hull = None if not intervals else (intervals[0][0], intervals[-1][1])
    retained_branches = [name for name, checked in branches.items() if checked['hull'] is not None]
    evaluated = sum(checked['evaluated'] for checked in branches.values())
    unresolved = sum(checked['budget_leaves'] for checked in branches.values())
    same(outer['source_leaves'], source, f'{where}: whole-scene branch provenance')
    require(isinstance(outer['retained_intervals'], list) and
            [pair(value, where) for value in outer['retained_intervals']] == intervals,
            f'{where}: incorrect closed depth union')
    require((outer['hull'] is None and hull is None) or
            (outer['hull'] is not None and hull is not None and pair(outer['hull'], where) == hull), f'{where}: false union hull')
    require(type(outer['evaluated_boxes']) is int and outer['evaluated_boxes'] == evaluated and
            type(outer['unresolved_budget_leaves']) is int and outer['unresolved_budget_leaves'] == unresolved and
            outer['retained_branches'] == retained_branches, f'{where}: false union counts')
    require(outer['scope'] == ('prior_only_identity_no_image_solver' if prior_only else
                             'union_of_branch_depth_outer_leaves_then_convex_hull'), f'{where}: changed union scope')
    return {'hull': hull, 'intervals': intervals, 'retained_branches': retained_branches,
            'evaluated': evaluated, 'budget_leaves': unresolved}


def verify_decision(decision, incumbent, hull, where):
    # Historical independent endpoint arithmetic has no box-budget dependency.
    require(decision.get('method') in METHODS, f'{where}: unknown decision method')
    return _arithmetic.verify_decision(decision, incumbent, hull, where)


def make_row(task, private, decision, checked):
    row = _historical.make_row(task, private, decision, checked)
    row['auxiliary_samples_used'] = 0
    return row


def reconstruct_aggregates(rows):
    return _historical.reconstruct_aggregates(rows)


def verify_csv(path, records, json_columns=()):
    return _historical.verify_csv(Path(path), records, json_columns)


def scientific_files():
    files = dict(_historical.scientific_files())
    paths = {ROOT / 'PROTOCOL.md', *ROOT.glob('*.py')}
    files.update({str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)})
    return dict(sorted(files.items()))


def frozen_hashes():
    return {str((OLD_OUTPUT / relative).resolve()): hashlib.sha256((OLD_OUTPUT / relative).read_bytes()).hexdigest()
            for relative in FROZEN_FILES}


def registration():
    return {'methods': list(METHODS),
            'method_configs': {method: {'mode': METHOD_MODES.get(method, 'identity'), 'max_boxes': MAX_BOXES,
                                        'rounds': ROUNDS} for method in METHODS},
            'epsilon': str(EPSILON), 'tolerance_mm': str(TOLERANCE), 'max_boxes_per_branch': MAX_BOXES,
            'expected_observations': 54, 'expected_tasks': 486, 'expected_decisions': 2916,
            'allowed_branches': list(BRANCHES), 'cpu_threads': 1, 'gpu_count': 0,
            'wall_time_limit_seconds': 5400, 'source_root': str(OLD_OUTPUT),
            'sampling_note': 'same 54 synthetic configurations replayed; not a new confirmation or independent scenes',
            'cache_policy': 'clear all geometry/model caches before each arm/observation; reuse within its three branches',
            'old_255_results': 'frozen prior artifacts, not rerun or overwritten'}


def verify_source_lock(lock):
    require(isinstance(lock, dict) and set(lock) == {'files', 'source_root', 'lock_scope', 'utc'},
            'SOURCE_LOCK: invalid schema')
    require(lock['source_root'] == str(ROOT) and lock['lock_scope'] == 'scientific sources and frozen protocol',
            'SOURCE_LOCK: changed exact source identity/scope')
    require(type(lock['utc']) is str and datetime.fromisoformat(lock['utc']).utcoffset() is not None,
            'SOURCE_LOCK: timezone-aware timestamp required')
    # Validate the old source identities and their complete inventory with the
    # pre-existing independent verifier; no historical solve is rerun.
    _historical.verify_source_lock(load(OLD_OUTPUT / 'SOURCE_LOCK.json'))
    sources = scientific_files()
    same(lock['files'], sources, 'SOURCE_LOCK independent complete inventory and hashes')
    return sources


def verify_inputs(lock):
    require(isinstance(lock, dict) and set(lock) == {'source_root', 'files'} and
            lock['source_root'] == str(OLD_OUTPUT), 'INPUTS: changed frozen source identity/schema')
    inputs = frozen_hashes()
    same(lock['files'], inputs, 'INPUTS eight exact historical output byte hashes')
    previous = load(OLD_OUTPUT / 'verification/VERIFY.json')
    require(isinstance(previous, dict) and previous.get('status') == 'PASS' and previous.get('host') == 'liekkas' and
            previous.get('output') == str(OLD_OUTPUT) and previous.get('observations') == 54 and
            previous.get('tasks') == 486 and previous.get('decisions') == 1944,
            'INPUTS: prior verification record does not establish the frozen population')
    return inputs


def verify_frozen_population(model_class):
    observation_list = load(OLD_OUTPUT / 'observations.json')
    task_list = load(OLD_OUTPUT / 'tasks.json')
    truth_list = load(OLD_OUTPUT / 'evaluation/truth.json')
    for records, count, name in ((observation_list, 54, 'observations'), (task_list, 486, 'tasks'), (truth_list, 54, 'truths')):
        require(isinstance(records, list) and len(records) == count, f'population: expected{count} {name}')
    observations = indexed(observation_list, lambda row: row['observation_id'], 'observations')
    truths = indexed(truth_list, lambda row: row['observation_id'], 'truths')
    tasks = indexed(task_list, lambda row: row['task_id'], 'tasks')
    ids = [f'o{i:03d}' for i in range(54)]
    require(list(observations) == list(truths) == ids, 'population: changed frozen observation IDs/order')
    expected_tasks, index = [], 0
    for case, branch, colors in _historical.CASES:
        for truth in _historical.TRUTHS:
            for seed in _historical.SEEDS:
                oid = ids[index]
                public, private = _historical.regenerate(case, branch, colors, truth, seed, model_class)
                same(observations[oid], {'observation_id': oid, **public}, f'{oid}: regenerated frozen observed RGB and samples')
                same(truths[oid], {'observation_id': oid, **private}, f'{oid}: regenerated frozen actual truth/random tape')
                expected_tasks.extend({'task_id': f'{oid}-s{j}', 'observation_id': oid,
                                       'incumbent': encoded(truth + offset)} for j, offset in enumerate(_historical.OFFSETS))
                index += 1
    same(task_list, expected_tasks, 'all frozen initial task offsets')
    return observations, tasks, truths


def verify_timing(output, outers, summary):
    timing = load(output / 'TIMING.json')
    require(isinstance(timing, dict) and set(timing) == {'observations', 'cold_cache_note'}, 'TIMING: invalid schema')
    same(timing['cold_cache_note'], registration()['cache_policy'], 'TIMING registered cold-cache scope')
    timings = indexed(timing['observations'], lambda row: row['observation_id'], 'TIMING')
    require(list(timings) == list(outers), 'TIMING: changed observation IDs/order')
    same(load(output / 'TIMING_PARTIAL.json'), {'observations': timing['observations']}, 'TIMING complete checkpoint')
    for oid, record in timings.items():
        require(set(record) == {'observation_id', 'outer_seconds_by_method', 'selector_seconds_by_method'},
                f'{oid}: invalid timing record')
        require(list(record['outer_seconds_by_method']) == list(record['selector_seconds_by_method']) == list(METHODS),
                f'{oid}: changed timing arm order/population')
        for method in METHODS:
            measured = finite_seconds(record['outer_seconds_by_method'][method], f'{oid}/{method}: measured solve time')
            finite_seconds(record['selector_seconds_by_method'][method], f'{oid}/{method}: measured action time')
            require(measured == outers[oid]['arms'][method]['outer_seconds'], f'{oid}/{method}: inconsistent solve time')
    costs = {}
    for method in METHODS:
        outer = sum(record['outer_seconds_by_method'][method] for record in timing['observations'])
        action = sum(record['selector_seconds_by_method'][method] for record in timing['observations'])
        costs[method] = {'outer_seconds': outer, 'selector_seconds': action, 'total_seconds': outer + action,
                         'evaluated_boxes': sum(record['arms'][method]['outer']['evaluated_boxes'] for record in outers.values()),
                         'unresolved_budget_leaves': sum(record['arms'][method]['outer']['unresolved_budget_leaves'] for record in outers.values())}
    same(summary['per_method_timing'], costs, 'SUMMARY independently summed measured arm costs')
    verify_csv(output / 'evaluation/TIMING.csv', [{'method': method, **cost} for method, cost in costs.items()])


def verify_evaluation(output, rows, coverage, decisions, outers, status):
    same(load(output / 'evaluation/ROWS.json'), rows, 'independent exact actual-truth evaluation rows')
    verify_csv(output / 'evaluation/ROWS.csv', rows)
    same(load(output / 'evaluation/COVERAGE.json'), coverage, 'independent retained-set truth membership')
    verify_csv(output / 'evaluation/COVERAGE.csv', coverage, ('retained_branches', 'branch_coverage'))
    summary = load(output / 'evaluation/SUMMARY.json')
    fields = {'observation_configurations', 'task_count', 'decision_count', 'rows', 'per_method_timing',
              'sampling_note', 'timing_scope', 'elapsed_seconds'}
    require(isinstance(summary, dict) and set(summary) == fields, 'SUMMARY: invalid complete schema')
    for field, value in {'observation_configurations': 54, 'task_count': 486, 'decision_count': 2916}.items():
        same(summary[field], value, f'SUMMARY {field}')
    aggregates = reconstruct_aggregates(rows)
    require(len(aggregates) == 540, 'SUMMARY: incorrect grouped endpoint denominator')
    same(summary['rows'], aggregates, 'SUMMARY all independently reconstructed counts/errors/coverage/costs')
    verify_csv(output / 'evaluation/SUMMARY.csv', aggregates)
    same(summary['sampling_note'], registration()['sampling_note'], 'SUMMARY fixed same-input replay scope')
    same(summary['timing_scope'], registration()['cache_policy'], 'SUMMARY cold-cache timing scope')
    elapsed = finite_seconds(summary['elapsed_seconds'], 'SUMMARY elapsed')
    require(elapsed <= status['elapsed_seconds'], 'SUMMARY: elapsed exceeds completed run time')
    verify_timing(output, outers, summary)
    return aggregates


def verify(output):
    output = Path(output).resolve()
    expected_root = Path('/srv/slam-research/grf/map-denoise/runs/lovelace-shared-ablation-20261011T055200Z')
    require(socket.gethostname() == 'liekkas' and ROOT == expected_root and output.is_relative_to(ROOT) and
            output != ROOT and output.is_dir(), 'target identity: exact existing ablation output on liekkas required')
    sources = verify_source_lock(load(output / 'SOURCE_LOCK.json'))
    inputs = verify_inputs(load(output / 'INPUTS.json'))
    same(load(output / 'PROTOCOL.json'), registration(), 'independent registered511-budget protocol snapshot')
    status = load(output / 'RUN_STATUS.json')
    require(isinstance(status, dict) and set(status) == {'status', 'observations_complete', 'decisions_complete', 'elapsed_seconds'},
            'RUN_STATUS: invalid complete schema')
    same({field: status[field] for field in ('status', 'observations_complete', 'decisions_complete')},
         {'status': 'COMPLETE', 'observations_complete': 54, 'decisions_complete': 2916}, 'RUN_STATUS complete population')
    require(finite_seconds(status['elapsed_seconds'], 'RUN_STATUS elapsed') <= 5400, 'RUN_STATUS: registered wall budget exceeded')
    model_class, geometry_bounds = _historical.shared_kernels()
    observations, tasks, truths = verify_frozen_population(model_class)
    outer_list, decision_list = load(output / 'outers.json'), load(output / 'decisions.json')
    require(isinstance(outer_list, list) and len(outer_list) == 54 and
            isinstance(decision_list, list) and len(decision_list) == 2916, 'population: incomplete outer/action records')
    outers = indexed(outer_list, lambda row: row['observation_id'], 'outers')
    decisions = indexed(decision_list, lambda row: (row['task_id'], row['method']), 'decisions')
    require(list(outers) == list(observations), 'outers: changed observation IDs/order')
    expected_decisions = [(f'{oid}-s{j}', method) for oid in observations for method in METHODS for j in range(9)]
    require(list(decisions) == expected_decisions, 'decisions: changed complete task/arm population or order')
    checkpoints = output / 'checkpoint'
    require(checkpoints.is_dir() and sorted(path.name for path in checkpoints.iterdir()) ==
            [f'{oid}.json' for oid in observations], 'checkpoint: absent, partial or additional observation records')
    unions, branches = {}, {}
    calls = channel_witnesses = certificates = proof_updates = leaves = group_tests = 0
    for oid, public in observations.items():
        record = outers[oid]
        require(set(record) == {'observation_id', 'arms'} and isinstance(record['arms'], dict) and
                list(record['arms']) == list(METHODS), f'{oid}: changed arm schema/order')
        same(load(checkpoints / f'{oid}.json'), record, f'{oid}: final outer/checkpoint byte-value consistency')
        image = tuple(exact(value) for value in public['observation'])
        require(exact(public['radius']) == RADIUS, f'{oid}: changed public pose radius')
        unions[oid], branches[oid] = {}, {}
        for method in METHODS:
            arm, where = record['arms'][method], f'{oid}/{method}'
            require(isinstance(arm, dict) and set(arm) == {'branches', 'outer', 'branch_seconds', 'outer_seconds'},
                    f'{where}: invalid arm schema')
            allowed = () if method == 'keep' else BRANCHES
            require(isinstance(arm['branches'], dict) and isinstance(arm['branch_seconds'], dict) and
                    list(arm['branches']) == list(arm['branch_seconds']) == list(allowed), f'{where}: changed whole-scene branch population')
            total = finite_seconds(arm['outer_seconds'], f'{where}: total solve time')
            measured = sum(finite_seconds(value, f'{where}: branch solve time') for value in arm['branch_seconds'].values())
            require(total >= measured and (method != 'keep' or total == 0), f'{where}: inconsistent measured solve costs')
            checked_branches = {}
            for branch in allowed:
                mode, outer = METHOD_MODES[method], arm['branches'][branch]
                checked = verify_trace(outer, image, f'{where}/{branch}', mode=mode,
                                       branch=branch, geometry_bounds=geometry_bounds)
                model = model_class(branch, camera_radius=RADIUS, colors=((Q(0), Q(1)),) * 3)
                n_calls, n_witnesses = verify_kernel_boxes(outer, checked, image, model, f'{where}/{branch}', mode != 'channel')
                calls += n_calls
                channel_witnesses += n_witnesses
                certificates += len(checked['shared_excluded'])
                proof_updates += checked['proof_updates']
                group_tests += outer['shared_tests']
                leaves += sum(node['status'] != 'split' for node in outer['trace'])
                checked_branches[branch] = checked
            branches[oid][method] = checked_branches
            unions[oid][method] = verify_union(arm['outer'], checked_branches, where,
                                               prior_only=method == 'keep', mode=METHOD_MODES.get(method, 'channel'))
    rows = []
    for decision in decision_list:
        task, method = tasks[decision['task_id']], decision['method']
        oid, checked = task['observation_id'], unions[task['observation_id']][method]
        result = verify_decision(decision, task['incumbent'], checked['hull'], f'{task["task_id"]}/{method}')
        row = make_row(task, truths[oid], decision, checked)
        if method != 'keep' and checked['hull'] is not None and row['covered']:
            truth, incumbent = exact(truths[oid]['truth']), exact(task['incumbent'])
            actual_gain = (incumbent - truth) ** 2 - (result - truth) ** 2
            require(actual_gain >= exact(decision['gain_lower']) >= 0, f'{task["task_id"]}/{method}: actual gain below certificate')
        rows.append(row)
    coverage = []
    for oid, private in truths.items():
        truth = exact(private['truth'])
        for method in METHODS:
            checked, branch_checks = unions[oid][method], branches[oid][method]
            hull = checked['hull']
            coverage.append({'observation_id': oid, 'case': private['case'], 'seed': private['seed'], 'method': method,
                             'covered': _historical.contains(checked['intervals'], truth), 'incompatible': hull is None,
                             'hull_width_mm': None if hull is None else str(hull[1] - hull[0]),
                             'retained_branches': checked['retained_branches'],
                             'branch_coverage': {name: _historical.contains(branch['intervals'], truth)
                                                 for name, branch in branch_checks.items()}})
    require(len(rows) == 2916 and len(coverage) == 324, 'evaluation: incorrect row/coverage denominator')
    aggregates = verify_evaluation(output, rows, coverage, decision_list, outers, status)
    claims = safety_claims(rows, unions, branches, truths)
    same(scientific_files(), sources, 'scientific source inventory unchanged throughout verification')
    same(frozen_hashes(), inputs, 'eight frozen historical output files unchanged throughout verification')
    return {'status': 'PASS', 'mechanical_verification': 'PASS', 'host': socket.gethostname(), 'output': str(output),
            'observations': 54, 'tasks': 486, 'decisions': 2916, 'coverage_rows': 324, 'summary_rows': len(aggregates),
            'branch_outer_traces': 810, 'terminal_partition_leaves': leaves, 'unresolved_budget_leaves': 0,
            'all_group_exclusion_certificates_verified': certificates, 'exact_propagation_updates_verified': proof_updates,
            'group_tests_accounted': group_tests, 'historical_kernel_evaluated_boxes_recomputed': calls,
            'all_channel_witnesses_recomputed': channel_witnesses, 'source_files_verified': len(sources),
            'frozen_output_inputs_verified': len(inputs), 'claims': claims,
            'checks': ['exact host/output identity, source inventory and eight frozen output inputs before/after',
                       'same54 frozen images/truth/tasks regenerated independently from historical point kernels',
                       'all511-budget BFS trees, bisections, full terminal partitions and zero unresolved leaves',
                       'all grouped exclusion rows and exact eight-round proof steps with independent group derivation',
                       'all old channel witnesses recomputed from historical continuous enclosures',
                       'whole-scene branch unions, retained-set truth membership and all frozen endpoint hybrid actions',
                       'all2916 actual-truth scores,324coverage records,540grouped summaries and CSV counterparts',
                       'recorded measured timing accounting and all54 per-arm correct KEEP/coverage,72negative KEEP,zero harm'],
            'limitations': ['Mechanical PASS does not imply the scientific C1 result is positive.',
                            'Historical point/enclosure/area kernels are shared; geometry soundness is not independently proved here.',
                            'Exclusion certificates are replayed; retained grouped boxes are not proved jointly feasible.',
                            'Nonexcluded group propagation and prior groups before a contradiction are not replayed; group test counts follow recorded valid group identities and fixed iteration order.',
                            'Measured times are checked and summed, not independently reproduced or used to claim a speed advantage.',
                            'Source/input hashes do not prove operating-system blindness or independent historical timestamps.',
                            'This is same-input synthetic replay with repeated seeds, not new independent scenes or real-MVS transfer.']}


def safety_claims(rows, unions, branches, truths):
    safety = {}
    for method in METHODS:
        values = [row for row in rows if row['method'] == method]
        correct = [row for row in values if row['state_group'] == 'correct']
        negative = [row for row in values if row['case'] == 'equal_full_negative' and row['state_group'] == 'small']
        harm = [row for row in values if exact(row['ae_mm']) > exact(row['initial_ae_mm'])]
        coverage = sum(_historical.contains(unions[oid][method]['intervals'], exact(private['truth']))
                       for oid, private in truths.items())
        require(len(values) == 486 and coverage == 54 and len(correct) == 54 and
                all(row['status'] == 'KEEP' and exact(row['ae_mm']) == 0 for row in correct) and not harm,
                f'safety: {method} failed true coverage/correct KEEP/zero harm')
        require(len(negative) == 72 and all(row['status'] == 'KEEP' for row in negative),
                f'negative control: {method} did not KEEP all72 offset states')
        require(sum(unions[oid][method]['budget_leaves'] for oid in truths) == 0,
                f'budget: {method} has unresolved leaves')
        for oid, private in truths.items():
            if private['case'] == 'equal_full_negative':
                require(unions[oid][method]['intervals'] == [DOMAIN], f'negative control: {method} lost full domain')
                if method != 'keep':
                    require(_arithmetic.merged_intervals(branches[oid][method]['full']['intervals']) == [DOMAIN],
                            f'negative control: {method} full branch lost domain')
        safety[method] = {'coverage': coverage, 'correct_kept': len(correct), 'harm': len(harm),
                          'negative_small_kept': len(negative), 'unresolved_budget_leaves': 0}
    endpoint = {}
    for method in METHODS:
        values = [row for row in rows if row['method'] == method and row['state_group'] == 'small']
        require(len(values) == 432, f'endpoints: {method} wrong offset denominator')
        errors = [exact(row['ae_mm']) for row in values]
        endpoint[method] = {'n': 432, 'improved': sum(error < exact(row['initial_ae_mm']) for error, row in zip(errors, values)),
                            'same': sum(error == exact(row['initial_ae_mm']) for error, row in zip(errors, values)),
                            'harm': sum(error > exact(row['initial_ae_mm']) for error, row in zip(errors, values)),
                            'mae_exact_mm': str(sum(errors, Q(0)) / 432),
                            'mse_exact_mm2': str(sum((error * error for error in errors), Q(0)) / 432)}
    global_gray, channel = endpoint['global_gray511'], endpoint['channel511']
    supported = global_gray['improved'] > channel['improved'] and exact(global_gray['mae_exact_mm']) < exact(channel['mae_exact_mm'])
    return {'contract_safety': safety,
            'C1': {'status': 'SUPPORTED' if supported else 'NEGATIVE', 'offset_endpoints': endpoint,
                   'negative_scientific_result_is_not_mechanical_verification_failure': True},
            'C2': {'fixed_deletion_comparisons': ['pixel_gray511', 'view_gray511', 'global_rgb511'],
                   'endpoint_policy': 'report fixed arms; no required per-task error monotonicity'},
            'scope': 'same frozen synthetic input replay, not new independent data or real-scene transfer'}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Independently verify frozen511-budget shared-material ablation evidence.')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/ablation-v1')
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if not output.is_dir() or not output.is_relative_to(ROOT) or output == ROOT:
        parser.error('existing output directory inside this exact experiment required')
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
        report = {'status': 'FAIL', 'host': socket.gethostname(), 'output': str(output),
                  'error': f'{type(error).__name__}: {error}'}
    report['utc'] = datetime.now(timezone.utc).isoformat()
    report['verification_seconds'] = time.perf_counter() - started
    with (destination / 'VERIFY.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""Independent exact A6 point-depth necessary-relaxation verification.

Reuse sealed A5 exact proof arithmetic and frozen A4-V independently authored
geometry/rows. Never import the new point solver or runner.
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
for _point_verifier_thread in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_point_verifier_thread] = '1'

ROOT = Path(__file__).resolve().parent
EXPECTED_ROOT = Path('/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/point-feasibility-1')
A5 = ROOT.parent / 'joint-feasibility-1'
A5_OUTPUT = A5 / 'outputs/polytope-v1'
_spec = importlib.util.spec_from_file_location('_point_frozen_independent_polytope', A5 / 'verify_polytope.py')
_a5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_a5)
_old = _a5._old
VerificationError = _a5.VerificationError
require, same, exact, encoded, pair = _a5.require, _a5.same, _a5.exact, _a5.encoded, _a5.pair
load, indexed, finite_seconds = _a5.load, _a5.indexed, _a5.finite_seconds
BRANCHES, METHODS, STATUSES = _a5.BRANCHES, _a5.METHODS, _a5.STATUSES
verify_certificate, verify_witness, verify_classification = _a5.verify_certificate, _a5.verify_witness, _a5.verify_classification
LIMIT = 3600
BUDGET_SCOPE = 'A6 exact incumbent-point diagnostic main plus independent verification; no reset'
RECORD_FIELDS = {'task_id', 'observation_id', 'unit_id', 'variation_index', 'method', 'incumbent',
                 'source_hull_contains', 'source_union_contains', 'local', 'global', 'box_status', 'local_status', 'global_status',
                 'a4_action', 'a5_action', 'diagnostic_seconds', 'lp_seconds'}


def point_geometry(public, branch, depth, node=None):
    """Reconstruct z,z with the same public camera uncertainty; no sampling."""
    depth = exact(depth)
    require(branch in BRANCHES and public['clipped'] is True and exact(public['baseline']) == 80 and
            exact(public['radius']) == _old.RADIUS and _old.DOMAIN[0] <= depth <= _old.DOMAIN[1],
            'point geometry: frozen B80/radius/clip and legal exact depth required')
    point = _old.independent_geometry(branch, _old.RADIUS, depth, depth, Q(80))
    if node is not None:
        require(node['status'] in ('retained_tolerance', 'unresolved_budget'), 'point geometry: original retained leaf required')
        low, high = exact(node['lo']), exact(node['hi'])
        require(low < high and low <= depth <= high, 'point geometry: original closed leaf must contain exact depth')
        interval = _old.independent_geometry(branch, _old.RADIUS, low, high, Q(80))
        require(len(point) == len(interval) == 144, 'point geometry: complete144 area bounds required')
        for pixel, (fixed, enclosing) in enumerate(zip(point, interval)):
            require(len(fixed) == len(enclosing) == 6 and
                    all(0 <= enclosing[index] <= fixed[index] <= fixed[index + 1] <= enclosing[index + 1] <= 1 for index in (0, 2, 4)),
                    f'point geometry: depth-point area bounds not nested in original leaf at pixel {pixel}')
    return point


def point_problem(public, arm, method, branch, depth, node=None):
    """Same necessary rows at fixed depth, global priors or exact old leaf box."""
    require(method in METHODS and arm.get('appearance_model') == 'bounded' and exact(arm.get('drift_bound')) == Q(1, 10),
            'point problem: registered bounded method/drift required')
    geometry = point_geometry(public, branch, depth, node)
    rows = _old.necessary_rows(tuple(map(exact, public['observation'])), geometry, True, method,
                              arm['calibration_intervals'], Q(1, 10))
    if node is None:
        bounds = [[encoded(0), encoded(1)] for _ in range(9)]
        if method == 'joint':
            bounds.extend(arm['calibration_intervals'])
    else:
        require(len(node['darkness_bounds']) == 9 and len(node['t_bounds']) == 3, 'point problem: original9 materials/3 biases required')
        bounds = list(node['darkness_bounds']) + (list(node['t_bounds']) if method == 'joint' else [])
    _a5.normalized_problem(rows, bounds)
    return rows, bounds


def containing_leaves(arm, depth):
    """All original closed-interval leaves, in original branch/ID list order."""
    depth = exact(depth)
    require(isinstance(arm['branches'], dict) and list(arm['branches']) == list(BRANCHES), 'containing leaves: frozen branch order required')
    result = []
    for branch in BRANCHES:
        value = arm['branches'][branch]
        nodes = indexed(value['trace'], lambda node: node['id'], 'containing leaves: original trace')
        ids = value['retained_ids']
        require(isinstance(ids, list) and all(type(ident) is int and ident in nodes for ident in ids) and len(ids) == len(set(ids)),
                'containing leaves: unique original integer IDs required')
        for ident in ids:
            node = nodes[ident]
            require(node['status'] in ('retained_tolerance', 'unresolved_budget'), 'containing leaves: original retained status required')
            if exact(node['lo']) <= depth <= exact(node['hi']):
                result.append((branch, node))
    return result


def aggregate_status(results, empty='SOURCE_EXCLUDED'):
    """OR branches/closed leaves: witness dominates; all proofs exclude."""
    require(isinstance(results, list) and empty in ('SOURCE_EXCLUDED', 'UNKNOWN'), 'aggregate: explicit valid empty scope required')
    statuses = [value['status'] for value in results]
    require(all(value in STATUSES for value in statuses), 'aggregate: valid exact statuses required')
    if not statuses:
        return empty
    if 'FEASIBLE_WITNESS' in statuses:
        return 'FEASIBLE_WITNESS'
    if all(value == 'CERTIFIED_INFEASIBLE' for value in statuses):
        return 'CERTIFIED_INFEASIBLE'
    return 'UNKNOWN'


def verify_global_lifting(public, arm, method, depth, branch, witness):
    """A global witness must also satisfy same-branch old interval/leaf boxes."""
    global_rows, global_box = point_problem(public, arm, method, branch, depth)
    verify_witness(global_rows, global_box, witness, 'lifting: exact global point witness')
    lifted = []
    for name, node in containing_leaves(arm, depth):
        if name != branch:
            continue
        point_rows, local_box = point_problem(public, arm, method, branch, depth, node)
        verify_witness(point_rows, local_box, witness, f'lifting/{branch}/{node["id"]}: same point in propagated old box')
        interval_rows, interval_box = _a5.leaf_problem(public, arm, method, branch, node)
        verify_witness(interval_rows, interval_box, witness, f'lifting/{branch}/{node["id"]}: enclosing original interval rows')
        lifted.append(node['id'])
    require(lifted, f'lifting/{branch}: global point witness lacks a containing original source leaf')
    return lifted


def verify_record(record, public, task, arm, a5_arm, a4_action, a5_action, where='record'):
    """Verify every original containing leaf and all three global branches."""
    require(isinstance(record, dict) and set(record) == RECORD_FIELDS and record['method'] in METHODS, f'{where}: changed point record schema')
    method, depth = record['method'], exact(task['incumbent'])
    expected = {'task_id': task['task_id'], 'observation_id': public['observation_id'], 'unit_id': public['unit_id'],
                'variation_index': public['variation_index'], 'incumbent': task['incumbent']}
    same({key: record[key] for key in expected}, expected, f'{where}: exact public task/image/incumbent identity')
    same(record['a4_action'], {key: a4_action[key] for key in ('status', 'output')}, f'{where}: unchanged A4 frozen hybrid context')
    same(record['a5_action'], {key: a5_action[key] for key in ('status', 'output')}, f'{where}: unchanged A5 frozen hybrid context')
    leaves = containing_leaves(arm, depth)
    hull = arm['outer']['hull']
    hull_contains = hull is not None and exact(hull[0]) <= depth <= exact(hull[1])
    require(type(record['source_hull_contains']) is bool and type(record['source_union_contains']) is bool,
            f'{where}: Boolean source containment required')
    same(record['source_hull_contains'], hull_contains, f'{where}: original hull containment')
    same(record['source_union_contains'], bool(leaves), f'{where}: original retained union containment without filling gaps')
    local, global_records = record['local'], record['global']
    require(isinstance(local, list) and len(local) == len(leaves), f'{where}: every original containing closed leaf must appear once')
    require(isinstance(global_records, list) and len(global_records) == 3, f'{where}: every global branch must appear once')
    same([value.get('branch') for value in global_records], list(BRANCHES), f'{where}: all global branches in original order')
    seconds, lp_seconds = 0.0, 0.0
    counts, local_by_branch = Counter({'local_checks': 0, 'global_checks': 0}), {branch: [] for branch in BRANCHES}
    original_boxes = {branch: indexed(a5_arm['polytope_records'][branch], lambda value: value['node_id'], 'sealed A5 classification IDs')
                      for branch in BRANCHES}
    for value, (branch, node) in zip(local, leaves):
        require(isinstance(value, dict) and set(value) == {'branch', 'node_id', 'lo', 'hi', 'box_classification', 'point_classification'} and
                type(value['node_id']) is int, f'{where}: invalid local item schema')
        same({key: value[key] for key in ('branch', 'node_id', 'lo', 'hi')},
             {'branch': branch, 'node_id': node['id'], 'lo': node['lo'], 'hi': node['hi']}, f'{where}: original containing leaf identity/order')
        same(value['box_classification'], original_boxes[branch][node['id']]['classification'], f'{where}: unchanged original A5 box evidence')
        interval_rows, interval_box = _a5.leaf_problem(public, arm, method, branch, node)
        verify_classification(interval_rows, interval_box, value['box_classification'], f'{where}/{branch}/{node["id"]}: frozen box classification')
        point_rows, local_box = point_problem(public, arm, method, branch, depth, node)
        result = value['point_classification']
        verify_classification(point_rows, local_box, result, f'{where}/{branch}/{node["id"]}: local fixed-depth classification')
        if result['status'] == 'FEASIBLE_WITNESS':
            global_rows, global_box = point_problem(public, arm, method, branch, depth)
            verify_witness(global_rows, global_box, result['witness'], f'{where}/{branch}/{node["id"]}: local witness also satisfies global point rows')
            verify_witness(interval_rows, interval_box, result['witness'], f'{where}/{branch}/{node["id"]}: local point witness satisfies enclosing interval rows')
            require(value['box_classification']['status'] != 'CERTIFIED_INFEASIBLE', f'{where}: valid local point witness conflicts with original box certificate')
        local_by_branch[branch].append(result)
        seconds += result['seconds']
        lp_seconds += result['lp_seconds']
        counts['local_' + result['status']] += 1
        counts['local_checks'] += 1
    for value in global_records:
        require(isinstance(value, dict) and set(value) == {'branch', 'point_classification'}, f'{where}: invalid global item schema')
        branch, result = value['branch'], value['point_classification']
        rows, bounds = point_problem(public, arm, method, branch, depth)
        verify_classification(rows, bounds, result, f'{where}/{branch}: global fixed-depth classification')
        if result['status'] == 'FEASIBLE_WITNESS':
            verify_global_lifting(public, arm, method, depth, branch, result['witness'])
            require(all(local['status'] != 'CERTIFIED_INFEASIBLE' for local in local_by_branch[branch]),
                    f'{where}/{branch}: global point witness conflicts with same-branch local certificate')
        if result['status'] == 'CERTIFIED_INFEASIBLE':
            require(all(local['status'] != 'FEASIBLE_WITNESS' for local in local_by_branch[branch]),
                    f'{where}/{branch}: local point witness conflicts with same-branch global certificate')
        seconds += result['seconds']
        lp_seconds += result['lp_seconds']
        counts['global_' + result['status']] += 1
        counts['global_checks'] += 1
    same(record['box_status'], aggregate_status([value['box_classification'] for value in local]), f'{where}: original box evidence OR aggregation')
    same(record['local_status'], aggregate_status([value['point_classification'] for value in local]), f'{where}: local point evidence OR aggregation')
    same(record['global_status'], aggregate_status([value['point_classification'] for value in global_records]), f'{where}: global point evidence OR aggregation')
    require(finite_seconds(record['lp_seconds'], where + ': total LP seconds') == lp_seconds and
            finite_seconds(record['diagnostic_seconds'], where + ': diagnostic seconds') >= seconds, f'{where}: false new-check cumulative timing')
    return dict(counts)


def check_target(output=None):
    require(socket.gethostname() == 'liekkas' and ROOT == EXPECTED_ROOT and ROOT.resolve() == EXPECTED_ROOT,
            'target: exact A6 child on liekkas required')
    if output is not None:
        output = Path(output).resolve()
        require(output.is_dir() and output.parent == ROOT / 'outputs', 'target: existing output at exact A6 child required')
    return output


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scientific_files():
    files = dict(_a5.scientific_files())
    for path in sorted({ROOT / 'AGENTS.md', ROOT / 'PROTOCOL.md', ROOT / 'REGISTRATION.json', *ROOT.glob('*.py')}):
        files[str(path.resolve())] = sha(path)
    return dict(sorted(files.items()))


def input_files():
    files = dict(_a5.input_files())
    for relative in ('SOURCE_LOCK.json', 'INPUTS.json', 'RUN_STATUS.json', 'STAGE_STATUS.json', 'verification/VERIFY.json', 'outers.json', 'decisions.json'):
        path = A5_OUTPUT / relative
        files[str(path.resolve())] = sha(path)
    return dict(sorted(files.items()))


def verify_locks(output):
    sources, inputs = scientific_files(), input_files()
    same(load(output / 'SOURCE_LOCK.json'), {'source_root': str(ROOT), 'files': sources}, 'SOURCE_LOCK: complete independent A6/frozen source graph')
    same(load(output / 'INPUTS.json'), {'files': inputs}, 'INPUTS: complete independent sealed input graph')
    _a5.verify_locks(A5_OUTPUT)
    receipt, stage, status = (load(A5_OUTPUT / relative) for relative in ('verification/VERIFY.json', 'STAGE_STATUS.json', 'RUN_STATUS.json'))
    require(receipt.get('status') == 'PASS' and receipt.get('host') == 'liekkas' and receipt.get('output') == str(A5_OUTPUT) and
            stage.get('status') == 'COMPLETE' and stage.get('independently_verified') is True and status.get('status') == 'COMPLETE',
            'INPUTS: exact frozen A5 COMPLETE/PASS target required')
    return sources, inputs


def verify_registration(registration):
    expected = {'stage': 'A6', 'host': 'liekkas', 'data_root': str(_a5.OLD / 'data'), 'source_output': str(A5_OUTPUT),
                'eta_levels': ['0', '1/10'], 'drift_bound': '1/10', 'observations': 36, 'tasks': 324, 'methods': list(METHODS),
                'diagnoses': 648, 'local_checks': 980, 'local_by_method': {'calibrated': 494, 'joint': 486}, 'global_checks': 1944,
                'new_checks': 2924, 'primary': {'eta_index': 1, 'depth_mm': '9007/14', 'n_per_method': 40},
                'budget_seconds': 3600, 'depth_splits_added': 0, 'new_geometry_outputs': 0,
                'source_stage': 'COMPLETE/PASS', 'science_success_required_for_mechanical_pass': False}
    same(registration, expected, 'REGISTRATION: fixed A6 diagnostic denominator/contract')


def verify_record_population(records, observations, tasks):
    require(isinstance(records, list) and len(records) == 648, 'records: complete648 saved diagnostic records required')
    indexed_records = indexed(records, lambda value: (value['task_id'], value['method']), 'saved point diagnosis identities')
    expected = [(f'{oid}s{state}', method) for oid in observations for method in METHODS for state in range(9)]
    same(list(indexed_records), expected, 'records: exact complete ordered36/two-method/nine-state matrix')
    require({ident for ident, _method in expected} == set(tasks), 'records: all public tasks required')
    require(sum(len(value['global']) for value in records) == 1944 and
            {method: sum(len(value['local']) for value in records if value['method'] == method) for method in METHODS} == {'calibrated': 494, 'joint': 486},
            'records: complete global1944/local494+486 evidence required')
    return indexed_records


def audit_runner_source(source=None):
    """Inspect the actual full648 save/read-back gate and sole private route."""
    path = ROOT / 'run_point.py'
    tree = ast.parse(path.read_text(encoding='utf-8') if source is None else source)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    require({'run', 'evaluate', 'check_records', 'diagnose', 'classify'} <= set(functions), 'runner audit: missing registered entry point')
    dump = lambda node: ast.dump(node, include_attributes=False)
    evaluations = [(function.name, node) for function in functions.values() for node in ast.walk(function)
                   if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'evaluate']
    require(len(evaluations) == 1 and evaluations[0][0] == 'run', 'runner audit: one sole post-diagnostic evaluation route required')
    evaluation = evaluations[0][1].lineno
    trials = [node for node in functions['run'].body if isinstance(node, ast.Try)]
    require(len(trials) == 1, 'runner audit: changed public gate flow')
    expressions = ("save(output/'diagnoses.json',records)", "records=load(output/'diagnoses.json')", 'check_records(records,ts)')
    positions = []
    for expression in expressions:
        matches = [node for node in trials[0].body if dump(node) == dump(ast.parse(expression).body[0])]
        require(len(matches) == 1 and matches[0].end_lineno < evaluation, 'runner audit: saved/read-back648 records gate missing before private')
        positions.append(matches[0].lineno)
    require(positions == sorted(positions), 'runner audit: reordered save/read-back gate')
    for predicate in ('len(records)!=648 or len(set(keys))!=648 or set(keys)!=expected',
                      "globals_count!=1944 or locals_count!={'calibrated':494,'joint':486}"):
        require(any(isinstance(node, ast.If) and dump(node.test) == dump(ast.parse(predicate, mode='eval').body) and
                    any(isinstance(child, ast.Raise) for child in node.body) for node in functions['check_records'].body),
                'runner audit: complete unique648/global1944/local980 population gate changed')
    private = [(function.name, node) for function in functions.values() for node in ast.walk(function)
               if isinstance(node, ast.Constant) and isinstance(node.value, str) and 'truth.json' in node.value]
    require([name for name, _node in private] == ['evaluate'] and len(private) ==
            sum(isinstance(node, ast.Constant) and isinstance(node.value, str) and 'truth.json' in node.value for node in ast.walk(tree)),
            'runner audit: private path outside sole post-gate evaluation')
    parses = [node for node in ast.walk(functions['evaluate']) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'load' and
              node.args and dump(node.args[0]) == dump(ast.parse("DATA/'private/truth.json'", mode='eval').body)]
    require(len(parses) == 1, 'runner audit: exact frozen private parse path required')
    return {'status': 'PASS', 'runner': str(path), 'post_readback_evaluation_line': evaluation,
            'private_parse_line': parses[0].lineno, 'persisted_gate_lines': positions}


def audit_executor_source(source=None):
    path = ROOT / 'execute_point.py'
    tree = ast.parse(path.read_text(encoding='utf-8') if source is None else source)
    execute = next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'execute'), None)
    require(execute is not None, 'executor audit: missing entry point')
    trials = [node for node in execute.body if isinstance(node, ast.Try)]
    require(len(trials) == 1, 'executor audit: changed owned stage flow')
    dump = lambda node: ast.dump(node, include_attributes=False)
    expressions = ('start=time.time()', 'deadline=start+runner.LIMIT',
                   "runner.save(output/'BUDGET.json',{'start_epoch':start,'deadline_epoch':deadline,'limit_seconds':runner.LIMIT,'scope':runner.BUDGET_SCOPE})",
                   'runner.run(output)', 'remaining=deadline-time.time()',
                   "subprocess.run([sys.executable,'-B',str(ROOT/'verify_point.py'),'--output',str(output)],check=True,timeout=remaining)")
    positions = []
    for expression in expressions:
        matches = [node for node in trials[0].body if dump(node) == dump(ast.parse(expression).body[0])]
        require(len(matches) == 1, 'executor audit: exact shared absolute main/verify chain required')
        positions.append(matches[0].lineno)
    require(positions == sorted(positions) and
            sum(isinstance(node, ast.Constant) and node.value == 'BUDGET.json' for node in ast.walk(execute)) == 1 and
            sum(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'run' and
                isinstance(node.func.value, ast.Name) and node.func.value.id == 'subprocess' for node in ast.walk(execute)) == 1,
            'executor audit: reordered/reset/additional budget or subprocess')
    require(not any(isinstance(node, ast.Constant) and isinstance(node.value, str) and 'truth.json' in node.value for node in ast.walk(tree)),
            'executor audit: private route')
    return {'status': 'PASS', 'executor': str(path), 'common_deadline_stage_lines': positions, 'budget_resets': 0}


def evaluation_row(record, task, public, fact):
    depth, truth = exact(task['incumbent']), exact(fact['truth'])
    return {'task_id': task['task_id'], 'observation_id': public['observation_id'], 'pair_id': public['pair_id'],
            'unit_id': public['unit_id'], 'variation_index': public['variation_index'], 'method': record['method'],
            'case': fact['case'], 'depth_mm': str(truth), 'incumbent_mm': str(depth), 'initial_ae_mm': str(abs(depth - truth)),
            'source_hull_contains': record['source_hull_contains'], 'source_union_contains': record['source_union_contains'],
            'box_status': record['box_status'], 'local_status': record['local_status'], 'global_status': record['global_status'],
            'local_checks': len(record['local']), 'a4_status': record['a4_action']['status'],
            'a4_output_mm': str(exact(record['a4_action']['output'])), 'a5_status': record['a5_action']['status'],
            'a5_output_mm': str(exact(record['a5_action']['output']))}


def summary_rows(rows):
    groups = {}
    for row in rows:
        prefix = (row['unit_id'], row['method'])
        shifted = exact(row['initial_ae_mm']) != 0
        cells = [('all', ''), ('all_shifted' if shifted else 'correct', '')]
        if shifted:
            informative = row['case'] != 'equal_full_negative'
            cells += [('informative_shifted' if informative else 'negative_shifted', ''), ('case_shifted', row['case'])]
            if informative:
                cells.append(('depth_informative', row['depth_mm']))
        for cell in cells:
            groups.setdefault(prefix + cell, []).append(row)
    result = []
    for (unit, method, group, value), selected in sorted(groups.items()):
        result.append({'unit_id': unit, 'method': method, 'group': group, 'value': value, 'n': len(selected),
                       **{label: dict(sorted(Counter(row[label] for row in selected).items())) for label in ('box_status', 'local_status', 'global_status')},
                       'new_point_excluded': sum(row['box_status'] not in ('CERTIFIED_INFEASIBLE', 'SOURCE_EXCLUDED') and
                                                 row['local_status'] == 'CERTIFIED_INFEASIBLE' for row in selected),
                       'a5_moves': sum(row['a5_status'] == 'MOVE' for row in selected)})
    return result


def verify_summary_denominators(summary):
    require(isinstance(summary, list), 'summary: list required')
    for unit in ('appearance_eta_0', 'appearance_eta_1_10'):
        for method in METHODS:
            cells = {(row['group'], row['value']): row for row in summary if (row['unit_id'], row['method']) == (unit, method)}
            for group, count in (('all', 162), ('all_shifted', 144), ('correct', 18), ('informative_shifted', 120), ('negative_shifted', 24)):
                require(cells.get((group, ''), {}).get('n') == count, f'{unit}/{method}: changed {group} denominator')
            depths = [row for (group, _value), row in cells.items() if group == 'depth_informative']
            cases = [row for (group, _value), row in cells.items() if group == 'case_shifted']
            require(len(depths) == 3 and all(row['n'] == 40 for row in depths) and len(cases) == 6 and all(row['n'] == 24 for row in cases) and
                    cells.get(('depth_informative', '9007/14'), {}).get('n') == 40, 'summary: changed full depth/template/primary40 matrix')


def actual_coverage(public, fact, arm, method, records):
    depth, branch = exact(fact['truth']), fact['rear_branch']
    darkness = [1 - exact(value) for value in fact['view_colors']]
    bias = [_old.BETA - exact(value) for value in fact['actual_view_bias']]
    witness = list(map(encoded, darkness + (bias if method == 'joint' else [])))
    leaves = verify_global_lifting(public, arm, method, depth, branch, witness)
    correct = [value for value in records if value['method'] == method and exact(value['incumbent']) == depth]
    require(len(correct) == 1 and correct[0]['source_union_contains'] is True and
            correct[0]['local_status'] not in ('CERTIFIED_INFEASIBLE', 'SOURCE_EXCLUDED') and correct[0]['global_status'] != 'CERTIFIED_INFEASIBLE',
            'coverage: actual correct-input point diagnosis must not be certified excluded')
    require(all(value['point_classification']['status'] != 'CERTIFIED_INFEASIBLE' for value in correct[0]['local'] if value['branch'] == branch) and
            all(value['point_classification']['status'] != 'CERTIFIED_INFEASIBLE' for value in correct[0]['global'] if value['branch'] == branch),
            'coverage: actual same-branch point proof conflicts with true necessary witness')
    return {'observation_id': public['observation_id'], 'method': method, 'generating_branch': branch,
            'global_actual_point_witness': True, 'all_containing_actual_leaf_witnesses': True, 'actual_leaf_ids': leaves}


def verify_budget(budget):
    require(isinstance(budget, dict) and set(budget) == {'start_epoch', 'deadline_epoch', 'limit_seconds', 'scope'}, 'BUDGET: invalid shared window')
    start, deadline = (finite_seconds(budget[key], 'BUDGET ' + key) for key in ('start_epoch', 'deadline_epoch'))
    require(type(budget['limit_seconds']) is int and budget['limit_seconds'] == LIMIT and deadline == start + LIMIT and budget['scope'] == BUDGET_SCOPE,
            'BUDGET: main and verifier must share one absolute3600-second window')
    if time.time() >= deadline:
        raise TimeoutError('A6 shared main/verification deadline exhausted')
    return deadline


@contextmanager
def deadline_guard(output):
    deadline = verify_budget(load(Path(output) / 'BUDGET.json'))
    require(not signal.getitimer(signal.ITIMER_REAL)[0], 'BUDGET: existing alarm cannot be reset')
    def reached(_signal, _frame):
        raise TimeoutError('A6 shared main/verification deadline reached')
    previous = signal.signal(signal.SIGALRM, reached)
    signal.setitimer(signal.ITIMER_REAL, max(0.000001, deadline - time.time()))
    try:
        yield deadline
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def verify(output):
    output = check_target(output)
    require(Path(sys.executable) == Path('/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python'), 'target: exact existing CPU runtime required')
    budget = load(output / 'BUDGET.json')
    deadline = verify_budget(budget)
    sources, inputs = verify_locks(output)
    registration = load(ROOT / 'REGISTRATION.json')
    verify_registration(registration)
    same(load(output / 'REGISTRATION.json'), registration, 'REGISTRATION: exact saved source copy')
    runner_audit, executor_audit = audit_runner_source(), audit_executor_source()
    observations, tasks, original_public, _original_tasks = _old.verify_public_population(output)
    records = load(output / 'diagnoses.json')
    verify_record_population(records, observations, tasks)
    prior, a4 = load(A5_OUTPUT / 'outers.json'), load(_a5.OLD_OUTPUT / 'outers.json')
    same(list(prior), list(observations), 'source outers: all36 exact frozen image identities/order')
    actions = load(A5_OUTPUT / 'decisions.json')
    old_actions = {name: indexed([value for value in actions if value['appearance_model'] == name],
                                lambda value: (value['task_id'], value['method']), 'sealed hybrid context ' + name)
                   for name in ('box', 'polytope')}
    counts, local_by_method = Counter(), Counter()
    for index, (oid, public) in enumerate(observations.items()):
        for method in METHODS:
            arm = prior[oid]['box'][method]
            same(arm, a4[oid]['bounded'][method], f'{oid}/{method}: all original A4 box/trace/variable bounds unchanged')
            for state in range(9):
                task = tasks[f'{oid}s{state}']
                record = records[index * 18 + METHODS.index(method) * 9 + state]
                verified = verify_record(record, public, task, arm, prior[oid]['polytope'][method],
                                         old_actions['box'][task['task_id'], method], old_actions['polytope'][task['task_id'], method],
                                         f'{task["task_id"]}/{method}')
                counts.update(verified)
                counts['diagnoses'] += 1
                local_by_method[method] += verified['local_checks']
        if index % 3 == 2:
            print(json.dumps({'stage': 'independent_point_verification', 'observations_complete': index + 1,
                              'remaining_seconds': round(deadline - time.time(), 1)}), flush=True)
    require(counts['diagnoses'] == 648 and counts['global_checks'] == 1944 and counts['local_checks'] == 980 and
            dict(local_by_method) == {'calibrated': 494, 'joint': 486}, 'evidence: complete648/1944/980 independent checks required')
    checkpoint = output / 'checkpoint'
    require(checkpoint.is_dir() and sorted(path.name for path in checkpoint.iterdir()) == sorted(f'{oid}.json' for oid in observations),
            'checkpoint: complete36 exact saved image pages required')
    for oid in observations:
        same(load(checkpoint / f'{oid}.json'), {'observation_id': oid, 'records': [value for value in records if value['observation_id'] == oid]},
             f'{oid}: complete persisted diagnostic checkpoint readback')
    # All648 public records, global1944/local980 exact evidence and saved
    # checkpoint readbacks have passed. Only now parse the frozen private facts.
    facts, actual_forward_witnesses = _old.verify_private_population(observations, tasks, original_public)
    rows = [evaluation_row(value, tasks[value['task_id']], observations[value['observation_id']], facts[value['observation_id']]) for value in records]
    coverage = [actual_coverage(public, facts[oid], prior[oid]['box'][method], method,
                                [value for value in records if value['observation_id'] == oid])
                for oid, public in observations.items() for method in METHODS]
    require(len(rows) == 648 and len(coverage) == 72, 'evaluation: complete648 rows/72 actual necessary witness coverage required')
    same(load(output / 'results.json'), rows, 'evaluation: all648 exact independent diagnostic strata/unchanged hybrid context')
    _old._baseline.verify_csv(output / 'results.csv', rows)
    same(load(output / 'coverage.json'), coverage, 'evaluation: all72 independent actual global/same-leaf necessary witnesses')
    summary = summary_rows(rows)
    verify_summary_denominators(summary)
    same(load(output / 'summary.json'), summary, 'evaluation: exact complete eta/method/state/depth/template diagnostics')
    status = load(output / 'RUN_STATUS.json')
    require(isinstance(status, dict) and set(status) == {'status', 'observations_complete', 'diagnoses_complete', 'local_checks',
            'global_checks', 'elapsed_seconds', 'deadline_epoch'}, 'RUN_STATUS: changed complete schema')
    same({key: value for key, value in status.items() if key != 'elapsed_seconds'},
         {'status': 'COMPLETE', 'observations_complete': 36, 'diagnoses_complete': 648, 'local_checks': 980,
          'global_checks': 1944, 'deadline_epoch': deadline}, 'RUN_STATUS: complete fixed diagnostic/common window')
    elapsed = finite_seconds(status['elapsed_seconds'], 'RUN_STATUS elapsed')
    require(elapsed < LIMIT and elapsed <= time.time() - budget['start_epoch'] and
            sum(finite_seconds(value['diagnostic_seconds'], 'record diagnostic seconds') for value in records) <= elapsed,
            'TIMING: impossible main/diagnostic duration')
    same(scientific_files(), sources, 'SOURCE_LOCK: old/new source bytes unchanged through verification')
    same(input_files(), inputs, 'INPUTS: frozen source/input bytes unchanged through verification')
    verify_budget(budget)
    return {'status': 'PASS', 'mechanical_verification': 'PASS', 'host': socket.gethostname(), 'output': str(output),
            'observations': 36, 'tasks': 324, 'diagnoses': 648, 'local_checks': 980, 'local_by_method': dict(local_by_method),
            'global_checks': 1944, 'new_checks': 2924, 'classification_counts': dict(sorted(counts.items())),
            'evaluation_rows': 648, 'actual_coverage_rows': 72, 'source_files_verified': len(sources), 'input_files_verified': len(inputs),
            'source_parse_order_audit': runner_audit, 'shared_budget_source_audit': executor_audit,
            'actual_forward_witnesses': actual_forward_witnesses,
            'scientific_scope': 'exact necessary point-depth linear relaxation; no physical forward feasibility or new geometry actions claimed',
            'shared_budget': {'start_epoch': budget['start_epoch'], 'deadline_epoch': deadline,
                              'elapsed_main_verification_seconds': time.time() - budget['start_epoch']}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=ROOT / 'outputs/point-v1')
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

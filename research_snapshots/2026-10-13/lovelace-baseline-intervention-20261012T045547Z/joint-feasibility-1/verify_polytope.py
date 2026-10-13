#!/usr/bin/env python3
"""Independent exact A5 weighted-contradiction and witness verification.

Only frozen A4-V verification geometry/necessary rows are reused. This module
does not import the new LP filter or runner and imports no private facts.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from copy import deepcopy
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
for _polytope_verifier_thread in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_polytope_verifier_thread] = '1'

ROOT = Path(__file__).resolve().parent
EXPECTED_ROOT = Path('/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/joint-feasibility-1')
OLD = ROOT.parent / 'appearance-variation-1'
OLD_OUTPUT = OLD / 'outputs/variation-v1'
_spec = importlib.util.spec_from_file_location('_polytope_frozen_independent_variation', OLD / 'verify_variation.py')
_old = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_old)
VerificationError = _old.VerificationError
require, same, exact, encoded, pair = _old.require, _old.same, _old.exact, _old.encoded, _old.pair
load, indexed, finite_seconds = _old.load, _old.indexed, _old.finite_seconds
BRANCHES, METHODS = _old.BRANCHES, _old.METHODS
ARMS = ('box', 'polytope')
STATUSES = ('CERTIFIED_INFEASIBLE', 'FEASIBLE_WITNESS', 'UNKNOWN')
CLASSIFICATION_FIELDS = {'status', 'certificate', 'witness', 'lp_status', 'seconds', 'lp_seconds',
                         'single_row_contradiction', 'single_row_contradiction_rows', 'row_count', 'dimension', 'reason'}
CERTIFICATE_FIELDS = {'sparse_weights', 'aggregate_coefficients', 'weighted_bound', 'box_min', 'margin'}
LIMIT = 3600
BUDGET_SCOPE = 'A5 full sealed-leaf main plus independent verification; no reset'


def rational(value, where='rational'):
    """Require the canonical exact JSON rational encoding, never a float."""
    require(isinstance(value, dict) and set(value) == {'numerator', 'denominator'} and
            type(value['numerator']) is int and type(value['denominator']) is int and value['denominator'] > 0,
            f'{where}: canonical integer rational required')
    result = Q(value['numerator'], value['denominator'])
    same(value, encoded(result), f'{where}: reduced positive-denominator rational required')
    return result


def normalized_problem(rows, bounds):
    """Independently negate lower rows to obtain A x <= b."""
    require(isinstance(bounds, (list, tuple)) and len(bounds) > 0, 'problem: nonempty box required')
    box = tuple(pair(value, 'problem box') for value in bounds)
    require(all(low <= high for low, high in box), 'problem: inverted box')
    require(isinstance(rows, (list, tuple)), 'problem: rows must be a sequence')
    normalized = []
    for index, row in enumerate(rows):
        require(isinstance(row, (tuple, list)) and len(row) == 3, f'problem row {index}: side/coefficient/bound triple required')
        side, coefficients, bound = row
        require(side in ('upper', 'lower') and isinstance(coefficients, (tuple, list)) and len(coefficients) == len(box),
                f'problem row {index}: invalid side/dimension')
        coefficients, bound = tuple(map(exact, coefficients)), exact(bound)
        if side == 'lower':
            coefficients, bound = tuple(-value for value in coefficients), -bound
        normalized.append((coefficients, bound))
    return tuple(normalized), box


def minimum_over_box(coefficients, box):
    return sum((coefficient * (low if coefficient >= 0 else high)
                for coefficient, (low, high) in zip(coefficients, box)), Q(0))


def single_row_contradictions(rows, bounds):
    normalized, box = normalized_problem(rows, bounds)
    return [index for index, (coefficients, bound) in enumerate(normalized)
            if minimum_over_box(coefficients, box) > bound]


def verify_certificate(rows, bounds, certificate, where='certificate'):
    """Recompute lambda A, lambda b and min(lambda A * box) using Fraction."""
    normalized, box = normalized_problem(rows, bounds)
    require(isinstance(certificate, dict) and set(certificate) == CERTIFICATE_FIELDS, f'{where}: missing/extra certificate fields')
    weights = certificate['sparse_weights']
    require(isinstance(weights, list) and weights, f'{where}: nonempty sparse weights required')
    aggregate, weighted_bound, seen = [Q(0)] * len(box), Q(0), set()
    for item in weights:
        require(isinstance(item, dict) and set(item) == {'row', 'weight'} and type(item['row']) is int and
                0 <= item['row'] < len(normalized) and item['row'] not in seen, f'{where}: invalid/duplicate row index')
        seen.add(item['row'])
        weight = rational(item['weight'], where + ': weight')
        require(weight >= 0, f'{where}: negative multiplier')
        coefficients, bound = normalized[item['row']]
        for index, coefficient in enumerate(coefficients):
            aggregate[index] += weight * coefficient
        weighted_bound += weight * bound
    supplied = certificate['aggregate_coefficients']
    require(isinstance(supplied, list) and len(supplied) == len(box), f'{where}: aggregate dimension mismatch')
    require(tuple(rational(value, where + ': aggregate') for value in supplied) == tuple(aggregate), f'{where}: changed weighted coefficients')
    minimum = minimum_over_box(aggregate, box)
    require(rational(certificate['weighted_bound'], where + ': bound') == weighted_bound, f'{where}: changed weighted bound')
    require(rational(certificate['box_min'], where + ': minimum') == minimum, f'{where}: changed signed box minimum')
    require(rational(certificate['margin'], where + ': margin') == minimum - weighted_bound, f'{where}: changed exact margin')
    require(minimum > weighted_bound, f'{where}: strict positive exact contradiction required')
    return {'aggregate_coefficients': tuple(aggregate), 'weighted_bound': weighted_bound,
            'box_min': minimum, 'margin': minimum - weighted_bound}


def verify_witness(rows, bounds, witness, where='witness'):
    """A witness must satisfy every original box and every normalized row."""
    normalized, box = normalized_problem(rows, bounds)
    require(isinstance(witness, list) and len(witness) == len(box), f'{where}: witness dimension mismatch')
    point = tuple(rational(value, where + ': coordinate') for value in witness)
    require(all(low <= value <= high for value, (low, high) in zip(point, box)), f'{where}: witness outside box')
    for index, (coefficients, bound) in enumerate(normalized):
        value = sum((coefficient * x for coefficient, x in zip(coefficients, point)), Q(0))
        require(value <= bound, f'{where}: witness violates row {index}')
    return point


def verify_classification(rows, bounds, result, where='classification'):
    normalized, box = normalized_problem(rows, bounds)
    require(isinstance(result, dict) and set(result) == CLASSIFICATION_FIELDS, f'{where}: changed classification schema')
    status = result['status']
    require(status in STATUSES and type(result['row_count']) is int and result['row_count'] == len(normalized) and
            type(result['dimension']) is int and result['dimension'] == len(box), f'{where}: changed status/problem size')
    require(result['lp_status'] is None or type(result['lp_status']) is int, f'{where}: invalid LP diagnostic')
    seconds, lp_seconds = finite_seconds(result['seconds'], where + ': seconds'), finite_seconds(result['lp_seconds'], where + ': LP seconds')
    require(lp_seconds <= seconds and isinstance(result['reason'], str) and result['reason'], f'{where}: invalid timing/reason')
    singles = single_row_contradictions(rows, bounds)
    require(type(result['single_row_contradiction']) is bool, f'{where}: Boolean single-row diagnostic required')
    same(result['single_row_contradiction'], bool(singles), f'{where}: false single-row contradiction diagnostic')
    same(result['single_row_contradiction_rows'], singles, f'{where}: complete independent single-row indices')
    if status == 'CERTIFIED_INFEASIBLE':
        require(result['witness'] is None, f'{where}: infeasible result carries witness')
        verify_certificate(rows, bounds, result['certificate'], where + ': certificate')
        return False
    require(result['certificate'] is None, f'{where}: retained result carries deletion certificate')
    if status == 'FEASIBLE_WITNESS':
        verify_witness(rows, bounds, result['witness'], where + ': witness')
    else:
        require(result['witness'] is None, f'{where}: UNKNOWN must have no asserted witness')
    return True


def leaf_problem(public, frozen_arm, method, branch, node):
    """Reconstruct the necessary rows from frozen public geometry and leaf box."""
    require(method in METHODS and branch in BRANCHES and node['status'] in ('retained_tolerance', 'unresolved_budget'),
            'leaf problem: retained frozen leaf required')
    low, high = exact(node['lo']), exact(node['hi'])
    require(low < high, 'leaf problem: invalid original depth interval')
    geometry = _old.independent_geometry(branch, exact(public['radius']), low, high, exact(public['baseline']))
    rows = _old.necessary_rows(tuple(map(exact, public['observation'])), geometry, public['clipped'], method,
                              frozen_arm['calibration_intervals'], Q(1, 10))
    bounds = list(node['darkness_bounds']) + (list(node['t_bounds']) if method == 'joint' else [])
    require(len(node['darkness_bounds']) == 9 and len(node['t_bounds']) == 3 and len(bounds) == (12 if method == 'joint' else 9),
            'leaf problem: frozen nine-material/three-bias dimensions required')
    normalized_problem(rows, bounds)
    return rows, bounds


def independent_union(frozen_outer, branches):
    """Rebuild the retained union independently while preserving frozen metadata."""
    source, budget_leaves = [], 0
    for branch in BRANCHES:
        value = branches[branch]
        ids = set(value['retained_ids'])
        for node in value['trace']:
            # Historical union metadata counts the full unchanged depth trace,
            # including budget leaves subsequently removed by an A5 certificate.
            budget_leaves += node['status'] == 'unresolved_budget'
            if node['id'] in ids:
                source.append({'branch': branch, 'id': node['id'], 'lo': node['lo'], 'hi': node['hi']})
    same(frozen_outer['unresolved_budget_leaves'], budget_leaves, 'union: frozen full-trace budget metadata')
    intervals = _old._arithmetic.merged_intervals((exact(node['lo']), exact(node['hi'])) for node in source)
    hull = None if not intervals else [encoded(intervals[0][0]), encoded(intervals[-1][1])]
    result = deepcopy(frozen_outer)
    result.update(source_leaves=source, retained_intervals=[[encoded(low), encoded(high)] for low, high in intervals],
                  hull=hull, retained_branches=[branch for branch in BRANCHES if branches[branch]['retained_ids']],
                  unresolved_budget_leaves=budget_leaves)
    return result


def verify_filtered_arm(arm, frozen_arm, public, method, where='filtered arm'):
    require(isinstance(arm, dict) and set(arm) == set(frozen_arm) | {'polytope_records', 'polytope_stats'}, f'{where}: changed arm schema')
    same({key: value for key, value in arm.items() if key not in ('branches', 'outer', 'polytope_records', 'polytope_stats')},
         {key: value for key, value in frozen_arm.items() if key not in ('branches', 'outer')}, f'{where}: frozen calibration/material/timing contract')
    require(isinstance(arm['polytope_records'], dict) and list(arm['polytope_records']) == list(BRANCHES) and
            isinstance(arm['branches'], dict) and list(arm['branches']) == list(BRANCHES), f'{where}: missing/reordered branch')
    branches, counts, seconds, lp_seconds = {}, Counter(), 0.0, 0.0
    for branch in BRANCHES:
        old_branch = frozen_arm['branches'][branch]
        nodes = indexed(old_branch['trace'], lambda node: node['id'], where + ': frozen nodes')
        ids = old_branch['retained_ids']
        records = arm['polytope_records'][branch]
        require(isinstance(records, list) and len(records) == len(ids), f'{where}/{branch}: every old retained leaf requires one record')
        same([record.get('node_id') for record in records], ids, f'{where}/{branch}: complete original leaf identity/order')
        retained = []
        for record in records:
            require(isinstance(record, dict) and set(record) == {'node_id', 'lo', 'hi', 'old_status', 'classification'} and
                    type(record['node_id']) is int, f'{where}/{branch}: invalid leaf record schema')
            node = nodes[record['node_id']]
            same({key: record[key] for key in ('lo', 'hi', 'old_status')},
                 {'lo': node['lo'], 'hi': node['hi'], 'old_status': node['status']}, f'{where}/{branch}/{node["id"]}: frozen leaf depth/status identity')
            rows, bounds = leaf_problem(public, frozen_arm, method, branch, node)
            result = record['classification']
            keep = verify_classification(rows, bounds, result, f'{where}/{branch}/{node["id"]}')
            if keep:
                retained.append(node['id'])
            counts[result['status']] += 1
            counts['leaf_checks'] += 1
            counts['single_row_contradiction'] += result['single_row_contradiction']
            counts['unresolved_budget_leaf_checks'] += node['status'] == 'unresolved_budget'
            seconds += result['seconds']
            lp_seconds += result['lp_seconds']
        expected = deepcopy(old_branch)
        intervals = [(exact(nodes[ident]['lo']), exact(nodes[ident]['hi'])) for ident in retained]
        expected.update(retained_ids=retained, hull=None if not intervals else [encoded(min(low for low, _ in intervals)), encoded(max(high for _, high in intervals))])
        same(arm['branches'][branch], expected, f'{where}/{branch}: only valid certified leaves may be removed; frozen trace unchanged')
        branches[branch] = {'retained': [nodes[ident] for ident in retained], 'intervals': intervals,
                            'hull': None if expected['hull'] is None else pair(expected['hull'], where)}
    union = independent_union(frozen_arm['outer'], arm['branches'])
    same(arm['outer'], union, f'{where}: exact filtered union/endpoints')
    stats = arm['polytope_stats']
    require(isinstance(stats, dict) and set(stats) == {'original_leaves', 'status_counts', 'single_row_contradictions',
            'budget_leaves_rebuilt', 'lp_seconds', 'filter_seconds'}, f'{where}: changed stats schema')
    same({key: value for key, value in stats.items() if key not in ('filter_seconds', 'lp_seconds')},
         {'original_leaves': counts['leaf_checks'],
          'status_counts': dict(sorted((status, counts[status]) for status in STATUSES if counts[status])),
          'single_row_contradictions': counts['single_row_contradiction'],
          'budget_leaves_rebuilt': counts['unresolved_budget_leaf_checks']}, f'{where}: independently recomputed leaf/status/diagnostic counts')
    require(finite_seconds(stats['lp_seconds'], where + ': cumulative LP') == lp_seconds and
            finite_seconds(stats['filter_seconds'], where + ': filter') >= seconds, f'{where}: false cumulative LP/filter timings')
    return {'union': {'hull': None if union['hull'] is None else pair(union['hull'], where),
                      'intervals': [pair(value, where) for value in union['retained_intervals']]},
            'branches': branches, 'counts': dict(counts), 'seconds': seconds, 'lp_seconds': lp_seconds}


def check_target(output=None):
    require(socket.gethostname() == 'liekkas' and ROOT == EXPECTED_ROOT and ROOT.resolve() == EXPECTED_ROOT,
            'target: exact A5 child on liekkas required')
    if output is not None:
        output = Path(output).resolve()
        require(output.is_dir() and output != ROOT and output.is_relative_to(ROOT), 'target: existing output inside exact A5 child required')
    return output


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scientific_files():
    files = dict(_old.scientific_files())
    for path in sorted({ROOT / 'AGENTS.md', ROOT / 'PROTOCOL.md', ROOT / 'REGISTRATION.json', *ROOT.glob('*.py')}):
        files[str(path.resolve())] = sha(path)
    return dict(sorted(files.items()))


def input_files():
    files = dict(_old.input_files())
    for relative in ('SOURCE_LOCK.json', 'INPUTS.json', 'RUN_STATUS.json', 'STAGE_STATUS.json', 'verification/VERIFY.json', 'outers.json', 'decisions.json'):
        path = OLD_OUTPUT / relative
        files[str(path.resolve())] = sha(path)
    return dict(sorted(files.items()))


def verify_locks(output):
    sources, inputs = scientific_files(), input_files()
    same(load(output / 'SOURCE_LOCK.json'), {'source_root': str(ROOT), 'files': sources}, 'SOURCE_LOCK: complete old/new exact source graph')
    same(load(output / 'INPUTS.json'), {'files': inputs}, 'INPUTS: complete independently enumerated sealed input graph')
    _old.verify_source_lock(load(OLD_OUTPUT / 'SOURCE_LOCK.json'))
    _old.verify_inputs(load(OLD_OUTPUT / 'INPUTS.json'))
    receipt, status, stage = (load(OLD_OUTPUT / relative) for relative in ('verification/VERIFY.json', 'RUN_STATUS.json', 'STAGE_STATUS.json'))
    require(receipt.get('status') == 'PASS' and receipt.get('host') == 'liekkas' and receipt.get('output') == str(OLD_OUTPUT) and
            status.get('status') == 'COMPLETE' and stage.get('status') == 'COMPLETE' and stage.get('independently_verified') is True,
            'INPUTS: exact frozen A4-V COMPLETE/PASS target required')
    return sources, inputs


def verify_registration(registration):
    expected = {'stage': 'A5', 'host': 'liekkas', 'data_root': str(OLD / 'data'), 'source_output': str(OLD_OUTPUT),
                'eta_levels': ['0', '1/10'], 'drift_bound': '1/10', 'observations': 36, 'tasks': 324,
                'case_count': 72, 'retained_leaf_count': 5103, 'new_actions': 648, 'baseline_actions': 648,
                'keep_actions': 324, 'evaluation_rows': 1620, 'primary': {'eta_index': 1, 'depth_mm': '9007/14', 'n_per_method': 40},
                'methods': list(METHODS), 'arms': list(ARMS), 'budget_seconds': 3600, 'depth_splits_added': 0,
                'row_changes': 0, 'source_stage': 'COMPLETE/PASS', 'science_success_required_for_mechanical_pass': False}
    same(registration, expected, 'REGISTRATION: exact fixed A5 matrix/scientific contract')


def verify_action_population(records, observations, tasks):
    require(isinstance(records, list) and len(records) == 1296, 'actions: complete1296 persisted method actions required')
    actions = indexed(records, lambda row: (row['task_id'], row['appearance_model'], row['method']), 'persisted A5 actions')
    expected = [(f'{oid}s{state}', name, method) for oid in observations for method in METHODS for name in ARMS for state in range(9)]
    same(list(actions), expected, 'actions: exact complete ordered36/two-method/two-arm/nine-state matrix')
    require({ident for ident, _name, _method in expected} == set(tasks), 'actions: incomplete public task mapping')
    require(Counter(row.get('reused') for row in records) == {True: 648, False: 648} and
            all(type(row.get('reused')) is bool and row['reused'] is (row['appearance_model'] == 'box') for row in records),
            'actions: exact648new/648baseline origin split required')
    return actions


def verify_action(decision, task, public, union, name, where='action'):
    action = _old._arithmetic.hybrid_action(union['hull'], task['incumbent'])
    expected = {'task_id': task['task_id'], 'unit_id': public['unit_id'], 'method': decision['method'],
                'appearance_model': name, 'reused': name == 'box',
                **{key: None if value is None else encoded(value) if key in ('output', 'gain_lower') else value for key, value in action.items()}}
    same(decision, expected, f'{where}: independent frozen hybrid action and exact endpoint certificate')


def checked_frozen_arm(arm, where='frozen arm'):
    branches = {}
    for branch in BRANCHES:
        value = arm['branches'][branch]
        nodes = indexed(value['trace'], lambda node: node['id'], where + ': trace')
        retained = [nodes[ident] for ident in value['retained_ids']]
        branches[branch] = {'retained': retained, 'intervals': [(exact(node['lo']), exact(node['hi'])) for node in retained]}
    outer = arm['outer']
    return {'union': {'hull': None if outer['hull'] is None else pair(outer['hull'], where),
                      'intervals': [pair(value, where) for value in outer['retained_intervals']]}, 'branches': branches}


def audit_runner_source(source=None):
    """Require the complete saved/read-back action gate before the only parse."""
    path = ROOT / 'run_polytope.py'
    tree = ast.parse(path.read_text(encoding='utf-8') if source is None else source)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    require({'run', 'evaluate', 'check_actions', 'check_keep', 'population', 'filter_arm', 'leaf_rows'} <= set(functions), 'runner audit: missing registered entry point')
    dump = lambda node: ast.dump(node, include_attributes=False)
    run = functions['run']
    evaluations = [node for node in ast.walk(run) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'evaluate']
    require(len(evaluations) == 1, 'runner audit: one private evaluation required')
    evaluation = evaluations[0].lineno
    callers = [(function.name, node.lineno) for function in functions.values() for node in ast.walk(function)
               if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'evaluate']
    require(callers == [('run', evaluation)], 'runner audit: additional evaluation route')
    trials = [node for node in run.body if isinstance(node, ast.Try)]
    require(len(trials) == 1, 'runner audit: changed public gate flow')
    expressions = ("save(output/'outers.json',arms)", "save(output/'decisions.json',ds)", "save(output/'keep/decisions.json',ks)",
                   "save(output/'TIMING.json',timing)", "ds=load(output/'decisions.json')", "ks=load(output/'keep/decisions.json')",
                   "arms=load(output/'outers.json')", 'check_actions(ds,ts)', 'check_keep(ks,ts)',
                   "if calls!=72 or len(arms)!=36:raise RuntimeError('incomplete72 cases')",
                   "if sum(c['original_leaves'] for c in timing)!=5103:raise RuntimeError('changed leaf population')")
    positions = []
    for expression in expressions:
        target = dump(ast.parse(expression).body[0])
        matches = [node for node in trials[0].body if dump(node) == target]
        require(len(matches) == 1 and matches[0].end_lineno < evaluation, 'runner audit: complete saved/read-back1296+324 gate missing before private evaluation')
        positions.append(matches[0].lineno)
    require(positions == sorted(positions), 'runner audit: reordered saved/read-back gates')
    predicates = (('check_actions', 'len(ds)!=1296 or len(set(keys))!=1296 or set(keys)!=expected'),
                  ('check_actions', "any(d['reused']!=(d['appearance_model']=='box') for d in ds)"),
                  ('check_keep', "len(ks)!=324 or len({d['task_id'] for d in ks})!=324 or {d['task_id'] for d in ks}!=set(ids)"),
                  ('check_keep', "any(d['method']!='keep' or d['status']!='KEEP' or d['output']!=ids[d['task_id']] for d in ks)"))
    for function, predicate in predicates:
        require(any(isinstance(node, ast.If) and dump(node.test) == dump(ast.parse(predicate, mode='eval').body) and
                    any(isinstance(child, ast.Raise) for child in node.body) for node in functions[function].body),
                'runner audit: exact unique complete action/KEEP population gate changed')
    private = [(function.name, node) for function in functions.values() for node in ast.walk(function)
               if isinstance(node, ast.Constant) and isinstance(node.value, str) and 'truth.json' in node.value]
    require([name for name, _node in private] == ['evaluate'] and len(private) ==
            sum(isinstance(node, ast.Constant) and isinstance(node.value, str) and 'truth.json' in node.value for node in ast.walk(tree)),
            'runner audit: private path outside sole post-action evaluation')
    parses = [node for node in ast.walk(functions['evaluate']) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'load' and
              node.args and dump(node.args[0]) == dump(ast.parse("DATA/'private/truth.json'", mode='eval').body)]
    require(len(parses) == 1, 'runner audit: exact frozen private parse path required')
    calls = [node for node in ast.walk(run) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'filter_arm']
    require(len(calls) == 1 and [dump(arg) for arg in calls[0].args] == [dump(ast.parse(expression, mode='eval').body) for expression in ('p', 'b', 'method', 'deadline')],
            'runner audit: filter only receives public/frozen arm/method/deadline')
    return {'status': 'PASS', 'runner': str(path), 'post_action_evaluation_call_line': evaluation,
            'private_parse_line': parses[0].lineno, 'persisted_matrix_gate_lines': positions}


def audit_executor_source(source=None):
    path = ROOT / 'execute_polytope.py'
    tree = ast.parse(path.read_text(encoding='utf-8') if source is None else source)
    execute = next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'execute'), None)
    require(execute is not None, 'executor audit: missing entry point')
    trials = [node for node in execute.body if isinstance(node, ast.Try)]
    require(len(trials) == 1, 'executor audit: changed owned stage flow')
    dump = lambda node: ast.dump(node, include_attributes=False)
    expressions = ('start=time.time()', 'deadline=start+runner.LIMIT',
                   "runner.save(output/'BUDGET.json',{'start_epoch':start,'deadline_epoch':deadline,'limit_seconds':runner.LIMIT,'scope':runner.BUDGET_SCOPE})",
                   'runner.run(output)', 'remaining=deadline-time.time()',
                   "subprocess.run([sys.executable,'-B',str(ROOT/'verify_polytope.py'),'--output',str(output)],check=True,timeout=remaining)")
    positions = []
    for expression in expressions:
        matches = [node for node in trials[0].body if dump(node) == dump(ast.parse(expression).body[0])]
        require(len(matches) == 1, 'executor audit: exact shared absolute main/verification chain changed')
        positions.append(matches[0].lineno)
    require(positions == sorted(positions), 'executor audit: reordered shared window')
    require(sum(isinstance(node, ast.Constant) and node.value == 'BUDGET.json' for node in ast.walk(execute)) == 1 and
            sum(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'run' and
                isinstance(node.func.value, ast.Name) and node.func.value.id == 'subprocess' for node in ast.walk(execute)) == 1,
            'executor audit: additional budget or subprocess')
    require(not any(isinstance(node, ast.Constant) and isinstance(node.value, str) and 'truth.json' in node.value for node in ast.walk(tree)),
            'executor audit: private data route')
    return {'status': 'PASS', 'executor': str(path), 'common_deadline_stage_lines': positions, 'budget_resets': 0}


def make_coverage(public, fact, method, checked, name):
    result = _old.make_coverage(public, fact, method, checked, 'bounded')
    result['appearance_model'] = name
    require(all(result[key] is True for key in ('covered', 'generating_branch_covered', 'latent_material_covered', 'joint_witness_covered')) and
            result['incompatible'] is False and (result['latent_bias_covered'] is True if method == 'joint' else result['latent_bias_covered'] is None),
            f'{public["observation_id"]}/{name}/{method}: physical simultaneous generating witness lost')
    return result


def verify_same_leaf_witness(public, fact, method, frozen_arm, before, after):
    truth, branch = exact(fact['truth']), fact['rear_branch']
    darkness = [1 - exact(value) for value in fact['view_colors']]
    bias = [_old.BETA - exact(value) for value in fact['actual_view_bias']]
    point = list(map(encoded, darkness + (bias if method == 'joint' else [])))
    candidates = []
    for node in before['branches'][branch]['retained']:
        if exact(node['lo']) <= truth <= exact(node['hi']):
            box = node['darkness_bounds'] + (node['t_bounds'] if method == 'joint' else [])
            if all(exact(low) <= value <= exact(high) for (low, high), value in zip(box, darkness + (bias if method == 'joint' else []))):
                rows, bounds = leaf_problem(public, frozen_arm, method, branch, node)
                verify_witness(rows, bounds, point, f'{public["observation_id"]}/{method}/{branch}/{node["id"]}: actual same-leaf necessary witness')
                candidates.append(node['id'])
    require(candidates, f'{public["observation_id"]}/{method}: old simultaneous original leaf witness missing')
    surviving = {node['id'] for node in after['branches'][branch]['retained']}
    require(set(candidates) <= surviving, f'{public["observation_id"]}/{method}: certified deletion removed an actual same-leaf witness')
    return {'observation_id': public['observation_id'], 'method': method, 'generating_branch': branch,
            'old_simultaneous_witness_leaf_ids': candidates, 'surviving_same_leaf_ids': [ident for ident in candidates if ident in surviving],
            'same_original_depth_material_bias_leaf_covered': True}


def verify_summary_denominators(summary):
    require(isinstance(summary, dict) and set(summary) == {'methods', 'keep'}, 'summary: changed groups')
    for name, cells in summary.items():
        configurations = [(method, arm) for arm in ARMS for method in METHODS] if name == 'methods' else [('keep', 'keep')]
        for unit in ('appearance_eta_0', 'appearance_eta_1_10'):
            for method, condition in configurations:
                selected = {(row['group'], row['value']): row for row in cells if (row['unit_id'], row['method'], row['condition']) == (unit, method, condition)}
                for group, count in (('all_shifted', 144), ('correct', 18), ('informative_shifted', 120), ('negative_shifted', 24)):
                    require(selected.get((group, ''), {}).get('n') == count, f'{name}/{unit}/{method}/{condition}: changed {group} denominator')
                depths = [row for (group, _value), row in selected.items() if group == 'depth_informative']
                cases = [row for (group, _value), row in selected.items() if group == 'case_shifted']
                require(len(depths) == 3 and all(row['n'] == 40 for row in depths) and len(cases) == 6 and all(row['n'] == 24 for row in cases),
                        'summary: changed registered depth/template matrix')
                require(selected.get(('depth_informative', '9007/14'), {}).get('n') == 40, 'summary: changed primary40 denominator')


def verify(output):
    output = check_target(output)
    require(Path(sys.executable) == Path('/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python'), 'target: exact existing CPU runtime required')
    budget = load(output / 'BUDGET.json')
    deadline = verify_budget(budget)
    sources, inputs = verify_locks(output)
    registration = load(ROOT / 'REGISTRATION.json')
    verify_registration(registration)
    same(load(output / 'REGISTRATION.json'), registration, 'REGISTRATION: complete saved fixed registration')
    runner_audit, executor_audit = audit_runner_source(), audit_executor_source()
    observations, tasks, old_public, _old_tasks = _old.verify_public_population(output)
    records, keeps = load(output / 'decisions.json'), load(output / 'keep/decisions.json')
    verify_action_population(records, observations, tasks)
    _old.verify_keep_population(keeps, observations, tasks)
    outers, frozen = load(output / 'outers.json'), load(OLD_OUTPUT / 'outers.json')
    require(isinstance(outers, dict) and list(outers) == list(observations), 'outers: complete ordered36 image configurations required')
    old_actions = indexed([row for row in load(OLD_OUTPUT / 'decisions.json') if row['appearance_model'] == 'bounded'],
                          lambda row: (row['task_id'], row['method']), 'frozen bounded actions')
    timing_records = load(output / 'TIMING.json')
    require(isinstance(timing_records, list) and len(timing_records) == 72, 'TIMING: complete72 post-filter calls required')
    timing = indexed(timing_records, lambda row: (row['observation_id'], row['method']), 'filter timing')
    same(list(timing), [(oid, method) for oid in observations for method in METHODS], 'TIMING: exact fixed72 call order')
    checked, counts = {}, Counter()
    for index, (oid, public) in enumerate(observations.items()):
        require(isinstance(outers[oid], dict) and list(outers[oid]) == list(ARMS) and
                all(isinstance(outers[oid][name], dict) and list(outers[oid][name]) == list(METHODS) for name in ARMS),
                f'{oid}: exact box/polytope two-method configurations required')
        checked[oid] = {name: {} for name in ARMS}
        for method in METHODS:
            old_arm = frozen[oid]['bounded'][method]
            same(outers[oid]['box'][method], old_arm, f'{oid}/{method}: complete sealed baseline arm identity')
            checked[oid]['box'][method] = checked_frozen_arm(old_arm, f'{oid}/box/{method}')
            result = verify_filtered_arm(outers[oid]['polytope'][method], old_arm, public, method, f'{oid}/polytope/{method}')
            checked[oid]['polytope'][method] = result
            counts.update(result['counts'])
            counts['filter_cases'] += 1
            same(timing[oid, method], {'observation_id': oid, 'variation_index': public['variation_index'], 'method': method,
                                      **outers[oid]['polytope'][method]['polytope_stats']}, f'{oid}/{method}: exact persisted filter timing/stats')
        if index % 3 == 2:
            print(json.dumps({'stage': 'independent_polytope_verification', 'observations_complete': index + 1,
                              'remaining_seconds': round(deadline - time.time(), 1)}), flush=True)
    require(counts['filter_cases'] == 72 and counts['leaf_checks'] == 5103, 'outers: complete fixed72 cases/5103 old leaves required')
    require(counts['unresolved_budget_leaf_checks'] == 0, 'outers: frozen matrix budget-leaf count changed')
    checkpoint = output / 'checkpoint'
    require(checkpoint.is_dir() and sorted(path.name for path in checkpoint.iterdir()) == sorted(f'{oid}.json' for oid in observations),
            'checkpoint: complete36 exact image checkpoints required')
    for oid in observations:
        same(load(checkpoint / f'{oid}.json'), {'observation_id': oid, 'arms': outers[oid]}, f'{oid}: persisted checkpoint readback')
    for record in records:
        task = tasks[record['task_id']]
        oid, name, method = task['observation_id'], record['appearance_model'], record['method']
        verify_action(record, task, observations[oid], checked[oid][name][method]['union'], name, f'{task["task_id"]}/{name}/{method}')
        if name == 'box':
            old = old_actions[task['task_id'], method]
            same({key: record[key] for key in ('status', 'output', 'gain_lower', 'reason', 'branch')},
                 {key: old[key] for key in ('status', 'output', 'gain_lower', 'reason', 'branch')}, 'baseline: all648 frozen actions exact')
    # Only after all5103 public leaf checks, all1296 saved method actions and
    # all324 KEEP identities pass may frozen private facts be parsed.
    facts, actual_witnesses = _old.verify_private_population(observations, tasks, old_public)
    results, keep_results, coverage, survival = [], [], [], []
    for record in records:
        task = tasks[record['task_id']]
        oid, name, method = task['observation_id'], record['appearance_model'], record['method']
        union = checked[oid][name][method]['union']
        _old._appearance.verify_safe_result(task, facts[oid], record, union, f'{task["task_id"]}/{name}/{method}')
        row = _old._appearance.make_result(task, facts[oid], record, union, name)
        row['variation_index'] = observations[oid]['variation_index']
        results.append(row)
    for record in keeps:
        task = tasks[record['task_id']]
        row = _old._appearance.make_result(task, facts[task['observation_id']], record, {'hull': None}, 'keep')
        row['variation_index'] = observations[task['observation_id']]['variation_index']
        keep_results.append(row)
    for oid, public in observations.items():
        for name in ARMS:
            for method in METHODS:
                coverage.append(make_coverage(public, facts[oid], method, checked[oid][name][method], name))
        for method in METHODS:
            survival.append(verify_same_leaf_witness(public, facts[oid], method, frozen[oid]['bounded'][method],
                                                     checked[oid]['box'][method], checked[oid]['polytope'][method]))
    require(len(coverage) == 144 and len(survival) == 72, 'coverage: complete same-leaf witness matrix required')
    _old.verify_registered_keeps(results, facts, tasks)
    same(load(output / 'results.json'), results, 'evaluation: all1296 exact independent method endpoints/errors')
    same(load(output / 'keep/results.json'), keep_results, 'evaluation: all324 exact KEEP reference endpoints/errors')
    same(load(output / 'coverage.json'), coverage, 'evaluation: all144 independent simultaneous physical coverage records')
    for relative, rows in (('results.csv', results), ('keep/results.csv', keep_results)):
        _old._baseline.verify_csv(output / relative, rows)
    summary = {'methods': _old._appearance.aggregate_results(results), 'keep': _old._appearance.aggregate_results(keep_results)}
    verify_summary_denominators(summary)
    same(load(output / 'summary.json'), summary, 'evaluation: all exact eta/arm/method/depth/template cells')
    status = load(output / 'RUN_STATUS.json')
    require(isinstance(status, dict) and set(status) == {'status', 'observations_complete', 'cases_complete', 'decisions_complete',
            'leaf_checks', 'elapsed_seconds', 'deadline_epoch'}, 'RUN_STATUS: changed schema')
    same({key: value for key, value in status.items() if key != 'elapsed_seconds'},
         {'status': 'COMPLETE', 'observations_complete': 36, 'cases_complete': 72, 'decisions_complete': 1296,
          'leaf_checks': 5103, 'deadline_epoch': deadline}, 'RUN_STATUS: complete fixed matrix/common window')
    elapsed = finite_seconds(status['elapsed_seconds'], 'RUN_STATUS elapsed')
    require(elapsed < LIMIT and elapsed <= time.time() - budget['start_epoch'] and
            sum(finite_seconds(row['filter_seconds'], 'TIMING filter') for row in timing_records) <= elapsed,
            'TIMING: impossible main/filter duration')
    same(scientific_files(), sources, 'SOURCE_LOCK: old/new bytes unchanged through verification')
    same(input_files(), inputs, 'INPUTS: frozen input bytes unchanged through verification')
    verify_budget(budget)
    return {'status': 'PASS', 'mechanical_verification': 'PASS', 'host': socket.gethostname(), 'output': str(output),
            'observations': 36, 'tasks': 324, 'filter_cases': 72, 'old_retained_leaves_verified': 5103,
            'classification_counts': dict(sorted((status, counts[status]) for status in STATUSES if counts[status])),
            'single_row_contradictions_verified': counts['single_row_contradiction'], 'budget_leaves_rebuilt': counts['unresolved_budget_leaf_checks'],
            'new_decisions': 648, 'baseline_decisions': 648, 'keep_decisions': 324, 'evaluation_rows': 1620, 'coverage_rows': 144,
            'source_files_verified': len(sources), 'input_files_verified': len(inputs), 'source_parse_order_audit': runner_audit,
            'shared_budget_source_audit': executor_audit, 'actual_truth_witnesses': actual_witnesses, 'same_original_leaf_witness_survival': survival,
            'scientific_results': 'no positive gain or directional subset acceptance condition; exact linear relaxation only',
            'shared_budget': {'start_epoch': budget['start_epoch'], 'deadline_epoch': deadline,
                              'elapsed_main_verification_seconds': time.time() - budget['start_epoch']}}


def verify_budget(budget):
    require(isinstance(budget, dict) and set(budget) == {'start_epoch', 'deadline_epoch', 'limit_seconds', 'scope'}, 'BUDGET: invalid shared window')
    start, deadline = (finite_seconds(budget[key], 'BUDGET ' + key) for key in ('start_epoch', 'deadline_epoch'))
    require(type(budget['limit_seconds']) is int and budget['limit_seconds'] == LIMIT and deadline == start + LIMIT and
            budget['scope'] == BUDGET_SCOPE, 'BUDGET: main and verifier must share one absolute3600-second window')
    if time.time() >= deadline:
        raise TimeoutError('A5 shared main/verification deadline exhausted')
    return deadline


@contextmanager
def deadline_guard(output):
    deadline = verify_budget(load(Path(output) / 'BUDGET.json'))
    require(not signal.getitimer(signal.ITIMER_REAL)[0], 'BUDGET: existing alarm cannot be reset')
    def reached(_signal, _frame):
        raise TimeoutError('A5 shared main/verification deadline reached')
    previous = signal.signal(signal.SIGALRM, reached)
    signal.setitimer(signal.ITIMER_REAL, max(0.000001, deadline - time.time()))
    try:
        yield deadline
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=ROOT / 'outputs/polytope-v1')
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

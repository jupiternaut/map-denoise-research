#!/usr/bin/env python3
"""Independent arithmetic verification of the registered hybrid confirmation.

The old verifier supplies unchanged arithmetic, trace and aggregation helpers
through an explicit file import. Neither experiment's selector or runner is
imported. The disclosed image kernels check seeded truth images and a bounded
sample of enclosure witnesses; residual grid optimality is not rerun.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import socket
import sys
import time

sys.dont_write_bytecode = True
Q = Fraction
ROOT = Path(__file__).resolve().parents[1]
DEPENDENCY_ROOT = Path('/srv/slam-research/grf/map-denoise/runs/lovelace-challenge-20261010T205013Z')
SOURCE_COMMIT = 'bd8d05aad65e2c55cd9cd288beae4ef949606984'
_spec = importlib.util.spec_from_file_location(
    '_frozen_challenge_arithmetic', DEPENDENCY_ROOT / 'verification' / 'verify_results.py')
_old = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_old)

# The dependency's constants and functions are never monkeypatched.
VerificationError = _old.VerificationError
require, exact, pair, encoded = _old.require, _old.exact, _old.pair, _old.encoded
indexed, verify_outer = _old.indexed, _old.verify_outer
expected_row, verify_row = _old.expected_row, _old.verify_row
verify_summary_rows = _old.verify_summary_rows
recompute_summary, finite_number = _old.recompute_summary, _old.finite_number
unique_object = _old.unique_object

KINDS = ('lovelace_triangle', 'rectangle', 'layered_contrast', 'layered_same', 'layered_hidden')
RADII = (Q(0), Q(1, 5), Q(1))
TRUTHS = (Q(6217, 11), Q(7864, 13), Q(8441, 13))
SEEDS = (41001, 41002, 41003)
OFFSETS = (Q(0), Q(-1, 4), Q(1, 4), Q(-3, 4), Q(3, 4), Q(-2), Q(2), Q(-60), Q(60))
METHODS = ('keep', 'certificate_midpoint', 'certificate_projection', 'certificate_hybrid', 'pose_profiled_residual')
CERTIFICATES = METHODS[1:4]
DOMAIN = (Q(540), Q(660))
EPSILON, TOLERANCE, MAX_BOXES = Q(1, 100), Q(15, 32), 511
WEIGHTS = (Q(-1), Q(-1, 2), Q(0), Q(1, 2), Q(1))
CHANNELS = 3 * 8 * 6 * 3
REQUIRED = {**_old.REQUIRED, 'protocol': 'PROTOCOL.json', 'source_lock': 'SOURCE_LOCK.json',
            'run_status': 'RUN_STATUS.json'}


def certificate_action(hull, incumbent, method):
    """Recompute the two endpoint gains and midpoint-first branch independently."""
    require(method in CERTIFICATES, 'unregistered certificate method')
    if method != 'certificate_hybrid':
        return _old.certificate_action(hull, incumbent, method)
    midpoint = _old.certificate_action(hull, incumbent, 'certificate_midpoint')
    if midpoint['status'] == 'MOVE':
        return {**midpoint, 'branch': 'midpoint'}
    projected = _old.certificate_action(hull, incumbent, 'certificate_projection')
    return {**projected, 'branch': {'MOVE': 'projection', 'KEEP': 'keep',
                                  'INCOMPATIBLE': 'incompatible'}[projected['status']]}


def verify_decision(decision, incumbent, hull, where):
    method = decision['method']
    require(method in METHODS, f'{where}: unregistered method')
    if method != 'certificate_hybrid':
        return _old.verify_decision(decision, incumbent, hull, where)
    fields = {'task_id', 'method', 'status', 'output', 'gain_lower', 'reason', 'branch'}
    require(set(decision) == fields, f'{where}: incorrect hybrid decision schema')
    expected = certificate_action(hull, incumbent, method)
    for field, value in expected.items():
        actual = decision[field]
        if field in {'output', 'gain_lower'} and actual is not None:
            actual = exact(actual, f'{where}.{field}')
        require(actual == value, f'{where}: incorrect {field}; got {actual!r}, expected {value!r}')
    return exact(decision['output'])


def verify_paired(task_id, decisions, truth):
    """Verify C1 task by task, rather than comparing aggregate means/counts."""
    midpoint = decisions[(task_id, 'certificate_midpoint')]
    projection = decisions[(task_id, 'certificate_projection')]
    hybrid = decisions[(task_id, 'certificate_hybrid')]
    require(abs(exact(hybrid['output']) - truth) <= abs(exact(midpoint['output']) - truth),
            f'{task_id}: hybrid error exceeds midpoint error')
    if midpoint['status'] == 'MOVE':
        require(hybrid['status'] == 'MOVE' and hybrid['branch'] == 'midpoint' and
                hybrid['output'] == midpoint['output'], f'{task_id}: accepted midpoint output changed')
    require((hybrid['status'] == 'MOVE') == (projection['status'] == 'MOVE'),
            f'{task_id}: hybrid MOVE set differs from projection')


def source_hashes():
    """Independently enumerate both source trees using absolute canonical keys."""
    files = list(DEPENDENCY_ROOT.glob('*.py'))
    files += list((DEPENDENCY_ROOT / 'source_snapshot').rglob('*.py'))
    files += list((DEPENDENCY_ROOT / 'source_snapshot').rglob('*.json'))
    files += list((DEPENDENCY_ROOT / 'verification').rglob('*.py'))
    files += list((DEPENDENCY_ROOT / 'verification').rglob('*.json'))
    files += [DEPENDENCY_ROOT / 'PROTOCOL.md']
    files += list(ROOT.glob('*.py')) + list((ROOT / 'verification').rglob('*.py'))
    files += list((ROOT / 'verification').rglob('*.json'))
    files += [ROOT / 'PROTOCOL.md', ROOT / 'THEORY.md']
    require(all(p.is_file() for p in files), 'source inventory: missing registered input')
    require(len(files) == len(set(files)), 'source inventory: duplicate input path')
    return {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def verify_source_lock(lock, current):
    require(isinstance(lock, dict), 'SOURCE_LOCK: expected object')
    require(lock.get('source_commit') == SOURCE_COMMIT, 'SOURCE_LOCK: changed source commit')
    require(lock.get('new_experiment_root') == str(ROOT.resolve()) and
            lock.get('read_only_dependency_root') == str(DEPENDENCY_ROOT.resolve()) and
            lock.get('path_format') == 'absolute' and lock.get('historical_outputs_used') is False,
            'SOURCE_LOCK: changed target/dependency/path provenance')
    files = lock.get('files')
    require(isinstance(files, dict) and files, 'SOURCE_LOCK: no source inventory')
    require(all(isinstance(p, str) and Path(p).is_absolute() for p in files),
            'SOURCE_LOCK: every source key must be absolute')
    require(set(files) == set(current), 'SOURCE_LOCK: missing/extra registered source files')
    for path, digest in current.items():
        require(files[path] == digest, f'SOURCE_LOCK: scientific source changed: {path}')
    return len(current)


def verify_protocol(protocol):
    require(isinstance(protocol, dict), 'PROTOCOL: expected object')
    expected = {'kinds': list(KINDS), 'camera_radii_mm': list(map(str, RADII)),
                'truths_evaluation_only': list(map(str, TRUTHS)), 'seeds': list(SEEDS),
                'offsets_evaluation_only': list(map(str, OFFSETS)), 'epsilon': str(EPSILON),
                'tolerance_mm': str(TOLERANCE), 'max_boxes': MAX_BOXES,
                'expected_observations': 135, 'expected_tasks': 1215, 'methods': list(METHODS),
                'expected_decisions': 6075, 'random_tape_prefix': 'hybrid-v1',
                'depth_grid_step_mm': '1/2', 'pose_grid_multipliers': list(map(str, WEIGHTS)),
                'tie_threshold': 1e-12, 'depth_domain_mm': ['540', '660'], 'cpu_threads': 1,
                'gpu_count': 0, 'wall_time_limit_seconds': 1800,
                'scope': 'known-shape known-radiance scalar depth; lateral camera error; partial opaque-layer visibility'}
    require(set(protocol) == set(expected), 'PROTOCOL: missing/extra registered parameters')
    for field, value in expected.items():
        require(protocol.get(field) == value, f'PROTOCOL: changed/missing {field}')
        if type(value) in (int, float):
            require(type(protocol[field]) == type(value), f'PROTOCOL: incorrect numeric type for {field}')


def paired_statistics(rows, decisions):
    """Independently filter the recorded tasks into all registered paired strata."""
    rows_by_key = indexed(rows, lambda r: (r['task_id'], r['method']), 'derived paired rows')
    hybrid_rows = [r for r in rows if r['method'] == 'certificate_hybrid']
    result = []
    for information in ('hidden', 'informative', 'overall'):
        for prior in ('all', 'inside_prior', 'outside_prior'):
            for state in ('all', 'correct', 'large', 'small'):
                selected = [r for r in hybrid_rows if
                            (information == 'overall' or (r['kind'] == 'layered_hidden') == (information == 'hidden')) and
                            (prior == 'all' or r['prior_position'] == prior) and
                            (state == 'all' or r['state_group'] == state)]
                if not selected:
                    continue
                counts, branches = Counter(), Counter()
                for hybrid in selected:
                    tid = hybrid['task_id']
                    midpoint = rows_by_key[(tid, 'certificate_midpoint')]
                    projection = rows_by_key[(tid, 'certificate_projection')]
                    h, m = exact(hybrid['ae_mm']), exact(midpoint['ae_mm'])
                    counts['ae_no_worse_than_midpoint'] += h <= m
                    counts['mse_no_worse_than_midpoint'] += h*h <= m*m
                    counts['strictly_lower_ae_than_midpoint'] += h < m
                    counts['projection_move_membership_equal'] += ((hybrid['status'] == 'MOVE') == (projection['status'] == 'MOVE'))
                    h_decision = decisions[(tid, 'certificate_hybrid')]
                    m_decision = decisions[(tid, 'certificate_midpoint')]
                    if midpoint['status'] == 'MOVE':
                        counts['midpoint_moves'] += 1
                        counts['midpoint_move_decision_preserved'] += (
                            {k: v for k, v in h_decision.items() if k not in ('method', 'branch')} ==
                            {k: v for k, v in m_decision.items() if k != 'method'})
                    branches[h_decision['branch']] += 1
                result.append({'information': information, 'prior_position': prior, 'state_group': state,
                               'n': len(selected), **{name: counts[name] for name in
                               ('ae_no_worse_than_midpoint', 'mse_no_worse_than_midpoint', 'strictly_lower_ae_than_midpoint',
                                'projection_move_membership_equal', 'midpoint_moves', 'midpoint_move_decision_preserved')},
                               'hybrid_branch_counts': dict(sorted(branches.items()))})
    return result


def verify_extended_summary(summary, derived_rows, decisions):
    strata = summary['information_strata']
    require(isinstance(strata, dict) and set(strata) == {'informative', 'hidden'},
            'SUMMARY.information_strata: missing/extra information groups')
    checked = 0
    for name in ('informative', 'hidden'):
        selected = [r for r in derived_rows if (r['kind'] == 'layered_hidden') == (name == 'hidden')]
        checked += len(verify_summary_rows(strata[name], selected))
    branches = dict(sorted(Counter(d['branch'] for d in decisions.values() if d['method'] == 'certificate_hybrid').items()))
    require(summary['hybrid_branch_counts'] == branches and
            all(type(v) is int for v in summary['hybrid_branch_counts'].values()),
            'SUMMARY.hybrid_branch_counts: incorrect branch totals')
    paired = paired_statistics(derived_rows, decisions)
    require(summary['paired_relations'] == paired, 'SUMMARY.paired_relations: incorrect paired strata')
    for row in summary['paired_relations']:
        require(all(type(value) is int for key, value in row.items() if key not in
                    {'information', 'prior_position', 'state_group', 'hybrid_branch_counts'}) and
                all(type(v) is int for v in row['hybrid_branch_counts'].values()),
                'SUMMARY.paired_relations: noninteger count')
    return checked, len(paired)


def model_spot_checks(model, observation, truth_record, verified_outer, where, witness_samples):
    """Rebuild the new RNG tape; never reuse the old challenge prefix."""
    truth, radius = exact(truth_record['truth']), exact(observation['radius'])
    seed = truth_record['seed']
    rng = random.Random(f"hybrid-v1:{observation['kind']}:{radius}:{truth}:{seed}")
    expected_offsets = (radius * rng.choice(WEIGHTS), Q(0), radius * rng.choice(WEIGHTS))
    offsets = tuple(exact(v) for v in truth_record['actual_offsets'])
    require(offsets == expected_offsets, f'{where}: camera offsets do not match registered seed')
    prediction = model.point_prediction(truth, offsets)
    obs = tuple(exact(v) for v in observation['observation'])
    require(len(prediction) == len(obs), f'{where}: shared prediction dimension mismatch')
    noise = tuple(value - predicted for value, predicted in zip(obs, prediction))
    require(all(abs(value) <= EPSILON for value in noise), f'{where}: actual observation noise exceeds bound')
    require(noise == tuple(EPSILON * rng.choice(WEIGHTS) for _ in obs),
            f'{where}: observation noise does not match registered hybrid-v1 seed')
    require(exact(truth_record['noise_linf']) == max(map(abs, noise)), f'{where}: false noise_linf')
    excluded = verified_outer['excluded']
    selected = sorted(excluded, key=lambda n: hashlib.sha256(f"{where}:{n['id']}".encode()).digest())[:witness_samples]
    for node in selected:
        low, high = verified_outer['boxes'][node['id']]
        enclosure = model.enclosure(low, high)
        require(len(enclosure) == len(obs), f'{where}: shared enclosure dimension mismatch')
        channel = node['witness_channel']
        claimed = pair(node['prediction_interval'], where)
        recomputed = tuple(exact(v) for v in enclosure[channel])
        require(claimed == recomputed, f"{where}: forged exclusion interval at node {node['id']}")
    return len(selected)


def verify_csv_rows(fieldnames, csv_rows, json_rows):
    require(bool(json_rows), 'ROWS.csv: empty JSON population')
    require(fieldnames == list(json_rows[0]), 'ROWS.csv: incorrect/duplicate/reordered columns')
    expected = [{field: str(value) for field, value in row.items()} for row in json_rows]
    require(csv_rows == expected, 'ROWS.csv: CSV records differ from JSON rows')
    return len(expected)


def verify_bundle(bundle, *, models=None, witness_samples=2):
    require(type(witness_samples) is int and 1 <= witness_samples <= 8, 'witness sample count must be 1..8')
    counts = {'observations': 135, 'truth': 135, 'outers': 135, 'tasks': 1215,
              'decisions': 6075, 'rows': 6075}
    for name, expected in counts.items():
        records = bundle[name]
        require(isinstance(records, list) and len(records) == expected,
                f'{name}: expected {expected} nonempty registered records')
    observations = indexed(bundle['observations'], lambda r: r['observation_id'], 'observations')
    truths = indexed(bundle['truth'], lambda r: r['observation_id'], 'truth')
    outers = indexed(bundle['outers'], lambda r: r['observation_id'], 'outers')
    tasks = indexed(bundle['tasks'], lambda r: r['task_id'], 'tasks')
    decisions = indexed(bundle['decisions'], lambda r: (r['task_id'], r['method']), 'decisions')
    rows = indexed(bundle['rows'], lambda r: (r['task_id'], r['method']), 'ROWS')
    require(set(observations) == set(truths) == set(outers), 'observation/truth/outer IDs differ')
    specs = [(kind, radius, truth, seed) for kind in KINDS for radius in RADII for truth in TRUTHS for seed in SEEDS]
    expected_task_ids, verified_outers = set(), {}
    for index, (kind, radius, truth, seed) in enumerate(specs):
        oid = f'o{index:03d}'
        require(oid in observations, f'missing registered observation {oid}')
        obs, record = observations[oid], truths[oid]
        require(set(obs) == {'observation_id', 'kind', 'radius', 'observation'}, f'{oid}: invalid observation schema')
        require(obs['kind'] == kind and obs['radius'] == str(radius), f'{oid}: wrong template/radius label')
        require(set(record) == {'observation_id', 'truth', 'seed', 'actual_offsets', 'noise_linf'},
                f'{oid}: invalid truth schema')
        require(exact(record['truth']) == truth and type(record['seed']) is int and record['seed'] == seed,
                f'{oid}: wrong registered truth/seed')
        require(isinstance(obs['observation'], list) and len(obs['observation']) == CHANNELS,
                f'{oid}: wrong observation channel count')
        observation = tuple(exact(v, oid) for v in obs['observation'])
        require(all(-EPSILON <= v <= 1 + EPSILON for v in observation), f'{oid}: observation outside radiance bounds')
        require(isinstance(record['actual_offsets'], list) and len(record['actual_offsets']) == 3,
                f'{oid}: invalid actual offsets')
        offsets = tuple(exact(v) for v in record['actual_offsets'])
        require(offsets[1] == 0 and all(offsets[j] in {radius * v for v in WEIGHTS} for j in (0, 2)),
                f'{oid}: camera pose violates registered radius/gauge/grid')
        require(0 <= exact(record['noise_linf']) <= EPSILON, f'{oid}: invalid noise_linf')
        require(set(outers[oid]) == {'observation_id', 'outer'}, f'{oid}: invalid outer record schema')
        verified_outers[oid] = verify_outer(outers[oid]['outer'], observation, truth, oid)
        for j, off in enumerate(OFFSETS):
            tid = f'{oid}-s{j}'
            expected_task_ids.add(tid)
            require(tid in tasks and set(tasks[tid]) == {'task_id', 'observation_id', 'incumbent'},
                    f'missing/invalid task {tid}')
            require(tasks[tid]['observation_id'] == oid and exact(tasks[tid]['incumbent']) == truth + off,
                    f'{tid}: incorrect association/initial offset')
    require(set(tasks) == expected_task_ids, 'tasks: extra/missing task IDs')
    expected_keys = {(tid, method) for tid in tasks for method in METHODS}
    require(set(decisions) == set(rows) == expected_keys, 'decisions/ROWS: missing/extra task-method pairs')
    derived_rows = []
    for tid, method in sorted(expected_keys):
        key = (tid, method)
        task, decision = tasks[tid], decisions[key]
        oid = task['observation_id']
        truth, a = exact(truths[oid]['truth']), exact(task['incumbent'])
        b = verify_decision(decision, a, verified_outers[oid]['hull'], f'{tid}/{method}')
        expected = expected_row(task, observations[oid], truth, decision, b)
        verify_row(rows[key], expected, f'ROWS {tid}/{method}')
        if method in CERTIFICATES:
            actual_gain = (a - truth) ** 2 - (b - truth) ** 2
            require(actual_gain >= 0, f'{tid}/{method}: observed certificate harm')
            require(actual_gain >= exact(decision['gain_lower']), f'{tid}/{method}: actual gain below certificate')
            if a == truth:
                require(decision['status'] == 'KEEP' and b == a, f'{tid}/{method}: correct initial point not preserved')
            if observations[oid]['kind'] == 'layered_hidden' and DOMAIN[0] <= a <= DOMAIN[1]:
                require(decision['status'] == 'KEEP', f'{tid}/{method}: hidden inside-prior certificate must KEEP')
        derived_rows.append(expected)
    for tid, task in tasks.items():
        verify_paired(tid, decisions, exact(truths[task['observation_id']]['truth']))
    summary = bundle['summary']
    require(isinstance(summary, dict), 'SUMMARY must be object')
    top_expected = {'observation_configurations': 135, 'template_count': 5, 'task_count': 1215, 'decision_count': 6075,
                    'evaluated_boxes': sum(v['evaluated'] for v in verified_outers.values()),
                    'unresolved_budget_leaves': sum(v['budget_leaves'] for v in verified_outers.values()), 'incompatible': 0}
    require(set(summary) == set(top_expected) | {'rows', 'elapsed_seconds', 'information_strata',
                                               'hybrid_branch_counts', 'paired_relations'},
            'SUMMARY: incorrect top-level schema')
    for field, value in top_expected.items():
        require(type(summary[field]) is int and summary[field] == value, f'SUMMARY: incorrect {field}')
    require(finite_number(summary['elapsed_seconds'], 'SUMMARY.elapsed_seconds') >= 0, 'SUMMARY: negative time')
    aggregates = verify_summary_rows(summary['rows'], derived_rows)
    information_groups, paired_groups = verify_extended_summary(summary, derived_rows, decisions)
    if models is None:
        models = _old.shared_models()
    sampled = sum(model_spot_checks(models[(obs['kind'], exact(obs['radius']))], obs, truths[oid],
                                   verified_outers[oid], oid, witness_samples) for oid, obs in sorted(observations.items()))
    totals = {}
    for method in METHODS:
        selected = [row for row in derived_rows if row['method'] == method]
        errors, initial = [exact(row['ae_mm']) for row in selected], [exact(row['initial_ae_mm']) for row in selected]
        totals[method] = {'n': len(selected), 'improved': sum(e < a for e, a in zip(errors, initial)),
                          'harm': sum(e > a for e, a in zip(errors, initial)), 'same': sum(e == a for e, a in zip(errors, initial)),
                          'moved': sum(row['status'] == 'MOVE' for row in selected),
                          'mae_mm_exact': str(sum(errors, Q(0)) / len(selected)),
                          'mse_mm2_exact': str(sum((e * e for e in errors), Q(0)) / len(selected)),
                          'max_error_mm_exact': str(max(errors))}
    return {'counts': top_expected, 'summary_groups_checked': len(aggregates), 'method_totals': totals,
            'information_summary_groups_checked': information_groups, 'paired_groups_checked': paired_groups,
            'paired_checks': {'tasks_checked': 1215, 'hybrid_error_le_midpoint': True,
                              'accepted_midpoint_outputs_preserved': True, 'hybrid_MOVE_set_equals_projection': True},
            'hybrid_branches': {branch: sum(d.get('branch') == branch for d in decisions.values())
                                for branch in ('midpoint', 'projection', 'keep', 'incompatible')},
            'model_checks': {'truth_images_recomputed': 135, 'recorded_exclusion_witnesses_checked':
                             sum(len(v['excluded']) for v in verified_outers.values()),
                             'shared_enclosure_witnesses_recomputed': sampled,
                             'witnesses_per_observation_cap': witness_samples}}


def load_bundle(output):
    bundle, hashes = {}, {}
    for name, relative in REQUIRED.items():
        raw = (output / relative).read_bytes()
        hashes[relative] = hashlib.sha256(raw).hexdigest()
        bundle[name] = json.loads(raw, object_pairs_hook=unique_object,
                                  parse_constant=lambda value: (_ for _ in ()).throw(VerificationError(f'nonfinite JSON {value}')))
    return bundle, hashes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path, help='Existing hybrid output directory')
    parser.add_argument('--witnesses-per-observation', type=int, default=2, help='Shared-model witness sample cap (1..8)')
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if not output.is_dir():
        parser.error(f'existing output directory required: {output}')
    if socket.gethostname() != 'liekkas' or not output.is_relative_to(ROOT / 'outputs'):
        parser.error(f'exact liekkas hybrid experiment output required: {output}')
    report_path = output / 'verification' / 'VERIFY.json'
    if report_path.exists():
        parser.error(f'refusing to overwrite existing verification: {report_path}')
    started = time.perf_counter()
    report = {'schema': 'independent-hybrid-verification-v1', 'status': 'FAIL', 'output_directory': str(output),
              'host': socket.gethostname(), 'verifier_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'arithmetic_helper': str(_spec.origin), 'model_source_root': str(DEPENDENCY_ROOT),
              'independent_arithmetic': True,
              'limitations': ['Image predictions and sampled enclosures reuse the disclosed geometry kernels; their enclosure/physical contracts are not independently reproved.',
                              'Every recorded exclusion witness is checked for noise conflict; a bounded sample is recomputed from the shared enclosure kernel.',
                              'Residual output grid membership, tie consistency and geometric outcomes are checked; finite residual grid optimality is not rerun.',
                              'Stored runtime is checked for finite nonnegative values, not independently benchmarked.']}
    try:
        bundle, hashes = load_bundle(output)
        report['input_sha256'] = hashes
        current = source_hashes()
        report['source_files_checked'] = verify_source_lock(bundle['source_lock'], current)
        verify_protocol(bundle['protocol'])
        status = bundle['run_status']
        require(isinstance(status, dict) and status.get('status') == 'COMPLETE' and
                status.get('observations_complete') == 135 and status.get('decisions_complete') == 6075,
                'RUN_STATUS: incomplete/incorrect registered population')
        require(finite_number(status['elapsed_seconds'], 'RUN_STATUS.elapsed_seconds') >= 0,
                'RUN_STATUS: negative elapsed time')
        report.update(verify_bundle(bundle, witness_samples=args.witnesses_per_observation))
        csv_path = output / 'evaluation' / 'ROWS.csv'
        report['input_sha256']['evaluation/ROWS.csv'] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        with csv_path.open(encoding='utf-8', newline='') as stream:
            reader = csv.DictReader(stream)
            report['csv_rows_checked'] = verify_csv_rows(reader.fieldnames, list(reader), bundle['rows'])
        require(source_hashes() == current, 'scientific sources changed during verification')
        report['source_lock_unchanged'] = True
        report['status'] = 'PASS'
        report['checks'] = ['complete 135/1215/6075 new registered population and unique associations',
                            'exact actions, gains, hybrid branches and all geometric evaluation rows',
                            'all trace trees, bisections, terminal-domain partitions and recorded witness conflicts',
                            'truth covered by retained leaves for every observation',
                            'every seeded new pose/noise image and sampled shared-model exclusion intervals',
                            'six aggregation families and CSV equality to JSON',
                            'per-task C1 relations, correct certificate KEEP and hidden inside-prior KEEP',
                            'absolute source inventory equals both frozen trees and remains unchanged']
    except (VerificationError, OSError, KeyError, TypeError, ValueError, ImportError, ArithmeticError) as error:
        report['error'] = f'{type(error).__name__}: {error}'
    report['verification_seconds'] = time.perf_counter() - started
    report_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with report_path.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
    except FileExistsError:
        parser.error(f'refusing to overwrite concurrent verification: {report_path}')
    print(json.dumps({'status': report['status'], 'report': str(report_path), 'error': report.get('error')}, ensure_ascii=False))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())

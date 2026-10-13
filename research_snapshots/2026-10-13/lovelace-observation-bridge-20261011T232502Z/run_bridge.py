"""Fixed observation/propagation bridge with validated historical reuse."""
import os
import sys
sys.dont_write_bytecode = True
for _thread_setting in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_thread_setting] = '1'

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from fractions import Fraction as Q
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import socket
import statistics
import time

ROOT = Path(__file__).resolve().parent
EXPECTED_ROOT = Path('/srv/slam-research/grf/map-denoise/runs/lovelace-observation-bridge-20261011T232502Z')
BASE = ROOT.parent / 'lovelace-shared-appearance-20261011T052300Z/outputs/confirmation-v1'
ABLATION = ROOT.parent / 'lovelace-shared-ablation-20261011T055200Z/outputs/ablation-v1'
BIAS = ROOT.parent / 'lovelace-view-bias-20261011T092300Z/outputs/view-bias-v1'
_spec = importlib.util.spec_from_file_location('frozen_bridge_helpers', ABLATION.parent.parent / 'run_ablation.py')
helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(helpers)
old = helpers.old
decode, encode, load = helpers.decode, helpers.encode, helpers.load
solve_arm, clear_geometry_cache = helpers.solve_arm, helpers.clear_geometry_cache
sys.path.insert(0, str(ROOT))

METHODS = ('keep', 'shared')
EPSILON, DELTA, BETA = Q(1, 100), Q(1, 50), Q(1, 100)
MAX_BOXES, TOLERANCE, TIMEOUT, IMAGE_SIZE = 511, Q(15, 32), 5400, 432
UNITS = (
    ('R0epsilon', 'raw', EPSILON, 8, True),
    ('C0epsilon', 'clip', EPSILON, 8, False),
    ('R0delta', 'raw', DELTA, 8, False),
    ('C0delta', 'clip', DELTA, 8, False),
    ('RBdelta', 'biased_raw', DELTA, 8, False),
    ('CBdelta', 'biased_clip', DELTA, 8, True),
    ('CBdelta4', 'biased_clip', DELTA, 4, False),
    ('CBdelta16', 'biased_clip', DELTA, 16, False),
)
UNIT = {name: {'image_kind': kind, 'bound': str(bound), 'rounds': rounds, 'reused': reused}
        for name, kind, bound, rounds, reused in UNITS}
REUSE_SOURCES = {'R0epsilon': (ABLATION, 'global_gray511'), 'CBdelta': (BIAS, 'global_shared_wide')}
CONTRASTS = (('R0epsilon', 'C0epsilon'), ('R0epsilon', 'R0delta'), ('C0epsilon', 'C0delta'),
             ('R0delta', 'RBdelta'), ('C0delta', 'CBdelta'), ('RBdelta', 'CBdelta'),
             ('R0delta', 'C0delta'), ('CBdelta', 'CBdelta4'), ('CBdelta', 'CBdelta16'))
PROGRESS = {'decisions.json', 'TIMING_PARTIAL.json', 'RUN_STATUS.json'}
META = ('observation_id', 'base_observation_id', 'unit_id', 'bound', 'rounds', 'reused')


def exact(value):
    if isinstance(value, (float, bool)):
        raise TypeError('exact rational values required')
    return decode(value) if isinstance(value, dict) else Q(value)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def check_target():
    if socket.gethostname() != 'liekkas' or ROOT != EXPECTED_ROOT or ROOT.resolve() != EXPECTED_ROOT:
        raise RuntimeError('requires exact local liekkas observation-bridge root')


def validate_output(output):
    check_target()
    path = Path(output).resolve()
    if path == ROOT or not path.is_relative_to(ROOT):
        raise ValueError('output must be below exact new root')
    if path.exists():
        raise FileExistsError('refuse overwriting existing output')
    return path


def save(path, value, *, replace=False):
    path = Path(path)
    if path.resolve() == ROOT or not path.resolve().is_relative_to(ROOT):
        raise ValueError('write outside exact new root')
    if replace and path.name not in PROGRESS:
        raise ValueError('only owned progress files may be replaced')
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    with old.finish_record_before_alarm():
        path.parent.mkdir(parents=True, exist_ok=True)
        destination = path.with_name(path.name+'.pending') if replace else path
        with destination.open('x', encoding='utf-8') as stream:
            stream.write(text)
        if replace:
            destination.replace(path)


def csv_save(path, rows, json_columns=()):
    path = Path(path)
    if not path.resolve().is_relative_to(ROOT):
        raise ValueError('CSV outside exact new root')
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if key in json_columns and isinstance(value, (dict, list)) else value
                          for key, value in row.items()} for row in rows)


def transform_image(observation, image_kind):
    if image_kind not in ('raw', 'clip', 'biased_raw', 'biased_clip'):
        raise ValueError('unregistered observation transformation')
    image = tuple(map(exact, observation))
    if len(image) != IMAGE_SIZE:
        raise ValueError('requires three views, 48 RGB pixels per view')
    bias = (-BETA, Q(0), BETA)
    result = []
    for index, value in enumerate(image):
        if image_kind.startswith('biased_'):
            value += bias[index // 144]
        if image_kind in ('clip', 'biased_clip'):
            value = min(Q(1), max(Q(0), value))
        result.append(encode(value))
    return result


def generate_public(base_observations, base_tasks):
    ids = {record['observation_id'] for record in base_observations}
    if len(ids) != len(base_observations):
        raise ValueError('duplicate base observations')
    counts = Counter(task['observation_id'] for task in base_tasks)
    if set(counts) != ids or set(counts.values()) != {9}:
        raise ValueError('every base observation requires nine initial states')
    if len({task['task_id'] for task in base_tasks}) != len(base_tasks):
        raise ValueError('duplicate base tasks')
    observations, tasks = [], []
    for unit_id, kind, bound, rounds, reused in UNITS:
        for item in base_observations:
            radius = exact(item['radius'])
            if radius < 0:
                raise ValueError('negative public radius')
            observations.append({'observation_id': f'{unit_id}-{item["observation_id"]}',
                'base_observation_id': item['observation_id'], 'unit_id': unit_id,
                'bound': str(bound), 'rounds': rounds, 'reused': reused, 'radius': str(radius),
                'observation': transform_image(item['observation'], kind)})
        for task in base_tasks:
            tasks.append({'task_id': f'{unit_id}-{task["task_id"]}',
                'observation_id': f'{unit_id}-{task["observation_id"]}', 'base_task_id': task['task_id'],
                'incumbent': encode(exact(task['incumbent']))})
    return observations, tasks


def execute_shared(observation, radius, epsilon, rounds, max_boxes, tolerance):
    """Public scientific interface; raw observations may be outside [0,1]."""
    image, radius, epsilon, tolerance = tuple(map(exact, observation)), exact(radius), exact(epsilon), exact(tolerance)
    if len(image) != IMAGE_SIZE or radius < 0:
        raise ValueError('wrong image layout or radius')
    if epsilon not in (EPSILON, DELTA) or type(rounds) is not int or rounds not in (4, 8, 16):
        raise ValueError('unregistered epsilon or propagation rounds')
    if type(max_boxes) is not int or max_boxes != MAX_BOXES or tolerance != TOLERANCE:
        raise ValueError('registered full tree budget and tolerance required')
    clear_geometry_cache()
    return solve_arm(image, radius, 'global_gray', epsilon=epsilon, rounds=rounds,
                     max_boxes=max_boxes, tolerance=tolerance)


def input_paths():
    base = [BASE / relative for relative in helpers.INPUT_RELATIVE]
    common = ('INPUTS.json', 'SOURCE_LOCK.json', 'PROTOCOL.json', 'RUN_STATUS.json',
              'outers.json', 'decisions.json', 'TIMING.json', 'verification/VERIFY.json')
    return base + [ABLATION / rel for rel in common] + [BIAS / rel for rel in common + ('observations.json', 'tasks.json')]


def input_lock():
    return {str(path.resolve()): sha(path) for path in input_paths()}


def science_lock():
    files = {}
    for source in (ABLATION, BIAS):
        for path, digest in load(source / 'SOURCE_LOCK.json')['files'].items():
            if path in files and files[path] != digest:
                raise RuntimeError('historical source lineages disagree: '+path)
            if sha(path) != digest:
                raise RuntimeError('historical source changed: '+path)
            files[path] = digest
    for path in [*sorted(ROOT.glob('*.py')), ROOT / 'PROTOCOL.md']:
        files[str(path.resolve())] = sha(path)
    return files


def configuration():
    return {'units': UNIT, 'methods': list(METHODS), 'base_noise_epsilon': str(EPSILON),
        'bias_magnitude_generation_only': str(BETA), 'bias_pattern_generation_only': ['-1/100', '0', '1/100'],
        'layout': {'views': 3, 'pixels_per_view': 48, 'channels': 3},
        'mode': 'global_gray', 'allowed_branches': list(old.BRANCHES),
        'max_boxes_per_branch': MAX_BOXES, 'tolerance_mm': str(TOLERANCE),
        'expected_observations': 432, 'expected_tasks': 3888, 'expected_decisions': 7776,
        'expected_new_shared_observations': 324, 'expected_reused_shared_observations': 108,
        'expected_new_shared_actions': 2916, 'expected_reused_shared_actions': 972, 'expected_keep_actions': 3888,
        'expected_per_unit_method': {'observations': 54, 'correct': 54, 'small': 432, 'negative_small': 72},
        'contrasts': [list(pair) for pair in CONTRASTS], 'cpu_threads': 1, 'gpu_count': 0,
        'wall_time_limit_seconds_main_plus_verification': TIMEOUT,
        'source_root': str(BASE), 'reused_sources': {unit: {'output_root': str(root), 'method': method}
            for unit, (root, method) in REUSE_SOURCES.items()},
        'cache_policy': 'clear all geometry/model caches once before each new shared image solve',
        'sampling_note': 'same 54 base synthetic observations, related initial states and transformations; no new independent scenes'}


def validate_reused_entry(entry, bound, rounds):
    if set(entry['branches']) != set(old.BRANCHES):
        raise RuntimeError('reuse has wrong scene branches')
    for outer in [entry['outer'], *entry['branches'].values()]:
        if (exact(outer['epsilon']) != bound or exact(outer['tolerance']) != TOLERANCE
                or outer['mode'] != 'global_gray' or outer['rounds'] != rounds or outer['budget'] != MAX_BOXES):
            raise RuntimeError('reuse has mismatched scientific parameters')
    if entry['outer']['max_boxes_per_branch'] != MAX_BOXES:
        raise RuntimeError('reuse is not the complete 511-box arm')
    for outer in entry['branches'].values():
        if outer['max_boxes'] != MAX_BOXES or outer['propagation_rounds'] != rounds:
            raise RuntimeError('reuse branch budget or propagation mismatch')


def _unique(records, key):
    mapping = {record[key]: record for record in records}
    if len(mapping) != len(records):
        raise RuntimeError('duplicate historical '+key)
    return mapping


def prepare_reuse(observations, tasks, base_observations, base_tasks, inputs):
    """Validate source graph/public data/parameters/actions, without truth parsing."""
    base_obs = _unique(base_observations, 'observation_id')
    base_task = _unique(base_tasks, 'task_id')
    new_tasks = defaultdict(list)
    for task in tasks:
        new_tasks[task['observation_id']].append(task)
    result, provenance = {}, []
    if load(BASE / 'verification/VERIFY.json')['status'] != 'PASS':
        raise RuntimeError('base experiment is not verified')
    for unit_id, (source, method) in REUSE_SOURCES.items():
        verify, status, protocol = (load(source / name) for name in
                                   ('verification/VERIFY.json', 'RUN_STATUS.json', 'PROTOCOL.json'))
        if verify['status'] != 'PASS' or verify['host'] != 'liekkas' or verify['output'] != str(source) or status['status'] != 'COMPLETE':
            raise RuntimeError('historical reuse is not verified at exact target')
        historical_inputs = load(source / 'INPUTS.json')
        if historical_inputs['source_root'] != str(BASE):
            raise RuntimeError('reuse source observation lineage differs')
        for path, digest in historical_inputs['files'].items():
            if path not in inputs or inputs[path] != digest:
                raise RuntimeError('historical input lineage mismatch: '+path)
        for rel in helpers.INPUT_RELATIVE:
            path = str(BASE / rel)
            if historical_inputs['files'].get(path) != inputs[path]:
                raise RuntimeError('base input omitted from reuse lineage: '+rel)
        if (protocol['max_boxes_per_branch'] != MAX_BOXES or exact(protocol['tolerance_mm']) != TOLERANCE
                or tuple(protocol['allowed_branches']) != tuple(old.BRANCHES)):
            raise RuntimeError('historical protocol mismatch')
        mconfig = protocol['method_configs'][method]
        if mconfig['mode'] != 'global_gray' or mconfig['max_boxes'] != MAX_BOXES or mconfig['rounds'] != 8:
            raise RuntimeError('wrong reuse method configuration')
        if unit_id == 'R0epsilon':
            if exact(protocol['epsilon']) != EPSILON:
                raise RuntimeError('wrong original residual bound')
            historical_observations, historical_tasks = base_obs, base_task
        else:
            if exact(mconfig['epsilon_by_beta']['1/100']) != DELTA:
                raise RuntimeError('wrong biased residual bound')
            historical_observations = _unique(load(source / 'observations.json'), 'observation_id')
            historical_tasks = _unique(load(source / 'tasks.json'), 'task_id')
        source_outers = {key: record['arms'][method]
                         for key, record in _unique(load(source / 'outers.json'), 'observation_id').items()}
        source_actions = [record for record in load(source / 'decisions.json') if record['method'] == method]
        actions = _unique(source_actions, 'task_id')
        timings = _unique(load(source / 'TIMING.json')['observations'], 'observation_id')
        for item in [record for record in observations if record['unit_id'] == unit_id]:
            source_oid = item['base_observation_id'] if unit_id == 'R0epsilon' else 'b1-'+item['base_observation_id']
            original = historical_observations[source_oid]
            if (original['observation'] != item['observation'] or exact(original['radius']) != exact(item['radius'])):
                raise RuntimeError('reuse image/radius differs: '+source_oid)
            if unit_id == 'CBdelta' and (original['base_observation_id'] != item['base_observation_id'] or original['beta'] != '1/100'):
                raise RuntimeError('wrong bias source identity')
            entry = source_outers[source_oid]
            validate_reused_entry(entry, Q(item['bound']), item['rounds'])
            time_row = timings[source_oid]
            if time_row['outer_seconds_by_method'][method] != entry['outer_seconds']:
                raise RuntimeError('historical timing does not match reused outer')
            selected, mappings = {}, []
            for task in new_tasks[item['observation_id']]:
                source_tid = task['base_task_id'] if unit_id == 'R0epsilon' else 'b1-'+task['base_task_id']
                historical_task = historical_tasks[source_tid]
                if historical_task['observation_id'] != source_oid or historical_task['incumbent'] != task['incumbent']:
                    raise RuntimeError('reuse initial state differs: '+source_tid)
                if unit_id == 'CBdelta' and historical_task['base_task_id'] != task['base_task_id']:
                    raise RuntimeError('reuse task lineage differs')
                original_action = actions[source_tid]
                action = {key: value for key, value in original_action.items() if key not in ('task_id', 'method')}
                if action != old.select_action(entry['outer'], decode(task['incumbent'])):
                    raise RuntimeError('historical action cannot be replayed from its outer')
                selected[task['task_id']] = action
                mappings.append({'task_id': task['task_id'], 'source_task_id': source_tid,
                                 'action_sha256': json_sha(original_action)})
            source_reference = {'output_root': str(source), 'observation_id': source_oid, 'method': method}
            result[item['observation_id']] = {'entry': entry, 'actions': selected, 'source': source_reference}
            provenance.append({**{key: item[key] for key in META}, 'source': source_reference,
                'observation_sha256': json_sha(original), 'outer_sha256': json_sha(entry),
                'source_selector_seconds': time_row['selector_seconds_by_method'][method], 'tasks': mappings})
    return result, provenance


def private_truths(observations):
    base = _unique(load(BASE / 'evaluation/truth.json'), 'observation_id')
    if len(base) != 54:
        raise RuntimeError('wrong private population')
    return [{**base[item['base_observation_id']], **{key: item[key] for key in META}}
            for item in observations]


def aggregate(rows):
    return [{**record, 'unit_id': unit_id, **{key: UNIT[unit_id][key] for key in ('bound', 'rounds')},
             'reused': UNIT[unit_id]['reused'] and record['method'] == 'shared'}
            for unit_id in UNIT for record in helpers.aggregate([row for row in rows if row['unit_id'] == unit_id])]


def group_mean_sd(aggregates):
    result = []
    for unit_id in UNIT:
        for method in METHODS:
            for state in ('all', 'correct', 'small'):
                for grouping in ('by_case', 'by_seed'):
                    part = [row for row in aggregates if row['unit_id'] == unit_id and row['method'] == method
                            and row['state_group'] == state and row['grouping'] == grouping]
                    result.append({'unit_id': unit_id, 'method': method, 'state_group': state,
                        'grouping': grouping, 'groups': len(part), 'ddof': 1,
                        'metrics': {metric: {'mean': statistics.mean(row[metric] for row in part),
                            'sd': statistics.stdev(row[metric] for row in part)}
                            for metric in ('mae_mm', 'mse_mm2', 'improved', 'harm')}})
    return result


def make_pairs(rows):
    indexed = {(row['unit_id'], row['base_task_id'], row['method']): row for row in rows}
    if len(indexed) != len(rows):
        raise RuntimeError('duplicate paired rows')
    pairs, interactions = [], []
    comparisons = [(unit, 'keep', unit, 'shared') for unit in UNIT]
    comparisons += [(left, 'shared', right, 'shared') for left, right in CONTRASTS]
    for base_task_id in sorted({row['base_task_id'] for row in rows}):
        for left_unit, left_method, right_unit, right_method in comparisons:
            left = indexed[left_unit, base_task_id, left_method]
            right = indexed[right_unit, base_task_id, right_method]
            a, b = Q(left['ae_mm']), Q(right['ae_mm'])
            pairs.append({'base_task_id': base_task_id, 'case': right['case'], 'seed': right['seed'],
                'state_group': right['state_group'], 'left_unit': left_unit, 'left_method': left_method,
                'right_unit': right_unit, 'right_method': right_method, 'left_ae_mm': str(a), 'right_ae_mm': str(b),
                'mae_gain_mm': str(a-b), 'mse_gain_mm2': str(a*a-b*b), 'better': b < a, 'same': b == a, 'worse': b > a})
        e = {unit: Q(indexed[unit, base_task_id, 'shared']['ae_mm'])
             for unit in ('R0delta', 'C0delta', 'RBdelta', 'CBdelta')}
        no_bias, with_bias = e['C0delta']-e['R0delta'], e['CBdelta']-e['RBdelta']
        interactions.append({'base_task_id': base_task_id, 'case': right['case'], 'seed': right['seed'],
            'state_group': right['state_group'], 'clipping_error_change_no_bias_mm': str(no_bias),
            'clipping_error_change_with_bias_mm': str(with_bias), 'difference_in_error_changes_mm': str(with_bias-no_bias),
            'formula': '(AE_CBdelta-AE_RBdelta)-(AE_C0delta-AE_R0delta)'})
    return pairs, interactions


def evaluate(output, observations, tasks, truths, decisions, outers, timing, started):
    tm, pm = _unique(tasks, 'task_id'), _unique(truths, 'observation_id')
    om = _unique(outers, 'observation_id')
    rows, coverage = [], []
    for decision in decisions:
        task = tm[decision['task_id']]
        private = pm[task['observation_id']]
        outer = om[task['observation_id']]['arms'][decision['method']]['outer']
        truth, incumbent, selected = decode(private['truth']), decode(task['incumbent']), decode(decision['output'])
        initial, error = abs(incumbent-truth), abs(selected-truth)
        rows.append({'task_id': task['task_id'], 'base_task_id': task['base_task_id'],
            **{key: private[key] for key in META}, 'reused': decision['reused'],
            'case': private['case'], 'seed': private['seed'], 'method': decision['method'],
            'state_group': 'correct' if initial == 0 else 'small',
            'prior_position': 'inside_prior' if 540 <= incumbent <= 660 else 'outside_prior',
            'incumbent': str(incumbent), 'output': str(selected), 'status': decision['status'],
            'initial_ae_mm': str(initial), 'ae_mm': str(error), 'se_mm2': str(error*error),
            'actual_gain_mm2': str(initial*initial-error*error), 'covered': helpers.covered(outer, truth),
            'hull_width_mm': None if outer['hull'] is None else str(decode(outer['hull'][1])-decode(outer['hull'][0])),
            'hull_lo_mm': None if outer['hull'] is None else str(decode(outer['hull'][0])),
            'hull_hi_mm': None if outer['hull'] is None else str(decode(outer['hull'][1])),
            'retained_branch_count': len(outer['retained_branches']), 'retained_interval_count': len(outer['retained_intervals']),
            'evaluated_boxes': outer['evaluated_boxes'], 'unresolved_budget_leaves': outer['unresolved_budget_leaves'],
            'auxiliary_samples_used': 0})
    for record in outers:
        private, truth = pm[record['observation_id']], decode(pm[record['observation_id']]['truth'])
        for method, entry in record['arms'].items():
            outer = entry['outer']
            coverage.append({**{key: record[key] for key in META}, 'reused': record['reused'] and method == 'shared',
                'case': private['case'], 'seed': private['seed'], 'method': method,
                'covered': helpers.covered(outer, truth), 'incompatible': outer['hull'] is None,
                'hull': outer['hull'], 'retained_intervals': outer['retained_intervals'],
                'hull_width_mm': None if outer['hull'] is None else str(decode(outer['hull'][1])-decode(outer['hull'][0])),
                'retained_branches': outer['retained_branches'],
                'branch_coverage': {branch: any(decode(node['lo']) <= truth <= decode(node['hi'])
                    for node in tree['trace'] if node['id'] in tree['retained_ids'])
                    for branch, tree in entry['branches'].items()}})
    if len(rows) != 7776 or len(coverage) != 864:
        raise RuntimeError('wrong evaluation population')
    for unit in UNIT:
        for method in METHODS:
            part = [row for row in rows if row['unit_id'] == unit and row['method'] == method]
            if Counter(row['state_group'] for row in part) != {'correct': 54, 'small': 432}:
                raise RuntimeError('wrong correct/offset denominator')
            if sum(row['case'] == 'equal_full_negative' and row['state_group'] == 'small' for row in part) != 72:
                raise RuntimeError('wrong negative-control denominator')
    save(output / 'evaluation/truth.json', truths)
    save(output / 'evaluation/ROWS.json', rows)
    csv_save(output / 'evaluation/ROWS.csv', rows)
    save(output / 'evaluation/COVERAGE.json', coverage)
    csv_save(output / 'evaluation/COVERAGE.csv', coverage, ('hull', 'retained_intervals', 'retained_branches', 'branch_coverage'))
    pairs, interactions = make_pairs(rows)
    save(output / 'evaluation/PAIRED.json', pairs)
    csv_save(output / 'evaluation/PAIRED.csv', pairs)
    save(output / 'evaluation/INTERACTION.json', interactions)
    csv_save(output / 'evaluation/INTERACTION.csv', interactions)
    costs = []
    for unit in UNIT:
        times = [row for row in timing if row['unit_id'] == unit]
        for method in METHODS:
            new = sum(row['new_solver_seconds_by_method'][method] for row in times)
            action = sum(row['selector_seconds_by_method'][method] for row in times)
            costs.append({'unit_id': unit, 'method': method, 'reused': UNIT[unit]['reused'] and method == 'shared',
                'new_solver_seconds': new, 'selector_seconds': action, 'new_total_seconds': new+action,
                'historical_outer_seconds': sum(row['historical_outer_seconds_by_method'][method] for row in times),
                'evaluated_boxes': sum(record['arms'][method]['outer']['evaluated_boxes'] for record in outers if record['unit_id'] == unit),
                'unresolved_budget_leaves': sum(record['arms'][method]['outer']['unresolved_budget_leaves'] for record in outers if record['unit_id'] == unit)})
    aggregates = aggregate(rows)
    summary = {'observation_configurations': len(observations), 'task_count': len(tasks), 'decision_count': len(decisions),
        'new_shared_actions': sum(row['method'] == 'shared' and not row['reused'] for row in rows),
        'reused_shared_actions': sum(row['method'] == 'shared' and row['reused'] for row in rows),
        'keep_actions': sum(row['method'] == 'keep' for row in rows), 'rows': aggregates,
        'group_mean_sd': group_mean_sd(aggregates), 'per_unit_method_timing': costs,
        'elapsed_seconds': time.perf_counter()-started, 'sampling_note': configuration()['sampling_note'],
        'timing_scope': 'new cold solver/selector cost separate from recorded historical outer time; reuse validation is setup overhead'}
    save(output / 'evaluation/SUMMARY.json', summary)
    csv_save(output / 'evaluation/SUMMARY.csv', aggregates, ('condition',))
    csv_save(output / 'evaluation/TIMING.csv', costs)
    return summary


def _run(output, started, budget, state):
    inputs, sources = input_lock(), science_lock()
    base_observations, base_tasks = load(BASE / 'observations.json'), load(BASE / 'tasks.json')
    if (len(base_observations), len(base_tasks)) != (54, 486):
        raise RuntimeError('wrong base population')
    observations, tasks = generate_public(base_observations, base_tasks)
    reuse, provenance = prepare_reuse(observations, tasks, base_observations, base_tasks, inputs)
    if len(observations) != 432 or len(tasks) != 3888 or len(reuse) != 108:
        raise RuntimeError('wrong bridge population')
    output.mkdir(parents=True, exist_ok=False)
    state['owns_output'] = True
    save(output / 'INPUTS.json', {'source_root': str(BASE), 'files': inputs})
    save(output / 'SOURCE_LOCK.json', {'source_root': str(ROOT), 'files': sources,
        'lock_scope': 'both frozen historical scientific source graphs plus new root scientific sources and protocol',
        'utc': datetime.now(timezone.utc).isoformat()})
    save(output / 'BUDGET.json', budget)
    save(output / 'PROTOCOL.json', configuration())
    save(output / 'observations.json', observations)
    save(output / 'tasks.json', tasks)
    save(output / 'REUSE.json', provenance)
    grouped = defaultdict(list)
    for task in tasks:
        grouped[task['observation_id']].append(task)
    decisions, outers, timing = [], [], []
    save(output / 'RUN_STATUS.json', {'status': 'RUNNING', 'observations_complete': 0, 'decisions_complete': 0,
        'deadline_unix': budget['deadline_unix'], 'elapsed_seconds': time.perf_counter()-started}, replace=True)
    setup_seconds = time.perf_counter()-started
    for item in observations:
        oid = item['observation_id']
        keep = helpers.identity_arm()
        if item['reused']:
            shared = reuse[oid]['entry']
        else:
            shared = execute_shared(item['observation'], item['radius'], item['bound'], item['rounds'], MAX_BOXES, TOLERANCE)
        arms, action_seconds, completed = {'keep': keep, 'shared': shared}, {}, []
        for method in METHODS:
            tick = time.perf_counter()
            for task in grouped[oid]:
                if method == 'shared' and item['reused']:
                    action = reuse[oid]['actions'][task['task_id']]
                else:
                    action = old.select_action(arms[method]['outer'], decode(task['incumbent']), keep=method == 'keep')
                completed.append({'task_id': task['task_id'], 'method': method, 'unit_id': item['unit_id'],
                    'bound': item['bound'], 'rounds': item['rounds'], 'reused': item['reused'] and method == 'shared', **action})
            action_seconds[method] = time.perf_counter()-tick
        with old.finish_record_before_alarm():
            record = {**{key: item[key] for key in META}, 'arms': arms,
                      'reused_source': reuse[oid]['source'] if item['reused'] else None}
            save(output / f'checkpoint/{oid}.json', record)
            outers.append(record)
            decisions.extend(completed)
            timing.append({**{key: item[key] for key in META},
                'new_solver_seconds_by_method': {'keep': 0., 'shared': 0. if item['reused'] else shared['outer_seconds']},
                'historical_outer_seconds_by_method': {'keep': 0., 'shared': shared['outer_seconds'] if item['reused'] else 0.},
                'selector_seconds_by_method': action_seconds})
            save(output / 'decisions.json', decisions, replace=True)
            save(output / 'TIMING_PARTIAL.json', {'observations': timing, 'setup_seconds': setup_seconds}, replace=True)
            save(output / 'RUN_STATUS.json', {'status': 'RUNNING', 'observations_complete': len(outers),
                'decisions_complete': len(decisions), 'deadline_unix': budget['deadline_unix'],
                'elapsed_seconds': time.perf_counter()-started}, replace=True)
        print(json.dumps({'observation_id': oid, 'observations_complete': len(outers),
            'decisions_complete': len(decisions), 'elapsed_seconds': round(time.perf_counter()-started, 2)}), flush=True)
    save(output / 'outers.json', outers)
    if len(decisions) != 7776 or len({(row['task_id'], row['method']) for row in decisions}) != 7776 or load(output / 'decisions.json') != decisions:
        raise RuntimeError('all 7776 unique actions must be saved and read back before truth parsing')
    if Counter((row['method'], row['reused']) for row in decisions) != {('keep', False): 3888, ('shared', False): 2916, ('shared', True): 972}:
        raise RuntimeError('wrong new/reused action denominator')
    save(output / 'TIMING.json', {'observations': timing, 'setup_seconds': setup_seconds,
        'cache_policy': configuration()['cache_policy'], 'budget_deadline_unix': budget['deadline_unix']})
    truths = private_truths(observations)
    summary = evaluate(output, observations, tasks, truths, decisions, outers, timing, started)
    if input_lock() != inputs or science_lock() != sources:
        raise RuntimeError('historical inputs or frozen source changed')
    save(output / 'RUN_STATUS.json', {'status': 'COMPLETE', 'observations_complete': 432, 'decisions_complete': 7776,
        'new_shared_observations': 324, 'reused_shared_observations': 108,
        'deadline_unix': budget['deadline_unix'], 'elapsed_seconds': time.perf_counter()-started,
        'scientific_support': 'requires separate independent verification and paired claim analysis'}, replace=True)
    return summary


def deadline_reached(_signum, _frame):
    raise TimeoutError('shared main-plus-verification absolute deadline reached')


def run(output, deadline_unix=None):
    output = validate_output(output)
    wall = time.time()
    deadline = wall+TIMEOUT if deadline_unix is None else float(deadline_unix)
    if not wall < deadline <= wall+TIMEOUT:
        raise ValueError('deadline must be future and at most 5400 seconds away')
    if signal.getitimer(signal.ITIMER_REAL)[0]:
        raise RuntimeError('refuse replacing existing process deadline')
    budget = {'started_unix': wall, 'deadline_unix': deadline, 'limit_seconds': TIMEOUT,
              'scope': 'main plus independent verification; verification must not reset deadline'}
    previous = signal.signal(signal.SIGALRM, deadline_reached)
    state, started = {}, time.perf_counter()
    signal.setitimer(signal.ITIMER_REAL, deadline-wall)
    try:
        return _run(output, started, budget, state)
    except (Exception, KeyboardInterrupt) as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if state.get('owns_output'):
            progress = load(output / 'RUN_STATUS.json') if (output / 'RUN_STATUS.json').exists() else {}
            save(output / 'RUN_STATUS.json', {**progress, 'status': 'INCOMPLETE', 'reason': str(error),
                'deadline_unix': deadline, 'elapsed_seconds': time.perf_counter()-started}, replace=True)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def smoke():
    check_target()
    public, _ = old.generate_observation('smoke_only', 'partial', (Q(1, 4), Q(1, 2), Q(3, 4)),
        Q(600), 99124, radius=Q(1, 5), prefix='observation-bridge-smoke-v1')
    for name, kind, bound, rounds, reused in UNITS:
        if reused:
            continue
        image = transform_image(public['observation'], kind)
        result = execute_shared(image, public['radius'], bound, rounds, MAX_BOXES, TOLERANCE)
        if not helpers.covered(result['outer'], Q(600)):
            raise AssertionError('nonconfirmation smoke lost truth')
        for incumbent in (Q(600), Q(2401, 4), Q(597)):
            action = old.select_action(result['outer'], incumbent)
            if abs(decode(action['output'])-600) > abs(incumbent-600):
                raise AssertionError('nonconfirmation smoke harmful action')
        print(json.dumps({'smoke': name, 'rounds': rounds, 'bound': str(bound),
            'seconds': result['outer_seconds'], 'hull': result['outer']['hull'],
            'evaluated_boxes': result['outer']['evaluated_boxes']}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/bridge-v1')
    parser.add_argument('--deadline-unix', type=float)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(argv)
    if args.smoke:
        smoke()
    else:
        summary = run(args.output, args.deadline_unix)
        print(json.dumps({'stage': 'complete', 'elapsed_seconds': summary['elapsed_seconds'],
            'main': [row for row in summary['rows'] if row['grouping'] == 'overall' and row['state_group'] == 'small']}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Independent registration, random tape, evaluation and source-lock checks.

No runner, interval solver or action function is imported. Image generation and
enclosures use the disclosed shared ColoredModel kernel, not a second proof of
that kernel's geometry or continuous enclosure validity.
"""
import ast
from collections import Counter, defaultdict
import csv
from datetime import datetime
from fractions import Fraction as Q
import hashlib
import json
import math
from pathlib import Path
import random

from verification.arithmetic import DOMAIN, EPSILON, TOLERANCE, MAX_BOXES, exact, encoded, require

ROOT = Path('/srv/slam-research/grf/map-denoise/runs/lovelace-nuisance-20261011T044500Z')
HYBRID_ROOT = ROOT.parent / 'lovelace-hybrid-20261010T214217Z'
DEPENDENCY_ROOT = ROOT.parent / 'lovelace-challenge-20261010T205013Z'
METHODS = ('keep', 'frozen_nominal', 'expanded_unanchored', 'calibrated_isolated', 'calibrated_union')
BRANCHES = ('none', 'partial', 'full')
SOURCES = ('front-left', 'front-right', 'rear')
RADIUS = Q(1, 5)
WEIGHTS = (Q(-1), Q(-1, 2), Q(0), Q(1, 2), Q(1))
TRUTHS = (Q(6199, 11), Q(7853, 13), Q(8999, 14))
SEEDS = (51001, 51002, 51003)
OFFSETS = (Q(0), Q(-1, 4), Q(1, 4), Q(-3, 4), Q(3, 4), Q(-2), Q(2), Q(-3), Q(3))
CASES = (
    ('uniform_isolated', 'none', (Q(1, 5), Q(1, 5), Q(2, 5))),
    ('textured_isolated', 'none', (Q(1, 5), Q(3, 5), Q(2, 5))),
    ('textured_partial', 'partial', (Q(1, 5), Q(3, 5), Q(2, 5))),
    ('equal_partial', 'partial', (Q(1, 5), Q(1, 5), Q(1, 5))),
    ('weak_partial', 'partial', (Q(3, 10), Q(3, 10), Q(31, 100))),
    ('equal_full_negative', 'full', (Q(1, 5), Q(1, 5), Q(1, 5))),
)


def same(actual, expected, where):
    """JSON type-sensitive equality, including boolean versus integer counts."""
    require(json.dumps(actual, sort_keys=True, allow_nan=False) ==
            json.dumps(expected, sort_keys=True, allow_nan=False), f'{where}: differs from independent reconstruction')


def load(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def registration():
    return {
        'cases_evaluation_only': [{'case': case, 'rear_branch': branch, 'colors': list(map(str, colors))}
                                  for case, branch, colors in CASES],
        'truths_evaluation_only': list(map(str, TRUTHS)), 'seeds_evaluation_only': list(SEEDS),
        'offsets_evaluation_only': list(map(str, OFFSETS)), 'camera_radius_mm': str(RADIUS),
        'epsilon': str(EPSILON), 'tolerance_mm': str(TOLERANCE), 'max_boxes_per_branch': MAX_BOXES,
        'depth_domain_mm': list(map(str, DOMAIN)), 'expected_observations': 54,
        'expected_tasks': 486, 'expected_decisions': 2430, 'methods': list(METHODS),
        'allowed_occlusion_branches': list(BRANCHES), 'calibration_sources': list(SOURCES),
        'calibration_samples_per_observation': 3,
        'calibration_samples_used_per_method': {m: 3 if m.startswith('calibrated_') else 0 for m in METHODS},
        'random_tape_prefix': 'nuisance-v1',
        'random_draw_order': 'left pose; right pose; each image RGB channel; front-left, front-right, rear calibration noise',
        'random_weights': list(map(str, WEIGHTS)), 'cpu_threads': 1, 'gpu_count': 0,
        'wall_time_limit_seconds': 1800,
        'scope': 'shared front depth; two continuous piecewise-constant front grays and one rear gray; finite branch competition',
        'information_note': 'calibrated arms use the same three additional sourced material observations; other arms do not use their values',
    }


def regenerate(case, branch, colors, truth, seed, model_class):
    tape = f'nuisance-v1:{case}:{RADIUS}:{truth}:{seed}'
    rng = random.Random(tape)
    offsets = (RADIUS * rng.choice(WEIGHTS), Q(0), RADIUS * rng.choice(WEIGHTS))
    model = model_class(branch, camera_radius=RADIUS, colors=((Q(0), Q(1)),) * 3)
    prediction = model.point_prediction(truth, offsets, colors=colors)
    require(len(prediction) == 432, 'regeneration: changed image dimension')
    image_noise = tuple(EPSILON * rng.choice(WEIGHTS) for _ in range(432))
    observation = tuple(value + noise for value, noise in zip(prediction, image_noise))
    calibration_noise = tuple(EPSILON * rng.choice(WEIGHTS) for _ in SOURCES)
    public = {'radius': str(RADIUS), 'observation': list(map(encoded, observation)),
              'calibration': [{'source': source, 'value': encoded(color + noise)}
                              for source, color, noise in zip(SOURCES, colors, calibration_noise)]}
    private = {'case': case, 'rear_branch': branch, 'truth': encoded(truth),
               'colors': list(map(encoded, colors)), 'seed': seed, 'random_tape': tape,
               'actual_offsets': list(map(encoded, offsets)), 'image_noise': list(map(encoded, image_noise)),
               'calibration_noise': list(map(encoded, calibration_noise)),
               'noise_linf': encoded(max(map(abs, image_noise))),
               'calibration_noise_linf': encoded(max(map(abs, calibration_noise)))}
    if case == 'equal_full_negative':
        for depth in (DOMAIN[0], (DOMAIN[0] + DOMAIN[1]) / 2, DOMAIN[1]):
            require(model.point_prediction(depth, offsets, colors=colors) == prediction,
                    'equal-full: shared point kernel fails depth-independent negative control')
    return public, private


def arm_contracts(calibration):
    require(isinstance(calibration, list) and len(calibration) == 3 and
            [c.get('source') for c in calibration] == list(SOURCES) and
            all(set(c) == {'source', 'value'} for c in calibration), 'calibration: invalid source labels/schema')
    anchored = tuple((max(Q(0), exact(c['value']) - EPSILON),
                      min(Q(1), exact(c['value']) + EPSILON)) for c in calibration)
    require(all(low <= high for low, high in anchored), 'calibration: incompatible sample interval')
    return {'keep': ((), None),
            'frozen_nominal': (('none',), ((Q(0), Q(0)), (Q(0), Q(0)), (Q(0), Q(1)))),
            'expanded_unanchored': (BRANCHES, ((Q(0), Q(1)),) * 3),
            'calibrated_isolated': (('none',), anchored), 'calibrated_union': (BRANCHES, anchored)}


def contains(intervals, value):
    return any(low <= value <= high for low, high in intervals)


def make_row(task, private, decision, checked):
    truth, incumbent, result = exact(private['truth']), exact(task['incumbent']), exact(decision['output'])
    initial, error = abs(incumbent - truth), abs(result - truth)
    hull = checked['hull']
    method = decision['method']
    return {'task_id': task['task_id'], 'observation_id': task['observation_id'], 'case': private['case'],
            'method': method, 'state_group': 'correct' if not initial else 'small',
            'prior_position': 'inside_prior' if DOMAIN[0] <= incumbent <= DOMAIN[1] else 'outside_prior',
            'incumbent': str(incumbent), 'output': str(result), 'status': decision['status'],
            'initial_ae_mm': str(initial), 'ae_mm': str(error), 'se_mm2': str(error * error),
            'actual_gain_mm2': str(initial * initial - error * error),
            'covered': contains(checked['intervals'], truth),
            'hull_width_mm': None if hull is None else str(hull[1] - hull[0]),
            'retained_branch_count': len(checked['retained_branches']), 'evaluated_boxes': checked['evaluated'],
            'unresolved_budget_leaves': checked['budget_leaves'],
            'auxiliary_samples_used': 3 if method.startswith('calibrated_') else 0}


def reconstruct_aggregates(rows):
    grouped = defaultdict(list)
    for row in rows:
        conditions = [('overall', 'all'), ('by_case', row['case']),
                      ('by_information', 'negative_control' if row['case'] == 'equal_full_negative' else 'informative')]
        for family, condition in conditions:
            for state in ('all', row['state_group']):
                grouped[family, condition, state, row['method']].append(row)
    result = []
    for (family, condition, state, method), values in sorted(grouped.items()):
        n = len(values)
        errors = [exact(r['ae_mm']) for r in values]
        initial = [exact(r['initial_ae_mm']) for r in values]
        widths = [exact(r['hull_width_mm']) for r in values if r['hull_width_mm'] is not None]
        unique = {r['observation_id']: r for r in values}
        result.append({'grouping': family, 'condition': condition, 'state_group': state, 'method': method,
                       'n': n, 'observations': len(unique), 'covered_observations': sum(r['covered'] for r in unique.values()),
                       'empty_observations': sum(r['hull_width_mm'] is None for r in unique.values()),
                       'wrongly_excluded_observations': sum(not r['covered'] for r in unique.values()),
                       'improved': sum(a < b for a, b in zip(errors, initial)),
                       'harm': sum(a > b for a, b in zip(errors, initial)),
                       'same': sum(a == b for a, b in zip(errors, initial)),
                       'moved': sum(r['status'] == 'MOVE' for r in values),
                       'kept': sum(r['status'] == 'KEEP' for r in values),
                       'incompatible': sum(r['status'] == 'INCOMPATIBLE' for r in values),
                       'correct_initial_preserved': sum(r['state_group'] == 'correct' and exact(r['ae_mm']) == 0 for r in values),
                       'correct_initial_kept': sum(r['state_group'] == 'correct' and r['status'] == 'KEEP' for r in values),
                       'mae_mm': float(sum(errors) / n), 'mse_mm2': float(sum(a * a for a in errors) / n),
                       'initial_mae_mm': float(sum(initial) / n),
                       'initial_mse_mm2': float(sum(a * a for a in initial) / n), 'max_error_mm': float(max(errors)),
                       'mean_hull_width_mm': None if not widths else float(sum(widths) / len(widths)),
                       'evaluated_boxes': sum(r['evaluated_boxes'] for r in unique.values()),
                       'unresolved_budget_leaves': sum(r['unresolved_budget_leaves'] for r in unique.values()),
                       'auxiliary_samples_used': sum(r['auxiliary_samples_used'] for r in unique.values())})
    return result


def verify_csv(path, records, json_columns=()):
    require(records, f'{path.name}: no records')
    with path.open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == list(records[0]), f'{path.name}: changed CSV header order')
        observed = list(reader)
    expected = [{key: json.dumps(value) if key in json_columns else '' if value is None else str(value)
                 for key, value in row.items()} for row in records]
    same(observed, expected, path.name)


def finite_seconds(value, where):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0, f'{where}: invalid observed time')
    return value


def scientific_files():
    paths = set(ROOT.glob('*.py')) | {ROOT / 'PROTOCOL.md'}
    if (ROOT / 'THEORY.md').exists():
        paths.add(ROOT / 'THEORY.md')
    for directory in (ROOT, HYBRID_ROOT, DEPENDENCY_ROOT):
        paths.update(directory.glob('*.py'))
        for pattern in ('*.py', '*.json'):
            paths.update((directory / 'verification').rglob(pattern))
        if directory != ROOT:
            for pattern in ('*.py', '*.json'):
                paths.update((directory / 'source_snapshot').rglob(pattern))
            paths.add(directory / 'PROTOCOL.md')
            if (directory / 'THEORY.md').exists():
                paths.add(directory / 'THEORY.md')
    return {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


def verify_source_lock(lock):
    require(set(lock) == {'utc', 'source_commit', 'new_experiment_root', 'read_only_dependency_roots',
                         'path_format', 'historical_outputs_used', 'files'}, 'SOURCE_LOCK: invalid schema')
    require(lock['source_commit'] == 'bd8d05aad65e2c55cd9cd288beae4ef949606984' and
            lock['new_experiment_root'] == str(ROOT) and
            lock['read_only_dependency_roots'] == [str(HYBRID_ROOT), str(DEPENDENCY_ROOT)] and
            lock['path_format'] == 'absolute' and lock['historical_outputs_used'] is False,
            'SOURCE_LOCK: altered target or dependency contract')
    require(datetime.fromisoformat(lock['utc']).utcoffset() is not None, 'SOURCE_LOCK: UTC timestamp required')
    require(all(Path(path).is_absolute() for path in lock['files']), 'SOURCE_LOCK: nonabsolute source identity')
    current = scientific_files()
    same(lock['files'], current, 'SOURCE_LOCK hashes and complete file inventory')
    return current


def verify_strategy_boundary():
    """A focused source check, not an information-flow proof of all Python."""
    tree = ast.parse((ROOT / 'run_nuisance.py').read_text(encoding='utf-8'))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    expected = {'solve_arm': ['observation', 'radius', 'allowed_branches', 'color_intervals'],
                'select_action': ['outer', 'incumbent', 'keep'], 'arm_specifications': ['calibration'],
                'calibration_intervals': ['calibration']}
    forbidden = {'truth', 'truths', 'private', 'case', 'rear_branch', 'actual_offsets', 'random_tape',
                 'image_noise', 'calibration_noise'}
    for name, arguments in expected.items():
        node = functions[name]
        require([argument.arg for argument in node.args.args] == arguments and
                node.args.vararg is None and node.args.kwarg is None,
                f'information boundary: changed {name} inputs')
        require(not ({part.id for part in ast.walk(node) if isinstance(part, ast.Name)} & forbidden),
                f'information boundary: private generator/evaluator value used in {name}')
    imported = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module == 'hybrid']
    require(len(imported) == 1 and [(name.name, name.asname) for name in imported[0].names] ==
            [('certificate_hybrid', None)], 'information boundary: frozen hybrid import changed')
    return 'public solve/action signatures checked; generator-only truth used for generation and evaluation; not a complete static noninterference proof'


def branch_counts(decisions):
    return {method: dict(sorted(Counter(d.get('branch', 'identity') for d in decisions if d['method'] == method).items()))
            for method in METHODS}

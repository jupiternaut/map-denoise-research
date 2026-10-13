"""Read-only A14 artifact audit; no estimator, executor, renderer or solver calls.

Run with the specified existing Python, -B, and four thread limits set to one.
The only nonstandard import is the frozen standard-library certificate verifier.
Output is stdout only; the caller persists it via apply_patch.
"""
import collections
import csv
import datetime
from fractions import Fraction as Q
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import resource
import signal
import socket
import sys
import time

START_WALL, START_CPU = time.perf_counter(), time.process_time()
resource.setrlimit(resource.RLIMIT_CPU, (8, 8))
ROOT = Path('/srv/slam-research/grf/map-denoise/runs/lovelace-joint-boundary-20261013T123336Z')
FORMAL = ROOT / 'outputs/formal-v1'
ARMS = [('base', 6.), ('fine', 6.), ('joint', 6.), ('legacy', 6.), ('legacy', 24.)]
SEEDS = [20261013141, 20261013142, 20261013143]
assert socket.gethostname() == 'liekkas'
assert Path(__file__).resolve().parent == ROOT.resolve()
assert all(os.environ.get(k) == '1' for k in (
    'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'))
assert sys.dont_write_bytecode and os.environ.get('PYTHONDONTWRITEBYTECODE') == '1'
checks = collections.Counter()
problems = []
hashes = {}
max_numeric_difference = 0.


def check(condition, category, detail):
    checks[category] += 1
    if not condition:
        problems.append({'category': category, 'detail': detail})


def read(path):
    path = Path(path)
    data = path.read_bytes()
    hashes[str(path)] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(expected, actual, category, path):
    global max_numeric_difference
    if isinstance(expected, dict):
        check(isinstance(actual, dict) and set(expected) == set(actual), category, path + ': keys')
        if isinstance(actual, dict):
            for key in expected.keys() & actual.keys():
                compare(expected[key], actual[key], category, path + '.' + key)
    elif isinstance(expected, list):
        check(isinstance(actual, list) and len(expected) == len(actual), category, path + ': length')
        if isinstance(actual, list):
            for index, (a, b) in enumerate(zip(expected, actual)):
                compare(a, b, category, path + '[' + str(index) + ']')
    elif isinstance(expected, float) and isinstance(actual, (float, int)) and not isinstance(actual, bool):
        difference = abs(expected - actual)
        max_numeric_difference = max(max_numeric_difference, difference)
        check(math.isfinite(actual) and math.isclose(expected, actual, rel_tol=1e-12, abs_tol=1e-12),
              category, path + ': ' + repr((expected, actual)))
    else:
        check(expected == actual and type(expected) is type(actual), category, path + ': ' + repr((expected, actual)))


def exact(value):
    return Q(value['numerator'], value['denominator']) if isinstance(value, dict) else Q(value)


def enc(value):
    return {'numerator': value.numerator, 'denominator': value.denominator}


def float_tree(value):
    if isinstance(value, dict):
        if set(value) == {'numerator', 'denominator'}:
            return float(exact(value))
        return {k: float_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [float_tree(v) for v in value]
    return value


def projected(rectangle, depth, camera):
    if rectangle is None:
        return None
    x0, x1, y0, y1 = rectangle
    return ((Q(x0) - camera) / depth, (Q(x1) - camera) / depth, Q(y0) / depth, Q(y1) / depth)


def intersection(first, second):
    if first is None or second is None:
        return None
    return (max(first[0], second[0]), min(first[1], second[1]),
            max(first[2], second[2]), min(first[3], second[3]))


def pixel_fraction(rectangle, pixel):
    if rectangle is None:
        return Q(0)
    return (max(Q(0), min(rectangle[1], pixel[1]) - max(rectangle[0], pixel[0])) *
            max(Q(0), min(rectangle[3], pixel[3]) - max(rectangle[2], pixel[2])) * 750)


def independent_forward(fact):
    depth = exact(fact['truth'])
    offsets = list(map(exact, fact['actual_offsets']))
    colors = list(map(exact, fact['colors']))
    eta = exact(fact['eta'])
    view_colors = [color - eta * sign * min(color, 1 - color)
                   for pattern in ((-1, 0, 1), (0, 1, -1), (1, -1, 0))
                   for color, sign in zip(colors, pattern)]
    compare(list(map(enc, view_colors)), fact['view_colors'], 'exact_forward', fact['observation_id'] + '.view_colors')
    spread = max(max(view_colors[k::3]) - min(view_colors[k::3]) for k in range(3))
    compare(enc(spread), fact['actual_material_spread'], 'exact_forward', fact['observation_id'] + '.spread')
    rear = {'none': None, 'partial': (-20, 36, -17, 23), 'full': (-100, 100, -80, 80)}[fact['rear_branch']]
    pixels = [(Q(-1, 5) + Q(i, 20), Q(-1, 5) + Q(i + 1, 20),
               Q(-2, 25) + Q(2 * j, 75), Q(-2, 25) + Q(2 * (j + 1), 75))
              for j in range(6) for i in range(8)]
    ideal = []
    for view, nominal in enumerate((-80, 0, 80)):
        camera = Q(nominal) + offsets[view]
        front = projected((-20, 20, -15, 15), depth, camera)
        left = projected((-20, 0, -15, 15), depth, camera)
        right = projected((0, 20, -15, 15), depth, camera)
        background = projected(rear, Q(720), camera)
        overlap = intersection(front, background)
        for pixel in pixels:
            visible = (pixel_fraction(left, pixel), pixel_fraction(right, pixel),
                       pixel_fraction(background, pixel) - pixel_fraction(overlap, pixel))
            gray = 1 - sum(area * (1 - color) for area, color in zip(visible, view_colors[3 * view:3 * view + 3]))
            ideal.extend([gray] * 3)
    clip = lambda value: max(Q(0), min(Q(1), value))
    noise = list(map(exact, fact['image_noise']))
    bias = list(map(exact, fact['actual_view_bias']))
    photo = [enc(clip(gray + noise[i] + bias[i // 144])) for i, gray in enumerate(ideal)]
    compare(photo, fact['photo_exact'], 'exact_forward', fact['observation_id'] + '.photo_exact')
    tape = fact['anchor_tape']
    survey = exact(tape['camera_k'])
    local = list(map(exact, tape['camera_noise']))
    appearance_bias = list(map(exact, tape['appearance_t']))
    transfer = list(map(exact, tape['coupon_transfer']))
    coupon_noise = list(map(exact, tape['coupon_noise']))
    ref_noise = list(map(exact, tape['reference_noise']))
    anchors = {
        'observation_id': fact['observation_id'],
        'camera': {'u': [enc((-Q(nominal) - offsets[v] - survey - local[v]) / 1000)
                         for v, nominal in enumerate((-80, 0, 80))],
                   'Xf': enc(Q(0)), 'Zf': enc(Q(1000)), 'survey_bound': enc(Q(1, 100)),
                   'localization_bound': enc(Q(1, 100))},
        'appearance': {'coupons': [enc(clip(view_colors[i] + Q(1, 100) - appearance_bias[i // 3] + transfer[i] + coupon_noise[i]))
                                    for i in range(9)],
                       'references': [enc(clip(Q(1, 2) + Q(1, 100) - appearance_bias[i] + ref_noise[i])) for i in range(3)],
                       'bias_bound': enc(Q(1, 100)), 'transfer_bound': enc(Q(1, 200)), 'noise_bound': enc(Q(1, 200))}}
    compare(anchors, fact['anchor_exact'], 'exact_forward', fact['observation_id'] + '.anchor_exact')


freeze = read(ROOT / 'SOURCE_FREEZE.json')
for path, expected in freeze['files'].items():
    check(Path(path).is_file(), 'source_freeze', path + ': existence')
    if Path(path).is_file():
        check(digest(path) == expected, 'source_freeze', path + ': digest')
compare(SEEDS, freeze['seeds'], 'source_freeze', 'seeds')
compare([list(x) for x in ARMS], freeze['arms'], 'source_freeze', 'arms')
ancestor = ROOT.parent / 'lovelace-certified-boundary-20261013T093552Z/SOURCE_FREEZE.json'
check(digest(ancestor) == freeze['ancestor_freeze_sha256'], 'source_freeze', 'ancestor digest')
generation = read(ROOT / 'data/confirm/GENERATION.json')
public = read(ROOT / 'data/confirm/public.json')['observations']
facts = read(ROOT / 'data/confirm/private/truth.json')
tasks = read(ROOT / 'data/confirm/tasks.json')
ready = read(FORMAL / 'PREDICTIONS_READY.json')
manifest = read(FORMAL / 'MANIFEST.json')
saved_summary = read(FORMAL / 'SUMMARY.json')
saved_costs = read(FORMAL / 'COSTS.json')
saved_analysis = read(FORMAL / 'ANALYSIS.json')
saved_verification = read(FORMAL / 'INDEPENDENT_VERIFICATION.json')
for name, record in [('generation', generation), ('predictions', ready)]:
    check(record['source_freeze_sha256'] == digest(ROOT / 'SOURCE_FREEZE.json'), 'source_freeze', name)
check(len(public) == len(facts) == generation['worlds'] == 18, 'population', 'world count')
check(len(tasks) == generation['tasks'] == 126, 'population', 'task count')
compare(SEEDS, generation['seeds'], 'population', 'generation seeds')
observations = {x['observation_id']: x['observation'] for x in public}
truth = {x['observation_id']: x for x in facts}
task_index = {x['task_id']: x for x in tasks}
grouped = collections.defaultdict(list)
for task in tasks:
    grouped[task['observation_id']].append(task)
check(len(observations) == len(truth) == 18 and set(observations) == set(truth), 'population', 'unique worlds')
check(len(task_index) == 126, 'population', 'unique task IDs')

# Independent replay of RNG draws only. No forward generator or inference imports.
fact_index = 0
for block, seed in enumerate(SEEDS):
    rng = random.Random(seed)
    for branch in ('none', 'partial', 'full'):
        for stratum in ('contrast', 'lowcontrast'):
            fact = facts[fact_index]
            oid = 'w_' + format(rng.getrandbits(96), '024x')
            depth = Q(550) + Q(rng.randint(1, 100 * 100003 - 1), 100003)
            while depth in {Q(6203, 11), Q(7861, 13), Q(9007, 14)}:
                depth = Q(550) + Q(rng.randint(1, 100 * 100003 - 1), 100003)
            negative = branch == 'full' and stratum == 'lowcontrast'
            if negative:
                colors = [Q(rng.randint(15000, 65000), 100000)] * 3
            elif stratum == 'contrast':
                colors = [Q(rng.randint(a, b), 100000) for a, b in ((10000, 25000), (65000, 85000), (35000, 55000))]
            else:
                center = rng.randint(20000, 60000)
                colors = [Q(center + rng.randint(-500, 500), 100000) for _ in range(3)]
            draw = lambda n: Q(rng.randint(-n, n), 100000)
            expected = {'observation_id': oid, 'truth': enc(depth), 'rear_branch': branch,
                        'colors': list(map(enc, colors)), 'baseline': '80',
                        'actual_offsets': list(map(enc, (draw(20000), Q(0), draw(20000)))),
                        'actual_view_bias': [enc(draw(1000)) for _ in range(3)],
                        'image_noise': [enc(draw(1000)) for _ in range(432)],
                        'eta': enc(Q(0) if negative else Q(1, 10)), 'stratum': stratum,
                        'negative_control': negative, 'split': 'holdout', 'seed_block': block, 'seed': seed}
            tape = {'camera_k': enc(draw(1000)), 'camera_noise': [enc(draw(1000)) for _ in range(3)],
                    'appearance_t': [enc(Q(1, 100) - draw(1000)) for _ in range(3)],
                    'coupon_transfer': [enc(draw(500)) for _ in range(9)],
                    'coupon_noise': [enc(draw(500)) for _ in range(9)],
                    'reference_noise': [enc(draw(500)) for _ in range(3)]}
            expected['anchor_tape'] = tape
            for field, value in expected.items():
                compare(value, fact[field], 'rng_replay', oid + '.' + field)
            while True:
                shifts = [rng.choice((-1, 1)) * Q(rng.randint(1000, 80000), 10000) for _ in range(6)]
                if len(set(map(abs, shifts))) == 6 and sum(shifts) != 0 and sorted([Q(0)] + shifts)[3] != 0:
                    break
            incumbents = [depth] + [depth + shift for shift in shifts]
            rng.shuffle(incumbents)
            expected_tasks = [{'task_id': 't_' + format(rng.getrandbits(96), '024x'),
                               'observation_id': oid, 'incumbent': float(x), 'incumbent_exact': enc(x)} for x in incumbents]
            compare(expected_tasks, grouped[oid], 'rng_replay', oid + '.tasks')
            check(550 < depth < 650, 'generation_ranges', oid + ': depth')
            check(len(grouped[oid]) == 7 and sum(exact(t['incumbent_exact']) == depth for t in grouped[oid]) == 1,
                  'generation_ranges', oid + ': one correct plus six shifted')
            check(all(Q(1, 10) <= abs(s) <= 8 for s in shifts), 'generation_ranges', oid + ': shift magnitudes')
            check(all(0 <= exact(x) <= 1 for x in fact['photo_exact']), 'generation_ranges', oid + ': clipped photo')
            check(exact(fact['actual_material_spread']) <= Q(1, 10), 'generation_ranges', oid + ': material spread')
            expected_public = {'photo': list(map(float, map(exact, fact['photo_exact']))),
                               'camera': float_tree(fact['anchor_exact']['camera']),
                               'appearance': float_tree(fact['anchor_exact']['appearance'])}
            compare(expected_public, observations[oid], 'public_transport', oid)
            independent_forward(fact)
            check(public[fact_index]['observation_id'] == oid, 'population', 'public order')
            fact_index += 1
check(generation['exact_forward_verified'] == all(f['independent_exact_verified'] for f in facts),
      'generation_record', 'recorded exact forward flags; not renderer replay')
compare(max(f['float_forward_max_abs_error'] for f in facts), generation['float_forward_max_abs_error'],
        'generation_record', 'recorded max forward discrepancy')

expected_manifest = []
for index, observation in enumerate(public):
    order = ARMS[index % 5:] + ARMS[:index % 5]
    oid = observation['observation_id']
    for arm, budget in order:
        expected_manifest.append({'arm': arm, 'budget_seconds': budget, 'observation_id': oid,
                                  'observation': observations[oid],
                                  'tasks': [{k: t[k] for k in ('task_id', 'incumbent')} for t in grouped[oid]]})
compare(expected_manifest, manifest, 'manifest', 'expected full matrix')
check(ready['status'] == 'COMPLETE' and ready['calls'] == len(ready['receipts']) == len(manifest) == 90,
      'matrix', 'complete 90 jobs')
check(not (FORMAL / 'INCOMPLETE.json').exists(), 'matrix', 'no INCOMPLETE marker')
for suffix in ('job', 'result', 'receipt'):
    expected_names = {f'{i:03}.{suffix}.json' for i in range(90)}
    check({p.name for p in (FORMAL / 'jobs').glob('*.' + suffix + '.json')} == expected_names,
          'matrix', suffix + ': exact file inventory')
rows, costs, outputs, serial_costs = [], [], [], []
coverage = collections.defaultdict(list)
exact_coverage = collections.defaultdict(list)
for index, entry in enumerate(ready['receipts']):
    stem = FORMAL / 'jobs' / f'{index:03}'
    job_path, result_path, receipt_path = (Path(str(stem) + '.' + x + '.json') for x in ('job', 'result', 'receipt'))
    job, result, receipt = read(job_path), read(result_path), read(receipt_path)
    compare(manifest[index], job, 'job_integrity', f'{index}: manifest/job')
    compare(entry, receipt, 'receipt_integrity', f'{index}: ready/receipt')
    check(receipt['sha256'] == digest(result_path), 'result_integrity', f'{index}: result digest')
    check(receipt['job'] == str(job_path.relative_to(ROOT)) and receipt['result'] == str(result_path.relative_to(ROOT)),
          'receipt_integrity', f'{index}: relative targets')
    check(receipt['index'] == index and receipt['returncode'] == 0, 'matrix', f'{index}: index and returncode')
    for key in ('arm', 'budget_seconds', 'observation_id'):
        check(job[key] == receipt[key] == result[key], 'job_integrity', f'{index}: {key}')
    for phase in ('running', 'terminal'):
        record = read(FORMAL / 'state' / f'{index:03}-{phase}.json')
        check(record == {'index': index, 'status': 'running' if phase == 'running' else 'completed'},
              'matrix', f'{index}: {phase}')
    check(len(result['predictions']) == 7, 'prediction_integrity', f'{index}: prediction count')
    check([{k: p[k] for k in ('task_id', 'incumbent')} for p in result['predictions']] == job['tasks'],
          'prediction_integrity', f'{index}: task/input order and values')
    arm, budget, oid = result['arm'], result['budget_seconds'], result['observation_id']
    fact, state = truth[oid], result['state']
    z = float(exact(fact['truth']))
    hull = None if state is None else (state.get('hull') if arm == 'legacy' else [state['lower'], state['upper']])
    coverage[(arm, budget)].append(hull is not None and hull[0] <= z <= hull[1])
    exact_inside = hull is not None and Q(hull[0]) <= exact(fact['truth']) <= Q(hull[1])
    exact_coverage[f'{arm}@{budget:g}'].append(exact_inside)
    check(exact_inside, 'rational_truth_coverage', f'{index}: binary64 hull endpoints vs original rational truth')
    cost = {k: receipt[k] for k in ('arm', 'budget_seconds', 'observation_id', 'process_seconds', 'process_cpu_seconds', 'returncode')}
    cost.update(result['cost'])
    costs.append(cost)
    check(cost['overrun_gt_0_25'] == (cost['prepare_seconds'] > budget + .25), 'cost_integrity', f'{index}: overrun')
    serial = json.loads(receipt['stdout'])
    check(serial['output_bytes'] == result_path.stat().st_size, 'cost_integrity', f'{index}: serialized bytes')
    serial_costs.append(serial)
    for prediction in result['predictions']:
        a, b = abs(prediction['incumbent'] - z), abs(prediction['selected'] - z)
        rows.append({'arm': arm, 'budget_seconds': budget, 'observation_id': oid,
                     'task_id': prediction['task_id'], 'truth': z, 'incumbent': prediction['incumbent'],
                     'selected': prediction['selected'], 'initial_ae': a, 'final_ae': b,
                     'initial_se': a * a, 'final_se': b * b, 'correct_input': a <= 1e-9,
                     'failure': prediction['failure'] is not None, 'branch': fact['rear_branch'],
                     'stratum': fact['stratum'], 'negative_control': fact['negative_control'], 'seed_block': fact['seed_block']})
        expected_selection = prediction['incumbent'] if prediction['failure'] is not None or state is None else min(hull[1], max(hull[0], prediction['incumbent']))
        check(prediction['selected'] == expected_selection, 'projection', f'{index}: {prediction["task_id"]}')
    outputs.append(result)
check(len(rows) == ready['prediction_count'] == 630, 'matrix', '630 predictions')
check(len({(r['arm'], r['budget_seconds'], r['task_id']) for r in rows}) == 630, 'matrix', 'prediction uniqueness')
with (FORMAL / 'METRICS.csv').open() as stream:
    csv_rows = list(csv.DictReader(stream))
check(len(csv_rows) == len(rows), 'metrics', 'row count')
for index, (expected, actual) in enumerate(zip(rows, csv_rows)):
    parsed = {k: actual[k] == 'True' if isinstance(v, bool) else type(v)(actual[k]) for k, v in expected.items()}
    compare(expected, parsed, 'metrics', str(index))
compare(costs, saved_costs, 'costs', 'COSTS.json')


def aggregate(group):
    if not group:
        return {'n': 0}
    n = len(group)
    initial = math.fsum(r['initial_se'] for r in group) / n
    final = math.fsum(r['final_se'] for r in group) / n
    return {'n': n, 'worlds': len({r['observation_id'] for r in group}), 'mse': final,
            'mae': math.fsum(r['final_ae'] for r in group) / n, 'identity_mse': initial,
            'mse_improvement_percent': 100 * (1 - final / initial) if initial else None,
            'improved': sum(r['initial_ae'] - r['final_ae'] > 1e-9 for r in group),
            'same': sum(abs(r['initial_ae'] - r['final_ae']) <= 1e-9 for r in group),
            'harmed': sum(r['final_ae'] - r['initial_ae'] > 1e-9 for r in group),
            'max_error': max(r['final_ae'] for r in group), 'failures': sum(r['failure'] for r in group)}


summary = {}
for arm, budget in ARMS:
    group = [r for r in rows if (r['arm'], r['budget_seconds']) == (arm, budget)]
    cost_group = [c for c in costs if (c['arm'], c['budget_seconds']) == (arm, budget)]
    summary[f'{arm}@{budget:g}'] = {
        'all': aggregate(group), 'shifted': aggregate([r for r in group if not r['correct_input']]),
        'correct': aggregate([r for r in group if r['correct_input']]),
        'negative': aggregate([r for r in group if r['negative_control']]),
        'by_branch': {b: aggregate([r for r in group if r['branch'] == b]) for b in ('none', 'partial', 'full')},
        'covered_worlds': sum(coverage[(arm, budget)]), 'worlds': len(observations),
        'mean_cold_seconds': math.fsum(c['process_seconds'] for c in cost_group) / len(cost_group),
        'cpu_seconds': math.fsum(c['process_cpu_seconds'] for c in cost_group),
        'overruns': sum(c.get('overrun_gt_0_25', False) for c in cost_group)}
compare(summary, saved_summary, 'summary', 'SUMMARY.json')
indexed = {(r['arm'], r['budget_seconds'], r['task_id']): r for r in rows}
comparisons = {}
classify = lambda a, b: 'better' if a < b - 1e-9 else 'worse' if a > b + 1e-9 else 'same'
for other, budget in (('base', 6.), ('fine', 6.), ('legacy', 6.), ('legacy', 24.)):
    counts, world = collections.Counter(), collections.defaultdict(lambda: [[], []])
    for row in rows:
        if row['arm'] == 'joint' and not row['correct_input']:
            a, b = row['final_se'], indexed[(other, budget, row['task_id'])]['final_se']
            counts[classify(a, b)] += 1
            world[row['observation_id']][0].append(a)
            world[row['observation_id']][1].append(b)
    joint, baseline = summary['joint@6'], summary[f'{other}@{budget:g}']
    comparisons[f'joint_vs_{other}@{budget:g}'] = {
        'shifted_input_counts': dict(counts),
        'world_sse_counts': dict(collections.Counter(classify(math.fsum(a), math.fsum(b)) for a, b in world.values())),
        'mse_reduction_percent': 100 * (1 - joint['shifted']['mse'] / baseline['shifted']['mse']),
        'baseline_over_joint_cold_time': baseline['mean_cold_seconds'] / joint['mean_cold_seconds']}
states = {(o['arm'], o['observation_id']): o['state'] for o in outputs if o['arm'] != 'legacy' and o['state'] is not None}
proofs, components, termination = (collections.defaultdict(collections.Counter) for _ in range(3))
for (arm, oid), state in states.items():
    label = arm + '@6'
    termination[label][state['reason']] += 1
    for attempt in state['certificate_attempts']:
        mode = attempt.get('mode', 'whole')
        proofs[label][mode + '_attempts'] += 1
        proofs[label][mode + '_accepted'] += bool(attempt['verified'])
    for key, value in state['component_seconds'].items():
        components[label][key] += value
hulls = []
for oid in sorted(observations):
    b, f, j = (states[(arm, oid)] for arm in ('base', 'fine', 'joint'))
    hulls.append({'observation_id': oid, 'status': 'available', 'base_width': b['upper'] - b['lower'],
                  'fine_width': f['upper'] - f['lower'], 'joint_width': j['upper'] - j['lower'],
                  'joint_subset_base': b['lower'] <= j['lower'] <= j['upper'] <= b['upper'],
                  'joint_subset_fine': f['lower'] <= j['lower'] <= j['upper'] <= f['upper']})
analysis = {'comparisons': comparisons, 'proof_counts': {k: dict(v) for k, v in proofs.items()},
            'component_seconds': {k: dict(v) for k, v in components.items()},
            'termination': {k: dict(v) for k, v in termination.items()}, 'hull_comparisons': hulls,
            'batch_wall_seconds': ready['seconds'],
            'total_child_cpu_seconds': math.fsum(r['process_cpu_seconds'] for r in ready['receipts'])}
compare(analysis, saved_analysis, 'analysis', 'ANALYSIS.json')
gap = summary['base@6']['shifted']['mse'] - summary['legacy@24']['shifted']['mse']
report_values = {
    'joint_gap_recovery_percent': 100 * (summary['base@6']['shifted']['mse'] - summary['joint@6']['shifted']['mse']) / gap,
    'fine_gap_recovery_percent': 100 * (summary['base@6']['shifted']['mse'] - summary['fine@6']['shifted']['mse']) / gap,
    'joint_cold_increase_over_base_percent': 100 * (summary['joint@6']['mean_cold_seconds'] / summary['base@6']['mean_cold_seconds'] - 1),
    'joint_strictly_narrower_worlds_than_base': sum(h['joint_width'] < h['base_width'] for h in hulls),
    'joint_subset_base_worlds': sum(h['joint_subset_base'] for h in hulls)}
report_text = (ROOT / 'REPORT.md').read_text()
check(round(report_values['joint_gap_recovery_percent'], 4) == 86.3584 and '86.3584' in report_text, 'report_arithmetic', 'gap recovery')
check(round(report_values['fine_gap_recovery_percent'], 2) == 35.80 and '35.80' in report_text, 'report_arithmetic', 'fine gap recovery')
check(report_values['joint_strictly_narrower_worlds_than_base'] == 13 and '13个更窄' in report_text, 'report_arithmetic', 'narrower worlds')
check(round(report_values['joint_cold_increase_over_base_percent'], 2) == 41.67 and '41.67' in report_text, 'report_arithmetic', 'cold increase')

# Frozen verifier replay only, not semantic certification of the verifier itself.
verification_cpu = time.process_time()
spec = importlib.util.spec_from_file_location('a14_audit_verifier', ROOT / 'joint_verify.py')
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)
original = verifier.old.verify_certificate
certificate_counts = collections.Counter()


def dispatch(observation, entry):
    cert = entry.get('certificate', entry)
    if cert.get('schema') == verifier.SCHEMA:
        checked = verifier.verify_certificate(observation, entry)
        certificate_counts['four_tile_exclusions'] += 1
        certificate_counts['individual_tiles'] += len(cert.get('tiles', []))
        return {**checked, 'valid': checked['verified'], 'reason': checked.get('error', 'four exact tiles')}
    certificate_counts['whole_exclusions'] += 1
    return original(observation, entry)


verifier.old.verify_certificate = dispatch
verification_records = []
for (arm, oid), state in states.items():
    record = verifier.old.verify_state(observations[oid], state)
    check(record['valid'] and record.get('partitioned') is True, 'certificate_replay', arm + ':' + oid)
    verification_records.append({'arm': arm, 'observation_id': oid, **record})
verifier.old.verify_certificate = original
check(len(verification_records) == 54, 'certificate_replay', 'all 54 states')
check(sum(r.get('checked_exclusions', 0) for r in verification_records) == 1091,
      'certificate_replay', 'all 1091 exclusions')
check(certificate_counts == {'whole_exclusions': 1025, 'four_tile_exclusions': 66, 'individual_tiles': 264},
      'certificate_replay', 'whole/four-tile proof count')
saved_checks = {(r['arm'], r['observation_id']): r for r in saved_verification['records']}
for record in verification_records:
    saved = saved_checks[(record['arm'], record['observation_id'])]
    for key in ('valid', 'partitioned', 'conservative', 'checked_exclusions', 'retained_cells', 'hull'):
        compare(record[key], saved[key], 'saved_verification', record['observation_id'] + '.' + key)
verification_cpu = time.process_time() - verification_cpu

timeline_paths = ['SOURCE_FREEZE.json', 'data/confirm/public.json', 'data/confirm/tasks.json',
                  'data/confirm/private/truth.json', 'data/confirm/GENERATION.json',
                  'outputs/formal-v1/MANIFEST.json', 'outputs/formal-v1/PREDICTIONS_READY.json',
                  'outputs/formal-v1/INDEPENDENT_VERIFICATION.json', 'outputs/formal-v1/METRICS.csv',
                  'outputs/formal-v1/SUMMARY.json', 'outputs/formal-v1/COSTS.json', 'outputs/formal-v1/ANALYSIS.json']
mtime = {name: (ROOT / name).stat().st_mtime_ns for name in timeline_paths}
check(mtime[timeline_paths[0]] <= min(mtime[p] for p in timeline_paths[1:5]), 'chronology', 'freeze before generation')
check(max(mtime[p] for p in timeline_paths[1:5]) <= mtime[timeline_paths[5]], 'chronology', 'generation before manifest')
previous_terminal = mtime[timeline_paths[5]]
for index in range(90):
    path = FORMAL / 'jobs' / f'{index:03}'
    running = (FORMAL / 'state' / f'{index:03}-running.json').stat().st_mtime_ns
    job, result, receipt = (Path(str(path) + '.' + s + '.json').stat().st_mtime_ns for s in ('job', 'result', 'receipt'))
    terminal = (FORMAL / 'state' / f'{index:03}-terminal.json').stat().st_mtime_ns
    check(previous_terminal <= running <= job <= result <= receipt <= terminal,
          'chronology', f'{index}: sequential job/result/receipt/terminal')
    previous_terminal = terminal
check(previous_terminal <= mtime[timeline_paths[6]] <= mtime[timeline_paths[7]] <= mtime[timeline_paths[8]],
      'chronology', 'all predictions before verification and scoring')
check(mtime[timeline_paths[8]] <= mtime[timeline_paths[9]] <= mtime[timeline_paths[10]] <= mtime[timeline_paths[11]],
      'chronology', 'metrics summary costs analysis order')
freeze_epoch = datetime.datetime.fromisoformat(freeze['utc']).timestamp()
generation_epoch = datetime.datetime.fromisoformat(generation['utc']).timestamp()
check(freeze_epoch < generation_epoch <= mtime[timeline_paths[5]] / 1e9, 'chronology', 'record UTCs')
check(analysis['total_child_cpu_seconds'] < 570, 'cost_limits', 'formal child CPU cap')
check(ready['seconds'] < 1200, 'cost_limits', 'formal wall cap')
used_cpu = time.process_time() - START_CPU
check(used_cpu < 8, 'audit_budget', 'single process CPU below 8 seconds')
result = {
    'status': 'PASS' if not problems else 'FAIL', 'host': socket.gethostname(),
    'logical_root': str(ROOT), 'canonical_root': str(ROOT.resolve()),
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'scope': 'Independent arithmetic, RNG-draw replay, exact rectangle/occlusion and anchor forward reconstruction, transport, hashes and recorded chronology; frozen verifier replay. No original generator, renderer or solver executed.',
    'limitations': ['File mtimes and recorded UTCs are supporting chronology, not tamper-proof proof of historical reads.',
                    'No historical manifest/job/receipt hashes are stored; current bytes are hashed and linked contents checked.',
                    'Certificate replay uses the frozen verifier and does not replace independent semantic review of its math.',
                    'The freeze check covers the 56 registered files, not the complete dynamic import closure.'],
    'checks_by_category': dict(checks), 'problems': problems, 'frozen_files': len(freeze['files']),
    'matrix': {'worlds': len(public), 'tasks': len(tasks), 'jobs': len(outputs), 'predictions': len(rows),
               'receipt_result_hashes_checked': len(outputs), 'projection_checks': checks['projection']},
    'numeric_comparison': {'relative_tolerance': 1e-12, 'absolute_tolerance': 1e-12,
                           'max_absolute_difference': max_numeric_difference, 'aggregation': 'math.fsum'},
    'recomputed_summary': summary, 'recomputed_analysis': analysis,
    'recomputed_report_values': report_values,
    'independent_exact_forward': {'worlds': len(facts), 'photo_components': 432 * len(facts),
                                  'checks': checks['exact_forward'], 'original_generator_imported': False},
    'rational_truth_coverage': {k: {'covered': sum(v), 'worlds': len(v)} for k, v in exact_coverage.items()},
    'certificate_replay': {'states': len(verification_records), 'all_valid': all(r['valid'] for r in verification_records),
                           'all_partitioned': all(r.get('partitioned') for r in verification_records),
                           'checked_exclusions': sum(r.get('checked_exclusions', 0) for r in verification_records),
                           'counts': dict(certificate_counts), 'cpu_seconds': verification_cpu},
    'recorded_cost_totals': {'child_cpu_seconds': analysis['total_child_cpu_seconds'],
                             'sum_cold_seconds': math.fsum(c['process_seconds'] for c in costs),
                             'batch_wall_seconds': ready['seconds'],
                             'serialize_seconds': math.fsum(s['serialize_seconds'] for s in serial_costs),
                             'write_seconds': math.fsum(s['write_seconds'] for s in serial_costs),
                             'serialized_bytes': sum(s['output_bytes'] for s in serial_costs)},
    'generation_ranges_observed': {'depth': [min(float(exact(f['truth'])) for f in facts), max(float(exact(f['truth'])) for f in facts)],
                                   'seed_blocks': dict(collections.Counter(f['seed_block'] for f in facts)),
                                   'branches': dict(collections.Counter(f['rear_branch'] for f in facts)),
                                   'strata': dict(collections.Counter(f['stratum'] for f in facts)),
                                   'negative_worlds': sum(f['negative_control'] for f in facts)},
    'mtime_ns': mtime, 'source_freeze_utc': freeze['utc'], 'generation_utc': generation['utc'],
    'critical_sha256': {str(p): digest(ROOT / p) for p in timeline_paths},
    'all_read_json_files': len(hashes),
    'read_json_digest_map_sha256': hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
    'script_sha256': digest(__file__), 'process_cpu_seconds': used_cpu,
    'process_wall_seconds': time.perf_counter() - START_WALL,
    'process_rusage_seconds': resource.getrusage(resource.RUSAGE_SELF).ru_utime + resource.getrusage(resource.RUSAGE_SELF).ru_stime,
}
print(json.dumps(result, indent=2, allow_nan=False))

"""Read-only exact paired analysis; imports no experiment or solver code."""
from collections import defaultdict
from fractions import Fraction as Q
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
UNITS = ('R0epsilon', 'C0epsilon', 'R0delta', 'C0delta', 'RBdelta', 'CBdelta', 'CBdelta4', 'CBdelta16')
PAIRS = (
    ('R0epsilon', 'C0epsilon', 'clipping_at_narrow_bound'),
    ('R0epsilon', 'R0delta', 'widen_raw_bound'),
    ('C0epsilon', 'C0delta', 'widen_clipped_bound'),
    ('R0delta', 'RBdelta', 'bias_raw_at_fixed_bound'),
    ('C0delta', 'CBdelta', 'bias_clipped_at_fixed_bound'),
    ('R0delta', 'C0delta', 'clipping_without_bias_at_wide_bound'),
    ('RBdelta', 'CBdelta', 'clipping_with_bias_at_wide_bound'),
    ('CBdelta', 'CBdelta4', 'rounds_8_to_4'),
    ('CBdelta', 'CBdelta16', 'rounds_8_to_16'),
    ('R0epsilon', 'CBdelta', 'full_registered_bridge'),
)


def load(path):
    return json.loads(path.read_text())


def exact(value):
    if isinstance(value, dict):
        return Q(value['numerator'], value['denominator'])
    return Q(value)


def csv_save(path, rows):
    if not rows:
        raise ValueError('empty table')
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                          for key, value in row.items()} for row in rows)


def endpoint(unit, method, rows, coverage):
    values = [row for row in rows if row['unit_id'] == unit and row['method'] == method]
    offsets = [row for row in values if row['state_group'] == 'small']
    correct = [row for row in values if row['state_group'] == 'correct']
    negative = [row for row in offsets if row['case'] == 'equal_full_negative']
    cov = [row for row in coverage if row['unit_id'] == unit and row['method'] == method]
    assert (len(values), len(offsets), len(correct), len(negative), len(cov)) == (486, 432, 54, 72, 54)
    errors, initial = [exact(row['ae_mm']) for row in offsets], [exact(row['initial_ae_mm']) for row in offsets]
    widths = [exact(row['hull_width_mm']) for row in cov if row['hull_width_mm'] is not None]
    mae, mse = sum(errors, Q(0))/432, sum((error*error for error in errors), Q(0))/432
    base_mae, base_mse = sum(initial, Q(0))/432, sum((error*error for error in initial), Q(0))/432
    return {'unit_id': unit, 'method': method, 'offset_n': 432,
            'improved': sum(error < start for error, start in zip(errors, initial)),
            'same': sum(error == start for error, start in zip(errors, initial)),
            'worse': sum(error > start for error, start in zip(errors, initial)),
            'mae_exact_mm': str(mae), 'mae_mm': float(mae), 'mse_exact_mm2': str(mse), 'mse_mm2': float(mse),
            'mae_reduction_percent': float(100*(base_mae-mae)/base_mae),
            'mse_reduction_percent': float(100*(base_mse-mse)/base_mse),
            'coverage': sum(row['covered'] for row in cov), 'coverage_n': len(cov),
            'empty_observations': sum(row['incompatible'] for row in cov),
            'correct_keep': sum(row['status'] == 'KEEP' and exact(row['ae_mm']) == 0 for row in correct),
            'correct_harm': sum(exact(row['ae_mm']) > 0 for row in correct),
            'negative_keep': sum(row['status'] == 'KEEP' for row in negative),
            'incompatible_decisions': sum(row['status'] == 'INCOMPATIBLE' for row in values),
            'all_harm': sum(exact(row['ae_mm']) > exact(row['initial_ae_mm']) for row in values),
            'mean_hull_width_mm': None if not widths else float(sum(widths, Q(0))/len(widths)),
            'evaluated_boxes': None}


def analysis(output, destination):
    assert ROOT == Path('/srv/slam-research/grf/map-denoise/runs/lovelace-observation-bridge-20261011T232502Z')
    assert output.is_relative_to(ROOT) and destination.is_relative_to(ROOT / 'analysis')
    assert load(output / 'RUN_STATUS.json')['status'] == 'COMPLETE'
    verified = load(output / 'verification/VERIFY.json')
    assert verified['status'] == 'PASS' and verified['output'] == str(output)
    paths = [output / name for name in ('observations.json', 'tasks.json', 'decisions.json', 'outers.json',
             'TIMING.json', 'evaluation/ROWS.json', 'evaluation/COVERAGE.json', 'verification/VERIFY.json')]
    input_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    rows, coverage = load(output / 'evaluation/ROWS.json'), load(output / 'evaluation/COVERAGE.json')
    tasks = {row['task_id']: row for row in load(output / 'tasks.json')}
    public = {row['observation_id']: row for row in load(output / 'observations.json')}
    outers = {row['observation_id']: row for row in load(output / 'outers.json')}
    assert (len(rows), len(coverage), len(tasks), len(outers)) == (7776, 864, 3888, 432)
    endpoints = [endpoint(unit, method, rows, coverage) for unit in UNITS for method in ('keep', 'shared')]
    for entry in endpoints:
        arms = [outer['arms'][entry['method']]['outer'] for outer in outers.values() if outer['unit_id'] == entry['unit_id']]
        entry['evaluated_boxes'] = sum(arm['evaluated_boxes'] for arm in arms)
        entry['unresolved_budget_leaves'] = sum(arm['unresolved_budget_leaves'] for arm in arms)
    paired = defaultdict(dict)
    for row in rows:
        if row['method'] == 'shared':
            base_task = tasks[row['task_id']]['base_task_id']
            assert row['unit_id'] not in paired[base_task]
            paired[base_task][row['unit_id']] = row
    assert len(paired) == 486 and all(set(values) == set(UNITS) for values in paired.values())
    pair_rows, pair_summary, hull_pairs = [], [], []
    for baseline, unit, label in PAIRS:
        subset = []
        for base_task, values in paired.items():
            left, right = values[baseline], values[unit]
            assert left['incumbent'] == right['incumbent'] and left['initial_ae_mm'] == right['initial_ae_mm']
            a, b = exact(left['ae_mm']), exact(right['ae_mm'])
            start = exact(left['initial_ae_mm'])
            item = {'comparison': label, 'baseline': baseline, 'unit_id': unit, 'base_task_id': base_task,
                    'base_observation_id': left['base_observation_id'], 'case': left['case'], 'seed': left['seed'],
                    'state_group': left['state_group'], 'baseline_ae_mm': str(a), 'ae_mm': str(b),
                    'mae_change_mm': str(b-a), 'mse_change_mm2': str(b*b-a*a),
                    'better': b<a, 'same': b==a, 'worse': b>a,
                    'lost_repair': a<start and b==start,
                    'gained_repair': a==start and b<start,
                    'both_repair': a<start and b<start,
                    'same_output': exact(left['output']) == exact(right['output']),
                    'same_status': left['status'] == right['status']}
            pair_rows.append(item)
            if left['state_group'] == 'small':
                subset.append(item)
        assert len(subset) == 432
        delta_mae = sum((exact(row['mae_change_mm']) for row in subset), Q(0))/432
        delta_mse = sum((exact(row['mse_change_mm2']) for row in subset), Q(0))/432
        pair_summary.append({'comparison': label, 'baseline': baseline, 'unit_id': unit, 'offset_n':432,
            'mae_change_exact_mm': str(delta_mae), 'mae_change_mm': float(delta_mae),
            'mse_change_exact_mm2': str(delta_mse), 'mse_change_mm2': float(delta_mse),
            'better': sum(row['better'] for row in subset), 'same': sum(row['same'] for row in subset),
            'worse': sum(row['worse'] for row in subset), 'output_changed': sum(not row['same_output'] for row in subset),
            'status_changed': sum(not row['same_status'] for row in subset),
            'lost_repair':sum(row['lost_repair'] for row in subset), 'gained_repair':sum(row['gained_repair'] for row in subset),
            'both_repair':sum(row['both_repair'] for row in subset)})
        for base_oid in sorted({row['base_observation_id'] for row in public.values()}):
            left, right = outers[f'{baseline}-{base_oid}']['arms']['shared']['outer'], outers[f'{unit}-{base_oid}']['arms']['shared']['outer']
            hull_pairs.append({'comparison': label, 'base_observation_id': base_oid,
                'same_hull': left['hull'] == right['hull'], 'same_intervals':left['retained_intervals'] == right['retained_intervals'],
                'same_retained_branches':left['retained_branches'] == right['retained_branches'],
                'baseline_width_mm': None if left['hull'] is None else str(exact(left['hull'][1])-exact(left['hull'][0])),
                'width_mm': None if right['hull'] is None else str(exact(right['hull'][1])-exact(right['hull'][0])),
                'baseline_boxes': left['evaluated_boxes'], 'boxes': right['evaluated_boxes']})
    stratified, seeds = [], []
    for unit in UNITS:
        offset = [row for row in rows if row['unit_id'] == unit and row['method'] == 'shared' and row['state_group'] == 'small']
        for grouping, field in (('case', 'case'), ('seed', 'seed')):
            for condition in sorted({row[field] for row in offset}):
                part = [row for row in offset if row[field] == condition]
                errors = [exact(row['ae_mm']) for row in part]
                entry = {'unit_id': unit, 'grouping': grouping, 'condition': condition, 'n':len(part),
                    'mae_mm':float(sum(errors,Q(0))/len(part)), 'mse_mm2':float(sum((e*e for e in errors),Q(0))/len(part)),
                    'improved':sum(e<exact(row['initial_ae_mm']) for e,row in zip(errors,part)),
                    'same':sum(e==exact(row['initial_ae_mm']) for e,row in zip(errors,part)),
                    'worse':sum(e>exact(row['initial_ae_mm']) for e,row in zip(errors,part))}
                stratified.append(entry)
        part = [row for row in stratified if row['unit_id']==unit and row['grouping']=='seed']
        assert len(part)==3 and all(row['n']==144 for row in part)
        seeds.append({'unit_id':unit, 'seeds':[row['condition'] for row in part], 'n_seeds':3,
            'mae_mean_mm':statistics.mean(row['mae_mm'] for row in part), 'mae_sample_sd_mm':statistics.stdev(row['mae_mm'] for row in part),
            'mse_mean_mm2':statistics.mean(row['mse_mm2'] for row in part), 'mse_sample_sd_mm2':statistics.stdev(row['mse_mm2'] for row in part)})
    metrics = {row['unit_id']:row for row in endpoints if row['method']=='shared'}
    means = {unit:exact(metrics[unit]['mae_exact_mm']) for unit in UNITS}
    path = {'total_CBd_minus_R0e':str(means['CBdelta']-means['R0epsilon']),
            'clipped_path_terms':[str(means['C0epsilon']-means['R0epsilon']),str(means['C0delta']-means['C0epsilon']),str(means['CBdelta']-means['C0delta'])],
            'raw_path_terms':[str(means['R0delta']-means['R0epsilon']),str(means['RBdelta']-means['R0delta']),str(means['CBdelta']-means['RBdelta'])],
            'clipping_bias_interaction':str((means['CBdelta']-means['RBdelta'])-(means['C0delta']-means['R0delta'])),
            'clipping_bound_interaction':str((means['C0delta']-means['R0delta'])-(means['C0epsilon']-means['R0epsilon']))}
    assert sum(map(Q,path['clipped_path_terms']),Q(0)) == sum(map(Q,path['raw_path_terms']),Q(0)) == Q(path['total_CBd_minus_R0e'])
    summary = {'input_hashes':input_hashes, 'endpoints':endpoints, 'paired':pair_summary,
        'by_case_and_seed':stratified, 'seed_mean_sd':seeds, 'mae_paths':path}
    # Boolean-keyed counts are kept as explicit records for portable JSON.
    summary['hull_pair_counts'] = [{ 'comparison':label, 'n':54,
        'same_hull':sum(row['same_hull'] for row in hull_pairs if row['comparison']==label),
        'same_intervals':sum(row['same_intervals'] for row in hull_pairs if row['comparison']==label),
        'same_retained_branches':sum(row['same_retained_branches'] for row in hull_pairs if row['comparison']==label)}
        for _,_,label in PAIRS]
    destination.mkdir(exist_ok=False,parents=True)
    for name, table in (('ENDPOINTS.csv',endpoints),('PAIR_SUMMARY.csv',pair_summary),('PAIR_ROWS.csv',pair_rows),
                        ('HULL_PAIRS.csv',hull_pairs),('STRATIFIED.csv',stratified),('SEED_STATS.csv',seeds)):
        csv_save(destination/name,table)
    with (destination/'SUMMARY.json').open('x') as stream:
        json.dump(summary,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')
    assert input_hashes == {str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    print(json.dumps({'endpoints':endpoints, 'paired':pair_summary, 'paths':path,'hulls':summary['hull_pair_counts']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/bridge-v1')
    parser.add_argument('--destination',type=Path,default=ROOT/'analysis/analysis-v1')
    args=parser.parse_args()
    analysis(args.output.resolve(),args.destination.resolve())

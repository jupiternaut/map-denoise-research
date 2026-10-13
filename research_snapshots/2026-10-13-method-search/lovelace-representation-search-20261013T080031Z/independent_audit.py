"""Independent saved-artifact arithmetic audit; never runs a method or generator.

Invoke only after the coordinator confirms scoring complete. Uses only the
already registered source/result hashes; creates no replacement freeze.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import csv
import hashlib
import json
import math
from pathlib import Path
import socket
import time

ROOT = Path(__file__).resolve().parent
LOGICAL = Path('/srv/slam-research/grf/map-denoise/runs/lovelace-representation-search-20261013T080031Z')
TRACE = ROOT / '.aris/traces/experiment-audit/2026-10-13_run01'
TOL = 1e-9


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(actual, expected, label):
    if isinstance(expected, dict):
        assert set(actual) == set(expected), (label, set(actual) ^ set(expected))
        for key, value in expected.items():
            compare(actual[key], value, label + '.' + str(key))
    elif isinstance(expected, list):
        assert len(actual) == len(expected), (label, len(actual), len(expected))
        for index, value in enumerate(expected):
            compare(actual[index], value, label + '.' + str(index))
    elif isinstance(expected, float):
        assert math.isfinite(actual) and math.isclose(actual, expected, abs_tol=1e-10, rel_tol=1e-10), (label, actual, expected)
    else:
        assert actual == expected, (label, actual, expected)


def aggregate(rows):
    if not rows:
        return {'n': 0}
    count = len(rows)
    initial = sum(row['initial_se'] for row in rows) / count
    final = sum(row['final_se'] for row in rows) / count
    return {
        'n': count, 'worlds': len({row['observation_id'] for row in rows}),
        'mae': sum(row['final_ae'] for row in rows) / count, 'mse': final,
        'identity_mae': sum(row['initial_ae'] for row in rows) / count,
        'identity_mse': initial,
        'mse_improvement_percent': None if initial == 0 else 100 * (initial-final) / initial,
        'improved': sum(row['final_ae'] < row['initial_ae']-TOL for row in rows),
        'harmed': sum(row['final_ae'] > row['initial_ae']+TOL for row in rows),
        'same': sum(abs(row['initial_ae']-row['final_ae']) <= TOL for row in rows),
        'max_error': max(row['final_ae'] for row in rows),
        'failures': sum(row['failure'] for row in rows),
    }


def hull(result):
    state = result['state']
    if state is None:
        return None
    return [state['lower'], state['upper']] if result['arm'] == 'free' else state['hull']


def classifications(value):
    if isinstance(value, dict):
        if 'certificate' in value and 'lp_seconds' in value:
            yield value
        for item in value.values():
            yield from classifications(item)
    elif isinstance(value, list):
        for item in value:
            yield from classifications(item)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scoring-complete', action='store_true')
    parser.add_argument('--evidence-name', choices=('arithmetic.json', 'arithmetic-v2.json'), default='arithmetic.json')
    args = parser.parse_args()
    assert args.scoring_complete, 'Coordinator scoring-complete handoff required'
    wall0, cpu0 = time.perf_counter(), time.process_time()
    assert socket.gethostname() == 'liekkas' and ROOT == LOGICAL.resolve()
    folder = ROOT / 'outputs/formal-v1'
    ready = read(folder / 'PREDICTIONS_READY.json')
    assert ready['status'] == 'COMPLETE' and ready['calls'] == 108 and ready['prediction_count'] == 756
    reported = read(folder / 'SUMMARY.json')  # Required before opening truth.
    freeze = read(ROOT / 'SOURCE_FREEZE.json')
    assert all(digest(ROOT / name) == value for name, value in freeze['files'].items())
    assert digest(ROOT / 'SOURCE_FREEZE.json') == ready['source_freeze_sha256']
    public = read(ROOT / 'data/holdout/public.json')['observations']
    assert len(public) == 18 and len({p['observation_id'] for p in public}) == 18
    public_by = {p['observation_id']: p['observation'] for p in public}
    for observation in public_by.values():
        assert set(observation) == {'photo', 'camera', 'appearance'}
        assert len(observation['photo']) == 432
        assert all(math.isfinite(x) and 0 <= x <= 1 for x in observation['photo'])
        assert set(observation['camera']) == {'u', 'Xf', 'Zf', 'survey_bound', 'localization_bound'}
        assert set(observation['appearance']) == {'coupons', 'references', 'bias_bound', 'transfer_bound', 'noise_bound'}

    # First private truth read in this independent verifier.
    facts_list = read(ROOT / 'data/holdout/private/truth.json')
    facts = {f['observation_id']: f for f in facts_list}
    assert len(facts_list) == len(facts) == 18 and set(facts) == set(public_by)
    assert Counter((f['seed_block'], f['rear_branch'], f['stratum']) for f in facts.values()) == Counter(
        (block, branch, contrast) for block in range(3) for branch in ('none', 'partial', 'full') for contrast in ('contrast', 'lowcontrast'))
    truth = {oid: f['truth']['numerator']/f['truth']['denominator'] for oid, f in facts.items()}
    tasks = read(ROOT / 'data/holdout/private/tasks.json')
    task_by = {task['task_id']: task for task in tasks}
    assert len(tasks) == len(task_by) == 126
    tasks_by = defaultdict(list)
    for task in tasks:
        tasks_by[task['observation_id']].append(task)
    for oid, population in tasks_by.items():
        z = Fraction(**facts[oid]['truth'])
        offsets = [Fraction(**task['incumbent_exact'])-z for task in population]
        assert len(population) == 7 and offsets.count(0) == 1
        assert all(Fraction(1, 10) <= abs(x) <= 8 for x in offsets if x)
        assert sum(offsets) != 0 and sorted(offsets)[3] != 0
        assert len({abs(x) for x in offsets if x}) == 6
        assert all(540 <= task['incumbent'] <= 660 for task in population)
    generation = read(ROOT / 'data/holdout/private/GENERATION.json')
    assert generation['freeze']['sha256'] == digest(ROOT / 'SOURCE_FREEZE.json')
    assert datetime.fromisoformat(generation['utc']) > datetime.fromisoformat(freeze['utc'])
    assert generation['all_exact_forward_verified']
    assert all(f['independent_exact_verified'] for f in facts.values())
    assert max(generation['float_forward_max_abs_error'], generation['float_observation_max_abs_error']) <= 1e-12

    rows, costs, result_by, receipt_by, coverage = [], [], {}, {}, defaultdict(list)
    internal_fallbacks, solver_failures, atomic_overruns, legacy_classifications = [], [], [], []
    components = defaultdict(lambda: defaultdict(float))
    expected_calls = {(arm, budget, oid) for arm in ('fixed', 'library', 'free') for budget in (6., 24.) for oid in facts}
    for receipt in ready['receipts']:
        result = read(ROOT / receipt['result'])
        job = read(ROOT / receipt['job'])
        assert digest(ROOT / receipt['result']) == receipt['result_sha256']
        key = (receipt['arm'], receipt['budget_seconds'], receipt['observation_id'])
        assert key not in result_by and key in expected_calls
        arm, budget, oid = key
        assert not receipt['ablation'] and not result['ablation']
        assert (result['arm'], result['budget_seconds'], result['observation_id']) == key
        assert (job['arm'], job['budget_seconds'], job['observation_id']) == key
        compare(job['observation'], public_by[oid], 'job.observation')
        compare(job['tasks'], [{'task_id':t['task_id'], 'incumbent':t['incumbent']} for t in tasks_by[oid]], 'job.tasks')
        assert len(result['predictions']) == 7
        assert {p['task_id'] for p in result['predictions']} == {t['task_id'] for t in tasks_by[oid]}
        result_by[key], receipt_by[key] = result, receipt
        h = hull(result)
        coverage[f'{arm}@{budget:g}'].append(h is not None and h[0]-TOL <= truth[oid] <= h[1]+TOL)
        state = result['state']
        label = f'{arm}@{budget:g}'
        for field in ('import_seconds', 'decode_seconds', 'worker_seconds_before_serialize'):
            components[label][field] += result['cost'][field]
        for field in ('serialize_seconds', 'write_seconds', 'output_bytes'):
            components[label][field] += receipt['finish'][field]
        assert result['cost']['prepare_seconds'] <= budget+2
        assert result['cost']['overrun_gt_0_25'] == (result['cost']['prepare_seconds'] > budget+.25)
        assert receipt['process_seconds'] >= result['cost']['worker_seconds_before_serialize']
        legacy_classifications.extend(classifications(state))
        if state is not None and arm == 'free':
            if state['reason'] in ('encoding_or_solver_error', 'empty_enclosure_identity_fallback'):
                internal_fallbacks.append(key)
            solver_failures.extend([{'key': key, **f} for f in state['stats']['solver_failures']])
            atomic_overruns.extend([{'key': key, 'seconds': t} for t in state['stats']['atomic_over_025_seconds']])
        if state is not None and arm != 'free':
            atomic_overruns.extend([{'key':key, 'stage':x} for x in state['stage_deadline_overruns']])
        for prediction in result['predictions']:
            task = task_by[prediction['task_id']]
            assert prediction['incumbent'] == task['incumbent']
            assert math.isfinite(prediction['selected']) and 540 <= prediction['selected'] <= 660
            expected = prediction['incumbent'] if prediction['failure'] is not None or h is None else max(h[0], min(h[1], prediction['incumbent']))
            assert math.isclose(prediction['selected'], expected, rel_tol=0, abs_tol=1e-12)
            initial, final = abs(prediction['incumbent']-truth[oid]), abs(prediction['selected']-truth[oid])
            rows.append({'arm':arm, 'budget_seconds':budget, 'observation_id':oid,
                'task_id':prediction['task_id'], 'truth':truth[oid], 'incumbent':prediction['incumbent'],
                'selected':prediction['selected'], 'initial_ae':initial, 'final_ae':final,
                'initial_se':initial*initial, 'final_se':final*final, 'correct_input':initial <= TOL,
                'failure':prediction['failure'] is not None, 'branch':facts[oid]['rear_branch'],
                'stratum':facts[oid]['stratum'], 'negative_control':facts[oid]['negative_control'],
                'seed_block':facts[oid]['seed_block']})
        costs.append({**{key: receipt[key] for key in ('arm', 'budget_seconds', 'observation_id', 'process_seconds', 'process_cpu_seconds', 'returncode', 'finish')}, **result['cost']})
    assert set(result_by) == expected_calls
    assert len(rows) == len({(r['arm'], r['budget_seconds'], r['task_id']) for r in rows}) == 756
    reconstructed = {}
    for arm in ('fixed', 'library', 'free'):
        for budget in (6., 24.):
            subset = [r for r in rows if (r['arm'], r['budget_seconds']) == (arm, budget)]
            cs = [c for c in costs if (c['arm'], c['budget_seconds']) == (arm, budget)]
            summary = {'all':aggregate(subset), 'shifted':aggregate([r for r in subset if not r['correct_input']]),
                'correct':aggregate([r for r in subset if r['correct_input']]),
                'negative':aggregate([r for r in subset if r['negative_control']]),
                'by_branch':{b:aggregate([r for r in subset if r['branch'] == b]) for b in ('none', 'partial', 'full')},
                'by_stratum':{s:aggregate([r for r in subset if r['stratum'] == s]) for s in ('contrast', 'lowcontrast')},
                'by_seed':{str(b):aggregate([r for r in subset if r['seed_block'] == b]) for b in range(3)},
                'cost':{'process_seconds_total':sum(c['process_seconds'] for c in cs),
                    'process_cpu_seconds_total':sum(c['process_cpu_seconds'] for c in cs),
                    'prepare_seconds_total':sum(c.get('prepare_seconds', 0) for c in cs),
                    'prepare_cpu_seconds_total':sum(c['prepare_cpu_seconds'] for c in cs if 'prepare_cpu_seconds' in c),
                    'missing_prepare_cpu_records':sum('prepare_cpu_seconds' not in c for c in cs),
                    'overruns_gt_0_25':sum(c.get('overrun_gt_0_25', False) for c in cs),
                    'process_failures':sum(c['returncode'] != 0 for c in cs),
                    'max_rss_kib':max(c.get('max_rss_kib', 0) for c in cs)}}
            reconstructed[f'{arm}@{budget:g}'] = summary
    compare(reported, reconstructed, 'SUMMARY')
    compare(read(folder / 'COSTS.json'), costs, 'COSTS')
    with (folder / 'METRICS.csv').open(newline='') as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == len(rows)
    for actual, expected in zip(csv_rows, rows):
        assert set(actual) == set(expected)
        for key, value in expected.items():
            converted = (actual[key] == 'True') if isinstance(value, bool) else float(actual[key]) if isinstance(value, float) else int(actual[key]) if isinstance(value, int) else actual[key]
            compare(converted, value, 'METRICS.'+key)

    analysis = read(folder / 'ANALYSIS.json')
    pair_summary = {}
    for budget in (6., 24.):
        for first, second in (('fixed', 'free'), ('library', 'free'), ('fixed', 'library')):
            a = {r['task_id']:r for r in rows if (r['arm'],r['budget_seconds']) == (first,budget)}
            b = {r['task_id']:r for r in rows if (r['arm'],r['budget_seconds']) == (second,budget)}
            selected = [key for key in a if not a[key]['correct_input']]
            result = {'shifted_n':len(selected),
                'better':sum(b[key]['final_ae'] < a[key]['final_ae']-TOL for key in selected),
                'same':sum(abs(b[key]['final_ae']-a[key]['final_ae']) <= TOL for key in selected),
                'worse':sum(b[key]['final_ae'] > a[key]['final_ae']+TOL for key in selected),
                'mean_delta_se':sum(b[key]['final_se']-a[key]['final_se'] for key in selected)/len(selected)}
            label = f'{first}_to_{second}@{budget:g}'
            compare({key:analysis['pairs'][label][key] for key in result}, result, 'ANALYSIS.pairs.'+label)
            pair_summary[label] = result
    for label, flags in coverage.items():
        assert analysis['coverage'][label]['n'] == len(flags) == 18
        assert analysis['coverage'][label]['covered'] == sum(flags)
        arm, budget = label.split('@')
        widths = [hull(value)[1]-hull(value)[0] for key,value in result_by.items() if key[:2] == (arm,float(budget)) and hull(value) is not None]
        compare(analysis['coverage'][label]['mean_width'], sum(widths)/len(widths), 'ANALYSIS.coverage.mean_width')

    supplement_ready = read(ROOT / 'outputs/supplement-v1/PREDICTIONS_READY.json')
    assert supplement_ready['status'] == 'COMPLETE' and supplement_ready['selected_public_indices'] == [0, 2, 4]
    assert len(supplement_ready['receipts']) == 3
    supplement = []
    for index, receipt in zip((0, 2, 4), supplement_ready['receipts']):
        assert digest(ROOT / receipt['result']) == receipt['result_sha256']
        result = read(ROOT / receipt['result'])
        job = read(ROOT / receipt['job'])
        oid = public[index]['observation_id']
        assert result['observation_id'] == receipt['observation_id'] == oid
        assert result['arm'] == 'free' and result['budget_seconds'] == 24. and result['ablation']
        assert facts[oid]['seed_block'] == 0 and facts[oid]['stratum'] == 'contrast'
        compare(job['observation'], public_by[oid], 'supplement.job.observation')
        compare(job['tasks'], [{'task_id':t['task_id'], 'incumbent':t['incumbent']} for t in tasks_by[oid]], 'supplement.job.tasks')
        full = result_by[('free', 24., oid)]
        full_by = {p['task_id']:p for p in full['predictions']}
        assert len(result['predictions']) == 7 and {p['task_id'] for p in result['predictions']} == set(full_by)
        deltas = [abs(p['selected']-full_by[p['task_id']]['selected']) for p in result['predictions']]
        for prediction in result['predictions']:
            assert prediction['failure'] is None and prediction['incumbent'] == full_by[prediction['task_id']]['incumbent']
            compare(prediction['selected'], max(hull(result)[0], min(hull(result)[1], prediction['incumbent'])), 'supplement.decode')
        supplement.append({'observation_id':oid, 'branch':facts[oid]['rear_branch'],
            'full_mse':sum((p['selected']-truth[oid])**2 for p in full['predictions'])/7,
            'ablated_mse':sum((p['selected']-truth[oid])**2 for p in result['predictions'])/7,
            'prediction_difference_count':sum(delta > TOL for delta in deltas),
            'max_prediction_delta':max(deltas),
            'full_process_seconds':receipt_by[('free', 24., oid)]['process_seconds'],
            'ablated_process_seconds':receipt['process_seconds']})
    compare(analysis['supplement'], supplement, 'ANALYSIS.supplement')
    assert sum(c['process_seconds'] for c in costs) <= ready['seconds'] < 2400
    compact = {key:{'all':value['all'], 'shifted':value['shifted'], 'correct':value['correct'], 'cost':value['cost']} for key,value in reconstructed.items()}
    evidence = {'status':'PASS', 'host':socket.gethostname(), 'logical_root':str(LOGICAL),
        'canonical_root':str(ROOT), 'generated_at':datetime.now(timezone.utc).isoformat(),
        'source_freeze_files_verified':len(freeze['files']), 'main_calls':108,
        'main_predictions':756, 'unique_worlds':18, 'unique_incumbent_tasks':126,
        'raw_summary_cost_and_csv_match':True, 'saved_decoders_match_hull_projection':True,
        'scoring_handoff_required':True, 'holdout_generation_after_freeze':True,
        'summary':compact, 'pairs':pair_summary, 'coverage':{key:sum(value) for key,value in coverage.items()},
        'supplement':supplement, 'free_internal_fallbacks':internal_fallbacks,
        'free_solver_failures':solver_failures, 'recorded_atomic_or_stage_overruns':atomic_overruns,
        'formal_batch_seconds':ready['seconds'], 'holdout_generation_seconds':generation['generation_wall_seconds'],
        'all_process_seconds':sum(c['process_seconds'] for c in costs),
        'all_process_cpu_seconds':sum(c['process_cpu_seconds'] for c in costs),
        'timing_components':dict(components),
        'legacy_classification_count':len(legacy_classifications),
        'legacy_classification_statuses':dict(Counter(c['status'] for c in legacy_classifications)),
        'legacy_classification_max_seconds':max(c['seconds'] for c in legacy_classifications),
        'legacy_classification_over_025':sum(c['seconds'] > .25 for c in legacy_classifications),
        'audit_cpu_seconds':time.process_time()-cpu0, 'audit_wall_seconds':time.perf_counter()-wall0}
    TRACE.mkdir(parents=True, exist_ok=True)
    with (TRACE / args.evidence_name).open('x') as stream:
        json.dump(evidence, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'status':evidence['status'], 'evidence':str(TRACE / args.evidence_name),
        'summary':compact, 'pairs':pair_summary, 'coverage':evidence['coverage'], 'supplement':supplement,
        'free_internal_fallbacks':internal_fallbacks, 'free_solver_failures':solver_failures,
        'recorded_atomic_or_stage_overruns':atomic_overruns,
        'audit_cpu_seconds':evidence['audit_cpu_seconds'], 'audit_wall_seconds':evidence['audit_wall_seconds']}))


if __name__ == '__main__':
    main()

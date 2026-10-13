#!/usr/bin/env python3
"""Read-only seal and A14 arithmetic replication for the A11--A14 archive.

Standard library only; --repo can point to any checkout. Absolute source paths
in the manifest are provenance text and are never opened. No archived Python
is imported. This does not rerun experiments, forward models, or proof kernels.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from fractions import Fraction
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ARCHIVE_ROOT = 'research_snapshots/2026-10-13-method-search'
MANIFEST_PATH = 'publication/LOVELACE_METHODS_20261013_MANIFEST.json'
ROOTS = (
    'lovelace-evidence-accumulation-20261013T051931Z',
    'lovelace-representation-search-20261013T080031Z',
    'lovelace-certified-boundary-20261013T093552Z',
    'lovelace-joint-boundary-20261013T123336Z',
)
A14 = ROOTS[-1]
FORMAL = 'outputs/formal-v1/'
ARMS = (('base', 6.0), ('fine', 6.0), ('joint', 6.0), ('legacy', 6.0), ('legacy', 24.0))
EPSILON = 1e-9  # Published improved/same/harmed and paired comparison rule.

# Load the existing trusted publication seal in an isolated module, so its
# archive constants do not change for callers of the older A0--A10 verifier.
_spec = importlib.util.spec_from_file_location(
    '_lovelace_methods_seal', Path(__file__).with_name('verify_lovelace_20261013.py'))
_seal = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_seal)
_seal.ARCHIVE_ROOT, _seal.MANIFEST_PATH = ARCHIVE_ROOT, MANIFEST_PATH
AuditError, require, rational = _seal.AuditError, _seal.require, _seal.rational


class Snapshot(_seal.Snapshot):
    def source_bytes(self, relative):
        """Resolve only a sealed source-relative identity, never provenance."""
        _seal.canonical_relative(relative)
        require(relative in self.sources, 'missing archived record: ' + relative)
        return self.read_bytes(self.sources[relative])

    def source_json(self, relative):
        return json.loads(self.source_bytes(relative))


class Comparison:
    def __init__(self):
        self.leaves = 0
        self.max_numeric_difference = 0.0

    def equal(self, expected, actual, label):
        if isinstance(expected, dict):
            require(isinstance(actual, dict) and set(expected) == set(actual), label + ': keys differ')
            for key, value in expected.items():
                self.equal(value, actual[key], label + '.' + key)
        elif isinstance(expected, list):
            require(isinstance(actual, list) and len(expected) == len(actual), label + ': length differs')
            for i, (left, right) in enumerate(zip(expected, actual)):
                self.equal(left, right, f'{label}[{i}]')
        elif type(expected) is float:
            finite(actual, label)
            difference = abs(expected - actual)
            self.max_numeric_difference = max(self.max_numeric_difference, difference)
            require(math.isfinite(expected) and math.isclose(expected, actual, rel_tol=1e-12, abs_tol=1e-12),
                    label + ': numeric mismatch')
            self.leaves += 1
        else:
            require(type(expected) is type(actual) and expected == actual, label + ': value/type mismatch')
            self.leaves += 1


def finite(value, label):
    require(type(value) in (int, float) and math.isfinite(value), label + ': finite number required')
    return value


def index_unique(records, fields, label):
    require(isinstance(records, list), label + ': array required')
    indexed = {}
    for record in records:
        require(isinstance(record, dict), label + ': object required')
        key = tuple(record[field] for field in fields)
        require(key not in indexed, label + ': duplicate identity')
        indexed[key] = record
    return indexed


def aggregate(rows):
    if not rows:
        return {'n': 0}
    n = len(rows)
    initial = math.fsum(r['initial_se'] for r in rows) / n
    final = math.fsum(r['final_se'] for r in rows) / n
    return {
        'n': n, 'worlds': len({r['observation_id'] for r in rows}),
        'mse': final, 'mae': math.fsum(r['final_ae'] for r in rows) / n,
        'identity_mse': initial, 'mse_improvement_percent': 100 * (1 - final / initial) if initial else None,
        'improved': sum(r['initial_ae'] - r['final_ae'] > EPSILON for r in rows),
        'same': sum(abs(r['initial_ae'] - r['final_ae']) <= EPSILON for r in rows),
        'harmed': sum(r['final_ae'] - r['initial_ae'] > EPSILON for r in rows),
        'max_error': max(r['final_ae'] for r in rows), 'failures': sum(r['failure'] for r in rows),
    }


def parse_metrics(raw, expected_rows):
    reader = csv.DictReader(io.StringIO(raw.decode('utf-8'), newline=''))
    fields = list(expected_rows[0])
    require(reader.fieldnames == fields, 'METRICS.csv: columns differ')
    records = list(reader)
    require(len(records) == len(expected_rows), 'METRICS.csv: row count differs')
    converted = []
    for record, expected in zip(records, expected_rows):
        require(set(record) == set(fields) and all(value is not None for value in record.values()),
                'METRICS.csv: malformed row')
        row = {}
        for key, value in expected.items():
            if type(value) is bool:
                require(record[key] in ('True', 'False'), 'METRICS.csv: invalid boolean')
                row[key] = record[key] == 'True'
            else:
                row[key] = type(value)(record[key])
        converted.append(row)
    return converted


def paired_comparisons(rows, summary):
    indexed = index_unique(rows, ('arm', 'budget_seconds', 'task_id'), 'scored rows')
    classify = lambda a, b: 'better' if a < b - EPSILON else 'worse' if a > b + EPSILON else 'same'
    comparisons = {}
    for arm, budget in (ARMS[0], ARMS[1], ARMS[3], ARMS[4]):
        counts, worlds = Counter(), defaultdict(lambda: [[], []])
        for row in rows:
            if row['arm'] != 'joint' or row['correct_input']:
                continue
            a, b = row['final_se'], indexed[(arm, budget, row['task_id'])]['final_se']
            counts[classify(a, b)] += 1
            worlds[row['observation_id']][0].append(a)
            worlds[row['observation_id']][1].append(b)
        joint, other = summary['joint@6'], summary[f'{arm}@{budget:g}']
        require(other['shifted']['mse'] > 0 and joint['mean_cold_seconds'] > 0,
                'paired comparison denominator must be positive')
        comparisons[f'joint_vs_{arm}@{budget:g}'] = {
            'shifted_input_counts': dict(counts),
            'world_sse_counts': dict(Counter(classify(math.fsum(a), math.fsum(b)) for a, b in worlds.values())),
            'mse_reduction_percent': 100 * (1 - joint['shifted']['mse'] / other['shifted']['mse']),
            'baseline_over_joint_cold_time': other['mean_cold_seconds'] / joint['mean_cold_seconds'],
        }
    return comparisons


def numerical_audit(snapshot):
    """Reproduce floating-point scoring, never import archived research code."""
    load = lambda name: snapshot.source_json(A14 + '/' + name)
    check = Comparison()
    tasks = load('data/confirm/tasks.json')
    facts = load('data/confirm/private/truth.json')
    observations = load('data/confirm/public.json')['observations']
    task_index = index_unique(tasks, ('task_id',), 'tasks')
    truth = index_unique(facts, ('observation_id',), 'evaluation truth')
    public = index_unique(observations, ('observation_id',), 'public observations')
    require(len(truth) == len(public) == 18 and set(truth) == set(public), '18 matching worlds required')
    require(len(task_index) == 126 and Counter(t['observation_id'] for t in tasks) ==
            Counter({oid: 7 for oid, in truth}), '126 tasks, seven per world required')
    grouped = defaultdict(list)
    for task in tasks:
        incumbent = finite(task['incumbent'], 'task incumbent')
        require(incumbent == float(rational(task['incumbent_exact'])), 'task incumbent differs from exact input')
        grouped[task['observation_id']].append({'task_id': task['task_id'], 'incumbent': incumbent})
    ready = load(FORMAL + 'PREDICTIONS_READY.json')
    require(ready['status'] == 'COMPLETE' and ready['calls'] == 90 and ready['prediction_count'] == 630,
            'complete 90-call, 630-prediction matrix required')
    receipts = ready['receipts']
    receipt_index = index_unique(receipts, ('arm', 'budget_seconds', 'observation_id'), 'receipts')
    require(set(receipt_index) == {(arm, budget, oid) for arm, budget in ARMS for oid, in truth},
            'receipt matrix differs from five configurations by 18 worlds')
    manifest = load(FORMAL + 'MANIFEST.json')
    require(isinstance(manifest, list) and len(manifest) == 90, '90 frozen jobs required')
    rows, costs, outputs = [], [], []
    coverage, exact_coverage = defaultdict(list), defaultdict(list)
    for i, receipt in enumerate(receipts):
        stem = FORMAL + f'jobs/{i:03}'
        require(receipt['index'] == i and receipt['job'] == stem + '.job.json' and
                receipt['result'] == stem + '.result.json', 'receipt target/index mismatch')
        check.equal(receipt, load(stem + '.receipt.json'), f'receipt {i}')
        job = load(stem + '.job.json')
        check.equal(manifest[i], job, f'frozen job {i}')
        raw = snapshot.source_bytes(A14 + '/' + stem + '.result.json')
        require(hashlib.sha256(raw).hexdigest() == receipt['sha256'], 'receipt result digest mismatch')
        out = json.loads(raw)
        arm, budget, oid = (out[k] for k in ('arm', 'budget_seconds', 'observation_id'))
        for key in ('arm', 'budget_seconds', 'observation_id'):
            check.equal(receipt[key], out[key], f'result {i}.{key}')
            check.equal(receipt[key], job[key], f'job {i}.{key}')
        check.equal(grouped[oid], job['tasks'], f'job {i} inputs')
        check.equal(public[(oid,)]['observation'], job['observation'], f'job {i} public observation')
        predictions = out['predictions']
        require(len(predictions) == 7, 'seven predictions per call required')
        check.equal(grouped[oid], [{k: p[k] for k in ('task_id', 'incumbent')} for p in predictions],
                    f'prediction {i} input identities')
        fact, state = truth[(oid,)], out['state']
        exact_z = rational(fact['truth'])
        z = finite(float(exact_z), 'truth')
        hull = None if state is None else state.get('hull') if arm == 'legacy' else [state['lower'], state['upper']]
        if hull is not None:
            require(isinstance(hull, list) and len(hull) == 2, 'two hull endpoints required')
            require(finite(hull[0], 'lower hull') <= finite(hull[1], 'upper hull'), 'reversed hull')
        coverage[(arm, budget)].append(hull is not None and hull[0] <= z <= hull[1])
        exact_coverage[f'{arm}@{budget:g}'].append(
            hull is not None and Fraction(hull[0]) <= exact_z <= Fraction(hull[1]))
        cost = {k: receipt[k] for k in ('arm', 'budget_seconds', 'observation_id',
                                      'process_seconds', 'process_cpu_seconds', 'returncode')}
        require(not set(cost).intersection(out['cost']), 'worker cost overrides receipt identity/cost')
        cost.update(out['cost'])
        for key, value in cost.items():
            if key.endswith('_seconds'):
                require(finite(value, 'cost ' + key) >= 0, 'negative cost')
        require(type(cost['returncode']) is int, 'integer return code required')
        if 'overrun_gt_0_25' in cost:
            require(type(cost['overrun_gt_0_25']) is bool and
                    cost['overrun_gt_0_25'] == (cost['prepare_seconds'] > budget + 0.25), 'cost overrun flag mismatch')
        costs.append(cost)
        for prediction in predictions:
            incumbent = finite(prediction['incumbent'], 'prediction incumbent')
            selected = finite(prediction['selected'], 'prediction selected')
            fallback = prediction['failure'] is not None or state is None
            require(fallback or hull is not None, 'successful output requires a hull')
            expected = incumbent if fallback else min(hull[1], max(hull[0], incumbent))
            require(selected == expected, 'prediction differs from saved hull projection/fallback')
            initial, final = abs(incumbent - z), abs(selected - z)
            rows.append(dict(arm=arm, budget_seconds=budget, observation_id=oid,
                             task_id=prediction['task_id'], truth=z, incumbent=incumbent, selected=selected,
                             initial_ae=initial, final_ae=final, initial_se=initial * initial, final_se=final * final,
                             correct_input=initial <= EPSILON, failure=prediction['failure'] is not None,
                             branch=fact['rear_branch'], stratum=fact['stratum'],
                             negative_control=fact['negative_control'], seed_block=fact['seed_block']))
        outputs.append(out)
    require(len(index_unique(rows, ('arm', 'budget_seconds', 'task_id'), 'predictions')) == 630,
            '630 unique task outputs required')
    check.equal(rows, parse_metrics(snapshot.source_bytes(A14 + '/' + FORMAL + 'METRICS.csv'), rows), 'METRICS.csv')
    check.equal(costs, load(FORMAL + 'COSTS.json'), 'COSTS.json')
    summary = {}
    for arm, budget in ARMS:
        group = [r for r in rows if (r['arm'], r['budget_seconds']) == (arm, budget)]
        cc = [c for c in costs if (c['arm'], c['budget_seconds']) == (arm, budget)]
        require(len(group) == 126 and sum(r['correct_input'] for r in group) == 18 and len(cc) == 18,
                'each configuration requires 18 correct and 108 shifted inputs, and 18 cold calls')
        require(all(sum(r['correct_input'] for r in group if r['observation_id'] == oid) == 1 for oid, in truth),
                'exactly one correct input per world required')
        summary[f'{arm}@{budget:g}'] = {
            'all': aggregate(group), 'shifted': aggregate([r for r in group if not r['correct_input']]),
            'correct': aggregate([r for r in group if r['correct_input']]),
            'negative': aggregate([r for r in group if r['negative_control']]),
            'by_branch': {b: aggregate([r for r in group if r['branch'] == b]) for b in ('none', 'partial', 'full')},
            'covered_worlds': sum(coverage[(arm, budget)]), 'worlds': 18,
            'mean_cold_seconds': math.fsum(c['process_seconds'] for c in cc) / 18,
            'cpu_seconds': math.fsum(c['process_cpu_seconds'] for c in cc),
            'overruns': sum(c.get('overrun_gt_0_25', False) for c in cc),
        }
    check.equal(summary, load(FORMAL + 'SUMMARY.json'), 'SUMMARY.json')
    comparisons = paired_comparisons(rows, summary)
    analysis = load(FORMAL + 'ANALYSIS.json')
    check.equal(comparisons, analysis['comparisons'], 'ANALYSIS.json.comparisons')
    child_cpu = math.fsum(c['process_cpu_seconds'] for c in costs)
    check.equal(child_cpu, analysis['total_child_cpu_seconds'], 'ANALYSIS.json.total_child_cpu_seconds')
    check.equal(ready['seconds'], analysis['batch_wall_seconds'], 'ANALYSIS.json.batch_wall_seconds')
    states = {(o['arm'], o['observation_id']): o['state'] for o in outputs if o['arm'] != 'legacy'}
    hulls = []
    for oid, in sorted(truth):
        missing = [arm for arm in ('base', 'fine', 'joint') if states[(arm, oid)] is None]
        if missing:
            hulls.append({'observation_id': oid, 'status': 'unavailable', 'missing_arms': missing})
            continue
        base, fine, joint = (states[(arm, oid)] for arm in ('base', 'fine', 'joint'))
        hulls.append(dict(observation_id=oid, status='available', base_width=base['upper'] - base['lower'],
                          fine_width=fine['upper'] - fine['lower'], joint_width=joint['upper'] - joint['lower'],
                          joint_subset_base=base['lower'] <= joint['lower'] <= joint['upper'] <= base['upper'],
                          joint_subset_fine=fine['lower'] <= joint['lower'] <= joint['upper'] <= fine['upper']))
    check.equal(hulls, analysis['hull_comparisons'], 'ANALYSIS.json.hull_comparisons')
    base, fine, joint, legacy = (summary[k] for k in ('base@6', 'fine@6', 'joint@6', 'legacy@24'))
    gap = base['shifted']['mse'] - legacy['shifted']['mse']
    require(gap != 0 and base['mean_cold_seconds'] > 0, 'report ratio denominator is zero')
    values = {
        'joint_gap_recovery_percent': 100 * (base['shifted']['mse'] - joint['shifted']['mse']) / gap,
        'fine_gap_recovery_percent': 100 * (base['shifted']['mse'] - fine['shifted']['mse']) / gap,
        'joint_cold_increase_over_base_percent': 100 * (joint['mean_cold_seconds'] / base['mean_cold_seconds'] - 1),
        'joint_strictly_narrower_worlds_than_base': sum(h['joint_width'] < h['base_width'] for h in hulls if h['status'] == 'available'),
        'joint_subset_base_worlds': sum(h['joint_subset_base'] for h in hulls if h['status'] == 'available'),
    }
    return {
        'worlds': 18, 'tasks': 126, 'cold_calls': 90, 'task_outputs_recomputed': len(rows),
        'all_metric_rows_match': True, 'all_summary_groups_match': True, 'all_cold_cost_records_match': True,
        'paired_comparisons_and_hull_arithmetic_match': True,
        'numeric_relative_tolerance': 1e-12, 'numeric_absolute_tolerance': 1e-12,
        'classification_tolerance': EPSILON, 'compared_leaves': check.leaves,
        'max_numeric_difference': check.max_numeric_difference,
        'exact_rational_truth_in_saved_hulls': {k: sum(v) for k, v in exact_coverage.items()},
        'per_configuration': {k: {field: v[field] for field in
                                 ('shifted', 'correct', 'negative', 'mean_cold_seconds', 'cpu_seconds', 'overruns')}
                              for k, v in summary.items()},
        'paired_comparisons': comparisons, 'report_key_values': values,
        'total_child_cpu_seconds': child_cpu, 'batch_wall_seconds': ready['seconds'],
        'limits': ['Arithmetic replication from saved predictions, task inputs, evaluation truth, and receipts.',
                   'No experiment, estimator, forward generator, or certificate proof kernel was executed.',
                   'Cold costs are reconciled saved measurements; they are not remeasured.',
                   'A11-A13 receive archive integrity checks only. A14 arithmetic is recomputed.',
                   'Report key values are derived here; free-form Markdown claims are not parsed or certified.',
                   'Archive hashes establish consistency with this manifest, not independent authenticity.'],
    }


def verify(repository):
    snapshot = Snapshot(repository)
    require(snapshot.manifest.get('source_roots') == list(ROOTS), 'unexpected four source roots')
    require({name.split('/')[0] for name in snapshot.sources} == set(ROOTS), 'archive roots differ from A11-A14')
    seal = snapshot.seal()
    return {'status': 'PASS', 'archive_root': ARCHIVE_ROOT, 'seal': seal,
            'a14_arithmetic': numerical_audit(snapshot), 'read_only': True, 'source_provenance_paths_opened': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        result = verify(args.repo)
    except (AuditError, OSError, ValueError, KeyError, TypeError, IndexError, OverflowError, EOFError) as error:
        print(json.dumps({'status': 'FAIL', 'error': str(error)[:2000]}, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

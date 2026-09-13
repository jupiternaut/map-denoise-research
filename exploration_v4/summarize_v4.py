"""Read saved V4 results; report paired effects without modifying the run."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import statistics

METRICS = (
    'surface_accuracy_mean_mm', 'surface_accuracy_rms_mm',
    'surface_accuracy_p95_mm', 'matched_point_rms_mm',
    'reference_sample_coverage_1mm', 'reference_sample_coverage_2mm',
    'fitted_gap_at_same_xy_error_mm', 'source_group_gap_error_mm',
    'method_seconds', 'rigid_equivariance_rms_mm',
)
PAIRS = (
    ('pool_independent', 'pool_compatible'),
    ('pool_global', 'pool_compatible'),
    ('graph_shared', 'pool_compatible'),
    ('graph_projected_bias_only', 'warm_balanced_2'),
    ('graph_projected_bias_only', 'warm_unbalanced_2'),
    ('zero_balanced_6', 'warm_balanced_2'),
    ('zero_unbalanced_6', 'warm_unbalanced_2'),
)


def stats(values):
    return {'n': len(values), 'mean': statistics.mean(values),
            'median': statistics.median(values), 'min': min(values),
            'max': max(values)} if values else None


def aggregate(rows):
    methods = sorted({r['method'] for r in rows})
    by_method = {}
    for method in methods:
        selected = [r for r in rows if r['method'] == method]
        by_method[method] = {
            key: stats([float(r[key]) for r in selected if r.get(key)])
            for key in METRICS}
    paired = {}
    lookup = {(r['case'], r['method']): r for r in rows}
    cases = sorted({r['case'] for r in rows})
    for first, second in PAIRS:
        metrics = {}
        for key in METRICS:
            observations = []
            for case in cases:
                a, b = lookup.get((case, first)), lookup.get((case, second))
                if a and b and a.get(key) and b.get(key):
                    observations.append((case, float(a[key]), float(b[key])))
            if not observations:
                continue
            # Lower is better except coverage; time is cost, not accuracy.
            direction = -1 if 'coverage' in key else 1
            gains = [(case, direction * (a-b)) for case, a, b in observations]
            metrics[key] = {
                'n': len(gains), 'positive_gain_is_better': True,
                'gain': stats([x[1] for x in gains]),
                'improved': sum(x[1] > 1e-10 for x in gains),
                'tied': sum(abs(x[1]) <= 1e-10 for x in gains),
                'worse': sum(x[1] < -1e-10 for x in gains),
                'cases': [{'case': case, 'gain': gain} for case, gain in gains],
            }
        paired[first + ' -> ' + second] = metrics
    return {'inputs': len(cases), 'methods': by_method, 'paired': paired}


def summarize(path):
    with (path/'RESULTS.csv').open(newline='') as handle:
        all_rows = list(csv.DictReader(handle))
    if any(r['ok'] != 'True' for r in all_rows):
        raise ValueError('Failures must be reported before aggregation')
    rows = [r for r in all_rows if r.get('geometry_scored') == 'True']
    full = [r for r in rows if r['sampling'] == 'full']
    groups = {'full_identifiable': full,
              'diagnostics_all': [r for r in rows if r['sampling'] != 'full']}
    for gap in sorted({float(r['gap_mm']) for r in full}):
        groups[f'full_gap_{gap:g}'] = [r for r in full if float(r['gap_mm']) == gap]
    for bias in sorted({float(r['bias_rms_mm']) for r in full}):
        groups[f'full_bias_{bias:g}'] = [r for r in full if float(r['bias_rms_mm']) == bias]
    for sampling in sorted({r['sampling'] for r in rows} - {'full'}):
        groups[sampling] = [r for r in rows if r['sampling'] == sampling]
    return {'run': str(path.resolve()), 'rows': len(all_rows),
            'excluded_unidentifiable_rows': len(all_rows)-len(rows),
            'note': 'Per-case descriptive statistics; cases share three seeds. No independence or significance claim.',
            'groups': {key: aggregate(selected) for key, selected in groups.items() if selected}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()
    result = summarize(args.run)
    if args.save:
        with (args.run/'AGGREGATES.json').open('x') as handle:
            json.dump(result, handle, indent=2)
    group = result['groups']['full_identifiable']
    print('Full identifiable cases:', group['inputs'])
    print('method | mean surface MAE mm | median surface MAE mm | median time ms')
    for name, metrics in group['methods'].items():
        score = metrics['surface_accuracy_mean_mm']
        print(f"{name} | {score['mean']:.6f} | {score['median']:.6f} | {1000*metrics['method_seconds']['median']:.3f}")
    print('\nPaired surface changes, positive means improvement:')
    for pair, metrics in group['paired'].items():
        score = metrics['surface_accuracy_mean_mm']
        print(f"{pair}: {score['gain']['mean']:.6f} mm; improved/tied/worse={score['improved']}/{score['tied']}/{score['worse']}")

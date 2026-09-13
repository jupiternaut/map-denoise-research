"""Additional independent all-output layer/threshold checks and K counts."""
import collections
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    run = ROOT / 'synthetic_results'
    rows = json.loads((run / 'results.json').read_text())['records']
    biggest = 0.
    checked = 0
    groups = collections.defaultdict(collections.Counter)
    for row in rows:
        if row['execution'] != 'ok':
            raise ValueError('Cannot hide execution failures')
        with np.load(run / row['output_path'], allow_pickle=False) as data:
            z = data['xyz_mm'][:, 2]
        with np.load(ROOT / 'data' / row['eval_path'], allow_pickle=False) as data:
            gt = data['clean_xyz_mm'][:, 2]
            labels = data['labels']
        if row['gap_mm'] > 0:
            residual = z - gt
            gap_error = abs(float(np.average(residual[labels == 1]) - np.average(residual[labels == 0])))
            half = float(np.count_nonzero(np.abs(residual) > row['gap_mm'] / 2) / len(z))
            biggest = max(biggest, abs(gap_error - row['metrics']['layer_gap_error_mm']),
                          abs(half - row['metrics']['half_gap_error_fraction']))
            checked += 1
        arm = 'correlated' if row['arm'] == 'correlated' else 'shuffled'
        groups[(row['method'], row['sigma_protocol'], row['family'], row['amplitude_mm'], arm)][str(row['selected_k'])] += 1
    records = []
    for (method, mode, family, amplitude, arm), counts in sorted(groups.items()):
        records.append({'method': method, 'sigma_protocol': mode, 'family': family,
                        'amplitude_mm': amplitude, 'arm': arm, 'selected_k_counts': dict(counts)})
    result = {'checked_dual_outputs': checked, 'max_metric_difference': biggest, 'k_counts': records}
    with (ROOT / 'analysis' / 'geometry_audit.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in result.items() if k != 'k_counts'}))
    if biggest > 1e-10:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

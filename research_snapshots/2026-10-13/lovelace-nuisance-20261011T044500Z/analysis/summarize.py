"""Post-confirmation descriptive statistics from saved truth-evaluation rows.

This script never creates actions or changes scientific sources. Noise-seed
summaries describe repeated measurements of the fixed synthetic settings,
not uncertainty over independently sampled real scenes.
"""
import json
from collections import defaultdict
from fractions import Fraction as Q
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'outputs/confirmation-v1'


def stats(rows):
    initial = [Q(row['initial_ae_mm']) for row in rows]
    errors = [Q(row['ae_mm']) for row in rows]
    n = len(rows)
    return {'n': n, 'improved': sum(a < b for a, b in zip(errors, initial)),
            'same': sum(a == b for a, b in zip(errors, initial)),
            'harm': sum(a > b for a, b in zip(errors, initial)),
            'mae_mm': float(sum(errors) / n),
            'mse_mm2': float(sum(a*a for a in errors) / n),
            'initial_mae_mm': float(sum(initial) / n),
            'mae_reduction_percent': 0.0 if not sum(initial) else float(100 * (1-sum(errors)/sum(initial)))}


def main():
    rows = json.loads((OUTPUT / 'evaluation/ROWS.json').read_text())
    private = {row['observation_id']: row for row in json.loads((OUTPUT / 'evaluation/truth.json').read_text())}
    groups = defaultdict(list)
    for row in rows:
        if row['state_group'] == 'small':
            for scope in ('all', row['case'], 'negative' if row['case'] == 'equal_full_negative' else 'informative'):
                groups[scope, row['method']].append(row)
    records = []
    for (scope, method), values in sorted(groups.items()):
        seeded = defaultdict(list)
        for row in values:
            seeded[private[row['observation_id']]['seed']].append(row)
        seed_stats = [{'seed': seed, **stats(sample)} for seed, sample in sorted(seeded.items())]
        means = [s['mae_mm'] for s in seed_stats]
        records.append({'scope': scope, 'method': method, **stats(values),
                        'noise_seed_mae_mean_mm': mean(means),
                        'noise_seed_mae_sample_std_mm': stdev(means), 'by_noise_seed': seed_stats})
    indexed = {(r['task_id'], r['method']): r for r in rows}
    paired = []
    for first, second in (('calibrated_union', 'expanded_unanchored'),
                          ('calibrated_union', 'calibrated_isolated')):
        for scope in ('all', 'informative', 'negative'):
            tasks = [r for r in rows if r['method'] == first and r['state_group'] == 'small' and
                     (scope == 'all' or (r['case'] == 'equal_full_negative') == (scope == 'negative'))]
            differences = [Q(r['ae_mm']) - Q(indexed[r['task_id'], second]['ae_mm']) for r in tasks]
            paired.append({'first': first, 'second': second, 'scope': scope, 'n': len(tasks),
                           'first_better': sum(x < 0 for x in differences),
                           'same': sum(x == 0 for x in differences), 'first_worse': sum(x > 0 for x in differences),
                           'first_minus_second_mae_mm': float(sum(differences)/len(differences))})
    result = {'descriptive_statistics': records, 'paired_comparisons': paired,
              'sampling_note': 'Three noise repeats per fixed case/depth; shared initial states and arms. Std is over noise-seed means, not independent-scene uncertainty.'}
    with (OUTPUT / 'evaluation/ANALYSIS.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'all_small': [r for r in records if r['scope'] == 'all'],
                      'paired': paired}, ensure_ascii=False))


if __name__ == '__main__':
    main()

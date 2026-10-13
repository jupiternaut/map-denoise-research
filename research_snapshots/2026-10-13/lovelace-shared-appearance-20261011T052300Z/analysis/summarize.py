"""Descriptive analysis only; no new decisions or tuned scientific parameters."""
import json
from fractions import Fraction as Q
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/confirmation-v1'


def main():
    rows = json.loads((OUT/'evaluation/ROWS.json').read_text())
    outers = json.loads((OUT/'outers.json').read_text())
    summary = json.loads((OUT/'evaluation/SUMMARY.json').read_text())
    methods = ['keep','channelwise_union','shared_union','calibrated_union']
    by_seed = {}
    for method in methods:
        means = []
        per_seed = []
        for seed in sorted({r['seed'] for r in rows}):
            selected = [r for r in rows if r['method']==method and r['seed']==seed and r['state_group']=='small']
            errors = [Q(r['ae_mm']) for r in selected]
            initial = [Q(r['initial_ae_mm']) for r in selected]
            mean = sum(errors)/len(errors)
            means.append(float(mean))
            per_seed.append({'seed':seed,'n':len(selected),'mae_mm':str(mean),
                             'improved':sum(a<b for a,b in zip(errors,initial)),
                             'harm':sum(a>b for a,b in zip(errors,initial))})
        by_seed[method] = {'seeds':per_seed, 'mean_mae_mm':statistics.mean(means),
                           'sample_std_across_noise_seed_means':statistics.stdev(means)}
    by_task = {}
    for row in rows:
        by_task.setdefault(row['task_id'],{})[row['method']] = row
    pairs = {}
    for baseline in ('channelwise_union','calibrated_union'):
        arms = [a for a in by_task.values() if a['shared_union']['state_group']=='small']
        shared = [Q(a['shared_union']['ae_mm']) for a in arms]
        base = [Q(a[baseline]['ae_mm']) for a in arms]
        pairs[baseline] = {'n':len(arms),'shared_better':sum(a<b for a,b in zip(shared,base)),
                           'same':sum(a==b for a,b in zip(shared,base)),
                           'shared_worse':sum(a>b for a,b in zip(shared,base)),
                           'mae_difference_shared_minus_baseline_mm':str(sum(a-b for a,b in zip(shared,base))/len(arms)),
                           'information_equal':baseline=='channelwise_union'}
    shared_nodes = [n for entry in outers for b in entry['arms']['shared_union']['branches'].values() for n in b['trace']]
    excluded = [n for n in shared_nodes if n['status']=='excluded_shared']
    informative_widths = []
    truths = {r['observation_id']:r for r in json.loads((OUT/'evaluation/truth.json').read_text())}
    for entry in outers:
        if truths[entry['observation_id']]['case']!='equal_full_negative':
            hull = entry['arms']['shared_union']['outer']['hull']
            q = lambda d: Q(d['numerator'],d['denominator'])
            informative_widths.append(q(hull[1])-q(hull[0]))
    result = {'noise_seed_statistics':by_seed, 'paired_small_tasks':pairs,
              'shared_constraint_certificates':len(excluded),
              'shared_certificate_bound_updates':sum(len(n['certificate']['steps']) for n in excluded),
              'informative_shared_mean_hull_width_mm':str(sum(informative_widths)/len(informative_widths)),
              'main_rows':[r for r in summary['rows'] if r['grouping'] in ('overall','by_case','by_information') and r['state_group']=='small'],
              'scope':'Descriptive repeat-noise statistics on fixed synthetic templates; no inference to independent real scenes'}
    path = OUT/'evaluation/ANALYSIS.json'
    with path.open('x') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()

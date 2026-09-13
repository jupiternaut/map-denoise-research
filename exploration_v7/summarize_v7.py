"""Paired public-case summaries, exposing budget and geometry tradeoffs."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np

METRICS = ('surface_accuracy_mean_mm', 'matched_point_rms_mm',
           'fitted_gap_at_same_xy_error_mm', 'evaluation_only_weightless_group_impurity')

def number(row, key):
    value = row.get(key)
    return float(value) if value not in (None, '', 'None') else None

def compare(rows, left, right, subset):
    by = {(r['case'], r['method']): r for r in rows}
    cases = sorted({r['case'] for r in rows if subset(r)})
    result = dict(left=left, right=right, lower_is_better=True)
    for metric in METRICS:
        pairs = [(number(by[(c, left)], metric), number(by[(c, right)], metric)) for c in cases]
        pairs = [(a,b) for a,b in pairs if a is not None and b is not None]
        if not pairs: continue
        a, b = np.asarray(pairs).T; delta = a-b
        result[metric] = dict(n=len(a), left_mean=float(a.mean()), right_mean=float(b.mean()),
             mean_difference=float(delta.mean()), wins=int((delta < -1e-8).sum()),
             ties=int((abs(delta) <= 1e-8).sum()), losses=int((delta > 1e-8).sum()))
    ratios = {}
    for key in ('info_search_work_proxy', 'measured_fit_seconds', 'shared_freeze_plus_fit_seconds'):
        pairs = [(number(by[(c,left)],key), number(by[(c,right)],key)) for c in cases]
        values = [a/b for a,b in pairs if a is not None and b not in (None, 0.)]
        if values: ratios[key] = dict(min=float(min(values)), median=float(np.median(values)), max=float(max(values)))
    result['left_over_right_cost_ratios'] = ratios
    return result

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('run', type=Path)
    args = parser.parse_args(); run=args.run.resolve()
    rows=list(csv.DictReader((run/'RESULTS.csv').open()))
    config=json.loads((run/'CONFIG.json').read_text())
    comparisons={}
    subsets={'all':lambda r: True, 'dual':lambda r: float(r['gap_mm'])>0,
             'single':lambda r: float(r['gap_mm'])==0}
    subsets.update({f'gap{g}':(lambda r, g=g: float(r['gap_mm'])==g) for g in (2,4,8)})
    for label, subset in subsets.items():
        if not any(subset(r) for r in rows): continue
        comparisons[label]=[]
        for budget in config['budgets']:
            for sharing in ('independent','shared'):
                left=f'reassociate_b{budget}_{sharing}'
                for right in (f'local_multistart_b{budget}_{sharing}', f'original_b0_{sharing}'):
                    comparisons[label].append(compare(rows,left,right,subset))
            comparisons[label].append(compare(rows,f'reassociate_b{budget}_shared',
                                                f'reassociate_b{budget}_independent',subset))
    with (run/'PAIRED_COMPARISONS.json').open('x') as stream:
        json.dump(comparisons,stream,indent=2,allow_nan=False)
    print(json.dumps(comparisons['all'],indent=2))

if __name__=='__main__':main()

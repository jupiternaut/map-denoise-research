"""Read-only endpoint decomposition; no new solver or selector is imported."""
from fractions import Fraction as Q
from pathlib import Path
import csv
import json
import statistics

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'outputs/ablation-v1'
PRIOR = ROOT.parent / 'lovelace-shared-appearance-20261011T052300Z/outputs/confirmation-v1'
DEST = ROOT / 'outputs/analysis-v1'
METHODS = ('keep', 'channel511', 'global_gray511', 'pixel_gray511', 'view_gray511', 'global_rgb511')


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def table(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def endpoint(rows):
    errors = [Q(r['ae_mm']) for r in rows]
    mse = sum((v*v for v in errors), Q(0)) / len(rows)
    mae = sum(errors, Q(0)) / len(rows)
    initial_mae = sum((Q(r['initial_ae_mm']) for r in rows), Q(0)) / len(rows)
    initial_mse = sum((Q(r['initial_ae_mm'])**2 for r in rows), Q(0)) / len(rows)
    return {'n': len(rows), 'improved': sum(v < Q(r['initial_ae_mm']) for v,r in zip(errors,rows)),
            'same': sum(v == Q(r['initial_ae_mm']) for v,r in zip(errors,rows)),
            'harm': sum(v > Q(r['initial_ae_mm']) for v,r in zip(errors,rows)),
            'mae_exact_mm': str(mae), 'mae_mm': float(mae), 'mse_exact_mm2': str(mse), 'mse_mm2': float(mse),
            'mae_reduction_percent_vs_keep': None if initial_mae == 0 else float(100*(1-mae/initial_mae)),
            'mse_reduction_percent_vs_keep': None if initial_mse == 0 else float(100*(1-mse/initial_mse)),
            'midpoint': sum(r['status'] == 'MOVE' and r.get('action_branch') == 'midpoint' for r in rows),
            'projection': sum(r['status'] == 'MOVE' and r.get('action_branch') == 'projection' for r in rows)}


def paired(reference, compared, label):
    a = {r['task_id']:r for r in reference}
    b = {r['task_id']:r for r in compared}
    assert a.keys() == b.keys()
    differences = [Q(b[k]['ae_mm'])-Q(a[k]['ae_mm']) for k in a]
    mse_diff = [Q(b[k]['se_mm2'])-Q(a[k]['se_mm2']) for k in a]
    return {'comparison':label, 'n':len(a), 'compared_better':sum(v<0 for v in differences),
            'same_error':sum(v==0 for v in differences), 'compared_worse':sum(v>0 for v in differences),
            'identical_output':sum(a[k]['output']==b[k]['output'] for k in a),
            'mae_delta_exact_mm':str(sum(differences,Q(0))/len(a)),
            'mae_delta_mm':float(sum(differences,Q(0))/len(a)),
            'mse_delta_exact_mm2':str(sum(mse_diff,Q(0))/len(a)),
            'mse_delta_mm2':float(sum(mse_diff,Q(0))/len(a))}


def main():
    assert load(RUN/'verification/VERIFY.json')['status']=='PASS'
    rows, old_rows = load(RUN/'evaluation/ROWS.json'), load(PRIOR/'evaluation/ROWS.json')
    decisions = {(r['task_id'],r['method']):r for r in load(RUN/'decisions.json')}
    assert len(rows)==2916 and len(old_rows)==1944 and len(decisions)==2916
    rows = [{**r,'action_branch':decisions[r['task_id'],r['method']].get('branch')} for r in rows]
    small=[r for r in rows if r['state_group']=='small']
    cases=list(dict.fromkeys(r['case'] for r in small))
    seeds=sorted({r['seed'] for r in small})
    overall=[{'method':m,**endpoint([r for r in small if r['method']==m])} for m in METHODS]
    by_case=[{'case':c,'method':m,**endpoint([r for r in small if r['method']==m and r['case']==c])}
             for c in cases for m in METHODS]
    by_seed=[{'seed':s,'method':m,**endpoint([r for r in small if r['method']==m and r['seed']==s])}
             for s in seeds for m in METHODS]
    seed_stats=[]
    for m in METHODS:
        values=[r for r in by_seed if r['method']==m]
        seed_stats.append({'method':m,'noise_seeds':len(seeds),'mae_mean_mm':statistics.mean(r['mae_mm'] for r in values),
                           'mae_sd_across_seeds_mm':statistics.stdev(r['mae_mm'] for r in values),
                           'mse_mean_mm2':statistics.mean(r['mse_mm2'] for r in values),
                           'mse_sd_across_seeds_mm2':statistics.stdev(r['mse_mm2'] for r in values)})
    comparisons=[]
    for m in ('channel511','pixel_gray511','view_gray511','global_rgb511'):
        for condition in ('all','informative',*cases):
            selected=lambda r: condition=='all' or (condition=='informative' and r['case']!='equal_full_negative') or r['case']==condition
            ref=[r for r in small if r['method']==m and selected(r)]
            new=[r for r in small if r['method']=='global_gray511' and selected(r)]
            comparisons.append({'condition':condition,**paired(ref,new,m+' -> global_gray511')})
    budget=[]
    for old_name,new_name in (('channelwise_union','channel511'),('shared_union','global_gray511')):
        for condition in ('all','informative',*cases):
            selected=lambda r: condition=='all' or (condition=='informative' and r['case']!='equal_full_negative') or r['case']==condition
            ref=[r for r in old_rows if r['method']==old_name and r['state_group']=='small' and selected(r)]
            new=[r for r in small if r['method']==new_name and selected(r)]
            budget.append({'condition':condition,**paired(ref,new,old_name+'255 -> '+new_name)})
    coverage=load(RUN/'evaluation/COVERAGE.json')
    widths=[]
    for m in METHODS:
        for condition in ('all','informative','equal_full_negative'):
            subset=[r for r in coverage if r['method']==m and (condition=='all' or
                    condition=='informative' and r['case']!='equal_full_negative' or r['case']==condition)]
            w=[Q(r['hull_width_mm']) for r in subset if r['hull_width_mm'] is not None]
            widths.append({'method':m,'condition':condition,'observations':len(subset),'covered':sum(r['covered'] for r in subset),
                           'mean_hull_width_mm':float(sum(w,Q(0))/len(w)),
                           'median_hull_width_mm':float(statistics.median(w)),
                           'max_hull_width_mm':float(max(w))})
    DEST.mkdir(parents=True,exist_ok=False)
    for name,value in (('OVERALL',overall),('BY_CASE',by_case),('BY_SEED',by_seed),('SEED_STATS',seed_stats),
                       ('PAIRED',comparisons),('BUDGET',budget),('WIDTHS',widths)):
        save(DEST/(name+'.json'),value)
        table(DEST/(name+'.csv'),value)
    save(DEST/'ANALYSIS_SCOPE.json',{'source_run':str(RUN),'prior_frozen_run':str(PRIOR),
          'same_observations':54,'small_states_per_arm':432,'correct_states_per_arm':54,
          'noise_seed_count':3,'uncertainty_note':'SD across repeated noise seeds on fixed six templates; not an independent-scene confidence interval',
          'generated_after_run':True,'new_method_runs':0,'new_thresholds':0})
    print(json.dumps({'overall':overall,'seed_stats':seed_stats,'budget_all':[r for r in budget if r['condition']=='all'],
                      'paired_all':[r for r in comparisons if r['condition']=='all']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()

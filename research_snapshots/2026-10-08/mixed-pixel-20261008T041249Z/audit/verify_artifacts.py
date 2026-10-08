"""Read-only independent artifact checks; never imports experiment code.

Only audit/VERIFICATION_v2.json is created. Does not generate data or rerun inference.
"""
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import socket
import numpy as np

ROOT = Path('/srv/slam-research/grf/map-denoise/runs/mixed-pixel-20261008T041249Z')
assert socket.gethostname() == 'liekkas'
assert Path(__file__).resolve().parent == ROOT / 'audit'
checked = {}
checks = []

def digest(path):
    path = Path(path)
    value = hashlib.sha256(path.read_bytes()).hexdigest()
    checked[str(path)] = value
    return value

def read(path):
    path = Path(path)
    digest(path)
    return json.loads(path.read_text())

def record(name, condition, detail=None):
    checks.append(dict(name=name, passed=bool(condition), detail=detail))

def close(a, b):
    if isinstance(a, dict):
        return set(a) == set(b) and all(close(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
    if isinstance(a, (float, int)) and not isinstance(a, bool):
        return b is not None and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-9)
    return a == b

def summary(rows):
    errors = [r['error'] for r in rows]
    initial = [r['initial_error'] for r in rows]
    return dict(n=len(rows), mae=sum(errors)/len(rows),
                mse=sum(x*x for x in errors)/len(rows),
                initial_mae=sum(initial)/len(rows),
                improved=sum(e < e0-1e-9 for e,e0 in zip(errors,initial)),
                worsened=sum(e > e0+1e-9 for e,e0 in zip(errors,initial)),
                unchanged=sum(abs(e-e0) <= 1e-9 for e,e0 in zip(errors,initial)),
                moved=sum(abs(r['selected_depth']-r['initial_depth'])>1e-9 for r in rows),
                empty_support=sum(not r['intervals'] for r in rows))

def comparison(rows, first, second):
    left={(r['id'],r['initial_depth']):r for r in rows if r['arm']==first}
    right={(r['id'],r['initial_depth']):r for r in rows if r['arm']==second}
    assert set(left)==set(right)
    delta=[right[k]['error']-left[k]['error'] for k in left]
    return dict(n=len(delta), mae_gain=sum(delta)/len(delta),
                better=sum(d>1e-9 for d in delta), worse=sum(d < -1e-9 for d in delta),
                same=sum(abs(d)<=1e-9 for d in delta))

def support_intervals(grid, accepted, padding):
    indices=np.flatnonzero(accepted)
    groups=[]
    for index in indices:
        if not groups or index != groups[-1][-1]+1:
            groups.append([int(index)])
        else:
            groups[-1].append(int(index))
    half=(grid[1]-grid[0])/2 if len(grid)>1 else 0.
    expanded=[[float(max(grid[0],grid[g[0]]-half-padding)),
               float(min(grid[-1],grid[g[-1]]+half+padding))] for g in groups]
    merged=[]
    for left,right in expanded:
        if merged and left <= merged[-1][1]+1e-9:
            merged[-1][1]=max(merged[-1][1],right)
        else:
            merged.append([left,right])
    return merged

def independent_rank(data, true_depth, candidates):
    if 'loss' not in data:
        return None
    values=[float(data['loss'][min(range(len(data['grid'])),key=lambda i:abs(data['grid'][i]-z))]) for z in candidates]
    if not all(math.isfinite(v) for v in values):
        return dict(valid=False)
    correct=min(range(len(candidates)),key=lambda i:abs(candidates[i]-true_depth))
    wrong=[v for i,v in enumerate(values) if i!=correct]
    margin=min(wrong)-values[correct]
    errors=[abs(candidates[i]-true_depth) for i,v in enumerate(values) if abs(v-min(values))<=1e-9]
    return dict(valid=True,flat=max(values)-min(values)<=1e-9,margin=margin,
                correct_strict_best=margin>1e-9,correct_tied_best=abs(margin)<=1e-9,
                wrong_strictly_better=sum(v<values[correct]-1e-9 for v in wrong),
                diagnostic_argmin_error_min=min(errors),diagnostic_argmin_error_max=max(errors),
                candidate_losses=values,whole_grid_range=float(np.ptp(data['loss'])))

lock=read(ROOT/'LOCK.json')
for category in ('sources','inputs','historical_hashes'):
    bad=[p for p,h in lock[category].items() if digest(p)!=h]
    record(category+'_hashes_match', not bad, dict(count=len(lock[category]), mismatches=bad))
e0=read(ROOT/'e0/RESULTS.json')
integration=read(ROOT/'e0/INTEGRATION.json')
legacy=read(ROOT/'legacy_replay/RESULTS.json')
record('E0_result_and_source_hash', e0['passed'] and digest(ROOT/'e0/RESULTS.json')==lock['e0_sha256'] and digest(ROOT/'e0.py')==e0['source_sha256'])
record('E0_integration_sources', integration['passed'] and all(digest(p)==h for p,h in integration['source_sha256'].items()))
record('legacy_30_replay_aggregate', legacy['n']==len(legacy['rows'])==30 and close(legacy['mae'],sum(r['absolute_error'] for r in legacy['rows'])/30) and legacy['max_score_difference']<1e-12 and all(digest(p)==h for p,h in legacy['input_hashes'].items()))
method=read(ROOT/'METHOD.json')
expected_grid=np.arange(method['grid'][0],method['grid'][1]+.1,method['grid'][2])
inputs=read(ROOT/'data/observed/inputs.json')
truth=read(ROOT/'data/truth/metadata.json')
oracle=read(ROOT/'data/oracle/manifest.json')
truthmap={r['id']:r for r in truth}
record('manifest_ids_match', {r['id'] for r in inputs}==set(truthmap)=={r['id'] for r in oracle})
record('expected_design_counts', len(inputs)==36 and len({r['group_id'] for r in truth})==24 and sum(r['background_pair'] for r in truth)==12)
tensor_hashes=defaultdict(list)
for row in inputs:
    record('observed_manifest_hash_'+row['id'], digest(row['image_file'])==row['sha256'])
    record('observed_metadata_schema_'+row['id'], set(row)=={'id','cameras','image_file','sha256'})
    with np.load(row['image_file'],allow_pickle=False) as f:
        record('observed_npz_schema_'+row['id'], f.files==['images'] and f['images'].shape==(3,128,128))
        tensor_hashes[hashlib.sha256(f['images'].tobytes()).hexdigest()].append(row['id'])
        if truthmap[row['id']]['mechanism']=='flat_equal':
            record('equal_color_exact_constant_'+row['id'], np.ptp(f['images'])==0)
for row in oracle:
    record('oracle_manifest_hash_'+row['id'], digest(row['aux_file'])==row['sha256'])
    with np.load(row['aux_file'],allow_pickle=False) as f:
        record('oracle_excludes_target_answer_'+row['id'], not set(f.files)&{'true_depth','target_depth','source_alpha','seed','phases','mechanism'})

stages={}
curve_data={}
for stage in ('ordinary_stage','oracle_stage'):
    seal=read(ROOT/stage/'SEAL.json')
    mismatches=[p for p,h in seal['files'].items() if digest(p)!=h]
    record(stage+'_seal', not mismatches, dict(count=len(seal['files']),mismatches=mismatches))
    obs=read(ROOT/stage/'OBSERVATIONS.json')
    stages[stage]=obs
    expected_source=hashlib.sha256(json.dumps(lock['sources'],sort_keys=True).encode()).hexdigest()
    record(stage+'_source_link', obs['source_lock_sha256']==expected_source)
    forbidden=[p for p in obs['read_log'] if '/data/truth/' in p or (stage=='ordinary_stage' and '/data/oracle/' in p)]
    record(stage+'_guard_log_excludes_privileged_content', not forbidden, forbidden)
    for row in obs['rows']:
        path=row['curve_file']
        with np.load(path,allow_pickle=False) as f:
            data={k:f[k].copy() for k in f.files}
        curve_data[path]=data
        record('public_grid_'+Path(path).name,np.array_equal(data['grid'],expected_grid))
        accepted=data['accepted']
        if row['arm']=='N':
            recalculated=np.all(np.isfinite(data['scores'])&(data['scores']>=.6),axis=0)
        else:
            pooled=np.average(data['fold_loss'],axis=0,weights=data['pixel_counts'])
            record('fold_weighting_'+Path(path).name,np.allclose(pooled,data['loss'],equal_nan=True,rtol=1e-12,atol=1e-12))
            if row['scale_valid']:
                sigma2=np.average(data['sigma_values']**2,weights=data['pixel_counts'])
                normalized=data['loss']/sigma2
                record('normalization_'+Path(path).name,np.allclose(normalized,data['normalized_loss'],equal_nan=True,rtol=1e-12,atol=1e-12))
            recalculated=np.zeros(len(expected_grid),dtype=bool)
            if row['reason']=='ok':
                norm=data['normalized_loss']
                recalculated=(norm<=method['residual_abs_max'])&(norm-np.min(norm)<=method['residual_excess_max'])
        record('accepted_'+Path(path).name,np.array_equal(accepted,recalculated))
        record('intervals_'+Path(path).name,close(support_intervals(data['grid'],accepted,method['padding_mm']),row['intervals']))
for first,second in zip(stages['ordinary_stage']['auxiliary'],stages['oracle_stage']['auxiliary']):
    record('shared_estimated_scale_'+first['id'], first['id']==second['id'] and first['sigma_values']==second['sigma_values'] and first['sigma_valid']==second['sigma_valid'])
for identifier in truthmap:
    modelrows=[r for obs in stages.values() for r in obs['rows'] if r['id']==identifier and r['arm']!='N']
    counts={tuple(r['pixel_counts']) for r in modelrows}
    record('same_pixel_denominators_'+identifier,len(counts)==1)

decision_seal=read(ROOT/'decisions/SEAL.json')
record('decision_hash',digest(ROOT/'decisions/PREDICTIONS.json')==decision_seal['sha256'])
for stage,h in decision_seal['observation_seals'].items():
    record('decision_'+stage+'_link',digest(ROOT/stage/'SEAL.json')==h)
predictions=read(ROOT/'decisions/PREDICTIONS.json')
rows=read(ROOT/'evaluation/ROWS.json')
result=read(ROOT/'evaluation/SUMMARY.json')
evaluation_seal=read(ROOT/'evaluation/SEAL.json')
record('evaluation_seal',all(digest(p)==h for p,h in evaluation_seal['files'].items()))
expected_keys={(r['id'],a,z) for r in inputs for a in method['arms'] for z in method['incumbents']}
prediction_keys=[(r['id'],r['arm'],r['initial_depth']) for r in predictions]
row_keys=[(r['id'],r['arm'],r['initial_depth']) for r in rows]
record('complete_unique_648_predictions',len(prediction_keys)==len(set(prediction_keys))==648 and set(prediction_keys)==expected_keys)
record('complete_unique_648_evaluation_rows',len(row_keys)==len(set(row_keys))==648 and set(row_keys)==expected_keys)
rowmap={(r['id'],r['arm'],r['initial_depth']):r for r in rows}
for p in predictions:
    initial=p['initial_depth'];intervals=p['intervals'];width=sum(b-a for a,b in intervals)
    target=sum((b-a)*(a+b)/2 for a,b in intervals)/width if width else None
    if target is None:
        chosen=initial;gain=0.
    else:
        candidate=min(enumerate(method['candidates']),key=lambda iz:(iz[1]-target)**2)[1]
        rawgain=(initial-target)**2-(candidate-target)**2
        chosen=candidate if rawgain>0 else initial;gain=max(0.,rawgain)
    record('P_'+str((p['id'],p['arm'],initial)),close(chosen,p['selected_depth']) and close(target,p['support_mean']) and close(gain,p['estimated_squared_gain']))
    r=rowmap[p['id'],p['arm'],initial];gt=truthmap[p['id']]['true_depth']
    record('geometry_'+str((p['id'],p['arm'],initial)),close(r['error'],abs(chosen-gt)) and close(r['initial_error'],abs(initial-gt)) and all(close(value,r[k]) for k,value in p.items()))
    if 'curve_file' in p:
        rank=independent_rank(curve_data[p['curve_file']],gt,method['candidates'])
        record('ranking_'+str((p['id'],p['arm'],initial)),close(rank,r['ranking']))

recomputed={}
for initial in method['incumbents']:
    subset=[r for r in rows if r['initial_depth']==initial]
    one={a:summary([r for r in subset if r['arm']==a]) for a in method['arms']}
    one['ED_vs_N']=comparison(subset,'ED','N');one['ED_vs_EF']=comparison(subset,'ED','EF')
    recomputed[str(int(initial))]=one
record('all_summary_metrics',close(recomputed,result['summary']))
for mechanism,expected in result['by_mechanism'].items():
    actual={str(int(z)):{a:summary([r for r in rows if r['mechanism']==mechanism and r['initial_depth']==z and r['arm']==a]) for a in method['arms']} for z in method['incumbents']}
    record('by_mechanism_'+mechanism,close(actual,expected))
correctN={(r['id'],r['initial_depth']) for r in rows if r['arm']=='N' and r['error']<=1e-9}
retention={a:dict(n=len(correctN),retained=sum(r['error']<=1e-9 for r in rows if r['arm']==a and (r['id'],r['initial_depth']) in correctN)) for a in method['arms']}
record('full9_success_retention',close(retention,result['full9_success_retention']))
reasons={a:dict(Counter(r.get('reason','baseline_or_keep') for r in rows if r['arm']==a)) for a in method['arms']}
record('reason_accounting',close(reasons,result['reasons']))
ranking={}
for mechanism in result['by_mechanism']:
    for arm in ('EF','ED','OF','OD'):
        one=[r for r in rows if r['mechanism']==mechanism and r['arm']==arm and (not arm.endswith('D') or r['initial_depth']==600)]
        valid=[independent_rank(curve_data[r['curve_file']],r['true_depth'],method['candidates']) for r in one]
        valid=[r for r in valid if r and r['valid']]
        ranking[mechanism+'/'+arm]=dict(n=len(one),valid=len(valid),strict_correct=sum(r['correct_strict_best'] for r in valid),
            flat=sum(r['flat'] for r in valid),tied_correct=sum(r['correct_tied_best'] for r in valid),
            mean_margin=sum(r['margin'] for r in valid)/len(valid) if valid else None,
            mean_argmin_error_min=sum(r['diagnostic_argmin_error_min'] for r in valid)/len(valid) if valid else None,
            mean_argmin_error_max=sum(r['diagnostic_argmin_error_max'] for r in valid)/len(valid) if valid else None)
record('all_ranking_aggregates',close(ranking,result['ranking']))
nullrows=[r for r in rows if r['mechanism']=='flat_equal' and r['arm'] in ('EF','ED','OF','OD')]
dynamic=ranking['flat_contrast/OD'];fixed=ranking['flat_contrast/OF']
directions=[recomputed[str(z)]['ED_vs_N']['mae_gain'] for z in (540,660)]
gate=dict(oracle_boundary_separation=dynamic['valid']>0 and dynamic['mean_margin']>0 and dynamic['mean_margin']>fixed['mean_margin'],
    equal_color_no_information=all(r['ranking']['valid'] and r['ranking']['flat'] for r in nullrows),
    estimated_correct_input_no_harm=recomputed['600']['ED']['worsened']==0,
    estimated_ncc_each_direction_nonworse=min(directions)>=-1e-9,
    estimated_ncc_some_direction_better=max(directions)>1e-9,
    estimated_dynamic_beats_fixed=comparison([r for r in rows if r['initial_depth']!=600],'ED','EF')['mae_gain']>1e-9)
gate['proceed_E2']=all(gate.values())
record('E1_gate_recomputed',close(gate,result['gate']))
with (ROOT/'evaluation/RESULTS.csv').open(newline='') as stream:
    csvrows=list(csv.DictReader(stream))
record('csv_has_648_rows',len(csvrows)==648)
for c in csvrows:
    r=rowmap[c['id'],c['arm'],float(c['initial_depth'])]
    record('csv_row_'+str((r['id'],r['arm'],r['initial_depth'])), all(close(float(c[k]),r[k]) for k in ('initial_depth','true_depth','selected_depth','initial_error','error')))

payload=dict(created=datetime.now(timezone.utc).isoformat(),host=socket.gethostname(),
    verifier='independent audit-only arithmetic and hash checks; no experiment code imported',
    passed=all(c['passed'] for c in checks),checks_run=len(checks),failures=[c for c in checks if not c['passed']],
    recomputed_summary=recomputed,recomputed_ranking=ranking,recomputed_gate=gate,
    verifier_correction='v1 omitted historical half-grid expansion and clipping/merging in support_intervals; v2 independently implements the preserved historical rule. v1 evidence retained unchanged.',
    physical_unique_observed_tensors=len(tensor_hashes),
    duplicate_observed_tensor_groups=[ids for ids in tensor_hashes.values() if len(ids)>1],
    verified_hashes=checked,checks=checks)
with (ROOT/'audit/VERIFICATION_v2.json').open('x') as stream:
    json.dump(payload,stream,indent=2,allow_nan=False)
print(json.dumps({k:payload[k] for k in ('passed','checks_run','failures','physical_unique_observed_tensors','duplicate_observed_tensor_groups')},indent=2))

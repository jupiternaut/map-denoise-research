"""Read sealed predictions first, then producer truth; no method selection here."""
from collections import Counter, defaultdict
import csv
import io
import numpy as np
from run import ROOT,load,sha,dump,verify_lock,now

def stats(rows):
    if not rows:return None
    e=np.asarray([r['error'] for r in rows]);e0=np.asarray([r['initial_error'] for r in rows])
    tol=1e-9
    return dict(n=len(rows),mae=float(e.mean()),mse=float(np.mean(e*e)),
        initial_mae=float(e0.mean()),improved=int(np.sum(e<e0-tol)),
        worsened=int(np.sum(e>e0+tol)),unchanged=int(np.sum(abs(e-e0)<=tol)),
        moved=sum(abs(r['selected_depth']-r['initial_depth'])>tol for r in rows),
        empty_support=sum(not r['intervals'] for r in rows))

def compare(rows,a,b):
    aa={(r['id'],r['initial_depth']):r for r in rows if r['arm']==a}
    bb={(r['id'],r['initial_depth']):r for r in rows if r['arm']==b}
    keys=sorted(aa.keys()&bb.keys())
    if not keys:return None
    delta=np.array([bb[k]['error']-aa[k]['error'] for k in keys])
    return dict(n=len(keys),mae_gain=float(delta.mean()),better=int(sum(delta>1e-9)),
                worse=int(sum(delta< -1e-9)),same=int(sum(abs(delta)<=1e-9)))

def rank_curve(path,true_depth,candidates):
    with np.load(path) as f:
        if 'loss' not in f:return None
        grid=f['grid'];loss=f['loss']
    v=np.array([loss[np.argmin(abs(grid-z))] for z in candidates])
    t=int(np.argmin(abs(np.asarray(candidates)-true_depth)))
    if not np.isfinite(v).all():return dict(valid=False)
    wrong=np.delete(v,t);margin=float(wrong.min()-v[t])
    best=np.flatnonzero(abs(v-v.min())<=1e-9)
    best_error=[abs(candidates[i]-true_depth) for i in best]
    return dict(valid=True,flat=bool(np.ptp(v)<=1e-9),margin=margin,
        correct_strict_best=margin>1e-9,correct_tied_best=abs(margin)<=1e-9,
        wrong_strictly_better=int(sum(wrong<v[t]-1e-9)),
        diagnostic_argmin_error_min=float(min(best_error)),
        diagnostic_argmin_error_max=float(max(best_error)),
        candidate_losses=v.tolist(),whole_grid_range=float(np.ptp(loss)))

def evaluate():
    lock=verify_lock();params=load(ROOT/'METHOD.json')
    ds=load(ROOT/'decisions/SEAL.json')
    assert sha(ROOT/'decisions/PREDICTIONS.json')==ds['sha256']
    for name,h in ds['observation_seals'].items():
        assert sha(ROOT/name/'SEAL.json')==h
        for p,h2 in load(ROOT/name/'SEAL.json')['files'].items():assert sha(p)==h2,p
    predictions=load(ROOT/'decisions/PREDICTIONS.json')
    truth={r['id']:r for r in load(ROOT/'data/truth/metadata.json')}
    assert len(truth)==36 and len(predictions)==648
    rows=[];curve_cache={}
    for p in predictions:
        t=truth[p['id']];z=t['true_depth'];e=abs(p['selected_depth']-z)
        r={**p,'true_depth':z,'group_id':t['group_id'],'mechanism':t['mechanism'],
           'background_pair':t['background_pair'],'initial_offset':p['initial_depth']-z,
           'error':e,'initial_error':abs(p['initial_depth']-z)}
        if 'curve_file' in p:
            key=(p['curve_file'],z)
            if key not in curve_cache:curve_cache[key]=rank_curve(key[0],z,params['candidates'])
            r['ranking']=curve_cache[key]
        rows.append(r)
    arms=params['arms'];summary={};mechanisms={}
    for inc in params['incumbents']:
        rr=[r for r in rows if r['initial_depth']==inc]
        summary[str(int(inc))]={a:stats([r for r in rr if r['arm']==a]) for a in arms}
        summary[str(int(inc))]['ED_vs_N']=compare(rr,'ED','N')
        summary[str(int(inc))]['ED_vs_EF']=compare(rr,'ED','EF')
    for mechanism in sorted({t['mechanism'] for t in truth.values()}):
        mechanisms[mechanism]={str(int(inc)):{a:stats([r for r in rows if r['mechanism']==mechanism and r['initial_depth']==inc and r['arm']==a]) for a in arms} for inc in params['incumbents']}
    ranking={}
    for mechanism in mechanisms:
        for a in ('EF','ED','OF','OD'):
            rr=[r for r in rows if r['mechanism']==mechanism and r['arm']==a]
            # Dynamic curves reused across incumbents: count each image once.
            if a.endswith('D'):rr=[r for r in rr if r['initial_depth']==600]
            kk=[r['ranking'] for r in rr if r.get('ranking') and r['ranking']['valid']]
            ranking[mechanism+'/'+a]=dict(n=len(rr),valid=len(kk),strict_correct=sum(k['correct_strict_best'] for k in kk),
                flat=sum(k['flat'] for k in kk),tied_correct=sum(k['correct_tied_best'] for k in kk),
                mean_margin=float(np.mean([k['margin'] for k in kk])) if kk else None,
                mean_argmin_error_min=float(np.mean([k['diagnostic_argmin_error_min'] for k in kk])) if kk else None,
                mean_argmin_error_max=float(np.mean([k['diagnostic_argmin_error_max'] for k in kk])) if kk else None)
    # Paired backgrounds are dependent, report changes without doubling groups.
    grouped=defaultdict(dict)
    for r in rows:grouped[(r['group_id'],r['arm'],r['initial_depth'])][r['background_pair']]=r
    pairs=[]
    for (g,a,inc),rr in sorted(grouped.items()):
        if 0 in rr and 1 in rr:pairs.append(dict(group_id=g,arm=a,initial_depth=inc,
            selection_shift=rr[1]['selected_depth']-rr[0]['selected_depth'],error_shift=rr[1]['error']-rr[0]['error']))
    correctN={(r['id'],r['initial_depth']) for r in rows if r['arm']=='N' and r['error']<=1e-9}
    retention={a:dict(n=len(correctN),retained=sum(r['error']<=1e-9 for r in rows if r['arm']==a and (r['id'],r['initial_depth']) in correctN)) for a in arms}
    # E1 operational gate: pre-result method constant, no threshold tuning.
    odds=ranking['flat_contrast/OD'];fixed=ranking['flat_contrast/OF']
    c1=odds['valid']>0 and odds['mean_margin']>0 and odds['mean_margin']>fixed['mean_margin']
    null_ok=all(r.get('ranking') is not None and r['ranking']['valid'] and r['ranking']['flat'] for r in rows if r['mechanism']=='flat_equal' and r['arm'] in ('ED','EF','OD','OF'))
    safe=summary['600']['ED']['worsened']==0
    directional=[summary[str(x)]['ED_vs_N']['mae_gain'] for x in (540,660)]
    ed_inc=compare([r for r in rows if r['initial_depth']!=600],'ED','EF')['mae_gain']>1e-9
    gate=dict(oracle_boundary_separation=bool(c1),equal_color_no_information=bool(null_ok),
        estimated_correct_input_no_harm=bool(safe),estimated_ncc_each_direction_nonworse=bool(min(directional)>=-1e-9),
        estimated_ncc_some_direction_better=bool(max(directional)>1e-9),estimated_dynamic_beats_fixed=bool(ed_inc))
    gate['proceed_E2']=all(gate.values())
    reasons={a:dict(Counter(r.get('reason','baseline_or_keep') for r in rows if r['arm']==a)) for a in arms}
    historical_mismatches=[p for p,h in lock['historical_hashes'].items() if sha(p)!=h]
    assert not historical_mismatches
    result=dict(created=now(),evaluation_type='simulation_only',n_worlds=len(truth),n_groups=len(set(t['group_id'] for t in truth.values())),
        n_decisions=len(rows),scope='E1 development; background pairs and folds are not independent samples',
        summary=summary,by_mechanism=mechanisms,ranking=ranking,full9_success_retention=retention,
        reasons=reasons,gate=gate,historical_hashes_unchanged=True,prediction_sha256=ds['sha256'])
    dump(ROOT/'evaluation/ROWS.json',rows);dump(ROOT/'evaluation/SUMMARY.json',result)
    dump(ROOT/'evaluation/BACKGROUND_PAIRS.json',pairs)
    cols=['id','group_id','mechanism','background_pair','arm','initial_depth','true_depth','selected_depth','initial_error','error','reason']
    s=io.StringIO();w=csv.DictWriter(s,fieldnames=cols);w.writeheader()
    w.writerows({k:r.get(k,'') for k in cols} for r in rows)
    with (ROOT/'evaluation/RESULTS.csv').open('x') as f:f.write(s.getvalue())
    dump(ROOT/'evaluation/SEAL.json',dict(created=now(),files={str(p):sha(p) for p in (ROOT/'evaluation').iterdir() if p.is_file()}))
    print(__import__('json').dumps(dict(summary=summary,gate=gate)))

if __name__=='__main__':evaluate()

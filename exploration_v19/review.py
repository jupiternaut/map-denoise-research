"""Representation-aware audit and descriptive paired summaries, no tuning."""
from pathlib import Path
import sys,json,argparse
import numpy as np
from common import HERE,ROOT,OLD,sha,save,load,unpack,world_support,v14,import_path
check_lock=import_path('v19_shared_run',HERE/'run.py').check_lock

FIELDS=('surface_mae_mm','balanced_source_mae_mm','matched_rms_mm','risk_rms_mm','source_gap_error_mm',
        'fitted_gap_error_mm','vertical_w1_mm','upper_mass_error','cross_midplane_mass','posterior_brier','coverage_1mm')

def aggregate(rows):
    out={}
    for kind in ('statistical','bridge','sigma_stress'):
        for group in ('dual','single','dual_selected_two'):
            selected=[r for r in rows if r['kind']==kind and ((r['gap']>0)==(group!='single')) and
                      (group!='dual_selected_two' or r.get('model_k')==2)]
            if not selected:continue
            key=kind+'_'+group;out[key]={}
            for method in sorted({r['method'] for r in selected}):
                rr=[r for r in selected if r['method']==method];ok=[r for r in rr if r['status']=='OK']
                result=dict(n=len(rr),failures=len(rr)-len(ok),seeds=len({r['seed'] for r in rr}))
                for field in FIELDS:
                    vals=[r[field] for r in ok if r.get(field) is not None]
                    if vals:result[field]=float(np.mean(vals))
                for flag in ('dual_missed','single_false_split'):
                    if any(flag in r for r in ok):result[flag]=sum(r.get(flag,0) for r in ok)
                out[key][method]=result
    return out

def paired(rows):
    out={}
    for kind in ('statistical','bridge','sigma_stress'):
        for candidate,baseline in [('a_crossfit_gate_map','b_map'),('a_crossfit_gate_map','a_insample_gate_map'),
                                   ('b_quota_map','b_map'),('b_weighted_measure','b_map'),('b_mean','b_map')]:
            aa={r['case']:r for r in rows if r['kind']==kind and r['gap']>0 and r['method']==candidate and r['status']=='OK'}
            bb={r['case']:r for r in rows if r['kind']==kind and r['gap']>0 and r['method']==baseline and r['status']=='OK'}
            keys=sorted(aa.keys()&bb.keys())
            if not keys:continue
            res={'cases':len(keys),'direction':'candidate minus baseline; negative favors candidate'}
            for field in FIELDS:
                if not all(field in aa[k] and field in bb[k] for k in keys):continue
                delta=np.array([aa[k][field]-bb[k][field] for k in keys])
                seeds=sorted({aa[k]['seed'] for k in keys})
                seedmeans=[float(np.mean([aa[k][field]-bb[k][field] for k in keys if aa[k]['seed']==s])) for s in seeds]
                res[field]=dict(mean=float(delta.mean()),improved=int(np.sum(delta < -1e-9)),worse=int(np.sum(delta>1e-9)),
                               unchanged=int(np.sum(abs(delta)<=1e-9)),seed_means=dict(zip(map(str,seeds),seedmeans)))
            out[f'{kind}:{candidate}-vs-{baseline}']=res
    return out

def audit(dest):
    check_lock(dest);checked=set();total=0;max_replay=0.;max_moment=0.;max_source_gap=0.
    oldrecords=json.loads((OLD/'bridge_SEALED_BEFORE_GT.json').read_text())
    oldmap={r['case']:r for r in oldrecords if r['method']=='retained_map'}
    for phase in ('development','confirmation'):
        jobs={j['case']:j for j in json.loads((dest/f'{phase}_JOBS.json').read_text())}
        for j in jobs.values():
            for key in ('input','evaluation','cache'):
                assert sha(j[key])==j[key+'_sha256'];checked.add(j[key])
        b_rows=json.loads((dest/f'{phase}_b_ROWS.json').read_text())
        rowmap={(r['case'],r['method']):r for r in b_rows}
        for track in ('a','b'):
            records=json.loads((dest/f'{phase}_{track}_SEALED.json').read_text())
            for r in records:
                assert r['status']=='OK',r
                assert sha(r['output'])==r['output_sha256'];checked.add(r['output']);total+=1
                j=jobs[r['case']];c,x,y,base=unpack(j);m=json.loads(Path(r['output']).read_text())
                points,w,isweighted=world_support(load(j['input']),c,m)
                np.testing.assert_allclose(w.sum(1),1.,rtol=0,atol=1e-10)
                assert np.isfinite(points).all() and (w>=0).all()
                if m.get('identity'):
                    np.testing.assert_array_equal(points[:,0],load(j['input'])['xyz_world']*1000.)
                    continue
                assert int(m['k'])==int(base['k'])
                levels=np.asarray(m['means'])[None,:]+(x[:,1:]@m['slope'])[:,None]
                if track=='a' or r['method'] in ('b_map','b_quota_map','b_posterior_draw'):
                    np.testing.assert_allclose(m['prediction'],levels[np.arange(len(y)),m['groups']],rtol=0,atol=1e-10)
                if track=='b':
                    for field in ('means','slope','posterior'):np.testing.assert_array_equal(m[field],base[field])
                if r['method']=='b_weighted_measure':
                    meanfile=dest/'outputs'/f'{j["case"]}__b_mean.json';mean=json.loads(meanfile.read_text())
                    delta=float(np.max(abs(np.sum(np.array(m['support_weights'])*m['support_heights'],axis=1)-mean['prediction'])))
                    max_moment=max(max_moment,delta);assert delta<1e-10
                    if j['gap']:
                        a=rowmap[(j['case'],'b_weighted_measure')]['source_gap_error_mm'];b=rowmap[(j['case'],'b_mean')]['source_gap_error_mm']
                        max_source_gap=max(max_source_gap,abs(a-b));assert abs(a-b)<1e-10
                if r['method']=='b_quota_map' and m['k']==2:
                    target=int(np.floor(np.asarray(m['posterior'])[:,1].sum()+.5))
                    assert np.sum(m['groups'])==target
                if phase=='development' and j['kind']=='bridge' and r['method']=='b_map':
                    orig=load(oldmap[j['case']]['output'])['xyz_world']*1000.
                    delta=float(np.max(abs(points[:,0]-orig)));max_replay=max(max_replay,delta);assert delta<1e-8
    for name in ('CHECKPOINT_BEFORE.json','V18_BEFORE.json'):
        for p,h in json.loads((dest/name).read_text()).items():assert sha(p)==h,p
    return dict(status='PASS',outputs=total,unique_artifacts_checked=len(checked),
         max_historical_bridge_map_difference_mm=max_replay,max_weighted_mean_moment_difference_mm=max_moment,
         max_weighted_mean_source_gap_error_difference_mm=max_source_gap,
         old_project_and_v18_unchanged=True,confirmation_source_lock_unchanged=True,
         scope='Hashes, representation algebra, output support and historical replay; not independent real geometry proof')

def main():
    p=argparse.ArgumentParser();p.add_argument('--dest',required=True);a=p.parse_args();dest=Path(a.dest)
    result=audit(dest);save(dest/'AUDIT.json',result)
    summary={}
    for phase in ('development','confirmation'):
        rows=[]
        for track in ('a','b'):rows+=json.loads((dest/f'{phase}_{track}_ROWS.json').read_text())
        summary[phase]=dict(aggregates=aggregate(rows),paired=paired(rows))
        v14.csvsave(dest/f'{phase}_ALL_RESULTS.csv',rows)
        # Full condition slices retained; descriptive comparisons, no post-hoc selector.
        slices={}
        for key in sorted({(r['kind'],r['gap'],r['sigma'],r.get('ratio'),r.get('dependence'),r['sigma_factor']) for r in rows},key=str):
            rr=[r for r in rows if (r['kind'],r['gap'],r['sigma'],r.get('ratio'),r.get('dependence'),r['sigma_factor'])==key]
            slices[str(key)]=aggregate(rr)
        save(dest/f'{phase}_SLICES.json',slices)
    save(dest/'SUMMARY.json',summary)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()

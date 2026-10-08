"""Evaluation-only repair: duplicate id/arm kwargs when merging sealed rows.
Locked run.py and all predictions remain unchanged. Only dict(**o, **d, ...)
is changed to dict(o, **d, ...); the rest is the original evaluate function.
"""
from run import *

def evaluate():
    verify_source_lock();obs=observations();ds=load(ROOT/'decisions/SEAL.json')
    assert sha(ROOT/'decisions/DECISIONS.json')==ds['decisions_sha256']
    for d,h in ds['observation_seals'].items():assert sha(ROOT/d/'SEAL.json')==h
    decisions={(d['id'],d['arm']):d for d in load(ROOT/'decisions/DECISIONS.json')}
    truth={t['id']:t for t in load(ROOT/'TRUTH.json')};method=load(ROOT/'METHOD.json')
    inputs={r['id']:r for r in load(ROOT/'OBS_INPUTS.json')}
    from oracle import _renderer
    renderer=_renderer()
    rows=[];mapping={(o['id'],o['arm']):o for o in obs}
    for o in obs:
        t=truth[o['id']];cam=renderer.deserialize_cam(t['actual_cameras'][0])
        z=float(renderer.intersect(t['scene'],cam,np.asarray(method['xy']))['depth'])
        d=decisions[o['id'],o['arm']]
        with np.load(o['curve_file']) as curve:
            ti=int(np.argmin(abs(curve['grid']-z)))
            true_mass=float(curve['mass'][ti]);true_ess=float(curve['ess'][ti]);true_count=int(curve['count'][ti])
        err=abs(d['selected_depth']-z)
        rows.append(dict(o,**d,name=t['name'],seed=t['seed'],true_depth=z,absolute_error=err,squared_error=err**2,
            true_in_support=any(a<=z<=b for a,b in o['intervals']),exact=err<1e-9,empty=not bool(o['intervals']),
            true_mass=true_mass,true_ess=true_ess,true_count=true_count))
    summary=[]
    for group,predicate in [('all',lambda r:True),('ring',lambda r:r['name'].startswith('ring')),('other',lambda r:not r['name'].startswith('ring'))]:
        base={r['id']:r for r in rows if r['arm']=='full9' and predicate(r)}
        for arm in method['arms']:
            sub=[r for r in rows if r['arm']==arm and predicate(r)]
            dif=[r['absolute_error']-base[r['id']]['absolute_error'] for r in sub]
            summary.append(dict(group=group,arm=arm,n=len(sub),mae=float(np.mean([r['absolute_error'] for r in sub])),mse=float(np.mean([r['squared_error'] for r in sub])),
                exact=sum(r['exact'] for r in sub),truth_in_support=sum(r['true_in_support'] for r in sub),empty=sum(r['empty'] for r in sub),
                improved_vs_full9=sum(x < -1e-9 for x in dif),worsened_vs_full9=sum(x > 1e-9 for x in dif),unchanged_vs_full9=sum(abs(x)<=1e-9 for x in dif),
                retained_full9_exact=sum(r['exact'] and base[r['id']]['exact'] for r in sub),baseline_exact=sum(r['exact'] for r in base.values()),
                improved_vs_incumbent=sum(r['absolute_error']<abs(900-r['true_depth'])-1e-9 for r in sub),worsened_vs_incumbent=sum(r['absolute_error']>abs(900-r['true_depth'])+1e-9 for r in sub)))
    pairs=[]
    for size in ('ring9','ring25'):
        for seed in (11,29,47):
            f=next(t for t in truth.values() if t['name']==size+'_flat' and t['seed']==seed)
            b=next(t for t in truth.values() if t['name']==size+'_textured' and t['seed']==seed)
            with np.load(inputs[f['id']]['image_file']) as a,np.load(inputs[b['id']]['image_file']) as c:
                center_diff=float(np.max(abs(a['images'][0,63:66,63:66]-c['images'][0,63:66,63:66])))
            assert center_diff==0
            for arm in method['arms']:
                oa=mapping[f['id'],arm];ob=mapping[b['id'],arm]
                with np.load(oa['curve_file']) as a,np.load(ob['curve_file']) as bcurve:
                    xor=int(np.sum(a['accepted']!=bcurve['accepted']))
                    wd=float(np.max(abs(a['weights']-bcurve['weights'])))
                    both=np.isfinite(a['scores'])&np.isfinite(bcurve['scores'])
                    score_delta=float(np.max(abs(a['scores'][both]-bcurve['scores'][both]))) if both.any() else None
                    values_same=bool(np.allclose(a['reference'],bcurve['reference'],atol=1e-10) and np.allclose(a['source'],bcurve['source'],atol=1e-10))
                pairs.append(dict(pair=size,seed=seed,arm=arm,reference_center_difference=center_diff,accepted_xor=xor,
                  support_symmetric_difference=interval_distance(oa['intervals'],ob['intervals']),weight_max_difference=wd,finite_score_max_difference=score_delta,
                  both_empty=not oa['intervals'] and not ob['intervals'],both_nonempty=bool(oa['intervals']) and bool(ob['intervals']),
                  selected_depth_difference=abs(decisions[f['id'],arm]['selected_depth']-decisions[b['id'],arm]['selected_depth']),
                  component_arrays_same=values_same))
    old={(r['fixture_id'],r['arm']):r for r in load(OLDM/'EVALUATION.json')}
    checks=[]
    for r in rows:
        if r['arm'] not in ('full9','connected9'):continue
        archived=old[truth[r['id']]['historical_id'],'plane_'+r['arm']]
        assert r['intervals']==archived['intervals']
        assert r['selected_depth']==archived['selected_depth']
        checks.append([r['id'],r['arm']])
    # Acceptance is fixed in the plan; no thresholds selected using this result.
    primary=next(s for s in summary if s['group']=='ring' and s['arm']==method['primary_arm'])
    base=next(s for s in summary if s['group']=='ring' and s['arm']=='full9')
    flips={a:sum(p['selected_depth_difference']>1e-9 for p in pairs if p['arm']==a) for a in method['arms']}
    success=primary['mae']<base['mae'] and primary['retained_full9_exact']==base['exact'] and flips[method['primary_arm']]<flips['full9']
    dump(ROOT/'evaluation/ROWS.json',rows);dump(ROOT/'evaluation/PAIRS.json',pairs)
    dump(ROOT/'evaluation/RESULTS.json',dict(created=now(),summary=summary,baseline_reproductions=len(checks),primary_success=success,selection_flips=flips))
    print(json.dumps(dict(stage='evaluate',summary=summary,selection_flips=flips,primary_success=success),indent=2))



if __name__ == '__main__':
    evaluate()

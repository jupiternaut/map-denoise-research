"""Independent post-seal metric/routing audit; never used to tune policies."""
from common import *
import csv,time
from scipy.io import loadmat
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

FIELDS=('source_MSE_mm2','source_MAE_mm','source_p95_mm','improved_fraction','harmed_fraction',
        'accepted_fraction','accepted_support_fraction','moved_fraction','move_RMS_mm','benefit_sum_mm2','harm_sum_mm2')


def metric(d0,d,mask,support,movement):
    values=d0.copy();values[mask]=d[mask]
    change=values[support]-d0[support];gain=d0[support]**2-values[support]**2
    motion=movement.copy();motion[~mask]=0
    return dict(zip(FIELDS,(float(np.mean(values[support]**2)),float(np.mean(values[support])),
        float(np.quantile(values[support],.95)),float(np.mean(change<-.1)),float(np.mean(change>.1)),
        float(mask.mean()),float(mask[support].mean()),float(np.mean(motion>1e-14)),
        float(np.sqrt(motion.mean())),float(np.maximum(gain,0).sum()),float(np.maximum(-gain,0).sum()))))


def main():
    check_host();start=time.monotonic();verify_seal(ROOT/'evaluation')
    with (ROOT/'evaluation/METRICS.csv').open() as f:
        saved={(int(r['scene']),r['roi'],r['condition'],r['arm']):r for r in csv.DictReader(f)}
    summary=json.loads((ROOT/'evaluation/SUMMARY.json').read_text())
    complement=json.loads((ROOT/'evaluation/COMPLEMENT.json').read_text())
    comp_lookup={(r['scene'],r['roi'],r['condition']):r for r in complement['cases']}
    assert len(saved)==2640 and len(comp_lookup)==60
    checks=[];computed={};comps={};direct=[];max_metric=0.;max_summary=0.
    for sid in SCENES:
        paths=cases_for_scene(sid)
        for case_index,path in enumerate(paths):
            rid,condition=path.name.split('__');dest=ROOT/'evaluation'/path.name
            with np.load(dest/'point_errors.npz',allow_pickle=False) as z:
                d0,da,db,support,ma,mb=(z[k] for k in ('d0','dA','dB','support','move2_A','move2_B'))
            assert all(v.shape==d0.shape for v in (da,db,support,ma,mb)) and support.dtype==bool
            assert all(np.isfinite(v).all() and np.min(v)>=0 for v in (d0,da,db,ma,mb))
            with np.load(BASE/'evaluation'/path.name/'point_errors.npz',allow_pickle=False) as z:
                assert np.array_equal(z['support'],support) and np.array_equal(z['movement_squared'],ma)
                assert np.max(abs(d0-z['d0']))<=1e-8 and np.max(abs(da-z['d1']))<=1e-8
            with np.load(ROOT/'inference'/f'scan{sid}'/path.name/'decisions.npz',allow_pickle=False) as z:masks={k:z[k] for k in z.files}
            with np.load(dest/'diagnostic_decisions.npz',allow_pickle=False) as z:
                route=z['oracle_AB_route'];diagnostics={k:z[k] for k in z.files if k!='oracle_AB_route'}
            assert len(masks)==21 and len(diagnostics)==23
            assert np.array_equal(diagnostics['oracle_A'],da<d0)
            assert np.array_equal(diagnostics['oracle_B'],db<d0)
            expected_route=np.argmin(np.column_stack([d0,da,db]),axis=1).astype(np.uint8)
            assert np.array_equal(route,expected_route)
            assert np.array_equal(diagnostics['oracle_AB'],route!=0)
            lead=masks['B_R__balanced'];bins=np.digitize(np.sqrt(mb),[1e-7,.25,.5,1,2,3,4,6.000001])
            for mode in ('count','bin'):
                group=support.astype(int)+(2*bins if mode=='bin' else 0)
                for seed in range(10):
                    random=diagnostics[f'random__B_R__balanced__{mode}__seed{seed}']
                    for g in np.unique(group):assert int(random[group==g].sum())==int(lead[group==g].sum())
            masks.update(diagnostics)
            assert len(masks)==44
            for arm,mask in masks.items():
                if arm=='oracle_AB':
                    d=np.minimum(da,db);movement=np.where(db<da,mb,ma)
                elif arm.startswith(('B_','random__')) or arm=='oracle_B':d,movement=db,mb
                else:d,movement=da,ma
                got=metric(d0,d,mask,support,movement);key=(sid,rid,condition,arm)
                error=max(abs(got[k]-float(saved[key][k])) for k in FIELDS)
                assert error<=1e-8,(key,error)
                max_metric=max(max_metric,error);computed[key]=got
            incremental=np.maximum(np.minimum(d0**2,da**2)-db**2,0)
            c=dict(B_better_than_A_KEEP_fraction=float(np.mean((db<np.minimum(d0,da))[support])),
                B_incremental_MSE_mm2=float(incremental[support].mean()),
                B_R_captured_incremental_MSE_mm2=float((incremental*lead)[support].mean()),
                B_R_selected_complement_fraction=float(np.mean(((incremental>0)&lead)[support])),
                identity_MSE_mm2=float(np.mean(d0[support]**2)),
                oracle_A_MSE_mm2=float(np.mean(np.minimum(d0,da)[support]**2)),
                oracle_AB_MSE_mm2=float(np.mean(np.minimum.reduce([d0,da,db])[support]**2)))
            assert all(abs(v-comp_lookup[(sid,rid,condition)][k])<=1e-10 for k,v in c.items())
            assert abs(c['oracle_A_MSE_mm2']-c['oracle_AB_MSE_mm2']-c['B_incremental_MSE_mm2'])<=1e-10
            comps[(sid,rid,condition)]=c
            checks.append(dict(case=path.name,arms=44,oracle_route_exact=True,random_groups_matched=True))
            if case_index==0:
                raw=CLOSEOUT/'confirmation'/f'scan{sid}'
                roi=next(r for r in json.loads((raw/'ROIS.json').read_text()) if r['id']==rid)
                evalroot=DATA/'closeout-confirmation-v1/evaluation_only'/f'scan{sid}'
                laser=read_points(evalroot/f'stl{sid:03d}_total.ply');obs=loadmat(evalroot/f'ObsMask{sid}_10.mat')
                ref=voxel(laser[in_box(laser,roi['lo'],roi['hi'])&observed(laser,obs)])
                tree=cKDTree(ref);b=read_points(path/'B_all.ply');ids=np.unique(np.linspace(0,len(b)-1,9).astype(int))
                err=float(np.max(abs(tree.query(b[ids],workers=1)[0]-db[ids])))
                assert err<=1e-8
                direct.append(dict(case=path.name,rows=ids.tolist(),B_distance_max_error_mm=err))
        print('AUDIT METRICS',sid,flush=True)
    for condition in CONDS:
        for arm in summary['methods']:
            scene_values=[];all_before=[];all_after=[]
            for sid in SCENES:
                keys=[k for k in computed if k[0]==sid and k[2]==condition and k[3]==arm]
                assert len(keys)==4
                got={f:float(np.mean([computed[k][f] for k in keys])) for f in FIELDS}
                before=[computed[(sid,k[1],condition,'identity')]['source_MSE_mm2'] for k in keys]
                got['MSE_gain_percent']=100*(1-got['source_MSE_mm2']/np.mean(before))
                recorded=summary['per_scene'][str(sid)][condition][arm]
                error=max(abs(got[k]-recorded[k]) for k in got);assert error<=1e-8
                max_summary=max(max_summary,error);scene_values.append(got)
                all_before.extend(before);all_after.extend([computed[k]['source_MSE_mm2'] for k in keys])
            pooled={f:float(np.mean([v[f] for v in scene_values])) for f in FIELDS}
            pooled['MSE_gain_percent']=100*(1-pooled['source_MSE_mm2']/np.mean(all_before))
            before=np.array(all_before);after=np.array(all_after)
            pooled.update(wins=int(np.sum(after<before-1e-12)),ties=int(np.sum(abs(after-before)<=1e-12)),losses=int(np.sum(after>before+1e-12)),n_rois=12)
            recorded=summary['exposed_replay'][condition][arm]
            error=max(abs(pooled[k]-recorded[k]) for k in pooled);assert error<=1e-8
            max_summary=max(max_summary,error)
        for policy in ('balanced','native_priority','natural','pair'):
            suffix='_pair' if policy=='pair' else '__'+policy
            mse={k:summary['exposed_replay'][condition][k+suffix]['source_MSE_mm2'] for k in ('A_F','A_R','B_F','B_R')}
            base=summary['exposed_replay'][condition]['identity']['source_MSE_mm2']
            expected=100*((mse['B_F']-mse['B_R'])-(mse['A_F']-mse['A_R']))/base
            assert abs(expected-summary['interactions'][condition][policy]['candidate_by_evidence_interaction_gain_pp'])<=1e-10
        for field in next(iter(comps.values())):
            value=np.mean([np.mean([c[field] for (s,_,co),c in comps.items() if s==sid and co==condition]) for sid in SCENES])
            assert abs(value-complement['exposed_replay'][condition][field])<=1e-10
    save_json(ROOT/'AUDIT_METRICS.json',dict(status='PASS',case_count=len(checks),metric_rows=len(computed),
        checks=checks,sampled_reference_recomputations=direct,maximum_metric_difference=max_metric,
        maximum_summary_difference=max_summary,oracle_routes_and_movement_correct=True,
        all_20_random_controls_match_support_groups=True,complement_and_interactions_reproduced=True,
        source_sha256={str(ROOT/'audit_metrics.py'):sha(ROOT/'audit_metrics.py'),str(ROOT/'evaluation/SEALED.json'):sha(ROOT/'evaluation/SEALED.json')},
        reference_read_after_evaluation_seal=True,policy_changed=False,wall_seconds=time.monotonic()-start))


if __name__=='__main__':
    with threadpool_limits(limits=1):main()

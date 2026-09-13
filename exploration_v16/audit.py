"""Post-run independent numerical checks, input pairing and immutable checkpoints."""
from pathlib import Path
import sys, json
import numpy as np
from scipy.spatial import cKDTree
from run import v14


def main(path, audit_name='audit'):
    run=Path(path)
    dest=run/audit_name;dest.mkdir()
    jobs=json.loads((run/'JOBS_LOCK.json').read_text())
    records=json.loads((run/'mechanism_SEALED_BEFORE_GT.json').read_text())
    rows=json.loads((run/'mechanism_ROWS.json').read_text())
    ix={(r['case'],r['method']):r for r in rows}
    maxima={}
    def check(key,a,b,tol=1e-9):
        delta=float(np.max(abs(np.asarray(a)-np.asarray(b))))
        maxima[key]=max(maxima.get(key,0.),delta)
        if delta>tol:
            raise AssertionError((key,a,b,delta))
    fingerprints={}
    def hashcheck(p,h):
        assert v14.sha(p)==h, p
        fingerprints[p]=h
    repair=json.loads((run/'SCORING_REPAIR.json').read_text()) if (run/'SCORING_REPAIR.json').exists() else dict(changes=[])
    corrected={r['path']:r for r in repair['changes']}
    for f in ('SOURCES_BEFORE.json','V15_HASHES_BEFORE.json'):
        for p,h in json.loads((run/f).read_text()).items():
            if f=='SOURCES_BEFORE.json' and p in corrected:
                archived=run/'source'/Path(p).relative_to(Path(__file__).resolve().parents[1])
                hashcheck(str(archived),h)
                hashcheck(p,corrected[p]['after'])
            else:hashcheck(p,h)
    byseed={}
    for j in jobs:
        hashcheck(j['input'],j['input_sha256']);hashcheck(j['evaluation'],j['evaluation_sha256'])
        inp,e=v14.load(j['input']),v14.load(j['evaluation'])
        assert set(inp)=={'design','height_mm','xyz_world','scan_id','sigma_mm'}
        np.testing.assert_array_equal(inp['design'][:,0],1.)
        np.testing.assert_array_equal(inp['height_mm']/1000.,inp['xyz_world'][:,2])
        assert np.isfinite(inp['xyz_world']).all()
        if j['gap']:
            for f in (0,1):
                counts=np.bincount(e['gt_layer'][inp['scan_id']==f],minlength=2)
                np.testing.assert_array_equal(counts,[192,192])
        ref=byseed.setdefault(j['seed'],(inp,e['eps_mm']/j['sigma']))
        np.testing.assert_array_equal(inp['design'],ref[0]['design'])
        np.testing.assert_allclose(e['eps_mm']/j['sigma'],ref[1],atol=0,rtol=0)
        byseed[j['seed']]=ref
    for r in records:
        if r['status']!='OK':continue
        hashcheck(r['output'],r['output_sha256'])
        inp,e,a=map(v14.load,(r['input'],r['evaluation'],r['output']))
        row=ix[r['case'],r['method']]
        q=a['xyz_world']*1000.;ref=e['gt_clean_xyz_world']*1000.;label=e['gt_layer'];gap=r['gap']
        k=int(a['k']);levels=e['true_means_mm']
        check('surface_mae_mm',np.min(abs(q[:,2,None]-levels),axis=1).mean(),row['surface_mae_mm'])
        check('matched_rms_mm',np.sqrt(np.mean(np.sum((q-ref)**2,axis=1))),row['matched_rms_mm'])
        check('balanced_source_mae_mm',np.mean([np.mean(abs(q[label==c,2]-levels[c])) for c in np.unique(label)]),row['balanced_source_mae_mm'])
        check('coverage_1mm',np.mean(cKDTree(q).query(ref)[0]<=1.+1e-9),row['coverage_1mm'])
        check('prediction_reconstruction',a['means'][a['groups']]+inp['design'][:,1:]@a['slope'],a['prediction'])
        check('posterior_unit_sum',a['posterior'].sum(1),np.ones(len(q)))
        np.testing.assert_array_equal(a['groups'],np.argmax(a['posterior'],axis=1))
        if r['method'].endswith('oracle_slope'):
            np.testing.assert_array_equal(a['slope'],e['true_slope'])
            assert r['info']['oracle_fields']==['shared_slope']
        else:assert r['info']['oracle_fields']==[]
        if gap:
            sourcegap=q[label==1,2].mean()-q[label==0,2].mean()
            check('source_gap_mm',sourcegap,row['source_gap_mm'])
            intercept_gap=np.ptp(a['means'])
            du=inp['design'][label==1,1:].mean(0)-inp['design'][label==0,1:].mean(0)
            b=float(du@a['slope'])
            if k==2:
                alpha=np.mean(a['groups'][label==0]==1);beta=np.mean(a['groups'][label==1]==0)
                check('assignment_error',(alpha+beta)/2,row['assignment_error'])
                check('gap_identity',intercept_gap*(1-alpha-beta)+b,sourcegap)
                cf=ref.copy();cf[:,:2]=q[:,:2];cf[:,2]=a['means'][label]+inp['design'][:,1:]@a['slope']
            else:
                check('single_gap_identity',b,sourcegap)
                cf=q.copy()
            aa=cf-ref;bb=q-cf
            check('counterfactual_geometry_mse_mm2',np.mean(np.sum(aa**2,axis=1)),row['counterfactual_geometry_mse_mm2'])
            check('assignment_delta_mse_mm2',np.mean(np.sum(bb**2,axis=1)),row['assignment_delta_mse_mm2'])
            check('geometry_assignment_cross_mm2',2*np.mean(np.sum(aa*bb,axis=1)),row['geometry_assignment_cross_mm2'])
            check('total_mse_decomposition',row['counterfactual_geometry_mse_mm2']+row['assignment_delta_mse_mm2']+row['geometry_assignment_cross_mm2'],row['matched_rms_mm']**2)
    replays=json.loads((run/'replay_SEALED_BEFORE_GT.json').read_text())
    replay_rows={(r['case'],r['method']):r for r in json.loads((run/'replay_ROWS.json').read_text())}
    for r in replays:
        if r['status']!='OK':continue
        hashcheck(r['output'],r['output_sha256'])
        hashcheck(r['previous_output'],r['previous_output_sha256'])
        old,new=map(v14.load,(r['previous_output'],r['output']))
        np.testing.assert_array_equal(old['xyz_world'],new['xyz_world'])
        if r['gap']>0:
            e=v14.load(r['evaluation']);label=e['gt_layer'];ref=e['gt_clean_xyz_world']
            row=replay_rows[r['case'],r['method']]
            diag=v14.load(run/'diagnostics'/f'{r["case"]}__{r["method"]}.npz')
            q=new['xyz_world'];cf=diag['counterfactual_xyz_world']
            aa=(cf-ref)*1000.;bb=(q-cf)*1000.
            check('replay_geometry_mse',np.mean(np.sum(aa*aa,axis=1)),row['counterfactual_geometry_mse_mm2'])
            check('replay_assignment_mse',np.mean(np.sum(bb*bb,axis=1)),row['assignment_delta_mse_mm2'])
            check('replay_cross_term',2*np.mean(np.sum(aa*bb,axis=1)),row['geometry_assignment_cross_mm2'])
            check('replay_total_mse',np.mean(np.sum(((q-ref)*1000.)**2,axis=1)),row['total_mse_mm2'])
            v=new['basis']@np.r_[-new['slope']/new['common_scale'],1.]
            mu=new['means'];world_order=np.argsort(mu/v[2])
            coeff=mu[world_order[label]] if len(mu)==2 else np.full(len(label),mu[0])
            d=abs(((ref-new['center'])*100.)@(v*10.)-coeff)/np.linalg.norm(v)
            check('replay_plane_mae',d.mean(),row['fitted_source_plane_mae_mm'])
            check('replay_fitted_gap',np.ptp(mu)/abs(v[2]),row['fitted_vertical_gap_mm'])
            check('replay_source_gap',1000.*(q[label==1,2].mean()-q[label==0,2].mean()),row['source_gap_mm'])
            take=np.flatnonzero(new['support']);ids=new['order'][take]
            inv=np.argsort(world_order)
            g=np.empty(len(q),int);g[new['order']]=inv[new['groups']]
            if len(mu)==2:
                check('replay_assignment_rate',.5*sum(np.mean(g[label==k]!=k) for k in (0,1)),row['assignment_error'])
            inp=v14.load(r['input']);expected=inp['xyz_world'].copy()
            coeff_local=mu[world_order[label[ids]]] if len(mu)==2 else np.full(len(ids),mu[0])
            new_height=coeff_local+new['design'][take,1:]@new['slope']
            expected[ids] += ((new_height-new['local'][take,2])/1000.)[:,None]*new['normal']
            check('replay_counterfactual_points_m',expected,cf)
    failures=[dict(case=r['case'],method=r['method'],error=r.get('error')) for r in records+replays if r['status']!='OK']
    summary=dict(status='PASS' if not failures else 'FAILURES_RETAINED',mechanism_scored=len(records),replay_compared=len(replays),
                 failed=len(failures),failures=failures,unique_files_hashed=len(fingerprints),
                 new_independent_seeds=len(byseed),maximum_differences=maxima,
                 scope='arithmetic/input provenance/replay verification, not physical validity or generalization proof')
    v14.save(dest/'AUDIT.json',summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main(sys.argv[1],sys.argv[2] if len(sys.argv)>2 else 'audit')

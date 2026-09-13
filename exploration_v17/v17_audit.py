"""Independent output geometry / selection / held-out density recomputation."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
from scipy.special import logsumexp

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as f:return {k:f[k] for k in f.files}
def nll(m,x,y,sigma):
    if int(m['k'])==1:return .5*((y-x@np.asarray(m['plane']))/sigma)**2
    z=(x[:,1:]-m['feature_center'])/m['feature_scale']
    f=np.ones((len(x),1))
    if m['family'] in ('L','R'):f=np.c_[f,z]
    if m['family']=='R':
        for i in (-.75,0,.75):
            for j in (-.75,0,.75):f=np.c_[f,np.exp(-((z[:,0]-i)**2+(z[:,1]-j)**2)/(2*.6**2))]
    a=f@np.asarray(m['gate'])
    prior=np.c_[-np.logaddexp(0,a),-np.logaddexp(0,-a)]
    prediction=np.asarray(m['means'])[None,:]+(x[:,1:]@np.asarray(m['slope']))[:,None]
    return -logsumexp(prior-.5*((y[:,None]-prediction)/sigma)**2,axis=1)

def main(dest):
    checked=set();count=0;cv_count=0;max_error=0.;start_count=0;nonconv=0;max_gradient=0.;counts={}
    params=dict(S=3,C=5,L=7,R=16);families=tuple(params)
    for phase in ('development','confirmation','bridge'):
        records=json.loads((dest/f'{phase}_SEALED_BEFORE_GT.json').read_text())
        rows={(r['case'],r['method']):r for r in json.loads((dest/f'{phase}_ROWS.json').read_text())}
        assert len(records)==len(rows)
        for r in records:
            assert r['status']=='OK',r
            for key in ('input','evaluation','output','diagnostic'):
                path=r[key]
                if path not in checked:
                    assert sha(path)==r[key+'_sha256'],path;checked.add(path)
            inp,truth,art=map(load,(r['input'],r['evaluation'],r['output']))
            if 'json' in truth:truth.update(json.loads(truth.pop('json').tobytes().decode()))
            q=art['xyz_world']*1000.;ref=truth['gt_clean_xyz_world']*1000.;labels=truth['gt_layer']
            if phase=='bridge':
                distances=[]
                for z,x0,x1,y0,y1 in truth['surface_rectangles_mm']:
                    distances.append(np.sqrt((q[:,0]-np.clip(q[:,0],x0,x1))**2+(q[:,1]-np.clip(q[:,1],y0,y1))**2+(q[:,2]-z)**2))
                d=np.asarray(distances).T
                unsupported=art['order'][~art['support']]
                np.testing.assert_array_equal(art['xyz_world'][unsupported],inp['xyz_world'][unsupported])
                x,y=art['design'],art['corrected']
            else:
                d=abs(q[:,2,None]-truth['true_means_mm'])
                np.testing.assert_array_equal(art['xyz_world'][:,:2],inp['xyz_world'][:,:2])
                x,y=inp['design'],inp['height_mm']
            expected=dict(surface_mae_mm=float(d.min(1).mean()),
                balanced_source_mae_mm=float(np.mean([d[labels==k,k].mean() for k in np.unique(labels)])),
                matched_rms_mm=float(np.sqrt(np.mean(np.sum((q-ref)**2,axis=1)))))
            if r['gap']:expected['source_gap_error_mm']=abs(q[labels==1,2].mean()-q[labels==0,2].mean()-r['gap'])
            row=rows[r['case'],r['method']]
            for k,v in expected.items():
                delta=abs(row[k]-v);max_error=max(max_error,delta);assert delta<1e-8,(k,delta,r['case'])
            assert row['model_k']==int(art['k'])
            count+=1
            if r['diagnostic'] in counts:continue
            dg=json.loads(Path(r['diagnostic']).read_text());counts[r['diagnostic']]=1
            pool=dg['pool'];scores={f:m['bic'] for f,m in pool.items()}
            for f,m in pool.items():
                loss=nll(m,x,y,r['sigma']);assert abs(loss.sum()-m['nll'])<1e-7
                expected_bic=2*loss.sum()+.1*np.sum(np.asarray(m['gate'])[1:]**2)+params[f]*np.log(len(y))
                assert abs(expected_bic-m['bic'])<1e-7
                for st in m['solver_diagnostics']:
                    start_count+=1;nonconv+=not st['success'];max_gradient=max(max_gradient,st['gradient_inf'])
            fold=np.asarray(dg['cv']['folds']);cv_scores={f:0. for f in families}
            for record in dg['cv']['details']:
                ti=np.asarray(record['train_indices']);vi=np.asarray(record['validation_indices'])
                assert not set(ti)&set(vi)
                assert set(ti)|set(vi)==set(range(len(y)))
                np.testing.assert_array_equal(vi,np.flatnonzero(fold==record['fold']))
                for f,m in record['families'].items():
                    loss=nll(m,x[vi],y[vi],r['sigma'])
                    assert abs(loss.sum()-m['validation_nll'])<1e-7
                    np.testing.assert_allclose(loss,np.asarray(dg['cv']['per_point'][f])[vi],rtol=1e-8,atol=1e-8)
                    cv_scores[f]+=float(loss.sum());cv_count+=1
                    for st in m['solver_diagnostics']:
                        start_count+=1;nonconv+=not st['success'];max_gradient=max(max_gradient,st['gradient_inf'])
            pick=lambda sc,ff:min(ff,key=lambda f:(round(sc[f],10),params[f],families.index(f)))
            expected_choices=dict(constant_only=pick(scores,('S','C')),linear_only=pick(scores,('S','L')),rbf_only=pick(scores,('S','R')),
                adaptive_bic=pick(scores,families),adaptive_cv=pick(cv_scores,families))
            assert expected_choices==dg['choices']
            for method,f in expected_choices.items():
                rr=rows[r['case'],method];assert rr['selected_family']==f
                output=load(dest/'outputs'/f'{r["case"]}__{method}.npz')
                for k in ('means','slope','groups','prediction'):np.testing.assert_array_equal(output[k],pool[f][k])
    old=json.loads((dest/'OLD_HASHES_BEFORE.json').read_text())
    for p,h in old.items():assert sha(p)==h,p
    lock=json.loads((dest/'CONFIRMATION_LOCK.json').read_text())
    for p,h in lock['sources'].items():assert sha(p)==h,p
    result=dict(status='PASS',geometry_outputs=count,unique_diagnostics=len(counts),held_out_family_folds=cv_count,
        verified_unique_artifacts=len(checked),old_files_unchanged=len(old),maximum_geometry_difference_mm=max_error,
        optimizer_starts=start_count,nonconverged_starts=nonconv,max_terminal_gradient_inf=max_gradient,
        note='Nonconverged starts retained; finite incumbent kept. No global optimum certificate. Independent evaluator formula replay, not independently acquired data.')
    with (dest/'AUDIT.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main(Path(sys.argv[1]))

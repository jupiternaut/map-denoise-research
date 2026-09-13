"""Independent numerical replay, paired summaries, and immutable result seal."""
from pathlib import Path
import sys,json,hashlib,subprocess,os,shutil
import numpy as np
from scipy.special import logsumexp
from scipy.stats import t
HERE=Path(__file__).resolve().parent
METHODS=('identity','old_spatial36','v17_map','retained_map','retained_half','retained_mean')
def load(p):
    with np.load(p,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2)

def aggregate(rows):
    out={}
    for method in METHODS:
        rr=[r for r in rows if r['method']==method];du=[r for r in rr if r['gap']];si=[r for r in rr if not r['gap']]
        fields=('surface_mae_mm','balanced_source_mae_mm','matched_rms_mm','source_gap_error_mm','fitted_gap_error_mm','distance_to_fitted_surface_mm','coverage_1mm')
        m={k:float(np.mean([r[k] for r in du])) for k in fields if du and all(k in r for r in du)}
        m.update(dual_n=len(du),dual_k1=None if method=='identity' else sum(r['model_k']==1 for r in du),
            single_n=len(si),single_k2=None if method=='identity' else sum(r['model_k']==2 for r in si),
            single_mae_mm=float(np.mean([r['surface_mae_mm'] for r in si])) if si else None)
        out[method]=m
    return out

def paired(rows,a,b,key):
    x={r['case']:r for r in rows if r['method']==a and r['gap']};y={r['case']:r for r in rows if r['method']==b and r['gap']}
    assert x.keys()==y.keys();seeds=sorted({r['seed'] for r in x.values()})
    means=[float(np.mean([x[c][key]-y[c][key] for c in x if x[c]['seed']==s])) for s in seeds]
    mean=float(np.mean(means));half=float(t.ppf(.975,len(means)-1)*np.std(means,ddof=1)/np.sqrt(len(means)))
    return dict(a=a,b=b,metric=key,mean_a_minus_b=mean,ci95=[mean-half,mean+half],seed_differences=means,seeds=seeds)

def review(dest):
    summary={};seen=set();n=0;max_error=0.;replacement={}
    for phase in ('development','confirmation','bridge'):
        records=json.loads((dest/f'{phase}_SEALED_BEFORE_GT.json').read_text());rows=json.loads((dest/f'{phase}_ROWS.json').read_text())
        lookup={(r['case'],r['method']):r for r in rows}
        assert len(records)==len(rows)
        for kind in sorted({r['kind'] for r in rows}):
            rr=[r for r in rows if r['kind']==kind];name=phase+'_'+kind
            summary[name]=dict(overall=aggregate(rr),paired=[],slices={})
            for a,b in [('old_spatial36','retained_map'),('v17_map','retained_map'),('retained_map','retained_half'),('retained_map','retained_mean')]:
                for k in ('surface_mae_mm','balanced_source_mae_mm','matched_rms_mm','source_gap_error_mm'):
                    summary[name]['paired'].append(paired(rr,a,b,k))
            for value in sorted({r['dependence'] for r in rr if r['gap'] and 'dependence' in r}):
                summary[name]['slices'][str(value)]=aggregate([r for r in rr if r['gap'] and r['dependence']==value])
        for r in records:
            assert r['status']=='OK',r
            for key in ('input','evaluation','output','diagnostic'):
                path=r[key]
                if path not in seen:assert sha(path)==r[key+'_sha256'],path;seen.add(path)
            inp,truth,art=map(load,(r['input'],r['evaluation'],r['output']))
            if 'json' in truth:truth.update(json.loads(truth.pop('json').tobytes().decode()))
            q=art['xyz_world']*1000;ref=truth['gt_clean_xyz_world']*1000;labels=truth['gt_layer'];row=lookup[r['case'],r['method']]
            if r['kind']=='statistical':
                d=abs(q[:,2,None]-truth['true_means_mm']);x,y=inp['design'],inp['height_mm']
            else:
                columns=[]
                for z,x0,x1,y0,y1 in truth['surface_rectangles_mm']:
                    target=np.c_[np.clip(q[:,0],x0,x1),np.clip(q[:,1],y0,y1),np.full(len(q),z)]
                    columns.append(np.linalg.norm(q-target,axis=1))
                d=np.column_stack(columns);x,y=art['design'],art['corrected']
            expected=dict(surface_mae_mm=float(d.min(1).mean()),balanced_source_mae_mm=float(np.mean([d[labels==k,k].mean() for k in np.unique(labels)])),
                matched_rms_mm=float(np.linalg.norm(q-ref)/np.sqrt(len(q))))
            if r['gap']:expected['source_gap_error_mm']=abs(q[labels==1,2].mean()-q[labels==0,2].mean()-r['gap'])
            for key,value in expected.items():
                error=abs(value-row[key]);max_error=max(max_error,error);assert error<1e-8
            if r['method']=='identity':np.testing.assert_array_equal(art['xyz_world'],inp['xyz_world'])
            elif r['kind']=='bridge':
                unsupported=art['order'][~art['support']];np.testing.assert_array_equal(art['xyz_world'][unsupported],inp['xyz_world'][unsupported])
            n+=1
            if r['method']!='retained_map':continue
            dg=json.loads(Path(r['diagnostic']).read_text());pool=dg['pool'];m=dg['retained'];old=dg['old']
            assert dg['r_bic_after']<=dg['r_bic_before']+1e-8
            assert dg['selected_bic']<=min(p['bic'] for p in pool.values())+1e-8
            if int(old['k'])==2:
                z=(x[:,1:]-pool['R']['feature_center'])/pool['R']['feature_scale'];f=np.c_[np.ones(len(x)),z]
                for i in (-.75,0,.75):
                    for j in (-.75,0,.75):f=np.c_[f,np.exp(-np.sum((z-[i,j])**2,axis=1)/(2*.6**2))]
                logits=f@np.asarray(old['gate']);levels=np.asarray(old['means'])[None,:]+(x[:,1:]@old['slope'])[:,None]
                lp=-.5*((y[:,None]-levels)/r['sigma'])**2+np.c_[-np.logaddexp(0,logits),-np.logaddexp(0,-logits)]
                bic=float(-2*logsumexp(lp,axis=1).sum()+.1*np.sum(np.asarray(old['gate'])[1:]**2)+16*np.log(len(y)))
                assert abs(bic-dg['old_r_bic'])<1e-6
                assert dg['r_bic_after']<=bic+1e-8
            levels=np.asarray(m['means'])[None,:]+(x[:,1:]@m['slope'])[:,None];post=np.asarray(m['posterior']);mean=np.sum(post*levels,axis=1)
            hard=levels[np.arange(len(x)),m['groups']]
            for method,gamma in [('retained_map',1.),('retained_half',.5),('retained_mean',0.)]:
                a=load(dest/'outputs'/f'{r["case"]}__{method}.npz')
                np.testing.assert_allclose(a['prediction'],gamma*hard+(1-gamma)*mean,rtol=0,atol=1e-12)
                for key in ('means','slope','groups','posterior'):np.testing.assert_array_equal(a[key],m[key])
                risk=np.sum(post*(a['prediction'][:,None]-levels)**2,axis=1)
                variance=np.sum(post*(mean[:,None]-levels)**2,axis=1)
                np.testing.assert_allclose(risk,variance+(a['prediction']-mean)**2,atol=1e-10)
            replacement[r['case']]=dict(phase=phase,kind=r['kind'],replaced=dg['replaced_r'],bic_drop=dg['r_bic_before']-dg['r_bic_after'])
    for p,h in json.loads((dest/'V17_HASHES_BEFORE.json').read_text()).items():assert sha(p)==h,p
    for p,h in json.loads((dest/'CONFIRMATION_LOCK.json').read_text()).items():assert sha(p)==h,p
    old17=dest.parent/'adaptive-v17-mklh_h2j'
    oldmiss=json.loads((old17/'INCUMBENT_AUDIT.json').read_text())['bridge']['rows']
    development={(r['case'],r['method']):r for r in json.loads((dest/'development_ROWS.json').read_text())}
    repaired=[]
    for r in oldmiss:
        if r['new_minus_old']<=1:continue
        c=r['case'];a=load(dest/'outputs'/f'{c}__retained_map.npz')['xyz_world'];b=load(dest/'outputs'/f'{c}__old_spatial36.npz')['xyz_world']
        delta=float(np.max(abs(a-b))*1000.);assert delta<1e-9
        repaired.append(dict(case=c,max_output_difference_mm=delta,before_mae_mm=development[c,'v17_map']['surface_mae_mm'],after_mae_mm=development[c,'retained_map']['surface_mae_mm']))
    save(dest/'FIVE_REPAIRS.json',repaired)
    save(dest/'SUMMARY.json',summary);save(dest/'REPLACEMENTS.json',replacement)
    audit=dict(status='PASS',outputs=n,unique_artifacts=len(seen),max_geometry_error_mm=max_error,
        V17_unchanged=True,posterior_risk_identity=True,pool_objective_never_worse=True,
        warning='Objective monotonicity is not geometry monotonicity. Cached development and fresh confirmation separated.')
    save(dest/'AUDIT.json',audit)
    print(json.dumps(audit,indent=2));print(json.dumps({k:v['overall'] for k,v in summary.items() if not k.startswith('development')},indent=2))

def seal(dest):
    tests=[];env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for suite in ('exploration_v18','exploration_v17','exploration_v16','exploration_v15','exploration_v14'):
        p=subprocess.run([sys.executable,'-m','unittest','discover','-s',suite,'-p','test_*.py','-v'],cwd=HERE.parent,env=env,text=True,capture_output=True)
        tests.append(dict(suite=suite,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr));print(suite,p.returncode,flush=True)
    save(dest/'TEST_RESULTS.json',tests);assert all(r['returncode']==0 for r in tests)
    delivery=dest/'final_delivery';delivery.mkdir()
    for p in HERE.iterdir():
        if p.is_file() and p.suffix in ('.py','.md'):shutil.copyfile(p,delivery/p.name)
    files={str(p.relative_to(dest)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in dest.rglob('*') if p.is_file()}
    save(dest/'FINAL_MANIFEST.json',dict(files=files,total_bytes=sum(x['bytes'] for x in files.values())))
    print('SEALED',len(files),'files')

if __name__=='__main__':
    (seal if len(sys.argv)>2 and sys.argv[2]=='seal' else review)(Path(sys.argv[1]))

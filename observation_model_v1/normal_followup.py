import json,sys,tempfile,csv,shutil
from pathlib import Path
import numpy as np
from source_filter import build,reference_id
from run import rms,sha,ambiguity,PATCHES,RUNS,HERE

PRIOR=RUNS/'observation-model-v1-7riqb9c4'

def scalar_fit(j,r):
    w=np.ones(len(r));beta=0.
    for _ in range(8):
        beta=float(np.sum(w*j*r)/max(np.sum(w*j*j),1e-20))
        e=r-j*beta;scale=max(1.4826*float(np.median(np.abs(e-np.median(e)))),1e-6)
        w=np.minimum(1.,1.345*scale/np.maximum(np.abs(e),1e-20))
    return beta

def estimate_normal(q,scan,valid_planes=False,local_normal=False):
    s=build(q,scan);a=q[s['anchor']];a=a-a.mean(0)
    _,basis=np.linalg.eigh(a.T@a);normal=basis[:,0]
    take=s['support'].copy()
    if valid_planes:
        e=s['plane_eigenvalues'];take&=e[:,1]>=.05*np.maximum(e[:,2],1e-20)
    if local_normal and np.any(take):
        normals=s['normals'][take];_,nb=np.linalg.eigh(normals.T@normals);normal=nb[:,-1]
    j=s['normals']@normal;folds=[];beta=0.;out=q.copy()
    if take.sum()>=12:
        beta=scalar_fit(j[take],s['residual'][take]);out[s['idx']]-=beta*normal
        for fold in (0,1):
            train=take&(s['fold']==fold);test=take&(s['fold']!=fold)
            if train.sum()<12 or test.sum()<12:
                folds.append(dict(pass_check=False,reason='too_few_points'));continue
            b=scalar_fit(j[train],s['residual'][train])
            before=float(np.mean(np.abs(s['residual'][test])))
            after=float(np.mean(np.abs(s['residual'][test]-b*j[test])))
            folds.append(dict(pass_check=bool(after<before),before_m=before,after_m=after,beta_m=b))
    accepted=len(folds)==2 and all(f['pass_check'] for f in folds)
    outputs=dict(normal_all=out,normal_confirmed=out.copy() if accepted else q.copy())
    for candidate in outputs.values():
        np.testing.assert_array_equal(candidate[s['anchor']],q[s['anchor']])
        delta=candidate[s['idx']]-q[s['idx']]
        np.testing.assert_allclose(delta,np.tile(delta[0],(len(delta),1)),atol=1e-12)
    return outputs,dict(accepted=accepted,beta_mm=1000*beta,normal=normal.tolist(),folds=folds,support=int(take.sum()))

def main():
    dest=Path(tempfile.mkdtemp(prefix='observation-normal-v1-',dir=RUNS));print('RUN',dest,flush=True)
    src=dest/'source';src.mkdir()
    for p in HERE.iterdir():
        if p.is_file():shutil.copy2(p,src/p.name)
    old=json.loads((PRIOR/'RESULTS.json').read_text());hashes={str(p):sha(p) for p in PRIOR.iterdir() if p.is_file()}
    rng=np.random.default_rng(916701);j=rng.uniform(.3,1,50)
    assert abs(scalar_fit(j,j*.005)-.005)<1e-12
    assert scalar_fit(j,np.zeros(50))==0
    records=[];diags=[];baselines={};refs={};counts={}
    for d in old['diagnostics']:
        file=PRIOR/f"{d['case']}.npz"
        with np.load(file) as a:q=a['input'].copy();scan=a['scan_id'].copy()
        outputs,info=estimate_normal(q,scan)
        np.savez_compressed(dest/f"{d['case']}.npz",**outputs)
        records.append((d,q,scan,outputs,info));diags.append(dict(case=d['case'],**info))
        counts.setdefault(d['kind'],set()).add(sha(file)) # file hash includes old candidate outputs too
        if d['kind']=='unperturbed':
            for method,out in outputs.items():baselines[(d['patch'],method)]=out
    # No measured original reference supplied to any estimator.
    for p in PATCHES.glob('*/*.npz'):
        with np.load(p) as a:refs[p.stem]=a['xyz_world'].copy()
    rows=list(old['rows'])
    for d,q,scan,outputs,info in records:
        reference=refs[d['patch']];anchor=scan==reference_id(scan)
        for method,out in outputs.items():
            rows.append(dict(patch=d['patch'],kind=d['kind'],seed=d['seed'],case=d['case'],method=method,n=len(q),
                input_rms_mm=rms(q,reference),recovery_rms_mm=rms(out,reference),moving_station_rms_mm=rms(out[~anchor],reference[~anchor]),
                edit_rms_mm=rms(out,q),response_to_own_baseline_rms_mm=rms(out,baselines[(d['patch'],method)]),
                crossfit_accepted=info['accepted'],support=info['support'],rank=1))
    aq,asc,arefs=ambiguity();aout,ainfo=estimate_normal(aq,asc);ar=list(old['ambiguity_rows'])
    for world,ref in arefs.items():
        for method,out in aout.items():
            ar.append(dict(world=world,method=method,geometry_rms_mm=rms(out,ref),
                station_centroid_gap_mm=float(np.linalg.norm(out[asc==1].mean(0)-out[asc==2].mean(0))*1000)))
    np.savez_compressed(dest/'ambiguity.npz',**aout)
    summary=[]
    for kind in dict.fromkeys(r['kind'] for r in rows):
        for method in dict.fromkeys(r['method'] for r in rows):
            rr=[r for r in rows if r['kind']==kind and r['method']==method]
            summary.append(dict(kind=kind,method=method,rows=len(rr),
                mean_recovery_rms_mm=float(np.mean([r['recovery_rms_mm'] for r in rr])),
                median_recovery_rms_mm=float(np.median([r['recovery_rms_mm'] for r in rr])),
                wins_vs_identity=sum(r['recovery_rms_mm']<r['input_rms_mm']-1e-8 for r in rr),
                confirmed=sum(r['crossfit_accepted'] for r in rr)))
    # Unique *input arrays*, not seed count, define repeated geometry here.
    import hashlib
    unique={}
    for d,q,scan,outputs,info in records:
        unique.setdefault(d['kind'],set()).add(hashlib.sha256(q.tobytes()+scan.tobytes()).hexdigest())
    for p,h in hashes.items():assert sha(Path(p))==h
    result=dict(rows=rows,summary=summary,diagnostics=diags,ambiguity_rows=ar,ambiguity_info=ainfo,
        prior=str(PRIOR),prior_unchanged=True,prior_hashes=hashes,scalar_tests_passed=2,
        unique_inputs_per_kind={k:len(v) for k,v in unique.items()},
        scope='Exposed-data mechanism followup; no new independent real geometry')
    (dest/'RESULTS.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    with (dest/'RESULTS.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('SUMMARY',json.dumps([r for r in summary if r['method'].startswith('normal')]),flush=True)
    print('AMBIGUITY',json.dumps(ar[-4:]),flush=True)
    print('UNIQUE',result['unique_inputs_per_kind'],flush=True)
    print('COMPLETE',dest,flush=True)

if __name__=='__main__':main()

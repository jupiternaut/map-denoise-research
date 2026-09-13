import sys,json,time,csv,hashlib,tempfile,shutil
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from source_filter import estimate,reference_id,robust_solve,transform

HERE=Path(__file__).resolve().parent
PATCHES=Path('/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/patches')
RUNS=Path('/srv/slam-research/grf/map-denoise/runs')
METHODS=('identity','point_projection','rigid_all','rigid_confirmed','open3d_icp')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def rms(q,r):return float(1000*np.sqrt(np.mean(np.sum((q-r)**2,axis=1))))

def tests():
    rng=np.random.default_rng(916701);p=rng.normal(size=(200,3));n=np.tile([0.,0.,1.],(200,1))
    j=np.column_stack((np.cross(p,n),n));truth=np.array([.001,-.002,0.,0.,0.,.005])
    beta,rank=robust_solve(j,j@truth)
    np.testing.assert_allclose(beta,truth,atol=1e-12);assert rank==3
    beta0,_=robust_solve(j,np.zeros(200));np.testing.assert_allclose(beta0,0,atol=1e-12)
    s=dict(center=p.mean(0),radius=1.);out=transform(p,truth,s)
    np.testing.assert_allclose(np.linalg.norm(out-np.roll(out,1,axis=0),axis=1),
        np.linalg.norm(p-np.roll(p,1,axis=0),axis=1),atol=1e-12)
    assert reference_id(np.array([2,1,2,1]))==1
    return dict(passed=5,total=5)

def variants(q,scan):
    anchor=scan==reference_id(scan);center=q[~anchor].mean(0)
    _,b=np.linalg.eigh((q[anchor]-q[anchor].mean(0)).T@(q[anchor]-q[anchor].mean(0)))
    yield 'unperturbed',0,q.copy()
    for seed in (916701,916709,916721):
        rng=np.random.default_rng(seed);sign=rng.choice([-1.,1.])
        angle=np.deg2rad(.05)*sign;axis=b[:,2]
        rot=Rotation.from_rotvec(angle*axis).as_matrix()
        for kind in ('normal_shift','rotation','shift_rotation','iid_same_rms'):
            out=q.copy()
            if kind in ('rotation','shift_rotation'):
                out[~anchor]=(q[~anchor]-center)@rot.T+center
            if kind in ('normal_shift','shift_rotation'):out[~anchor]+=.005*sign*b[:,0]
            if kind=='iid_same_rms':
                noise=rng.normal(size=out[~anchor].shape);noise*=.005/np.sqrt(np.mean(np.sum(noise**2,axis=1)))
                out[~anchor]+=noise
            yield kind,seed,out

def ambiguity():
    # Same observations under two distinct worlds; evaluator references differ.
    rng=np.random.default_rng(916733);uv=rng.uniform(-.1,.1,(512,2))
    base=np.column_stack((uv,np.zeros(512)))
    noise=rng.normal(0,.0002,(512,3))
    anchor=base+noise;moving=base+noise+[0,0,.006]
    q=np.concatenate((anchor,moving));scan=np.repeat([1,2],512)
    return q,scan,dict(single_surface_with_bias=np.concatenate((base,base)),
        true_two_surfaces=np.concatenate((base,base+[0,0,.006])))

def main():
    start=time.perf_counter();dest=Path(tempfile.mkdtemp(prefix='observation-model-v1-',dir=RUNS))
    print('RUN',dest,flush=True)
    src=dest/'source';src.mkdir()
    for p in HERE.iterdir():
        if p.is_file():shutil.copy2(p,src/p.name)
    inputs=sorted(PATCHES.glob('*/*.npz'))
    protected=inputs+sorted((HERE.parent/'published_outputs_v3').glob('*.py'))+sorted((RUNS/'published-outputs-v3').glob('*'))
    hashes={str(p):sha(p) for p in protected if p.is_file()}
    test_result=tests();pending=[];diagnostics=[]
    # Generate all candidates without passing clean coordinates to the operator.
    for file in inputs:
        with np.load(file) as a:q=a['xyz_world'].copy();scan=a['scan_id'].copy()
        anchor=scan==reference_id(scan)
        for kind,seed,visible in variants(q,scan):
            outputs,info=estimate(visible,scan)
            case=f'{file.stem}_{kind}_{seed}'
            np.savez_compressed(dest/f'{case}.npz',input=visible,scan_id=scan,**outputs)
            diagnostics.append(dict(case=case,patch=file.stem,kind=kind,seed=seed,**info))
            pending.append((file.stem,kind,seed,case,q,anchor,visible,outputs,info))
        print('GENERATED',file.stem,flush=True)
    aq,asc,refs=ambiguity();aout,ainfo=estimate(aq,asc)
    aout2,_=estimate(aq.copy(),asc.copy())
    for k in METHODS:np.testing.assert_allclose(aout[k],aout2[k],atol=1e-12,rtol=0)
    np.savez_compressed(dest/'ambiguity.npz',input=aq,scan_id=asc,**aout)
    # Evaluator accesses originals only after all candidates have been generated.
    baselines={(patch,k):out for patch,kind,seed,case,q,anchor,vis,outputs,info in pending if kind=='unperturbed' for k,out in outputs.items()}
    rows=[]
    for patch,kind,seed,case,q,anchor,vis,outputs,info in pending:
        for method,out in outputs.items():
            rows.append(dict(patch=patch,kind=kind,seed=seed,case=case,method=method,n=len(q),
                input_rms_mm=rms(vis,q),recovery_rms_mm=rms(out,q),moving_station_rms_mm=rms(out[~anchor],q[~anchor]),
                edit_rms_mm=rms(out,vis),response_to_own_baseline_rms_mm=rms(out,baselines[(patch,method)]),
                crossfit_accepted=info['accepted'],support=info['support'],rank=info['rank']))
    ar=[]
    for world,ref in refs.items():
        for method,out in aout.items():
            ar.append(dict(world=world,method=method,geometry_rms_mm=rms(out,ref),
                station_centroid_gap_mm=float(np.linalg.norm(out[asc==1].mean(0)-out[asc==2].mean(0))*1000)))
    summary=[]
    for kind in dict.fromkeys(r['kind'] for r in rows):
        for method in METHODS:
            rr=[r for r in rows if r['kind']==kind and r['method']==method]
            summary.append(dict(kind=kind,method=method,cases=len(rr),
                mean_recovery_rms_mm=float(np.mean([r['recovery_rms_mm'] for r in rr])),
                median_recovery_rms_mm=float(np.median([r['recovery_rms_mm'] for r in rr])),
                wins_vs_identity=sum(r['recovery_rms_mm']<r['input_rms_mm']-1e-8 for r in rr),
                confirmed=sum(r['crossfit_accepted'] for r in rr)))
    for p,h in hashes.items():assert sha(Path(p))==h,p
    result=dict(rows=rows,summary=summary,diagnostics=diagnostics,ambiguity_rows=ar,ambiguity_info=ainfo,
        tests=test_result,old_hashes_unchanged=True,input_hashes=hashes,
        scope='ETH3D injection recovery vs measured original, not independent physical geometry; no scan24 or 3DGS claim',
        seconds=time.perf_counter()-start)
    (dest/'RESULTS.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    with (dest/'RESULTS.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('SUMMARY',json.dumps(summary),flush=True)
    print('AMBIGUITY',json.dumps(ar),flush=True)
    print('COMPLETE',dest,flush=True)

if __name__=='__main__':main()

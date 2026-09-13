"""Fixed-support/fixed-association surface-family diagnostic. No GT in fit()."""
import sys, json, csv, time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.io import loadmat

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parent
sys.path[:0]=[str(PROJECT/'published_outputs_v1'),str(PROJECT/'published_outputs_v2')]
from test_outputs import read, audit, write_points, incident_faces, triangle_metrics
from eval_reference import xyz, observed, sha, REF
from spatial_filter import solve

V1=Path('/srv/slam-research/grf/map-denoise/runs/published-outputs-v1/scan24')
OUT=Path('/srv/slam-research/grf/map-denoise/runs/published-outputs-v3')
SOURCE=Path('/srv/slam-research/grf/map-denoise/datasets/published-outputs-v1/scan24_mesh.ply')
FAMILIES=('shared_affine','separate_affine','separate_quadratic','single_quadratic')

def design(uv,groups,family):
    # Scaling improves conditioning without changing the polynomial spaces.
    uv=uv/np.maximum(np.std(uv,axis=0),1e-12)
    u,v=uv.T
    linear=np.column_stack((np.ones(len(u)),u,v))
    quad=np.column_stack((linear,u*u,u*v,v*v))
    labels=np.unique(groups)
    membership=np.column_stack([groups==g for g in labels]).astype(float)
    if family=='shared_affine':return np.column_stack((membership,u,v))
    if family=='single_quadratic':return quad
    basis=linear if family=='separate_affine' else quad
    return np.column_stack([basis*membership[:,j,None] for j in range(len(labels))])

def fit(uv,y,groups,family):
    a=design(uv,groups,family)
    coef,_,rank,s=np.linalg.lstsq(a,y,rcond=1e-12)
    prediction=a@coef
    return prediction,dict(parameters=a.shape[1],rank=int(rank),
        condition=float(s[0]/s[-1]),input_residual_rms_spacing=float(np.sqrt(np.mean((prediction-y)**2))))

def unit_tests():
    rng=np.random.default_rng(4821)
    uv=rng.normal(size=(240,2));g=np.arange(240)%2
    for family in FAMILIES:
        a=design(uv,g,family);y=a@rng.normal(size=a.shape[1])
        p,_=fit(uv,y,g,family);np.testing.assert_allclose(p,y,atol=1e-10)
    y=rng.normal(size=240)
    errors=[np.sum((fit(uv,y,g,f)[0]-y)**2) for f in FAMILIES[:3]]
    assert errors[0]>=errors[1]-1e-10>=errors[2]-2e-10
    assert design(uv,np.zeros(240,int),'shared_affine').shape[1]==3
    return dict(tests=6,passed=6)

def main():
    start=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    tests=unit_tests()
    files=[SOURCE,REF/'cameras_surfels.npz',REF/'ObsMask24_10.mat',REF/'stl024_total.ply']
    files+=list(V1.glob('patch*_source_ids.npy'))+list(V1.glob('patch*.ply'))
    hashes={str(p):sha(p) for p in files}
    world,faces,_=audit(read(SOURCE));pending=[];diagnostics=[]
    for ix in range(3):
        ids=np.load(V1/f'patch{ix}_source_ids.npy');q=world[ids].copy()
        np.testing.assert_array_equal(q,xyz(V1/f'patch{ix}_identity.ply'))
        dist=cKDTree(q).query(q,k=2)[0][:,1];h=float(np.median(dist[dist>0]))
        center=q.mean(0);_,basis=np.linalg.eigh((q-center).T@(q-center))
        normal=basis[:,0];basis=basis[:,[2,1,0]];local=(q-center)@basis/h
        x=np.column_stack((np.ones(len(q)),local[:,:2]));y=local[:,2]
        old=solve(x,y,sigma=1.,iterations=36)
        reproduced=q+(old['prediction']-y)[:,None]*h*normal
        saved=xyz(V1/f'patch{ix}_spatial_s1.ply')
        np.testing.assert_allclose(reproduced,saved,rtol=0,atol=1e-12)
        groups=old['groups'];np.save(OUT/f'patch{ix}_frozen_groups.npy',groups)
        tri=incident_faces(faces,ids,len(world));results={}
        internal=np.all(np.isin(tri,ids),axis=1)
        for method in ('identity','plane','spatial_s1','apss4'):
            results[method]=xyz(V1/f'patch{ix}_{method}.ply')
        info=[]
        for family in FAMILIES:
            pred,di=fit(local[:,:2],y,groups,family)
            out=q+(pred-y)[:,None]*h*normal
            np.testing.assert_allclose((out-q)@basis[:,:2],0,atol=1e-12)
            assert np.isfinite(out).all()
            results[family]=out;info.append(dict(method=family,**di))
        nested=[a['input_residual_rms_spacing'] for a in info[:3]]
        assert nested[0]>=nested[1]-1e-10>=nested[2]-2e-10
        for method,out in results.items():
            write_points(OUT/f'patch{ix}_{method}.ply',out)
            row=dict(patch=ix,method=method,n=len(q),frozen_association_groups=int(old['k']),
                displacement_rms_scene_units=float(np.sqrt(np.mean(np.sum((out-q)**2,axis=1)))),
                **triangle_metrics(world,ids,out,tri))
            for region,take in (('interior',internal),('boundary',~internal)):
                row[region+'_reversals']=triangle_metrics(world,ids,out,tri[take])['orientation_reversals'] if np.any(take) else 0
            pending.append((row,q,out))
        diagnostics.append(dict(patch=ix,groups=np.bincount(groups).tolist(),spacing=h,
            original_reproduction_max=float(np.max(np.abs(reproduced-saved))),models=info))
        print('GENERATED',ix,diagnostics[-1],flush=True)
    # Candidate generation is complete. Independent reference is accessed only here.
    matrix=np.load(REF/'cameras_surfels.npz')['scale_mat_0'].astype(float)
    transform=lambda a:a@matrix[:3,:3].T+matrix[:3,3]
    tree=cKDTree(xyz(REF/'stl024_total.ply'));obs=loadmat(REF/'ObsMask24_10.mat')
    rows=[]
    for row,q,out in pending:
        mask=observed(transform(q),obs);d=tree.query(transform(out))[0][mask]
        row.update(observed_points=int(mask.sum()),mean_mm=float(d.mean()),p95_mm=float(np.quantile(d,.95)),
            displacement_rms_mm=row.pop('displacement_rms_scene_units')*float(matrix[0,0]))
        rows.append(row)
    summary=[]
    for method in dict.fromkeys(r['method'] for r in rows):
        rr=[r for r in rows if r['method']==method]
        summary.append(dict(method=method,mean_mm=float(np.average([r['mean_mm'] for r in rr],weights=[r['observed_points'] for r in rr])),
            orientation_reversals=sum(r['orientation_reversals'] for r in rr),
            interior_reversals=sum(r['interior_reversals'] for r in rr),
            boundary_reversals=sum(r['boundary_reversals'] for r in rr)))
    for p,hsh in hashes.items():assert sha(p)==hsh,p
    result=dict(rows=rows,summary=summary,diagnostics=diagnostics,tests=tests,
        old_hashes_unchanged=True,input_hashes=hashes,seconds=time.perf_counter()-start,
        evaluation='Fixed original observed support; local one-way STL distance; v2 conditional coordinate transform; not official DTU score',
        fitting='No reference coordinates, true normals or associations; labels from frozen spatial_s1; all candidates retained')
    (OUT/'RESULTS.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    with (OUT/'RESULTS.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('SUMMARY',json.dumps(summary),flush=True)

if __name__=='__main__':main()

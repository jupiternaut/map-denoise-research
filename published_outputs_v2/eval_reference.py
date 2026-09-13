"""Frozen v1 local outputs against official DTU scan24 STL observations.

Not the official DTU leaderboard protocol: no mesh surface resampling, local
point-to-reference accuracy only, no full-scene completeness claim.
"""
import sys,json,csv,hashlib,time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.io import loadmat
sys.path.insert(0,'/srv/slam-research/grf/map-denoise/tools/published-plyfile')
from plyfile import PlyData
REF=Path('/srv/slam-research/grf/map-denoise/datasets/published-outputs-v2-reference')
V1=Path('/srv/slam-research/grf/map-denoise/runs/published-outputs-v1/scan24')
OUT=Path('/srv/slam-research/grf/map-denoise/runs/published-outputs-v2/scan24')

def xyz(p):
    a=PlyData.read(str(p),known_list_len={'face':{'vertex_indices':3}})['vertex'].data
    return np.column_stack([a[k] for k in ('x','y','z')]).astype(float)

def observed(q,obs):
    grid=np.around((q-obs['BB'][:1])/obs['Res']).astype(np.int32)
    valid=np.all((grid>=0)&(grid<np.asarray(obs['ObsMask'].shape)),axis=1)
    keep=np.zeros(len(q),bool);g=grid[valid];keep[valid]=obs['ObsMask'][g[:,0],g[:,1],g[:,2]].astype(bool)
    return keep

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def main():
    OUT.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    cameras=np.load(REF/'cameras_surfels.npz');m=cameras['scale_mat_0'].astype(float)
    for k in cameras.files:
        if k.startswith('scale_mat_') and 'inv' not in k:np.testing.assert_array_equal(cameras[k],m)
    assert np.allclose(m[:3,:3],np.eye(3)*m[0,0])
    transform=lambda x:x@m[:3,:3].T+m[:3,3]
    gt=xyz(REF/'stl024_total.ply');tree=cKDTree(gt);obs=loadmat(REF/'ObsMask24_10.mat')
    rows=[];hashes={}
    for patch in range(3):
        original=transform(xyz(V1/f'patch{patch}_identity.ply'));keep=observed(original,obs)
        for method in ('identity','plane','spatial_s05','spatial_s1','spatial_s2','apss4'):
            path=V1/f'patch{patch}_{method}.ply';hashes[str(path)]=sha(path)
            q=transform(xyz(path));d=tree.query(q)[0];dk=d[keep]
            row=dict(patch=patch,method=method,n=len(q),fixed_observed_points=int(keep.sum()),
                all_points_mean_mm=float(d.mean()),all_points_p95_mm=float(np.quantile(d,.95)))
            if len(dk):
                row.update(observed_mean_mm=float(dk.mean()),observed_median_mm=float(np.median(dk)),
                    observed_p95_mm=float(np.quantile(dk,.95)),observed_fraction_below_1mm=float(np.mean(dk<1)),
                    observed_fraction_ge_20mm=float(np.mean(dk>=20)),
                    observed_mean_below20mm=float(dk[dk<20].mean()) if np.any(dk<20) else None)
            else:row.update(status='NO_ORIGINAL_OBSERVED_SUPPORT')
            rows.append(row);np.save(OUT/f'patch{patch}_{method}_distances_mm.npy',d)
            print(row,flush=True)
    summary=[]
    for method in ('identity','plane','spatial_s05','spatial_s1','spatial_s2','apss4'):
        rr=[r for r in rows if r['method']==method and r['fixed_observed_points']]
        summary.append(dict(method=method,observed_points=sum(r['fixed_observed_points'] for r in rr),
            observed_mean_mm=float(np.average([r['observed_mean_mm'] for r in rr],weights=[r['fixed_observed_points'] for r in rr]))))
    result=dict(rows=rows,summary=summary,scale_matrix=m.tolist(),matrix_source='turandai/gaussian-surfels-dtu/scan24/cameras.npz',
        coordinate_status='public DTU normalization metadata; no GT-based ICP or fitted scale; exact GeoSVR input cameras not acquired',
        gt_points=len(gt),ground_truth_source='official DTU Points.zip/Points/stl/stl024_total.ply',
        fixed_mask='official ObsMask24_10 evaluated on original points; same rows for every candidate',
        scope='local point-to-STL distances conditional on the metadata coordinate mapping; not official DTU overall score',
        no_claims=['full-scene completeness','millimeter OSM accuracy','same-camera photometric improvement','OSM import approval'],
        seconds=time.perf_counter()-start,output_hashes=hashes)
    for path,h in hashes.items():assert sha(path)==h
    (OUT/'GEOMETRY_RESULTS.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/'GEOMETRY_RESULTS.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    print('SUMMARY',summary,flush=True)
if __name__=='__main__':main()

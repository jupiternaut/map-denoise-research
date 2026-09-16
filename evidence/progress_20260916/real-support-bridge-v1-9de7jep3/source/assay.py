"""Photo-fixed real-object scope assay. Reference is diagnostic only."""
from pathlib import Path
import sys, json, hashlib, tempfile, time, socket, shutil
import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT/'published_outputs_v2'), str(ROOT/'loss_alignment_v23')]
from eval_reference import xyz
import calibration

DATA = Path('/srv/slam-research/grf/map-denoise/datasets')
RUNS = DATA.parent/'runs'
PHOTO = DATA/'loss-alignment-v23/scan24/image/0000.png'
CAMERA = DATA/'real-closure-v21/cameras_geosvr_linked.npz'
MESH = DATA/'published-outputs-v1/scan24_mesh.ply'
REFERENCE = DATA/'published-outputs-v2-reference/stl024_total.ply'
ROIS = {'window_a': (640,371,736,472), 'window_b': (750,432,842,526),
        'wall_control': (720,419,756,450)}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()

def save(path, value):
    def convert(v):
        if isinstance(v, np.ndarray): return v.tolist()
        if isinstance(v, np.generic): return v.item()
        return str(v)
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False, default=convert)

def project(q, P):
    h = q@P[:,:3].T + P[:,3]
    return h[:,:2]/h[:,2,None], h[:,2]

def rectangle(uv, z, box):
    l,t,r,b = box
    return (z>0)&(uv[:,0]>=l)&(uv[:,0]<r)&(uv[:,1]>=t)&(uv[:,1]<b)

def range_gap(q, camera_center, ray):
    delta = q-camera_center
    distance = np.linalg.norm(delta,axis=1)
    rays = np.column_stack([np.broadcast_to(camera_center,q.shape),delta/distance[:,None]])
    hit = ray.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))['t_hit'].numpy()
    return np.where(np.isfinite(hit), abs(hit-distance), 1e12)

def planes(q, tolerance, max_planes=6):
    """Greedy descriptor, not physical object segmentation or an estimator baseline."""
    o3d.utility.random.seed(932413)
    remaining = np.arange(len(q)); descriptors = []; labels = np.full(len(q),-1,int)
    for k in range(max_planes):
        if len(remaining)<30: break
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(q[remaining]))
        coefficients, local_ids = cloud.segment_plane(distance_threshold=tolerance,
            ransac_n=3, num_iterations=1000, probability=.999)
        local_ids = np.asarray(local_ids,int)
        if len(local_ids)<30: break
        ids = remaining[local_ids]; labels[ids] = k
        n = np.asarray(coefficients[:3],float); d = float(coefficients[3])
        if n[np.argmax(abs(n))]<0: n=-n;d=-d
        residual = q[ids]@n+d
        descriptors.append(dict(normal=n,offset=d,n=len(ids),fraction=len(ids)/len(q),
            residual_rms_mm=float(np.sqrt(np.mean(residual**2))),
            residual_p95_mm=float(np.quantile(abs(residual),.95)),center=q[ids].mean(0)))
        mask = np.ones(len(remaining),bool); mask[local_ids] = False; remaining=remaining[mask]
    return descriptors, labels

def assess(descriptors, center):
    if len(descriptors)<2:
        return dict(provisional_parallel_pair=False,reason='fewer than two supported planes')
    a,b = descriptors[:2]
    dot = float(np.dot(a['normal'],b['normal'])); sign = 1 if dot>=0 else -1
    angle = float(np.degrees(np.arccos(np.clip(abs(dot),0,1))))
    n = a['normal']
    # Intersections of both fitted planes with the same line through ROI center.
    t1 = -(center@n+a['offset'])
    t2 = -(center@b['normal']+b['offset'])/(n@b['normal']) if abs(n@b['normal'])>1e-9 else None
    gap = None if t2 is None else abs(float(t2-t1))
    criteria = dict(both_at_least_15pct=min(a['fraction'],b['fraction'])>=.15,
        union_at_least_70pct=a['fraction']+b['fraction']>=.70,
        normals_within_5deg=angle<=5, separation_at_least_05mm=gap is not None and gap>=.5)
    return dict(provisional_parallel_pair=all(criteria.values()),criteria=criteria,
        angle_deg=angle,line_separation_at_center_mm=gap,
        top_two_coverage=a['fraction']+b['fraction'],fractions=[a['fraction'],b['fraction']])

def main():
    assert socket.gethostname()=='liekkas'
    assert ROOT == Path('/home/grf/Documents/Codex/2026-09-12/map-denoise-dataset-pilot-v1')
    dest=Path(tempfile.mkdtemp(prefix='real-support-bridge-v1-',dir=RUNS))
    print('RUN',dest,flush=True)
    for name in ('inputs','reference','source','figures'): (dest/name).mkdir()
    files=[p for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.parts
           and '__pycache__' not in p.parts and 'evidence' not in p.parts]
    folder=DATA/'loss-alignment-v23/scan24/sparse/0'
    files += [PHOTO,CAMERA,MESH,REFERENCE]+list(folder.glob('*.bin'))
    lock={str(p):sha(p) for p in files};save(dest/'SOURCE_LOCK.json',lock)
    for p in HERE.glob('*'):
        if p.is_file():shutil.copyfile(p,dest/'source'/p.name)
    start=time.perf_counter()
    cal,_=calibration.load(folder);c=cal['0000.png'];scale=np.load(CAMERA)['scale_mat_0'].astype(float)
    im=np.asarray(Image.open(PHOTO));assert (im.shape[1],im.shape[0])==(c['width'],c['height'])
    P=c['P']@np.linalg.inv(scale);center=-np.linalg.solve(P[:,:3],P[:,3])
    linked=np.load(CAMERA)['world_mat_0'];linked_center=-np.linalg.solve(linked[:3,:3],linked[:3,3])
    np.testing.assert_allclose(center,linked_center,atol=1e-3,rtol=0)
    mesh=o3d.io.read_triangle_mesh(str(MESH));mesh.transform(scale)
    q=np.asarray(mesh.vertices).copy()
    ray=o3d.t.geometry.RaycastingScene();ray.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    uv,z=project(q,P);jobs=[]
    for name,box in ROIS.items():
        ids=np.flatnonzero(rectangle(uv,z,box));gap=range_gap(q[ids],center,ray)
        ids=ids[gap<=.75]
        assert len(ids)>=30,(name,len(ids))
        np.savez_compressed(dest/'inputs'/f'{name}.npz',points_mm=q[ids],source_ids=ids,uv=uv[ids])
        jobs.append(dict(name=name,box=box,mesh_points=len(ids),input=str(dest/'inputs'/f'{name}.npz')))
    save(dest/'INPUTS_FIXED_BEFORE_REFERENCE.json',dict(jobs=jobs,P=P,camera_center=center,
        calibration_center_error_mm=float(np.linalg.norm(center-linked_center))))
    # No fitting method has been run. This is the explicitly reference-informed task assay.
    ref=xyz(REFERENCE);ruv,rz=project(ref,P);report=[]
    for job in jobs:
        ids=np.flatnonzero(rectangle(ruv,rz,job['box']));gap=range_gap(ref[ids],center,ray)
        counts={str(t):int(np.sum(gap<=t)) for t in (5,10,20)}
        ids=ids[gap<=20];r=ref[ids];assert len(r)>=30
        descriptors={};labels={};assessments={}
        for tol in (.15,.3,.6):
            d,l=planes(r,tol);descriptors[str(tol)]=d;labels[str(tol)]=l
            assessments[str(tol)]=assess(d,r.mean(0))
        inp=np.load(job['input'])['points_mm'];basis=np.linalg.eigh(np.cov(inp.T))[1][:,::-1]
        h=(r-inp.mean(0))@basis;surface_rms=float(np.sqrt(np.mean(h[:,2]**2)))
        np.savez_compressed(dest/'reference'/f'{job["name"]}.npz',points_mm=r,source_ids=ids,uv=ruv[ids],
            labels_015=labels['0.15'],labels_030=labels['0.3'],labels_060=labels['0.6'],frame=basis,origin=inp.mean(0))
        row=dict(**job,reference_points=len(r),reference_range_band_counts=counts,
            descriptors=descriptors,assessments=assessments,reference_pca_height_rms_mm=surface_rms,
            input_reference_distance_mean_mm=float(cKDTree(r).query(inp)[0].mean()))
        report.append(row);print(job['name'],len(inp),len(r),assessments['0.3'],flush=True)
    assert all(sha(p)==h for p,h in lock.items())
    save(dest/'ASSAY.json',dict(regions=report,seconds=time.perf_counter()-start,
         history_unchanged=True,reference_used_only_for_task_diagnostics=True))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig,ax=plt.subplots(figsize=(10,7.5));ax.imshow(im)
    for name,box in ROIS.items():
        l,t,r,b=box;ax.add_patch(Rectangle((l,t),r-l,b-t,fill=False,color='cyan',lw=1.6));ax.text(l,t-5,name,color='blue',fontsize=9,bbox=dict(fc='white',alpha=.8,ec='none'))
    ax.set_title('Photo-fixed semantic ROIs; no method-score selection');ax.set_axis_off();fig.tight_layout()
    fig.savefig(dest/'figures/regions.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(3,3,figsize=(13,10))
    for i,row in enumerate(report):
        name=row['name'];data=np.load(dest/'reference'/f'{name}.npz');r=data['points_mm'];l=data['labels_030'];loc=(r-data['origin'])@data['frame']
        inp=np.load(row['input'])['points_mm'];iloc=(inp-data['origin'])@data['frame']
        sample=np.linspace(0,len(r)-1,min(8000,len(r)),dtype=int)
        axs[i,0].scatter(loc[sample,0],loc[sample,1],c=l[sample],s=2,cmap='tab10',vmin=-1,vmax=8)
        axs[i,1].scatter(loc[sample,0],loc[sample,2],c=l[sample],s=2,cmap='tab10',vmin=-1,vmax=8)
        axs[i,1].scatter(iloc[:,0],iloc[:,2],c='black',s=.4,alpha=.25)
        axs[i,2].hist(loc[:,2],bins=70,density=True,alpha=.6,label='laser reference')
        axs[i,2].hist(iloc[:,2],bins=70,density=True,alpha=.6,label='input mesh')
        axs[i,0].set(title=name+' reference plane labels',xlabel='u (mm)',ylabel='v (mm)')
        axs[i,1].set(title='Reference cross-section + black input',xlabel='u (mm)',ylabel='height (mm)')
        axs[i,2].set(title='Height distribution, not physical layer GT',xlabel='height (mm)');axs[i,2].legend(fontsize=8)
    fig.tight_layout();fig.savefig(dest/'figures/reference_structure.png',dpi=170);plt.close(fig)
    print('DONE',dest,flush=True)

if __name__=='__main__':main()

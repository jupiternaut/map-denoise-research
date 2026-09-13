"""Real published artifacts: full schema audit and explicitly local transfer probes."""
import sys, json, time, resource, traceback, hashlib, csv
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import expit
sys.path.insert(0,'/srv/slam-research/grf/map-denoise/tools/published-plyfile')
from plyfile import PlyData, PlyElement
HERE=Path(__file__).resolve().parent; PROJECT=HERE.parent
sys.path[:0]=[str(PROJECT/'exploration_v11'),str(PROJECT/'exploration_v12')]
from spatial_filter import solve
from external import project
DATA=Path('/srv/slam-research/grf/map-denoise/datasets/published-outputs-v1')
OUT=Path('/srv/slam-research/grf/map-denoise/runs/published-outputs-v1')

def save(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False))

def write_points(path,q):
    a=np.empty(len(q),dtype=[(k,'f8') for k in ('x','y','z')])
    for j,k in enumerate(('x','y','z')):a[k]=q[:,j]
    PlyData([PlyElement.describe(a,'vertex')],text=False).write(str(path))

def read(path):
    return PlyData.read(str(path),mmap='r',known_list_len={'face':{'vertex_indices':3}})

def audit(ply):
    v=ply['vertex'].data; names=v.dtype.names;count=len(v)
    xyz=np.empty((count,3),dtype=np.float64)
    for j,k in enumerate(('x','y','z')):xyz[:,j]=v[k]
    assert np.isfinite(xyz).all()
    result=dict(vertices=count,fields=list(names),bbox_min=xyz.min(0).tolist(),bbox_max=xyz.max(0).tolist(),
                finite_xyz=True,has_scan_id='scan_id' in names,metric_units_verified=False)
    face=None
    if 'face' in ply:
        face=ply['face'].data['vertex_indices']
        assert face.min()>=0 and face.max()<count
        result.update(faces=len(face),face_indices_valid=True)
    if 'opacity' in names:
        for k in names:
            assert np.isfinite(v[k]).all(),k
        result['opacity_quantiles']=np.quantile(expit(v['opacity']),[0,.01,.5,.99,1]).tolist()
        scales=np.column_stack([v[f'scale_{j}'] for j in range(3)])
        result['log_scale_quantiles']=np.quantile(scales,[0,.01,.5,.99,1]).tolist()
        quat=np.column_stack([v[f'rot_{j}'] for j in range(4)])
        result['zero_quaternions']=int(np.sum(np.linalg.norm(quat,axis=1)==0))
    return xyz,face,result

def incident_faces(face,ids,n):
    if face is None:return None
    mask=np.zeros(n,bool);mask[ids]=True;chunks=[]
    for start in range(0,len(face),200000):
        f=face[start:start+200000];chunks.append(np.array(f[np.any(mask[f],axis=1)],copy=True))
    return np.concatenate(chunks)

def triangle_metrics(xyz,ids,q,tri):
    if tri is None:return {}
    old=xyz[tri];new=old.copy();order=np.argsort(ids);sortedids=ids[order]
    pos=np.searchsorted(sortedids,tri);match=(pos<len(ids)) & (sortedids[np.minimum(pos,len(ids)-1)]==tri)
    new[match]=q[order[pos[match]]]
    oldn=np.cross(old[:,1]-old[:,0],old[:,2]-old[:,0]);newn=np.cross(new[:,1]-new[:,0],new[:,2]-new[:,0])
    a=np.linalg.norm(oldn,axis=1);b=np.linalg.norm(newn,axis=1);valid=a>np.finfo(float).eps
    return dict(incident_triangles=len(tri),orientation_reversals=int(np.sum(np.sum(oldn[valid]*newn[valid],axis=1)<=0)),
                area_ratio_min=float(np.min(b[valid]/a[valid])),area_ratio_max=float(np.max(b[valid]/a[valid])))

def measures(q,out,h):
    _,nn=cKDTree(q).query(q,k=9);d0=np.linalg.norm(q[:,None]-q[nn[:,1:]],axis=2);d1=np.linalg.norm(out[:,None]-out[nn[:,1:]],axis=2)
    disp=np.linalg.norm(out-q,axis=1)/h;ratio=d1/np.maximum(d0,1e-15)
    return dict(displacement_median_spacing=float(np.median(disp)),displacement_p95_spacing=float(np.quantile(disp,.95)),
                displacement_max_spacing=float(disp.max()),edge_ratio_min=float(ratio.min()),edge_ratio_max=float(ratio.max()),
                moved_points=int(np.sum(disp>1e-7)),finite=bool(np.isfinite(out).all()))

def test_scene(name,path):
    t=time.monotonic();ply=read(path);xyz,face,meta=audit(ply);dest=OUT/name;dest.mkdir(parents=True,exist_ok=True)
    meta['bytes']=path.stat().st_size;meta['source_path']=str(path)
    print(name,'audit',meta['vertices'],meta.get('faces'),flush=True)
    tree=cKDTree(xyz);patches=[];rows=[]
    for ix,quantile in enumerate((.25,.5,.75)):
        anchor=int(tree.query(np.quantile(xyz,quantile,axis=0))[1]);_,ids=tree.query(xyz[anchor],k=1024)
        q=xyz[ids].copy();d=cKDTree(q).query(q,k=2)[0][:,1];h=float(np.median(d[d>0]));assert h>0
        center=q.mean(0);_,basis=np.linalg.eigh((q-center).T@(q-center));normal=basis[:,0];basis=basis[:,[2,1,0]]
        local=(q-center)@basis/h;x=np.column_stack((np.ones(len(q)),local[:,:2]));y=local[:,2]
        tri=incident_faces(face,ids,len(xyz));np.save(dest/f'patch{ix}_source_ids.npy',ids)
        patches.append(dict(patch=ix,anchor=anchor,spacing_scene_units=h,points=len(ids)))
        outputs={}
        for method in ('identity','plane','spatial_s05','spatial_s1','spatial_s2','apss4'):
            st=time.monotonic();row=dict(scene=name,patch=ix,method=method,n=len(q),spacing_scene_units=h)
            try:
                k=None
                if method=='identity':out=q.copy()
                elif method=='apss4':out=project(q,'apss',4.)
                else:
                    if method=='plane':prediction=x@np.linalg.lstsq(x,y,rcond=None)[0];k=1
                    else:
                        sigma={'spatial_s05':.5,'spatial_s1':1.,'spatial_s2':2.}[method]
                        model=solve(x,y,sigma,iterations=36);prediction=model['prediction'];k=model['k']
                    out=q+(prediction-y)[:,None]*h*normal
                assert np.isfinite(out).all()
                row.update(status='ok',model_layers=k,**measures(q,out,h),**triangle_metrics(xyz,ids,out,tri))
                write_points(dest/f'patch{ix}_{method}.ply',out);outputs[method]=out
            except Exception:
                row.update(status='error',error=traceback.format_exc())
            row['seconds']=time.monotonic()-st;rows.append(row)
            print(name,ix,method,row['status'],round(row['seconds'],2),flush=True)
        plot(dest/f'patch{ix}_comparison.png',q,outputs,center,basis,h,name)
    meta['patches']=patches
    if 'opacity' in ply['vertex'].data.dtype.names:
        v=ply['vertex'].data;keep=expit(v['opacity'])>=.01;filtered=v[keep]
        outfile=dest/'room_opacity001_baseline.ply'
        PlyData([PlyElement.describe(filtered,'vertex')],text=False).write(str(outfile))
        re=read(outfile)['vertex'].data
        assert re.dtype.names==v.dtype.names
        for field in v.dtype.names:np.testing.assert_array_equal(re[field],v[field][keep])
        meta['opacity_baseline']=dict(threshold=.01,retained=int(keep.sum()),removed=int((~keep).sum()),
            fraction_removed=float((~keep).mean()),all_retained_attributes_exact=True,path=str(outfile),rendering_tested=False)
    meta['elapsed_seconds']=time.monotonic()-t;meta['process_peak_rss_KiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    save(dest/'audit.json',meta);save(dest/'tests.json',rows)
    return meta,rows

def plot(path,q,outputs,center,basis,h,name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(12,6),sharex=True,sharey=True)
    for ax,(method,out) in zip(axes.ravel(),outputs.items()):
        p=(out-center)@basis/h;ax.scatter(p[:,0],p[:,2],s=2);ax.set_title(method);ax.set_xlabel('PCA tangent / spacing');ax.set_ylabel('PCA normal / spacing')
    fig.suptitle(name+' — geometric cross-section, NOT a rendered image or GT')
    fig.tight_layout();fig.savefig(path,dpi=140);plt.close(fig)

def main():
    OUT.mkdir(parents=True,exist_ok=True);allrows=[];audits=[]
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--scene',choices=('room','scan24','Courthouse'));args=parser.parse_args()
    scenes=[('room',DATA/'room/point_cloud/iteration_30000/point_cloud.ply'),('scan24',DATA/'scan24_mesh.ply'),('Courthouse',DATA/'Courthouse_mesh.ply')]
    for name,path in scenes:
        if args.scene and name!=args.scene:continue
        meta,rows=test_scene(name,path);audits.append(meta);allrows.extend(rows)
    print('COMPLETE',len(allrows),'runs',flush=True)

if __name__=='__main__':main()

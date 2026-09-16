"""Seal fixed proposals and proxy choices before independent reference evaluation."""
from pathlib import Path
import sys,json,hashlib,tempfile,time,socket,shutil
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from PIL import Image
import open3d as o3d
import loss_operator as op
import calibration
ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'published_outputs_v2'))
from eval_reference import xyz
DATA=Path('/srv/slam-research/grf/map-denoise/datasets');RUNS=DATA.parent/'runs';OLD=RUNS/'reconstruction-v22-qayc8gft'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False,default=lambda v:v.tolist() if isinstance(v,(np.ndarray,np.generic)) else str(v))

def setup(scene):
    camera=(DATA/'real-closure-v21/cameras_geosvr_linked.npz' if scene==24 else DATA/'reconstruction-v22-scan37/cameras.npz')
    mesh=(DATA/'published-outputs-v1/scan24_mesh.ply' if scene==24 else DATA/'reconstruction-v22-scan37/scan37_mesh.ply')
    cam=np.load(camera);paths=sorted((DATA/f'loss-alignment-v23/scan{scene}/image').glob('*.png'))
    assert len(paths)==49,(scene,len(paths));assert [int(p.stem) for p in paths]==list(range(49))
    images=[np.asarray(Image.open(p).convert('RGB')) for p in paths]
    folder=DATA/f'loss-alignment-v23/scan{scene}/sparse/0';cal,points=calibration.load(folder)
    assert set(cal)=={p.name for p in paths}
    scale=cam['scale_mat_0'].astype(float);matrices=[];center_errors=[];reprojection=[]
    for i,p in enumerate(paths):
        c=cal[p.name];h,w=images[i].shape[:2];assert (w,h)==(c['width'],c['height'])
        # COLMAP coordinates are exactly the published mesh's normalized frame.
        original=cam[f'world_mat_{i}'];cw=-np.linalg.solve(original[:3,:3],original[:3,3])
        cn=(cw-scale[:3,3])/scale[0,0];center_errors.append(float(np.linalg.norm(cn+c['R'].T@c['t'])))
        assert center_errors[-1]<1e-4
        matrix=np.eye(4);matrix[:3]=c['P']@np.linalg.inv(scale);matrices.append(matrix)
        tracks=c['tracks'];tracks=tracks[np.array([i in points for i in tracks['id']])]
        point=np.array([points[i] for i in tracks['id']]);uv,_=op.project(point,c['P'])
        reprojection.extend(np.linalg.norm(uv-np.column_stack([tracks['x'],tracks['y']]),axis=1))
    matrices=np.asarray(matrices);centers=np.array([-np.linalg.solve(m[:3,:3],m[:3,3]) for m in matrices])
    assert np.quantile(reprojection,.95)<5.
    m=o3d.io.read_triangle_mesh(str(mesh));m.transform(cam['scale_mat_0'].astype(float))
    ray=o3d.t.geometry.RaycastingScene();ray.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))
    return images,matrices,centers,ray,dict(camera=str(camera),camera_sha256=sha(camera),mesh=str(mesh),mesh_sha256=sha(mesh),image_shapes=[im.shape for im in images],images=[dict(path=str(p),sha256=sha(p)) for p in paths],calibration=[dict(path=str(p),sha256=sha(p)) for p in sorted(folder.glob('*.bin'))],normalized_center_max_difference=max(center_errors),sparse_reprojection_median_px=float(np.median(reprojection)),sparse_reprojection_p95_px=float(np.quantile(reprojection,.95)),matrices=matrices)

def support_for(q,normal,images,matrices,centers,ray):
    masks=[];angles=[];gaps=[]
    for im,m,c in zip(images,matrices,centers):
        uv,z=op.project(q,m);_,valid=op.sample(im,uv)
        direction=q-c;distance=np.linalg.norm(direction,axis=1);direction/=distance[:,None]
        rays=np.column_stack([np.broadcast_to(c,q.shape),direction]).astype(np.float32)
        hit=ray.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy().astype(float)
        gap=abs(hit-distance);cosine=abs(np.sum(normal*direction,axis=1))
        masks.append(valid&(z>0)&(gap<=.5)&(cosine>=.2));angles.append(cosine);gaps.append(np.where(np.isfinite(gap),gap,1e9))
    masks=np.asarray(masks);angles=np.asarray(angles)
    # At most four front-facing/high-incidence supports, frozen on original q.
    order=np.argsort(np.where(masks,angles,-1),axis=0)[-4:]
    keep=np.zeros_like(masks);np.put_along_axis(keep,order,True,axis=0);keep&=masks
    return keep,dict(eligible_points=int(np.sum(keep.sum(0)>=2)),available_views_per_point=masks.sum(0),chosen_views_per_point=keep.sum(0),ray_gap_mm=np.asarray(gaps))

def evaluate(q,out,tree,reference,keep):
    a=tree.query(out[keep])[0];b=cKDTree(out).query(reference)[0];p=float(np.mean(a<=1));r=float(np.mean(b<=1))
    return dict(accuracy_mm=float(a.mean()),completeness_mm=float(b.mean()),recall=r,precision=p,fscore=2*p*r/(p+r) if p+r else 0.,displacement_rms_mm=float(np.sqrt(np.mean(np.sum((out-q)**2,axis=1)))))

def main():
    assert socket.gethostname()=='liekkas';dest=Path(tempfile.mkdtemp(prefix='loss-alignment-v23-',dir=RUNS));print('RUN',dest,flush=True)
    for name in ('outputs','source','support','figures'):(dest/name).mkdir()
    lock={str(p):sha(p) for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts and 'evidence' not in p.parts}
    save(dest/'SOURCE_LOCK.json',lock)
    for p in HERE.glob('*'):
        if p.is_file():shutil.copyfile(p,dest/'source'/p.name)
    start=time.perf_counter();sealed=[]
    for scene,phase in [(24,'development'),(37,'confirmation')]:
        images,matrices,centers,ray,provenance=setup(scene);save(dest/f'scan{scene}_PROVENANCE.json',provenance)
        for bundle in json.loads((OLD/f'{phase}_SEALED.json').read_text()):
            job=bundle['job'];q=np.load(job['input']);assert sha(job['input'])==job['sha256'];outputs,fit,normal=op.actions(q)
            support,diag=support_for(q,normal,images,matrices,centers,ray);photo={};details={}
            for name,out in outputs.items():photo[name],details[name]=op.photo_loss(out,images,matrices,support)
            choices=dict(fit_selector=op.select(fit),photo_selector=op.select(photo))
            for name,choice in choices.items():outputs[name]=outputs[choice].copy()
            records=[]
            for name,out in outputs.items():
                p=dest/'outputs'/f'{job["case"]}__{name}.npy';np.save(p,out);records.append(dict(method=name,path=str(p),sha256=sha(p)))
            np.savez_compressed(dest/'support'/f'{job["case"]}.npz',support=support,normal=normal,**diag)
            sealed.append(dict(scene=scene,job=job,records=records,fit_loss=fit,photo_loss=photo,choices=choices,photo_details=details,eligible_points=diag['eligible_points']))
            print(scene,job['case'],diag['eligible_points'],choices,flush=True)
        del images,ray
    save(dest/'SEALED_BEFORE_GT.json',sealed)
    rows=[]
    for scene in (24,37):
        ref=xyz(DATA/'published-outputs-v2-reference/stl024_total.ply' if scene==24 else DATA/'reconstruction-v22-scan37/stl037_total.ply');tree=cKDTree(ref)
        for bundle in [b for b in sealed if b['scene']==scene]:
            case=bundle['job']['case'];q=np.load(bundle['job']['input']);ev=np.load(OLD/'evaluation'/f'{case}.npz');gt=ref[ev['reference_ids']];keep=ev['input_mask']
            for rec in bundle['records']:
                assert sha(rec['path'])==rec['sha256'];out=np.load(rec['path']);row=dict(scene=scene,case=case,method=rec['method'],**evaluate(q,out,tree,gt,keep));rows.append(row)
    save(dest/'RESULTS.json',rows);summary={};alignment=[]
    for scene in (24,37):
        rr=[r for r in rows if r['scene']==scene];methods=list(dict.fromkeys(r['method'] for r in rr));bycase={}
        for r in rr:bycase.setdefault(r['case'],{})[r['method']]=r
        stats={m:{k:float(np.mean([r[k] for r in rr if r['method']==m])) for k in ('accuracy_mm','completeness_mm','recall','fscore','displacement_rms_mm')} for m in methods}
        for b in [b for b in sealed if b['scene']==scene]:
            cc=bycase[b['job']['case']];base=cc['identity'];pool=list(b['fit_loss']);best=min(pool,key=lambda m:cc[m]['accuracy_mm'])
            entry=dict(scene=scene,case=b['job']['case'],oracle_choice=best,oracle_accuracy_mm=cc[best]['accuracy_mm'],identity_accuracy_mm=base['accuracy_mm'],eligible_points=b['eligible_points'])
            for proxy in ('fit','photo'):
                losses=b[f'{proxy}_loss'];valid=[m for m in pool if losses[m] is not None]
                x=[losses[m] for m in valid];y=[cc[m]['accuracy_mm'] for m in valid]
                corr=float(spearmanr(x,y).statistic) if len(set(x))>1 and len(set(y))>1 else None
                choice=b['choices'][proxy+'_selector'];entry[proxy]=dict(spearman=corr,choice=choice,regret_mm=cc[choice]['accuracy_mm']-cc[best]['accuracy_mm'],accuracy_delta_mm=cc[choice]['accuracy_mm']-base['accuracy_mm'],recall_delta=cc[choice]['recall']-base['recall'])
            alignment.append(entry)
        ar=[a for a in alignment if a['scene']==scene];summary[str(scene)]=dict(patches=len(bycase),methods=stats,oracle_accuracy_mm=float(np.mean([a['oracle_accuracy_mm'] for a in ar])),alignment={proxy:dict(mean_spearman=float(np.mean([a[proxy]['spearman'] for a in ar if a[proxy]['spearman'] is not None])),mean_regret_mm=float(np.mean([a[proxy]['regret_mm'] for a in ar])),wins=sum(a[proxy]['accuracy_delta_mm']<0 for a in ar),choices={m:sum(a[proxy]['choice']==m for a in ar) for m in methods}) for proxy in ('fit','photo')})
    save(dest/'ALIGNMENT.json',alignment)
    assert all(sha(p)==h for p,h in lock.items())
    save(dest/'SUMMARY.json',dict(scenes=summary,output_count=len(rows),wall_seconds=time.perf_counter()-start,source_hashes_unchanged=True))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for si,scene in enumerate((24,37)):
        for pi,proxy in enumerate(('fit','photo')):
            ax=axes[si,pi]
            for b in [b for b in sealed if b['scene']==scene]:
                cc={r['method']:r for r in rows if r['case']==b['job']['case']};losses=b[proxy+'_loss'];base=losses['identity']
                if base is None:continue
                for m,v in losses.items():
                    if m=='identity' or v is None:continue
                    ax.scatter(v-base,cc[m]['accuracy_mm']-cc['identity']['accuracy_mm'],s=9,alpha=.45)
            ax.axhline(0,color='k',lw=.7);ax.axvline(0,color='k',lw=.7);ax.set(xlabel=f'{proxy} loss change',ylabel='reference MAE change (mm)',title=f'scan{scene}: lower-left = aligned improvement')
    fig.tight_layout();fig.savefig(dest/'figures/loss_alignment.png',dpi=150);plt.close(fig)
    print('DONE',dest,flush=True);print(json.dumps(summary),flush=True)
if __name__=='__main__':main()

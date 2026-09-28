"""Independent B paired-evidence audit; only images/input geometry are read."""
from common import *
import time
from scene_adapter import load_scene,camera
from v28_closeout.graph_field import estimate_normals
from v28_closeout.direct_evidence import score_patches
from threadpoolctl import threadpool_limits


def summarize(scores):
    s=np.clip(np.asarray(scores,dtype=float),-1,1)
    assert s.shape[0]==4 and s.shape[2]==2 and not np.isinf(scores).any()
    assert np.all(np.abs(scores[np.isfinite(scores)])<=1.000001)
    p,q=s[:,:,0].T,s[:,:,1].T
    vp,vq=np.isfinite(p),np.isfinite(q);v=vp&vq
    cp,cq=np.where(vp,1-p,1),np.where(vq,1-q,1)
    margin=np.where(v,q-p,0);n=v.sum(1);den=np.maximum(n,1)
    mean=margin.sum(1)/den
    low=np.where(n>0,np.where(v,margin,np.inf).min(1),0)
    high=np.where(n>0,np.where(v,margin,-np.inf).max(1),0)
    std=np.sqrt(np.where(v,(margin-mean[:,None])**2,0).sum(1)/den)
    old=np.where(n>0,np.where(v,cp,0).sum(1)/den,1)
    new=np.where(n>0,np.where(v,cq,0).sum(1)/den,1)
    return np.column_stack([cp,cq,margin,v,vp,vq,mean,low,high,std,
                            ((margin>0)&v).sum(1)/den,n/4,old,new]).astype(np.float32)


def input_spec(sid):
    if sid not in (24,37):return dict(root=DATA/'closeout-confirmation-v1/inputs'/f'scan{sid}')
    return dict(spec=dict(images=DATA/f'loss-alignment-v23/scan{sid}/image',
        colmap=DATA/f'loss-alignment-v23/scan{sid}/sparse/0',
        cameras=DATA/('real-closure-v21/cameras_geosvr_linked.npz' if sid==24 else 'reconstruction-v22-scan37/cameras.npz'),
        mesh=DATA/('published-outputs-v1/scan24_mesh.ply' if sid==24 else 'reconstruction-v22-scan37/scan37_mesh.ply')))


def main():
    check_host();start=time.monotonic();checks=[];direct=[];views_checked=0
    for sid in (24,37,55,65,69):
        folder=ROOT/'evidence'/f'scan{sid}'
        verify_seal(folder);verify_seal(PREV/'evidence'/f'scan{sid}')
        views=json.loads((folder/'VIEWS.json').read_text())
        assert views==json.loads((PREV/'evidence'/f'scan{sid}'/'VIEWS.json').read_text())
        assert json.loads((folder/'CALIBRATION.json').read_text())==json.loads((PREV/'evidence'/f'scan{sid}'/'CALIBRATION.json').read_text())
        scene=load_scene(**input_spec(sid));cams={}
        for v in views.values():
            original,reserved=v['original_views'],v['reserved_views']
            assert len(original)==len(set(original))==5
            assert len(reserved)==len(set(reserved))==4 and not set(original)&set(reserved)
            for name in original+reserved:
                assert sha(scene['views'][name]['path'])==v['image_sha256'][name]
                if name not in cams:cams[name]=camera(scene['views'][name])
            views_checked+=1
        paths=cases_for_scene(sid);assert len(paths)==(12 if sid in (24,37) else 20)
        for index,path in enumerate(paths):
            dest=folder/path.name;v=views[path.name.split('__')[0]]
            meta=json.loads((dest/'META.json').read_text())
            assert meta['candidate']=='B_all' and meta['reference_access'] is False and meta['views']==v
            assert all(sha(path/name)==h for name,h in meta['source_sha256'].items())
            with np.load(dest/'PAIRED.npz',allow_pickle=False) as z:
                ids,F,R,fit,res,valid=(z[k] for k in ('row_ids','F','R','fit_scores','reserved_scores','ref_valid'))
            with np.load(PREV/'evidence'/f'scan{sid}'/path.name/'PAIRED.npz',allow_pickle=False) as z:
                assert np.array_equal(z['row_ids'],ids)
                assert np.array_equal(z['ref_valid'],valid)
                assert np.array_equal(z['fit_scores'][:,:,0],fit[:,:,0],equal_nan=True)
                assert np.array_equal(z['reserved_scores'][:,:,0],res[:,:,0],equal_nan=True)
            assert F.shape==R.shape==(len(ids),32) and fit.shape==res.shape==(4,len(ids),2)
            assert np.isfinite(F).all() and np.isfinite(R).all()
            error=max(float(np.max(abs(F-summarize(fit)))),float(np.max(abs(R-summarize(res)))))
            assert error<=1e-7
            p,b=read_points(path/'identity.ply'),read_points(path/'B_all.ply')
            assert p.shape==b.shape and len(np.unique(ids))==len(ids) and ids.min()>=0 and ids.max()<len(p)
            if sid not in (24,37):assert np.array_equal(ids,np.arange(len(p)))
            ref=cams[v['original_views'][0]];rays=p[ids]-ref['center'];rays/=np.linalg.norm(rays,axis=1,keepdims=True)
            offset=np.sum((b[ids]-p[ids])*rays,axis=1)
            residual=float(np.max(abs(p[ids]+offset[:,None]*rays-b[ids])))
            assert residual<=1e-8 and abs(residual-meta['max_ray_residual_mm'])<=1e-12
            checks.append(dict(case=path.name,rows=len(ids),summary_error=error,ray_residual_mm=residual,incumbent_exact=True))
            if index==0:
                sample=np.unique(np.linspace(0,len(ids)-1,9).astype(int));physical=ids[sample]
                normals=estimate_normals(p,24)
                ev=score_patches(p[physical],normals[physical],np.column_stack([np.zeros(len(sample)),offset[sample]]),
                    ref,[cams[n] for n in v['original_views'][1:]+v['reserved_views']],mode='tangent',patch_radius=3,batch_size=256)
                saved=np.concatenate([fit[:,sample],res[:,sample]],axis=0);got=ev['scores']
                assert np.array_equal(np.isnan(saved),np.isnan(got))
                finite=np.isfinite(saved);score_error=float(np.max(abs(saved[finite]-got[finite]))) if finite.any() else 0.
                assert score_error<=1e-7
                assert np.allclose(ev['candidates'][:,1],b[physical],rtol=0,atol=1e-8)
                direct.append(dict(case=path.name,physical_row_ids=physical.tolist(),maximum_score_error=score_error))
        print('AUDIT B EVIDENCE',sid,flush=True)
    assert len(checks)==84 and len(direct)==5 and views_checked==20
    save_json(ROOT/'AUDIT_EVIDENCE.json',dict(status='PASS',case_count=len(checks),roi_count=views_checked,
        checks=checks,direct_checks=direct,replay_labels_read=False,reference_geometry_read=False,
        original_and_reserved_view_ids_reused_exactly=True,A_B_incumbent_scores_exact=True,
        source_sha256={str(ROOT/'audit_evidence.py'):sha(ROOT/'audit_evidence.py')},wall_seconds=time.monotonic()-start))


if __name__=='__main__':
    with threadpool_limits(limits=1):main()

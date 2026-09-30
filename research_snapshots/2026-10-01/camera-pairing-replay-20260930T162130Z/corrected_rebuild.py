"""Corrected-camera Q-only plane sweep, with immutable old query pixels."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import argparse, hashlib, json, socket, time
import numpy as np
import cv2

ROOT=Path(__file__).resolve().parent
OLD=Path('/srv/slam-research/grf/map-denoise/runs/upstream-photo-holdout-20260930T113213Z')
sys.path.insert(0,str(OLD))
from dense_rebuild import plane_scores
from rebuild import half_gray, mutual_matches, triangulate, projection_center, sha, write_json

def run(sid, only_roi=None):
    assert socket.gethostname()=='liekkas'
    cv2.setNumThreads(1); cv2.setRNGSeed(20260930)
    spec=json.loads((ROOT/'PLAN_CORRECTED.json').read_text())['scenes'][str(sid)]
    mapping=json.loads((ROOT/'CAMERA_MAPPING.json').read_text())
    start=time.monotonic(); cameras=spec['cameras']; ref=spec['reference']
    names=[ref]+spec['Q']; images=[]; sources={}
    for name in names:
        c=cameras[name]; sources[c['image_path']]=sha(c['image_path'])
        images.append(half_gray(Path(c['image_path']),c))
    P=np.array(cameras[ref]['P']); Ps=[np.array(cameras[n]['P']) for n in spec['Q']]
    sift=cv2.SIFT_create(nfeatures=12000)
    kp,des=sift.detectAndCompute(images[0],None); zs=[]
    for im,S in zip(images[1:],Ps):
        skp,sdes=sift.detectAndCompute(im,None); matches=mutual_matches(des,sdes)
        if not matches: continue
        uv=np.array([kp[m.queryIdx].pt for m in matches]); xy=np.array([skp[m.trainIdx].pt for m in matches])
        pts,good,_,_=triangulate(P,S,uv,xy)
        zs.extend((pts[good]@P[2,:3]+P[2,3]).tolist())
    if len(zs)>=10:
        low,high=np.quantile(zs,[.05,.95])+[-20.,20.]; rule='Q-only SIFT q05/q95 +-20mm'
    else:
        axes=np.array([cameras[n]['P'][2][:3] for n in names]); axes/=np.linalg.norm(axes,axis=1)[:,None]
        proj=np.eye(3)[None]-axes[:,:,None]*axes[:,None,:]
        centers=np.array([cameras[n]['center'] for n in names])
        focus=np.linalg.solve(proj.sum(0),np.einsum('nij,nj->i',proj,centers))
        z=P[2,:3]@focus+P[2,3]; low,high=z-150,z+150; rule='Q-only rig focus +-150mm'
    assert np.isfinite([low,high]).all() and 0<low<high and high-low<2000
    depths=np.arange(np.floor(low),np.ceil(high)+.1,1.)
    print('DEPTH_RANGE',sid,float(low),float(high),len(depths),flush=True)
    for roi in spec['rois']:
        if only_roi and roi['id']!=only_roi: continue
        dest=ROOT/'initialization'/f'scan{sid}'/roi['id']
        dest.mkdir(parents=True,exist_ok=False); tick=time.monotonic()
        x0,y0,x1,y1=roi['box_xyxy_original']
        xx,yy=np.meshgrid(np.arange(x0/2+4,x1/2-4,8.),np.arange(y0/2+4,y1/2-4,8.))
        pixels=np.column_stack((xx.ravel(),yy.ravel()))
        old_path=OLD/'construction-dense'/f'scan{sid}'/roi['id']/'input.npz'
        with np.load(old_path) as a:
            fixed=a['reference_pixel_xy'][a['query_ids']].copy()
            old_context=a['reference_pixel_xy'].copy()
        lookup={tuple(x):i for i,x in enumerate(pixels)}
        fixed_grid=np.array([lookup[tuple(x)] for x in fixed])
        scores,var=plane_scores(images[0],images[1:],P,Ps,pixels,depths)
        finite=np.isfinite(scores); safe=np.where(finite,scores,-2); best=safe.argmax(0)
        best_score=safe[best,np.arange(len(pixels))]
        valid=finite.any(0)&(best_score>=.6)&(var>1e-5)
        z=depths[best].copy()
        for i in np.flatnonzero((best>0)&(best<len(depths)-1)):
            a,b,c=safe[best[i]-1:best[i]+2,i]; den=a-2*b+c
            if den<-1e-8 and min(a,b,c)>-1: z[i]+=np.clip(.5*(a-c)/den,-.5,.5)
        success=valid[fixed_grid]; old_ids=np.flatnonzero(success)
        reserved=set(fixed_grid[success].tolist())
        hashed=sorted(np.flatnonzero(valid),key=lambda i:hashlib.sha256(pixels[i].tobytes()).digest())
        selected=list(reserved)+[i for i in hashed if i not in reserved][:384-len(reserved)]
        # Stable context ordering; fixed queries reserved, nonqueries use old hash fill.
        order=np.array(sorted(selected,key=lambda i:hashlib.sha256(pixels[i].tobytes()).digest()),int)
        reverse={v:i for i,v in enumerate(order)}
        ids=np.array([reverse[g] for g in fixed_grid[success]],int)
        context_index=np.full(len(fixed),-1,int); context_index[success]=ids
        rays=np.column_stack((pixels[order],np.ones(len(order))))@np.linalg.inv(P[:,:3]).T
        points=projection_center(P)+z[order,None]*rays
        with (dest/'input.npz').open('xb') as f:
            np.savez_compressed(f,points_mm=points,reference_pixel_xy=pixels[order],query_ids=ids,
                query_old_ids=old_ids,query_pixel_xy=fixed[success],construction_success=success,
                fixed_query_pixel_xy=fixed,construction_zncc=best_score[order],optical_depth_mm=z[order],
                grid_uv=pixels,qualified_mask=valid,selected_context_grid_indices=order,
                query_context_indices=context_index,fixed_query_grid_indices=fixed_grid,
                hash_order=np.array(hashed,int),failure_reason=np.where(success,'','quality_threshold'))
        old_set={tuple(v) for v in old_context}; new_set={tuple(v) for v in pixels[order]}
        info=dict(scene=sid,roi=roi['id'],fixed_query_rows=len(fixed),successful_query_rows=int(success.sum()),
            failed_query_rows=int((~success).sum()),context_rows=len(order),passing_rows=int(valid.sum()),
            status='READY' if len(ids)>=64 and len(order)>=64 else 'INCOMPLETE',
            depth_range_mm=[float(low),float(high)],depth_steps=len(depths),depth_rule=rule,sift_depth_count=len(zs),
            fixed_query_score=best_score[fixed_grid].tolist(),fixed_query_variance=var[fixed_grid].tolist(),
            fixed_query_success=success.tolist(),query_old_ids=old_ids.tolist(),
            context_rule='reserve passing fixed queries; old SHA256 fill to384; hash sort',
            forced_queries_outside_pure_hash384=len(reserved-set(hashed[:384])),
            context_intersection_old=len(old_set&new_set),context_added=len(new_set-old_set),context_removed=len(old_set-new_set),
            mapping_sha256=sha(ROOT/'CAMERA_MAPPING.json'),plan_sha256=sha(ROOT/'PLAN_CORRECTED.json'),
            input_source_sha256=sha(old_path),source_files=sources,gt_accessed=False,
            construction_views=names,seconds=time.monotonic()-tick,scene_elapsed=time.monotonic()-start)
        write_json(dest/'REBUILD.json',info)
        print('REBUILT',roi['id'],len(ids),len(order),round(info['seconds'],2),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--scene',type=int,required=True); ap.add_argument('--roi')
    a=ap.parse_args(); run(a.scene,a.roi)

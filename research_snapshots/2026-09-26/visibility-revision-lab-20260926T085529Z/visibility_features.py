"""Current-input depth-support proxies; no ground truth or native parent surface."""
import numpy as np

VIEW_NAMES = (
    'old_project_valid','new_project_valid','old_center_known','new_center_known',
    'old_plane_known','new_plane_known','old_center_gap_mm','new_center_gap_mm',
    'old_plane_gap_mm','new_plane_gap_mm','old_plane_residual_mm','new_plane_residual_mm',
    'old_coverage_3x3','new_coverage_3x3','old_log_density','new_log_density',
    'old_self_front','new_self_front','parallax_sin','pixel_motion',
)
GEOMETRY_WIDTH = 80
FEATURE_NAMES = tuple(f'parallax_slot{v}_{n}' for v in range(4) for n in VIEW_NAMES) + tuple(
    f'tol{t}_{n}' for t in (0.5,2.0) for n in
    ('margin0','margin1','margin2','margin3','weighted_mean_margin','paired_fraction','mean_weight','weighted_win_fraction'))


def project(points, cam):
    h=np.column_stack([points,np.ones(len(points))]) @ np.asarray(cam['P']).T
    depth=h[:,2]/np.linalg.norm(np.asarray(cam['P'])[2,:3])
    uv=np.divide(h[:,:2],h[:,2,None],out=np.zeros((len(points),2)),where=abs(h[:,2,None])>1e-12)
    valid=(depth>0)&np.isfinite(depth)&np.isfinite(uv).all(1)&(uv[:,0]>=0)&(uv[:,0]<cam['width'])&(uv[:,1]>=0)&(uv[:,1]<cam['height'])
    return uv,depth,valid


def rasterize(points,cam):
    """One-pixel nearest positive camera-depth; original points retain owner IDs."""
    uv,z,valid=project(points,cam)
    width,height=cam['width'],cam['height'];size=width*height
    ids=np.flatnonzero(valid);pixels=np.floor(uv[ids]).astype(int)
    flat=pixels[:,1]*width+pixels[:,0]
    order=np.lexsort((ids,z[ids],flat))
    sorted_flat=flat[order]
    first=np.r_[True,np.diff(sorted_flat)!=0] if len(order) else np.zeros(0,bool)
    owner=np.full(size,-1,dtype=np.int64)
    owner[sorted_flat[first]]=ids[order[first]]
    density=np.bincount(flat,minlength=size)
    return dict(owner=owner,density=density,uv=uv,z=z,cam=cam)


def query_support(points,row_ids,raster):
    """Return projection and signed gaps; fit 1/z affine to 3x3 front points.

    Inverse depth is planar in image coordinates for a perspective plane. Actual
    subpixel front-point UV is used so raster rounding cannot create a fake edge.
    Failed/nonexistent fits remain unknown, not inferred free space.
    """
    cam=raster['cam'];width,height=cam['width'],cam['height']
    uv,z,valid=project(points,cam);n=len(points)
    pix=np.floor(uv).astype(int);safe=np.clip(pix,[0,0],[width-1,height-1])
    centerflat=safe[:,1]*width+safe[:,0]
    centerowner=raster['owner'][centerflat]
    known=valid&(centerowner>=0)
    centergap=np.where(known,z-raster['z'][np.maximum(centerowner,0)],0.)
    self_front=known&(centerowner==row_ids)
    owners=[]
    for dy in (-1,0,1):
        for dx in (-1,0,1):
            q=pix+np.array([dx,dy]);inside=valid&(q[:,0]>=0)&(q[:,0]<width)&(q[:,1]>=0)&(q[:,1]<height)
            clipped=np.clip(q,[0,0],[width-1,height-1]);idx=clipped[:,1]*width+clipped[:,0]
            owners.append(np.where(inside,raster['owner'][idx],-1))
    owners=np.stack(owners,axis=1);mask=owners>=0;safe_owner=np.maximum(owners,0)
    count=mask.sum(1);xy=raster['uv'][safe_owner]-uv[:,None,:]
    x=np.concatenate([np.ones((n,9,1)),xy],axis=2)
    x=np.where(mask[:,:,None],x,0.)
    frontz=raster['z'][safe_owner]
    invz=np.divide(mask,frontz,out=np.zeros_like(frontz),where=mask&(frontz>0))
    gram=np.einsum('nki,nkj->nij',x,x)
    rhs=np.einsum('nki,nk->ni',x,invz)
    # Degenerate support (one pixel, collinear points) must not become a plane.
    eig=np.linalg.eigvalsh(gram)
    fit=valid&(count>=3)&(eig[:,0]>1e-8)
    coeff=np.zeros((n,3))
    if fit.any():coeff[fit]=np.linalg.solve(gram[fit],rhs[fit,:,None])[:,:,0]
    fit &= coeff[:,0]>1e-12
    predicted=np.divide(1.,coeff[:,0],out=np.zeros(n),where=fit)
    planegap=np.where(fit,z-predicted,0.)
    invpred=np.einsum('nki,ni->nk',x,coeff)
    back=np.divide(1.,invpred,out=np.zeros_like(invpred),where=mask&(invpred>1e-12))
    residual=np.sqrt(np.sum(np.where(mask,(back-frontz)**2,0.),axis=1)/np.maximum(count,1))
    residual=np.where(fit,residual,0.)
    logdensity=np.where(known,np.log1p(raster['density'][centerflat]),0.)
    return dict(uv=uv,valid=valid,known=known,fit=fit,centergap=centergap,
        planegap=planegap,residual=residual,coverage=count/9.,density=logdensity,self_front=self_front)


def visibility_features(p, candidates, row_ids, cameras, ref_center, pair_scores, chunk=8192):
    """[N,2,96] features for A/B, using current full input p for every raster.

    candidates is [N,2,3], pair_scores [2,4,N,2] (incumbent/proposal ZNCC).
    Evidence slots sort by incumbent reference/source ray parallax, stable ties.
    """
    p=np.asarray(p,float);candidates=np.asarray(candidates,float);row_ids=np.asarray(row_ids,int)
    scores=np.asarray(pair_scores,float)
    n=len(row_ids)
    if candidates.shape!=(n,2,3) or scores.shape!=(2,4,n,2) or len(cameras)!=4:
        raise ValueError('candidate / score / camera dimensions differ')
    output=np.zeros((n,2,len(FEATURE_NAMES)),np.float32)
    rasters=[rasterize(p,cam) for cam in cameras]
    for start in range(0,n,chunk):
        end=min(start+chunk,n);ids=row_ids[start:end];old=p[ids];m=len(ids)
        rr=old-np.asarray(ref_center);rr/=np.maximum(np.linalg.norm(rr,axis=1,keepdims=True),1e-12)
        raw=np.zeros((m,2,4,20));parallax=np.zeros((m,4));gaps=np.zeros((m,2,4));support=np.zeros((m,2,4),bool)
        for v,raster in enumerate(rasters):
            sr=old-np.asarray(cameras[v]['center']);sr/=np.maximum(np.linalg.norm(sr,axis=1,keepdims=True),1e-12)
            parallax[:,v]=np.linalg.norm(np.cross(rr,sr),axis=1)
            o=query_support(old,ids,raster)
            for c in range(2):
                q=query_support(candidates[start:end,c],ids,raster)
                raw[:,c,v]=np.column_stack([o['valid'],q['valid'],o['known'],q['known'],o['fit'],q['fit'],
                    o['centergap'],q['centergap'],o['planegap'],q['planegap'],o['residual'],q['residual'],
                    o['coverage'],q['coverage'],o['density'],q['density'],o['self_front'],q['self_front'],
                    parallax[:,v],np.where(o['valid']&q['valid'],np.linalg.norm(o['uv']-q['uv'],axis=1),0.)])
                og=np.where(o['fit'],o['planegap'],o['centergap']);qg=np.where(q['fit'],q['planegap'],q['centergap'])
                gaps[:,c,v]=np.maximum(np.maximum(og,qg),0.)
                support[:,c,v]=(o['fit']|o['known'])&(q['fit']|q['known'])
        order=np.argsort(parallax,axis=1,kind='stable')
        raw=np.take_along_axis(raw,order[:,None,:,None],axis=2)
        output[start:end,:,:GEOMETRY_WIDTH]=raw.reshape(m,2,GEOMETRY_WIDTH)
        pair=scores[:,:,start:end].transpose(2,0,1,3)
        valid=np.isfinite(pair).all(3)&support
        margin=np.where(valid,pair[:,:,:,1]-pair[:,:,:,0],0.)
        for ti,tolerance in enumerate((.5,2.)):
            weight=np.where(valid,np.exp(-gaps/tolerance),0.)
            weighted=weight*margin;den=weight.sum(2);safe=np.maximum(den,1e-20)
            ordered=np.take_along_axis(weighted,order[:,None,:],axis=2)
            aggregate=np.stack([weighted.sum(2)/safe,valid.mean(2),weight.mean(2),
                (weight*(margin>0)).sum(2)/safe],axis=2)
            output[start:end,:,GEOMETRY_WIDTH+ti*8:GEOMETRY_WIDTH+(ti+1)*8]=np.concatenate([ordered,aggregate],axis=2)
    if not np.isfinite(output).all():raise AssertionError('nonfinite evidence')
    return output

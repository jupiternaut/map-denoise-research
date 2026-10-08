"""Shared image-only weighted NCC, explicit field interventions, frozen P."""
import importlib.util
from pathlib import Path
import numpy as np

OLD = Path('/srv/slam-research/grf/map-denoise/runs/surface-owned-support-20261008T022918Z')
spec = importlib.util.spec_from_file_location('readonly_support', OLD/'support.py')
support = importlib.util.module_from_spec(spec)
spec.loader.exec_module(support)


def cameras(records):
    return [{k:np.asarray(v,float) for k,v in c.items()} for c in records]


def geometry(cams, xy, grid):
    yy,xx=np.mgrid[-4:5,-4:5]
    ref_uv=np.asarray(xy)+np.column_stack((xx.ravel(),yy.ravel()))
    world=cams[0]['C']+grid[:,None,None]*support.rays(cams[0],ref_uv)[None,:,:]
    projected=[support.project(c,world) for c in cams[1:]]
    return ref_uv,[x[0] for x in projected],[x[1] for x in projected]


def gather(fields, ref_uv, source_uv):
    a,av=support.sample(np.asarray(fields[0],float),ref_uv)
    others=[support.sample(np.asarray(fields[j+1],float),uv) for j,uv in enumerate(source_uv)]
    return a,np.stack([p[0] for p in others]),av,np.stack([p[1] for p in others])


def weighted_ncc(a,b,weights,valid_samples,params):
    """a: GxP, b: 2xGxP, same nonnegative GxP weights for both views."""
    w=np.asarray(weights,float)
    mass=w.sum(axis=1)
    safe=np.maximum(mass,1e-300)
    ess=mass**2/np.maximum((w*w).sum(axis=1),1e-300)
    am=(w*a).sum(axis=1)/safe
    bm=(w[None,:,:]*b).sum(axis=2)/safe[None,:]
    ac=a-am[:,None];bc=b-bm[:,:,None]
    sa=np.sqrt(np.maximum(0,(w*ac*ac).sum(axis=1)/safe))
    sb=np.sqrt(np.maximum(0,(w[None,:,:]*bc*bc).sum(axis=2)/safe[None,:]))
    cov=(w[None,:,:]*ac[None,:,:]*bc).sum(axis=2)/safe[None,:]
    qualified=(mass>=params['min_mass']) & (ess>=params['min_ess']-1e-10) & (sa>=params['std_min'])
    # Invalid projections matter only for a sample given positive weight.
    qualified &= np.all(valid_samples | (w[None,:,:]<=1e-14),axis=(0,2))
    valid=qualified[None,:] & (sb>=params['std_min'])
    scores=np.full((2,len(mass)),np.nan)
    denominator=sa[None,:]*sb
    np.divide(cov,denominator,out=scores,where=valid)
    return dict(scores=scores,mass=mass,ess=ess,ref_std=sa,source_std=sb,
                count=(w>1e-14).sum(axis=1),qualified=qualified)


def score(images,cams,grid,params,arm,fields=None):
    ref_uv,source_uv,depths=geometry(cams,params['xy'],grid)
    a,b,av,bv=gather(images,ref_uv,source_uv)
    a=np.broadcast_to(a,(len(grid),81)).copy()
    valid=bv & av[None,None,:] & (np.stack(depths)>0) & (grid[None,:,None]>0)
    if arm in ('full9','connected9'):
        mask=support.center_mask(a[0].reshape(9,9),arm).ravel()
        w=np.broadcast_to(mask,(len(grid),81)).astype(float)
    else:
        field_key='center' if arm.endswith('_center') else 'footprint' if arm=='estimated_footprint' else 'area'
        qa,qb,qav,qbv=gather(fields[field_key],ref_uv,source_uv)
        valid &= qbv & qav[None,None,:]
        fraction=np.minimum(qa[None,:],np.min(qb,axis=0))
        fraction=np.clip(fraction,0.,1.)
        if arm=='oracle_fraction':
            w=fraction
        elif arm=='oracle_pure':
            w=(fraction>=params['purity_min']).astype(float)
        else:
            w=(fraction>=params['majority_min']).astype(float)
        if arm=='oracle_component':
            na,nb,nav,nbv=gather(fields['numerator'],ref_uv,source_uv)
            aclean=np.divide(na,qa,out=np.zeros_like(na),where=qa>1e-12)
            b=np.divide(nb,qb,out=np.zeros_like(nb),where=qb>1e-12)
            a=np.broadcast_to(aclean,(len(grid),81)).copy()
            valid &= nbv & nav[None,None,:]
    result=weighted_ncc(a,b,w,valid,params)
    accepted=np.all(np.isfinite(result['scores']) & (result['scores']>=params['ncc_min']),axis=0)
    result.update(grid=grid,weights=w,accepted=accepted,reference=a,source=b,valid_samples=valid)
    intervals=support.support_intervals(grid,accepted,padding=params['padding_mm'])
    return result,intervals


def select(intervals,config):
    mass=sum(b-a for a,b in intervals)
    current=config['incumbent_mm']
    if mass<=0:
        return dict(selected_depth=current,support_mean=None,estimated_squared_gain=0.)
    target=sum((b-a)*(a+b)/2 for a,b in intervals)/mass
    candidates=np.asarray(config['depths_mm'])
    idx=int(np.argmin((candidates-target)**2))
    new=float(candidates[idx]);gain=(current-target)**2-(new-target)**2
    if gain<=0:new=current
    return dict(selected_depth=new,support_mean=target,estimated_squared_gain=max(0.,float(gain)))

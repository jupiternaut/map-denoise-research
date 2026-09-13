"""New automatic XYZ-only projection-pursuit pair reference. No evaluator import.

Candidate planes are fitted along all three local PCA axes, not just the least
variance axis; valid pairs must actually be parallel, separated normal to their
fitted planes, and have planar component covariance. Two neighborhood scales
compete using observed normalized BIC improvement. This is a development
heuristic, not a calibrated posterior/model-selection theorem.
"""
import time
import numpy as np
from scipy.spatial import cKDTree
from parallel_geometry_v5.association.operator import _planes, _spacing, fit_modes, _log_component

CONFIG={'ks':(48,96),'iterations':2,'em_iterations':16,'step':.8,
        'bic_gain':6.,'separation_sigma':3.,'normal_agreement':.95,'planarity_ratio':.25}

def denoise(points, split=True, config=None):
    cfg=CONFIG|({} if config is None else config)
    source=np.asarray(points,dtype=float)
    if source.ndim!=2 or source.shape[1]!=3 or not np.isfinite(source).all(): raise ValueError('finite Nx3 required')
    p=source.copy(); started=time.perf_counter(); spacing=_spacing(p); history=[]
    if len(p)<8 or spacing<=0: return p,{'reason':'small_or_coincident'}
    floor=max(.05*spacing,1e-10)
    for _ in range(cfg['iterations']):
        tree=cKDTree(p); best_score=np.full(len(p),-np.inf); best_move=np.zeros_like(p)
        fallback=None; split_axis=np.full(len(p),-1); split_scale=np.full(len(p),-1)
        for si,k in enumerate(cfg['ks']):
            dist,ix=tree.query(p,k=min(k,len(p)),workers=1); nbr=p[ix]
            h=np.maximum(.5*dist[:,-1],floor); spatial=np.exp(-.5*(dist/h[:,None])**2)
            center,normal,vals=_planes(nbr,spatial)
            centered=nbr-center[:,None,:]
            cov=np.einsum('nk,nki,nkj->nij',spatial/spatial.sum(axis=1)[:,None],centered,centered)
            _,axes=np.linalg.eigh(cov)
            if fallback is None:
                heights=np.einsum('nkj,nj->nk',nbr-p[:,None,:],normal)
                w=spatial*np.exp(-.5*(heights/(.5*spacing))**2)
                fallback=normal*((w*heights).sum(axis=1)/np.maximum(w.sum(axis=1),1e-20))[:,None]
            if not split: continue
            for axis in range(3):
                direction=axes[:,:,axis]
                residual=np.einsum('nkj,nj->nk',centered,direction)
                modes=fit_modes(residual,floor**2,{'em_iterations':cfg['em_iterations'],
                    'bic_gain':cfg['bic_gain'],'separation_sigma':cfg['separation_sigma']})
                c0,n0,v0=_planes(nbr,spatial*modes['responsibility'][:,:,0])
                c1,n1,v1=_planes(nbr,spatial*modes['responsibility'][:,:,1])
                agreement=np.abs(np.einsum('nj,nj->n',n0,n1))
                fit_gap=np.abs(np.einsum('nj,nj->n',c1-c0,n0))
                thickness=np.sqrt(np.maximum((v0[:,0]+v1[:,0])/2,floor**2))
                valid=(modes['choose_two']&(agreement>=cfg['normal_agreement'])
                       &(fit_gap>=cfg['separation_sigma']*thickness)
                       &(v0[:,0]<cfg['planarity_ratio']*v0[:,1])
                       &(v1[:,0]<cfg['planarity_ratio']*v1[:,1]))
                qr=np.einsum('nj,nj->n',p-center,direction)
                lp=_log_component(qr[:,None],modes['means'],modes['variance'],modes['weights'])[:,0,:]
                post=np.exp(lp-np.logaddexp(lp[:,0],lp[:,1])[:,None]); choose=post[:,1]>post[:,0]
                c=np.where(choose[:,None],c1,c0); n=np.where(choose[:,None],n1,n0)
                move=n*np.einsum('nj,nj->n',c-p,n)[:,None]*np.abs(post[:,1]-post[:,0])[:,None]
                score=modes['bic_gain']/len(ix[0])
                take=valid&(score>best_score)
                best_score[take]=score[take]; best_move[take]=move[take]
                split_axis[take]=axis; split_scale[take]=si
        selected=np.isfinite(best_score)
        displacement=cfg['step']*np.where(selected[:,None],best_move,fallback)
        length=np.linalg.norm(displacement,axis=1)
        displacement*=np.minimum(1.,2*spacing/np.maximum(length,floor))[:,None]
        p+=displacement
        history.append({'split_count':int(selected.sum()),'nonminimal_axis_count':int(np.sum(split_axis>0)),
                        'large_scale_count':int(np.sum(split_scale==1))})
    return p,{'config':cfg,'split_enabled':split,'history':history,'elapsed_s':time.perf_counter()-started,
              'requires_ground_truth':False,'observed_spacing':spacing}

def oracle_association(points, labels, k=48, iterations=2):
    """Diagnostic only: GT component identity, no GT coordinates or normals."""
    p=np.asarray(points).copy()
    for _ in range(iterations):
        q=p.copy()
        for lab in np.unique(labels):
            subset=np.flatnonzero(labels==lab); cloud=p[subset]
            dist,ix=cKDTree(cloud).query(cloud,k=min(k,len(cloud)),workers=1)
            w=np.exp(-.5*(dist/np.maximum(.5*dist[:,-1,None],1e-10))**2)
            c,n,_=_planes(cloud[ix],w)
            q[subset]+=.8*n*np.einsum('nj,nj->n',c-cloud,n)[:,None]
        p=q
    return p,{'oracle':'GT association only; no GT positions/normals','deployable':False}

def oracle_normal(points, normals, k=48, iterations=2):
    """Diagnostic only: GT per-point direction, not GT association or position."""
    p=np.asarray(points).copy(); spacing=_spacing(p)
    for _ in range(iterations):
        dist,ix=cKDTree(p).query(p,k=min(k,len(p)),workers=1)
        w=np.exp(-.5*(dist/np.maximum(.5*dist[:,-1,None],1e-10))**2)
        heights=np.einsum('nkj,nj->nk',p[ix]-p[:,None,:],normals)
        w*=np.exp(-.5*(heights/(.5*spacing))**2)
        p+=.8*normals*((w*heights).sum(axis=1)/w.sum(axis=1))[:,None]
    return p,{'oracle':'GT normals only; no GT positions/association','deployable':False}

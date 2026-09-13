"""Incumbent-preserving BIC and fixed-parameter projection policies."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'exploration_v17'))
import v17_operator as base

METHODS=('identity','old_spatial36','v17_map','retained_map','retained_half','retained_mean')

def arrays(model):
    out=dict(model)
    for key in ('means','slope','gate','feature_center','feature_scale','plane','prediction','posterior','groups'):
        if key in out:out[key]=np.asarray(out[key])
    return out

def retain(pool,old,x,y,sigma):
    pool={f:arrays(m) for f,m in pool.items()};old=arrays(old)
    original=base.choose({f:m['bic'] for f,m in pool.items()})
    before=float(pool['R']['bic']);candidate=None;replaced=False
    if int(old['k'])==2:
        candidate=dict(old,family='R',feature_center=pool['R']['feature_center'],feature_scale=pool['R']['feature_scale'],
                       parameter_count=16,candidate_origin='old_spatial36',solver_diagnostics=[])
        candidate.update(base.evaluate(candidate,x,y,sigma))
        candidate['nll']=float(-candidate['log_density'].sum())
        candidate['ridge_penalty']=float(base.RIDGE*np.sum(candidate['gate'][1:]**2))
        candidate['bic']=float(2*candidate['nll']+candidate['ridge_penalty']+16*np.log(len(y)))
        if candidate['bic']<pool['R']['bic']:
            pool['R']=candidate;replaced=True
    selected=base.choose({f:m['bic'] for f,m in pool.items()})
    diagnostic=dict(original_family=original,selected_family=selected,replaced_r=replaced,
                    r_bic_before=before,r_bic_after=float(pool['R']['bic']),
                    old_r_bic=None if candidate is None else candidate['bic'],
                    selected_bic=float(pool[selected]['bic']),truth_fields_used=[])
    return pool[selected],diagnostic

def project(model,x,gamma):
    """gamma=1 MAP, 0 posterior mean; keeps fitting/selection fixed."""
    if not 0<=gamma<=1:raise ValueError('gamma must be between 0 and 1')
    model=arrays(model)
    levels=model['means'][None,:]+(x[:,1:]@model['slope'])[:,None]
    mean=np.sum(model['posterior']*levels,axis=1)
    hard=levels[np.arange(len(x)),model['groups']]
    output=hard if gamma==1 else mean if gamma==0 else gamma*hard+(1-gamma)*mean
    return dict(model,prediction=output,posterior_mean=mean,map_prediction=hard,gamma=gamma,
                output_distance_to_fitted_surface_mm=np.min(abs(output[:,None]-levels),axis=1),
                posterior_model_variance_mm2=np.sum(model['posterior']*(levels-mean[:,None])**2,axis=1))

def construct(x,y,sigma,cached_pool=None,cached_old=None):
    pool=base.fit_pool(x,y,sigma) if cached_pool is None else {f:arrays(m) for f,m in cached_pool.items()}
    old=base.old_fit(x,y,sigma,'spatial_free') if cached_old is None else arrays(cached_old)
    original=base.choose({f:m['bic'] for f,m in pool.items()})
    retained,diagnostic=retain(pool,old,x,y,sigma)
    outputs={'old_spatial36':old,'v17_map':dict(pool[original],gamma=1.)}
    for name,gamma in [('retained_map',1.),('retained_half',.5),('retained_mean',0.)]:
        outputs[name]=project(retained,x,gamma)
    diagnostic.update(pool=pool,old=old,retained=retained)
    return outputs,diagnostic

def filter_local(local_xyz_mm,sigma_mm,policy='retained_map'):
    if policy not in ('retained_map','retained_half','retained_mean'):raise ValueError(policy)
    xyz=np.asarray(local_xyz_mm,float)
    if xyz.ndim!=2 or xyz.shape[1]!=3:raise ValueError('N x 3 local-coordinate points in millimetres')
    x=np.c_[np.ones(len(xyz)),xyz[:,:2]/50.]
    models,diagnostic=construct(x,xyz[:,2],sigma_mm)
    out=xyz.copy();out[:,2]=models[policy]['prediction']
    return out,models[policy],diagnostic

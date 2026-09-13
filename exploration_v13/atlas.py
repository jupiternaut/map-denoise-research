from pathlib import Path
import sys,time
import numpy as np
from scipy.special import expit,logsumexp
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'exploration_v11'),str(ROOT/'exploration_v12')]
from spatial_filter import solve,features
from local_filter import corrected_world,windows

def plane(q,w=None,normal=None):
    w=np.ones(len(q)) if w is None else np.asarray(w,float)
    mass=float(w.sum())
    if mass<1e-8:raise ValueError('empty component')
    mu=np.sum(q*w[:,None],axis=0)/mass;a=q-mu
    if normal is None:
        _,v=np.linalg.eigh(a.T@(a*w[:,None])/mass);normal=v[:,0]
    return mu,np.asarray(normal)

def prepare(state,sigma):
    start=time.perf_counter();q,mask=corrected_world(state)
    model=solve(state['design'],state['corrected'],sigma,iterations=12)
    n=len(q);labels=np.empty(n,int);labels[state['order']]=model['groups']
    if model['k']==1:prior=np.ones((n,1))
    else:
        p=np.clip(expit(features(state['design'][:,1:])@model['gate']),.02,.98)
        prior=np.empty((n,2));prior[state['order']]=np.c_[1-p,p]
    mm=q*1000;reference=[]
    for g in range(model['k']):
        take=labels==g
        reference.append(plane(mm[take]) if take.sum()>=3 else plane(mm,prior[:,g]))
    return dict(mm=mm,mask=mask,labels=labels,prior=prior,reference=reference,k=model['k'],seconds=time.perf_counter()-start)

def project(q,surface):
    mu,normal=surface
    return q-((q-mu)@normal)[:,None]*normal

def decode(state,prepared,sigma,mode='atlas_hard',neighbors=96):
    if mode not in ('global_hard','atlas_hard','atlas_tied','atlas_relax'):raise ValueError(mode)
    if mode!='global_hard' and neighbors<6:raise ValueError('at least six neighbors required')
    start=time.perf_counter();q=prepared['mm'];labels=prepared['labels'];K=prepared['k'];prior=prepared['prior'];reference=prepared['reference']
    result=q.copy();final_labels=labels.copy();fallbacks=0;jobs=[]
    if mode=='global_hard':
        for g in range(K):result[labels==g]=project(q[labels==g],reference[g])
    else:
        for core,fit in windows(q,neighbors):
            surfaces=[]
            for g in range(K):
                take=fit[labels[fit]==g]
                if len(take)<6:surfaces.append(reference[g]);fallbacks+=1
                else:surfaces.append(plane(q[take],normal=reference[g][1] if mode=='atlas_tied' else None))
            assignment=labels[core].copy()
            if mode=='atlas_relax':
                for _ in range(8):
                    residual=np.column_stack([(q[fit]-mu)@normal for mu,normal in surfaces])
                    logp=-.5*(residual/sigma)**2+np.log(prior[fit])
                    r=np.exp(logp-logsumexp(logp,axis=1)[:,None])
                    for g in range(K):
                        if r[:,g].sum()>=6:surfaces[g]=plane(q[fit],r[:,g])
                residual=np.column_stack([(q[core]-mu)@normal for mu,normal in surfaces])
                assignment=np.argmax(-.5*(residual/sigma)**2+np.log(prior[core]),axis=1)
            for g in range(K):
                take=core[assignment==g];result[take]=project(q[take],surfaces[g])
            final_labels[core]=assignment;jobs.append(dict(core=len(core),fit=len(fit)))
    out=state['world'].copy();mask=prepared['mask'];out[mask]=result[mask]/1000.
    assert np.isfinite(out).all();np.testing.assert_array_equal(out[~mask],state['world'][~mask])
    info=dict(mode=mode,neighbors=neighbors,global_k=K,local_windows=len(jobs),fallback_components=fallbacks,
        association_change_fraction=float(np.mean(labels!=final_labels)),supported_fraction=float(mask.mean()),
        prepare_seconds=prepared['seconds'],decode_seconds=time.perf_counter()-start,truth_fields_used=[])
    return out,info,dict(support_mask=mask,initial_labels=labels,final_labels=final_labels)

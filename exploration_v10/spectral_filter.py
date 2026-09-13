"""Frame-power-spectrum gap estimation followed by actual surface projection.

Input-only prototype for a locally parallel one/two-surface family. Noise sigma
is supplied. No hard point grouping is used by the spectral gap estimator.
"""
from functools import lru_cache
import time
import numpy as np
from scipy.special import logsumexp

FREQUENCY=np.linspace(.15,2.5,64) # dimensionless angular frequency omega*sigma
GAP_GRID=np.linspace(.25,12.,300) # gap/sigma, same range for every input
NULL_REPLICATES=256
NULL_SEED=9131007

def frame_power(height_mm,scan_id,sigma_mm):
    if not np.isfinite(sigma_mm) or sigma_mm<=0:raise ValueError('positive sigma required')
    y=np.asarray(height_mm,float)/float(sigma_mm);scan=np.asarray(scan_id)
    if y.ndim!=1 or scan.shape!=y.shape or not np.isfinite(y).all():raise ValueError('finite heights and matching scans required')
    curves=[];counts=[];ids=[]
    for sid in np.unique(scan):
        h=y[scan==sid];n=len(h)
        if n<3:continue
        # Centering changes phase only, not modulus; reduces trig roundoff.
        h=h-np.median(h)
        z=np.exp(1j*h[:,None]*FREQUENCY)
        q=(np.abs(z.sum(axis=0))**2-n)/(n*(n-1))
        curves.append(q);counts.append(n);ids.append(sid)
    if not curves:raise ValueError('at least one scan with 3 observations required')
    return np.asarray(curves),np.asarray(counts),np.asarray(ids)

def fit_power(curves,counts):
    q=np.asarray(curves);counts=np.asarray(counts,float)
    envelope=np.exp(-FREQUENCY**2)
    shape=envelope[None,:]*np.sin(GAP_GRID[:,None]*FREQUENCY[None,:]/2)**2
    residual=envelope[None,:]-q
    norm=(shape*shape).sum(1)
    amplitude=np.clip((residual@shape.T)/norm[None,:],0.,1.)
    loss=((residual[:,None,:]-amplitude[:,:,None]*shape[None,:,:])**2).mean(2)
    weights=counts/counts.sum()
    profile=weights@loss
    index=int(np.argmin(profile))
    null_loss=float(weights@(residual**2).mean(1))
    return dict(gap_over_sigma=float(GAP_GRID[index]),amplitude=amplitude[:,index],
                gain=float(null_loss-profile[index]),null_loss=null_loss,
                fitted_loss=float(profile[index]),profile=profile)

@lru_cache(maxsize=64)
def null_gain_threshold(counts):
    """Input-independent Gaussian-null calibration, including frequency search.

This is model calibration, not GT scene tuning or a universal false-split bound.
Pair differences are dependent; simulation retains that dependence.
"""
    rng=np.random.default_rng(NULL_SEED);counts=tuple(int(n) for n in counts)
    scan=np.repeat(np.arange(len(counts)),counts);gains=[]
    for _ in range(NULL_REPLICATES):
        q,n,_=frame_power(rng.normal(size=len(scan)),scan,1.)
        gains.append(fit_power(q,n)['gain'])
    return float(np.quantile(gains,.99)),np.asarray(gains)

def estimate_gap(height_mm,scan_id,sigma_mm):
    start=time.perf_counter();q,n,ids=frame_power(height_mm,scan_id,sigma_mm)
    fit=fit_power(q,n);threshold,_=null_gain_threshold(tuple(n))
    k=2 if fit['gain']>threshold else 1
    info=dict(k=k,gap_mm=fit['gap_over_sigma']*sigma_mm if k==2 else 0.,
              unconstrained_gap_mm=fit['gap_over_sigma']*sigma_mm,
              power_gain=fit['gain'],null_gain_threshold=threshold,
              scan_amplitudes=fit['amplitude'].tolist(),scan_counts=n.tolist(),
              range_sigma=[.25,12.],frequency_count=len(FREQUENCY),
              null_replicates=NULL_REPLICATES,seconds=time.perf_counter()-start,
              model='parallel layers, common per-scan additive bias, supplied iid Gaussian sigma',
              truth_fields_used=[])
    return info,dict(power_curves=q,power_scan_ids=ids,power_scan_counts=n,
                     angular_frequency_per_mm=FREQUENCY/sigma_mm,
                     gap_grid_mm=GAP_GRID*sigma_mm,gap_profile=fit['profile'])

def _single(x,y,sigma):
    beta=np.linalg.lstsq(x,y,rcond=1e-12)[0];pred=x@beta
    ll=-.5*((y-pred)/sigma)**2
    return dict(k=1,beta=beta,means=np.array([beta[0]]),slope=beta[1:],
                prediction=pred,groups=np.zeros(len(y),int),score=float(-2*ll.sum()+3*np.log(len(y))))

def fit_surfaces(design,y,scan_id,sigma,gap=None,force_k=None,iterations=24):
    """Matched pooled likelihood backend: fixed-gap or free-gap parallel planes.

Frame-specific mixture proportions are estimated. Bias/axis remain the same
upstream for both methods. No hard-group refit is applied to final predictions.
"""
    x=np.asarray(design);y=np.asarray(y);_,scan=np.unique(scan_id,return_inverse=True)
    f=int(scan.max())+1;n=len(y);one=_single(x,y,sigma)
    if force_k==1:return one
    candidates=[]
    for slope_seed in (one['slope'],np.zeros(2)):
        residual=y-x[:,1:]@slope_seed
        for quantiles in ((.2,.8),(.1,.9)):
            levels=np.quantile(residual,quantiles)
            if gap is not None:levels=levels.mean()+np.array([-.5,.5])*gap
            slope=slope_seed.copy();pi=np.full((f,2),.5)
            for _ in range(iterations):
                prediction=levels[None,:]+(x[:,1:]@slope)[:,None]
                logp=-.5*((y[:,None]-prediction)/sigma)**2+np.log(pi[scan])
                r=np.exp(logp-logsumexp(logp,axis=1)[:,None])
                for j in range(f):
                    pi[j]=np.maximum(r[scan==j].mean(0),.001);pi[j]/=pi[j].sum()
                if gap is not None:
                    target=y-(r[:,1]-.5)*gap
                    beta=np.linalg.lstsq(x,target,rcond=1e-12)[0]
                    levels=beta[0]+np.array([-.5,.5])*gap;slope=beta[1:]
                else:
                    a=np.zeros((n*2,4));a[:,:2]=np.tile(np.eye(2),(n,1));a[:,2:]=np.repeat(x[:,1:],2,axis=0)
                    w=np.sqrt(r.ravel());beta=np.linalg.lstsq(a*w[:,None],np.repeat(y,2)*w,rcond=1e-12)[0]
                    levels=beta[:2];slope=beta[2:]
            prediction=levels[None,:]+(x[:,1:]@slope)[:,None]
            logp=-.5*((y[:,None]-prediction)/sigma)**2+np.log(pi[scan])
            ll=logsumexp(logp,axis=1);g=np.argmax(logp,axis=1)
            score=float(-2*ll.sum()+(4+f)*np.log(n))
            candidates.append(dict(k=2,means=levels,slope=slope,groups=g,
                 prediction=prediction[np.arange(n),g],score=score,
                 fitted_gap_mm=float(abs(levels[1]-levels[0])),mixture_pi=pi))
    best=min(candidates,key=lambda c:c['score'])
    return best if force_k==2 or best['score']<one['score'] else one

def filter_frozen(state,sigma_mm,method='spectrum_fixed',iterations=24):
    start=time.perf_counter()
    if method not in ('spectrum_fixed','pooled_em'):raise ValueError('unknown method')
    yraw=state['local'][:,2];scans=state['scan_id'][state['order']]
    diagnostics={};spectrum=None
    if method=='spectrum_fixed':
        spectrum,diagnostics=estimate_gap(yraw,scans,sigma_mm)
        model=fit_surfaces(state['design'],state['corrected'],scans,sigma_mm,
                          gap=spectrum['gap_mm'] if spectrum['k']==2 else None,
                          force_k=spectrum['k'],iterations=iterations)
    else:model=fit_surfaces(state['design'],state['corrected'],scans,sigma_mm,iterations=iterations)
    output=state['world'].copy();take=np.flatnonzero(state['support']);rows=state['order'][take]
    output[rows]+=(model['prediction'][take]-yraw[take])[:,None]*state['normal']/1000.
    support=np.zeros(len(output),bool);support[state['order']]=state['support']
    groups=np.empty(len(output),int);groups[state['order']]=model['groups']
    prediction=np.empty(len(output));prediction[state['order']]=model['prediction']
    np.testing.assert_array_equal(output[~support],state['world'][~support])
    assert np.isfinite(output).all()
    info=dict(method=method,iterations=iterations,k=model['k'],
       fitted_gap_mm=float(abs(np.diff(model['means'])[0])) if model['k']==2 else 0.,
       spectrum=spectrum,score=model['score'],truth_fields_used=[],
       fitted_rows=len(yraw),supported_fraction=float(support.mean()),
       model_scope='pooled local parallel surfaces; unit likelihood weights; same old axis and bias',
       seconds=time.perf_counter()-start)
    art=dict(xyz_world=output,support_mask=support,group_ids=groups,
             prediction_world_order_mm=prediction,means_mm=model['means'],slope=model['slope'],**diagnostics)
    return output,info,art

def estimate(xyz_world_m,scan_id,sigma_mm,method='spectrum_fixed',iterations=24):
    """Full legal-input API; inherits the old within-scan normal/bias front end."""
    from pathlib import Path
    import importlib.util
    spec=importlib.util.spec_from_file_location('_v10_v7',Path(__file__).resolve().parents[1]/'exploration_v7/algorithm/reassociation.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    start=time.perf_counter();state=module.freeze(xyz_world_m,scan_id,sigma_mm)
    if 'local' not in state:
        return np.array(xyz_world_m,copy=True),dict(status='UNSUPPORTED',truth_fields_used=[]),{}
    out,info,art=filter_frozen(state,sigma_mm,method,iterations)
    info['full_api_seconds']=time.perf_counter()-start
    return out,info,art

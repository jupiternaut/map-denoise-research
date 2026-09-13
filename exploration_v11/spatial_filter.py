"""Spatial mixture-of-experts decoding, optionally guided by frame spectra."""
from pathlib import Path
import sys,time
import numpy as np
from scipy.special import expit,logsumexp
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'exploration_v10'))
from spectral_filter import estimate_gap

METHODS=('global_free','spatial_free','spatial_fixed','spatial_seed')
RIDGE=.1

def features(xy,spatial=True):
    if not spatial:return np.ones((len(xy),1))
    center=np.median(xy,axis=0);scale=np.maximum(np.ptp(xy,axis=0)/2,1e-6)
    z=(xy-center)/scale
    anchors=np.array([(i,j) for i in (-.75,0,.75) for j in (-.75,0,.75)])
    rbf=np.exp(-np.sum((z[:,None,:]-anchors[None,:,:])**2,axis=2)/(2*.6**2))
    return np.column_stack((np.ones(len(z)),z,rbf))

def logistic(z,target,beta):
    beta=beta.copy();penalty=np.full(z.shape[1],RIDGE);penalty[0]=1e-6
    for _ in range(5):
        p=expit(z@beta);w=np.maximum(p*(1-p),1e-6)
        grad=z.T@(p-target)+penalty*beta
        h=z.T@(w[:,None]*z)+np.diag(penalty)
        step=np.linalg.solve(h,grad)
        # Bound the Newton step rather than allow saturated gates to overflow.
        step*=min(1.,3./max(float(np.linalg.norm(step)),1e-12));beta-=step
    return beta

def geometry(x,y,r,gap):
    if gap is not None:
        beta=np.linalg.lstsq(x,y-(r[:,1]-.5)*gap,rcond=1e-12)[0]
        return beta[0]+np.array([-.5,.5])*gap,beta[1:]
    n=len(y);a=np.column_stack((np.tile(np.eye(2),(n,1)),np.repeat(x[:,1:],2,axis=0)))
    w=np.sqrt(r.ravel());b=np.linalg.lstsq(a*w[:,None],np.repeat(y,2)*w,rcond=1e-12)[0]
    return b[:2],b[2:]

def solve(x,y,sigma,spatial=True,gap=None,seed_gap=None,force_k=None,iterations=24,preserve_seed_order=False):
    n=len(y);z=features(x[:,1:],spatial);pdim=z.shape[1]
    one=np.linalg.lstsq(x,y,rcond=1e-12)[0];onepred=x@one
    single=dict(k=1,means=one[:1],slope=one[1:],prediction=onepred,
        groups=np.zeros(n,int),score=float(np.sum(((y-onepred)/sigma)**2)+3*np.log(n)),gate=np.zeros(pdim))
    if force_k==1:return single
    residual=y-onepred
    starts=[expit((residual-np.quantile(residual,q))/sigma) for q in (.35,.65)]
    for axis in (0,1):
        coord=x[:,axis+1];coord=(coord-np.median(coord))/max(np.std(coord),1e-6)
        starts.extend([expit(3*coord),expit(-3*coord)])
    candidates=[]
    for probability in starts:
        r=np.column_stack((1-probability,probability));gate=logistic(z,probability,np.zeros(pdim))
        levels,slope=geometry(x,y,r,gap)
        if seed_gap is not None and gap is None:
            orientation=(1. if levels[1]>=levels[0] else -1.) if preserve_seed_order else 1.
            levels=levels.mean()+orientation*np.array([-.5,.5])*seed_gap
        for _ in range(iterations):
            p=np.clip(expit(z@gate),1e-9,1-1e-9)
            pred=levels[None,:]+(x[:,1:]@slope)[:,None]
            lp=-.5*((y[:,None]-pred)/sigma)**2+np.log(np.column_stack((1-p,p)))
            r=np.exp(lp-logsumexp(lp,axis=1)[:,None])
            levels,slope=geometry(x,y,r,gap)
            gate=logistic(z,r[:,1],gate)
        p=np.clip(expit(z@gate),1e-9,1-1e-9)
        pred=levels[None,:]+(x[:,1:]@slope)[:,None]
        lp=-.5*((y[:,None]-pred)/sigma)**2+np.log(np.column_stack((1-p,p)))
        g=np.argmax(lp,axis=1)
        score=float(-2*logsumexp(lp,axis=1).sum()+RIDGE*np.sum(gate[1:]**2)+(4+pdim)*np.log(n))
        candidates.append(dict(k=2,means=levels,slope=slope,prediction=pred[np.arange(n),g],groups=g,gate=gate,score=score))
    best=min(candidates,key=lambda m:m['score'])
    return best if force_k==2 or best['score']<single['score'] else single

def filter_frozen(state,sigma_mm,method='spatial_free',iterations=24):
    if method not in METHODS+('spatial_seed_aligned',):raise ValueError(method)
    start=time.perf_counter();yraw=state['local'][:,2];scan=state['scan_id'][state['order']]
    spectrum=None;diagnostics={}
    if method in ('spatial_fixed','spatial_seed','spatial_seed_aligned'):spectrum,diagnostics=estimate_gap(yraw,scan,sigma_mm)
    fixed=method=='spatial_fixed';seed=method in ('spatial_seed','spatial_seed_aligned')
    model=solve(state['design'],state['corrected'],sigma_mm,spatial=method!='global_free',
        gap=spectrum['gap_mm'] if fixed and spectrum['k']==2 else None,
        seed_gap=spectrum['unconstrained_gap_mm'] if seed else None,
        force_k=spectrum['k'] if fixed else None,iterations=iterations,preserve_seed_order=method=='spatial_seed_aligned')
    out=state['world'].copy();take=np.flatnonzero(state['support']);rows=state['order'][take]
    out[rows]+=(model['prediction'][take]-yraw[take])[:,None]*state['normal']/1000.
    support=np.zeros(len(out),bool);support[state['order']]=state['support']
    groups=np.empty(len(out),int);groups[state['order']]=model['groups']
    pred=np.empty(len(out));pred[state['order']]=model['prediction']
    np.testing.assert_array_equal(out[~support],state['world'][~support]);assert np.isfinite(out).all()
    info=dict(method=method,iterations=iterations,k=model['k'],fitted_gap_mm=float(abs(np.diff(model['means'])[0])) if model['k']==2 else 0.,
        spectrum=spectrum,seconds=time.perf_counter()-start,supported_fraction=float(support.mean()),
        score=model['score'],gate_parameters=len(model['gate']),starts=6,truth_fields_used=[],
        scope='old axis/bias/support; full-patch parallel surfaces; spatial mixture prior on measured XY')
    return out,info,dict(xyz_world=out,support_mask=support,group_ids=groups,prediction_world_order_mm=pred,
        means_mm=model['means'],slope=model['slope'],gate=model['gate'],**diagnostics)

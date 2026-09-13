"""Specialized local parallel-sheet filter with per-frame sampling proportions.

Legal inputs: aligned local XYZ, frame ids, coarse ROI, supplied scalar noise and
bias bound. No GT layer labels, coordinates, bias or layer gap are accepted.
ROI>0 retains the previous interface's declared stable single-plane reference.
All numerical values are millimetres. This is an EM prototype, not a new theorem.
"""
import time
import numpy as np
from scipy.special import logsumexp

METHODS=('framewise_joint_soft','framewise_joint_hard','framewise_joint_global_pi')
CONFIG={'iterations':160,'local_iterations':50,'pseudocount':.5,'tolerance':1e-6,
        'crossed_mass':6.,'crossed_confident':4,'separation_sigma':3.}

def _mean(y,f,nf):
    return np.bincount(f,weights=y,minlength=nf)/np.maximum(np.bincount(f,minlength=nf),1)

def _box_gauge(value,count,bound):
    """Euclidean common shift followed by clipping, weighted mean constrained 0."""
    lo,hi=float(np.min(value)-bound),float(np.max(value)+bound)
    for _ in range(48):
        middle=(lo+hi)/2
        if np.dot(np.clip(value-middle,-bound,bound),count)>0:lo=middle
        else:hi=middle
    return np.clip(value-(lo+hi)/2,-bound,bound)

def _local_two(y,sigma):
    best=None
    for q in ((.1,.9),(.25,.75)):
        mu=np.quantile(y,q);pi=np.array([.5,.5])
        for _ in range(CONFIG['local_iterations']):
            lp=-.5*((y[:,None]-mu)/sigma)**2+np.log(pi)
            r=np.exp(lp-logsumexp(lp,axis=1)[:,None]);mass=r.sum(0)+1e-12
            new=(r*y[:,None]).sum(0)/mass
            pi=(mass+CONFIG['pseudocount'])/(len(y)+2*CONFIG['pseudocount'])
            if np.max(abs(new-mu))<1e-5:mu=new;break
            mu=new
        lp=-.5*((y[:,None]-mu)/sigma)**2+np.log(pi)
        score=float(logsumexp(lp,axis=1).sum())
        if best is None or score>best[0]:best=(score,mu,pi)
    return best[1],best[2]

def _initializations(y,f,target,reference,nf,count,sigma,bound):
    starts=[('zero',np.zeros(nf)),('frame_mean',_mean(y[target],f[target],nf)-y[target].mean())]
    midpoint=np.full(nf,np.nan)
    for fid in range(nf):
        local=y[target&(f==fid)]
        if len(local)<12:continue
        mu,pi=_local_two(local,sigma)
        if np.ptp(mu)>2.5*sigma and pi.min()*len(local)>=3:
            midpoint[fid]=mu.mean()
    valid=np.isfinite(midpoint)
    if valid.sum()>=2:
        center=np.average(midpoint[valid],weights=count[valid])
        local_init=starts[1][1].copy();local_init[valid]=midpoint[valid]-center
        starts.append(('frame_local_midpoint',local_init))
    if reference.any():
        starts.append(('stable_reference',_mean(y[reference],f[reference],nf)-y[reference].mean()))
    return [(name,_box_gauge(value,count,bound)) for name,value in starts]

def _fit(inp,per_frame):
    xyz=np.asarray(inp.xyz_mm,dtype=float);original_frame=np.asarray(inp.frame)
    frame_ids,f=np.unique(original_frame,return_inverse=True);nf=len(frame_ids)
    target=np.asarray(inp.roi)==0;reference=~target;y=xyz[:,2]
    sigma=float(inp.sigma_mm);bound=float(inp.bias_bound_mm)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or not np.isfinite(xyz).all():raise ValueError('finite Nx3 required')
    if sigma<=0 or bound<=0 or not target.any():raise ValueError('positive noise/bias and target ROI required')
    count=np.bincount(f,minlength=nf);ft=f[target];yt=y[target];nt=len(yt)
    starts=_initializations(y,f,target,reference,nf,count,sigma,bound);best=None
    norm=-np.log(sigma*np.sqrt(2*np.pi));candidate_scores=[]
    for k in (1,2):
        # One-plane optimum does not need every two-layer start.
        for name,start in starts if k==2 else starts[:1]:
            b=start.copy();corrected=y-b[f]
            mu=np.quantile(corrected[target],[.2,.8]) if k==2 else np.array([corrected[target].mean()])
            anchor=float(corrected[reference].mean()) if reference.any() else 0.
            pi=np.full((nf,k),1/k)
            for iteration in range(CONFIG['iterations']):
                lp=-.5*((yt[:,None]-b[ft,None]-mu)/sigma)**2+np.log(pi[ft])
                r=np.exp(lp-logsumexp(lp,axis=1)[:,None]);mass=r.sum(0)+1e-12
                newmu=(r*(yt-b[ft])[:,None]).sum(0)/mass
                if per_frame and k==2:
                    frame_mass=np.stack([np.bincount(ft,weights=r[:,j],minlength=nf) for j in range(k)],axis=1)
                    newpi=(frame_mass+CONFIG['pseudocount'])/(frame_mass.sum(1,keepdims=True)+k*CONFIG['pseudocount'])
                else:
                    pp=(mass+CONFIG['pseudocount'])/(nt+k*CONFIG['pseudocount']);newpi=np.tile(pp,(nf,1))
                newanchor=float((y-b[f])[reference].mean()) if reference.any() else 0.
                pred=np.empty(len(y));pred[target]=r@newmu
                if reference.any():pred[reference]=newanchor
                newb=_box_gauge(_mean(y-pred,f,nf),count,bound)
                # The bias gauge is enforced by constrained update, not by moving
                # geometry afterwards. Refit surface positions to updated biases.
                change=max(float(np.max(abs(newb-b))),float(np.max(abs(newmu-mu))),float(np.max(abs(newpi-pi))))
                mu,b,pi,anchor=newmu,newb,newpi,newanchor
                if change<CONFIG['tolerance']:break
            lp=-.5*((yt[:,None]-b[ft,None]-mu)/sigma)**2+np.log(pi[ft])+norm
            ll=logsumexp(lp,axis=1);r=np.exp(lp-ll[:,None]);loglike=float(ll.sum())
            if reference.any():loglike+=float((-.5*((y[reference]-b[f[reference]]-anchor)/sigma)**2+norm).sum())
            dimensions=(nf-1)+k+(nf if per_frame else 1)*(k-1)+int(reference.any())
            bic=-2*loglike+dimensions*np.log(len(y))
            candidate_scores.append({'k':k,'start':name,'bic':float(bic),'iterations':iteration+1})
            if best is None or bic<best['score']:
                best={'score':float(bic),'k':k,'mu':mu.copy(),'b':b.copy(),'pi':pi.copy(),'r':r.copy(),
                      'anchor_mu':anchor,'start':name,'iterations':iteration+1,'frame_ids':frame_ids,'frame_index':f}
    crossed=0
    if best['k']==2:
        for fid in range(nf):
            rr=best['r'][ft==fid]
            required=min(CONFIG['crossed_mass'],max(3.,.05*len(rr)))
            confident=min(CONFIG['crossed_confident'],max(2,int(.025*len(rr))))
            if len(rr) and (rr.sum(0)>=required).all() and ((rr>.9).sum(0)>=confident).all():crossed+=1
    best['observable']=bool(reference.any() or (best['k']==2 and crossed>=2 and np.ptp(best['mu'])>CONFIG['separation_sigma']*sigma))
    best['crossed_frames']=crossed;best['candidate_scores']=candidate_scores
    return best

def estimate(inp,method):
    if method not in METHODS:raise KeyError(method)
    start=time.perf_counter();fit=_fit(inp,per_frame=method!='framewise_joint_global_pi')
    info={'status':'APPLY' if fit['observable'] else 'WAIT','k':fit['k'],'mu_mm':fit['mu'].tolist(),
          'gap_mm':float(np.ptp(fit['mu'])),'score':fit['score'],'start':fit['start'],
          'frame_ids':fit['frame_ids'].tolist(),'frame_pi':fit['pi'].tolist(),'crossed_frames':fit['crossed_frames'],
          'observable':fit['observable'],'iterations':fit['iterations'],'candidate_scores':fit['candidate_scores'],
          'projection':'MAP' if method.endswith('_hard') else 'posterior_mean','config':CONFIG.copy(),
          'scope':'known local normal axis; supplied sigma and bias bound; support guard is heuristic',
          'requires_ground_truth':False,'elapsed_s':time.perf_counter()-start}
    nf=int(np.max(inp.frame))+1;b=np.zeros(nf)
    if not fit['observable']:
        info['latent_bias_mm']=fit['b'].tolist()
        return np.asarray(inp.xyz_mm,dtype=float).copy(),b,info
    out=np.asarray(inp.xyz_mm,dtype=float).copy();target=np.asarray(inp.roi)==0
    out[target,2]=fit['mu'][np.argmax(fit['r'],axis=1)] if method.endswith('_hard') else fit['r']@fit['mu']
    if (~target).any():out[~target,2]=fit['anchor_mu']
    b[fit['frame_ids'].astype(int)]=fit['b']
    return out,b,info

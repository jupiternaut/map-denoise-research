"""Frame-local split -> gap/center consensus -> specialized normal denoising.

Input duck-types the frozen original Input. No evaluator truth is accepted.
No joint latent frame-bias/global-mixture EM: each frame is fitted independently;
only estimated layer distance and de-biased center are pooled afterwards.
"""
import numpy as np
from scipy.special import logsumexp

METHODS=("frame_split_consensus", "frame_split_soft", "frame_split_local")


def _fit(y,sigma):
    n=len(y);sigma=max(float(sigma),1e-6)
    mu1=float(np.mean(y));ll1=-.5*np.sum(((y-mu1)/sigma)**2)-n*np.log(sigma*np.sqrt(2*np.pi))
    best={"k":1,"mu":np.array([mu1]),"pi":np.array([1.]),"r":np.ones((n,1)),"bic":float(-2*ll1+np.log(max(n,2)))}
    if n<8:return best
    for quantiles in ((.2,.8),(.1,.9)):
        mu=np.quantile(y,quantiles);pi=np.full(2,.5)
        for _ in range(45):
            lp=-.5*((y[:,None]-mu)/sigma)**2+np.log(pi)
            r=np.exp(lp-logsumexp(lp,axis=1)[:,None]);mass=r.sum(0)+1e-9
            new=(r*y[:,None]).sum(0)/mass
            pi=np.clip(mass/n,.01,.99);pi/=pi.sum()
            delta=np.max(abs(new-mu));mu=new
            if delta<1e-5:break
        order=np.argsort(mu);mu=mu[order];pi=pi[order]
        lp=-.5*((y[:,None]-mu)/sigma)**2+np.log(pi)-np.log(sigma*np.sqrt(2*np.pi))
        ll=logsumexp(lp,axis=1);r=np.exp(lp-ll[:,None]);mass=r.sum(0)
        bic=float(-2*ll.sum()+3*np.log(n))
        if bic<best["bic"] and np.ptp(mu)>1.5*sigma and mass.min()>=max(3.,.03*n):
            best={"k":2,"mu":mu,"pi":pi,"r":r,"bic":bic}
    return best


def _fixed_gap(y,gap,sigma,center,pi):
    # Per-frame location and proportion refinement; distance remains frozen.
    pi=pi.copy()
    for _ in range(25):
        mu=center+np.array([-.5,.5])*gap
        lp=-.5*((y[:,None]-mu)/sigma)**2+np.log(pi)
        r=np.exp(lp-logsumexp(lp,axis=1)[:,None])
        newcenter=float(np.mean(y-(r[:,1]-.5)*gap))
        pi=np.clip(r.mean(0),.01,.99);pi/=pi.sum()
        delta=abs(center-newcenter);center=newcenter
        if delta<1e-5:break
    mu=center+np.array([-.5,.5])*gap
    lp=-.5*((y[:,None]-mu)/sigma)**2+np.log(pi)
    r=np.exp(lp-logsumexp(lp,axis=1)[:,None])
    return center,pi,r


def _weighted_median(v,w):
    order=np.argsort(v);v=np.asarray(v)[order];w=np.asarray(w)[order]
    return float(v[np.searchsorted(np.cumsum(w),.5*np.sum(w))])


def _emit(r,mu,soft):
    return r@mu if soft else mu[np.argmax(r,axis=1)]


def estimate(inp,method):
    if method not in METHODS:raise KeyError(method)
    xyz=np.asarray(inp.xyz_mm);out=xyz.copy();frame=np.asarray(inp.frame);roi=np.asarray(inp.roi)
    nf=int(frame.max())+1;b=np.zeros(nf);sigma=max(float(inp.sigma_mm),1e-6)
    target=roi==0;reference=roi==1;soft=method=="frame_split_soft";local=method=="frame_split_local"
    info={"status":"APPLY","method":method,"normal_supplied":True,"per_frame_pi":{},"supported_frames":[],"missing_layer_policy":"local denoise without claimed bias correction"}
    # A declared stable reference provides a direct, separate bias estimate.
    if reference.any():
        counts=np.bincount(frame[reference],minlength=nf)
        means=np.bincount(frame[reference],weights=xyz[reference,2],minlength=nf)/np.maximum(counts,1)
        valid=counts>0;refmean=float(np.average(means[valid],weights=counts[valid]))
        b[valid]=means[valid]-refmean
        corrected=xyz[target,2]-b[frame[target]]
        pooled=_fit(corrected,sigma);mu=pooled["mu"]
        for f in np.unique(frame[target]):
            m=target&(frame==f);y=xyz[m,2]-b[f]
            if pooled["k"]==1:out[m,2]=mu[0]
            else:
                pi=pooled["pi"].copy()
                for _ in range(15):
                    lp=-.5*((y[:,None]-mu)/sigma)**2+np.log(pi)
                    r=np.exp(lp-logsumexp(lp,axis=1)[:,None]);pi=np.clip(r.mean(0),.01,.99);pi/=pi.sum()
                out[m,2]=_emit(r,mu,soft)
                info["per_frame_pi"][str(int(f))]=pi.tolist()
        out[reference,2]=refmean
        info.update({"route":"reference_then_shared_split","k":pooled["k"],"mu_mm":mu.tolist(),"gap_mm":float(np.ptp(mu)),"reference_frames":np.flatnonzero(valid).tolist()})
        return out,b,info
    fits={}
    for f in np.unique(frame[target]):
        m=target&(frame==f);fit=_fit(xyz[m,2],sigma);fits[int(f)]=fit
        out[m,2]=_emit(fit["r"],fit["mu"],soft)
        info["per_frame_pi"][str(int(f))]=fit["pi"].tolist()
    supported=[f for f,fit in fits.items() if fit["k"]==2]
    info["supported_frames"]=supported
    if len(supported)<2:
        info.update({"route":"local_only_insufficient_crossed_support","gap_mm":None,"k":None})
        return out,b,info
    gaps=np.array([np.ptp(fits[f]["mu"]) for f in supported])
    # Effective two-group sample size weights stable frame-wise gap estimates.
    weights=np.array([1./np.sum(1./np.maximum(fits[f]["r"].sum(0),1e-9)) for f in supported])
    median=_weighted_median(gaps,weights)
    keep=np.abs(gaps-median)<=max(sigma,median*.35)
    gap=float(np.average(gaps[keep],weights=weights[keep]))
    centers=[];counts=[];refined={}
    for f in supported:
        m=target&(frame==f);fit=fits[f];usegap=float(np.ptp(fit["mu"])) if local else gap
        c,pi,r=_fixed_gap(xyz[m,2],usegap,sigma,float(fit["mu"].mean()),fit["pi"])
        refined[f]=(c,pi,r,usegap);centers.append(c);counts.append(int(m.sum()))
    globalcenter=float(np.average(centers,weights=counts))
    for f in supported:
        m=target&(frame==f);c,pi,r,usegap=refined[f]
        b[f]=c-globalcenter
        out[m,2]=_emit(r,globalcenter+np.array([-.5,.5])*usegap,soft)
        info["per_frame_pi"][str(f)]=pi.tolist()
    info.update({"route":"frame_split_gap_center_consensus","gap_mm":gap,"mu_mm":[globalcenter-gap/2,globalcenter+gap/2],"k":2,"pooled_frames":len(supported),"unsupported_frames":[f for f in fits if f not in supported],"global_gauge":"weighted mean estimated bias on supported frames is zero; missing frames do not anchor absolute position","bias_bound_usage":"not used: consensus estimates have no strict bound guarantee"})
    return out,b,info

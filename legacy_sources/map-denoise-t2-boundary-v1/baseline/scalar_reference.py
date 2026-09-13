"""Own strong scalar references, not JRMPC/BALM implementations.

Frame intercepts are eliminated for a sorted hard-assignment profile initializer.
An independent SciPy L-BFGS-B solver then optimizes the observed mixture likelihood
with a common layer separation and individual frame proportions. No EM updates.
"""
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logsumexp

METHODS=('scalar_profile_hard','scalar_profile_lbfgs')

def _data(inp):
    xyz=np.asarray(inp.xyz_mm,float); f=np.asarray(inp.frame,int); roi=np.asarray(inp.roi,int)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or len(xyz)!=len(f) or not np.isfinite(xyz).all(): raise ValueError('finite XYZ and frame required')
    if len(f)==0 or np.min(f)<0 or inp.sigma_mm<=0: raise ValueError('nonempty frames and positive noise required')
    nf=int(f.max())+1; target=roi==0; anchor=roi!=0
    if not target.any(): raise ValueError('target ROI required')
    if np.any(np.bincount(f[target],minlength=nf)==0): raise ValueError('each indexed frame must contain target observations')
    return xyz,f,target,anchor,nf

def _profile(y,f,nf,sigma):
    tables=[]; means=np.zeros(nf); counts=np.bincount(f,minlength=nf)
    for frame in range(nf):
        row=np.sort(y[f==frame]); n=len(row); mean=row.mean(); means[frame]=mean
        lo=np.arange(n+1,dtype=float); hi=n-lo
        prefix=np.r_[0.,np.cumsum(row)]; c=row.sum()-prefix-hi*mean
        q=lo*hi/n
        entropy=n*np.log(n)-np.where(lo>0,lo*np.log(np.maximum(lo,1)),0)-np.where(hi>0,hi*np.log(np.maximum(hi,1)),0)
        tables.append((np.sum((row-mean)**2),c,q,entropy,hi/n))
    upper=max(float(np.quantile(y[f==j],.99)-np.quantile(y[f==j],.01)) for j in range(nf))
    upper=max(upper,2*sigma)
    def profile(gap,details=False):
        cost=0.; positions=[]; probabilities=[]; cuts=[]
        for frame,(s,c,q,h,p) in enumerate(tables):
            values=s-2*gap*c+gap*gap*q+2*sigma*sigma*h
            cut=int(np.argmin(values)); cost+=values[cut]
            positions.append(means[frame]-p[cut]*gap);probabilities.append(p[cut]);cuts.append(cut)
        return (float(cost),np.asarray(positions),np.asarray(probabilities),cuts) if details else float(cost)
    grid=np.linspace(0,upper,33); scores=np.array([profile(gap) for gap in grid]); candidates=[(float(scores[0]),0.)]
    for j in np.argsort(scores)[:4]:
        lo=grid[max(0,j-1)];hi=grid[min(len(grid)-1,j+1)]
        if hi>lo:
            result=minimize_scalar(profile,bounds=(lo,hi),method='bounded',options={'xatol':1e-5,'maxiter':40})
            candidates.append((float(result.fun),float(result.x)))
    _,gap=min(candidates);cost,positions,pi,cuts=profile(gap,True)
    return {'gap':gap,'a':positions,'pi':np.clip(pi,.001,.999),'objective':cost,'upper':upper,'cuts':cuts}

def _objective(theta,y,f,ry,rf,nf,sigma,k):
    a=theta[:nf]; anchor=len(ry)>0; variance=sigma*sigma
    grad=np.zeros_like(theta)
    if k==1:
        residual=y-a[f]; value=.5*np.sum(residual**2)/variance
        grad[:nf]=-np.bincount(f,weights=residual/variance,minlength=nf)
        offset=nf
    else:
        gap=theta[nf]; logits=theta[nf+1:2*nf+1]; pi=expit(logits)
        e0=y-a[f]; e1=e0-gap
        lp0=-.5*e0**2/variance-np.logaddexp(0,logits[f])
        lp1=-.5*e1**2/variance-np.logaddexp(0,-logits[f])
        ll=np.logaddexp(lp0,lp1); r1=np.exp(lp1-ll)
        value=-ll.sum()
        grad[:nf]=-np.bincount(f,weights=((1-r1)*e0+r1*e1)/variance,minlength=nf)
        grad[nf]=-np.sum(r1*e1)/variance
        grad[nf+1:2*nf+1]=np.bincount(f,weights=pi[f]-r1,minlength=nf)
        offset=2*nf+1
    if anchor:
        r=ry-a[rf]-theta[offset]
        value+=.5*np.sum(r*r)/variance
        grad[:nf]-=np.bincount(rf,weights=r/variance,minlength=nf)
        grad[offset]=-r.sum()/variance
    return float(value),grad

def estimate(inp,method):
    if method not in METHODS: raise KeyError(method)
    xyz,f,target,anchor,nf=_data(inp); sigma=float(inp.sigma_mm)
    y=xyz[target,2];tf=f[target];ry=xyz[anchor,2];rf=f[anchor]
    counts=np.bincount(f,minlength=nf)
    frame_mean=np.bincount(tf,weights=y,minlength=nf)/np.bincount(tf,minlength=nf)
    init=_profile(y,tf,nf,sigma); has_anchor=bool(anchor.any()); fits=[]
    # Independent one-layer numerical reference, including any known stable ROI.
    t0=float(np.mean(ry-frame_mean[rf])) if has_anchor else None
    x0=np.r_[frame_mean,t0] if has_anchor else frame_mean.copy()
    one=minimize(_objective,x0,args=(y,tf,ry,rf,nf,sigma,1),jac=True,method='L-BFGS-B',options={'maxiter':150,'ftol':1e-12,'gtol':1e-7})
    fits.append({'k':1,'a':one.x[:nf],'gap':0.,'pi':np.zeros(nf),'anchor_delta':float(one.x[-1]) if has_anchor else 0.,
                 'score':float(2*one.fun+(nf+int(has_anchor))*np.log(len(xyz))),
                 'success':bool(one.success),'iterations':int(one.nit)})
    if method=='scalar_profile_hard' and not has_anchor:
        # Penalized complete-data classification likelihood, not marginal BIC.
        fits.append({'k':2,'a':init['a'],'gap':init['gap'],'pi':init['pi'],'anchor_delta':0.,
          'score':float(init['objective']/(sigma*sigma)+(2*nf+1)*np.log(len(xyz))),
          'success':True,'iterations':0})
    else:
        starts=[]
        for factor in (.65,1.,1.4):
            gap=max(factor*init['gap'],2*sigma)
            a=frame_mean-init['pi']*gap
            logits=np.log(init['pi']/(1-init['pi']))
            starts.append(np.r_[a,gap,logits,float(np.mean(ry-a[rf]))] if has_anchor else np.r_[a,gap,logits])
        if has_anchor:
            rc=np.bincount(rf,minlength=nf); rm=np.bincount(rf,weights=ry,minlength=nf)/np.maximum(rc,1)
            b=rm-np.average(rm,weights=rc); corrected=y-b[tf]
            mu=np.quantile(corrected,[.2,.8]);a=mu[0]+b
            starts.append(np.r_[a,max(mu[1]-mu[0],sigma),np.zeros(nf),float(np.mean(ry-a[rf]))])
        for start in starts:
            bounds=[(None,None)]*nf+[(1e-6,2*init['upper']+2*sigma)]+[(-7.,7.)]*nf+([(None,None)] if has_anchor else [])
            fit=minimize(_objective,start,args=(y,tf,ry,rf,nf,sigma,2),jac=True,method='L-BFGS-B',bounds=bounds,
                         options={'maxiter':200,'ftol':1e-11,'gtol':1e-6,'maxls':30})
            fits.append({'k':2,'a':fit.x[:nf],'gap':float(fit.x[nf]),'pi':expit(fit.x[nf+1:2*nf+1]),
                'anchor_delta':float(fit.x[-1]) if has_anchor else 0.,
                'score':float(2*fit.fun+(2*nf+1+int(has_anchor))*np.log(len(xyz))),
                'success':bool(fit.success),'iterations':int(fit.nit)})
    best=min(fits,key=lambda x:x['score']);a=best['a'];mu0=float(np.average(a,weights=counts));b=a-mu0
    out=xyz.copy()
    if best['k']==1:
        posterior=np.zeros(len(y));out[target,2]=mu0
    else:
        gap=best['gap'];pi=np.clip(best['pi'],1e-12,1-1e-12)
        e0=y-a[tf];e1=e0-gap
        posterior=expit(.5*(e0**2-e1**2)/(sigma*sigma)+np.log(pi[tf]/(1-pi[tf])))
        if method=='scalar_profile_hard':out[target,2]=mu0+gap*(posterior>=.5)
        else:out[target,2]=mu0+gap*posterior
    if has_anchor:out[anchor,2]=mu0+best['anchor_delta']
    crossed=sum(np.sum((tf==j)&(posterior>.9))>=8 and np.sum((tf==j)&(posterior<.1))>=8 for j in range(nf)) if best['k']==2 else 0
    info={'status':'APPLY','method':method,'k':best['k'],'mu_mm':[mu0]+([mu0+best['gap']] if best['k']==2 else []),
      'gap_mm':best['gap'],'frame_pi':best['pi'].tolist(),'crossed_frames':int(crossed),
      'score':best['score'],'optimizer_success':best['success'],'iterations':best['iterations'],
      'all_scores':[x['score'] for x in fits], 'profile_gap_mm':init['gap'],
      'scope':'own scalar profile/L-BFGS reference, not JRMPC/BALM; BIC-style selection heuristic; no ambiguity guard',
      'bias_bound_used':False,'normal_coordinate_provided':True,'noise_sigma_provided':True}
    if not np.isfinite(out).all() or not np.isfinite(b).all(): raise FloatingPointError('nonfinite output')
    return out,b,info

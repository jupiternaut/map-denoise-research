"""Specialized parallel-plane mixture with jointly estimated common tilt.

Independent L-BFGS solver; two geometric initializations; normal projection.
Requires approximate local z axis, not a supplied exact normal. No truth input.
"""
import importlib.util
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

_spec=importlib.util.spec_from_file_location('_frozen_scalar_profile',Path(__file__).with_name('scalar_reference.py'))
_scalar=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_scalar)
METHODS=('plane_profile_map',)

def _objective(theta,y,f,X,ry,rf,RX,nf,sigma,k):
    variance=sigma*sigma; a=theta[:nf]; offset=nf if k==1 else 2*nf+1
    slope=theta[offset:offset+2]; grad=np.zeros_like(theta)
    e0=y-a[f]-X@slope
    if k==1:
        value=.5*np.sum(e0*e0)/variance; weighted=e0/variance
    else:
        gap=theta[nf];logits=theta[nf+1:2*nf+1];e1=e0-gap
        lp0=-.5*e0*e0/variance-np.logaddexp(0,logits[f])
        lp1=-.5*e1*e1/variance-np.logaddexp(0,-logits[f])
        ll=np.logaddexp(lp0,lp1);r1=np.exp(lp1-ll)
        value=-ll.sum();weighted=((1-r1)*e0+r1*e1)/variance
        grad[nf]=-np.sum(r1*e1)/variance
        grad[nf+1:2*nf+1]=np.bincount(f,weights=expit(logits[f])-r1,minlength=nf)
    grad[:nf]=-np.bincount(f,weights=weighted,minlength=nf)
    grad[offset:offset+2]=-X.T@weighted
    if len(ry):
        er=ry-a[rf]-RX@slope-theta[offset+2];wr=er/variance
        value+=.5*np.sum(er*er)/variance
        grad[:nf]-=np.bincount(rf,weights=wr,minlength=nf)
        grad[offset:offset+2]-=RX.T@wr;grad[offset+2]=-wr.sum()
    return float(value),grad

def estimate(inp,method='plane_profile_map'):
    if method not in METHODS:raise KeyError(method)
    xyz,f,target,anchor,nf=_scalar._data(inp);sigma=float(inp.sigma_mm)
    center=xyz[:,:2].mean(0);scale=np.maximum(np.std(xyz[:,:2],axis=0),1.)
    XX=(xyz[:,:2]-center)/scale
    X=XX[target];RX=XX[anchor];y=xyz[target,2];ry=xyz[anchor,2];tf=f[target];rf=f[anchor]
    has_anchor=bool(anchor.any());counts=np.bincount(f,minlength=nf);tc=np.bincount(tf,minlength=nf)
    means=np.bincount(tf,weights=y,minlength=nf)/tc
    xmeans=np.array([X[tf==j].mean(0) for j in range(nf)])
    regression=np.linalg.lstsq(X-xmeans[tf],y-means[tf],rcond=None)[0]
    # Both zero-tilt and within-frame regression are used; the latter alone may
    # confuse the true layer/XY association with surface inclination.
    starts=[('zero_slope',np.zeros(2)),('within_frame_regression',regression)]
    fits=[]
    for label,slope in starts:
        detrended=y-X@slope
        init=_scalar._profile(detrended,tf,nf,sigma)
        frame_mean=np.bincount(tf,weights=detrended,minlength=nf)/tc
        for k in (1,2):
            factors=(1.,) if k==1 else (.65,1.,1.4)
            for factor in factors:
                if k==1:
                    a=frame_mean;theta=np.r_[a,slope]
                    bounds=[(None,None)]*(nf+2)
                else:
                    gap=max(factor*init['gap'],2*sigma);a=frame_mean-init['pi']*gap
                    theta=np.r_[a,gap,np.log(init['pi']/(1-init['pi'])),slope]
                    bounds=[(None,None)]*nf+[(1e-6,2*init['upper']+2*sigma)]+[(-7.,7.)]*nf+[(None,None)]*2
                if has_anchor:
                    theta=np.r_[theta,np.mean(ry-RX@slope-a[rf])];bounds.append((None,None))
                fit=minimize(_objective,theta,args=(y,tf,X,ry,rf,RX,nf,sigma,k),jac=True,
                    method='L-BFGS-B',bounds=bounds,options={'maxiter':250,'ftol':1e-11,'gtol':1e-6,'maxls':30})
                offset=nf if k==1 else 2*nf+1
                fits.append({'score':float(2*fit.fun+(nf+2+int(has_anchor)+(nf+1 if k==2 else 0))*np.log(len(xyz))),
                    'k':k,'a':fit.x[:nf].copy(),'gap':float(fit.x[nf]) if k==2 else 0.,
                    'pi':expit(fit.x[nf+1:2*nf+1]) if k==2 else np.zeros(nf),
                    'slope_scaled':fit.x[offset:offset+2].copy(),
                    'anchor_delta':float(fit.x[offset+2]) if has_anchor else 0.,
                    'success':bool(fit.success),'iterations':int(fit.nit),'initialization':label,'factor':factor})
    best=min(fits,key=lambda x:x['score']);a=best['a'];slope=best['slope_scaled']/scale
    # Equation n dot (p - [center_x,center_y,0]) = intercept.
    normal=np.r_[-slope,1.];normal_sq=float(normal@normal);normal_length=np.sqrt(normal_sq)
    mu0=float(np.average(a,weights=counts));bias=(a-mu0)/normal_length
    observed_coord=xyz[:,2]-(xyz[:,:2]-center)@slope
    predicted=np.full(len(xyz),mu0)
    posterior=np.zeros(int(target.sum()))
    if best['k']==2:
        residual=observed_coord[target]-a[tf];gap=best['gap'];pi=np.clip(best['pi'][tf],1e-12,1-1e-12)
        posterior=expit(.5*(residual**2-(residual-gap)**2)/(sigma*sigma)+np.log(pi/(1-pi)))
        predicted[target]+=gap*(posterior>=.5)
    if has_anchor:predicted[anchor]+=best['anchor_delta']
    out=xyz+(predicted-observed_coord)[:,None]*normal[None,:]/normal_sq
    crossed=sum(np.sum((tf==j)&(posterior>.9))>=8 and np.sum((tf==j)&(posterior<.1))>=8 for j in range(nf)) if best['k']==2 else 0
    info={'status':'APPLY','method':method,'k':best['k'],'gap_mm':best['gap']/normal_length,
          'mu_mm':[mu0/normal_length]+([(mu0+best['gap'])/normal_length] if best['k']==2 else []),
          'normal':(normal/normal_length).tolist(),'slope_xy':slope.tolist(),'center_xy_mm':center.tolist(),
          'frame_pi':best['pi'].tolist(),'crossed_frames':int(crossed),'score':best['score'],
          'optimizer_success':best['success'],'iterations':best['iterations'],'initialization':best['initialization'],
          'all_scores':[{'k':fit['k'],'initialization':fit['initialization'],'factor':fit['factor'],'score':fit['score']} for fit in fits],
          'scope':'common tilted parallel planes + normal frame shifts; MAP orthogonal projection; not curved-surface filter',
          'noise_model':'provided sigma in approximate z coordinate, no errors-in-variables covariance estimation',
          'bias_bound_used':False,'ground_truth_used':False}
    if not np.isfinite(out).all() or not np.isfinite(bias).all():raise FloatingPointError('nonfinite plane filter')
    return out,bias,info

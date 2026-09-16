"""Fixed-action loss selector. No independent reference is accepted here."""
import importlib.util
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location('surface_v22',Path(__file__).resolve().parents[1]/'reconstruction_v22/operator.py')
surface=importlib.util.module_from_spec(spec);spec.loader.exec_module(surface)

def project(points,matrix):
    q=np.column_stack([points,np.ones(len(points))])@matrix[:3].T
    uv=np.divide(q[:,:2],q[:,2,None],out=np.full((len(points),2),np.nan),where=abs(q[:,2,None])>1e-12)
    return uv,q[:,2]

def sample(image,uv):
    h,w=image.shape[:2];ok=np.isfinite(uv).all(1)&(uv[:,0]>=0)&(uv[:,0]<w-1)&(uv[:,1]>=0)&(uv[:,1]<h-1)
    xy=np.where(ok[:,None],uv,0);x=np.floor(xy[:,0]).astype(int);y=np.floor(xy[:,1]).astype(int)
    a=(xy[:,0]-x)[:,None];b=(xy[:,1]-y)[:,None]
    colors=((1-b)*((1-a)*image[y,x].astype(float)+a*image[y,x+1])+b*((1-a)*image[y+1,x]+a*image[y+1,x+1]))/255.
    return colors,ok

def actions(points):
    outputs,diag=surface.construct(points);q=np.asarray(points)
    full=outputs['multiscale_full']-q;small=outputs['multiscale_consensus']-q
    outputs['multiscale_matched']=q+np.linalg.norm(small)/max(np.linalg.norm(full),1e-30)*full
    outputs={'identity':q.copy(),**outputs};normal=diag['consensus']['normal']
    for delta in (-1.,-.5,-.25,.25,.5,1.):outputs[f'offset_{delta:+g}']=q+delta*normal
    target=diag['quadratics'][64]['correction']
    fit={k:float(np.mean((np.sum((out-q)*normal,axis=1)-target)**2)) for k,out in outputs.items()}
    return outputs,fit,normal

def photo_loss(points,images,matrices,support):
    """Fixed V x N support. Missing predictions cost worst normalized RGB variance (1).

    Real RGB variance is <=.25; invalid predictions must never improve a score by
    reducing the evaluated domain. Points with <2 initial views are not evaluated.
    """
    colors=[];valid=[]
    for image,matrix in zip(images,matrices):
        uv,z=project(points,matrix);c,ok=sample(image,uv);colors.append(c);valid.append(ok&(z>0))
    colors=np.asarray(colors);valid=np.asarray(valid);n=support.sum(0);eligible=n>=2
    if not eligible.any():return None,dict(points=0,invalid=0)
    denom=np.maximum(n,1)
    mean=np.sum(colors*support[:,:,None],axis=0)/denom[:,None]
    variance=np.sum((colors-mean[None])**2*support[:,:,None],axis=(0,2))/(3*denom)
    invalid=np.any(support&~valid,axis=0);variance[invalid]=1.
    return float(variance[eligible].mean()),dict(points=int(eligible.sum()),invalid=int((invalid&eligible).sum()))

def select(losses):
    """Identity-first tie breaking; unavailable photo evidence returns identity."""
    good=[k for k,v in losses.items() if v is not None and np.isfinite(v)]
    return min(good,key=lambda k:(losses[k],k!='identity',k)) if good else 'identity'

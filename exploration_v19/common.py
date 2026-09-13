"""Shared data, immutable fit caches and representation-aware V19 evaluation."""
from pathlib import Path
import sys,json,hashlib,importlib.util
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.spatial import cKDTree

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path[:0]=[str(ROOT/'exploration_v18'),str(ROOT/'exploration_v17')]
import v18_operator as v18
import v18_run as hist
v14=hist.v14
OLD=v14.RUNS/'retention-v18-ccmexx7q'
sha=v14.sha;save=v14.save;load=v14.load;npz=hist.helper.npz

def import_path(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def make(seed,gap,sigma,ratio,dependence):
    n=768;rng=np.random.default_rng(np.random.SeedSequence([seed,1901]))
    xy=rng.uniform([-60.,-50.],[60.,50.],(n,2));noise=rng.normal(size=n)
    scan=np.arange(n)%2;labels=np.zeros(n,int)
    if gap:
        for f in (0,1):
            ids=np.flatnonzero(scan==f);order=ids[np.argsort(xy[ids,0],kind='stable')]
            labels[order[-int(np.floor(len(ids)*ratio+.5)):]]=1
            if dependence==0:
                r=np.random.default_rng(np.random.SeedSequence([seed,1902,f]))
                labels[ids]=labels[r.permutation(ids)]
    clean=np.c_[xy,gap*labels];observed=clean.copy();observed[:,2]+=sigma*noise
    inp=dict(design=np.c_[np.ones(n),xy/50.],height_mm=observed[:,2],xyz_world=observed/1000.,scan_id=scan)
    truth=dict(gt_layer=labels,gt_clean_xyz_world=clean/1000.,true_means_mm=np.array([0.,gap]) if gap else np.zeros(1),
               true_slope=np.zeros(2),eps_mm=sigma*noise)
    return inp,truth

def compact(m):
    keys=('k','family','means','slope','gate','feature_center','feature_scale','plane','posterior','groups','prediction','bic')
    return {k:m[k] for k in keys if k in m}

def unpack(job):
    assert sha(job['cache'])==job['cache_sha256']
    c=json.loads(Path(job['cache']).read_text());x=np.asarray(c['x']);y=np.asarray(c['y'])
    return c,x,y,v18.arrays(c['baseline'])

def world_support(inp,c,model):
    n=len(inp['xyz_world']);weighted=model.get('representation')=='weighted'
    if model.get('identity'):
        return (inp['xyz_world']*1000.)[:,None,:],np.ones((n,1)),False
    h=np.asarray(model['support_heights'] if weighted else model['prediction'])
    if h.ndim==1:h=h[:,None]
    w=np.asarray(model['support_weights']) if weighted else np.ones(h.shape)
    assert h.shape==w.shape and h.shape[0]==n
    assert np.isfinite(h).all() and np.isfinite(w).all() and (w>=0).all()
    np.testing.assert_allclose(w.sum(1),1.,atol=1e-10)
    points=np.repeat((inp['xyz_world']*1000.)[:,None,:],h.shape[1],axis=1)
    state=c.get('state')
    if state is None:points[:,:,2]=h
    else:
        order=np.asarray(state['order']);support=np.asarray(state['support'],bool)
        take=np.flatnonzero(support);rows=order[take]
        delta=h[take]-np.asarray(state['local'])[take,2,None]
        points[rows]+=delta[:,:,None]*np.asarray(state['normal'])[None,None,:]
        # weights were in local ordering, as were posterior/levels.
        world_w=np.zeros_like(w);world_w[order]=w;w=world_w
        # Frozen-upstream unsupported observations remain singleton identity.
        un=order[~support];w[un]=0.;w[un,0]=1.
    return points,w,weighted

def score(job,model):
    c,x,y,base=unpack(job);inp=load(job['input']);truth=load(job['evaluation'])
    if 'json' in truth:truth.update(json.loads(truth.pop('json').tobytes().decode()))
    points,w,weighted=world_support(inp,c,model);n,k=w.shape
    ref=truth['gt_clean_xyz_world']*1000.;labels=truth['gt_layer']
    flat=points.reshape(-1,3);wf=w.ravel()/n
    if job['kind']=='bridge':
        ds=[]
        for z,x0,x1,y0,y1 in truth['surface_rectangles_mm']:
            ds.append(np.sqrt((flat[:,0]-np.clip(flat[:,0],x0,x1))**2+(flat[:,1]-np.clip(flat[:,1],y0,y1))**2+(flat[:,2]-z)**2))
        d=np.array(ds).T.reshape(n,k,-1)
    else:d=abs(points[:,:,2,None]-truth['true_means_mm'])
    out=dict(surface_mae_mm=float(np.sum(w*d.min(2))/n),
        balanced_source_mae_mm=float(np.mean([np.mean(np.sum(w[labels==i]*d[labels==i,:,i],axis=1)) for i in np.unique(labels)])),
        vertical_w1_mm=float(wasserstein_distance(flat[:,2],ref[:,2],u_weights=wf)),
        representation='weighted' if weighted else 'points',model_k=int(model['k']),n=n,
        risk_rms_mm=float(np.sqrt(np.sum(w*np.sum((points-ref[:,None,:])**2,axis=2))/n)))
    if not weighted:
        out['matched_rms_mm']=out['risk_rms_mm']
        out['coverage_1mm']=float(np.mean(cKDTree(flat).query(ref)[0]<=1+1e-9))
    else:
        # This is mass close to some reference sample, NOT reference coverage.
        out['weighted_reference_near_mass']=float(wf@(cKDTree(ref).query(flat)[0]<=1+1e-9))
    if job['gap']:
        mid=job['gap']/2.;mass=float(np.sum(w*(points[:,:,2]>mid))/n)
        gap=float(np.mean(np.sum(w[labels==1]*points[labels==1,:,2],axis=1))-np.mean(np.sum(w[labels==0]*points[labels==0,:,2],axis=1)))
        out.update(upper_mass=mass,upper_mass_error=abs(mass-float(labels.mean())),source_gap_error_mm=abs(gap-job['gap']),
                   cross_midplane_mass=float(np.mean([np.mean(np.sum(w[labels==i]*((points[labels==i,:,2]>mid)!=(i==1)),axis=1)) for i in (0,1)])))
        st=c.get('state');scale=1.
        if st is not None:
            basis=np.array(st['basis']);vec=basis[:,2]-basis[:,:2]@(np.array(model['slope'])/st['common_scale']);scale=max(abs(vec[2]),1e-12)
        if not model.get('identity'):
            out.update(fitted_gap_error_mm=abs(float(np.ptp(model['means']))/scale-job['gap']),dual_missed=int(model['k']==1))
    elif not model.get('identity'):out['single_false_split']=int(model['k']==2)
    if not model.get('identity'):
        prob=np.asarray(model['posterior'])[:,1] if model['k']==2 else np.zeros(n)
        if c.get('state') is not None:
            ordered=prob.copy();prob[np.asarray(c['state']['order'])]=ordered
        out['posterior_brier']=float(np.mean((prob-labels)**2))
    return out

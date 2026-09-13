"""Input-only 3D windows; shared windows for plane and spatial-mixture fits."""
from pathlib import Path
import sys,time
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'exploration_v11'))
from spatial_filter import solve

def corrected_world(state):
    q=state['world'].copy();order=state['order']
    q[order]+=(state['corrected']-state['local'][:,2])[:,None]*state['normal']/1000.
    mask=np.zeros(len(q),bool);mask[order]=state['support']
    return q,mask

def windows(q,k):
    n=len(q);k=min(k,n);count=max(1,int(np.ceil(2*n/k)))
    center=q.mean(0);first=int(np.argmin(np.sum((q-center)**2,axis=1)))
    anchors=[first];d=np.sum((q-q[first])**2,axis=1)
    for _ in range(1,count):
        a=int(np.argmax(d));anchors.append(a);d=np.minimum(d,np.sum((q-q[a])**2,axis=1))
    owners=cKDTree(q[anchors]).query(q)[1];tree=cKDTree(q)
    jobs=[]
    for j,a in enumerate(anchors):
        core=np.flatnonzero(owners==j)
        if not len(core):continue
        near=np.atleast_1d(tree.query(q[a],k=k)[1])
        # Include owned points so no core is extrapolated from another window.
        fit=np.unique(np.r_[near,core]);jobs.append((core,fit))
    return jobs

def estimate_frozen(state,sigma_mm,k=96,backend='spatial'):
    start=time.perf_counter();q,mask=corrected_world(state);mm=q*1000;result=mm.copy();records=[]
    for core,fit in windows(mm,k):
        mu=mm[fit].mean(0);a=mm[fit]-mu;_,vec=np.linalg.eigh(a.T@a/max(len(a),1))
        normal=vec[:,0];tangent=vec[:,1:];scale=max(float(np.sqrt(np.mean(np.sum((a@tangent)**2,axis=1)))),sigma_mm)
        x=np.c_[np.ones(len(fit)),a@tangent/scale];y=a@normal
        model=solve(x,y,sigma_mm,force_k=1 if backend=='plane' else None,iterations=12)
        ix=np.searchsorted(fit,core);assert np.array_equal(fit[ix],core)
        result[core]+=(model['prediction'][ix]-y[ix])[:,None]*normal
        records.append(dict(core=len(core),fit=len(fit),k=model['k'],radius_mm=float(np.max(np.linalg.norm(a,axis=1)))))
    out=state['world'].copy();out[mask]=result[mask]/1000
    assert np.isfinite(out).all();np.testing.assert_array_equal(out[~mask],state['world'][~mask])
    return out,dict(k_neighbors=k,backend=backend,windows=records,seconds=time.perf_counter()-start,
        supported_fraction=float(mask.mean()),truth_fields_used=[],scope='shared old frame correction; local PCA frame and disjoint core projection'),dict(support_mask=mask)

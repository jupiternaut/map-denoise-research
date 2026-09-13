"""Evaluator-owned procedural CAD-like surfaces; not ABC and not real scans."""
import numpy as np
from scipy.spatial import cKDTree

FAMILIES = ('parallel_plates', 'slotted_sheet', 'concentric_shells', 'rod_plate')

def rotation(seed):
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(3,3)))
    if np.linalg.det(q) < 0: q[:,0] *= -1
    return q

def sample(family, n, gap, balance, rng):
    lab = (rng.random(n) > balance).astype(int)
    side = 2*lab-1
    if family == 'parallel_plates':
        p = np.c_[rng.uniform(-.07,.07,(n,2)), side*gap/2]
        normal = np.tile([0.,0.,1.],(n,1))
    elif family == 'slotted_sheet':
        # Two finite, flat boards separated by a slot; each board has two faces.
        face = (rng.random(n) > .5).astype(int)
        p = np.c_[side*rng.uniform(gap/2,.065,n),rng.uniform(-.06,.06,n), (2*face-1)*.0025]
        lab = 2*lab+face
        normal = np.tile([0.,0.,1.],(n,1))
    elif family == 'concentric_shells':
        a = rng.uniform(0,2*np.pi,n)
        normal = np.c_[np.cos(a),np.sin(a),np.zeros(n)]
        p = normal*(.04+side*gap/2)[:,None]
        p[:,2] = rng.uniform(-.07,.07,n)
    elif family == 'rod_plate':
        lab = (rng.random(n) > .72).astype(int)
        p = np.c_[rng.uniform(-.065,.065,(n,2)),np.zeros(n)]
        normal = np.tile([0.,0.,1.],(n,1))
        take = lab == 1; a = rng.uniform(0,2*np.pi,take.sum())
        radius = gap/2
        p[take] = np.c_[rng.uniform(-.06,.06,take.sum()),.09+radius*np.cos(a),radius*np.sin(a)]
        normal[take] = np.c_[np.zeros(take.sum()),np.cos(a),np.sin(a)]
    else: raise ValueError(family)
    return p, normal, lab

def rect(p, center, hu, hv):
    closest = p.copy()
    closest[:,0] = np.clip(p[:,0], center[0]-hu, center[0]+hu)
    closest[:,1] = np.clip(p[:,1], center[1]-hv, center[1]+hv)
    closest[:,2] = center[2]
    return np.linalg.norm(p-closest,axis=1)

def distances(family,p,gap):
    if family == 'parallel_plates':
        return np.c_[rect(p,[0,0,-gap/2],.07,.07),rect(p,[0,0,gap/2],.07,.07)]
    if family == 'slotted_sheet':
        out=[]
        for side in (-1,1):
            for face in (-1,1):
                out.append(rect(p,[side*(.065+gap/2)/2,0,face*.0025],(.065-gap/2)/2,.06))
        return np.asarray(out).T
    if family == 'concentric_shells':
        r=np.linalg.norm(p[:,:2],axis=1); axial=np.maximum(np.abs(p[:,2])-.07,0)
        return np.c_[np.hypot(r-(.04-gap/2),axial),np.hypot(r-(.04+gap/2),axial)]
    if family == 'rod_plate':
        r=np.linalg.norm(p[:,1:]-[.09,0],axis=1); axial=np.maximum(np.abs(p[:,0])-.06,0)
        return np.c_[rect(p,[0,0,0],.065,.065),np.hypot(r-gap/2,axial)]
    raise ValueError(family)

def evaluate(family, output, gap, reference, ref_labels, n_input):
    if len(output)==0: return {'valid':False,'n_output':0}
    dd=distances(family,output,gap); ds=dd.min(axis=1); assigned=dd.argmin(axis=1)
    cov=cKDTree(output).query(reference,workers=1)[0]
    tol=min(.002,gap/4)
    out={'valid':True,'n_output':len(output),'retained_fraction':len(output)/n_input,
         'surface_mean_mm':float(ds.mean()*1000),'surface_p95_mm':float(np.quantile(ds,.95)*1000),
         'coverage_mean_mm':float(cov.mean()*1000),'combined_mm':float(500*(ds.mean()+cov.mean())),
         'tolerance_mm':tol*1000,'precision':float(np.mean(ds<=tol)),'recall':float(np.mean(cov<=tol)),
         'component_recall':{str(j):float(np.mean(cov[ref_labels==j]<=tol)) for j in np.unique(ref_labels)}}
    out['min_component_recall']=min(out['component_recall'].values())
    if family in ('parallel_plates','concentric_shells'):
        coord=output[:,2] if family=='parallel_plates' else np.linalg.norm(output[:,:2],axis=1)-.04
        values=[float(np.median(coord[assigned==j])) if np.sum(assigned==j)>=10 else None for j in (0,1)]
        out['estimated_gap_mm']=1000*abs(values[1]-values[0]) if None not in values else None
        out['gap_absolute_error_mm']=abs(out['estimated_gap_mm']-gap*1000) if out['estimated_gap_mm'] is not None else None
        out['gap_intrusion_fraction']=float(np.mean(np.abs(coord)<gap/4))
    if family=='slotted_sheet':
        out['slot_intrusion_fraction']=float(np.mean((np.abs(output[:,0])<gap/4)&(np.abs(output[:,1])<.05)))
        # Coverage of independent strips bordering the two slot edges.
        edge=(np.abs(reference[:,0])<gap/2+.003)
        out['slot_edge_recall']=float(np.mean(cov[edge]<=tol))
    if family=='rod_plate':
        rod=output[assigned==1]; radius=np.linalg.norm(rod[:,1:]-[.09,0],axis=1)
        out['rod_radius_mm']=float(np.median(radius)*1000) if len(radius) else None
        out['rod_radius_error_mm']=abs(out['rod_radius_mm']-gap*500) if len(radius) else None
        out['rod_recall']=out['component_recall']['1']
    return out

"""Local label coupling; retain lowest energy among eight synchronous sweeps."""
import numpy as np
from scipy.spatial import cKDTree


def edges_for_patches(uv, intensity, patches):
    edges=[]
    for patch in np.unique(patches):
        ids=np.flatnonzero(patches==patch)
        if patch<0 or len(ids)<2: continue
        _,near=cKDTree(uv[ids]).query(uv[ids],k=min(7,len(ids)),workers=1)
        candidate=ids[near]
        other=candidate!=ids[:,None]
        take=other & (np.cumsum(other,axis=1)<=6)
        a=np.broadcast_to(ids[:,None],candidate.shape)[take]; b=candidate[take]
        edges.append(np.sort(np.column_stack((a,b)),axis=1))
    if not edges:return np.empty((0,2),int),np.empty(0)
    edge=np.unique(np.concatenate(edges),axis=0)
    a,b=edge.T
    weight=np.exp(-np.sum((uv[a]-uv[b])**2,axis=1)/(2*8.**2)
                  -(intensity[a]-intensity[b])**2/(2*.1**2))
    return edge,weight


def couple(uv,intensity,patches,unary,supported,penalty=.04,iterations=8):
    unary=np.asarray(unary,float)
    edge,weight=edges_for_patches(np.asarray(uv),np.asarray(intensity),np.asarray(patches))
    labels=unary.argmin(axis=1); labels[~supported]=0
    row=np.arange(len(labels)); a,b=edge.T
    def energy(z):
        return float(unary[row,z].sum()+penalty*np.sum(weight*(z[a]!=z[b])))
    best=labels.copy(); best_energy=energy(labels); energies=[best_energy]
    initial=labels.copy()
    for _ in range(iterations):
        score=unary.copy()
        for k in range(unary.shape[1]):
            np.add.at(score[:,k],a,penalty*weight*(labels[b]!=k))
            np.add.at(score[:,k],b,penalty*weight*(labels[a]!=k))
        labels=score.argmin(axis=1); labels[~supported]=0
        value=energy(labels); energies.append(value)
        if value<best_energy:best=labels.copy();best_energy=value
    return best,dict(edges=len(edge),energies=energies,initial_energy=energies[0],
        final_energy=best_energy,changed=int(np.sum(best!=initial)),
        uv_scale_px=8.,gray_scale=.1,penalty=penalty,iterations=iterations,
        optimizer='best visited synchronous Potts states; no global certificate')

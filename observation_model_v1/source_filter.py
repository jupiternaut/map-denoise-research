"""Source-conditioned error correction; only current XYZ and station IDs enter."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

def reference_id(scan):
    ids,counts=np.unique(scan,return_counts=True)
    if len(ids)!=2:raise ValueError('This pilot requires exactly two stations')
    return ids[np.argmax(counts)]

def build(q,scan):
    ref=reference_id(scan);anchor=scan==ref;idx=np.flatnonzero(~anchor)
    a=q[anchor];p=q[idx];tree=cKDTree(a)
    dd=tree.query(a,k=2)[0][:,1];spacing=float(np.median(dd[dd>0]))
    d,nn=tree.query(p,k=min(16,len(a)))
    neighbors=a[nn];centers=neighbors.mean(axis=1);rel=neighbors-centers[:,None]
    eig,basis=np.linalg.eigh(np.einsum('nki,nkj->nij',rel,rel))
    normals=basis[:,:,0]
    support=(d[:,0]<=3*spacing)&(eig[:,0]/np.maximum(eig.sum(axis=1),1e-20)<=.05)
    center=np.mean(p,axis=0);radius=max(float(np.sqrt(np.mean(np.sum((p-center)**2,axis=1)))),1e-6)
    j=np.column_stack((np.cross((p-center)/radius,normals),normals))
    residual=np.sum((p-centers)*normals,axis=1)
    # Spatial checkerboard split, not a fresh station or independent sensor noise.
    _,b=np.linalg.eigh((p-center).T@(p-center));uv=(p-center)@b[:,[-1,-2]]
    fold=((uv[:,0]>np.median(uv[:,0]))^(uv[:,1]>np.median(uv[:,1]))).astype(int)
    return dict(reference=int(ref),idx=idx,anchor=anchor,p=p,centers=centers,normals=normals,
        support=support,center=center,radius=radius,j=j,residual=residual,fold=fold,spacing=spacing,plane_eigenvalues=eig)

def robust_solve(j,r):
    weights=np.ones(len(r));beta=np.zeros(6);rank=0
    for _ in range(8):
        w=np.sqrt(weights);u,s,vt=np.linalg.svd(j*w[:,None],full_matrices=False)
        take=s>.02*s[0];rank=int(take.sum())
        beta=vt[take].T@((u[:,take].T@(r*w))/s[take])
        e=r-j@beta;scale=max(1.4826*float(np.median(np.abs(e-np.median(e)))),1e-6)
        weights=np.minimum(1.,1.345*scale/np.maximum(np.abs(e),1e-20))
    return beta,rank

def transform(p,beta,state):
    rot=Rotation.from_rotvec(-beta[:3]/state['radius']).as_matrix()
    return (p-state['center'])@rot.T+state['center']-beta[3:]

def estimate(q,scan):
    s=build(q,scan);take=s['support'];idx=s['idx'];outputs={'identity':q.copy()}
    projected=q.copy();projected[idx[take]]-=s['residual'][take,None]*s['normals'][take]
    outputs['point_projection']=projected
    info=dict(reference_station=s['reference'],moving_points=len(idx),support=int(take.sum()),
        spacing_m=s['spacing'],rank=0,accepted=False,folds=[])
    rigid=q.copy()
    if take.sum()>=12:
        beta,rank=robust_solve(s['j'][take],s['residual'][take]);info['rank']=rank
        rigid[idx]=transform(q[idx],beta,s);info['beta']=beta.tolist()
        for fold in (0,1):
            train=take&(s['fold']==fold);test=take&(~(s['fold']==fold))
            if train.sum()<12 or test.sum()<12:
                info['folds'].append(dict(pass_check=False,reason='too_few_points'));continue
            b,r=robust_solve(s['j'][train],s['residual'][train])
            candidate=transform(q[idx[test]],b,s)
            before=float(np.mean(np.abs(s['residual'][test])))
            after=float(np.mean(np.abs(np.sum((candidate-s['centers'][test])*s['normals'][test],axis=1))))
            info['folds'].append(dict(pass_check=bool(after<before),before_m=before,after_m=after,rank=r))
        info['accepted']=all(f['pass_check'] for f in info['folds'])
    outputs['rigid_all']=rigid
    outputs['rigid_confirmed']=rigid.copy() if info['accepted'] else q.copy()
    import open3d as o3d
    source=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(q[idx]))
    target=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(q[s['anchor']]))
    reg=o3d.pipelines.registration.registration_icp(source,target,3*s['spacing'],np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30))
    out=q.copy();out[idx]=q[idx]@reg.transformation[:3,:3].T+reg.transformation[:3,3]
    outputs['open3d_icp']=out
    info['icp_fitness']=reg.fitness
    for name,out in outputs.items():
        assert np.isfinite(out).all();np.testing.assert_array_equal(out[s['anchor']],q[s['anchor']])
        if name in ('rigid_all','rigid_confirmed','open3d_icp'):
            # Rigid corrections preserve all station-internal distances (sample pairs).
            np.testing.assert_allclose(np.linalg.norm(out[idx]-np.roll(out[idx],1,axis=0),axis=1),
                np.linalg.norm(q[idx]-np.roll(q[idx],1,axis=0),axis=1),atol=1e-10,rtol=1e-10)
    return outputs,info

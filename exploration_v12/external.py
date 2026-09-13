"""Official PyMeshLab kernels, not a reimplementation of APSS/RIMLS."""
import sys,time
import numpy as np
sys.path.append('/srv/slam-research/grf/map-denoise/envs/spatial-v12-AXeJsv/lib/python3.12/site-packages')
import pymeshlab as pm
from local_filter import corrected_world

def project(q,kind='rimls',scale=2.):
    center=q.mean(0);x=(q-center)*1000;ms=pm.MeshSet();ms.add_mesh(pm.Mesh(vertex_matrix=x))
    ms.compute_normal_for_point_clouds(k=min(32,len(q)-1),smoothiter=0,flipflag=False)
    normals=ms.current_mesh().vertex_normal_matrix().copy()
    _,v=np.linalg.eigh(x.T@x);axis=v[:,0]
    normals[np.sum(normals*axis,axis=1)<0]*=-1
    ids=np.arange(len(q));tags=np.column_stack(((ids%256)/255.,((ids//256)%256)/255.,((ids//65536)%256)/255.,np.ones(len(q))))
    ms=pm.MeshSet();ms.add_mesh(pm.Mesh(vertex_matrix=x,v_normals_matrix=normals,v_color_matrix=tags))
    ms.compute_custom_radius_scalar_attribute_per_vertex(nbneighbors=16)
    # Distinct control/proxy meshes avoid a reproduced native auto-clone crash.
    ms.add_mesh(pm.Mesh(vertex_matrix=x,v_normals_matrix=normals,v_color_matrix=tags))
    kw=dict(controlmesh=0,proxymesh=1,filterscale=float(scale),maxprojectioniters=15,maxsubdivisions=0)
    if kind=='rimls':ms.compute_mls_projection_rimls(**kw,sigman=.75,maxrefittingiters=3)
    else:ms.compute_mls_projection_apss(**kw,sphericalparameter=1.,accuratenormal=True)
    out=ms.current_mesh().vertex_matrix().copy()/1000+center
    assert out.shape==q.shape and np.isfinite(out).all()
    # Color encoding checks that the official kernel preserves source row order.
    np.testing.assert_allclose(ms.current_mesh().vertex_color_matrix(),tags,atol=1e-7)
    return out

def estimate_frozen(state,kind='rimls',scale=2.,raw=False):
    start=time.perf_counter();q,mask=corrected_world(state)
    if raw:q=state['world'].copy()
    candidate=project(q,kind,scale);out=state['world'].copy();out[mask]=candidate[mask]
    return out,dict(implementation='PyMeshLab 2025.7.post1',kernel='compute_mls_projection_'+kind,
        scale=scale,raw=raw,normals_k=32,seconds=time.perf_counter()-start,supported_fraction=float(mask.mean()),
        scope='official kernel, input-only estimated oriented normals; shared support and optional frame correction'),dict(support_mask=mask)

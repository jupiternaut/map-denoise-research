"""Shared local fields and local label graph; inference never reads geometry GT."""
import common as c
import argparse,json,time,resource
import numpy as np
from pathlib import Path
from scipy.ndimage import map_coordinates
from v28_closeout.direct_evidence import _camera
from v28_closeout.graph_field import estimate_normals
from field_model import fit_local_fields
from field_evidence import candidate_offsets,score_positions,visibility_cost,load_cameras
from graph import couple

POOL_NAMES=('keep','old_A','old_B','half_A','half_B','minus6','minus3','plus3','plus6',
            'field_K1','field_K2a','field_K2b')


def choose(cost,offset,supported):
    objective=cost+.005*offset**2
    labels=objective.argmin(axis=1)
    labels[~supported]=0
    return labels,objective


def infer_case(p,a,b,ref,views):
    start=time.monotonic();times={}
    cam=_camera(ref);h=(p-cam.center)@cam.matrix.T
    uv=h[:,:2]/h[:,2,None];z=h[:,2]
    rays=p-cam.center;rays/=np.linalg.norm(rays,axis=1,keepdims=True)
    zrate=rays@cam.matrix[2]
    if np.any(z<=0) or np.any(zrate<=0):raise ValueError('points behind reference camera')
    normals=estimate_normals(p,24)
    offsets=candidate_offsets(p,a,b,ref['center'])
    ev=score_positions(p,offsets,normals,ref,views)
    times['pool_photometry']=time.monotonic()-start;tick=time.monotonic()
    depth=z[:,None]+offsets*zrate[:,None]
    valid=np.broadcast_to(ev['supported'][:,None],depth.shape)
    k1=fit_local_fields(uv,depth,ev['cost'],valid,input_depth=z,n_layers=1)
    k2=fit_local_fields(uv,depth,ev['cost'],valid,input_depth=z,n_layers=2)
    field_z=np.column_stack((k1.depth,k2.depth))
    field_offsets=np.clip((field_z-z[:,None])/zrate[:,None],-6.,6.)
    field_offsets[~ev['supported']]=0.
    new_offsets=np.column_stack((np.zeros(len(p)),field_offsets))
    times['fields']=time.monotonic()-tick;tick=time.monotonic()
    fresh=score_positions(p,new_offsets,normals,ref,views)
    times['field_photometry']=time.monotonic()-tick;tick=time.monotonic()
    row=np.arange(len(p)); selected={}
    selected['point_wta'],_=choose(ev['cost'],offsets,ev['supported'])
    s1=np.array([0,1]);s2=np.array([0,2,3])
    single,_=choose(fresh['cost'][:,s1],new_offsets[:,s1],fresh['supported'] & k1.supported)
    multi,_=choose(fresh['cost'][:,s2],new_offsets[:,s2],fresh['supported'] & k2.supported)
    outputs=dict(point_wta=ev['candidates'][row,selected['point_wta']],
        single_field=fresh['candidates'][row,s1[single]],
        multi_field=fresh['candidates'][row,s2[multi]])
    # The same four-candidate common source set is used for all field arms.
    vis=visibility_cost(fresh['scores'],fresh['candidates'],outputs['multi_field'],views)
    supported=fresh['supported'] & k2.supported
    visibility,unary=choose(vis['cost'][:,s2],new_offsets[:,s2],supported)
    outputs['multi_field_visibility']=fresh['candidates'][row,s2[visibility]]
    times['visibility']=time.monotonic()-tick;tick=time.monotonic()
    intensity=map_coordinates(cam.image,np.array([uv[:,1],uv[:,0]]),order=1,mode='nearest')
    graph,graphmeta=couple(uv,intensity,k2.patch_index,unary,supported)
    outputs['multi_field_graph']=fresh['candidates'][row,s2[graph]]
    times['graph']=time.monotonic()-tick
    pool=np.concatenate((ev['candidates'],fresh['candidates'][:,1:]),axis=1)
    arrays=dict(uv=uv,depth=z,normals=normals,pool_offsets=offsets,
        pool_scores=ev['scores'],pool_common_count=ev['common_count'],pool_cost=ev['cost'],
        field_offsets=new_offsets,field_scores=fresh['scores'],field_cost=fresh['cost'],
        field_common_count=fresh['common_count'],visibility_cost=vis['cost'],
        visibility_gap=vis['gap_mm'],visibility_known=vis['known'],visibility_plane_known=vis['plane_known'],
        visibility_self_front=vis['self_front'],visibility_weights=vis['normalized_weights'],
        k1_support=k1.supported,k2_support=k2.supported,k2_responsibility=k2.responsibility,
        patch_index=k2.patch_index,point_wta=selected['point_wta'],single_field=single,
        multi_field=multi,multi_field_visibility=visibility,multi_field_graph=graph)
    metadata=dict(seconds_by_stage=times,graph=graphmeta,rows=len(p),
        pool_two_view_rows=int(ev['supported'].sum()),field_two_view_rows=int(fresh['supported'].sum()),
        k1_supported=int(k1.supported.sum()),k2_supported=int(k2.supported.sum()),
        k1_patches=len(k1.patches),k2_patches=len(k2.patches),
        visibility_known_fraction=float(vis['known'].mean()),
        visibility_changes=int(np.sum(multi!=visibility)),graph_changes=int(np.sum(visibility!=graph)),
        reference_access=False,displacement_penalty=.005,max_new_field_ray_motion_mm=6,
        field_view_support='same KEEP/K1/K2a/K2b intersection',
        role='exposed old-scene replay; neither independent confirmation nor global topology search')
    return outputs,pool,arrays,metadata,dict(K1=k1.patches,K2=k2.patches)


def main():
    c.check_host();ap=argparse.ArgumentParser()
    ap.add_argument('--scene',required=True,type=int,choices=c.SCENES)
    ap.add_argument('--case');args=ap.parse_args()
    cases=c.cases_for_scene(args.scene)
    if args.case:
        cases=[x for x in cases if x.name==args.case]
        if len(cases)!=1:raise ValueError('exact smoke case missing')
    out=c.OUT/('smoke' if args.case else 'inference')/f'scan{args.scene}'
    out.mkdir(parents=True,exist_ok=False)
    viewsfile=c.RESERVED/'evidence'/f'scan{args.scene}'/'VIEWS.json'
    views=json.loads(viewsfile.read_text());cams,camfiles=load_cameras(args.scene,views,c)
    paths=[c.ROOT/x for x in ('infer.py','graph.py','common.py','field_model.py','field_evidence.py',
        'PROTOCOL.md','MODEL.md','test_graph.py','test_field_model.py','test_field_evidence.py')]
    paths += [viewsfile,c.CONT/'step_observation.py',c.CONT/'common.py',c.JOINT/'common.py',
        c.VIS_CODE/'visibility_features.py',c.CLOSEOUT/'scene_adapter.py',
        c.CLOSEOUT/'package/v28_closeout/direct_evidence.py',
        c.CLOSEOUT/'package/v28_closeout/graph_field.py',*camfiles]
    sources={str(x):c.sha(x) for x in paths}
    c.save_json(out/'LOCK.json',dict(source_sha256=sources,actual_arms=c.ACTUAL_ARMS,
        primary='multi_field',reference_access=False,prior_artifacts_read_only=True))
    records=[];start=time.monotonic()
    for case in cases:
        tick=time.monotonic();dest=out/case.name;dest.mkdir()
        geometry,route=c.geometry_and_route(case,sources)
        v=views[case.name.split('__')[0]]
        if set(v['reserved_views']) & set(v['original_views']):raise AssertionError('view split')
        outputs,pool,arrays,meta,fields=infer_case(*geometry.transpose(1,0,2),cams[v['reference']],
                                                [cams[x] for x in v['reserved_views']])
        outputs.update(identity=geometry[:,0],prior_recovery=c.selected_endpoint(geometry,route))
        for arm in c.ACTUAL_ARMS:c.write_points(dest/(arm+'.ply'),outputs[arm])
        c.save_npz(dest/'POOL.npz',points=pool,candidate_names=np.array(POOL_NAMES))
        c.save_npz(dest/'EVIDENCE.npz',**arrays)
        c.save_json(dest/'FIELDS.json',fields)
        meta.update(case=case.name,seconds=time.monotonic()-tick,candidate_names=list(POOL_NAMES),
                    moved_fraction={k:float(np.mean(np.linalg.norm(x-geometry[:,0],axis=1)>1e-7)) for k,x in outputs.items()})
        c.save_json(dest/'META.json',meta);records.append(meta)
        print('FIELD',case.name,round(meta['seconds'],2),'s',meta['k2_patches'],'patches',flush=True)
    if any(c.sha(p)!=h for p,h in sources.items()):raise AssertionError('source changed during inference')
    c.save_json(out/'SUMMARY.json',dict(records=records,seconds=time.monotonic()-start,
        peak_own_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,gpu=False))
    c.seal(out,sources);print('FIELD SEALED',args.scene,flush=True)

if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):main()

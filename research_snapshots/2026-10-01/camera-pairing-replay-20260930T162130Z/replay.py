"""Frozen candidates x camera-corrected photo witnesses; no GT reads."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import argparse, json, socket, time
import numpy as np
from PIL import Image
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
OLD=Path('/srv/slam-research/grf/map-denoise/runs/upstream-photo-holdout-20260930T113213Z')
sys.path.insert(0,str(OLD))
import adapter as a

def cameras(sid,names,group,version,mapping,log):
    out=[]
    for name in names:
        info=mapping['scenes'][str(sid)]['views'][name]
        path=log.bind(info['image_path'],group=group,view_id=name)
        with Image.open(path) as im: rgb=np.array(im.convert('RGB'),np.uint8)
        cam=a.half_resolution_camera(np.array(info['P_full_'+version]),rgb)
        assert np.allclose(cam['P'],info['P_half_'+version],atol=1e-10,rtol=0)
        assert np.allclose(cam['center'],info['center_mm'],atol=1e-8,rtol=0)
        out.append(cam)
    return out

def read_npz(path):
    with np.load(path,allow_pickle=False) as z: return {k:z[k].copy() for k in z.files}

def run(upstream,case_id):
    assert socket.gethostname()=='liekkas'
    start=time.monotonic(); a.install_input_guard(); log=a.ReadLog()
    for p in (Path(__file__),ROOT/'INITIALIZER_PROTOCOL.md',ROOT/'CAMERA_MAPPING.json',ROOT/'PLAN_CORRECTED.json'):
        log.bind(p)
    plan=json.loads((ROOT/'PLAN_CORRECTED.json').read_text())
    case=next(c for c in plan['cases'] if c['case_id']==case_id)
    sid=case['scene']; roi=case['roi']; condition=case['condition']; spec=plan['scenes'][str(sid)]
    mapping=json.loads((ROOT/'CAMERA_MAPPING.json').read_text()); groups=a.groups_from_plan(spec)
    dest=ROOT/'predictions'/upstream/case_id; dest.mkdir(parents=True,exist_ok=False)
    modules=a.frozen_modules(log); log.phase='construction'
    input_path=(OLD/'construction-dense' if upstream=='U0' else ROOT/'initialization')/f'scan{sid}'/roi/'input.npz'
    log.bind(input_path,group='construction_points'); initial=read_npz(input_path)
    if len(initial['query_ids'])<64 or len(initial['points_mm'])<64:
        a.write_json(dest/'INCOMPLETE.json',dict(case=case,upstream=upstream,
            valid_rows=len(initial['query_ids']),context_rows=len(initial['points_mm']),total_rows=128,
            reason='fewer_than64_qualified_queries_or_context',source_hashes=log.sources))
        return
    native,pixels,ids=a.input_arrays(input_path,log)
    old_ids=np.arange(len(ids)) if upstream=='U0' else initial['query_old_ids']
    if len(ids)<64:
        a.write_json(dest/'INCOMPLETE.json',dict(case=case,upstream=upstream,valid_rows=len(ids),total_rows=128))
        return
    version='old' if upstream=='U0' else 'corrected'
    ref=cameras(sid,groups['reference'],'reference',version,mapping,log)[0]
    points=a.perturb(native,ref['center'],condition)
    if upstream=='U0':
        old_dest=OLD/'inference'/f'scan{sid}'/case_id
        old_path=log.bind(old_dest/'candidates/CANDIDATES.npz',group='frozen_candidates')
        saved=read_npz(old_path)
        assert np.array_equal(ids,saved['query_ids'])
        assert np.max(np.abs(points[ids]-saved['geometry_mm'][:,0]))<1e-10
        with threadpool_limits(limits=1): normals=modules.graph.estimate_normals(points,24)
        value=dict(geometry=saved['geometry_mm'],x=saved['x'],scores=saved['scores'],routes=saved['restore_routes'],normals=normals)
    else:
        q=cameras(sid,groups['Q'],'Q',version,mapping,log)
        c=cameras(sid,groups['C'],'C',version,mapping,log)
        with threadpool_limits(limits=1): value=a.candidate_arrays(modules,points,ids,ref,q,c)
    candidate_dir=dest/'candidates'; candidate_dir.mkdir()
    a.write_npz(candidate_dir/'CANDIDATES.npz',geometry_mm=value['geometry'],x=value['x'],scores=value['scores'],
        restore_routes=value['routes'],query_ids=ids,query_old_ids=old_ids)
    a.write_npz(dest/'INPUT.npz',points_mm=points,reference_pixel_xy=pixels,query_ids=ids,query_old_ids=old_ids,normals=value['normals'])
    log.verify(); seal=a.seal(candidate_dir,log.sources); log.mark_sealed(seal)
    seal_digest=a.sha256(seal); chosen=value['geometry'][np.arange(len(ids)),value['routes']]
    counts={}; replay_checks={}
    for witness,version in [('W0','old'),('W1','corrected')]:
        folder=dest/witness; folder.mkdir()
        log.phase='witness_construct'
        wref=cameras(sid,groups['reference'],'reference',version,mapping,log)[0]
        wc=cameras(sid,groups['C'],'C',version,mapping,log)
        with threadpool_limits(limits=1): cs,cp=a.strict_pair_scores(modules.direct,points[ids],value['normals'][ids],chosen,wref,wc)
        cw=a.witness_accept(cs)
        log.phase='witness_heldout'; wh=cameras(sid,groups['H'],'H',version,mapping,log)
        with threadpool_limits(limits=1): hs,hp=a.strict_pair_scores(modules.direct,points[ids],value['normals'][ids],chosen,wref,wh)
        hw=a.witness_accept(hs)
        routes=dict(KEEP=np.zeros(len(ids),np.uint8),restore=value['routes'],
            construct_witness=a.veto_routes(value['routes'],cw['accepted']),heldout_witness=a.veto_routes(value['routes'],hw['accepted']))
        outputs={k:value['geometry'][np.arange(len(ids)),r] for k,r in routes.items()}
        a.write_npz(folder/'points_outputs.npz',**outputs)
        a.write_npz(folder/'routes.npz',**routes,query_ids=ids,query_old_ids=old_ids,original_routes=value['routes'],
            construct_accepted=cw['accepted'],heldout_accepted=hw['accepted'],
            construct_valid_views=cw['valid_views'],heldout_valid_views=hw['valid_views'])
        a.write_npz(folder/'witness.npz',construct_scores=cs,heldout_scores=hs,construct_shared_pixels=cp,heldout_shared_pixels=hp,
            construct_positive_views=cw['positive_views'],heldout_positive_views=hw['positive_views'],
            construct_mean_margin=cw['mean_margin'],heldout_mean_margin=hw['mean_margin'])
        if upstream=='U0' and witness=='W0':
            previous=read_npz(old_dest/'points_outputs.npz'); ps=read_npz(old_dest/'witness.npz')
            replay_checks={k:float(np.max(np.abs(outputs[k]-previous[k]))) for k in outputs}
            assert all(v==0 for v in replay_checks.values()),replay_checks
            assert np.array_equal(cs,ps['construct_scores'],equal_nan=True)
            assert np.array_equal(hs,ps['heldout_scores'],equal_nan=True)
        counts[witness]={k:int(np.count_nonzero(r)) for k,r in routes.items()}
    assert a.sha256(seal)==seal_digest
    log.phase='complete'; log.verify()
    a.write_json(dest/'READ_EVENTS.json',log.events)
    a.write_json(dest/'CASE.json',dict(case=case,upstream=upstream,valid_rows=len(ids),total_rows=128,context_rows=len(points),
        source_hashes=log.sources,candidate_sha256=seal_digest,U0W0_max_output_difference=replay_checks,
        labels_accessed=False,gt_accessed=False,candidate_sealed_before_H=True,modified=counts,seconds=time.monotonic()-start))
    a.seal(dest,log.sources)
    print(json.dumps(dict(case=case_id,upstream=upstream,rows=len(ids),moves=counts,seconds=time.monotonic()-start)),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--upstream',choices=['U0','U1'],required=True);ap.add_argument('--case',required=True)
    args=ap.parse_args();run(args.upstream,args.case)

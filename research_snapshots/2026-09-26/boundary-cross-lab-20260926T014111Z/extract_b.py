"""Score frozen B point pairs with the preceding, unchanged F/R evidence interface."""
from common import *
import argparse,time,resource
from threadpoolctl import threadpool_limits
from scene_adapter import load_scene,camera
from v28_closeout.graph_field import estimate_normals
from v28_closeout.direct_evidence import score_patches
from paired_features import summarize_pairs
from extract_evidence import scene_spec


def main():
    check_host();ap=argparse.ArgumentParser();ap.add_argument('--scene',type=int,required=True,choices=(24,37,55,65,69))
    sid=ap.parse_args().scene
    verify_seal(PREV/'evidence'/f'scan{sid}')
    out=ROOT/'evidence'/f'scan{sid}';out.mkdir(parents=True,exist_ok=False)
    source_paths=[ROOT/n for n in ('common.py','PROTOCOL.md','extract_b.py')]
    source_paths += [PREV/'extract_evidence.py',PREV/'paired_features.py',
        PREV/'evidence'/f'scan{sid}'/'SEALED.json',BASE/'data/train.npz',BASE/'data/TRAIN_MANIFEST.json',
        CLOSEOUT/'scene_adapter.py',CLOSEOUT/'package/v28_closeout/direct_evidence.py',
        CLOSEOUT/'package/v28_closeout/graph_field.py']
    sources={str(p):sha(p) for p in source_paths}
    save_json(out/'LOCK.json',dict(source_sha256=sources,scene=sid,reference_access=False,candidate='B_all'))
    scene=load_scene(**scene_spec(sid));save_json(out/'CALIBRATION.json',scene['audit'])
    prior_cal=json.loads((PREV/'evidence'/f'scan{sid}'/'CALIBRATION.json').read_text())
    if scene['audit']!=prior_cal:raise AssertionError('camera/input calibration changed')
    views=json.loads((PREV/'evidence'/f'scan{sid}'/'VIEWS.json').read_text())
    save_json(out/'VIEWS.json',views)
    train_ids={}
    if sid in (24,37):
        verify_seal(BASE/'data')
        manifest=json.loads((BASE/'data/TRAIN_MANIFEST.json').read_text())['records']
        with np.load(BASE/'data/train.npz',allow_pickle=False) as z:ci,ri=z['case_id'],z['row_id']
        train_ids={r['case']:ri[ci==r['case_id']] for r in manifest if r['scene']==sid}
    cams={}
    for v in views.values():
        if set(v['reserved_views'])&set(v['original_views']):raise AssertionError('view overlap')
        for name in v['original_views']+v['reserved_views']:
            if sha(scene['views'][name]['path'])!=v['image_sha256'][name]:raise AssertionError('image changed')
            if name not in cams:cams[name]=camera(scene['views'][name])
    records=[];start=time.monotonic()
    for path in cases_for_scene(sid):
        tick=time.monotonic();dest=out/path.name;dest.mkdir()
        rid=path.name.split('__')[0];v=views[rid]
        filenames=['identity.ply','B_all.ply','features.npz' if sid in (24,37) else 'FEATURES.npz',
                   'construction.json' if sid in (24,37) else 'CONSTRUCTION.json']
        hashes={f:sha(path/f) for f in filenames}
        frozen=json.loads((path.parent/'SEALED.json').read_text())['files']
        for f,h in hashes.items():
            if h!=frozen[path.name+'/'+f]:raise AssertionError('archived B input changed')
        construction=json.loads((path/filenames[-1]).read_text())
        if construction['views']!=v['original_views']:raise AssertionError('candidate view mismatch')
        p,b=read_points(path/'identity.ply'),read_points(path/'B_all.ply')
        if p.shape!=b.shape:raise AssertionError('B row mismatch')
        ids=train_ids[path.name] if sid in (24,37) else np.arange(len(p))
        normals=estimate_normals(p,24)
        ref=cams[v['original_views'][0]]
        ray=p[ids]-ref['center'];ray/=np.linalg.norm(ray,axis=1,keepdims=True)
        offset=np.sum((b[ids]-p[ids])*ray,axis=1)
        residual=float(np.max(np.abs(p[ids]+offset[:,None]*ray-b[ids])))
        if residual>1e-8:raise AssertionError('B ray coordinate mismatch')
        source_cams=[cams[n] for n in v['original_views'][1:]+v['reserved_views']]
        ev=score_patches(p[ids],normals[ids],np.column_stack([np.zeros(len(ids)),offset]),
                         ref,source_cams,mode='tangent',patch_radius=3,batch_size=256)
        if not np.allclose(ev['candidates'][:,1],b[ids],atol=1e-8,rtol=0):raise AssertionError('wrong scoredB')
        F,R=summarize_pairs(ev['scores'][:4]),summarize_pairs(ev['scores'][4:])
        with np.load(PREV/'evidence'/f'scan{sid}'/path.name/'PAIRED.npz',allow_pickle=False) as old:
            if not np.array_equal(old['row_ids'],ids):raise AssertionError('pair row support differs')
            if not np.array_equal(old['fit_scores'][:,:,0],ev['scores'][:4,:,0],equal_nan=True):
                raise AssertionError('F incumbent evidence differs from A')
            if not np.array_equal(old['reserved_scores'][:,:,0],ev['scores'][4:,:,0],equal_nan=True):
                raise AssertionError('R incumbent evidence differs from A')
        save_npz(dest/'PAIRED.npz',F=F,R=R,fit_scores=ev['scores'][:4],
                 reserved_scores=ev['scores'][4:],row_ids=ids,ref_valid=ev['ref_valid'])
        rec=dict(case=path.name,candidate='B_all',rows=len(ids),full_rows=len(p),source_path=str(path),
            source_sha256=hashes,views=v,max_ray_residual_mm=residual,incumbent_A_B_exact=True,
            fit_two_valid_fraction=float(np.mean(F[:,29]>=.5)),reserved_two_valid_fraction=float(np.mean(R[:,29]>=.5)),
            reference_access=False,wall_seconds=time.monotonic()-tick)
        save_json(dest/'META.json',rec);records.append(rec)
        print('B PAIRED',path.name,len(ids),round(rec['wall_seconds'],2),'s',flush=True)
    if any(sha(p)!=h for p,h in sources.items()):raise AssertionError('source changed during scoring')
    save_json(out/'SUMMARY.json',dict(records=records,wall_seconds=time.monotonic()-start,
        peak_own_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,gpu=False))
    seal(out,sources);print('B EVIDENCE SEALED',sid,flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()

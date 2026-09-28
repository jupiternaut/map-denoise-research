"""Direct final-point paired photometry; old and correction-reserved sources."""
from common import *
import argparse, time, resource
from threadpoolctl import threadpool_limits
from scene_adapter import load_scene, camera, project
from v28_closeout.graph_field import estimate_normals
from v28_closeout.direct_evidence import score_patches
from paired_features import summarize_pairs


def scene_spec(sid):
    if sid not in (24,37):
        return dict(root=DATA/'closeout-confirmation-v1/inputs'/f'scan{sid}')
    return dict(spec=dict(images=DATA/f'loss-alignment-v23/scan{sid}/image',
        colmap=DATA/f'loss-alignment-v23/scan{sid}/sparse/0',
        cameras=DATA/('real-closure-v21/cameras_geosvr_linked.npz' if sid==24 else 'reconstruction-v22-scan37/cameras.npz'),
        mesh=DATA/('published-outputs-v1/scan24_mesh.ply' if sid==24 else 'reconstruction-v22-scan37/scan37_mesh.ply')))


def choose_reserved(native, views, used):
    choices = []
    for name,v in views.items():
        if name in used: continue
        uv,z = project(native, v['P'])
        valid = (np.isfinite(uv).all(1)&(z>0)&(uv[:,0]>=8)&(uv[:,1]>=8)&
                 (uv[:,0]<v['width']-8)&(uv[:,1]<v['height']-8))
        if valid.sum()>=20: choices.append((int(valid.sum()),name))
    choices.sort(reverse=True)
    result = [name for _,name in choices[:4]]
    if len(result)!=4 or set(result)&set(used): raise RuntimeError('four disjoint sources unavailable')
    return result, choices


def main():
    check_host()
    ap=argparse.ArgumentParser();ap.add_argument('--scene',type=int,required=True,choices=(24,37,55,65,69))
    args=ap.parse_args();sid=args.scene
    out=ROOT/'evidence'/f'scan{sid}';out.mkdir(parents=True,exist_ok=False)
    paths=cases_for_scene(sid)
    if len(paths)!=(12 if sid in (24,37) else 20):raise AssertionError('incomplete case queue')
    source_paths=[ROOT/n for n in ('common.py','extract_evidence.py','paired_features.py','PROTOCOL.md')]
    source_paths += [PREV/'common.py',CLOSEOUT/'scene_adapter.py',CLOSEOUT/'package/v28_closeout/direct_evidence.py',
                     CLOSEOUT/'package/v28_closeout/graph_field.py']
    if sid in (24,37):
        verify_seal(PREV/'data')
        source_paths += [PREV/'data/train.npz',PREV/'data/TRAIN_MANIFEST.json',PREV/'data/SEALED.json']
    source_hashes={str(p):sha(p) for p in source_paths}
    save_json(out/'LOCK.json',dict(source_sha256=source_hashes,scene=sid,reference_access=False))
    scene=load_scene(**scene_spec(sid));save_json(out/'CALIBRATION.json',scene['audit'])
    train_ids={}
    if sid in (24,37):
        meta=json.loads((PREV/'data/TRAIN_MANIFEST.json').read_text())['records']
        with np.load(PREV/'data/train.npz',allow_pickle=False) as z:
            ci,ri=z['case_id'],z['row_id']
        train_ids={r['case']:ri[ci==r['case_id']] for r in meta if r['scene']==sid}
    camera_cache={};view_records={};records=[];start=time.monotonic()
    for native_path in [p for p in paths if p.name.endswith('__native')]:
        rid=native_path.name.split('__')[0]
        metadata_name='construction.json' if sid in (24,37) else 'CONSTRUCTION.json'
        frozen=json.loads((native_path.parent/'SEALED.json').read_text())['files']
        if sha(native_path/metadata_name)!=frozen[native_path.name+'/'+metadata_name]:
            raise AssertionError('archived construction metadata changed')
        meta=case_metadata(native_path);used=meta['views']
        if len(used)!=5 or len(set(used))!=5:raise AssertionError('original five views invalid')
        native=read_points(native_path/'identity.ply')
        reserved,rank=choose_reserved(native,scene['views'],used)
        images={name:sha(scene['views'][name]['path']) for name in used+reserved}
        if any(images[name]!=meta['image_hashes'][name] for name in used):raise AssertionError('original pixels changed')
        for name in used+reserved:
            if name not in camera_cache:camera_cache[name]=camera(scene['views'][name])
        record=dict(original_views=used,reserved_views=reserved,eligible_rank=rank,image_sha256=images,
                    reference=used[0],disjoint=True,scope='held-out from local A search only')
        view_records[rid]=record
    save_json(out/'VIEWS.json',view_records)
    for path in paths:
        tick=time.monotonic();rid=path.name.split('__')[0];v=view_records[rid]
        if case_metadata(path)['views']!=v['original_views']:raise AssertionError('condition changed views')
        dest=out/path.name;dest.mkdir()
        filenames=['identity.ply','A_all.ply','features.npz' if sid in (24,37) else 'FEATURES.npz',
                   'construction.json' if sid in (24,37) else 'CONSTRUCTION.json']
        hashes={f:sha(path/f) for f in filenames}
        # Verify the frozen input seal, not just a fresh hash.
        frozen=json.loads(((OLD/'real_results' if sid in (24,37) else path.parent)/'SEALED.json').read_text())['files']
        for f in filenames:
            if hashes[f]!=frozen[path.name+'/'+f]:raise AssertionError('frozen candidate changed')
        p,a=read_points(path/'identity.ply'),read_points(path/'A_all.ply')
        if p.shape!=a.shape:raise AssertionError('row mismatch')
        normals=estimate_normals(p,24)
        ids=train_ids[path.name] if sid in (24,37) else np.arange(len(p))
        ref=camera_cache[v['original_views'][0]]
        ray=p[ids]-ref['center'];ray/=np.linalg.norm(ray,axis=1,keepdims=True)
        offset=np.sum((a[ids]-p[ids])*ray,axis=1)
        residual=float(np.max(np.abs(p[ids]+offset[:,None]*ray-a[ids])))
        if residual>1e-8:raise AssertionError('A is not exact ray candidate')
        sources=[camera_cache[n] for n in v['original_views'][1:]+v['reserved_views']]
        evidence=score_patches(p[ids],normals[ids],np.column_stack([np.zeros(len(ids)),offset]),
                               ref,sources,mode='tangent',patch_radius=3,batch_size=256)
        if not np.allclose(evidence['candidates'][:,1],a[ids],rtol=0,atol=1e-8):raise AssertionError('scored wrong A')
        fit,res=evidence['scores'][:4],evidence['scores'][4:]
        F,R=summarize_pairs(fit),summarize_pairs(res)
        save_npz(dest/'PAIRED.npz',F=F,R=R,fit_scores=fit,reserved_scores=res,row_ids=ids,
                 ref_valid=evidence['ref_valid'])
        record=dict(case=path.name,rows=len(ids),full_rows=len(p),source_path=str(path),
            source_sha256=hashes,max_ray_residual_mm=residual,views=v,
            fit_two_valid_fraction=float(np.mean(F[:,29]>=.5)),
            reserved_two_valid_fraction=float(np.mean(R[:,29]>=.5)),
            wall_seconds=time.monotonic()-tick,reference_access=False)
        save_json(dest/'META.json',record);records.append(record)
        print('PAIRED',path.name,len(ids),'rows',round(record['wall_seconds'],2),'s',flush=True)
    if any(sha(p)!=h for p,h in source_hashes.items()):raise AssertionError('source changed')
    save_json(out/'SUMMARY.json',dict(records=records,wall_seconds=time.monotonic()-start,
              peak_own_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,gpu=False))
    seal(out,source_hashes)
    print('EVIDENCE SEALED',sid,flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()

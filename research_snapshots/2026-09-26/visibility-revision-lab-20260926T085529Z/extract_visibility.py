"""Fixed input-only visibility features from archived candidates and photo scores."""
from common import *
import argparse,resource,time
from threadpoolctl import threadpool_limits
from scene_adapter import colmap
from visibility_features import visibility_features,FEATURE_NAMES,GEOMETRY_WIDTH


def calibration_spec(sid):
    if sid not in (24,37):
        root=DATA/'closeout-confirmation-v1/inputs'/f'scan{sid}'
        return root/'cameras.npz',root/'sparse/0'
    return DATA/('real-closure-v21/cameras_geosvr_linked.npz' if sid==24 else 'reconstruction-v22-scan37/cameras.npz'),DATA/f'loss-alignment-v23/scan{sid}/sparse/0'


def calibrated_cameras(sid):
    """Load calibration only. Never open the native mesh/points for occlusion."""
    calibration,sparse=calibration_spec(sid)
    with np.load(calibration,allow_pickle=False) as z:inverse=np.linalg.inv(z['scale_mat_0'].astype(float))
    raw,_=colmap(sparse);result={}
    for name,view in raw.items():
        width,height=view['width']//2,view['height']//2
        sx,sy=width/view['width'],height/view['height']
        resize=np.array([[sx,0,(sx-1)/2],[0,sy,(sy-1)/2],[0,0,1.]])
        P=resize@view['P']@inverse
        result[name]=dict(P=P,center=-np.linalg.solve(P[:,:3],P[:,3]),width=width,height=height)
    return result,[calibration,*sorted(sparse.glob('*.bin'))]


def main():
    check_host();ap=argparse.ArgumentParser();ap.add_argument('--scene',type=int,required=True,choices=(24,37,55,65,69));args=ap.parse_args();sid=args.scene
    if not (ROOT/'PROTOCOL.md').is_file():raise RuntimeError('protocol required before extraction')
    out=OUT/'evidence'/f'scan{sid}';out.mkdir(parents=True,exist_ok=False)
    for source in (RESERVED,CROSS):verify_seal(source/'evidence'/f'scan{sid}')
    cameras,camera_files=calibrated_cameras(sid)
    views=json.loads((RESERVED/'evidence'/f'scan{sid}'/'VIEWS.json').read_text())
    assert views==json.loads((CROSS/'evidence'/f'scan{sid}'/'VIEWS.json').read_text())
    source_paths=[ROOT/n for n in ('common.py','PROTOCOL.md','visibility_features.py','extract_visibility.py','test_visibility.py','FEATURES.md')]
    source_paths += camera_files+[CLOSEOUT/'scene_adapter.py']+[source/'evidence'/f'scan{sid}'/'SEALED.json' for source in (RESERVED,CROSS)]
    sources={str(p):sha(p) for p in source_paths}
    save_json(out/'LOCK.json',dict(scene=sid,source_sha256=sources,geometry_width=GEOMETRY_WIDTH,feature_names=FEATURE_NAMES,
        camera_loading='COLMAP calibration and scale only; native mesh never loaded',raster='current case identity.ply only',reference_access=False))
    save_json(out/'VIEWS.json',views)
    start=time.monotonic();records=[]
    for path in cases_for_scene(sid):
        tick=time.monotonic();dest=out/path.name;dest.mkdir()
        files=[path/(n+'.ply') for n in ('identity','A_all','B_all')]
        frozen=json.loads((path.parent/'SEALED.json').read_text())['files']
        hashes={str(f):sha(f) for f in files}
        assert all(h==frozen[path.name+'/'+Path(f).name] for f,h in hashes.items())
        p,a,b=(read_points(f) for f in files);assert p.shape==a.shape==b.shape
        paired=[];scores=[];ids=None
        for source in (RESERVED,CROSS):
            file=source/'evidence'/f'scan{sid}'/path.name/'PAIRED.npz';paired.append(file)
            with np.load(file,allow_pickle=False) as z:
                if ids is None:ids=z['row_ids'].copy()
                else:assert np.array_equal(ids,z['row_ids'])
                scores.append(z['reserved_scores'].copy())
        view=views[path.name.split('__')[0]]
        assert not set(view['reserved_views'])&set(view['original_views'])
        selected=[cameras[n] for n in view['reserved_views']];ref=cameras[view['reference']]['center']
        features=visibility_features(p,np.stack([a[ids],b[ids]],axis=1),ids,selected,ref,np.stack(scores))
        save_npz(dest/'VISIBILITY.npz',row_ids=ids,features=features)
        record=dict(case=path.name,rows=len(ids),full_rows=len(p),shape=list(features.shape),reference_access=False,
            source_sha256=hashes,paired_source_sha256={str(f):sha(f) for f in paired},
            current_roi_input_only=True,empty_support_is_unknown=True,
            old_center_known_fraction=float(features[:,0,2:80:20].mean()),
            A_center_known_fraction=float(features[:,0,3:80:20].mean()),
            B_center_known_fraction=float(features[:,1,3:80:20].mean()),
            old_self_front_fraction=float(features[:,0,16:80:20].mean()),
            candidate_plane_fit_fraction=float(features[:,:,5:80:20].mean()),
            max_abs_mm=float(abs(features[:,:,[6+v*20 for v in range(4)]+[7+v*20 for v in range(4)]]).max()),
            wall_seconds=time.monotonic()-tick)
        save_json(dest/'META.json',record);records.append(record)
        print('VISIBILITY',path.name,len(ids),round(record['wall_seconds'],2),'s',flush=True)
    assert len(records)==(12 if sid in (24,37) else 20)
    assert all(sha(p)==h for p,h in sources.items())
    save_json(out/'SUMMARY.json',dict(records=records,rows=sum(r['rows'] for r in records),wall_seconds=time.monotonic()-start,
        peak_own_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,gpu=False))
    seal(out,sources);print('VISIBILITY SEALED',sid,flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()

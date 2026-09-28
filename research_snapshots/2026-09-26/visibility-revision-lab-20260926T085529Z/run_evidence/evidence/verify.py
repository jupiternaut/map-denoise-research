"""Read-only arrays/stage audit, then exclusive integrity record and evidence seal."""
from common import *
from visibility_features import FEATURE_NAMES, GEOMETRY_WIDTH

root=OUT/'evidence';scenes=[];rows=0;files=0;size=0;source_files={}
for sid in (24,37,55,65,69):
    folder=root/f'scan{sid}';manifest=verify_seal(folder)
    assert all(sha(p)==h for p,h in manifest['source'].items())
    source_files.update(manifest['source'])
    summary=json.loads((folder/'SUMMARY.json').read_text())
    scene_rows=0
    for record in summary['records']:
        f=folder/record['case']/'VISIBILITY.npz'
        with np.load(f,allow_pickle=False) as z:
            a=z['features'];ids=z['row_ids']
        assert a.shape==(record['rows'],2,96) and a.dtype==np.float32 and len(ids)==len(np.unique(ids))
        assert np.isfinite(a).all()
        raw=a[:,:,:80].reshape(len(a),2,4,20)
        flags=raw[:,:,:,[0,1,2,3,4,5,16,17]]
        assert np.isin(flags,[0,1]).all()
        assert ((raw[:,:,:,12:14]>=0)&(raw[:,:,:,12:14]<=1)).all()
        assert (raw[:,:,:,14:16]>=0).all()
        assert (np.diff(raw[:,:,:,18],axis=2)>=-1e-7).all()
        assert ((raw[:,:,:,18]>=0)&(raw[:,:,:,18]<=1.000001)).all()
        for k in (0,2,4,6,8,10,12,14,16,18):
            assert np.array_equal(raw[:,0,:,k],raw[:,1,:,k]),('incumbent candidate symmetry',k)
        for known,gap in ((2,6),(3,7),(4,8),(5,9)):
            assert not raw[:,:,:,gap][raw[:,:,:,known]==0].any()
        assert (abs(a[:,:,80:])<=2.000001).all()
        for block in (80,88):
            assert ((a[:,:,block+5:block+8]>=0)&(a[:,:,block+5:block+8]<=1.000001)).all()
        assert all(sha(p)==h for p,h in record['source_sha256'].items())
        assert all(sha(p)==h for p,h in record['paired_source_sha256'].items())
        scene_rows+=len(a);rows+=len(a);files+=1;size+=f.stat().st_size
    assert scene_rows==summary['rows']
    scenes.append(dict(scene=sid,rows=scene_rows,cases=len(summary['records']),wall_seconds=summary['wall_seconds'],peak_own_rss_mib=summary['peak_own_rss_mib']))
result=dict(status='PASS',scenes=scenes,rows=rows,cases=files,feature_width=len(FEATURE_NAMES),geometry_width=GEOMETRY_WIDTH,
    compressed_array_bytes=size,all_finite=True,incumbent_columns_identical_for_A_B=True,missing_gaps_zero_with_flags=True,
    source_hashes_unchanged=True,source_file_count=len(source_files),reference_access=False,
    scope='Current-case ROI-only zbuffer; self anchoring retained; not independent or certified visibility')
save_json(root/'VERIFICATION.json',result)
seal(root,source_files)
print(json.dumps(result,indent=2))

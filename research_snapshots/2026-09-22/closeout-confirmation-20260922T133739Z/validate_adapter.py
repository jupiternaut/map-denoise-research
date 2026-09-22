"""Replay old view selection and verify new input calibration without GT."""
import json
from pathlib import Path
from scene_adapter import load_scene,select_views,save
ROOT=Path(__file__).resolve().parent
BASE=Path('/srv/slam-research/grf/map-denoise/datasets')
OLD=Path('/home/grf/Documents/Codex/2026-09-22/v28-surface-experts-20260922T125505Z/real_results')
results=[]
for sid in (24,37):
    spec=dict(images=BASE/f'loss-alignment-v23/scan{sid}/image',colmap=BASE/f'loss-alignment-v23/scan{sid}/sparse/0',
        cameras=BASE/('real-closure-v21/cameras_geosvr_linked.npz' if sid==24 else 'reconstruction-v22-scan37/cameras.npz'),
        mesh=BASE/('published-outputs-v1/scan24_mesh.ply' if sid==24 else 'reconstruction-v22-scan37/scan37_mesh.ply'))
    scene=load_scene(spec=spec)
    for folder in sorted(OLD.glob(f'scan{sid}_*__native')):
        r=json.loads((folder/'roi.json').read_text());c=json.loads((folder/'construction.json').read_text())
        roi=dict(lo=r['aabb_min_mm'],hi=r['aabb_max_mm'],centroid=r['centroid_mm'],reference=r['ref_view'])
        selected=select_views(scene,roi)
        if selected!=c['views']:raise AssertionError((folder.name,selected,c['views']))
        results.append(dict(case=folder.name,views=selected,pass_exact=True))
    print('OLD_REPLAY',sid,scene['audit']['sparse_reprojection_p95_px'],flush=True)
new=load_scene(BASE/'closeout-confirmation-v1/inputs/scan40')
save(ROOT/'ADAPTER_VALIDATION.json',dict(old_view_replay=results,scan40_calibration=new['audit'],reference_access=False))
print('PASS',len(results),'old ROI lists; scan40 p95=',new['audit']['sparse_reprojection_p95_px'],flush=True)

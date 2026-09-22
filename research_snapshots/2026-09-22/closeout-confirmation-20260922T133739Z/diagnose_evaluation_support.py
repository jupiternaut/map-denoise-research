"""Post-evaluation diagnostic only; does NOT change any metric or selection."""
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from evaluate_closeout import points,observed,voxel
from scene_adapter import load_scene,project,in_box,save
ROOT=Path(__file__).resolve().parent
DATA=Path('/srv/slam-research/grf/map-denoise/datasets/closeout-confirmation-v1')
rows=[]
for sid in (55,65,69):
    scene=load_scene(DATA/'inputs'/f'scan{sid}')
    laser=points(DATA/'evaluation_only'/f'scan{sid}'/f'stl{sid:03d}_total.ply')
    obs=loadmat(DATA/'evaluation_only'/f'scan{sid}'/f'ObsMask{sid}_10.mat')
    for roi in json.loads((ROOT/'confirmation'/f'scan{sid}'/'ROIS.json').read_text()):
        ref=voxel(laser[in_box(laser,roi['lo'],roi['hi'])&observed(laser,obs)])
        uv,z=project(ref,scene['views'][roi['reference']]['P']);x0,y0,x1,y1=roi['box']
        inside=(z>0)&(uv[:,0]>=x0)&(uv[:,0]<x1)&(uv[:,1]>=y0)&(uv[:,1]<y1)
        rows.append(dict(scene=sid,roi=roi['id'],reference_points=len(ref),
                         reference_outside_photo_box_fraction=float(np.mean(~inside))))
save(ROOT/'SUPPORT_DIAGNOSTIC.json',dict(posthoc_diagnostic=True,metrics_unchanged=True,rows=rows,
    mean_outside_photo_box_fraction=float(np.mean([r['reference_outside_photo_box_fraction'] for r in rows]))))
print(json.dumps(rows,indent=2))

import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,importlib.util,json,hashlib,time
from pathlib import Path
import numpy as np
from scipy.optimize import check_grad
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent))
import common
import scalar_reference,plane_profile
RUN=Path('/srv/slam-research/grf/map-denoise/runs/t2-boundary-v1-20260911-1704/baseline-plane');RUN.mkdir(parents=True,exist_ok=False)
rows=[]
for family in ('dual','raycast','single'):
 for angle in (0.,3.,6.):
  cfg=common.Config(family=family,normal_degrees=angle,gap=6.,imbalance=.9,points_per_frame=160)
  inp,truth=common.generate(cfg,71331)
  for method,fn in [('scalar_profile_hard',scalar_reference.estimate),('scalar_profile_lbfgs',scalar_reference.estimate),('plane_profile_map',plane_profile.estimate)]:
   start=time.perf_counter();out,b,info=fn(inp,method);elapsed=time.perf_counter()-start
   filename=RUN/f'{family}-{angle:g}-{method}.npz';np.savez_compressed(filename,xyz_mm=out,bias_mm=b)
   saved=np.load(filename);metrics=common.evaluate(saved['xyz_mm'],saved['bias_mm'],truth,inp)
   row={'family':family,'angle':angle,'method':method,'seed':71331,'elapsed_s':elapsed,'metrics':metrics,'info':info};rows.append(row)
   print(json.dumps({'family':family,'angle':angle,'method':method,'mae':metrics['normal_mae_mm'],'rmse':metrics['point_rmse_mm'],'normal':info.get('normal'),'seconds':elapsed}),flush=True)
with (RUN/'results.json').open('x') as stream:json.dump({'source_sha256':hashlib.sha256((HERE/'plane_profile.py').read_bytes()).hexdigest(),'records':rows},stream,indent=2)

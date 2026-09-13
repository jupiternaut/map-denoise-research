import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent))
import common
import normal_then_profile
RUN=Path('/srv/slam-research/grf/map-denoise/runs/t2-boundary-v1-20260911-1704/baseline-staged');RUN.mkdir(parents=True,exist_ok=False)
rows=[]
for family in ('dual','raycast','single'):
 for angle in (0.,3.,6.):
  inp,truth=common.generate(common.Config(family=family,normal_degrees=angle,gap=6.,imbalance=.9,points_per_frame=160),71331)
  original=inp.xyz_mm.copy()
  for method in normal_then_profile.METHODS:
   start=time.perf_counter();out,b,info=normal_then_profile.estimate(inp,method);elapsed=time.perf_counter()-start
   np.testing.assert_array_equal(original,inp.xyz_mm)
   assert np.isfinite(out).all() and info['rotation_orthogonality_error']<1e-12
   path=RUN/f'{family}-{angle:g}-{method}.npz';np.savez_compressed(path,xyz_mm=out,bias_mm=b)
   saved=np.load(path);metrics=common.evaluate(saved['xyz_mm'],saved['bias_mm'],truth,inp)
   rows.append({'family':family,'angle':angle,'seed':71331,'method':method,'elapsed_s':elapsed,'info':info,'metrics':metrics})
   print(json.dumps({'family':family,'angle':angle,'method':method,'mae':metrics['normal_mae_mm'],'seconds':elapsed}),flush=True)
with (RUN/'results.json').open('x') as fp:json.dump({'source_sha256':hashlib.sha256((HERE/'normal_then_profile.py').read_bytes()).hexdigest(),'records':rows},fp,indent=2)

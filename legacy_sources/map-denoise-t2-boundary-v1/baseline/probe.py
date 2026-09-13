import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import hashlib,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
from scipy.optimize import check_grad

HERE=Path(__file__).resolve().parent
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
old=load(Path('/home/grf/Documents/Codex/2026-09-11/map-denoise-six-track-v1/tracks/t1_t2/experiment.py'),'old_scalar_probe')
new=load(HERE/'scalar_reference.py','strong_scalar_reference')
RUN=Path('/srv/slam-research/grf/map-denoise/runs/t2-boundary-v1-20260911-1704/baseline');RUN.mkdir(parents=True,exist_ok=False)
rows=[]
for case in old.CASES:
 for seed in (91113,91117):
  inp,truth=old.generate(case,seed)
  for method in ('frame_center_then_xyz','open3d_icp_then_xyz','joint_forced')+new.METHODS:
   start=time.perf_counter();out,b,info=(new.estimate if method in new.METHODS else old.estimate)(inp,method);elapsed=time.perf_counter()-start
   path=RUN/f'{case}-{seed}-{method}.npz';np.savez_compressed(path,xyz_mm=out,bias_mm=b)
   row={'case':case,'seed':seed,'method':method,'elapsed_s':elapsed,'metrics':old.evaluate(inp,truth,np.load(path)['xyz_mm'],np.load(path)['bias_mm']),'info':info};rows.append(row)
   print(json.dumps({'case':case,'method':method,'normal_mae_mm':row['metrics']['normal_mae_mm'],'seconds':elapsed}),flush=True)
with (RUN/'results.json').open('x') as fp:json.dump({'source_sha256':hashlib.sha256((HERE/'scalar_reference.py').read_bytes()).hexdigest(),'scope':'development feasibility on prior synthetic families','records':rows},fp,indent=2)

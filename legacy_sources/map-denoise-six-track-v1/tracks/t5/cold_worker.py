"""One fresh process per cold measurement, including import/load separately."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[key]='1'
import time
boot=time.perf_counter()
import json
from pathlib import Path
import resource
import sys
import numpy as np
from accelerated import FusedEM, make_denoiser

run=Path('/srv/slam-research/grf/map-denoise/runs/six-track-v1-20260911-1600/t5')
arm=sys.argv[1]
p=np.load(run/'thin_assembly-iid_1p5mm-1211-input.npy')
fit=None if arm=='reference' else FusedEM(run/'mixture_kernel.so',screen=arm=='fused_screen')
fn=make_denoiser(fit)
loaded=time.perf_counter()
out,diag=fn(p,{'fallback':'bilateral','em_iterations':32,'iterations':2})
ended=time.perf_counter()
print(json.dumps({'arm':arm,'points':len(p),'import_input_library_load_s':loaded-boot,
 'cold_denoise_s':ended-loaded,'in_process_total_s':ended-boot,
 'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
 'scope':'fresh Python process; OS filesystem cache not flushed; no GPU'}))

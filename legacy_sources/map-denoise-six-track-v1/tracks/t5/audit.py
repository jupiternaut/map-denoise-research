"""Run inherited unit properties against every implementation; cost scale probe."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[key]='1'
import importlib
import io
import json
from pathlib import Path
import sys
import unittest
import numpy as np
from accelerated import FusedEM, REFERENCE, make_denoiser, OLD
from run import RUN, measured, geometry_module, write_json, CONFIG

sys.path.insert(0,str(OLD.parents[2]))
testmodule=importlib.import_module('parallel_geometry_v5.association.test_operator')
results=[]
for arm,fit in [('reference',None),('fused',FusedEM(RUN/'mixture_kernel.so')),('fused_screen',FusedEM(RUN/'mixture_kernel.so',screen=True))]:
    testmodule.denoise=make_denoiser(fit); testmodule.fit_modes=fit or REFERENCE.fit_modes
    output=io.StringIO()
    result=unittest.TextTestRunner(stream=output,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(testmodule))
    results.append({'arm':arm,'tests_run':result.testsRun,'passed':result.wasSuccessful(),'text':output.getvalue()})
write_json(RUN/'inherited-unit-tests.json',results)
assert all(x['passed'] for x in results)

geo=geometry_module(); n=10000
clean,normal,_=geo.sample_surface('thin_assembly',n,np.random.default_rng(62021))
p=geo.corrupt(clean,normal,'iid_1p5mm',np.random.default_rng(62022))
np.save(RUN/'scale-10000-input.npy',p,allow_pickle=False)
scaling=[]; output={}
for arm,fit in [('reference',None),('fused',FusedEM(RUN/'mixture_kernel.so')),('fused_screen',FusedEM(RUN/'mixture_kernel.so',screen=True))]:
    fn=make_denoiser(fit); timings=[]
    for _ in range(3):
        out,timing=measured(fn,p); timings.append(timing)
    np.save(RUN/f'scale-10000-{arm}.npy',out,allow_pickle=False); output[arm]=out
    scaling.append({'arm':arm,'points':n,'timings':timings,
                    'max_output_delta_m':float(np.linalg.norm(out-output['reference'],axis=1).max())})
    print(json.dumps(scaling[-1]),flush=True)
write_json(RUN/'scale-results.json',scaling)
assert all(x['max_output_delta_m']<1e-9 for x in scaling)

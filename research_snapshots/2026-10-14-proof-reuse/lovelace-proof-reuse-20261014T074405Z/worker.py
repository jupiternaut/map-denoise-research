"""Cold, observation-only A15 invocation. No evaluation/truth imports."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time

ROOT = Path(__file__).resolve().parent
A14 = Path('/srv/slam-research/grf/map-denoise/runs/lovelace-joint-boundary-20261013T123336Z')
sys.path.append(str(A14))
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

def load_method(arm):
    if arm == 'original':
        spec=importlib.util.spec_from_file_location('a15_original', A14/'search_method.py')
        m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    elif arm == 'pure':
        import pure_search as m
    else:
        import reuse_search as m
    if hasattr(m, 'warm_load'):
        m.warm_load('joint') if arm in ('original','pure') else m.warm_load()
    return m

class Deadline(BaseException): pass
def alarm(*_): raise Deadline('outer cap exceeded')

def run(job):
    wall0,cpu0=time.perf_counter(),time.process_time()
    arm=job['arm']; m=load_method(arm); imported=time.perf_counter()-wall0
    budget=float(job['budget_seconds']); failure=None; state=None
    signal.signal(signal.SIGALRM,alarm); signal.setitimer(signal.ITIMER_REAL,budget+2.)
    start,cpu=time.perf_counter(),time.process_time()
    try:
        if arm in ('original','pure'):
            state=m.prepare(deepcopy(job['observation']),budget,'joint')
        else:
            state=m.prepare(deepcopy(job['observation']),budget,tolerance=.001,
                            hot=arm in ('hot','both'),inherit=arm in ('inherit','both'))
    except (Exception,Deadline) as exc:
        failure={'type':type(exc).__name__,'message':str(exc)}
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
    prepare=time.perf_counter()-start; prep_cpu=time.process_time()-cpu
    preds=[]
    for t in job['tasks']:
        selected=t['incumbent'] if failure or state is None else min(state['upper'],max(state['lower'],t['incumbent']))
        preds.append({**t,'selected':selected,'failure':failure})
    dependencies=sorted({str(Path(x.__file__).resolve()) for x in list(sys.modules.values())
                         if getattr(x,'__file__',None) and '/map-denoise/runs/' in x.__file__ and x.__file__.endswith('.py')})
    return {'schema':'a15-cold-v1','arm':arm,'budget_seconds':budget,
            'observation_id':job['observation_id'],'replicate':job.get('replicate',0),
            'state':state,'predictions':preds,'prepare_failure':failure,'dependencies':dependencies,
            'cost':{'import_seconds':imported,'prepare_seconds':prepare,'prepare_cpu_seconds':prep_cpu,
                    'worker_seconds':time.perf_counter()-wall0,'worker_cpu_seconds':time.process_time()-cpu0,
                    'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}}

if __name__=='__main__':
    if sys.argv[1]=='--deps':
        load_method(sys.argv[2])
        print(json.dumps(sorted({str(Path(x.__file__).resolve()) for x in list(sys.modules.values())
                               if getattr(x,'__file__',None) and '/map-denoise/runs/' in x.__file__ and x.__file__.endswith('.py')})))
    else:
        result=run(json.loads(Path(sys.argv[1]).read_text()))
        with Path(sys.argv[2]).open('x') as f: json.dump(result,f,allow_nan=False)

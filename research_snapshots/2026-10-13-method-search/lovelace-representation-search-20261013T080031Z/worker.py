"""One cold method invocation, without generator or evaluator imports."""
import argparse
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import resource
import signal
import sys
import time

ROOT=Path(__file__).resolve().parent

class SolveTimeout(BaseException):
    pass

def expired(signum,frame):
    raise SolveTimeout('prepare exceeded budget + 2 seconds')

def run(job):
    wall0=time.perf_counter(); cpu0=time.process_time()
    sys.path[:0]=[str(ROOT/'task_bundle'),str(ROOT/'library'),str(ROOT/'arms'/job['arm'])]
    path=ROOT/'arms'/job['arm']/'method.py'
    spec=importlib.util.spec_from_file_location('a12_selected_method',path)
    method=importlib.util.module_from_spec(spec); spec.loader.exec_module(method)
    import_seconds=time.perf_counter()-wall0
    budget=float(job['budget_seconds'])
    entry=method.prepare if not job.get('ablation') else method.prepare_ablation
    prepare0=time.perf_counter(); prepare_cpu0=time.process_time()
    state=None; failed=None
    signal.signal(signal.SIGALRM,expired)
    signal.setitimer(signal.ITIMER_REAL,budget+2.)
    try:
        state=entry(deepcopy(job['observation']),budget)
    except (Exception,SolveTimeout) as error:
        failed={'type':type(error).__name__,'message':str(error)}
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
    prepare_seconds=time.perf_counter()-prepare0
    prepare_cpu=time.process_time()-prepare_cpu0
    if prepare_seconds>budget+2.:
        failed={'type':'SolveTimeout','message':'measured prepare elapsed exceeded budget + 2 seconds'}
        state=None
    predictions=[]; decode0=time.perf_counter()
    for task in job['tasks']:
        incumbent=float(task['incumbent'])
        fault=failed
        try:
            selected=incumbent if fault else float(method.decode(deepcopy(state),incumbent))
            if not math.isfinite(selected) or not 540 <= selected <= 660:
                raise ValueError('decoded depth outside finite [540,660]')
        except Exception as error:
            selected=incumbent; fault={'type':type(error).__name__,'message':str(error)}
        predictions.append({'task_id':task['task_id'],'incumbent':incumbent,
                            'selected':selected,'failure':fault})
    decode_seconds=time.perf_counter()-decode0
    # Return serialized state as a persisted construction artifact, not only a score.
    return {'schema':'a12-prediction-v1','arm':job['arm'],'budget_seconds':budget,
            'observation_id':job['observation_id'],'ablation':job.get('ablation',False),
            'state':state,'predictions':predictions,'prepare_failure':failed,
            'cost':{'import_seconds':import_seconds,'prepare_seconds':prepare_seconds,
                    'prepare_cpu_seconds':prepare_cpu,'decode_seconds':decode_seconds,
                    'worker_seconds_before_serialize':time.perf_counter()-wall0,
                    'worker_cpu_seconds_before_serialize':time.process_time()-cpu0,
                    'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    'prepare_overrun_seconds':max(0,prepare_seconds-budget),
                    'overrun_gt_0_25':prepare_seconds-budget>0.25}}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('job'); ap.add_argument('output'); args=ap.parse_args()
    job=json.loads(Path(args.job).read_text())
    result=run(job)
    start=time.perf_counter(); payload=json.dumps(result,allow_nan=False)
    serialize_seconds=time.perf_counter()-start
    start=time.perf_counter()
    with Path(args.output).open('x') as f:
        f.write(payload)
    write_seconds=time.perf_counter()-start
    print(json.dumps({'arm':result['arm'],'budget':result['budget_seconds'],
                      'prepare_seconds':result['cost']['prepare_seconds'],
                      'failure':result['prepare_failure'],'serialize_seconds':serialize_seconds,
                      'write_seconds':write_seconds,'output_bytes':len(payload.encode())}),flush=True)

if __name__=='__main__':
    main()

"""One cold CPU invocation. No generator, labels or evaluation imported."""
from copy import deepcopy
import json
import math
from pathlib import Path
import resource
import signal
import sys
import time

class SolveTimeout(BaseException): pass
def expired(signum, frame): raise SolveTimeout('prepare exceeded cap + 2 seconds')

def run(job):
    wall0, cpu0 = time.perf_counter(), time.process_time()
    import search_method as method
    arm = job['arm']; method.warm_load(arm)
    import_seconds = time.perf_counter()-wall0
    start, cpu = time.perf_counter(), time.process_time()
    budget = float(job['budget_seconds']); state = None; failed = None
    signal.signal(signal.SIGALRM, expired); signal.setitimer(signal.ITIMER_REAL, budget+2.)
    try:
        state = method.prepare(deepcopy(job['observation']), budget, arm)
    except (Exception, SolveTimeout) as exc:
        failed = {'type': type(exc).__name__, 'message': str(exc)}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    elapsed = time.perf_counter()-start
    prepare_cpu = time.process_time()-cpu
    if elapsed > budget+2.:
        failed = {'type': 'SolveTimeout', 'message': 'measured deadline exceeded'}; state = None
    preds = []; start = time.perf_counter()
    for task in job['tasks']:
        value = float(task['incumbent']); fault = failed
        try:
            selected = value if fault else float(method.decode(state, value, arm))
            if not math.isfinite(selected) or not 540 <= selected <= 660:
                raise ValueError('invalid decoded depth')
        except Exception as exc:
            selected = value; fault = {'type':type(exc).__name__, 'message':str(exc)}
        preds.append({'task_id':task['task_id'], 'incumbent':value,
                      'selected':selected, 'failure':fault})
    return {'schema':'a13-result-v1', 'arm':arm, 'budget_seconds':budget,
            'observation_id':job['observation_id'], 'state':state,
            'predictions':preds, 'prepare_failure':failed,
            'cost':{'import_seconds':import_seconds, 'prepare_seconds':elapsed,
                    'prepare_cpu_seconds':prepare_cpu, 'decode_seconds':time.perf_counter()-start,
                    'worker_seconds_before_serialize':time.perf_counter()-wall0,
                    'worker_cpu_seconds_before_serialize':time.process_time()-cpu0,
                    'pre_serialization_max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    'overrun_gt_0_25':elapsed>budget+.25}}

if __name__ == '__main__':
    result=run(json.loads(Path(sys.argv[1]).read_text()))
    t=time.perf_counter(); payload=json.dumps(result, allow_nan=False)
    serial=time.perf_counter()-t; t=time.perf_counter()
    with Path(sys.argv[2]).open('x') as f: f.write(payload)
    print(json.dumps({'serialize_seconds':serial, 'write_seconds':time.perf_counter()-t,
                      'output_bytes':len(payload.encode())}), flush=True)

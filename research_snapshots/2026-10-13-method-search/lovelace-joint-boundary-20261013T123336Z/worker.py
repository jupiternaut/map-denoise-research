"""Reuse frozen cold-process runner; only search_method is the A14 adapter."""
import importlib.util
import json
from pathlib import Path
import sys
import time

SOURCE=Path('/srv/slam-research/grf/map-denoise/runs/lovelace-certified-boundary-20261013T093552Z/worker.py')
spec=importlib.util.spec_from_file_location('a14_worker_core',SOURCE)
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)

if __name__=='__main__':
    result=core.run(json.loads(Path(sys.argv[1]).read_text()))
    result['schema']='a14-result-v1'
    t=time.perf_counter();payload=json.dumps(result,allow_nan=False)
    serialization=time.perf_counter()-t;t=time.perf_counter()
    with Path(sys.argv[2]).open('x') as f:f.write(payload)
    print(json.dumps({'serialize_seconds':serialization,'write_seconds':time.perf_counter()-t,
                      'output_bytes':len(payload.encode())}),flush=True)

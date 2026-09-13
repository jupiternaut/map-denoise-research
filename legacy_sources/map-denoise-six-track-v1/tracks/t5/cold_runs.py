import json
import os
from pathlib import Path
import subprocess
import sys
import time

here=Path(__file__).resolve().parent
run=Path('/srv/slam-research/grf/map-denoise/runs/six-track-v1-20260911-1600/t5')
records=[]
for repeat in range(3):
    order=['reference','fused','fused_screen']
    if repeat%2: order.reverse()
    for arm in order:
        start=time.perf_counter()
        proc=subprocess.run([sys.executable,'-B',str(here/'cold_worker.py'),arm],capture_output=True,text=True,check=True,
                            env=os.environ | {'PYTHONDONTWRITEBYTECODE':'1'})
        elapsed=time.perf_counter()-start
        row=json.loads(proc.stdout); row.update({'repeat':repeat,'whole_subprocess_wall_s':elapsed})
        records.append(row); print(json.dumps(row),flush=True)
with (run/'cold-process-results.json').open('x') as stream: json.dump(records,stream,indent=2)

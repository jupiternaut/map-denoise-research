"""Small development only; operator outputs serialized before evaluation."""
import sys,json,tempfile,time
from pathlib import Path
import numpy as np
sys.path.insert(0,'/home/grf/Documents/Codex/2026-09-11/map-denoise-six-track-v1/tracks/t1_t2')
from experiment import generate,evaluate,estimate as old_estimate,CASES
from joint_framewise import METHODS,estimate

def main():
    root=Path('/srv/slam-research/grf/map-denoise/runs/t2-boundary-v1-20260911-1704/candidate');root.mkdir(parents=True,exist_ok=True)
    run=Path(tempfile.mkdtemp(prefix='development-',dir=root));(run/'outputs').mkdir();records=[]
    print(str(run),flush=True)
    for case in CASES:
      for seed in (311,313):
        inp,truth=generate(case,seed)
        for method in ('joint_forced','joint_guarded')+METHODS:
            st=time.perf_counter();out,b,info=(old_estimate(inp,method) if method.startswith('joint_') else estimate(inp,method));elapsed=time.perf_counter()-st
            file=run/'outputs'/f'{case}-{seed}-{method}.npz';np.savez_compressed(file,xyz_mm=out,bias_mm=b)
            with np.load(file) as d:metrics=evaluate(inp,truth,d['xyz_mm'],d['bias_mm'])
            row={'case':case,'seed':seed,'method':method,'elapsed_s':elapsed,'status':info['status'],'metrics':metrics,'info':info}
            records.append(row)
            with open(run/'records.jsonl','a') as fp:fp.write(json.dumps(row)+'\n')
        print(case,seed,'done',flush=True)
    (run/'results.json').write_text(json.dumps({'scope':'development only; old analytic family new seeds','records':records},indent=2))
    print('COMPLETED',run,flush=True)
if __name__=='__main__':main()

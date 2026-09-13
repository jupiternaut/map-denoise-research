import argparse
import json
from pathlib import Path
import time
import hashlib
import shutil
import numpy as np
from split_consensus import METHODS,estimate
from test_split_consensus import old

def run(destination):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False);rows=[]
    for case in old.CASES:
        for seed in (0,1):
            inp,truth=old.generate(case,seed)
            for method in METHODS:
                t=time.perf_counter();out,b,info=estimate(inp,method);elapsed=time.perf_counter()-t
                fn=destination/f'{case}_s{seed}__{method}.npz';np.savez_compressed(fn,xyz_mm=out,bias_mm=b)
                fn.with_suffix('.json').write_text(json.dumps(info,indent=2))
                saved=np.load(fn);row={'case':case,'seed':seed,'method':method,'elapsed_s':elapsed,**old.evaluate(inp,truth,saved['xyz_mm'],saved['bias_mm'])};rows.append(row)
    (destination/'results.json').write_text(json.dumps(rows,indent=2))
    src=Path(__file__).with_name('split_consensus.py');shutil.copy2(src,destination/src.name)
    (destination/'source_sha256.txt').write_text(hashlib.sha256(src.read_bytes()).hexdigest()+'\n')
    for case in old.CASES:
        print(case,{m:round(float(np.mean([r['normal_mae_mm'] for r in rows if r['case']==case and r['method']==m])),5) for m in METHODS})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();run(a.output)

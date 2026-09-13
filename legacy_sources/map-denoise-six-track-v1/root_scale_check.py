"""Add the newly identified simple scale control to all six replication cases.
This is post-exposure attribution, not a second independent holdout.
"""
import json
import time
import numpy as np
from root_geometry_check import DEST, CASES, sample, evaluate, old_a

if __name__ == '__main__':
    folder=DEST/'scale-control';folder.mkdir(exist_ok=False)
    rows=[]
    for family,gap,sigma,balance in CASES:
        for seed in [82019,82021]:
            key=f'{family}-{seed}';data=np.load(DEST/f'{key}-input.npz')
            p,rot=data['points'],data['rotation']
            ref,_,labels=sample(family,20000,gap,.5,np.random.default_rng(seed+991))
            start=time.perf_counter();out,_=old_a(p,{'fallback':'bilateral','k':96,'em_iterations':16})
            elapsed=time.perf_counter()-start
            path=folder/f'{key}-old_A_k96.npy';np.save(path,out)
            rows.append({'family':family,'seed':seed,'arm':'old_A_k96_em16','seconds':elapsed,
                         'metrics':evaluate(family,np.load(path)@rot,gap,ref,labels,len(p))})
    (folder/'results.json').write_text(json.dumps({'scope':__doc__,'records':rows},indent=2))
    print(json.dumps([{'family':r['family'],'seed':r['seed'],'combined_mm':r['metrics']['combined_mm']} for r in rows]))

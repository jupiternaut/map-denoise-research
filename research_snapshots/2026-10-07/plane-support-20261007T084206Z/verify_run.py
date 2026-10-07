"""Read-only reproduction of frozen decisions, coordinates and all aggregate metrics."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='1'
import sys
sys.dont_write_bytecode=True
import json
from collections import Counter
import numpy as np
from experiment import (ROOT, OLD, EVAL, PLAN, INPUTS, check, load, verify,
                        sanitize, read_neighbors, make_intervals, readcsv, clean)
from selector import decisions


def main():
    check()
    for seal in ('RUN_LOCK.json','PREDICTIONS_SEALED.json','evaluation/SEALED.json'):
        verify(ROOT/seal)
    protocol=load(ROOT/'PROTOCOL.json')
    frozen={(r['roi'],r['query']):r for r in load(ROOT/'DECISIONS.json')['rows']}
    rows=sanitize(load(INPUTS[0]))
    neighbors,_=read_neighbors()
    plan=load(PLAN)
    for row in rows:
        iv=make_intervals(row,neighbors[(row['roi'],row['query'])],plan,protocol)
        assert decisions(row['objects'],iv,protocol,row['baseline_id'])==frozen[(row['roi'],row['query'])]['decisions']
    # Only now load sealed evaluation artifacts.
    points=readcsv(ROOT/'evaluation/POINT_METRICS.csv')
    base={(p['roi'],int(p['query'])):p for p in readcsv(EVAL/'POINT_METRICS.csv') if p['arm']=='incumbent'}
    candidates={(p['roi'],int(p['query']),int(p['candidate_id'])):p for p in readcsv(EVAL/'CANDIDATE_DISTANCES.csv')}
    ic=cc=0
    for key,r in frozen.items():
        for o in r['objects']:
            q=base[key] if o['candidate_id']==-1 else candidates[(*key,o['candidate_id'])]
            np.testing.assert_array_equal(o['xyz_mm'],[float(q[k]) for k in ('x_mm','y_mm','z_mm')])
            ic+=o['candidate_id']==-1
            cc+=o['candidate_id']!=-1
    metrics=load(ROOT/'evaluation/RESULTS.json')['metrics']
    recomputed=[]
    for m in metrics:
        arm=m['arm']
        pp=[p for p in points if p['arm']==arm]
        assert len(pp)==512 and len({(p['roi'],p['query']) for p in pp})==512
        primary=[p for p in pp if p['primary']=='True']
        assert len(primary)==484
        pair=np.array([[float(p['incumbent_distance_mm']),float(p['distance_mm'])] for p in primary])
        rois=sorted({p['roi'] for p in primary})
        ds={roi:np.array([float(p['distance_mm']) for p in primary if p['roi']==roi]) for roi in rois}
        rm={r:float(np.mean(v*v)) for r,v in ds.items()}
        r=dict(mse_mm2=float(np.mean(list(rm.values()))),mae_mm=float(np.mean([np.mean(v) for v in ds.values()])),
               improved=int(np.sum(pair[:,1]<pair[:,0]-1e-9)),worsened=int(np.sum(pair[:,1]>pair[:,0]+1e-9)),
               unchanged=int(np.sum(abs(pair[:,1]-pair[:,0])<=1e-9)),severe_gt5=int(np.sum(pair[:,1]>5)),
               good_le1=int(np.sum(pair[:,1]<=1)),new_1mm_harm_vs_incumbent=int(np.sum((pair[:,0]<=1)&(pair[:,1]>1))),
               finite=sum(bool(p['distance_mm']) for p in pp),
               original_missing_finite=sum(p['primary']=='False' and bool(p['distance_mm']) for p in pp))
        for k,v in r.items():
            assert abs(v-m[k])<1e-10,(arm,k,v,m[k])
        for roi,v in rm.items():
            assert abs(v-m['roi_mse'][roi])<1e-10
        for sid in ('118','122'):
            assert abs(np.mean([v for roi,v in rm.items() if roi.startswith('scan'+sid+'_')])-m['scene_mse'][sid])<1e-10
        recomputed.append(dict(arm=arm,**r))
    print(json.dumps(dict(decisions_reproduced=len(rows),candidate_coordinates=cc,incumbent_coordinates=ic,
                         metric_arms=len(metrics),status='PASS',recomputed=recomputed),indent=2))


if __name__=='__main__':
    main()

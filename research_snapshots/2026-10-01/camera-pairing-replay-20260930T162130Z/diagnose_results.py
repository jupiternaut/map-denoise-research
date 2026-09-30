"""Post-hoc error-tail accounting; cannot select or change predictions."""
from pathlib import Path
import json,sys
sys.dont_write_bytecode=True
import numpy as np
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from evaluate_replay import prior,read,dump
assert (ROOT/'evaluation/RESULTS.json').exists()
plan=json.loads((ROOT/'PLAN_CORRECTED.json').read_text());records=[]
for sid,meta in prior.reference_metadata().items():
    tree=cKDTree(prior.read_ply_xyz(meta['path'],np))
    for case in plan['cases']:
        if case['scene']!=sid or case['condition']!='native':continue
        valid=read(ROOT/'predictions/U1'/case['case_id']/'candidates/CANDIDATES.npz')['query_old_ids']
        for u in ('U0','U1'):
            c=read(ROOT/'predictions'/u/case['case_id']/'candidates/CANDIDATES.npz')
            g=c['geometry_mm'][valid] if u=='U0' else c['geometry_mm']
            d=tree.query(g.reshape(-1,3),workers=1)[0].reshape(len(g),3)
            nearest=d[:,0];tail=np.sort(nearest**2)[-max(1,int(np.ceil(.05*len(g)))):]
            records.append(dict(upstream=u,roi=case['roi'],n=len(g),
                median_mm=float(np.median(nearest)),p90_mm=float(np.quantile(nearest,.9)),p95_mm=float(np.quantile(nearest,.95)),
                precision1mm=float(np.mean(nearest<=1)),over5mm=int((nearest>5).sum()),over10mm=int((nearest>10).sum()),
                mse_mm2=float(np.mean(nearest**2)),top5percent_squared_error_fraction=float(tail.sum()/(nearest**2).sum()),
                all_candidates_over5mm=int((d.min(1)>5).sum())))
dump(ROOT/'evaluation/POSTHOC_TAIL_DIAGNOSIS.json',dict(status='posthoc descriptive, no tuning',records=records))
for r in records:print(json.dumps(r))

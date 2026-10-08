"""Frozen full9 evidence and P; historical definitions are read-only."""
import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parent
OLD=Path('/srv/slam-research/grf/map-denoise/runs/footprint-support-20261008T025757Z')
SUPPORT=Path('/srv/slam-research/grf/map-denoise/runs/surface-owned-support-20261008T022918Z/support.py')
spec=importlib.util.spec_from_file_location('_readonly_surface_support',SUPPORT)
support=importlib.util.module_from_spec(spec)
sys.dont_write_bytecode=True
spec.loader.exec_module(support)

def cameras(records):
    return [{k:np.asarray(v,float) for k,v in c.items()} for c in records]

def full9_intervals(images,cams,grid):
    rs=[support.score_support(images[0],images[j],cams[0],cams[j],
        [64.,64.],grid,'plane','full9') for j in (1,2)]
    scores=np.stack([r['scores'] for r in rs])
    accepted=np.all(np.isfinite(scores)&(scores>=.6),axis=0)
    return dict(intervals=support.support_intervals(grid,accepted,padding=1.),
        scores=scores,accepted=accepted)

def choose(intervals,candidates,incumbent):
    mass=sum(b-a for a,b in intervals)
    if mass<=0:
        return dict(selected_depth=float(incumbent),support_mean=None,estimated_squared_gain=0.)
    target=sum((b-a)*(a+b)/2 for a,b in intervals)/mass
    candidates=np.asarray(candidates,float)
    new=float(candidates[int(np.argmin((candidates-target)**2))])
    gain=(incumbent-target)**2-(new-target)**2
    if gain<=0:new=float(incumbent)
    return dict(selected_depth=new,support_mean=float(target),estimated_squared_gain=max(0.,float(gain)))

def replay_legacy(output_dir):
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    inp=json.loads((OLD/'OBS_INPUTS.json').read_text())
    oldrows={r['id']:r for r in json.loads((OLD/'evaluation/ROWS.json').read_text()) if r['arm']=='full9'}
    rows=[];maxdelta=0.;hashes={}
    for row in inp:
        p=Path(row['image_file'])
        hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        assert hashes[str(p)]==row['sha256']
        with np.load(p) as f:images=f['images']
        z=np.arange(300.,1001.,2.)
        r=full9_intervals(images,cameras(row['cameras']),z)
        old=oldrows[row['id']]
        # Some old JSON locations were initially named score; resolve the sealed
        # archive by basename if that pre-rename path is not present.
        curve=Path(old['curve_file'])
        if not curve.exists():curve=OLD/'curves/observed'/curve.name
        with np.load(curve) as f:
            assert np.array_equal(np.isnan(r['scores']),np.isnan(f['scores']))
            finite=np.isfinite(r['scores'])
            delta=float(np.max(np.abs(r['scores'][finite]-f['scores'][finite]))) if finite.any() else 0.
            maxdelta=max(maxdelta,delta)
            assert delta<1e-12
            assert np.array_equal(r['accepted'],f['accepted'])
        assert r['intervals']==old['intervals']
        d=choose(r['intervals'],[450.,540.,600.,660.,900.],900.)
        assert d['selected_depth']==old['selected_depth']
        err=abs(d['selected_depth']-old['true_depth'])
        assert err==old['absolute_error']
        rows.append(dict(id=row['id'],**d,absolute_error=err,curve_max_difference=delta))
    result=dict(n=len(rows),mae=float(np.mean([r['absolute_error'] for r in rows])),
        max_score_difference=maxdelta,rows=rows,input_hashes=hashes,
        support_sha256=hashlib.sha256(SUPPORT.read_bytes()).hexdigest())
    with (out/'RESULTS.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    return result

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',required=True)
    result=replay_legacy(p.parse_args().out)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','input_hashes')}))

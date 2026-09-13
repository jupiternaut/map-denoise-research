"""Recompute saved scores and verify fixed-state and V4-replay contracts."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from schema import read_evaluation
from paths import RUNS


def run():
    parser=argparse.ArgumentParser();parser.add_argument('run_dir',type=Path);parser.add_argument('--tests',action='store_true')
    args=parser.parse_args();dest=args.run_dir.resolve();started=time.perf_counter()
    if (dest/'VERIFICATION.json').exists():raise FileExistsError('verification already exists')
    rows=list(csv.DictReader((dest/'RESULTS.csv').open()))
    assert len(rows)==len({(r['case'],r['method']) for r in rows})
    infos={};differences=[];old_checked=0;frozen_checked=0
    old_run=RUNS/'exploration-v4-dev-e0c_n4hk'
    for r in rows:
        assert r['ok']=='True',r
        path=Path(r['output'])
        assert hashlib.sha256(path.read_bytes()).hexdigest()==r['output_sha256']
        with np.load(path) as a:out=a['xyz_world'];canonical=a['xyz_world_canonical']*1000
        with np.load(dest/'inputs'/(r['case']+'.npz')) as a:inp=a['xyz_world']
        assert out.shape==inp.shape and np.isfinite(out).all()
        info=json.loads(path.with_suffix('.json').read_text())['info'];infos[r['case'],r['method']]=info
        if r['method'] in ('pool_compatible','shared_group_slope','node_intercepts'):
            unsupported=np.array(info['unsupported_point_indices'],dtype=int)
            np.testing.assert_array_equal(out[unsupported],inp[unsupported])
        if r.get('geometry_scored')=='True':
            ev=read_evaluation(Path(r['source']));d=np.full(len(out),np.inf)
            for z,x0,x1,y0,y1 in ev['surface_rectangles_mm']:
                dx=np.maximum.reduce([x0-canonical[:,0],canonical[:,0]-x1,np.zeros(len(out))])
                dy=np.maximum.reduce([y0-canonical[:,1],canonical[:,1]-y1,np.zeros(len(out))])
                d=np.minimum(d,np.sqrt(dx*dx+dy*dy+(canonical[:,2]-z)**2))
            err=max(abs(float(r[k])-v) for k,v in {
                'surface_accuracy_mean_mm':np.mean(d),'surface_accuracy_rms_mm':np.sqrt(np.mean(d*d)),
                'surface_accuracy_p95_mm':np.quantile(d,.95)}.items())
            assert err<1e-8;differences.append(float(err))
        old=old_run/'outputs'/(r['case']+'__'+r['method']+'.npz')
        if r['stage']!='confirmation' and old.exists() and r['method'] in (
                'identity','fast','open3d_icp_then_xyz','graph_shared','pool_independent','pool_compatible'):
            with np.load(old) as a:np.testing.assert_array_equal(a['xyz_world'],out)
            old_checked+=1
    for case in sorted({r['case'] for r in rows}):
        state=[infos[case,m] for m in ('pool_compatible','shared_group_slope','node_intercepts')]
        for key in ('initialization_sha256','grouping_sha256','support_mask_sha256','frozen_state_sha256',
                    'local_assignment_sha256','weights_sha256'):
            assert len({i[key] for i in state})==1,(case,key)
        frozen_checked+=1
    old=json.loads((dest/'PROTECTED_BEFORE.json').read_text())
    assert old==json.loads((dest/'PROTECTED_AFTER.json').read_text())
    frozen=json.loads((dest/'ALGORITHM_SOURCES.json').read_text())
    for p,h in {**old,**frozen}.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    tests=[]
    if args.tests:
        for label,cwd,opts in [('v5',HERE,['discover','-s','.','-p','test_*.py','-v']),
             ('v4',HERE.parent/'exploration_v4',['discover','-s','.','-p','test_*.py','-v']),
             ('v3',HERE.parent/'exploration_v3',['discover','-s','.','-p','test_*.py','-v']),
             ('original',HERE.parent,['discover','-s','tests','-v'])]:
            result=subprocess.run([sys.executable,'-m','unittest',*opts],cwd=cwd,capture_output=True,text=True)
            text=result.stdout+result.stderr
            with (dest/f'TESTS_{label}.log').open('x') as f:f.write(text)
            assert result.returncode==0,text
            tests.append(dict(suite=label,returncode=0,log=str(dest/f'TESTS_{label}.log')))
    report=dict(outputs_checked=len(rows),surface_outputs_recomputed=len(differences),
        max_surface_score_difference_mm=max(differences,default=0),fixed_state_triplets_checked=frozen_checked,
        exact_historical_v4_output_replays=old_checked,protected_files=len(old),historical_files_unchanged=True,
        frozen_sources_unchanged=True,tests=tests,elapsed_s=time.perf_counter()-started,
        scope='saved-output numerical/contracts audit; not proof of true real accuracy or novelty')
    with (dest/'VERIFICATION.json').open('x') as f:json.dump(report,f,indent=2)
    with (dest/'VERIFICATION_SOURCE.py').open('x') as f:f.write(Path(__file__).read_text())
    print(json.dumps(report,indent=2))


if __name__=='__main__':run()

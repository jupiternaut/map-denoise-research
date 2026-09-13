"""Independent distance recomputation and same-target/initialization audits."""
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


def run():
    parser=argparse.ArgumentParser();parser.add_argument('run_dir',type=Path)
    parser.add_argument('--tests',action='store_true');args=parser.parse_args()
    d=args.run_dir.resolve();start=time.perf_counter()
    if (d/'VERIFICATION.json').exists():raise FileExistsError('verification exists')
    rows=list(csv.DictReader((d/'RESULTS.csv').open()))
    assert len(rows)==len({(r['case'],r['method']) for r in rows})
    differences=[];infos={};scored=0
    for r in rows:
        assert r['ok']=='True',r
        with np.load(r['output']) as a:out=a['xyz_world'];p=a['xyz_world_canonical']*1000
        with np.load(d/'inputs'/(r['case']+'.npz')) as a:inp=a['xyz_world']
        assert out.shape==inp.shape and np.isfinite(out).all()
        info=json.loads(Path(r['output']).with_suffix('.json').read_text())['info']
        infos[(r['case'],r['method'])]=info
        if r.get('geometry_scored')!='True':continue
        ev=read_evaluation(Path(r['source']));nearest=np.full(len(p),np.inf)
        for z,x0,x1,y0,y1 in ev['surface_rectangles_mm']:
            dx=np.maximum.reduce([x0-p[:,0],p[:,0]-x1,np.zeros(len(p))])
            dy=np.maximum.reduce([y0-p[:,1],p[:,1]-y1,np.zeros(len(p))])
            nearest=np.minimum(nearest,np.sqrt(dx**2+dy**2+(p[:,2]-z)**2))
        value={'surface_accuracy_mean_mm':np.mean(nearest),'surface_accuracy_rms_mm':np.sqrt(np.mean(nearest**2)),
               'surface_accuracy_p95_mm':np.quantile(nearest,.95)}
        err=max(abs(float(r[k])-v) for k,v in value.items());assert err<1e-8
        differences.append(float(err));scored+=1
    targets_checked=0;pool_support_checked=0
    for case in sorted({r['case'] for r in rows}):
        family=[infos[(case,m)] for m in ('zero_balanced_6','zero_unbalanced_6','warm_balanced_2','warm_unbalanced_2','graph_projected_bias_only') if (case,m) in infos]
        fingerprints=[a['target_representation_sha256'] for a in family]
        assert len(set(fingerprints))<=1,(case,fingerprints)
        targets_checked+=bool(family)
        pool=[infos[(case,m)] for m in ('pool_independent','pool_global','pool_compatible') if (case,m) in infos]
        if pool:
            for a in pool[1:]:
                np.testing.assert_array_equal(a['bias_mm'],pool[0]['bias_mm'])
                assert a['supported_fraction']==pool[0]['supported_fraction']
                assert a.get('local_k')==pool[0].get('local_k')
            if pool[-1].get('status')=='APPLY':assert pool[-1]['complete_link_verified'] is True
            pool_support_checked+=1
    old=json.loads((d/'PROTECTED_BEFORE.json').read_text())
    assert old==json.loads((d/'PROTECTED_AFTER.json').read_text())
    for path,value in old.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==value,path
    source=json.loads((d/'ALGORITHM_SOURCES.json').read_text())
    for path,value in source.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==value,path
    tests=[]
    if args.tests:
        for label,cwd,test_args in [('v4',HERE,['discover','-s','.','-p','test_*.py','-v']),
                     ('v3',HERE.parent/'exploration_v3',['discover','-s','.','-p','test_*.py','-v']),
                     ('original',HERE.parent,['discover','-s','tests','-v'])]:
            p=subprocess.run([sys.executable,'-m','unittest',*test_args],cwd=cwd,capture_output=True,text=True)
            log=p.stdout+p.stderr
            with (d/f'TESTS_{label}.log').open('x') as f:f.write(log)
            assert p.returncode==0,log
            tests.append({'suite':label,'returncode':p.returncode,'log':str(d/f'TESTS_{label}.log')})
    report={'outputs_checked':len(rows),'independent_surface_scores_checked':scored,
            'max_surface_score_difference_mm':max(differences,default=0.),
            'identical_transport_targets_checked':targets_checked,'identical_pool_initialization_support_checked':pool_support_checked,
            'historical_files_unchanged':True,'protected_file_count':len(old),'frozen_algorithm_unchanged':True,
            'tests':tests,'elapsed_s':time.perf_counter()-start,
            'scope':'numerical and contract audit; not proof of real accuracy or novelty'}
    with (d/'VERIFICATION.json').open('x') as f:json.dump(report,f,indent=2)
    with (d/'VERIFICATION_SOURCE.py').open('x') as f:f.write(Path(__file__).read_text())
    print(json.dumps(report,indent=2))


if __name__=='__main__':run()

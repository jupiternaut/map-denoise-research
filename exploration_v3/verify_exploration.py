"""Recompute finite-surface scores directly and retain actual test output."""
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
    parser=argparse.ArgumentParser(); parser.add_argument('run_dir',type=Path)
    args=parser.parse_args(); d=args.run_dir.resolve()
    if (d/'VERIFICATION.json').exists(): raise FileExistsError('verification exists')
    start=time.perf_counter(); rows=list(csv.DictReader((d/'RESULTS.csv').open()))
    assert len(rows)==len({(r['case'],r['method']) for r in rows})
    max_error=0.; scored=0; checked=0
    for row in rows:
        assert row['ok']=='True',row
        with np.load(row['output']) as a:
            out=a['xyz_world']; p=a['xyz_world_canonical']*1000
        with np.load(d/'inputs'/(row['case']+'.npz')) as a:
            inp=a['xyz_world']; ids=a['scan_id']
        assert out.shape==inp.shape and np.isfinite(out).all()
        checked+=1
        if row.get('geometry_scored')=='True':
            ev=read_evaluation(Path(row['source']))
            # Independent scalar loop over finite rectangles, no evaluate_v2.
            nearest=np.full(len(p),np.inf)
            for z,xlo,xhi,ylo,yhi in ev['surface_rectangles_mm']:
                dx=np.maximum.reduce([xlo-p[:,0],p[:,0]-xhi,np.zeros(len(p))])
                dy=np.maximum.reduce([ylo-p[:,1],p[:,1]-yhi,np.zeros(len(p))])
                distance=np.sqrt(dx*dx+dy*dy+(p[:,2]-z)**2)
                nearest=np.minimum(nearest,distance)
            computed={'surface_accuracy_mean_mm':np.mean(nearest),
                      'surface_accuracy_rms_mm':np.sqrt(np.mean(nearest**2)),
                      'surface_accuracy_p95_mm':np.quantile(nearest,.95)}
            err=max(abs(float(row[k])-float(v)) for k,v in computed.items())
            max_error=max(max_error,err); assert err<1e-8,(row['case'],row['method'],err)
            scored+=1
    old_before=json.loads((d/'PROTECTED_BEFORE.json').read_text())
    old_after=json.loads((d/'PROTECTED_AFTER.json').read_text())
    assert old_before==old_after
    for path,value in old_before.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==value,path
    source_manifest=json.loads((d/'SOURCE_MANIFEST.json').read_text())
    for path,value in source_manifest.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==value,path
    test_runs=[]
    for name,cwd,args_test in [('new',HERE,['discover','-s','.', '-p','test_*.py','-v']),
                             ('existing',HERE.parent,['discover','-s','tests','-v'])]:
        proc=subprocess.run([sys.executable,'-m','unittest',*args_test],cwd=cwd,capture_output=True,text=True)
        log=proc.stdout+proc.stderr
        with (d/f'TESTS_{name}.log').open('x') as f: f.write(log)
        test_runs.append({'suite':name,'returncode':proc.returncode,'log':str(d/f'TESTS_{name}.log')})
        assert proc.returncode==0,log
    report={'saved_outputs_checked':checked,'independent_surface_scores_checked':scored,
            'max_surface_score_difference_mm':max_error,'historical_files_checked':len(old_before),
            'historical_files_unchanged':True,'frozen_sources_unchanged':True,'test_runs':test_runs,
            'elapsed_s':time.perf_counter()-start,
            'scope':'implementation/score consistency checks, not independent real accuracy or novelty'}
    with (d/'VERIFICATION.json').open('x') as f: json.dump(report,f,indent=2)
    with (d/'VERIFICATION_SOURCE.py').open('x') as f: f.write(Path(__file__).read_text())
    print(json.dumps(report,indent=2))


if __name__=='__main__':run()

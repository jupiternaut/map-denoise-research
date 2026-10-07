"""Portable read-only archive/table verification; optional frozen-selector replay."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
SNAP=ROOT/'research_snapshots/2026-10-07'
RUN=SNAP/'plane-support-20261007T084206Z'
DEP=SNAP/'dependencies'


def load(p):
    return json.loads(p.read_text())


def readcsv(p):
    with p.open(newline='') as f:
        return list(csv.DictReader(f))


def near(a,b):
    assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-9),(a,b)


def check_archive():
    m=load(ROOT/'publication/PLANE_SUPPORT_20261007_MANIFEST.json')
    expected={r['path'] for r in m['records']}
    actual={str(p.relative_to(ROOT)) for base in (RUN,DEP) for p in base.rglob('*') if p.is_file()}
    assert actual==expected,{'missing':sorted(expected-actual),'extra':sorted(actual-expected)}
    for r in m['records']:
        p=ROOT/r['path']
        assert not p.is_symlink(),p
        assert p.stat().st_size==r['bytes'],p
        assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],p
    return len(expected)


def tables():
    rows=readcsv(RUN/'evaluation/POINT_METRICS.csv')
    summary=load(RUN/'evaluation/RESULTS.json')
    assert len(rows)==3584
    byarm={}
    for metric in summary['metrics']:
        pp=[p for p in rows if p['arm']==metric['arm']]
        assert len(pp)==512 and len({(p['roi'],p['query']) for p in pp})==512
        primary=[p for p in pp if p['primary']=='True']
        assert len(primary)==484
        byarm[metric['arm']]={(p['roi'],p['query']):p for p in primary}
        roi={r:[float(p['distance_mm']) for p in primary if p['roi']==r] for r in sorted({p['roi'] for p in primary})}
        rm={r:sum(x*x for x in values)/len(values) for r,values in roi.items()}
        near(metric['mse_mm2'],sum(rm.values())/len(rm))
        near(metric['mae_mm'],sum(sum(v)/len(v) for v in roi.values())/len(roi))
        for r,v in rm.items():near(metric['roi_mse'][r],v)
        for sid in ('118','122'):
            v=[v for r,v in rm.items() if r.startswith('scan'+sid+'_')]
            near(metric['scene_mse'][sid],sum(v)/len(v))
        pairs=[(float(p['incumbent_distance_mm']),float(p['distance_mm'])) for p in primary]
        counts=dict(improved=sum(b<a-1e-9 for a,b in pairs),worsened=sum(b>a+1e-9 for a,b in pairs),
                    unchanged=sum(abs(b-a)<=1e-9 for a,b in pairs),severe_gt5=sum(b>5 for a,b in pairs),
                    good_le1=sum(b<=1 for a,b in pairs),new_1mm_harm_vs_incumbent=sum(a<=1 and b>1 for a,b in pairs),
                    finite=sum(bool(p['distance_mm']) for p in pp),
                    original_missing_finite=sum(p['primary']=='False' and bool(p['distance_mm']) for p in pp))
        for k,v in counts.items():assert metric[k]==v,(metric['arm'],k,v)
    harms=lambda p:float(p['incumbent_distance_mm'])<=1 and float(p['distance_mm'])>1
    extra=[(r,int(q)) for (r,q),p in byarm['joint_interval'].items()
           if harms(p) and not harms(byarm['photo_U11'][(r,q)])]
    assert sorted(map(tuple,summary['additional_harms_vs_photo']))==sorted(extra)
    mm={m['arm']:m['mse_mm2'] for m in summary['metrics']}
    assert summary['primary_success']==(mm['joint_interval']<mm['photo_U11'] and not extra)
    return dict(point_rows=len(rows),arms=len(summary['metrics']),primary_success=summary['primary_success'],
                photo_mse_mm2=mm['photo_U11'],joint_interval_mse_mm2=mm['joint_interval'])


def replay():
    try:
        import numpy
    except ImportError as e:
        raise SystemExit('--replay requires NumPy; archive/table verification needs only the standard library') from e
    sys.path.insert(0,str(RUN))
    spec=importlib.util.spec_from_file_location('archived_plane_experiment',RUN/'experiment.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Explicit publication-only data mapping; source files and original locks stay unchanged.
    module.INPUTS=[DEP/'official-mechanism-20261001T183124Z/EVIDENCE.json',
                   DEP/'official-mechanism-20261001T183124Z/neighborhood/NEIGHBORS.jsonl',
                   DEP/'rescue-local-20260930T201603Z/PLAN.json']
    rows=module.sanitize(load(module.INPUTS[0]))
    groups,_=module.read_neighbors()
    plan,protocol=load(module.INPUTS[2]),load(RUN/'PROTOCOL.json')
    saved={(r['roi'],r['query']):r for r in load(RUN/'DECISIONS.json')['rows']}
    assert len(rows)==len(saved)==21
    for row in rows:
        key=(row['roi'],row['query'])
        intervals=module.make_intervals(row,groups[key],plan,protocol)
        assert module.clean(intervals)==saved[key]['intervals'],key
        actual=module.decisions(row['objects'],intervals,protocol,row['baseline_id'])
        assert actual==saved[key]['decisions'],key
    return dict(decisions_reproduced=len(rows),candidates=sum(o['candidate_id']!=-1 for r in rows for o in r['objects']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay',action='store_true')
    args=parser.parse_args()
    result=dict(archive_files=check_archive(),**tables())
    if args.replay:result.update(replay())
    print(json.dumps(dict(status='PASS',**result),indent=2))
    print('Checks archived bytes, table arithmetic, and optional frozen selection only; no new MVS or raw-laser evaluation.')


if __name__=='__main__':
    main()

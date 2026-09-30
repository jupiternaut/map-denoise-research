"""CPU batch driver. Completed sealed cases are verified, never overwritten."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from pathlib import Path
import argparse, hashlib, json, subprocess, time
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parent
PY='/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def call(cmd,name):
    logs=ROOT/'logs';logs.mkdir(exist_ok=True)
    path=logs/(name+'.log')
    if path.exists(): path=logs/(name+'-'+str(time.time_ns())+'.log')
    with path.open('x') as f: ret=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
    print(name,ret.returncode,path,flush=True)
    if ret.returncode: raise RuntimeError(f'failed {name}: {path}')

def inference(task):
    u,c=task; dest=ROOT/'predictions'/u/c
    if (dest/'SEALED.json').exists():
        seal=json.loads((dest/'SEALED.json').read_text())
        assert all(sha(dest/p)==h for p,h in seal['files'].items())
        return
    call([PY,'-B',str(ROOT/'replay.py'),'--upstream',u,'--case',c],u+'_'+c)

def seal():
    plan=json.loads((ROOT/'PLAN_CORRECTED.json').read_text())
    for u in ('U0','U1'):
        for c in plan['cases']:
            d=ROOT/'predictions'/u/c['case_id']
            assert (d/'SEALED.json').exists() or (d/'INCOMPLETE.json').exists(),d
    paths=list((ROOT/'predictions').rglob('*'))+list((ROOT/'initialization').rglob('*'))
    paths += [ROOT/n for n in ('CAMERA_MAPPING.json','PLAN_CORRECTED.json','INITIALIZER_PROTOCOL.md','corrected_rebuild.py','replay.py','run_all.py')]
    record=dict(files={str(p.relative_to(ROOT)):sha(p) for p in sorted(paths) if p.is_file()},gt_accessed=False,
                sealed_at_unix=time.time(),expected_case_count=24)
    with (ROOT/'PREDICTIONS_SEALED.json').open('x') as f: json.dump(record,f,indent=2)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['inference','seal'],required=True);args=ap.parse_args()
    if args.stage=='seal':seal()
    else:
        plan=json.loads((ROOT/'PLAN_CORRECTED.json').read_text())
        tasks=[(u,c['case_id']) for u in ('U0','U1') for c in plan['cases']]
        with ThreadPoolExecutor(max_workers=2) as pool: list(pool.map(inference,tasks))

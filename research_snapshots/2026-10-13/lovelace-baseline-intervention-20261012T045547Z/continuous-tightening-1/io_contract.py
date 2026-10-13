"""A10 target, immutable input contracts and output I/O; no scientific solver."""
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parent
EXPECTED = Path('/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/continuous-tightening-1')
P = ROOT.parent
A9 = P/'continuous-anchors-1'
A8 = P/'independent-anchors-1'
OLD = A9/'outputs/continuous-v1'
OUT = ROOT/'outputs/tightening-v1'
PYTHON = Path('/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python')
SCIENCE_NAMES = ('AGENTS.md','PROTOCOL.md','REGISTRATION.json','io_contract.py',
    'tightening_solver.py','test_tightening_solver.py','verify_tightening.py',
    'test_verify_tightening.py','run_tightening.py','audit_tightening.py',
    'execute_tightening.py','test_tightening_pipeline.py')
def load(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(value, message):
    if not value: raise RuntimeError(message)
def target():
    require(socket.gethostname()=='liekkas' and ROOT==EXPECTED and ROOT.resolve()==EXPECTED
            and Path(sys.executable)==PYTHON and sys.flags.dont_write_bytecode, 'wrong exact A10 target/runtime')
def save(path,value,replace=False):
    path=Path(path)
    require(path.resolve().is_relative_to(OUT),'write outside A10 formal output')
    require(not replace or path.name=='RUN_STATUS.json','only live status may replace')
    path.parent.mkdir(parents=True,exist_ok=True)
    dest=path.with_suffix('.pending') if replace else path
    with dest.open('x') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False); f.write('\n')
    if replace: dest.replace(path)
def sources():
    files=load(OLD/'SOURCE_LOCK.json')['files'].copy()
    files.update({str(ROOT/n):sha(ROOT/n) for n in SCIENCE_NAMES})
    return dict(sorted(files.items()))
def inputs():
    paths=[A9/'ARTIFACTS.json',A9/'FINAL_SEAL.json',OLD/'sets.json',OLD/'observations.json',OLD/'tasks.json',
        OLD/'STAGE_STATUS.json',OLD/'verification/VERIFY.json',A8/'data/anchors.json',
        A8/'data/private/measurements.json',P/'appearance-variation-1/data/private/truth.json',
        P/'joint-feasibility-1/outputs/polytope-v1/outers.json']
    return {str(p):sha(p) for p in paths}
def validate_old():
    seal=load(A9/'FINAL_SEAL.json')
    require(sha(A9/'ARTIFACTS.json')==seal['artifact_index_sha256'],'A9 index drift')
    for rel,digest in load(A9/'ARTIFACTS.json')['files'].items():
        require(sha(A9/rel)==digest,'A9 artifact drift: '+rel)
    for path,digest in load(OLD/'SOURCE_LOCK.json')['files'].items():
        require(sha(path)==digest,'ancestral source drift: '+path)
    require(load(OLD/'STAGE_STATUS.json')['status']=='COMPLETE' and load(OLD/'verification/VERIFY.json')['status']=='PASS','A9 gate')
def public_inputs():
    observations=load(OLD/'observations.json')
    anchors={a['observation_id']:a for a in load(A8/'data/anchors.json')}
    oldarms=load(P/'joint-feasibility-1/outputs/polytope-v1/outers.json')
    baseline={s['observation_id']:s for s in load(OLD/'sets.json') if s['arm']=='both'}
    require(len(observations)==36 and len(baseline)==36 and set(anchors)==set(baseline)=={p['observation_id'] for p in observations},'fixed observation matrix')
    return observations,anchors,oldarms,baseline
def probe_positions(record,exact):
    """Only public retained extreme leaves; no incumbent or truth argument."""
    leaves=[(bi,b['branch'],n) for bi,b in enumerate(record['branches']) for n in b['trace'] if n['status'].startswith('retained_')]
    if not leaves:return []
    left=min(leaves,key=lambda x:(exact(x[2]['lo']),x[0],x[2]['id']))
    right=min(leaves,key=lambda x:(-exact(x[2]['hi']),x[0],x[2]['id']))
    return [{'side':side,'branch':b,'node_id':n['id'],'depth':str((exact(n['lo'])+exact(n['hi']))/2)}
            for side,(_bi,b,n) in zip(('left','right'),(left,right))]

"""A11 immutable provenance and exact target; no scientific solver."""
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
LOGICAL_ROOT = Path('/srv/slam-research/grf/map-denoise/runs/lovelace-evidence-accumulation-20261013T051931Z')
PHYSICAL_ROOT = Path('/mnt/data4t-usb/experiments/srv/slam-research/grf/map-denoise/runs/lovelace-evidence-accumulation-20261013T051931Z')
ROOT = Path(__file__).resolve().parent
P = Path('/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z').resolve()
A10 = P/'continuous-tightening-1'
A9, A8 = P/'continuous-anchors-1', P/'independent-anchors-1'
OLD = A10/'outputs/tightening-v1'
OLD9 = A9/'outputs/continuous-v1'
OUT = ROOT/'outputs/accumulation-v1'
PYTHON = Path('/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python')
SCIENCE_NAMES = ('AGENTS.md','PROTOCOL.md','REGISTRATION.json','a11_io_contract.py',
    'accumulation_solver.py','test_accumulation_solver.py','verify_accumulation.py',
    'test_verify_accumulation.py','run_accumulation.py','audit_accumulation.py',
    'execute_accumulation.py','test_accumulation_pipeline.py','check_tests.py','THEORY.md')
def load(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(value,message):
    if not value: raise RuntimeError(message)
def target():
    require(socket.gethostname()=='liekkas' and ROOT==PHYSICAL_ROOT and LOGICAL_ROOT.resolve()==ROOT,
            'wrong exact A11 host/root: logical alias must resolve to declared physical object')
    require(Path(sys.executable)==PYTHON and sys.flags.dont_write_bytecode,'wrong existing Python -B')
def save(path,value,replace=False):
    path=Path(path)
    require(path.resolve().is_relative_to(OUT),'write outside A11 output')
    require(not replace or path.name=='RUN_STATUS.json','only live status replaces')
    path.parent.mkdir(parents=True,exist_ok=True)
    dest=path.with_suffix('.pending') if replace else path
    with dest.open('x') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    if replace: dest.replace(path)
def sources():
    files=load(OLD/'SOURCE_LOCK.json')['files'].copy()
    files.update({str(ROOT/n):sha(ROOT/n) for n in SCIENCE_NAMES})
    return dict(sorted(files.items()))
def inputs():
    paths=(A10/'ARTIFACTS.json',A10/'FINAL_SEAL.json',OLD/'sets.json',OLD/'STAGE_STATUS.json',
        OLD/'verification/VERIFY.json',OLD9/'sets.json',OLD9/'observations.json',OLD9/'tasks.json',
        A8/'data/anchors.json',A8/'data/private/measurements.json',
        P/'appearance-variation-1/data/private/truth.json',P/'joint-feasibility-1/outputs/polytope-v1/outers.json')
    return {str(p):sha(p) for p in paths}
def validate_old():
    seal=load(A10/'FINAL_SEAL.json')
    require(sha(A10/'ARTIFACTS.json')==seal['artifact_index_sha256'],'old index drift')
    for rel,digest in load(A10/'ARTIFACTS.json')['files'].items():
        require(sha(A10/rel)==digest,'A10 artifact drift: '+rel)
    for p,digest in load(OLD/'SOURCE_LOCK.json')['files'].items():
        require(sha(p)==digest,'ancestral source drift: '+p)
    for p,digest in load(OLD/'INPUT_PRELOCK.json')['files'].items():
        require(sha(p)==digest,'ancestral input drift: '+p)
    require(load(OLD/'STAGE_STATUS.json')['status']=='COMPLETE' and
            load(OLD/'verification/VERIFY.json')['status']=='PASS','old A10 incomplete')
def public_inputs():
    photos=load(OLD9/'observations.json')
    anchors={a['observation_id']:a for a in load(A8/'data/anchors.json')}
    oldarms=load(P/'joint-feasibility-1/outputs/polytope-v1/outers.json')
    source={s['observation_id']:s for s in load(OLD/'sets.json') if s['arm']=='depth'}
    require(len(photos)==len(source)==36 and set(source)==set(anchors)=={p['observation_id'] for p in photos},'fixed matrix')
    return photos,anchors,oldarms,source
